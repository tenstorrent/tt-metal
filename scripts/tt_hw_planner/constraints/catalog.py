"""Constraint catalog.

A ``Constraint`` is a rule: given an ``OpCallContext`` (component name,
HF class name, captured input shapes/dtypes, opplan), return a
``Violation`` if the rule fires, or ``None``.

Constraints are intentionally lightweight — they're heuristics over
the captured-input metadata, not a full type-checker. They flag
"likely-failure" patterns so the LLM can pre-emptively reach for the
workaround recipe instead of discovering it by trial-and-error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class OpCallContext:
    """What the checker knows about a component before the LLM runs.

    Populated by ``checker.check_component`` from the on-disk manifest
    + opplan. The shape fields use a list-of-list-of-int representation
    so callers don't need torch to parse it.
    """

    component_name: str
    hf_class_name: str
    input_shapes: List[List[int]] = field(default_factory=list)
    input_dtypes: List[str] = field(default_factory=list)
    output_shape: Optional[List[int]] = None
    output_dtype: Optional[str] = None
    opplan: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Violation:
    """One firing rule.

    ``recipe_id`` keys into :data:`recipes.RECIPES`.
    ``details`` is a small dict the recipe template can substitute
    (e.g. the offending shape, the threshold value).
    """

    constraint_name: str
    description: str
    recipe_id: str
    details: Dict[str, Any] = field(default_factory=dict)


# --- Constraint definitions --------------------------------------------------

# Tile width is fixed by the Tenstorrent hardware (Wormhole/Blackhole).
TILE_WIDTH = 32


def _max_non_batch_dim(shape: List[int]) -> int:
    """Largest dim excluding the leading batch dim. Useful for spotting
    a layer-norm "feature dim" candidate without committing to a layout."""
    if len(shape) <= 1:
        return shape[0] if shape else 0
    return max(shape[1:])


def _min_non_batch_dim(shape: List[int]) -> int:
    """Smallest dim excluding the leading batch dim — the most likely
    tile-alignment violator when the model treats it as a feature axis."""
    if len(shape) <= 1:
        return shape[0] if shape else 0
    return min(shape[1:])


def _looks_like_layer_norm(ctx: OpCallContext) -> bool:
    name = (ctx.hf_class_name or "").lower() + " " + (ctx.component_name or "").lower()
    return "layernorm" in name or "layer_norm" in name


def _looks_like_position_embedding(ctx: OpCallContext) -> bool:
    name = (ctx.hf_class_name or "").lower() + " " + (ctx.component_name or "").lower()
    return "positionembedding" in name or "position_embedding" in name


def _looks_like_conv(ctx: OpCallContext) -> bool:
    name = (ctx.hf_class_name or "").lower() + " " + (ctx.component_name or "").lower()
    return "conv" in name or "downsampl" in name or "fuser" in name or "encoder" in name


def _check_layer_norm_small_dim(ctx: OpCallContext) -> Optional[Violation]:
    """Fires when the component looks like a LayerNorm AND any non-batch
    input dim is below TILE_WIDTH.

    Triggers for channels-first LN (where the small dim is the channel
    dim that the LN normalizes over) and for any LN whose last dim is
    sub-tile. The recipe explains both shape layouts.
    """
    if not _looks_like_layer_norm(ctx):
        return None
    if not ctx.input_shapes:
        return None
    shape = ctx.input_shapes[0]
    small = _min_non_batch_dim(shape)
    if small >= TILE_WIDTH or small <= 0:
        return None
    return Violation(
        constraint_name="layer_norm_sub_tile_dim",
        description=(
            f"`{ctx.hf_class_name}` will fail ttnn.layer_norm: input "
            f"shape {shape} has a non-batch dim < tile_width ({small} < {TILE_WIDTH}). "
            f"ttnn.layer_norm requires the normalized dim to be tile-aligned."
        ),
        recipe_id="small_dim_layer_norm",
        details={"shape": shape, "small_dim": small, "tile_width": TILE_WIDTH},
    )


def _check_layer_norm_last_dim_not_aligned(ctx: OpCallContext) -> Optional[Violation]:
    """Fires when the component looks like a LayerNorm AND the last dim
    is >= TILE_WIDTH but not a multiple of it (e.g. last_dim=96)."""
    if not _looks_like_layer_norm(ctx):
        return None
    if not ctx.input_shapes:
        return None
    shape = ctx.input_shapes[0]
    last = shape[-1] if shape else 0
    if last <= 0 or last < TILE_WIDTH or last % TILE_WIDTH == 0:
        return None
    return Violation(
        constraint_name="layer_norm_last_dim_not_tile_aligned",
        description=(
            f"`{ctx.hf_class_name}` last dim {last} is not a multiple of "
            f"tile_width ({TILE_WIDTH}). ttnn.layer_norm rejects this — "
            f"pad to {(last // TILE_WIDTH + 1) * TILE_WIDTH} before the op."
        ),
        recipe_id="padded_layer_norm",
        details={"shape": shape, "last_dim": last, "tile_width": TILE_WIDTH},
    )


def _check_conv_bias_dtype_mismatch(ctx: OpCallContext) -> Optional[Violation]:
    """Conv2d in HF reference often has fp32 bias even when the input
    is bf16, which raises `Input type and bias type should be the same`
    inside the torch reference forward — before any ttnn call.

    Heuristic: component name suggests conv work AND captured input
    dtype is bfloat16 (we cast inputs to model_dtype during capture, so
    this is the most common shape).
    """
    if not _looks_like_conv(ctx):
        return None
    if not ctx.input_dtypes:
        return None
    dt = (ctx.input_dtypes[0] or "").lower()
    if "bf" not in dt and "bfloat" not in dt:
        return None
    return Violation(
        constraint_name="conv_bias_dtype_mismatch",
        description=(
            f"`{ctx.hf_class_name}` takes a {dt} input but its conv "
            f"bias is likely fp32 — torch.nn.functional.conv2d will "
            f"reject the mismatch before your ttnn code runs."
        ),
        recipe_id="bias_dtype_match",
        details={"input_dtype": dt, "component": ctx.component_name},
    )


def _check_position_embedding_shape_kwarg(ctx: OpCallContext) -> Optional[Violation]:
    """Sinusoidal position-embedding modules often have a `shape`
    parameter; the test harness may pass shape both positionally
    (derived from the captured tensor) and as kwarg, raising
    `TypeError: got multiple values for argument 'shape'`.

    Not a ttnn constraint — a test-harness pattern. Surfaces a recipe
    that nudges the LLM to dedupe.
    """
    if not _looks_like_position_embedding(ctx):
        return None
    return Violation(
        constraint_name="position_embedding_shape_arg",
        description=(
            f"`{ctx.hf_class_name}` likely takes a `shape` argument that "
            f"the test scaffolder also passes — risks `TypeError: got "
            f"multiple values for argument 'shape'` at invocation."
        ),
        recipe_id="position_embedding_shape_dedup",
        details={"component": ctx.component_name},
    )


def _check_matmul_k_not_tile_aligned(ctx: OpCallContext) -> Optional[Violation]:
    """When the component is matmul-ish (Linear/MLP/Attention proj) and
    the last input dim is the K dimension, check K % TILE_WIDTH."""
    name = (ctx.hf_class_name or "").lower() + " " + (ctx.component_name or "").lower()
    if not any(t in name for t in ("linear", "mlp", "matmul", "projection", "attention")):
        return None
    if not ctx.input_shapes:
        return None
    shape = ctx.input_shapes[0]
    last = shape[-1] if shape else 0
    if last <= 0 or last % TILE_WIDTH == 0:
        return None
    return Violation(
        constraint_name="matmul_k_dim_not_tile_aligned",
        description=(
            f"`{ctx.hf_class_name}` K dim {last} is not a multiple of "
            f"tile_width ({TILE_WIDTH}). ttnn.matmul rejects this — pad "
            f"K and the corresponding weight axis to "
            f"{(last // TILE_WIDTH + 1) * TILE_WIDTH}."
        ),
        recipe_id="padded_matmul",
        details={"shape": shape, "k_dim": last, "tile_width": TILE_WIDTH},
    )


@dataclass
class Constraint:
    name: str
    check: Callable[[OpCallContext], Optional[Violation]]


@dataclass
class Catalog:
    """A bag of constraints. Run all of them against a context, collect
    every fired violation. Order is preserved so callers can see the
    rule sequence."""

    constraints: List[Constraint]

    def run(self, ctx: OpCallContext) -> List[Violation]:
        out: List[Violation] = []
        for c in self.constraints:
            try:
                v = c.check(ctx)
            except Exception:
                # A buggy constraint must NEVER take down the loop.
                continue
            if v is not None:
                out.append(v)
        return out


def default_catalog() -> Catalog:
    """Singleton-style factory; cheap to call repeatedly."""
    return Catalog(
        constraints=[
            Constraint("layer_norm_sub_tile_dim", _check_layer_norm_small_dim),
            Constraint("layer_norm_last_dim_not_tile_aligned", _check_layer_norm_last_dim_not_aligned),
            Constraint("conv_bias_dtype_mismatch", _check_conv_bias_dtype_mismatch),
            Constraint("position_embedding_shape_arg", _check_position_embedding_shape_kwarg),
            Constraint("matmul_k_dim_not_tile_aligned", _check_matmul_k_not_tile_aligned),
        ]
    )
