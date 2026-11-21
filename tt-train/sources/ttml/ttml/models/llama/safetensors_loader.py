# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load HuggingFace Llama safetensors weights into a Python Llama model."""

from __future__ import annotations

import os
from contextlib import ExitStack
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
from safetensors import safe_open

import ttnn
import ttml
from ttml.common.utils import resolve_padded_load_shape, unpermute_proj_rows

from .. import WeightTyingType
from . import Llama, LlamaConfig, compute_swiglu_intermediate_size

# TTML stores a weight as 4-D (1, 1, out_features, in_features).
ROW_DIM, COL_DIM = 2, 3

# Fills the vocab rows the model adds above the checkpoint's for tile alignment.
# They are never a real token, so small init-like noise; the fixed seed keeps two loads identical.
_PAD_SEED = 0
_PAD_STDDEV = 0.02

# Collapsed on read so the rules name each tensor once.
_NAME_ALIASES = {
    "wte.weight": "embed_tokens.weight",
    "transformer.wte.weight": "embed_tokens.weight",
}

# Float dtypes the loader converts to the parameter's own; anything else (FP8, int8, packed 4-bit) is
# a quantized checkpoint, which would fail mid-load or load raw integers.
_FLOAT_DTYPES = ("F32", "BF16", "F16")

# Shipped by some checkpoints; TTML derives these at runtime.
_NOT_WEIGHTS = ("rotary_emb.inv_freq",)

# Tying makes these one parameter, so the model keeps whichever name it walks first.
_TIED_NAMES = ("Llama/fc/weight", "Llama/tok_emb/weight")


def _canonical(name: str) -> str:
    name = name.removeprefix("model.")
    return _NAME_ALIASES.get(name, name)


class _Checkpoint:
    """The tensors in a directory of .safetensors files, by canonical name.
    Each tensor is read only when asked for, and comes back 2-D as ``[out, in]``."""

    def __init__(self, directory: str | os.PathLike) -> None:
        files = sorted(Path(directory).glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {Path(directory)}")

        self._files = ExitStack()
        self._where: dict[str, tuple[safe_open, str]] = {}
        try:
            for path in files:
                file = self._files.enter_context(safe_open(str(path), framework="np"))
                for name in file.keys():
                    canonical = _canonical(name)
                    if canonical in self._where:
                        raise RuntimeError(f"{name}: collides with another tensor already read as {canonical}")
                    self._where[canonical] = (file, name)
        except BaseException:
            self._files.close()
            raise
        self.names: frozenset[str] = frozenset(self._where)

    def __enter__(self) -> _Checkpoint:
        return self

    def __exit__(self, *exc) -> None:
        self._files.close()

    def shape(self, name: str) -> tuple[int, int]:
        """Return the 2-D shape of *name*, from the file header without reading the data.
        Raise if *name* is not stored as a float the loader supports."""
        file, raw = self._where[name]
        header = file.get_slice(raw)
        if (dtype := header.get_dtype()) not in _FLOAT_DTYPES:
            raise RuntimeError(f"{raw}: stored as {dtype}; this loader reads only {', '.join(_FLOAT_DTYPES)}")
        shape = tuple(header.get_shape())
        if len(shape) == 1:  # norm gammas; a 1-row weight downstream
            return 1, shape[0]
        if len(shape) != 2:
            raise RuntimeError(f"{raw}: expected a 1-D or 2-D tensor, got shape {shape}")
        return shape

    def __getitem__(self, name: str) -> np.ndarray:
        file, raw = self._where[name]
        return file.get_tensor(raw).reshape(self.shape(name))


def _assembled_shape(
    shapes: Sequence[tuple[int, int]], shard_dim: int | None, mesh_size: int, subject: str
) -> tuple[int, int]:
    """Return the shape ``_assemble`` would build from blocks of these shapes.
    Raise if the blocks differ in width, or if a row-sharded block can't be split evenly across devices."""
    width = shapes[0][1]
    for i, (rows, cols) in enumerate(shapes):
        # Stacking on rows needs one width, whatever the placement.
        if cols != width:
            raise RuntimeError(f"{subject}: block {i} has {cols} columns, expected {width}")
        # Only a row-shard splits the blocks: replicated ones are never cut.
        if shard_dim == ROW_DIM and len(shapes) > 1 and rows % mesh_size != 0:
            raise RuntimeError(f"{subject}: block {i} has {rows} rows, not divisible over {mesh_size} devices")
    return sum(rows for rows, _ in shapes), width


def _assemble(blocks: Sequence[np.ndarray], shard_dim: int | None, mesh_size: int) -> np.ndarray:
    """Lay out a parameter's source blocks as one array of rows."""
    if len(blocks) == 1:
        return blocks[0]
    if shard_dim != ROW_DIM:
        return np.concatenate(blocks, axis=0)
    slices_by_block = [np.split(block, mesh_size) for block in blocks]
    return np.concatenate([slices[rank] for rank in range(mesh_size) for slices in slices_by_block], axis=0)


def _pad_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Grow *arr* by rows to *shape*."""
    rows, cols = arr.shape
    if cols != shape[1]:
        raise RuntimeError(f"cannot pad {arr.shape} to {shape}: only rows are padded")
    if shape[0] == rows:
        return arr
    rng = np.random.default_rng(_PAD_SEED)
    bottom = rng.normal(0.0, _PAD_STDDEV, (shape[0] - rows, cols)).astype(arr.dtype)
    return np.concatenate([arr, bottom], axis=0)


def _host_array(rule: _Rule, checkpoint: _Checkpoint, shape: tuple[int, int], shard_dim: int | None, mesh_size: int):
    """*rule*'s sources as one array of *shape*, laid out over *mesh_size* ranks, in the checkpoint's dtype."""
    blocks = [checkpoint[name] for name in rule.sources]
    if rule.transform:
        blocks = rule.transform(*blocks)
    return _pad_to(_assemble(blocks, shard_dim, mesh_size), shape)


@dataclass(frozen=True)
class _TpAxis:
    index: int
    size: int


def _tp_axis() -> _TpAxis | None:
    """None without a mesh or a 'tp' axis: nothing is then sharded over 'tp'."""
    mesh = ttml.maybe_mesh()
    if mesh is None or not mesh.has_axis("tp"):
        return None
    return _TpAxis(mesh.axis_index("tp"), mesh.axis_size("tp"))


def _sharded_dim(param, tp: _TpAxis | None, subject: str) -> int | None:
    """Which tensor dim *param* shards over 'tp', or ``None`` if replicated. A shard over any other
    axis raises."""
    placements = ttml.Sharding.from_tensor(param).placements
    tp_index = tp.index if tp else None
    for axis, placement in enumerate(placements):
        if isinstance(placement, ttnn.PlacementShard) and axis != tp_index:
            raise RuntimeError(f"{subject}: sharded over mesh axis {axis}; this loader places weights over 'tp' only.")
    if tp is None or tp.index >= len(placements):  # a fully replicated tensor flattens to a single Replicate
        return None
    placement = placements[tp.index]
    if not isinstance(placement, ttnn.PlacementShard):
        return None
    if placement.dim not in (ROW_DIM, COL_DIM):
        raise RuntimeError(
            f"{subject}: sharded on dim {placement.dim} over 'tp'; expected {ROW_DIM} (rows) or {COL_DIM} (cols)."
        )
    return placement.dim


def _global_shape(param, shard_dim: int | None, mesh_size: int) -> tuple[int, int]:
    """The parameter's shape before sharding, i.e. the shape the checkpoint should supply."""
    rows, cols = param.shape()[-2:]
    return (
        rows * mesh_size if shard_dim == ROW_DIM else rows,
        cols * mesh_size if shard_dim == COL_DIM else cols,
    )


@dataclass(frozen=True)
class _Rule:
    """One parameter and the checkpoint tensors that feed it, in fused-block order."""

    param: str
    sources: tuple[str, ...]
    # (arrays) -> blocks of the same shapes, when the checkpoint layout is not what the parameter wants.
    transform: Callable[..., list[np.ndarray]] | None = None
    # What the assembled sources must measure when the parameter is tile-padded above them (the
    # vocab); None means they must match the parameter exactly.
    source_shape: tuple[int, int] | None = None
    # The rows each source must have, for a fused parameter: a short block next to a long one still
    # fills the parameter. None for a single source, whose rows the parameter's shape already fixes.
    block_rows: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.sources) > 1 and len(self.block_rows or ()) != len(self.sources):
            raise ValueError(f"{self.param}: a fused rule needs block_rows for each of its {len(self.sources)} sources")


def _load_plan(
    rule: _Rule, param, tp: _TpAxis | None, mesh_size: int, checkpoint: _Checkpoint
) -> tuple[int | None, tuple[int, int]]:
    """Return the dim *param* is sharded on over 'tp', and the shape to pad its sources to."""
    shard_dim = _sharded_dim(param, tp, rule.param)
    shapes = [checkpoint.shape(s) for s in rule.sources]
    for source, (rows, _), expected in zip(rule.sources, shapes, rule.block_rows or ()):
        if rows != expected:
            raise RuntimeError(f"{rule.param}: {source} has {rows} rows; the config implies {expected}")
    assembled = _assembled_shape(shapes, shard_dim, mesh_size, rule.param)
    param_shape = _global_shape(param, shard_dim, mesh_size)
    return shard_dim, resolve_padded_load_shape(
        assembled, param_shape, rule.source_shape or param_shape, name=rule.param
    )


def _rules(config: LlamaConfig, parameter_names: set[str]) -> Iterator[_Rule]:
    """The whole HF -> TTML mapping for a Llama built from *config*."""
    embedding_shape = (config.vocab_size, config.hidden_size)
    if config.weight_tying == WeightTyingType.Enabled:
        tied = next((name for name in _TIED_NAMES if name in parameter_names), _TIED_NAMES[0])
        yield _Rule(tied, ("embed_tokens.weight",), source_shape=embedding_shape)
    else:
        yield _Rule("Llama/tok_emb/weight", ("embed_tokens.weight",), source_shape=embedding_shape)
        yield _Rule("Llama/fc/weight", ("lm_head.weight",), source_shape=embedding_shape)
    yield _Rule("Llama/ln_fc/gamma", ("norm.weight",))

    # As the model sizes them: GroupedQueryAttention and LlamaMLP.
    head_dim = config.hidden_size // config.num_attention_heads
    q_rows, kv_rows = config.num_attention_heads * head_dim, config.num_key_value_heads * head_dim
    intermediate_size = config.intermediate_size or compute_swiglu_intermediate_size(config.hidden_size)

    def rope_unpermute(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> list[np.ndarray]:
        return [
            unpermute_proj_rows(q, config.num_attention_heads),
            unpermute_proj_rows(k, config.num_key_value_heads),
            v,  # V is not rotated
        ]

    for layer in range(config.num_hidden_layers):
        param, hf = f"Llama/blocks/{layer}", f"layers.{layer}"
        yield _Rule(f"{param}/attention_norm/gamma", (f"{hf}.input_layernorm.weight",))
        yield _Rule(f"{param}/mlp_norm/gamma", (f"{hf}.post_attention_layernorm.weight",))
        # Block order is the contract with heads_creation, which reads [Q | K | V].
        yield _Rule(
            f"{param}/attention/qkv_linear/weight",
            tuple(f"{hf}.self_attn.{p}_proj.weight" for p in ("q", "k", "v")),
            transform=rope_unpermute,
            block_rows=(q_rows, kv_rows, kv_rows),
        )
        yield _Rule(f"{param}/attention/out_linear/weight", (f"{hf}.self_attn.o_proj.weight",))
        # Block order is the contract with swiglu_packed, which reads [gate | up].
        yield _Rule(
            f"{param}/mlp/w_gate_up/weight",
            tuple(f"{hf}.mlp.{p}_proj.weight" for p in ("gate", "up")),
            block_rows=(intermediate_size, intermediate_size),
        )
        yield _Rule(f"{param}/mlp/w2/weight", (f"{hf}.mlp.down_proj.weight",))


def _biases(parameter_names: set[str]) -> set[str]:
    """The one exemption from coverage: HF Llama ships no biases, so a model configured with them
    keeps its init values. Read off the model, so a newly biased layer needs no change here;
    ``_check_coverage`` withdraws the exemption for a checkpoint that does ship them."""
    return {name for name in parameter_names if name.endswith("/bias")}


def _check_coverage(parameter_names: set[str], rules: Sequence[_Rule], checkpoint_names: frozenset[str]) -> None:
    """Every parameter must be fed by exactly one rule or be a bias the checkpoint does not carry,
    and every rule must land. A renamed or newly fused module shows up here instead of as a quietly
    untrained weight; so does a checkpoint that ships the biases the model was built with, and a
    rule table that would assign one parameter twice.
    """
    feeders = Counter(rule.param for rule in rules)
    targets = set(feeders)
    by_param = {rule.param: rule for rule in rules}

    def checkpoint_ships(bias: str) -> bool:
        """True if the checkpoint has a ``.bias`` beside a source of the sibling weight rule."""
        weight = by_param.get(bias.removesuffix("/bias") + "/weight")
        sources = weight.sources if weight else ()
        return any(source.removesuffix(".weight") + ".bias" in checkpoint_names for source in sources)

    biases = _biases(parameter_names)
    shipped_biases = sorted(bias for bias in biases if checkpoint_ships(bias))
    uncovered = sorted(parameter_names - targets - biases)
    unknown = sorted(targets - parameter_names)
    contested = sorted(name for name, count in feeders.items() if count > 1)
    if not uncovered and not unknown and not contested and not shipped_biases:
        return

    def line(label: str, name: str) -> str:
        return f"\n  {label:<20}{name}"

    detail = "".join(line("no rule feeds", name) for name in uncovered)
    detail += "".join(line("no such parameter", name) for name in unknown)
    detail += "".join(line(f"fed by {feeders[name]} rules", name) for name in contested)
    detail += "".join(line("bias shipped for", name) for name in shipped_biases)
    hints = []
    if uncovered or unknown or contested:
        hints.append(f"Update _rules() in {Path(__file__).name} to match the model.")
    if shipped_biases:
        hints.append("This loader leaves biases at their init values and cannot load the ones this checkpoint ships.")
    raise RuntimeError(f"the loader and this Llama disagree about its parameters:{detail}\n" + "\n".join(hints))


def _check_sources(rules: Sequence[_Rule], checkpoint: _Checkpoint) -> None:
    """Raise if the checkpoint is missing a source tensor, or is untied while the model is tied."""
    names = checkpoint.names
    missing = {rule.param: [s for s in rule.sources if s not in names] for rule in rules}
    missing = {param: sources for param, sources in missing.items() if sources}
    if missing:
        detail = "".join(
            f"\n  {param}: the checkpoint has no {', '.join(sources)}" for param, sources in missing.items()
        )
        hint = ""
        if "embed_tokens.weight" in names and all(s == ["lm_head.weight"] for s in missing.values()):
            hint = "\nA tied checkpoint ships no lm_head.weight; load it with weight_tying=Enabled."
        raise RuntimeError(f"the checkpoint lacks tensors the model needs:{detail}{hint}")

    # A tied checkpoint may still ship lm_head.weight as a copy of the embedding.
    # If it differs from the embedding, the checkpoint is really untied.
    if "lm_head.weight" in names and not any("lm_head.weight" in rule.sources for rule in rules):
        if not np.array_equal(checkpoint["lm_head.weight"], checkpoint["embed_tokens.weight"]):
            raise RuntimeError(
                "the checkpoint ships an lm_head.weight that differs from embed_tokens.weight, which a "
                "weight-tied model has no parameter for; load it with weight_tying=Disabled."
            )


@dataclass(frozen=True)
class LoadResult:
    loaded: tuple[str, ...]  # parameters assigned from the checkpoint
    left_at_init: tuple[str, ...]  # biases the checkpoint does not carry
    unused: tuple[str, ...]  # checkpoint tensors no parameter reads

    def __str__(self) -> str:
        lines = [f"Loaded {len(self.loaded)} parameters."]
        if self.left_at_init:
            lines.append(f"Left at initial values: {len(self.left_at_init)} biases the checkpoint does not carry.")
        if self.unused:
            lines.append(f"Note: {len(self.unused)} checkpoint tensors were not used:")
            lines += [f"  - {name}" for name in self.unused]
        return "\n".join(lines)


def load_from_safetensors(model: Llama, safetensors_path: str | os.PathLike) -> LoadResult:
    """Load a HuggingFace Llama checkpoint into *model*.

    *safetensors_path* is a directory of .safetensors files holding one complete model, with HF's tensor names.
    All checks run before any parameter is written, so a failed load leaves the model unchanged.

    Raises:
        FileNotFoundError: *safetensors_path* has no .safetensors files.
        RuntimeError: if
            - the model has a parameter no rule loads, or a rule loads a parameter the model lacks
            - the checkpoint is missing a tensor, or is tied while the model is untied (or vice versa)
            - the checkpoint ships a bias the model has, which this loader can't load; any other lands in ``unused``
            - a tensor appears in two files, or is not 1-D or 2-D
            - a tensor is stored as anything but F32, BF16 or F16 (a quantized checkpoint)
            - a tensor's shape differs from what the config implies (a transposed weight included)
            - a block of a fused parameter (q, k or v; gate or up) has other rows than the config implies
            - a parameter is sharded over a mesh axis other than 'tp', or over 'tp' on a dim other than rows or columns
    """
    parameters = model.parameters()
    parameter_names = set(parameters)
    rules = list(_rules(model.config, parameter_names))
    tp = _tp_axis()
    mesh_size = tp.size if tp else 1

    with _Checkpoint(safetensors_path) as checkpoint:
        _check_coverage(parameter_names, rules, checkpoint.names)
        _check_sources(rules, checkpoint)
        plans = [_load_plan(rule, parameters[rule.param], tp, mesh_size, checkpoint) for rule in rules]

        for rule, (shard_dim, shape) in zip(rules, plans):
            host = _host_array(rule, checkpoint, shape, shard_dim, mesh_size)
            param = parameters[rule.param]
            ttml.autograd.assign_numpy(param, host, mapper=ttml.Sharding.from_tensor(param).derive_mapper())
        unused = checkpoint.names - {source for rule in rules for source in rule.sources}

    return LoadResult(
        loaded=tuple(rule.param for rule in rules),
        left_at_init=tuple(sorted(_biases(parameter_names))),
        unused=tuple(sorted(name for name in unused if not name.endswith(_NOT_WEIGHTS))),
    )
