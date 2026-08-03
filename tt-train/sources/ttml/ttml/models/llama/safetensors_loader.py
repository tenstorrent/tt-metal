# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load HuggingFace Llama safetensors weights into a Python Llama model.

:func:`_rules` names the checkpoint tensors feeding each parameter; the driver walks parameters
and pulls, so a fused parameter fetches its sources together and nothing needs staging.
A parameter with no rule is an error.

Placement is read from the destination parameter, and fused block sizes from the source tensors,
so the only thing stated twice anywhere is the block *order*.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import ml_dtypes
import numpy as np

import ttnn
import ttml

from .. import WeightTyingType
from . import LlamaConfig

# TTML stores a weight as 4-D (1, 1, out_features, in_features).
ROW_DIM, COL_DIM = 2, 3

# Noise rather than zeros avoids dead neurons; the fixed seed keeps two loads identical.
_PAD_SEED = 0
_PAD_STDDEV = 0.02

# Collapsed on read so the rules name each tensor once.
_NAME_ALIASES = {
    "wte.weight": "embed_tokens.weight",
    "transformer.wte.weight": "embed_tokens.weight",
}

# Shipped by some checkpoints; TTML derives these at runtime.
_NOT_WEIGHTS = ("rotary_emb.inv_freq",)

# Tying makes these one parameter, so the model keeps whichever name it walks first.
_TIED_NAMES = ("Llama/fc/weight", "Llama/tok_emb/weight")


def _canonical(name: str) -> str:
    name = name.removeprefix("model.")
    return _NAME_ALIASES.get(name, name)


def _read_checkpoint(directory: str | os.PathLike) -> dict[str, np.ndarray]:
    """Every tensor in *directory*, keyed by canonical name and shaped 2-D as ``[out, in]``."""
    from safetensors.numpy import load_file

    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {Path(directory)}")

    tensors: dict[str, np.ndarray] = {}
    for path in files:
        print(f"Loading safetensors file: {path}")
        for name, array in load_file(str(path)).items():
            if array.ndim == 1:  # norm gammas; a 1-row weight downstream
                array = array.reshape(1, -1)
            elif array.ndim != 2:
                raise RuntimeError(f"{name}: expected a 1-D or 2-D tensor, got shape {array.shape}")
            canonical = _canonical(name)
            if canonical in tensors:
                raise RuntimeError(f"{name}: collides with another tensor already read as {canonical}")
            tensors[canonical] = array
    return tensors


def _unpermute_proj_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    """Reorder Q/K projection rows from HF grouped layout to interleaved pairs.

    HF stores rows as [first_half, second_half] per head.
    TTML's RoPE expects interleaved: [0, half, 1, half+1, ...].
    """
    rows, _ = w.shape
    if rows % n_heads != 0:
        raise RuntimeError(f"rows {rows} not divisible by n_heads {n_heads}")
    per_head = rows // n_heads
    if per_head % 2 != 0:
        raise RuntimeError(f"rows per head {per_head} must be even")

    half = per_head // 2
    return w.reshape(n_heads, 2, half, -1).transpose(0, 2, 1, 3).reshape(rows, -1)


def _assemble(blocks: Sequence[np.ndarray], shard_dim: int | None, mesh_size: int, what: str) -> np.ndarray:
    """Lay out a parameter's source blocks as one array of rows."""
    if len(blocks) == 1:
        return blocks[0]

    # Stacking on rows needs one width, whatever the placement.
    hidden = blocks[0].shape[1]
    for i, block in enumerate(blocks):
        if block.shape[1] != hidden:
            raise RuntimeError(f"{what}: block {i} has {block.shape[1]} columns, expected {hidden}")

    if shard_dim != ROW_DIM:
        return np.concatenate(blocks, axis=0)

    # Only a row-shard needs the rows to divide: replicated blocks never get split.
    slices_by_block = []
    for i, block in enumerate(blocks):
        if block.shape[0] % mesh_size != 0:
            raise RuntimeError(f"{what}: block {i} has {block.shape[0]} rows, not divisible over {mesh_size} devices")
        slices_by_block.append(np.split(block, mesh_size))
    return np.concatenate([slices_by_block[i][rank] for rank in range(mesh_size) for i in range(len(blocks))], axis=0)


def _require_shape(arr: np.ndarray, shape: tuple[int, int], what: str) -> np.ndarray:
    if arr.shape != shape:
        raise RuntimeError(
            f"{what}: the checkpoint gives {arr.shape} but the parameter is {shape}. "
            f"Check that the LlamaConfig matches the checkpoint."
        )
    return arr


def _pad_to(arr: np.ndarray, shape: tuple[int, int], what: str) -> np.ndarray:
    """Grow *arr* to *shape*. Never shrinks."""
    if arr.shape == shape:
        return arr
    if any(target < source for target, source in zip(shape, arr.shape)):
        raise RuntimeError(
            f"{what}: the checkpoint gives {arr.shape}, larger than the parameter's {shape}, so "
            f"loading would discard weights. Check vocab_size against the checkpoint."
        )
    rows, cols = arr.shape
    rng = np.random.default_rng(_PAD_SEED)
    out = np.empty(shape, arr.dtype)
    out[:rows, :cols] = arr
    # Randomize only the added region; padding a few rows must not cost a whole table of noise.
    if shape[0] > rows:
        out[rows:, :] = rng.normal(0.0, _PAD_STDDEV, (shape[0] - rows, shape[1])).astype(arr.dtype)
    if shape[1] > cols:
        out[:rows, cols:] = rng.normal(0.0, _PAD_STDDEV, (rows, shape[1] - cols)).astype(arr.dtype)
    return out


def _to_bf16_4d(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(1, 1, *arr.shape).astype(ml_dtypes.bfloat16, order="C")


def _sharded_dim(param, what: str) -> int | None:
    """Which tensor dim *param* shards over the 'tp' mesh axis, or ``None`` if replicated.
    No 'tp' axis is the single-device case, not an error.
    """
    mesh = ttml.maybe_mesh()
    if mesh is None or not mesh.has_axis("tp"):
        return None

    sharding = ttml.Sharding.from_tensor(param)
    placements = sharding.placements
    if placements is None:
        raise RuntimeError(
            f"{what}: could not read mesh placements; assuming replicated could scramble the shards."
        ) from sharding.read_error

    tp_axis = mesh.axis_index("tp")
    if tp_axis >= len(placements):  # a fully replicated tensor flattens to a single Replicate
        return None
    placement = placements[tp_axis]
    if not isinstance(placement, ttnn.PlacementShard):
        return None
    if placement.dim not in (ROW_DIM, COL_DIM):
        raise RuntimeError(
            f"{what}: sharded on dim {placement.dim} over 'tp'; expected {ROW_DIM} (rows) or {COL_DIM} (cols)."
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
    # (arrays) -> blocks, when the checkpoint layout is not what the parameter wants.
    transform: Callable[..., list[np.ndarray]] | None = None
    # Vocabulary parameters may be larger than the checkpoint's; everything else must match.
    pad: bool = False


def _tied_embedding_name(parameter_names: set[str]) -> str:
    """The one name the shared embedding/head parameter kept under weight tying."""
    present = [name for name in _TIED_NAMES if name in parameter_names]
    if len(present) != 1:
        raise RuntimeError(
            f"weight_tying=Enabled, so exactly one of {' / '.join(_TIED_NAMES)} should exist, but "
            f"the model has {len(present)}."
        )
    return present[0]


def _rules(config: LlamaConfig, parameter_names: set[str]) -> Iterator[_Rule]:
    """The whole HF -> TTML mapping for a Llama built from *config*."""
    if config.weight_tying == WeightTyingType.Enabled:
        yield _Rule(_tied_embedding_name(parameter_names), ("embed_tokens.weight",), pad=True)
    else:
        yield _Rule("Llama/tok_emb/weight", ("embed_tokens.weight",), pad=True)
        yield _Rule("Llama/fc/weight", ("lm_head.weight",), pad=True)
    yield _Rule("Llama/ln_fc/gamma", ("norm.weight",))

    def rope_unpermute(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> list[np.ndarray]:
        return [
            _unpermute_proj_rows(q, config.num_attention_heads),
            _unpermute_proj_rows(k, config.num_key_value_heads),
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
        )
        yield _Rule(f"{param}/attention/out_linear/weight", (f"{hf}.self_attn.o_proj.weight",))
        # Block order is the contract with swiglu_packed, which reads [gate | up].
        yield _Rule(
            f"{param}/mlp/w_gate_up/weight",
            tuple(f"{hf}.mlp.{p}_proj.weight" for p in ("gate", "up")),
        )
        yield _Rule(f"{param}/mlp/w2/weight", (f"{hf}.mlp.down_proj.weight",))


def _left_at_init(parameter_names: set[str]) -> set[str]:
    """HF Llama ships no biases, so a model configured with them keeps its init values. Read off
    the model, so a newly biased layer needs no change here."""
    return {name for name in parameter_names if name.endswith("/bias")}


def _check_coverage(parameter_names: set[str], rules: Sequence[_Rule]) -> None:
    """Every parameter must be fed by a rule or declared init-only, and every rule must land.

    A renamed or newly fused module shows up here instead of as a quietly untrained weight.
    """
    targets = {rule.param for rule in rules}
    uncovered = sorted(parameter_names - targets - _left_at_init(parameter_names))
    unknown = sorted(targets - parameter_names)
    if not uncovered and not unknown:
        return

    detail = "".join(f"\n  no rule feeds       {name}" for name in uncovered)
    detail += "".join(f"\n  no such parameter   {name}" for name in unknown)
    raise RuntimeError(
        f"the loader and this Llama disagree about its parameters:{detail}\n"
        f"Update _rules() in {Path(__file__).name} to match the model."
    )


def load_from_safetensors(
    model: ttml.modules.AbstractModuleBase,
    safetensors_path: str | os.PathLike,
    config: LlamaConfig,
) -> None:
    """Load HuggingFace Llama .safetensors weights into a Python Llama model.

    *safetensors_path* is a directory of ``.safetensors`` files holding one whole model in HF's
    canonical form.

    Raises:
        RuntimeError: for any of
            - a parameter no rule feeds
            - a rule naming a parameter the model does not have
            - a missing source tensor
            - a shape that disagrees with the config
    """
    checkpoint = _read_checkpoint(safetensors_path)
    parameters = model.parameters()
    # Only used for row-sharded parameters; from the mesh, so it cannot disagree with the model.
    mesh = ttml.maybe_mesh()
    mesh_size = mesh.axis_size("tp") if mesh is not None and mesh.has_axis("tp") else 1

    parameter_names = set(parameters)
    rules = list(_rules(config, parameter_names))
    _check_coverage(parameter_names, rules)

    consumed: set[str] = set()
    for rule in rules:
        missing = [name for name in rule.sources if name not in checkpoint]
        if missing:
            raise RuntimeError(f"{rule.param}: the checkpoint has no {', '.join(missing)}")

        param = parameters[rule.param]
        arrays = [checkpoint[name] for name in rule.sources]
        consumed.update(rule.sources)

        blocks = rule.transform(*arrays) if rule.transform else arrays
        shard_dim = _sharded_dim(param, rule.param)
        host = _assemble(blocks, shard_dim, mesh_size, rule.param)

        wanted = _global_shape(param, shard_dim, mesh_size)
        host = _pad_to(host, wanted, rule.param) if rule.pad else _require_shape(host, wanted, rule.param)

        mapper = ttml.mesh().axis_mapper("tp", tdim=shard_dim) if shard_dim is not None else None
        param.assign(
            ttml.autograd.Tensor.from_numpy(_to_bf16_4d(host), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper=mapper)
        )

    print(f"Loaded {len(rules)} parameters from {len(consumed)} checkpoint tensors.")
    if init_only := sorted(_left_at_init(set(parameters))):
        print(f"Left at initial values ({len(init_only)}): the checkpoint carries no biases.")
    leftover = set(checkpoint) - consumed
    if unused := sorted(n for n in leftover if not n.endswith(_NOT_WEIGHTS)):
        print(f"Note: {len(unused)} checkpoint tensors were not used:")
        for name in unused:
            print(f"  - {name}")
