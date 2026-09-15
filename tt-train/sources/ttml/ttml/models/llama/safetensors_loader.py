# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load HuggingFace Llama safetensors weights into a Python Llama model."""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import ml_dtypes
import numpy as np

import ttnn
import ttml
from ttml.common.utils import resolve_padded_load_shape

from .. import WeightTyingType
from . import LlamaConfig

if TYPE_CHECKING:
    from safetensors import safe_open

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


class _Checkpoint:
    """The tensors of a safetensors directory, indexed from the headers and read one at a time.

    Keyed by canonical name; a read comes back 2-D as ``[out, in]``."""

    def __init__(self, directory: str | os.PathLike) -> None:
        from safetensors import safe_open

        files = sorted(Path(directory).glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {Path(directory)}")

        self._files = ExitStack()
        self._where: dict[str, tuple[safe_open, str]] = {}
        try:
            for path in files:
                print(f"Reading safetensors file: {path}")
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

    def __getitem__(self, name: str) -> np.ndarray:
        file, raw = self._where[name]
        array = file.get_tensor(raw)
        if array.ndim == 1:  # norm gammas; a 1-row weight downstream
            return array.reshape(1, -1)
        if array.ndim != 2:
            raise RuntimeError(f"{raw}: expected a 1-D or 2-D tensor, got shape {array.shape}")
        return array


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


def _assemble(blocks: Sequence[np.ndarray], shard_dim: int | None, mesh_size: int, subject: str) -> np.ndarray:
    """Lay out a parameter's source blocks as one array of rows."""
    if len(blocks) == 1:
        return blocks[0]

    # Stacking on rows needs one width, whatever the placement.
    hidden = blocks[0].shape[1]
    for i, block in enumerate(blocks):
        if block.shape[1] != hidden:
            raise RuntimeError(f"{subject}: block {i} has {block.shape[1]} columns, expected {hidden}")

    if shard_dim != ROW_DIM:
        return np.concatenate(blocks, axis=0)

    # Only a row-shard needs the rows to divide: replicated blocks never get split.
    slices_by_block = []
    for i, block in enumerate(blocks):
        if block.shape[0] % mesh_size != 0:
            raise RuntimeError(
                f"{subject}: block {i} has {block.shape[0]} rows, not divisible over {mesh_size} devices"
            )
        slices_by_block.append(np.split(block, mesh_size))
    return np.concatenate([slices_by_block[i][rank] for rank in range(mesh_size) for i in range(len(blocks))], axis=0)


def _pad_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = arr.shape
    rng = np.random.default_rng(_PAD_SEED)
    if shape[1] > cols:
        right = rng.normal(0.0, _PAD_STDDEV, (rows, shape[1] - cols)).astype(arr.dtype)
        arr = np.concatenate([arr, right], axis=1)
    if shape[0] > rows:
        bottom = rng.normal(0.0, _PAD_STDDEV, (shape[0] - rows, shape[1])).astype(arr.dtype)
        arr = np.concatenate([arr, bottom], axis=0)
    return arr


def _fit(
    arr: np.ndarray, param_shape: tuple[int, int], source_shape: tuple[int, int] | None, subject: str
) -> np.ndarray:
    return _pad_to(arr, resolve_padded_load_shape(arr.shape, param_shape, source_shape or param_shape, name=subject))


def _to_bf16_4d(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(1, 1, *arr.shape).astype(ml_dtypes.bfloat16, order="C", copy=False)


@dataclass(frozen=True)
class _TpAxis:
    mesh: ttml.Mesh
    index: int
    size: int


def _tp_axis() -> _TpAxis | None:
    """None without a mesh or a 'tp' axis: the single-device case, where nothing is sharded."""
    mesh = ttml.maybe_mesh()
    if mesh is None or not mesh.has_axis("tp"):
        return None
    return _TpAxis(mesh, mesh.axis_index("tp"), mesh.axis_size("tp"))


def _sharded_dim(param, tp: _TpAxis | None, subject: str) -> int | None:
    """Which tensor dim *param* shards over 'tp', or ``None`` if replicated."""
    if tp is None:
        return None
    placements = ttml.Sharding.from_tensor(param).placements
    for axis, placement in enumerate(placements):
        if isinstance(placement, ttnn.PlacementShard) and axis != tp.index:
            raise RuntimeError(f"{subject}: sharded over mesh axis {axis}; this loader places weights over 'tp' only.")
    if tp.index >= len(placements):  # a fully replicated tensor flattens to a single Replicate
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
    # (arrays) -> blocks, when the checkpoint layout is not what the parameter wants.
    transform: Callable[..., list[np.ndarray]] | None = None
    # What the assembled sources must measure when the parameter is tile-padded above them (the
    # vocab); None means they must match the parameter exactly.
    source_shape: tuple[int, int] | None = None


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


def _biases(parameter_names: set[str]) -> set[str]:
    """The one exemption from coverage: HF Llama ships no biases, so a model configured with them
    keeps its init values. Read off the model, so a newly biased layer needs no change here;
    ``_check_coverage`` withdraws the exemption for a checkpoint that does ship them."""
    return {name for name in parameter_names if name.endswith("/bias")}


def _check_coverage(parameter_names: set[str], rules: Sequence[_Rule], checkpoint_names: frozenset[str]) -> None:
    """Every parameter must be fed by a rule or be a bias the checkpoint does not carry, and every
    rule must land. A renamed or newly fused module shows up here instead of as a quietly untrained
    weight; so does a checkpoint that ships the biases the model was built with.
    """
    targets = {rule.param for rule in rules}
    checkpoint_has_biases = any(name.endswith(".bias") for name in checkpoint_names)
    exempt = set() if checkpoint_has_biases else _biases(parameter_names)
    uncovered = sorted(parameter_names - targets - exempt)
    unknown = sorted(targets - parameter_names)
    if not uncovered and not unknown:
        return

    detail = "".join(f"\n  no rule feeds       {name}" for name in uncovered)
    detail += "".join(f"\n  no such parameter   {name}" for name in unknown)
    raise RuntimeError(
        f"the loader and this Llama disagree about its parameters:{detail}\n"
        f"Update _rules() in {Path(__file__).name} to match the model; a weight_tying or attention_bias "
        f"mismatch between the LlamaConfig and the model also lands here."
    )


def _check_sources(rules: Sequence[_Rule], checkpoint_names: frozenset[str]) -> None:
    """Every source must exist before anything is read or assigned, so a bad checkpoint fails whole."""
    missing = {rule.param: [s for s in rule.sources if s not in checkpoint_names] for rule in rules}
    missing = {param: sources for param, sources in missing.items() if sources}
    if not missing:
        return

    detail = "".join(f"\n  {param}: the checkpoint has no {', '.join(sources)}" for param, sources in missing.items())
    hint = ""
    if "embed_tokens.weight" in checkpoint_names and all(s == ["lm_head.weight"] for s in missing.values()):
        hint = "\nA tied checkpoint ships no lm_head.weight; load it with weight_tying=Enabled."
    raise RuntimeError(f"the checkpoint lacks tensors the model needs:{detail}{hint}")


def load_from_safetensors(
    model: ttml.modules.AbstractModuleBase,
    safetensors_path: str | os.PathLike,
    config: LlamaConfig,
) -> None:
    """Load HuggingFace Llama .safetensors weights into a Python Llama model.

    *safetensors_path* is a directory of ``.safetensors`` files holding one whole model in HF's
    canonical form. Every check that needs only names runs before the first tensor is read.

    Raises:
        RuntimeError: for any of
            - a parameter no rule feeds
            - a rule naming a parameter the model does not have
            - a missing source tensor
            - a shape that disagrees with the config
            - a parameter sharded over a mesh axis other than 'tp'
    """
    parameters = model.parameters()
    parameter_names = set(parameters)
    rules = list(_rules(config, parameter_names))
    tp = _tp_axis()
    mesh_size = tp.size if tp else 1

    with _Checkpoint(safetensors_path) as checkpoint:
        _check_coverage(parameter_names, rules, checkpoint.names)
        _check_sources(rules, checkpoint.names)

        for rule in rules:
            param = parameters[rule.param]
            blocks = [checkpoint[name] for name in rule.sources]
            if rule.transform:
                blocks = rule.transform(*blocks)
            shard_dim = _sharded_dim(param, tp, rule.param)
            host = _assemble(blocks, shard_dim, mesh_size, rule.param)
            host = _fit(host, _global_shape(param, shard_dim, mesh_size), rule.source_shape, rule.param)

            mapper = tp.mesh.axis_mapper("tp", tdim=shard_dim) if tp and shard_dim is not None else None
            param.assign(
                ttml.autograd.Tensor.from_numpy(
                    _to_bf16_4d(host), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper=mapper
                )
            )
        leftover = checkpoint.names - {source for rule in rules for source in rule.sources}

    print(f"Loaded {len(rules)} parameters from {len(checkpoint.names) - len(leftover)} checkpoint tensors.")
    if biases := _biases(parameter_names):
        print(f"Left at initial values: {len(biases)} biases the checkpoint does not carry.")
    if unused := sorted(n for n in leftover if not n.endswith(_NOT_WEIGHTS)):
        print(f"Note: {len(unused)} checkpoint tensors were not used:")
        for name in unused:
            print(f"  - {name}")
