# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pick how to shard a layer's weights across a 2D mesh. Host-only, no device ops.

A layer is column-parallel -> row-parallel (dim -> intermediate -> dim), and two choices decide
everything:

    intermediate_axis  mesh axis the intermediate dim splits on (attn heads / ffn inner dim), or None
    model_axis         mesh axis the model dim splits on, or None

The intermediate dim is the column matmul's output and the row matmul's input; the model dim is the
column matmul's input and the row matmul's output:

    wqkv [dim, qkv_size]         in = model, out = intermediate (heads)
    wo   [n_heads*head_dim, dim] in = intermediate (heads), out = model

MLP is the same shape with the ffn inner dim in place of the heads (w1/w3 like wqkv, w2 like wo).

Splitting a contraction leaves partial sums, so each matmul is followed by an all-reduce over the
axis that was split: the column matmul over model_axis, the row matmul over intermediate_axis (each
a no-op if that axis isn't split). ccl_auto_shard.all_reduce runs them.
"""

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from models.tt_transformers.tt import params
from models.tt_transformers.tt.auto_shard.cost_model.cost_models import DECODE_STEPS, PREFILL_LEN, cost_model

# ttnn weights are 4D [1, 1, in, out]. The column weight (wqkv / w1,w3) is [model, intermediate];
# the row weight (wo / w2) is [intermediate, model].
COL_INTERMEDIATE = 3
COL_MODEL = 2
ROW_INTERMEDIATE = 2
ROW_MODEL = 3
TILE = 32


@dataclass(frozen=True)
class AttentionShapes:
    n_heads: int
    n_kv_heads: int
    head_dim: int
    dim: int
    qkv_size: int
    tile: int = TILE


@dataclass(frozen=True)
class MLPShapes:
    dim: int
    hidden_dim: int  # the ffn inner dim, in the role attention gives the heads
    tile: int = TILE


@dataclass
class Placement:
    """Which mesh axis each dim splits on (None = replicated)."""

    intermediate_axis: Optional[int]
    model_axis: Optional[int]


@dataclass(frozen=True)
class Sharding:
    """What a module needs to build its weights and collectives.

    Neutral to attention/MLP: the column weight takes col_dims, the row weight takes row_dims.
    """

    placement: Placement
    col_dims: tuple  # ShardTensor2dMesh dims for the column weight
    row_dims: tuple  # ShardTensor2dMesh dims for the row weight
    num_intermediate_shards: int
    num_model_shards: int
    reduce_col_over: Optional[int]  # axis to all-reduce after the column matmul, or None
    reduce_row_over: Optional[int]  # axis to all-reduce after the row matmul, or None


## Legality


def _divides_tiled(total, n, tile):
    """n divides total, and each shard is a whole number of tiles."""
    return total % n == 0 and (total // n) % tile == 0


def _axis_size(mesh_shape, axis):
    """Chips along this axis, or 1 if it's replicated (None)."""
    return 1 if axis is None else mesh_shape[axis]


def is_legal(placement, shapes, mesh_shape):
    """Attention placement builds and tile-aligns. GQA means the KV heads must divide too."""
    if placement.intermediate_axis is not None and placement.intermediate_axis == placement.model_axis:
        return False  # one mesh axis can't carry two dims
    i = _axis_size(mesh_shape, placement.intermediate_axis)
    m = _axis_size(mesh_shape, placement.model_axis)
    heads_fit = shapes.n_heads % i == 0 and shapes.n_kv_heads % i == 0 and _divides_tiled(shapes.qkv_size, i, shapes.tile)
    return heads_fit and _divides_tiled(shapes.dim, m, shapes.tile)


def mlp_is_legal(placement, shapes, mesh_shape):
    """Same, without the GQA constraint."""
    if placement.intermediate_axis is not None and placement.intermediate_axis == placement.model_axis:
        return False
    i = _axis_size(mesh_shape, placement.intermediate_axis)
    m = _axis_size(mesh_shape, placement.model_axis)
    return _divides_tiled(shapes.hidden_dim, i, shapes.tile) and _divides_tiled(shapes.dim, m, shapes.tile)


def is_correct(placement, mesh_shape):
    """Reject attention layouts the KV-cache code can't handle.

    The cache is ReplicateTensorToMesh, a full copy on every chip, sharded only logically by the
    per-chip head count. Prefill can write it two ways:

      * 2D mesh, heads split (intermediate=0, model=1): prefill_prepare_tensor_for_kv_cache picks
        one group's chips by striding the device list by num_model_shards, which lands on the right
        chips only when the group axis is 1 and the heads are on axis 0.
      * Line mesh, heads replicated (intermediate=None): no orthogonal head axis to gather, so the
        cache is fully replicated and prefill writes every chip (see the num_devices_per_group
        guard, which skips the striding).

    Any other split model dim writes the wrong chips. A replicated model dim always works.
    """
    model_split = placement.model_axis is not None and mesh_shape[placement.model_axis] > 1
    if not model_split:
        return True
    if placement.model_axis == 1 and placement.intermediate_axis == 0:
        return True
    return 1 in mesh_shape and placement.intermediate_axis is None


## DRAM footprint

# Bytes per element from the 32x32 tile table, so bfloat8/4 block overhead is counted.
_BYTES_PER_ELEM = {"bfloat16": 2.0, "bfloat8_b": 1.0625, "bfloat4_b": 0.5625, "float32": 4.0}

_SOC_YAML = {"blackhole": "blackhole_140_arch.yaml", "wormhole": "wormhole_b0_80_arch.yaml"}


@functools.lru_cache(maxsize=None)
def _device_dram_bytes(arch_name):
    """One device's DRAM in bytes, from the SoC yaml (WH = 12 GiB, BH ~= 32 GiB).

    Cached because MemoryParams.from_config is evaluated as a select_sharding *argument*, so it runs
    once per layer whatever select_sharding's own cache does. Keyed on the arch name so it holds no
    reference to a config or device.
    """
    key = "blackhole" if "blackhole" in arch_name.lower() else "wormhole"
    root = Path(__file__).resolve().parents[5]
    d = yaml.safe_load(open(root / "tt_metal" / "soc_descriptors" / _SOC_YAML[key]))
    return d["dram_bank_size"] * len(d["dram"])


@dataclass(frozen=True)
class MemoryParams:
    """What attention_footprint needs on top of the weight shapes."""

    n_layers: int
    max_seq_len: int
    batch: int
    wqkv_bytes: float
    wo_bytes: float
    kv_bytes: float
    dram_bytes_per_device: int
    usable_fraction: float = 0.95  # headroom for activations and fragmentation

    @classmethod
    def from_config(cls, configuration):
        """Read these off a ModelArgs. Layer 0's dtypes stand in for all layers."""
        import ttnn
        from models.tt_transformers.tt.model_config import TensorGroup

        def dbytes(group):
            dtype = configuration.decoders_optimizations.get_tensor_dtype(0, group) or ttnn.bfloat16
            # str(ttnn.bfloat8_b) is "DataType.BFLOAT8_B" -> "bfloat8_b"
            return _BYTES_PER_ELEM[str(dtype).rsplit(".", 1)[-1].lower()]

        return cls(
            n_layers=configuration.n_layers,
            max_seq_len=configuration.max_seq_len,
            batch=configuration.batch_size_per_device_group,
            wqkv_bytes=dbytes(TensorGroup.WQKV),
            wo_bytes=dbytes(TensorGroup.WO),
            kv_bytes=dbytes(TensorGroup.KV_CACHE),
            dram_bytes_per_device=_device_dram_bytes(configuration.arch_name),
        )


def attention_footprint(placement, shapes, mesh_shape, memory_params):
    """DRAM one device holds under this placement: weights + KV cache, all layers.

    Weights split on both axes. The KV cache is replicated across the model (group) axis, so it only
    shrinks with the head split.
    """
    intermediate_shards = _axis_size(mesh_shape, placement.intermediate_axis)
    model_shards = _axis_size(mesh_shape, placement.model_axis)

    wqkv = (shapes.dim / model_shards) * (shapes.qkv_size / intermediate_shards) * memory_params.wqkv_bytes
    wo = (shapes.n_heads * shapes.head_dim / intermediate_shards) * (shapes.dim / model_shards) * memory_params.wo_bytes
    weights = (wqkv + wo) * memory_params.n_layers

    # 2 (k + v) * batch * kv_heads * seq_len * head_dim
    kv_cache = (
        2 * memory_params.batch * (shapes.n_kv_heads / intermediate_shards) * memory_params.max_seq_len * shapes.head_dim * memory_params.kv_bytes
    ) * memory_params.n_layers

    return {"weights": weights, "kv_cache": kv_cache, "total": weights + kv_cache}


def is_memory_legal(placement, shapes, mesh_shape, memory_params=None):
    """Footprint fits in DRAM with headroom. memory_params=None skips the check."""
    if memory_params is None:
        return True
    budget = memory_params.dram_bytes_per_device * memory_params.usable_fraction
    return attention_footprint(placement, shapes, mesh_shape, memory_params)["total"] <= budget


## Enumerate + pick


def _real_axes(mesh_shape):
    # splitting a size-1 axis is the same as replicating
    return [a for a in (0, 1) if mesh_shape[a] > 1]


def _drop_all_replicated(placements, mesh_shape):
    """Drop the split-nothing placement on a multichip mesh, unless it's the only option.

    Replicating both dims runs the whole layer on every chip, so each reads the full weights. The
    cost model prices that correctly now (split=1 is its most expensive case), so this is a guard
    rather than a correction. It mattered when the decode term was a flat floor and charged every
    split alike. Keep it until layers can take replicated activations.
    """
    if mesh_shape[0] * mesh_shape[1] <= 1:
        return placements
    sharded = [p for p in placements if not (p.intermediate_axis is None and p.model_axis is None)]
    return sharded or placements


def legal_correct_placements(shapes, mesh_shape, memory_params=None):
    """Attention placements that are legal, KV-correct, and (with memory_params) fit in DRAM."""
    axes = _real_axes(mesh_shape) + [None]
    return [
        p
        for intermediate_axis in axes
        for model_axis in axes
        if is_legal((p := Placement(intermediate_axis, model_axis)), shapes, mesh_shape)
        and is_correct(p, mesh_shape)
        and is_memory_legal(p, shapes, mesh_shape, memory_params)
    ]


def legal_mlp_placements(shapes, mesh_shape):
    """MLP placements that are legal. No KV-correctness rule, no DRAM check."""
    axes = _real_axes(mesh_shape) + [None]
    return [
        p
        for intermediate_axis in axes
        for model_axis in axes
        if mlp_is_legal((p := Placement(intermediate_axis, model_axis)), shapes, mesh_shape)
    ]


def _place(intermediate_axis, intermediate_dim, model_axis, model_dim):
    # put each dim's index on its mesh axis; the other axis stays None (replicated)
    dims = [None, None]
    if intermediate_axis is not None:
        dims[intermediate_axis] = intermediate_dim
    if model_axis is not None:
        dims[model_axis] = model_dim
    return tuple(dims)


def _build_sharding(placement, mesh_shape):
    return Sharding(
        placement=placement,
        col_dims=_place(placement.intermediate_axis, COL_INTERMEDIATE, placement.model_axis, COL_MODEL),
        row_dims=_place(placement.intermediate_axis, ROW_INTERMEDIATE, placement.model_axis, ROW_MODEL),
        num_intermediate_shards=_axis_size(mesh_shape, placement.intermediate_axis),
        num_model_shards=_axis_size(mesh_shape, placement.model_axis),
        reduce_col_over=placement.model_axis,  # column matmul splits the model dim -> reduce there
        reduce_row_over=placement.intermediate_axis,  # row matmul splits the intermediate -> reduce there
    )


def cache_tag(sharding, mesh_shape):
    """Filename tag for the on-disk layout a Sharding produces.

    weight_cache_path's device_name keys only on device count (4 chips -> "N150x4"), so a 1x4 and a
    2x2 already share a cache dir. The bytes depend on the placement too: it fixes which dim each
    weight splits on and num_intermediate_shards, which sets the MLP's fused gate_up interleave. One
    mesh can pick different placements for different workloads (workload_from_config reads
    max_seq_len/max_generated_tokens), so tagging by mesh shape alone would let two layouts collide
    on one filename and a run would silently load wrong-layout weights.
    """
    axis = lambda a: "R" if a is None else str(a)  # noqa: E731. R = replicated
    mesh = "x".join(str(d) for d in mesh_shape)
    return f"mesh{mesh}_i{axis(sharding.placement.intermediate_axis)}m{axis(sharding.placement.model_axis)}"


def _no_placement_message(shapes, mesh_shape, memory_params):
    """Error text for when nothing works; name DRAM if that's what ruled everything out."""
    msg = f"no legal sharding for shapes {shapes} on mesh {mesh_shape}"
    if memory_params is None:
        return msg
    feasible = legal_correct_placements(shapes, mesh_shape)  # ignore memory this time
    if not feasible:
        return msg  # nothing was legal anyway, so memory isn't the cause
    best = min(attention_footprint(p, shapes, mesh_shape, memory_params)["total"] for p in feasible)
    budget = memory_params.dram_bytes_per_device * memory_params.usable_fraction
    return (
        f"{msg}; smallest footprint {best / 2**30:.2f} GiB exceeds DRAM budget "
        f"{budget / 2**30:.2f} GiB ({memory_params.usable_fraction:.0%} of {memory_params.dram_bytes_per_device / 2**30:.0f} GiB)"
    )


def workload_from_config(configuration):
    """The real (prefill_len, decode_steps) to cost against, from a ModelArgs.

    The demo runs a prompt of up to (max_seq_len - max_generated_tokens) tokens then that many
    single-token decodes, so take the largest prompt that fits and the generated-token count. Falls
    back to the params.py defaults for callers that never plumbed max_generated_tokens.
    """
    decode_steps = getattr(configuration, "max_generated_tokens", None)
    if not decode_steps:
        return PREFILL_LEN, DECODE_STEPS
    prefill_len = max(configuration.max_seq_len - decode_steps, TILE)
    return prefill_len, decode_steps


# One decision per distinct (mesh, shapes, workload). Every layer of a Llama/Qwen stack passes
# identical arguments, so a 32-layer model would otherwise rank the same placements 64 times and
# re-query the live fabric on each. Keyed on the arguments rather than on an assumption that layers
# match, so a model that varied a layer still gets its own decision. Mesh identity, not just shape:
# data-parallel builds one model per submesh, and two submeshes of the same shape can sit on
# different chips with different links.
_CACHE = {}


def select_sharding(mesh_device, shapes, mesh_shape, memory_params=None, prefill_len=PREFILL_LEN, decode_steps=DECODE_STEPS):
    """Rank the legal placements with the cost model and build the winner.

    Takes AttentionShapes or MLPShapes. The workload is one prefill of prefill_len tokens plus
    decode_steps decodes. Pass memory_params (attention only) to also drop placements that overflow
    DRAM.
    """
    key = (id(mesh_device), shapes, tuple(mesh_shape), memory_params, prefill_len, decode_steps)
    kind = "MLP" if isinstance(shapes, MLPShapes) else "attention"
    if key not in _CACHE:
        _CACHE[key] = _select_sharding(mesh_device, shapes, mesh_shape, memory_params, prefill_len, decode_steps)
    else:
        logger.info(f"auto-shard: loaded cached {kind} sharding {_CACHE[key].placement} for mesh {tuple(mesh_shape)}")
    return _CACHE[key]


def _select_sharding(mesh_device, shapes, mesh_shape, memory_params, prefill_len, decode_steps):
    mesh_shape = tuple(mesh_shape)
    if isinstance(shapes, MLPShapes):
        placements = legal_mlp_placements(shapes, mesh_shape)
    else:
        placements = legal_correct_placements(shapes, mesh_shape, memory_params=memory_params)
    if not placements:
        raise ValueError(_no_placement_message(shapes, mesh_shape, memory_params))
    placements = _drop_all_replicated(placements, mesh_shape)

    placement = cost_model(mesh_device, placements, shapes, mesh_shape, prefill_len=prefill_len, decode_steps=decode_steps, debug=params.DEBUG)
    
    # To pin a placement by hand, assign it here, e.g. Placement(intermediate_axis=0, model_axis=1).
    # placement = Placement(intermediate_axis=1, model_axis=0)

    logger.info(f"auto-shard: picked {placement} from {placements} for {shapes} on mesh {mesh_shape}")
    return _build_sharding(placement, mesh_shape)
