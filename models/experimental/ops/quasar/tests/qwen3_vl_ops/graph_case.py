# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime for the graph-capture-derived per-op suite (``tests/graph_ops/``).

Every test file in this directory is **generated** by
``generate_from_graph_capture.py`` from a ttnn graph capture
(``generated/ttnn/reports/<run>/graph_capture.python_io.json``). The generated
files hold nothing but data: a ``CASES`` list of the *distinct* calls the model
actually made to one op, each carrying the exact shapes / dtypes / layouts /
memory configs / program configs that were recorded.

This module turns that data back into ttnn objects and runs it. All the
interpretation lives here, so a fix (a renamed program-config field, a better
index heuristic, a new golden reference) is made **once** and every generated
test picks it up without regeneration.

What a case run does
--------------------
1. materialize each input tensor: random torch data of the captured shape/dtype,
   uploaded with the captured layout + memory config (so sharded inputs are
   really sharded, DRAM-sharded weights are really DRAM-sharded, …);
2. rebuild the captured keyword arguments (memory_config, program_config, dtype,
   scalars, activations);
3. call the op;
4. check every returned tensor against its captured output spec — shape, dtype,
   layout and memory config (so a relayout op that hands back its input untouched
   fails), plus finiteness. A multi-output op such as ``nlp_create_qkv_heads`` has
   all of Q, K and V checked. Where the capture never observed an output, what the
   call itself pins down is used instead (``_derived_spec``), and ops whose result
   metadata cannot show they did anything get an explicit ``POSTCONDITION``;
5. on top of that, a torch golden (PCC) for the ops where a reference is
   unambiguous — see ``GOLDEN``.

Fidelity caveats (deliberate, documented, all visible in the generated data)
---------------------------------------------------------------------------
* ``compute_kernel_config`` is captured only as a python object repr
  (``<WormholeComputeKernelConfig object at 0x…>``) — its fields are NOT in the
  capture, so the kwarg is dropped and the op's default is used. This can move
  PCC (math fidelity / fp32 accumulation) but not shapes or program structure.
* Tensor *values* are not captured; inputs are random. Index-like tensors
  (page tables, cur_pos, update_idxs, embedding ids) would fault the device with
  random values, so they get semantic values from ``INDEX_VALUES`` below.
* Program configs are rebuilt field-for-field from the repr (including
  LayerNorm's ``legacy_reduction`` / ``legacy_rsqrt`` / ``use_welford`` and SDPA's
  ``max_cores_per_head_batch``). The only exceptions are the optional
  CoreRangeSet restrictions (``sub_core_grids``, ``allowed_worker_cores``): those
  print as ``std::nullopt`` in every capture so far, and a non-null value skips
  the case instead of running a different core set — see
  ``_PROGRAM_CONFIG_FIELDS`` / ``_MUST_BE_NULL``.
* Mesh placement is replicated onto the (1, 1) mesh the capture ran on.

Set ``TTNN_GRAPH_OPS_NO_GOLDEN=1`` to reduce every case to
"runs + right shape/dtype + finite" (useful while a golden reference is under
suspicion, or on an emulator where PCC of a bfloat8_b path is not the point).
"""

from __future__ import annotations

import math
import os
from typing import Any

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import op_utils as U

# Re-exported so generated files need one import only.
with_default_mesh = U.with_default_mesh
from_tt = U.from_tt

# =============================================================================
# Enum tables (capture string -> ttnn object)
# =============================================================================

DTYPE = {
    "BFLOAT16": ttnn.bfloat16,
    "BFLOAT8_B": ttnn.bfloat8_b,
    "BFLOAT4_B": ttnn.bfloat4_b,
    "FLOAT32": ttnn.float32,
    "UINT32": ttnn.uint32,
    "INT32": ttnn.int32,
    "UINT16": ttnn.uint16,
    "UINT8": ttnn.uint8,
}

LAYOUT = {"TILE": ttnn.TILE_LAYOUT, "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT}

MEM_LAYOUT = {
    "INTERLEAVED": ttnn.TensorMemoryLayout.INTERLEAVED,
    "HEIGHT_SHARDED": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "WIDTH_SHARDED": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "BLOCK_SHARDED": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
}

BUFFER_TYPE = {
    "L1": ttnn.BufferType.L1,
    "DRAM": ttnn.BufferType.DRAM,
    "L1_SMALL": ttnn.BufferType.L1_SMALL,
    "TRACE": ttnn.BufferType.TRACE,
}

ORIENTATION = {"ROW_MAJOR": ttnn.ShardOrientation.ROW_MAJOR, "COL_MAJOR": ttnn.ShardOrientation.COL_MAJOR}

# Integer dtypes get integer torch data (and semantic values where the op needs
# them); everything else gets bfloat16 noise.
_INT_DTYPES = {"UINT32", "INT32", "UINT16", "UINT8"}

# The bar is PCC > 0.999 against the torch reference. Every floor below it is a
# property of the call, not a blanket allowance, and each mirrors the number the
# hand-written tests/ops suite settled on for the same op on real hardware:
#
#   * block-float dtypes quantize to a per-face exponent, so the reference (built in
#     fp32) cannot be reproduced to 0.999 no matter how correct the op is;
#   * a K-deep matmul accumulates rounding over the reduction — tests/ops/test_linear.py:30
#     uses 0.99, and 0.98 for the K=8192 down-projection;
#   * rms_norm (reduction + rsqrt), embedding (gather of a quantized table) and
#     typecast (the captured casts are bf16 -> bfloat8_b) are 0.99 in tests/ops for
#     the same reasons — test_rms_norm.py:56, test_embedding.py:58, test_typecast.py:26.
#
# On top of that, ``compute_kernel_config`` is not reconstructible from a capture
# (see the module docstring), so these run at each op's default math fidelity rather
# than the fidelity the model asked for — which costs precision on exactly the
# reduction-heavy ops listed here.
_DEFAULT_PCC = 0.999
_PCC_BY_DTYPE = {"BFLOAT8_B": 0.97, "BFLOAT4_B": 0.90}
_PCC_BY_OP = {
    "ttnn.linear": 0.99,
    "ttnn.experimental.minimal_matmul": 0.99,
    "ttnn.rms_norm": 0.99,
    "ttnn.embedding": 0.99,
    "ttnn.typecast": 0.99,
}
# Ops that reduce along the input's last dimension, and the depth past which they get
# the looser floor tests/ops uses for the K=8192 down-projection.
_REDUCTION_OPS = frozenset({"ttnn.linear", "ttnn.experimental.minimal_matmul", "ttnn.rms_norm"})
_DEEP_K = 4096
_DEEP_K_PCC = 0.98

NO_GOLDEN = os.environ.get("TTNN_GRAPH_OPS_NO_GOLDEN", "") not in ("", "0", "false", "False")


# =============================================================================
# Memory configs
# =============================================================================


def _core_range_set(ranges):
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0, x1, y1) in ranges}
    )


def _shard_grid_fits(spec, mesh_device):
    """Does the captured shard grid exist on this device? Returns (fits, why not).

    A captured shard grid is absolute (the capture ran on an N150-class 8x8 grid),
    so on a smaller device — e.g. the 2-node Quasar emulator — it may name cores or
    DRAM banks that are not there. DRAM shard grids index banks instead of compute
    cores and are gated separately.
    """
    shard = spec.get("shard")
    if shard is None:
        return True, ""

    max_x = max(r[2] for r in shard["grid"])
    max_y = max(r[3] for r in shard["grid"])
    if spec["buffer"] == "L1":
        grid = mesh_device.compute_with_storage_grid_size()
        if max_x >= grid.x or max_y >= grid.y:
            return False, (
                f"captured L1 shard grid needs cores up to ({max_x},{max_y}); "
                f"device compute grid is {grid.x}x{grid.y}"
            )
    else:
        dram_grid = getattr(mesh_device, "dram_grid_size", None)
        if callable(dram_grid):
            g = dram_grid()
            if max_x >= g.x or max_y >= g.y:
                return False, (
                    f"captured DRAM shard grid needs banks up to ({max_x},{max_y}); " f"device DRAM grid is {g.x}x{g.y}"
                )
    return True, ""


def build_memory_config(spec, mesh_device):
    """Rebuild a captured MemoryConfig; skip the case if its grid exceeds the device.

    A config whose shard grid is not on this device cannot be allocated at all, so
    the case is SKIPPED rather than failed (see ``_shard_grid_fits``).
    """
    if spec is None:
        return None
    shard = spec.get("shard")
    if shard is None:
        return ttnn.MemoryConfig(MEM_LAYOUT[spec["layout"]], BUFFER_TYPE[spec["buffer"]])

    fits, why = _shard_grid_fits(spec, mesh_device)
    if not fits:
        pytest.skip(f"{why} — shape too large for this device")

    shard_spec = ttnn.ShardSpec(_core_range_set(shard["grid"]), list(shard["shape"]), ORIENTATION[shard["orientation"]])
    return ttnn.MemoryConfig(MEM_LAYOUT[spec["layout"]], BUFFER_TYPE[spec["buffer"]], shard_spec)


# =============================================================================
# Program configs
# =============================================================================

# Fields the (keyword-only) python constructors accept. Verified against the
# nanobind definitions:
#   matmul_nanobind.cpp:145,330,556 | layernorm_nanobind.cpp:52
#   transformer_nanobind.cpp:27     | minimal_matmul_nanobind.cpp:164
# Anything the C++ repr prints but the ctor does not accept is listed in
# _DROPPED_FIELDS, so the omission is explicit rather than silent.
_PROGRAM_CONFIG_FIELDS = {
    "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig": ("in0_block_w", "per_core_M", "per_core_N"),
    "MatmulMultiCoreReuseMultiCastProgramConfig": (
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "out_block_h",
        "out_block_w",
        "per_core_M",
        "per_core_N",
        "transpose_mcast",
        "fuse_batch",
    ),
    "MatmulMultiCoreReuseMultiCast1DProgramConfig": (
        "compute_with_storage_grid_size",
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "out_block_h",
        "out_block_w",
        "per_core_M",
        "per_core_N",
        "fuse_batch",
        "mcast_in0",
    ),
    "LayerNormShardedMultiCoreProgramConfig": (
        "compute_with_storage_grid_size",
        "subblock_w",
        "block_h",
        "block_w",
        "inplace",
        "legacy_reduction",
        "legacy_rsqrt",
        "use_welford",
    ),
    "SDPAProgramConfig": (
        "compute_with_storage_grid_size",
        "q_chunk_size",
        "k_chunk_size",
        "exp_approx_mode",
        "max_cores_per_head_batch",
    ),
    "MinimalMatmulConfig": (
        "M_block_size",
        "K_block_size",
        "N_block_size",
        "subblock_h",
        "subblock_w",
        "compute_with_storage_grid_size",
    ),
}

# Fields the repr prints that this runtime does not reconstruct. Both are
# std::nullopt in every capture generated so far; a non-null value makes the case
# skip (see build_program_config) rather than run a subtly different config.
_DROPPED_FIELDS = {
    "SDPAProgramConfig": ("sub_core_grids",),
    "MatmulMultiCoreReuseMultiCastProgramConfig": ("allowed_worker_cores",),
    "MatmulMultiCoreReuseMultiCast1DProgramConfig": ("allowed_worker_cores",),
}

# Optional CoreRangeSet fields that must be std::nullopt for a faithful rebuild.
_MUST_BE_NULL = ("sub_core_grids", "allowed_worker_cores")

# The C++ repr prints bools as 0/1, but these ctor args are bound `.noconvert()`,
# which rejects a python int. Coerce them back to bool.
_BOOL_FIELDS = frozenset(
    {
        "inplace",
        "legacy_reduction",
        "legacy_rsqrt",
        "use_welford",
        "transpose_mcast",
        "fuse_batch",
        "mcast_in0",
        "gather_in0",
        "untilize_out",
        "exp_approx_mode",
    }
)


def build_program_config(spec):
    """Rebuild a captured program config (e.g. SDPAProgramConfig(...))."""
    if spec is None:
        return None
    kind = spec["kind"]
    cls = getattr(ttnn, kind, None)
    if cls is None:
        pytest.skip(f"{kind} is not exposed on this ttnn build")

    fields = spec["fields"]
    allowed = _PROGRAM_CONFIG_FIELDS.get(kind)
    if allowed is None:
        pytest.skip(f"no python field mapping for {kind} — add it to graph_case._PROGRAM_CONFIG_FIELDS")

    kwargs: dict[str, Any] = {}
    for name in allowed:
        if name not in fields:
            continue
        value = fields[name]
        if name == "compute_with_storage_grid_size":
            value = ttnn.CoreCoord(*value)
        elif name in _BOOL_FIELDS:
            value = bool(value)
        kwargs[name] = value

    # A fused activation is reconstructible only through its UnaryOpType name.
    activation = fields.get("fused_activation")
    if activation is not None:
        op_type = getattr(ttnn.UnaryOpType, str(activation).upper(), None)
        if op_type is None:
            pytest.skip(f"{kind} fused_activation={activation!r} is not reconstructible from the capture")
        kwargs["fused_activation"] = ttnn.UnaryWithParam(op_type)

    # Optional core-grid restrictions are not reconstructed; a captured non-null
    # value would change which cores the op runs on, so skip instead of lying.
    for name in _MUST_BE_NULL:
        if fields.get(name) is not None:
            pytest.skip(f"{kind}.{name}={fields[name]!r} is not reconstructible from the capture")

    try:
        return cls(**kwargs)
    except TypeError as exc:  # ttnn version drift in the ctor signature
        pytest.skip(f"{kind}(**{sorted(kwargs)}) rejected by this ttnn build: {exc}")


# =============================================================================
# Input tensors
# =============================================================================


def _numel(shape):
    return math.prod(shape) if shape else 1


def _zeros(spec, case):
    return torch.zeros(spec["shape"], dtype=torch.int32)


def _arange_pages(spec, case):
    """Page table: distinct page ids, one per entry (values must index the cache)."""
    n = _numel(spec["shape"])
    return torch.arange(n, dtype=torch.int32).reshape(spec["shape"])


def _small_position(spec, case):
    """cur_pos / update_idxs: a valid, small KV position for every user."""
    return torch.full(spec["shape"], 8, dtype=torch.int32)


def _embedding_ids(spec, case):
    """Token ids must be < the embedding table's row count (args[1] is the table)."""
    table = case["args"][1]
    rows = table["shape"][-2]
    return torch.randint(0, rows, spec["shape"], dtype=torch.int32)


# A cache write is checked by looking for this value in the cache afterwards, so the
# tensor written into the cache is filled with it instead of random noise. It has to
# be a value the surrounding random data (standard normal, so |x| < 6) cannot produce
# even once across a multi-million-element cache, or the count below is meaningless.
# A power of two is exact in bfloat16/bfloat8_b, and the tolerance is one bfloat8_b
# mantissa step at this exponent (1024/128), which covers a block-float round-trip.
CACHE_SENTINEL = 1024.0
_CACHE_SENTINEL_TOL = 8.0


def _cache_sentinel(spec, case):
    """The tensor a paged-cache op writes: a value we can look for afterwards."""
    return torch.full(spec["shape"], CACHE_SENTINEL, dtype=torch.bfloat16)


# (op name, argument key) -> value generator. Random data in an index tensor
# faults the device, so these are the ops where values carry meaning.
INDEX_VALUES = {
    ("ttnn.embedding", "0"): _embedding_ids,
    ("ttnn.experimental.paged_update_cache", "1"): _cache_sentinel,
    ("ttnn.experimental.paged_update_cache", "page_table"): _arange_pages,
    ("ttnn.experimental.paged_update_cache", "update_idxs_tensor"): _small_position,
    ("ttnn.experimental.paged_fill_cache", "1"): _cache_sentinel,
    ("ttnn.experimental.paged_fill_cache", "2"): _arange_pages,
    ("ttnn.transformer.paged_scaled_dot_product_attention_decode", "page_table_tensor"): _arange_pages,
    ("ttnn.transformer.paged_scaled_dot_product_attention_decode", "cur_pos_tensor"): _small_position,
}


def _torch_data(spec, case, op_name, key):
    hook = INDEX_VALUES.get((op_name, key))
    if hook is not None:
        return hook(spec, case)
    if spec["dtype"] in _INT_DTYPES:
        return _zeros(spec, case)
    return U.torch_rand(spec["shape"])


# Ops that take a HOST tensor: the capture records the argument's spec after the
# upload, so uploading it here first would hand the op a tensor already on device.
HOST_INPUT = {("ttnn.to_device", "0")}


def _rows_of(shape):
    return math.prod(shape[:-1]) if len(shape) > 1 else 1


def _shard_row_capacity(mem):
    """How many rows the captured shard region holds (vs the tensor's logical rows)."""
    shard = mem["shard"]
    cores = sum((x1 - x0 + 1) * (y1 - y0 + 1) for x0, y0, x1, y1 in shard["grid"])
    shard_h = shard["shape"][0]
    return shard_h * cores if mem["layout"] == "HEIGHT_SHARDED" else shard_h


def _is_partial_shard(spec):
    """True when the logical shape does not fill its shard region.

    The model reaches this state by sharding a small tensor into a tile-sized shard
    — e.g. rope's cos/sin at logical [1, 1, 1, 64] inside a 32x64 shard, or decode's
    8 KV heads inside a 32-row shard. Handing such a memory config straight to
    ``from_torch`` pads the *logical* shape up to the shard, which changes what the
    op computes (rope then returns 32 rows instead of the captured 8). Building
    interleaved first and relaying out keeps the logical shape intact.
    """
    mem = spec.get("mem")
    if not mem or not mem.get("shard"):
        return False
    return _rows_of(spec["shape"]) < _shard_row_capacity(mem)


def build_tensor(spec, mesh_device, case, op_name, key):
    """Materialize one captured input tensor. Returns (ttnn tensor, torch source)."""
    data = _torch_data(spec, case, op_name, key)
    if (op_name, key) in HOST_INPUT:
        return ttnn.from_torch(data, dtype=DTYPE[spec["dtype"]], layout=LAYOUT[spec["layout"]]), data

    memory_config = build_memory_config(spec.get("mem"), mesh_device) or ttnn.DRAM_MEMORY_CONFIG
    partial = _is_partial_shard(spec)
    tt = ttnn.from_torch(
        data,
        dtype=DTYPE[spec["dtype"]],
        layout=LAYOUT[spec["layout"]],
        device=mesh_device,
        # a partially filled shard is reached in two steps, see _is_partial_shard
        memory_config=ttnn.DRAM_MEMORY_CONFIG if partial else memory_config,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    if partial:
        tt = ttnn.to_memory_config(tt, memory_config)
    return tt, data


def _build_value(spec, mesh_device, case, op_name, key, torch_sink):
    """Turn one captured argument spec into the object to pass to the op."""
    kind = spec["k"]
    if kind == "t":
        tt, data = build_tensor(spec, mesh_device, case, op_name, key)
        torch_sink[key] = data
        return tt
    if kind == "tlist":
        out = []
        for i, sub in enumerate(spec["tensors"]):
            tt, data = build_tensor(sub, mesh_device, case, op_name, f"{key}[{i}]")
            torch_sink[f"{key}[{i}]"] = data
            out.append(tt)
        return out
    if kind == "mem":
        return build_memory_config(spec, mesh_device)
    if kind == "cfg":
        return build_program_config(spec)
    if kind == "dtype":
        return DTYPE[spec["v"]]
    if kind == "layout":
        return LAYOUT[spec["v"]]
    if kind == "acts":
        return [getattr(ttnn.UnaryOpType, name) for name in spec["v"]]
    if kind == "device":
        return mesh_device
    if kind == "lit":
        return spec["v"]
    if kind == "slices":
        return tuple(slice(*s) if isinstance(s, (list, tuple)) else s for s in spec["v"])
    if kind == "skip":
        pytest.skip(f"argument {key} is not reconstructible from the capture: {spec.get('repr', '')[:120]}")
    raise AssertionError(f"unknown argument kind {kind!r}")


# =============================================================================
# Golden references
# =============================================================================


def _silu(x):
    return torch.nn.functional.silu(x)


_ACTIVATIONS = {"SILU": _silu, "RELU": torch.relu, "GELU": torch.nn.functional.gelu}


def _apply_activations(x, acts):
    for name in acts or ():
        fn = _ACTIVATIONS.get(name)
        if fn is None:
            return None  # unknown activation -> no golden
        x = fn(x)
    return x


def _ref_identity(inputs, kwargs, case):
    """Ops that move/relayout data without changing it."""
    return inputs["0"]


def _ref_matmul(inputs, kwargs, case):
    return inputs["0"].float() @ inputs["1"].float()


def _second_operand(inputs, case):
    """The right-hand operand of a binary op: a tensor, or a captured scalar literal.

    ``ttnn.multiply(cache, 0, output_tensor=cache)`` — the demo's KV-cache reset — passes
    a python scalar, which never becomes a torch input, so ``inputs`` holds only arg 0.
    Returns None when arg 1 is neither, so the case falls back to the structural checks.
    """
    if "1" in inputs:
        return inputs["1"].float()
    spec = case["args"][1] if len(case["args"]) > 1 else None
    if spec is None or spec.get("k") != "lit" or not isinstance(spec.get("v"), (int, float)):
        return None
    return float(spec["v"])


def _ref_binary(fn):
    def ref(inputs, kwargs, case):
        a = inputs["0"].float()
        b = _second_operand(inputs, case)
        if b is None:
            return None
        acts = case["kwargs"].get("input_tensor_a_activations")
        if acts is not None:
            a = _apply_activations(a, acts.get("v"))
            if a is None:
                return None
        return fn(a, b)

    return ref


def _ref_transpose(inputs, kwargs, case):
    dims = [a["v"] for a in case["args"][1:] if a["k"] == "lit"]
    if len(dims) != 2:
        return None
    return inputs["0"].float().transpose(dims[0], dims[1])


def _ref_concat(inputs, kwargs, case):
    dim = case["kwargs"].get("dim", {}).get("v")
    if dim is None:
        return None
    # keys are "0[0]", "0[1]", … — filter before sorting on the list index
    keys = [k for k in inputs if k.startswith("0[")]
    keys.sort(key=lambda s: int(s.split("[")[1].rstrip("]")))
    return torch.cat([inputs[k].float() for k in keys], dim=dim)


def _ref_slice(inputs, kwargs, case):
    lits = [a["v"] for a in case["args"][1:] if a["k"] == "lit"]
    if len(lits) != 2:
        return None
    start, end = lits
    x = inputs["0"].float()
    if len(start) != x.dim():
        return None
    return x[tuple(slice(s, e) for s, e in zip(start, end))]


def _ref_getitem(inputs, kwargs, case):
    key = next((a for a in case["args"][1:] if a["k"] == "slices"), None)
    if key is None:
        return None
    return inputs["0"].float()[tuple(slice(*s) for s in key["v"])]


def _ref_rms_norm(inputs, kwargs, case):
    x = inputs["0"].float()
    weight = inputs.get("weight")
    eps = case["kwargs"].get("epsilon", {}).get("v", 1e-5)
    gamma = weight.float().reshape(-1)[: x.shape[-1]] if weight is not None else None
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return normed * gamma if gamma is not None else normed


def _ref_embedding(inputs, kwargs, case):
    ids = inputs["0"].long()
    table = inputs["1"].float()
    return torch.nn.functional.embedding(ids.reshape(-1), table.reshape(-1, table.shape[-1]))


def _out_shapes(case):
    """Captured output shapes, or None when any of them was never observed."""
    outs = case.get("outs") or []
    if not outs or any(o is None for o in outs):
        return None
    return [tuple(o["shape"]) for o in outs]


def _ref_view(inputs, kwargs, case):
    """Ops that reinterpret the same row-major data: reshape, unsqueeze_to_4D.

    The captured output shape is the reference: same elements, same order. It can be
    *smaller* than the input — ``ttnn.reshape(x, (1,1,1,3072), (1,1,32,3072))`` keeps
    one logical row of a batch-padded decode tensor — in which case the reference is
    the leading elements, which is where row-major puts that row.
    """
    shapes = _out_shapes(case)
    if not shapes:
        return None
    flat = inputs["0"].float().reshape(-1)
    n = math.prod(shapes[0])
    if n > flat.numel():
        return None  # output larger than the input: the extra is padding we cannot define
    return flat[:n].reshape(shapes[0])


# The head ops below mirror the references the hand-written tests/ops suite verified
# on hardware — test_nlp_create_qkv_heads.py:57-60, test_nlp_create_qkv_heads_decode.py:110-113,
# test_nlp_concat_heads.py:51 — so a permutation or a mis-split of the fused QKV is
# caught here too, instead of only shape/dtype. Each returns None if the captured
# shapes do not match the layout it models, so an unfamiliar capture falls back to the
# structural checks rather than failing against a reference that does not apply.
def _qkv_dims(x, shapes):
    """(batch/seq-carrying view, q_dim, kv_dim) for a fused-QKV split, or None."""
    if len(shapes) != 3:
        return None
    q_shape, k_shape, v_shape = shapes
    if k_shape != v_shape or len(q_shape) != 4 or len(k_shape) != 4:
        return None
    head_dim = q_shape[3]
    q_dim, kv_dim = q_shape[1] * head_dim, k_shape[1] * head_dim
    return (q_dim, kv_dim) if x.shape[-1] == q_dim + 2 * kv_dim else None


def _ref_create_qkv_heads(inputs, kwargs, case):
    """Prefill: [1,1,seq,QKV] -> Q [1,heads,seq,hd], K/V [1,kv_heads,seq,hd]."""
    shapes = _out_shapes(case)
    if not shapes:
        return None
    x = inputs["0"].float()
    dims = _qkv_dims(x, [(s[0], s[1], s[2], s[3]) for s in shapes])
    if dims is None or case["kwargs"].get("transpose_k_heads", {}).get("v") is not False:
        return None
    q_dim, kv_dim = dims
    seq = shapes[0][2]

    def heads(chunk, count):
        return chunk.reshape(1, seq, count, shapes[0][3]).permute(0, 2, 1, 3)

    return [
        heads(x[..., :q_dim], shapes[0][1]),
        heads(x[..., q_dim : q_dim + kv_dim], shapes[1][1]),
        heads(x[..., q_dim + kv_dim :], shapes[2][1]),
    ]


def _ref_create_qkv_heads_decode(inputs, kwargs, case):
    """Decode: [1,1,batch,QKV] -> Q [1,batch,heads,hd], K/V [1,batch,kv_heads,hd]."""
    shapes = _out_shapes(case)
    if not shapes or len(shapes[0]) != 4:
        return None
    x = inputs["0"].float()
    batch, n_heads, head_dim = shapes[0][1], shapes[0][2], shapes[0][3]
    n_kv = shapes[1][2]
    q_dim, kv_dim = n_heads * head_dim, n_kv * head_dim
    if x.shape[-1] != q_dim + 2 * kv_dim or x.shape[2] < batch:
        return None
    rows = x[:, :, :batch, :]
    return [
        rows[..., :q_dim].reshape(1, batch, n_heads, head_dim),
        rows[..., q_dim : q_dim + kv_dim].reshape(1, batch, n_kv, head_dim),
        rows[..., q_dim + kv_dim :].reshape(1, batch, n_kv, head_dim),
    ]


def _ref_concat_heads(inputs, kwargs, case):
    """Prefill: [1,heads,seq,hd] -> [1,1,seq,heads*hd]."""
    x = inputs["0"].float()
    if x.dim() != 4:
        return None
    _, n_heads, seq, head_dim = x.shape
    return x.permute(0, 2, 1, 3).reshape(1, 1, seq, n_heads * head_dim)


def _ref_concat_heads_decode(inputs, kwargs, case):
    """Decode: [1,batch,heads,hd] -> [1,1,batch,heads*hd] (batch padded in the output).

    The captured output is padded up to a tile row, so the reference covers only the
    real batch; ``assert_pcc`` compares it against the leading elements, which is
    where row-major puts those users.
    """
    x = inputs["0"].float()
    n_heads = case["kwargs"].get("num_heads", {}).get("v")
    if x.dim() != 4 or n_heads is None or x.shape[2] != n_heads:
        return None
    batch, head_dim = x.shape[1], x.shape[3]
    return x.reshape(1, 1, batch, n_heads * head_dim)


def _ref_layer_norm(inputs, kwargs, case):
    """LayerNorm over the last dim (the vision tower's norm).

    weight/bias arrive tile-padded exactly as rms_norm's gamma does, so they are
    flattened and trimmed to the normalized width the same way.
    """
    x = inputs["0"].float()
    eps = case["kwargs"].get("epsilon", {}).get("v", 1e-5)
    width = x.shape[-1]
    out = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
    weight, bias = inputs.get("weight"), inputs.get("bias")
    if weight is not None:
        out = out * weight.float().reshape(-1)[:width]
    if bias is not None:
        out = out + bias.float().reshape(-1)[:width]
    return out


def _ref_split(inputs, kwargs, case):
    """``ttnn.split(x, split_size, dim=d)`` -> the same chunks torch.split returns."""
    x = inputs["0"].float()
    lits = [a["v"] for a in case["args"][1:] if a["k"] == "lit"]
    dim = case["kwargs"].get("dim", {}).get("v")
    if not lits or dim is None or x.shape[dim] % lits[0]:
        return None
    return list(torch.split(x, lits[0], dim=dim))


def _ref_plus_one(inputs, kwargs, case):
    """Increment, in place. ``skip_negative_entries`` leaves entries below zero alone."""
    x = inputs["0"].float()
    if case["kwargs"].get("skip_negative_entries", {}).get("v"):
        return torch.where(x < 0, x, x + 1.0)
    return x + 1.0


def _ref_expand(inputs, kwargs, case):
    """Broadcast to the requested shape (-1 keeps the input's extent on that dim)."""
    x = inputs["0"].float()
    target = next((a["v"] for a in case["args"][1:] if a["k"] == "lit"), None)
    if target is None:
        return None
    target = list(target) if isinstance(target, (list, tuple)) else [target]
    if len(target) != x.dim():
        return None
    return x.expand(*target).contiguous()


def _ref_zeros_like(inputs, kwargs, case):
    return torch.zeros_like(inputs["0"].float())


def _ref_mesh_partition(inputs, kwargs, case):
    """Identity on the single-device mesh the capture ran on.

    ``ttnn::mesh_partition`` returns its input unchanged when the cluster axis holds
    one device (mesh_partition.cpp:17). On a larger mesh it really partitions, so the
    reference applies only while the captured output keeps the input's shape.
    """
    shapes = _out_shapes(case)
    x = inputs["0"]
    if not shapes or tuple(x.shape) != shapes[0]:
        return None
    return x


# Ops whose output is a deterministic function of the inputs we generated. A
# reference here is worth more than the structural checks: it catches a permutation,
# a mis-split or a wrong reduction that every shape/dtype/placement assertion passes.
#
# Everything absent from this table is checked for shape / dtype / placement /
# finiteness only. What is left out, and why:
#   * rope — the Meta-format cos/sin plus the tile-wise transformation matrix make a
#     torch reference fiddly and easy to get subtly wrong; the hand-written
#     tests/ops/test_rotary_embedding_llama.py:33 came to the same conclusion;
#   * SDPA — the captured page tables/positions describe the KV geometry, but a
#     reference would have to reimplement chunked flash attention;
#   * the paged caches — value semantics are checked by POSTCONDITION instead, which
#     verifies the write landed where the page table says it should;
#   * argmax — random bfloat16 logits tie at the maximum often enough that the index
#     torch picks and the index the op picks are both correct and different;
#   * scatter and pad — the index/pad-value semantics would have to be re-derived, and
#     a reference that is subtly wrong is worse than the structural checks.
GOLDEN = {
    "ttnn.to_memory_config": _ref_identity,
    "ttnn.to_device": _ref_identity,
    "ttnn.sharded_to_interleaved": _ref_identity,
    "ttnn.interleaved_to_sharded": _ref_identity,
    "ttnn.untilize": _ref_identity,
    "ttnn.typecast": _ref_identity,
    "ttnn.linear": _ref_matmul,
    "ttnn.experimental.minimal_matmul": _ref_matmul,
    "ttnn.add": _ref_binary(torch.add),
    "ttnn.multiply": _ref_binary(torch.mul),
    "ttnn.transpose": _ref_transpose,
    "ttnn.concat": _ref_concat,
    "ttnn.slice": _ref_slice,
    "ttnn.Tensor.__getitem__": _ref_getitem,
    "ttnn.rms_norm": _ref_rms_norm,
    "ttnn.embedding": _ref_embedding,
    "ttnn.reshape": _ref_view,
    "ttnn.unsqueeze_to_4D": _ref_view,
    "ttnn.experimental.nlp_create_qkv_heads": _ref_create_qkv_heads,
    "ttnn.experimental.nlp_create_qkv_heads_decode": _ref_create_qkv_heads_decode,
    "ttnn.experimental.nlp_concat_heads": _ref_concat_heads,
    "ttnn.experimental.nlp_concat_heads_decode": _ref_concat_heads_decode,
    "ttnn.to_layout": _ref_identity,
    "ttnn.unsqueeze": _ref_view,
    "ttnn.layer_norm": _ref_layer_norm,
    "ttnn.split": _ref_split,
    "ttnn.plus_one": _ref_plus_one,
    "ttnn.expand": _ref_expand,
    "ttnn.zeros_like": _ref_zeros_like,
    "ttnn.mesh_partition": _ref_mesh_partition,
}


# =============================================================================
# Derived expectations (for outputs the capture never observed)
# =============================================================================

# An op's output spec is recovered only if that tensor is used again later in the
# capture. When it is not (``outs`` entry is None), shape/dtype/layout/placement
# would go unchecked and an op that returned its input untouched would pass on
# finiteness alone — which is precisely what an identity PCC cannot catch either.
# So what the *call itself* pins down is reconstructed instead:
#
#   * a ``memory_config`` argument is where the output must land, by definition;
#   * a layout-changing op fixes the output layout whatever else is unknown;
#   * a relayout/move op preserves the input's shape.
#
# Nothing here is a guess about the op's semantics: each rule is a property of the
# call as written. Anything not derivable stays unchecked (see POSTCONDITION for
# the ops whose real content check needs more than metadata).

_OUTPUT_LAYOUT = {
    "ttnn.untilize": "ROW_MAJOR",
    "ttnn.untilize_with_unpadding": "ROW_MAJOR",
    "ttnn.tilize": "TILE",
    "ttnn.tilize_with_val_padding": "TILE",
}

# Ops that move or relayout data without reshaping it (the identity goldens).
_SHAPE_PRESERVING = frozenset(op for op, ref in GOLDEN.items() if ref is _ref_identity)


def _derived_spec(case, op_name, index):
    """What the call itself says the output must be; {} when it says nothing."""
    spec = {}

    first = case["args"][0] if case["args"] else None
    if index == 0 and op_name in _SHAPE_PRESERVING and first is not None and first["k"] == "t":
        spec["shape"] = first["shape"]

    layout = _OUTPUT_LAYOUT.get(op_name)
    if layout is None:
        layout = next((v["v"] for v in case["kwargs"].values() if v["k"] == "layout"), None)
    if layout is not None:
        spec["layout"] = layout

    dtype = next((v["v"] for v in case["kwargs"].values() if v["k"] == "dtype"), None)
    if dtype is not None:
        spec["dtype"] = dtype

    mem = case["kwargs"].get("memory_config")
    if mem is not None and mem["k"] == "mem":
        spec["mem"] = {"layout": mem["layout"], "buffer": mem["buffer"], "shard": mem.get("shard")}

    return spec


# =============================================================================
# Postconditions (ops whose result metadata alone cannot show they did anything)
# =============================================================================


# Paged KV cache layout, from the op's own tests
# (tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py:449-530
# for the decode update, :641-700 for the prefill fill):
#
#   cache      [max_num_blocks, n_heads, block_size, head_dim]   (physical pages)
#   page_table [num_users, max_num_blocks_per_seq]               virtual -> physical
#   token at position p of user b lives in page_table[b, p // block_size], slot p % block_size
#
# The suite generates both the page table (``_arange_pages``) and the positions
# (``_small_position``), so the destination is known and can be asserted, not just
# the amount written. If a capture ever has a geometry this does not describe, the
# region comes back None and only the count is checked.
def _paged_write_region(torch_inputs, case, op_name, cache_shape):
    """Mask of the cache cells this call must write, or None if not modelled."""
    if len(cache_shape) != 4:
        return None
    n_pages, cache_heads, block_size, head_dim = cache_shape
    update = case["args"][1]["shape"]
    if len(update) != 4 or update[3] != head_dim:
        return None
    mask = torch.zeros(cache_shape, dtype=torch.bool)

    if op_name.endswith("paged_update_cache"):
        page_table = torch_inputs.get("page_table")
        positions = torch_inputs.get("update_idxs_tensor")
        if page_table is None or positions is None:
            return None
        batch, heads = update[1], update[2]
        if heads > cache_heads:
            return None
        page_table = page_table.reshape(batch, -1)
        positions = positions.reshape(-1)
        for user in range(batch):
            pos = int(positions[user])
            block = pos // block_size
            if block >= page_table.shape[1]:
                return None
            page = int(page_table[user, block])
            if not 0 <= page < n_pages:
                return None
            mask[page, :heads, pos % block_size, :] = True
        return mask

    # paged_fill_cache(cache, x, page_table, batch_idx=user): the whole sequence.
    page_table = torch_inputs.get("2")
    if page_table is None:
        return None
    user = case["kwargs"].get("batch_idx", {}).get("v", 0)
    heads, seq = update[1], update[2]
    if heads > cache_heads or page_table.dim() != 2 or user >= page_table.shape[0]:
        return None
    for start in range(0, seq, block_size):
        block = start // block_size
        if block >= page_table.shape[1]:
            return None
        page = int(page_table[user, block])
        if not 0 <= page < n_pages:
            return None
        mask[page, :heads, : min(block_size, seq - start), :] = True
    return mask


def _check_cache_written(args, torch_inputs, case, op_name, mesh_device):
    """A paged-cache op must write what it was given, where the page table says.

    The capture never sees these outputs again, so without this the case passes as
    long as the (unchanged, finite) cache comes back. The tensor being written is
    filled with ``CACHE_SENTINEL`` instead of noise, so the write is visible in the
    cache afterwards — and since the page table and positions are generated here, so
    is the destination: a write to the wrong page or slot shows up as sentinel values
    outside the region, which is what the second assertion is for.
    """
    cache_shape = case["args"][0]["shape"]
    host = from_tt(args[0], mesh_device)
    hits = (host - CACHE_SENTINEL).abs() <= _CACHE_SENTINEL_TOL

    region = _paged_write_region(torch_inputs, case, op_name, cache_shape)
    if region is None or host.numel() != region.numel():
        want = _numel(case["args"][1]["shape"])
        written = int(hits.sum())
        assert written >= want, (
            f"{op_name}: {written} of the {want} written value(s) are in the cache afterwards "
            f"— the op did not write what it was given (cache holds {host.numel()} elements)"
        )
        return

    hits = hits.reshape(cache_shape)
    want = int(region.sum())
    inside = int((hits & region).sum())
    outside = int((hits & ~region).sum())
    assert inside == want, (
        f"{op_name}: {inside} of the {want} cell(s) the page table maps hold the written value "
        f"— the op did not write the region "
        f"[pages/heads/slots the generated page table and positions select] (see _paged_write_region)"
    )
    assert outside == 0, (
        f"{op_name}: {outside} written value(s) landed outside the region the page table maps "
        f"— wrong page or slot (see _paged_write_region for the layout this assumes)"
    )


POSTCONDITION = {
    "ttnn.experimental.paged_update_cache": _check_cache_written,
    "ttnn.experimental.paged_fill_cache": _check_cache_written,
}


# =============================================================================
# Case runner
# =============================================================================


def _input_elements(case):
    """Total elements across all captured input tensors."""
    total = 0
    for spec in list(case["args"]) + list(case["kwargs"].values()):
        if spec.get("k") == "t":
            total += _numel(spec["shape"])
        elif spec.get("k") == "tlist":
            total += sum(_numel(s["shape"]) for s in spec["tensors"])
    return total


def _check_finite(host, case, op_name, index):
    """Assert the region the op can actually define is finite.

    An output with more elements than *all* of its inputs combined cannot be fully
    written by any implementation — decode ops are the common case, where a batch-1
    result lives in a tensor padded to 32 tile rows and the remaining rows keep
    whatever the L1 shard held before. Requiring every element to be finite would
    fail on that untouched padding forever.

    So for those outputs the check narrows to the *leading* elements the inputs can
    account for. Row-major puts the logical data first (user 0's row, the first
    head, …), which is exactly the region the op writes; a NaN/Inf there still
    fails, and checking the leading slice rather than merely counting finite values
    means finite garbage in the padding cannot mask it.
    """
    finite = torch.isfinite(host)
    n_finite, n_total = int(finite.sum()), host.numel()
    if n_finite == n_total:
        return

    budget = _input_elements(case)
    if n_total > budget:
        head = finite.reshape(-1)[:budget]
        bad = int(head.numel() - head.sum())
        if bad == 0:
            return  # only the padding the op cannot define is non-finite
        assert False, (
            f"{op_name}: output[{index}] has {bad} non-finite value(s) in the leading {budget} "
            f"element(s) the op must define ({n_total - n_finite} of {n_total} non-finite overall; "
            f"the tail beyond {budget} is batch/tile padding the op does not write)"
        )

    assert False, (
        f"{op_name}: output[{index}] has {n_total - n_finite} non-finite of {n_total} element(s) "
        f"(inputs account for {budget}, so the whole output should be written)"
    )


def _check_placement(got, spec, mesh_device, op_name, index, source="captured"):
    """Assert the output landed in the expected layout / buffer type / shard.

    Without this a no-op ``to_memory_config`` or ``interleaved_to_sharded`` — one
    that hands back its input untouched — passes on shape and dtype alone, which is
    exactly the regression those cases exist to catch.

    The shard *detail* (grid, per-core shape, orientation) is compared only when the
    expected grid exists on this device: on a smaller device the op derives its own
    grid, and the capture's absolute core coordinates are then not the answer to
    compare against (the memory layout and buffer type still are).
    """
    if "layout" in spec:
        want_layout = LAYOUT[spec["layout"]]
        assert got.layout == want_layout, f"{op_name}: output[{index}] layout {got.layout} != {source} {want_layout}"

    mem = spec.get("mem")
    if mem is None or not ttnn.is_tensor_storage_on_device(got):
        return

    got_mem = got.memory_config()
    want_mem_layout = MEM_LAYOUT[mem["layout"]]
    assert got_mem.memory_layout == want_mem_layout, (
        f"{op_name}: output[{index}] memory layout {got_mem.memory_layout} != {source} {want_mem_layout} "
        f"— the op did not place its output where the model expects it"
    )
    want_buffer = BUFFER_TYPE[mem["buffer"]]
    assert (
        got_mem.buffer_type == want_buffer
    ), f"{op_name}: output[{index}] buffer type {got_mem.buffer_type} != {source} {want_buffer}"

    if mem.get("shard") is None or not _shard_grid_fits(mem, mesh_device)[0]:
        return
    want_mem = build_memory_config(mem, mesh_device)
    assert got_mem == want_mem, f"{op_name}: output[{index}] memory config {got_mem} != {source} {want_mem}"


def _check_output(out, case, mesh_device, op_name):
    """Check every returned tensor against its captured spec, then its values.

    ``case["outs"]`` holds one spec per tensor the op returned in the capture, in
    order — multi-output ops (``nlp_create_qkv_heads``: Q, K, V) get all of theirs
    checked, not just the first. An entry is None when that output was never
    consumed again in the capture, so its spec was never observed; the entry is
    still present, which is what makes the output *count* checkable, and the checks
    fall back to ``_derived_spec`` rather than to nothing.
    """
    expected = case.get("outs") or []
    tensors = list(out) if isinstance(out, (list, tuple)) else [out]

    if expected:
        assert len(tensors) == len(
            expected
        ), f"{op_name}: op returned {len(tensors)} tensor(s), capture recorded {len(expected)}"

    for i, got in enumerate(tensors):
        spec = expected[i] if i < len(expected) else None
        source = "captured"
        if spec is None:
            # Fall back to what the call itself pins down, so an unobserved output
            # is still held to its memory config / layout / shape (see _derived_spec).
            spec, source = _derived_spec(case, op_name, i), "implied by the call"
        if not spec:
            continue

        if "shape" in spec:
            want_shape = tuple(spec["shape"])
            assert (
                tuple(got.shape) == want_shape
            ), f"{op_name}: output[{i}] shape {tuple(got.shape)} != {source} {want_shape}"
        if "dtype" in spec:
            want_dtype = DTYPE[spec["dtype"]]
            assert got.dtype == want_dtype, f"{op_name}: output[{i}] dtype {got.dtype} != {source} {want_dtype}"
        _check_placement(got, spec, mesh_device, op_name, i, source)

    for i, t in enumerate(tensors):
        _check_finite(from_tt(t, mesh_device), case, op_name, i)


def _golden_pcc(case, op_name):
    """Floor for this case: 0.999 unless the call itself costs precision.

    Every reason to go below the bar is spelled out where the tables are defined
    (``_PCC_BY_DTYPE`` / ``_PCC_BY_OP`` / ``_DEEP_K``); the lowest applicable floor
    wins, so a bfloat8_b matmul is judged on its dtype rather than its op.
    """
    floors = [_DEFAULT_PCC, _PCC_BY_OP.get(op_name, _DEFAULT_PCC)]

    dtypes = [out["dtype"] for out in case.get("outs") or [] if out is not None]
    dtypes += [a["dtype"] for a in case["args"] if a["k"] == "t"]
    floors += [_PCC_BY_DTYPE.get(d, _DEFAULT_PCC) for d in dtypes]

    if op_name in _REDUCTION_OPS and case["args"] and case["args"][0]["k"] == "t":
        shape = case["args"][0]["shape"]
        if shape and shape[-1] >= _DEEP_K:
            floors.append(_DEEP_K_PCC)

    return min(floors)


def run_case(op, case, mesh_device, *, op_name=None, pcc=None):
    """Materialize one captured call, run it, and check the result.

    ``op`` is the callable (``ttnn.linear``, or a small lambda for operators like
    ``Tensor.__getitem__``); ``case`` is one entry of a generated ``CASES`` list.
    """
    op_name = op_name or case["op"]
    torch_inputs: dict[str, torch.Tensor] = {}

    args = [_build_value(spec, mesh_device, case, op_name, str(i), torch_inputs) for i, spec in enumerate(case["args"])]
    kwargs = {
        name: _build_value(spec, mesh_device, case, op_name, name, torch_inputs)
        for name, spec in case["kwargs"].items()
    }

    out = op(*args, **kwargs)

    _check_output(out, case, mesh_device, op_name)

    post_fn = POSTCONDITION.get(op_name)
    if post_fn is not None:
        post_fn(args, torch_inputs, case, op_name, mesh_device)

    ref_fn = None if NO_GOLDEN else GOLDEN.get(op_name)
    if ref_fn is not None:
        ref = ref_fn(torch_inputs, kwargs, case)
        if ref is not None:
            # A reference may cover every output (the qkv splits return Q, K and V),
            # so compare position by position rather than only the first tensor.
            refs = list(ref) if isinstance(ref, (list, tuple)) else [ref]
            tensors = list(out) if isinstance(out, (list, tuple)) else [out]
            floor = pcc or _golden_pcc(case, op_name)
            for i, one in enumerate(refs):
                if one is not None and i < len(tensors):
                    U.assert_pcc(one, tensors[i], pcc=floor, mesh_device=mesh_device)

    return out
