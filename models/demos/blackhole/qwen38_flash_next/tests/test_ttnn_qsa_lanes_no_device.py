# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The batched-lanes QSA layer without a device: B lanes at their own positions bitwise the B 1-row steps.

The whole decode QSA layer (``forward_decode_lanes``) and the 1-row generic body (``forward_decode_generic``) run on
a torch-backed ttnn whose every op is deterministic per row (float64 arithmetic rounded once to the op's dtype, so no
result depends on the tensor shape it sits in).  For B in 1, 2, 4, 8 lanes at positions spread over 2k..32k (one
residue class) the lane body's row u of every kept intermediate, its output row, lane u's KV block, compressed cache,
staging tile and ring are compared bit for bit with the 1-row body run at P_u on the same warm caches, over two
consecutive steps (the second after the in-trace advance), for both indexer forms.  The lane state contract, the
residue rule at the position row and the source pins (the 1-row and chunk bodies untouched, the lane body free of
host I/O and per-call ints) are checked alongside.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen36.tt.attention import rope_tp as rope_module
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as decode_matmul_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    MAX_LANES,
    Qwen38TTNNDevicePosition,
    Qwen38TTNNDevicePositionRow,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    CACHE_WRITE_ROWS,
    COMPRESS_RATIO,
    HEAD_DIM,
    HIDDEN_SIZE,
    INDEX_HEAD_DIM,
    INDEXER_FORMS,
    LOCAL_QUERY_WIDTH,
    MASKED_INDEX,
    ROPE_DIM,
    Qwen38TTNNQSA,
    Qwen38TTNNQSAChunkConstants,
    Qwen38TTNNQSALaneConstants,
    Qwen38TTNNQSALaneState,
    Qwen38TTNNQSAPositionConstants,
    Qwen38TTNNQSAWeights,
    derive_qsa_lane_inputs,
    derive_qsa_position_inputs,
)

TESTS = Path(__file__).resolve().parent
TP = 4
ALLOCATED_CONTEXT = 32768
BLOCKS = ALLOCATED_CONTEXT // COMPRESS_RATIO
# One lane per 4k span of the 32k cache, all in residue class 3 (the block-closing compressed write is real), each in
# its own 32-row block (the QSA lane micro-test's positions).
LANE_POSITIONS = (2051, 6147, 10243, 14339, 18435, 22531, 26627, 30723)
LANE_COUNTS = (1, 2, 4, 8)
STEPS = 2
LAYER = 3


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TESTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step4 = _load("test_mtp_v2_step4_rows_no_device")
BF16, FP32, TILE, ROW_MAJOR = step4.BF16, step4.FP32, step4.TILE, step4.ROW_MAJOR
FakeContract, FakeTensor = step4.FakeContract, step4.FakeTensor
U32 = step4._DType("uint32", torch.int64)  # UINT32 values as int64 in [0, 2**32)
I32 = step4._DType("int32", torch.int32)
U32_MASK = 0xFFFFFFFF


# --------------------------------------------------------------------------- the QSA fake


def _padded_shape(self) -> tuple[int, ...]:
    shape = self.shape
    if self.layout != TILE or len(shape) < 2:
        return shape
    return (*shape[:-2], -(-shape[-2] // 32) * 32, -(-shape[-1] // 32) * 32)


FakeTensor.padded_shape = property(_padded_shape)
# The QSA constructor validates the DRAM bank layout of its query-gate and output weights against the layout its
# decode linears run on (decode_matmul.validate_dram_sharded_weight, two readers per bank on the served head): a fake
# weight carries the memory config it was uploaded with and its device members report it.
FakeTensor.memory_config = lambda self: getattr(self, "memory_config_value", "DRAM_MEMORY_CONFIG")


def _get_device_tensors(tensor):
    members = step4._get_device_tensors(tensor)
    for member in members:
        member.memory_config_value = getattr(tensor, "memory_config_value", "DRAM_MEMORY_CONFIG")
    return members


def _flat_topology() -> step4.TensorTopology:
    """The legacy flat mesh ``ttnn.zeros`` reports (what ``_canonicalize_flat_replicated_topology`` accepts)."""

    return step4.TensorTopology((TP,), [step4.PlacementReplicate()], [(0, column) for column in range(TP)])


def _host(host: torch.Tensor, dtype) -> torch.Tensor:
    return host.to(torch.int64) & U32_MASK if dtype is U32 else host.to(dtype.torch)


def _from_torch(host, *, dtype, layout, device=None, memory_config=None, mesh_mapper):
    host = _host(host, dtype)
    if mesh_mapper == "replicate":
        return FakeTensor([host.clone() for _ in range(TP)], dtype, layout)
    kind, dims = mesh_mapper
    assert kind == "shard"
    return FakeTensor([piece.clone() for piece in torch.chunk(host, TP, dim=dims[1])], dtype, layout, dims[1])


def _zeros(shape, *, dtype, layout, device, memory_config):
    tensor = FakeTensor([torch.zeros(tuple(shape), dtype=dtype.torch) for _ in range(TP)], dtype, layout)
    tensor.topology = _flat_topology()
    return tensor


def _operand(b, index):
    return b.torch_shards()[index] if isinstance(b, FakeTensor) else b


def _round(value: torch.Tensor, dtype) -> torch.Tensor:
    return (value & U32_MASK) if dtype is U32 else value.to(dtype.torch)


def _binary(torch_op):
    def op(a, b, *, memory_config=None, output_tensor=None, dtype=None, fast_and_approximate_mode=None):
        out_dtype = dtype or a.dtype
        results = []
        for index, x in enumerate(a.torch_shards()):
            y = _operand(b, index)
            if a.dtype is U32:
                result = torch_op(x, y)
            else:
                y64 = y.double() if isinstance(y, torch.Tensor) else y
                result = torch_op(x.double(), y64)
                if torch_op is torch.mul:
                    # binary_ng forces the product to +0.0 whenever an input is zero (the device's rule).
                    result = torch.where((x.double() == 0) | (y64 == 0), torch.zeros_like(result), result)
            results.append(_round(result, out_dtype))
        if output_tensor is not None:
            assert output_tensor.dtype is out_dtype and output_tensor.shape == tuple(results[0].shape), (
                output_tensor.shape,
                tuple(results[0].shape),
            )
            for target, result in zip(output_tensor.locals, results):
                target.copy_(result)
            return output_tensor
        return FakeTensor(results, out_dtype, a.layout)

    return op


def _compare(torch_op):
    def op(a, b, *, dtype, memory_config=None):
        assert dtype is U32
        return FakeTensor(
            [torch_op(x, _operand(b, index)).to(torch.int64) for index, x in enumerate(a.torch_shards())], U32, a.layout
        )

    return op


def _shift(torch_op):
    return lambda a, count, memory_config=None: FakeTensor(
        [torch_op(x, count) & U32_MASK for x in a.torch_shards()], U32, a.layout
    )


def _mapped(fn):
    def op(tensor, *args, memory_config=None, **kwargs):
        return FakeTensor([fn(t, *args, **kwargs) for t in tensor.torch_shards()], tensor.dtype, tensor.layout)

    return op


def _rowwise64(fn):
    """Apply ``fn`` to every last-dim row in float64 and round once to the tensor's dtype."""

    def op(tensor, *args, memory_config=None, **kwargs):
        results = []
        for x in tensor.torch_shards():
            rows = x.reshape(-1, x.shape[-1]).double()
            results.append(torch.stack([fn(row, *args, **kwargs) for row in rows]).reshape(x.shape).to(x.dtype))
        return FakeTensor(results, tensor.dtype, tensor.layout)

    return op


def _typecast(t, dtype, memory_config=None):
    if dtype is I32:
        assert all(int(x.max()) < 2**31 for x in t.torch_shards())
    return FakeTensor([_round(x if dtype is U32 else x.double(), dtype) for x in t.torch_shards()], dtype, t.layout)


def _reshape(t, shape, padded_shape=None, memory_config=None):
    """A same-volume reshape, or the padded tile view: a smaller logical shape reads the leading rows, a larger one
    (the 1-row query as its 32-row tile) reads the tile's zero padding past row 0."""

    shape = tuple(shape)
    results = []
    for x in t.torch_shards():
        if x.numel() == torch.Size(shape).numel():
            results.append(x.reshape(shape).clone())
        else:
            assert padded_shape is not None and shape[:2] == tuple(x.shape[:2]) and shape[3] == x.shape[3]
            if shape[2] < x.shape[2]:
                results.append(x[:, :, : shape[2]].clone())
            else:
                results.append(F.pad(x, (0, 0, 0, shape[2] - x.shape[2])))
    return FakeTensor(results, t.dtype, t.layout)


def _view(t, shape):
    """``ttnn.experimental.view``: the same buffer under a new shape.  ROW_MAJOR pages and whole-tile-row TILE
    regroupings keep the torch element order, so the result aliases the input (a torch view); any other TILE view
    re-reads the tile pages in the new grid (a copy, caught by the aliasing tests if the body relied on it)."""

    shape = tuple(shape)
    aligned = t.layout == ROW_MAJOR or (
        shape[-1] == t.shape[-1] and shape[-2] % 32 == 0 and t.shape[-2] % 32 == 0 and len(shape) == 4
    )
    if aligned:
        return FakeTensor([x.reshape(shape) for x in t.torch_shards()], t.dtype, t.layout)
    return FakeTensor(
        [step4._from_tile_pages(step4._tile_pages(x), shape) for x in t.torch_shards()], t.dtype, t.layout
    )


def _linear(a, w, *, memory_config=None, program_config=None, compute_kernel_config=None):
    return FakeTensor(
        [
            (x.double() @ wt.reshape(wt.shape[-2], wt.shape[-1]).double()).to(x.dtype)
            for x, wt in zip(a.torch_shards(), w.torch_shards())
        ],
        a.dtype,
        a.layout,
    )


def _matmul(a, b, *, memory_config=None, compute_kernel_config=None, optional_output_tensor=None):
    results = [torch.matmul(x.double(), y.double()).to(x.dtype) for x, y in zip(a.torch_shards(), b.torch_shards())]
    if optional_output_tensor is not None:
        for target, result in zip(optional_output_tensor.locals, results):
            target.copy_(result)
        return optional_output_tensor
    return FakeTensor(results, a.dtype, a.layout)


def _sum(t, dim, keepdim, *, memory_config=None, compute_kernel_config=None, scalar=1.0):
    assert keepdim and dim in (2, 3)
    return FakeTensor(
        [(x.double().sum(dim=dim, keepdim=True) * scalar).to(x.dtype) for x in t.torch_shards()], t.dtype, t.layout
    )


def _rms_norm(t, *, epsilon, weight, memory_config=None, compute_kernel_config=None):
    results = []
    for x, w in zip(t.torch_shards(), weight.torch_shards()):
        x64 = x.double()
        y = x64 * torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + epsilon) * w.reshape(-1).double()
        results.append(y.to(x.dtype))
    return FakeTensor(results, t.dtype, t.layout)


def _rotary_embedding_hf(x, cos, sin, *, is_decode_mode, memory_config=None):
    results = []
    for xs, cs, ss in zip(x.torch_shards(), cos.torch_shards(), sin.torch_shards()):
        x64, c64, s64 = xs.double(), cs.double(), ss.double()
        half = x64.shape[-1] // 2
        rotated = torch.cat([-x64[..., half:], x64[..., :half]], dim=-1)
        results.append((x64 * c64 + rotated * s64).to(xs.dtype))
    return FakeTensor(results, x.dtype, x.layout)


def _all_gather(t, *, dim, cluster_axis, memory_config=None):
    full = torch.cat(t.torch_shards(), dim=dim)
    return FakeTensor([full.clone() for _ in range(TP)], t.dtype, t.layout)


def _all_reduce(t, *, cluster_axis, memory_config=None, topology=None):
    total = sum(x.double() for x in t.torch_shards()).to(t.dtype.torch)
    return FakeTensor([total.clone() for _ in range(TP)], t.dtype, t.layout)


def _reduce_scatter(t, *, dim, cluster_axis, memory_config=None, topology=None):
    total = sum(x.double() for x in t.torch_shards()).to(t.dtype.torch)
    return FakeTensor([piece.clone() for piece in torch.chunk(total, TP, dim=dim)], t.dtype, t.layout, dim)


def _indexer_score_dsa(
    q, k, gate, *, chunk_start_idx, compute_kernel_config, seq_shard_axes, cache_batch_idx=None, kv_len=None
):
    """Per head ReLU(q . k) times the gate, summed over the local heads, causal past ``chunk_start_idx + s``."""

    results = []
    for qs, ks, gs in zip(q.torch_shards(), k.torch_shards(), gate.torch_shards()):
        keys = ks[0 if cache_batch_idx is None else cache_batch_idx, 0].double()  # [T, D]
        scores = torch.zeros(qs.shape[2], keys.shape[0], dtype=torch.float64)
        for head in range(qs.shape[1]):
            scores += torch.relu(qs[0, head].double() @ keys.T) * gs[0, head].double()
        visible = torch.arange(keys.shape[0])[None, :] <= chunk_start_idx + torch.arange(qs.shape[2])[:, None]
        results.append(torch.where(visible, scores, torch.zeros_like(scores)).to(qs.dtype).reshape(1, 1, *scores.shape))
    return FakeTensor(results, q.dtype, ROW_MAJOR)


def _topk_large_indices(scores, *, k):
    return FakeTensor(
        [
            torch.topk(x.float(), k, dim=-1, largest=True, sorted=True).indices.to(torch.int64)
            for x in scores.torch_shards()
        ],
        U32,
        ROW_MAJOR,
    )


def _sparse_sdpa(q, kv, indices, v_dim, *, kv_format, scale, k_chunk_size, compute_kernel_config):
    """Per head and query row: softmax(q . kv[idx] * scale) over the valid ids times kv[idx, :v_dim]."""

    results = []
    for qs, kvs, ids in zip(q.torch_shards(), kv.torch_shards(), indices.torch_shards()):
        heads, rows = qs.shape[1], qs.shape[2]
        out = torch.zeros(1, heads, rows, v_dim, dtype=torch.float64)
        cache = kvs[0, 0].double()
        for row in range(rows):
            valid = ids[0, 0, row]
            valid = valid[valid != MASKED_INDEX]
            if valid.numel() == 0:
                continue
            assert int(valid.max()) < cache.shape[0], (int(valid.max()), cache.shape[0])
            gathered = cache[valid]  # [n, K_DIM]
            for head in range(heads):
                query = qs[0, head, row].double()
                if not torch.any(query != 0):
                    continue
                weights = torch.softmax((gathered @ query) * scale, dim=0)
                out[0, head, row] = weights @ gathered[:, :v_dim]
        results.append(out.to(qs.dtype))
    return FakeTensor(results, q.dtype, ROW_MAJOR)


def _paged_update_cache(cache, rows, *, update_idxs_tensor):
    for cs, rs, ids in zip(cache.locals, rows.torch_shards(), update_idxs_tensor.torch_shards()):
        for user, index in enumerate(ids.tolist()):
            cs[user, 0, index] = rs[0, user, 0]
    return cache


def _update_padded_kv_cache(cache, slab, slot, start, layer_idx, num_layers, cluster_axis):
    for cs, ss, st in zip(cache.locals, slab.torch_shards(), start.torch_shards()):
        begin = int(st.reshape(-1)[0])
        assert begin % 32 == 0 and begin + ss.shape[2] <= cs.shape[2], (begin, cs.shape)
        cs[0, 0, begin : begin + ss.shape[2]] = ss[0, 0]
    return cache


def _embedding(indices, table, *, layout, dtype, memory_config=None):
    assert indices.dtype is U32 and indices.shape == (1, 1, 32)
    return FakeTensor(
        [
            row_table[0, 0][index.reshape(-1)].reshape(1, 32, -1).clone()
            for index, row_table in zip(indices.torch_shards(), table.torch_shards())
        ],
        dtype,
        layout,
    )


def _fill(tensor, value, *, output_tensor):
    assert output_tensor is tensor
    for x in tensor.locals:
        x.fill_(value)
    return tensor


def make_qsa_fake() -> SimpleNamespace:
    experimental = SimpleNamespace(
        view=_view,
        rotary_embedding_hf=_rotary_embedding_hf,
        indexer_score_dsa=_indexer_score_dsa,
        topk_large_indices=_topk_large_indices,
        paged_update_cache=_paged_update_cache,
        deepseek_prefill=SimpleNamespace(update_padded_kv_cache=_update_padded_kv_cache),
    )
    return SimpleNamespace(
        uint32=U32,
        int32=I32,
        bfloat16=BF16,
        float32=FP32,
        TILE_LAYOUT=TILE,
        ROW_MAJOR_LAYOUT=ROW_MAJOR,
        TILE_SIZE=32,
        DRAM_MEMORY_CONFIG="DRAM_MEMORY_CONFIG",
        L1_MEMORY_CONFIG="L1",
        L1_WIDTH_SHARDED_MEMORY_CONFIG="L1WS",
        Topology=SimpleNamespace(Linear="linear"),
        MathFidelity=SimpleNamespace(HiFi4="HiFi4"),
        ShardStrategy=SimpleNamespace(HEIGHT="HEIGHT", WIDTH="WIDTH"),
        ShardOrientation=SimpleNamespace(ROW_MAJOR="ROW_MAJOR"),
        Shape=lambda values: tuple(values),
        MeshShape=lambda *values: tuple(values),
        CoreGrid=lambda *, x, y: ("grid", x, y),
        CoreCoord=lambda x, y: ("core", x, y),
        CoreRange=lambda start, end: ("range", start, end),
        CoreRangeSet=lambda ranges: ("ranges", tuple(sorted(ranges))),
        ShardSpec=lambda grid, shape, orientation: ("shard_spec", grid, tuple(shape), orientation),
        MemoryConfig=lambda layout, buffer_type, shard_spec=None: ("memory_config", layout, buffer_type, shard_spec),
        TensorMemoryLayout=SimpleNamespace(
            WIDTH_SHARDED="WIDTH_SHARDED", HEIGHT_SHARDED="HEIGHT_SHARDED", INTERLEAVED="INTERLEAVED"
        ),
        BufferType=SimpleNamespace(DRAM="DRAM", L1="L1"),
        num_cores_to_corerangeset=lambda cores, grid, row_wise: ("cores", cores, grid),
        create_sharded_memory_config=lambda shape, grid, strategy, orientation, use_height_and_width_as_shard_shape: (
            "sharded",
            tuple(shape),
            grid,
            strategy,
        ),
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig=lambda **fields: ("dram_sharded", tuple(sorted(fields))),
        init_device_compute_kernel_config=lambda arch, **fields: ("compute", arch, tuple(sorted(fields.items()))),
        TensorTopology=step4.TensorTopology,
        PlacementReplicate=step4.PlacementReplicate,
        PlacementShard=step4.PlacementShard,
        ShardTensor2dMesh=lambda device, mesh_shape, dims: ("shard", dims),
        experimental=experimental,
        transformer=SimpleNamespace(sparse_sdpa=_sparse_sdpa, SparseKVFormat=SimpleNamespace(BF16="BF16")),
        from_torch=_from_torch,
        zeros=_zeros,
        fill=_fill,
        copy=step4._copy,
        copy_host_to_device_tensor=step4._copy,
        deallocate=step4._deallocate,
        get_device_tensors=_get_device_tensors,
        to_torch=lambda t, mesh_composer=None: t.torch_shards()[0].clone(),
        to_layout=lambda t, layout, memory_config=None: FakeTensor(
            [x.clone() for x in t.torch_shards()], t.dtype, layout
        ),
        to_memory_config=lambda t, cfg: FakeTensor([x.clone() for x in t.torch_shards()], t.dtype, t.layout),
        reshape=_reshape,
        unsqueeze_to_4D=lambda t: FakeTensor([x.reshape(1, *x.shape) for x in t.torch_shards()], t.dtype, t.layout),
        slice=step4._slice,
        pad=step4._pad,
        concat=step4._concat,
        repeat=_mapped(lambda x, multipliers: x.repeat(*multipliers)),
        repeat_interleave=_mapped(lambda x, repeats, dim: torch.repeat_interleave(x, repeats, dim=dim)),
        typecast=_typecast,
        add=_binary(torch.add),
        subtract=_binary(torch.sub),
        multiply=_binary(torch.mul),
        mul=_binary(torch.mul),
        minimum=lambda a, b, memory_config=None: FakeTensor(
            [torch.clamp(x, max=b) for x in a.torch_shards()], a.dtype, a.layout
        ),
        rsub=lambda t, value, memory_config=None: FakeTensor(
            [(value - x.double()).to(x.dtype) for x in t.torch_shards()], t.dtype, t.layout
        ),
        bitwise_and=_binary(torch.bitwise_and),
        bitwise_or=_binary(torch.bitwise_or),
        bitwise_left_shift=_shift(torch.bitwise_left_shift),
        bitwise_right_shift=_shift(torch.bitwise_right_shift),
        eq=_compare(torch.eq),
        lt=_compare(torch.lt),
        ge=_compare(torch.ge),
        sum=_sum,
        rms_norm=_rms_norm,
        sigmoid=_rowwise64(torch.sigmoid),
        linear=_linear,
        matmul=_matmul,
        all_gather=_all_gather,
        all_reduce=_all_reduce,
        reduce_scatter=_reduce_scatter,
        embedding=_embedding,
    )


class FakeMesh:
    def arch(self):
        return "BLACKHOLE"

    def dram_grid_size(self):
        return SimpleNamespace(x=8, y=1)


@pytest.fixture
def fake(monkeypatch):
    # The six QSA programs serve by default and bind the fused lane body at construction; the fake ttnn runs the
    # chains only, so this fixture switches them off (the 1-row helpers under test are the chain's regardless).
    monkeypatch.setenv(
        "QWEN38_FUSED_OFF",
        "qsa_index_tail,qsa_main_tail,qsa_post_attention,qsa_widen_partial,qsa_selection_row,qsa_score_merge",
    )
    fake = make_qsa_fake()
    for module in (qsa_module, contracts_module, model_module, rope_module, decode_matmul_module):
        monkeypatch.setattr(module, "ttnn", fake)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    # the fake's bf16 is the dense plan's bf16 for the projection weights (dense_dtype_tag / the fidelity name)
    monkeypatch.setitem(decode_matmul_module.DENSE_DTYPE_TAGS, BF16, "bf16")
    monkeypatch.setitem(decode_matmul_module.DENSE_MATH_FIDELITY_NAMES, BF16, "HiFi4")
    return fake


# --------------------------------------------------------------------------- the layer under the fake


def _replicated(host: torch.Tensor, layout=TILE) -> FakeTensor:
    return FakeTensor([host.to(torch.bfloat16).clone() for _ in range(TP)], BF16, layout)


def _sharded(host: torch.Tensor, dim: int, layout=TILE) -> FakeTensor:
    return FakeTensor([piece.to(torch.bfloat16).clone() for piece in torch.chunk(host, TP, dim=dim)], BF16, layout, dim)


def _dram_sharded_weight(host: torch.Tensor, dim: int, k: int, n: int) -> FakeTensor:
    """A decode linear's weight in the one-reader DRAM bank layout the module validates (decode_matmul)."""

    tensor = _sharded(host, dim)
    tensor.memory_config_value = decode_matmul_module.dram_sharded_weight_memory_config(FakeMesh(), k, n)
    return tensor


def _weights(generator) -> Qwen38TTNNQSAWeights:
    def randn(*shape, scale):
        return torch.randn(*shape, generator=generator) * scale

    return Qwen38TTNNQSAWeights(
        source_kind="backbone",
        layer_index=LAYER,
        qg=_dram_sharded_weight(
            randn(1, 1, HIDDEN_SIZE, TP * 2 * LOCAL_QUERY_WIDTH, scale=0.02), 3, HIDDEN_SIZE, 2 * LOCAL_QUERY_WIDTH
        ),
        k_pair_grouped=_sharded(randn(1, 1, HIDDEN_SIZE, TP * HEAD_DIM, scale=0.02), 3),
        v_pair_grouped=_sharded(randn(1, 1, HIDDEN_SIZE, TP * HEAD_DIM, scale=0.02), 3),
        out=_dram_sharded_weight(
            randn(1, 1, TP * LOCAL_QUERY_WIDTH, HIDDEN_SIZE, scale=0.02), 2, LOCAL_QUERY_WIDTH, HIDDEN_SIZE
        ),
        q_norm=_replicated(1.0 + randn(1, 1, 1, HEAD_DIM, scale=0.1)),
        k_norm=_replicated(1.0 + randn(1, 1, 1, HEAD_DIM, scale=0.1)),
        index_q=_sharded(randn(1, 1, HIDDEN_SIZE, TP * INDEX_HEAD_DIM, scale=0.02), 3),
        index_k=_replicated(randn(1, 1, HIDDEN_SIZE, INDEX_HEAD_DIM, scale=0.02)),
        index_q_norm=_replicated(1.0 + randn(1, 1, 1, INDEX_HEAD_DIM, scale=0.1)),
        index_k_norm=_replicated(1.0 + randn(1, 1, 1, INDEX_HEAD_DIM, scale=0.1)),
        weight_dtype=BF16,
    )


class _Layer:
    """The QSA layer, its RoPE table, the position constants and the warm cache images, all on the fake."""

    def __init__(self, seed: int = 20260903) -> None:
        self.generator = torch.Generator().manual_seed(seed)
        self.contract = FakeContract()
        self.mesh = FakeMesh()
        self.qsa = Qwen38TTNNQSA(
            self.mesh, self.contract, _weights(self.generator), layer_index=LAYER, allocated_context=ALLOCATED_CONTEXT
        )
        cos, sin = (torch.randn(1, 1, ALLOCATED_CONTEXT, ROPE_DIM, generator=self.generator) for _ in range(2))
        self.rope_table = model_module.Qwen38TTNNRoPETable(
            _replicated(cos, ROW_MAJOR), _replicated(sin, ROW_MAJOR), ALLOCATED_CONTEXT, None, self.contract
        )
        self.position_constants = Qwen38TTNNQSAPositionConstants.build(self.mesh, self.contract, BLOCKS)
        self.chunk_constants = Qwen38TTNNQSAChunkConstants.build(self.mesh, self.contract, BLOCKS)
        self.kv_host = torch.randn(1, TP, ALLOCATED_CONTEXT, 2 * HEAD_DIM, generator=self.generator).to(torch.bfloat16)
        self.compressed_host = torch.cat(
            [
                torch.randn(1, 1, BLOCKS, INDEX_HEAD_DIM, generator=self.generator).to(torch.bfloat16),
                torch.zeros(1, 1, 32, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            ],
            dim=2,
        )

    def hidden(self, rows: int) -> torch.Tensor:
        return torch.randn(1, 1, rows, HIDDEN_SIZE, generator=self.generator).to(torch.bfloat16)

    def write_generic(self, state) -> None:
        for host, target in (
            (self.kv_host, state.packed_kv_cache),
            (self.compressed_host.expand(1, TP, -1, -1), state.compressed_index_cache),
            (torch.zeros(1, TP, CACHE_WRITE_ROWS, 2 * HEAD_DIM), state.kv_staging),
            (torch.zeros(1, TP, CACHE_WRITE_ROWS, INDEX_HEAD_DIM), state.raw_key_ring),
        ):
            step4._copy(_sharded(host, 1), target)

    def write_lanes(self, state: Qwen38TTNNQSALaneState) -> None:
        lanes = state.lanes
        for host, target in (
            (self.kv_host.repeat(1, 1, lanes, 1), state.packed_kv_cache),
            (self.compressed_host.repeat(lanes, TP, 1, 1), state.compressed_index_cache),
            (torch.zeros(1, TP * lanes, CACHE_WRITE_ROWS, 2 * HEAD_DIM), state.kv_staging),
            (torch.zeros(1, TP * lanes, CACHE_WRITE_ROWS, INDEX_HEAD_DIM), state.raw_key_ring),
        ):
            step4._copy(_sharded(host, 1), target)


def _shards(tensor: FakeTensor) -> list[torch.Tensor]:
    return [x.clone() for x in tensor.torch_shards()]


def _one_row_step(layer: _Layer, state, hidden_row: torch.Tensor, position: int) -> dict[str, list[torch.Tensor]]:
    """forward_decode_generic at ``position`` stage by stage; every intermediate and the caches after it."""

    qsa = layer.qsa
    scalar = Qwen38TTNNDevicePosition.allocate(layer.mesh, layer.contract, position=position)
    inputs = derive_qsa_position_inputs(scalar.scalar, layer.position_constants)
    index_row = scalar.index_row()
    rope = layer.rope_table.rows(index_row, scalar.block_start_index_row(index_row))
    hidden = _sharded(hidden_row, 3)
    fields: dict[str, list[torch.Tensor]] = {}
    full_hidden = qsa._all_gather_hidden(hidden)
    index_query, raw_key = qsa._index_projection(full_hidden, rope.cos, rope.sin)
    fields["index_query"], fields["raw_key"] = _shards(index_query), _shards(raw_key)
    qsa._write_compressed_index_generic(state, raw_key, inputs, rope.block_start_cos, rope.block_start_sin)
    masked_scores = qsa._score_blocks_generic(index_query, state, inputs)
    fields["masked_scores"] = _shards(masked_scores)
    sparse_indices = qsa._materialize_row_generic(masked_scores, inputs)
    fields["sparse_indices"] = _shards(sparse_indices)
    query, gate, key, value = qsa._main_projection(full_hidden, rope.cos, rope.sin)
    for name, tensor in (("q", query), ("gate", gate), ("k", key), ("v", value)):
        fields[name] = _shards(tensor)
    qsa._write_packed_kv_generic(state, key, value, inputs)
    local_attention = qsa._sparse_value_attention(query, gate, sparse_indices, state)
    fields["local_attention"] = _shards(local_attention)
    fields["output"] = _shards(qsa._project_output(local_attention, full_hidden))
    start = position & ~(CACHE_WRITE_ROWS - 1)
    fields["kv_block"] = [x[:, :, start : start + CACHE_WRITE_ROWS] for x in _shards(state.packed_kv_cache)]
    fields["compressed_cache"] = _shards(state.compressed_index_cache)
    fields["staging"], fields["ring"] = _shards(state.kv_staging), _shards(state.raw_key_ring)
    return fields


def _lane_step(
    layer: _Layer, state: Qwen38TTNNQSALaneState, lane_constants, row: Qwen38TTNNDevicePositionRow, hidden_rows
) -> dict[str, list[torch.Tensor]]:
    """forward_decode_lanes stage by stage (the body's own order); every intermediate and the caches after it."""

    qsa = layer.qsa
    inputs = derive_qsa_lane_inputs(row.row, layer.position_constants, layer.chunk_constants, lane_constants)
    index_row = row.index_row()
    rope = layer.rope_table.rows_chunk(index_row, row.block_start_index_row(index_row))
    hidden = _sharded(hidden_rows, 3)
    fields: dict[str, list[torch.Tensor]] = {}
    full_hidden = qsa._all_gather_hidden_rows(hidden, layer.chunk_constants)
    index_query, raw_key = qsa._index_projection_rows(full_hidden, None, rope.cos, rope.sin, layer.chunk_constants)
    fields["index_query"], fields["raw_key"] = _shards(index_query), _shards(raw_key)
    qsa._write_compressed_index_lanes(
        state, raw_key, rope.block_start_cos, rope.block_start_sin, inputs, lane_constants
    )
    masked_scores = qsa._score_blocks_lanes(index_query, state, inputs, lane_constants)
    fields["masked_scores"] = _shards(masked_scores)
    sparse_indices = qsa._materialize_rows_lanes(masked_scores, inputs, layer.chunk_constants, lane_constants)
    fields["sparse_indices"] = _shards(sparse_indices)
    query, gate, key, value = qsa._main_projection_rows(full_hidden, None, rope.cos, rope.sin, layer.chunk_constants)
    for name, tensor in (("q", query), ("gate", gate), ("k", key), ("v", value)):
        fields[name] = _shards(tensor)
    qsa._write_packed_kv_lanes(state, key, value, inputs)
    local_attention = qsa._sparse_value_attention_rows(query, gate, sparse_indices, state, layer.chunk_constants)
    fields["local_attention"] = _shards(local_attention)
    fields["output"] = _shards(qsa._project_output_rows(local_attention, full_hidden, layer.chunk_constants))
    # The caches themselves (read before the next step mutates them): no copies of the B x 32k regions.
    fields["kv_cache"] = list(state.packed_kv_cache.torch_shards())
    fields["compressed_cache"] = list(state.compressed_index_cache.torch_shards())
    fields["staging"], fields["ring"] = list(state.kv_staging.torch_shards()), list(state.raw_key_ring.torch_shards())
    inputs.deallocate()
    return fields


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype == torch.bfloat16:
        return torch.equal(left.view(torch.int16), right.view(torch.int16))
    return torch.equal(left, right)


def _lane_view(name: str, shard: torch.Tensor, lane: int, position: int) -> torch.Tensor:
    """Lane u's 1-row image out of a lane-body field: row u of a rows tensor (the sparse ids with lane u's KV offset
    taken back off every non-sentinel slot), tile u of the slot tensors, user u of the compressed cache, lane u's
    block of the flat KV cache."""

    if name == "sparse_indices":
        row = shard[:, :, lane : lane + 1]
        return torch.where(row == MASKED_INDEX, row, row - lane * ALLOCATED_CONTEXT)
    if name in ROW_FIELDS:
        return shard[:, :, lane : lane + 1]
    if name in ("staging", "ring"):
        return shard[:, lane : lane + 1]
    if name == "compressed_cache":
        return shard[lane : lane + 1]
    assert name == "kv_cache"
    start = lane * ALLOCATED_CONTEXT + (position & ~(CACHE_WRITE_ROWS - 1))
    return shard[:, :, start : start + CACHE_WRITE_ROWS]


ROW_FIELDS = (
    "index_query",
    "raw_key",
    "masked_scores",
    "sparse_indices",
    "q",
    "gate",
    "k",
    "v",
    "local_attention",
    "output",
)


def _lane_body_cases():
    cases = [(lanes, "wide") for lanes in LANE_COUNTS]
    cases += [(lanes, "per_lane") for lanes in (2, 8)]
    return cases


@pytest.mark.parametrize("lanes,indexer_form", _lane_body_cases())
def test_lane_body_rows_and_regions_are_bitwise_the_one_row_steps(fake, lanes: int, indexer_form: str) -> None:
    layer = _Layer()
    qsa = layer.qsa
    positions = LANE_POSITIONS[:lanes]
    lane_constants = Qwen38TTNNQSALaneConstants.build(
        layer.mesh, layer.contract, lanes=lanes, allocated_context=ALLOCATED_CONTEXT, indexer_form=indexer_form
    )
    hidden = [layer.hidden(lanes) for _ in range(STEPS)]  # step t: row u is lane u's hidden row

    # The B 1-row chains: lane u's STEPS steps on its own warm caches (staging and ring empty at the start).
    reference: list[list[dict[str, list[torch.Tensor]]]] = []
    generic = qsa.allocate_generic_state()
    for lane, position in enumerate(positions):
        layer.write_generic(generic)
        reference.append(
            [
                _one_row_step(layer, generic, hidden[step][:, :, lane : lane + 1], position + step)
                for step in range(STEPS)
            ]
        )
    qsa.release_generic_state(generic)

    # The lane body on the same warm caches, every lane in its own region, the row advanced between the steps.
    state = qsa.allocate_lane_state(lanes)
    layer.write_lanes(state)
    row = Qwen38TTNNDevicePositionRow.allocate(layer.mesh, layer.contract, positions, lanes=lanes)
    for step in range(STEPS):
        rows = torch.zeros(1, 1, MAX_LANES, HIDDEN_SIZE, dtype=torch.bfloat16)
        rows[:, :, :lanes] = hidden[step]
        fields = _lane_step(layer, state, lane_constants, row, rows)
        for lane, position in enumerate(positions):
            expected = reference[lane][step]
            for name in ROW_FIELDS:
                for coordinate, (got, want) in enumerate(zip(fields[name], expected[name])):
                    view = _lane_view(name, got, lane, position + step)
                    assert _bits_equal(view, want), (
                        f"B={lanes} {indexer_form} step {step} lane {lane} (P={position + step}) {name} coordinate "
                        f"{coordinate}: lane row {tuple(view.shape)} {view.flatten()[:6].tolist()} vs 1-row "
                        f"{tuple(want.shape)} {want.flatten()[:6].tolist()}"
                    )
            for name, one_row_name in (
                ("kv_cache", "kv_block"),
                ("compressed_cache", "compressed_cache"),
                ("staging", "staging"),
                ("ring", "ring"),
            ):
                for coordinate, (got, want) in enumerate(zip(fields[name], expected[one_row_name])):
                    assert _bits_equal(_lane_view(name, got, lane, position + step), want), (
                        f"B={lanes} {indexer_form} step {step} lane {lane} (P={position + step}) {name} coordinate "
                        f"{coordinate} differs from the 1-row {one_row_name}"
                    )
        # Every other lane's region is untouched by lane u's writes: the KV rows outside every lane's block stay warm.
        for coordinate, cache in enumerate(fields["kv_cache"]):
            warm = layer.kv_host[:, coordinate : coordinate + 1].repeat(1, 1, lanes, 1)
            touched = torch.zeros(lanes * ALLOCATED_CONTEXT, dtype=torch.bool)
            for lane, position in enumerate(positions):
                start = lane * ALLOCATED_CONTEXT + ((position + step) & ~(CACHE_WRITE_ROWS - 1))
                touched[start : start + CACHE_WRITE_ROWS] = True
            assert _bits_equal(cache[:, :, ~touched], warm[:, :, ~touched]), f"coordinate {coordinate}"
        row.advance()
    assert row.positions[:lanes] == [position + STEPS for position in positions]
    qsa.release_lane_state(state)
    lane_constants.deallocate()


def test_lane_state_contract_allocates_the_layout_and_refuses_foreign_states(expect_error, fake) -> None:
    layer = _Layer()
    qsa = layer.qsa
    state = qsa.allocate_lane_state(8)
    assert state.lanes == 8 and state.layer_index == LAYER
    assert state.packed_kv_cache.shape == (1, 1, 8 * ALLOCATED_CONTEXT, 2 * HEAD_DIM)
    assert state.packed_kv_cache.layout == ROW_MAJOR and state.packed_kv_cache.dtype is BF16
    assert state.compressed_index_cache.shape == (8, 1, BLOCKS + 32, INDEX_HEAD_DIM)
    assert state.compressed_index_cache_flat.shape == (1, 1, 8 * (BLOCKS + 32), INDEX_HEAD_DIM)
    assert state.kv_staging.shape == (1, 8, CACHE_WRITE_ROWS, 2 * HEAD_DIM) and state.kv_staging.layout == TILE
    assert state.raw_key_ring.shape == (1, 8, CACHE_WRITE_ROWS, INDEX_HEAD_DIM)
    # The flat view reads the users back to back: user u's rows are rows [u*R, (u+1)*R) of the view.
    layer.write_lanes(state)
    flat = fake.experimental.view(state.compressed_index_cache, (1, 1, 8 * (BLOCKS + 32), INDEX_HEAD_DIM))
    for user in range(8):
        rows = flat.torch_shards()[0][0, 0, user * (BLOCKS + 32) : (user + 1) * (BLOCKS + 32)]
        assert _bits_equal(rows, state.compressed_index_cache.torch_shards()[0][user, 0])
    for x in state.kv_staging.locals:
        x.fill_(2.0)
    qsa.reset_lane_state_inplace(state)
    assert all(float(x.abs().sum()) == 0.0 for x in state.kv_staging.locals + state.raw_key_ring.locals)
    with expect_error(ValueError, match=r"QSA lane count must be in \[1,32\], got 33"):  # allow-pytest.raises
        qsa.allocate_lane_state(33)
    foreign = Qwen38TTNNQSALaneState(**{**state.__dict__, "layer_index": LAYER + 1})
    with expect_error(ValueError, match="belongs to layer 4"):  # allow-pytest.raises: pure contract test
        qsa._validate_lane_state(foreign)
    with expect_error(ValueError, match="epoch"):  # allow-pytest.raises: pure contract test
        qsa._validate_lane_state(Qwen38TTNNQSALaneState(**{**state.__dict__, "epoch": 10**6}))
    qsa.release_lane_state(state)
    with expect_error(ValueError, match="was not allocated"):  # allow-pytest.raises: released epochs are dead
        qsa.release_lane_state(state)
    # The lane count of the constants, the inputs and the state must agree.
    two = qsa.allocate_lane_state(2)
    constants = {
        lanes: Qwen38TTNNQSALaneConstants.build(
            layer.mesh, layer.contract, lanes=lanes, allocated_context=ALLOCATED_CONTEXT
        )
        for lanes in (2, 4)
    }
    row = Qwen38TTNNDevicePositionRow.allocate(layer.mesh, layer.contract, [3, 7], lanes=2)
    inputs = derive_qsa_lane_inputs(row.row, layer.position_constants, layer.chunk_constants, constants[2])
    index_row = row.index_row()
    rope = layer.rope_table.rows_chunk(index_row, row.block_start_index_row(index_row))
    with expect_error(ValueError, match="lane constants were built for 4 lanes"):  # allow-pytest.raises: contract
        qsa.forward_decode_lanes(
            _sharded(torch.zeros(1, 1, MAX_LANES, HIDDEN_SIZE), 3),
            two,
            cos=rope.cos,
            sin=rope.sin,
            block_start_cos=rope.block_start_cos,
            block_start_sin=rope.block_start_sin,
            lanes=inputs,
            constants=layer.chunk_constants,
            lane_constants=constants[4],
        )
    with expect_error(ValueError, match="carry 2 slab rows, expected 4"):  # allow-pytest.raises: contract
        qsa._validate_lane_inputs(inputs, 4)
    qsa.release_lane_state(two)


def test_lane_body_entry_point_runs_the_layer_and_the_residue_rule_gates_the_row(expect_error, fake) -> None:
    layer = _Layer()
    qsa = layer.qsa
    lane_constants = Qwen38TTNNQSALaneConstants.build(
        layer.mesh, layer.contract, lanes=2, allocated_context=ALLOCATED_CONTEXT
    )
    state = qsa.allocate_lane_state(2)
    layer.write_lanes(state)
    row = Qwen38TTNNDevicePositionRow.allocate(layer.mesh, layer.contract, [2051, 6147], lanes=2)
    inputs = derive_qsa_lane_inputs(row.row, layer.position_constants, layer.chunk_constants, lane_constants)
    index_row = row.index_row()
    rope = layer.rope_table.rows_chunk(index_row, row.block_start_index_row(index_row))
    output = qsa.forward_decode_lanes(
        _sharded(layer.hidden(MAX_LANES), 3),
        state,
        cos=rope.cos,
        sin=rope.sin,
        block_start_cos=rope.block_start_cos,
        block_start_sin=rope.block_start_sin,
        lanes=inputs,
        constants=layer.chunk_constants,
        lane_constants=lane_constants,
    )
    assert output.shape == (1, 1, MAX_LANES, HIDDEN_SIZE // TP) and output.dtype is BF16 and output.layout == TILE
    assert all(torch.isfinite(x.float()).all() for x in output.torch_shards())
    # The residue rule lives at the position row: a lane at another residue is refused with the wait it needs, and a
    # lane admitted at the row's residue joins; the derive itself is residue-agnostic (test_batched_lanes_no_device).
    with expect_error(ValueError, match="residue 0, expected the row's residue 3; admit it 1 steps later"):
        row.admit(1, 4096)
    row.admit(1, 4099)
    assert row.positions[:2] == [2051, 4099]
    with expect_error(ValueError, match="lane 1 position 8 has residue 0, expected the row's residue 3"):
        Qwen38TTNNDevicePositionRow.allocate(layer.mesh, layer.contract, [2051, 8], lanes=2)
    inputs.deallocate()
    qsa.release_lane_state(state)


# --------------------------------------------------------------------------- source pins

LANE_BODY = (
    "forward_decode_lanes",
    "_write_compressed_index_lanes",
    "_score_blocks_lanes",
    "_materialize_rows_lanes",
    "_write_packed_kv_lanes",
)


def _dedent(function) -> str:
    lines = inspect.getsource(function).splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] for line in lines)


def _ttnn_op_walk(name: str) -> list[str]:
    tree = ast.parse(_dedent(getattr(Qwen38TTNNQSA, name)))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = ast.unparse(node.func.value)
            if owner.startswith("ttnn") and node.func.attr not in ("deallocate", "Shape"):
                calls.append((node.lineno, node.col_offset, node.func.attr))
    return [attr for _line, _col, attr in sorted(calls)]


def test_lane_body_is_the_decode_walk_on_the_lane_layout_and_the_other_bodies_are_untouched() -> None:
    for name in LANE_BODY:
        source = inspect.getsource(getattr(Qwen38TTNNQSA, name))
        for forbidden in ("from_torch", "to_torch", ".item()", "copy_host_to_device_tensor", "synchronize", ".shape["):
            assert forbidden not in source, (name, forbidden)
        for argument in ast.parse(_dedent(getattr(Qwen38TTNNQSA, name))).body[0].args.args:
            annotation = ast.unparse(argument.annotation) if argument.annotation is not None else ""
            assert annotation != "int", (name, argument.arg)
    body = inspect.getsource(Qwen38TTNNQSA.forward_decode_lanes)
    order = (
        "self._all_gather_hidden_rows(hidden_rows, constants)",
        "self._index_projection_rows(full_hidden, None, cos, sin, constants)",
        "self._write_compressed_index_lanes(state, raw_key, block_start_cos, block_start_sin, lanes, lane_constants)",
        "self._score_blocks_lanes(index_query, state, lanes, lane_constants)",
        "self._materialize_rows_lanes(masked_scores, lanes, constants, lane_constants)",
        "self._main_projection_rows(full_hidden, None, cos, sin, constants)",
        "self._write_packed_kv_lanes(state, key, value, lanes)",
        "self._sparse_value_attention_rows(query, gate, sparse_indices, state, constants)",
        "self._project_output_rows(local_attention, full_hidden, constants)",
    )
    found = [body.index(fragment) for fragment in order]
    assert found == sorted(found)
    # The compressed write: one selection matmul, the in-place ring add, the 1-row reduce per lane tile, the row
    # RoPE, one paged_update_cache for every user; no per-lane loop.
    compressed = _ttnn_op_walk("_write_compressed_index_lanes")
    assert compressed.count("paged_update_cache") == 1 and compressed.count("matmul") == 1
    assert compressed.count("sum") == 1 and "for lane in range" not in inspect.getsource(
        Qwen38TTNNQSA._write_compressed_index_lanes
    )
    assert compressed.count("view") == 2 and compressed.count("to_layout") == 4
    # The KV write: one selection matmul, the in-place staging add, one untilize, then per lane one slab slice and
    # one update_padded_kv_cache read from the lane's metadata scalar.
    # (the untilize, the per-lane slices and the slab writes moved into ``_write_kv_slabs_lanes``, shared with the
    # MTP lane verify's block writes; the walk covers both.)
    kv = _ttnn_op_walk("_write_packed_kv_lanes") + _ttnn_op_walk("_write_kv_slabs_lanes")
    assert kv.count("matmul") == 1 and kv.count("update_padded_kv_cache") == 1 and kv.count("to_layout") == 1
    kv_source = inspect.getsource(Qwen38TTNNQSA._write_packed_kv_lanes) + inspect.getsource(
        Qwen38TTNNQSA._write_kv_slabs_lanes
    )
    assert "lanes.kv_row_start_lanes" in kv_source and "row_starts[lane]" in kv_source and "self.slot_zero" in kv_source
    # The scores: one wide indexer on the flat view or one per lane with cache_batch_idx, lane u's row sliced.
    score_source = inspect.getsource(Qwen38TTNNQSA._score_blocks_lanes)
    assert "state.compressed_index_cache_flat" in score_source and "cache_batch_idx=lane" in score_source
    assert (
        'lane_constants.indexer_form == "wide"' in score_source
        and _ttnn_op_walk("_score_blocks_lanes").count("all_reduce") == 1
    )
    materialize = _ttnn_op_walk("_materialize_rows_lanes")
    assert materialize == [
        "topk_large_indices",
        "bitwise_left_shift",
        "repeat_interleave",
        "add",
        "concat",
        "bitwise_and",
        "bitwise_or",
    ]
    assert "lane_constants.block_offsets_lanes" in inspect.getsource(Qwen38TTNNQSA._materialize_rows_lanes)
    # The 1-row generic body, its helpers and the chunk body are not edited by the lane path.
    for name in (
        "forward_decode_generic",
        "_write_compressed_index_generic",
        "_score_blocks_generic",
        "_materialize_row_generic",
        "_write_packed_kv_generic",
        "forward_chunk_generic",
        "_write_compressed_index_chunk",
        "_score_blocks_chunk",
        "_materialize_rows_chunk",
        "_write_packed_kv_chunk",
        "_sparse_value_attention_rows",
        "_project_output_rows",
        "_main_projection_rows",
        "_index_projection_rows",
        "_all_gather_hidden_rows",
    ):
        source = inspect.getsource(getattr(Qwen38TTNNQSA, name))
        assert "lane" not in source and "Lane" not in source, name
    assert INDEXER_FORMS == ("wide", "per_lane")
    assert tuple(Qwen38TTNNQSALaneState.__dataclass_fields__) == (
        "layer_index",
        "epoch",
        "lanes",
        "packed_kv_cache",
        "compressed_index_cache",
        "compressed_index_cache_flat",
        "kv_staging",
        "raw_key_ring",
        "kv_scratch_rows",  # the MTP lane verify's redirect region past the last lane (0 for the plain lanes)
    )
