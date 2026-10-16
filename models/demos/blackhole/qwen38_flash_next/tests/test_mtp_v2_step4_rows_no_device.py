# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP v2 step 4: the GDN and PLE multi-row device paths against their 1-row paths, on a torch-backed ttnn.

No device.  ``ttnn`` is replaced in ``ttnn/gdn.py``, ``ttnn/ple.py`` and ``ttnn/layer.py`` by a mesh-of-four
torch model of the op surface the modules use (elementwise ops in the operand dtype, row-serial matmuls,
norms and transcendental ops so a row's result does not depend on how many rows share its tensor, collectives
over the four locals).  The chunk kernel is pluggable: the step form (the 1-row path's fp32 arithmetic) shows
that everything around the recurrence is bitwise the 1-row path at R = 1; the torch chunk reference (fp32 WY
form, bf16 q scaling as the composite does it) shows R = 5 rows against five sequential steps and every
commit(a) against the a + 1 committed steps.  The source pins hold the trace contract of the new methods
(one full chunk, device-resident constants, no host upload, no per-pass host ints, the 1-row bodies untouched).
"""

from __future__ import annotations

import ast
import inspect
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.tools.mtp_v2_verify_reference import row_serial_torch
from models.demos.blackhole.qwen38_flash_next.tt.gdn import (
    Qwen38GDN,
    Qwen38GDNDimensions,
    Qwen38GDNState,
    Qwen38GDNWeights,
)
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38PLEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_wrap
from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import chunk_gated_delta_rule

ROOT = Path(__file__).resolve().parents[1]
GDN_SOURCE = ROOT / "ttnn" / "gdn.py"
PLE_SOURCE = ROOT / "ttnn" / "ple.py"
LAYER_SOURCE = ROOT / "ttnn" / "layer.py"
TP = 4
ROWS = 5
HEADS = 12
HEAD_DIM = 128


# --------------------------------------------------------------------------- torch-backed ttnn


class _DType:
    def __init__(self, name: str, torch_dtype: torch.dtype) -> None:
        self.name, self.torch = name, torch_dtype

    def __repr__(self) -> str:
        return self.name


BF16 = _DType("bfloat16", torch.bfloat16)
FP32 = _DType("float32", torch.float32)
TILE, ROW_MAJOR = "TILE_LAYOUT", "ROW_MAJOR_LAYOUT"
_ids = itertools.count(1)


class PlacementReplicate:
    pass


class PlacementShard:
    def __init__(self, dim: int) -> None:
        self.dim = dim


class TensorTopology:
    def __init__(self, distribution_shape, placements, mesh_coords) -> None:
        self._shape, self._placements, self._coords = tuple(distribution_shape), list(placements), list(mesh_coords)

    def distribution_shape(self):
        return self._shape

    def placements(self):
        return self._placements

    def mesh_coords(self):
        return self._coords


def _topology(shard_dim: int | None) -> TensorTopology:
    second = PlacementReplicate() if shard_dim is None else PlacementShard(shard_dim)
    return TensorTopology((1, TP), [PlacementReplicate(), second], [(0, column) for column in range(TP)])


class FakeTensor:
    """One tensor per mesh device; ``locals`` are the four torch shards."""

    def __init__(self, locals_, dtype: _DType, layout: str = TILE, shard_dim: int | None = None) -> None:
        self.locals = list(locals_)
        assert len(self.locals) == TP and all(t.dtype == dtype.torch for t in self.locals), dtype
        self.dtype, self.layout = dtype, layout
        self.topology = _topology(shard_dim)
        self.tensor_id = next(_ids)
        self.alive = True

    @property
    def shape(self):
        return tuple(self.locals[0].shape)

    def device(self):
        return "mesh"

    def memory_config(self):
        return "DRAM_MEMORY_CONFIG"

    def tensor_topology(self):
        return self.topology

    def update_tensor_topology(self, topology) -> None:
        self.topology = topology

    def torch_shards(self) -> list[torch.Tensor]:
        assert self.alive, "read of a deallocated tensor"
        return self.locals


def _rowwise(fn, x: torch.Tensor) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    return torch.stack([fn(row) for row in rows]).reshape(x.shape)


def _unary(fn, dtype_keep=True):
    def op(tensor, *args, memory_config=None, output_tensor=None, **kwargs):
        results = [_rowwise(fn, t) for t in tensor.torch_shards()]
        if output_tensor is not None:
            assert output_tensor.dtype is tensor.dtype and output_tensor.shape == tuple(results[0].shape)
            for target, result in zip(output_tensor.locals, results):
                target.copy_(result)
            return output_tensor
        return FakeTensor(results, tensor.dtype, tensor.layout)

    return op


def _softplus(tensor, *, beta, threshold, memory_config=None):
    """``ttnn.softplus`` as fp32 torch softplus; the device op's SFPU error (4.7e-4 relative, never a flush to 0 on
    4x p150) is below the 1e-3 envelope the gate probe asserts, so the fake stands in for it honestly."""

    assert tensor.dtype is FP32 and (beta, threshold) == (1.0, 20.0), (tensor.dtype, beta, threshold)
    return FakeTensor(
        [_rowwise(lambda row: F.softplus(row, beta, threshold), t) for t in tensor.torch_shards()], FP32, tensor.layout
    )


def _binary(torch_op):
    def op(
        a,
        b,
        *,
        memory_config=None,
        output_tensor=None,
        dtype=None,
        input_tensor_b_activations=None,
        activations=None,
        fast_and_approximate_mode=None,
    ):
        # Mixed float dtypes compute in fp32; the output keeps input_a's dtype unless ``dtype`` says otherwise.
        out_dtype = dtype or a.dtype
        compute = (
            FP32 if dtype is FP32 or a.dtype is FP32 or (isinstance(b, FakeTensor) and b.dtype is FP32) else a.dtype
        )
        results = []
        for index, x in enumerate(a.torch_shards()):
            y = b.torch_shards()[index] if isinstance(b, FakeTensor) else b
            if isinstance(y, torch.Tensor):
                if input_tensor_b_activations:
                    assert input_tensor_b_activations == ["EXP"]
                    y = torch.exp(y)
                x, y = x.to(compute.torch), y.to(compute.torch)
            result = torch_op(x, y).to(out_dtype.torch)
            # No fused output activation is modeled: the device's fused SOFTPLUS flushed to 0 below about -5.
            assert not activations, activations
            results.append(result)
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
    def op(a, b, *, memory_config=None, dtype=None):
        out_dtype = dtype or a.dtype
        results = []
        for index, x in enumerate(a.torch_shards()):
            y = b.torch_shards()[index] if isinstance(b, FakeTensor) else b
            results.append(torch_op(x, y).to(out_dtype.torch))
        return FakeTensor(results, out_dtype, a.layout)

    return op


def _mapped(fn):
    def op(tensor, *args, memory_config=None, **kwargs):
        return FakeTensor([fn(t, *args, **kwargs) for t in tensor.torch_shards()], tensor.dtype, tensor.layout)

    return op


def _slice(tensor, start, end, *, memory_config=None, output_tensor=None):
    index = tuple(slice(s, e) for s, e in zip(start, end))
    pieces = [t[index].clone() for t in tensor.torch_shards()]
    if output_tensor is not None:
        for target, piece in zip(output_tensor.locals, pieces):
            assert target.shape == piece.shape, (target.shape, piece.shape)
            target.copy_(piece)
        return output_tensor
    return FakeTensor(pieces, tensor.dtype, tensor.layout)


def _pad(tensor, padding, value=0.0, *, memory_config=None):
    flat = [amount for pair in reversed(padding) for amount in pair]
    return FakeTensor([F.pad(t, flat, value=value) for t in tensor.torch_shards()], tensor.dtype, tensor.layout)


def _concat(tensors, dim, *, memory_config=None):
    return FakeTensor(
        [torch.cat([t.torch_shards()[index] for t in tensors], dim=dim) for index in range(TP)],
        tensors[0].dtype,
        tensors[0].layout,
    )


def _linear_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    out = torch.stack([(row[None].float() @ w.float())[0] for row in rows])
    return out.reshape(*x.shape[:-1], w.shape[-1]).to(x.dtype)


def _linear(a, w, *, memory_config=None, program_config=None, compute_kernel_config=None):
    return FakeTensor(
        [_linear_rows(x, wt.reshape(wt.shape[-2], wt.shape[-1])) for x, wt in zip(a.torch_shards(), w.torch_shards())],
        a.dtype,
        a.layout,
    )


def _matmul(
    a,
    b,
    *,
    memory_config=None,
    dtype=None,
    program_config=None,
    compute_kernel_config=None,
    optional_output_tensor=None,
):
    """fp32 accumulation over the products, one rounding into the output dtype (the fp32_dest_acc_en form)."""

    out_dtype = dtype or a.dtype
    results = [
        torch.matmul(x.float(), y.float()).to(out_dtype.torch) for x, y in zip(a.torch_shards(), b.torch_shards())
    ]
    if optional_output_tensor is not None:
        assert optional_output_tensor.dtype is out_dtype and optional_output_tensor.shape == tuple(results[0].shape)
        for target, result in zip(optional_output_tensor.locals, results):
            target.copy_(result)
        return optional_output_tensor
    return FakeTensor(results, out_dtype, a.layout)


def _tile_pages(x: torch.Tensor) -> torch.Tensor:
    """The TILE-layout page order of a logical tensor: batch dims, then the 32x32 tiles row by row (zero padded)."""

    *batch, rows, cols = x.shape
    padded_rows, padded_cols = -(-rows // 32) * 32, -(-cols // 32) * 32
    padded = F.pad(x.reshape(-1, rows, cols), (0, padded_cols - cols, 0, padded_rows - rows))
    tiles = padded.reshape(-1, padded_rows // 32, 32, padded_cols // 32, 32).permute(0, 1, 3, 2, 4)
    return tiles.reshape(-1, 32, 32)


def _from_tile_pages(pages: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    *batch, rows, cols = shape
    padded_rows, padded_cols = -(-rows // 32) * 32, -(-cols // 32) * 32
    count = 1
    for size in batch:
        count *= size
    assert pages.shape[0] == count * (padded_rows // 32) * (padded_cols // 32), (pages.shape, shape)
    tiles = pages.reshape(count, padded_rows // 32, padded_cols // 32, 32, 32).permute(0, 1, 3, 2, 4)
    return tiles.reshape(count, padded_rows, padded_cols)[:, :rows, :cols].reshape(shape).clone()


def _view(tensor, shape, *, memory_config=None):
    """``ttnn.experimental.view``: the same device pages read back under a new shape (a metadata change), i.e.
    the tile pages of the input reinterpreted in the output's tile grid.  Modelled faithfully so a view whose
    logical (row-major) meaning differs from a torch reshape is caught."""

    return FakeTensor(
        [_from_tile_pages(_tile_pages(x), tuple(shape)) for x in tensor.torch_shards()], tensor.dtype, tensor.layout
    )


def _rms_norm(tensor, *, epsilon, weight=None, memory_config=None):
    def per_device(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        def norm_row(row: torch.Tensor) -> torch.Tensor:
            row32 = row.float()
            y = row32 * torch.rsqrt(row32.square().mean() + epsilon)
            return y if w is None else y * w.reshape(-1).float()

        return _rowwise(norm_row, x).to(x.dtype)

    weights = weight.torch_shards() if weight is not None else [None] * TP
    return FakeTensor([per_device(x, w) for x, w in zip(tensor.torch_shards(), weights)], tensor.dtype, tensor.layout)


def _all_gather(tensor, *, dim, cluster_axis, memory_config=None):
    full = torch.cat(tensor.torch_shards(), dim=dim)
    return FakeTensor([full.clone() for _ in range(TP)], tensor.dtype, tensor.layout)


def _reduce_scatter(tensor, *, dim, cluster_axis, memory_config=None, topology=None):
    total = sum(t.float() for t in tensor.torch_shards()).to(tensor.dtype.torch)
    return FakeTensor([piece.clone() for piece in torch.chunk(total, TP, dim=dim)], tensor.dtype, tensor.layout, dim)


def _pre_all_gather(tensor, *, dtype, memory_config=None, compute_kernel_config=None):
    def stats(x: torch.Tensor) -> torch.Tensor:
        sums = x.float().square().sum(dim=-1, keepdim=True)  # one stat per (token, branch) row
        return F.pad(sums, (0, 31))

    return FakeTensor([stats(x) for x in tensor.torch_shards()], FP32, tensor.layout)


def _post_all_gather(tensor, gathered, *, epsilon, weight, memory_config=None, compute_kernel_config=None, dtype):
    outputs = []
    for x, stats, w in zip(tensor.torch_shards(), gathered.torch_shards(), weight.torch_shards()):
        total = stats[..., ::32].sum(dim=-1, keepdim=True)  # the four devices' column-0 stats
        mean = total / (TP * x.shape[-1])
        outputs.append((x.float() * torch.rsqrt(mean + epsilon) * w.float()).to(dtype.torch))
    return FakeTensor(outputs, dtype, tensor.layout)


def _sum_last(tensor, dim, keepdim, *, memory_config=None, compute_kernel_config=None):
    assert dim in (3, -1) and keepdim
    return FakeTensor(
        [
            torch.stack([row.sum() for row in x.reshape(-1, x.shape[-1])]).reshape(*x.shape[:-1], 1)
            for x in tensor.torch_shards()
        ],
        tensor.dtype,
        tensor.layout,
    )


def _from_torch(host, *, dtype, layout, device, memory_config, mesh_mapper):
    host = host.to(dtype.torch)
    if mesh_mapper == "replicate":
        return FakeTensor([host.clone() for _ in range(TP)], dtype, layout)
    kind, dims = mesh_mapper
    assert kind == "shard"
    return FakeTensor([piece.clone() for piece in torch.chunk(host, TP, dim=dims[1])], dtype, layout, dims[1])


def _moreh_full(shape, value, device, *, dtype, layout, memory_config):
    return FakeTensor([torch.full(tuple(shape), value, dtype=dtype.torch) for _ in range(TP)], dtype, layout)


def _get_device_tensors(tensor):
    return [FakeTensor([t] * TP, tensor.dtype, tensor.layout) for t in tensor.torch_shards()]


def _copy(source, target):
    for s, t in zip(source.torch_shards(), target.locals):
        t.copy_(s)
    return target


def _deallocate(tensor):
    assert tensor.alive, "double deallocation"
    tensor.alive = False


class FakeChunk:
    """``ttnn.transformer.chunk_gated_delta_rule`` with a pluggable per-device kernel model."""

    def __init__(self) -> None:
        self.impl = torch_chunk_kernel
        self.calls: list[dict] = []

    def chunk_gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
        eye,
        tril,
        ones,
        masks,
        output_head_major=False,
    ):
        # v may arrive rank-3 token-major flat [B, T, HV * V] (the composite's flat-v form: the prep reader
        # addresses head h's tiles at columns 128h..; no head split), or head-split [B, T, HV, V].
        flat_v = len(v.shape) == 3
        heads_v = HEADS if flat_v else v.shape[2]
        self.calls.append(
            {
                "chunk_size": chunk_size,
                "rows": q.shape[1],
                "heads": (q.shape[2], k.shape[2], heads_v),
                "flat_v": flat_v,
                "output_head_major": output_head_major,
                "constants": (eye, tril, ones, masks),
            }
        )
        # T is one full chunk (32) or the long prefill chunk's four chunks (128): pad == 0 either way.
        rows = q.shape[1]
        assert output_final_state and chunk_size == 32 and rows in (32, 128), (chunk_size, q.shape)
        assert q.dtype is BF16 and k.dtype is BF16 and v.dtype is BF16 and g.dtype is FP32 and beta.dtype is FP32
        assert initial_state.dtype is FP32
        if flat_v:
            assert v.shape == (1, rows, HEADS * HEAD_DIM), v.shape
        outputs, states = [], []
        for index in range(TP):
            o, s = self.impl(
                q.torch_shards()[index],
                k.torch_shards()[index],
                v.torch_shards()[index].reshape(1, rows, HEADS, HEAD_DIM) if flat_v else v.torch_shards()[index],
                g.torch_shards()[index],
                beta.torch_shards()[index],
                scale,
                initial_state.torch_shards()[index],
            )
            outputs.append(o)
            states.append(s)
        if output_head_major:
            # [B*HV, T, V] TILE: the kernel's own head-major layout, fp32 on the pinned runtime.
            heads = [o.permute(0, 2, 1, 3).reshape(HEADS, rows, HEAD_DIM).contiguous() for o in outputs]
            return FakeTensor(heads, FP32, TILE), FakeTensor(states, FP32, TILE)
        return FakeTensor(outputs, FP32, ROW_MAJOR), FakeTensor(states, FP32, TILE)  # fp32 token-major, as on device


def torch_chunk_kernel(q, k, v, g, beta, scale, state):
    """The kernel's algorithm in torch: bf16 q scaled in bf16 (the composite's ``multiply(q, scale)``), GQA
    expand of the 4 key heads to the 12 value heads, fp32 WY chunk of 32 rows from ``state``; fp32 output
    (the kernel's output dtype on the pinned runtime, measured on 4x p150)."""

    q_scaled = (q.float() * scale).to(torch.bfloat16).float().repeat_interleave(HEADS // q.shape[2], dim=2)
    k_full = k.float().repeat_interleave(HEADS // k.shape[2], dim=2)
    output, final = chunk_gated_delta_rule(
        q_scaled,
        k_full,
        v.float(),
        g.float(),
        beta.float(),
        chunk_size=32,
        scale=1.0,
        initial_state=state,
        output_final_state=True,
    )
    return output.float(), final


def step_chunk_kernel(q, k, v, g, beta, scale, state):
    """Row 0 through the 1-row path's fp32 step arithmetic (the same torch calls the fake step ops make); rows
    past 0 are zero.  Used to show that everything around the recurrence is bitwise the 1-row path."""

    q_row = q[:, 0].repeat_interleave(HEADS // q.shape[2], dim=1).reshape(1, HEADS, 1, HEAD_DIM).float() * scale
    k_row = k[:, 0].repeat_interleave(HEADS // k.shape[2], dim=1).reshape(1, HEADS, 1, HEAD_DIM).float()
    v_row = v[:, 0].reshape(1, HEADS, 1, HEAD_DIM)
    beta_row = beta[:, 0].reshape(1, HEADS, 1, 1)
    g_row = g[:, 0].reshape(1, HEADS, 1, 1)
    decayed = state * torch.exp(g_row)
    v_read = torch.matmul(k_row, decayed)
    delta = v_row.to(torch.float32) - v_read
    outer = torch.matmul(k_row.transpose(2, 3), delta)
    new_state = decayed + outer * beta_row
    o_row = torch.matmul(
        q_row, new_state
    )  # [1, HEADS, 1, HEAD_DIM] fp32; the rows path casts to bf16 as the 1-row path does
    output = torch.zeros(1, q.shape[1], HEADS, HEAD_DIM, dtype=torch.float32)
    output[:, 0] = o_row[:, :, 0]
    return output, new_state


def make_fake_ttnn(chunk: FakeChunk) -> SimpleNamespace:
    return SimpleNamespace(
        bfloat16=BF16,
        float32=FP32,
        TILE_LAYOUT=TILE,
        ROW_MAJOR_LAYOUT=ROW_MAJOR,
        TILE_SIZE=32,
        DRAM_MEMORY_CONFIG="DRAM_MEMORY_CONFIG",
        L1_MEMORY_CONFIG="L1",
        L1_WIDTH_SHARDED_MEMORY_CONFIG="L1WS",
        Topology=SimpleNamespace(Linear="linear"),
        UnaryOpType=SimpleNamespace(EXP="EXP"),
        UnaryWithParam=lambda op, a, b: (op, a, b),
        MathFidelity=SimpleNamespace(HiFi4="HiFi4", HiFi2="HiFi2"),
        WormholeComputeKernelConfig=lambda **fields: ("compute_kernel_config", tuple(sorted(fields.items()))),
        experimental=SimpleNamespace(view=_view),
        TensorTopology=TensorTopology,
        PlacementReplicate=PlacementReplicate,
        PlacementShard=PlacementShard,
        ShardTensor2dMesh=lambda device, mesh_shape, dims: ("shard", dims),
        from_torch=_from_torch,
        moreh_full=_moreh_full,
        get_device_tensors=_get_device_tensors,
        deallocate=_deallocate,
        copy=_copy,
        all_gather=_all_gather,
        reduce_scatter=_reduce_scatter,
        linear=_linear,
        matmul=_matmul,
        to_memory_config=lambda t, cfg: FakeTensor([x.clone() for x in t.torch_shards()], t.dtype, t.layout),
        to_layout=lambda t, layout, memory_config=None: FakeTensor(
            [x.clone() for x in t.torch_shards()], t.dtype, layout
        ),
        slice=_slice,
        pad=_pad,
        concat=_concat,
        reshape=lambda t, shape, pad_value=None, memory_config=None: FakeTensor(
            [x.reshape(tuple(shape)).clone() for x in t.torch_shards()], t.dtype, t.layout
        ),
        repeat_interleave=_mapped(lambda x, n, dim: torch.repeat_interleave(x, n, dim=dim)),
        repeat=_mapped(lambda x, multipliers: x.repeat(*multipliers)),
        transpose=_mapped(lambda x, a, b: x.transpose(a, b).contiguous()),
        permute=_mapped(lambda x, dims: x.permute(*dims).contiguous()),
        typecast=lambda t, dtype, memory_config=None: FakeTensor(
            [x.to(dtype.torch) for x in t.torch_shards()], dtype, t.layout
        ),
        multiply=_binary(torch.mul),
        add=_binary(torch.add),
        subtract=_binary(torch.sub),
        mac=lambda a, b, c: _binary(torch.add)(_binary(torch.mul)(a, b), c),
        le=_compare(torch.le),
        eq=_compare(torch.eq),
        sigmoid=_unary(torch.sigmoid),
        softplus=_softplus,
        silu=_unary(F.silu),
        sqrt=_unary(torch.sqrt),
        abs=_unary(torch.abs),
        sign=_unary(torch.sign),
        clamp=lambda t, min=None, max=None, memory_config=None: FakeTensor(
            [x.clamp(min=min, max=max) for x in t.torch_shards()], t.dtype, t.layout
        ),
        sum=_sum_last,
        rms_norm=_rms_norm,
        rms_norm_pre_all_gather=_pre_all_gather,
        rms_norm_post_all_gather=_post_all_gather,
        transformer=chunk,
    )


def install_source_chunk_fake(monkeypatch, chunk):
    """Composition tests replace only the numerical recurrence boundary.

    The real model-owned descriptors have independent device/source-word tests.
    This retains the host oracle for projection, history, masks and ownership.
    """

    def source_chunk(q, k, v, g, beta, initial_state, constants, *, rows_total, scale):
        assert q.shape[1] == rows_total
        eye, tril, ones, masks = constants
        return chunk.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=32,
            output_head_major=True,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )

    monkeypatch.setattr(gdn_module.fused.gdn_source_chunk, "chunk_token_major", source_chunk)


class FakeContract:
    """Shape/topology validation is the modules' own ``_require_shape``; placement checks are no-ops here."""

    mesh_shape = (1, TP)
    physical_ids = (0, 1, 2, 3)

    def validate_mesh(self, mesh_device) -> None:
        pass

    def validate_tensor(self, tensor, *, placement, shard_dim=None, require_device=True) -> None:
        assert tensor.alive

    def mark_local_partial(self, tensor, *, replicated_reference, expected_shape) -> None:
        assert tensor.shape == tuple(expected_shape), (tensor.shape, expected_shape)

    def mark_collective_shard(self, tensor, *, replicated_reference, shard_dim, expected_local_shape) -> None:
        assert tensor.shape == tuple(expected_local_shape), (tensor.shape, expected_local_shape)


# --------------------------------------------------------------------------- GDN under the fake


def _bf16(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


def _gdn_oracle_weights() -> Qwen38GDNWeights:
    torch.manual_seed(41)
    dims = Qwen38GDNDimensions(
        hidden_size=2560, q_heads=16, value_heads=48, key_head_dim=128, value_head_dim=128, conv_kernel=4
    )
    return Qwen38GDNWeights(
        layer_idx=0,
        dimensions=dims,
        rms_norm_eps=1e-6,
        output_gate="sigmoid",
        qkv=_bf16(dims.qkv_width, 2560, scale=2560**-0.5),
        z=_bf16(dims.value_width, 2560, scale=2560**-0.5),
        a=_bf16(48, 2560, scale=2560**-0.5),
        b=_bf16(48, 2560, scale=2560**-0.5),
        out=_bf16(2560, dims.value_width, scale=dims.value_width**-0.5),
        conv=_bf16(dims.qkv_width, 1, 4, scale=0.5),
        dt_bias=_bf16(48, scale=0.1),
        A_log=_bf16(48, scale=0.1),
        norm=_bf16(128, scale=0.1),
    )


def _device_gdn_weights(oracle: Qwen38GDNWeights) -> gdn_module.Qwen38TTNNGDNWeights:
    """The host packing of ``Qwen38TTNNGDNWeights.from_checkpoint`` on the fake mesh."""

    shards = [oracle.device_shard(index) for index in range(TP)]
    qkvzab = torch.chunk(gdn_module.pack_projection_columns(shards), TP, dim=3)
    out = [shard.out.transpose(0, 1).contiguous().reshape(1, 1, 1536, 2560) for shard in shards]
    taps = tuple(
        FakeTensor([shard.conv[:, 0, tap].to(torch.bfloat16).reshape(1, 1, 1, 2560) for shard in shards], BF16, TILE, 3)
        for tap in range(4)
    )
    return gdn_module.Qwen38TTNNGDNWeights(
        layer_index=0,
        qkvzab=FakeTensor([piece.clone() for piece in qkvzab], BF16, TILE, 3),
        out=FakeTensor(out, BF16, TILE, 2),
        conv_taps=taps,
        dt_bias=FakeTensor([shard.dt_bias.float().reshape(1, 1, 1, 12) for shard in shards], FP32, TILE, 3),
        neg_exp_A=FakeTensor(
            [(-torch.exp(shard.A_log.float())).reshape(1, 1, 1, 12) for shard in shards], FP32, TILE, 3
        ),
        norm=FakeTensor([oracle.norm.reshape(1, 1, 1, 128).clone() for _ in range(TP)], BF16, TILE),
        projection_dtype=BF16,
    )


def _gdn_module(weights) -> gdn_module.Qwen38TTNNGDN:
    module = object.__new__(gdn_module.Qwen38TTNNGDN)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.weights = weights
    module.collective_topology = "linear"
    for name in (
        "compute_config",
        "projection_compute_config",
        "in_proj_act_memory_config",
        "in_proj_program_config",
        "out_proj_act_memory_config",
        "out_proj_program_config",
        "recurrent_matmul_program_config",
        "recurrent_read_compute_config",
        "recurrent_write_compute_config",
    ):
        setattr(module, name, name)
    return module


@pytest.fixture
def fake(monkeypatch):
    chunk = FakeChunk()
    fake_ttnn = make_fake_ttnn(chunk)
    install_source_chunk_fake(monkeypatch, chunk)
    for module in (gdn_module, ple_module, layer_module):
        monkeypatch.setattr(module, "ttnn", fake_ttnn)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    return SimpleNamespace(ttnn=fake_ttnn, chunk=chunk)


def test_softplus_gate_never_flushes_to_zero_and_tracks_torch_softplus(fake) -> None:
    """The decay gate over a + dt_bias in [-8, 0] (the served range; layer-0 heads reach -5.5): no exact 0,
    within 1e-3 of torch softplus, FP32 out.  The fused SOFTPLUS activation it replaced returned 0 below -5.02."""

    xs = torch.linspace(-8.0, 0.0, 4 * 32, dtype=torch.float32).reshape(1, 1, 4, 32)
    a_fp32 = FakeTensor([xs.clone() for _ in range(TP)], FP32)
    dt_bias = FakeTensor([torch.zeros(1, 1, 1, 32) for _ in range(TP)], FP32)
    gate = gdn_module.softplus_gate(a_fp32, dt_bias, memory_config="L1")
    assert gate.dtype is FP32 and gate.shape == (1, 1, 4, 32)
    for local in gate.torch_shards():
        assert bool((local > 0.0).all()), float(local.min())
        assert float((local - F.softplus(xs)).abs().max()) <= 1e-3
    assert a_fp32.alive and dt_bias.alive  # the helper frees only its own intermediate (the add's output)


@pytest.fixture(scope="module")
def gdn_weights():
    oracle = _gdn_oracle_weights()
    return oracle, _device_gdn_weights(oracle)


def _hidden_sharded(hidden: torch.Tensor) -> FakeTensor:
    """``[1, T, 2560]`` bf16 -> hidden-sharded ``[1, 1, T, 640]``."""

    return FakeTensor(
        [piece.clone() for piece in torch.chunk(hidden.reshape(1, 1, -1, 2560), TP, dim=3)], BF16, TILE, 3
    )


def _cat(tensor: FakeTensor, dim: int) -> torch.Tensor:
    """The four device locals back to one host tensor along their shard dim (a copy)."""

    return torch.cat(tensor.torch_shards(), dim=dim)


def _scalar(tensor: FakeTensor) -> float:
    return float(tensor.torch_shards()[0].reshape(-1)[0])


def _seed_state(module: gdn_module.Qwen38TTNNGDN, seed: int):
    """A non-zero recurrent state and non-zero ring history so the paths exercise the real transition."""

    state = module.allocate_state()
    torch.manual_seed(seed)
    for local in state.recurrent.locals:
        local.copy_(torch.randn(1, 12, 128, 128) * 0.3)
    for slot in state.conv:
        for local in slot.locals:
            local.copy_(torch.randn(1, 1, 1, 2560).to(torch.bfloat16))
    return state


def _clone_state(module, state):
    clone = module.allocate_state()
    for source, target in ((state.recurrent, clone.recurrent), *zip(state.conv, clone.conv)):
        for s, t in zip(source.locals, target.locals):
            t.copy_(s)
    clone.conv_phase = state.conv_phase
    return clone


def _sequential(module, state, hidden: torch.Tensor) -> list[torch.Tensor]:
    outputs = []
    for row in range(hidden.shape[1]):
        result = module.forward_decode(_hidden_sharded(hidden[:, row : row + 1]), state)
        outputs.append(_cat(result.hidden_sharded, 3))
    return outputs


def _rows_run(module, state, hidden: torch.Tensor, constants):
    rows_state = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state, rows_state)
    result = module.forward_rows(_hidden_sharded(hidden), state, rows_state)
    return result, rows_state


def test_forward_rows_full_tile_is_the_row_slice_with_exact_zero_padding_rows(fake, gdn_weights) -> None:
    """``full_tile=True`` returns the whole 32-row output (a fresh buffer): rows below R are the persistent slice's,
    rows past R are exact zeros (zero q rows -> zero recurrent output -> zero norm, gate and projection)."""

    fake.chunk.impl = step_chunk_kernel
    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(17)
    rows = 5
    hidden = torch.randn(1, rows, 2560).to(torch.bfloat16)
    state_slice = _seed_state(module, 13)
    state_tile = _clone_state(module, state_slice)
    constants = module.allocate_rows_constants(rows)
    sliced, _ = _rows_run(module, state_slice, hidden, constants)
    rows_state = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state_tile, rows_state)
    tile = module.forward_rows(_hidden_sharded(hidden), state_tile, rows_state, full_tile=True)
    output = _cat(tile.hidden_rows, 3)
    assert output.shape == (1, 1, CHUNK_ROWS, 2560) and tile.hidden_rows is not rows_state.output
    assert torch.equal(output[:, :, :rows], _cat(sliced.hidden_rows, 3))
    assert torch.count_nonzero(output[:, :, rows:]) == 0
    assert torch.equal(_cat(tile.final_state, 1), _cat(sliced.final_state, 1))
    # The verify layer body takes the tile (no slice landing, no pad) and owns it.
    source = inspect.getsource(mtp_v2._forward_layer_verify)
    assert "full_tile=True" in source and 'label="GDN verify output rows"' not in source
    assert "attention_owner = attention_hidden" in source


def test_forward_rows_at_one_row_is_bitwise_the_one_row_path_around_the_recurrence(fake, gdn_weights) -> None:
    fake.chunk.impl = step_chunk_kernel
    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(7)
    hidden = torch.randn(1, 1, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 11)
    state_rows = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(1)

    initial = _cat(state_step.recurrent, 1)
    step_output = _sequential(module, state_step, hidden)[0]
    result, rows_state = _rows_run(module, state_rows, hidden, constants)

    assert result.hidden_rows.shape == (1, 1, 1, 640)
    assert torch.equal(_cat(result.hidden_rows, 3), step_output)
    assert torch.equal(_cat(result.final_state, 1), _cat(state_step.recurrent, 1))
    # The committed state was not written by the pass; the ring and its phase were not touched.
    assert torch.equal(_cat(state_rows.recurrent, 1), initial)
    assert (
        state_rows.conv_phase == 0 and fake.chunk.calls[-1]["chunk_size"] == 32 and fake.chunk.calls[-1]["rows"] == 32
    )
    # The kernel saw the GQA-expanded q/k (H = HV = 12, no repeat inside the composite), v token-major flat
    # (no head split) and was asked for its head-major output (no untilize / row-major permute round trip).
    assert fake.chunk.calls[-1]["heads"] == (HEADS, HEADS, HEADS) and fake.chunk.calls[-1]["output_head_major"]
    assert fake.chunk.calls[-1]["flat_v"]
    # Rows past the real one reached the kernel as zeros: k = v = 0, beta = 0, g = 0.
    for name, rows_axis, shard_dim in (("q", 1, 2), ("k", 1, 2), ("v", 2, 3), ("beta", 2, 3), ("g", 2, 3)):
        tensor = _cat(getattr(rows_state, name), shard_dim)
        assert torch.count_nonzero(tensor.narrow(rows_axis, 1, 31)) == 0, name


def test_forward_rows_takes_a_zero_padded_input_tile_without_the_pad(fake, gdn_weights) -> None:
    """A caller that hands the full 32-row tile (rows past R already zero) gets the same rows, bitwise, and
    the path skips ``ttnn.pad`` (the persistent zero-padded input tile of the verify body)."""

    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(17)
    hidden = torch.randn(1, ROWS, 2560).to(torch.bfloat16)
    state_a = _seed_state(module, 18)
    state_b = _clone_state(module, state_a)
    constants = module.allocate_rows_constants(ROWS)
    rows_a, rows_b = module.allocate_rows_state(constants), module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state_a, rows_a)
    module.sync_rows_history_from_state(state_b, rows_b)  # the eager sync pads the history tile; not counted
    pads = []
    real_pad = fake.ttnn.pad
    fake.ttnn.pad = lambda *args, **kwargs: pads.append(args[1]) or real_pad(*args, **kwargs)

    result_a = module.forward_rows(_hidden_sharded(hidden), state_a, rows_a)
    assert pads == [[(0, 0), (0, 0), (0, 32 - ROWS), (0, 0)]]
    tile = _hidden_sharded(F.pad(hidden, (0, 0, 0, 32 - ROWS)))
    assert tile.shape == (1, 1, 32, 640)
    result_b = module.forward_rows(tile, state_b, rows_b)
    assert pads == [[(0, 0), (0, 0), (0, 32 - ROWS), (0, 0)]]  # no second pad
    assert torch.equal(_cat(result_b.hidden_rows, 3), _cat(result_a.hidden_rows, 3))
    assert torch.equal(_cat(result_b.final_state, 1), _cat(result_a.final_state, 1))
    for name, shard_dim in (("q", 2), ("k", 2), ("v", 3), ("beta", 3), ("g", 3), ("qkv", 3)):
        assert torch.equal(_cat(getattr(rows_b, name), shard_dim), _cat(getattr(rows_a, name), shard_dim)), name


def _oracle_rows(oracle: Qwen38GDNWeights, hidden: torch.Tensor, state: gdn_module.Qwen38TTNNGDNState):
    # Device d holds q/k heads 4d..4d+3 and v heads 12d..12d+11 as its local [q|k|v] columns, in checkpoint order.
    stacked = torch.stack([_cat(slot, 3).reshape(TP, 2560) for slot in state.conv_window()[:3]], dim=-1)  # [4, 2560, 3]
    q, k, v = (
        stacked[:, :512].reshape(2048, 3),
        stacked[:, 512:1024].reshape(2048, 3),
        stacked[:, 1024:].reshape(6144, 3),
    )
    oracle_state = Qwen38GDNState(
        conv=torch.cat([q, k, v], dim=0).reshape(1, 10240, 3), recurrent=_cat(state.recurrent, 1)
    )
    with row_serial_torch():
        return Qwen38GDN(oracle).forward(hidden, oracle_state)


def test_forward_rows_matches_five_sequential_steps_and_the_cpu_oracle(fake, gdn_weights) -> None:
    oracle, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(8)
    hidden = torch.randn(1, ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 12)
    state_rows = _clone_state(module, state_step)
    oracle_output, oracle_state = _oracle_rows(oracle, hidden, state_rows)
    constants = module.allocate_rows_constants(ROWS)

    step_outputs = torch.cat(_sequential(module, state_step, hidden), dim=2)  # [1, 1, 5, 2560]
    result, _ = _rows_run(module, state_rows, hidden, constants)
    rows_output = _cat(result.hidden_rows, 3)
    final_state = _cat(result.final_state, 1)
    step_state = _cat(state_step.recurrent, 1)

    out_err = (rows_output.float() - step_outputs.float()).abs().max().item()
    state_err = (final_state - step_state).abs().max().item()
    print(
        f"rows vs 5 steps: output max abs {out_err:.2e} (|out| max {step_outputs.float().abs().max().item():.2f}); "
        f"state max abs {state_err:.2e} (|state| max {step_state.abs().max().item():.2f})"
    )
    assert rows_output.shape == (1, 1, ROWS, 2560)
    # The kernel scales q in bf16 (a 2**-9 relative rounding the fp32 step does not have) and the layer output
    # is bf16: allow four bf16 ulps of the output scale on top of the design's 2e-4 recurrence budget.
    assert out_err <= 2e-4 + 4 * 2.0**-8 * step_outputs.float().abs().max().item()
    torch.testing.assert_close(final_state, step_state, rtol=1e-4, atol=1e-5)  # step-2 model: rel <= 1.2e-5
    # The CPU oracle normalizes q/k with its own bf16 rounding (a 1-ulp flip in k moves the state by 4e-3), so
    # this is a wiring check (head order, conv mapping, gate/output path), not a precision gate.
    oracle_err = (
        (rows_output.float().reshape(ROWS, 2560) - oracle_output.float().reshape(ROWS, 2560)).abs().max().item()
    )
    oracle_state_err = (final_state - oracle_state.recurrent).abs().max().item()
    print(f"rows vs CPU oracle: output max abs {oracle_err:.2e}; state max abs {oracle_state_err:.2e}")
    assert oracle_err <= 0.1 * oracle_output.float().abs().max().item()
    assert oracle_state_err <= 0.1 * oracle_state.recurrent.abs().max().item()


@pytest.mark.parametrize("form", ("chunk", "step_on_full_rejection", "step_committed_rows"))
@pytest.mark.parametrize("accepted", range(ROWS))
def test_commit_rows_lands_the_committed_prefix_state_and_history(fake, gdn_weights, accepted: int, form: str) -> None:
    step_form = form == "step_on_full_rejection"
    anchor_form = form == "step_committed_rows"
    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(9)
    hidden = torch.randn(1, ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 13)
    state_rows = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(ROWS)

    _sequential(module, state_step, hidden[:, : accepted + 1])  # the committed prefix, one step at a time
    result, rows_state = _rows_run(module, state_rows, hidden, constants)
    accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
    selectors = gdn_module.build_rows_selectors(accepted_tensor, constants)
    assert [_scalar(t) for t in selectors.onehot_bf16] == [float(j == accepted) for j in range(ROWS)]
    assert selectors.committed_mask.torch_shards()[0].reshape(-1).tolist() == [float(j <= accepted) for j in range(32)]
    # The per-pass history select is the stack row of this accept count: rows 0..2 pick logical window rows
    # accepted + 1 .. accepted + 3 (history rows 0..2 sit at buffer rows 0..2, new row r at 32 + r).
    select = selectors.history_select.torch_shards()[0].float().reshape(32, 64)
    assert selectors.history_select.dtype is BF16 and torch.count_nonzero(select[3:]) == 0
    for index in range(3):
        logical = accepted + 1 + index
        assert select[index].nonzero().reshape(-1).tolist() == [logical if logical < 3 else 32 + logical - 3]
    chunk_calls = len(fake.chunk.calls)
    module.commit_rows(
        state_rows, rows_state, selectors, step_on_full_rejection=step_form, step_committed_rows=anchor_form
    )
    # The re-anchor does not rerun the chunk kernel; the other two forms rerun it once (the masked catch-up).
    assert len(fake.chunk.calls) - chunk_calls == (0 if anchor_form else 1)

    committed = _cat(state_rows.recurrent, 1)
    expected = _cat(state_step.recurrent, 1)  # accepted + 1 committed rows: row 0 (the base token) always is
    err = (committed - expected).abs().max().item()
    print(f"commit a={accepted} form={form}: state vs {accepted + 1} steps max abs {err:.2e}")
    torch.testing.assert_close(committed, expected, rtol=1e-4, atol=1e-5)
    if (accepted == 0 and step_form) or anchor_form:
        # The 1-row step arithmetic over the committed rows is bitwise the 1-row path's state (every a with the
        # re-anchor; a = 0 with the full-rejection switch).
        assert torch.equal(committed, expected)
    if accepted == ROWS - 1 and not anchor_form:
        assert torch.equal(committed, _cat(result.final_state, 1))
    # The history tile's rows 0..2 are the last three q|k|v rows of the committed stream: the 1-row ring's three
    # history slots; the selection matmul leaves exact zeros in rows 3..31.
    expected_history = torch.cat([_cat(slot, 3) for slot in state_step.conv_window()[:3]], dim=2)
    history = _cat(rows_state.history, 3)  # the four device tiles side by side
    assert history.shape == (1, 1, 32, 4 * 2560)
    assert torch.equal(history[:, :, :3], expected_history) and torch.count_nonzero(history[:, :, 3:]) == 0
    # Round trip back to the ring form for the 1-row path.
    module.sync_state_from_rows_history(rows_state, state_rows)
    for slot, expected_slot in zip(state_rows.conv_window()[:3], state_step.conv_window()[:3]):
        assert torch.equal(_cat(slot, 3), _cat(expected_slot, 3))


def test_step_anchor_keeps_the_committed_state_on_the_one_row_path_over_many_passes(fake, gdn_weights) -> None:
    """Twenty-four R = 5 passes with random accept counts (about 70 committed rows; 60 passes ran past the 300 s
    per-test timeout on a loaded host): with ``step_committed_rows`` the committed state stays bitwise the 1-row
    path's after every pass, and the FIR history the ring's; the chunk form's state is reported beside it (the fake
    kernel is the fp32 torch WY form, so its drift here is the algorithm's, not the device's TF32 drift that motivated
    the re-anchor)."""

    _, weights = gdn_weights
    module = _gdn_module(weights)
    generator = torch.Generator().manual_seed(180)
    state_step = _seed_state(module, 21)
    state_anchor = _clone_state(module, state_step)
    state_chunk = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(ROWS)
    rows_anchor = module.allocate_rows_state(constants)
    rows_chunk = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state_anchor, rows_anchor)
    module.sync_rows_history_from_state(state_chunk, rows_chunk)
    committed_rows = 0
    chunk_drift = []
    for index in range(24):
        hidden = torch.randn(1, ROWS, 2560, generator=generator).to(torch.bfloat16)
        accepted = int(torch.randint(0, ROWS, (1,), generator=generator))
        accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
        selectors = gdn_module.build_rows_selectors(accepted_tensor, constants)
        _sequential(module, state_step, hidden[:, : accepted + 1])
        for state, rows_state, anchor in ((state_anchor, rows_anchor, True), (state_chunk, rows_chunk, False)):
            module.forward_rows(_hidden_sharded(hidden), state, rows_state)
            module.commit_rows(state, rows_state, selectors, step_committed_rows=anchor)
        committed_rows += accepted + 1
        expected = _cat(state_step.recurrent, 1)
        anchored = _cat(state_anchor.recurrent, 1)
        chunk_drift.append((_cat(state_chunk.recurrent, 1) - expected).abs().max().item())
        assert torch.equal(anchored, expected), (index, accepted, (anchored - expected).abs().max().item())
        expected_history = torch.cat([_cat(slot, 3) for slot in state_step.conv_window()[:3]], dim=2)
        assert torch.equal(_cat(rows_anchor.history, 3)[:, :, :3], expected_history)
    assert committed_rows >= 60, committed_rows
    scale = _cat(state_step.recurrent, 1).abs().max().item()
    print(
        f"{committed_rows} committed rows over 24 passes: anchored state bitwise the 1-row path's at every pass; "
        f"chunk-form drift max abs {max(chunk_drift):.2e} (state scale {scale:.2f})"
    )


def test_rows_paths_never_upload_or_take_per_pass_host_ints() -> None:
    """Trace contract of the new GDN methods: one full chunk with device-resident constants, no host tensor
    creation, the row count only from the constants, and the 1-row state's ring untouched."""

    tree = ast.parse(GDN_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGDN")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    rows_methods = [
        "forward_rows",
        "commit_rows",
        "_all_gather_rows",
        "_project_rows",
        "_conv_window_rows",
        "_select_rows",
        "_causal_conv_rows",
        "_make_chunk_inputs",
        "_chunk_rows",
        "_chunk_rows_composite",
        "_project_rows_linear",
        "_land_rows_qkv",
        "_gate_and_project_rows",
        "_rows_output_tile",
        "_advance_history_rows",
    ]
    calls = {
        name: [ast.unparse(node.func) for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        for name in rows_methods
    }
    for name, names in calls.items():
        for forbidden in (
            "ttnn.zeros",
            "ttnn.from_torch",
            "ttnn.as_tensor",
            "torch.",
            "ttnn.copy_host_to_device_tensor",
        ):
            assert not any(called.startswith(forbidden) for called in names), (name, forbidden)
        assert "state.advance_conv_window" not in names and "state.conv_window" not in names, name
    for name in ("forward_rows", "commit_rows"):  # the row count comes from the constants, never per call
        for argument in methods[name].args.args:
            annotation = ast.unparse(argument.annotation) if argument.annotation is not None else ""
            assert annotation != "int", f"{name} takes a per-call host int {argument.arg}"
    # The owned rows path hands its views to the public composite with the original scale.
    dispatch = [
        node
        for node in ast.walk(methods["_chunk_rows"])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self._chunk_rows_composite"
    ]
    assert len(dispatch) == 1 and [ast.unparse(argument) for argument in dispatch[0].args] == [
        "q_rows",
        "k_rows",
        "v_rows",
        "g_rows",
        "beta_rows",
        "initial_state",
        "constants",
    ]
    assert calls["_chunk_rows"].count("ttnn.transformer.chunk_gated_delta_rule") == 0
    chunk_call = next(
        node
        for node in ast.walk(methods["_chunk_rows_composite"])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "ttnn.transformer.chunk_gated_delta_rule"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in chunk_call.keywords}
    assert keywords["chunk_size"] == "CHUNK_SIZE" and keywords["output_final_state"] == "True"
    assert keywords["output_head_major"] == "True"  # the kernel's TILE layout; no untilize / RM permute
    assert keywords["initial_state"] == "initial_state" and keywords["scale"] == "HEAD_DIM ** (-0.5)"
    assert all(keywords[name] == f"constants.{name}" for name in ("eye", "tril", "ones", "masks"))
    # q_rows / k_rows are rows_state.q / rows_state.k, or their rank-3 token-major flat views under the slab's
    # gdn_qk_flat glue form (test_ttnn_prefill_glue_no_device pins the conditional); v_rows is the flat view of v.
    assert [ast.unparse(argument) for argument in chunk_call.args] == [
        "q_rows",
        "k_rows",
        "v_rows",
        "g_rows",
        "beta_rows",
    ]
    assert calls["_chunk_rows_composite"].count("ttnn.transformer.chunk_gated_delta_rule") == 1
    assert calls["_chunk_rows_composite"].count("fused.gdn_source_chunk.chunk_token_major") == 1
    source_call = next(
        node
        for node in ast.walk(methods["_chunk_rows_composite"])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "fused.gdn_source_chunk.chunk_token_major"
    )
    source_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in source_call.keywords}
    assert source_keywords == {"rows_total": "constants.tile_rows", "scale": "HEAD_DIM ** (-0.5)"}
    # The forward pass reads the committed state and writes only rows buffers; the commit is the only writer.
    assert "output_tensor=state.recurrent" not in ast.get_source_segment(
        GDN_SOURCE.read_text(), methods["forward_rows"]
    )
    assert "_copy_inplace" in calls["commit_rows"] and "output_tensor=state.recurrent" in ast.get_source_segment(
        GDN_SOURCE.read_text(), methods["commit_rows"]
    )
    # Exact selects: every row read from the [history | qkv] window is a 0/1 selection matmul (HiFi4, fp32
    # accumulation), never a row-unaligned slice (a ROW_MAJOR round trip on this runtime) and never a reduce.
    source = GDN_SOURCE.read_text()
    select = ast.get_source_segment(source, methods["_select_rows"])
    assert "ttnn.matmul(" in select and "compute_kernel_config=self.compute_config" in select
    for name in ("_advance_history_rows", "_causal_conv_rows"):
        body = ast.get_source_segment(source, methods[name])
        assert "self._select_rows(" in body and "ttnn.slice" not in body and "ttnn.sum" not in body, name
    assert "ttnn.slice" not in ast.get_source_segment(source, methods["_conv_window_rows"])
    assert calls["_causal_conv_rows"].count("self._select_rows") == 1  # one per tap in the comprehension
    assert (
        calls["_make_chunk_inputs"].count("ttnn.matmul") == 1
        and "ttnn.repeat_interleave" not in calls["_make_chunk_inputs"]
    )
    gate = ast.get_source_segment(source, methods["_gate_and_project_rows"])
    assert "ttnn.experimental.view(" in gate and "ttnn.to_layout" not in gate and "ttnn.permute" not in gate
    assert "ttnn.pad" not in ast.get_source_segment(source, methods["_causal_conv_rows"])
    # The history select lands through the matmul's own output tensor (no copy); the state copy remains.
    assert "output_tensor=rows_state.history" in ast.get_source_segment(source, methods["_advance_history_rows"])
    assert "_copy_inplace" not in calls["_advance_history_rows"]
    assert gdn_module.CHUNK_SIZE == 32 and gdn_module.CONV_HISTORY_ROWS == 3 and gdn_module.CONV_WINDOW_ROWS == 35
    assert gdn_module.CONV_WINDOW_TILE_ROWS == 64
    tiles = gdn_module.chunk_constant_tiles()
    assert tiles["masks"].shape == (1, 1, 32, 96) and tiles["masks"].sum() == 3 * 16 * 16
    assert torch.equal(tiles["eye"][0, 0], torch.eye(32)) and tiles["tril"].sum() == 32 * 33 / 2


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and torch.equal(left.view(torch.int16), right.view(torch.int16))


def test_rows_selection_matmuls_reproduce_torch_indexing_bitwise() -> None:
    """The 0/1 selection matrices pick exactly the rows torch indexing picks, under the device's arithmetic
    model (HiFi4: exact bf16 x bf16 products; fp32 accumulation; one rounding to bf16).  Each output element
    has a single nonzero term (x * 1.0, the other products exact zeros), so the fp32 sum is x and the bf16
    result is the selected bf16 value, whatever the accumulation order."""

    tiles = gdn_module.rows_window_select_tiles(ROWS)
    taps, stack, expand = tiles["conv_taps"], tiles["history_select_stack"], tiles["qk_expand"]
    for matrix in (*taps, stack, expand):
        assert set(matrix.unique().tolist()) <= {0.0, 1.0}
        assert torch.equal(matrix.to(torch.bfloat16).float(), matrix)  # exactly representable constants
    assert (taps != 0).sum(dim=-1).eq(1).all()  # every tap row selects exactly one window row
    assert (expand != 0).sum(dim=0).eq(1).all()  # every expanded column comes from exactly one q/k column
    for accepted in range(ROWS):
        select = stack[accepted].reshape(32, 64)
        assert (select[:3] != 0).sum(dim=-1).eq(1).all() and torch.count_nonzero(select[3:]) == 0
    assert torch.count_nonzero(stack[ROWS:]) == 0

    torch.manual_seed(29)
    history_tile = _bf16(1, 1, 32, 2560)  # rows 3..31 are never read: random here on purpose
    qkv = _bf16(1, 1, 32, 2560)
    window = torch.cat([history_tile, qkv], dim=2)  # the device window: two whole tiles
    logical = torch.cat([history_tile[:, :, :3], qkv], dim=2)  # [3 history | 32 new] = 35 rows

    def select_rows(select: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        products = select.float().unsqueeze(-1) * rows.float().unsqueeze(-3)  # every product, exact in fp32
        return products.sum(dim=-2).to(torch.bfloat16)  # fp32 accumulation, one rounding

    for tap in range(3):
        assert _bits_equal(select_rows(taps[tap], window[0, 0]), logical[0, 0, tap : tap + 32]), tap
    for accepted in range(ROWS):
        onehot = torch.zeros(1, 32)
        onehot[0, accepted] = 1.0
        select = select_rows(onehot.to(torch.bfloat16), stack.to(torch.bfloat16)).float().reshape(32, 64)
        assert torch.equal(select, stack[accepted].reshape(32, 64))
        selected = select_rows(select, window[0, 0])
        assert _bits_equal(selected[:3], logical[0, 0, accepted + 1 : accepted + 4]), accepted
        assert torch.count_nonzero(selected[3:].float()) == 0
    q = _bf16(1, 1, 32, 512)
    expanded = (q[0, 0].float() @ expand.float()).to(torch.bfloat16)  # one nonzero term per output element
    assert _bits_equal(
        expanded.reshape(1, 32, HEADS, HEAD_DIM), q.reshape(1, 32, 4, HEAD_DIM).repeat_interleave(3, dim=2)
    )


def test_fake_view_is_the_device_tile_order_and_folds_head_major_rows_for_free() -> None:
    """``ttnn.experimental.view`` reinterprets the tile pages: the fake models that, and on it the head-major
    ``[1, 12, 32, 128]`` -> token-major ``[1, 1, 32, 1536]`` fold the gate path uses is a permute + reshape in
    logical terms, at zero data movement on device (tile (h, c) sits at page 4h + c in both shapes)."""

    torch.manual_seed(30)
    head_major = torch.randn(1, HEADS, 32, HEAD_DIM).to(torch.bfloat16)
    viewed = _view(FakeTensor([head_major.clone() for _ in range(TP)], BF16), (1, 1, 32, HEADS * HEAD_DIM))
    expected = head_major.permute(0, 2, 1, 3).reshape(1, 1, 32, HEADS * HEAD_DIM)
    assert _bits_equal(viewed.torch_shards()[0], expected)
    assert not torch.equal(viewed.torch_shards()[0].float(), head_major.reshape(1, 1, 32, -1).float())
    # The GR shapes the model already runs through the view on device, and a plain row-count view.
    branch_major = torch.randn(1, 4, 1, 640).to(torch.bfloat16)
    assert torch.equal(
        _view(FakeTensor([branch_major] * TP, BF16), (1, 1, 1, 2560)).torch_shards()[0],
        branch_major.reshape(1, 1, 1, 2560),
    )
    rows = torch.randn(12, 32, 128).to(torch.bfloat16)
    assert torch.equal(
        _view(FakeTensor([rows] * TP, BF16), (1, 12, 32, 128)).torch_shards()[0], rows.reshape(1, 12, 32, 128)
    )


def test_rows_bodies_are_the_pinned_walk() -> None:
    """The rows-path call order (the trace body's op order) and the stage boundaries the micro-test times."""

    tree = ast.parse(GDN_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGDN")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}

    def walk(name: str) -> list[str]:
        return [
            ast.unparse(node.func).removeprefix("self.")
            for node in sorted(
                (n for n in ast.walk(methods[name]) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
            )
            if ast.unparse(node.func).startswith("self._")
        ]

    # forward_rows gathers and hands the body to the resolved rows form; the chain's own order moved verbatim into
    # the registry's composed body (ttnn/fused/gdn_rows_wrap.rows_body_composed), which is walked right below.
    assert walk("forward_rows") == [
        "_validate_state",
        "_validate_rows_state",
        "_all_gather_rows",
        # the slab body, resolved once: the fused gdn_prefill_rows pair, or the chain's four calls below (which are
        # what the composed callable runs); the 32-row and 128-row bodies take those four directly
        "_gdn_prefill_rows",
        "_rows_body",
    ]
    composed = ast.parse(inspect.getsource(gdn_rows_wrap.rows_body_composed))
    assert [
        ast.unparse(node.func).removeprefix("gdn.")
        for node in sorted(
            (n for n in ast.walk(composed) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
        )
        if ast.unparse(node.func).startswith("gdn._")
    ] == [
        "_project_rows",
        "_causal_conv_rows",
        "_make_chunk_inputs",
        "_chunk_rows",
        "_gate_and_project_rows",
    ]
    assert walk("_project_rows") == ["_project_rows_linear", "_land_rows_qkv"]
    assert walk("_gate_and_project_rows")[-1] == "_rows_output_tile"
    assert walk("commit_rows") == [
        "_validate_state",
        "_validate_rows_state",
        "_advance_history_rows",  # the verify-rows fold's form: the prefix-state pick (a fused call), then the history
        "_step_committed_rows_state",  # the state re-anchor form: no chunk rerun
        "_advance_history_rows",
        "_chunk_rows",
        "_step_row_state",  # the step-on-full-rejection form's row 0
        "_advance_history_rows",
    ]
    assert walk("_step_committed_rows_state") == ["_step_row_state"]
    # the recurrence: the wrap's prim-layout form when it owns the rows state, else the resolved composite / prims
    # direct call; both return through the one validation
    assert walk("_chunk_rows") == ["_validate_chunk_rows", "_chunk_rows_composite", "_validate_chunk_rows"]
    assert walk("_validate_chunk_rows") == []
    assert walk("_chunk_rows_composite") == []
    assert walk("_causal_conv_rows") == ["_shifted_rows_slab", "_conv_window_rows", "_select_rows"]
    assert walk("_advance_history_rows") == ["_conv_window_rows", "_select_rows"]


def test_one_row_gdn_bodies_are_the_pinned_walk() -> None:
    """The 1-row production path is untouched: its method set and call order are the op-diet pins' walk."""

    tree = ast.parse(GDN_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGDN")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward_decode")
    called = [
        ast.unparse(node.func).removeprefix("self.")
        for node in sorted(
            (n for n in ast.walk(forward) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
        )
        if ast.unparse(node.func).startswith("self._")
    ]
    # the conv, recurrent inputs, recurrent step and gate run inside the resolved gdn_step chain (ttnn/fused/gdn_step)
    assert called == ["_validate_state", "_all_gather_hidden", "_project", "_gdn_step", "_out_project"]
    assert "rows" not in ast.get_source_segment(GDN_SOURCE.read_text(), forward)


# --------------------------------------------------------------------------- PLE under the fake

EOS = 2
PLE_TABLE_ROWS = 512


class _FakeResidentLookup:
    """A deterministic stand-in for the n-gram table: the row of (token, context) is a hash of the three ids."""

    def __init__(self) -> None:
        torch.manual_seed(21)
        self.table = _bf16(PLE_TABLE_ROWS, 2560, scale=0.5)

    def _row(self, token: int, context) -> torch.Tensor:
        c0, c1 = (EOS, EOS) if context is None else context
        return self.table[(token * 7919 + c1 * 104729 + c0 * 1299709) % PLE_TABLE_ROWS]

    def lookup_token(self, token: int, context):
        c0, c1 = (EOS, EOS) if context is None else context
        row = self._row(token, context).contiguous()
        return bytearray(row.view(torch.int16).numpy().tobytes()), (c1, token)

    def lookup_tokens(self, tokens, context):
        payload = bytearray()
        contexts = [context]
        for token in tokens:
            row, context = self.lookup_token(int(token), context)
            payload += row
            contexts.append(context)
        return payload, tuple(contexts)

    def lookup(self, input_ids: torch.Tensor, previous_context):
        context = None if previous_context is None else tuple(int(v) for v in previous_context[0])
        token = int(input_ids.reshape(-1)[0])
        row = self._row(token, context)
        _, next_context = self.lookup_token(token, context)
        return row.reshape(1, 1, 2560), torch.tensor([list(next_context)], dtype=torch.long)


def _ple_weights():
    torch.manual_seed(22)
    source = Qwen38PLEWeights(
        layer_idx=1,
        hidden_size=2560,
        residual_branches=4,
        embedding_width=2560,
        conv_kernel=4,
        conv_dilation=3,
        rms_norm_eps=1e-6,
        key=_bf16(4 * 2560, 2560, scale=2560**-0.5),
        value=_bf16(2560, 2560, scale=2560**-0.5),
        norm_key=_bf16(4 * 2560, scale=0.1),
        norm_query=_bf16(4 * 2560, scale=0.1),
        norm_conv=_bf16(4 * 2560, scale=0.1),
        conv=_bf16(4 * 2560, 1, 4, scale=0.5),
    )

    def shard(host: torch.Tensor, dtype: _DType) -> FakeTensor:
        return FakeTensor([piece.clone() for piece in torch.chunk(host.to(dtype.torch), TP, dim=3)], dtype, TILE, 3)

    conv = source.conv.reshape(4, 2560, 4)
    return ple_module.Qwen38TTNNPLEWeights(
        key=shard(ple_module._prepare_key_weight(source.key), BF16),
        value=shard(source.value.transpose(0, 1).reshape(1, 1, 2560, 2560), BF16),
        norm_key=shard((1.0 + source.norm_key.float()).reshape(1, 1, 4, 2560), FP32),
        norm_query=shard((1.0 + source.norm_query.float()).reshape(1, 1, 4, 2560), FP32),
        norm_conv=shard((1.0 + source.norm_conv.float()).reshape(1, 1, 4, 2560), FP32),
        conv_taps=tuple(shard(conv[:, :, tap].reshape(1, 1, 4, 2560), BF16) for tap in range(4)),
        replicated_anchor=FakeTensor([torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE),
        layer_index=1,
    )


def _ple_module():
    module = object.__new__(ple_module.Qwen38TTNNPLE)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.weights = _ple_weights()
    module.collective_topology = "linear"
    module._poisoned_prepared_inputs = []
    module._resident_lookup = _FakeResidentLookup()
    module.host_embedding = module._resident_lookup
    module.compute_config = "compute_config"
    return module


def _residual_rows(residual: torch.Tensor) -> FakeTensor:
    """``[R, 4, 2560]`` bf16 -> token-major hidden-sharded ``[1, R, 4, 640]``."""

    return FakeTensor(
        [piece.clone() for piece in torch.chunk(residual.reshape(1, *residual.shape), TP, dim=3)], BF16, TILE, 3
    )


def _seed_ple_state(module):
    state = module.allocate_state()
    torch.manual_seed(23)
    for slot in state.conv:
        for local in slot.locals:
            local.copy_(torch.randn(1, 1, 4, 640).to(torch.bfloat16))
    return state


def test_ple_rows_equal_sequential_prepared_steps_and_commit_selects_the_history(fake) -> None:
    module = _ple_module()
    torch.manual_seed(24)
    residual = torch.randn(ROWS, 4, 2560).to(torch.bfloat16)
    tokens = (17, 15, 16, 95859, 20)
    state = _seed_ple_state(module)
    rows_state = module.allocate_rows_state(ROWS)
    rows_state.load_from_state(state)
    assert rows_state.token_context is None

    # 1-row path, threaded through the nine slots and the host context.
    deltas, histories, contexts = [], [], [None]
    for row, token in enumerate(tokens):
        prepared = module.prepare_decode_input(torch.tensor([[token]], dtype=torch.long), state)
        result = module.forward_prepared(_residual_rows(residual[row : row + 1]), prepared, state)
        deltas.append(_cat(result.residual_delta, 3))
        histories.append(torch.cat([_cat(slot, 3) for slot in state.conv], dim=1))
        contexts.append(tuple(int(v) for v in state.token_context[0]))

    prepared_rows = module.prepare_rows_input(tokens, rows_state)
    assert prepared_rows.contexts == tuple(contexts) and prepared_rows.embedding_rows.layout == ROW_MAJOR
    result = module.forward_prepared_rows(_residual_rows(residual), prepared_rows, rows_state)
    delta_rows = _cat(result.residual_delta, 3)
    assert delta_rows.shape == (1, ROWS, 4, 2560)
    for row in range(ROWS):
        assert torch.equal(delta_rows[:, row : row + 1], deltas[row]), row

    for accepted in range(ROWS):
        fresh = module.allocate_rows_state(ROWS)
        fresh.load_from_state(_seed_ple_state(module))
        prepared_fresh = module.prepare_rows_input(tokens, fresh)
        module.forward_prepared_rows(_residual_rows(residual), prepared_fresh, fresh)
        constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", module.mesh_contract, rows=ROWS)
        accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
        module.commit_rows(fresh, gdn_module.build_rows_selectors(accepted_tensor, constants))
        module.commit_rows_host(fresh, prepared_fresh, accepted)
        assert torch.equal(_cat(fresh.history, 3), histories[accepted]), accepted
        assert fresh.token_context == contexts[accepted + 1], accepted
        # Round trip to the nine-slot form.
        back = module.allocate_state()
        fresh.store_to_state(back)
        assert torch.equal(torch.cat([_cat(slot, 3) for slot in back.conv], dim=1), histories[accepted])
        assert back.token_context.tolist() == [list(contexts[accepted + 1])]


def test_ple_rows_source_pins() -> None:
    tree = ast.parse(PLE_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNPLE")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    source = PLE_SOURCE.read_text(encoding="utf-8")
    for name in (
        "forward_prepared_rows",
        "_project_rows",
        "_gate_rows",
        "_convolve_rows",
        "_conv_window_rows",
        "commit_rows",
    ):
        calls = [ast.unparse(node.func) for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        assert not any(
            called.startswith(("ttnn.from_torch", "ttnn.zeros", "torch.", "self.resident_lookup")) for called in calls
        ), name
    inject = [ast.unparse(node.func) for node in ast.walk(methods["inject_rows"]) if isinstance(node, ast.Call)]
    assert inject.count("ttnn.permute") == 2 and "self.forward_prepared_rows" in inject and "ttnn.add" in inject
    convolve = ast.get_source_segment(source, methods["_convolve_rows"])
    assert "tap * CONV_DILATION" in convolve and convolve.count("ttnn.slice") == 1
    commit = ast.get_source_segment(source, methods["commit_rows"])
    assert "committed + CONV_STATE_LENGTH" in commit and "fast_and_approximate_mode=False" in commit
    # The 1-row body is untouched: it still shifts the nine slots and consumes the prepared host context.
    convolve_one = ast.get_source_segment(source, methods["_convolve"])
    assert "_copy_inplace(state.conv[index + 1], state.conv[index]" in convolve_one
    forward_one = ast.get_source_segment(source, methods["forward_prepared"])
    assert "state.token_context = prepared.next_token_context" in forward_one


def test_layer_rows_entry_points_are_thin_and_typed() -> None:
    tree = ast.parse(LAYER_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNDecoderLayer")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    for name in ("forward_gdn_rows", "commit_gdn_rows", "forward_ple_rows", "commit_ple_rows"):
        assert name in methods, name
    calls = [ast.unparse(node.func) for node in ast.walk(methods["forward_ple_rows"]) if isinstance(node, ast.Call)]
    assert "ple.inject_rows" in calls and "ttnn.permute" not in calls  # the layer permutes only in _apply_ple
    gdn_calls = [ast.unparse(node.func) for node in ast.walk(methods["forward_gdn_rows"]) if isinstance(node, ast.Call)]
    assert "gdn.forward_rows" in gdn_calls
    # The 1-row layer bodies do not call into the rows path.
    for name in ("forward_decode", "forward_decode_generic", "_apply_ple"):
        called = [ast.unparse(node.func) for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        assert not any(called_name.endswith("_rows") for called_name in called), name


# --------------------------------------------------------------------------- prefill chunk: R = 32
# The chunk step of the C = 32 prefill is forward_rows at the full tile plus commit_rows with every row
# accepted (a = 31; a = r - 1 for the padded last chunk).  These pins hold the same properties as the
# R = 5 tests above at the chunk's row count, under the tolerance model (output <= 2e-4 + 4 bf16
# ulp of the output scale, state <= 3e-3 + 2^-7 of the state scale).

CHUNK_ROWS = 32
CHUNK_COMMITS = (31, 0, 1, 2, 8)


def _output_tolerance(reference: torch.Tensor) -> float:
    return 2e-4 + 4 * 2.0**-8 * reference.float().abs().max().item()


def _state_tolerance(reference: torch.Tensor) -> float:
    return 3e-3 + 2.0**-7 * reference.abs().max().item()


def test_forward_rows_at_the_chunk_size_matches_32_sequential_steps_and_pads_nothing(fake, gdn_weights) -> None:
    oracle, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(31)
    hidden = torch.randn(1, CHUNK_ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 32)
    state_rows = _clone_state(module, state_step)
    oracle_output, oracle_state = _oracle_rows(oracle, hidden, state_rows)
    constants = module.allocate_rows_constants(CHUNK_ROWS)
    assert torch.count_nonzero(constants.row_mask_fp32.torch_shards()[0]) == CHUNK_ROWS  # no padding rows
    pads = []
    real_pad = fake.ttnn.pad
    fake.ttnn.pad = lambda *args, **kwargs: pads.append(args[1]) or real_pad(*args, **kwargs)

    step_outputs = torch.cat(_sequential(module, state_step, hidden), dim=2)
    result, rows_state = _rows_run(module, state_rows, hidden, constants)
    rows_output = _cat(result.hidden_rows, 3)
    final_state = _cat(result.final_state, 1)
    step_state = _cat(state_step.recurrent, 1)

    # The history sync pads its 3-row tile; the forward pass of a full tile pads nothing.
    assert pads == [[(0, 0), (0, 0), (0, 32 - 3), (0, 0)]]
    assert rows_output.shape == (1, 1, CHUNK_ROWS, 2560) and result.hidden_rows is rows_state.output
    out_err = (rows_output.float() - step_outputs.float()).abs().max().item()
    state_err = (final_state - step_state).abs().max().item()
    print(
        f"chunk rows vs 32 steps: output max abs {out_err:.2e} (tol {_output_tolerance(step_outputs):.2e}); state max abs {state_err:.2e} (tol {_state_tolerance(step_state):.2e})"
    )
    assert out_err <= _output_tolerance(step_outputs)
    assert state_err <= _state_tolerance(step_state)
    oracle_err = (
        (rows_output.float().reshape(CHUNK_ROWS, 2560) - oracle_output.float().reshape(CHUNK_ROWS, 2560))
        .abs()
        .max()
        .item()
    )
    assert oracle_err <= 0.1 * oracle_output.float().abs().max().item()
    assert (final_state - oracle_state.recurrent).abs().max().item() <= 0.1 * oracle_state.recurrent.abs().max().item()
    # The committed state and the ring are untouched by the forward pass.
    assert torch.equal(_cat(state_rows.recurrent, 1), _cat(_clone_state(module, state_rows).recurrent, 1))
    assert fake.chunk.calls[-1]["rows"] == 32 and fake.chunk.calls[-1]["chunk_size"] == 32


@pytest.mark.parametrize("accepted", CHUNK_COMMITS)
def test_commit_rows_at_the_chunk_size_lands_the_prefix_state_and_the_history(fake, gdn_weights, accepted: int) -> None:
    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(33)
    hidden = torch.randn(1, CHUNK_ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 34)
    state_rows = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(CHUNK_ROWS)

    _sequential(module, state_step, hidden[:, : accepted + 1])
    result, rows_state = _rows_run(module, state_rows, hidden, constants)
    accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
    selectors = gdn_module.build_rows_selectors(accepted_tensor, constants)
    assert len(selectors.onehot_bf16) == CHUNK_ROWS
    assert selectors.committed_mask.torch_shards()[0].reshape(-1).tolist() == [float(j <= accepted) for j in range(32)]
    module.commit_rows(state_rows, rows_state, selectors)

    committed = _cat(state_rows.recurrent, 1)
    expected = _cat(state_step.recurrent, 1)
    err = (committed - expected).abs().max().item()
    print(
        f"chunk commit a={accepted}: state vs {accepted + 1} steps max abs {err:.2e} (tol {_state_tolerance(expected):.2e})"
    )
    assert err <= _state_tolerance(expected)
    if accepted == CHUNK_ROWS - 1:
        # A full chunk commits the forward pass's own final state: no second numerics.
        assert torch.equal(committed, _cat(result.final_state, 1))
    expected_history = torch.cat([_cat(slot, 3) for slot in state_step.conv_window()[:3]], dim=2)
    history = _cat(rows_state.history, 3)
    assert torch.equal(history[:, :, :3], expected_history) and torch.count_nonzero(history[:, :, 3:]) == 0
    state_rows.conv_phase = (accepted + 1) % 4  # the hand-off: the host sets the phase, the sync fills the ring
    module.sync_state_from_rows_history(rows_state, state_rows)
    for slot, expected_slot in zip(state_rows.conv_window()[:3], state_step.conv_window()[:3]):
        assert torch.equal(_cat(slot, 3), _cat(expected_slot, 3))
    # The next 1-row step agrees on both sides (the ring, phase and state are the decode's).
    torch.manual_seed(35)
    next_hidden = torch.randn(1, 1, 2560).to(torch.bfloat16)
    step_next = _sequential(module, state_step, next_hidden)[0]
    rows_next = _sequential(module, state_rows, next_hidden)[0]
    assert (rows_next.float() - step_next.float()).abs().max().item() <= _output_tolerance(step_next)


@pytest.mark.parametrize("accepted", (CHUNK_ROWS - 1, 4))
def test_chunk_path_commit_with_the_step_anchor_is_bitwise_the_one_row_path(fake, gdn_weights, accepted: int) -> None:
    """The prefill chunk's GDN commit (rows = 32 = CHUNK_ROWS: the accept scalar 31 for a full chunk, r - 1 for the
    padded tail of r real rows) with ``step_committed_rows`` (the layer's ``gdn_step_anchor``): the committed state
    is bitwise the 1-row path's after the committed rows and the FIR history is the ring's; the chunk-kernel form is
    reported beside it (the fake kernel is the fp32 torch WY form, so its drift is the algorithm's, not the device's
    TF32 drift that grows with the prompt length and motivated the re-anchor)."""

    _, weights = gdn_weights
    module = _gdn_module(weights)
    torch.manual_seed(32)
    hidden = torch.randn(1, CHUNK_ROWS, 2560).to(torch.bfloat16)
    state_step = _seed_state(module, 33)
    state_anchor = _clone_state(module, state_step)
    state_chunk = _clone_state(module, state_step)
    constants = module.allocate_rows_constants(CHUNK_ROWS)
    accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)

    _sequential(module, state_step, hidden[:, : accepted + 1])
    committed = {}
    for state, anchor in ((state_anchor, True), (state_chunk, False)):
        result, rows_state = _rows_run(module, state, hidden, constants)
        selectors = gdn_module.build_rows_selectors(accepted_tensor, constants)
        assert selectors.committed_mask.torch_shards()[0].reshape(-1).tolist() == [
            float(j <= accepted) for j in range(CHUNK_ROWS)
        ]
        chunk_calls = len(fake.chunk.calls)
        module.commit_rows(state, rows_state, selectors, step_committed_rows=anchor)
        assert len(fake.chunk.calls) - chunk_calls == (0 if anchor else 1)
        _deallocate(result.final_state)
        committed[anchor] = _cat(state.recurrent, 1)
        if anchor:
            expected_history = torch.cat([_cat(slot, 3) for slot in state_step.conv_window()[:3]], dim=2)
            history = _cat(rows_state.history, 3)
            assert torch.equal(history[:, :, :3], expected_history) and torch.count_nonzero(history[:, :, 3:]) == 0
    expected = _cat(state_step.recurrent, 1)
    chunk_err = (committed[False] - expected).abs().max().item()
    print(f"chunk commit a={accepted}: anchored state bitwise the 1-row path's; chunk-form max abs {chunk_err:.2e}")
    assert torch.equal(committed[True], expected)
    torch.testing.assert_close(committed[False], expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("accepted", (31, 0, 1, 8))
def test_ple_rows_at_the_chunk_size_equal_32_prepared_steps_and_commit_selects_the_history(fake, accepted: int) -> None:
    module = _ple_module()
    torch.manual_seed(36)
    residual = torch.randn(CHUNK_ROWS, 4, 2560).to(torch.bfloat16)
    tokens = tuple(int(v) for v in torch.randint(0, 200_000, (CHUNK_ROWS,)))
    state = _seed_ple_state(module)
    rows_state = module.allocate_rows_state(CHUNK_ROWS)
    rows_state.load_from_state(state)

    deltas, histories, contexts = [], [], [None]
    for row, token in enumerate(tokens):
        prepared = module.prepare_decode_input(torch.tensor([[token]], dtype=torch.long), state)
        result = module.forward_prepared(_residual_rows(residual[row : row + 1]), prepared, state)
        deltas.append(_cat(result.residual_delta, 3))
        histories.append(torch.cat([_cat(slot, 3) for slot in state.conv], dim=1))
        contexts.append(tuple(int(v) for v in state.token_context[0]))

    prepared_rows = module.prepare_rows_input(tokens, rows_state)
    assert prepared_rows.contexts == tuple(contexts) and prepared_rows.embedding_rows.shape == (1, 1, CHUNK_ROWS, 640)
    result = module.forward_prepared_rows(_residual_rows(residual), prepared_rows, rows_state)
    delta_rows = _cat(result.residual_delta, 3)
    assert delta_rows.shape == (1, CHUNK_ROWS, 4, 2560)
    for row in range(CHUNK_ROWS):
        assert torch.equal(delta_rows[:, row : row + 1], deltas[row]), row

    constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", module.mesh_contract, rows=CHUNK_ROWS)
    accepted_tensor = FakeTensor([torch.full((1, 1, 1, 1), float(accepted)) for _ in range(TP)], FP32, TILE)
    module.commit_rows(rows_state, gdn_module.build_rows_selectors(accepted_tensor, constants))
    module.commit_rows_host(rows_state, prepared_rows, accepted)
    assert torch.equal(_cat(rows_state.history, 3), histories[accepted])
    assert rows_state.token_context == contexts[accepted + 1]
    back = module.allocate_state()
    rows_state.store_to_state(back)
    assert torch.equal(torch.cat([_cat(slot, 3) for slot in back.conv], dim=1), histories[accepted])
    assert back.token_context.tolist() == [list(contexts[accepted + 1])]
