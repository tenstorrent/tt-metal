# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The torch-backed ttnn of the no-device tests: a four-device mesh of torch shards and the ttnn ops the model calls.

An op has more than one form where the tests model different device arithmetic or aliasing: the fp32 row-serial
forms of the rows paths (``make_rows_ttnn`` and its extensions), the exact-integer lane forms (``make_lanes_ttnn``),
the fp64 forms of the QSA lanes (``make_qsa_ttnn``) and the single-tensor integer fake of the chunk inputs
(``make_int_ttnn``).  The model stand-ins and fixtures several test files share follow the namespaces.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.tt.gdn import Qwen38GDNDimensions, Qwen38GDNWeights
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38PLEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as decode_matmul_module
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as final_mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp as mtp_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import chunk_gated_delta_rule

TP = 4
U32_MASK = 0xFFFFFFFF
TILE, ROW_MAJOR = "TILE_LAYOUT", "ROW_MAJOR_LAYOUT"
GDN_HEADS = 12  # value heads per device (the key heads are GQA-expanded to these)
GDN_HEAD_DIM = 128


# --------------------------------------------------------------------------- mesh tensors


class FakeDType:
    def __init__(self, name: str, torch_dtype: torch.dtype) -> None:
        self.name, self.torch = name, torch_dtype

    def __repr__(self) -> str:
        return self.name


BF16 = FakeDType("bfloat16", torch.bfloat16)
FP32 = FakeDType("float32", torch.float32)
U32 = FakeDType("uint32", torch.int64)  # UINT32 values as int64 in [0, 2**32)
I32 = FakeDType("int32", torch.int32)
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


class FakeTensor:
    """One tensor per mesh device; ``locals`` are the four torch shards."""

    memory_config_value = "DRAM_MEMORY_CONFIG"  # a decode weight carries the DRAM bank layout it was uploaded with
    # padded_shape is the logical shape (so the fused programs' tile contract refuses a host fake and the rows paths
    # take the composed chains) unless a test's fake pads to whole tiles (the GR, QSA and lane checks read it).
    tile_padded = False

    def __init__(self, locals_, dtype: FakeDType, layout: str = TILE, shard_dim: int | None = None) -> None:
        self.locals = list(locals_)
        assert len(self.locals) == TP and all(t.dtype == dtype.torch for t in self.locals), dtype
        self.dtype, self.layout = dtype, layout
        second = PlacementReplicate() if shard_dim is None else PlacementShard(shard_dim)
        self.topology = TensorTopology((1, TP), [PlacementReplicate(), second], [(0, column) for column in range(TP)])
        self.tensor_id = next(_ids)
        self.alive = True

    @property
    def shape(self):
        return tuple(self.locals[0].shape)

    @property
    def padded_shape(self):
        shape = self.shape
        if not self.tile_padded or self.layout != TILE or len(shape) < 2:
            return shape
        return (*shape[:-2], -(-shape[-2] // 32) * 32, -(-shape[-1] // 32) * 32)

    def device(self):
        return "mesh"

    def memory_config(self):
        return self.memory_config_value

    def tensor_topology(self):
        return self.topology

    def update_tensor_topology(self, topology) -> None:
        self.topology = topology

    def torch_shards(self) -> list[torch.Tensor]:
        assert self.alive, "read of a deallocated tensor"
        return self.locals


def use_tile_padding(monkeypatch) -> None:
    """This test's TILE fake tensors report their last two dims rounded up to whole tiles as ``padded_shape``."""

    monkeypatch.setattr(FakeTensor, "tile_padded", True)


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


class FakeMesh:
    def arch(self):
        return "BLACKHOLE"

    def dram_grid_size(self):
        return SimpleNamespace(x=8, y=1)


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
    count = math.prod(batch)
    assert pages.shape[0] == count * (padded_rows // 32) * (padded_cols // 32), (pages.shape, shape)
    tiles = pages.reshape(count, padded_rows // 32, padded_cols // 32, 32, 32).permute(0, 1, 3, 2, 4)
    return tiles.reshape(count, padded_rows, padded_cols)[:, :rows, :cols].reshape(shape).clone()


# --------------------------------------------------------------------------- ops every form shares


def _operand(b, index: int):
    return b.torch_shards()[index] if isinstance(b, FakeTensor) else b


def _mapped(fn):
    def op(tensor, *args, memory_config=None, **kwargs):
        return FakeTensor([fn(t, *args, **kwargs) for t in tensor.torch_shards()], tensor.dtype, tensor.layout)

    return op


def _elementwise(torch_op):
    """``torch_op`` on each shard and ``b``'s matching shard (or the scalar ``b``); the dtype is kept."""

    def op(a, b, *, memory_config=None):
        return FakeTensor([torch_op(x, _operand(b, i)) for i, x in enumerate(a.torch_shards())], a.dtype, a.layout)

    return op


def _from_torch_with(convert):
    def from_torch(host, *, dtype, layout, device=None, memory_config=None, mesh_mapper):
        host = convert(host, dtype)
        if mesh_mapper == "replicate":
            return FakeTensor([host.clone() for _ in range(TP)], dtype, layout)
        kind, dims = mesh_mapper
        assert kind == "shard"
        return FakeTensor([piece.clone() for piece in torch.chunk(host, TP, dim=dims[1])], dtype, layout, dims[1])

    return from_torch


_from_torch = _from_torch_with(lambda host, dtype: host.to(dtype.torch))
_from_torch_wrapping = _from_torch_with(
    lambda host, dtype: host.to(torch.int64) & U32_MASK if dtype is U32 else host.to(dtype.torch)
)


def copy(source, target):
    for s, t in zip(source.torch_shards(), target.locals):
        t.copy_(s)
    return target


def deallocate(tensor):
    assert tensor.alive, "double deallocation"
    tensor.alive = False


def _get_device_tensors(tensor):
    members = [FakeTensor([t] * TP, tensor.dtype, tensor.layout) for t in tensor.torch_shards()]
    for member in members:
        member.memory_config_value = tensor.memory_config_value
    return members


def _zeros(shape, *, dtype, layout, device, memory_config):
    tensor = FakeTensor([torch.zeros(tuple(shape), dtype=dtype.torch) for _ in range(TP)], dtype, layout)
    # The legacy flat topology ttnn.zeros reports (what _canonicalize_flat_replicated_topology accepts).
    tensor.topology = TensorTopology((TP,), [PlacementReplicate()], [(0, column) for column in range(TP)])
    return tensor


def _to_layout(t, layout, memory_config=None):
    return FakeTensor([x.clone() for x in t.torch_shards()], t.dtype, layout)


def _to_memory_config(t, cfg):
    return FakeTensor([x.clone() for x in t.torch_shards()], t.dtype, t.layout)


def _to_torch(t, mesh_composer=None, **kwargs):
    return t.torch_shards()[0].clone()


def _typecast(t, dtype, memory_config=None):
    if dtype is I32:
        assert all(int(x.max()) < 2**31 for x in t.torch_shards())
    return FakeTensor([x.to(dtype.torch) for x in t.torch_shards()], dtype, t.layout)


def _slice_copy(tensor, start, end, *, memory_config=None, output_tensor=None):
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


def _all_gather(tensor, *, dim, cluster_axis, memory_config=None):
    full = torch.cat(tensor.torch_shards(), dim=dim)
    return FakeTensor([full.clone() for _ in range(TP)], tensor.dtype, tensor.layout)


def _argmax(t, dim=-1, keepdim=False):
    return FakeTensor([torch.argmax(x, dim=dim, keepdim=keepdim) for x in t.torch_shards()], U32, t.layout)


def _gather(t, dim, index, memory_config=None):
    return FakeTensor(
        [torch.gather(x, dim, i.to(torch.int64)) for x, i in zip(t.torch_shards(), index.torch_shards())],
        t.dtype,
        t.layout,
    )


def _paged_update_cache(cache, row, *, update_idxs_tensor):
    """Row ``user`` of ``row`` into cache user ``user`` at that user's index."""

    for target, source, index in zip(cache.locals, row.torch_shards(), update_idxs_tensor.torch_shards()):
        users = target.shape[0]
        rows = source.reshape(users, -1, source.shape[-1])
        for user, position in enumerate(index.reshape(-1).tolist()[:users]):
            target[user, 0, int(position)] = rows[user, 0]
    return cache


_minimum = _elementwise(lambda x, y: torch.clamp(x, max=y))
_repeat = _mapped(lambda x, multipliers: x.repeat(*multipliers))
_repeat_interleave = _mapped(lambda x, repeats, dim: torch.repeat_interleave(x, repeats, dim=dim))


def _unsqueeze_to_4d_view(t):
    return FakeTensor([x.reshape(1, *x.shape) for x in t.torch_shards()], t.dtype, t.layout)


def fill_in_place(tensor, value, *, output_tensor=None, memory_config=None):
    assert output_tensor is tensor
    for local in tensor.torch_shards():
        local.fill_(value)
    return tensor


# --------------------------------------------------------------------------- fp32 row-serial ops (the rows paths)


def _rowwise(fn, x: torch.Tensor) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    return torch.stack([fn(row) for row in rows]).reshape(x.shape)


def _unary(fn):
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
    """``ttnn.softplus`` as fp32 torch softplus; the device op's error is below the gate probe's 1e-3 envelope."""

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
            y = _operand(b, index)
            if isinstance(y, torch.Tensor):
                if input_tensor_b_activations:
                    assert input_tensor_b_activations == ["EXP"]
                    y = torch.exp(y)
                x, y = x.to(compute.torch), y.to(compute.torch)
            result = torch_op(x, y).to(out_dtype.torch)
            # No fused output activation is modeled.
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
        return FakeTensor(
            [torch_op(x, _operand(b, index)).to(out_dtype.torch) for index, x in enumerate(a.torch_shards())],
            out_dtype,
            a.layout,
        )

    return op


def _linear(a, w, *, memory_config=None, program_config=None, compute_kernel_config=None, dtype=None):
    out_dtype = dtype or a.dtype
    results = []
    for x, wt in zip(a.torch_shards(), w.torch_shards()):
        weight = wt.reshape(wt.shape[-2], wt.shape[-1]).float()
        rows = x.reshape(-1, x.shape[-1])
        out = torch.stack([(row[None].float() @ weight)[0] for row in rows])
        results.append(out.reshape(*x.shape[:-1], weight.shape[-1]).to(out_dtype.torch))
    return FakeTensor(results, out_dtype, a.layout)


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


def view_tile_pages(tensor, shape, *, memory_config=None):
    """``ttnn.experimental.view``: the input's tile pages reinterpreted in the output's tile grid, so a view whose
    logical meaning differs from a torch reshape is caught."""

    return FakeTensor(
        [_from_tile_pages(_tile_pages(x), tuple(shape)) for x in tensor.torch_shards()], tensor.dtype, tensor.layout
    )


def _reshape(t, shape, pad_value=None, memory_config=None):
    return FakeTensor([x.reshape(tuple(shape)).clone() for x in t.torch_shards()], t.dtype, t.layout)


def _padded_reshape(view_rule):
    """``ttnn.reshape(t, logical, padded)``: ``view_rule`` reads the logical shape out of the padded tensor."""

    def reshape(t, shape, padded_shape=None, *, pad_value=None, memory_config=None):
        shape = tuple(int(item) for item in shape)
        if padded_shape is not None and not isinstance(padded_shape, (int, float)):
            return FakeTensor([view_rule(x, shape) for x in t.torch_shards()], t.dtype, t.layout)
        return _reshape(t, shape)

    return reshape


_reshape_leading = _padded_reshape(lambda x, shape: x.reshape(-1)[: math.prod(shape)].reshape(shape).clone())
# The logical extents inside the padded tile grid: row 0 of every lane's tile for [1,B,32,128] -> [1,B,1,128].
_reshape_extents = _padded_reshape(lambda x, shape: x[tuple(slice(0, size) for size in shape)].clone())


def _rms_norm(tensor, *, epsilon, weight=None, memory_config=None, compute_kernel_config=None):
    def per_device(x: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
        def norm_row(row: torch.Tensor) -> torch.Tensor:
            row32 = row.float()
            y = row32 * torch.rsqrt(row32.square().mean() + epsilon)
            return y if w is None else y * w.reshape(-1).float()

        return _rowwise(norm_row, x).to(x.dtype)

    weights = weight.torch_shards() if weight is not None else [None] * TP
    return FakeTensor([per_device(x, w) for x, w in zip(tensor.torch_shards(), weights)], tensor.dtype, tensor.layout)


def _reduce_scatter(tensor, *, dim, cluster_axis, memory_config=None, topology=None):
    total = sum(t.float() for t in tensor.torch_shards()).to(tensor.dtype.torch)
    return FakeTensor([piece.clone() for piece in torch.chunk(total, TP, dim=dim)], tensor.dtype, tensor.layout, dim)


def _rms_norm_pre_all_gather(tensor, *, dtype, memory_config=None, compute_kernel_config=None):
    def stats(x: torch.Tensor) -> torch.Tensor:
        sums = x.float().square().sum(dim=-1, keepdim=True)  # one stat per (token, branch) row
        return F.pad(sums, (0, 31))

    return FakeTensor([stats(x) for x in tensor.torch_shards()], FP32, tensor.layout)


def _rms_norm_post_all_gather(
    tensor, gathered, *, epsilon, weight=None, memory_config=None, compute_kernel_config=None, dtype
):
    if weight is None:  # the final mixer's unscaled unit: gamma is applied afterwards on the flat row
        weight = FakeTensor([torch.ones(1, 1, 1, tensor.shape[-1]) for _ in range(TP)], FP32, TILE)
    outputs = []
    for x, stats, w in zip(tensor.torch_shards(), gathered.torch_shards(), weight.torch_shards()):
        total = stats[..., ::32].sum(dim=-1, keepdim=True)  # the four devices' column-0 stats
        mean = total / (TP * x.shape[-1])
        outputs.append((x.float() * torch.rsqrt(mean + epsilon) * w.float()).to(dtype.torch))
    return FakeTensor(outputs, dtype, tensor.layout)


def _sum(t, dim, keepdim=True, *, memory_config=None, compute_kernel_config=None, scalar=None):
    """The last-dim sum row by row; any other dim, or a scaled sum, in fp32."""

    if dim in (3, -1) and scalar is None:
        assert keepdim
        return FakeTensor(
            [
                torch.stack([row.sum() for row in x.reshape(-1, x.shape[-1])]).reshape(*x.shape[:-1], 1)
                for x in t.torch_shards()
            ],
            t.dtype,
            t.layout,
        )
    results = []
    for x in t.torch_shards():
        y = x.float().sum(dim=dim, keepdim=keepdim)
        results.append((y if scalar is None else y * scalar).to(t.dtype.torch))
    return FakeTensor(results, t.dtype, t.layout)


def _moreh_full(shape, value, device, *, dtype, layout, memory_config):
    return FakeTensor([torch.full(tuple(shape), value, dtype=dtype.torch) for _ in range(TP)], dtype, layout)


def _embedding(indices, table, *, layout=None, dtype=None, memory_config=None):
    results = []
    for index, rows in zip(indices.torch_shards(), table.torch_shards()):
        flat = rows.reshape(-1, rows.shape[-1])
        results.append(flat[index.reshape(-1).long()].reshape(1, index.numel(), rows.shape[-1]).clone())
    return FakeTensor(results, dtype or table.dtype, layout or table.layout)


def _update_padded_kv_cache(cache, staging, slot, block_start, layer_idx, num_layers, axis):
    """One 32-row ROW_MAJOR block at ``block_start``."""

    assert layer_idx == 0 and num_layers == 1 and staging.layout == ROW_MAJOR and cache.layout == ROW_MAJOR
    for target, source, start in zip(cache.locals, staging.torch_shards(), block_start.torch_shards()):
        begin = int(start.reshape(-1)[0])
        assert begin % 32 == 0 and begin + 32 <= target.shape[2], (begin, target.shape)
        target[0, 0, begin : begin + 32] = source[0, 0]
    return cache


def _fast_reduce_nc(t, *, dims, output=None, compute_kernel_config=None, memory_config=None):
    (dim,) = dims
    return FakeTensor(
        [x.float().sum(dim=dim, keepdim=True).to(t.dtype.torch) for x in t.torch_shards()], t.dtype, t.layout
    )


def _sequential_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """fp32 accumulation in index order, one bf16/fp32 rounding at the end (fast_reduce_nc's contract)."""

    pieces = x.float().unbind(dim)
    acc = pieces[0]
    for piece in pieces[1:]:
        acc = acc + piece
    return acc.unsqueeze(dim).to(x.dtype)


def _matmul_per_tile(a, b, *, memory_config=None, dtype=None, program_config=None, compute_kernel_config=None):
    """Every (batch) matmul as its own float64 2-D product rounded once: the same call for a tile in a 1-row
    operand and for the same tile at batch index u of a lanes operand."""

    out_dtype = dtype or a.dtype
    results = []
    for x, y in zip(a.torch_shards(), b.torch_shards()):
        xs, ys = x.double().reshape(-1, *x.shape[-2:]), y.double().reshape(-1, *y.shape[-2:])
        if ys.shape[0] == 1 and xs.shape[0] > 1:
            ys = ys.expand(xs.shape[0], -1, -1)
        product = torch.stack([xi @ yi for xi, yi in zip(xs, ys)])
        results.append(product.reshape(*x.shape[:-2], x.shape[-2], y.shape[-1]).to(out_dtype.torch))
    return FakeTensor(results, out_dtype, a.layout)


def _multiply_device_zero(base_multiply):
    """``base_multiply`` with binary_ng's rule: the product is +0.0 whenever an input is zero (so the keep-mask reset
    of a lane leaves +0.0, the zero source's bits, as the device does; torch alone gives -0.0 for x < 0)."""

    def op(a, b, **kwargs):
        result = base_multiply(a, b, **kwargs)
        for index, x in enumerate(result.torch_shards()):
            if x.dtype.is_floating_point:
                a_shard = a.torch_shards()[index]
                zero = a_shard == 0
                if isinstance(b, FakeTensor):
                    zero = zero | (b.torch_shards()[index] == 0)
                elif b == 0:
                    zero = torch.ones_like(zero)
                x.copy_(torch.where(zero, torch.zeros_like(x), x))
        return result

    return op


# --------------------------------------------------------------------------- integer and fp64 ops (the lanes)


def _arith_u32(torch_op):
    """UINT32 arithmetic wraps; float arithmetic runs in fp32 with binary_ng's +0.0 product."""

    def op(a, b, *, memory_config=None, dtype=None):
        results = []
        for index, x in enumerate(a.torch_shards()):
            y = _operand(b, index)
            if a.dtype is U32:
                result = torch_op(x, y) & U32_MASK
            else:
                y = y.to(torch.float32) if isinstance(y, torch.Tensor) else y
                result = torch_op(x.float(), y)
                if torch_op is torch.mul:
                    result = torch.where(x.float() == 0, torch.zeros_like(result), result)
                result = result.to(a.dtype.torch)
            results.append(result)
        return FakeTensor(results, a.dtype, a.layout)

    return op


def _wrap(value: torch.Tensor, dtype) -> torch.Tensor:
    return (value & U32_MASK) if dtype is U32 else value.to(dtype.torch)


def _binary_fp64(torch_op):
    """UINT32 ops exact and wrapped; float ops in fp64 rounded once, with binary_ng's +0.0 product."""

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
                    result = torch.where((x.double() == 0) | (y64 == 0), torch.zeros_like(result), result)
            results.append(_wrap(result, out_dtype))
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


def _compare_u32(torch_op):
    def op(a, b, *, dtype, memory_config=None):
        assert dtype is U32
        return FakeTensor(
            [torch_op(x, _operand(b, index)).to(torch.int64) for index, x in enumerate(a.torch_shards())], U32, a.layout
        )

    return op


def _shift_u32(torch_op):
    return lambda a, count, memory_config=None: FakeTensor(
        [torch_op(x, count) & U32_MASK for x in a.torch_shards()], U32, a.layout
    )


def _typecast_wrapping(t, dtype, memory_config=None):
    if dtype is I32:
        assert all(int(x.max()) < 2**31 for x in t.torch_shards())
    return FakeTensor([_wrap(x if dtype is U32 else x.double(), dtype) for x in t.torch_shards()], dtype, t.layout)


def _embedding_u32_rows(indices, table, *, layout, dtype, memory_config=None):
    assert indices.dtype is U32 and indices.shape == (1, 1, 32)
    return FakeTensor(
        [
            row_table[0, 0][index.reshape(-1)].reshape(1, 32, -1).clone()
            for index, row_table in zip(indices.torch_shards(), table.torch_shards())
        ],
        dtype,
        layout,
    )


def _sum_fp64(t, dim, keepdim, *, memory_config=None, compute_kernel_config=None, scalar=1.0):
    assert keepdim and dim in (2, 3)
    return FakeTensor(
        [(x.double().sum(dim=dim, keepdim=True) * scalar).to(t.dtype.torch) for x in t.torch_shards()],
        t.dtype,
        t.layout,
    )


def _rowwise_fp64(fn):
    """``fn`` on every last-dim row in fp64, rounded once to the tensor's dtype."""

    def op(tensor, *args, memory_config=None, **kwargs):
        results = []
        for x in tensor.torch_shards():
            rows = x.reshape(-1, x.shape[-1]).double()
            results.append(torch.stack([fn(row, *args, **kwargs) for row in rows]).reshape(x.shape).to(x.dtype))
        return FakeTensor(results, tensor.dtype, tensor.layout)

    return op


def _reshape_tile_rows(t, shape, padded_shape=None, memory_config=None):
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


def _view_aliasing_rows(t, shape):
    """``ttnn.experimental.view``: the same buffer under a new shape.  ROW_MAJOR pages and whole-tile-row TILE
    regroupings keep the torch element order, so the result aliases the input (a torch view); any other TILE view
    re-reads the tile pages in the new grid (a copy, caught by the aliasing tests if the body relied on it)."""

    shape = tuple(shape)
    aligned = t.layout == ROW_MAJOR or (
        shape[-1] == t.shape[-1] and shape[-2] % 32 == 0 and t.shape[-2] % 32 == 0 and len(shape) == 4
    )
    if aligned:
        return FakeTensor([x.reshape(shape) for x in t.torch_shards()], t.dtype, t.layout)
    return FakeTensor([_from_tile_pages(_tile_pages(x), shape) for x in t.torch_shards()], t.dtype, t.layout)


def _linear_fp64(a, w, *, memory_config=None, program_config=None, compute_kernel_config=None):
    return FakeTensor(
        [
            (x.double() @ wt.reshape(wt.shape[-2], wt.shape[-1]).double()).to(x.dtype)
            for x, wt in zip(a.torch_shards(), w.torch_shards())
        ],
        a.dtype,
        a.layout,
    )


def _matmul_fp64(a, b, *, memory_config=None, compute_kernel_config=None, optional_output_tensor=None):
    results = [torch.matmul(x.double(), y.double()).to(x.dtype) for x, y in zip(a.torch_shards(), b.torch_shards())]
    if optional_output_tensor is not None:
        for target, result in zip(optional_output_tensor.locals, results):
            target.copy_(result)
        return optional_output_tensor
    return FakeTensor(results, a.dtype, a.layout)


def _rms_norm_fp64(t, *, epsilon, weight, memory_config=None, compute_kernel_config=None):
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


def _all_reduce_fp64(t, *, cluster_axis, memory_config=None, topology=None):
    total = sum(x.double() for x in t.torch_shards()).to(t.dtype.torch)
    return FakeTensor([total.clone() for _ in range(TP)], t.dtype, t.layout)


def _reduce_scatter_fp64(t, *, dim, cluster_axis, memory_config=None, topology=None):
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
            valid = valid[valid != qsa_module.MASKED_INDEX]
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


def _update_padded_kv_slab(cache, slab, slot, start, layer_idx, num_layers, cluster_axis):
    for cs, ss, st in zip(cache.locals, slab.torch_shards(), start.torch_shards()):
        begin = int(st.reshape(-1)[0])
        assert begin % 32 == 0 and begin + ss.shape[2] <= cs.shape[2], (begin, cs.shape)
        cs[0, 0, begin : begin + ss.shape[2]] = ss[0, 0]
    return cache


# --------------------------------------------------------------------------- the ttnn namespaces


def make_rows_ttnn(chunk: FakeChunk) -> SimpleNamespace:
    """The fp32 row-serial ttnn of the GDN/PLE/layer rows paths; ``chunk`` models the chunk kernel."""

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
        experimental=SimpleNamespace(view=view_tile_pages),
        TensorTopology=TensorTopology,
        PlacementReplicate=PlacementReplicate,
        PlacementShard=PlacementShard,
        ShardTensor2dMesh=lambda device, mesh_shape, dims: ("shard", dims),
        from_torch=_from_torch,
        moreh_full=_moreh_full,
        get_device_tensors=_get_device_tensors,
        deallocate=deallocate,
        copy=copy,
        all_gather=_all_gather,
        reduce_scatter=_reduce_scatter,
        linear=_linear,
        matmul=_matmul,
        to_memory_config=_to_memory_config,
        to_layout=_to_layout,
        slice=_slice_copy,
        pad=_pad,
        concat=_concat,
        reshape=_reshape,
        repeat_interleave=_repeat_interleave,
        repeat=_repeat,
        transpose=_mapped(lambda x, a, b: x.transpose(a, b).contiguous()),
        permute=_mapped(lambda x, dims: x.permute(*dims).contiguous()),
        typecast=_typecast,
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
        sum=_sum,
        rms_norm=_rms_norm,
        rms_norm_pre_all_gather=_rms_norm_pre_all_gather,
        rms_norm_post_all_gather=_rms_norm_post_all_gather,
        transformer=chunk,
    )


def make_verify_ttnn(chunk: FakeChunk) -> SimpleNamespace:
    """The rows ttnn plus the integer, embedding, cache-write and reduce ops of the MTP verify and draft bodies."""

    fake = make_rows_ttnn(chunk)
    fake.uint32, fake.int32 = U32, I32
    fake.Shape = tuple
    fake.MeshShape = lambda *dims: tuple(dims)
    fake.reshape = _reshape_leading
    fake.embedding = _embedding
    fake.unsqueeze_to_4D = lambda t: FakeTensor(
        [x.reshape(1, 1, *x.shape[-2:]).clone() for x in t.torch_shards()], t.dtype, t.layout
    )
    fake.zeros = _zeros
    fake.lt, fake.ge, fake.ne = _compare(torch.lt), _compare(torch.ge), _compare(torch.ne)
    fake.bitwise_and, fake.bitwise_or = _elementwise(torch.bitwise_and), _elementwise(torch.bitwise_or)
    fake.bitwise_left_shift = _elementwise(lambda x, n: x << n)
    fake.bitwise_right_shift = _elementwise(lambda x, n: x >> n)
    fake.minimum = _minimum
    fake.rsub = lambda t, value, memory_config=None: FakeTensor(
        [(value - x).to(t.dtype.torch) for x in t.torch_shards()], t.dtype, t.layout
    )
    fake.argmax = _argmax
    fake.gather = _gather
    fake.max = lambda t, dim=-1, keepdim=False: FakeTensor(
        [x.max(dim=dim, keepdim=keepdim).values for x in t.torch_shards()], t.dtype, t.layout
    )
    fake.all_reduce = lambda t, *, cluster_axis, memory_config=None, topology=None: FakeTensor(
        [sum(x.float() for x in t.torch_shards()).to(t.dtype.torch).clone() for _ in range(TP)], t.dtype, t.layout
    )
    fake.mesh_partition = lambda t, *, dim, cluster_axis, memory_config=None: FakeTensor(
        [torch.chunk(x, TP, dim=dim)[i].clone() for i, x in enumerate(t.torch_shards())], t.dtype, t.layout, dim
    )
    fake.fill = lambda t, value, *, output_tensor=None: ([x.fill_(value) for x in t.locals] and (output_tensor or t))
    fake.copy_host_to_device_tensor = copy
    fake.to_torch = _to_torch
    fake.experimental = SimpleNamespace(
        view=view_tile_pages,
        paged_update_cache=_paged_update_cache,
        deepseek_prefill=SimpleNamespace(update_padded_kv_cache=_update_padded_kv_cache),
        fast_reduce_nc=_fast_reduce_nc,
    )
    return fake


def make_mtp_lanes_ttnn(chunk: FakeChunk) -> SimpleNamespace:
    """The verify ttnn on the lane layouts: a logical-extent padded reshape, the aliasing whole-tensor slice."""

    fake = make_verify_ttnn(chunk)

    def slice_(tensor, start, end, *, memory_config=None, output_tensor=None):
        # The device op returns its input over the whole tensor (``slice.cpp``'s no-op path): the alias the owned
        # slice guards against.  Into an output tensor it copies.
        if output_tensor is None and all(v == 0 for v in start) and tuple(end) == tensor.shape:
            return tensor
        return _slice_copy(tensor, start, end, memory_config=memory_config, output_tensor=output_tensor)

    fake.slice = slice_
    fake.clone = lambda t, memory_config=None: FakeTensor([x.clone() for x in t.torch_shards()], t.dtype, t.layout)
    fake.reshape = _reshape_extents
    fake.num_cores_to_corerangeset = lambda count, grid, row_wise: ("cores", count)
    fake.CoreCoord = lambda x, y: (x, y)
    fake.create_sharded_memory_config = lambda *args, **kwargs: ("sharded", args[0])
    fake.ShardStrategy = SimpleNamespace(HEIGHT="HEIGHT")
    fake.ShardOrientation = SimpleNamespace(ROW_MAJOR="ROW_MAJOR")
    return fake


def install_lane_admission_ops(fake_ttnn) -> None:
    """The ops the lane import adds: ``fill_cache``, both argument forms of ``update_padded_kv_cache``, and a
    ``view`` that aliases its source when the tile-page reinterpretation is a plain reshape, as on device."""

    def fill_cache(cache, source, batch_idx: int):
        for target, part in zip(cache.locals, source.torch_shards()):
            assert part.shape[0] == 1 and tuple(part.shape[1:]) == tuple(target.shape[1:]), (part.shape, target.shape)
            target[batch_idx] = part[0]
        return cache

    def update_padded_kv_cache(cache, staging, slot_idx, a, b, c, axis, valid_global=None, tp_axis=None):
        for index, (target, source) in enumerate(zip(cache.locals, staging.torch_shards())):
            begin = int(c) if isinstance(slot_idx, int) else int(a.torch_shards()[index].reshape(-1)[0])
            rows = source.shape[2]
            assert begin % 32 == 0 and begin + rows <= target.shape[2], (begin, rows, target.shape)
            # The op's block-cyclic invariant: the cache rows are a multiple of the rows one call writes.
            assert target.shape[2] % rows == 0, (target.shape, rows)
            target[0, 0, begin : begin + rows] = source[0, 0]
        return cache

    def view(t, shape, memory_config=None):
        shape = tuple(int(item) for item in shape)
        faithful = [_from_tile_pages(_tile_pages(x), shape) for x in t.torch_shards()]
        if all(torch.equal(f, x.reshape(shape)) for f, x in zip(faithful, t.torch_shards())):
            return FakeTensor([x.reshape(shape) for x in t.torch_shards()], t.dtype, t.layout)  # the owner's storage
        return FakeTensor(faithful, t.dtype, t.layout)

    fake_ttnn.fill_cache = fill_cache
    fake_ttnn.experimental.deepseek_prefill.update_padded_kv_cache = update_padded_kv_cache
    fake_ttnn.experimental.view = view


def make_gdn_lanes_ttnn() -> SimpleNamespace:
    """The rows ttnn whose results never depend on the batch a row sits in: per-tile fp64 matmuls, binary_ng's
    +0.0 product, and a same-shape reshape that is a view of its input (as on device)."""

    fake = make_rows_ttnn(FakeChunk())
    fake.matmul = _matmul_per_tile
    fake.multiply = _multiply_device_zero(fake.multiply)
    fake.empty_like = lambda t, dtype=None, layout=None, memory_config=None: FakeTensor(
        [torch.empty_like(x) for x in t.torch_shards()], dtype or t.dtype, layout or t.layout
    )
    # Releasing the input after a same-shape reshape is then a read of a deallocated tensor here too.
    fake.reshape = lambda t, shape, **kwargs: t if tuple(shape) == t.shape else _reshape(t, shape, **kwargs)
    return fake


def make_gr_ttnn() -> SimpleNamespace:
    """The rows ttnn plus the GR ops of the 1-row path; every op is per row or per element."""

    fake = make_rows_ttnn(FakeChunk())
    fake.UnaryOpType.SIGMOID = "SIGMOID"
    base_multiply = fake.multiply

    def multiply(a, b, *, memory_config=None, dtype=None, input_tensor_a_activations=None, output_tensor=None):
        if input_tensor_a_activations:
            assert input_tensor_a_activations == [("SIGMOID", 4.0, 0.0)]
            a = FakeTensor([torch.sigmoid(x) for x in a.torch_shards()], a.dtype, a.layout)
        return base_multiply(a, b, memory_config=memory_config, dtype=dtype, output_tensor=output_tensor)

    def post_all_gather(
        tensor, gathered, *, epsilon, weight=None, memory_config=None, compute_kernel_config=None, dtype
    ):
        outputs = []
        for x, stats in zip(tensor.torch_shards(), gathered.torch_shards()):
            total = stats[..., ::32].sum(dim=-1, keepdim=True)
            mean = total / (TP * x.shape[-1])
            outputs.append((x.float() * torch.rsqrt(mean + epsilon)).to(dtype.torch))
        return FakeTensor(outputs, dtype, tensor.layout)

    def all_gather_async(tensor, *, persistent_output_buffer, dim, **kwargs):
        return _all_gather(tensor, dim=dim, cluster_axis=None)

    def fast_reduce_nc(tensor, *, dims, output, compute_kernel_config, memory_config):
        # The device reports the 32-row tile as the row count of a TILE result below 32 rows (zero pad rows):
        # the rows bodies restore the logical rows with ttnn.reshape(t, logical, padded).
        (dim,) = dims
        shards = [_sequential_sum(x, dim) for x in tensor.torch_shards()]
        if tensor.layout == TILE and dim != 2 and shards[0].shape[2] % 32:
            shards = [F.pad(x, (0, 0, 0, 32 - x.shape[2] % 32)) for x in shards]
        return FakeTensor(shards, tensor.dtype, tensor.layout)

    def reshape(tensor, shape, padded_shape=None, memory_config=None):
        shards = tensor.torch_shards()
        if padded_shape is not None and tuple(shape) != tuple(shards[0].shape):
            assert tuple(padded_shape) == tuple(shards[0].shape), (padded_shape, shards[0].shape)
            shards = [x.narrow(2, 0, shape[2]) for x in shards]
        return FakeTensor([x.reshape(tuple(shape)).clone() for x in shards], tensor.dtype, tensor.layout)

    fake.multiply = multiply
    fake.reshape = reshape
    fake.rms_norm_post_all_gather = post_all_gather
    # ttnn.experimental.view as the tile pages under the new shape (a torch reshape would give the 1-row shapes the
    # same values but not the rows' flat rows, which is the claim under test); torch_view is that torch reshape.
    fake.experimental = SimpleNamespace(
        all_gather_async=all_gather_async,
        fast_reduce_nc=fast_reduce_nc,
        view=view_tile_pages,
        torch_view=lambda tensor, shape: FakeTensor(
            [x.reshape(tuple(shape)) for x in tensor.torch_shards()], tensor.dtype, tensor.layout
        ),
    )
    return fake


def make_final_mixer_ttnn() -> SimpleNamespace:
    """The GR ttnn with the final mixer's all-reduce: the devices' partials summed in device order, one rounding."""

    fake = make_gr_ttnn()

    def all_reduce(tensor, *, cluster_axis, memory_config=None, topology=None):
        total = _sequential_sum(torch.stack(tensor.torch_shards()), 0).squeeze(0)
        return FakeTensor([total.clone() for _ in range(TP)], tensor.dtype, tensor.layout)

    fake.all_reduce = all_reduce
    return fake


def make_lanes_ttnn() -> SimpleNamespace:
    """The exact UINT32/FP32 op surface of the lane position rows, the lane derive and the lane greedy resolve."""

    return SimpleNamespace(
        uint32=U32,
        int32=I32,
        bfloat16=BF16,
        float32=FP32,
        TILE_LAYOUT=TILE,
        ROW_MAJOR_LAYOUT=ROW_MAJOR,
        TILE_SIZE=32,
        DRAM_MEMORY_CONFIG="DRAM_MEMORY_CONFIG",
        Topology=SimpleNamespace(Linear="linear"),
        ShardTensor2dMesh=lambda device, mesh_shape, dims: ("shard", dims),
        from_torch=_from_torch_wrapping,
        copy_host_to_device_tensor=copy,
        copy=copy,
        deallocate=deallocate,
        get_device_tensors=_get_device_tensors,
        to_torch=_to_torch,
        add=_arith_u32(torch.add),
        subtract=_arith_u32(torch.sub),
        multiply=_arith_u32(torch.mul),
        minimum=_minimum,
        rsub=lambda t, value, memory_config=None: FakeTensor(
            [(value - x.float()).to(t.dtype.torch) for x in t.torch_shards()], t.dtype, t.layout
        ),
        bitwise_and=_arith_u32(torch.bitwise_and),
        bitwise_or=_arith_u32(torch.bitwise_or),
        bitwise_left_shift=_shift_u32(torch.bitwise_left_shift),
        bitwise_right_shift=_shift_u32(torch.bitwise_right_shift),
        eq=_compare_u32(torch.eq),
        lt=_compare_u32(torch.lt),
        ge=_compare_u32(torch.ge),
        typecast=_typecast,
        reshape=lambda t, shape, memory_config=None: FakeTensor(
            [x.reshape(tuple(shape)) for x in t.torch_shards()], t.dtype, t.layout
        ),
        repeat=_repeat,
        to_layout=_to_layout,
        unsqueeze_to_4D=_unsqueeze_to_4d_view,
        embedding=_embedding_u32_rows,
        all_gather=_all_gather,
        argmax=_argmax,
        gather=_gather,
        pad=_pad,
        slice=_slice_copy,
        sum=_sum_fp64,
        CoreCoord=lambda x, y: ("core", x, y),
        num_cores_to_corerangeset=lambda cores, grid, row_wise: ("cores", cores, grid),
        ShardStrategy=SimpleNamespace(HEIGHT="HEIGHT"),
        ShardOrientation=SimpleNamespace(ROW_MAJOR="ROW_MAJOR"),
        create_sharded_memory_config=lambda shape, grid, strategy, orientation, use_height_and_width_as_shard_shape: (
            "sharded",
            tuple(shape),
            grid,
        ),
    )


def make_qsa_ttnn() -> SimpleNamespace:
    """The QSA layer's ttnn, deterministic per row (fp64 rounded once to the op's dtype), so no result depends on
    the tensor shape it sits in."""

    experimental = SimpleNamespace(
        view=_view_aliasing_rows,
        rotary_embedding_hf=_rotary_embedding_hf,
        indexer_score_dsa=_indexer_score_dsa,
        topk_large_indices=_topk_large_indices,
        paged_update_cache=_paged_update_cache,
        deepseek_prefill=SimpleNamespace(update_padded_kv_cache=_update_padded_kv_slab),
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
        TensorTopology=TensorTopology,
        PlacementReplicate=PlacementReplicate,
        PlacementShard=PlacementShard,
        ShardTensor2dMesh=lambda device, mesh_shape, dims: ("shard", dims),
        experimental=experimental,
        transformer=SimpleNamespace(sparse_sdpa=_sparse_sdpa, SparseKVFormat=SimpleNamespace(BF16="BF16")),
        from_torch=_from_torch_wrapping,
        zeros=_zeros,
        fill=fill_in_place,
        copy=copy,
        copy_host_to_device_tensor=copy,
        deallocate=deallocate,
        get_device_tensors=_get_device_tensors,
        to_torch=_to_torch,
        to_layout=_to_layout,
        to_memory_config=_to_memory_config,
        reshape=_reshape_tile_rows,
        unsqueeze_to_4D=_unsqueeze_to_4d_view,
        slice=_slice_copy,
        pad=_pad,
        concat=_concat,
        repeat=_repeat,
        repeat_interleave=_repeat_interleave,
        typecast=_typecast_wrapping,
        add=_binary_fp64(torch.add),
        subtract=_binary_fp64(torch.sub),
        multiply=_binary_fp64(torch.mul),
        mul=_binary_fp64(torch.mul),
        minimum=_minimum,
        rsub=lambda t, value, memory_config=None: FakeTensor(
            [(value - x.double()).to(x.dtype) for x in t.torch_shards()], t.dtype, t.layout
        ),
        bitwise_and=_binary_fp64(torch.bitwise_and),
        bitwise_or=_binary_fp64(torch.bitwise_or),
        bitwise_left_shift=_shift_u32(torch.bitwise_left_shift),
        bitwise_right_shift=_shift_u32(torch.bitwise_right_shift),
        eq=_compare_u32(torch.eq),
        lt=_compare_u32(torch.lt),
        ge=_compare_u32(torch.ge),
        sum=_sum_fp64,
        rms_norm=_rms_norm_fp64,
        sigmoid=_rowwise_fp64(torch.sigmoid),
        linear=_linear_fp64,
        matmul=_matmul_fp64,
        all_gather=_all_gather,
        all_reduce=_all_reduce_fp64,
        reduce_scatter=_reduce_scatter_fp64,
        embedding=_embedding_u32_rows,
    )


class IntTensor:
    """One device tensor of the integer fake: int64 storage for UINT32/INT32, bf16 for the mask."""

    def __init__(self, value: torch.Tensor, dtype: FakeDType, layout: str) -> None:
        self.value, self.dtype, self.layout = value, dtype, layout
        self.tensor_id = next(_ids)
        self.alive = True

    @property
    def shape(self):
        return tuple(self.value.shape)

    def _read(self) -> torch.Tensor:
        assert self.alive, "read of a deallocated tensor"
        return self.value


class IntContract:
    def validate_tensor(self, tensor, *, placement, shard_dim=None, require_device=True) -> None:
        assert tensor.alive


def int_u32(values: torch.Tensor) -> IntTensor:
    return IntTensor(values.to(torch.int64).clone(), U32, ROW_MAJOR)


def _int_op(torch_op):
    def op(a, b, *, memory_config=None, dtype=None):
        result = torch_op(a._read(), b._read() if isinstance(b, IntTensor) else b)
        tag = dtype or a.dtype
        if tag is BF16:
            result = result.to(torch.bfloat16)
        elif result.dtype != torch.int64:
            result = result.to(torch.int64)
        return IntTensor(result, tag, a.layout)

    return op


def make_int_ttnn() -> SimpleNamespace:
    """The exact integer ops of the chunk-input derivation on single-device ``IntTensor``s."""

    def typecast(tensor, dtype, *, memory_config=None):
        value = tensor._read()
        return IntTensor(value.to(torch.bfloat16) if dtype is BF16 else value.clone(), dtype, tensor.layout)

    def from_torch(host, *, dtype, layout, device=None, memory_config=None, mesh_mapper=None):
        return IntTensor(host.to(torch.int64) if dtype is U32 else host.to(torch.bfloat16), dtype, layout)

    def copy_int(source, target):
        target._read().copy_(source._read())
        return target

    return SimpleNamespace(
        uint32=U32,
        int32=I32,
        bfloat16=BF16,
        ROW_MAJOR_LAYOUT=ROW_MAJOR,
        TILE_LAYOUT=TILE,
        TILE_SIZE=32,
        DRAM_MEMORY_CONFIG="DRAM",
        bitwise_and=_int_op(torch.bitwise_and),
        bitwise_or=_int_op(torch.bitwise_or),
        bitwise_right_shift=_int_op(torch.bitwise_right_shift),
        bitwise_left_shift=_int_op(torch.bitwise_left_shift),
        add=_int_op(torch.add),
        subtract=_int_op(torch.sub),
        multiply=_int_op(torch.mul),
        minimum=_int_op(lambda a, b: torch.clamp(a, max=b)),
        lt=_int_op(torch.lt),
        ge=_int_op(torch.ge),
        rsub=_int_op(lambda a, b: b - a),
        typecast=typecast,
        reshape=lambda t, shape, memory_config=None: IntTensor(t._read().reshape(tuple(shape)), t.dtype, t.layout),
        from_torch=from_torch,
        copy_host_to_device_tensor=copy_int,
        copy=copy_int,
        deallocate=deallocate,
    )


# --------------------------------------------------------------------------- host-side tensors and comparisons


def bf16_randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


def replicated(host: torch.Tensor, dtype: FakeDType, layout: str = TILE) -> FakeTensor:
    return FakeTensor([host.to(dtype.torch).clone() for _ in range(TP)], dtype, layout)


def sharded(host: torch.Tensor, dim: int = 3, dtype: FakeDType = BF16, layout: str = TILE) -> FakeTensor:
    return FakeTensor([piece.to(dtype.torch).clone() for piece in torch.chunk(host, TP, dim=dim)], dtype, layout, dim)


def hidden_sharded(hidden: torch.Tensor) -> FakeTensor:
    """``[1, T, 2560]`` bf16 -> hidden-sharded ``[1, 1, T, 640]``."""

    return sharded(hidden.reshape(1, 1, -1, 2560))


def residual_rows(residual: torch.Tensor) -> FakeTensor:
    """``[R, 4, 2560]`` bf16 -> token-major hidden-sharded ``[1, R, 4, 640]``."""

    return sharded(residual.reshape(1, *residual.shape))


def u32_scalar(value: int) -> FakeTensor:
    return replicated(torch.full((1, 1, 1, 1), value, dtype=torch.int64), U32, ROW_MAJOR)


def fp32_lanes(values, fill: float = -1.0, width: int = 32) -> FakeTensor:
    """A replicated FP32 ROW_MAJOR ``[1, 1, 1, width]`` lane row: ``values`` then ``fill``."""

    host = torch.full((1, 1, 1, width), fill, dtype=torch.float32)
    host[..., : len(values)] = torch.tensor([float(v) for v in values])
    return replicated(host, FP32, ROW_MAJOR)


def cat_shards(tensor: FakeTensor, dim: int = 3) -> torch.Tensor:
    """The four device locals back to one host tensor along their shard dim (a copy)."""

    return torch.cat(tensor.torch_shards(), dim=dim)


def fake_row(tensor: FakeTensor, row: int, dim: int = 2) -> FakeTensor:
    return FakeTensor([x.narrow(dim, row, 1).clone() for x in tensor.torch_shards()], tensor.dtype, tensor.layout)


def same_zero(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(left.float() + 0.0, right.float() + 0.0)  # -0.0 == +0.0 (the bf16 multiply-by-mask note)


def bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Same shape, dtype and bits (-0.0 is not +0.0)."""

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.is_floating_point():
        width = torch.int32 if left.dtype == torch.float32 else torch.int16
        return torch.equal(left.view(width), right.view(width))
    return torch.equal(left, right)


def assert_rows_bitwise(batched: FakeTensor, per_row: list[FakeTensor], *, dim: int, label: str) -> None:
    """``batched`` is the ``per_row`` results stacked along ``dim``, bitwise on every device."""

    for device, local in enumerate(batched.torch_shards()):
        stacked = torch.cat([row.torch_shards()[device] for row in per_row], dim=dim)
        assert stacked.shape == local.shape, (label, device, stacked.shape, local.shape)
        assert torch.equal(stacked.view(torch.int16), local.view(torch.int16)), (label, device)


# --------------------------------------------------------------------------- GDN and PLE stand-ins


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
        heads_v = GDN_HEADS if flat_v else v.shape[2]
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
            assert v.shape == (1, rows, GDN_HEADS * GDN_HEAD_DIM), v.shape
        outputs, states = [], []
        for index in range(TP):
            o, s = self.impl(
                q.torch_shards()[index],
                k.torch_shards()[index],
                (
                    v.torch_shards()[index].reshape(1, rows, GDN_HEADS, GDN_HEAD_DIM)
                    if flat_v
                    else v.torch_shards()[index]
                ),
                g.torch_shards()[index],
                beta.torch_shards()[index],
                scale,
                initial_state.torch_shards()[index],
            )
            outputs.append(o)
            states.append(s)
        if output_head_major:
            # [B*HV, T, V] TILE: the kernel's own head-major layout, fp32 on the pinned runtime.
            heads = [o.permute(0, 2, 1, 3).reshape(GDN_HEADS, rows, GDN_HEAD_DIM).contiguous() for o in outputs]
            return FakeTensor(heads, FP32, TILE), FakeTensor(states, FP32, TILE)
        return FakeTensor(outputs, FP32, ROW_MAJOR), FakeTensor(states, FP32, TILE)  # fp32 token-major, as on device


def torch_chunk_kernel(q, k, v, g, beta, scale, state):
    """The kernel's algorithm in torch: q scaled in bf16, GQA expand of the 4 key heads to the 12 value heads, fp32
    WY chunk of 32 rows from ``state``; fp32 output."""

    q_scaled = (q.float() * scale).to(torch.bfloat16).float().repeat_interleave(GDN_HEADS // q.shape[2], dim=2)
    k_full = k.float().repeat_interleave(GDN_HEADS // k.shape[2], dim=2)
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

    heads, head_dim = GDN_HEADS, GDN_HEAD_DIM
    q_row = q[:, 0].repeat_interleave(heads // q.shape[2], dim=1).reshape(1, heads, 1, head_dim).float() * scale
    k_row = k[:, 0].repeat_interleave(heads // k.shape[2], dim=1).reshape(1, heads, 1, head_dim).float()
    v_row = v[:, 0].reshape(1, heads, 1, head_dim)
    beta_row = beta[:, 0].reshape(1, heads, 1, 1)
    g_row = g[:, 0].reshape(1, heads, 1, 1)
    decayed = state * torch.exp(g_row)
    v_read = torch.matmul(k_row, decayed)
    delta = v_row.to(torch.float32) - v_read
    outer = torch.matmul(k_row.transpose(2, 3), delta)
    new_state = decayed + outer * beta_row
    o_row = torch.matmul(q_row, new_state)  # [1, HEADS, 1, HEAD_DIM] fp32; the rows path casts to bf16 as 1-row does
    output = torch.zeros(1, q.shape[1], heads, head_dim, dtype=torch.float32)
    output[:, 0] = o_row[:, :, 0]
    return output, new_state


def gdn_oracle_weights() -> Qwen38GDNWeights:
    torch.manual_seed(41)
    dims = Qwen38GDNDimensions(
        hidden_size=2560, q_heads=16, value_heads=48, key_head_dim=128, value_head_dim=128, conv_kernel=4
    )
    return Qwen38GDNWeights(
        layer_idx=0,
        dimensions=dims,
        rms_norm_eps=1e-6,
        output_gate="sigmoid",
        qkv=bf16_randn(dims.qkv_width, 2560, scale=2560**-0.5),
        z=bf16_randn(dims.value_width, 2560, scale=2560**-0.5),
        a=bf16_randn(48, 2560, scale=2560**-0.5),
        b=bf16_randn(48, 2560, scale=2560**-0.5),
        out=bf16_randn(2560, dims.value_width, scale=dims.value_width**-0.5),
        conv=bf16_randn(dims.qkv_width, 1, 4, scale=0.5),
        dt_bias=bf16_randn(48, scale=0.1),
        A_log=bf16_randn(48, scale=0.1),
        norm=bf16_randn(128, scale=0.1),
    )


def device_gdn_weights(oracle: Qwen38GDNWeights) -> gdn_module.Qwen38TTNNGDNWeights:
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


def make_gdn_module(weights) -> gdn_module.Qwen38TTNNGDN:
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


def seed_gdn_state(module: gdn_module.Qwen38TTNNGDN, seed: int):
    """A non-zero recurrent state and non-zero ring history so the paths exercise the real transition."""

    state = module.allocate_state()
    torch.manual_seed(seed)
    for local in state.recurrent.locals:
        local.copy_(torch.randn(1, 12, 128, 128) * 0.3)
    for slot in state.conv:
        for local in slot.locals:
            local.copy_(torch.randn(1, 1, 1, 2560).to(torch.bfloat16))
    return state


def clone_gdn_state(module, state):
    clone = module.allocate_state()
    for source, target in ((state.recurrent, clone.recurrent), *zip(state.conv, clone.conv)):
        for s, t in zip(source.locals, target.locals):
            t.copy_(s)
    clone.conv_phase = state.conv_phase
    return clone


def run_gdn_steps(module, state, hidden: torch.Tensor) -> list[torch.Tensor]:
    """The 1-row decode over every row of ``hidden``: the per-row outputs."""

    outputs = []
    for row in range(hidden.shape[1]):
        result = module.forward_decode(hidden_sharded(hidden[:, row : row + 1]), state)
        outputs.append(cat_shards(result.hidden_sharded, 3))
    return outputs


def gdn_output_tolerance(reference: torch.Tensor) -> float:
    return 2e-4 + 4 * 2.0**-8 * reference.float().abs().max().item()


def gdn_state_tolerance(reference: torch.Tensor) -> float:
    return 3e-3 + 2.0**-7 * reference.abs().max().item()


def gdn_rows_selectors(accepted: int, constants):
    """The commit selects of a replicated FP32 accept count ``accepted``."""

    return gdn_module.build_rows_selectors(replicated(torch.full((1, 1, 1, 1), float(accepted)), FP32), constants)


def gdn_ring_history(state) -> torch.Tensor:
    """The 1-row ring's three history slots in window order, ``[1, 1, 3, 4 * 2560]`` (the devices side by side)."""

    return torch.cat([cat_shards(slot, 3) for slot in state.conv_window()[:3]], dim=2)


EOS = 2
PLE_TABLE_ROWS = 512


class FakeResidentLookup:
    """A deterministic stand-in for the n-gram table: the row of (token, context) is a hash of the three ids."""

    def __init__(self) -> None:
        torch.manual_seed(21)
        self.table = bf16_randn(PLE_TABLE_ROWS, 2560, scale=0.5)

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

    def lookup_lanes(self, tokens, contexts):
        """The batched lane read: one payload of every lane's row, lane-major."""

        payloads, next_contexts = zip(
            *(self.lookup_token(int(token), context) for token, context in zip(tokens, contexts))
        )
        return bytearray().join(payloads), tuple(next_contexts)

    def lookup(self, input_ids: torch.Tensor, previous_context):
        context = None if previous_context is None else tuple(int(v) for v in previous_context[0])
        token = int(input_ids.reshape(-1)[0])
        row = self._row(token, context)
        _, next_context = self.lookup_token(token, context)
        return row.reshape(1, 1, 2560), torch.tensor([list(next_context)], dtype=torch.long)


def make_ple_module():
    torch.manual_seed(22)
    source = Qwen38PLEWeights(
        layer_idx=1,
        hidden_size=2560,
        residual_branches=4,
        embedding_width=2560,
        conv_kernel=4,
        conv_dilation=3,
        rms_norm_eps=1e-6,
        key=bf16_randn(4 * 2560, 2560, scale=2560**-0.5),
        value=bf16_randn(2560, 2560, scale=2560**-0.5),
        norm_key=bf16_randn(4 * 2560, scale=0.1),
        norm_query=bf16_randn(4 * 2560, scale=0.1),
        norm_conv=bf16_randn(4 * 2560, scale=0.1),
        conv=bf16_randn(4 * 2560, 1, 4, scale=0.5),
    )
    conv = source.conv.reshape(4, 2560, 4)
    module = object.__new__(ple_module.Qwen38TTNNPLE)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.weights = ple_module.Qwen38TTNNPLEWeights(
        key=sharded(ple_module._prepare_key_weight(source.key)),
        value=sharded(source.value.transpose(0, 1).reshape(1, 1, 2560, 2560)),
        norm_key=sharded((1.0 + source.norm_key.float()).reshape(1, 1, 4, 2560), dtype=FP32),
        norm_query=sharded((1.0 + source.norm_query.float()).reshape(1, 1, 4, 2560), dtype=FP32),
        norm_conv=sharded((1.0 + source.norm_conv.float()).reshape(1, 1, 4, 2560), dtype=FP32),
        conv_taps=tuple(sharded(conv[:, :, tap].reshape(1, 1, 4, 2560)) for tap in range(4)),
        replicated_anchor=FakeTensor([torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE),
        layer_index=1,
    )
    module.collective_topology = "linear"
    module._poisoned_prepared_inputs = []
    module._resident_lookup = FakeResidentLookup()
    module.host_embedding = module._resident_lookup
    module.compute_config = "compute_config"
    return module


def seed_ple_state(module):
    state = module.allocate_state()
    torch.manual_seed(23)
    for slot in state.conv:
        for local in slot.locals:
            local.copy_(torch.randn(1, 1, 4, 640).to(torch.bfloat16))
    return state


# --------------------------------------------------------------------------- QSA verify, heads and GR stand-ins

VERIFY_CONTEXT = 2048  # the smallest admitted resident cache: 512 compressed blocks (+ the indexer's query tile)
VERIFY_BLOCKS = VERIFY_CONTEXT // 4
SMALL_IDS = (17, 15, 16, 21, 12, 20, 11, 0, 2047)
LARGE_IDS = (2048, 95859, 62086, 248044, 248319)


def partial_rope_reference(x, cos, sin, n_heads, rope_dim):
    """A deterministic torch stand-in for ``apply_partial_rope_prefill`` (rotate-half on the first ``rope_dim``)."""

    results = []
    for value, c, s in zip(x.torch_shards(), cos.torch_shards(), sin.torch_shards()):
        head = value[..., :rope_dim].float()
        half = rope_dim // 2
        rotated = torch.cat([-head[..., half:], head[..., :half]], dim=-1)
        roped = (head * c.float() + rotated * s.float()).to(value.dtype)
        results.append(torch.cat([roped, value[..., rope_dim:]], dim=-1))
    return FakeTensor(results, x.dtype, x.layout)


def qsa_verify_constants(rows: int):
    contract = FakeContract()
    position_constants = qsa_module.Qwen38TTNNQSAPositionConstants.build("mesh", contract, VERIFY_BLOCKS)
    chunk_constants = qsa_module.Qwen38TTNNQSAChunkConstants.build("mesh", contract, VERIFY_BLOCKS)
    verify_constants = qsa_module.Qwen38TTNNQSAVerifyConstants.build("mesh", contract, chunk_constants, rows=rows)
    return position_constants, chunk_constants, verify_constants


def make_qsa_module(fake_ttnn) -> qsa_module.Qwen38TTNNQSA:
    module = object.__new__(qsa_module.Qwen38TTNNQSA)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.layer_index = 3
    module.allocated_context = VERIFY_CONTEXT
    module.allocated_compressed_blocks = VERIFY_BLOCKS
    module.rms_norm_eps = 1e-6
    module.collective_topology = "linear"
    module.compute_config = "compute_config"
    module.projection_compute_config = "compute_config"
    module.compressed_row_memory_config = "compressed_row"
    module._next_epoch = 1
    module._live_epochs = set()
    module._live_generic_epochs = set()
    module._live_views = {}
    module._protected_views = set()
    torch.manual_seed(51)
    module.weights = SimpleNamespace(index_k_norm=replicated(1.0 + 0.1 * torch.randn(1, 1, 1, 128), BF16))
    module.slot_zero = u32_scalar(0)
    return module


class KVStream:
    """The sequential 1-row reference: per-position packed KV rows, raw keys and RoPE tables (per device)."""

    def __init__(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.packed = [bf16_randn(VERIFY_CONTEXT, 512) for _ in range(TP)]  # [v | k] per position, one per device
        self.raw = bf16_randn(VERIFY_CONTEXT, 128)  # replicated raw index keys
        self.cos, self.sin = bf16_randn(VERIFY_CONTEXT, 64, scale=0.7), bf16_randn(VERIFY_CONTEXT, 64, scale=0.7)

    def rope_rows(self, positions: list[int]) -> tuple[FakeTensor, FakeTensor]:
        index = torch.tensor(positions).clamp(max=VERIFY_CONTEXT - 1)
        return (
            replicated(self.cos[index].reshape(1, 1, 32, 64), BF16),
            replicated(self.sin[index].reshape(1, 1, 32, 64), BF16),
        )

    def compressed_block(self, module, fake_ttnn, block: int) -> torch.Tensor:
        """The reference compressed row of ``block``: the 0.25-select sum in fp32 (one bf16 rounding), the norm and
        the block-start RoPE, through the same fake ops the device path runs."""

        keys = self.raw[4 * block : 4 * block + 4].float()
        pooled = replicated((0.25 * keys).sum(dim=0, keepdim=True).to(torch.bfloat16).reshape(1, 1, 1, 128), BF16)
        normalized = fake_ttnn.rms_norm(pooled, epsilon=module.rms_norm_eps, weight=module.weights.index_k_norm)
        cos, sin = self.rope_rows([4 * block] * 32)
        cos = replicated(cos.torch_shards()[0][:, :, :1], BF16)
        sin = replicated(sin.torch_shards()[0][:, :, :1], BF16)
        return partial_rope_reference(normalized, cos, sin, 1, 64).torch_shards()[0].reshape(128)


def seed_qsa_caches(module, fake_ttnn, stream: KVStream, state, verify_state, position: int, lane: int = 0) -> None:
    """The device caches of ``lane`` after the sequential decode of positions 0 .. P - 1 (finite garbage past P)."""

    base = lane * VERIFY_CONTEXT
    for target, source in zip(state.packed_kv_cache.locals, stream.packed):
        target[0, 0, base : base + position] = source[:position]
        target[0, 0, base + position : base + VERIFY_CONTEXT] = 7.0  # stale rows: never named until rewritten
    complete_blocks = position // 4
    for block in range(complete_blocks):
        row = stream.compressed_block(module, fake_ttnn, block)
        for target in state.compressed_index_cache.locals:
            target[lane, 0, block] = row
    for target in state.compressed_index_cache.locals:
        target[lane, 0, complete_blocks:] = 9.0
    history = torch.zeros(32, 128, dtype=torch.bfloat16)
    for back in range(1, 4):
        if position - back >= 0:
            history[3 - back] = stream.raw[position - back]
    for target in verify_state.raw_history.locals:
        target[0, lane] = history


def run_qsa_verify_pass(
    module, fake_ttnn, stream: KVStream, state, verify_state, position: int, rows: int, constants, *, single_row=False
) -> None:
    """The two verify cache writers at ``position`` on the stream's rows P .. P + R - 1 (rows past R zero)."""

    position_constants, chunk_constants, verify_constants = constants
    verify = qsa_module.derive_qsa_verify_inputs(
        u32_scalar(position), position_constants, verify_constants, single_row=single_row
    )
    raw = torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16)
    raw[0, 0, :rows] = stream.raw[position : position + rows]
    block_start_cos, block_start_sin = stream.rope_rows(
        [4 * (position // 4) + 4 * i for i in range(8)] + [4 * (position // 4)] * 24
    )
    module._write_compressed_index_verify(
        state, verify_state, replicated(raw, BF16), block_start_cos, block_start_sin, verify, chunk_constants
    )
    keys, values = [], []
    for device in range(TP):
        packed = torch.zeros(32, 512, dtype=torch.bfloat16)
        packed[:rows] = stream.packed[device][position : position + rows]
        values.append(packed[:, :256].reshape(1, 1, 32, 256).clone())
        keys.append(packed[:, 256:].reshape(1, 1, 32, 256).clone())
    module._write_packed_kv_verify(state, FakeTensor(keys, BF16, TILE, 1), FakeTensor(values, BF16, TILE, 1), verify)
    verify.deallocate()


def accept_pattern_rows(pattern: tuple[int, ...], ids: tuple[int, ...]):
    """Target, draft and alignment ids whose draft j matches its target exactly where ``pattern[j]`` is 1."""

    k = len(pattern)
    cycle = itertools.cycle(ids)
    targets = [next(cycle) for _ in range(k + 1)]
    drafts = []
    for j, match in enumerate(pattern):
        drafts.append(targets[j] if match else next(value for value in ids if value != targets[j]))
    alignment = [next(cycle) for _ in range(k + 1)]
    return targets, drafts, alignment


def make_mtp_input(fake_ttnn) -> mtp_module.Qwen38TTNNMTPInput:
    mixer = object.__new__(mtp_module.Qwen38TTNNMTPInput)
    mixer.mesh_device = "mesh"
    mixer.mesh_contract = FakeContract()
    mixer.collective_topology = "linear"
    mixer.compute_config = "compute_config"
    mixer.projection_compute_config = "compute_config"
    torch.manual_seed(93)
    mixer.weights = SimpleNamespace(
        released=False,
        epsilon=1e-6,
        embedding_norm_scale=sharded(1.0 + 0.1 * torch.randn(1, 1, 1, 2560), dtype=FP32),
        hidden_norm_scale=sharded(1.0 + 0.1 * torch.randn(1, 1, 4, 2560), dtype=FP32),
        fc_embedding=sharded(bf16_randn(1, 1, 2560, 2560, scale=2560**-0.5)),
        fc_hidden=sharded(bf16_randn(1, 1, 2560, 2560, scale=2560**-0.5)),
    )
    return mixer


def make_gr_module(fake_ttnn) -> gr_module.Qwen38TTNNGatedResidual:
    torch.manual_seed(7)
    width, partial = gr_module.FLAT_LOCAL_WIDTH, gr_module.PARTIAL_WIDTH
    weights = gr_module.Qwen38TTNNGatedResidualWeights(
        norm_scale=FakeTensor([(torch.randn(1, 4, 1, 640) * 0.1 + 0.25) for _ in range(TP)], FP32, TILE, 3),
        down_inject=FakeTensor(
            [bf16_randn(1, 1, width, partial, scale=width**-0.5) for _ in range(TP)], BF16, TILE, 2
        ),
        up=FakeTensor([bf16_randn(1, 1, partial, width, scale=partial**-0.5) for _ in range(TP)], BF16, TILE, 3),
        replicated_anchor=FakeTensor([torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE),
        epsilon=1e-6,
        layer_index=0,
        block="attn",
        namespace="backbone",
    )
    module = object.__new__(gr_module.Qwen38TTNNGatedResidual)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.weights = weights
    module.tt_ccl = SimpleNamespace(
        get_num_links=lambda axis: 1,
        get_and_cycle_ag_semaphore_handles=lambda axis: "ag",
        get_and_cycle_barrier_semaphore_handle=lambda axis: "barrier",
    )
    module.collective_topology = "linear"
    module.compute_config = "compute"
    module.down_inject_act_memory_config = fake_ttnn.DRAM_MEMORY_CONFIG
    module.down_inject_program_config = "down"
    module.up_act_memory_config = fake_ttnn.DRAM_MEMORY_CONFIG
    module.up_program_config = "up"
    module.norm_scale_flat = fake_ttnn.experimental.view(weights.norm_scale, (1, 1, 1, width))
    return module


# --------------------------------------------------------------------------- the MTP draft stand-ins

DRAFT_VOCAB = 64  # the stub model's vocabulary (ids stay small; the id plumbing is exact copies, not arithmetic)
DRAFT_WIDTH = 640


class DraftWorld:
    """The deterministic stand-ins around the draft body: an embedding table, a row-wise 'layer', a vocabulary
    projection for the greedy resolve, and recorders of the per-row positions the body derives."""

    def __init__(self, seed: int = 11) -> None:
        torch.manual_seed(seed)
        self.table = bf16_randn(DRAFT_VOCAB, DRAFT_WIDTH * TP, scale=0.5)
        self.layer_scale = (1.0 + 0.25 * torch.rand(1, 4, 1, DRAFT_WIDTH * TP)).to(torch.bfloat16)
        self.vocab = bf16_randn(DRAFT_WIDTH * TP, DRAFT_VOCAB, scale=DRAFT_WIDTH**-0.5)
        self.positions: list[dict[str, int]] = []
        self.rope_requests: list[list[int]] = []

    def embed(self, token_row: FakeTensor) -> FakeTensor:
        assert token_row.dtype is FP32 and token_row.layout == TILE and token_row.shape == (1, 1, 1, 32)
        ids = token_row.torch_shards()[0].reshape(-1).long()
        rows = torch.zeros(32, DRAFT_WIDTH * TP, dtype=torch.bfloat16)
        keep = ids >= 0
        rows[keep] = self.table[ids[keep]]
        return sharded(rows.reshape(1, 1, 32, DRAFT_WIDTH * TP))

    def embed_one(self, token_row: FakeTensor) -> FakeTensor:
        """The 1-row form: lane 0's embedding as ``[1,1,1,W]`` (row 0 of :meth:`embed`)."""

        return sharded(cat_shards(self.embed(token_row))[:, :, :1])

    def mix(self, embedding_rows: FakeTensor, residual_rows: FakeTensor) -> FakeTensor:
        rows = embedding_rows.shape[2]
        assert embedding_rows.shape == (1, 1, rows, DRAFT_WIDTH) and residual_rows.shape == (1, 4, rows, DRAFT_WIDTH)
        embedding = cat_shards(embedding_rows).float().repeat(1, 4, 1, 1)
        return sharded((cat_shards(residual_rows).float() + embedding).to(torch.bfloat16))

    def layer(self, residual: FakeTensor) -> FakeTensor:
        rows = cat_shards(residual).float()
        return sharded((rows * self.layer_scale.float() + torch.roll(rows, 1, dims=3) * 0.125).to(torch.bfloat16))

    def final(self, residual: FakeTensor) -> FakeTensor:
        return sharded(cat_shards(residual).float().sum(dim=1, keepdim=True).to(torch.bfloat16))

    def resolve(self, hidden: FakeTensor) -> FakeTensor:
        logits = cat_shards(hidden).float() @ self.vocab.float()  # [1,1,rows,VOCAB]
        return replicated(logits.argmax(dim=3).float().reshape(1, 1, 1, -1), FP32, ROW_MAJOR)

    # The world is the 1-row LM head the draft rows use: logits -> candidates -> the FP32 TILE token row (column 0 =
    # the id).
    def __call__(self, hidden: FakeTensor):
        assert hidden.shape == (1, 1, 1, DRAFT_WIDTH)
        return SimpleNamespace(tensor=replicated(cat_shards(hidden).float() @ self.vocab.float(), FP32, TILE))

    def greedy_candidates(self, logits, *, values_by_gather=False):
        assert values_by_gather, "the draft rows take the gathered row maxima"
        token = float(logits.tensor.torch_shards()[0].argmax(dim=3).reshape(-1)[0])
        return SimpleNamespace(
            local_indices=replicated(torch.full((1, 1, 1), token), FP32, ROW_MAJOR),
            local_values=replicated(torch.zeros(1, 1, 1, 1), FP32, ROW_MAJOR),
        )

    def resolve_greedy_on_device(self, candidates) -> FakeTensor:
        row = torch.zeros(1, 1, 1, 32)
        row[..., 0] = candidates.local_indices.torch_shards()[0].reshape(-1)[0]
        return replicated(row, FP32, TILE)


class RowsMixer:
    """A stand-in mixer with the 1-row call and the 32-row ``rows`` form over the same row math."""

    def __init__(self, rows_form) -> None:
        self.rows = rows_form

    def __call__(self, *operands):
        return self.rows(*operands)


def draft_body_model(world: DraftWorld, contract, owner, rows_chunk) -> SimpleNamespace:
    """The model the draft bodies read: the world's embedding forms and LM head, ``rows_chunk`` as the RoPE table's
    rows, ``owner`` as the state owner; a poisoning re-raises its error."""

    def poisoned(operation, processed, error):
        raise error

    embedding = SimpleNamespace(
        embed_device_token_rows=world.embed,
        embed_device_token=world.embed_one,
        validate_token_row=lambda token_row, label="": embedding_module._validate_token_row(
            token_row, mesh_contract=contract, label=label
        ),
        host_verify_token_rows=embedding_module.Qwen38TTNNTokenEmbedding.host_verify_token_rows,
    )
    return SimpleNamespace(
        _state_owner=owner,
        poisoned=False,
        mesh_device="mesh",
        mesh_contract=contract,
        allocated_context=VERIFY_CONTEXT,
        model_io=SimpleNamespace(embedding=embedding, lm_head=world),
        rope_table=SimpleNamespace(rows_chunk=rows_chunk),
        qsa_position_constants=qsa_module.Qwen38TTNNQSAPositionConstants.build("mesh", contract, VERIFY_BLOCKS),
        layers=(),
        _mark_poisoned=poisoned,
    )


def draft_history(history: torch.Tensor, raw_rows: torch.Tensor, accepted: int, advances: int) -> torch.Tensor:
    """The draft raw history ``[32, 128]``: rows a + 1 .. a + 3 of the window ``[history[:3] | raw_rows]``, advanced
    ``advances`` times with zeros shifting in (the stub layer writes no raw rows)."""

    window = torch.cat([history[:3], raw_rows])
    expected = torch.zeros(32, 128, dtype=torch.bfloat16)
    expected[: 3 - advances] = window[accepted + 1 + advances : accepted + 4]
    return expected


def eager_draft_rows(world: DraftWorld, residual: FakeTensor, tokens: list[int], steps: int) -> list[int]:
    """The reference: ``steps`` MTP rows fed host-known tokens through the same stand-ins, one 1-row token row each."""

    rows = FakeTensor([F.pad(piece, (0, 0, 0, 31)) for piece in residual.torch_shards()], BF16, TILE, 3)
    ids: list[int] = []
    token = tokens[0]
    for _ in range(steps):
        token_row = replicated(embedding_module.Qwen38TTNNTokenEmbedding.host_verify_token_rows([token]), FP32)
        rows = world.layer(world.mix(world.embed(token_row), rows))
        token = int(world.resolve(world.final(rows)).torch_shards()[0].reshape(-1)[0])
        ids.append(token)
    return ids


# --------------------------------------------------------------------------- chunk-input constants and source pins

RESIDENT_BLOCKS = 32768 // qsa_module.COMPRESS_RATIO
CHUNK_UINT32_TEMPLATES = (
    "arange32_lanes",
    "block_start_lanes",
    "row_index_blocks",
    "arange_blocks_rows",
    "row_index_slots",
    "arange_slots_rows",
    "all_ones_rows",
)
HOST_TENSOR_CALLS = (
    "ttnn.from_torch",
    "ttnn.zeros",
    "ttnn.as_tensor",
    "ttnn.copy_host_to_device_tensor",
    "ttnn.to_torch",
    "torch.",
)


def source_functions(source: Path) -> dict[str, ast.FunctionDef]:
    """Every function of ``source`` by name (the first definition of a name)."""

    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef):
            found.setdefault(node.name, node)
    return found


def source_methods(source: Path, class_name: str) -> dict[str, ast.FunctionDef]:
    """The methods of the top-level class ``class_name`` of ``source`` by name."""

    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def source_segment(source: Path, node: ast.AST) -> str:
    """The function's source with every whitespace run collapsed (black may wrap a call over several lines)."""

    return " ".join(ast.get_source_segment(source.read_text(encoding="utf-8"), node).split()).replace("( ", "(")


def ordered_calls(node: ast.AST) -> list[str]:
    """The callees under ``node`` in source order."""

    return [
        ast.unparse(call.func)
        for call in sorted(
            (n for n in ast.walk(node) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
        )
    ]


def function_calls(function) -> list[str]:
    """The callees of a live function or method (its dedented source), in ``ast.walk`` order."""

    lines = inspect.getsource(function).splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    tree = ast.parse("\n".join(line[indent:] for line in lines))
    return [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]


# --------------------------------------------------------------------------- fixtures


def patch_ttnn(monkeypatch, fake_ttnn, *modules) -> None:
    """``modules`` run on ``fake_ttnn``; their replicate mesh mapper hands the fake's "replicate" tag."""

    for module in modules:
        monkeypatch.setattr(module, "ttnn", fake_ttnn)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)


@pytest.fixture
def rows_fake(monkeypatch):
    """The rows ttnn on the GDN, PLE, layer and decode-matmul modules, and its chunk kernel model."""

    chunk = FakeChunk()
    fake_ttnn = make_rows_ttnn(chunk)
    patch_ttnn(monkeypatch, fake_ttnn, gdn_module, ple_module, layer_module, decode_matmul_module)
    return SimpleNamespace(ttnn=fake_ttnn, chunk=chunk)


@pytest.fixture(scope="module")
def gdn_weights():
    oracle = gdn_oracle_weights()
    return oracle, device_gdn_weights(oracle)


@pytest.fixture
def verify_fake(monkeypatch):
    """The verify ttnn on the QSA, GDN, embedding, final mixer and MTP modules, with the torch partial RoPE."""

    fake_ttnn = make_verify_ttnn(FakeChunk())
    patch_ttnn(monkeypatch, fake_ttnn, qsa_module, gdn_module, embedding_module, final_mixer_module, mtp_module, mtp_v2)
    monkeypatch.setattr(qsa_module, "apply_partial_rope_prefill", partial_rope_reference)
    return fake_ttnn


@pytest.fixture
def gr_fake(monkeypatch):
    use_tile_padding(monkeypatch)
    fake_ttnn = make_gr_ttnn()
    monkeypatch.setattr(gr_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(decode_matmul_module, "ttnn", fake_ttnn)
    return fake_ttnn
