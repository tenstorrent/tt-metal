# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The torch-backed ttnn of the no-device tests: a four-device mesh of torch shards and the ttnn ops the model calls.

Every namespace starts from one base (constants, config tags, tensor plumbing) and adds the arithmetic and aliasing
of the device paths it stands in for: the fp32 row-serial ops of the rows paths (``make_rows_ttnn`` and its variants)
and the exact ops of the lanes and the QSA layer (``make_qsa_ttnn``: UINT32 arithmetic wraps, float ops run in fp64
rounded once, so no result depends on the shape a row sits in).  The model stand-ins several test files share and
the installers of conftest.py's fake fixtures follow the namespaces.
"""

from __future__ import annotations

import ast
import functools
import inspect
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.tt.gdn import Qwen38GDNDimensions, Qwen38GDNWeights
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38PLEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as decode_matmul_module
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as final_mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
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


def _tile_pages(x: torch.Tensor) -> torch.Tensor:
    """The TILE-layout page order of a logical tensor: batch dims, then the 32x32 tiles row by row (zero padded)."""

    rows, cols = x.shape[-2:]
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


# --------------------------------------------------------------------------- tensor plumbing (every namespace)


def _operand(b, index: int):
    return b.torch_shards()[index] if isinstance(b, FakeTensor) else b


def _into(results, dtype, layout, output=None):
    """``results`` as a new tensor, or copied into ``output`` (of their dtype and shape)."""

    if output is None:
        return FakeTensor(results, dtype, layout)
    assert output.dtype is dtype and output.shape == tuple(results[0].shape), (output.shape, tuple(results[0].shape))
    for target, result in zip(output.locals, results):
        target.copy_(result)
    return output


def _mapped(fn):
    """``fn`` on every shard; dtype and layout kept."""

    def op(tensor, *args, memory_config=None, **kwargs):
        return FakeTensor([fn(t, *args, **kwargs) for t in tensor.torch_shards()], tensor.dtype, tensor.layout)

    return op


def _elementwise(torch_op, wrap=False):
    """``torch_op`` on each shard and ``b``'s matching shard (or the scalar ``b``); ``wrap``: the result wraps to
    UINT32."""

    def op(a, b, *, memory_config=None):
        results = [torch_op(x, _operand(b, index)) for index, x in enumerate(a.torch_shards())]
        return FakeTensor([r & U32_MASK for r in results] if wrap else results, a.dtype, a.layout)

    return op


def _from_torch(wrap=False):
    """``ttnn.from_torch`` with the "replicate" tag or a ``("shard", dims)`` mapper; ``wrap``: UINT32 values wrap."""

    def from_torch(host, *, dtype, layout, device=None, memory_config=None, mesh_mapper):
        host = host.to(torch.int64) & U32_MASK if wrap and dtype is U32 else host.to(dtype.torch)
        if mesh_mapper == "replicate":
            return FakeTensor([host.clone() for _ in range(TP)], dtype, layout)
        kind, dims = mesh_mapper
        assert kind == "shard"
        return FakeTensor([piece.clone() for piece in torch.chunk(host, TP, dim=dims[1])], dtype, layout, dims[1])

    return from_torch


def _typecast(wrap=False):
    """``ttnn.typecast``; ``wrap``: a UINT32 result wraps."""

    def typecast(t, dtype, memory_config=None):
        if dtype is I32:
            assert all(int(x.max()) < 2**31 for x in t.torch_shards())
        results = [x.to(dtype.torch) for x in t.torch_shards()]
        return FakeTensor([r & U32_MASK for r in results] if wrap and dtype is U32 else results, dtype, t.layout)

    return typecast


def copy(source, target):
    for s, t in zip(source.torch_shards(), target.locals):
        t.copy_(s)
    return target


def deallocate(tensor):
    assert tensor.alive, "double deallocation"
    tensor.alive = False


def _fill(tensor, value, *, output_tensor=None, memory_config=None):
    assert output_tensor is tensor  # the callers fill in place
    for local in tensor.torch_shards():
        local.fill_(value)
    return tensor


def _fill_cache(cache, source, batch_idx: int):
    for target, part in zip(cache.locals, source.torch_shards()):
        assert part.shape[0] == 1 and tuple(part.shape[1:]) == tuple(target.shape[1:]), (part.shape, target.shape)
        target[batch_idx] = part[0]
    return cache


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


def _slice(tensor, start, end, *, memory_config=None, output_tensor=None):
    index = tuple(slice(s, e) for s, e in zip(start, end))
    return _into([t[index].clone() for t in tensor.torch_shards()], tensor.dtype, tensor.layout, output_tensor)


def _pad(tensor, padding, value=0.0, *, memory_config=None):
    flat = [amount for pair in reversed(padding) for amount in pair]
    return FakeTensor([F.pad(t, flat, value=value) for t in tensor.torch_shards()], tensor.dtype, tensor.layout)


def _concat(tensors, dim, *, memory_config=None):
    return FakeTensor(
        [torch.cat([t.torch_shards()[index] for t in tensors], dim=dim) for index in range(TP)],
        tensors[0].dtype,
        tensors[0].layout,
    )


def _reshape(alias=None):
    """``ttnn.reshape``.  A same-volume reshape is a copy, a torch view (``alias="view"``) or, to the input's own
    shape, the input itself (``alias="same"``: releasing the input then releases the result too, as on device).  The
    padded form ``reshape(t, logical, padded)`` reads the logical extents out of the padded tile grid (row 0 of every
    lane's tile for [1,B,32,128] -> [1,B,1,128]) and zero rows past a shorter input (the 1-row query as its tile)."""

    def reshape(t, shape, padded_shape=None, *, pad_value=None, memory_config=None):
        shape = tuple(int(item) for item in shape)
        if alias == "same" and shape == t.shape:
            return t
        shards = t.torch_shards()
        if math.prod(shape) == math.prod(t.shape):
            return FakeTensor(
                [x.reshape(shape) if alias == "view" else x.reshape(shape).clone() for x in shards], t.dtype, t.layout
            )
        padded = tuple(map(max, shape, t.shape))
        assert padded_shape is not None and len(shape) == len(t.shape) and tuple(padded_shape) == padded, (
            shape,
            padded_shape,
            t.shape,
        )
        extents = tuple(slice(0, size) for size in shape)
        pad = [amount for size, have in zip(reversed(shape), reversed(t.shape)) for amount in (0, max(size - have, 0))]
        return FakeTensor([F.pad(x[extents].clone(), pad) for x in shards], t.dtype, t.layout)

    return reshape


def view_tile_pages(tensor, shape, *, memory_config=None):
    """``ttnn.experimental.view``: the input's tile pages reinterpreted in the output's tile grid (a copy), so a view
    whose logical meaning differs from a torch reshape is caught."""

    return FakeTensor(
        [_from_tile_pages(_tile_pages(x), tuple(shape)) for x in tensor.torch_shards()], tensor.dtype, tensor.layout
    )


def _view_aliasing(t, shape, memory_config=None):
    """``ttnn.experimental.view`` on the same buffer: a torch view of the input where re-reading its tile pages in the
    new grid is a plain reshape (ROW_MAJOR pages always are), else the re-read pages (a copy, so a body that relied on
    the alias is caught)."""

    shape = tuple(int(item) for item in shape)
    shards = t.torch_shards()
    if t.layout != ROW_MAJOR:
        faithful = [_from_tile_pages(_tile_pages(x), shape) for x in shards]
        if not all(torch.equal(f, x.reshape(shape)) for f, x in zip(faithful, shards)):
            return FakeTensor(faithful, t.dtype, t.layout)
    return FakeTensor([x.reshape(shape) for x in shards], t.dtype, t.layout)


def use_aliasing_view(fake_ttnn) -> None:
    """This test's ``ttnn.experimental.view`` aliases its source where the tile-page view is a plain reshape (as on
    device; the lane import's views rely on it)."""

    fake_ttnn.experimental.view = _view_aliasing


def _all_gather(tensor, *, dim, cluster_axis, memory_config=None):
    full = torch.cat(tensor.torch_shards(), dim=dim)
    return FakeTensor([full.clone() for _ in range(TP)], tensor.dtype, tensor.layout)


def _gather(t, dim, index, memory_config=None):
    return FakeTensor(
        [torch.gather(x, dim, i.to(torch.int64)) for x, i in zip(t.torch_shards(), index.torch_shards())],
        t.dtype,
        t.layout,
    )


def _embedding(indices, table, *, layout=None, dtype=None, memory_config=None, lane_row=False):
    """The rows of ``table`` (flattened to ``[V, W]``) at ``indices`` as ``[1, n, W]``; ``lane_row``: the indices are
    the UINT32 ``[1, 1, 32]`` lane row."""

    assert not lane_row or (indices.dtype is U32 and indices.shape == (1, 1, 32)), (indices.dtype, indices.shape)
    results = []
    for index, rows in zip(indices.torch_shards(), table.torch_shards()):
        flat = rows.reshape(-1, rows.shape[-1])
        results.append(flat[index.reshape(-1).long()].reshape(1, index.numel(), rows.shape[-1]).clone())
    return FakeTensor(results, dtype or table.dtype, layout or table.layout)


def _paged_update_cache(cache, row, *, update_idxs_tensor):
    """Row ``user`` of ``row`` into cache user ``user`` at that user's index."""

    for target, source, index in zip(cache.locals, row.torch_shards(), update_idxs_tensor.torch_shards()):
        users = target.shape[0]
        rows = source.reshape(users, -1, source.shape[-1])
        for user, position in enumerate(index.reshape(-1).tolist()[:users]):
            target[user, 0, int(position)] = rows[user, 0]
    return cache


def _update_padded_kv_cache(cache, staging, slot, *args, valid_global=None, tp_axis=None):
    """The staging rows into the one-layer ROW_MAJOR cache at ``start``, in both argument forms: ``(slot, start,
    layer 0, 1 layer, axis)`` with device scalars and ``(slot 0, layer 0, 1 layer, start, axis)`` with ints."""

    if isinstance(slot, int):
        layer_idx, num_layers, start = args[:3]
        begins = [int(start)] * TP
    else:
        start, layer_idx, num_layers = args[:3]
        begins = [int(s.reshape(-1)[0]) for s in start.torch_shards()]
    assert (layer_idx, num_layers) == (0, 1) and staging.layout == ROW_MAJOR and cache.layout == ROW_MAJOR
    for target, source, begin in zip(cache.locals, staging.torch_shards(), begins):
        rows = source.shape[2]
        # The op's block-cyclic invariant: the cache rows are a multiple of the rows one call writes.
        assert begin % 32 == 0 and begin + rows <= target.shape[2] and target.shape[2] % rows == 0, (
            begin,
            rows,
            target.shape,
        )
        target[0, 0, begin : begin + rows] = source[0, 0]
    return cache


# --------------------------------------------------------------------------- arithmetic


def _unary(fn, acc=None):
    """``fn`` on every last-dim row (cast to ``acc`` when given), rounded to the tensor's dtype."""

    def op(tensor, *args, memory_config=None, output_tensor=None, **kwargs):
        results = []
        for x in tensor.torch_shards():
            rows = x.reshape(-1, x.shape[-1])
            rows = rows if acc is None else rows.to(acc)
            results.append(torch.stack([fn(row) for row in rows]).reshape(x.shape).to(x.dtype))
        return _into(results, tensor.dtype, tensor.layout, output_tensor)

    return op


def _softplus(tensor, *, beta, threshold, memory_config=None):
    """fp32 torch softplus; the device op's error is below the gate probe's 1e-3 envelope."""

    assert tensor.dtype is FP32 and (beta, threshold) == (1.0, 20.0), (tensor.dtype, beta, threshold)
    return _unary(lambda row: F.softplus(row, beta, threshold))(tensor)


def _binary(torch_op, zero_product=False):
    """Mixed float dtypes compute in fp32; the output keeps input_a's dtype unless ``dtype`` says otherwise.
    ``zero_product``: binary_ng's rule that a product is +0.0 whenever an input is zero (so a lane's keep-mask reset
    leaves +0.0, the zero source's bits, as the device does; torch alone gives -0.0 for x < 0)."""

    def op(
        a,
        b,
        *,
        memory_config=None,
        output_tensor=None,
        dtype=None,
        input_tensor_a_activations=None,
        input_tensor_b_activations=None,
        activations=None,
        fast_and_approximate_mode=None,
    ):
        assert not activations, activations  # no fused output activation is modeled
        out_dtype = dtype or a.dtype
        fp32 = dtype is FP32 or a.dtype is FP32 or (isinstance(b, FakeTensor) and b.dtype is FP32)
        compute = FP32 if fp32 else a.dtype
        results = []
        for index, x in enumerate(a.torch_shards()):
            y = _operand(b, index)
            zero = (x == 0) | (y == 0) if zero_product else None
            if input_tensor_a_activations:
                assert input_tensor_a_activations == [("SIGMOID", 4.0, 0.0)], input_tensor_a_activations
                x = torch.sigmoid(x)
            if isinstance(y, torch.Tensor):
                if input_tensor_b_activations:
                    assert input_tensor_b_activations == ["EXP"], input_tensor_b_activations
                    y = torch.exp(y)
                x, y = x.to(compute.torch), y.to(compute.torch)
            result = torch_op(x, y).to(out_dtype.torch)
            if zero is not None and result.is_floating_point():
                result = torch.where(zero, torch.zeros_like(result), result)
            results.append(result)
        return _into(results, out_dtype, a.layout, output_tensor)

    return op


def _binary_exact(torch_op):
    """UINT32 ops exact and wrapped; float ops in fp64 rounded once, with binary_ng's +0.0 product."""

    def op(a, b, *, memory_config=None, output_tensor=None, dtype=None, fast_and_approximate_mode=None):
        out_dtype = dtype or a.dtype
        results = []
        for index, x in enumerate(a.torch_shards()):
            y = _operand(b, index)
            if a.dtype is U32:
                result = torch_op(x, y)
            else:
                x, y = x.double(), (y.double() if isinstance(y, torch.Tensor) else y)
                result = torch_op(x, y)
                if torch_op is torch.mul:
                    result = torch.where((x == 0) | (y == 0), torch.zeros_like(result), result)
            results.append(result & U32_MASK if out_dtype is U32 else result.to(out_dtype.torch))
        return _into(results, out_dtype, a.layout, output_tensor)

    return op


def _compare(torch_op, u32=False):
    """``torch_op`` as ``dtype`` (default input_a's); ``u32``: the callers must ask for a UINT32 result."""

    def op(a, b, *, memory_config=None, dtype=None):
        assert dtype is U32 or not u32, dtype
        out_dtype = dtype or a.dtype
        return FakeTensor(
            [torch_op(x, _operand(b, index)).to(out_dtype.torch) for index, x in enumerate(a.torch_shards())],
            out_dtype,
            a.layout,
        )

    return op


def _linear(acc):
    """Every row on its own in ``acc``, rounded once to the output dtype."""

    def op(a, w, *, memory_config=None, program_config=None, compute_kernel_config=None, dtype=None):
        out_dtype = dtype or a.dtype
        results = []
        for x, wt in zip(a.torch_shards(), w.torch_shards()):
            weight = wt.reshape(wt.shape[-2], wt.shape[-1]).to(acc)
            out = torch.stack([(row[None] @ weight)[0] for row in x.reshape(-1, x.shape[-1]).to(acc)])
            results.append(out.reshape(*x.shape[:-1], weight.shape[-1]).to(out_dtype.torch))
        return FakeTensor(results, out_dtype, a.layout)

    return op


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
    return _into(results, out_dtype, a.layout, optional_output_tensor)


def _matmul_per_tile(
    a,
    b,
    *,
    memory_config=None,
    dtype=None,
    program_config=None,
    compute_kernel_config=None,
    optional_output_tensor=None,
):
    """Every (batch) 2-D product on its own in fp64, rounded once: the same call for a tile in a 1-row operand and
    for the same tile at batch index u of a lanes operand."""

    out_dtype = dtype or a.dtype
    results = []
    for x, y in zip(a.torch_shards(), b.torch_shards()):
        batch = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2])
        xs = x.double().expand(*batch, *x.shape[-2:]).reshape(-1, *x.shape[-2:])
        ys = y.double().expand(*batch, *y.shape[-2:]).reshape(-1, *y.shape[-2:])
        product = torch.stack([p @ q for p, q in zip(xs, ys)])
        results.append(product.reshape(*batch, x.shape[-2], y.shape[-1]).to(out_dtype.torch))
    return _into(results, out_dtype, a.layout, optional_output_tensor)


def _rms_norm(acc):
    """Every last-dim row normalized in ``acc``, rounded once to the tensor's dtype."""

    def op(tensor, *, epsilon, weight=None, memory_config=None, compute_kernel_config=None):
        weights = weight.torch_shards() if weight is not None else [None] * TP
        results = []
        for x, w in zip(tensor.torch_shards(), weights):
            rows = x.reshape(-1, x.shape[-1]).to(acc)
            y = torch.stack([row * torch.rsqrt(row.square().mean() + epsilon) for row in rows]).reshape(x.shape)
            results.append((y if w is None else y * w.reshape(-1).to(acc)).to(x.dtype))
        return FakeTensor(results, tensor.dtype, tensor.layout)

    return op


def _rms_norm_pre_all_gather(tensor, *, dtype, memory_config=None, compute_kernel_config=None):
    def stats(x: torch.Tensor) -> torch.Tensor:
        sums = x.float().square().sum(dim=-1, keepdim=True)  # one stat per (token, branch) row
        return F.pad(sums, (0, 31))

    return FakeTensor([stats(x) for x in tensor.torch_shards()], FP32, tensor.layout)


def _rms_norm_post_all_gather(
    tensor, gathered, *, epsilon, weight=None, memory_config=None, compute_kernel_config=None, dtype
):
    if weight is None:  # the GR's and the final mixer's unscaled unit: gamma is applied afterwards on the flat row
        weight = FakeTensor([torch.ones(1, 1, 1, tensor.shape[-1]) for _ in range(TP)], FP32, TILE)
    outputs = []
    for x, stats, w in zip(tensor.torch_shards(), gathered.torch_shards(), weight.torch_shards()):
        total = stats[..., ::32].sum(dim=-1, keepdim=True)  # the four devices' column-0 stats
        mean = total / (TP * x.shape[-1])
        outputs.append((x.float() * torch.rsqrt(mean + epsilon) * w.float()).to(dtype.torch))
    return FakeTensor(outputs, dtype, tensor.layout)


def _sum(acc=None):
    """``ttnn.sum`` rounded once to the tensor's dtype: the unscaled last-dim sum row by row (in the tensor's own
    dtype when ``acc`` is None), any other in ``acc`` (fp32 when None)."""

    def op(t, dim, keepdim=True, *, memory_config=None, compute_kernel_config=None, scalar=None):
        assert keepdim
        results = []
        for x in t.torch_shards():
            if dim in (3, -1) and scalar is None:
                rows = x.reshape(-1, x.shape[-1])
                y = torch.stack([(row if acc is None else row.to(acc)).sum() for row in rows])
                y = y.reshape(*x.shape[:-1], 1)
            else:
                y = x.to(acc or torch.float32).sum(dim=dim, keepdim=True) * (1.0 if scalar is None else scalar)
            results.append(y.to(x.dtype))
        return FakeTensor(results, t.dtype, t.layout)

    return op


def _device_sum(t, acc) -> torch.Tensor:
    """The devices' partials summed in device order in ``acc``, one rounding."""

    return functools.reduce(torch.add, [x.to(acc) for x in t.torch_shards()]).to(t.dtype.torch)


def _all_reduce(acc):
    def op(t, *, cluster_axis, memory_config=None, topology=None):
        total = _device_sum(t, acc)
        return FakeTensor([total.clone() for _ in range(TP)], t.dtype, t.layout)

    return op


def _reduce_scatter(acc):
    def op(t, *, dim, cluster_axis, memory_config=None, topology=None):
        total = _device_sum(t, acc)
        return FakeTensor([piece.clone() for piece in torch.chunk(total, TP, dim=dim)], t.dtype, t.layout, dim)

    return op


def _fast_reduce_nc(pad_rows=False):
    """fp32 accumulation in index order, one rounding at the end (fast_reduce_nc's contract).  ``pad_rows``: a TILE
    result below 32 rows reports the 32-row tile (zero pad rows), as on device; the rows bodies restore the logical
    rows with ttnn.reshape(t, logical, padded)."""

    def op(t, *, dims, output=None, compute_kernel_config=None, memory_config=None):
        (dim,) = dims
        shards = [
            functools.reduce(torch.add, x.float().unbind(dim)).unsqueeze(dim).to(x.dtype) for x in t.torch_shards()
        ]
        if pad_rows and t.layout == TILE and dim != 2 and shards[0].shape[2] % 32:
            shards = [F.pad(x, (0, 0, 0, 32 - x.shape[2] % 32)) for x in shards]
        return FakeTensor(shards, t.dtype, t.layout)

    return op


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


def _rotary_embedding_hf(x, cos, sin, *, is_decode_mode, memory_config=None):
    results = []
    for xs, cs, ss in zip(x.torch_shards(), cos.torch_shards(), sin.torch_shards()):
        x64, c64, s64 = xs.double(), cs.double(), ss.double()
        half = x64.shape[-1] // 2
        rotated = torch.cat([-x64[..., half:], x64[..., :half]], dim=-1)
        results.append((x64 * c64 + rotated * s64).to(xs.dtype))
    return FakeTensor(results, x.dtype, x.layout)


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


# --------------------------------------------------------------------------- the ttnn namespaces


def _ttnn(**ops) -> SimpleNamespace:
    """The constants, config tags and tensor plumbing every namespace below shares, with ``ops`` on top."""

    base = dict(
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
        MathFidelity=SimpleNamespace(HiFi4="HiFi4", HiFi2="HiFi2"),
        UnaryOpType=SimpleNamespace(EXP="EXP", SIGMOID="SIGMOID"),
        ShardStrategy=SimpleNamespace(HEIGHT="HEIGHT", WIDTH="WIDTH"),
        ShardOrientation=SimpleNamespace(ROW_MAJOR="ROW_MAJOR"),
        TensorMemoryLayout=SimpleNamespace(WIDTH_SHARDED="WIDTH_SHARDED", HEIGHT_SHARDED="HEIGHT_SHARDED"),
        BufferType=SimpleNamespace(DRAM="DRAM", L1="L1"),
        TensorTopology=TensorTopology,
        PlacementReplicate=PlacementReplicate,
        PlacementShard=PlacementShard,
        Shape=tuple,
        MeshShape=lambda *dims: tuple(dims),
        UnaryWithParam=lambda op, a, b: (op, a, b),
        ShardTensor2dMesh=lambda device, mesh_shape, dims: ("shard", dims),
        CoreGrid=lambda *, x, y: ("grid", x, y),
        CoreCoord=lambda x, y: ("core", x, y),
        CoreRange=lambda start, end: ("range", start, end),
        CoreRangeSet=lambda ranges: ("ranges", tuple(sorted(ranges))),
        ShardSpec=lambda grid, shape, orientation: ("shard_spec", grid, tuple(shape), orientation),
        MemoryConfig=lambda layout, buffer_type, shard_spec=None: ("memory_config", layout, buffer_type, shard_spec),
        num_cores_to_corerangeset=lambda cores, grid, row_wise: ("cores", cores, grid),
        create_sharded_memory_config=lambda shape, grid, strategy, orientation, use_height_and_width_as_shard_shape: (
            "sharded",
            tuple(shape),
            grid,
            strategy,
        ),
        WormholeComputeKernelConfig=lambda **fields: ("compute_kernel_config", tuple(sorted(fields.items()))),
        init_device_compute_kernel_config=lambda arch, **fields: ("compute", arch, tuple(sorted(fields.items()))),
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig=lambda **fields: ("dram_sharded", tuple(sorted(fields))),
        from_torch=_from_torch(),
        to_torch=lambda t, mesh_composer=None, **kwargs: t.torch_shards()[0].clone(),
        copy=copy,
        copy_host_to_device_tensor=copy,
        deallocate=deallocate,
        get_device_tensors=_get_device_tensors,
        zeros=_zeros,
        moreh_full=lambda shape, value, device, *, dtype, layout, memory_config: FakeTensor(
            [torch.full(tuple(shape), value, dtype=dtype.torch) for _ in range(TP)], dtype, layout
        ),
        fill=_fill,
        fill_cache=_fill_cache,
        clone=_mapped(torch.clone),
        to_layout=lambda t, layout, memory_config=None: FakeTensor(
            [x.clone() for x in t.torch_shards()], t.dtype, layout
        ),
        to_memory_config=_mapped(lambda x, cfg: x.clone()),
        typecast=_typecast(),
        reshape=_reshape(),
        unsqueeze_to_4D=_mapped(lambda x: x.reshape(1, *x.shape)),  # a view of the input, as on device
        slice=_slice,
        pad=_pad,
        concat=_concat,
        repeat=_mapped(lambda x, multipliers: x.repeat(*multipliers)),
        repeat_interleave=_mapped(lambda x, repeats, dim: torch.repeat_interleave(x, repeats, dim=dim)),
        transpose=_mapped(lambda x, a, b: x.transpose(a, b).contiguous()),
        permute=_mapped(lambda x, dims: x.permute(*dims).contiguous()),
        embedding=_embedding,
        gather=_gather,
        argmax=lambda t, dim=-1, keepdim=False: FakeTensor(
            [torch.argmax(x, dim=dim, keepdim=keepdim) for x in t.torch_shards()], U32, t.layout
        ),
        max=_mapped(lambda x, dim=-1, keepdim=False: x.max(dim=dim, keepdim=keepdim).values),
        minimum=_elementwise(lambda x, y: torch.clamp(x, max=y)),
        rsub=_mapped(lambda x, value: (value - x.double()).to(x.dtype)),
        all_gather=_all_gather,
        all_reduce=_all_reduce(torch.float32),
        mesh_partition=lambda t, *, dim, cluster_axis, memory_config=None: FakeTensor(
            [torch.chunk(x, TP, dim=dim)[i].clone() for i, x in enumerate(t.torch_shards())], t.dtype, t.layout, dim
        ),
        experimental=SimpleNamespace(
            view=view_tile_pages,
            fast_reduce_nc=_fast_reduce_nc(),
            paged_update_cache=_paged_update_cache,
            deepseek_prefill=SimpleNamespace(update_padded_kv_cache=_update_padded_kv_cache),
        ),
    )
    return SimpleNamespace(**(base | ops))


def make_rows_ttnn(chunk: FakeChunk) -> SimpleNamespace:
    """The fp32 row-serial ttnn of the GDN/PLE/layer rows paths and the MTP verify and draft bodies; ``chunk`` models
    the chunk kernel."""

    return _ttnn(
        multiply=_binary(torch.mul),
        add=_binary(torch.add),
        subtract=_binary(torch.sub),
        mac=lambda a, b, c: _binary(torch.add)(_binary(torch.mul)(a, b), c),
        sigmoid=_unary(torch.sigmoid),
        silu=_unary(F.silu),
        sqrt=_unary(torch.sqrt),
        abs=_unary(torch.abs),
        sign=_unary(torch.sign),
        softplus=_softplus,
        clamp=_mapped(lambda x, min=None, max=None: x.clamp(min=min, max=max)),
        le=_compare(torch.le),
        lt=_compare(torch.lt),
        ge=_compare(torch.ge),
        eq=_compare(torch.eq),
        bitwise_and=_elementwise(torch.bitwise_and),
        bitwise_or=_elementwise(torch.bitwise_or),
        bitwise_left_shift=_elementwise(torch.bitwise_left_shift),
        bitwise_right_shift=_elementwise(torch.bitwise_right_shift),
        unsqueeze_to_4D=_mapped(lambda x: x.reshape(1, 1, *x.shape[-2:]).clone()),
        sum=_sum(),
        linear=_linear(torch.float32),
        matmul=_matmul,
        rms_norm=_rms_norm(torch.float32),
        rms_norm_pre_all_gather=_rms_norm_pre_all_gather,
        rms_norm_post_all_gather=_rms_norm_post_all_gather,
        reduce_scatter=_reduce_scatter(torch.float32),
        transformer=chunk,
    )


def make_mtp_lanes_ttnn(chunk: FakeChunk) -> SimpleNamespace:
    """The rows ttnn with the device slice's no-op: over the whole tensor it returns its input (``slice.cpp``; the
    alias the owned slice guards against), into an output tensor it copies."""

    fake = make_rows_ttnn(chunk)

    def slice_(tensor, start, end, *, memory_config=None, output_tensor=None):
        if output_tensor is None and all(v == 0 for v in start) and tuple(end) == tensor.shape:
            return tensor
        return _slice(tensor, start, end, memory_config=memory_config, output_tensor=output_tensor)

    fake.slice = slice_
    return fake


def make_gdn_lanes_ttnn() -> SimpleNamespace:
    """The rows ttnn whose results never depend on the batch a row sits in: per-tile fp64 matmuls, binary_ng's +0.0
    product, and a same-shape reshape that is its input (releasing the input then releases the result too)."""

    fake = make_rows_ttnn(FakeChunk())
    fake.matmul = _matmul_per_tile
    fake.multiply = _binary(torch.mul, zero_product=True)
    fake.reshape = _reshape("same")
    fake.empty_like = lambda t, dtype=None, layout=None, memory_config=None: FakeTensor(
        [torch.empty_like(x) for x in t.torch_shards()], dtype or t.dtype, layout or t.layout
    )
    return fake


def make_gr_ttnn() -> SimpleNamespace:
    """The rows ttnn plus the GR ops of the 1-row path; every op is per row or per element."""

    fake = make_rows_ttnn(FakeChunk())
    fake.experimental.all_gather_async = lambda tensor, *, persistent_output_buffer, dim, **kwargs: _all_gather(
        tensor, dim=dim, cluster_axis=None
    )
    fake.experimental.fast_reduce_nc = _fast_reduce_nc(pad_rows=True)
    # The torch reshape the tile-page view is checked against: it gives the 1-row shapes the same values, but not the
    # rows' flat rows, which is the claim under test.
    fake.experimental.torch_view = lambda tensor, shape: FakeTensor(
        [x.reshape(tuple(shape)) for x in tensor.torch_shards()], tensor.dtype, tensor.layout
    )
    return fake


def make_qsa_ttnn() -> SimpleNamespace:
    """The exact ttnn of the QSA layer: UINT32 arithmetic wraps, float ops run per row in fp64 rounded once to the
    op's dtype, so no result depends on the tensor shape it sits in."""

    fake = _ttnn(
        from_torch=_from_torch(wrap=True),
        typecast=_typecast(wrap=True),
        add=_binary_exact(torch.add),
        subtract=_binary_exact(torch.sub),
        multiply=_binary_exact(torch.mul),
        mul=_binary_exact(torch.mul),
        bitwise_and=_elementwise(torch.bitwise_and, wrap=True),
        bitwise_or=_elementwise(torch.bitwise_or, wrap=True),
        bitwise_left_shift=_elementwise(torch.bitwise_left_shift, wrap=True),
        bitwise_right_shift=_elementwise(torch.bitwise_right_shift, wrap=True),
        eq=_compare(torch.eq, u32=True),
        lt=_compare(torch.lt, u32=True),
        ge=_compare(torch.ge, u32=True),
        sigmoid=_unary(torch.sigmoid, torch.float64),
        sum=_sum(torch.float64),
        rms_norm=_rms_norm(torch.float64),
        linear=_linear(torch.float64),
        matmul=_matmul_per_tile,
        all_reduce=_all_reduce(torch.float64),
        reduce_scatter=_reduce_scatter(torch.float64),
        embedding=functools.partial(_embedding, lane_row=True),
        transformer=SimpleNamespace(sparse_sdpa=_sparse_sdpa, SparseKVFormat=SimpleNamespace(BF16="BF16")),
    )
    fake.experimental.view = _view_aliasing
    fake.experimental.rotary_embedding_hf = _rotary_embedding_hf  # qwen36's rope_tp
    fake.experimental.indexer_score_dsa = _indexer_score_dsa
    fake.experimental.topk_large_indices = lambda scores, *, k: FakeTensor(
        [torch.topk(x.float(), k, dim=-1, largest=True, sorted=True).indices for x in scores.torch_shards()],
        U32,
        ROW_MAJOR,
    )
    return fake


def make_lanes_ttnn() -> SimpleNamespace:
    """The exact ttnn of the lane position rows, the lane derive and the lane greedy resolve: the QSA ops with a
    same-volume reshape that is a view of its input."""

    fake = make_qsa_ttnn()
    fake.reshape = _reshape("view")
    return fake


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
        row = self._row(token, context).contiguous()
        return bytearray(row.view(torch.int16).numpy().tobytes()), (EOS if context is None else context[1], token)

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


# --------------------------------------------------------------------------- QSA verify stand-ins

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


def make_qsa_module() -> qsa_module.Qwen38TTNNQSA:
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
    module, stream: KVStream, state, verify_state, position: int, rows: int, constants, *, single_row=False
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

    # The world is the 1-row LM head the draft rows use: logits -> candidates -> the FP32 TILE token row (id in col 0).
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


# --------------------------------------------------------------------------- source pins

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


# --------------------------------------------------------------------------- installing the fakes


def patch_ttnn(monkeypatch, fake_ttnn, *modules) -> None:
    """``modules`` run on ``fake_ttnn``; their replicate mesh mapper hands the fake's "replicate" tag."""

    for module in modules:
        monkeypatch.setattr(module, "ttnn", fake_ttnn)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)


def install_rows_fake(monkeypatch):
    """The rows ttnn on the GDN, PLE, layer and decode-matmul modules, and its chunk kernel model."""

    chunk = FakeChunk()
    fake_ttnn = make_rows_ttnn(chunk)
    patch_ttnn(monkeypatch, fake_ttnn, gdn_module, ple_module, layer_module, decode_matmul_module)
    return SimpleNamespace(ttnn=fake_ttnn, chunk=chunk)


def install_verify_fake(monkeypatch):
    """The rows ttnn on the QSA, GDN, embedding, final mixer and MTP modules, with the torch partial RoPE."""

    fake_ttnn = make_rows_ttnn(FakeChunk())
    patch_ttnn(monkeypatch, fake_ttnn, qsa_module, gdn_module, embedding_module, final_mixer_module, mtp_module, mtp_v2)
    monkeypatch.setattr(qsa_module, "apply_partial_rope_prefill", partial_rope_reference)
    return fake_ttnn
