# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 decode routing: the gate's selection and the dispatch's routing setup shaped for one tile of users.

The batched decode step (``tt/v41/batch_decode.py``) runs the prefill MoE (``TtMoe``) on a tile whose first
``users_per_rank`` rows are real. Its gate and routing setup are chunk-shaped: ``moe_grouped_topk`` sorts a tile
row's 384 columns on one core (~80 us), ``masked_bincount`` histograms the rows on a 64-core grid and wants a multiple
of 64 of them (~90 us), ``offset_cumsum`` all-gathers the histogram over the SP column and prefix-sums it on one core
(~85 us): ~250 us a layer for 4 tokens a chip. This module replaces them with the same results from two small
programs and one tile's all-gather:

* :func:`router_select`: one core per token row, exact fp32 top-k of ``score + bias`` and the normalised, scaled
  weights of the unbiased scores (``kernels/router_select.cpp``, after ``ssinghal/dsv4p1``'s router); the pad rows
  of the tile get the sentinel index. The scores are ``sqrt(softplus(logits))`` on the SFPU as the gate kernel
  computes them (``softplus_tile(1, 1, 20)``, ``sqrt_tile``), done by one fused eltwise op before the program.
* :func:`route_offsets`: the index tiles of the SP column's chips, all-gathered (one 2 KB page a chip), to the
  dispatch offsets, totals and region offsets of ``offset_cumsum`` in one program on one core
  (``kernels/route_offsets.cpp``); the chip's position in the column comes from a per-chip pin page.

:func:`install_decode_routing` swaps a ``TtV41Moe``'s gate and routing setup for :class:`DecodeGate` and
:class:`DecodeRoutingSetup`, which keep ``TtMoe.forward``'s contract (weights row-major bf16 ``[T, K]``, indices
uint16 TILE ``[T, K]``, three ``[1, E]`` uint32 row-major pages), so the dispatch, experts, combine and reduce run
unchanged. The tile needs no 64-row padding any more: ``tt/v41/decode.py`` sets ``MOE_ROWS`` to the decode tile's
32 rows when the path is on (``V41_DECODE_ROUTING=1``).
"""

import os
import struct

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import _cores
from models.demos.deepseek_v3_d_p.tt.v41.layout import SP_AXIS

_KERNELS = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_TILE = 32
# the index tile is gathered repeated: ``ttnn.all_gather`` of one or two pages a chip hangs the traced step now and
# then; four pages (two a link) have not. The knobs are for bisecting it.
_GATHER_TILES = int(os.environ.get("V41_DECODE_ROUTING_GATHER_TILES", 4))
_GATHER_LINKS = int(os.environ.get("V41_DECODE_ROUTING_GATHER_LINKS", 0))  # 0: the fabric's links


def _bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _accessors(*tensors) -> list:
    args = []
    for t in tensors:
        args += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    return args


def _cb(index: int, size: int, cores, dtype=ttnn.uint32) -> ttnn.CBDescriptor:
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)
    return ttnn.CBDescriptor(total_size=size, core_ranges=cores, format_descriptors=[fmt])


def router_select(rank, score, k: int, eps: float, scale: float, real_rows: int):
    """``rank``, ``score``: fp32 TILE ``[T, E]`` DRAM (``T <= 32``, ``E % 32 == 0``; ``rank`` = ``score`` + bias).
    Rows below ``real_rows`` get the exact top-``k`` of ``rank`` (lowest index first on a tie) and bf16 weights
    ``score[idx] / (sum + eps) * scale``; the rows after get weight 0 and index ``E``, the gate's padding sentinel.
    Returns ``(weights, indices)``: row-major bf16 ``[T, k]`` and uint16 TILE ``[T, k]``, DRAM."""
    T, E = int(rank.shape[-2]), int(rank.shape[-1])
    assert len(rank.shape) == 2 and tuple(score.shape) == (T, E), (tuple(rank.shape), tuple(score.shape))
    assert (T <= _TILE or T % _TILE == 0) and E % _TILE == 0 and k <= 16, (T, E, k)
    assert rank.dtype == ttnn.float32 and score.dtype == ttnn.float32, (rank.dtype, score.dtype)
    assert rank.layout == ttnn.TILE_LAYOUT and score.layout == ttnn.TILE_LAYOUT, (rank.layout, score.layout)
    assert 0 < real_rows <= T, (real_rows, T)
    device = rank.device()
    weights = ttnn.allocate_tensor_on_device(
        ttnn.Shape([T, k]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    indices = ttnn.allocate_tensor_on_device(
        ttnn.Shape([T, k]), ttnn.uint16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    coords, cores = _cores(device, T)
    args = ttnn.RuntimeArgs()
    for t, (cx, cy) in enumerate(coords):
        args[cx][cy] = [
            rank.buffer_address(),
            score.buffer_address(),
            weights.buffer_address(),
            indices.buffer_address(),
            t,
        ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNELS}/router_select.cpp",
        core_ranges=cores,
        compile_time_args=[0, 1, 2, E // _TILE, k, _bits(eps), _bits(scale), real_rows]
        + _accessors(rank, score, weights, indices),
        runtime_args=args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cbs = [_cb(0, E * 4, cores, ttnn.float32), _cb(1, k * 64, cores, ttnn.float32), _cb(2, 128, cores)]
    ttnn.generic_op([rank, score, weights, indices], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs))
    return weights, indices


def route_offsets(gathered, mask, pin, num_experts: int, k: int, real_rows: int, experts_per_chip: int, rows: int):
    """``gathered``: uint16 TILE ``[1, 1, H * T, k]`` DRAM (``T`` a multiple of 32), the SP all-gather of the
    column's index tiles (pages ``h * T / 32 ..`` = chip ``h``'s; rows below ``real_rows`` of each chip real). ``mask``: int32 row-major ``[1, E]`` or ``[1, E + 1]``
    (this group's row of the expert dispatch table). ``pin``: uint32 row-major ``[1, 1, 1, 8]`` per chip, element 0
    the chip's position in the column. Returns ``(offsets, totals, region)``, uint32 row-major ``[1, E]`` DRAM, as
    ``offset_cumsum`` returns them."""
    E = num_experts
    tpc = -(-rows // _TILE)  # index tiles per chip
    H = int(gathered.shape[2]) // (tpc * _TILE)
    assert gathered.dtype == ttnn.uint16 and gathered.layout == ttnn.TILE_LAYOUT, (gathered.dtype, gathered.layout)
    assert int(gathered.shape[2]) % _TILE == 0 and int(gathered.shape[3]) == k, tuple(gathered.shape)
    assert mask.dtype == ttnn.int32 and mask.layout == ttnn.ROW_MAJOR_LAYOUT and int(mask.shape[-1]) >= E, mask
    assert pin.dtype == ttnn.uint32 and pin.layout == ttnn.ROW_MAJOR_LAYOUT, pin
    assert E % experts_per_chip == 0, (E, experts_per_chip)
    device = gathered.device()
    outs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, E]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(3)
    ]
    offsets, totals, region = outs
    coords, cores = _cores(device, 1)
    args = ttnn.RuntimeArgs()
    cx, cy = coords[0]
    args[cx][cy] = [
        gathered.buffer_address(),
        mask.buffer_address(),
        pin.buffer_address(),
        offsets.buffer_address(),
        totals.buffer_address(),
        region.buffer_address(),
    ]
    mask_bytes = int(mask.shape[-1]) * 4
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNELS}/route_offsets.cpp",
        core_ranges=cores,
        compile_time_args=[0, E, k, real_rows, H, experts_per_chip, mask_bytes, tpc]
        + _accessors(gathered, mask, pin, offsets, totals, region),
        runtime_args=args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    scratch = 64 + H * tpc * 2048 + (mask_bytes + 63) // 64 * 64 + 64 + 3 * E * 4
    cbs = [_cb(0, scratch, cores)]
    ttnn.generic_op(
        [gathered, mask, pin, offsets, totals, region], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
    )
    return offsets, totals, region


class DecodeGate:
    """``TtMoEGatePrefill`` for the decode tile: its gate matmul and TP all-reduce, then the scores on the SFPU
    and :func:`router_select`. ``real_rows``: the tile's real rows when a padding config is given (the batched
    step's ``users_per_rank``; every row is real without one). Everything else is the wrapped gate's."""

    def __init__(self, gate, real_rows: int | None):
        self._gate = gate
        self._real_rows = real_rows
        self._zero = None  # fp32 zeros [T, E], the operand the scores' activations ride on
        self._bias32 = None  # the gate's bias as fp32, rebuilt when the bias tensor is swapped (VL rows)
        self._bias_src = None

    def __getattr__(self, name):
        return getattr(self._gate, name)

    @property
    def bias(self):
        return self._gate.bias

    @bias.setter
    def bias(self, value):
        self._gate.bias = value

    def _constants(self, logits):
        bias = self._gate.bias
        if self._bias_src is not bias or self._bias32 is None:
            self._bias32 = ttnn.typecast(bias, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._bias_src = bias
        if self._zero is None:
            T, E = int(logits.shape[-2]), int(logits.shape[-1])
            self._zero = ttnn.from_torch(
                torch.zeros(T, E),
                device=logits.device(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(logits.device()),
            )
        return self._zero, self._bias32

    def __call__(self, x, actual_isl=None, padding_side="right", padding_config=None, input_ids=None, actual_start=0):
        gate, cfg = self._gate, self._gate.config
        assert input_ids is None, "the decode gate selects by top-k, not by hash"
        logits = gate._device_matmul(x)  # bf16 TILE [T, E], TP all-reduced
        zero, bias32 = self._constants(logits)
        T = int(logits.shape[-2])
        rows = self._real_rows if padding_config is not None and self._real_rows else T
        l32 = ttnn.typecast(logits, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        activations = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT),
        ]
        score = ttnn.add(l32, zero, input_tensor_a_activations=activations, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(l32)
        rank = ttnn.add(score, bias32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        weights, indices = router_select(rank, score, cfg.n_activated_experts, 1e-20, cfg.route_scale, rows)
        ttnn.deallocate(rank)
        ttnn.deallocate(score)
        return weights, indices, logits

    forward = __call__


class DecodeRoutingSetup:
    """``TtMoERoutingSetup`` for the decode tile: the index tile all-gathered over the SP column and
    :func:`route_offsets`. The fourth return (the chip's own histogram) is None: ``TtMoe.forward`` discards it."""

    def __init__(self, setup, mesh_device, tokens: int, k: int, real_rows: int | None):
        """``tokens``, ``k``: the index tile's rows (the MoE's tokens per chip) and columns (experts per token)."""
        self._setup = setup
        self._real_rows = real_rows
        sp = tuple(mesh_device.shape)[SP_AXIS]
        pins = torch.zeros(sp, 1, 1, 8, dtype=torch.int32)
        pins[:, 0, 0, 0] = torch.arange(sp)
        self._pin = ttnn.from_torch(
            pins,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None)),
        )
        self._num_links = fabric_num_links()
        # the all-gather keeps two semaphores in main L1 per program-cache entry, placed when the entry is first
        # built: built inside a decode step (under the step's transient L1) they land low and the routed experts'
        # CBs, which take nearly all of L1, clash with them at 40 layers; so the entry is built here, on an empty L1,
        # with a tile of the step's shape (offset_cumsum's gather has the prefill's shape and is built there)
        warm = ttnn.from_torch(
            torch.zeros(1, 1, tokens, k, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.deallocate(self._gather(warm))
        ttnn.deallocate(warm)

    def __getattr__(self, name):
        return getattr(self._setup, name)

    def _gather(self, tile):
        """The SP column's index tiles on every chip, each repeated ``_GATHER_TILES`` times:
        ``[1, 1, H * rows * _GATHER_TILES, k]`` (pages ``h * _GATHER_TILES ..`` = chip h's)."""
        if _GATHER_TILES > 1:
            tile = ttnn.concat([tile] * _GATHER_TILES, dim=2)
        return ttnn.all_gather(
            tile,
            dim=2,
            cluster_axis=SP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=_GATHER_LINKS or self._num_links,
        )

    def __call__(self, ttnn_top_k_experts_indices, num_routed_experts: int, num_experts_per_tok: int):
        indices = ttnn_top_k_experts_indices
        assert indices.layout == ttnn.TILE_LAYOUT and indices.dtype == ttnn.uint16, (indices.layout, indices.dtype)
        T, K = int(indices.shape[-2]), int(indices.shape[-1])
        assert (T <= _TILE or T % _TILE == 0) and K == num_experts_per_tok, (T, K, num_experts_per_tok)
        rows = self._real_rows or T
        # ``ttnn.all_gather`` as ``offset_cumsum`` gathers its histogram (its own resident semaphores, see __init__);
        # on ``TtCCL``'s two cycled ``all_gather_async`` semaphore pairs of the SP axis, which the sampler's gather
        # shares, 40 of these a step hung the captured step's back-to-back replays at 40 layers
        gathered = self._gather(ttnn.reshape(indices, (1, 1, T, K)))
        offsets, totals, region = route_offsets(
            gathered,
            self._setup.experts_in_dispatch_group,
            self._pin,
            num_routed_experts,
            K,
            rows,
            self._setup.experts_per_chip,
            T * _GATHER_TILES,
        )
        ttnn.deallocate(gathered)
        return offsets, totals, region, None

    forward = __call__


def install_decode_routing(v41_moe, real_rows: int | None) -> None:
    """Swap ``v41_moe.moe``'s gate and routing setup for the decode-shaped ones. ``real_rows``: the tile's real
    rows under the batched step's padding config (``users_per_rank``); None when every row is real."""
    moe = v41_moe.moe
    if isinstance(moe.gate, DecodeGate):
        return
    moe.gate = DecodeGate(moe.gate, real_rows)
    moe.routing_setup = DecodeRoutingSetup(
        moe.routing_setup, moe.mesh_device, moe.gate.config.sp_dim, moe.num_experts_per_tok, real_rows
    )
