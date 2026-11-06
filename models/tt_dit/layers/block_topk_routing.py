# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Block top-k routing for block-sparse ring attention, as two ``ttnn.generic_op`` programs around one all-gather.

Per device, from the local Q/K/V shards ``[1, H, rows, D]`` of an SP ring of ``S`` devices (``rows`` a multiple of 32,
cut into 64-row blocks, the last one possibly 32 rows), produces the inputs of ``ttnn.transformer.sparse_ring_sdpa``
and the coarse branch:

* ``block_pool`` (``kernels/block_pool_*.cpp``): per-block means of Q, K and V from per-block valid counts, into
  ``q_bar [1, H, nb_pad, D]`` and the stacked ``kv_bar [1, 2H, nb_pad, D]`` (K heads, then V heads).
* one all-gather of ``kv_bar`` over SP into the CCL manager's persistent ping-pong buffer.
* ``block_topk_route`` (``kernels/block_topk_route_*.cpp``): per (head, 32 local Q blocks): fp32 scores against
  every global block, a running top-k over the candidate columns, the k-th value, the uint32 block mask and,
  optionally, the coarse attention output ``softmax(Q̄ K̄ᵀ / sqrt(D)) V̄`` per block.

Global padded block ``s * nb_pad + b`` is block ``b`` of shard ``s``. A scored Q block keeps every candidate block
whose fp32 score reaches its k-th largest candidate score (ties at the k-th value are all kept) plus every exempt
block; an unscored Q block keeps nothing.

``update`` writes a request's geometry and ``k`` into buffers whose shapes depend only on the constructor
arguments, in place when traced, so one capture replays for every request; the sort always runs ``k_max`` wide.
"""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch

import ttnn

from ..utils.tensor import from_torch
from ..utils.tracing import StateTensor

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
TILE = 32
BLOCK = 64
TILE_BYTES_BF16 = 2048
TILE_BYTES_FP32 = 4096
# Powers of two, so they are exact in bf16, tf32 and fp32 and swamp any score they are added to.
EXCLUDED = -(2.0**100)
INCLUDED = 2.0**100
UNSCORED = 2.0**102
SENTINEL = -(2.0**126)
MASK_PAGE_TILES = 8
KTH_INDEX_WORDS = 16
N_CONST_TILES = 3 + TILE
L1_CB_BUDGET = 1_300_000


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@dataclass(frozen=True)
class RoutingRequest:
    """One request's routing geometry, per shard ``[S, nb_pad]`` in the padded block numbering."""

    counts: torch.Tensor
    candidate: torch.Tensor
    exempt: torch.Tensor
    scored: torch.Tensor
    k: int


def torch_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, request: RoutingRequest) -> dict:
    """The math of the device op on shards ``[S, H, rows, D]``.

    Returns ``q_bar``, ``k_bar``, ``v_bar`` ([S, H, nb_pad, D], bf16-rounded block means), ``scores``
    ([S, H, nb_pad, S * nb_pad] fp32, unscaled), ``block_mask`` (same shape, 0/1) and ``coarse``
    ([S, H, nb_pad, D] fp32, per local block).
    """
    S, H, rows, D = q.shape
    nb_pad = request.counts.shape[1]
    counts = request.counts.long()
    row = torch.arange(rows)
    block = row // BLOCK
    recip = torch.where(counts > 0, 1.0 / counts.clamp(min=1).double(), 0.0).float().bfloat16().float()
    valid = torch.zeros(S, rows, dtype=torch.bool)
    weight = torch.zeros(S, rows)
    for s in range(S):
        valid[s] = row - block * BLOCK < counts[s, block]
        weight[s] = torch.where(valid[s], recip[s, block], 0.0)

    def pool(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(S, H, nb_pad, D)
        weighted = torch.where(valid[:, None, :, None], x.float() * weight[:, None, :, None], 0.0)
        out.index_add_(2, block, weighted)
        return out.bfloat16().float()

    q_bar, k_bar, v_bar = pool(q), pool(k), pool(v)
    k_glob = k_bar.permute(1, 0, 2, 3).reshape(H, S * nb_pad, D)
    v_glob = v_bar.permute(1, 0, 2, 3).reshape(H, S * nb_pad, D)
    scores = torch.einsum("shbd,hcd->shbc", q_bar, k_glob)

    cand = request.candidate.reshape(-1)
    exempt = request.exempt.reshape(-1)
    real = (request.counts > 0).reshape(-1)
    cand_scores = scores[..., cand]
    kth = cand_scores.topk(request.k, dim=-1).values[..., -1:]
    kept = torch.zeros_like(scores, dtype=torch.bool)
    kept[..., cand] = cand_scores >= kth
    kept |= exempt
    kept &= request.scored[:, None, :, None]

    probs = torch.softmax(scores.masked_fill(~real, -float("inf")) / math.sqrt(D), dim=-1)
    coarse = torch.einsum("shbc,hcd->shbd", probs, v_glob)
    return {
        "q_bar": q_bar,
        "k_bar": k_bar,
        "v_bar": v_bar,
        "scores": scores,
        "block_mask": kept.float(),
        "coarse": coarse,
    }


class BlockTopKRouting:
    """Block top-k routing for one shape (local heads, head dim, local rows, SP factor, ``k_max``)."""

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        sp_axis: int,
        num_heads: int,
        head_dim: int,
        rows: int,
        k_max: int,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    ) -> None:
        assert rows % TILE == 0 and head_dim % TILE == 0, (rows, head_dim)
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.sp_factor = tuple(mesh_device.shape)[sp_axis]
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rows = rows
        self.nb_pad = _round_up(-(-rows // BLOCK), TILE)
        self.n_blocks_pad = self.sp_factor * self.nb_pad
        self.k_max = k_max
        self.kt = -(-k_max // TILE)
        assert self.kt * TILE <= self.n_blocks_pad, (k_max, self.n_blocks_pad)
        self.math_fidelity = math_fidelity

        self.pool_counts = StateTensor()
        self.block_counts = StateTensor()
        self.cand_t = StateTensor()
        self.col_bias = StateTensor()
        self.pad_row = StateTensor()
        self.row_bias = StateTensor()
        self.kth_index = StateTensor()

        # eye, SENTINEL, ones, then row_sel[r] with column r set: row_sel[r] @ T copies row r of T to every row.
        consts = torch.zeros(N_CONST_TILES, TILE, TILE)
        consts[0] = torch.eye(TILE)
        consts[1] = SENTINEL
        consts[2] = 1.0
        for r in range(TILE):
            consts[3 + r, :, r] = 1.0
        self.consts = from_torch(
            consts.reshape(1, 1, N_CONST_TILES * TILE, TILE), device=mesh_device, dtype=ttnn.float32
        )

        self._q_bar: ttnn.Tensor | None = None
        self._kv_bar: ttnn.Tensor | None = None
        self._pool_prog = None
        self._route_progs: dict[bool, ttnn.ProgramDescriptor] = {}

    # ------------------------------------------------------------------ per-request constants

    def update(self, request: RoutingRequest, *, traced: bool) -> None:
        """Write one request's geometry and ``k`` into the persistent buffers (in place when ``traced``)."""
        S, nb_pad, nbg = self.sp_factor, self.nb_pad, self.n_blocks_pad
        assert request.counts.shape == (S, nb_pad), (request.counts.shape, S, nb_pad)
        assert 1 <= request.k <= min(self.k_max, int(request.candidate.sum())), (request.k, self.k_max)
        device = self.mesh_device
        sharded = [self.sp_axis, None, None, None]

        counts = request.counts.to(torch.int32)
        cand = request.candidate.reshape(-1)
        exempt = request.exempt.reshape(-1)
        real = (request.counts > 0).reshape(-1)
        cand_bias = torch.where(cand, 0.0, EXCLUDED)
        col_bias = torch.where(exempt, INCLUDED, cand_bias)
        pad_bias = torch.where(real, 0.0, EXCLUDED)
        row_bias = torch.where(request.scored, 0.0, UNSCORED)
        kth_index = torch.zeros(1, 1, 1, KTH_INDEX_WORDS, dtype=torch.int32)
        kth_index[..., 0] = request.k - 1

        def upload(state: StateTensor, x: torch.Tensor, *, mesh_axes=None, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
            state.update(from_torch(x, device=device, dtype=dtype, layout=layout, mesh_axes=mesh_axes), traced=traced)

        rm = ttnn.ROW_MAJOR_LAYOUT
        upload(self.pool_counts, counts.reshape(S, 1, 1, nb_pad), mesh_axes=sharded, dtype=ttnn.uint32, layout=rm)
        upload(self.block_counts, counts.reshape(1, 1, 1, nbg), dtype=ttnn.uint32, layout=rm)
        upload(self.cand_t, cand_bias.reshape(1, 1, nbg, 1).expand(1, 1, nbg, TILE).contiguous())
        upload(self.col_bias, col_bias.reshape(1, 1, 1, nbg).expand(1, 1, TILE, nbg).contiguous())
        upload(self.pad_row, pad_bias.reshape(1, 1, 1, nbg).expand(1, 1, TILE, nbg).contiguous())
        upload(
            self.row_bias,
            row_bias.reshape(S, 1, nb_pad, 1).expand(S, 1, nb_pad, TILE).contiguous(),
            mesh_axes=sharded,
        )
        upload(self.kth_index, kth_index, dtype=ttnn.uint32, layout=rm)

    # ------------------------------------------------------------------ program helpers

    def _cores(self):
        g = self.mesh_device.compute_with_storage_grid_size()
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        return g.x, g.y, ranges

    @staticmethod
    def _cb(idx: int, pages: int, cores, page_size: int, dtype) -> ttnn.CBDescriptor:
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page_size)
        return ttnn.CBDescriptor(total_size=pages * page_size, core_ranges=cores, format_descriptors=[fmt])

    @staticmethod
    def _kernel(name: str, cores, ct_args: list[int], common: list[int], config) -> ttnn.KernelDescriptor:
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=list(ct_args),
            runtime_args=[],
            common_runtime_args=list(common),
            config=config,
        )

    @staticmethod
    def _accessor_args(*tensors: ttnn.Tensor) -> list[int]:
        return [a for t in tensors for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]

    def _compute_config(self, *, full_sync: bool) -> ttnn.ComputeConfigDescriptor:
        return ttnn.ComputeConfigDescriptor(
            math_fidelity=self.math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=full_sync,
        )

    def _alloc(self, shape: list[int], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), dtype, layout, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )

    # ------------------------------------------------------------------ block_pool

    def _pool_program(self, x: ttnn.Tensor, outs: list[ttnn.Tensor]) -> ttnn.ProgramDescriptor:
        H, D = self.num_heads, self.head_dim
        RT, DT, NBPT = self.rows // TILE, D // TILE, self.nb_pad // TILE
        nx, ny, cores = self._cores()
        ct = [H, RT, DT, NBPT, nx, ny, 3 * H * NBPT]
        acc_in = self._accessor_args(x, x, x, self.pool_counts.value)
        acc_out = self._accessor_args(*outs)
        bf16 = ttnn.bfloat16
        cbs = [
            self._cb(0, 3 * 4 * DT, cores, TILE_BYTES_BF16, bf16),
            self._cb(2, 3 * 4, cores, TILE_BYTES_BF16, bf16),
            self._cb(3, 1, cores, TILE_BYTES_BF16, bf16),
            self._cb(4, 1, cores, self.nb_pad * 4, ttnn.uint32),
            self._cb(5, 32, cores, TILE_BYTES_BF16, bf16),
            self._cb(16, 2 * DT, cores, TILE_BYTES_BF16, bf16),
        ]
        reader = self._kernel("block_pool_reader.cpp", cores, ct + acc_in, [0] * 4, ttnn.ReaderConfigDescriptor())
        writer = self._kernel("block_pool_writer.cpp", cores, ct + acc_out, [0] * 2, ttnn.WriterConfigDescriptor())
        compute = self._kernel("block_pool_compute.cpp", cores, ct, [], self._compute_config(full_sync=False))
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    def _run_pool(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, outs: list[ttnn.Tensor]) -> None:
        assert tuple(q.shape) == (1, self.num_heads, self.rows, self.head_dim), (q.shape, self.rows)
        if self._pool_prog is None:
            self._pool_prog = self._pool_program(q, outs)
        prog = self._pool_prog
        counts = self.pool_counts.value
        prog.kernels[0].common_runtime_args = [
            q.buffer_address(),
            k.buffer_address(),
            v.buffer_address(),
            counts.buffer_address(),
        ]
        prog.kernels[1].common_runtime_args = [o.buffer_address() for o in outs]
        ttnn.generic_op([q, k, v, counts, *outs], prog)

    def _pool_outputs(self) -> list[ttnn.Tensor]:
        H, D = self.num_heads, self.head_dim
        return [self._alloc([1, H, self.nb_pad, D]), self._alloc([1, 2 * H, self.nb_pad, D])]

    def pool(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Block means into fresh tensors: ``q_bar [1, H, nb_pad, D]`` and ``kv_bar [1, 2H, nb_pad, D]``."""
        outs = self._pool_outputs()
        self._run_pool(q, k, v, outs)
        return outs[0], outs[1]

    # ------------------------------------------------------------------ block_topk_route

    def _route_program(
        self, q_bar: ttnn.Tensor, kv_g: ttnn.Tensor, mask: ttnn.Tensor, coarse_out: ttnn.Tensor, coarse: bool
    ) -> ttnn.ProgramDescriptor:
        H, D = self.num_heads, self.head_dim
        DT, NBPT, NBGT, KT = D // TILE, self.nb_pad // TILE, self.n_blocks_pad // TILE, self.kt
        MC = min(MASK_PAGE_TILES, NBGT)
        assert NBGT % MC == 0 and DT <= 4, (NBGT, DT)
        nx, ny, cores = self._cores()
        units = H * NBPT
        scale = _f32_bits(1.0 / math.sqrt(D))
        ct = [H, NBPT, NBGT, DT, KT, MC, nx, ny, units, int(coarse), scale, self.rows // TILE]
        ins = self._route_inputs(q_bar, kv_g)
        acc_in = self._accessor_args(*ins)
        outs = [mask, coarse_out, self.kth_index.value]
        acc_out = self._accessor_args(*outs)
        bf16, f32 = ttnn.bfloat16, ttnn.float32
        mask_page = TILE * MC * TILE * 4
        sizes = [
            (0, DT, bf16),
            (1, 2 * DT, bf16),
            (2, N_CONST_TILES, f32),
            (7, 2, f32),
            (8, 2, f32),
            (9, 1, f32),
            (19, KT, f32),
            (20, 1, f32),
            (24, NBGT, f32),
            (25, KT, f32),
            (26, 1, f32),
            (28, 1, f32),
        ]
        if coarse:
            sizes += [(10, 2, f32), (11, 2 * DT, bf16), (12, 1, f32), (13, 1, f32), (14, 1, f32), (15, 1, f32)]
            sizes += [(17, 2 * DT, bf16), (18, 1, f32), (22, DT, bf16), (29, NBGT, f32)]
        cbs = [self._cb(i, n, cores, TILE_BYTES_FP32 if dt == f32 else TILE_BYTES_BF16, dt) for i, n, dt in sizes]
        cbs.append(self._cb(16, 2, cores, mask_page, ttnn.uint32))
        cbs.append(self._cb(21, 1, cores, KTH_INDEX_WORDS * 4, ttnn.uint32))
        total = sum(n * (TILE_BYTES_FP32 if dt == f32 else TILE_BYTES_BF16) for _, n, dt in sizes) + 2 * mask_page
        if total > L1_CB_BUDGET:
            raise ValueError(f"block_topk_route circular buffers need {total} B of L1 (> {L1_CB_BUDGET})")
        reader = self._kernel(
            "block_topk_route_reader.cpp", cores, ct + acc_in, [0] * len(ins), ttnn.ReaderConfigDescriptor()
        )
        writer = self._kernel(
            "block_topk_route_writer.cpp", cores, ct + acc_out, [0] * len(outs), ttnn.WriterConfigDescriptor()
        )
        config = self._compute_config(full_sync=True)
        # Exact fp32 copies and transposes for every tile the selection reads (scores, sort buffers, biases).
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (7, 8, 9, 19, 20, 24, 25, 26, 28):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        compute = self._kernel("block_topk_route_compute.cpp", cores, ct, [], config)
        return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    def _route_inputs(self, q_bar: ttnn.Tensor, kv_g: ttnn.Tensor) -> list[ttnn.Tensor]:
        return [
            q_bar,
            kv_g,
            self.consts,
            self.cand_t.value,
            self.col_bias.value,
            self.row_bias.value,
            self.pad_row.value,
        ]

    def _run_route(self, q_bar: ttnn.Tensor, kv_g: ttnn.Tensor, coarse: bool) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        H, D = self.num_heads, self.head_dim
        mask = self._alloc([1, H, self.nb_pad, self.n_blocks_pad], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        coarse_out = self._alloc([1, 1, self.rows, H * D]) if coarse else mask
        if coarse not in self._route_progs:
            self._route_progs[coarse] = self._route_program(q_bar, kv_g, mask, coarse_out, coarse)
        prog = self._route_progs[coarse]
        ins = self._route_inputs(q_bar, kv_g)
        prog.kernels[0].common_runtime_args = [t.buffer_address() for t in ins]
        kth_index = self.kth_index.value
        prog.kernels[1].common_runtime_args = [
            mask.buffer_address(),
            coarse_out.buffer_address(),
            kth_index.buffer_address(),
        ]
        ttnn.generic_op([*ins, kth_index, mask] + ([coarse_out] if coarse else []), prog)
        return mask, coarse_out

    # ------------------------------------------------------------------ forward

    def __call__(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        *,
        coarse: bool = True,
        section: Callable[[str], AbstractContextManager] = lambda name: nullcontext(),
    ) -> dict:
        """``block_mask`` uint32 [1, H, nb_pad, S * nb_pad], ``block_counts`` uint32 [1, 1, 1, S * nb_pad] and
        ``coarse_out`` [1, 1, rows, H * D] (each block's coarse output on all its rows, heads concatenated; None
        without ``coarse``). The caller owns and deallocates ``block_mask`` and ``coarse_out``. ``section(name)`` wraps
        the pool, gather and route steps (for timing)."""
        if self._q_bar is None:
            self._q_bar, self._kv_bar = self._pool_outputs()
        with section("pool"):
            self._run_pool(q, k, v, [self._q_bar, self._kv_bar])
        with section("gather"):
            kv_g = self.ccl_manager.all_gather_persistent_buffer(self._kv_bar, dim=2, mesh_axis=self.sp_axis)
        with section("route"):
            mask, coarse_out = self._run_route(self._q_bar, kv_g, coarse)
        return {
            "block_mask": mask,
            "block_counts": self.block_counts.value,
            "coarse_out": coarse_out if coarse else None,
        }
