# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Block top-k routing for block-sparse ring attention, as one ``ttnn.generic_op`` program.

Per device, from the local Q/K/V shards ``[1, H, rows, D]`` of an SP ring of ``S`` devices (``rows`` a multiple of 32,
cut into 64-row blocks, the last one possibly 32 rows), produces the inputs of ``ttnn.transformer.sparse_ring_sdpa``
and the coarse branch. The program's reader, compute and writer (``kernels/block_topk_route_*.cpp``) run two phases on
every core but the senders:

* pool (``kernels/block_pool_*.cpp``): per-block means of Q, K and V from per-block valid counts, into
  ``q_bar [1, H, nb_pad, D]`` and this device's shard slot of the gathered ``kv_g [1, 2H, S * nb_pad, D]`` (K heads,
  then V heads), the CCL manager's ping-pong buffer. Each core starts on one unit and claims the rest from a queue on
  the first sender core (``block_pool_queue.cpp``), K and V units first; sender cores (``block_pool_send.cpp``, one per
  link and direction) push each K/V unit over the fabric to the same slot of every peer's ``kv_g`` as soon as its
  writer flags it, while the Q units are still pooled, then bump this shard's arrival slot on the peer.
* route, once every pool writer has signalled a program-wide barrier: shards are scored as they arrive, this device's
  first; per (head, 32 local Q blocks): fp32 scores against every global block, a running top-k over the candidate
  columns, the k-th value, the uint32 block mask and, optionally, the coarse attention output
  ``softmax(Q̄ K̄ᵀ / sqrt(D)) V̄`` per block.

Global padded block ``s * nb_pad + b`` is block ``b`` of shard ``s``. A scored Q block keeps every candidate block
whose fp32 score reaches its k-th largest candidate score (ties at the k-th value are all kept) plus every exempt
block; an unscored Q block keeps every block (a dense row) when it holds a real token, and nothing when it is a
pad block.

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
# Row bias: ``thr - DENSE`` is below every column, ``thr + DENSE`` above every column.
DENSE = 2.0**102
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
    """One request's routing geometry, per shard ``[S, nb_pad]`` in the padded block numbering.

    ``scored = False`` on a block holding a real token is a dense query block (every key block kept).
    """

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
    unscored = ~request.scored[:, None, :].expand(kept.shape[:3])
    kept[unscored] = (request.counts > 0)[:, None, :, None].expand_as(kept)[unscored]

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
        self.topology = ccl_manager.topology
        self.num_links = ccl_manager.num_links
        assert self.topology in (ttnn.Topology.Ring, ttnn.Topology.Linear), self.topology

        # One arrival slot per shard of the ring, living for the run: wait-then-decrement carries a fast peer's early
        # increment for the next call over, so nothing is reset or allocated per call. All S slots share one L1 page
        # per core (slot s at +4 s): the pipeline keeps an instance per rung, and each page lowers the L1 top that
        # programs size their circular buffers against.
        nx, ny, cores = self._cores()
        self.arrived = from_torch(
            torch.zeros(nx * ny, self.sp_factor, dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [1, self.sp_factor], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        ttnn.synchronize_device(mesh_device)

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
            consts.reshape(1, 1, N_CONST_TILES * TILE, TILE), device=mesh_device, dtype=ttnn.bfloat16
        )

        self._q_bar: ttnn.Tensor | None = None
        self._route_progs: dict[bool, tuple] = {}

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
        row_bias = torch.where(request.scored, 0.0, torch.where(request.counts > 0, -DENSE, DENSE))
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
    def _kernel(
        name: str, cores, ct_args: list[int], common: list[int], config, runtime_args=None
    ) -> ttnn.KernelDescriptor:
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=list(ct_args),
            runtime_args=[] if runtime_args is None else runtime_args,
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

    def _devices(self) -> list[tuple[ttnn.MeshCoordinate, int]]:
        """Every device of the mesh with its ring index, its coordinate along the SP axis."""
        rows, cols = tuple(self.mesh_device.shape)
        coords = [ttnn.MeshCoordinate(i, j) for i in range(rows) for j in range(cols)]
        return [(c, c[self.sp_axis]) for c in coords]

    def _peer(self, coord: ttnn.MeshCoordinate, step: int) -> ttnn.MeshCoordinate:
        c = [coord[0], coord[1]]
        c[self.sp_axis] = (c[self.sp_axis] + step) % self.sp_factor
        return ttnn.MeshCoordinate(*c)

    def _targets(self, r: int) -> dict[int, list[tuple[int, bool]]]:
        """Per direction (+1, -1) the hop counts ring position ``r`` sends its shard to, each with whether that target
        takes only this direction's half of the tiles (the opposite position of an even ring, reached both ways)."""
        S = self.sp_factor
        if self.topology == ttnn.Topology.Linear:
            return {1: [(d, False) for d in range(1, S - r)], -1: [(d, False) for d in range(1, r + 1)]}
        near = [(d, False) for d in range(1, (S - 1) // 2 + 1)]
        if S % 2 == 0:
            near.append((S // 2, True))
        return {1: near, -1: near}

    def _hop_order(self, r: int) -> list[int]:
        """The ring's shards by hop distance from ``r``: ``r``, r + 1, r - 1, r + 2, r - 2, ..."""
        S, ring = self.sp_factor, self.topology == ttnn.Topology.Ring
        order = [r]
        for d in range(1, S):
            for s in (r + d, r - d):
                s = s % S if ring else s
                if 0 <= s < S and s not in order:
                    order.append(s)
        return order

    def _expected_arrivals(self, m: int) -> list[int]:
        """Increments ring position ``m`` receives per call on each shard's arrival slot: one per sender core (link
        and direction) whose targets include ``m``."""
        S = self.sp_factor
        expected = [0] * S
        for s in range(S):
            for step, targets in self._targets(s).items():
                expected[s] += self.num_links * sum((s + step * d) % S == m for d, _ in targets)
        return expected

    def _ring_position(self, r: int) -> ttnn.MeshCoordinateRange:
        """Every device at ring position ``r``: the line across the other mesh axis."""
        rows, cols = tuple(self.mesh_device.shape)
        lo, hi = [0, 0], [rows - 1, cols - 1]
        lo[self.sp_axis] = hi[self.sp_axis] = r
        return ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*lo), ttnn.MeshCoordinate(*hi))

    @staticmethod
    def _mesh_program(progs: list[tuple]) -> ttnn.MeshProgramDescriptor:
        """Programs per device (a coordinate) or per ring position (a range), kept for the run and rebound in place
        per call. Each carries its structural hash, so a call does not rehash every descriptor."""
        mesh = ttnn.MeshProgramDescriptor()
        for where, prog in progs:
            prog.custom_program_hash = ttnn.compute_program_descriptor_hash(prog)
            if isinstance(where, ttnn.MeshCoordinate):
                where = ttnn.MeshCoordinateRange(where, where)
            mesh[where] = prog
        return mesh

    # ------------------------------------------------------------------ program

    def pool(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Block means and their gather into fresh tensors: ``q_bar [1, H, nb_pad, D]`` and
        ``kv_g [1, 2H, S * nb_pad, D]``. The route phase runs too, its mask dropped: its gate consumes the call's arrival
        counts, so the next call waits for its own shards."""
        H, D = self.num_heads, self.head_dim
        q_bar = self._alloc([1, H, self.nb_pad, D])
        kv_g = self._alloc([1, 2 * H, self.n_blocks_pad, D])
        mask, _ = self._run(q, k, v, q_bar, kv_g, coarse=False)
        ttnn.deallocate(mask)
        return q_bar, kv_g

    def _sender_cores(self) -> list[ttnn.CoreCoord]:
        """The last ``2 * num_links`` cores of the grid: per link a forward and a backward sender."""
        nx, ny, _ = self._cores()
        n = 2 * self.num_links
        assert n < nx and ny > 1, (n, nx, ny)
        return [ttnn.CoreCoord(x, ny - 1) for x in range(nx - n, nx)]

    def _programs(
        self,
        x: ttnn.Tensor,
        q_bar: ttnn.Tensor,
        kv_g: ttnn.Tensor,
        mask: ttnn.Tensor,
        coarse_out: ttnn.Tensor,
        coarse: bool,
    ) -> tuple[ttnn.MeshProgramDescriptor, list[tuple[list[int], list[int], list[int], list[int]]]]:
        """The program per ring position, or per device where the fabric connection args of a ring position's devices
        differ, and, in the same order, the reader's common args after the addresses, the writer's after the addresses
        and the writer's after ``q_bar`` and ``kv_g``, and the pool reader's after its addresses. Pool and route run on
        every core but the senders, which push this device's shard to the peers."""
        H, D, L = self.num_heads, self.head_dim, self.num_links
        RT, DT, NBPT, NBGT, KT = self.rows // TILE, D // TILE, self.nb_pad // TILE, self.n_blocks_pad // TILE, self.kt
        MC = min(MASK_PAGE_TILES, NBGT)
        assert NBGT % MC == 0 and DT <= 4, (NBGT, DT)
        nx, ny, grid = self._cores()
        senders = self._sender_cores()
        cores = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 2)),
                ttnn.CoreRange(ttnn.CoreCoord(0, ny - 1), ttnn.CoreCoord(senders[0].x - 1, ny - 1)),
            ]
        )
        send_cores = ttnn.CoreRangeSet([ttnn.CoreRange(senders[0], senders[-1])])
        ncores = nx * ny - len(senders)
        units = H * NBPT
        pool_units, kv_units = 3 * H * NBPT, 2 * H * NBPT
        scale = _f32_bits(1.0 / math.sqrt(D))
        pool_ct = [H, RT, DT, NBPT, nx, ny, pool_units, ncores, NBGT]
        ins = self._route_inputs(q_bar, kv_g)
        outs = [mask, coarse_out, self.kth_index.value]
        bf16, f32 = ttnn.bfloat16, ttnn.float32
        mask_page = TILE * MC * TILE * 4
        sizes = [
            (0, DT, bf16),
            (1, 2 * DT, bf16),
            (2, N_CONST_TILES, bf16),
            (7, 2, f32),
            (8, 2, f32),
            (9, 1, f32),
            (19, 1, f32),
            (20, 1, f32),
            (24, NBGT, f32),
            (25, KT, f32),
            (26, 1, f32),
            (28, 1, f32),
        ]
        if coarse:
            sizes += [(10, 2, f32), (11, 2 * DT, bf16), (12, 1, f32), (13, 1, f32), (14, 1, f32), (15, 1, f32)]
            sizes += [(17, 2 * DT, bf16), (18, 1, f32), (22, DT, bf16)]
        # The pool phase's buffers: data, pooling tiles, zero tile, full-block pooling table, output.
        sizes += [(3, 3 * 4 * DT, bf16), (4, 3 * 4, bf16), (5, 1, bf16), (27, 32, bf16), (30, 2 * DT, bf16)]

        # p = exp(...) is produced and consumed in chunks of PC column tiles; between chunks the DT + 1 partial
        # sums spill to cb 6 (on the senders, cb 6 is their packet buffer).
        def chunk_sizes(pc: int) -> list[tuple]:
            if not coarse:
                return []
            return [(29, pc, f32)] + ([(6, DT + 1, f32)] if pc < NBGT else [])

        # The claimed units' pages and the end page.
        units_bytes = (pool_units + 1) * 16
        # The senders' page (block_pool_units.hpp SenderPage): the per-unit completion flags, the page's address in its
        # own 16 B word, then for the queue one request word and one 16 B reply word per pool core.
        flag_bytes = (_round_up(kv_units, 4) + 4 + _round_up(ncores, 4) + 4 * ncores) * 4

        def total(pc: int) -> int:
            cb_bytes = sum(
                n * (TILE_BYTES_FP32 if dt == f32 else TILE_BYTES_BF16) for _, n, dt in sizes + chunk_sizes(pc)
            )
            return cb_bytes + 2 * mask_page + KTH_INDEX_WORDS * 4 + self.nb_pad * 4 + units_bytes + flag_bytes

        chunks = [pc for pc in range(NBGT, 0, -4) if NBGT % pc == 0]
        PC = next((pc for pc in chunks if total(pc) <= L1_CB_BUDGET), chunks[-1])
        if total(PC) > L1_CB_BUDGET:
            raise ValueError(f"block_topk_route circular buffers need {total(PC)} B of L1 (> {L1_CB_BUDGET})")
        sizes += chunk_sizes(PC)
        ct = [H, NBPT, NBGT, DT, KT, MC, nx, ny, units, int(coarse), scale, RT, PC, ncores]
        reader_ct = ct + self._accessor_args(*ins) + pool_ct + self._accessor_args(x, x, x, self.pool_counts.value)
        writer_ct = ct + self._accessor_args(*outs) + pool_ct + self._accessor_args(q_bar, kv_g)
        cbs = [self._cb(i, n, cores, TILE_BYTES_FP32 if dt == f32 else TILE_BYTES_BF16, dt) for i, n, dt in sizes]
        cbs.append(self._cb(16, 2, cores, mask_page, ttnn.uint32))
        cbs.append(self._cb(21, 1, cores, KTH_INDEX_WORDS * 4, ttnn.uint32))
        cbs.append(self._cb(23, 1, cores, self.nb_pad * 4, ttnn.uint32))
        cbs.append(self._cb(31, pool_units + 1, cores, 16, ttnn.uint32))
        packet_tiles = min(4, ttnn.get_tt_fabric_max_payload_size_bytes() // TILE_BYTES_BF16)
        assert DT <= packet_tiles, (DT, packet_tiles)
        cbs.append(self._cb(6, 2 * DT, send_cores, TILE_BYTES_BF16, bf16))
        cbs.append(self._cb(3, 1, send_cores, flag_bytes, ttnn.uint32))
        config = self._compute_config(full_sync=True)
        # Exact fp32 copies and transposes for every tile the selection reads (scores, sort buffers, biases) and for
        # the reloaded coarse partial sums.
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (6, 7, 8, 9, 19, 20, 24, 25, 26, 28):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        # 0: the gate's release count, multicast from core (0, 0) as the shards come in. 1: the pool-to-route barrier,
        # one increment per pool core on the gate, then multicast by it. 2: one increment per sender once its page is
        # zeroed. 3: the queue's reply to the pool reader's latest request. 4: the unit pages the pool reader has
        # written. 5: the senders' page address. 6 (senders): the page is zeroed, for the queue. The senders' fabric
        # connections take their next free ids.
        semaphores = [
            ttnn.SemaphoreDescriptor(id=0, core_ranges=grid, initial_value=0),
            ttnn.SemaphoreDescriptor(id=1, core_ranges=grid, initial_value=0),
            ttnn.SemaphoreDescriptor(id=2, core_ranges=cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=3, core_ranges=cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=4, core_ranges=cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=5, core_ranges=cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=6, core_ranges=send_cores, initial_value=0),
        ]
        lo = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        hi = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(nx - 1, ny - 1))
        sender_nocs = [self.mesh_device.worker_core_from_logical_core(c) for c in senders]
        slots = [self.arrived.buffer_address() + 4 * s for s in range(self.sp_factor)]
        send_ct = [H, RT, NBPT, NBGT, DT, ncores, L] + self._accessor_args(kv_g)
        queue_core = ttnn.CoreRangeSet([ttnn.CoreRange(senders[0], senders[0])])
        # The pool cores as two rectangles, each noc corners and core count: rows 0..ny-2, and the top row left of
        # the senders.
        pool_rects = []
        for (x0, y0), (x1, y1), count in (
            ((0, 0), (nx - 1, ny - 2), nx * (ny - 1)),
            ((0, ny - 1), (senders[0].x - 1, ny - 1), senders[0].x),
        ):
            a = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x0, y0))
            b = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x1, y1))
            pool_rects += [a.x, a.y, b.x, b.y, count]

        progs, tails = [], []
        for r in range(self.sp_factor):
            hop = self._hop_order(r)
            reader_tail = [lo.x, lo.y, hi.x, hi.y, *hop, *slots, *self._expected_arrivals(r)]
            writer_tail = [*hop, lo.x, lo.y]
            pool_tail = [r, len(senders), *[a for c in sender_nocs for a in (c.x, c.y)]]
            pool_reader_tail = [sender_nocs[0].x, sender_nocs[0].y]
            targets = self._targets(r)
            slot = slots[r]
            built = []
            for coord in [c for c, ring in self._devices() if ring == r]:
                send_args = ttnn.RuntimeArgs()
                for i, core in enumerate(senders):
                    link, step = i // 2, 1 if i % 2 == 0 else -1
                    pairs = [a for d, half in targets[step] for a in (d, int(half))]
                    send_args[core.x][core.y] = [r, slot, link, i % 2, len(pairs) // 2, *pairs]
                kernels = [
                    self._kernel(
                        "block_topk_route_reader.cpp",
                        cores,
                        reader_ct,
                        [0] * (len(ins) + len(reader_tail) + 4 + len(pool_reader_tail)),
                        ttnn.ReaderConfigDescriptor(),
                    ),
                    self._kernel(
                        "block_topk_route_writer.cpp",
                        cores,
                        writer_ct,
                        [0] * (len(outs) + len(writer_tail) + 2 + len(pool_tail)),
                        ttnn.WriterConfigDescriptor(),
                    ),
                    self._kernel("block_topk_route_compute.cpp", cores, ct + pool_ct, [], config),
                    self._kernel(
                        "block_pool_send.cpp",
                        send_cores,
                        send_ct,
                        [0, lo.x, lo.y, *pool_rects],
                        ttnn.WriterConfigDescriptor(),
                        runtime_args=send_args,
                    ),
                    self._kernel(
                        "block_pool_queue.cpp",
                        queue_core,
                        [H, RT, NBPT, ncores],
                        [],
                        ttnn.ReaderConfigDescriptor(),
                    ),
                ]
                prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
                src = self.mesh_device.get_fabric_node_id(coord)
                connections = []
                for i, core in enumerate(senders):
                    link, step = i // 2, 1 if i % 2 == 0 else -1
                    if not targets[step]:
                        continue
                    dst = self.mesh_device.get_fabric_node_id(self._peer(coord, step))
                    fabric = list(ttnn.setup_fabric_connection(self.mesh_device, src, dst, link, prog, core))
                    prog.kernels[3].runtime_args[core.x][core.y].extend(
                        [1, *fabric, 0] if step == 1 else [0, 1, *fabric]
                    )
                    connections.append(fabric)
                built.append((coord, prog, connections))
            if all(c == built[0][2] for _, _, c in built):
                built = [(self._ring_position(r), built[0][1], None)]
            for where, prog, _ in built:
                progs.append((where, prog))
                tails.append((reader_tail, writer_tail, pool_tail, pool_reader_tail))
        return self._mesh_program(progs), tails

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

    def _run(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, q_bar: ttnn.Tensor, kv_g: ttnn.Tensor, coarse: bool
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        H, D = self.num_heads, self.head_dim
        assert tuple(q.shape) == (1, H, self.rows, D), (q.shape, self.rows)
        mask = self._alloc([1, H, self.nb_pad, self.n_blocks_pad], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        coarse_out = self._alloc([1, 1, self.rows, H * D]) if coarse else mask
        if coarse not in self._route_progs:
            self._route_progs[coarse] = self._programs(q, q_bar, kv_g, mask, coarse_out, coarse)
        mesh, tails = self._route_progs[coarse]
        counts = self.pool_counts.value
        pool_ins = [q, k, v, counts]
        ins = self._route_inputs(q_bar, kv_g)
        addresses = [t.buffer_address() for t in ins]
        pool_addresses = [t.buffer_address() for t in pool_ins]
        kth_index = self.kth_index.value
        writer_args = [mask.buffer_address(), coarse_out.buffer_address(), kth_index.buffer_address()]
        for (_, prog), (reader_tail, writer_tail, pool_tail, pool_reader_tail) in zip(mesh, tails):
            prog.kernels[0].common_runtime_args = [*addresses, *reader_tail, *pool_addresses, *pool_reader_tail]
            prog.kernels[1].common_runtime_args = [*writer_args, *writer_tail, *addresses[:2], *pool_tail]
            sender = prog.kernels[3]
            sender.common_runtime_args = [addresses[1], *sender.common_runtime_args[1:]]
        ttnn.generic_op([*pool_ins, *ins, kth_index, mask] + ([coarse_out] if coarse else []), mesh)
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
        the one program, section ``routing`` (for timing).

        The gathered ``kv_g`` is the CCL manager's ping-pong buffer, fetched per call: consecutive calls alternate
        between its two buffers, so a peer's push for the next call never lands in a buffer this device still reads
        (to push call i + 2 into call i's buffer, a peer must have finished call i + 1, which waited for this device's
        shard i + 1, sent only after its call i)."""
        H, D = self.num_heads, self.head_dim
        if self._q_bar is None:
            self._q_bar = self._alloc([1, H, self.nb_pad, D])
        kv_g = self.ccl_manager.get_ag_ping_pong_buffer((1, 2 * H, self.nb_pad, D), 2, self.sp_axis)
        with section("routing"):
            mask, coarse_out = self._run(q, k, v, self._q_bar, kv_g, coarse)
        return {
            "block_mask": mask,
            "block_counts": self.block_counts.value,
            "coarse_out": coarse_out if coarse else None,
        }
