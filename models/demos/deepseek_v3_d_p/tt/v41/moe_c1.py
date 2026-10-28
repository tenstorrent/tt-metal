# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 decode MoE for one live row (``V41_DECODE_C1_MOE=1``).

The batched decode step runs the MoE on a tile whose first ``users_per_rank`` rows of every rank are real, whatever
is live: at c1 the 31 idle rows (token 0) select 6 experts each and go through the dispatch, the experts and the
combine ( ~0.5 to 0.65 ms a layer). With one live row the fabric chain is not needed:
:class:`C1Moe` gathers the row to every chip (a one-hot matmul picks it out of the live rank's tile into row 0 of
an otherwise zero tile, an SP all-gather of the tiles and a one-hot sum over the gathered rows broadcast it, a TP
all-gather completes it), runs the gate and the exact top-k redundantly, selects the row's experts that are this
chip's (:func:`c1_select`, one core) into the routed expert FFN's ``counts`` and the weights of the local sum,
runs ``unified_routed_expert_moe`` on a 12-region buffer that holds the row at the head of each region (one tile
row an expert), sums the local outputs with a ``[32, 384] x [384, 5120]`` matmul, sums the partials over the 32
chips (TP reduce-scatter, then the SP all-gather and one-hot sum again), adds the shared expert's output and
places the result at the live row of the live rank's tile with a second one-hot matmul (the other rows zero). The
gate, ``router_select``, ``fp8_qdq``, the shared expert and the routed expert FFN are the MoE's own; the
collectives are ``V41Collectives``' and the pool's. Measured (``tests/v41/test_v41_moe_c1.py`` under trace): ~500
us a layer against ~1,120 for the 32-row path on random rows; the routed expert FFN is ~90 us an expert (20 MB of
bfp4 weights), the shared expert 72, the two SP all-gathers 44 each, the gate 45, 28 programs.

The one-hot tiles ``sel`` (row 0 = e_u on the live rank) and ``put`` (row u = e_0 on the live rank) come from
two integer rows of the step's host inputs (:meth:`C1Moe.host_rows`) through :meth:`C1Moe.onehots` (two
``ttnn.embedding`` lookups of an identity table), once a step.

The shared expert enters exactly once, by either of two contracts. With
the MoE's own ``TtSharedExpert`` it returns its TP-reduce-scattered ``[32, hidden/tp]`` output and the c1 path
adds it after the 32-chip sum (step 12). With ``V41_SHARED_IN_FFN=rs`` (the served catalog)
``DecodeSharedExpert`` returns ``None`` and parks its ``[32, hidden]`` partial (this chip's TP slice of the
intermediate, not yet summed over TP) in the decode tail's ``DecodeReduce.pending_shared``; the c1 path takes
it from there, masks it to the chips of SP rank 0 (``sp0 @ partial``: a per-chip one-hot constant, so the SP
all-gather-and-sum of step 11 counts it once) and adds it to the local routed partial before step 11's TP
reduce-scatter, which is what the 32-row tail does with it. That replaces the shared expert's own reduce-scatter
(~16 us) and the final add with a one-hot matmul and an add (~4 us).

``rows`` above 1 (the speculative verify step, ``tt/v41/spec_verify.py``: ``1 + K`` rows of one user) runs the same chain on ``T <= rows`` live rows at rows ``0 .. T - 1`` of the tile: ``sel`` picks them,
the SP gather-sum keeps row ``j`` of every rank (the ``sum8`` matrix is the identity repeated), ``router_select``
ranks ``T`` real rows, ``c1_select`` (``T`` a runtime argument) marks an expert of any row with count ``T`` and writes
row ``t``'s weight at ``(t, 32 l)``, ``R`` puts all the rows at the head of every region (the FFN computes a selected
expert on the ``T`` rows at once: one tile of M, the same cost as one row; a row that did not select the expert
has weight 0), the shared expert runs row-wise, and ``put`` places rows ``0 .. T - 1``. The composite FFN carries
the rows (``C1FfnMatvec`` is one row's); the one-row call is unchanged.

With ``V41_DECODE_C1_SHARED=split`` the c1 path runs the shared expert itself on a 32-way
split of its intermediate instead of the chip's TP slice: :class:`C1SharedSplit` holds, per chip, the SP rank's
tile-aligned piece of the chip's TP shard of ``gate_proj`` / ``up_proj`` (columns) and ``down_proj`` (rows), cut on
device at construction with one-hot matmuls (bit-exact: whole tiles keep their bfp8 blocks; +1.6 MB a chip a
layer), and the layer runs three matmuls on ``[5120, 96]`` / ``[96, 5120]`` instead of ``[5120, 576]`` /
``[576, 5120]``. Every chip's partial then holds a different piece, so no ``sp0`` mask: the TP reduce-scatter and
the SP gather-sum of step 11 add all 32. The tile's row 0 is the only live row either way. The MoE's own shared
expert is not called (``shared_fold`` reads ``split``; the tail's ``pending_shared`` stays empty). The chain is
on every chip's critical path, so the saving (~35 us a layer: the 9-core and 12-core matmuls read 3.1 MB each at
~19 GB/s a core) comes off the busiest chip's path as much as device 0's.
"""

import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import _fused_compute_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import clamped_silu_glu, get_program_configs
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import _cores
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp8_qdq
from models.demos.deepseek_v3_d_p.tt.v41.routing import DecodeGate, router_select

_KERNELS = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_TILE = 32
_SENTINEL = 32  # the identity table's zero row
# the routed expert FFN op: ``fused`` (moe_fused_swiglu, the in-tree op for few tokens an expert) or ``composite``
# (unified_routed_expert_moe, today's prefill op: ~90 us an expert at one token) or ``matvec`` (C1FfnMatvec, the
# decode-shaped reader: every core reads, the row and the activated tiles multicast)
_FFN = os.environ.get("V41_DECODE_C1_FFN", "composite")
assert _FFN in ("fused", "composite", "matvec"), _FFN
# the shared expert on the c1 path: ``fold`` (the MoE's own shared expert, its partial folded into the tail) or
# ``split`` (C1SharedSplit)
_SHARED = os.environ.get("V41_DECODE_C1_SHARED", "fold")
assert _SHARED in ("fold", "split"), _SHARED
# c1_select's weight tile: ``fresh`` (allocated and zeroed by the kernel each call) or ``persist`` (one zeroed
# tensor a layer at construction, the kernel writes its EPC head words)
_SELECT = os.environ.get("V41_DECODE_C1_SELECT", "fresh")
assert _SELECT in ("fresh", "persist"), _SELECT
_HIFI4 = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False
)


def _accessors(*tensors) -> list:
    args = []
    for t in tensors:
        args += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    return args


C1_SELECT_ROWS = 4  # the most live rows c1_select handles (the verify step's 1 + K, K <= 3)


def c1_select(indices, weights, table, num_experts: int, w=None, rows: int = 1):
    """``indices``: uint16 TILE ``[T, K]`` (``T <= 32``; rows ``0 .. rows - 1`` the live rows), ``weights``: bf16
    row-major ``[T, K]``, ``table``: this chip's ``global_expert_idx_table`` (uint32 row-major, ``EPC`` words).
    Returns ``(counts, w)``: uint32 row-major ``[1, E]`` (``rows`` at the experts some live row selected that are
    this chip's) and bf16 TILE ``[32, 32 * EPC]`` (row ``t``, column ``32 l`` = row ``t``'s weight of local expert
    ``l``, else 0). ``w``: a persistent, zeroed tensor of that shape to write the head words into (rows
    ``0 .. C1_SELECT_ROWS - 1`` of column ``32 l``; the kernel's ``PERSIST``), instead of a fresh tensor zeroed by
    the kernel."""
    E = num_experts
    T, K = int(indices.shape[-2]), int(indices.shape[-1])
    assert T <= _TILE and K <= 16, (T, K)
    assert 1 <= rows <= min(T, C1_SELECT_ROWS), (rows, T)
    assert indices.dtype == ttnn.uint16 and indices.layout == ttnn.TILE_LAYOUT, (indices.dtype, indices.layout)
    assert weights.dtype == ttnn.bfloat16 and weights.layout == ttnn.ROW_MAJOR_LAYOUT, (weights.dtype, weights.layout)
    assert tuple(weights.shape)[-2:] == (T, K), tuple(weights.shape)
    assert table.dtype == ttnn.uint32 and table.layout == ttnn.ROW_MAJOR_LAYOUT, (table.dtype, table.layout)
    epc = int(table.shape[-1])
    device = indices.device()
    counts = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, E]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    persist = w is not None
    if persist:
        assert tuple(w.shape) == (1, 1, _TILE, _TILE * epc) and w.dtype == ttnn.bfloat16, (tuple(w.shape), w.dtype)
    else:
        w = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, _TILE, _TILE * epc]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
    coords, cores = _cores(device, 1)
    args = ttnn.RuntimeArgs()
    cx, cy = coords[0]
    args[cx][cy] = [
        indices.buffer_address(),
        weights.buffer_address(),
        table.buffer_address(),
        counts.buffer_address(),
        w.buffer_address(),
        rows,
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNELS}/c1_select.cpp",
        core_ranges=cores,
        compile_time_args=[0, E, K, epc, int(persist), C1_SELECT_ROWS] + _accessors(indices, weights, table, counts, w),
        runtime_args=args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    w_stride = C1_SELECT_ROWS * 32 if persist else 2048
    scratch = 64 + 2048 + 64 * C1_SELECT_ROWS + (epc * 4 + 63) // 64 * 64 + (E * 4 + 63) // 64 * 64 + epc * w_stride
    fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)
    cbs = [ttnn.CBDescriptor(total_size=scratch, core_ranges=cores, format_descriptors=[fmt])]
    ttnn.generic_op(
        [indices, weights, table, counts, w], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
    )
    return counts, w


class C1SharedSplit:
    """The shared expert's intermediate split over the 32 chips (``V41_DECODE_C1_SHARED=split``; the module
    docstring). ``shared``: the layer's ``TtSharedExpert`` (its ``gate_proj`` / ``up_proj`` ``[hidden, inter/tp]``
    and ``down_proj`` ``[inter/tp, hidden]`` per chip, TP-sharded, SP-replicated). The TP shard's ``inter/tp/32``
    tiles go to the SP ranks in order, the first ranks one more when they do not divide (18 tiles over 8 ranks: 3,
    3, 2, 2, 2, 2, 2, 2), every chip's piece padded to ``width`` (the largest) with zero columns / rows, which add
    nothing (``silu(min(0, L)) * clamp(0) = 0``)."""

    def __init__(self, shared, device):
        two_d = lambda t: ttnn.reshape(t, tuple(t.shape)[-2:]) if len(t.shape) != 2 else t
        gate, up, down = two_d(shared.gate_proj), two_d(shared.up_proj), two_d(shared.down_proj)
        self.hidden, inter_tp = int(gate.shape[-2]), int(gate.shape[-1])
        assert tuple(up.shape) == (self.hidden, inter_tp) and tuple(down.shape) == (inter_tp, self.hidden)
        sp, tp = tuple(device.shape)
        n_tiles = inter_tp // _TILE
        base, extra = divmod(n_tiles, sp)
        self.counts = [base + (1 if s < extra else 0) for s in range(sp)]
        self.offsets = [sum(self.counts[:s]) for s in range(sp)]
        self.width = max(self.counts) * _TILE
        S = torch.zeros(sp, tp, inter_tp, self.width)
        for s in range(sp):
            for j in range(self.counts[s]):
                for i in range(_TILE):
                    S[s, :, _TILE * (self.offsets[s] + j) + i, _TILE * j + i] = 1.0
        shard = lambda t: ttnn.reshape(
            ttnn.from_torch(
                t,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=device.shape, dims=(0, 1)),
            ),
            tuple(t.shape)[-2:],
        )
        S_dev, ST_dev = shard(S), shard(S.transpose(-1, -2).contiguous())
        cut = lambda a, b, dtype: ttnn.matmul(
            a, b, compute_kernel_config=_HIFI4, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.gate = cut(gate, S_dev, gate.dtype)  # [hidden, width]
        self.up = cut(up, S_dev, up.dtype)
        self.down = cut(ST_dev, down, down.dtype)  # [width, hidden]
        ttnn.deallocate(S_dev)
        ttnn.deallocate(ST_dev)
        self.limit = float(shared.clamped_silu_glu_limit)
        self.compute = shared.compute_kernel_config
        grid = device.compute_with_storage_grid_size()
        w = self.width // _TILE
        self.gu_cfg, _, self.d_cfg = get_program_configs(
            grid, 1, self.hidden // _TILE, w, w, self.hidden // _TILE, False
        )

    def __call__(self, x):
        """``x``: bf16 TILE ``[32, hidden]`` (the all-gathered row at row 0) -> this chip's partial of the shared
        expert's output, bf16 TILE ``[32, hidden]``; ``x`` is not consumed."""
        mm = lambda a, b, cfg: ttnn.matmul(a, b, program_config=cfg, compute_kernel_config=self.compute)
        act = clamped_silu_glu(mm(x, self.gate, self.gu_cfg), mm(x, self.up, self.gu_cfg), self.limit)
        out = mm(act, self.down, self.d_cfg)
        ttnn.deallocate(act)
        return out


class C1FfnMatvec:
    """The decode-shaped routed expert FFN (``V41_DECODE_C1_FFN=matvec``): one token row
    through this chip's selected experts with every core of the grid reading the expert's weights, instead of
    ``unified_routed_expert_moe``'s 11 column senders multicasting them down the grid at one tile of M (~90 us an
    expert, 222 GB/s). Per expert: core c < NT_GU owns gate/up column tile c (its two dot
    products over K from the resident row, fp32 accumulation, the clamped SiLU-GLU in DST); the activated tiles are
    staged on core 0 and multicast into every core's L1; core c owns down column tiles {c, c + cores, ..} < NT_D.
    The row ``x`` (bf16 TILE ``[32, hidden]``, row 0 live) is read by a few cores and multicast once by core 0.
    Output: tile row l of the persistent ``y`` (bf16 ``[EPC * 32, hidden]``, zeroed once and shared by the layers;
    the other regions keep stale finite values that the local sum's zero weights drop). The kernels are
    ``kernels/c1_ffn_{reader,compute,writer}.cpp`` (the reader's header has the dataflow). Measured: the reads at ~380 GB/s (52 us an expert), the compute hidden, ~25 us of sync an expert;
    the one-row step -0.8 ms."""

    BLK_GU = 32
    BLK_D = 24
    RING_GU = 5  # gate/up ring blocks: the whole K, so the reader runs an expert ahead of the compute
    RING_D = 3  # down ring blocks of two columns: the whole K, issued before the activation gather
    ACT_SLOTS = 3  # activated-row slots (an expert's slot is reused three experts later, after a done barrier)
    NONE = 0xFFFFFFFF
    _shared_y = {}  # (device, epc, hidden) -> the output buffer shared by the layers

    def __init__(self, routed, device, hidden: int, epc: int, E: int, fidelity=None):
        self.routed, self.epc, self.E, self.hidden = routed, epc, E, hidden
        gate0, down0 = routed.gate_projs[0], routed.down_projs[0]
        assert gate0.dtype == ttnn.bfloat4_b and down0.dtype == ttnn.bfloat4_b, (gate0.dtype, down0.dtype)
        gs, ds = tuple(gate0.padded_shape), tuple(down0.padded_shape)
        inter = int(gate0.shape[-1])
        assert gs[-2] == hidden and ds[-2] == inter and int(down0.shape[-1]) == hidden, (gs, ds, hidden, inter)
        self.KT, self.NGU_FULL, self.NT_GU = hidden // _TILE, gs[-1] // _TILE, inter // _TILE
        self.KD, self.ND_FULL, self.NT_D = inter // _TILE, ds[-1] // _TILE, hidden // _TILE
        # timing probes (wrong outputs): ``noread`` skips the weight reads, ``nocompute`` the matmuls; ``cols<n>``
        #  runs the TP-row split's quarter geometry on the whole-expert tensors: n gate/up
        # column tiles on n cores and n down K tiles (the quarter's 18 of 72), the reads of a quarter expert
        probe = os.environ.get("V41_DECODE_C1_FFN_PROBE", "")
        self.cols = 0
        if probe.startswith("cols"):  # ``cols18`` or ``cols18:noread``
            cols, _, probe = probe[4:].partition(":")
            self.cols = int(cols)
        self.probe = {"": 0, "noread": 1, "nocompute": 2, "none": 3}[probe]
        if self.cols:
            assert 0 < self.cols <= self.NT_GU, (self.cols, self.NT_GU)
            self.NT_GU = self.KD = self.BLK_D = self.cols
        assert self.KT % self.BLK_GU == 0 and self.KD % self.BLK_D == 0, (self.KT, self.KD)
        grid = device.compute_with_storage_grid_size()
        self.gx, self.gy = grid.x, grid.y
        self.n_cores = self.gx * self.gy
        assert self.n_cores >= self.NT_GU and 2 * self.n_cores >= self.NT_D, (self.n_cores, self.NT_GU, self.NT_D)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.gx - 1, self.gy - 1))])
        noc = lambda x, y: device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        c00, c11 = noc(0, 0), noc(self.gx - 1, self.gy - 1)
        self.rect = [c00.x, c00.y, c11.x, c11.y]
        fid = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}
        self.fidelity = fidelity or os.environ.get("V41_DECODE_C1_FFN_FIDELITY", "lofi")
        self.X_SENDERS = self.gy  # one x stripe reader a grid row
        assert self.KT % self.X_SENDERS == 0, (self.KT, self.X_SENDERS)
        self.X_PER_SENDER = self.KT // self.X_SENDERS
        self.tpc = int(os.environ.get("V41_DECODE_C1_FFN_TPC", "1"))  # gate/up column tiles a core (1 or 2)
        self.dpair = os.environ.get("V41_DECODE_C1_FFN_DPAIR", "1") == "1"  # down: adjacent column pairs a core
        shard_w = lambda t: (
            int(t.memory_config().nd_shard_spec.shard_shape[-1]) // _TILE if t.memory_config().nd_shard_spec else 0
        )
        self.shard_gu, self.shard_d = shard_w(gate0), shard_w(down0)
        self.compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=fid[self.fidelity], fp32_dest_acc_en=True, math_approx_mode=False
        )
        # one output buffer for every layer of the decoder (the layers run in order and the local sum reads it
        # before the next layer's FFN writes it): 3.9 MB a chip instead of 157 at 40 layers
        key = (id(device), epc, hidden)
        if key not in C1FfnMatvec._shared_y:
            C1FfnMatvec._shared_y[key] = ttnn.from_torch(
                torch.zeros(1, 1, epc * _TILE, hidden),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        self.y = C1FfnMatvec._shared_y[key]
        self.table = routed.global_expert_idx_table
        self.waddr = [t.buffer_address() for w in (routed.gate_projs, routed.up_projs, routed.down_projs) for t in w]
        assert len(self.waddr) == 3 * epc
        # per core: the owned gate/up column and the owned down columns
        self.per_core = []
        ngu_cores = -(-self.NT_GU // self.tpc)
        for i in range(self.n_cores):
            x, y = i % self.gx, i // self.gx
            if self.dpair:
                nd = [n for n in (2 * i, 2 * i + 1) if n < self.NT_D]
            else:
                nd = [n for n in (i, i + self.n_cores) if n < self.NT_D]
            self.per_core.append(
                {
                    "xy": (x, y),
                    "ngu": i if i < ngu_cores else self.NONE,
                    "nd": (nd + [self.NONE, self.NONE])[:2],
                    "nd_count": len(nd),
                    "x_sender": int(i < self.X_SENDERS),
                    "x_first": (i % self.X_SENDERS) * self.X_PER_SENDER,
                }
            )
        self.scr_page = 64 + (E * 4 + epc * 4 + 63) // 64 * 64  # a 64-byte aligned base inside

    def __call__(self, x, counts):
        """``x``: bf16 TILE ``[1, 1, 32, hidden]`` (DRAM-interleaved; row 0 the live row, the others zero),
        ``counts``: uint32 row-major ``[1, E]`` -> ``self.y`` with tile row l = local expert l's output of row 0
        (rows 1..31 zero) for every l with ``counts[table[l]] > 0``."""
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
        assert tuple(x.shape)[-2:] == (_TILE, self.hidden), tuple(x.shape)
        X_TILE, W_TILE, ACT_TILE = 2048, 576, 2048
        cb_x, cb_gate, cb_up, cb_down, cb_act, cb_actf, cb_scr, cb_done, cb_stage, cb_out = (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            16,
        )
        rd, wr, cm = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c in self.per_core:
            cx, cy = c["xy"]
            rd[cx][cy] = (
                [
                    x.buffer_address(),
                    counts.buffer_address(),
                    self.table.buffer_address(),
                    c["ngu"],
                    c["nd"][0],
                    c["nd"][1],
                    int(cx == 0 and cy == 0),
                    self.rect[0],
                    self.rect[1],
                ]
                + self.rect
                + [0, 1, 2, 3, 4, c["x_sender"], c["x_first"]]
                + self.waddr
            )
            wr[cx][cy] = [self.y.buffer_address(), c["nd"][0], c["nd"][1]]
            cm[cx][cy] = [c["ngu"], c["nd_count"]]
        common = [self.epc, self.E, self.KT, self.NT_GU, self.NGU_FULL, self.KD, self.NT_D, self.ND_FULL]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNELS}/c1_ffn_reader.cpp",
                core_ranges=self.cores,
                compile_time_args=[cb_x, cb_gate, cb_up, cb_down, cb_act, cb_actf, cb_scr, cb_done, cb_stage]
                + common
                + [self.BLK_GU, self.BLK_D, self.n_cores, X_TILE, W_TILE, ACT_TILE, self.ACT_SLOTS, self.probe]
                + [self.tpc, self.shard_gu, self.shard_d, self.RING_GU, self.RING_D, self.X_SENDERS, self.X_PER_SENDER]
                + _accessors(x, self.routed.gate_projs[0], self.routed.up_projs[0], self.routed.down_projs[0])
                + _accessors(counts, self.table),
                runtime_args=rd,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNELS}/c1_ffn_writer.cpp",
                core_ranges=self.cores,
                compile_time_args=[cb_scr, cb_out, self.epc, self.E, self.NT_D, ACT_TILE] + _accessors(self.y),
                runtime_args=wr,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNELS}/c1_ffn_compute.cpp",
                core_ranges=self.cores,
                compile_time_args=[cb_x, cb_gate, cb_up, cb_down, cb_act, cb_actf, cb_scr, cb_done, cb_out]
                + [self.epc, self.E, self.KT, self.NT_GU, self.KD, self.BLK_GU, self.BLK_D, self.probe, self.tpc],
                runtime_args=cm,
                config=self.compute,
            ),
        ]

        def cb(index, fmt, pages, page):
            f = ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.cores, format_descriptors=[f])

        bf16, bfp4, u32 = ttnn.bfloat16, ttnn.bfloat4_b, ttnn.uint32
        cbs = [
            cb(cb_x, bf16, self.KT, X_TILE),
            cb(cb_gate, bfp4, self.RING_GU * self.BLK_GU * self.tpc, W_TILE),
            cb(cb_up, bfp4, self.RING_GU * self.BLK_GU * self.tpc, W_TILE),
            cb(cb_down, bfp4, self.RING_D * self.BLK_D * 2, W_TILE),
            cb(cb_act, bf16, 2 * self.tpc, ACT_TILE),
            cb(cb_actf, bf16, self.ACT_SLOTS * self.NT_GU, ACT_TILE),
            cb(cb_scr, u32, 1, self.scr_page),
            cb(cb_done, u32, 2, 64),
            cb(cb_stage, bf16, self.NT_GU, ACT_TILE),
            cb(cb_out, bf16, 4, ACT_TILE),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.cores, initial_value=0) for i in range(5)]
        ttnn.generic_op(
            [x, counts, self.table, self.y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        )
        return self.y


class C1Moe:
    """The c1 MoE of one decode layer's ``TtV41Moe`` (its gate a :class:`DecodeGate`): see the module docstring.
    ``ccl``: the model's ``V41Collectives``; ``pool``: the decoder's ``V41DecodePool`` (its ``tt_ccl``)."""

    def __init__(self, v41_moe, ccl, pool, rows: int = 1):
        """``rows``: the most live rows a call takes (1: the served c1 step; the verify step's ``1 + K``, the
        module docstring); above 1 the composite FFN's entry is built beside the matvec one."""
        moe = v41_moe.moe
        assert isinstance(moe.gate, DecodeGate), "the c1 MoE needs the decode routing (V41_DECODE_ROUTING=1)"
        assert 1 <= rows <= C1_SELECT_ROWS, rows
        self.moe, self.ccl, self.pool, self.rows = moe, ccl, pool, rows
        self.gate: DecodeGate = moe.gate
        self.routed = moe.routed_expert
        self.k = moe.num_experts_per_tok
        self.E = moe.num_routed_experts
        self.epc = self.routed.experts_per_chip
        self.table = self.routed.global_expert_idx_table  # uint32 [EPC] per chip, the chip's global expert ids
        assert self.table.layout == ttnn.ROW_MAJOR_LAYOUT and int(self.table.shape[-1]) == self.epc, self.table
        device = moe.mesh_device
        # the region of local expert l starts at row 32 l of the FFN buffer: a constant per chip, built from the
        # chip's expert table (the FFN reads region[global id] for each of its experts; the other entries unread)
        hidden = int(self.routed.gate_projs[0].shape[-2])
        self.hidden = hidden
        region = []
        for chip in ttnn.get_device_tensors(self.table):
            ids = ttnn.to_torch(chip).flatten().to(torch.int64)
            row = torch.zeros(1, self.E, dtype=torch.int32)
            row[0, ids] = torch.arange(self.epc, dtype=torch.int32) * _TILE
            region.append(row)
        self.region = ttnn.from_torch(
            torch.stack(region).view(*tuple(device.shape), 1, self.E),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=device.shape, dims=(0, 1)),
        )
        self.region = ttnn.reshape(self.region, (1, self.E))
        # R[32 l + j, j] = 1 (j < rows): ``R @ x_full`` puts the live rows at the head of each expert's region
        R = torch.zeros(self.epc * _TILE, _TILE)
        for j in range(rows):
            R[j::_TILE, j] = 1.0
        self.R = self._replicated(R.view(1, 1, self.epc * _TILE, _TILE), ttnn.bfloat16)
        # sum8[j, 32 r + j] = 1 (j < rows): ``sum8 @ gathered`` sums the SP column's tiles (their other rows zero)
        # row by row
        sum8 = torch.zeros(_TILE, _TILE * pool.sp)
        for j in range(rows):
            sum8[j, j::_TILE] = 1.0
        self.sum8 = self._replicated(sum8.view(1, 1, _TILE, _TILE * pool.sp), ttnn.bfloat16)
        # sp0[0, 0] = 1 on the chips of SP rank 0 (mesh row 0), zero elsewhere: ``sp0 @ partial`` keeps a
        # replicated row-0 partial on one SP rank only, so the SP all-gather-and-sum of step 11 counts it once
        # (the shared expert's partial under V41_SHARED_IN_FFN, the module docstring)
        sp, tp = tuple(device.shape)
        sp0 = torch.zeros(sp, tp, _TILE, _TILE)
        for j in range(rows):
            sp0[0, :, j, j] = 1.0
        self.sp0 = ttnn.from_torch(
            sp0,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=device.shape, dims=(0, 1)),
        )
        self.sp0 = ttnn.reshape(self.sp0, (1, 1, _TILE, _TILE))
        # c1_select's persistent weight tile (``V41_DECODE_C1_SELECT=persist``): zeroed once, before any capture
        self.W0 = (
            self._replicated(torch.zeros(1, 1, _TILE, _TILE * self.epc), ttnn.bfloat16)
            if _SELECT == "persist"
            else None
        )
        self.shared_fold = None  # "pending" once the shared expert has parked a partial, "split" (tests read it)
        self.shared_split = None
        if _SHARED == "split":
            shared = moe.shared_expert
            shared = getattr(shared, "_shared", shared)  # DecodeSharedExpert wraps the MoE's TtSharedExpert
            self.shared_split = C1SharedSplit(shared, device)
        self.debug = None  # a dict: the call keeps its ``counts`` and ``indices`` there (tests)
        self.matvec = C1FfnMatvec(self.routed, device, hidden, self.epc, self.E) if _FFN == "matvec" else None
        # the FFN's program-cache entry for this shape is built here, on an empty L1: ``ttnn.reduce_scatter`` (the
        # decode tail's TP reduce-scatter) leaves a 512-byte L1 buffer at 914,176 after its first call, and the
        # routed experts' static CBs (to 1,004,864 in this mode) are rejected at compile against it; today's
        # 32-row FFN entry is built before any reduce-scatter runs and never meets it
        zeros = ttnn.from_torch(
            torch.zeros(1, 1, self.epc * _TILE, hidden),
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        none = self._replicated(torch.zeros(1, self.E, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        if self.matvec is not None:
            row = self._replicated(torch.zeros(1, 1, _TILE, hidden), ttnn.bfloat16)
            self.matvec(row, none)
            ttnn.deallocate(row)
        if self.matvec is None or rows > 1:  # the composite FFN carries the multi-row calls
            ttnn.deallocate(self._ffn(ttnn.reshape(zeros, (self.epc * _TILE, hidden)), none))
        ttnn.deallocate(none)

    def _replicated(self, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t,
            device=self.moe.mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.moe.mesh_device),
        )

    # --- the one-hot tiles, once a step ------------------------------------------------------------------------
    @staticmethod
    def host_rows(sp: int, rows: int, live: tuple[int, int] | None) -> dict:
        """The two integer rows per rank (``[sp, rows]`` int32): ``c1sel`` (word 0 = u on the live rank) and
        ``c1put`` (word u = 0 on the live rank), every other word the sentinel. ``live``: (rank, u) or None."""
        sel = torch.full((sp, rows), _SENTINEL, dtype=torch.int32)
        put = torch.full((sp, rows), _SENTINEL, dtype=torch.int32)
        if live is not None:
            r, u = live
            sel[r, 0] = u
            put[r, u] = 0
        return {"c1sel": sel, "c1put": put}

    @staticmethod
    def identity_table(mesh_device):
        """``[33, 32]`` bf16 row-major: the identity and a zero row (the sentinel), replicated."""
        eye = torch.cat([torch.eye(_TILE), torch.zeros(1, _TILE)]).to(torch.bfloat16)
        return ttnn.from_torch(
            eye,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @staticmethod
    def onehots(ids_sel, ids_put, table):
        """``ids_*``: uint32 row-major ``[1, 1, 1, 32]`` (the device slices of ``host_rows``) -> the ``sel`` and
        ``put`` tiles, bf16 TILE ``[1, 1, 32, 32]``."""
        out = []
        for ids in (ids_sel, ids_put):
            flat = ttnn.reshape(ids, (1, 1, _TILE))
            tile = ttnn.embedding(flat, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.append(ttnn.reshape(tile, (1, 1, _TILE, _TILE)))
        return tuple(out)

    # --- the layer -----------------------------------------------------------------------------------------------
    def _ffn(self, x_in, counts):
        """The routed expert FFN on the 12-region buffer (bfp8 TILE ``[EPC * 32, hidden]``): local expert ``l``'s
        rows at ``32 l`` where ``counts`` says so. ``fused``: ``moe_fused_swiglu`` into a zeroed output (the
        unselected regions stay zero); ``composite``: ``unified_routed_expert_moe`` in place."""
        routed = self.routed
        if _FFN == "fused":
            out = ttnn.zeros_like(x_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                x_in,
                routed.gate_projs,
                routed.up_projs,
                routed.down_projs,
                counts,
                self.table,
                input_m_tiles=1,
                core_grid=ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID,
                compute_kernel_config=_fused_compute_config(routed.compute_kernel_config),
                activation=routed.activation,
                output=out,
                expert_region_offsets=self.region,
                read_x_at_offset=True,
                max_active_tokens=_TILE,
                gate_biases=routed.gate_biases,
                up_biases=routed.up_biases,
                down_biases=routed.down_biases,
            )
            ttnn.deallocate(x_in)
            return out
        return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x_in,
            self.region,
            counts,
            self.table,
            routed.gate_projs,
            routed.up_projs,
            routed.down_projs,
            max_dispatched_tokens_per_expert=_TILE,
            compute_kernel_config=routed.compute_kernel_config,
            activation=routed.activation,
            gate_biases=routed.gate_biases,
            up_biases=routed.up_biases,
            down_biases=routed.down_biases,
        )

    def _sp_all_reduce(self, t):
        """``t`` ``[1, 1, 32, W]`` with rows 1..31 zero -> the same shape on every chip of the SP column, row 0 the
        sum of the column's row 0s: one all-gather of the tiles (dim 2) and a one-hot matmul over the 32 x sp
        gathered rows. A ring reduce-scatter and all-gather over the 8 chips cost 46 us each (7 serialized hops);
        the all-gather alone is one of them, the matmul 3 us."""
        if self.pool.sp == 1:
            return t
        gathered = self.pool.gather_sp(t, dim=2)
        out = ttnn.matmul(self.sum8, gathered, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gathered)
        return out

    def __call__(self, h, sel, put, rows: int = 1):
        """``h``: bf16 TILE ``[1, 1, 32, hidden/tp]`` (the normed residual tile), ``sel`` / ``put`` of
        :meth:`onehots` -> the MoE output of the live row at its row of the live rank's tile, the other rows zero;
        the same shape as ``h``. ``rows``: the live rows (``sel`` picks them into rows ``0 .. rows - 1``)."""
        moe, gate, cfg = self.moe, self.gate, self.gate.config
        assert 1 <= rows <= self.rows, (rows, self.rows)
        # 1, 2: the live row's TP shard to every chip
        picked = ttnn.matmul(sel, h, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xs = self._sp_all_reduce(picked)
        ttnn.deallocate(picked)
        # 3, 4: the gate and the exact top-k of row 0 (rows 1..31 the sentinel)
        xs2 = ttnn.reshape(xs, (_TILE, int(xs.shape[-1])))
        logits = gate._gate._device_matmul(xs2)
        zero, bias32 = gate._constants(logits)
        l32 = ttnn.typecast(logits, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(logits)
        activations = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT),
        ]
        score = ttnn.add(l32, zero, input_tensor_a_activations=activations, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(l32)
        rank = ttnn.add(score, bias32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        weights, indices = router_select(rank, score, self.k, 1e-20, cfg.route_scale, rows)
        ttnn.deallocate(rank)
        ttnn.deallocate(score)
        # 5: the row's experts that are this chip's
        counts, W = c1_select(indices, weights, self.table, self.E, self.W0, rows)
        if self.debug is not None:
            self.debug.update(counts=counts, indices=indices, weights=weights)
        else:
            ttnn.deallocate(indices)
            ttnn.deallocate(weights)
        # 6: the experts' input: the QDQ'd row, completed over TP
        xq = fp8_qdq(xs)
        ttnn.deallocate(xs)
        x_full = self.ccl.tp_all_gather(xq, dim=3)
        ttnn.deallocate(xq)
        # 7: the shared expert (rows 1..31 zero in, zero out). ``TtSharedExpert`` returns its reduce-scattered
        # [32, hidden/tp] output (added at step 12); ``DecodeSharedExpert`` (V41_SHARED_IN_FFN) returns None and
        # parks its [32, hidden] partial in the decode tail's reduce: taken here, folded into step 11's
        # reduce-scatter on SP rank 0 only (the module docstring)
        shared = None if self.shared_split is not None else moe.shared_expert(ttnn.squeeze(x_full, dim=0))
        if self.shared_split is not None:
            # 7': this chip's piece of the shared expert on the all-gathered row (C1SharedSplit); every chip's
            # partial differs, so step 11's TP reduce-scatter and SP gather-sum add all 32 of them: no sp0 mask
            shared = self.shared_split(ttnn.reshape(x_full, (_TILE, self.hidden)))
            shared = ttnn.reshape(shared, (1, 1, _TILE, self.hidden))
            self.shared_fold = "split"
        elif shared is None:
            reduce = moe.reduce_module
            partial, reduce.pending_shared = reduce.pending_shared, None
            assert partial is not None, "the shared expert returned None and parked no partial"
            partial = ttnn.reshape(partial, (1, 1, _TILE, self.hidden))
            shared = ttnn.matmul(self.sp0, partial, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(partial)
            self.shared_fold = "pending"
        else:
            shared = ttnn.unsqueeze(shared, dim=0)
            self.shared_fold = "returned"
        folded = self.shared_fold in ("pending", "split")
        if self.debug is not None:
            self.debug["x_full"] = x_full  # the experts' input row (tests build the reference from it)
        matvec = self.matvec is not None and rows == 1
        if matvec:
            # 8', 9': the decode-shaped FFN reads the bf16 row itself (no bfp8 region buffer) into its persistent y
            y = self.matvec(x_full, counts)
            if self.debug is None:
                ttnn.deallocate(x_full)
        else:
            # 8, 9: the row at the head of every local expert's region; the selected experts run in place
            x_in = ttnn.matmul(
                self.R,
                x_full,
                compute_kernel_config=_HIFI4,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if self.debug is None:
                ttnn.deallocate(x_full)
            x_in = ttnn.reshape(x_in, (self.epc * _TILE, self.hidden))
            y = ttnn.reshape(self._ffn(x_in, counts), (1, 1, self.epc * _TILE, self.hidden))
        if self.debug is None:
            ttnn.deallocate(counts)
        # 10: the local weighted sum (row 0)
        part = ttnn.matmul(W, y, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if not matvec:
            ttnn.deallocate(y)
        if self.W0 is None:
            ttnn.deallocate(W)
        if folded:
            # the shared partial (SP rank 0 only) joins the local routed partial before the TP reduce-scatter,
            # as the 32-row tail does (moe_tail.DecodeReduce)
            total = ttnn.add(part, shared, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(part)
            ttnn.deallocate(shared)
            part = total
        # 11: the sum over the 32 chips, this chip's TP slice
        tp_part = self.ccl.tp_reduce_scatter(part, dim=3)
        ttnn.deallocate(part)
        summed = self._sp_all_reduce(tp_part)
        ttnn.deallocate(tp_part)
        # 12: + the shared expert (unless folded above), placed at the live row
        if folded:
            full = summed
        else:
            full = ttnn.add(summed, shared, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(summed)
            ttnn.deallocate(shared)
        out = ttnn.matmul(put, full, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(full)
        return out
