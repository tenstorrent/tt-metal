# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Bank-streamed fused MoE expert step for Inkling-2-Mini decode (one Blackhole p100a, 7 DRAM banks).

Same math, inputs and output as ``FusedMoeExperts`` (tt/fused/moe_expert.py, stick layout): for the 8 routed experts
gate_up matmul -> softcapped SwiGLU x routing weight -> down matmul -> sum over experts, one program. What changes is
who reads what: the weights are in the bank layout (convert/bank_layout.py), where each down column and each
(gate, up) column pair always sits in one DRAM bank, and every worker core sits on its bank's row next to the bank's
optimal NOC0 reader and reads only that bank, as the gpt-oss fused decode streams do
(models/demos/gpt_oss/tt/fused_decode/stream.py, row_placement: ~96 % of DRAM peak on Blackhole).

Workers per bank b (row_placement): the gate core and the up core of each base pair j = b, b + 7, ... (10), then on
banks 2..6 the gate and up cores of the extra pair 35 (they only compute the experts whose extra pair
convert/bank_layout.extra_bank puts in that bank; their other blocks are pushed unread and dropped), or on banks 0, 1
a down-only core. The first ones of each bank also own its down columns b, b + 7, ... (11 on banks 0, 1, 10 on the
others): 82 workers, each reading one gate_up column and / or one down column of the 8 experts. As in the fused step,
an up core hands its 8 u tiles to its gate core, the gate core computes the SwiGLU and sends h to the collector, the
collector multicasts the hidden [8][36] to the whole grid, and every down core runs its column over K = 8 x 36 hidden
tiles in the same order as the stick-layout FusedMoeExperts, so the output is the same.

Measured on p100a (traced, per op; the module default has both changes below, the output is the same either way):
the op ends when the slowest bank has streamed its share, so the start and the NOC paths matter, the compute
(~35 us of the BFP8 step) does not.
* x: the collector reads x once and multicasts it (X_MCAST); before, all 82 workers read the x tensor's core
  (2 x 72 reads each) before their first weight read, which delayed the streams by 3-8 us.
* bank 0's workers (its whole row) read on NOC1, their writers on NOC0: BFP8 179 -> 164 us, BFP4 94 -> 88,
  mixed 127 -> 118 (any other single bank: about half of that; two or more banks: no better, or much worse).
Fewer readers per bank (gate and up of a pair on one core, 6 per bank) read slower (340-367 GB/s vs ~400), as did
any other row for a bank.
"""

import os

import torch

import ttnn
from models.demos.inkling2_mini.convert import bank_layout as BL
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.moe_expert import (
    CB_G,
    CB_G32,
    CB_H16,
    CB_HID,
    CB_O,
    CB_SCR,
    CB_U16,
    CB_W,
    CB_WD,
    CB_WS16,
    CB_X,
    CB_Z,
    HID_TILES,
    HIDDEN,
    KDIR,
    KT_DN,
    KT_GU,
    N_GATE,
    NT_DN,
    SEM_G,
    SEM_R,
    SEM_U,
    TOP_K,
    _range_set,
)

F_GU, F_GATE, F_EXTRA, F_DN = 1, 2, 4, 8  # worker flags (moe_bank_*.cpp)
# The output row also goes to a dense twin (tt/fused/common.py set_twin): the fused residual consumer (conv_mlp)
# reads 3 pages instead of gathering the row. INKLING_MOE_TWIN=0: row only (A/B runs).
DENSE_OUT = os.environ.get("INKLING_MOE_TWIN", "1") != "0"
SEM_X = 3  # x multicast done (X_MCAST)
NOC1_BANKS = (0,)  # banks whose workers read on NOC1 (module doc)
X_LO = 2  # convert/bank_layout.extra_bank: first bank holding extra pairs


def bank_weights(device, w, dtype, name):
    """Device-layout expert weight ``w`` [E, K, N] (torch) -> DRAM interleaved tensor in the bank layout (tests;
    the model loads the converter's bank cache). Same tiles as the TILE tensor."""
    E, K, N = w.shape
    pages = BL.bank_pages(name, [1, E, K, N])
    tiles = w.bfloat16().reshape(E, K // 32, 32, N // 32, 32).permute(0, 1, 3, 2, 4).reshape(-1, 32, 32)
    out = torch.zeros(len(pages), 32, 32, dtype=torch.bfloat16)
    out[torch.from_numpy(pages >= 0)] = tiles[torch.from_numpy(pages[pages >= 0])]
    return ttnn.from_torch(out.reshape(1, 1, -1, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


def _nd(device, t, dtype, banks):
    """[1, E, K, n*32] torch -> DRAM tensor ND-sharded one 32-column shard [E, K, 32] per bank, round robin over
    banks 0 .. banks-1: shard m (columns m*32..) sits in bank m % banks, the shards of a bank one after another, each
    receiver-contiguous (expert e's K tiles of that column in a row), as the tensor prefetcher streams grouped weights.
    """
    spec = ttnn.NdShardSpec(
        ttnn.Shape([t.shape[1], t.shape[2], 32]),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
        ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )  # fmt: skip
    return ttnn.from_torch(t.bfloat16(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec))  # fmt: skip


def pf_columns(name, nb=BL.N_BANKS):
    """Source column tile of ring position m (bank m % nb, worker i = m // nb of that bank) in the PF layout's main
    tensor: gate_up worker i is the gate (i even) or up column of base pair b + nb * (i // 2); down worker i owns
    column b + nb * i = m."""
    P = N_GATE
    base, _ = BL.gu_pair_counts(P, nb)
    if name.endswith("w_gate_up"):
        out = []
        for m in range(nb * 2 * base):
            b, i = m % nb, m // nb
            j = b + nb * (i // 2)
            out.append(j if i % 2 == 0 else P + j)
        return out
    return list(range(nb * (NT_DN // nb)))


def pf_weights(device, w, dtype, name):
    """Device-layout expert weight ``w`` [E, K, N] (torch) -> (main, extra) DRAM tensors in the PF layout (tests).
    main: the base columns (pf_columns), ND-sharded per ring position (_nd). extra: gate_up's extra pair as the bank
    layout stores it (bank extra_bank(e), slot e // (nb - X_LO), gate column then up column), DRAM interleaved; down's
    leftover columns (70, 71 of 72) in banks 0, 1 at slot e * kt, DRAM interleaved. Same tiles as the TILE tensor."""
    E, K, N = w.shape
    nb = BL.N_BANKS
    w = w.bfloat16()
    main = _nd(device, w.reshape(E, K, N // 32, 32)[:, :, pf_columns(name)].reshape(1, E, K, -1), dtype, nb)
    if name.endswith("w_down"):
        lo = nb * (NT_DN // nb)
        # shortcut: DRAM interleaved with banks 2..6 left empty (~5/7 of a 2-column tensor wasted); an ND-shard over
        # banks 0, 1 alone read back wrong columns in the kernel. Upgrade if DRAM gets tight.
        kt = K // 32
        tiles = w[:, :, lo * 32 :].reshape(E, kt, 32, NT_DN - lo, 32).permute(0, 1, 3, 2, 4)  # [E, kt, 2, 32, 32]
        out = torch.zeros(E * kt, nb, 32, 32, dtype=torch.bfloat16)
        out[:, : NT_DN - lo] = tiles.reshape(E * kt, NT_DN - lo, 32, 32)
        return main, ttnn.from_torch(out.reshape(1, 1, -1, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    P, kt = N_GATE, K // 32
    tiles = w.reshape(E, kt, 32, N // 32, 32).permute(0, 1, 3, 2, 4)  # [E, kt, nt, 32, 32]
    per = nb - X_LO
    slots = -(-E // per) * 2 * kt
    out = torch.zeros(slots, nb, 32, 32, dtype=torch.bfloat16)
    for e in range(E):
        b, s0 = BL.extra_bank(e, 0, nb), (e // per) * 2 * kt
        out[s0 : s0 + kt, b] = tiles[e, :, P - 1]
        out[s0 + kt : s0 + 2 * kt, b] = tiles[e, :, 2 * P - 1]
    extra = ttnn.from_torch(out.reshape(1, 1, -1, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    return main, extra


def row_placement(device, per_bank):
    """per_bank[b] logical cores for DRAM bank b on the row of the bank's optimal NOC0 worker: its own row segment
    first (nearest the worker), then the rest of the row, then the rows above / below. Adapted from
    gpt_oss/tt/fused_decode/stream.py row_placement (single device)."""
    opt = [(int(c.x), int(c.y)) for c in device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    grid = device.compute_with_storage_grid_size()
    used, out, cand = set(), {}, {}
    for b, (wx, wy) in enumerate(opt):
        mates = sorted(x for x, y in opt if y == wy)
        k = mates.index(wx)
        lo, hi = (wx if k else 0), (mates[k + 1] if k + 1 < len(mates) else grid.x)
        rows = [(wy, range(lo, hi)), (wy, range(grid.x))]
        rows += [(wy + d, range(grid.x)) for dd in range(1, grid.y) for d in (-dd, dd)]
        cand[b], out[b] = rows, []
    for phase in range(max(len(c) for c in cand.values())):  # every bank's segment before any bank takes leftovers
        for b, (wx, wy) in enumerate(opt):
            if phase >= len(cand[b]):
                continue
            y, xs = cand[b][phase]
            if not 0 <= y < grid.y:
                continue
            for x in sorted(xs, key=lambda x: (abs(x - wx), x)):
                if len(out[b]) < per_bank[b] and (x, y) not in used:
                    used.add((x, y))
                    out[b].append((x, y))
    assert all(len(out[b]) == per_bank[b] for b in out), out
    return out


class BankMoeExperts:
    def __init__(
        self, device, w_gate_up, w_down, sbk=18, nbuf=3, bfp8_fidelity=ttnn.MathFidelity.HiFi2, x_mcast=True,
        noc1_banks=NOC1_BANKS, dense_out=None,
    ):  # fmt: skip
        """``w_gate_up`` / ``w_down``: bank-layout DRAM tensors (``bank_weights`` or the bank cache), BFP4 or BFP8
        (each its own). Placement and flags: module doc. x_mcast: the collector reads x once and multicasts it
        (otherwise every gate_up core reads it). noc1_banks: banks whose workers read on NOC1 (writer on NOC0)."""
        self.pf = isinstance(w_gate_up, tuple)  # PF layout (pf_weights): (main, extra) per tensor
        if self.pf:
            (w_gate_up, self.gu_x), (w_down, self.dn_x) = w_gate_up, w_down
        self.x_mcast, self.noc1_banks = x_mcast, tuple(noc1_banks)
        self.dense_out = DENSE_OUT if dense_out is None else dense_out
        self.dense, self._last = None, None  # the twin (one per instance) and the row tensor it is registered for
        nb = device.dram_grid_size().x
        assert nb == BL.N_BANKS, nb
        assert (2 * KT_GU) % sbk == 0 and HID_TILES % sbk == 0, sbk
        self.device, self.w_gate_up, self.w_down, self.sbk, self.nbuf, self.nb = (
            device,
            w_gate_up,
            w_down,
            sbk,
            nbuf,
            nb,
        )
        self.base, n_extra = BL.gu_pair_counts(N_GATE, nb)
        assert n_extra == 1, n_extra
        self.dn_slots = BL.dn_slots(NT_DN, nb)
        # Per bank b: the gate and the up core of base pairs b, b + 7, ... (10), then the extra pair's gate and up
        # cores (banks >= X_LO) or a down-only core (the 11th down column of banks 0, 1); the first ones get the
        # bank's down columns b, b + 7, ...
        units = []
        for b in range(nb):
            u = [(F_GU | F_GATE, k, b + nb * k, 0) for k in range(self.base)]
            u = [x for g in u for x in (g, (F_GU, g[1], g[2], KT_GU))]  # gate, up of each pair
            if b >= X_LO:
                u += [
                    (F_GU | F_GATE | F_EXTRA, self.base, N_GATE - 1, 0),
                    (F_GU | F_EXTRA, self.base, N_GATE - 1, KT_GU),
                ]
            n_dn = len(range(b, NT_DN, nb))
            u += [(0, 0, 0, 0)] * max(0, n_dn - len(u))
            units.append([(f | (F_DN if i < n_dn else 0), k, j, off, i) for i, (f, k, j, off) in enumerate(u)])
        place = row_placement(device, [len(u) for u in units])
        self.workers_info = []  # (core, flags, bank, kslot, j, col_off, dslot, n, partner gate core)
        for b in range(nb):
            gate_of = {}
            for (f, k, j, off, i), c in zip(units[b], place[b]):
                if f & F_GATE:
                    gate_of[j] = c
            for (f, k, j, off, i), c in zip(units[b], place[b]):
                self.workers_info.append((c, f, b, k, j, off, i, b + nb * i, gate_of.get(j, c)))
        grid = device.compute_with_storage_grid_size()
        allc = [(x, y) for y in range(grid.y) for x in range(grid.x)]
        used = {w[0] for w in self.workers_info}
        self.collector = next(c for c in reversed(allc) if c not in used)
        self.workers = _range_set(list(used))
        self.all_cores = _range_set(allc)
        self.noc = {}
        for c in allc:
            p = device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
            self.noc[c] = (p.x, p.y)
        self.mcast_dests = len(allc) - 1
        self.mcast_rect = (self.noc[(0, 0)], self.noc[(grid.x - 1, grid.y - 1)])
        bfp8 = ttnn.bfloat8_b
        self.std_gu = bfp8_fidelity is not None and w_gate_up.dtype == bfp8
        self.std_dn = bfp8_fidelity is not None and w_down.dtype == bfp8
        self.cbs = self._cbs(w_gate_up.dtype, w_down.dtype)
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=bfp8_fidelity if (self.std_gu or self.std_dn) else ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        )

    def _cbs(self, gu_dtype, dn_dtype):
        tiny = ttnn.TileDescriptor(ttnn.Tile([1, 32]))
        bf = ttnn.bfloat16
        t32 = ttnn.TileDescriptor(ttnn.Tile([32, 32]))
        gtile = ttnn.Tile([32, 32]).get_tile_size(gu_dtype)
        dtile = ttnn.Tile([32, 32]).get_tile_size(dn_dtype)

        def cb(cores, total, *fmts):
            return ttnn.CBDescriptor(
                total_size=total,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=d, page_size=ps, tile=t) for i, d, ps, t in fmts
                ],
            )

        w = self.workers
        wc = _range_set([c for c, *_ in self.workers_info] + [self.collector])  # the collector multicasts x
        return [
            cb(wc, KT_GU * 64, (CB_X, bf, 64, tiny)),
            cb(w, self.nbuf * self.sbk * gtile, (CB_W, gu_dtype, gtile, t32)),
            cb(w, HID_TILES * dtile, (CB_WD, dn_dtype, dtile, t32)),  # the whole down column of the 8 experts
            cb(w, 2048, (CB_G, bf, 64, tiny), (CB_G32, bf, 2048, t32)),
            cb(w, 2048, (CB_U16, bf, 2048, t32)),
            cb(w, 2048, (CB_WS16, bf, 2048, t32)),
            cb(w, 2048, (CB_H16, bf, 2048, t32)),
            cb(self.all_cores, HID_TILES * 64, (CB_HID, bf, 64, tiny)),
            cb(w, 64, (CB_O, bf, 64, tiny)),
            cb(w, 2048, (CB_Z, bf, 2048, t32)),
            cb(w, 128, (CB_SCR, ttnn.uint16, 64, tiny)),  # ids + weights (reader), ids (writer)
        ]

    def __call__(self, x, weights, indices):
        """x [1,1,1,H] bf16 (L1), weights / indices from Inkling2MoE.route -> [1,1,1,H] bf16 L1."""
        for t in (x, weights, indices):
            assert t.memory_config().buffer_type == ttnn.BufferType.L1, t.memory_config()
            assert t.layout == ttnn.TILE_LAYOUT
        assert x.dtype == ttnn.bfloat16 and weights.dtype == ttnn.bfloat16 and indices.dtype == ttnn.uint16
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, HIDDEN]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG
        )
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        common = {"topk": TOP_K, "hid_tiles": HID_TILES}
        r_named = {
            "cb_x": CB_X, "cb_w": CB_W, "cb_wd": CB_WD, "cb_ws16": CB_WS16, "cb_scr": CB_SCR, "sbk": self.sbk,
            "nbuf": self.nbuf, "kt_gu": KT_GU, "kt_dn": KT_DN, "x_tiles": KT_GU, "n_banks": self.nb, "base": self.base,
            "x_lo": X_LO, "dn_slots": self.dn_slots, **common,
        }  # fmt: skip
        if self.pf:
            r_named["n_pf"] = len(pf_columns("experts.w_down", self.nb))
        w_named = {
            "cb_g": CB_G, "cb_u16": CB_U16, "sem_u": SEM_U,
            "cb_h16": CB_H16, "cb_hid": CB_HID, "cb_o": CB_O, "cb_z": CB_Z, "cb_scr": CB_SCR, "sem_g": SEM_G,
            "sem_r": SEM_R, "n_banks": self.nb, "x_lo": X_LO, "cb_x": CB_X, "x_tiles": KT_GU, "sem_x": SEM_X, **common,
        }  # fmt: skip
        c_named = {
            "cb_x": CB_X, "cb_w": CB_W, "cb_wd": CB_WD, "cb_g": CB_G, "cb_g32": CB_G32, "cb_u16": CB_U16,
            "cb_ws16": CB_WS16, "cb_h16": CB_H16, "cb_hid": CB_HID, "cb_o": CB_O, "sbk": self.sbk, "kt_gu": KT_GU,
            **common,
        }  # fmt: skip
        if self.dense_out and self.dense is None:  # first call is eager (from host: zeros, pad stays zero)
            self.dense = F.dense_tensor(self.device, torch.zeros(1, HIDDEN), ttnn.bfloat16)
        cx, cy = self.noc[self.collector]
        r_args, w_args, c_args = [], [], []
        for c, f, b, k, j, off, dslot, n, gate in self.workers_info:
            core = ttnn.CoreCoord(*c)
            px, py = self.noc[gate]
            r_args.append((core, [x.buffer_address(), weights.buffer_address(), indices.buffer_address(),
                                  self.w_gate_up.buffer_address(), self.w_down.buffer_address(), f, b, k, off, dslot]
                           + ([self.gu_x.buffer_address(), self.dn_x.buffer_address()] if self.pf else [])))  # fmt: skip
            w_args.append((core, [f, j, px, py, cx, cy, out.buffer_address(), n, indices.buffer_address(), b]
                           + ([self.dense.buffer_address()] if self.dense_out else [])))  # fmt: skip
            c_args.append((core, [f]))
        (x0, y0), (x1, y1) = self.mcast_rect
        dm = ttnn.DataMovementConfigDescriptor
        xdef = [("X_MCAST", "1")] * self.x_mcast
        flip = {(w[0][0], w[0][1]) for w in self.workers_info if w[2] in self.noc1_banks}
        dm_kernels = []
        for sel, rnoc, wnoc in ((False, ttnn.NOC.NOC_0, ttnn.NOC.NOC_1), (True, ttnn.NOC.NOC_1, ttnn.NOC.NOC_0)):
            cores = [c for c, *_ in self.workers_info if ((c[0], c[1]) in flip) == sel]
            if not cores:
                continue
            pick = lambda args: [(k, v) for k, v in args if ((k.x, k.y) in flip) == sel]  # noqa: E731
            dm_kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/moe_bank_reader.cpp",
                    core_ranges=_range_set(cores),
                    compile_time_args=acc(x)
                    + acc(weights)
                    + acc(indices)
                    + acc(self.w_gate_up)
                    + acc(self.w_down)
                    + (acc(self.gu_x) + acc(self.dn_x) if self.pf else []),
                    named_compile_time_args=list(r_named.items()),
                    defines=xdef + [("PF_LAYOUT", "1")] * self.pf,
                    runtime_args=pick(r_args),
                    config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rnoc),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/moe_bank_writer.cpp",
                    core_ranges=_range_set(cores),
                    compile_time_args=acc(out) + acc(indices) + (acc(self.dense) if self.dense_out else []),
                    named_compile_time_args=list(w_named.items()),
                    defines=xdef + [("DENSE_OUT", "1")] * self.dense_out,
                    runtime_args=pick(w_args),
                    config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=wnoc),
                ),
            ]
        kernels = dm_kernels + [
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/moe_bank_compute.cpp",
                core_ranges=self.workers,
                named_compile_time_args=list(c_named.items()),
                defines=[("STD_MM_GU", "1")] * self.std_gu + [("STD_MM_DN", "1")] * self.std_dn,
                runtime_args=c_args,
                config=self.compute_config,
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/moe_expert_collector.cpp",
                core_ranges=_range_set([self.collector]),
                compile_time_args=acc(x) if self.x_mcast else [],
                defines=xdef,
                named_compile_time_args=[
                    ("cb_hid", CB_HID),
                    ("sem_g", SEM_G),
                    ("sem_r", SEM_R),
                    ("n_senders", HID_TILES),  # one count per hidden tile written
                    ("hid_bytes", HID_TILES * 64),
                    ("cb_x", CB_X),
                    ("x_tiles", KT_GU),
                    ("sem_x", SEM_X),
                ],
                runtime_args=[
                    (ttnn.CoreCoord(*self.collector), [x0, y0, x1, y1, self.mcast_dests, x.buffer_address()])
                ],
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        sems = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all_cores, initial_value=0)
            for i in (SEM_U, SEM_G, SEM_R, SEM_X)
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=self.cbs)
        io = (
            [x, weights, indices, self.w_gate_up, self.w_down]
            + ([self.gu_x, self.dn_x] if self.pf else [])
            + ([self.dense] if self.dense_out else [])
            + [out]
        )
        y = ttnn.generic_op(io, prog)
        if self.dense_out:  # register the twin for this output; drop the previous output's entry (no leak)
            if self._last is not None:
                F._TWIN.pop(id(self._last), None)
            F.set_twin(y, self.dense, 1)
            self._last = y
        return y
