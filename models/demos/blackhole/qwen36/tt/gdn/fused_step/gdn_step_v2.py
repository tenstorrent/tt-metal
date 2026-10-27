# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused Gated DeltaNet DECODE step, v2 (QWEN36_GDN_FUSED_STEP_V2=1 together with QWEN36_GDN_FUSED_STEP=1).

Same contract as gdn_step.GDNFusedDecodeStep (v1): one program between the GDN in-projection and its out-projection,
the composite's tensors used in place (conv_states, rec_state, conv_taps), batch 1, fp32 state. What changes:

* Vt cores per value head (Nv * Vt cores; 16 on the TP8 Qwen3.5-9B pup): core (h, j) owns value-column tile j of head
  h. The recurrence is separable by value columns: Sd = decay * S[:, j], vread_j = k^T Sd, u_j = beta (v_j - vread_j),
  Snew[:, j] = Sd + k u_j^T, o_j = q^T Snew[:, j] -- Kt state tiles per core instead of Kt * Vt.
* Dense per-token vectors. A decode vector lives in row 0 of a TILE (31 padding rows). The reader packs row 0 of
  every channel tile a core needs (q: Kt, k: Kt, v_j, z_j) into ONE dense tile, one 32-element chunk per even row
  (64 B NOC reads of rows 0-1 of faces 0/1), for each of the 3 conv-history slots and the new input. The conv, SiLU
  (also of z, whose conv taps are 0, 0, 0, 1), and both L2 norms then run on one tile instead of 2 Kt + 2 Vt tiles;
  the group sums of the L2 norms are two 1-tile matmuls (x^2 @ ones, then a group-indicator matrix @ that).
  The reader splits the normalised dense tile back into row-0 tiles (q_t, k_t, v_j, silu(z_j)) and k_t columns
  (the rank-1 update's left operand) with RISC-V copies (exact fp32; no transposes on the compute core).
* The gated RMSNorm needs sum(o^2) over the whole head: each core reduces its 32 columns and the 4 cores of a head
  swap the partial sums (inline 4-byte NOC writes + a semaphore), each core summing all four in the same order.
* The conv shift register (st0 <- st1 <- st2 <- st3 <- x, full tiles, bit-identical to the composite's ttnn.copy)
  is spread over the cores: every q/k channel tile of a key head has one owner among the rf * Vt cores that read it,
  v_j's owner is core (h, j). Owners batch their reads, wait until every reader of the group has finished, then batch
  the writes, all on the writer RISC while the compute core works.
* Taps and other constants are prebuilt host-side in the dense layout (one tensor, one read per tile).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import torch

import ttnn

KERNELS = Path(os.environ.get("QWEN36_GDN_FUSED_STEP_V2_KERNELS") or (Path(__file__).parent / "kernels"))
TILE = 32
# CB ids shared with kernels/gdn_step_v2_*.cpp
CB = dict(
    h=0,
    x=1,
    tap=2,
    abraw=3,
    wscr=4,
    out=5,  # bf16
    const=6,
    c=7,
    t1=8,
    t2=9,
    n=10,
    aq=11,
    ak=12,
    kcol=13,
    v=14,
    sz=15,
    a=16,
    b=17,
    na=18,
    dt=19,
    decay=20,
    beta=21,
    S=22,
    sd=23,
    vread=24,
    u=25,
    snew=26,
    state=27,
    o=28,
    sq=29,
    ps=30,
    gather=31,
    r=32,
    w=33,
)
# const tile order in the replicated const tensor (fp32)
CONST_ONES, CONST_MCOL, CONST_ONESM, CONST_G, CONST_MQK, CONST_MVZ = range(6)
SEM_RD, SEM_X = 0, 1


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _per_device_values(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float().flatten() for d in ttnn.get_device_tensors(t)]


def dense_row(slot: int) -> int:
    """Tile row of dense slot s: even rows of faces 0/1 (s < 8), then of faces 2/3."""
    return 2 * slot if slot < 8 else 16 + 2 * (slot - 8)


class GDNFusedDecodeStepV2:
    def __init__(self, gdn, math_fidelity=ttnn.MathFidelity.HiFi4):
        self.gdn = gdn
        mesh = gdn.mesh
        self.mesh = mesh
        self.Nk, self.Nv, self.Dk, self.Dv = gdn.Nk, gdn.Nv, gdn.Dk, gdn.Dv
        assert gdn.K == 4, "four conv taps"
        assert self.Dk % TILE == 0 and self.Dv % TILE == 0 and self.Dk <= 128 and self.Dv <= 128
        self.Kt, self.Vt = self.Dk // TILE, self.Dv // TILE
        Kt, Vt = self.Kt, self.Vt
        assert 2 * Kt + 2 <= 16, "q, k, v_j, z_j must fit the 16 dense slots"
        assert gdn.key_dim_tp % TILE == 0 and gdn.qkv_dim_tp % TILE == 0
        self.k_off = gdn.key_dim_tp // TILE
        self.v_off = 2 * gdn.key_dim_tp // TILE
        self.z_off = gdn.qkv_dim_tp // TILE
        self.ab_tile = gdn.qkvz_dim_tp // TILE
        self.a_col = gdn.qkvz_dim_tp % TILE
        assert self.a_col + 2 * self.Nv <= 16, "a|b must sit in face 0, row 0 of one tile"
        self.rf = self.Nv // self.Nk
        assert self.rf * self.Nk == self.Nv
        self.G = self.rf * Vt  # cores per key head (readers of its q/k channels)
        self.math_fidelity = math_fidelity
        rep = ttnn.ReplicateTensorToMesh(mesh)
        shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)
        ndev = mesh.get_num_devices()
        self.ncores = self.Nv * Vt

        # per-head gate constants (v1 layout): tile 2h = -exp(A_log)[h] at [0, 0], tile 2h + 1 = dt_bias[h] at [0, 0]
        na = _per_device_values(gdn.tw["neg_exp_A"])
        dt = _per_device_values(gdn.tw["dt_bias"])
        g = torch.zeros(ndev, 1, TILE, 2 * self.Nv * TILE)
        for d in range(ndev):
            for h in range(self.Nv):
                g[d, 0, 0, 2 * h * TILE] = na[d][h]
                g[d, 0, 0, (2 * h + 1) * TILE] = dt[d][h]
        self.gconst = ttnn.from_torch(
            g,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=shard0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w = _per_device_values(gdn.tw["norm_w"])[0][: self.Dv]
        nw = torch.zeros(1, 1, TILE, self.Dv)
        nw[0, 0, 0] = w
        self.normw = ttnn.from_torch(
            nw,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # replicated fp32 constants: reduce scaler (ones), mcol (1 at [0,0]), all-ones matrix, group matrix (q rows:
        # Dk, so rsqrt(Dk |q|^2 + eps) = Dk^-0.5 / |q| folds the q scale in, exact in tf32; k rows: 1), q|k row mask,
        # v|z row mask (pass-through rows)
        cm = torch.zeros(6, TILE, TILE)
        cm[CONST_ONES] = 1.0
        cm[CONST_MCOL, 0, 0] = 1.0
        cm[CONST_ONESM] = 1.0
        qrows = [dense_row(t) for t in range(Kt)]
        krows = [dense_row(Kt + t) for t in range(Kt)]
        for grp, wgt in ((qrows, float(self.Dk)), (krows, 1.0)):
            for r in grp:
                for r2 in grp:
                    cm[CONST_G, r, r2] = wgt
        for r in qrows + krows:
            cm[CONST_MQK, r, :] = 1.0
        for s in (2 * Kt, 2 * Kt + 1):
            cm[CONST_MVZ, dense_row(s), :] = 1.0
        cmt = cm.permute(1, 0, 2).reshape(1, 1, TILE, 6 * TILE)
        self.consts = ttnn.from_torch(
            cmt,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # dense conv taps per core (h, j): 4 bf16 tiles, tap k of the channel of each dense slot in its row
        taps = [_per_device_values(gdn.tw["conv_taps"][k]) for k in range(4)]
        dense = torch.zeros(ndev, 1, TILE, self.ncores * 4 * TILE)
        for d in range(ndev):
            for h in range(self.Nv):
                kh = h // self.rf
                for j in range(Vt):
                    ci = h * Vt + j
                    chans = self._channels(h, kh, j)
                    for k in range(4):
                        col0 = (ci * 4 + k) * TILE
                        for s, ch in enumerate(chans):
                            dense[d, 0, dense_row(s), col0 : col0 + TILE] = taps[k][d][ch * TILE : (ch + 1) * TILE]
                        dense[d, 0, dense_row(2 * Kt + 1), col0 : col0 + TILE] = 1.0 if k == 3 else 0.0
        self.taps_dense = ttnn.from_torch(
            dense,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=shard0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # core grid: x = value-column tile j, y = value head h
        self.cores = [(h, j, ttnn.CoreCoord(j, h)) for h in range(self.Nv) for j in range(Vt)]
        self.core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(Vt - 1, self.Nv - 1))])
        self.phys = {(h, j): mesh.worker_core_from_logical_core(c) for h, j, c in self.cores}
        self.mesh_range = ttnn.MeshCoordinateRange(
            ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(mesh.shape[0] - 1, mesh.shape[1] - 1)
        )
        # q/k channel ownership inside a key group: channel index c in [0, 2 Kt) (q tiles then k tiles) is shifted
        # by group member c % G (member g = (h - kh * rf) * Vt + j)
        self.n_own_qk_max = -(-2 * Kt // self.G)

    def _channels(self, h, kh, j):
        """Conv channel tiles (in the qkv block) of the dense slots of core (h, j): q_t, k_t of kh, then v_j."""
        Kt = self.Kt
        return (
            [kh * Kt + t for t in range(Kt)]
            + [self.k_off + kh * Kt + t for t in range(Kt)]
            + [self.v_off + h * self.Vt + j]
        )

    def _owned(self, h, kh, j):
        member = (h - kh * self.rf) * self.Vt + j
        qk = self._channels(h, kh, j)[: 2 * self.Kt]
        own = [qk[c] for c in range(2 * self.Kt) if c % self.G == member]
        return own + [self.v_off + h * self.Vt + j]

    def _cbs(self):
        Kt = self.Kt
        nown = (self.n_own_qk_max + 1) * 4
        bf = {"h": 3, "x": 1, "tap": 4, "abraw": 1, "wscr": nown, "out": 1}
        f32 = {
            "const": 6,
            "c": 1,
            "t1": 1,
            "t2": 1,
            "n": 1,
            "aq": Kt,
            "ak": Kt,
            "kcol": Kt,
            "v": 1,
            "sz": 1,
            "a": 1,
            "b": 1,
            "na": 1,
            "dt": 1,
            "decay": 1,
            "beta": 1,
            "S": Kt,
            "sd": Kt,
            "vread": 1,
            "u": 1,
            "snew": Kt,
            "state": Kt,
            "o": 1,
            "sq": 1,
            "ps": 1,
            "gather": 1,
            "r": 1,
            "w": 1,
        }
        cbs = []
        for names, dtype, page in ((bf, ttnn.bfloat16, 2048), (f32, ttnn.float32, 4096)):
            for name, n in names.items():
                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=n * page,
                        core_ranges=self.core_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=CB[name], data_format=dtype, page_size=page)
                        ],
                    )
                )
        return cbs

    def __call__(self, proj, out=None):
        gdn = self.gdn
        st = gdn.conv_states
        rec = gdn.rec_state
        assert rec.dtype == ttnn.float32, "fp32 recurrent state (QWEN35_GDN_STATE_BF16 unset)"
        assert proj.shape[-1] == gdn.qkvz_dim_tp + 2 * self.Nv, f"in-projection width {proj.shape[-1]}"
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, gdn.value_dim_tp]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG
            )
        Kt, Vt = self.Kt, self.Vt
        reader_ct = [Kt, Vt, self.z_off, self.ab_tile, self.a_col, self.Nv, SEM_RD, self.G]
        for t in (proj, st[1], st[2], st[3], self.taps_dense, self.gconst, self.normw, self.consts, rec):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [Kt, Vt, SEM_RD, SEM_X, self.G]
        for t in (out, rec, st[0], st[1], st[2], st[3], proj):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [
            Kt,
            f32_bits(1e-6),
            f32_bits(1.0 / self.Dv),
            f32_bits(1e-6),
            f32_bits(1.0),
            f32_bits(20.0),
        ]
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        a = lambda t: t.buffer_address()  # noqa: E731
        for h, j, core in self.cores:
            kh = h // self.rf
            chans = self._channels(h, kh, j)
            group = [self.phys[(hh, jj)] for hh in range(kh * self.rf, (kh + 1) * self.rf) for jj in range(Vt)]
            peers = [self.phys[(h, jj)] for jj in range(Vt) if jj != j]
            rd[core.x][core.y] = (
                [
                    a(proj),
                    a(st[1]),
                    a(st[2]),
                    a(st[3]),
                    a(self.taps_dense),
                    a(self.gconst),
                    a(self.normw),
                    a(self.consts),
                    a(rec),
                    h,
                    j,
                    (h * Vt + j),
                ]
                + chans
                + [c for p in group for c in (p.x, p.y)]
            )
            owned = self._owned(h, kh, j)
            wr[core.x][core.y] = (
                [a(out), a(rec), a(st[0]), a(st[1]), a(st[2]), a(st[3]), a(proj), h, j, len(owned)]
                + owned
                + [c for p in peers for c in (p.x, p.y)]
            )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gdn_step_v2_reader.cpp"),
                core_ranges=self.core_set,
                compile_time_args=reader_ct,
                runtime_args=rd,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gdn_step_v2_writer.cpp"),
                core_ranges=self.core_set,
                compile_time_args=writer_ct,
                runtime_args=wr,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gdn_step_v2_compute.cpp"),
                core_ranges=self.core_set,
                compile_time_args=compute_ct,
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.math_fidelity, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=[
                ttnn.SemaphoreDescriptor(SEM_RD, ttnn.CoreType.WORKER, self.core_set, 0),
                ttnn.SemaphoreDescriptor(SEM_X, ttnn.CoreType.WORKER, self.core_set, 0),
            ],
            cbs=self._cbs(),
        )
        mesh_program = ttnn.MeshProgramDescriptor()
        mesh_program[self.mesh_range] = program
        io = [proj, st[0], st[1], st[2], st[3], self.taps_dense, self.gconst, self.normw, self.consts, rec, out]
        ttnn.generic_op(io, mesh_program)
        return out
