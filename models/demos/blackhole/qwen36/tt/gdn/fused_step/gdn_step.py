# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One fused device program for a Gated DeltaNet DECODE step (batch 1, tensor parallel, one device of the TP ring).

Replaces everything between the GDN in-projection and its out-projection in ``TPGatedDeltaNet.forward_decode``
(~60 small device programs per layer: the conv shift register, SiLU, q/k/v split, GQA repeat, L2 norms, gates, the
recurrence, the gated RMSNorm) with ONE program, one core per local value head (kernels/gdn_step_*.cpp):

  reader   the head's conv channel tiles of the projection output and of the 3 history slots, the 4 taps, z, the
           a|b tile (a_h, b_h unpacked to scalar tiles), the gate constants, the norm weight, the recurrent state
  compute  conv + SiLU, q/k L2 norm (q scaled), decay = exp(-exp(A_log) softplus(a + dt_bias)), beta = sigmoid(b),
           the delta-rule recurrence of ttnn.transformer.fused_recurrent_gated_delta_rule (T = 1, fp32 state, HiFi4),
           RMSNorm(o) * norm_w * SiLU(z)
  writer   the gated output (bf16, the out-projection's input), the state (in place), the conv shift register
           st0 <- st1 <- st2 <- st3 <- x of the channels the core owns (identical to the composite's ttnn.copy calls)

The composite's tensors are used as they are (conv_states, rec_state, conv_taps), so the fused and composite paths can
alternate. Selected by QWEN36_GDN_FUSED_STEP=1 (gdn/tp.py). Batch 1 and the fused qkvzab in-projection only.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import torch

import ttnn

# QWEN36_GDN_FUSED_STEP_KERNELS: an alternative kernel directory (profiling copies); default the kernels next to this file
KERNELS = Path(os.environ.get("QWEN36_GDN_FUSED_STEP_KERNELS") or (Path(__file__).parent / "kernels"))
# CB ids shared with the kernels
CB = dict(
    h=0,
    x=1,
    tap=2,
    z=3,
    a=4,
    b=5,
    na=6,
    dt=7,
    w=8,
    mcol=9,
    one=10,
    S=11,
    c=12,
    sz=13,
    sq=14,
    r=15,
    qn=16,
    kn=17,
    v=18,
    decay=19,
    beta=20,
    sd=21,
    vread=22,
    u=23,
    kcol=24,
    snew=25,
    state=26,
    o=27,
    out=28,
    abraw=29,
    wscr=30,
)
SEM = 0
TILE = 32


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _per_device_values(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float().flatten() for d in ttnn.get_device_tensors(t)]


class GDNFusedDecodeStep:
    def __init__(self, gdn, math_fidelity=ttnn.MathFidelity.HiFi4):
        self.gdn = gdn
        mesh = gdn.mesh
        self.mesh = mesh
        self.Nk, self.Nv, self.Dk, self.Dv = gdn.Nk, gdn.Nv, gdn.Dk, gdn.Dv
        assert gdn.K == 4, "four conv taps"
        assert self.Dk % TILE == 0 and self.Dv % TILE == 0 and self.Dk <= 128 and self.Dv <= 128
        self.Kt, self.Vt = self.Dk // TILE, self.Dv // TILE
        self.NC = 2 * self.Kt + self.Vt
        assert gdn.key_dim_tp % TILE == 0 and gdn.qkv_dim_tp % TILE == 0
        self.k_off = gdn.key_dim_tp // TILE
        self.v_off = 2 * gdn.key_dim_tp // TILE
        self.z_off = gdn.qkv_dim_tp // TILE
        self.ab_tile = gdn.qkvz_dim_tp // TILE
        self.a_col = gdn.qkvz_dim_tp % TILE
        assert self.a_col + 2 * self.Nv <= 16, "a|b must sit in face 0, row 0 of one tile"
        self.rf = self.Nv // self.Nk
        assert self.rf * self.Nk == self.Nv
        self.math_fidelity = math_fidelity
        rep = ttnn.ReplicateTensorToMesh(mesh)
        ndev = mesh.get_num_devices()

        # per-head gate constants: tile 2h = -exp(A_log)[h] at [0, 0], tile 2h + 1 = dt_bias[h] at [0, 0]
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
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
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
        cm = torch.zeros(1, 1, TILE, 2 * TILE)
        cm[0, 0, :, :TILE] = 1.0  # reduce scaler: ones
        cm[0, 0, 0, TILE] = 1.0  # mcol: 1 at (0, 0)
        self.cmask = ttnn.from_torch(
            cm,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.cores = [ttnn.CoreCoord(h, 0) for h in range(self.Nv)]
        self.core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.Nv - 1, 0))])
        self.phys = [mesh.worker_core_from_logical_core(c) for c in self.cores]
        self.mesh_range = ttnn.MeshCoordinateRange(
            ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(mesh.shape[0] - 1, mesh.shape[1] - 1)
        )

    def _cbs(self):
        Kt, Vt, NC, kv = self.Kt, self.Vt, self.NC, self.Kt * self.Vt
        bf = {"h": 3 * NC, "x": NC, "tap": 4 * NC, "z": Vt, "out": Vt, "abraw": 1, "wscr": 1}
        f32 = {
            "a": 1,
            "b": 1,
            "na": 1,
            "dt": 1,
            "w": Vt,
            "mcol": 1,
            "one": 1,
            "S": kv,
            "c": NC,
            "sz": Vt,
            "sq": 1,
            "r": 1,
            "qn": Kt,
            "kn": Kt,
            "v": Vt,
            "decay": 1,
            "beta": 1,
            "sd": kv,
            "vread": Vt,
            "u": Vt,
            "kcol": Kt,
            "snew": kv,
            "state": kv,
            "o": Vt,
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
        """proj: the fused qkvzab in-projection output [1, 1, qkvz + 2 Nv] bf16 TILE (row 0 = the token);
        returns the gated output [1, 1, value_dim_tp] bf16 TILE (L1), the out-projection's input."""
        gdn = self.gdn
        st = gdn.conv_states
        taps = gdn.tw["conv_taps"]
        rec = gdn.rec_state
        assert rec.dtype == ttnn.float32, "fp32 recurrent state (QWEN35_GDN_STATE_BF16 unset)"
        assert proj.shape[-1] == gdn.qkvz_dim_tp + 2 * self.Nv, f"in-projection width {proj.shape[-1]}"
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, gdn.value_dim_tp]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG
            )
        Kt, Vt, NC = self.Kt, self.Vt, self.NC
        reader_ct = [Kt, Vt, NC, self.k_off, self.v_off, self.z_off, self.ab_tile, self.a_col, self.Nv, SEM]
        for t in (
            proj,
            st[1],
            st[2],
            st[3],
            taps[0],
            taps[1],
            taps[2],
            taps[3],
            self.gconst,
            self.normw,
            self.cmask,
            rec,
        ):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [Kt, Vt, self.k_off, self.v_off, SEM]
        for t in (out, rec, st[0], st[1], st[2], st[3], proj):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [
            Kt,
            Vt,
            NC,
            f32_bits(self.Dk**-0.5),
            f32_bits(1e-6),
            f32_bits(1.0 / self.Dv),
            f32_bits(1e-6),
            f32_bits(1.0),
            f32_bits(20.0),
        ]
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        a = lambda t: t.buffer_address()  # noqa: E731
        for h, core in enumerate(self.cores):
            kh = h // self.rf
            owner = kh * self.rf
            signal = 1 if (self.rf > 1 and h != owner) else 0
            p = self.phys[owner]
            rd[core.x][core.y] = [
                a(proj),
                a(st[1]),
                a(st[2]),
                a(st[3]),
                a(taps[0]),
                a(taps[1]),
                a(taps[2]),
                a(taps[3]),
                a(self.gconst),
                a(self.normw),
                a(self.cmask),
                a(rec),
                h,
                kh,
                signal,
                p.x,
                p.y,
            ]
            wr[core.x][core.y] = [
                a(out),
                a(rec),
                a(st[0]),
                a(st[1]),
                a(st[2]),
                a(st[3]),
                a(proj),
                h,
                kh,
                1 if h == owner else 0,
                self.rf - 1 if h == owner else 0,
            ]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gdn_step_reader.cpp"),
                core_ranges=self.core_set,
                compile_time_args=reader_ct,
                runtime_args=rd,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gdn_step_writer.cpp"),
                core_ranges=self.core_set,
                compile_time_args=writer_ct,
                runtime_args=wr,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gdn_step_compute.cpp"),
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
            semaphores=[ttnn.SemaphoreDescriptor(SEM, ttnn.CoreType.WORKER, self.core_set, 0)],
            cbs=self._cbs(),
        )
        mesh_program = ttnn.MeshProgramDescriptor()
        mesh_program[self.mesh_range] = program
        io = [
            proj,
            st[0],
            st[1],
            st[2],
            st[3],
            taps[0],
            taps[1],
            taps[2],
            taps[3],
            self.gconst,
            self.normw,
            self.cmask,
            rec,
            out,
        ]
        ttnn.generic_op(io, mesh_program)
        return out
