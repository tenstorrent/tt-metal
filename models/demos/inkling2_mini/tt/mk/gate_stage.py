# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""M2 staging of the message gates' F / G weights (tt/fused/gate.py combine program) into L1.

A combine core reads its kz x nt fp32 tiles of F and of G (pages kt * nct + col0 + j) from DRAM: 48 KB per core for
the QKV gate, 16 KB for the mlp gate, 3.5 / 1.2 MB per program, DRAM bound. ``GateStage`` copies them earlier
into the core's L1 staging shard (kernels/stage_pages_fill.cpp, riding in a DRAM-light program, tt/mk/overlap.py)
and runs the combine with staged copies of its reader / writer (kernels/stage_gate_{reader,writer}.cpp) that take
them from there. Same bytes in the same CB pages, compute unchanged: bit-equal. Cores that cannot hold staging
(the hosting programs' cores, other projections' staging cores) read DRAM as before.
"""

import os

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.mk.stage import coalesce

MK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


class GateStage:
    def __init__(self, gate, forbidden):
        """gate: a tt/fused/gate._FusedGate (FusedQKVGate / FusedMessageGate). forbidden: (x, y) cores that must not
        stage (hosts, other staging)."""
        self.gate = gate
        cores, _ = gate.comb_cores
        self.cores = [(int(c.x), int(c.y)) for c in cores]
        self.kz, self.nt, self.nct = gate.kz, gate.nt, gate.ntc
        self.staged = [i for i, c in enumerate(self.cores) if c not in forbidden]
        self.stage_cores = {self.cores[i] for i in self.staged}
        self.bytes = 2 * self.kz * self.nt * 4096  # F then G
        self.grid = coalesce(sorted(self.stage_cores))
        self.stage = None  # the shared staging tensor (set by the owner)

    def fill_kernels(self, part=None):
        """F on the reader RISC, G on the writer RISC of every staging core."""
        g, s = self.gate, self.stage.buffer_address()
        named = [("kz", self.kz), ("nt", self.nt), ("nct", self.nct)]
        out = []
        for t, off, proc, noc in ((g.f, 0, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
                                  (g.g, self.kz * self.nt * 4096, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1)):  # fmt: skip
            rt = [(ttnn.CoreCoord(*self.cores[i]), [t.buffer_address(), i * self.nt, s + off]) for i in self.staged]
            out.append(ttnn.KernelDescriptor(
                kernel_source=f"{MK}/stage_pages_fill.cpp", core_ranges=self.grid, named_compile_time_args=named,
                runtime_args=rt, config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
            ))  # fmt: skip
        return out

    @property
    def fill_io(self):
        return [self.gate.f, self.gate.g, self.stage]

    def attach(self, prog, part=None):
        from models.demos.inkling2_mini.tt.mk.stage import program_cores

        clash = program_cores(prog) & self.stage_cores
        assert not clash, f"gate staging cores also used by the host program: {sorted(clash)[:8]}"
        return ttnn.ProgramDescriptor(
            kernels=list(prog.kernels) + self.fill_kernels(), semaphores=list(prog.semaphores), cbs=list(prog.cbs)
        )

    def combine(self, z, h, m, staged):
        """tt/fused/gate._FusedGate._combine with the staged reader / writer (staged: F / G of the staging cores are
        in L1)."""
        gt = self.gate
        cores, crs = gt.comb_cores
        out, out_dense = gt.out.get_twin("out", gt.rows, gt.c, ttnn.float32)
        block = int(gt.block)

        def src(t):
            tw = F.twin(t)
            if tw is not None and t.dtype == ttnn.float32:
                return tw[0].buffer_address(), 1, tw[1]
            return t.buffer_address(), 0, 0

        (h_a, h_d, h_n), (m_a, m_d, m_n) = src(h), src(m)
        s_addr = self.stage.buffer_address() if staged else 0
        on = set(self.staged)
        rd, wr = {}, {}
        for i, core in enumerate(cores):
            col0 = i * gt.nt
            st = s_addr if i in on else 0
            rd[core] = [z.buffer_address(), 0, gt.f.buffer_address(), gt.g.buffer_address(), h_a, m_a, col0]
            rd[core] += [h_d, m_d, h_n, m_n, i % 8, st]
            wr[core] = [out.buffer_address(), col0, out_dense.buffer_address(), gt.g.buffer_address()]
            wr[core] += [gt.f.buffer_address(), st]
        named = dict(kz=gt.kz, nt=gt.nt, nct=gt.ntc, has_mask=0, f_w=gt.kz // 2 if gt.kz > 2 else 0, block=block,
                     h_tile=F.tile_bytes(h.dtype), rows=gt.rows, pages=F.pages(gt.c))  # fmt: skip
        fp32_cbs = [0, 1] + ([4, 5] if h.dtype == ttnn.float32 else [])
        rel = os.path.relpath(MK, F.KDIR)
        kernels = [
            F.kernel(f"{rel}/stage_gate_reader.cpp", ttnn.ReaderConfigDescriptor(), named, rd, crs),
            F.kernel(f"{rel}/stage_gate_writer.cpp", ttnn.WriterConfigDescriptor(), named, wr, crs),
            F.kernel("gate_combine_compute.cpp", F.compute_config(fp32_cbs), named, None, crs),
        ]
        f32 = ttnn.float32
        cbs = [
            F.cb(0, f32, gt.kz, cores=crs), F.cb(1, f32, gt.kz, cores=crs), F.cb(2, f32, gt.kz * gt.nt, cores=crs),
            F.cb(3, f32, gt.kz * gt.nt, cores=crs), F.cb(4, h.dtype, gt.nt, cores=crs), F.cb(5, h.dtype, gt.nt, cores=crs),
            F.cb(6, ttnn.bfloat16, 1, 32, cores=crs), F.cb(16, ttnn.bfloat16, gt.kz, cores=crs),
            F.cb(17, f32, gt.nt, cores=crs),
        ]  # fmt: skip
        tws = [t[0] for t in (F.twin(h), F.twin(m)) if t is not None]
        io = [z, h, m, *tws, gt.f, gt.g, self.stage, out_dense, out]
        F.run(io, kernels, cbs)
        return out
