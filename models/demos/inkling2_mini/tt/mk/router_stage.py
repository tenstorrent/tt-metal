# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Stage each app's router weights into its router cores' L1 one program early.

The fused router (tt/fused/router.py) reads 8 fp32 W tiles (32 KB) per core from DRAM before its partial GEMM
tile, ~4 us of the router's critical path. Each app's conv program (one core on row 0, DRAM-light, already a w_all
fill host) also gets data-movement kernels on the router cores (rows 2..5, disjoint) that copy the NEXT app's W
tiles into one L1 shard per router core (kernels/stage_pages_fill.cpp, the page order the reader uses); that app's
router then reads them from local L1. Same bytes in the same CB pages: output unchanged.

    rs = install(model, overlap)   # after tt/mk/overlap.install; rs.uninstall() before overlap.uninstall

The conv of app i runs after the router of app i and before the router of app i + 1 (model.APPS order), so the
shard is never overwritten while in use. A router call whose fill did not run (first eager step) reads DRAM.
Rides on overlap's dispatch hook (overlap._orig_generic_op): host programs are recognized there, after overlap
attached its own pieces.
"""

import os

import ttnn
from models.demos.inkling2_mini.tt import model as MD
from models.demos.inkling2_mini.tt.mk import overlap as OV
from models.demos.inkling2_mini.tt.mk.stage import coalesce, program_cores

MK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
HOST = "conv_compute"  # kernel basename of the host program


class _Call:
    def __init__(self, rs, router):
        self.rs, self.router = rs, router

    def __getattr__(self, name):
        return getattr(self.router, name)

    def __call__(self, x):
        return self.rs.consume(self.router, x)


class RouterStage:
    def __init__(self, model):
        self.model = model
        self.layers = [lay for lay in model.layers if lay.moe.router is not None]
        r = self.layers[0].moe.router
        self.cores = [(int(c.x), int(c.y)) for c in r.cores]
        assert all([(int(c.x), int(c.y)) for c in lay.moe.router.cores] == self.cores for lay in self.layers)
        self.core_set = set(self.cores)
        self.kt, self.nt = r.kt, r.nt
        words = self.kt * 4096 // 4
        grid = coalesce(self.cores)  # every core reads its own shard at the same local address: shard order is free
        spec = ttnn.ShardSpec(grid, (1, words), ttnn.ShardOrientation.ROW_MAJOR)
        self.stage = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, len(self.cores), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, model.device,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec),
        )  # fmt: skip
        self.grid = grid
        self.order = [L for L, _ in MD.APPS]
        self.ptr, self.filled_for = 0, None
        self.n, self.staged = 0, 0
        self.calls = {}
        for lay in self.layers:
            self.calls[lay.L] = lay.moe.router
            lay.moe.router = _Call(self, lay.moe.router)
        self.prev = OV._orig_generic_op
        OV._orig_generic_op = self._op
        self._held = None  # (L1 address, host copy, memory config, shape) at a step boundary, from the first suspend()
        self.suspended = False

    def uninstall(self):
        OV._orig_generic_op = self.prev
        for lay in self.layers:
            lay.moe.router = self.calls[lay.L]
        if not self.suspended:
            ttnn.deallocate(self.stage)

    def suspend(self):
        """As Overlap.suspend (the hooks go off there): free the stage's L1 between two decode steps."""
        if self._held is None:
            st = self.stage
            self._held = (st.buffer_address(), ttnn.from_device(st), st.memory_config(), st.shape)
        self.suspended = True
        ttnn.deallocate(self.stage)

    def resume(self, retake):
        """As Overlap.resume: the stage back at its address with its step-boundary content, or False."""
        addr, host, mem, shape = self._held
        dev = self.model.device
        make = lambda: ttnn.allocate_tensor_on_device(shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, mem)  # noqa: E731
        stage = retake(make, lambda t: t.buffer_address(), ttnn.deallocate, addr)
        if stage is None:
            return False
        ttnn.copy_host_to_device_tensor(host, stage)
        self.stage, self.suspended = stage, False
        return True

    def fill_kernels(self, router):
        """The router's W tiles of every core: k 0..kt/2-1 on the reader RISC, the rest on the writer RISC."""
        half = self.kt // 2
        out = []
        dm = ttnn.DataMovementProcessor
        halves = ((0, half, dm.RISCV_1, ttnn.NOC.NOC_0), (half, self.kt - half, dm.RISCV_0, ttnn.NOC.NOC_1))
        for k0, kz, proc, noc in halves:
            rt = []
            for i, c in enumerate(self.cores):
                chunk, n = i // self.nt, i % self.nt
                col0 = (chunk * self.kt + k0) * self.nt + n  # page (chunk * kt + k) * nt + n for k = k0 ..
                rt.append(
                    (ttnn.CoreCoord(*c), [router.w.buffer_address(), col0, self.stage.buffer_address() + k0 * 4096])
                )
            out.append(ttnn.KernelDescriptor(
                kernel_source=f"{MK}/stage_pages_fill.cpp", core_ranges=self.grid,
                named_compile_time_args=[("kz", kz), ("nt", 1), ("nct", self.nt)], runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
            ))  # fmt: skip
        return out

    def _op(self, io, prog):
        names = {os.path.basename(k.kernel_source).rsplit(".", 1)[0] for k in prog.kernels}
        if HOST in names and self.filled_for is None:
            L = self.order[self.ptr]
            router = self.calls.get(L)
            if router is not None and not (program_cores(prog) & self.core_set):
                prog = ttnn.ProgramDescriptor(
                    kernels=list(prog.kernels) + self.fill_kernels(router), semaphores=list(prog.semaphores),
                    cbs=list(prog.cbs),
                )  # fmt: skip
                io = list(io[:-1]) + [router.w, self.stage] + [io[-1]]
                self.filled_for = self.ptr
        return self.prev(io, prog)

    def consume(self, router, x):
        L = self.order[self.ptr]
        assert self.calls[L] is router, ("router order out of step", self.ptr, L)
        staged = self.filled_for == self.ptr
        out = router(x, w_stage=self.stage if staged else None)
        self.n += 1
        self.staged += int(staged)
        self.ptr = (self.ptr + 1) % len(self.order)
        self.filled_for = None
        return out

    def stats(self):
        return {"calls": self.n, "staged": self.staged}


def install(model):
    return RouterStage(model)
