# SPDX-License-Identifier: Apache-2.0
"""tp32-prefill2: host side of the TP32 grouped-GEMM prefill kernels (tt/tp32pf_kernels/k1b_*, k2b_*), JIT generic_op.

Per chip (v1 layout: this chip holds intermediate columns [64 c, 64 c + 64) of every expert, W01 [bank][e][b][k][j] and W2 v1 order):
  K1b gate/up: rows of the chunk (x [Tt, 6144] row-major, replicated) gathered per expert (pair list), tilized, matmul with the expert
       slice resident in L1, SwiGLU, scaled by the pair score -> act tiles [mt][2] (DRAM, 64 columns of the intermediate)
  K2b down (6 passes S, columns 24 b + 4 S + n): act @ W2 slice, untilized rows are written by NoC to the owner core of the token
       (owner o = token // 32 < Tt / 32), owners sum the 8 slot rows, tilize, write [Tt, 6144] TILE partial (this chip's 1/32 of every expert).
The caller reduces the partial over the 32 chips (reduce-scatter).
"""
import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.skinny import _acc, _core_set
from models.demos.blackhole.deepseek_v41_flash.tt.skinny import _hash as _hash0
from models.demos.blackhole.deepseek_v41_flash.tt.skinny_shared import _fb
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import TB4
from models.demos.blackhole.hy4_preview.tt.hy4_tp32pf_plan import mt_cap

NOADDR = (
    os.environ.get("TP32PF_HASH", "noaddr") == "noaddr"
)  # program hash without buffer addresses (rely on runtime-arg override on a cache hit)


def _hash(*vals):
    return _hash0(*[v for v in vals if not (NOADDR and isinstance(v, int) and v >= (1 << 24))])


KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp32pf_kernels")
LIMIT = 10.0
HID = 6144


def tile_cb(cs, idx, ntiles, fmt, page=2048):
    return ttnn.CBDescriptor(
        total_size=ntiles * page,
        core_ranges=cs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=idx, data_format=fmt, page_size=page, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32]))
            )
        ],
    )


def u32_cb(cs, idx, nbytes, page=2048):
    return ttnn.CBDescriptor(
        total_size=nbytes,
        core_ranges=cs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.uint32, page_size=page)],
    )


class PlanTensors:
    """device copies of one Plan (replicated per chip by the caller's mapper), or (from the device builder) the tensors it filled:
    plan [nc, 512] u32, gidx / dest [cap / 16, 512] u32 (row r = word r), sc [cap / 32, 1024] bf16 (row r = element r); all row-major, pages of 2 KB
    """

    def __init__(self, md=None, pl=None, mapper=None, tensors=None):
        if tensors is not None:
            self.plan, self.gidx, self.dest, self.sc = tensors
            return
        kw = dict(device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if mapper is not None:
            kw["mesh_mapper"] = mapper
        self.plan = ttnn.from_torch(pl.plan, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
        self.gidx = ttnn.from_torch(pl.gidx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
        self.dest = ttnn.from_torch(pl.dest, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
        self.sc = ttnn.from_torch(pl.sc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)

    def free(self):
        for t_ in (self.plan, self.gidx, self.dest, self.sc):
            ttnn.deallocate(t_)


class Tp32PlanBuilder:
    """device plan builder (tt/tp32pf_kernels/plan_build.cpp, one core): route words [1,1,Tt/32,512] int32 RM (per token 8 ids then 8 bf16 score
    bit words) -> PlanTensors (same content as hy4_tp32pf_plan.build_plan(xl1=False))"""

    def __init__(self, md, nc=110):
        self.md, self.nc = md, nc
        self.cs = _core_set([ttnn.CoreCoord(0, 0)])

    def alloc(self, tt):
        cap = mt_cap(tt)
        md, A = self.md, ttnn.allocate_tensor_on_device
        mk = lambda shp, dt: A(ttnn.Shape(shp), dt, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
        return PlanTensors(
            tensors=(
                mk([self.nc, 512], ttnn.uint32),
                mk([cap // 16, 512], ttnn.uint32),
                mk([cap // 16, 512], ttnn.uint32),
                mk([cap // 32, 1024], ttnn.bfloat16),
            )
        )

    def __call__(self, route, tt, pt=None):
        pt = pt or self.alloc(tt)
        cap = mt_cap(tt)
        rows = cap * 32
        nbytes = tt * 64 + rows * 10 + self.nc * 2048 + 8 * 256 * 4 + self.nc * 8 + 64
        npages = (nbytes + 2047) // 2048
        CB = 0
        cbs = [u32_cb(self.cs, CB, npages * 2048)]
        impl = os.environ.get("TP32PF_PLAN_IMPL", "2")
        common = [
            route.buffer_address(),
            pt.plan.buffer_address(),
            pt.gidx.buffer_address(),
            pt.dest.buffer_address(),
            pt.sc.buffer_address(),
        ]
        accs = _acc(route) + _acc(pt.plan) + _acc(pt.gidx) + _acc(pt.dest) + _acc(pt.sc)
        if impl == "2":  # two RISCs of one core (plan_build2.cpp)
            kerns = [
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/plan_build2.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.cs,
                    compile_time_args=[CB, tt, self.nc, cap, npages, role] + accs,
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=common,
                    config=cfg,
                )
                for role, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor()))
            ]
        else:
            kerns = [
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/plan_build.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.cs,
                    compile_time_args=[CB, tt, self.nc, cap, npages, int(os.environ.get("TP32PF_PLAN_MODE", "0"))]
                    + accs,
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=common,
                    config=ttnn.ReaderConfigDescriptor(),
                )
            ]
        prog = ttnn.ProgramDescriptor(kernels=kerns, semaphores=[], cbs=cbs)
        prog.custom_program_hash = _hash(
            0x7F24,
            impl,
            int(os.environ.get("TP32PF_PLAN_MODE", "0")),
            tt,
            self.nc,
            route.buffer_address(),
            pt.plan.buffer_address(),
            pt.gidx.buffer_address(),
            pt.dest.buffer_address(),
            pt.sc.buffer_address(),
        )
        ttnn.generic_op([route, pt.plan, pt.gidx, pt.dest, pt.sc], prog)
        return pt


class Tp32PrefillExperts:
    def __init__(self, md, w01, w2, nc=110, nown=None, limit=LIMIT, scale=1):
        self.md, self.w01, self.w2, self.nc, self.limit, self.scale = md, w01, w2, nc, limit, scale
        g = md.compute_with_storage_grid_size()
        allc = [ttnn.CoreCoord(xx, yy) for yy in range(g.y) for xx in range(g.x)]
        self.cores = allc[:nc]
        self.cs = _core_set(self.cores)
        self.phys = [md.worker_core_from_logical_core(ttnn.CoreCoord(c.x, c.y)) for c in self.cores]
        self.nown = nown
        self.dbg = int(os.environ.get("TP32PF_DBG", "0"))
        self.var = int(os.environ.get("TP32PF_VAR", "0"))
        self.perf = int(os.environ.get("TP32PF_PERF", "0"))
        self.xl1 = int(os.environ.get("TP32PF_XL1", "1"))
        self.split = int(os.environ.get("TP32PF_SPLIT", "0"))
        self.k1v = os.environ.get("TP32PF_K1", "c")
        self.k2v = os.environ.get("TP32PF_K2", "f")

    def alloc(self, tt):
        cap = mt_cap(tt)
        md = self.md
        act = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, cap * 2 * 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, tt, HID]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        return act, out

    # ------------------------------------------------------------------ K1c (full-row gather, single-buffered weights; needs a plan with xl1=False)
    def _core_rt(self, cores):
        """per-core runtime args [core index] of the K1/K2 kernels. HY4_B49_RTCACHE=1: built once (the 110 x proxy assignments cost ~0.4 ms of host time per launch)."""
        if os.environ.get("HY4_B49_RTCACHE", "0") == "1":
            rt = getattr(self, "_rt_cached", None)
            if rt is None:
                rt = self._rt_cached = self._build_rt(cores)
            return rt
        return self._build_rt(cores)

    @staticmethod
    def _build_rt(cores):
        rt = ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            rt[c.x][c.y] = [i]
        return rt

    def k1c(self, x, pt, act):
        cs, cores = self.cs, self.cores
        CB_X, CB_W, CB_O, CB_META, CB_WS, CB_RM, CB_SC, CB_O2, CB_IX = range(9)
        cbs = [
            tile_cb(cs, CB_X, 192, ttnn.bfloat16),
            tile_cb(cs, CB_W, 768, ttnn.bfloat4_b, TB4),
            tile_cb(cs, CB_O, 2, ttnn.bfloat16),
            u32_cb(cs, CB_META, 2048),
            u32_cb(cs, CB_WS, 2048),
            tile_cb(cs, CB_RM, 192, ttnn.bfloat16),
            tile_cb(cs, CB_SC, 2, ttnn.bfloat16),
            tile_cb(cs, CB_O2, 4, ttnn.bfloat16),
            u32_cb(cs, CB_IX, 2048),
        ]
        rt = self._core_rt(cores)
        rd = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k1c_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_RM, CB_META, CB_IX, CB_SC] + _acc(pt.plan) + _acc(pt.gidx) + _acc(x) + _acc(pt.sc),
            runtime_args=rt,
            common_runtime_args=[
                x.buffer_address(),
                pt.plan.buffer_address(),
                pt.gidx.buffer_address(),
                pt.sc.buffer_address(),
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        wr = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k1c_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_O2, CB_WS, CB_W] + _acc(pt.plan) + _acc(act),
            runtime_args=rt,
            common_runtime_args=[pt.plan.buffer_address(), act.buffer_address(), self.w01.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        L = self.limit
        cp = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k1c_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_X, CB_W, CB_O, CB_META, _fb(L), _fb(-L), _fb(-3.0e38), CB_RM, CB_SC, CB_O2],
            runtime_args=rt,
            common_runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=bool(int(os.environ.get("TP32PF_K1_FP32", "1"))),
                math_approx_mode=True,
            ),
        )
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=cbs)
        prog.custom_program_hash = _hash(
            0x7F23,
            self.nc,
            int(os.environ.get("TP32PF_K1_FP32", "1")),
            self.w01.buffer_address(),
            x.buffer_address(),
            pt.plan.buffer_address(),
            pt.gidx.buffer_address(),
            pt.sc.buffer_address(),
            act.buffer_address(),
        )
        ttnn.generic_op([x, self.w01, pt.plan, pt.gidx, pt.sc, act], prog)

    # ------------------------------------------------------------------ K1b
    def k1(self, x, pt, act):
        assert (
            pt.gidx.shape[-1] == 64
        ), "legacy K1b needs the old 256 B-page gidx plan (hy4_tp32pf_plan before the 2 KB-row tables); use k1c"
        cs, cores = self.cs, self.cores
        CB_X, CB_W, CB_O, CB_META, CB_WS, CB_RM, CB_PART, CB_SC, CB_O2, CB_IX, CB_XS, CB_RM1 = range(12)
        tok = int(x.shape[2])
        tpc = (tok + self.nc - 1) // self.nc
        cbs = [
            tile_cb(cs, CB_X, 48, ttnn.bfloat16),
            tile_cb(cs, CB_W, 2 * 768, ttnn.bfloat4_b, TB4),
            tile_cb(cs, CB_O, 4, ttnn.bfloat16),
            u32_cb(cs, CB_META, 2048),
            u32_cb(cs, CB_WS, 4096),
            tile_cb(cs, CB_RM, 48 if (self.xl1 or self.split) else 96, ttnn.bfloat16),
            tile_cb(cs, CB_PART, 8, ttnn.bfloat16),
            tile_cb(cs, CB_SC, 4, ttnn.bfloat16),
            tile_cb(cs, CB_O2, 8, ttnn.bfloat16),
            u32_cb(cs, CB_IX, 2048),
        ]
        cbs.append(tile_cb(cs, CB_RM1, 48 if self.split else 1, ttnn.bfloat16))
        if self.xl1:
            cbs.append(tile_cb(cs, CB_XS, tpc * 6, ttnn.bfloat16))
        rt = self._core_rt(cores)
        rd = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k1b_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[
                CB_W,
                CB_RM,
                CB_META,
                CB_IX,
                CB_SC,
                self.perf,
                self.xl1,
                CB_XS,
                0,
                self.nc,
                tpc,
                self.split,
            ]
            + _acc(pt.plan)
            + _acc(pt.gidx)
            + _acc(x)
            + _acc(pt.sc),
            runtime_args=rt,
            common_runtime_args=[
                self.w01.buffer_address(),
                x.buffer_address(),
                pt.plan.buffer_address(),
                pt.gidx.buffer_address(),
                pt.sc.buffer_address(),
                tok,
            ]
            + [v for q in self.phys for v in (q.x, q.y)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        wr = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k1b_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_O2, CB_WS, CB_RM1, CB_XS, self.xl1, self.nc, tpc, self.split, 0, CB_W]
            + _acc(pt.plan)
            + _acc(act)
            + _acc(pt.gidx)
            + _acc(x),
            runtime_args=rt,
            common_runtime_args=[
                pt.plan.buffer_address(),
                act.buffer_address(),
                pt.gidx.buffer_address(),
                x.buffer_address(),
                tok,
                self.w01.buffer_address(),
            ]
            + [v for q in self.phys for v in (q.x, q.y)],
            config=ttnn.WriterConfigDescriptor(),
        )
        L = self.limit
        cp = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k1b_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[
                CB_X,
                CB_W,
                CB_O,
                CB_META,
                _fb(L),
                _fb(-L),
                _fb(-3.0e38),
                CB_RM,
                CB_PART,
                CB_SC,
                CB_O2,
                self.scale,
                self.dbg,
                self.var,
                self.perf,
                CB_RM1,
                self.split,
            ],
            runtime_args=rt,
            common_runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=True
            ),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=[rd, wr, cp],
            semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0)],
            cbs=cbs,
        )
        prog.custom_program_hash = _hash(
            0x7F21,
            self.xl1,
            self.split,
            tok,
            self.nc,
            self.scale,
            self.dbg,
            self.var,
            self.perf,
            self.w01.buffer_address(),
            x.buffer_address(),
            pt.plan.buffer_address(),
            pt.gidx.buffer_address(),
            pt.sc.buffer_address(),
            act.buffer_address(),
        )
        ttnn.generic_op([x, self.w01, pt.plan, pt.gidx, pt.sc, act], prog)

    # ------------------------------------------------------------------ K2b (one pass S)
    def k2(self, S, pt, act, out, tt):
        cs, cores = self.cs, self.cores
        nown = tt // 32
        assert nown <= self.nc
        CB_A, CB_W, CB_RM, CB_META, CB_SLOT, CB_SUM, CB_TIL, CB_WS = range(8)
        cbs = [
            tile_cb(cs, CB_A, 8, ttnn.bfloat16),
            tile_cb(cs, CB_W, 2 * 64, ttnn.bfloat4_b, TB4),
            tile_cb(cs, CB_RM, 128, ttnn.bfloat16),
            u32_cb(cs, CB_META, 2048),
            tile_cb(cs, CB_SLOT, 256, ttnn.bfloat16),
            tile_cb(cs, CB_SUM, 32, ttnn.bfloat16),
            tile_cb(cs, CB_TIL, 32, ttnn.bfloat16),
            u32_cb(cs, CB_WS, 8192, 2048),
        ]
        rt = self._core_rt(cores)
        rd = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k2b_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_W, CB_A, CB_META, S] + _acc(pt.plan) + _acc(act),
            runtime_args=rt,
            common_runtime_args=[self.w2.buffer_address(), pt.plan.buffer_address(), act.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        common_w = [pt.plan.buffer_address(), pt.dest.buffer_address(), out.buffer_address()]
        for o in range(nown):
            common_w += [self.phys[o].x, self.phys[o].y]
        wr = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k2b_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_RM, CB_WS, CB_SLOT, CB_TIL, nown, S, 0, self.dbg]
            + _acc(pt.plan)
            + _acc(pt.dest)
            + _acc(out),
            runtime_args=rt,
            common_runtime_args=common_w,
            config=ttnn.WriterConfigDescriptor(),
        )
        cp = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k2b_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_A, CB_W, CB_RM, CB_META, CB_SLOT, CB_SUM, CB_TIL, nown, self.dbg, self.perf],
            runtime_args=rt,
            common_runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=True
            ),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=[rd, wr, cp],
            semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0)],
            cbs=cbs,
        )
        prog.custom_program_hash = _hash(
            0x7F22,
            self.nc,
            self.dbg,
            self.perf,
            S,
            nown,
            self.w2.buffer_address(),
            pt.plan.buffer_address(),
            pt.dest.buffer_address(),
            act.buffer_address(),
            out.buffer_address(),
        )
        ttnn.generic_op([act, self.w2, pt.plan, pt.dest, out], prog)

    def k2f(self, pt, act, out, tt):
        """all 6 column passes in one launch (HY4_TP32PF_K2=f)"""
        cs, cores = self.cs, self.cores
        nown = tt // 32
        assert nown <= self.nc and nown <= 64
        CB_A, CB_W, CB_RM, CB_META, CB_SLOT, CB_SUM, CB_TIL, CB_WS = range(8)
        cbs = [
            tile_cb(cs, CB_A, 8, ttnn.bfloat16),
            tile_cb(cs, CB_W, 2 * 64, ttnn.bfloat4_b, TB4),
            tile_cb(cs, CB_RM, 128, ttnn.bfloat16),
            u32_cb(cs, CB_META, 2048),
            tile_cb(cs, CB_SLOT, 256, ttnn.bfloat16),
            tile_cb(cs, CB_SUM, 32, ttnn.bfloat16),
            tile_cb(cs, CB_TIL, 32, ttnn.bfloat16),
            u32_cb(cs, CB_WS, 8192, 2048),
        ]
        rt = self._core_rt(cores)
        rd = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k2f_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_W, CB_A, CB_META, 6] + _acc(pt.plan) + _acc(act),
            runtime_args=rt,
            common_runtime_args=[self.w2.buffer_address(), pt.plan.buffer_address(), act.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        common_w = [pt.plan.buffer_address(), pt.dest.buffer_address(), out.buffer_address()]
        for q in self.phys:
            common_w += [q.x, q.y]
        wr = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k2f_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_RM, CB_WS, CB_SLOT, CB_TIL, nown, 6, 0, self.nc, 1]
            + _acc(pt.plan)
            + _acc(pt.dest)
            + _acc(out),
            runtime_args=rt,
            common_runtime_args=common_w,
            config=ttnn.WriterConfigDescriptor(),
        )
        cp = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/k2f_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cs,
            compile_time_args=[CB_A, CB_W, CB_RM, CB_META, CB_SLOT, CB_SUM, CB_TIL, nown, 0, 0, 6],
            runtime_args=rt,
            common_runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=True
            ),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=[rd, wr, cp],
            semaphores=[
                ttnn.SemaphoreDescriptor(id=0, core_ranges=cs, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=cs, initial_value=0),
            ],
            cbs=cbs,
        )
        prog.custom_program_hash = _hash(
            0x7F25,
            self.nc,
            nown,
            self.w2.buffer_address(),
            pt.plan.buffer_address(),
            pt.dest.buffer_address(),
            act.buffer_address(),
            out.buffer_address(),
        )
        ttnn.generic_op([act, self.w2, pt.plan, pt.dest, out], prog)

    def __call__(self, x, pt, act, out, tt):
        (self.k1c if self.k1v == "c" else self.k1)(x, pt, act)
        if self.k2v == "f":
            self.k2f(pt, act, out, tt)
        else:
            for S in range(6):
                self.k2(S, pt, act, out, tt)
        return out
