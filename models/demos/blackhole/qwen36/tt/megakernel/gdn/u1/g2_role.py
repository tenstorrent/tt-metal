"""G2: the GDN role of the megakernel engine (spec §17 / §20, v1.33) -- the batched step (GDN_BATCH, type 4 / 11 from
one binary via GDN_RT_BM) looped over the pass's GDN layers on the 24 GDN cores (rows 8-9, head i // 4, value block
i % 4), with its I/O on the engine's device-local semaphores (e1_shim.build_engine).

Bring-up scope (in-proj / prefix / wo are still engine stand-ins): the role owns its per-layer buffers. `pre` (the
prefix output: conv + SiLU + L2 roots + gates + batched coefficients + gate page) is filled by the u1 test producers
(tp / l2r / tg) OUTSIDE the engine pass; the step inside the pass reads it once in_ready says the layer started.
Validation: every GDN layer's out, ring (stash pages' unread columns masked) and next window bit-identical vs the
stock r13 op (kernel_opts 4094, VB 4) on an identical copy of the layer state.

Flow: GdnRole(...) allocates + primes (one FRESH step through the stock op on both copies, so the checked step
replays a real stash), then prepare() writes the step's rows / ctrl and runs the producers; the engine program
(kernels / cbs / io) runs ONCE; check() runs the stock op on the reference copies and compares."""
import importlib.util, os, struct, sys
import torch, ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1_shim  # noqa: E402

bits = lambda f: struct.unpack("<I", struct.pack("<f", f))[0]
PBIT, PGBIT = 1 << 21, 1 << 22
f32, b16 = ttnn.float32, ttnn.bfloat16


def _test_module():
    TM = os.environ["TT_METAL_HOME"]
    spec = importlib.util.spec_from_file_location(
        "tgss", f"{TM}/tests/ttnn/unit_tests/operations/experimental/kda/test_gdn_spec_step.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _host(t):
    if hasattr(ttnn, "get_device_tensors"):
        try:
            ds = ttnn.get_device_tensors(t)
            if len(ds) > 1:
                return ttnn.to_torch(ds[0])
        except Exception:
            pass
    return ttnn.to_torch(t)


class GdnRole:
    def __init__(self, dev, cores, NG, B, sems, out_target, prod_cores, gen_dir, kdir=None, opt=4094, seed=901,
                 hold_users=(), ref=True, attn_n=0):
        """ref=False: no stock-op reference (no priming step, check() skipped) -- the stock op's program does not fit
        the kernel config buffer with device-profiler zones compiled in; the checked step is then FRESH."""
        self.ref = ref
        m = self.m = _test_module()
        self.dev, self.cores, self.NG, self.B, self.T, self.K = dev, list(cores), NG, B, 4, 4
        g = self.g = m.GEOS[0]
        self.Nv, self.Nk = g.Nv, g.Nk
        self.Kt, self.Vt = m.Dk // 32, m.Dv // 32
        self.S = 2 * self.Kt + 2 * self.Vt + 16
        self.Ct, self.KV, self.rf = g.C // 32, self.Kt * self.Vt, self.Nv // self.Nk
        self.opt, self.hold_users = opt, list(hold_users)
        assert len(self.cores) == self.Nv * 4, len(self.cores)
        T, Nv = self.T, self.Nv
        mkz = lambda shp, dt: ttnn.from_torch(torch.zeros(*shp), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.qkv = mkz((1, 32, g.W), b16)
        ctrl0 = m._ctrl_lazy(g, B, [T] * B, 0, [])
        mkc = lambda: ttnn.from_torch(ctrl0, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        self.LY = [dict(cA=m.LazyCase(dev, g, B, T, seed=seed + 7 * l), cE=m.LazyCase(dev, g, B, T, seed=seed + 7 * l),
                        pre=mkz((1, 1, 32, Nv * self.S * 32), f32), outE=mkz((1, 32, Nv * m.Dv), b16), ctrl=mkc())
                   for l in range(NG)]
        # producer placement (separate programs, run before the engine pass): any cores off the GDN rows
        self._jobs()
        nA, nB, nG = len(self.jobs), 4, Nv
        assert len(prod_cores) >= nA + nB + nG, (len(prod_cores), nA + nB + nG)
        self.coresA, self.coresB, self.coresG = (prod_cores[:nA], prod_cores[nA:nA + nB], prod_cores[nA + nB:nA + nB + nG])
        mk = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])
        self.crsA, self.crsB, self.crsG = mk(self.coresA), mk(self.coresB), mk(self.coresG)
        self.gen = torch.Generator().manual_seed(5)
        # prime: one FRESH step through the stock op on both copies (identical states, a real stash to replay)
        self.mi_ctrl = [T] * B
        if ref:
            self._rows()
            for ly in self.LY:
                self._write_ctrl(ly, [])
                for c in (ly["cA"], ly["cE"]):
                    ttnn.deallocate(self._plain(c, ly["ctrl"]))
                    c.par = 1 - c.par
            ttnn.synchronize_device(dev)
            self.mi_ctrl = [int(x) for x in torch.randint(0, T, (B,), generator=self.gen)]
        self.tl = [dict(qkv=self.qkv, win_a=ly["cE"].win_a, win_b=ly["cE"].win_b, ring=ly["cE"].ring, ctrl=ly["ctrl"],
                        taps=ly["cE"].taps_d, dtb=ly["cE"].dtb_d, nea=ly["cE"].nea_d, weight=ly["cE"].w_d,
                        pre=ly["pre"], out=ly["outE"]) for ly in self.LY]
        self.role = e1_shim.build_engine(dev, self.cores, g, B, T, self.Kt, self.Vt, 4, self.tl, m.SCALE, m.L2_EPS,
                                         m.NORM_EPS, gen_dir, self.S, sems, out_target, kdir=kdir,
                                         attn_n=attn_n)
        # the window parity is a per-step input of the producers: the role's window tensors stay fixed (win_a / win_b),
        # the kernels pick by ctrl parity

    # ---------------- producers (verbatim from test_b8_loop.py, B / T generic) ----------------
    def _jobs(self):
        Nk, Kt, Vt, Nv, S, rf, Ct = self.Nk, self.Kt, self.Vt, self.Nv, self.S, self.rf, self.Ct
        jobs = []
        for hk in range(Nk):
            hs = range(hk * rf, hk * rf + rf)
            for j in range(Kt):
                jobs.append((0, hk * Kt + j, [h * S + Kt + j for h in hs]))
                jobs.append((0, Nk * Kt + hk * Kt + j, [h * S + j for h in hs]))
        for h in range(Nv):
            for j in range(Vt):
                jobs.append((0, 2 * Nk * Kt + h * Vt + j, [h * S + 2 * Kt + j]))
                jobs.append((1, Ct + h * Vt + j, [h * S + 2 * Kt + Vt + j]))
        self.jobs = jobs
        self.l2jobs = []
        for hk in range(Nk):
            hs = list(range(hk * rf, hk * rf + rf))
            self.l2jobs.append((hs[0] * S + 0, 1.0, [h * S + 0 for h in hs]))
            self.l2jobs.append((hs[0] * S + Kt, self.m.SCALE, [h * S + Kt for h in hs]))

    @staticmethod
    def _cbd(crs, idx, dt, pages, page):
        return ttnn.CBDescriptor(total_size=pages * page, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])

    @staticmethod
    def _ccfg():
        return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

    @staticmethod
    def _dm(proc, noc):
        return ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)

    def _build_A(self, cA, pre, win, winn, mi_win, hold):
        crsA, T, K, B, Ct = self.crsA, self.T, self.K, self.B, self.Ct
        cbd = self._cbd
        cbs = [cbd(crsA, 0, b16, 2, 2048), cbd(crsA, 3, b16, 1, 2048), cbd(crsA, 4, b16, K, 2048), cbd(crsA, 7, b16, 1, 2048),
               cbd(crsA, 5, b16, 1, 2048), cbd(crsA, 6, b16, 1, 2048), cbd(crsA, 24, f32, 1, 4096), cbd(crsA, 16, f32, 1, 4096)]
        rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        hm = sum(1 << u for u in hold)
        assert not (hold and B == 1), "tp producers: HOLD needs NB > 1 (arg 5 = the single user's mi otherwise)"
        for (kind, col, dst), c in zip(self.jobs, self.coresA):
            rrt[c.x][c.y] = [win.buffer_address(), self.qkv.buffer_address(), cA.taps_d.buffer_address(), kind, col, mi_win[0],
                             winn.buffer_address(), 1, B] + mi_win + [Ct, hm]
            wrt[c.x][c.y] = [pre.buffer_address(), len(dst)] + dst + [0] * (3 - len(dst))
            crt[c.x][c.y] = [kind]
        acc = (ttnn.TensorAccessorArgs(win).get_compile_time_args() + ttnn.TensorAccessorArgs(self.qkv).get_compile_time_args()
               + ttnn.TensorAccessorArgs(cA.taps_d).get_compile_time_args())
        ks = [ttnn.KernelDescriptor(kernel_source=f"{HERE}/tp_reader.cpp", core_ranges=crsA, compile_time_args=[T, K] + acc,
                                    runtime_args=rrt, config=self._dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)),
              ttnn.KernelDescriptor(kernel_source=f"{HERE}/tp_writer.cpp", core_ranges=crsA,
                                    compile_time_args=[16, 1] + ttnn.TensorAccessorArgs(pre).get_compile_time_args(),
                                    runtime_args=wrt, config=self._dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)),
              ttnn.KernelDescriptor(kernel_source=f"{HERE}/tp_compute.cpp", core_ranges=crsA, compile_time_args=[K],
                                    runtime_args=crt, config=self._ccfg())]
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)

    def _build_B(self, pre):
        crsB, N, cbd = self.crsB, self.Kt, self._cbd
        cbs = [cbd(crsB, 0, f32, N, 4096), cbd(crsB, 1, f32, N, 4096), cbd(crsB, 2, f32, 1, 4096), cbd(crsB, 3, f32, 1, 4096),
               cbd(crsB, 4, b16, 1, 2048), cbd(crsB, 5, b16, 1, 2048), cbd(crsB, 6, f32, 1, 4096), cbd(crsB, 7, f32, 1, 4096),
               cbd(crsB, 16, f32, N, 4096)]
        rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (src0, post, dst), c in zip(self.l2jobs, self.coresB):
            rrt[c.x][c.y] = [pre.buffer_address(), src0, bits(self.m.L2_EPS)]
            wrt[c.x][c.y] = [pre.buffer_address(), len(dst)] + dst
            crt[c.x][c.y] = [bits(post)]
        ks = [ttnn.KernelDescriptor(kernel_source=f"{HERE}/l2r_reader.cpp", core_ranges=crsB,
                                    compile_time_args=[N, self.B * self.T] + ttnn.TensorAccessorArgs(pre).get_compile_time_args(),
                                    runtime_args=rrt, config=self._dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)),
              ttnn.KernelDescriptor(kernel_source=f"{HERE}/tp_writer.cpp", core_ranges=crsB,
                                    compile_time_args=[16, N] + ttnn.TensorAccessorArgs(pre).get_compile_time_args(),
                                    runtime_args=wrt, config=self._dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)),
              ttnn.KernelDescriptor(kernel_source=f"{HERE}/l2r_compute.cpp", core_ranges=crsB, compile_time_args=[N],
                                    runtime_args=crt, config=self._ccfg())]
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)

    def _build_G(self, cA, pre, ring, par, hold):
        crsG, cbd, B, T, Nv, KV, Kt, Vt, S = self.crsG, self._cbd, self.B, self.T, self.Nv, self.KV, self.Kt, self.Vt, self.S
        cbs = [cbd(crsG, 0, b16, 1, 2048), cbd(crsG, 16, f32, 2, 4096), cbd(crsG, 24, f32, 16, 4096), cbd(crsG, 25, f32, 1, 4096)]
        rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for h, c in enumerate(self.coresG):
            per = []
            for u in range(B):
                r_rep = 0 if (u in hold or self.mi_ctrl[u] >= T) else self.mi_ctrl[u] + 1
                stash_rd = ((1 + par) * B + u) * Nv + h
                per += [r_rep, stash_rd * KV + Kt + Vt]
            rrt[c.x][c.y] = [self.qkv.buffer_address(), self.g.QKVZ // 32 + (h >> 5)]
            wrt[c.x][c.y] = [ring.buffer_address(), pre.buffer_address(), h, h * S + 2 * Kt + 2 * Vt, B, 1] + per
            crt[c.x][c.y] = [bits(float(cA.dtb[h])), bits(float(cA.nea[h])), 2]
        ks = [ttnn.KernelDescriptor(kernel_source=f"{HERE}/tg_reader.cpp", core_ranges=crsG,
                                    compile_time_args=ttnn.TensorAccessorArgs(self.qkv).get_compile_time_args(),
                                    runtime_args=rrt, config=self._dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)),
              ttnn.KernelDescriptor(kernel_source=f"{HERE}/tg_writer.cpp", core_ranges=crsG,
                                    compile_time_args=[T, Nv] + ttnn.TensorAccessorArgs(ring).get_compile_time_args()
                                    + ttnn.TensorAccessorArgs(pre).get_compile_time_args(),
                                    runtime_args=wrt, config=self._dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)),
              ttnn.KernelDescriptor(kernel_source=f"{HERE}/tg_compute.cpp", core_ranges=crsG, compile_time_args=[],
                                    runtime_args=crt, config=self._ccfg())]
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)

    # ---------------- step I/O ----------------
    def _rows(self):
        rows = self.m.make_rows(self.g, 32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(rows.reshape(1, 32, self.g.W), dtype=b16, layout=ttnn.TILE_LAYOUT),
                                        self.qkv)

    def _write_ctrl(self, ly, hold):
        c = self.m._ctrl_lazy(self.g, self.B, self.mi_ctrl, ly["cE"].par, hold)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(c, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), ly["ctrl"])

    def _plain(self, c, ctrl):
        m, g = self.m, self.g
        return ttnn.experimental.kda.gdn_spec_step(
            self.qkv, c.win_a, c.win_b, c.ring, ctrl, c.taps_d, c.dtb_d, c.nea_d, c.w_d, g.Nv, g.Nk, m.Dk, m.Dv, self.T,
            self.B, g.QKVZ, conv_kernel=self.K, scale=m.SCALE, hnew_depth=2, memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=c.out_dtype, fast_mode=True, fp32_state=True, value_blocks=4, kernel_opts=self.opt)

    def prepare(self, hold=()):
        """This step's rows / ctrl, then the producers for every GDN layer (before the engine pass)."""
        self.hold = list(hold)
        self._rows()
        mi_win = [0 if mm >= self.T else mm for mm in self.mi_ctrl]
        for ly in self.LY:
            cA, cE, pre = ly["cA"], ly["cE"], ly["pre"]
            assert cA.par == cE.par
            self._write_ctrl(ly, self.hold)
            winE, winEn = [cE.win_a, cE.win_b][cE.par], [cE.win_a, cE.win_b][1 - cE.par]
            ttnn.generic_op([winE, self.qkv, cA.taps_d, winEn, pre], self._build_A(cA, pre, winE, winEn, mi_win, self.hold))
            ttnn.generic_op([pre, pre], self._build_B(pre))
            ttnn.generic_op([self.qkv, cE.ring, pre], self._build_G(cA, pre, cE.ring, cE.par, self.hold))
        ttnn.synchronize_device(self.dev)

    def check(self, tag="g2"):
        """After ONE engine pass: the stock op on the reference copies, then out / ring / next window per GDN layer."""
        if not self.ref:
            print(f"[{tag}] check skipped (no stock-op reference in this run: device profiler on)", flush=True)
            return True
        Kt, Vt, T, B, Nv = self.Kt, self.Vt, self.T, self.B, self.Nv
        m = self.m
        iv = lambda t: t.contiguous().view(torch.int32)

        def ringbits(r):
            b = iv(r).clone()
            p = Kt + Vt
            r0, c0 = (p // Vt) * 32, (p % Vt) * 32
            b[1:3, :, :, r0:r0 + 32, c0 + 1:c0 + 32] = 0
            return b
        oAs = [self._plain(ly["cA"], ly["ctrl"]) for ly in self.LY]
        ttnn.synchronize_device(self.dev)
        ok_all = True
        for l, ly in enumerate(self.LY):
            cA, cE = ly["cA"], ly["cE"]
            outB = _host(ly["outE"]).float()
            outA = _host(oAs[l]).float().reshape(outB.shape)
            rA = _host(cA.ring).reshape(T, B, Nv, m.Dk, m.Dv).float()
            rE = _host(cE.ring).reshape(T, B, Nv, m.Dk, m.Dv).float()
            wA = _host([cA.win_a, cA.win_b][1 - cA.par]).float()
            wE = _host([cE.win_a, cE.win_b][1 - cE.par]).float()
            e_out, e_ring, e_win = torch.equal(iv(outA), iv(outB)), torch.equal(ringbits(rA), ringbits(rE)), torch.equal(iv(wA), iv(wE))
            print(f"[{tag}] GDN layer {l} hold={self.hold}: out {e_out} (max|d| {(outA - outB).abs().max().item():.3e}) | "
                  f"ring {e_ring} | win {e_win}", flush=True)
            ok_all = ok_all and e_out and e_ring and e_win
            ttnn.deallocate(oAs[l])
            cA.par, cE.par = 1 - cA.par, 1 - cE.par
        self.mi_ctrl = [int(x) for x in torch.randint(0, T, (B,), generator=self.gen)]
        return ok_all
