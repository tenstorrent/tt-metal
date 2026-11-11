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
                 hold_users=(), ref=True, attn_n=0, gdnout=None, gin_k=None, gb_cores=None, gb=None):
        """ref=False: no stock-op reference (no priming step, check() skipped) -- the stock op's program does not fit
        the kernel config buffer with device-profiler zones compiled in; the checked step is then FRESH."""
        self.ref = ref
        # MKA_GDNOUT with E1_GO_LOCAL=2: the step's out tiles go straight into gdn_out's A (never to the out tensor), so
        # check() can't compare out (got = 0); ring / window still are. E1_GO_LOCAL=1 keeps the out write for the full check.
        self.out_unwritten = bool(gdnout) and os.environ.get("E1_GO_LOCAL", "2") == "2"
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
        mkz = lambda shp, dt, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(torch.zeros(*shp), dtype=dt,
                                                                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        self.qkv = mkz((1, 32, g.W), b16)
        ctrl0 = m._ctrl_lazy(g, B, [T] * B, 0, [])
        # MKA_GDN_L1 (G2 lever 2): the role's per-layer I/O tensors in L1 (interleaved over every L1 bank) instead of
        # DRAM -- `pre` (the prefix output the role reads), `out` (gdn_out, the wo input) and the ctrl page. A comma list of
        # pre / out / ctrl (default: none = DRAM). The kernels read / write them through TensorAccessor either way.
        l1 = set(x for x in os.environ.get("MKA_GDN_L1", "").split(",") if x)
        assert l1 <= {"pre", "out", "ctrl"}, l1
        if gdnout and os.environ.get("E1_GO_LOCAL", "2") == "2" and "out" in l1:
            # gdn_out local handoff without the out write: `out` is never written, keep it in DRAM (frees its L1 range,
            # which otherwise lowers the GDN-core CB floor below the role + gdn_out CBs at u8)
            l1.discard("out")
            print("[g2] MKA_GDN_L1: 'out' kept in DRAM (gdn_out E1_GO_LOCAL=2 never writes it)", flush=True)
        mc = lambda k: ttnn.L1_MEMORY_CONFIG if k in l1 else ttnn.DRAM_MEMORY_CONFIG
        mkc = lambda: ttnn.from_torch(ctrl0, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                      memory_config=mc("ctrl"))
        self.LY = [dict(cA=m.LazyCase(dev, g, B, T, seed=seed + 7 * l), cE=m.LazyCase(dev, g, B, T, seed=seed + 7 * l),
                        pre=mkz((1, 1, 32, Nv * self.S * 32), f32, mc("pre")), outE=mkz((1, 32, Nv * m.Dv), b16, mc("out")),
                        ctrl=mkc())
                   for l in range(NG)]
        if l1:
            lo = min(ly[k].buffer_address() for ly in self.LY for k, kk in (("pre", "pre"), ("outE", "out"), ("ctrl", "ctrl"))
                     if kk in l1)
            kb = sum((ly[k].buffer().size() if hasattr(ly[k], "buffer") and hasattr(ly[k].buffer(), "size") else 0)
                     for ly in self.LY for k, kk in (("pre", "pre"), ("outE", "out"), ("ctrl", "ctrl")) if kk in l1) // 1024
            print(f"[g2] MKA_GDN_L1={sorted(l1)}: lowest L1 address of the role's L1 tensors {lo} ({kb} KB total over all "
                  f"banks); role CB end must stay below the floor", flush=True)
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
        # G-b (E1_GB): the prefix runs inside the role from G-a's staging tiles; no producers. Harness (gb_cores = the GDN
        # cores' CoreRangeSet): a staging tensor at one L1 address on every GDN core (4 tiles), filled by the stand-in per
        # layer with the qkv row tiles (G-a's tile -> core / slot map). Engine: gb = dict(gin_addr = CB 41, gin_map).
        self.gb = gb
        self.gin_stage = None
        if gb_cores is not None and gb is None:
            self.gin_stage = ttnn.from_torch(torch.zeros(len(self.cores), 2048, dtype=torch.int32), dtype=ttnn.uint32,
                                             layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.MemoryConfig(
                                                 ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                                 ttnn.ShardSpec(gb_cores, [1, 2048], ttnn.ShardOrientation.ROW_MAJOR)))
            self.gb = dict(gin_addr=self.gin_stage.buffer_address(), gin_map=e1_shim.gin_map_engine())
        self.role = e1_shim.build_engine(dev, self.cores, g, B, T, self.Kt, self.Vt, 4, self.tl, m.SCALE, m.L2_EPS,
                                         m.NORM_EPS, gen_dir, self.S, sems, out_target, kdir=kdir,
                                         attn_n=attn_n, gdnout=gdnout, gin_k=gin_k, gb=self.gb)
        if self.gin_stage is not None:
            self.role["io"].append(self.gin_stage)
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

    def _plain(self, c, ctrl, qkv=None):
        m, g = self.m, self.g
        return ttnn.experimental.kda.gdn_spec_step(
            self.qkv if qkv is None else qkv, c.win_a, c.win_b, c.ring, ctrl, c.taps_d, c.dtb_d, c.nea_d, c.w_d, g.Nv, g.Nk, m.Dk, m.Dv, self.T,
            self.B, g.QKVZ, conv_kernel=self.K, scale=m.SCALE, hnew_depth=2, memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=c.out_dtype, fast_mode=True, fp32_state=True, value_blocks=4, kernel_opts=self.opt)

    def standin_gin(self):
        """g2_standin's G-a stand-in args: [n, qkv addr, staging addr, (xy, slot) per tile] (n = 0 without G-b staging)."""
        if self.gin_stage is None:
            return [0, 0, 0]
        fp = [self.dev.worker_core_from_logical_core(c) for c in self.cores]
        tab = []
        for ci, sl in self.gb["gin_map"]:
            tab += [fp[ci].x | (fp[ci].y << 16), sl]
        return [len(self.gb["gin_map"]), self.qkv.buffer_address(), self.gb["gin_addr"]] + tab

    def prepare(self, hold=()):
        """This step's rows / ctrl, then the producers for every GDN layer (before the engine pass). G-b: no producers."""
        self.hold = list(hold)
        self._rows()
        if self.gb is not None:
            cmp_ = os.environ.get("G2_GB_CMP", "0") == "1"
            mi_win = [0 if mm >= self.T else mm for mm in self.mi_ctrl]
            for ly in self.LY:
                assert ly["cA"].par == ly["cE"].par
                self._write_ctrl(ly, self.hold)
                if cmp_:   # debug: the u1 producers' conv / silu / gate tiles (no L2 roots: q / k stay raw) into pre_ref
                    cA, cE = ly["cA"], ly["cE"]
                    if "pre_ref" not in ly:
                        ly["pre_ref"] = ttnn.from_torch(torch.zeros(1, 1, 32, self.Nv * self.S * 32), dtype=f32, layout=ttnn.TILE_LAYOUT,
                                                        device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    pr = ly["pre_ref"]
                    winE, winEn = [cE.win_a, cE.win_b][cE.par], [cE.win_a, cE.win_b][1 - cE.par]
                    ttnn.generic_op([winE, self.qkv, cA.taps_d, winEn, pr], self._build_A(cA, pr, winE, winEn, mi_win, self.hold))
                    ttnn.generic_op([self.qkv, cE.ring, pr], self._build_G(cA, pr, cE.ring, cE.par, self.hold))
            ttnn.synchronize_device(self.dev)
            return
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
        if self.gb is not None and os.environ.get("G2_GB_CMP", "0") == "1":
            S, names = self.S, ["kn"] * Kt + ["qn"] * Kt + ["vc"] * Vt + ["zs"] * Vt + ["gate"] * 16
            for l, ly in enumerate(self.LY):
                a = _host(ly["pre"]).float().reshape(32, self.Nv * S, 32).transpose(0, 1).contiguous().view(torch.int32)
                b = _host(ly["pre_ref"]).float().reshape(32, self.Nv * S, 32).transpose(0, 1).contiguous().view(torch.int32)
                bad = [(h, j, names[j], int((a[h * S + j] != b[h * S + j]).sum()),
                        float((a[h * S + j].view(torch.float32) - b[h * S + j].view(torch.float32)).abs().max()))
                       for h in range(self.Nv) for j in range(S) if not torch.equal(a[h * S + j], b[h * S + j])]
                import collections
                cnt = collections.Counter((b_[2], b_[1]) for b_ in bad)
                print(f"[{tag}] pre vs producers layer {l}: {len(bad)} tiles differ; (slot, j): n heads {sorted(cnt.items())}; "
                      f"max|d| {max([b_[4] for b_ in bad] or [0]):.3e}", flush=True)
        qkvs = [None] * len(self.LY)
        if self.gb is not None and self.gb.get("dbg") is not None:
            # engine G-b: the step's input is G-a's real in-proj (CB 41), not the role's synthetic rows -> the reference reads
            # the bench-only DRAM copy G-a writes of every layer's tiles ([32 rows, NG x 65 tiles], same bits as staging)
            d = _host(self.gb["dbg"]).reshape(32, -1)
            W = self.g.W
            assert d.shape[1] == len(self.LY) * W, (d.shape, len(self.LY), W)
            qkvs = [ttnn.from_torch(d[:, l * W:(l + 1) * W].reshape(1, 32, W).bfloat16(), dtype=b16, layout=ttnn.TILE_LAYOUT,
                                    device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG) for l in range(len(self.LY))]
        oAs = [self._plain(ly["cA"], ly["ctrl"], qkvs[l]) for l, ly in enumerate(self.LY)]
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
            if self.out_unwritten:   # gdn_out E1_GO_LOCAL=2: the out tensor is never written -> out not compared (ring / win are)
                e_out = "NOT CHECKED (gdn_out E1_GO_LOCAL=2; =1 checks)"
            elif os.environ.get("E1_HOLD_SKIP", "0") == "1" and self.hold and B > 1:
                # HOLD_SKIP: HOLD rows' out is not the reference's (no state term) -- live rows bit-identical, HOLD rows finite
                W_ = outB.shape[-1]
                oa, ob = iv(outA).reshape(-1, W_), iv(outB).reshape(-1, W_)
                lv = torch.tensor([(r // T) not in self.hold for r in range(oa.shape[0])])
                fin = bool(torch.isfinite(outB.reshape(-1, W_)[~lv]).all())
                e_out = torch.equal(oa[lv], ob[lv]) and fin
                outA = outA.reshape(-1, W_)[lv]; outB = outB.reshape(-1, W_)[lv]
                if outA.numel() == 0:   # every user HOLD: nothing live to compare
                    outA = outB = torch.zeros(1)
                print(f"[{tag}]   HOLD_SKIP: live rows {int(lv.sum())} compared, HOLD rows finite {fin}", flush=True)
            print(f"[{tag}] GDN layer {l} hold={self.hold}: out {e_out} (max|d| {(outA - outB).abs().max().item():.3e}) | "
                  f"ring {e_ring} | win {e_win}", flush=True)
            if e_out is False:  # where: shape, # differing elements, first indices (row = user*T + t)
                d = (iv(outA) != iv(outB)).reshape(outB.shape)
                nz = d.nonzero()
                print(f"[{tag}]   out diff: shape {tuple(outB.shape)} n {nz.shape[0]} first {nz[:12].tolist()}", flush=True)
                c0 = int(nz[0, -1]) // 32 * 32  # first differing 32-column block: per-row diff count + value pairs
                dd = d.reshape(-1, outB.shape[-1])[:, c0:c0 + 32]
                print(f"[{tag}]   block cols {c0}..{c0 + 31}: diffs per row {dd.sum(1).tolist()}", flush=True)
                oa = outA.reshape(-1, outB.shape[-1])[:, c0:c0 + 32]; ob = outB.reshape(-1, outB.shape[-1])[:, c0:c0 + 32]
                idx = dd.nonzero()[:8].tolist()
                print(f"[{tag}]   pairs (row, col, ref, got): {[(r, c, float(oa[r, c]), float(ob[r, c])) for r, c in idx]}", flush=True)
                n2 = int(((iv(outA) != iv(outB)).reshape(-1, outB.shape[-1]).sum(0) > 0).reshape(-1, 32).any(1).sum())
                print(f"[{tag}]   blocks with diffs: {n2}", flush=True)
            if not e_ring:  # which (ring slot, user) blocks differ, and in how many heads
                da, db = ringbits(rA), ringbits(rE)
                bad = [(sl, u, int((da[sl, u] != db[sl, u]).flatten(1).any(1).sum())) for sl in range(da.shape[0]) for u in range(da.shape[1])
                       if not torch.equal(da[sl, u], db[sl, u])]
                print(f"[{tag}]   ring diff (slot, user, heads): {bad}", flush=True)
                for sl, u, _ in bad[:3]:   # which 32x32 tiles of the (Dk, Dv) block differ (head 0..): page = tile row * Vt + col
                    dd = (da[sl, u] != db[sl, u]).reshape(da.shape[2], da.shape[3] // 32, 32, da.shape[4] // 32, 32).any(4).any(2)
                    print(f"[{tag}]     slot {sl} user {u}: differing pages per head {[[int(i) for i in dd[hh].flatten().nonzero().flatten()] for hh in range(dd.shape[0])][:2]}", flush=True)
            ok_all = ok_all and (e_out is not False) and e_ring and e_win
            ttnn.deallocate(oAs[l])
            cA.par, cE.par = 1 - cA.par, 1 - cE.par
        self.mi_ctrl = [int(x) for x in torch.randint(0, T, (B,), generator=self.gen)]
        return ok_all
