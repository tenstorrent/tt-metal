# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full Kokoro-82M text->audio pipeline running entirely on a single P150.

Every neural stage runs on-device via ttnn: plbert (OptimizedDecoder),
bert_encoder, DurationEncoder, ProsodyPredictor (duration + F0Ntrain), the
on-device duration->alignment, TextEncoder, and the ISTFTNet decoder + Generator
(SineGen, forward-STFT, snake AdaINResBlocks, conv-transpose upsampling,
conv_post, iSTFT). SineGen's interpolate/cumsum/mod and the STFT magnitude/phase
run on-device too (frac/cumsum/sin + matmul-interp, sqrt/atan2).

Weights are taken from a torch KModel (host load only); all forward compute is
ttnn on the P150. Fidelity is spectral, not sample-exact: end-to-end the produced
waveform matches the torch reference at STFT log-magnitude PCC ~0.98
(``tests/test_device_pipeline.py``). Raw-waveform PCC is ~0.12 and is NOT a valid
metric here — ``sinegen_device`` uses a deterministic harmonic-phase model that
omits the reference SineGen's random initial phase and voiced/unvoiced phase
reset, so the signals are perceptually equivalent but phase-decorrelated.

Two entrypoints:
- ``synthesize()`` — hybrid: acoustic front half on the host torch KModel, ISTFTNet
  back half on device. Simplest; used by the demo.
- ``synthesize_device()`` — fully on device: plbert (TT OptimizedDecoder),
  bert_encoder, DurationEncoder, prosody predictor (LSTM/duration_proj/F0Ntrain), and
  TextEncoder all run in ttnn via ``front_half_device()``, feeding the on-device
  decoder. Only the duration->alignment scatter and embedding lookup stay host indexing
  (no compute). Validated per stage on p150: pred_dur exact, asr/F0/N PCC ~1.0, audio
  log-mag PCC ~0.98 (``tests/test_device_pipeline.py``).
"""

import math

import numpy as np
import torch
import torch.nn.functional as Fn

import ttnn


class KokoroDevicePipeline:
    def __init__(self, kmodel, mesh_device):
        self.km = kmodel
        self.mesh = mesh_device
        self.dt = ttnn.bfloat16
        self.ck = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )
        # STFT bases (n_fft=20, hop=5)
        NF, HOP = 20, 5
        self.NF, self.HOP, self.FB, self.CP = NF, HOP, NF // 2 + 1, 32
        win = torch.hann_window(NF, periodic=True)
        n = np.arange(NF)
        k = np.arange(self.FB)
        ang = 2 * np.pi * np.outer(k, n) / NF
        self.fwd_r = torch.from_numpy((np.cos(ang) * win.numpy()).astype(np.float32))
        self.fwd_i = torch.from_numpy((-np.sin(ang) * win.numpy()).astype(np.float32))
        dbl = np.ones(self.FB)
        dbl[1 : self.FB - 1] = 2
        self.bwd_r = torch.from_numpy(((1 / NF) * dbl[:, None] * np.cos(ang) * win.numpy()).astype(np.float32))
        self.bwd_i = torch.from_numpy((-(1 / NF) * dbl[:, None] * np.sin(ang) * win.numpy()).astype(np.float32))
        self.w2 = torch.from_numpy((win.numpy() ** 2).astype(np.float32))
        # Constant iSTFT conv-transpose weights, built once (per-call rebuilds would be
        # fresh tensors that miss the weight cache and leak L1_SMALL).
        _wr = torch.zeros(self.CP, 1, 1, NF)
        _wr[: self.FB, 0, 0] = self.bwd_r
        _wi = torch.zeros(self.CP, 1, 1, NF)
        _wi[: self.FB, 0, 0] = self.bwd_i
        _w2 = torch.zeros(self.CP, 1, 1, NF)
        _w2[0, 0, 0] = self.w2
        self._istft_wr, self._istft_wi, self._istft_w2 = _wr, _wi, _w2
        # On-device weight cache keyed by (host data_ptr, shape). Fixed conv/matmul
        # weights then upload+prepare once and are reused across synths, so repeated
        # generate()/stream() calls on one device don't grow L1_SMALL.
        self._wcache = {}
        # weight_norm recomputes `.weight` on every access (a fresh tensor with a new
        # data_ptr), which defeats the weight cache and leaks L1_SMALL as ttnn
        # re-prepares the vocoder conv weights each call. Materialize it once so every
        # conv `.weight` is a stable parameter (value unchanged) -> cache hits.
        self._materialize_weight_norm(kmodel)

    @staticmethod
    def _materialize_weight_norm(module):
        import torch.nn.utils as _U

        try:
            from torch.nn.utils.parametrize import remove_parametrizations
        except Exception:
            remove_parametrizations = None
        for m in module.modules():
            if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
                try:
                    _U.remove_weight_norm(m)
                except Exception:
                    pass
            elif remove_parametrizations is not None and "weight" in getattr(m, "parametrizations", {}):
                try:
                    remove_parametrizations(m, "weight", leave_parametrized=True)
                except Exception:
                    pass

    # ---- tensor helpers ----
    def H(self, t, l=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t.contiguous(), dtype=self.dt, layout=l, device=self.mesh)

    def Hw(self, t):
        # Cache by host tensor identity so fixed conv weights upload once and reuse.
        key = (t.data_ptr(), tuple(t.shape))
        cached = self._wcache.get(key)
        if cached is None:
            cached = ttnn.from_torch(t.contiguous(), dtype=self.dt, layout=ttnn.ROW_MAJOR_LAYOUT)
            self._wcache[key] = cached
        return cached

    def mm(self, a, b):
        return ttnn.matmul(a, b, compute_kernel_config=self.ck)

    def cl(self, t):  # torch [1,C,L] -> ttnn [1,1,L,C]
        return self.H(t.transpose(1, 2).reshape(1, 1, t.shape[2], t.shape[1]))

    def to_t(self, x, L, C):
        return ttnn.to_torch(x).float().reshape(1, L, C).transpose(1, 2)

    def conv1d(self, x, Ci, Co, w, b, L, k, pad, stride=1, groups=1, dil=1):
        x = ttnn.to_memory_config(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.Hw(w),
            bias_tensor=(self.Hw(b.reshape(1, 1, 1, Co)) if b is not None else None),
            device=self.mesh,
            in_channels=Ci,
            out_channels=Co,
            batch_size=1,
            input_length=L,
            kernel_size=k,
            stride=stride,
            padding=pad,
            dilation=dil,
            groups=groups,
            compute_config=self.ck,
            dtype=self.dt,
        )
        o = o[0] if isinstance(o, (tuple, list)) else o
        Lo = (L + 2 * pad - dil * (k - 1) - 1) // stride + 1
        return (
            ttnn.to_memory_config(
                ttnn.reshape(ttnn.to_layout(o, ttnn.TILE_LAYOUT), (1, 1, Lo, Co)), ttnn.DRAM_MEMORY_CONFIG
            ),
            Lo,
        )

    def ctrans(self, x, Ci, Co, w, b, L, k, stride, pad, opad, groups=1):
        x = ttnn.to_memory_config(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.Hw(w.reshape(Ci, Co // groups, 1, k)),
            bias_tensor=(self.Hw(b.reshape(1, 1, 1, Co)) if b is not None else None),
            device=self.mesh,
            in_channels=Ci,
            out_channels=Co,
            batch_size=1,
            input_height=1,
            input_width=L,
            kernel_size=(1, k),
            stride=(1, stride),
            padding=(0, pad),
            output_padding=(0, opad),
            dilation=(1, 1),
            groups=groups,
            compute_config=self.ck,
            dtype=self.dt,
        )
        o = o[0] if isinstance(o, (tuple, list)) else o
        Lo = (L - 1) * stride - 2 * pad + (k - 1) + opad + 1
        return (
            ttnn.to_memory_config(
                ttnn.reshape(ttnn.to_layout(o, ttnn.TILE_LAYOUT), (1, 1, Lo, Co)), ttnn.DRAM_MEMORY_CONFIG
            ),
            Lo,
        )

    def lrelu(self, x, ns=0.2):
        return ttnn.leaky_relu(x, ns)

    def bilstm(self, xTC, sd, Hd, T):
        def dirp(pfx, rev):
            XW = ttnn.add(
                self.mm(xTC, self.H(sd[f"weight_ih_l0{pfx}"].t())),
                self.H((sd[f"bias_ih_l0{pfx}"] + sd[f"bias_hh_l0{pfx}"]).reshape(1, 4 * Hd)),
            )
            Whh = self.H(sd[f"weight_hh_l0{pfx}"].t())
            h = self.H(torch.zeros(1, Hd))
            c = self.H(torch.zeros(1, Hd))
            o_ = {}
            for t in range(T - 1, -1, -1) if rev else range(T):
                g = ttnn.add(ttnn.slice(XW, [t, 0], [t + 1, 4 * Hd]), self.mm(h, Whh))
                i = ttnn.sigmoid(ttnn.slice(g, [0, 0], [1, Hd]))
                f = ttnn.sigmoid(ttnn.slice(g, [0, Hd], [1, 2 * Hd]))
                gg = ttnn.tanh(ttnn.slice(g, [0, 2 * Hd], [1, 3 * Hd]))
                oo = ttnn.sigmoid(ttnn.slice(g, [0, 3 * Hd], [1, 4 * Hd]))
                c = ttnn.add(ttnn.mul(f, c), ttnn.mul(i, gg))
                h = ttnn.mul(oo, ttnn.tanh(c))
                o_[t] = h
            return ttnn.concat([o_[t] for t in range(T)], dim=0)

        return ttnn.concat([dirp("", False), dirp("_reverse", True)], dim=1)

    def adain(self, x, C, norm, s_):
        mean = ttnn.mean(x, dim=2, keepdim=True)
        xc = ttnn.subtract(x, mean)
        var = ttnn.mean(ttnn.mul(xc, xc), dim=2, keepdim=True)
        xn = ttnn.mul(xc, ttnn.rsqrt(ttnn.add(var, 1e-5)))
        xn = ttnn.add(
            ttnn.mul(xn, self.H(norm.norm.weight.detach().reshape(1, 1, 1, C))),
            self.H(norm.norm.bias.detach().reshape(1, 1, 1, C)),
        )
        hh = ttnn.to_torch(
            ttnn.add(
                self.mm(self.H(s_), self.H(norm.fc.weight.detach().t())),
                self.H(norm.fc.bias.detach().reshape(1, 2 * C)),
            )
        ).float()
        return ttnn.add(
            ttnn.mul(ttnn.add(self.H(hh[:, :C].reshape(1, 1, 1, C)), 1.0), xn), self.H(hh[:, C:].reshape(1, 1, 1, C))
        )

    def snake(self, x, alpha):
        a = self.H(alpha.detach().reshape(1, 1, 1, -1))
        ar = self.H((1.0 / alpha.detach()).reshape(1, 1, 1, -1))
        sn = ttnn.sin(ttnn.mul(x, a))
        return ttnn.add(x, ttnn.mul(ar, ttnn.mul(sn, sn)))

    def adaresblk1(self, x, blk, C, L, s_):
        for c1, c2, n1, n2, a1, a2 in zip(blk.convs1, blk.convs2, blk.adain1, blk.adain2, blk.alpha1, blk.alpha2):
            xt = self.snake(self.adain(x, C, n1, s_), a1)
            xt, _ = self.conv1d(
                xt, C, C, c1.weight.detach(), c1.bias.detach(), L, c1.kernel_size[0], c1.padding[0], dil=c1.dilation[0]
            )
            xt = self.snake(self.adain(xt, C, n2, s_), a2)
            xt, _ = self.conv1d(
                xt, C, C, c2.weight.detach(), c2.bias.detach(), L, c2.kernel_size[0], c2.padding[0], dil=c2.dilation[0]
            )
            x = ttnn.add(xt, x)
        return x

    def adainresblk1d(self, x, blk, Cin, Cout, L, s_):
        up = blk.upsample_type != "none"
        r = self.lrelu(self.adain(x, Cin, blk.norm1, s_))
        Lr = L
        if up:
            r, Lr = self.ctrans(
                r, Cin, Cin, blk.pool.weight.detach(), blk.pool.bias.detach(), Lr, 3, 2, 1, 1, groups=Cin
            )
        r, Lr = self.conv1d(r, Cin, Cout, blk.conv1.weight.detach(), blk.conv1.bias.detach(), Lr, 3, 1)
        r = self.lrelu(self.adain(r, Cout, blk.norm2, s_))
        r, Lr = self.conv1d(r, Cout, Cout, blk.conv2.weight.detach(), blk.conv2.bias.detach(), Lr, 3, 1)
        sc = x
        if up:
            sc = self.cl(Fn.interpolate(self.to_t(x, L, Cin), scale_factor=2, mode="nearest"))
        Ls = 2 * L if up else L
        if blk.learned_sc:
            sc, Ls = self.conv1d(sc, Cin, Cout, blk.conv1x1.weight.detach(), None, Ls, 1, 0)
        return ttnn.mul(ttnn.add(r, sc), 1.0 / math.sqrt(2)), Lr

    # ---- STFT on device ----
    def padc(self, t, C):
        tt = t.transpose(1, 2).reshape(1, 1, -1, C)
        tt = Fn.pad(tt, (0, self.CP - C))
        return self.H(tt, ttnn.ROW_MAJOR_LAYOUT)

    def stft_fwd(self, wav):  # wav ttnn? here torch [1,Lw] -> device conv -> mag,phase device[1,1,fr,FB-pad]
        NF, HOP, FB = self.NF, self.HOP, self.FB
        wp = Fn.pad(wav, (NF // 2, NF // 2), mode="reflect")
        x = self.H(wp.reshape(1, 1, -1, 1), ttnn.ROW_MAJOR_LAYOUT)
        fr = (wp.shape[-1] - NF) // HOP + 1

        def cvf(wt):
            o = ttnn.conv1d(
                input_tensor=x,
                weight_tensor=self.Hw(wt.reshape(FB, 1, 1, NF)),
                device=self.mesh,
                in_channels=1,
                out_channels=FB,
                batch_size=1,
                input_length=wp.shape[-1],
                kernel_size=NF,
                stride=HOP,
                padding=0,
                dilation=1,
                groups=1,
                compute_config=self.ck,
                dtype=self.dt,
            )
            o = o[0] if isinstance(o, (tuple, list)) else o
            return ttnn.reshape(ttnn.to_layout(o, ttnn.TILE_LAYOUT), (1, 1, fr, FB))

        rr = cvf(self.fwd_r)
        ii = cvf(self.fwd_i)
        mag = ttnn.sqrt(ttnn.add(ttnn.add(ttnn.mul(rr, rr), ttnn.mul(ii, ii)), 1e-9))
        ph = ttnn.atan2(ii, rr)
        return mag, ph, fr  # device [1,1,fr,FB]

    def istft(self, spec, phase, T):  # spec,phase torch [1,FB,T] -> device -> torch wav
        NF, HOP, FB, CP = self.NF, self.HOP, self.FB, self.CP
        Xr = spec * torch.cos(phase)
        Xi = spec * torch.sin(phase)
        xr = self.padc(Xr, FB)
        xi = self.padc(Xi, FB)
        wr = self._istft_wr
        wi = self._istft_wi

        def ct(inp, w):
            o = ttnn.conv_transpose2d(
                input_tensor=inp,
                weight_tensor=self.Hw(w),
                device=self.mesh,
                in_channels=CP,
                out_channels=1,
                batch_size=1,
                input_height=1,
                input_width=T,
                kernel_size=(1, NF),
                stride=(1, HOP),
                padding=(0, 0),
                output_padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                compute_config=self.ck,
                dtype=self.dt,
            )
            o = o[0] if isinstance(o, (tuple, list)) else o
            return ttnn.to_torch(ttnn.to_layout(o, ttnn.TILE_LAYOUT)).float().flatten()

        y = ct(xr, wr) + ct(xi, wi)
        onep = torch.zeros(1, 1, T, CP)
        onep[..., 0] = 1.0
        ws = ct(self.H(onep, ttnn.ROW_MAJOR_LAYOUT), self._istft_w2)
        L = (T - 1) * HOP + NF
        return (y[:L] / (ws[:L] + 1e-6))[NF // 2 : L - NF // 2]

    # ---- linear-interp matrices matching F.interpolate(mode='linear', align_corners=False) ----
    @staticmethod
    def _lin_interp_mat(Lin, Lout):
        sf = Lout / Lin
        M = np.zeros((Lout, Lin), dtype=np.float32)
        for i in range(Lout):
            src = (i + 0.5) / sf - 0.5
            src = min(max(src, 0.0), Lin - 1)
            lo = int(np.floor(src))
            hi = min(lo + 1, Lin - 1)
            fr = src - lo
            M[i, lo] += 1 - fr
            M[i, hi] += fr
        return torch.from_numpy(M)

    def sinegen_device(self, F0_curve, ups=300, harm=8, SR=24000):
        """Fully on-device SineGen+source: F0_curve torch [1,nF] -> merged source torch [1,Lf].
        interpolate(linear) via device matmul; frac/cumsum/sin/tanh on device.
        Runs in fp32: after cumsum the phase reaches ~1e4 rad, which bf16's 8-bit
        mantissa cannot represent for sin()."""
        # Fully on-device (ttnn) in fp32: after cumsum the phase reaches ~1e4 rad,
        # which bf16 cannot represent for sin(); and (fn/SR) modulo must use
        # x-floor(x) (ttnn.frac keeps sign, wrong for the negative F0 curve).
        f32 = ttnn.float32

        def Hf(t):
            return ttnn.from_torch(t.contiguous(), dtype=f32, layout=ttnn.TILE_LAYOUT, device=self.mesh)

        def mmf(a, b):
            return ttnn.matmul(a, b, compute_kernel_config=self.ck, dtype=f32)

        def modulo(x):  # x - floor(x) == x % 1 (correct for negatives)
            return ttnn.subtract(x, ttnn.floor(x))

        nF = F0_curve.shape[-1]
        Lf = nF * ups
        Rnn = torch.zeros(Lf, nF, dtype=torch.float32)
        for i in range(Lf):
            Rnn[i, min(i // ups, nF - 1)] = 1.0
        f0u = mmf(Hf(Rnn), Hf(F0_curve.reshape(nF, 1)))  # nearest upsample [Lf,1]
        fn = mmf(f0u, Hf(torch.arange(1, harm + 2).float().reshape(1, harm + 1)))  # [Lf,9]
        rad = modulo(ttnn.mul(fn, 1.0 / SR))
        rad_d = mmf(Hf(self._lin_interp_mat(Lf, nF)), rad)  # [nF,9]
        phase = ttnn.mul(ttnn.cumsum(rad_d, dim=0), 2 * math.pi)
        phase_u = mmf(Hf(self._lin_interp_mat(nF, Lf)), ttnn.mul(phase, float(ups)))  # [Lf,9]
        pr = ttnn.mul(modulo(ttnn.mul(phase_u, 1.0 / (2 * math.pi))), 2 * math.pi)  # range-reduce for sin
        sines = ttnn.mul(ttnn.mul(ttnn.sin(pr), 0.1), ttnn.gt(f0u, 10.0))  # * uv voicing
        M = self.km.decoder.generator.m_source
        merged = ttnn.tanh(
            ttnn.add(mmf(sines, Hf(M.l_linear.weight.detach().t())), Hf(M.l_linear.bias.detach().reshape(1, 1)))
        )
        return ttnn.to_torch(merged).float().reshape(1, Lf)

    def generator(self, xd, L, F0_curve, s):
        G = self.km.decoder.generator
        har = self.sinegen_device(F0_curve)  # [1,Lf] (device-computed)
        mag, ph, fr = self.stft_fwd(har)  # device [1,1,fr,FB]
        magt = self.to_t(mag, fr, self.FB)
        pht = self.to_t(ph, fr, self.FB)
        harcat = torch.cat([magt, pht], dim=1)  # [1,22,fr]
        xg = xd
        Lg = L
        ch = [256, 128]
        for i in range(2):
            xg = self.lrelu(xg, 0.1)
            nk = G.noise_convs[i]
            xs, _ = self.conv1d(
                self.cl(harcat),
                22,
                ch[i],
                nk.weight.detach(),
                nk.bias.detach(),
                harcat.shape[-1],
                nk.kernel_size[0],
                nk.padding[0],
                stride=nk.stride[0],
            )
            u = G.ups[i]
            xg, Lg = self.ctrans(
                xg,
                u.in_channels,
                u.out_channels,
                u.weight.detach(),
                u.bias.detach(),
                Lg,
                u.kernel_size[0],
                u.stride[0],
                u.padding[0],
                0,
            )
            if i == 1:
                xgt = Fn.pad(self.to_t(xg, Lg, ch[i]), (1, 0), mode="reflect")
                xg = self.cl(xgt)
                Lg += 1
            xst = self.to_t(xs, ttnn.to_torch(xs).shape[2], ch[i])
            if xst.shape[-1] < Lg:
                xst = Fn.pad(xst, (0, Lg - xst.shape[-1]))
            xst = xst[..., :Lg]
            xs2 = self.adaresblk1(self.cl(xst), G.noise_res[i], ch[i], Lg, s)
            xg = ttnn.add(xg, xs2)
            acc = None
            for j in range(3):
                rb = self.adaresblk1(xg, G.resblocks[i * 3 + j], ch[i], Lg, s)
                acc = rb if acc is None else ttnn.add(acc, rb)
            xg = ttnn.mul(acc, 1.0 / 3.0)
        xg = self.lrelu(xg, 0.01)  # torch F.leaky_relu default slope is 0.01, not 0.2
        xg, Lg = self.conv1d(xg, 128, 22, G.conv_post.weight.detach(), G.conv_post.bias.detach(), Lg, 7, 3)
        xgt = self.to_t(xg, Lg, 22)
        spec = torch.exp(xgt[:, :11, :])
        phase = torch.sin(xgt[:, 11:, :])
        return self.istft(spec, phase, Lg)

    def decoder(self, asr, F0_curve, N_curve, s):
        D = self.km.decoder
        F = asr.shape[-1]
        F0c, _ = self.conv1d(
            self.cl(F0_curve.unsqueeze(1)),
            1,
            1,
            D.F0_conv.weight.detach(),
            D.F0_conv.bias.detach(),
            F0_curve.shape[-1],
            3,
            1,
            stride=2,
        )
        Nc, _ = self.conv1d(
            self.cl(N_curve.unsqueeze(1)),
            1,
            1,
            D.N_conv.weight.detach(),
            D.N_conv.bias.detach(),
            N_curve.shape[-1],
            3,
            1,
            stride=2,
        )
        F0c_t = self.to_t(F0c, F, 1)
        Nc_t = self.to_t(Nc, F, 1)
        xd = self.cl(torch.cat([asr, F0c_t, Nc_t], dim=1))
        L = F
        xd, L = self.adainresblk1d(xd, D.encode, 514, 1024, L, s)
        asr_res, _ = self.conv1d(
            self.cl(asr), 512, 64, D.asr_res[0].weight.detach(), D.asr_res[0].bias.detach(), F, 1, 0
        )
        asr_res_t = self.to_t(asr_res, F, 64)
        res = True
        for blk in D.decode:
            cur = torch.cat([self.to_t(xd, L, 1024), asr_res_t, F0c_t, Nc_t], dim=1)
            Cout = 1024 if blk.upsample_type == "none" else 512
            xd, L = self.adainresblk1d(self.cl(cur), blk, 1090, Cout, L, s)
        return self.generator(xd, L, F0_curve, s)

    # ---- front-half helpers (prosody predictor + text encoder) ----
    def _lin(self, x_torch, W, b):
        """Linear on device: x_torch [N,In] -> torch [N,Out]."""
        n = x_torch.shape[0]
        o = self.mm(self.H(x_torch.reshape(n, -1)), self.H(W.detach().t()))
        if b is not None:
            o = ttnn.add(o, self.H(b.detach().reshape(1, -1)))
        return ttnn.to_torch(o).float()

    def _lstm(self, x_torch, lstm):
        """nn.LSTM (bidirectional) on device via bilstm: x_torch [T,In] -> torch [T,2*Hd]."""
        T = x_torch.shape[0]
        Hd = lstm.hidden_size
        sd = {k: v.detach() for k, v in lstm.state_dict().items()}
        return ttnn.to_torch(self.bilstm(self.H(x_torch.reshape(T, -1)), sd, Hd, T)).float()

    def _chan_ln(self, x_torch, C, gamma, beta):
        """LayerNorm over the channel dim on device: x_torch [S,C] -> torch [S,C].

        gamma/beta may be None (AdaLayerNorm normalizes without a learned affine).
        """
        S = x_torch.shape[0]
        x = self.H(x_torch.reshape(1, 1, S, C))
        mean = ttnn.mean(x, dim=3, keepdim=True)
        xc = ttnn.subtract(x, mean)
        var = ttnn.mean(ttnn.mul(xc, xc), dim=3, keepdim=True)
        xn = ttnn.mul(xc, ttnn.rsqrt(ttnn.add(var, 1e-5)))
        if gamma is not None:
            xn = ttnn.add(
                ttnn.mul(xn, self.H(gamma.detach().reshape(1, 1, 1, C))),
                self.H(beta.detach().reshape(1, 1, 1, C)),
            )
        return ttnn.to_torch(xn).float().reshape(S, C)

    def _adaln(self, x_torch, ada, s_):
        """AdaLayerNorm on device: LN over channels + (1+gamma)*x+beta from fc(style)."""
        S, C = x_torch.shape
        xn = self._chan_ln(x_torch, C, None, None)  # [S,C]
        h = ttnn.to_torch(
            ttnn.add(
                self.mm(self.H(s_), self.H(ada.fc.weight.detach().t())),
                self.H(ada.fc.bias.detach().reshape(1, 2 * C)),
            )
        ).float()
        gamma = h[:, :C]
        beta = h[:, C:]
        return (1.0 + gamma) * xn + beta  # broadcast [1,C] over S

    def _duration_encoder(self, d_en_feat, s_dur):
        """DurationEncoder on device. d_en_feat [1,S,512], s_dur [1,128] -> d [S,640]."""
        de = self.km.predictor.text_encoder
        S = d_en_feat.shape[1]
        s_exp = s_dur.expand(S, -1)  # [S,128]
        x = torch.cat([d_en_feat.reshape(S, -1), s_exp], dim=1)  # [S,640]
        for i in range(0, len(de.lstms), 2):
            lstm, ada = de.lstms[i], de.lstms[i + 1]
            x = self._lstm(x, lstm)  # [S,512]
            x = self._adaln(x, ada, s_dur)  # [S,512]
            x = torch.cat([x, s_exp], dim=1)  # [S,640]
        return x  # [S,640]

    @staticmethod
    def _wn_weight(conv):
        """Effective weight of a (possibly weight_norm'd) conv, robust to stale .weight."""
        if hasattr(conv, "weight_g") and hasattr(conv, "weight_v"):
            g, v = conv.weight_g.detach(), conv.weight_v.detach()
            return g * v / (v.norm(dim=(1, 2), keepdim=True) + 1e-12)
        return conv.weight.detach()

    def _text_encoder(self, input_ids):
        """TextEncoder on device: input_ids [1,S] -> t_en torch [1,512,S]."""
        te = self.km.text_encoder
        S = input_ids.shape[1]
        emb = te.embedding.weight.detach()[input_ids[0]]  # [S,512] lookup
        xcs = emb.t().unsqueeze(0).contiguous()  # [1,512,S]
        for c in te.cnn:
            conv, ln = c[0], c[1]
            o, _ = self.conv1d(self.cl(xcs), 512, 512, self._wn_weight(conv), conv.bias.detach(), S, 5, 2)
            x_sc = self.to_t(o, S, 512)[0].t().contiguous()  # [S,512]
            x_sc = self._chan_ln(x_sc, 512, ln.gamma, ln.beta)  # LayerNorm over channels
            xd = self.lrelu(self.H(x_sc.reshape(1, 1, S, 512)), 0.2)  # LeakyReLU(0.2) on device
            x_sc = ttnn.to_torch(xd).float().reshape(S, 512)
            xcs = x_sc.t().unsqueeze(0).contiguous()
        h = self._lstm(xcs[0].t().contiguous(), te.lstm)  # [S,512]
        return h.t().unsqueeze(0).contiguous()  # [1,512,S]

    def _f0ntrain(self, en, s_dur):
        """ProsodyPredictor.F0Ntrain on device. en [640,T] -> (F0 [1,2T], N [1,2T])."""
        P = self.km.predictor
        T = en.shape[-1]
        h = self._lstm(en.t().contiguous(), P.shared)  # [T,512]
        base = h.t().unsqueeze(0).contiguous()  # [1,512,T]

        def branch(blocks, proj):
            L, Cin = T, 512
            x = self.cl(base)
            for blk in blocks:
                Cout = blk.conv1.out_channels
                x, L = self.adainresblk1d(x, blk, Cin, Cout, L, s_dur)
                Cin = Cout
            o, L = self.conv1d(x, Cin, 1, self._wn_weight(proj), proj.bias.detach(), L, 1, 0)
            return self.to_t(o, L, 1).reshape(1, L)

        return branch(P.F0, P.F0_proj), branch(P.N, P.N_proj)

    def _plbert(self, input_ids):
        """TT plbert encoder on device: input_ids [1,S] -> last_hidden_state [1,S,768].

        Built once from the host KModel's own bert weights (no re-download) via the
        validated single-chip OptimizedDecoder. attention_mask=None (single utterance,
        no padding) matches the reference all-ones mask.
        """
        from models.demos.audio.kokoro.tt.optimized_decoder import OptimizedDecoder

        if getattr(self, "_plbert_dec", None) is None:
            sd = {k: v.detach() for k, v in self.km.bert.state_dict().items()}
            self._plbert_dec = OptimizedDecoder.from_state_dict(
                sd, hf_config=self.km.bert.config, mesh_device=self.mesh
            )
        prep = OptimizedDecoder.prepare_inputs(input_ids, self.mesh, attention_mask=None)
        out = self._plbert_dec.prefill_forward(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            batch=prep["batch"],
            seq_len=prep["padded_seq_len"],
        )
        return ttnn.to_torch(out)[:, : prep["seq_len"], :].float()

    def front_half_device(self, input_ids, ref_s, speed: float = 1.0, tt_plbert: bool = True, pred_dur=None):
        """Acoustic front half on device -> (asr [1,512,T], F0 [1,2T], N [1,2T], s_dec [1,128], pred_dur [S]).

        With ``tt_plbert=True`` (default) plbert runs on device too via the TT
        ``OptimizedDecoder`` — the entire compute path is on device. Set
        ``tt_plbert=False`` to run plbert on the host KModel (isolates the rest of the
        front half for apples-to-apples PCC against the reference). Everything else —
        bert_encoder, DurationEncoder, predictor LSTM/duration_proj, F0Ntrain,
        TextEncoder — always runs on device. Only the duration->alignment scatter and
        the embedding lookup are host indexing (no compute).
        """
        km = self.km
        S = input_ids.shape[1]
        s_dur, s_dec = ref_s[:, 128:], ref_s[:, :128]
        if tt_plbert:
            bert_dur = self._plbert(input_ids)  # TT plbert on device -> [1,S,768]
        else:
            text_mask = torch.zeros(1, S, dtype=torch.bool)
            bert_dur = km.bert(input_ids, attention_mask=(~text_mask).int())  # host plbert
        d_en = self._lin(bert_dur.reshape(S, -1), km.bert_encoder.weight, km.bert_encoder.bias).reshape(1, S, 512)
        d = self._duration_encoder(d_en, s_dur)  # [S,640]

        if pred_dur is None:
            x = self._lstm(d, km.predictor.lstm)  # [S,512]
            dp = km.predictor.duration_proj.linear_layer
            dur = self._lin(x, dp.weight, dp.bias)  # [S,50]
            dur = torch.sigmoid(dur).sum(dim=-1) / speed
            pred_dur = torch.round(dur).clamp(min=1).long()  # [S]
        else:
            pred_dur = pred_dur.reshape(-1).long()  # caller-pinned alignment

        Tt = int(pred_dur.sum())
        idx = torch.repeat_interleave(torch.arange(S), pred_dur)
        aln = torch.zeros(S, idx.shape[0])
        aln[idx, torch.arange(idx.shape[0])] = 1.0  # [S,T]

        en = ttnn.to_torch(self.mm(self.H(d.t().contiguous()), self.H(aln))).float()  # [640,T]
        F0, N = self._f0ntrain(en, s_dur)  # [1,2T] each
        t_en = self._text_encoder(input_ids)  # [1,512,S]
        asr = ttnn.to_torch(self.mm(self.H(t_en[0].contiguous()), self.H(aln))).float().reshape(1, 512, Tt)
        return asr, F0, N, s_dec, pred_dur

    def synthesize_device(self, input_ids, ref_s, speed: float = 1.0, pred_dur=None):
        """Fully-on-device (front half + ISTFTNet back half) phonemes->audio.

        pred_dur optionally pins the duration/alignment (else predicted on device).
        """
        asr, F0, N, s_dec, _ = self.front_half_device(input_ids, ref_s, speed, pred_dur=pred_dur)
        return self.decoder(asr, F0, N, s_dec).reshape(-1)

    def synthesize(self, input_ids, ref_s, speed: float = 1.0):
        """End-to-end phonemes->audio for a single utterance.

        The acoustic-feature front half (plbert, ``bert_encoder``, prosody
        predictor, duration->alignment, ``text_encoder``) runs on the host torch
        ``KModel`` exactly as the reference ``forward_with_tokens`` does; the
        ISTFTNet decoder + generator + iSTFT back half runs on device via
        :meth:`decoder`. Returns a 1-D torch waveform at 24 kHz.

        (Moving the front half on-device is a follow-up; the on-device stage
        primitives it needs — ``bilstm``/``adain`` — already live in this class.)
        """
        km = self.km
        dev = input_ids.device
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=dev, dtype=torch.long)
        text_mask = (
            torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(dev)
        bert_dur = km.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = km.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = ref_s[:, 128:]
        d = km.predictor.text_encoder(d_en, s_pred, input_lengths, text_mask)
        x, _ = km.predictor.lstm(d)
        duration = km.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=dev), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=dev)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(dev)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = km.predictor.F0Ntrain(en, s_pred)
        t_en = km.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])  # on device
        return audio.reshape(-1)
