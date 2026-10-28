# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""FlowState r1.1 in TTNN on one Blackhole chip.

Host (fp32): trim, NaN mask, appended [0, 1] steps, causal RevIN (reference/dataflow.front_end); CPM combine,
de-normalization, torch.quantile, weighted mean (dataflow.back_end).
Device: activations as (1, 1, L * Bp, 512) with rows (t, b) and Bp = the batch padded to 32 (route K's unit is a
32 x 32 tile of (b, p)). Embedding as one K = 32 matmul over [x~, flag, 0...]; per layer the B projection as one
N = 1024 matmul writing [Bu_re | Bu_im] fp32, the LTI scan (route K, fp32 state), Re(C x) as one K = 1024 matmul
over [x_re | x_im] with [C_r | -C_i]^T, the gate, + D u, the MLP, LayerNorm, the residual; the last layer cut to
the rows the CPM combine reads (from the first read row to the end); the decoder folded with the Legendre basis.
Precision per role: tt/common.py.
"""

from __future__ import annotations

import torch
import ttnn

from models.experimental.flowstate_forecast.ops.lti_scan import op as lti
from models.experimental.flowstate_forecast.reference import dataflow as dfl
from models.experimental.flowstate_forecast.tt import common

TILE = 32


def _put(device, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device)


class FlowStateTT:
    def __init__(self, model, device, precision="accuracy"):
        """model: the HF FlowStateForPrediction (weights only, fp32)."""
        self.device = device
        self.p = common.policy(device, precision)
        m = model.model
        self.layers_ref = list(m.encoder.layers)
        self.dec_lin_w = m.decoder.lin.weight.detach().clone()
        self.dec_lin_b = m.decoder.lin.bias.detach().clone()
        w = m.embed.embed.weight.detach()  # (512, 2)
        we = torch.zeros(TILE, 512)
        we[:2] = w.T
        self.embed_w = _put(device, we, ttnn.float32)
        self.embed_b = _put(device, m.embed.embed.bias.detach().view(1, 512), ttnn.float32)
        wd = self.p.dense_weight_dtype
        self.layers = []
        for lay in self.layers_ref:
            ssm, mlp = lay.ssm, lay.mlp
            T = lambda x: x.detach().T  # noqa: E731
            self.layers.append(
                {
                    "C": _put(device, torch.cat((ssm.C_tilde_r, -ssm.C_tilde_i), 1).detach().T, self.p.c_weight_dtype),
                    "gate_w": _put(device, T(ssm.output_gate.weight), wd),
                    "gate_b": _put(device, ssm.output_gate.bias.detach().view(1, 512), wd),
                    "D": _put(device, ssm.D.detach().view(1, 512), ttnn.bfloat16),
                    "rms_w": _put(device, mlp.rms_weight.detach().view(1, 512), ttnn.bfloat16),
                    "w1": _put(device, T(mlp.w1.weight), wd),
                    "b1": _put(device, mlp.w1.bias.detach().view(1, 1024), wd),
                    "w2": _put(device, T(mlp.w2.weight), wd),
                    "b2": _put(device, mlp.w2.bias.detach().view(1, 1024), wd),
                    "w3": _put(device, T(mlp.w3.weight), wd),
                    "b3": _put(device, mlp.w3.bias.detach().view(1, 512), wd),
                    "ln_w": _put(device, lay.norm.weight.detach().view(1, 512), ttnn.bfloat16),
                    "ln_b": _put(device, lay.norm.bias.detach().view(1, 512), ttnn.bfloat16),
                }
            )
        self._per_s = {}  # (s, L, P) -> per-scale constants, kept alive (skill §4 item 9)

    # ------------------------------------------------------------------ per-scale constants

    def constants(self, s, L, P):
        key = (float(s), int(L), int(P))
        if key not in self._per_s:
            per_layer = []
            for lay in self.layers_ref:
                _, Bbar, K = dfl.ssm_constants(lay.ssm, s, 2)  # Bbar and Lbar = K[1] do not depend on L
                bcat = torch.cat((Bbar.real, Bbar.imag), 0).T  # (512, 1024): u @ bcat = [Bu_re | Bu_im]
                per_layer.append(
                    {"B": _put(self.device, bcat, self.p.b_weight_dtype), "lbar": lti.lbar_tiles(self.device, K[1])}
                )
            W = dfl.legendre_w(s, P).double()  # (P, 256)
            lw = self.dec_lin_w.double().view(9, 256, 512)
            fold_w = torch.einsum("ph,qhn->qpn", W, lw).reshape(9 * P, 512).T.float()  # (512, 9P)
            fold_b = (self.dec_lin_b.double().view(9, 256) @ W.T).reshape(1, -1).float()
            n_pad = -(-9 * P // TILE) * TILE
            fw = torch.zeros(512, n_pad)
            fw[:, : 9 * P] = fold_w
            fb = torch.zeros(1, n_pad)
            fb[:, : 9 * P] = fold_b
            dec = (_put(self.device, fw, self.p.decoder_weight_dtype), _put(self.device, fb, ttnn.float32))
            self._per_s[key] = (per_layer, dec)
        return self._per_s[key]

    # ------------------------------------------------------------------ device forward

    def layer(self, i, u, consts, L, Bp, r0):
        """u (1, 1, L * Bp, 512) bf16 -> out (1, 1, rows * Bp, 512) bf16; rows = L for i < 5, L - r0 for the last."""
        p, w = self.p, self.layers[i]
        bu = ttnn.linear(u, consts["B"], dtype=ttnn.float32, compute_kernel_config=p.b_proj, memory_config=p.mem)
        x = lti.lti_scan_fused(bu, consts["lbar"], L)
        ttnn.deallocate(bu)
        if r0:
            M = L * Bp
            x_rows = ttnn.slice(x, (0, 0, r0 * Bp, 0), (1, 1, M, 1024))
            ttnn.deallocate(x)
            x = x_rows
            u = ttnn.slice(u, (0, 0, r0 * Bp, 0), (1, 1, M, 512))
        if p.c_input_dtype != ttnn.float32:
            x16 = ttnn.typecast(x, p.c_input_dtype)
            ttnn.deallocate(x)
            x = x16
        y = ttnn.linear(x, w["C"], dtype=ttnn.bfloat16, compute_kernel_config=p.c_proj, memory_config=p.mem)
        ttnn.deallocate(x)
        g = ttnn.linear(y, w["gate_w"], bias=w["gate_b"], compute_kernel_config=p.dense, memory_config=p.mem)
        g = ttnn.sigmoid(g)
        y = ttnn.multiply(g, y)
        y = ttnn.add(y, ttnn.multiply(u, w["D"]))
        h = ttnn.rms_norm(y, weight=w["rms_w"], epsilon=1e-6, compute_kernel_config=p.norm)
        a = ttnn.linear(h, w["w1"], bias=w["b1"], activation="silu", compute_kernel_config=p.dense, memory_config=p.mem)
        b = ttnn.linear(h, w["w2"], bias=w["b2"], compute_kernel_config=p.dense, memory_config=p.mem)
        ttnn.deallocate(h)
        ab = ttnn.multiply(a, b)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        z = ttnn.linear(ab, w["w3"], bias=w["b3"], compute_kernel_config=p.dense, memory_config=p.mem)
        ttnn.deallocate(ab)
        z = ttnn.add(z, y)
        ttnn.deallocate(y)
        n = ttnn.layer_norm(z, weight=w["ln_w"], bias=w["ln_b"], epsilon=1e-5, compute_kernel_config=p.norm)
        ttnn.deallocate(z)
        return ttnn.add(u, n)

    def forward_device(self, tok_dev, s, L, Bp, r0, P, keep=False):
        """tok_dev (1, 1, L * Bp, 32) fp32 -> decoder output (1, 1, rows * Bp, 9P padded) fp32 (and every layer)."""
        per_layer, (dw, db) = self.constants(s, L, P)
        p = self.p
        h = ttnn.linear(tok_dev, self.embed_w, bias=self.embed_b, dtype=ttnn.bfloat16, compute_kernel_config=p.embed)
        kept = [h] if keep else None
        n = len(self.layers)
        for i in range(n):
            h_next = self.layer(i, h, per_layer[i], L, Bp, r0 if i == n - 1 else 0)
            if keep:
                kept.append(h_next)
            else:
                ttnn.deallocate(h)
            h = h_next
        dec = ttnn.linear(h, dw, bias=db, dtype=ttnn.float32, compute_kernel_config=p.decoder)
        return dec, kept

    # ------------------------------------------------------------------ host wrapper

    def prepare(self, x, s, horizon, host_only=False):
        """Host front end: x (B, T, 1) -> tokens (on device, or a host tile tensor for a traced upload), bookkeeping."""
        tok, mean, stdev, bk = dfl.front_end(x, s, horizon)
        B, L = tok.shape[:2]
        Bp = -(-B // TILE) * TILE
        t = torch.zeros(L, Bp, TILE)
        t[:, :B, :2] = tok.transpose(0, 1)
        r0 = dfl.first_read_row(L, bk["min_context"])
        meta = dict(bk, B=B, Bp=Bp, r0=r0, mean=mean, stdev=stdev)
        host = ttnn.from_torch(t.view(1, 1, L * Bp, TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        if host_only:
            return host, meta
        return ttnn.to_device(host, self.device), meta

    def finish(self, dec, meta, horizon):
        """Device decoder output -> (quantile_outputs, prediction_outputs, combined_normalized, decoder_pre_denorm)."""
        B, Bp, P, L, r0 = meta["B"], meta["Bp"], meta["P"], meta["L"], meta["r0"]
        rows = L - r0
        d = ttnn.to_torch(dec).float().reshape(rows, Bp, -1)[:, :B, : 9 * P].reshape(rows, B, 9, P, 1)
        q, pt = dfl.back_end(d, meta["mean"], meta["stdev"], horizon)
        return {
            "quantile_outputs": q,
            "prediction_outputs": pt,
            "combined_normalized": dfl.combine_cpm(d, horizon),
            "decoder_pre_denorm": d,
        }

    def __call__(self, x, s, horizon, keep=False):
        tok_dev, meta = self.prepare(x, s, horizon)
        dec, kept = self.forward_device(tok_dev, s, meta["L"], meta["Bp"], meta["r0"], meta["P"], keep=keep)
        out = self.finish(dec, meta, horizon)
        if keep:
            out["layers"] = [ttnn.to_torch(t).float().reshape(-1, meta["Bp"], 512)[:, : meta["B"]] for t in kept]
        return out
