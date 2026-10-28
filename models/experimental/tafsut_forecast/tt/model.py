# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tafsut-FM in TTNN (one Blackhole chip). Decomposition: `reference/dataflow.py`.

Host: bucketing, fallback, instance norm, asinh, patching (fp32), per-shape constants (time-encoding terms of the
patch embedding in fp32, future + REG rows, RoPE tables, key masks); sinh and rescale after.
Device: patch embedding on ``[values | mask]`` (K = 64), 14 pre-norm blocks over ``[context | future | REG | pad]``
with a ``[B, 1, S_pad, S_pad]`` additive key mask (finite fill; ttnn SDPA requires the mask's query dim to equal
Q's, so a ``[B, 1, 1, S]`` broadcast mask is rejected), the read-out cut in the last layer, and the head on the 32
future rows.
"""

from __future__ import annotations

import math
import os

import torch
import ttnn

from models.experimental.tafsut_forecast import ops
from models.experimental.tafsut_forecast.ops.qkv_heads_rope.op import fits_l1
from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tt import program_configs
from models.experimental.tafsut_forecast.tt.common import MASK_FILL, Precision, ckc

D, NH, DH, DFF, NL = df.D, df.NH, df.DH, df.DFF, df.NL
NREAD = df.NFUT  # read-out rows: all 32 future tokens (one tile row)
DRAM = ttnn.DRAM_MEMORY_CONFIG
MM_2D = os.environ.get("TAFSUT_MM_2D", "1") != "0"  # R1-2; 0 = the auto-config (screening)
FUSED_ROPE = os.environ.get("TAFSUT_FUSED_ROPE", "1") != "0"  # R1-1; 0 = the round-0 matmul RoPE (screening)
# screening knobs (D7 (b)), default off: embed + head linears on the auto-config, fp32 cos/sin tables, bf16 RoPE DST
MM_2D_SKIP_HIFI4 = os.environ.get("TAFSUT_MM_2D_SKIP_HIFI4", "0") == "1"
ROPE_FP32_TABLE = os.environ.get("TAFSUT_ROPE_FP32_TABLE", "0") == "1"
ROPE_FP32_DEST = os.environ.get("TAFSUT_ROPE_FP32_DEST", "1") != "0"
# round 1 candidates (screening knobs, default off until kept)
ADD_NORM = os.environ.get("TAFSUT_ADD_NORM", "1") != "0"  # R1-4 + R1-7b: residual add fused into the next norm, gamma folded; 0 = off
SDPA_CONCAT = os.environ.get("TAFSUT_SDPA_CONCAT", "1") != "0"  # R1-5: SDPA writes the heads concatenated; 0 = off
PERF_BF16_DEST = os.environ.get("TAFSUT_PERF_BF16_DEST", "0") == "1"  # R1-7a: performance linears without fp32 DST


def sdpa_chunks(s_pad: int, ctx_len: int | None) -> int:
    """Masked k-chunk from the per-shape table (CONTRACT §7 item 1); never chosen by catching exceptions."""
    if ctx_len in df.PAD_TABLE and df.PAD_TABLE[ctx_len][0] == s_pad:
        chunk = df.PAD_TABLE[ctx_len][1]
    else:
        chunk = max(c for c in (32, 64, 128) if df.chunk_count_ok(s_pad, c))
    assert df.chunk_count_ok(s_pad, chunk), (s_pad, chunk)  # the odd-chunk-count SDPA bug (dataflow.PAD_TABLE)
    return chunk


def rope_matrix() -> torch.Tensor:
    """``x @ R == rotate_half(x)`` for a 64-wide head."""
    h = DH // 2
    r = torch.zeros(DH, DH)
    r[h:, :h] = -torch.eye(h)
    r[:h, h:] = torch.eye(h)
    return r


class TtTafsut:
    def __init__(self, device, sd: dict | None = None, precision: Precision | None = None, n_layers: int = NL):
        self.device = device
        self.p = precision or Precision.accuracy()
        self.n_layers = n_layers
        self.w = df.Weights(sd)
        self._consts: dict = {}
        self._mm = ckc("LoFi" if self.p.lofi else "HiFi2", fp32_acc=not (self.p.lofi and PERF_BF16_DEST))
        self._hifi4 = ckc("HiFi4")
        self._norm = ckc("HiFi4")
        self._sdpa = ckc(self.p.sdpa_fidelity)
        self._qkv = ckc(self.p.qkv_fidelity)
        sd = self.w.sd

        def up(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=DRAM)

        def gamma(g):
            return None if ADD_NORM else up(g.view(1, 1, D // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)

        def fold(g, w):  # R1-7b: rms_norm(x) * g @ W = rms_norm(x) @ (g[:, None] * W), folded in fp32
            return g.float()[:, None] * w.float() if ADD_NORM else w

        e = "input_patch_embedding"
        self.emb_hid = up(sd[f"{e}.hidden_layer.weight"][:, 32:].T)  # [64, 3072]: values | mask columns
        self.emb_out = up(sd[f"{e}.output_layer.weight"].T)
        self.emb_res = up(sd[f"{e}.residual_layer.weight"][:, 32:].T)
        self.layers = []
        for L in self.w.layers[:n_layers]:
            self.layers.append(
                dict(
                    ln1=gamma(L["ln1"]),
                    wqkv=up(fold(L["ln1"], L["wqkv"]), self.p.in_weight_dtype),
                    wo=up(L["wo"]),
                    ln2=gamma(L["ln2"]),
                    wi=up(fold(L["ln2"], L["wi"]), self.p.in_weight_dtype),
                    wo2=up(L["wo2"]),
                )
            )
        self.final_ln = gamma(self.w.final_ln)
        h = "output_patch_embedding"
        self.head_hid = up(fold(self.w.final_ln, sd[f"{h}.hidden_layer.weight"].T))
        self.head_hid_b = up(sd[f"{h}.hidden_layer.bias"][None])
        self.head_out = up(sd[f"{h}.output_layer.weight"].T)
        self.head_res = up(fold(self.w.final_ln, sd[f"{h}.residual_layer.weight"].T))
        self.head_b = up((sd[f"{h}.output_layer.bias"] + sd[f"{h}.residual_layer.bias"])[None], ttnn.float32)
        self.rot = up(rope_matrix())

    # ------------------------------------------------------------------------------------------ host side
    @staticmethod
    def prepare_inputs(x) -> df.Prepared:
        return df.prepare_inputs(x, bucket=True)

    def constants(self, B: int, ctx_len: int) -> dict:
        """Per-shape device constants, keyed on shape and kept alive (skill §4 item 9)."""
        key = (B, ctx_len)
        if key in self._consts:
            return self._consts[key]
        sd, P = self.w.sd, ctx_len // df.PATCH
        S = P + df.NFUT + 1
        s_pad = df.PAD_TABLE.get(ctx_len, (math.ceil(S / 128) * 128, 128))[0]
        e = "input_patch_embedding"
        t = (torch.arange(-ctx_len, 0, dtype=torch.float32) / df.TIME_SCALE).view(P, df.PATCH)
        c_hid = t @ sd[f"{e}.hidden_layer.weight"][:, :32].T + sd[f"{e}.hidden_layer.bias"]
        c_res = t @ sd[f"{e}.residual_layer.weight"][:, :32].T + sd[f"{e}.residual_layer.bias"]
        c_res = c_res + sd[f"{e}.output_layer.bias"]
        tail = torch.zeros(s_pad - P, D)
        tail[: df.NFUT] = self.w.future
        tail[df.NFUT] = self.w.reg
        cos, sin = df.rope_tables(df.layout(P, s_pad))

        def up(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
                                   memory_config=DRAM)

        rope_dt = ttnn.float32 if ROPE_FP32_TABLE else ttnn.bfloat16
        c = dict(
            P=P, S=S, s_pad=s_pad, chunk=sdpa_chunks(s_pad, ctx_len),
            c_hid=up(c_hid[None]), c_res=up(c_res[None]),
            tail=up(tail[None].expand(B, -1, -1)),
            cos=up(cos[None, None], rope_dt), sin=up(sin[None, None], rope_dt),
            fused_rope=FUSED_ROPE and fits_l1(s_pad, DH, rope_dt, rope_dt),
        )
        self._consts[key] = c
        return c

    def host_inputs(self, prep: df.Prepared) -> dict:
        B, P, _ = prep.tokens.shape
        c = self.constants(B, prep.ctx_len)
        kv = torch.zeros(B, c["s_pad"], dtype=torch.bool)
        kv[:, :P] = prep.key_valid
        kv[:, P : c["S"]] = True
        bias = torch.where(kv, 0.0, MASK_FILL)[:, None, None, :]
        return dict(
            vm=prep.tokens[:, :, 32:].to(torch.bfloat16),
            mask=bias.expand(B, 1, c["s_pad"], c["s_pad"]).to(torch.bfloat16),
            mask_read=bias.expand(B, 1, NREAD, c["s_pad"]).to(torch.bfloat16),
        )

    def upload_inputs(self, host: dict, dev: dict | None = None) -> dict:
        """Upload (or copy into ``dev``'s existing buffers, for a trace)."""
        out = {}
        for k, v in host.items():
            t = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if dev is None:
                out[k] = ttnn.to_device(t, self.device, memory_config=DRAM)
            else:
                ttnn.copy_host_to_device_tensor(t, dev[k])
                out[k] = dev[k]
        return out

    @staticmethod
    def deallocate_inputs(dev: dict):
        for v in dev.values():
            ttnn.deallocate(v)

    @staticmethod
    def postprocess(head: torch.Tensor, prep: df.Prepared, horizon: int) -> torch.Tensor:
        return df.postprocess(head[:, : math.ceil(horizon / df.PATCH)].float(), prep, horizon)

    # -------------------------------------------------------------------------------------------- device
    def _linear(self, x, w, **kw):
        kw.setdefault("compute_kernel_config", self._mm)
        if MM_2D and not (MM_2D_SKIP_HIFI4 and kw["compute_kernel_config"] is self._hifi4):  # R1-2: 2D multicast config over the whole grid, relu fused into it (R1-3)
            pc = program_configs.for_linear(x, w, kw["compute_kernel_config"], relu=kw.get("activation") == "relu")
            if pc is not None:
                kw.pop("activation", None)
                kw["program_config"] = pc
        return ttnn.linear(x, w, memory_config=DRAM, **kw)

    def embed(self, vm, c):
        B = vm.shape[0]
        P = c["P"]
        flat = ttnn.reshape(vm, (1, 1, B * P, 64))
        hid = self._linear(flat, self.emb_hid, compute_kernel_config=self._hifi4)
        hid = ttnn.reshape(hid, (B, P, DFF))
        hid = ttnn.relu(ttnn.add(hid, c["c_hid"]))
        out = self._linear(ttnn.reshape(hid, (1, 1, B * P, DFF)), self.emb_out, compute_kernel_config=self._hifi4)
        ttnn.deallocate(hid)
        res = self._linear(flat, self.emb_res, compute_kernel_config=self._hifi4)
        out = ttnn.add(ttnn.reshape(ttnn.add(out, res), (B, P, D)), c["c_res"])
        ttnn.deallocate(res)
        h = ttnn.concat([out, c["tail"]], dim=1, memory_config=DRAM)
        ttnn.deallocate(out)
        return h

    def rope(self, x, cos, sin):
        r = ttnn.matmul(x, self.rot, compute_kernel_config=self._hifi4, memory_config=DRAM)
        y = ttnn.add(ttnn.mul(x, cos), ttnn.mul(r, sin))
        ttnn.deallocate(r)
        return y

    def norm(self, h, g):
        if ADD_NORM:
            return ops.rms_norm(h, epsilon=df.NORM_EPS, memory_config=DRAM)
        return ttnn.rms_norm(h, epsilon=df.NORM_EPS, weight=g, compute_kernel_config=self._norm)

    def add_norm(self, h, d, g):
        """(h + d, rms_norm(h + d)); consumes h and d."""
        if ADD_NORM:  # R1-4: one pass, both outputs
            y, x = ops.add_rms_norm(h, d, epsilon=df.NORM_EPS, memory_config=DRAM)
        else:
            y = ttnn.add(h, d)
            x = ttnn.rms_norm(y, epsilon=df.NORM_EPS, weight=g, compute_kernel_config=self._norm)
        ttnn.deallocate(h)
        ttnn.deallocate(d)
        return y, x

    def layer(self, h, x, L, inp, c, last: bool, g_next):
        """h, its ln1 norm x -> h', its next norm (ln1 of the next layer, or the final norm)."""
        B, S = h.shape[0], h.shape[1]
        qkv = self._linear(ttnn.reshape(x, (1, 1, B * S, D)), L["wqkv"], compute_kernel_config=self._qkv)
        ttnn.deallocate(x)
        if c["fused_rope"]:  # R1-1: head split + RoPE in one program (chronos qkv_heads_rope)
            q, k, v = ops.qkv_heads_rope(ttnn.reshape(qkv, (B, S, 3 * D)), c["cos"], c["sin"], num_heads=NH,
                                         memory_config=DRAM, fp32_dest_acc_en=ROPE_FP32_DEST)
            ttnn.deallocate(qkv)
        else:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn.reshape(qkv, (B, S, 3 * D)), num_heads=NH, transpose_key=False, memory_config=DRAM
            )
            ttnn.deallocate(qkv)
            q, k = self.rope(q, c["cos"], c["sin"]), self.rope(k, c["cos"], c["sin"])
        mask, q_chunk = inp["mask"], c["chunk"]
        if last:  # read-out cut: queries, O and the FFN only on the 32 future rows
            P = c["P"]
            q = ttnn.slice(q, (0, 0, P, 0), (B, NH, P + NREAD, DH))
            mask, q_chunk, S = inp["mask_read"], NREAD, NREAD
            hs = ttnn.slice(h, (0, P, 0), (B, P + NREAD, D))
            ttnn.deallocate(h)
            h = hs
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk, k_chunk_size=c["chunk"], exp_approx_mode=self.p.exp_approx,
        )
        a = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, scale=1.0, program_config=pc,
            compute_kernel_config=self._sdpa, memory_config=DRAM,
            **({"output_concat_heads": True} if SDPA_CONCAT else {}),
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        if not SDPA_CONCAT:
            a = ttnn.transformer.concatenate_heads(a, memory_config=DRAM)
        o = self._linear(ttnn.reshape(a, (1, 1, B * S, D)), L["wo"])
        ttnn.deallocate(a)
        h, x = self.add_norm(h, ttnn.reshape(o, (B, S, D)), L["ln2"])
        f = self._linear(ttnn.reshape(x, (1, 1, B * S, D)), L["wi"], activation="relu")
        ttnn.deallocate(x)
        f2 = self._linear(f, L["wo2"])
        ttnn.deallocate(f)
        return self.add_norm(h, ttnn.reshape(f2, (B, S, D)), g_next)

    def head(self, x):
        """The final norm's output [B, 32, D] -> head."""
        B = x.shape[0]
        x = ttnn.reshape(x, (1, 1, B * NREAD, D))
        hid = self._linear(x, self.head_hid, bias=self.head_hid_b, activation="relu", compute_kernel_config=self._hifi4)
        o = self._linear(hid, self.head_out, compute_kernel_config=self._hifi4, dtype=ttnn.float32)
        r = self._linear(x, self.head_res, compute_kernel_config=self._hifi4, dtype=ttnn.float32)
        out = ttnn.add(ttnn.add(o, r), self.head_b)
        for t in (x, hid, o, r):
            ttnn.deallocate(t)
        return ttnn.reshape(out, (B, NREAD, df.NQ * df.PATCH))

    def forward_device(self, inp: dict, c: dict, n_layers: int | None = None, taps: list | None = None):
        """Device inputs -> device ``[B, 32, 288]`` fp32 head output (asinh space)."""
        n = self.n_layers if n_layers is None else n_layers
        h = self.embed(inp["vm"], c)
        if taps is not None:
            taps.append(ttnn.to_torch(h).float())
        x = self.norm(h, self.layers[0]["ln1"] if n else self.final_ln)
        for i in range(n):
            g_next = self.layers[i + 1]["ln1"] if i + 1 < n else self.final_ln
            h, x = self.layer(h, x, self.layers[i], inp, c, last=(i == n - 1 and n == NL), g_next=g_next)
            if taps is not None:
                taps.append(ttnn.to_torch(h).float())
        ttnn.deallocate(h)
        if n != NL:  # reduced-layer runs (profile / screening): read the future rows without the cut
            P = c["P"]
            xr = ttnn.slice(x, (0, P, 0), (x.shape[0], P + NREAD, D))
            ttnn.deallocate(x)
            x = xr
        out = self.head(x)
        ttnn.deallocate(x)
        return out

    def forward(self, prep: df.Prepared, taps: list | None = None) -> torch.Tensor:
        """One forward, uncached inputs: host ``[B, 32, 288]`` fp32."""
        c = self.constants(prep.tokens.shape[0], prep.ctx_len)
        inp = self.upload_inputs(self.host_inputs(prep))
        out = self.forward_device(inp, c, taps=taps)
        host = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        self.deallocate_inputs(inp)
        return host

    def forecast(self, x, horizon: int, heads: list | None = None) -> torch.Tensor:
        """`tafsut.forecast`: median feedback above 1024 steps, context trimmed to 32768."""
        x = torch.as_tensor(x, dtype=torch.float32)
        out, remaining = [], horizon
        while remaining > 0:
            take = min(remaining, df.NFUT * df.PATCH)
            prep = self.prepare_inputs(x)
            head = self.forward(prep)
            if heads is not None:
                heads.append(head)
            cur = self.postprocess(head, prep, take)
            out.append(cur)
            remaining -= take
            if remaining > 0:
                x = torch.cat([x, cur[:, :, df.MEDIAN_IDX]], -1)[:, -df.MAX_CTX :]
        return torch.cat(out, 1)
