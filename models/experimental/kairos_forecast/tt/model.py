# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Kairos-50M forward in TTNN on one Blackhole chip: packed embedding, 6 + 6 T5 blocks with instance-wise RoPE,
and the residual head. The routing, patch division and RoPE frequencies stay on the host (tt/host_frontend.py);
generate() drives the flip halves and the rollout with the reference's router-bias sequence.

Layout: per-token activations are flat (1, 1, B * S, 512), with S = S_PAD (96: N <= 64 tokens in rows 0..N-1,
masked pad rows up to 63, [REG] in row 64 with RoPE position N, as the reference has it at index N) in
the encoder and D_PAD (32: 2 tokens, pad) in the decoder, viewed as (B, S, 512) around attention.
RoPE: the q/k projection rows of each head are permuted at load from pair order (2i, 2i+1) to half order
(i, i + 32), so the rotation is q * cos + (q @ R) * sin with R the rotate-half matrix (QK^T is unchanged).
q @ R is not a matmul: its signed-permuted columns are extra projection columns, folded at load (R1-1, fold_stored).
"""

from __future__ import annotations

import torch

from models.experimental.kairos_forecast.tt.common import (
    bmm_config,
    linear_config,
    D_PAD,
    NEG,
    S_PAD,
    TILE,
    Precision,
    upload,
    upload_linear,
    upload_norm,
)

D, H, DK, LAYERS, N_TOK, N_SEG, N_Q, H_SEG = 512, 8, 64, 6, 64, 2, 9, 64
K_EMB = 800  # 3 x 256 masked expert inputs + 3 coefficient columns (the expert biases), padded to tiles
HALF = torch.cat([torch.arange(0, DK, 2), torch.arange(1, DK, 2)])  # pair order -> half order, per head


def _half_rows(w):
    """q/k weight (out, in): permute each head's 64 output rows to half order."""
    return w.view(H, DK, -1)[:, HALF].reshape(H * DK, -1)


def rotate_cols(t):
    """Stored (in, out) half-order q/k weight -> the weight of x @ R (R1-1): per head, col i = -col i + 32,
    col i + 32 = col i. A signed permutation of whole 32-column halves."""
    h = t.reshape(t.shape[0], H, DK)
    return torch.cat([-h[..., DK // 2 :], h[..., : DK // 2]], -1).reshape(t.shape)


def fold_stored(device, w, groups):
    """w: a stored [g0 | g1 | ...] (in, 512 * n) weight -> the stored weight [f0(g0) | f1(g1) | ...], with f = "r"
    rotate_cols, "=" the group as is. Built from the stored (already quantized) values and re-uploaded in the same
    dtype, so the folded columns are bitwise the permuted and negated stored ones (device-side bfp8 conversion is not
    sign-symmetric; values on the bfp8 grid re-convert exactly)."""
    import ttnn

    t = ttnn.to_torch(w).float().reshape(-1, w.shape[-1])
    g = t.split(D, -1)
    out = torch.cat([rotate_cols(g[i]) if f == "r" else g[i] for i, f in groups], -1)
    return upload(device, out, dtype=w.dtype)


def positions(n_tok, pad, reg_row=None):
    """RoPE position of every row: tokens 0..n_tok-1 in rows 0..n_tok-1, [REG] (position n_tok) in reg_row,
    0 for the masked pad rows."""
    pos = torch.zeros(pad)
    pos[:n_tok] = torch.arange(n_tok, dtype=torch.float32)
    if reg_row is not None:
        pos[reg_row] = n_tok
    return pos


def rope_tables(freqs, pos):
    """freqs [B, 32], pos [S] or [B, S] -> cos, sin [B, 1, S, 64] in half order."""
    pos = pos.expand(freqs.shape[0], -1) if pos.dim() == 1 else pos
    ang = pos[:, :, None] * freqs[:, None, :]
    ang = torch.cat([ang, ang], -1)[:, None]
    return torch.cos(ang), torch.sin(ang)


def pack_embedding_weight(w, b):
    """MultiInSizeLinear weight [3, out, 256] and bias [3, out] -> [K_EMB, out] for the packed input."""
    cols = torch.arange(256)
    keep = [((cols % 128) < s).float() for s in (128, 64, 32)]
    wp = torch.cat([w[e] * keep[e] for e in range(3)], -1)  # [out, 768]
    out = torch.zeros(K_EMB, w.shape[1])
    out[:768] = wp.T
    out[768:771] = b
    return out


class TtKairos:
    def __init__(self, device, W, precision: Precision | None = None, layers=(LAYERS, LAYERS)):
        import ttnn

        self.device, self.W, self.p = device, W, precision or Precision.accuracy()
        self.n_enc, self.n_dec = layers
        wd = self.p.weight_dtype
        up = lambda t, dt=None: upload_linear(device, t, dt or wd())  # noqa: E731
        bias = lambda t: upload(device, t.reshape(1, -1))  # noqa: E731
        p = "input_patch_embedding"
        self.emb_h = upload(
            device, pack_embedding_weight(W[f"{p}.hidden_layer.weight"], W[f"{p}.hidden_layer.bias"]), dtype=wd()
        )
        self.emb_r = upload(
            device, pack_embedding_weight(W[f"{p}.residual_layer.weight"], W[f"{p}.residual_layer.bias"]), dtype=wd()
        )
        self.emb_o, self.emb_ob = up(W[f"{p}.output_layer.weight"]), bias(W[f"{p}.output_layer.bias"])
        self.enc = []
        for i in range(self.n_enc):
            a = f"encoder.block.{i}.layer.0.SelfAttention"
            f = f"encoder.block.{i}.layer.1"
            q, k, v = _half_rows(W[f"{a}.q.weight"]), _half_rows(W[f"{a}.k.weight"]), W[f"{a}.v.weight"]
            self.enc.append(
                {
                    "n1": upload_norm(device, W[f"encoder.block.{i}.layer.0.layer_norm.weight"]),
                    "qkv": (qkv := up(torch.cat([q, k, v]))),
                    "qkvr": fold_stored(device, qkv, ((0, "r"), (1, "r"), (2, "="))),
                    "o": up(W[f"{a}.o.weight"], wd(keep_bf16=True)),
                    "n2": upload_norm(device, W[f"{f}.layer_norm.weight"]),
                    "wi": up(W[f"{f}.DenseReluDense.wi.weight"]),
                    "wo": up(W[f"{f}.DenseReluDense.wo.weight"], wd(keep_bf16=True)),
                }
            )
        self.enc_norm = upload_norm(device, W["encoder.final_layer_norm.weight"])
        self.dec = []
        for i in range(self.n_dec):
            s, c, f = (f"decoder.block.{i}.layer.{j}" for j in range(3))
            sa, ca = f"{s}.SelfAttention", f"{c}.EncDecAttention"
            q, k, v = _half_rows(W[f"{sa}.q.weight"]), _half_rows(W[f"{sa}.k.weight"]), W[f"{sa}.v.weight"]
            cq, ck = _half_rows(W[f"{ca}.q.weight"]), _half_rows(W[f"{ca}.k.weight"])
            self.dec.append(
                {
                    "n1": upload_norm(device, W[f"{s}.layer_norm.weight"]),
                    "qkv": (qkv := up(torch.cat([q, k, v]))),
                    "qkvr": fold_stored(device, qkv, ((0, "r"), (1, "r"), (2, "="))),
                    "o": up(W[f"{sa}.o.weight"], wd(keep_bf16=True)),
                    "n2": upload_norm(device, W[f"{c}.layer_norm.weight"]),
                    "q": fold_stored(device, up(cq), ((0, "="), (0, "r"), (0, "="))),
                    "kv": fold_stored(device, up(torch.cat([ck, W[f"{ca}.v.weight"]])), ((0, "="), (1, "="), (0, "r"))),
                    "co": up(W[f"{ca}.o.weight"], wd(keep_bf16=True)),
                    "n3": upload_norm(device, W[f"{f}.layer_norm.weight"]),
                    "wi": up(W[f"{f}.DenseReluDense.wi.weight"]),
                    "wo": up(W[f"{f}.DenseReluDense.wo.weight"], wd(keep_bf16=True)),
                }
            )
        self.dec_norm = upload_norm(device, W["decoder.final_layer_norm.weight"])
        p = "output_patch_embedding"
        hd = ttnn.bfloat16  # the head stays bf16 weights in both precisions, fp32 out
        self.head_h, self.head_hb = up(W[f"{p}.hidden_layer.weight"], hd), bias(W[f"{p}.hidden_layer.bias"])
        self.head_o = up(W[f"{p}.output_layer.weight"], hd)
        self.head_r = up(W[f"{p}.residual_layer.weight"], hd)
        self.head_b = upload(
            device, (W[f"{p}.output_layer.bias"] + W[f"{p}.residual_layer.bias"]).reshape(1, -1), dtype=ttnn.float32
        )
        causal = torch.zeros(D_PAD, D_PAD)
        causal[:, N_SEG:] = NEG
        causal[torch.triu(torch.ones(N_SEG, N_SEG, dtype=torch.bool), 1).nonzero(as_tuple=True)] = NEG
        self._causal = causal  # uploaded per batch size (the fused softmax wants the mask's batch = B)
        self._const = {}

    # ---- host side -------------------------------------------------------------------------------------------
    def module_freqs(self, s_features):
        """RoPE frequencies of the 18 attention modules, [B, 32] each (host fp32, a 3-layer MLP per module)."""
        from models.experimental.kairos_forecast.reference.dataflow import rope_freqs

        enc = [rope_freqs(self.W, f"encoder.block.{i}.layer.0.SelfAttention", s_features) for i in range(self.n_enc)]
        dec = [
            (
                rope_freqs(self.W, f"decoder.block.{i}.layer.0.SelfAttention", s_features),
                rope_freqs(self.W, f"decoder.block.{i}.layer.1.EncDecAttention", s_features),
            )
            for i in range(self.n_dec)
        ]
        return enc, dec

    def prepare(self, front):
        """Host tensors of one forward -> device inputs."""

        inp = self.build_inputs(front, lambda x, dt: upload(self.device, x, dtype=dt))
        inp.update(self._consts(inp["B"]))
        return inp

    def prepare_host(self, front):
        """The same inputs as host ttnn tensors (tile layout, not on device), for refreshing trace inputs."""
        import ttnn

        return self.build_inputs(
            front,
            lambda x, dt: ttnn.from_torch(
                x.to(torch.float32 if dt == ttnn.float32 else torch.bfloat16).contiguous(),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
            ),
        )

    def build_inputs(self, front, leaf):
        """Per-forward inputs; leaf(torch tensor, ttnn dtype) makes each one (device upload or host tensor)."""
        import ttnn
        from models.experimental.kairos_forecast.reference.dataflow import expert_coeffs

        xf, w, idx, am = front["x_final"], front["weights"], front["indices"], front["attn_mask"]
        B, N = am.shape
        assert N <= N_TOK, N
        c = expert_coeffs(w, idx)
        xs = torch.zeros(B, N_TOK, K_EMB)
        for e in range(3):
            xs[:, :N, 256 * e : 256 * (e + 1)] = xf[e].nan_to_num(0.0) * c[..., e : e + 1]
        xs[:, :N, 768:771] = c
        m = torch.zeros(B, S_PAD)
        m[:, :N] = am.float()
        m[:, N_TOK] = 1.0  # [REG]
        pe = positions(N, S_PAD, N_TOK).expand(B, -1).clone()
        pe[:, N_TOK] = front["reg_pos"].float() if "reg_pos" in front else N  # [REG] sits at index N of its forward
        pd = positions(N_SEG, D_PAD)
        key_add = (1.0 - m) * NEG
        enc_f, dec_f = self.module_freqs(front["s_features"])
        t = lambda x, dt=ttnn.bfloat16: leaf(x, dt)  # noqa: E731
        return {
            "B": B,
            "xs": t(xs.reshape(1, 1, B * N_TOK, K_EMB)),
            "enc_mask": t(key_add[:, None, None, :].expand(B, 1, TILE, S_PAD)),  # 0 / NEG: exact in bf16; one tile row
            "cross_mask": t(key_add[:, None, None, :].expand(B, 1, TILE, S_PAD)),  # (the softmax broadcasts it)
            "enc_rope": [tuple(map(t, rope_tables(f, pe))) for f in enc_f],
            "dec_rope": [
                (
                    tuple(map(t, rope_tables(fs, pd))),
                    tuple(map(t, rope_tables(fc, pd))),
                    tuple(map(t, rope_tables(fc, pe))),
                )
                for fs, fc in dec_f
            ],
        }

    def _consts(self, B):
        pass

        if B not in self._const:
            reg = torch.zeros(B, TILE, D)
            reg[:, 0] = self.W["shared.weight"][1]
            dec_in = torch.zeros(1, 1, B * D_PAD, D)
            dec_in.view(B, D_PAD, D)[:, :N_SEG] = self.W["shared.weight"][0]
            causal = self._causal.reshape(1, 1, D_PAD, D_PAD).expand(B, 1, D_PAD, D_PAD)
            self._const[B] = {
                "reg": upload(self.device, reg),
                "dec_in": upload(self.device, dec_in),
                "causal": upload(self.device, causal),
            }
        return self._const[B]

    # ---- device side -----------------------------------------------------------------------------------------
    def _lin(self, x, w, *, bias=None, act=None, fid=None, dtype=None):
        import ttnn

        M = 1
        for d in tuple(x.shape)[:-1]:
            M *= d
        cfg = linear_config(M, x.shape[-1], w.shape[-1], act == "relu", self.p.fp32_dest_acc_en)  # R1-4
        return ttnn.linear(
            x,
            w,
            bias=bias,
            activation=act if cfg is None else None,
            program_config=cfg,
            compute_kernel_config=self.p.mm(fid) if fid is None else fid,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _norm(self, x, w):
        import ttnn

        return ttnn.rms_norm(
            x, epsilon=1e-6, weight=w, compute_kernel_config=self.p.hifi4(), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _rope(self, x, xr, cs):
        """x * cos + (x @ R) * sin; x @ R comes from the folded weight columns (R1-1)."""
        import ttnn

        cos, sin = cs
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(xr, sin))

    def _split(self, x, w, B, S):
        """Normed x and a [a | b | c] weight -> three [B, 8, S, 64] head tensors."""
        import ttnn

        y = ttnn.reshape(self._lin(x, w), (B, S, 3 * D))
        return ttnn.transformer.split_query_key_value_and_split_heads(y, num_heads=H, transpose_key=False)

    def _attend(self, q, k, v, mask, B, Sq):
        """q [B, 8, Sq, 64], k / v [B, 8, Sk, 64] (rotated) -> (1, 1, B * Sq, 512). T5: no score scaling."""
        import ttnn

        Sk, fp32 = k.shape[-2], self.p.fp32_dest_acc_en
        s = ttnn.matmul(
            q,
            k,
            transpose_b=True,
            dtype=ttnn.float32,
            compute_kernel_config=self.p.hifi4(),
            program_config=bmm_config(self.device, Sq, DK, Sk, fp32),
        )  # R1-2b: full grid
        if Sq == Sk == D_PAD:  # decoder self-attention: a per-row causal mask, which the fused softmax cannot take
            s = ttnn.softmax(ttnn.add(s, mask), dim=-1, numeric_stable=True, compute_kernel_config=self.p.hifi4())
        else:  # R1-5a: a key mask (one row, broadcast) fused into the softmax (bf16 mask, fp32 scores)
            s = ttnn.scale_mask_softmax(s, None, mask, numeric_stable=True, compute_kernel_config=self.p.hifi4())
        o = ttnn.matmul(
            ttnn.typecast(s, ttnn.bfloat16),
            v,
            compute_kernel_config=self.p.mm(),
            program_config=bmm_config(self.device, Sq, Sk, DK, fp32),
        )
        o = ttnn.transformer.concatenate_heads(o)  # [B, Sq, 512]
        return ttnn.reshape(o, (1, 1, B * Sq, D))

    def _qkv(self, x, blk, cs, B, S):
        """Normed x -> rotated q, rotated k, v [B, 8, S, 64] (q / k in half order)."""
        import ttnn

        q, k, v = self._split(x, blk["qkv"], B, S)
        qr, kr, vd = self._split(x, blk["qkvr"], B, S)  # [qR | kR | v]: v again, unused
        ttnn.deallocate(vd)
        return self._rope(q, qr, cs), self._rope(k, kr, cs), v

    def _self_attn(self, x, blk, cs, mask, B, S):
        return self._attend(*self._qkv(x, blk, cs, B, S), mask, B, S)

    # Block pieces (module tests call each alone; the residual adds stay in the blocks).
    def self_attention(self, h, blk, cs, mask, B, S):
        """Self-attention sublayer output, before the residual add: o(attn(norm(h)))."""
        return self._lin(self._self_attn(self._norm(h, blk["n1"]), blk, cs, mask, B, S), blk["o"])

    def cross_attention(self, h, enc, blk, cs_q, cs_k, mask, B):
        """Decoder cross-attention sublayer output (q 32 rows x k 96 rows), before the residual add."""
        import ttnn

        q, qr, qd = self._split(self._norm(h, blk["n2"]), blk["q"], B, D_PAD)  # [q | qR | q]: q again, unused
        k, v, kr = self._split(enc, blk["kv"], B, S_PAD)  # R1-3: no reshape + permute head split
        ttnn.deallocate(qd)
        return self._lin(self._attend(self._rope(q, qr, cs_q), self._rope(k, kr, cs_k), v, mask, B, D_PAD), blk["co"])

    def _ff(self, h, blk, norm):
        import ttnn

        return ttnn.add(h, self._lin(self._lin(self._norm(h, blk[norm]), blk["wi"], act="relu"), blk["wo"]))

    def enc_block(self, h, blk, cs, mask, B):
        import ttnn

        h = ttnn.add(h, self.self_attention(h, blk, cs, mask, B, S_PAD))
        return self._ff(h, blk, "n2")

    def dec_block(self, h, enc, blk, rope, cross_mask, B):
        import ttnn

        cs_self, cs_q, cs_k = rope
        h = ttnn.add(h, self.self_attention(h, blk, cs_self, self._consts(B)["causal"], B, D_PAD))
        h = ttnn.add(h, self.cross_attention(h, enc, blk, cs_q, cs_k, cross_mask, B))
        return self._ff(h, blk, "n3")

    def embed(self, xs):
        """Packed patch embedding: (1, 1, B * 64, K_EMB) -> (1, 1, B * 64, 512)."""
        import ttnn

        hid = self._lin(xs, self.emb_h, act="relu", fid=self.p.hifi4())
        return ttnn.add(self._lin(hid, self.emb_o, bias=self.emb_ob), self._lin(xs, self.emb_r, fid=self.p.hifi4()))

    def encode(self, inp, mark=None):
        import ttnn

        mark = mark or (lambda _name: None)
        B = inp["B"]
        mark("embed")
        xs = inp["xs"]
        e = self.embed(xs)
        e = ttnn.concat([ttnn.reshape(e, (B, N_TOK, D)), inp["reg"]], dim=1)  # [B, 96, 512], [REG] at row 64
        h = ttnn.reshape(e, (1, 1, B * S_PAD, D))
        for blk, cs in zip(self.enc, inp["enc_rope"]):
            mark("enc_block")
            h = self.enc_block(h, blk, cs, inp["enc_mask"], B)
        mark("enc_norm")
        return self._norm(h, self.enc_norm)

    def decode(self, enc, inp, mark=None):
        pass

        mark = mark or (lambda _name: None)
        B = inp["B"]
        h = inp["dec_in"]
        for blk, rope in zip(self.dec, inp["dec_rope"]):
            mark("dec_block")
            h = self.dec_block(h, enc, blk, rope, inp["cross_mask"], B)
        mark("dec_norm")
        return self._norm(h, self.dec_norm)

    def head(self, h):
        import ttnn

        k = self.p.hifi4()
        hid = self._lin(h, self.head_h, bias=self.head_hb, act="relu", fid=k)
        out = ttnn.add(
            self._lin(hid, self.head_o, fid=k, dtype=ttnn.float32), self._lin(h, self.head_r, fid=k, dtype=ttnn.float32)
        )
        return ttnn.add(out, self.head_b)

    def forward_device(self, inp, mark=None):
        out = self.decode(self.encode(inp, mark), inp, mark)
        if mark:
            mark("head")
        return self.head(out)

    def forward(self, front):
        """front (host_frontend.front_end) -> head output [B, 2, 576] fp32 on the host."""
        import ttnn

        inp = self.prepare(front)
        out = ttnn.to_torch(self.forward_device(inp)).float()
        return out.view(inp["B"], D_PAD, -1)[:, :N_SEG]
