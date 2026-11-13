# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The audio tokenizer encoder (mel -> 25 Hz hidden states -> RVQ codes) on one 1x8 row.

    tok = TtAudioTokenizerEncoder.from_checkpoint(row_mesh, model_path, cache_dir)
    tok.warm()                      # every bucket's programs, before the LLM captures its traces
    codes = tok.codes(mel)          # fp32 mel [T, 128] -> int64 [n_codes, 20], as HostAudioBackend.codes(mel)

The reference is ``reference/audio.AudioTokenizerEncoder`` for one item (``tokenize_audio_batch([mel])``: 6000-frame
segments, one padded batch). Each segment runs as its own pass: conv1 (k 3) + GELU, conv2 (k 3, stride 2) + GELU,
24 pre-LN layers (causal; even layers also window 128), the skip connection of layer 2's output, LayerNorm, the
avg-pool conv (k 2, stride 2) + GELU + LayerNorm. The pass returns the hidden states to the host, which runs the
20-stage RVQ in fp32 (the reference's ``quantize``).

Numerics (measured on Blackhole; tests/perf/audio_device_matmul_codes.py): fp32 elementwise ops (add, sub, mul, exp,
rsqrt, division, GELU, ``ttnn.sum``, ``ttnn.max``, typecast with round-to-nearest-even) are fp32-exact to ~1e-7; a
matmul is not: products of bf16 operands are exact and the accumulation across K tiles in the fp32 dest is exact, but
each 32-term tile dot product keeps only 11 bits below its largest product (~3e-4 relative; no math fidelity or packer
setting changes it). ``ttnn.all_reduce`` adds on the FPU too (~6e-4), and an fp32 ``ttnn.all_gather`` on the row
races (rows that change run to run). So:

* every matmul takes its fp32 activation as two bf16 parts (hi = bf16(x), lo = bf16(x - hi)): 2 matmuls per weight
  matmul, and q.k / P.v as hi.hi + hi.lo + lo.hi (the three parts concatenated along K: one matmul each);
* LayerNorm and the softmax row sums use ``ttnn.sum`` in fp32; the softmax is exp(s - max), P.V unnormalized, then
  divided by the row sums; biases, residual adds, RoPE and GELU are fp32 elementwise ops;
* tensor parallelism uses only all-gathers of bf16 tensors (exact data movement): QKV and FC1 are column-parallel (2
  heads, 512 FFN columns per chip); the attention output and the FC1 activation are gathered as their bf16 parts, the
  output projection and FC2 are column-parallel too (128 output columns per chip), and their fp32 outputs are gathered
  as bf16 parts and re-added into the replicated fp32 residual stream. The convolutions are replicated.

Even so, the codes are not the fp32 reference's (tests/test_audio_tokenizer_tp8.py, bhg2): 3.9-5.6 % of the 25 Hz
code frames change on 5 to 300 s of speech (an fp64 encoder changes ~0.01 %), at RVQ decision margins up to ~1e-3
(median frame margin ~9e-4), and the LLM embeddings reach PCC 0.9989-0.9991 against the fp32 path. The outputs are
deterministic. Time on one row (eager): 94 / 122 / 729 / 1587 ms for 5 / 30 / 120 / 300 s of audio.

Device memory per chip: 113 MB of weights (layers sharded, convolutions replicated, LayerNorm parameters and biases
fp32), ~19 MB of RoPE tables for all buckets, and per pass the attention tensors and two [rows, rows] fp32 masks built
for it (~0.35 GB at a full 6000-frame segment, estimated from the tensor sizes).
"""

import os
import time

import torch

import ttnn

from ...reference import audio as ref
from ..common import MeshPlan, as_weight, cache_fill, cache_name, compute_config

# Rows (50 Hz frames) of one segment pass: each is padded to the smallest bucket holding its frames plus one zero row.
SEGMENT_BUCKETS = (256, 512, 768, 1024, 1536, 2048, 3072)
MASK_NEG = -1.0e9


def padded_rows(n, buckets=SEGMENT_BUCKETS):
    for b in buckets:
        if b > n:
            return b
    raise ValueError(f"{n} frames: more than a 6000-frame segment ({buckets[-1]} rows)")


def default_cache_dir(model_path):
    root = os.environ.get("MIMO_AUDIO_CACHE")
    return os.path.join(root, os.path.basename(os.path.normpath(model_path))) if root else None


class Ops:
    """fp32 building blocks on the row (replicated tensors unless said otherwise)."""

    def __init__(self, plan):
        self.plan = plan
        self.mm = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=True)
        self.sfpu = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)

    @staticmethod
    def free(*ts):
        for t in ts:
            t.deallocate(True)

    def split(self, x):
        """fp32 -> (bf16 hi, bf16 lo), x - hi - lo ~ 2^-17 |x|."""
        hi = ttnn.typecast(x, ttnn.bfloat16)
        hi32 = ttnn.typecast(hi, ttnn.float32)
        rest = ttnn.subtract(x, hi32, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lo = ttnn.typecast(rest, ttnn.bfloat16)
        self.free(hi32, rest)
        return hi, lo

    def linear_parts(self, hi, lo, w, b=None):
        """(hi + lo) @ w (+ b) in fp32: two matmuls and exact adds."""
        kw = dict(dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.mm)
        y1 = ttnn.matmul(hi, w, **kw)
        y2 = ttnn.matmul(lo, w, **kw)
        y = ttnn.add(y1, y2, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.free(y1, y2)
        if b is not None:
            yb = ttnn.add(y, b, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.free(y)
            y = yb
        return y

    def linear(self, x, w, b=None):
        hi, lo = self.split(x)
        y = self.linear_parts(hi, lo, w, b)
        self.free(hi, lo)
        return y

    def layer_norm(self, x, w, b, eps=1e-5):
        """nn.LayerNorm over the last dim (1024) in fp32: two-pass mean / variance with ttnn.sum."""
        d = x.shape[-1]
        kw = dict(dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s = ttnn.sum(x, dim=-1, keepdim=True, compute_kernel_config=self.sfpu)
        mean = ttnn.multiply(s, 1.0 / d, **kw)  # 1/1024: exact
        xc = ttnn.subtract(x, mean, **kw)
        self.free(s, mean)
        sq = ttnn.multiply(xc, xc, **kw)
        v = ttnn.sum(sq, dim=-1, keepdim=True, compute_kernel_config=self.sfpu)
        self.free(sq)
        var = ttnn.multiply(v, 1.0 / d, **kw)
        ve = ttnn.add(var, eps, **kw)
        rstd = ttnn.rsqrt(ve, fast_and_approximate_mode=False)
        self.free(v, var, ve)
        xn = ttnn.multiply(xc, rstd, **kw)
        self.free(xc, rstd)
        xw = ttnn.multiply(xn, w, **kw)
        y = ttnn.add(xw, b, **kw)
        self.free(xn, xw)
        return y

    def gelu(self, x):
        y = ttnn.gelu(x, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return y

    FP32_GATHER = os.environ.get("MIMO_AUDIO_TOK_FP32_GATHER", "parts")  # parts | sync | raw (experiments)

    def gather_last(self, x):
        """[.., n] per chip -> [.., 8 n] in chip order on every chip (exact).

        An fp32 tensor is gathered as its two bf16 parts and re-added (exact): an fp32 all-gather on the row raced
        like the fp32 all-reduce of tt/vision/vit.py (rows that changed run to run)."""
        gather = lambda t: ttnn.all_gather(
            t, dim=3, cluster_axis=self.plan.tp_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if x.dtype != ttnn.float32 or self.FP32_GATHER == "raw":
            return gather(x)
        if self.FP32_GATHER == "sync":
            out = gather(x)
            ttnn.synchronize_device(self.plan.mesh_device)
            return out
        hi, lo = self.split(x)
        ghi, glo = gather(hi), gather(lo)
        self.free(hi, lo)
        a, b = ttnn.typecast(ghi, ttnn.float32), ttnn.typecast(glo, ttnn.float32)
        out = ttnn.add(a, b, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.free(ghi, glo, a, b)
        return out

    def rows(self, t, index, width):
        """Rows ``index`` (device uint32 [1, n]) of a bf16 [R, width] tensor -> bf16 [n, width] tiled."""
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.embedding(
            index,
            ttnn.reshape(rm, (rm.shape[-2], width)),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.free(rm)
        return out

    def im2col(self, x, index, taps):
        """fp32 [R, d] -> (bf16 hi, bf16 lo) [n, taps * d]: rows index[taps*t .. taps*t+taps) side by side."""
        d = x.shape[-1]
        parts = []
        for p in self.split(x):
            g = self.rows(p, index, d)  # [taps * n, d] row-major
            n = g.shape[-2] // taps
            cols = ttnn.reshape(g, (1, 1, n, taps * d))
            parts.append(ttnn.to_layout(cols, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            self.free(g, p)
        return parts


class TtAudioTokenizerLayer:
    def __init__(self, plan, ops, cfg, i, sd, cache_dir):
        self.plan, self.ops, self.cfg, self.i = plan, ops, cfg, i
        tp = plan.tp
        heads, hd = cfg.encoder_heads, cfg.d_model // cfg.encoder_heads
        self.local_heads = heads // tp
        per = self.local_heads * hd  # 128
        self.window = cfg.window_for_layer(i)
        name = lambda n: cache_name(cache_dir, f"audio_tok_tp{tp}_l{i}", n)
        w = {}
        if sd is not None:
            p = f"encoder.layers.{i}."
            g = lambda k: sd[p + k].float()
            scale = hd**-0.5  # 1/8: exact
            q_w, q_b = g("self_attn.q_proj.weight") * scale, g("self_attn.q_proj.bias") * scale
            k_w, k_b = g("self_attn.k_proj.weight"), torch.zeros(cfg.d_model)
            v_w, v_b = g("self_attn.v_proj.weight"), g("self_attn.v_proj.bias")

            def rot(t):  # rotate_half on the output features, per head: [-x2, x1]
                t = t.reshape(heads, hd, *t.shape[1:])
                return torch.cat([-t[:, hd // 2 :], t[:, : hd // 2]], dim=1).reshape(heads * hd, *t.shape[2:])

            cols_w, cols_b = [], []
            for c in range(tp):
                s = slice(c * per, (c + 1) * per)
                cols_w += [q_w[s], k_w[s], rot(q_w)[s], rot(k_w)[s], v_w[s]]
                cols_b += [q_b[s], k_b[s], rot(q_b)[s], rot(k_b)[s], v_b[s]]
            w["wqkv"] = torch.cat(cols_w).T.contiguous()  # [d, tp * 5 * per]
            w["bqkv"] = torch.cat(cols_b).reshape(1, 1, 1, -1)
            w["wo"] = g("self_attn.out_proj.weight").T.contiguous()  # [d_in, d_out]: column shards
            w["bo"] = g("self_attn.out_proj.bias").reshape(1, 1, 1, -1)
            w["w1"] = g("fc1.weight").T.contiguous()
            w["b1"] = g("fc1.bias").reshape(1, 1, 1, -1)
            w["w2"] = g("fc2.weight").T.contiguous()  # [ffn, d]: column shards over d
            w["b2"] = g("fc2.bias").reshape(1, 1, 1, -1)
            for k in ("self_attn_layer_norm", "final_layer_norm"):
                w[k + ".w"] = g(k + ".weight").reshape(1, 1, 1, -1)
                w[k + ".b"] = g(k + ".bias").reshape(1, 1, 1, -1)

        def load(key, mapper, dtype):
            return as_weight(w.get(key), plan, mapper, dtype, name(key))

        bf16, f32 = ttnn.bfloat16, ttnn.float32
        self.wqkv, self.bqkv = load("wqkv", plan.shard(-1), bf16), load("bqkv", plan.shard(-1), f32)
        self.wo, self.bo = load("wo", plan.shard(-1), bf16), load("bo", plan.shard(-1), f32)
        self.w1, self.b1 = load("w1", plan.shard(-1), bf16), load("b1", plan.shard(-1), f32)
        self.w2, self.b2 = load("w2", plan.shard(-1), bf16), load("b2", plan.shard(-1), f32)
        rep = plan.replicate()
        self.ln1 = (load("self_attn_layer_norm.w", rep, f32), load("self_attn_layer_norm.b", rep, f32))
        self.ln2 = (load("final_layer_norm.w", rep, f32), load("final_layer_norm.b", rep, f32))
        self.per = per

    def __call__(self, x, inp):
        ops = self.ops
        kw = dict(dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = ops.layer_norm(x, *self.ln1)
        attn = self.attention(h, inp)
        ops.free(h)
        x2 = ttnn.add(x, attn, **kw)
        ops.free(attn)
        h = ops.layer_norm(x2, *self.ln2)
        f1 = ops.linear(h, self.w1, self.b1)  # [R, ffn / tp]
        ops.free(h)
        g = ops.gelu(f1)
        ops.free(f1)
        hi, lo = ops.split(g)
        ops.free(g)
        hi_all, lo_all = ops.gather_last(hi), ops.gather_last(lo)  # [R, ffn]
        ops.free(hi, lo)
        f2 = ops.linear_parts(hi_all, lo_all, self.w2, self.b2)  # [R, d / tp]
        ops.free(hi_all, lo_all)
        f2_all = ops.gather_last(f2)
        ops.free(f2)
        out = ttnn.add(x2, f2_all, **kw)
        ops.free(x2, f2_all)
        return out

    def attention(self, h, inp):
        ops, per = self.ops, self.per
        kw = dict(dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ops.linear(h, self.wqkv, self.bqkv)  # [R, 5 per]: q k qR kR v
        rows = qkv.shape[-2]
        qk = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, rows, 2 * per))
        qk_r = ttnn.slice(qkv, (0, 0, 0, 2 * per), (1, 1, rows, 4 * per))
        v = ttnn.slice(qkv, (0, 0, 0, 4 * per), (1, 1, rows, 5 * per))
        ops.free(qkv)
        a = ttnn.multiply(qk, inp.cos, **kw)
        b = ttnn.multiply(qk_r, inp.sin, **kw)
        rope = ttnn.add(a, b, **kw)
        ops.free(qk, qk_r, a, b)
        qkv_r = ttnn.concat([rope, v], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ops.free(rope, v)
        hi, lo = ops.split(qkv_r)
        ops.free(qkv_r)
        heads = self.local_heads
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            hi, num_heads=heads, num_kv_heads=heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ql, kl, vl = ttnn.experimental.nlp_create_qkv_heads(
            lo, num_heads=heads, num_kv_heads=heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ops.free(hi, lo)
        dram = dict(memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q3 = ttnn.concat([qh, qh, ql], dim=-1, **dram)  # [1, H, R, 3 hd]: hi.hi + hi.lo + lo.hi
        k3 = ttnn.concat([kh, kl, kh], dim=-1, **dram)
        v3 = ttnn.concat([vh, vl, vh], dim=-2, **dram)  # [1, H, 3R, hd]
        ops.free(qh, ql, kh, kl, vh, vl)
        s = ttnn.matmul(q3, k3, transpose_b=True, compute_kernel_config=ops.mm, **kw)  # [1, H, R, R]
        ops.free(q3, k3)
        sm = ttnn.add(s, inp.mask[self.window > 0], **kw)
        ops.free(s)
        mx = ttnn.max(sm, dim=-1, keepdim=True)
        z = ttnn.subtract(sm, mx, **kw)
        ops.free(sm, mx)
        e = ttnn.exp(z, fast_and_approximate_mode=False)
        ops.free(z)
        ssum = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=ops.sfpu)  # fp32-exact (a matmul is not)
        e_hi, e_lo = ops.split(e)
        ops.free(e)
        e3 = ttnn.concat([e_hi, e_hi, e_lo], dim=-1, **dram)  # [1, H, R, 3R]
        ops.free(e_hi, e_lo)
        o_un = ttnn.matmul(e3, v3, compute_kernel_config=ops.mm, **kw)  # [1, H, R, hd]: P.V unnormalized
        ops.free(e3, v3)
        o = ttnn.divide(o_un, ssum, **kw)
        ops.free(o_un, ssum)
        o_hi, o_lo = ops.split(o)
        ops.free(o)
        cat_hi = ttnn.experimental.nlp_concat_heads(o_hi, **dram)
        cat_lo = ttnn.experimental.nlp_concat_heads(o_lo, **dram)
        ops.free(o_hi, o_lo)
        hi_all, lo_all = ops.gather_last(cat_hi), ops.gather_last(cat_lo)  # [R, d]
        ops.free(cat_hi, cat_lo)
        out = ops.linear_parts(hi_all, lo_all, self.wo, self.bo)  # [R, d / tp]
        ops.free(hi_all, lo_all)
        out_all = ops.gather_last(out)
        ops.free(out)
        return out_all


class PassInputs:
    """Per-bucket RoPE tables (q heads then k heads; kept: 6 MB per chip at 3072 rows), and per-pass attention masks
    (causal, and causal within the window; built on the device for each pass and freed after it: 75 MB at 3072
    rows)."""

    def __init__(self, tok, rows):
        cfg, mesh, rep = tok.cfg, tok.plan.mesh_device, tok.plan.replicate()
        heads = cfg.encoder_heads // tok.plan.tp

        def dev(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        cos, sin = ref._rope_cos_sin(tok.inv_freq, torch.arange(rows), torch.float32)  # [rows, hd], as the reference
        self.cos = dev(cos.repeat(1, 2 * heads).reshape(1, 1, rows, -1))
        self.sin = dev(sin.repeat(1, 2 * heads).reshape(1, 1, rows, -1))
        self.rows, self.window, self.mesh = rows, cfg.window, mesh
        self.mask = None

    def build_masks(self):
        """{windowed: additive mask [1, 1, rows, rows]}: MASK_NEG above the diagonal, and below the window."""
        self.free_masks()  # a pass that failed before freeing its masks: not two sets of them in DRAM
        neg = ttnn.full(
            (1, 1, self.rows, self.rows),
            MASK_NEG,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        causal = ttnn.triu(neg, diagonal=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        far = ttnn.tril(neg, diagonal=-(self.window + 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        windowed = ttnn.add(causal, far, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        Ops.free(neg, far)
        self.mask = {False: causal, True: windowed}

    def free_masks(self):
        if self.mask is not None:
            Ops.free(*self.mask.values())
            self.mask = None

    def deallocate(self):
        self.free_masks()
        Ops.free(self.cos, self.sin)


class HostRVQ:
    """The reference's 20-stage residual VQ in fp32 on the host (``AudioTokenizerEncoder.quantize``, same code), on
    blocks of ``rows`` rows: the ops are per row and give every row the codes of one call over all rows
    (tests/test_audio_host_backend_host.py: 300 s of speech in blocks of 1000-4096 rows at 1 and 8 threads), while the
    blocks' [rows, 1024] temporaries stay in cache: 41K frames (a 27 min video's track) took 0.87 s instead of 3.27 s
    at 8 threads on bhg2, with the same codes."""

    reference_quantize = ref.AudioTokenizerEncoder.quantize
    rows = 4096

    def quantize(self, h):
        if h.shape[0] <= self.rows:
            return self.reference_quantize(h)
        blocks = [self.reference_quantize(h[i : i + self.rows]) for i in range(0, h.shape[0], self.rows)]
        return torch.cat(blocks, dim=1)

    def __init__(self, weights, cfg):
        self.rvq_dtype = torch.float32
        n = len(cfg.codebook_sizes)
        self.codebooks = [
            weights[f"encoder.quantizer.vq.layers.{i}._codebook.embed"].float().contiguous() for i in range(n)
        ]


class TtAudioTokenizerEncoder:
    """``encoder.*`` of ``audio_tokenizer`` on one 1x8 mesh. ``state_dict``: ``reference.audio.load_audio_weights``
    output (``encoder.*`` keys), or None to load from a warm ``cache_dir``. The RVQ codebooks stay on the host."""

    def __init__(self, mesh_device, cfg, state_dict=None, cache_dir=None, buckets=SEGMENT_BUCKETS):
        assert tuple(mesh_device.shape)[0] == 1, f"one row (1 x TP), got {tuple(mesh_device.shape)}"
        self.plan = MeshPlan(mesh_device)
        self.ops = Ops(self.plan)
        self.cfg, self.buckets = cfg, tuple(sorted(buckets))
        tp = self.plan.tp
        assert cfg.encoder_heads % tp == 0 and cfg.encoder_ffn % (32 * tp) == 0 and cfg.d_model % (32 * tp) == 0
        name = lambda n: cache_name(cache_dir, f"audio_tok_tp{tp}", n)
        sd = state_dict
        w = {}
        if sd is not None:
            g = lambda k: sd["encoder." + k].float()
            c1, c2, dn = g("conv1.weight"), g("conv2.weight"), g("down_sample_layer.0.weight")  # [out, in, k]
            w["conv1"] = c1.permute(2, 1, 0).reshape(-1, c1.shape[0]).contiguous()  # [k * in, out]
            w["conv2"] = c2.permute(2, 1, 0).reshape(-1, c2.shape[0]).contiguous()
            w["down"] = dn.permute(2, 1, 0).reshape(-1, dn.shape[0]).contiguous()
            w["conv1.b"] = g("conv1.bias").reshape(1, 1, 1, -1)
            w["conv2.b"] = g("conv2.bias").reshape(1, 1, 1, -1)
            for k in ("layer_norm", "down_sample_norm"):
                w[k + ".w"] = g(k + ".weight").reshape(1, 1, 1, -1)
                w[k + ".b"] = g(k + ".bias").reshape(1, 1, 1, -1)
        rep = self.plan.replicate()
        load = lambda key, dtype: as_weight(w.get(key), self.plan, rep, dtype, name(key))
        self.conv1, self.conv2, self.down = (load(k, ttnn.bfloat16) for k in ("conv1", "conv2", "down"))
        self.conv1_b, self.conv2_b = load("conv1.b", ttnn.float32), load("conv2.b", ttnn.float32)
        self.ln = (load("layer_norm.w", ttnn.float32), load("layer_norm.b", ttnn.float32))
        self.down_ln = (load("down_sample_norm.w", ttnn.float32), load("down_sample_norm.b", ttnn.float32))
        self.layers = [
            TtAudioTokenizerLayer(self.plan, self.ops, cfg, i, sd, cache_dir) for i in range(cfg.encoder_layers)
        ]
        hd = cfg.d_model // cfg.encoder_heads
        self.inv_freq = 1.0 / (cfg.encoder_rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
        self._inputs = {}

    @classmethod
    def from_checkpoint(cls, mesh_device, model_path, cache_dir=None, **kwargs):
        cfg = ref.MiMoAudioConfig.from_model_path(model_path)
        tp = tuple(mesh_device.shape)[1]
        weights = ref.load_audio_weights(model_path, patch=False)  # the codebooks are needed on the host anyway
        with cache_fill(cache_dir, f"audio_tok_tp{tp}") as need:
            enc = cls(mesh_device, cfg, weights if need else None, cache_dir, **kwargs)
        enc.quantizer = HostRVQ(weights, cfg)
        return enc

    def _pass_inputs(self, rows):
        if rows not in self._inputs:
            self._inputs[rows] = PassInputs(self, rows)
        return self._inputs[rows]

    # -- one segment ----------------------------------------------------------------------------------------------

    def _dev(self, t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.plan.mesh_device,
            mesh_mapper=self.plan.replicate(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def segment_plan(self, seg_len, max_len):
        """Host layout of one segment of ``seg_len`` mel frames in a batch padded to ``max_len`` (the reference's
        padded batch: a segment shorter than ``max_len`` sees conv1 outputs of zero frames past its end and repeats its
        last row into the avg-pool; the longest sees zero padding)."""
        cfg = self.cfg
        n50 = int(ref.encoder_output_length(seg_len, cfg))  # conv2 rows
        tgt = int(ref.encoder_output_length(max_len, cfg))
        n25 = int(ref.pooled_length(n50, cfg))
        rows = padded_rows(n50, self.buckets)
        c1_rows = min(max_len, seg_len + 1)  # conv1 positions the conv2 rows read (zero mel past the segment)
        zero50 = 2 * rows - 1  # a conv1 row forced to zero (2 * rows conv1 rows; c1_rows < 2 * rows)
        idx2 = []
        for t in range(rows):
            for j in (2 * t - 1, 2 * t, 2 * t + 1):
                idx2.append(j if 0 <= j < c1_rows else zero50)
        zero25 = rows - 1  # an encoder row forced to zero (n50 < rows)
        idxp = []
        for t in range(rows // 2):
            for j in (2 * t, 2 * t + 1):
                if j < n50:
                    idxp.append(j)
                elif j == n50 and n50 % 2 and n50 < tgt:
                    idxp.append(n50 - 1)  # repeated last row (the batch's unpacking index)
                else:
                    idxp.append(zero25)
        return dict(n50=n50, n25=n25, rows=rows, c1_rows=c1_rows, idx2=idx2, idxp=idxp)

    def segment_hidden(self, mel, max_len, trace=None):
        """fp32 mel [L, 128] of one segment -> host fp32 [pooled_length, d]: the reference's ``features`` rows.
        ``trace``: a list that collects the 50 Hz stream after the convolutions and after each layer (host fp32,
        ``n50`` rows), for tests."""
        sp = self.segment_plan(mel.shape[0], max_len)
        inp = self._pass_inputs(sp["rows"])
        inp.build_masks()
        try:
            return self._segment_hidden(mel, sp, inp, trace)
        finally:
            # Also when an op of the pass fails (device out of memory next to the LLM's traces): the two
            # [rows, rows] fp32 masks (75 MB at 3072 rows) belong to ``inp``, which outlives the pass, so nothing
            # else would free them before this row count's next item. The pass's own temporaries are locals and go
            # with its frame.
            inp.free_masks()

    def _segment_hidden(self, mel, sp, inp, trace):
        """``segment_hidden`` with the segment's plan and its pass inputs (masks built)."""
        ops, cfg = self.ops, self.cfg
        L = mel.shape[0]
        rows = sp["rows"]
        kw = dict(dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # conv1 (k 3, padding 1) on 2 * rows mel frames: im2col on the host, split into bf16 parts there.
        m = torch.zeros(2 * rows + 2, mel.shape[1])
        m[1 : 1 + L] = mel.float()  # zero frames past the segment (the padded batch's zeros)
        cols = torch.cat([m[0:-2], m[1:-1], m[2:]], dim=1).reshape(1, 1, 2 * rows, -1)  # [2 rows, 3 * 128]
        hi = cols.to(torch.bfloat16)
        lo = (cols - hi.float()).to(torch.bfloat16)
        hi_d, lo_d = self._dev(hi, ttnn.bfloat16), self._dev(lo, ttnn.bfloat16)
        c1 = ops.linear_parts(hi_d, lo_d, self.conv1, self.conv1_b)
        ops.free(hi_d, lo_d)
        g1 = ops.gelu(c1)
        ops.free(c1)
        keep = torch.zeros(1, 1, 2 * rows, 1)
        keep[..., : sp["c1_rows"], :] = 1.0
        keep_d = self._dev(keep, ttnn.float32)
        g1m = ttnn.multiply(g1, keep_d, **kw)  # conv1 rows past c1_rows -> 0 (the zero row of the gathers)
        ops.free(g1, keep_d)
        # conv2 (k 3, stride 2, padding 1): im2col by a row gather on the device
        idx2 = self._dev(torch.tensor(sp["idx2"], dtype=torch.int32).reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        p_hi, p_lo = ops.im2col(g1m, idx2, 3)
        ops.free(g1m, idx2)
        c2 = ops.linear_parts(p_hi, p_lo, self.conv2, self.conv2_b)
        ops.free(p_hi, p_lo)
        x = ops.gelu(c2)
        ops.free(c2)
        rows_of = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(rows, -1)[: sp["n50"]]
        if trace is not None:
            trace.append(rows_of(x))
        skip = None
        for i, layer in enumerate(self.layers):
            x_new = layer(x, inp)
            if x is not skip:
                ops.free(x)
            if i == cfg.skip_layer_id - 1:
                skip = x_new  # the reference's skip connection: layer 2's output, added after the last layer
            x = x_new
            if trace is not None:
                trace.append(rows_of(x))
        inp.free_masks()
        hs = ttnn.add(x, skip, **kw)
        ops.free(x, skip)
        h = ops.layer_norm(hs, *self.ln)
        ops.free(hs)
        keep = torch.zeros(1, 1, rows, 1)
        keep[..., : sp["n50"], :] = 1.0
        keep_d = self._dev(keep, ttnn.float32)
        hm = ttnn.multiply(h, keep_d, **kw)
        ops.free(h, keep_d)
        idxp = self._dev(torch.tensor(sp["idxp"], dtype=torch.int32).reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        q_hi, q_lo = ops.im2col(hm, idxp, 2)
        ops.free(hm, idxp)
        pooled = ops.linear_parts(q_hi, q_lo, self.down)
        ops.free(q_hi, q_lo)
        pg = ops.gelu(pooled)
        ops.free(pooled)
        out = ops.layer_norm(pg, *self.down_ln)
        ops.free(pg)
        host = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(rows // 2, -1)
        ops.free(out)
        return host[: sp["n25"]]

    # -- items ----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def hidden(self, mel):
        """fp32 mel [T, 128] -> fp32 [n_codes, d] for one item (its 6000-frame segments as one padded batch)."""
        segs = ref.segment_lengths(mel.shape[0], self.cfg.segment_size)
        max_len, outs, start = max(segs), [], 0
        for L in segs:
            outs.append(self.segment_hidden(mel[start : start + L], max_len))
            start += L
        return torch.cat(outs)

    @torch.no_grad()
    def codes(self, mel):
        """fp32 mel [T, 128] -> int64 [n_codes, 20]: the RVQ of ``hidden`` in fp32 on the host."""
        return self.quantizer.quantize(self.hidden(mel)).t().contiguous()

    def warm(self, buckets=None):
        """Compile every bucket's programs (one silent segment each). Returns {rows: seconds}."""
        times = {}
        for rows in buckets if buckets is not None else self.buckets:
            n50 = rows - 1
            L = 2 * n50 - 1  # mel frames whose conv2 output has n50 rows
            t0 = time.time()
            self.segment_hidden(torch.zeros(L, self.cfg.n_mels), L)
            times[rows] = time.time() - t0
        return times
