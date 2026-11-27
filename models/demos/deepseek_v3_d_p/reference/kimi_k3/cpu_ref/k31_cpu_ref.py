# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 full-model CPU reference forward over the 10K vLLM trace prompt (study tool, CPU only).

usage: python k31_cpu_ref.py --tracks fp32,tf --out /data/.../cpu_ref [--rows 8192] [--last-layer 92]

One pass streams the 93 layers' bf16 weights from the dequantized checkpoint (one ~60 GB file per
layer, read with parallel preads and prefetched one layer ahead) and runs every requested track on
each layer before moving on, so the weights are read once per pass:

  fp32  free-running forward from the embeddings, fp32 math (C1, the reference).
  tf    teacher-forced per layer from the trace's own bf16 residual state (C3, the vLLM yardstick):
        attention sees AttnRes read #1 over the trace state, the FFN sees read #2 over the trace state
        plus the trace's attention output; both are scored against the trace's outputs.
  bf16  free-running forward with bf16 weights and activations, bf16 GEMMs with fp32 accumulate (C2).

Math follows the HF model (modeling_kimi_linear.py; vendored pieces in reference/kimi_k3): AttnRes
reads (attn_res/hf_walk.py schedule), KDA as reference/kda (q/k L2 norm, causal conv K=4 + SiLU, gate
-5*sigmoid(exp(A_log)*(f_b+dt_bias)), delta rule, sigmoid-gated o_norm), MLA NoPE with the DSA
indexer (top-2048 of sum_h w_h relu(q_h.k)), latent MoE (sigmoid router + bias, renormalised top-16,
SiTU experts in the 3584 latent, routed_expert_norm, up-projection, 2 shared experts), layer-0 dense
MLP, output AttnRes read + norm + lm_head.

Precision of the fp32/tf tracks: every GEMM fp32 (MKL sgemm, fp32 accumulate) on fp32 upcasts of the
bf16 checkpoint weights; norms, gates, activations fp32. fp64: AttnRes reads (norm, scores, softmax,
mix), the KDA gate and recurrence (state and outputs), the MLA attention softmax (scores come from an
fp32 GEMM; probabilities cast to fp32 for the PV GEMM), the final log-sum-exp. Not fp64: everything
else (listed in RESULTS.md).

Outputs per track under <out>/<track>/: entry_LXX.pt (live stream entering layer XX, fp32; the AttnRes
plane sealed at layer 12b equals entry_L(12b)), attn_LXX.pt / ffn_LXX.pt (module outputs), route_LXX.pt
(top-16 ids int16 + weights), dsa_LXX.pt (indexer top-2048 int16, -1 = pad), done_LXX markers,
metrics.jsonl (per-layer dB vs the trace), final.pt + logits.pt at the end. Resumable per layer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

CKPT = "/mnt/weka/model-weights/llm/moonshotai/Kimi-K3.1-Pro-dequantized"
TRACE = "/mnt/weka/model-weights/llm/ref-data/debug_traces/kimi_k31_pro_10k_v2/vllm-d06690a4-10240tok"

# text_config of Kimi-K3.1-Pro (checked against config.json at start-up)
H = 7168
N_LAYERS = 93
BLOCK = 12
EPS = 1e-5
KDA_HEADS, KDA_D, CONV_K, GATE_LB, L2_EPS = 96, 128, 4, -5.0, 1e-6
MLA_HEADS, NOPE, ROPE, KV_LORA, V_DIM = 96, 128, 64, 512, 128
IDX_HEADS, IDX_D, TOPK = 32, 128, 2048
N_EXP, TOP_E, LATENT = 896, 16, 3584
SITU_BETA, SITU_LBETA = 4.0, 25.0
RENORM_EPS = 1e-20
VOCAB = 163840

_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I64": torch.int64, "I32": torch.int32}

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ───────────────────────────── checkpoint and trace I/O ─────────────────────────────


class Checkpoint:
    """Safetensors reader: one layer's tensors read with parallel preads into one buffer (views)."""

    def __init__(self, root=CKPT, threads=16, chunk=64 << 20):
        self.root = Path(root)
        self.wm = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.threads, self.chunk = threads, chunk
        self._hdr = {}
        self.cfg = json.loads((self.root / "config.json").read_text())["text_config"]

    def header(self, fname):
        if fname not in self._hdr:
            with open(self.root / fname, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._hdr[fname] = (json.loads(fh.read(n)), 8 + n)
        return self._hdr[fname]

    def load(self, keys):
        by_file = {}
        for k in keys:
            by_file.setdefault(self.wm[k], []).append(k)
        out = {}
        for fname, ks in by_file.items():
            hdr, base = self.header(fname)
            lo = min(hdr[k]["data_offsets"][0] for k in ks)
            hi = max(hdr[k]["data_offsets"][1] for k in ks)
            buf = torch.empty(hi - lo, dtype=torch.uint8)
            arr = buf.numpy()
            fd = os.open(self.root / fname, os.O_RDONLY)
            try:

                def rd(a):
                    b = min(hi - lo, a + self.chunk)
                    mv = memoryview(arr[a:b])
                    off = 0
                    while off < b - a:
                        n = os.preadv(fd, [mv[off:]], base + lo + a + off)
                        if n <= 0:
                            raise IOError(f"short read {fname} @ {base + lo + a + off}")
                        off += n

                with ThreadPoolExecutor(self.threads) as ex:
                    list(ex.map(rd, range(0, hi - lo, self.chunk)))
            finally:
                os.close(fd)
            for k in ks:
                m = hdr[k]
                a, b = m["data_offsets"][0] - lo, m["data_offsets"][1] - lo
                dt = _DT[m["dtype"]]
                es = torch.empty(0, dtype=dt).element_size()
                if a % es == 0:
                    t = buf[a:b].view(dt)
                else:  # misaligned for the dtype: copy (only small tensors)
                    t = buf[a:b].clone().view(dt)
                out[k] = t.view(m["shape"])
        return out

    def layer_keys(self, layer):
        pre = f"model.layers.{layer}."
        return [k for k in self.wm if k.startswith(pre)]


class Trace:
    """Rows [0, n) of the 10K vLLM trace streams (chunked safetensors, index.json)."""

    def __init__(self, root=TRACE, n=8192):
        from safetensors import safe_open

        self._open = safe_open
        self.root = Path(root)
        self.ix = json.loads((self.root / "index.json").read_text())
        self.meta = json.loads((self.root / "metadata.json").read_text())
        self.streams = self.ix["tensor_streams"]
        self.n = n

    def has(self, key):
        return key in self.streams

    def rows(self, key, lo=0, hi=None):
        hi = self.n if hi is None else hi
        parts = []
        for ch in self.streams[key]["chunks"]:
            cs, ce = int(ch["row_start"]), int(ch["row_end"])
            a, b = max(lo, cs), min(hi, ce)
            if a < b:
                with self._open(str(self.root / ch["path"]), framework="pt", device="cpu") as fh:
                    names = list(fh.keys())
                    name = key if key in names else names[0]
                    parts.append(fh.get_slice(name)[a - cs : b - cs])
        out = torch.cat(parts, 0)
        assert out.shape[0] == hi - lo, (key, out.shape, lo, hi)
        return out

    def layer(self, l, is_mla):
        d = {"attn": self.rows(f"attn_output_layer_{l}"), "ffn": self.rows(f"decoder_output_layer_{l}")}
        if self.has(f"expert_ids_layer_{l}"):
            d["ids"] = self.rows(f"expert_ids_layer_{l}").long()
            d["w"] = self.rows(f"expert_weights_layer_{l}").float()
        if is_mla:
            for k in ("dsa_topk_indices", "kv_post_transform", "indexer_k"):
                if self.has(f"{k}_layer_{l}"):
                    d[k] = self.rows(f"{k}_layer_{l}")
        if l % BLOCK == 0 and self.has(f"residual_layer_{l}"):
            d["residual"] = self.rows(f"residual_layer_{l}")[:, l // BLOCK]
        return d


class Writer:
    """Background torch.save so compute never waits on NFS."""

    def __init__(self, depth=6):
        self.q = queue.Queue(depth)
        self.err = None
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            path, obj = item
            if self.err is not None:
                self.q.task_done()
                continue
            try:
                if isinstance(obj, str):
                    Path(path).write_text(obj)
                else:
                    tmp = str(path) + ".tmp"
                    torch.save(obj, tmp)
                    os.replace(tmp, path)
            except Exception as e:  # surfaced on the next put / flush; later items (done markers) are skipped
                self.err = e
            finally:
                self.q.task_done()

    def put(self, path, obj):
        if self.err:
            raise self.err
        self.q.put((path, obj))

    def flush(self):
        self.q.join()
        if self.err:
            raise self.err


# ───────────────────────────── numerics ─────────────────────────────


def snr(ref, got):
    r, g = ref.double().flatten(), got.double().flatten()
    n = ((g - r) ** 2).sum()
    return round(float(10 * math.log10(float((r * r).sum()) / float(n))), 3) if n > 0 else float("inf")


def segs(ref, got, n):
    out = {"all": snr(ref, got)}
    if n > 4096:
        out["c0"] = snr(ref[:4096], got[:4096])
        out["c1"] = snr(ref[4096:], got[4096:])
    return out


def bf(t):
    return t.to(torch.bfloat16).float()


class Prec:
    """Precision policy. fp32: identity rounding, fp32 GEMMs. bf16: bf16 storage between ops, bf16 GEMMs (fp32 acc)."""

    def __init__(self, name):
        self.name = name
        self.bf16 = name == "bf16"

    def rb(self, t):
        return bf(t) if self.bf16 else t

    def mm(self, x, w):
        """x [.., in] fp32 container, w [out, in] checkpoint weight (bf16/fp32) -> [.., out] fp32 container.

        bf16: a bf16 x bf16 product is exact in fp32, so an fp32 GEMM on bf16-rounded operands with the output
        rounded to bf16 is a bf16 GEMM with fp32 accumulate (as cuBLAS on the GPU), independent of oneDNN's kernels.
        """
        if self.bf16:
            wf = w.float() if w.dtype == torch.bfloat16 else bf(w.float())
            return bf(torch.matmul(bf(x), wf.t()))
        return torch.matmul(x, w.float().t())


def rmsnorm(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def situ(g, u):
    return SITU_BETA * torch.tanh(g / SITU_BETA) * torch.sigmoid(g) * (SITU_LBETA * torch.tanh(u / SITU_LBETA))


def attn_res_read(run, blocks, norm_w, proj_w, p, chunk=1024):
    """HF _apply_attn_res over [blocks..., run]. fp32 policy: all in fp64, fp32 out. bf16: HF's fp32 math, bf16 out."""
    cands = list(blocks) + [run]
    N = run.shape[0]
    dt = torch.float32 if p.bf16 else torch.float64
    sw = norm_w.to(dt) * proj_w.to(dt).flatten()
    out = torch.empty(N, H)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        v = torch.stack([c[a:b].to(dt) for c in cands], 1)  # [n, S, H]
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
        pr = (k * sw).sum(-1).softmax(-1)  # [n, S]
        out[a:b] = torch.matmul(pr.unsqueeze(1), v).squeeze(1).float()
    return p.rb(out)


# ───────────────────────────── modules ─────────────────────────────


def kda(l, x, W, p):
    """Kimi Delta Attention, fresh state over all rows. x [N, H] (AttnRes read #1, un-normed)."""
    pre = f"model.layers.{l}.self_attn."
    N = x.shape[0]
    NL, D = KDA_HEADS, KDA_D
    xn = p.rb(rmsnorm(x, W[f"model.layers.{l}.input_layernorm.weight"]))

    def branch(name, l2):
        proj = p.rb(p.mm(xn, W[pre + f"{name}_proj.weight"]))
        taps = W[pre + f"{name}_conv1d.weight"].float().reshape(NL * D, CONV_K)
        acc = proj * taps[:, CONV_K - 1]
        for s in range(1, CONV_K):  # tap CONV_K-1-s multiplies row t-s (zero history before row 0)
            acc[s:] += proj[:-s] * taps[:, CONV_K - 1 - s]
        act = p.rb(F.silu(acc)).view(N, NL, D)
        if l2:
            act = act * torch.rsqrt(act.pow(2).sum(-1, keepdim=True) + L2_EPS)
        return act

    q = branch("q", True) * (D**-0.5)
    k = branch("k", True)
    v = branch("v", False)
    rdt = torch.float32 if p.bf16 else torch.float64
    f = p.rb(p.mm(p.rb(p.mm(xn, W[pre + "f_a_proj.weight"])), W[pre + "f_b_proj.weight"])).view(N, NL, D)
    a_log = W[pre + "A_log"].float().flatten()[:NL]
    dt_bias = W[pre + "dt_bias"].float().reshape(NL, D)
    g = GATE_LB * torch.sigmoid(torch.exp(a_log.to(rdt)).view(1, NL, 1) * (f.to(rdt) + dt_bias.to(rdt)))
    alpha = torch.exp(g)
    del f, g
    beta = torch.sigmoid(p.rb(p.mm(xn, W[pre + "b_proj.weight"]))).to(rdt)  # [N, NL]
    qd, kd, vd = q.to(rdt), k.to(rdt), v.to(rdt)
    del q, k, v
    S = torch.zeros(NL, D, D, dtype=rdt)
    o = torch.empty(N, NL, D, dtype=rdt)
    for t in range(N):  # token-ordered delta rule, state [heads, D_k, D_v]
        S.mul_(alpha[t].unsqueeze(-1))
        pr = torch.bmm(kd[t].unsqueeze(1), S).squeeze(1)
        u = beta[t].unsqueeze(-1) * (vd[t] - pr)
        S.baddbmm_(kd[t].unsqueeze(2), u.unsqueeze(1))
        o[t] = torch.bmm(qd[t].unsqueeze(1), S).squeeze(1)
    del qd, kd, vd, alpha, beta, S
    o = p.rb(o.float())
    gate = p.rb(p.mm(xn, W[pre + "g_proj.weight"])).view(N, NL, D)
    del xn
    o = p.rb(
        o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * W[pre + "o_norm.weight"].float() * torch.sigmoid(gate)
    )
    return p.rb(p.mm(o.reshape(N, NL * D), W[pre + "o_proj.weight"])), {}


def mla(l, x, W, p, qb=64):
    """K3.1 MLA (NoPE, output gate) with the DSA indexer; dense masked attention per query block."""
    pre = f"model.layers.{l}.self_attn."
    N = x.shape[0]
    NH = MLA_HEADS
    xn = p.rb(rmsnorm(x, W[f"model.layers.{l}.input_layernorm.weight"]))
    qan = p.rb(rmsnorm(p.rb(p.mm(xn, W[pre + "q_a_proj.weight"])), W[pre + "q_a_layernorm.weight"]))
    q = p.rb(p.mm(qan, W[pre + "q_b_proj.weight"])).view(N, NH, NOPE + ROPE)
    kva = p.rb(p.mm(xn, W[pre + "kv_a_proj_with_mqa.weight"]))
    kv = torch.cat(
        [p.rb(rmsnorm(kva[:, :KV_LORA], W[pre + "kv_a_layernorm.weight"])), kva[:, KV_LORA:]], -1
    )  # [N, 576]
    del kva
    wkv = W[pre + "kv_b_proj.weight"].float().view(NH, NOPE + V_DIM, KV_LORA)
    W_UK, W_UV = wkv[:, :NOPE], wkv[:, NOPE:]  # [h, 128, 512]
    q_lat = p.rb(torch.einsum("nhd,hdc->nhc", q[..., :NOPE], W_UK))
    q_full = torch.cat([q_lat, q[..., NOPE:]], -1)  # [N, h, 576]
    del q, q_lat
    gate = p.rb(torch.sigmoid(p.rb(p.mm(xn, W[pre + "g_proj.weight"]))))  # [N, h*128]
    # indexer
    qi = p.rb(p.mm(qan, W[pre + "indexer.wq_b.weight"])).view(N, IDX_HEADS, IDX_D)
    ki = p.rb(
        F.layer_norm(
            p.rb(p.mm(xn, W[pre + "indexer.wk.weight"])),
            (IDX_D,),
            W[pre + "indexer.k_norm.weight"].float(),
            W[pre + "indexer.k_norm.bias"].float(),
            EPS,
        )
    )
    wts = p.rb(p.mm(xn, W[pre + "indexer.weights_proj.weight"]))  # [N, 32]
    del xn, qan
    topk = torch.full((N, TOPK), -1, dtype=torch.int64)
    for a in range(0, N, 256):
        b = min(N, a + 256)
        if b <= TOPK:
            continue
        lg = torch.matmul(qi[a:b].reshape(-1, IDX_D), ki[:b].t()).view(b - a, IDX_HEADS, b).clamp_min_(0)
        sc = torch.einsum("ths,th->ts", lg, wts[a:b]) / 64.0
        del lg
        sc.masked_fill_(torch.arange(b).unsqueeze(0) > torch.arange(a, b).unsqueeze(1), float("-inf"))
        r0 = max(a, TOPK)
        topk[r0:b] = sc[r0 - a :].topk(TOPK, dim=-1).indices
    small = torch.arange(min(N, TOPK))
    topk[: min(N, TOPK)] = torch.where(
        small.unsqueeze(1) >= torch.arange(TOPK).unsqueeze(0), torch.arange(TOPK).unsqueeze(0), -1
    )[:, :TOPK]
    # attention over the selected positions
    scale = (NOPE + ROPE) ** -0.5
    out = torch.empty(N, H)
    o_proj_f = W[pre + "o_proj.weight"].float()
    for a in range(0, N, qb):
        b = min(N, a + qb)
        ii = topk[a:b]
        allowed = torch.zeros(b - a, b + 1, dtype=torch.bool)  # column b collects the -1 pads
        allowed.scatter_(1, torch.where(ii >= 0, ii, b), True)
        allowed = allowed[:, :b]
        s = torch.matmul(q_full[a:b].reshape(-1, KV_LORA + ROPE), kv[:b].t()).view(b - a, NH, b) * scale
        if p.bf16:  # flash-attention style: fp32 scores/softmax, bf16 probabilities into the PV GEMM
            s.masked_fill_(~allowed.unsqueeze(1), float("-inf"))
            pr = bf(torch.softmax(s, -1))
        else:
            s64 = s.double().masked_fill_(~allowed.unsqueeze(1), float("-inf"))
            pr = torch.softmax(s64, -1).float()
            del s64
        del s
        o_lat = p.rb(torch.matmul(pr.view(-1, b), kv[:b, :KV_LORA]).view(b - a, NH, KV_LORA))
        del pr
        o = p.rb(torch.einsum("bhc,hvc->bhv", o_lat, W_UV)).reshape(b - a, NH * V_DIM)
        og = p.rb(o * gate[a:b])
        out[a:b] = p.rb(p.mm(og, o_proj_f))
    extra = {"topk": topk, "kv": kv, "idx_k": ki}
    return out, extra


def router(l, x_h, W):
    pre = f"model.layers.{l}.block_sparse_moe.gate."
    logits = torch.matmul(x_h.float(), W[pre + "weight"].float().t())  # HF: fp32 router GEMM
    scores = torch.sigmoid(logits)
    choice = scores + W[pre + "e_score_correction_bias"].float()
    ids = torch.topk(choice, TOP_E, dim=-1, sorted=True).indices
    return ids, scores


def route_weights(scores, ids):
    w = scores.gather(1, ids)
    return w / (w.sum(-1, keepdim=True) + RENORM_EPS)


def experts(l, latent, variants, W, p):
    """Routed experts over the union of (row, expert) pairs of several routings; one weighted sum per routing."""
    pre = f"model.layers.{l}.block_sparse_moe.experts."
    N = latent.shape[0]
    outs = [torch.zeros(N, LATENT) for _ in variants]
    flat = []
    for ids, w in variants:
        e = ids.reshape(-1)
        r = torch.arange(N).repeat_interleave(TOP_E)
        ww = w.reshape(-1)
        order = torch.argsort(e * N + r)
        e, r, ww = e[order], r[order], ww[order]
        off = torch.zeros(N_EXP + 1, dtype=torch.long)
        off[1:] = torch.cumsum(torch.bincount(e, minlength=N_EXP), 0)
        flat.append((r, ww, off.tolist()))
    for ex in range(N_EXP):
        rows_v = [(r[o[ex] : o[ex + 1]], ww[o[ex] : o[ex + 1]]) for r, ww, o in flat]
        if all(rv.numel() == 0 for rv, _ in rows_v):
            continue
        rows_u = torch.unique(torch.cat([rv for rv, _ in rows_v]))
        xe = latent[rows_u]
        g = p.rb(p.mm(xe, W[pre + f"{ex}.w1.weight"]))
        u = p.rb(p.mm(xe, W[pre + f"{ex}.w3.weight"]))
        oe = p.rb(p.mm(p.rb(situ(g, u)), W[pre + f"{ex}.w2.weight"]))
        for vi, (rv, wv) in enumerate(rows_v):
            if rv.numel():
                pos = torch.searchsorted(rows_u, rv)
                outs[vi].index_add_(0, rv, oe[pos] * wv.unsqueeze(1))
    return outs


def moe(l, x, W, p, trace_ids=None):
    """Latent MoE block; x = AttnRes read #2 (un-normed). Returns out, extras (routing, optional trace-routing out)."""
    pre = f"model.layers.{l}.block_sparse_moe."
    x_h = p.rb(rmsnorm(x, W[f"model.layers.{l}.post_attention_layernorm.weight"]))
    ids, scores = router(l, x_h, W)
    w = route_weights(scores, ids)
    variants = [(ids, w)]
    if trace_ids is not None:
        variants.append((trace_ids, route_weights(scores, trace_ids)))
    latent = p.rb(p.mm(x_h, W[pre + "routed_expert_down_proj.weight"]))
    routed = experts(l, latent, variants, W, p)
    del latent

    def tail(y):
        y = p.rb(y)  # HF moe_infer: fp32 weighted sum cast back to the activation dtype
        y = p.rb(rmsnorm(y, W[pre + "routed_expert_norm.weight"]))
        return p.rb(p.mm(y, W[pre + "routed_expert_up_proj.weight"]))

    sp = pre + "shared_experts."
    sh = p.rb(
        p.mm(
            p.rb(situ(p.rb(p.mm(x_h, W[sp + "gate_proj.weight"])), p.rb(p.mm(x_h, W[sp + "up_proj.weight"])))),
            W[sp + "down_proj.weight"],
        )
    )
    out = p.rb(tail(routed[0]) + sh)
    extra = {"ids": ids, "w": w}
    if trace_ids is not None:
        extra["out_trace_route"] = p.rb(tail(routed[1]) + sh)
    return out, extra


def dense_mlp(l, x, W, p, chunk=2048):
    pre = f"model.layers.{l}.mlp."
    x_h = p.rb(rmsnorm(x, W[f"model.layers.{l}.post_attention_layernorm.weight"]))
    out = torch.empty_like(x_h)
    for a in range(0, x.shape[0], chunk):
        xc = x_h[a : a + chunk]
        g = p.rb(p.mm(xc, W[pre + "gate_proj.weight"]))
        u = p.rb(p.mm(xc, W[pre + "up_proj.weight"]))
        out[a : a + chunk] = p.rb(p.mm(p.rb(situ(g, u)), W[pre + "down_proj.weight"]))
    return out, {}


# ───────────────────────────── tracks ─────────────────────────────


def overlap_sets(a, b):
    """per-row |a & b| for [N, k] int sets (negative entries ignored)."""
    rows = torch.arange(a.shape[0]).unsqueeze(1) << 32
    hit = torch.isin(rows + a.long(), rows + b.long()) & (a >= 0)
    return hit.sum(1)


class Track:
    def __init__(self, name, out, n, mla_layers, writer, save=True):
        self.name, self.n, self.mla_layers, self.writer, self.save = name, n, mla_layers, writer, save
        self.dir = Path(out) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.p = Prec("bf16" if name == "bf16" else "fp32")
        self.tf = name == "tf"
        self.run, self.blocks = None, []
        self.metrics = self.dir / "metrics.jsonl"

    def done(self, l):
        return (self.dir / f"done_L{l:02d}").exists()

    def first_todo(self, last):
        for l in range(last + 1):
            if not self.done(l):
                return l
        return last + 1

    def restore(self, l, emb, trace):
        """State entering layer l (l = first layer not done)."""
        if self.tf:
            self.run, self.blocks = emb, []
            for j in range(l):  # replay the trace stream through layers < l (no compute)
                if j % BLOCK == 0:
                    self.blocks.append(self.run)
                    self.run = None
                a = trace.rows(f"attn_output_layer_{j}")
                self.run = a if self.run is None else bf(self.run + a.float())
                self.run = bf(self.run + trace.rows(f"decoder_output_layer_{j}").float())
            return
        if l == 0:
            self.run, self.blocks = (bf(emb) if self.p.bf16 else emb.float()), []
            return
        ld = lambda j: torch.load(self.dir / f"entry_L{j:02d}.pt")["run"].float()
        self.blocks = [ld(b * BLOCK) for b in range((l - 1) // BLOCK + 1)]
        self.run = ld(l)

    def put(self, fname, obj):
        if self.save:
            self.writer.put(self.dir / fname, obj)

    def step(self, l, W, tr):
        p, n = self.p, self.n
        m = {"layer": l, "track": self.name}
        t0 = time.time()
        if not self.tf:
            self.put(f"entry_L{l:02d}.pt", {"run": self.run.clone() if not p.bf16 else self.run.to(torch.bfloat16)})
        x_attn = (
            self.run.float()
            if not self.blocks
            else attn_res_read(
                self.run,
                self.blocks,
                W[f"model.layers.{l}.self_attention_res_norm.weight"],
                W[f"model.layers.{l}.self_attention_res_proj.weight"],
                p,
            )
        )
        if l % BLOCK == 0:
            self.blocks.append(self.run)
            self.run = None
        is_mla = l in self.mla_layers
        a, ax = (mla if is_mla else kda)(l, x_attn, W, p)
        del x_attn
        m["t_attn"] = round(time.time() - t0, 1)
        m["attn_vs_trace"] = segs(tr["attn"].float(), a, n)
        if self.tf:
            m["attn_bf16_floor"] = snr(a, bf(a))
        if is_mla:
            ttk = tr.get("dsa_topk_indices")
            if ttk is not None and n > TOPK:
                ov = overlap_sets(ax["topk"][TOPK:n], ttk[TOPK:n].long()).float()
                m["dsa_recall_vs_trace"] = {
                    "mean": round(float(ov.mean()) / TOPK, 4),
                    "min": round(float(ov.min()) / TOPK, 4),
                }
            if "kv_post_transform" in tr:
                kt = tr["kv_post_transform"].float()
                m["kv_vs_trace"] = {
                    "latent": snr(kt[:, :KV_LORA], ax["kv"][:, :KV_LORA]),
                    "rope": snr(kt[:, KV_LORA:], ax["kv"][:, KV_LORA:]),
                }
            if "indexer_k" in tr:
                m["idx_k_vs_trace"] = snr(tr["indexer_k"].float(), ax["idx_k"])
            self.put(f"dsa_L{l:02d}.pt", {"topk": ax["topk"].to(torch.int16)})
        self.put(f"attn_L{l:02d}.pt", a.to(torch.bfloat16) if p.bf16 else a)
        # residual add (fp32 track: fp32; bf16 track and trace replay: bf16 adds)
        a_state = tr["attn"] if self.tf else a
        if self.run is None:
            self.run = a_state.float() if not (p.bf16 or self.tf) else bf(a_state.float())
        else:
            self.run = (self.run + a_state.float()) if not (p.bf16 or self.tf) else bf(self.run + a_state.float())
        del a
        t1 = time.time()
        x_ffn = attn_res_read(
            self.run,
            self.blocks,
            W[f"model.layers.{l}.mlp_res_norm.weight"],
            W[f"model.layers.{l}.mlp_res_proj.weight"],
            p,
        )
        if l == 0:
            f, fx = dense_mlp(l, x_ffn, W, p)
        else:
            f, fx = moe(l, x_ffn, W, p, trace_ids=tr["ids"] if (self.tf and "ids" in tr) else None)
        del x_ffn
        m["t_ffn"] = round(time.time() - t1, 1)
        m["ffn_vs_trace"] = segs(tr["ffn"].float(), f, n)
        if "ids" in fx:
            tid = tr.get("ids")
            if tid is not None:
                ov = overlap_sets(fx["ids"], tid)
                same = ov == TOP_E
                m["route_vs_trace"] = {
                    "identical": round(float(same.float().mean()), 4),
                    "overlap": round(float(ov.float().mean()), 3),
                }
                if self.tf:
                    tf_ = tr["ffn"].float()
                    m["ffn_same_route_vs_trace"] = {"snr": snr(tf_[same], f[same]), "rows": int(same.sum())}
                    m["ffn_trace_route_vs_trace"] = segs(tf_, fx["out_trace_route"], n)
            self.put(f"route_L{l:02d}.pt", {"ids": fx["ids"].to(torch.int16), "w": fx["w"]})
        self.put(f"ffn_L{l:02d}.pt", f.to(torch.bfloat16) if p.bf16 else f)
        f_state = tr["ffn"] if self.tf else f
        self.run = (self.run + f_state.float()) if not (p.bf16 or self.tf) else bf(self.run + f_state.float())
        del f
        if self.tf and "residual" in tr:  # replay check: sealed plane vs the tracer's own snapshot
            m["replay_block_maxdiff"] = float((self.blocks[l // BLOCK] - tr["residual"].float()).abs().max())
        m["run_rms"] = round(float(self.run.pow(2).mean().sqrt()), 4)
        m["t_layer"] = round(time.time() - t0, 1)
        with open(self.metrics, "a") as fh:
            fh.write(json.dumps(m) + "\n")
        log(f"{self.name} L{l}: " + json.dumps({k: v for k, v in m.items() if k not in ("layer", "track")}))

    def mark_done(self, l):
        # through the writer's FIFO, so the marker lands only after this layer's files (skipped if one failed)
        self.writer.put(self.dir / f"done_L{l:02d}", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def final(self, Wh):
        p = self.p
        h = attn_res_read(
            self.run, self.blocks, Wh["model.output_attn_res_norm.weight"], Wh["model.output_attn_res_proj.weight"], p
        )
        hn = p.rb(rmsnorm(h, Wh["model.norm.weight"]))
        lmf = Wh["lm_head.weight"].float()
        N = hn.shape[0]
        logits = torch.empty(N, VOCAB)
        logits32 = torch.empty(N, VOCAB) if p.bf16 else None
        for a in range(0, N, 512):
            b = min(N, a + 512)
            if p.bf16:  # vLLM: lm_head in the model dtype (bf16 logits, fp32 accumulate); also kept unrounded
                logits32[a:b] = torch.matmul(hn[a:b], lmf.t())
                logits[a:b] = bf(logits32[a:b])
            else:
                logits[a:b] = torch.matmul(hn[a:b], lmf.t())
        lse = torch.logsumexp(logits.double(), -1)
        res = {"hidden": h.to(torch.bfloat16) if p.bf16 else h, "lse": lse, "top1": logits.argmax(-1)}
        if logits32 is not None:
            res["lse_fp32head"] = torch.logsumexp(logits32.double(), -1)
            self.writer.put(self.dir / "logits_fp32head.pt", logits32)
        self.writer.put(self.dir / "final.pt", res)
        self.writer.put(self.dir / "logits.pt", logits)
        self.writer.flush()
        (self.dir / "done_final").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log(f"{self.name} final: logits saved, mean lse {float(lse.mean()):.4f}")


# ───────────────────────────── driver ─────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="fp32,tf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--last-layer", type=int, default=N_LAYERS - 1)
    ap.add_argument("--threads", type=int, default=30)
    ap.add_argument("--io-threads", type=int, default=16)
    ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--heartbeat", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    n = args.rows
    ck = Checkpoint(threads=args.io_threads)
    cfg = ck.cfg
    assert cfg["hidden_size"] == H and cfg["num_hidden_layers"] == N_LAYERS and cfg["attn_res_block_size"] == BLOCK
    assert (
        cfg["num_experts"] == N_EXP
        and cfg["num_experts_per_token"] == TOP_E
        and cfg["routed_expert_hidden_size"] == LATENT
    )
    assert cfg["activation_situ_beta"] == SITU_BETA and cfg["activation_situ_linear_beta"] == SITU_LBETA
    assert cfg["routed_scaling_factor"] == 1.0 and cfg["moe_renormalize"] and cfg["rms_norm_eps"] == EPS
    mla_layers = {i - 1 for i in cfg["linear_attn_config"]["full_attn_layers"]}
    assert {i - 1 for i in cfg["linear_attn_config"]["kda_layers"]} | mla_layers == set(range(N_LAYERS))
    trace = Trace(n=n)
    assert sorted(trace.ix["dsa_layers"]) == sorted(mla_layers), "trace DSA layers != config full-attention layers"
    tokens = torch.tensor(trace.meta["token_ids"][:n], dtype=torch.long)
    assert len(tokens) == n
    writer = Writer()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else out / "heartbeat.txt"
    tracks = [Track(t, out, n, mla_layers, writer) for t in args.tracks.split(",")]
    last = args.last_layer
    starts = {t.name: t.first_todo(last) for t in tracks}
    log(f"rows {n}, tracks {starts}, last layer {last}, threads {args.threads}")
    head_keys = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.output_attn_res_norm.weight",
        "model.output_attn_res_proj.weight",
    ]
    Wh = ck.load(head_keys)
    emb_bf = Wh["model.embed_tokens.weight"][tokens].clone()
    emb_tr = trace.rows("decoder_input_layer_0")
    log(f"embedding vs trace decoder_input_layer_0: equal={bool(torch.equal(emb_bf, emb_tr))}")
    emb = emb_bf.float()
    for t in tracks:
        t.restore(starts[t.name], emb_bf if t.tf else emb, trace)
    first = min(starts.values())
    pool = ThreadPoolExecutor(1)

    def fetch(l):
        ts = time.time()
        W = ck.load(ck.layer_keys(l))
        tr = trace.layer(l, l in mla_layers)
        return W, tr, time.time() - ts

    fut = pool.submit(fetch, first) if first <= last else None
    for l in range(first, last + 1):
        tw = time.time()
        W, tr, t_io = fut.result()
        wait = time.time() - tw
        fut = pool.submit(fetch, l + 1) if l + 1 <= last else None
        log(
            f"L{l}: weights ready (load {t_io:.0f}s, waited {wait:.0f}s, {sum(v.numel() * v.element_size() for v in W.values()) / 2**30:.1f} GiB)"
        )
        for t in tracks:
            if l < starts[t.name]:
                continue
            hb.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {t.name} L{l}\n")
            t.step(l, W, tr)
            t.mark_done(l)
        del W, tr
    if not args.no_final and last == N_LAYERS - 1:
        for t in tracks:
            if not (t.dir / "done_final").exists():
                hb.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {t.name} final\n")
                t.final(Wh)
    writer.flush()
    hb.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} done\n")
    log("done")


if __name__ == "__main__":
    main()
