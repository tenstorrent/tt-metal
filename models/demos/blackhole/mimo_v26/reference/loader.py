"""MiMo-V2.6-Flash checkpoint loader (CPU, layer streaming). Shared by all tracks.

Checkpoint dir: env MIMO_CKPT (HF snapshot with model.safetensors.index.json + shards).
Facts (see bringup/mimo_v26/LOG.md decision entries):
  * qkv_proj is stored per TP4 rank: rows = [q_r (16 heads*192) | k_r (nkv/4*192) | v_r (nkv/4*128)] x 4, and the fp8 128x128
    scale blocks run over each rank's rows (GA 3392 rows -> 27 blocks, SWA 3712 -> 29). `deinterleave_qkv` returns the HF layout
    (contiguous Q|K|V, heads in order) that modeling_mimo_v2.py expects.
  * fp4 experts: DSV4 convention (low nibble first, FP4 E2M1 table, e8m0 scale per 32 along K). UNVERIFIED on MiMo tensors until
    a checkpoint is available (see LOG).
"""
import json
import os

import torch

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)
TP = 4  # checkpoint tp_size (index metadata)
HEAD_QK, HEAD_V = 192, 128
GLOBAL_LAYERS = (0, 5, 11, 17, 23, 29, 35, 41, 47)


def is_swa(i):
    return i not in GLOBAL_LAYERS


def dequant_fp8_block(w, s, block=128):
    """w fp8 [R,C], s fp32 [ceil(R/b),ceil(C/b)] -> fp32 [R,C] (contiguous row blocks)."""
    R, C = w.shape
    se = s.repeat_interleave(block, 0)[:R].repeat_interleave(block, 1)[:, :C]
    return w.float() * se


def dequant_fp8_qkv_ranked(w, s, nq, nk, nv, tp=TP, block=128):
    """Dequant the rank-blocked fused qkv: per rank, rows (nq+nk+nv)//tp use that rank's scale blocks. Returns fp32 in CHECKPOINT row order."""
    per = (nq + nk + nv) // tp
    nb = -(-per // block)
    assert w.shape[0] == per * tp and s.shape[0] == nb * tp, (w.shape, s.shape)
    return torch.cat(
        [dequant_fp8_block(w[r * per : (r + 1) * per], s[r * nb : (r + 1) * nb], block) for r in range(tp)], 0
    )


def deinterleave_qkv(w, nq, nk, nv, tp=TP):
    """Checkpoint row order [q_r|k_r|v_r]xtp -> contiguous (Q [nq], K [nk], V [nv]) rows (HF order)."""
    per = (nq + nk + nv) // tp
    qs, ks, vs = nq // tp, nk // tp, nv // tp
    w = w.view(tp, per, *w.shape[1:])
    return torch.cat(
        [
            w[:, :qs].reshape(nq, *w.shape[2:]),
            w[:, qs : qs + ks].reshape(nk, *w.shape[2:]),
            w[:, qs + ks :].reshape(nv, *w.shape[2:]),
        ],
        0,
    )


def qkv_dims(i, nh=64, swa_kv=8, ga_kv=4):
    nkv = swa_kv if is_swa(i) else ga_kv
    return nh * HEAD_QK, nkv * HEAD_QK, nkv * HEAD_V


def dequant_qkv(i, w, s):
    """Layer i fused qkv fp8 + scale -> fp32 [q_size+k_size+v_size, 4096] in HF contiguous order."""
    nq, nk, nv = qkv_dims(i)
    return deinterleave_qkv(dequant_fp8_qkv_ranked(w, s, nq, nk, nv), nq, nk, nv)


def dequant_fp4_weight(w, s, block=32):
    """packed fp4 [N,K/2] uint8 (low nibble first) + e8m0 scale [N,K/32] -> fp32 [N,K]."""
    b = w.view(torch.uint8)
    pair = torch.stack([FP4_TABLE[(b & 0x0F).long()], FP4_TABLE[(b >> 4).long()]], dim=-1)
    n = b.size(0)
    return (pair.reshape(n, -1).view(n, -1, block) * s.float().unsqueeze(-1)).reshape(n, -1)


class Ckpt:
    def __init__(self, path=None):
        self.path = path or os.environ["MIMO_CKPT"]
        with open(os.path.join(self.path, "model.safetensors.index.json")) as f:
            self.wm = json.load(f)["weight_map"]
        self._open = {}

    def get(self, name):
        from safetensors import safe_open

        fn = self.wm[name]
        if fn not in self._open:
            self._open[fn] = safe_open(os.path.join(self.path, fn), "pt")
        return self._open[fn].get_tensor(name)

    def has(self, name):
        return name in self.wm


def _e8m0(t):
    return t.view(torch.float8_e8m0fnu).float() if t.dtype == torch.uint8 else t.float()


def load_layer(i, dtype=torch.bfloat16, ckpt=None, experts=True):
    """Dequantized HF-layout state dict for decoder layer i (keys relative to model.layers.{i}.). Streams; experts ~13 GB bf16/layer."""
    ck = ckpt or Ckpt()
    p = f"model.layers.{i}."
    sd = {}
    for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        sd[n] = ck.get(p + n).to(dtype)
    sd["self_attn.qkv_proj.weight"] = dequant_qkv(
        i, ck.get(p + "self_attn.qkv_proj.weight"), ck.get(p + "self_attn.qkv_proj.weight_scale_inv")
    ).to(dtype)
    sd["self_attn.o_proj.weight"] = ck.get(p + "self_attn.o_proj.weight").to(dtype)
    if ck.has(p + "self_attn.attention_sink_bias"):
        sd["self_attn.attention_sink_bias"] = ck.get(p + "self_attn.attention_sink_bias").to(dtype)
    if i == 0:
        for n in ("gate_proj", "up_proj", "down_proj"):
            sd[f"mlp.{n}.weight"] = dequant_fp8_block(
                ck.get(p + f"mlp.{n}.weight"), ck.get(p + f"mlp.{n}.weight_scale_inv")
            ).to(dtype)
    else:
        sd["mlp.gate.weight"] = ck.get(p + "mlp.gate.weight").to(dtype)
        sd["mlp.gate.e_score_correction_bias"] = ck.get(p + "mlp.gate.e_score_correction_bias").float()
        if experts:
            for e in range(256):
                for n in ("gate_proj", "up_proj", "down_proj"):
                    k = p + f"mlp.experts.{e}.{n}."
                    sd[f"mlp.experts.{e}.{n}.weight"] = dequant_fp4_weight(
                        ck.get(k + "weight"), _e8m0(ck.get(k + "weight_scale"))
                    ).to(dtype)
    return sd
