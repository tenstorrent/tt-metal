#!/usr/bin/env python3
"""
bf16-DeltaNet variant of hf_ref.py.

Mimics the TT kernel's bfloat16 on-core storage for the DeltaNet (linear_attention)
path ONLY, while keeping everything else (weights, matmuls, full_attention, MLP,
residual stream) in fp32 -- to decisively answer whether the ~1.5%-per-DeltaNet-layer
PCC decay seen in the TT port is consistent with bf16 intermediate precision or
indicates a real formula bug.

bf16 round-trips are inserted at:
  * conv1d output
  * silu output
  * L2-normed q/k
  * beta, decay (g)
  * recurrent state and per-step kv_mem/delta/out accumulation
  * gated-norm inputs (core_attn_out and z)

Optional full-bf16 hidden-state casting between layers via env FULL_BF16=1.

Run inside docker image qwen36-wh-test:latest:
  docker run --rm -e HF_HUB_OFFLINE=1 \
    -v /home/yito/tt-metal:/home/yito/tt-metal -v /home/yito/work:/home/yito/work \
    -w /home/yito/tt-metal qwen36-wh-test:latest \
    /opt/venv/bin/python3 models/demos/qwen36_27b/t3k/hf_ref_bf16.py
"""

import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.manual_seed(0)

HF_DIR = "/home/yito/work/qwen36_27b_hf"
OUT_PT = "/home/yito/work/hf_ref_acts_bf16.pt"

HIDDEN = 5120
N_LAYERS = 64
VOCAB = 248320
EPS = 1e-6
INTERMEDIATE = 17408

N_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
ATTN_BIAS = False
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY)
ROPE_THETA = 1.0e7
MROPE_SECTION = [11, 11, 10]
MROPE_INTERLEAVED = True
ATTN_SCALE = HEAD_DIM ** -0.5

LIN_N_K_HEADS = 16
LIN_N_V_HEADS = 48
LIN_K_HEAD_DIM = 128
LIN_V_HEAD_DIM = 128
LIN_CONV_K = 4
KEY_DIM = LIN_N_K_HEADS * LIN_K_HEAD_DIM
VALUE_DIM = LIN_N_V_HEADS * LIN_V_HEAD_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
V_PER_K = LIN_N_V_HEADS // LIN_N_K_HEADS

LAYER_TYPES = []
for i in range(N_LAYERS):
    LAYER_TYPES.append("full_attention" if (i + 1) % 4 == 0 else "linear_attention")

DTYPE = torch.float32
BF16 = torch.bfloat16
FULL_BF16 = os.environ.get("FULL_BF16", "0") == "1"

NORM_PLUS1 = os.environ.get("NORM_PLUS1", "1") == "1"
GATE_MODE = os.environ.get("GATE_MODE", "sigmoid")
QK_NORM_PLUS1 = os.environ.get("QK_NORM_PLUS1", "1") == "1"


def bf(x):
    """bf16 round-trip: store as bfloat16 then back to fp32 (mimic TT on-core storage)."""
    return x.to(BF16).to(torch.float32)


class Weights:
    def __init__(self, base):
        self.base = base
        idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]
        self._handles = {}

    def _h(self, fname):
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.base, fname), framework="pt")
        return self._handles[fname]

    def get(self, key):
        if key not in self.wm:
            raise KeyError(key)
        t = self._h(self.wm[key]).get_tensor(key)
        return t.to(DTYPE)


def L(layer_idx, suffix):
    return f"model.language_model.layers.{layer_idx}.{suffix}"


def rmsnorm(x, weight, eps=EPS, plus1=True):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    w = (1.0 + weight.float()) if plus1 else weight.float()
    out = out * w
    return out.type_as(x)


def rmsnorm_w_plus_1(x, weight, eps=EPS):
    return rmsnorm(x, weight, eps, plus1=NORM_PLUS1)


def rmsnorm_gated(x, weight, gate, eps=EPS):
    """Qwen3NextRMSNormGated: norm-before-gate, weight DIRECT (no +1), gate via silu.
    bf16 variant: inputs x (core) and gate (z) are already bf16-rounded by caller."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = weight.float() * out
    out = out * F.silu(gate.float())
    return out.type_as(x)


def build_rope(position_ids_1d):
    dim = ROTARY_DIM
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    seq = position_ids_1d.shape[0]
    pos3 = position_ids_1d.float()[None, :].expand(3, seq)
    freqs = pos3[:, :, None] * inv_freq[None, None, :]
    if MROPE_INTERLEAVED:
        freqs_t = freqs[0].clone()
        for d, offset in enumerate((1, 2), start=1):
            length = MROPE_SECTION[d] * 3
            idx = slice(offset, length, 3)
            freqs_t[:, idx] = freqs[d][:, idx]
        freqs_use = freqs_t
    else:
        freqs_use = freqs[0]
    emb = torch.cat((freqs_use, freqs_use), dim=-1)
    cos = emb.cos()[None]
    sin = emb.sin()[None]
    return cos.to(DTYPE), sin.to(DTYPE)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rot = cos.shape[-1]
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    k_rot, k_pass = k[..., :rot], k[..., rot:]
    q_emb = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_emb = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_emb, q_pass], dim=-1), torch.cat([k_emb, k_pass], dim=-1)


def full_attention(hidden, w, layer_idx, cos, sin):
    b, seq, _ = hidden.shape
    q_proj = w.get(L(layer_idx, "self_attn.q_proj.weight"))
    k_proj = w.get(L(layer_idx, "self_attn.k_proj.weight"))
    v_proj = w.get(L(layer_idx, "self_attn.v_proj.weight"))
    o_proj = w.get(L(layer_idx, "self_attn.o_proj.weight"))
    q_norm_w = w.get(L(layer_idx, "self_attn.q_norm.weight"))
    k_norm_w = w.get(L(layer_idx, "self_attn.k_norm.weight"))

    q = F.linear(hidden, q_proj)
    q = q.view(b, seq, N_HEADS, HEAD_DIM * 2)
    query, gate = torch.chunk(q, 2, dim=-1)
    gate = gate.reshape(b, seq, -1)

    query = rmsnorm(query, q_norm_w, plus1=QK_NORM_PLUS1)
    query = query.transpose(1, 2)

    key = F.linear(hidden, k_proj).view(b, seq, N_KV_HEADS, HEAD_DIM)
    key = rmsnorm(key, k_norm_w, plus1=QK_NORM_PLUS1).transpose(1, 2)
    value = F.linear(hidden, v_proj).view(b, seq, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

    query, key = apply_rope(query, key, cos, sin)

    n_rep = N_HEADS // N_KV_HEADS
    key = key.repeat_interleave(n_rep, dim=1)
    value = value.repeat_interleave(n_rep, dim=1)

    attn = torch.matmul(query, key.transpose(2, 3)) * ATTN_SCALE
    causal = torch.full((seq, seq), float("-inf")).triu(1)
    attn = attn + causal[None, None]
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn, value)
    out = out.transpose(1, 2).reshape(b, seq, -1)

    if GATE_MODE == "sigmoid":
        out = out * torch.sigmoid(gate)
    else:
        out = out * F.silu(gate)

    return F.linear(out, o_proj)


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gated_delta_rule_bf16(query, key, value, g, beta):
    """bf16-storage variant of the recurrence.
    L2-norm q/k, beta, decay, recurrent state, and per-step accumulators are
    bf16-rounded each step to mimic TT on-core bf16 storage."""
    # L2-normed q/k stored bf16
    query = bf(l2norm(query, dim=-1, eps=1e-6))
    key = bf(l2norm(key, dim=-1, eps=1e-6))

    q = query.transpose(1, 2).contiguous().float()
    k = key.transpose(1, 2).contiguous().float()
    v = bf(value).transpose(1, 2).contiguous().float()  # conv/silu already bf16 upstream
    bb = bf(beta).transpose(1, 2).contiguous().float()    # beta stored bf16
    gg = bf(g).transpose(1, 2).contiguous().float()        # decay stored bf16

    bsz, nh, slen, k_dim = k.shape
    v_dim = v.shape[-1]
    scale = 1.0 / (k_dim ** 0.5)
    q = bf(q * scale)

    out = torch.zeros(bsz, nh, slen, v_dim, dtype=torch.float32)
    state = torch.zeros(bsz, nh, k_dim, v_dim, dtype=torch.float32)
    for i in range(slen):
        q_t = q[:, :, i]
        k_t = k[:, :, i]
        v_t = v[:, :, i]
        g_t = bf(gg[:, :, i].exp())[..., None, None]   # decay factor bf16
        beta_t = bb[:, :, i][..., None]
        state = bf(state * g_t)                          # recurrent state bf16
        kv_mem = bf((state * k_t.unsqueeze(-1)).sum(dim=-2))
        delta = bf((v_t - kv_mem) * beta_t)
        state = bf(state + k_t.unsqueeze(-1) * delta.unsqueeze(-2))  # recurrent state bf16
        out[:, :, i] = bf((state * q_t.unsqueeze(-1)).sum(dim=-2))
    return out.transpose(1, 2).contiguous()


def linear_attention(hidden, w, layer_idx):
    b, seq, _ = hidden.shape
    in_qkv = w.get(L(layer_idx, "linear_attn.in_proj_qkv.weight"))
    in_z = w.get(L(layer_idx, "linear_attn.in_proj_z.weight"))
    in_b = w.get(L(layer_idx, "linear_attn.in_proj_b.weight"))
    in_a = w.get(L(layer_idx, "linear_attn.in_proj_a.weight"))
    conv_w = w.get(L(layer_idx, "linear_attn.conv1d.weight"))
    A_log = w.get(L(layer_idx, "linear_attn.A_log"))
    dt_bias = w.get(L(layer_idx, "linear_attn.dt_bias"))
    norm_w = w.get(L(layer_idx, "linear_attn.norm.weight"))
    out_proj = w.get(L(layer_idx, "linear_attn.out_proj.weight"))

    mixed_qkv = F.linear(hidden, in_qkv)
    z = F.linear(hidden, in_z)
    b_proj = F.linear(hidden, in_b)
    a_proj = F.linear(hidden, in_a)

    # causal depthwise conv1d over [q|k|v] then silu -- both stored bf16
    x = mixed_qkv.transpose(1, 2)
    x = F.conv1d(x, conv_w, bias=None, padding=LIN_CONV_K - 1, groups=CONV_DIM)
    x = x[:, :, :seq]
    x = bf(x)              # conv1d output bf16
    x = F.silu(x)
    x = bf(x)              # silu output bf16
    mixed_qkv = x.transpose(1, 2)

    query, key, value = torch.split(mixed_qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    query = query.reshape(b, seq, LIN_N_K_HEADS, LIN_K_HEAD_DIM)
    key = key.reshape(b, seq, LIN_N_K_HEADS, LIN_K_HEAD_DIM)
    value = value.reshape(b, seq, LIN_N_V_HEADS, LIN_V_HEAD_DIM)

    beta = b_proj.sigmoid()
    g = -A_log.float().exp() * F.softplus(a_proj.float() + dt_bias)

    query = query.repeat_interleave(V_PER_K, dim=2)
    key = key.repeat_interleave(V_PER_K, dim=2)

    core = recurrent_gated_delta_rule_bf16(query, key, value, g, beta)

    # gated RMSNorm with z: both inputs stored bf16
    z = z.reshape(b, seq, LIN_N_V_HEADS, LIN_V_HEAD_DIM)
    core = rmsnorm_gated(bf(core), norm_w, bf(z))
    core = core.reshape(b, seq, -1)

    return F.linear(core, out_proj)


def decoder_layer(hidden, w, layer_idx, cos, sin):
    in_ln = w.get(L(layer_idx, "input_layernorm.weight"))
    post_ln = w.get(L(layer_idx, "post_attention_layernorm.weight"))

    residual = hidden
    x = rmsnorm_w_plus_1(hidden, in_ln)
    if LAYER_TYPES[layer_idx] == "linear_attention":
        x = linear_attention(x, w, layer_idx)
    else:
        x = full_attention(x, w, layer_idx, cos, sin)
    hidden = residual + x

    residual = hidden
    x = rmsnorm_w_plus_1(hidden, post_ln)
    gate_w = w.get(L(layer_idx, "mlp.gate_proj.weight"))
    up_w = w.get(L(layer_idx, "mlp.up_proj.weight"))
    down_w = w.get(L(layer_idx, "mlp.down_proj.weight"))
    x = F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)
    hidden = residual + x

    if FULL_BF16:
        hidden = bf(hidden)
    return hidden


def main():
    w = Weights(HF_DIR)

    input_ids = torch.tensor([[760, 6511, 314, 9338, 369]], dtype=torch.long)
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, dtype=torch.long)
    cos, sin = build_rope(position_ids)

    embed = w.get("model.language_model.embed_tokens.weight")
    hidden = F.embedding(input_ids, embed)
    input_embeds = hidden.clone()

    layer_acts = torch.empty(N_LAYERS, 1, seq, HIDDEN, dtype=DTYPE)
    for li in range(N_LAYERS):
        hidden = decoder_layer(hidden, w, li, cos, sin)
        layer_acts[li] = hidden.detach()
        print(f"  layer {li:2d} ({LAYER_TYPES[li]:16s}) "
              f"mean={hidden.mean().item():.5f} std={hidden.std().item():.5f}")

    norm_w = w.get("model.language_model.norm.weight")
    final_norm = rmsnorm_w_plus_1(hidden, norm_w)

    lm_head = w.get("lm_head.weight")
    logits = F.linear(final_norm, lm_head)
    last_logits = logits[:, -1, :]

    topv, topi = torch.topk(last_logits[0], 5)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True)
        decoded = [tok.decode([int(i)]) for i in topi]
    except Exception as e:
        decoded = [f"<id {int(i)}>" for i in topi]
        print("tokenizer load failed:", e)

    print("\n=== TOP-5 NEXT TOKEN (FULL_BF16=%s) ===" % FULL_BF16)
    for v, i, d in zip(topv.tolist(), topi.tolist(), decoded):
        print(f"  id={i:7d} logit={v:9.4f}  {d!r}")

    torch.save(
        {
            "input_embeds": input_embeds,
            "layers": layer_acts,
            "final_norm": final_norm,
            "logits": last_logits,
            "input_ids": input_ids,
            "top5_ids": topi,
            "top5_logits": topv,
        },
        OUT_PT,
    )
    print(f"\nSaved activations to {OUT_PT}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
