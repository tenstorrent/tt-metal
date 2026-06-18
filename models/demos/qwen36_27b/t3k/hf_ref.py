#!/usr/bin/env python3
"""
Faithful pure-PyTorch CPU reference forward for the Qwen3.6-27B text model.

HF arch:  Qwen3_5ForConditionalGeneration  (text backbone model_type=qwen3_5_text)
This is a correctness oracle (NOT Tenstorrent hardware). Pure CPU torch / float32.

The qwen3_5 architecture is a hybrid of:
  * Qwen3Next gated-DeltaNet linear-attention layers + full-attention layers
    (modeling_qwen3_next.py is the authoritative recurrence / gate / norm math), and
  * Qwen3VL-style INTERLEAVED MRoPE for the full-attention RoPE
    (apply_interleaved_mrope from modeling_qwen3_vl.py),
with two config knobs that do not yet exist in transformers 4.57.6:
  * attn_output_gate=True; config says output_gate_type="swish" but the real
    weights want SIGMOID gating: attn_out * sigmoid(gate)  (verified empirically)
  * partial_rotary_factor=0.25                        -> rotary_dim = 64 (first 64 dims)

The projections for DeltaNet are stored ALREADY SPLIT
(in_proj_qkv / in_proj_z / in_proj_b / in_proj_a) rather than fused
(in_proj_qkvz / in_proj_ba) as in stock qwen3_next, so the per-head interleaving
of `fix_query_key_value_ordering` does NOT apply: q|k|v come straight out of the
[key_dim, key_dim, value_dim] split, z straight out of in_proj_z.

Run inside docker image qwen36-wh-test:latest:
  docker run --rm -e HF_HUB_OFFLINE=1 \
    -v /home/yito/tt-metal:/home/yito/tt-metal -v /home/yito/work:/home/yito/work \
    -w /home/yito/tt-metal qwen36-wh-test:latest \
    /opt/venv/bin/python3 models/demos/qwen36_27b/t3k/hf_ref.py
"""

import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.manual_seed(0)

# --------------------------------------------------------------------------- #
# Config (from text_config of config.json -- verified by reading the file)
# --------------------------------------------------------------------------- #
HF_DIR = "/home/yito/work/qwen36_27b_hf"
OUT_PT = "/home/yito/work/hf_ref_acts.pt"

HIDDEN = 5120
N_LAYERS = 64
VOCAB = 248320
EPS = 1e-6
INTERMEDIATE = 17408

# full attention
N_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
ATTN_BIAS = False
# RoPE
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY)  # 64
ROPE_THETA = 1.0e7
MROPE_SECTION = [11, 11, 10]  # sums to 32 = ROTARY_DIM // 2
MROPE_INTERLEAVED = True
ATTN_SCALE = HEAD_DIM ** -0.5

# linear attention (gated DeltaNet)
LIN_N_K_HEADS = 16
LIN_N_V_HEADS = 48
LIN_K_HEAD_DIM = 128
LIN_V_HEAD_DIM = 128
LIN_CONV_K = 4
KEY_DIM = LIN_N_K_HEADS * LIN_K_HEAD_DIM          # 2048
VALUE_DIM = LIN_N_V_HEADS * LIN_V_HEAD_DIM        # 6144
CONV_DIM = KEY_DIM * 2 + VALUE_DIM                # 10240
V_PER_K = LIN_N_V_HEADS // LIN_N_K_HEADS          # 3

LAYER_TYPES = []
for i in range(N_LAYERS):
    LAYER_TYPES.append("full_attention" if (i + 1) % 4 == 0 else "linear_attention")

DTYPE = torch.float32

# Switchable hypotheses (override via env for diagnostics).
#   NORM_PLUS1   : main RMSNorm uses (1+w) [qwen3_next] vs direct w [qwen3/qwen3_vl]
#   GATE_MODE    : "swish" (silu) | "sigmoid"
#   QK_NORM_PLUS1: q_norm/k_norm use (1+w) vs direct
# Empirically determined by sweep against the real weights (prompt -> " Paris"):
#   * main RMSNorm uses (1 + weight)            [qwen3_next semantics]   -> NORM_PLUS1=1
#   * q_norm / k_norm use (1 + weight)          [qwen3_next semantics]   -> QK_NORM_PLUS1=1
#   * attention output gate is SIGMOID, not swish/silu, even though
#     config says output_gate_type="swish"      [verified empirically]   -> GATE_MODE=sigmoid
NORM_PLUS1 = os.environ.get("NORM_PLUS1", "1") == "1"
GATE_MODE = os.environ.get("GATE_MODE", "sigmoid")
QK_NORM_PLUS1 = os.environ.get("QK_NORM_PLUS1", "1") == "1"

# --------------------------------------------------------------------------- #
# Lazy weight loader
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# RMSNorm variants
# --------------------------------------------------------------------------- #
def rmsnorm(x, weight, eps=EPS, plus1=True):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    w = (1.0 + weight.float()) if plus1 else weight.float()
    out = out * w
    return out.type_as(x)


def rmsnorm_w_plus_1(x, weight, eps=EPS):
    return rmsnorm(x, weight, eps, plus1=NORM_PLUS1)


def rmsnorm_gated(x, weight, gate, eps=EPS):
    """Qwen3NextRMSNormGated: norm-before-gate, weight DIRECT (no +1), gate via silu."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = weight.float() * out
    out = out * F.silu(gate.float())
    return out.type_as(x)


# --------------------------------------------------------------------------- #
# RoPE (interleaved MRoPE, partial rotary over first ROTARY_DIM dims)
# --------------------------------------------------------------------------- #
def build_rope(position_ids_1d):
    """position_ids_1d: LongTensor [seq]. Returns cos,sin of shape [1, seq, ROTARY_DIM].
    For text-only input, the 3 MRoPE position rows (T,H,W) are identical, so the
    interleaved selection is a no-op (selecting freqs[dim] == freqs[0]); we still
    implement the interleaved path faithfully so the code matches the true model."""
    dim = ROTARY_DIM
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # [dim//2]=32
    seq = position_ids_1d.shape[0]
    # 3 rows of identical positions for text-only
    pos3 = position_ids_1d.float()[None, :].expand(3, seq)  # [3, seq]
    # freqs[dim_idx, seq, n_freq]
    freqs = pos3[:, :, None] * inv_freq[None, None, :]  # [3, seq, 32]
    if MROPE_INTERLEAVED:
        freqs_t = freqs[0].clone()  # [seq, 32]
        for d, offset in enumerate((1, 2), start=1):  # H, W
            length = MROPE_SECTION[d] * 3
            idx = slice(offset, length, 3)
            freqs_t[:, idx] = freqs[d][:, idx]
        freqs_use = freqs_t
    else:
        freqs_use = freqs[0]
    emb = torch.cat((freqs_use, freqs_use), dim=-1)  # [seq, 64]
    cos = emb.cos()[None]  # [1, seq, 64]
    sin = emb.sin()[None]
    return cos.to(DTYPE), sin.to(DTYPE)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """q,k: [b, heads, seq, head_dim]. cos/sin: [1, seq, ROTARY_DIM].
    Partial rotary: rotate first ROTARY_DIM dims, pass through the rest."""
    cos = cos.unsqueeze(1)  # [1,1,seq,rot]
    sin = sin.unsqueeze(1)
    rot = cos.shape[-1]
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    k_rot, k_pass = k[..., :rot], k[..., rot:]
    q_emb = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_emb = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_emb, q_pass], dim=-1), torch.cat([k_emb, k_pass], dim=-1)


# --------------------------------------------------------------------------- #
# Full attention layer
# --------------------------------------------------------------------------- #
def full_attention(hidden, w, layer_idx, cos, sin):
    """hidden: [b, seq, HIDDEN]. Returns [b, seq, HIDDEN]."""
    b, seq, _ = hidden.shape
    q_proj = w.get(L(layer_idx, "self_attn.q_proj.weight"))   # [12288, 5120]
    k_proj = w.get(L(layer_idx, "self_attn.k_proj.weight"))   # [1024, 5120]
    v_proj = w.get(L(layer_idx, "self_attn.v_proj.weight"))
    o_proj = w.get(L(layer_idx, "self_attn.o_proj.weight"))   # [5120, 6144]
    q_norm_w = w.get(L(layer_idx, "self_attn.q_norm.weight"))  # [256]
    k_norm_w = w.get(L(layer_idx, "self_attn.k_norm.weight"))

    q = F.linear(hidden, q_proj)  # [b, seq, N_HEADS*HEAD_DIM*2]
    q = q.view(b, seq, N_HEADS, HEAD_DIM * 2)
    query, gate = torch.chunk(q, 2, dim=-1)  # each [b, seq, N_HEADS, HEAD_DIM]
    gate = gate.reshape(b, seq, -1)          # [b, seq, N_HEADS*HEAD_DIM]

    query = rmsnorm(query, q_norm_w, plus1=QK_NORM_PLUS1)  # q_norm over head_dim
    query = query.transpose(1, 2)              # [b, N_HEADS, seq, HEAD_DIM]

    key = F.linear(hidden, k_proj).view(b, seq, N_KV_HEADS, HEAD_DIM)
    key = rmsnorm(key, k_norm_w, plus1=QK_NORM_PLUS1).transpose(1, 2)  # [b, N_KV, seq, HEAD_DIM]
    value = F.linear(hidden, v_proj).view(b, seq, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

    query, key = apply_rope(query, key, cos, sin)

    # GQA repeat
    n_rep = N_HEADS // N_KV_HEADS
    key = key.repeat_interleave(n_rep, dim=1)
    value = value.repeat_interleave(n_rep, dim=1)

    attn = torch.matmul(query, key.transpose(2, 3)) * ATTN_SCALE  # [b, H, seq, seq]
    causal = torch.full((seq, seq), float("-inf")).triu(1)
    attn = attn + causal[None, None]
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn, value)            # [b, H, seq, HEAD_DIM]
    out = out.transpose(1, 2).reshape(b, seq, -1)  # [b, seq, N_HEADS*HEAD_DIM]

    # output gate: config says output_gate_type="swish" but the real weights want
    # SIGMOID (verified empirically: silu produces gibberish, sigmoid -> " Paris").
    if GATE_MODE == "sigmoid":
        out = out * torch.sigmoid(gate)
    else:
        out = out * F.silu(gate)

    return F.linear(out, o_proj)


# --------------------------------------------------------------------------- #
# Linear attention (gated DeltaNet) -- recurrent reference (exact)
# --------------------------------------------------------------------------- #
def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gated_delta_rule(query, key, value, g, beta):
    """Adapted from torch_recurrent_gated_delta_rule (qwen3_next).
    query,key: [b, seq, n_v_heads, head_k_dim]; value: [b, seq, n_v_heads, head_v_dim]
    g, beta: [b, seq, n_v_heads]. l2-norm q,k applied here (use_qk_l2norm_in_kernel=True).
    Returns core_attn_out [b, seq, n_v_heads, head_v_dim]."""
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    # -> [b, heads, seq, dim]
    q = query.transpose(1, 2).contiguous().float()
    k = key.transpose(1, 2).contiguous().float()
    v = value.transpose(1, 2).contiguous().float()
    bb = beta.transpose(1, 2).contiguous().float()  # [b, heads, seq]
    gg = g.transpose(1, 2).contiguous().float()      # [b, heads, seq]

    bsz, nh, slen, k_dim = k.shape
    v_dim = v.shape[-1]
    scale = 1.0 / (k_dim ** 0.5)
    q = q * scale

    out = torch.zeros(bsz, nh, slen, v_dim, dtype=torch.float32)
    state = torch.zeros(bsz, nh, k_dim, v_dim, dtype=torch.float32)
    for i in range(slen):
        q_t = q[:, :, i]                          # [b, h, k_dim]
        k_t = k[:, :, i]
        v_t = v[:, :, i]                          # [b, h, v_dim]
        g_t = gg[:, :, i].exp()[..., None, None]  # [b, h, 1, 1]
        beta_t = bb[:, :, i][..., None]           # [b, h, 1]
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)   # [b, h, v_dim]
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2).contiguous()  # [b, seq, heads, v_dim]


def linear_attention(hidden, w, layer_idx):
    b, seq, _ = hidden.shape
    in_qkv = w.get(L(layer_idx, "linear_attn.in_proj_qkv.weight"))  # [10240, 5120]
    in_z = w.get(L(layer_idx, "linear_attn.in_proj_z.weight"))      # [6144, 5120]
    in_b = w.get(L(layer_idx, "linear_attn.in_proj_b.weight"))      # [48, 5120]
    in_a = w.get(L(layer_idx, "linear_attn.in_proj_a.weight"))      # [48, 5120]
    conv_w = w.get(L(layer_idx, "linear_attn.conv1d.weight"))       # [10240, 1, 4]
    A_log = w.get(L(layer_idx, "linear_attn.A_log"))                # [48]
    dt_bias = w.get(L(layer_idx, "linear_attn.dt_bias"))            # [48]
    norm_w = w.get(L(layer_idx, "linear_attn.norm.weight"))         # [128]
    out_proj = w.get(L(layer_idx, "linear_attn.out_proj.weight"))   # [5120, 6144]

    mixed_qkv = F.linear(hidden, in_qkv)  # [b, seq, 10240]  (q | k | v already in order)
    z = F.linear(hidden, in_z)            # [b, seq, 6144]
    b_proj = F.linear(hidden, in_b)       # [b, seq, 48]
    a_proj = F.linear(hidden, in_a)       # [b, seq, 48]

    # causal depthwise conv1d over [q|k|v] then silu
    x = mixed_qkv.transpose(1, 2)  # [b, 10240, seq]
    x = F.conv1d(x, conv_w, bias=None, padding=LIN_CONV_K - 1, groups=CONV_DIM)
    x = x[:, :, :seq]
    x = F.silu(x)
    mixed_qkv = x.transpose(1, 2)  # [b, seq, 10240]

    query, key, value = torch.split(mixed_qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    query = query.reshape(b, seq, LIN_N_K_HEADS, LIN_K_HEAD_DIM)
    key = key.reshape(b, seq, LIN_N_K_HEADS, LIN_K_HEAD_DIM)
    value = value.reshape(b, seq, LIN_N_V_HEADS, LIN_V_HEAD_DIM)

    beta = b_proj.sigmoid()  # [b, seq, 48]
    g = -A_log.float().exp() * F.softplus(a_proj.float() + dt_bias)  # [b, seq, 48]

    # expand k-heads to v-heads
    query = query.repeat_interleave(V_PER_K, dim=2)  # [b, seq, 48, 128]
    key = key.repeat_interleave(V_PER_K, dim=2)

    core = recurrent_gated_delta_rule(query, key, value, g, beta)  # [b, seq, 48, 128]

    # gated RMSNorm with z (direct weight, gate via silu)
    z = z.reshape(b, seq, LIN_N_V_HEADS, LIN_V_HEAD_DIM)
    core = rmsnorm_gated(core, norm_w, z)   # [b, seq, 48, 128]
    core = core.reshape(b, seq, -1)         # [b, seq, 6144]

    return F.linear(core, out_proj)


# --------------------------------------------------------------------------- #
# Decoder layer + full forward
# --------------------------------------------------------------------------- #
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
    return hidden


def main():
    w = Weights(HF_DIR)

    input_ids = torch.tensor([[760, 6511, 314, 9338, 369]], dtype=torch.long)  # "The capital of France is"
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, dtype=torch.long)
    cos, sin = build_rope(position_ids)

    embed = w.get("model.language_model.embed_tokens.weight")  # [VOCAB, HIDDEN]
    hidden = F.embedding(input_ids, embed)  # [1, seq, HIDDEN]
    input_embeds = hidden.clone()

    layer_acts = torch.empty(N_LAYERS, 1, seq, HIDDEN, dtype=DTYPE)
    for li in range(N_LAYERS):
        hidden = decoder_layer(hidden, w, li, cos, sin)
        layer_acts[li] = hidden.detach()
        print(f"  layer {li:2d} ({LAYER_TYPES[li]:16s}) "
              f"mean={hidden.mean().item():.5f} std={hidden.std().item():.5f}")

    norm_w = w.get("model.language_model.norm.weight")
    final_norm = rmsnorm_w_plus_1(hidden, norm_w)

    lm_head = w.get("lm_head.weight")  # [VOCAB, HIDDEN]
    logits = F.linear(final_norm, lm_head)  # [1, seq, VOCAB]
    last_logits = logits[:, -1, :]

    # top-5
    topv, topi = torch.topk(last_logits[0], 5)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True)
        decoded = [tok.decode([int(i)]) for i in topi]
    except Exception as e:
        decoded = [f"<id {int(i)}>" for i in topi]
        print("tokenizer load failed:", e)

    print("\n=== TOP-5 NEXT TOKEN ===")
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
