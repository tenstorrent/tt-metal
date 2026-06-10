"""TTNN port of the XTTS ConditioningEncoder (Phase 2).

Reference: TTS/tts/layers/tortoise/{autoregressive,arch_utils}.py. Encodes the
conditioning mel spectrogram (b, 80, S) into (b, 1024, S):

    h = init(x)                       # Conv1d(80, 1024, k=1)
    for block in 6 x AttentionBlock:  # h <- block(h)
        x_norm = GroupNorm32(h)       # 32 groups over 1024 channels
        qkv    = qkv_conv(x_norm)     # Conv1d(1024, 3072, k=1)
        a      = QKVAttentionLegacy(qkv)   # non-causal MHA, 16 heads x 64
        h      = x_norm + proj_out(a) # tortoise_norm=False -> residual on x_norm

All work is done in [1, S, C] layout (k=1 convs == linear over channels). The
QKV conv output is reordered at load time from QKVAttentionLegacy's head-major
[h0:qkv | h1:qkv | ...] channel layout to [all-q | all-k | all-v] so standard
head-splitting + ttnn SDPA apply. GroupNorm is computed manually (group stats via
last-dim reductions) to avoid the sharded ttnn.group_norm kernel's grid setup.
"""

import torch
import ttnn

SPEC = 80
DIM = 1024
HEADS = 16
HEAD_DIM = 64  # 1024 // (3*16) per QKVAttentionLegacy
GROUPS = 32
CH_PER_GROUP = DIM // GROUPS  # 32
GN_EPS = 1e-5


def _lin(t, device):
    return ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _conv1d_k1_to_linear(weight, device):
    """Conv1d(out,in,1) weight -> ttnn.linear weight [in, out]."""
    return _lin(weight.squeeze(-1).t().contiguous(), device)


def _reorder_qkv(weight, bias):
    """Reorder qkv Conv1d out-channels from QKVAttentionLegacy head-major
    [head h: q(64) k(64) v(64)] to type-major [all q | all k | all v]."""
    w = weight.squeeze(-1)  # [3072, 1024]
    idx = torch.empty(3 * DIM, dtype=torch.long)
    for h in range(HEADS):
        for d in range(HEAD_DIM):
            old = h * (3 * HEAD_DIM) + d  # q within head h
            idx[h * HEAD_DIM + d] = old
            idx[DIM + h * HEAD_DIM + d] = old + HEAD_DIM  # k
            idx[2 * DIM + h * HEAD_DIM + d] = old + 2 * HEAD_DIM  # v
    return w[idx].contiguous(), bias[idx].contiguous()


def _compute_config(device):
    # HiFi4 + fp32 accumulation: full (non-causal) attention over large-magnitude
    # activations compounds bf16 error across the 6 blocks otherwise.
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def load_encoder_params(sd, device, prefix="gpt.conditioning_encoder."):
    g = lambda k: sd[prefix + k]
    n_blocks = 1 + max(int(k[len(prefix + "attn.") :].split(".")[0]) for k in sd if k.startswith(prefix + "attn."))
    blocks = []
    for i in range(n_blocks):
        bp = f"attn.{i}."
        qkv_w, qkv_b = _reorder_qkv(g(bp + "qkv.weight"), g(bp + "qkv.bias"))
        blocks.append(
            {
                "norm_w": _lin(g(bp + "norm.weight"), device),
                "norm_b": _lin(g(bp + "norm.bias"), device),
                "qkv_w": _lin(qkv_w.t().contiguous(), device),  # [1024, 3072]
                "qkv_b": _lin(qkv_b, device),
                "proj_w": _conv1d_k1_to_linear(g(bp + "proj_out.weight"), device),
                "proj_b": _lin(g(bp + "proj_out.bias"), device),
            }
        )
    return {
        "init_w": _conv1d_k1_to_linear(g("init.weight"), device),  # [80, 1024]
        "init_b": _lin(g("init.bias"), device),
        "blocks": blocks,
        "ckc": _compute_config(device),
    }


def _group_norm(x, weight, bias):
    """GroupNorm32 over [1, S, 1024]: 32 groups of 32 consecutive channels,
    stats pooled over (channels-in-group x S)."""
    b, s, _ = x.shape
    # reference casts to float32 for the stats; do likewise (bf16 sums over
    # 32*S elements lose too much precision and compound across blocks)
    xf = ttnn.typecast(x, ttnn.float32)
    x4 = ttnn.reshape(xf, (b, s, GROUPS, CH_PER_GROUP))  # [1,S,32,32]
    n = s * CH_PER_GROUP
    sum_g = ttnn.sum(ttnn.sum(x4, dim=3), dim=1, keepdim=True)  # [1,1,32]
    sq_g = ttnn.sum(ttnn.sum(ttnn.mul(x4, x4), dim=3), dim=1, keepdim=True)
    mean = ttnn.multiply(sum_g, 1.0 / n)  # [1,1,32]
    var = ttnn.sub(ttnn.multiply(sq_g, 1.0 / n), ttnn.mul(mean, mean))
    inv = ttnn.rsqrt(ttnn.add(var, GN_EPS))
    mean4 = ttnn.reshape(mean, (b, 1, GROUPS, 1))
    inv4 = ttnn.reshape(inv, (b, 1, GROUPS, 1))
    x4n = ttnn.mul(ttnn.sub(x4, mean4), inv4)
    xn = ttnn.typecast(ttnn.reshape(x4n, (b, s, DIM)), ttnn.bfloat16)
    return ttnn.add(ttnn.mul(xn, weight), bias)  # per-channel affine


def _split_heads(t):
    b, n, _ = t.shape
    return ttnn.permute(ttnn.reshape(t, (b, n, HEADS, HEAD_DIM)), (0, 2, 1, 3))


def _merge_heads(t):
    b, _, n, _ = t.shape
    return ttnn.reshape(ttnn.permute(t, (0, 2, 1, 3)), (b, n, DIM))


def _attention_block(x, p, ckc):
    x_norm = _group_norm(x, p["norm_w"], p["norm_b"])
    qkv = ttnn.linear(x_norm, p["qkv_w"], bias=p["qkv_b"], compute_kernel_config=ckc)  # [1,S,3072]
    q = _split_heads(ttnn.slice(qkv, [0, 0, 0], [qkv.shape[0], qkv.shape[1], DIM]))
    k = _split_heads(ttnn.slice(qkv, [0, 0, DIM], [qkv.shape[0], qkv.shape[1], 2 * DIM]))
    v = _split_heads(ttnn.slice(qkv, [0, 0, 2 * DIM], [qkv.shape[0], qkv.shape[1], 3 * DIM]))
    a = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, compute_kernel_config=ckc)
    a = ttnn.linear(_merge_heads(a), p["proj_w"], bias=p["proj_b"], compute_kernel_config=ckc)
    return ttnn.add(x_norm, a)  # tortoise_norm=False


def conditioning_encoder(x, p):
    """x: mel [1, 80, S] -> conds [1, 1024, S] (returned in [1, S, 1024])."""
    # input arrives [1, 80, S]; move to [1, S, 80] for channel-wise linear
    h = ttnn.permute(x, (0, 2, 1))
    h = ttnn.linear(h, p["init_w"], bias=p["init_b"], compute_kernel_config=p["ckc"])  # [1, S, 1024]
    for bp in p["blocks"]:
        h = _attention_block(h, bp, p["ckc"])
    return h
