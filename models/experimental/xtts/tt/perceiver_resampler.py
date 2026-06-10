"""TTNN port of the XTTS PerceiverResampler (Phase 2).

Reference: TTS/tts/layers/xtts/perceiver_encoder.py. Resamples the variable-length
conditioning sequence (b, S, 1024) down to a fixed (b, 32, 1024) set of latents.

    latents = self.latents                       # learned [32, 1024]
    for attn, ff in layers:                       # 2 layers
        latents = attn(latents, context) + latents   # cross-attention
        latents = ff(latents) + latents              # GEGLU feed-forward
    return RMSNorm(latents)

Details matched here:
  * Attention: to_q/to_kv/to_out are nn.Linear(bias=False), 8 heads x 64 dim;
    scale = 1/sqrt(64) (ttnn SDPA default), non-causal cross-attention.
  * FeedForward: Linear -> GEGLU (x * gelu(gates)) -> Linear; exact (erf) GELU.
  * RMSNorm: F.normalize(x)*sqrt(dim)*gamma == (x/RMS)*gamma == ttnn.rms_norm.
"""

import torch
import ttnn

DIM = 1024
HEADS = 8
HEAD_DIM = 64
INNER = HEADS * HEAD_DIM  # 512
RMS_EPS = 1e-6


def _lin(t, device, *, transpose=False):
    if transpose:
        t = t.t().contiguous()
    return ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def load_perceiver_params(sd, device, prefix="gpt.conditioning_perceiver."):
    g = lambda k: sd[prefix + k]
    n_layers = 1 + max(int(k[len(prefix + "layers.") :].split(".")[0]) for k in sd if k.startswith(prefix + "layers."))
    layers = []
    for i in range(n_layers):
        lp = f"layers.{i}."
        layers.append(
            {
                # Attention (nn.Linear bias=False -> transpose to [in,out])
                "to_q": _lin(g(lp + "0.to_q.weight"), device, transpose=True),
                "to_kv": _lin(g(lp + "0.to_kv.weight"), device, transpose=True),
                "to_out": _lin(g(lp + "0.to_out.weight"), device, transpose=True),
                # FeedForward Sequential: 0 = Linear(dim, inner*2), GEGLU, 2 = Linear(inner, dim)
                "ff0_w": _lin(g(lp + "1.0.weight"), device, transpose=True),
                "ff0_b": _lin(g(lp + "1.0.bias"), device),
                "ff3_w": _lin(g(lp + "1.2.weight"), device, transpose=True),
                "ff3_b": _lin(g(lp + "1.2.bias"), device),
            }
        )
    latents = g("latents").unsqueeze(0)  # [1, 32, 1024]
    return {
        "latents": ttnn.from_torch(
            latents.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
        ),
        "layers": layers,
        "norm_gamma": _lin(g("norm.gamma"), device),
    }


def _split_heads(t):  # [1, n, INNER] -> [1, HEADS, n, HEAD_DIM]
    b, n, _ = t.shape
    return ttnn.permute(ttnn.reshape(t, (b, n, HEADS, HEAD_DIM)), (0, 2, 1, 3))


def _merge_heads(t):  # [1, HEADS, n, HEAD_DIM] -> [1, n, INNER]
    b, _, n, _ = t.shape
    return ttnn.reshape(ttnn.permute(t, (0, 2, 1, 3)), (b, n, INNER))


def _attention(latents, context, p):
    q = _split_heads(ttnn.linear(latents, p["to_q"]))
    kv = ttnn.linear(context, p["to_kv"])
    k = _split_heads(ttnn.slice(kv, [0, 0, 0], [kv.shape[0], kv.shape[1], INNER]))
    v = _split_heads(ttnn.slice(kv, [0, 0, INNER], [kv.shape[0], kv.shape[1], 2 * INNER]))
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
    return ttnn.linear(_merge_heads(attn), p["to_out"])


def _feedforward(x, p):
    h = ttnn.linear(x, p["ff0_w"], bias=p["ff0_b"])
    inner = h.shape[-1] // 2
    a = ttnn.slice(h, [0, 0, 0], [h.shape[0], h.shape[1], inner])
    gates = ttnn.slice(h, [0, 0, inner], [h.shape[0], h.shape[1], 2 * inner])
    h = ttnn.mul(a, ttnn.gelu(gates))  # GEGLU (exact GELU)
    return ttnn.linear(h, p["ff3_w"], bias=p["ff3_b"])


def perceiver_resampler(context, p):
    """context: [1, S, 1024] -> latents [1, 32, 1024]."""
    latents = p["latents"]
    for lp in p["layers"]:
        latents = ttnn.add(latents, _attention(latents, context, lp))
        latents = ttnn.add(latents, _feedforward(latents, lp))
    return ttnn.rms_norm(latents, weight=p["norm_gamma"], epsilon=RMS_EPS)
