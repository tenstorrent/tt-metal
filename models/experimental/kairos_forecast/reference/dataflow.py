# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU fp32 dataflow of the Kairos forward, in the decomposition the TTNN port uses (pure torch, no tsfm import).

The routing, patch division and x_final gather stay on the host (they come in as `front`); this file covers what
goes to the device: the packed embedding, the encoder and decoder stacks with instance-wise RoPE, the head, and the
host epilogue (de-normalize, flip average, clamp). Checked against the reference to 1e-5 relative on the test goldens
(tests/pcc/test_dataflow.py).
"""
import torch

D_MODEL, N_HEADS, D_KV, N_LAYERS, N_SEG, N_Q, H_SEG = 512, 8, 64, 6, 2, 9, 64
N_REAL = 3
EPS = 1e-6
NEG = torch.finfo(torch.float32).min


def load_weights(path=None):
    from safetensors.torch import load_file

    if path is None:
        import glob

        path = glob.glob(
            "/home/andy/.cache/huggingface/hub/models--mldi-lab--Kairos_50m/snapshots/"
            "b788434dd95f9893be002e6858bfb2afb0e3f360/model.safetensors"
        )[0]
    return {k: v.float() for k, v in load_file(path).items()}


def rms_norm(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


def rope_freqs(W, pre, s_features):
    """Per-series frequencies [B, 32] of one attention module (InstanceWiseParamNet + log-period scaling)."""
    p = f"{pre}.rotary_emb.instance_wise_param_net.mlp"
    h = torch.relu(s_features @ W[f"{p}.0.weight"].T + W[f"{p}.0.bias"])
    h = torch.relu(h @ W[f"{p}.2.weight"].T + W[f"{p}.2.bias"])
    params = h @ W[f"{p}.4.weight"].T + W[f"{p}.4.bias"]
    gamma, beta = params[:, : D_KV // 2], params[:, D_KV // 2 :]
    return torch.exp(gamma * W[f"{pre}.rotary_emb.log_freqs"] + beta)


def rope(x, freqs):
    """x [B, H, L, 64]; adjacent pairs (2i, 2i+1) rotate by angle pos * freqs[i], positions 0..L-1."""
    L = x.shape[-2]
    ang = torch.arange(L, dtype=torch.float32)[None, :, None] * freqs[:, None, :]  # [B, L, 32]
    cos, sin = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


def attention(W, pre, h, kv, mask_add, s_features):
    B, Lq, _ = h.shape
    heads = lambda t: t.view(B, -1, N_HEADS, D_KV).transpose(1, 2)  # noqa: E731
    q = heads(h @ W[f"{pre}.q.weight"].T)
    k = heads(kv @ W[f"{pre}.k.weight"].T)
    v = heads(kv @ W[f"{pre}.v.weight"].T)
    f = rope_freqs(W, pre, s_features)
    q, k = rope(q, f), rope(k, f)
    s = q @ k.transpose(-1, -2) + mask_add  # T5: no 1/sqrt(d) scaling
    o = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, Lq, N_HEADS * D_KV)
    return o @ W[f"{pre}.o.weight"].T


def ff(W, pre, h):
    return torch.relu(h @ W[f"{pre}.wi.weight"].T) @ W[f"{pre}.wo.weight"].T


def expert_coeffs(weights, indices):
    """[B, N, 3] coefficient of each real expert, as MultiInSizeLinear forms it (padded slots included)."""
    real = (indices < N_REAL).float()
    rs = (weights * real).sum(-1, keepdim=True)
    rs = torch.where(rs == 0, torch.ones_like(rs), rs)
    return torch.stack([((indices == e).float() * weights).sum(-1) for e in range(N_REAL)], -1) / rs


def embed(W, x_final, weights, indices):
    """Packed form: one [B, N, 3*256] x [3*256, out] matmul per MultiInSizeLinear (CONTRACT §7).

    x_final [3, B, N, 256]. The 128/64-wide experts' unused input columns are masked to zero."""
    c = expert_coeffs(weights, indices)  # [B, N, 3]
    xs = torch.cat([x_final[e].nan_to_num(0.0) * c[..., e : e + 1] for e in range(N_REAL)], -1)

    def multi(name):
        w, b = W[f"input_patch_embedding.{name}.weight"], W[f"input_patch_embedding.{name}.bias"]
        cols = torch.arange(256)
        keep = [((cols % 128) < s).float() for s in (128, 64, 32)]
        wp = torch.cat([w[e] * keep[e] for e in range(N_REAL)], -1)  # [out, 768]
        return xs @ wp.T + c @ b

    hid = torch.relu(multi("hidden_layer"))
    p = "input_patch_embedding.output_layer"
    return hid @ W[f"{p}.weight"].T + W[f"{p}.bias"] + multi("residual_layer")


def encoder(W, emb, attn_mask, s_features):
    B = emb.shape[0]
    h = torch.cat([emb, W["shared.weight"][1].expand(B, 1, D_MODEL)], 1)
    m = torch.cat([attn_mask.float(), torch.ones(B, 1)], 1)
    mask_add = ((1.0 - m) * NEG)[:, None, None, :]
    for i in range(N_LAYERS):
        p = f"encoder.block.{i}.layer"
        n = rms_norm(h, W[f"{p}.0.layer_norm.weight"])
        h = h + attention(W, f"{p}.0.SelfAttention", n, n, mask_add, s_features)
        h = h + ff(W, f"{p}.1.DenseReluDense", rms_norm(h, W[f"{p}.1.layer_norm.weight"]))
    return rms_norm(h, W["encoder.final_layer_norm.weight"]), mask_add


def decoder(W, enc, enc_mask_add, s_features):
    B = enc.shape[0]
    h = W["shared.weight"][0].expand(B, N_SEG, D_MODEL).clone()
    causal = torch.triu(torch.full((N_SEG, N_SEG), NEG), 1)[None, None]
    for i in range(N_LAYERS):
        p = f"decoder.block.{i}.layer"
        n = rms_norm(h, W[f"{p}.0.layer_norm.weight"])
        h = h + attention(W, f"{p}.0.SelfAttention", n, n, causal, s_features)
        n = rms_norm(h, W[f"{p}.1.layer_norm.weight"])
        h = h + attention(W, f"{p}.1.EncDecAttention", n, enc, enc_mask_add, s_features)
        h = h + ff(W, f"{p}.2.DenseReluDense", rms_norm(h, W[f"{p}.2.layer_norm.weight"]))
    return rms_norm(h, W["decoder.final_layer_norm.weight"])


def head(W, h):
    p = "output_patch_embedding"
    hid = torch.relu(h @ W[f"{p}.hidden_layer.weight"].T + W[f"{p}.hidden_layer.bias"])
    out = hid @ W[f"{p}.output_layer.weight"].T + W[f"{p}.output_layer.bias"]
    return out + h @ W[f"{p}.residual_layer.weight"].T + W[f"{p}.residual_layer.bias"]


def forward(W, front):
    """front: s_features [B, 128], x_final [3, B, N, 256], weights / indices [B, N, 3], attn_mask [B, N].
    Returns the head output [B, 2, 576] (the pcc_normalized space)."""
    emb = embed(W, front["x_final"], front["weights"], front["indices"])
    enc, mask_add = encoder(W, emb, front["attn_mask"], front["s_features"])
    return head(W, decoder(W, enc, mask_add, front["s_features"]))


def head_to_quantiles(raw, loc, scale):
    """[B, 2, 576] -> data-space [B, 9, 128]."""
    B = raw.shape[0]
    q = raw.view(B, N_SEG, N_Q, H_SEG).permute(0, 2, 1, 3).reshape(B, N_Q, N_SEG * H_SEG)
    return (q.reshape(B, -1) * scale + loc).view(B, N_Q, -1)
