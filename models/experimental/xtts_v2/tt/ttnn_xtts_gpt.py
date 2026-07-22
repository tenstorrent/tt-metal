# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN port of the XTTS-v2 GPT transformer core (Block 3), PREFILL path.

Ports the sub-computation validated in reference/xtts_gpt_ref.py:
    inputs_embeds [1, S, 1024]
      -> 30x GPT-2 block (LayerNorm -> causal MHA -> LayerNorm -> GELU MLP, residuals)
      -> ln_f (LayerNorm) -> final_norm (LayerNorm) = latents [1, S, 1024]

This is a first, single-device, functional port (correctness first; sharding/trace later).
Embeddings, positions, and the mel_head live on host (they are cheap and outside this block).

HF GPT-2 uses Conv1D weights stored as [in, out], which feed ttnn.linear directly (NO transpose),
unlike nn.Linear. GELU is the tanh approximation (HF "gelu_new").

Validate against the golden from reference/xtts_gpt_ref.py:
    PYTHONPATH=<repo> python models/experimental/xtts_v2/tt/ttnn_xtts_gpt.py
"""

import os

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_gpt_ref import (
    DEFAULT_CKPT,
    LN_EPS,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    load_gpt_core_state,
    pcc,
)

HEAD_DIM = N_EMBD // N_HEAD  # 64
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "gpt")

# Activation/weight dtype. bf16 with fp32-accumulate compute config; bump to fp32 tensors if PCC is low.
DTYPE = ttnn.bfloat16


def _to_dev(t, device, dtype=DTYPE):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def load_ttnn_weights(device, ckpt_path=DEFAULT_CKPT):
    """Convert the GPT-core checkpoint tensors to on-device ttnn tensors.

    HF Conv1D weights (c_attn/c_proj/c_fc) are [in, out] -> used as-is by ttnn.linear.
    LayerNorm weights/bias kept per layer, plus ln_f and the extra final_norm."""
    core = load_gpt_core_state(ckpt_path)  # keys: h.{i}.*, ln_f.*, final_norm.*
    f = lambda k: core[k].float()

    layers = []
    for i in range(N_LAYER):
        p = f"h.{i}."
        layers.append(
            {
                "ln_1_w": _to_dev(f(p + "ln_1.weight"), device),
                "ln_1_b": _to_dev(f(p + "ln_1.bias"), device),
                "attn_w": _to_dev(f(p + "attn.c_attn.weight"), device),  # [1024, 3072] (in,out)
                "attn_b": _to_dev(f(p + "attn.c_attn.bias"), device),
                "proj_w": _to_dev(f(p + "attn.c_proj.weight"), device),  # [1024, 1024]
                "proj_b": _to_dev(f(p + "attn.c_proj.bias"), device),
                "ln_2_w": _to_dev(f(p + "ln_2.weight"), device),
                "ln_2_b": _to_dev(f(p + "ln_2.bias"), device),
                "fc_w": _to_dev(f(p + "mlp.c_fc.weight"), device),  # [1024, 4096]
                "fc_b": _to_dev(f(p + "mlp.c_fc.bias"), device),
                "mproj_w": _to_dev(f(p + "mlp.c_proj.weight"), device),  # [4096, 1024]
                "mproj_b": _to_dev(f(p + "mlp.c_proj.bias"), device),
            }
        )
    tail = {
        "ln_f_w": _to_dev(f("ln_f.weight"), device),
        "ln_f_b": _to_dev(f("ln_f.bias"), device),
        "fn_w": _to_dev(f("final_norm.weight"), device),
        "fn_b": _to_dev(f("final_norm.bias"), device),
    }
    return layers, tail


def _attention(x, w, seq_len, causal_mask):
    """Causal multi-head self-attention. x: [1, S, 1024] -> [1, S, 1024]."""
    qkv = ttnn.linear(x, w["attn_w"], bias=w["attn_b"])  # [1, S, 3072]

    # split fused QKV on the last dim, then split heads -> [1, n_head, S, head_dim]
    q = ttnn.slice(qkv, [0, 0, 0], [1, seq_len, N_EMBD])
    k = ttnn.slice(qkv, [0, 0, N_EMBD], [1, seq_len, 2 * N_EMBD])
    v = ttnn.slice(qkv, [0, 0, 2 * N_EMBD], [1, seq_len, 3 * N_EMBD])

    def heads(t):
        t = ttnn.reshape(t, [1, seq_len, N_HEAD, HEAD_DIM])
        return ttnn.permute(t, [0, 2, 1, 3])  # [1, n_head, S, head_dim]

    q, k, v = heads(q), heads(k), heads(v)

    scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))  # [1, n_head, S, S]
    scores = ttnn.multiply(scores, 1.0 / (HEAD_DIM**0.5))
    scores = ttnn.add(scores, causal_mask)  # additive -inf above the diagonal
    attn = ttnn.softmax(scores, dim=-1)
    out = ttnn.matmul(attn, v)  # [1, n_head, S, head_dim]

    out = ttnn.permute(out, [0, 2, 1, 3])  # [1, S, n_head, head_dim]
    out = ttnn.reshape(out, [1, seq_len, N_EMBD])
    return ttnn.linear(out, w["proj_w"], bias=w["proj_b"])


def _mlp(x, w):
    x = ttnn.linear(x, w["fc_w"], bias=w["fc_b"])  # [1, S, 4096]
    x = ttnn.gelu(x)  # HF gelu_new (tanh approx) — verify approximate mode on device
    return ttnn.linear(x, w["mproj_w"], bias=w["mproj_b"])  # [1, S, 1024]


def run_prefill(device, inputs_embeds, weights=None, ckpt_path=DEFAULT_CKPT):
    """inputs_embeds: torch [1, S, 1024] -> latents: torch [1, S, 1024]."""
    if weights is None:
        weights = load_ttnn_weights(device, ckpt_path)
    layers, tail = weights
    seq_len = inputs_embeds.shape[1]

    # additive causal mask [1, 1, S, S]: 0 on/below diagonal, large-negative above.
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), -1e9)
    causal_mask = _to_dev(mask.reshape(1, 1, seq_len, seq_len), device)

    x = _to_dev(inputs_embeds, device)
    for w in layers:
        h = ttnn.layer_norm(x, weight=w["ln_1_w"], bias=w["ln_1_b"], epsilon=LN_EPS)
        x = ttnn.add(x, _attention(h, w, seq_len, causal_mask))
        h = ttnn.layer_norm(x, weight=w["ln_2_w"], bias=w["ln_2_b"], epsilon=LN_EPS)
        x = ttnn.add(x, _mlp(h, w))

    x = ttnn.layer_norm(x, weight=tail["ln_f_w"], bias=tail["ln_f_b"], epsilon=LN_EPS)
    x = ttnn.layer_norm(x, weight=tail["fn_w"], bias=tail["fn_b"], epsilon=LN_EPS)
    return ttnn.to_torch(x).float()


def main():
    inputs_embeds = torch.load(os.path.join(GOLDEN_DIR, "inputs_embeds.pt"))
    golden = torch.load(os.path.join(GOLDEN_DIR, "latents.pt"))

    device = ttnn.open_device(device_id=0)
    try:
        out = run_prefill(device, inputs_embeds)
    finally:
        ttnn.close_device(device)

    print(f"[ttnn] latents {tuple(out.shape)} vs golden {tuple(golden.shape)}")
    print(f"[ttnn] PCC vs reference latents = {pcc(out, golden):.6f}")


if __name__ == "__main__":
    main()
