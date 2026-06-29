# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic: compare TTNN's fused-QKV-projection-then-split output against
HF's own q_proj / k_proj / v_proj linear outputs, for decoder layer 0
self-attention.

Context: [1] self_attn PCC = 0.8294 (bfloat16, baseline).
The fp32-mask-softmax test showed PCC barely changed (0.8295) -- so the
masking/softmax step is NOT the cause. This test isolates the step before
it: the fused QKV projection (ttnn.linear with qkv_weight) followed by
ttnn.transformer.split_query_key_value_and_split_heads.

Method:
  1. Hook HF's self_attn.q_proj / k_proj / v_proj directly -- these are
     nn.Linear layers, so the hook captures the raw post-projection output,
     BEFORE HF reshapes into heads. Shape: [B, T, 26], two 13-wide heads
     packed contiguously (head0 = cols 0:13, head1 = cols 13:26).
  2. Run TTNN's fused QKV projection + split on the SAME input (dec_emb,
     padded to 64 wide, matching production). This yields per-head TTNN
     tensors shaped [B, NUM_HEADS, T, HEAD_DIM_PADDED=32], where each head's
     REAL data lives in columns 0:13 of its own 32-wide slot (the rest is
     zero padding from _pad_weight_per_head).
  3. Re-assemble TTNN's two padded 32-wide heads into one contiguous 26-wide
     tensor (head0 real cols 0:13, head1 real cols 13:26) so it directly
     matches HF's raw q_proj/k_proj/v_proj layout. Padding lanes are
     dropped, not compared (they are zero by construction on both sides --
     including them would inflate PCC artificially with trivial matches).
  4. PCC each of Q, K, V independently against HF's q_proj/k_proj/v_proj
     output.

If Q/K/V PCC are all near 1.0: the fused projection+split is fine, and the
error must be in the Q@K^T or attn@V matmuls (next thing to check).
If any of Q/K/V PCC is low: that specific projection (or the split/reshape
step) is the root cause.

Run (after copying into tests/):
PYTHONPATH=/root/tt-metal/ttnn:/root/tt-metal/tools:/root/tt-metal/build_Release/lib:. \
TT_METAL_HOME=/root/tt-metal \
LD_LIBRARY_PATH=/root/tt-metal/build_Release/lib \
ARCH_NAME=wormhole_b0 \
pytest tests/test_qkv_split_pcc.py -v -s --noconftest
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_attention import HEAD_DIM_PADDED, NUM_HEADS
from tt.tst_model import load_weights

import ttnn
from models.common.utility_functions import comp_pcc

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
D_MODEL = 26
HEAD_DIM_TRUE = D_MODEL // NUM_HEADS  # 13
PADDED_WIDTH = NUM_HEADS * HEAD_DIM_PADDED  # 64


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_hf_dec_emb_and_qkv(hf_model, inputs):
    """Capture decoder layernorm_embedding output AND layer-0 self-attn's
    raw q_proj/k_proj/v_proj outputs in a single forward pass."""
    captured = {}

    def dec_hook(m, i, o):
        captured["dec_emb"] = o.detach()

    def make_proj_hook(name):
        def fn(module, inp, out):
            captured[name] = out.detach()

        return fn

    self_attn = hf_model.model.decoder.layers[0].self_attn
    handles = [
        hf_model.model.decoder.layernorm_embedding.register_forward_hook(dec_hook),
        self_attn.q_proj.register_forward_hook(make_proj_hook("hf_q")),
        self_attn.k_proj.register_forward_hook(make_proj_hook("hf_k")),
        self_attn.v_proj.register_forward_hook(make_proj_hook("hf_v")),
    ]

    with torch.no_grad():
        hf_model(
            past_values=inputs["input_past_values"],
            past_time_features=inputs["input_past_time_features"],
            past_observed_mask=inputs["input_past_observed_mask"],
            future_time_features=inputs["input_future_time_features"],
            future_values=inputs["input_future_values"],
            static_categorical_features=inputs["input_static_categorical_features"].long(),
            static_real_features=inputs["input_static_real_features"],
        )
    for h in handles:
        h.remove()

    return captured["dec_emb"], captured["hf_q"], captured["hf_k"], captured["hf_v"]


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    yield device, hf_model, weights, inputs
    ttnn.close_device(device)


def _unpad_heads_to_contiguous(torch_tensor_padded, num_heads, head_dim_padded, head_dim_true, key_is_transposed=False):
    """
    torch_tensor_padded: per-head TTNN layout, each head individually padded
    from head_dim_true to head_dim_padded with zeros.

    IMPORTANT: ttnn.transformer.split_query_key_value_and_split_heads defaults
    to transpose_key=True (confirmed in docs), meaning K may come back as
    [B, H, head_dim_padded, T] instead of [B, H, T, head_dim_padded] like Q
    and V. We do NOT assume which -- caller must tell us via key_is_transposed,
    set based on an explicit shape check, not a guess.

    Returns: [B, T, num_heads * head_dim_true] -- heads' REAL columns only,
    concatenated contiguously, matching HF's raw q_proj/k_proj/v_proj layout.
    """
    if key_is_transposed:
        # [B, H, head_dim_padded, T] -> [B, H, T, head_dim_padded]
        torch_tensor_padded = torch_tensor_padded.permute(0, 1, 3, 2)

    B, H, T, _ = torch_tensor_padded.shape
    x = torch_tensor_padded.permute(0, 2, 1, 3)  # -> [B, T, H, head_dim_padded]
    real = x[..., :head_dim_true]  # [B, T, H, head_dim_true]
    return real.reshape(B, T, H * head_dim_true)


def test_qkv_projection_split_pcc(setup):
    """Compare TTNN fused-QKV-then-split Q/K/V against HF's raw q/k/v_proj outputs."""
    device, hf_model, weights, inputs = setup

    dec_emb, hf_q, hf_k, hf_v = get_hf_dec_emb_and_qkv(hf_model, inputs)
    logger.info(f"dec_emb shape: {dec_emb.shape}, hf_q shape: {hf_q.shape}")

    # Build TTNN input exactly as production does
    h = dec_emb.float()
    pad = PADDED_WIDTH - h.shape[-1]
    if pad > 0:
        h = F.pad(h, (0, pad))
    hidden_states = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    w0 = weights["decoder.layers.0"]["self_attn"]

    # Run the EXACT same fused QKV + split as tst_self_attention does
    fused_qkv = ttnn.linear(hidden_states, w0["qkv_weight"], bias=w0["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)
    # query/key/value: [B, NUM_HEADS, T, HEAD_DIM_PADDED]

    query_torch = ttnn.to_torch(query).float()
    key_torch = ttnn.to_torch(key).float()
    value_torch = ttnn.to_torch(value).float()

    logger.info(f"TTNN query raw shape: {query_torch.shape}")
    logger.info(f"TTNN key   raw shape: {key_torch.shape}")
    logger.info(f"TTNN value raw shape: {value_torch.shape}")

    # Detect K's layout explicitly rather than assuming transpose_key=True
    # actually applies here. Q and V should both be [B, H, T, head_dim_padded].
    # If K's last two dims are swapped relative to Q, it's pre-transposed.
    key_is_transposed = key_torch.shape[-2:] != query_torch.shape[-2:]
    logger.info(
        f"Detected key_is_transposed={key_is_transposed} "
        f"(query last-2-dims={query_torch.shape[-2:]}, key last-2-dims={key_torch.shape[-2:]})"
    )

    q_unpadded = _unpad_heads_to_contiguous(query_torch, NUM_HEADS, HEAD_DIM_PADDED, HEAD_DIM_TRUE)
    k_unpadded = _unpad_heads_to_contiguous(
        key_torch, NUM_HEADS, HEAD_DIM_PADDED, HEAD_DIM_TRUE, key_is_transposed=key_is_transposed
    )
    v_unpadded = _unpad_heads_to_contiguous(value_torch, NUM_HEADS, HEAD_DIM_PADDED, HEAD_DIM_TRUE)

    logger.info(f"Unpadded TTNN query shape: {q_unpadded.shape} vs HF q shape: {hf_q.shape}")

    assert q_unpadded.shape == hf_q.shape, f"Q shape mismatch: {q_unpadded.shape} vs {hf_q.shape}"
    assert k_unpadded.shape == hf_k.shape, f"K shape mismatch: {k_unpadded.shape} vs {hf_k.shape}"
    assert v_unpadded.shape == hf_v.shape, f"V shape mismatch: {v_unpadded.shape} vs {hf_v.shape}"

    passing_q, pcc_q = comp_pcc(q_unpadded, hf_q, 0.99)
    passing_k, pcc_k = comp_pcc(k_unpadded, hf_k, 0.99)
    passing_v, pcc_v = comp_pcc(v_unpadded, hf_v, 0.99)

    logger.info(f"[Q] PCC: {pcc_q} (passing={passing_q})")
    logger.info(f"[K] PCC: {pcc_k} (passing={passing_k})")
    logger.info(f"[V] PCC: {pcc_v} (passing={passing_v})")
    logger.info(
        "If all three are near 1.0, the fused QKV projection+split is fine -- "
        "look at Q@K^T / attn@V matmuls next. If any is low, that projection "
        "(or the split/reshape) is the root cause."
    )
