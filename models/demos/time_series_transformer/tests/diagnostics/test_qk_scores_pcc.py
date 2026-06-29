# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic: PCC of raw (unscaled, pre-mask, pre-softmax) Q@K^T scores,
TTNN vs hand-computed torch reference.

Context so far (all measured, not theorized):
  [1] self_attn (full, bfloat16)          PCC = 0.8294
  [1] self_attn (fp32 mask+softmax only)  PCC = 0.8295  -> masking/softmax cleared
  Q projection+split                      PCC = 0.99999 -> cleared
  K projection+split                      PCC = 0.99999 -> cleared
  V projection+split                      PCC = 0.99985 -> cleared

Remaining candidates: Q@K^T matmul, the scale factor, attn@V matmul,
concatenate_heads, out_proj. This script isolates Q@K^T specifically,
BEFORE any scaling, masking, or softmax -- and separately reports what
scale factor production code actually uses vs. what HF uses, since a
scale mismatch would be a systematic bug, not noise.

Method:
  1. Re-derive TTNN's already-verified, unpadded Q and K (same extraction
     as test_qkv_split_pcc.py) -- shape [B, T, D_MODEL=26], HF-layout.
  2. Reshape into per-head form using HF's OWN convention (view + transpose,
     not TTNN's padded-then-permuted convention) to build a pure-torch
     reference: ref_scores = Q_hf_shape @ K_hf_shape^T, per head, UNSCALED.
  3. Run ttnn.matmul(query, key) directly on TTNN's own padded/permuted
     Q, K (production's actual call), unpad the result's real 13x13 region
     per head (padding columns/rows are not meaningful and must be excluded),
     and compare against the same unscaled per-head reference.
  4. Print what scale factor production applies (HEAD_DIM_PADDED**-0.5 = 32**-0.5)
     vs. what HF would apply (HEAD_DIM_TRUE**-0.5 = 13**-0.5) -- NOT inferred,
     just both numbers shown so a mismatch is obvious if present.

Run (after copying into tests/):
PYTHONPATH=/root/tt-metal/ttnn:/root/tt-metal/tools:/root/tt-metal/build_Release/lib:. \
TT_METAL_HOME=/root/tt-metal \
LD_LIBRARY_PATH=/root/tt-metal/build_Release/lib \
ARCH_NAME=wormhole_b0 \
pytest tests/test_qk_scores_pcc.py -v -s --noconftest
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


def get_hf_dec_emb_and_qk(hf_model, inputs):
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
    return captured["dec_emb"], captured["hf_q"], captured["hf_k"]


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    yield device, hf_model, weights, inputs
    ttnn.close_device(device)


def test_qk_scores_pcc(setup):
    """Compare TTNN's unscaled Q@K^T against a hand-built torch reference."""
    device, hf_model, weights, inputs = setup

    dec_emb, hf_q, hf_k = get_hf_dec_emb_and_qk(hf_model, inputs)
    B, T, _ = hf_q.shape
    logger.info(f"hf_q shape: {hf_q.shape}")

    # ---- Build pure-torch reference scores (HF's own head convention) ----
    # HF: [B, T, D_MODEL] -> [B, T, NUM_HEADS, HEAD_DIM_TRUE] -> [B, NUM_HEADS, T, HEAD_DIM_TRUE]
    hf_q_heads = hf_q.view(B, T, NUM_HEADS, HEAD_DIM_TRUE).transpose(1, 2)
    hf_k_heads = hf_k.view(B, T, NUM_HEADS, HEAD_DIM_TRUE).transpose(1, 2)
    ref_scores_unscaled = hf_q_heads @ hf_k_heads.transpose(-2, -1)  # [B, NUM_HEADS, T, T]
    logger.info(f"Reference (torch) unscaled scores shape: {ref_scores_unscaled.shape}")

    # ---- Build TTNN's actual production Q, K and run its actual matmul ----
    h = dec_emb.float()
    pad = PADDED_WIDTH - h.shape[-1]
    if pad > 0:
        h = F.pad(h, (0, pad))
    hidden_states = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    w0 = weights["decoder.layers.0"]["self_attn"]
    fused_qkv = ttnn.linear(hidden_states, w0["qkv_weight"], bias=w0["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)
    # This is the EXACT production call -- unscaled, raw matmul output.
    scores = ttnn.matmul(query, key)
    scores_torch = ttnn.to_torch(scores).float()
    logger.info(f"TTNN raw scores shape: {scores_torch.shape}")

    # scores_torch shape depends on whether key came back transposed
    # (confirmed transposed in the prior run: key shape [B,H,head_dim_padded,T]).
    # So query @ key directly gives [B, H, T_q, T_k] already -- but the
    # PADDED head dimension (32) was contracted over, including 19 zero-padded
    # rows/cols per head. Those zero-padded lanes contribute exactly 0 to the
    # dot product (since both Q and K are zero there), so the contracted
    # result for the REAL T x T positions should already equal the true
    # 13-dim dot product -- padding does NOT need to be sliced out of the
    # T x T score matrix itself (unlike Q/K/V, which had padding in the
    # feature dimension; here the feature dim has already been summed over).
    assert (
        scores_torch.shape == ref_scores_unscaled.shape
    ), f"scores shape mismatch: {scores_torch.shape} vs {ref_scores_unscaled.shape}"

    passing, pcc_val = comp_pcc(scores_torch, ref_scores_unscaled, 0.99)
    logger.info(f"[Q@K^T unscaled] PCC: {pcc_val} (passing={passing})")

    # ---- Scale factor check: report both, do not assume which is "right" ----
    prod_scale = HEAD_DIM_PADDED**-0.5
    hf_scale = HEAD_DIM_TRUE**-0.5
    logger.info(f"Production scale factor (uses HEAD_DIM_PADDED=32): {prod_scale:.6f}")
    logger.info(f"HF-equivalent scale factor (uses HEAD_DIM_TRUE=13): {hf_scale:.6f}")
    logger.info(
        f"Ratio (prod/hf): {prod_scale / hf_scale:.6f} -- "
        f"if not 1.0, this is a SYSTEMATIC scale mismatch, not noise."
    )

    # Also report max abs diff for context (PCC alone can hide a small but
    # uniform scale-only mismatch since PCC is scale-invariant by design --
    # this is exactly why pcc_val above could look fine even with a real
    # scale bug, so the explicit ratio above is the real check for that).
    max_abs_diff = (scores_torch - ref_scores_unscaled).abs().max().item()
    logger.info(f"Max abs diff (unscaled scores): {max_abs_diff:.6f}")
