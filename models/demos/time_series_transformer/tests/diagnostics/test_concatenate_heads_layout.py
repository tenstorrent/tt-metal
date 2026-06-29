# tests/test_concatenate_heads_layout.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DIAGNOSTIC -- not part of the real bounty test suite.

Context: test_attn_v_outproj_pcc.py localized the remaining decoder
self-attention error precisely:
    [1] post-softmax probs        PCC 0.99971  PASS
    [2] attn@V context (unpadded, MANUAL per-head slice+concat)  PCC 0.99991  PASS
    [4] final output (ttnn concatenate_heads + real out_proj)     PCC 0.81936  FAIL

Since [2]'s manual unpad+concat is correct (matches HF) but [4]'s path
(ttnn.transformer.concatenate_heads + out_proj) is not, the bug is
somewhere in concatenate_heads' layout, out_proj_weight's construction,
or a mismatch between the two.

THIS TEST: run ttnn.transformer.concatenate_heads on the EXACT SAME
context tensor used in [2], and diff it element-wise against the manual
per-head unpad+concat from [2]. No HF reference needed here -- this is
purely "do these two reassembly methods agree with each other", which
isolates the layout question from any out_proj weight-construction
question.

If they DISAGREE: concatenate_heads uses a different internal layout
than the simple "head h -> padded columns [h*32:(h+1)*32]" assumption
baked into _pad_weight_per_head (tst_model.py) when it built
out_proj_weight. That would explain [4]'s failure directly: out_proj_weight
expects one layout, concatenate_heads produces another, so the matmul
contracts the wrong columns against the wrong weight rows.

If they AGREE: the layout is NOT the problem, and the bug must be in
out_proj_weight's actual values (i.e. _pad_weight_per_head or
_pad_bias_per_head have a real construction bug, not just a layout
mismatch) -- a different, more specific next hypothesis.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_attention import HEAD_DIM_PADDED, HEAD_DIM_TRUE, NUM_HEADS, build_causal_mask, causal_softmax
from tt.tst_model import load_weights

import ttnn

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_hf_decoder_input(hf_model, inputs):
    captured = {}

    def dec_emb_hook(m, i, o):
        captured["dec_emb"] = o.detach()

    h1 = hf_model.model.decoder.layernorm_embedding.register_forward_hook(dec_emb_hook)
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
    h1.remove()
    return captured["dec_emb"]


def manual_unpad_concat(context_torch):
    """The SAME logic used (and verified, PCC 0.99991) in [2] of
    test_attn_v_outproj_pcc.py: for each head, take its real first
    HEAD_DIM_TRUE=13 columns out of its own 32-wide padded block,
    then concatenate heads side by side -> [B, T, NUM_HEADS*13] = [B, T, 26].
    This is NOT padded back up -- it's the true, unpadded 26-wide tensor.
    """
    B, H, T, _ = context_torch.shape
    real_heads = [context_torch[:, h, :, :HEAD_DIM_TRUE] for h in range(H)]
    return torch.cat(real_heads, dim=-1)  # [B, T, 26]


def manual_unpad_concat_padded(context_torch):
    """Same as above, but instead of dropping padding, keep each head's
    full 32-wide slot (so output is [B, T, NUM_HEADS*32] = [B, T, 64]),
    matching the shape concatenate_heads returns -- for an apples-to-apples
    shape comparison against ttnn's concatenate_heads output."""
    B, H, T, D = context_torch.shape
    heads = [context_torch[:, h, :, :] for h in range(H)]  # each [B, T, 32]
    return torch.cat(heads, dim=-1)  # [B, T, H*32] = [B, T, 64]


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    yield device, hf_model, weights, inputs
    ttnn.close_device(device)


def test_concatenate_heads_layout(setup):
    device, hf_model, weights, inputs = setup

    dec_emb = get_hf_decoder_input(hf_model, inputs)
    T_dec = dec_emb.shape[1]
    logger.info(f"dec_emb shape: {dec_emb.shape}, T_dec={T_dec}")

    w = weights["decoder.layers.0"]["self_attn"]

    pad = 64 - dec_emb.shape[-1]
    h = F.pad(dec_emb.float(), (0, pad)) if pad > 0 else dec_emb.float()
    hidden = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    fused_qkv = ttnn.linear(hidden, w["qkv_weight"], bias=w["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)

    scores = ttnn.matmul(query, key)
    scale = HEAD_DIM_TRUE**-0.5
    mask = build_causal_mask(device, T_dec)
    probs = causal_softmax(scores, mask, scale)

    context = ttnn.matmul(probs, value)  # [B, NUM_HEADS, T_dec, HEAD_DIM_PADDED=32]
    context_torch = ttnn.to_torch(context).float()
    logger.info(f"context (pre-concat) shape: {context_torch.shape}")

    # Path A: ttnn's own concatenate_heads
    context_concat_ttnn = ttnn.transformer.concatenate_heads(context)
    concat_ttnn_torch = ttnn.to_torch(context_concat_ttnn).float()
    logger.info(f"ttnn concatenate_heads output shape: {concat_ttnn_torch.shape}")

    # Path B: manual concat, keeping full padded width per head (for shape parity)
    concat_manual_padded = manual_unpad_concat_padded(context_torch)
    logger.info(f"manual padded concat shape: {concat_manual_padded.shape}")

    # Direct element-wise comparison -- same input, two reassembly methods
    if concat_ttnn_torch.shape != concat_manual_padded.shape:
        logger.info(
            f"SHAPE MISMATCH: ttnn={concat_ttnn_torch.shape} vs "
            f"manual={concat_manual_padded.shape} -- cannot diff directly, "
            f"this alone indicates a layout difference."
        )
    else:
        max_abs_diff = (concat_ttnn_torch - concat_manual_padded).abs().max().item()
        mean_abs_diff = (concat_ttnn_torch - concat_manual_padded).abs().mean().item()
        identical = torch.allclose(concat_ttnn_torch, concat_manual_padded, atol=1e-2)
        logger.info(
            f"concatenate_heads vs manual padded concat: max_abs_diff={max_abs_diff:.6f}, "
            f"mean_abs_diff={mean_abs_diff:.6f}, allclose(atol=1e-2)={identical}"
        )

        # Spot-check: print first row, first 8 cols of head 0's region (cols 0:8)
        # and head 1's region (cols 32:40) for both, to see qualitatively
        # whether values just got reordered/shifted vs. genuinely different.
        logger.info(f"ttnn   [0,0,0:8]   (head0 region): {concat_ttnn_torch[0,0,0:8].tolist()}")
        logger.info(f"manual [0,0,0:8]   (head0 region): {concat_manual_padded[0,0,0:8].tolist()}")
        logger.info(f"ttnn   [0,0,32:40] (head1 region): {concat_ttnn_torch[0,0,32:40].tolist()}")
        logger.info(f"manual [0,0,32:40] (head1 region): {concat_manual_padded[0,0,32:40].tolist()}")

        if not identical:
            # Check if it's a head-order swap: does ttnn's [0:32] match manual's [32:64]?
            swapped_diff = (concat_ttnn_torch[..., :32] - concat_manual_padded[..., 32:64]).abs().max().item()
            logger.info(
                f"Hypothesis check -- head-order swap: ttnn[:32] vs manual[32:64] "
                f"max_abs_diff={swapped_diff:.6f} (low value would confirm swapped head order)"
            )

    # Also compare against the UNPADDED manual concat (true 26-wide), since
    # that's what we know matches HF (PCC 0.99991 in [2] of the prior test).
    concat_manual_unpadded = manual_unpad_concat(context_torch)  # [B, T, 26]
    # Slice ttnn's concat output the same naive way out_proj construction assumes:
    # heads contiguous, real data first 13 of each 32-wide block.
    B, T, _ = concat_ttnn_torch.shape
    ttnn_sliced_as_if_simple = torch.cat(
        [concat_ttnn_torch[:, :, h * HEAD_DIM_PADDED : h * HEAD_DIM_PADDED + HEAD_DIM_TRUE] for h in range(NUM_HEADS)],
        dim=-1,
    )  # [B, T, 26], assuming ttnn's concat used the simple contiguous-padded-block layout
    diff_vs_known_good = (ttnn_sliced_as_if_simple - concat_manual_unpadded).abs().max().item()
    logger.info(
        f"ttnn concat sliced under SIMPLE layout assumption vs known-good manual "
        f"unpadded concat: max_abs_diff={diff_vs_known_good:.6f} "
        f"(near-zero confirms ttnn DOES use the simple contiguous layout; "
        f"large value means the layout assumption itself is wrong)"
    )
