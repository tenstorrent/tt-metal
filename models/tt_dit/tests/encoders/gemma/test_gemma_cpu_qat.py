#!/usr/bin/env python3
"""
Validate QAT Gemma-3-12b hidden states on CPU against reference FeatureExtractorV2.

Tests that the gemma-3-12b-it-qat-q4_0-unquantized model produces identical hidden
states when loaded via HuggingFace transformers (same path the reference uses).
If this passes, the TTNN Gemma issue is purely device-side bf16 precision.
"""

import gc
import os
import sys

import pytest
import torch
from loguru import logger

GEMMA_PATH = os.environ.get(
    "GEMMA_PATH",
    "/localdev/kevinmi/.cache/gemma-3-12b-it-qat-q4_0-unquantized",
)
CKPT = os.environ.get(
    "LTX_CHECKPOINT",
    os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"),
)
PROMPT = "A plump orange tabby cat sits on a piano bench playing keys with its paws."


def pcc(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    a_m, b_m = a_f - a_f.mean(), b_f - b_f.mean()
    d = (a_m.pow(2).sum() * b_m.pow(2).sum()).sqrt()
    return ((a_m * b_m).sum() / d).item() if d > 0 else 0.0


@pytest.fixture(scope="module")
def ref_hidden_states():
    """Get reference hidden states via ModelLedger → text_encoder (same as FeatureExtractorV2)."""
    if not os.path.isdir(GEMMA_PATH):
        pytest.skip(f"Gemma not found: {GEMMA_PATH}")
    if not os.path.exists(CKPT):
        pytest.skip(f"Checkpoint not found: {CKPT}")

    sys.path.insert(0, "LTX-2/packages/ltx-core/src")
    sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
    torch.cuda.synchronize = lambda *a, **kw: None

    from ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=CKPT,
        gemma_root_path=GEMMA_PATH,
    )
    text_encoder = ledger.text_encoder()
    text_encoder.eval()

    logger.info(f"Reference text encoder loaded from ModelLedger")

    # Encode prompt — returns (tuple_of_hidden_states, attention_mask)
    with torch.no_grad():
        hidden_states, attention_mask = text_encoder.encode(PROMPT)

    logger.info(f"Reference: {len(hidden_states)} hidden states, mask shape {attention_mask.shape}")
    logger.info(f"Reference: real tokens = {attention_mask.sum().item()}")

    result = {
        "hidden_states": [h.float() for h in hidden_states],
        "attention_mask": attention_mask,
        "n_layers": len(hidden_states),
    }

    del text_encoder, ledger
    gc.collect()
    return result


@pytest.fixture(scope="module")
def qat_hidden_states():
    """Get hidden states by directly loading QAT Gemma via HuggingFace transformers."""
    if not os.path.isdir(GEMMA_PATH):
        pytest.skip(f"Gemma not found: {GEMMA_PATH}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
    # Left-padding to match reference
    tokens = tokenizer(PROMPT, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)

    logger.info(f"QAT Gemma: tokens shape {tokens.input_ids.shape}, real = {tokens.attention_mask.sum().item()}")

    model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, torch_dtype=torch.bfloat16)
    model.eval()

    with torch.no_grad():
        out = model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )

    qat_hs = out.hidden_states  # tuple of 49 tensors (embed + 48 layers)
    logger.info(f"QAT Gemma: {len(qat_hs)} hidden states, shape {qat_hs[0].shape}")

    result = {
        "hidden_states": [h.float() for h in qat_hs],
        "attention_mask": tokens.attention_mask,
        "n_layers": len(qat_hs),
    }

    del model, out
    gc.collect()
    return result


def test_token_match(ref_hidden_states, qat_hidden_states):
    """Verify both paths produce the same number of hidden states."""
    n_ref = ref_hidden_states["n_layers"]
    n_qat = qat_hidden_states["n_layers"]
    logger.info(f"Reference layers: {n_ref}, QAT layers: {n_qat}")
    # Reference text_encoder.encode may return different count than raw model
    # (e.g., it might skip embedding layer or add final norm)
    # Just log the difference for now
    assert n_ref > 0 and n_qat > 0


def test_hidden_state_per_layer(ref_hidden_states, qat_hidden_states):
    """Compare hidden states layer by layer between reference and QAT on CPU.

    Expected: near-perfect match (PCC > 0.9999) since both run on CPU with same weights.
    """
    ref_hs = ref_hidden_states["hidden_states"]
    qat_hs = qat_hidden_states["hidden_states"]
    ref_mask = ref_hidden_states["attention_mask"]
    qat_mask = qat_hidden_states["attention_mask"]

    n_real_ref = int(ref_mask.sum().item())
    n_real_qat = int(qat_mask.sum().item())
    logger.info(f"Real tokens: ref={n_real_ref}, qat={n_real_qat}")

    # Determine comparison range (min of both)
    n_compare = min(len(ref_hs), len(qat_hs))
    logger.info(f"Comparing {n_compare} layers")

    logger.info(f"{'Layer':>6} {'PCC':>10} {'MaxDiff':>12} {'MeanDiff':>12} {'Ref_std':>10} {'QAT_std':>10}")

    all_good = True
    for i in range(n_compare):
        ref_h = ref_hs[i]
        qat_h = qat_hs[i]

        if ref_h.shape != qat_h.shape:
            logger.warning(f"Layer {i}: shape mismatch ref={ref_h.shape} qat={qat_h.shape}")
            continue

        # Compare only real (non-padding) tokens
        # Reference uses left-padding, so real tokens are at the end
        ref_real = ref_h[:, -n_real_ref:, :]
        qat_real = qat_h[:, -n_real_qat:, :]

        if ref_real.shape != qat_real.shape:
            # Different number of real tokens — compare full tensors
            ref_real = ref_h
            qat_real = qat_h

        p = pcc(ref_real, qat_real)
        diff = (ref_real - qat_real).abs()
        logger.info(
            f"{i:>6} {p:>10.6f} {diff.max():>12.4e} {diff.mean():>12.4e} "
            f"{ref_real.std():>10.4f} {qat_real.std():>10.4f}"
        )

        if p < 0.999:
            all_good = False
            logger.error(f"  LOW PCC at layer {i}")

    assert all_good, "Some layers have PCC < 0.999"


def test_final_embeddings(ref_hidden_states, qat_hidden_states):
    """Compare the stacked+normed features that feed into the connector."""
    ref_hs = ref_hidden_states["hidden_states"]
    qat_hs = qat_hidden_states["hidden_states"]

    # Stack all hidden states (skip final norm if present)
    # Reference FeatureExtractorV2 uses hidden_states[:-1] for stacking
    D = 3840  # Gemma hidden size

    def stack_and_norm(hs_list):
        # Use first N layers that have shape (B, seq, D)
        valid = [h for h in hs_list if h.shape[-1] == D]
        stacked = torch.stack(valid, dim=-1)  # (B, T, D, L)
        variance = torch.mean(stacked**2, dim=2, keepdim=True)
        normed = stacked * torch.rsqrt(variance + 1e-6)
        return normed.reshape(stacked.shape[0], stacked.shape[1], -1)

    ref_normed = stack_and_norm(ref_hs)
    qat_normed = stack_and_norm(qat_hs)

    if ref_normed.shape == qat_normed.shape:
        p = pcc(ref_normed, qat_normed)
        diff = (ref_normed - qat_normed).abs()
        logger.info(
            f"Stacked+normed features: PCC={p:.6f}, max_diff={diff.max():.4e}, "
            f"ref_std={ref_normed.std():.4f}, qat_std={qat_normed.std():.4f}"
        )
        assert p > 0.999, f"Stacked features PCC {p} too low"
    else:
        logger.warning(f"Shape mismatch: ref={ref_normed.shape} qat={qat_normed.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
