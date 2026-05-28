# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference verification for SeamlessM4Tv2SinusoidalPositionalEmbedding.

Compares the standalone PyTorch reference in ``functional.py`` against the
HuggingFace ``transformers`` implementation, and generates a golden tensor
saved to ``reference/golden/sinusoidal_positional_embedding.pt``.

Run with:

    cd <tt-metal>
    source python_env/bin/activate
    export PYTHONPATH=$(pwd) TT_METAL_HOME=$(pwd)
    python -m pytest models/demos/facebook_seamless_m4t_v2_large/reference/test_functional_sinusoidal_positional_embedding.py -s
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import (
    build_sinusoidal_positional_embedding_weights,
    sinusoidal_positional_embedding_forward,
)

# SeamlessM4T-v2 text encoder / decoder defaults (config.max_position_embeddings=4096,
# config.hidden_size=1024, config.pad_token_id=1).
HIDDEN_SIZE = 1024
PADDING_IDX = 1
MAX_POSITIONS = 4096

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_PATH = GOLDEN_DIR / "sinusoidal_positional_embedding.pt"


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    if torch.equal(a, b):
        return 1.0
    # Pearson correlation coefficient.
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = torch.sqrt((a_c * a_c).sum() * (b_c * b_c).sum())
    if denom.item() == 0.0:
        return float("nan")
    return (a_c * b_c).sum().item() / denom.item()


def _hf_module():
    """Instantiate the HF SeamlessM4Tv2SinusoidalPositionalEmbedding directly."""
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2SinusoidalPositionalEmbedding

    return SeamlessM4Tv2SinusoidalPositionalEmbedding(
        num_positions=MAX_POSITIONS,
        embedding_dim=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
    )


def test_weights_match_hf_buffer():
    """Reference weight table must equal HF's `weights` buffer bit-for-bit."""
    hf = _hf_module()

    # HF stores `num_positions + offset(=2)` rows.
    ref_weights = build_sinusoidal_positional_embedding_weights(
        num_embeddings=MAX_POSITIONS + 2,
        embedding_dim=HIDDEN_SIZE,
        padding_idx=PADDING_IDX,
    )

    assert ref_weights.shape == hf.weights.shape, (ref_weights.shape, hf.weights.shape)
    assert ref_weights.dtype == hf.weights.dtype, (ref_weights.dtype, hf.weights.dtype)
    # Bit-equivalence expected (same code path).
    assert torch.equal(ref_weights, hf.weights), "Sinusoidal weight tables differ"

    # Sanity: padding row is zero.
    assert torch.all(ref_weights[PADDING_IDX] == 0)


def test_forward_matches_hf_input_ids():
    """Reference forward must match HF forward for a padded input_ids batch."""
    hf = _hf_module()
    ref_weights = hf.weights.clone()

    torch.manual_seed(0)
    bsz, seq_len = 2, 17
    # Build input_ids: vocab range avoiding pad in some rows; first row has
    # leading-pad and second row has trailing-pad so we exercise the
    # padding-aware position logic in both directions.
    input_ids = torch.randint(low=3, high=200, size=(bsz, seq_len), dtype=torch.long)
    # Row 0: 3 leading padding tokens.
    input_ids[0, :3] = PADDING_IDX
    # Row 1: 5 trailing padding tokens.
    input_ids[1, -5:] = PADDING_IDX
    # Some interior padding too.
    input_ids[0, 10] = PADDING_IDX

    ref_out = sinusoidal_positional_embedding_forward(
        weights=ref_weights,
        input_ids=input_ids,
        padding_idx=PADDING_IDX,
    )
    hf_out = hf(input_ids=input_ids)

    assert ref_out.shape == hf_out.shape == (bsz, seq_len, HIDDEN_SIZE)
    assert torch.equal(ref_out, hf_out), "Reference forward does not match HF (input_ids)"
    pcc = _pcc(ref_out, hf_out)
    assert pcc > 0.99, f"PCC {pcc} < 0.99"


def test_forward_matches_hf_inputs_embeds():
    """Reference forward must match HF forward when called with inputs_embeds."""
    hf = _hf_module()
    ref_weights = hf.weights.clone()

    torch.manual_seed(1)
    bsz, seq_len = 3, 11
    inputs_embeds = torch.randn(bsz, seq_len, HIDDEN_SIZE)

    ref_out = sinusoidal_positional_embedding_forward(
        weights=ref_weights,
        inputs_embeds=inputs_embeds,
        padding_idx=PADDING_IDX,
    )
    hf_out = hf(inputs_embeds=inputs_embeds)

    assert ref_out.shape == hf_out.shape == (bsz, seq_len, HIDDEN_SIZE)
    assert torch.equal(ref_out, hf_out), "Reference forward does not match HF (inputs_embeds)"
    pcc = _pcc(ref_out, hf_out)
    assert pcc > 0.99, f"PCC {pcc} < 0.99"


def test_forward_matches_hf_with_past_kv_length():
    """Reference forward must match HF forward when past_key_values_length > 0 (decoder-style)."""
    hf = _hf_module()
    ref_weights = hf.weights.clone()

    torch.manual_seed(2)
    bsz, seq_len, past = 2, 1, 37  # incremental decode step
    input_ids = torch.randint(low=3, high=200, size=(bsz, seq_len), dtype=torch.long)

    ref_out = sinusoidal_positional_embedding_forward(
        weights=ref_weights,
        input_ids=input_ids,
        padding_idx=PADDING_IDX,
        past_key_values_length=past,
    )
    hf_out = hf(input_ids=input_ids, past_key_values_length=past)

    assert ref_out.shape == hf_out.shape == (bsz, seq_len, HIDDEN_SIZE)
    assert torch.equal(ref_out, hf_out), "Reference forward does not match HF (past_kv_length)"


def test_generate_and_save_golden():
    """Save a golden tensor for downstream TTNN block verification."""
    hf = _hf_module()
    ref_weights = hf.weights.clone()

    torch.manual_seed(0)

    # Representative inputs for the TTNN block: a typical text-encoder input
    # (batch=1, seq=128, no padding inside, no past KV) plus a single-step
    # decoder input (batch=1, seq=1, past=64).
    enc_input_ids = torch.randint(low=3, high=2000, size=(1, 128), dtype=torch.long)
    dec_input_ids = torch.randint(low=3, high=2000, size=(1, 1), dtype=torch.long)

    enc_ref = sinusoidal_positional_embedding_forward(
        weights=ref_weights, input_ids=enc_input_ids, padding_idx=PADDING_IDX
    )
    enc_hf = hf(input_ids=enc_input_ids)
    enc_pcc = _pcc(enc_ref, enc_hf)
    assert enc_pcc > 0.99, f"Encoder-style PCC {enc_pcc} < 0.99"

    dec_ref = sinusoidal_positional_embedding_forward(
        weights=ref_weights,
        input_ids=dec_input_ids,
        padding_idx=PADDING_IDX,
        past_key_values_length=64,
    )
    dec_hf = hf(input_ids=dec_input_ids, past_key_values_length=64)
    dec_pcc = _pcc(dec_ref, dec_hf)
    assert dec_pcc > 0.99, f"Decoder-style PCC {dec_pcc} < 0.99"

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "hidden_size": HIDDEN_SIZE,
                "padding_idx": PADDING_IDX,
                "max_positions": MAX_POSITIONS,
                "num_embedding_rows": MAX_POSITIONS + 2,
            },
            "weights": ref_weights,
            "encoder": {
                "input_ids": enc_input_ids,
                "past_key_values_length": 0,
                "output": enc_ref,
            },
            "decoder_incremental": {
                "input_ids": dec_input_ids,
                "past_key_values_length": 64,
                "output": dec_ref,
            },
            "pcc": {"encoder": enc_pcc, "decoder": dec_pcc},
        },
        GOLDEN_PATH,
    )
    print(f"Saved golden tensor to {GOLDEN_PATH}")
    print(f"PCC (encoder, seq=128): {enc_pcc}")
    print(f"PCC (decoder, seq=1, past=64): {dec_pcc}")


if __name__ == "__main__":
    test_weights_match_hf_buffer()
    test_forward_matches_hf_input_ids()
    test_forward_matches_hf_inputs_embeds()
    test_forward_matches_hf_with_past_kv_length()
    test_generate_and_save_golden()
    print("All tests passed.")
