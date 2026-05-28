# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""KV-cache correctness tests for the SeamlessM4T-v2 text decoder.

We exercise two properties:

1. ``test_self_attn_kv_cache_step_by_step_matches_batch`` — running the
   text decoder N times one new token at a time with the cached path
   produces the same final hidden state per step as a single batched
   non-cached forward over the full N-token sequence. PCC > 0.99
   required at every step.

2. ``test_cross_attn_cache_populates_once`` — after populating the
   cross-attention cache once, every subsequent decode step reads back
   the same K/V tensors (the cross-attn cache is static across decode
   steps).

Both tests use the real HF SeamlessM4T-v2-Large weights (cached at module
scope via the standard ``hf_sd`` fixture) and a reduced 2-layer text
decoder so the test stays fast.
"""

from __future__ import annotations

import re

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
from models.demos.facebook_seamless_m4t_v2_large.tt.kv_cache import SelfAttentionKVCache  # noqa: F401
from models.demos.facebook_seamless_m4t_v2_large.tt.kv_cache import (  # noqa: F401  (re-exported for downstream callers)
    CrossAttentionKVCache,
    TextDecoderKVCache,
)
from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder import TextDecoder

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

NUM_LAYERS = 2
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
EPS = 1e-5
PADDING_IDX = 0
PCC_THRESHOLD = 0.99
MAX_DECODE_SEQ_LEN = 32  # tile-aligned, ample for the 4-step test
NUM_DECODE_STEPS = 4
ENCODER_SEQ_LEN_LOGICAL = 16  # Decoder will tile-pad this to 32 internally.
_TILE = 32


def _padded(n: int) -> int:
    return n + ((_TILE - n % _TILE) % _TILE)


ENCODER_SEQ_LEN_PADDED = _padded(ENCODER_SEQ_LEN_LOGICAL)


@pytest.fixture(scope="session")
def hf_sd():
    return wl.load_hf_state_dict()


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def _pcc_value(ref_t: torch.Tensor, tt_t: torch.Tensor) -> float:
    """Run ``comp_pcc`` and parse the returned PCC value as a float."""
    _, msg = comp_pcc(ref_t, tt_t, PCC_THRESHOLD)
    s = str(msg).strip()
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", s)
        return float(m.group(0)) if m else float("nan")


def _build_decoder(device, hf_sd):
    sd = wl.text_decoder_weights(hf_sd, num_layers=NUM_LAYERS)
    decoder = TextDecoder(
        device=device,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        embed_tokens_weight=sd["embed_tokens"]["weight"],
        embed_positions_weights=sd["embed_positions_weights"],
        layers_state_dict=sd["layers"],
        final_layer_norm_state_dict=sd["layer_norm"],
        eps=EPS,
        padding_idx=PADDING_IDX,
        weight_dtype=ttnn.bfloat16,
    )
    return decoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_self_attn_kv_cache_step_by_step_matches_batch(device, hf_sd):
    """Step-by-step cached decode == batched non-cached forward (PCC > 0.99)."""
    torch.manual_seed(123)

    # NUM_DECODE_STEPS distinct decoder token ids (avoid padding_idx 0).
    decoder_ids = torch.randint(low=2, high=512, size=(1, NUM_DECODE_STEPS), dtype=torch.long)
    encoder_hidden_states = torch.randn(1, ENCODER_SEQ_LEN_LOGICAL, EMBED_DIM, dtype=torch.float32)

    decoder = _build_decoder(device, hf_sd)

    # ----- Reference: single batched non-cached forward over all N tokens -----
    ref_out_tt = decoder(
        decoder_ids,
        encoder_hidden_states_torch=encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
    )
    ref_out = ttnn.to_torch(ref_out_tt).to(torch.float32).reshape(1, NUM_DECODE_STEPS, EMBED_DIM)
    ttnn.deallocate(ref_out_tt)

    # ----- Cached: NUM_DECODE_STEPS one-token decode_step calls -----
    past_key_values = TextDecoderKVCache(
        device=device,
        num_layers=NUM_LAYERS,
        batch=1,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        max_decode_seq_len=MAX_DECODE_SEQ_LEN,
        # The decoder tile-pads the encoder hidden states internally
        # (here 16 -> 32). The cross-attn cache must match that padded
        # length, not the logical one.
        encoder_seq_len=ENCODER_SEQ_LEN_PADDED,
    )

    # Pre-populate cross-attn cache from the encoder hidden states.
    decoder.populate_cross_attention_cache(
        past_key_values=past_key_values,
        encoder_hidden_states_torch=encoder_hidden_states,
    )
    # Sanity: every layer's cross-attn cache must be populated now.
    for i in range(NUM_LAYERS):
        assert past_key_values.cross_attn.is_populated(i), f"cross_attn layer {i} not populated"

    # Step-by-step decode.
    per_step_pccs = []
    for step in range(NUM_DECODE_STEPS):
        new_id = decoder_ids[:, step : step + 1]  # [1, 1]
        step_out_tt = decoder.decode_step(
            input_ids=new_id,
            position=step,
            past_key_values=past_key_values,
            encoder_attention_mask=None,
            encoder_seq_len_logical=ENCODER_SEQ_LEN_LOGICAL,
        )
        step_out = ttnn.to_torch(step_out_tt).to(torch.float32).reshape(1, 1, EMBED_DIM)
        ttnn.deallocate(step_out_tt)
        pcc = _pcc_value(ref_out[:, step : step + 1, :], step_out)
        per_step_pccs.append(pcc)
        print(f"[decode step {step}] PCC = {pcc:.6f}")

    worst = min(per_step_pccs)
    assert worst > PCC_THRESHOLD, f"per-step PCCs={per_step_pccs}; worst={worst} <= {PCC_THRESHOLD}"


def test_cross_attn_cache_populates_once(device, hf_sd):
    """Verify the cross-attn cache is static across decode steps.

    After populating it once, the read() result for every layer must be
    the same tensor handle on every subsequent decode step. We also do a
    deep numerical comparison: read the K/V tensors before step 0 and
    after step NUM_DECODE_STEPS-1 and assert they are bit-identical.
    """
    torch.manual_seed(456)

    decoder_ids = torch.randint(low=2, high=512, size=(1, NUM_DECODE_STEPS), dtype=torch.long)
    encoder_hidden_states = torch.randn(1, ENCODER_SEQ_LEN_LOGICAL, EMBED_DIM, dtype=torch.float32)

    decoder = _build_decoder(device, hf_sd)

    past_key_values = TextDecoderKVCache(
        device=device,
        num_layers=NUM_LAYERS,
        batch=1,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        max_decode_seq_len=MAX_DECODE_SEQ_LEN,
        encoder_seq_len=ENCODER_SEQ_LEN_PADDED,
    )

    decoder.populate_cross_attention_cache(
        past_key_values=past_key_values,
        encoder_hidden_states_torch=encoder_hidden_states,
    )

    # Snapshot K/V per layer right after populate, before any decode step.
    snapshot_k = []
    snapshot_v = []
    for i in range(NUM_LAYERS):
        k, v = past_key_values.cross_attn.read(i)
        snapshot_k.append(ttnn.to_torch(k).to(torch.float32).clone())
        snapshot_v.append(ttnn.to_torch(v).to(torch.float32).clone())
        # Tensor handle identity check.
        k2, v2 = past_key_values.cross_attn.read(i)
        assert k is k2, f"cross_attn.read(layer={i}) returned different K handle"
        assert v is v2, f"cross_attn.read(layer={i}) returned different V handle"

    # Run all decode steps; cross-attn cache must not mutate.
    for step in range(NUM_DECODE_STEPS):
        new_id = decoder_ids[:, step : step + 1]
        out_tt = decoder.decode_step(
            input_ids=new_id,
            position=step,
            past_key_values=past_key_values,
            encoder_attention_mask=None,
            encoder_seq_len_logical=ENCODER_SEQ_LEN_LOGICAL,
        )
        ttnn.deallocate(out_tt)

    # After all decode steps, read again and compare bit-for-bit.
    for i in range(NUM_LAYERS):
        k_after, v_after = past_key_values.cross_attn.read(i)
        k_after_t = ttnn.to_torch(k_after).to(torch.float32)
        v_after_t = ttnn.to_torch(v_after).to(torch.float32)
        assert torch.equal(k_after_t, snapshot_k[i]), f"cross_attn K layer {i} mutated during decode!"
        assert torch.equal(v_after_t, snapshot_v[i]), f"cross_attn V layer {i} mutated during decode!"


# ---------------------------------------------------------------------------
# Standalone smoke runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        sd = wl.load_hf_state_dict()
        # Reuse the test bodies inline by calling them with the same dev/sd.
        results = {}
        try:
            test_self_attn_kv_cache_step_by_step_matches_batch.__wrapped__ if hasattr(
                test_self_attn_kv_cache_step_by_step_matches_batch, "__wrapped__"
            ) else None
            test_self_attn_kv_cache_step_by_step_matches_batch(dev, sd)
            results["self_attn"] = "ok"
        except Exception as exc:
            results["self_attn"] = f"FAIL: {exc!r}"

        try:
            test_cross_attn_cache_populates_once(dev, sd)
            results["cross_attn"] = "ok"
        except Exception as exc:
            results["cross_attn"] = f"FAIL: {exc!r}"

        print(json.dumps(results))
        sys.exit(0 if all(v == "ok" for v in results.values()) else 1)
    finally:
        ttnn.close_device(dev)
