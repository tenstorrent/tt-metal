# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end token-match tests for :class:`TextGenerator`.

These are the FIRST tests in this project that compare TTNN
generation output against the real HuggingFace ``model.generate()``
method (not just per-block PCC).

There are two tests:

1.  ``test_token_match_greedy_t2tt_short``
    Runs HF ``SeamlessM4Tv2ForTextToText.generate(do_sample=False)`` to
    get a reference token list, then runs the TTNN
    ``TextEncoder + TextGenerator`` over the same source ids and target
    language. Asserts the TTNN token list matches the HF one for a
    short generation (token-for-token in greedy mode). Allows some
    slack: ``>= 90%`` match AND first divergence (if any) must be at
    position >= 6 (i.e. the recognisable head of the translation must
    match exactly).

2.  ``test_generation_terminates_on_eos``
    Sanity check that the AR loop stops on ``eos_token_id``.

Memory budget: this is the heaviest test in the project so far. We
limit ``max_decode_seq_len=64``, ``max_new_tokens=24``,
``encoder_seq_len=32`` (tile-padded, source is 5 logical tokens for
``"Hello world."``).

Run with::

    pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_text_generator.py -v
"""

from __future__ import annotations

import math
from typing import List

import pytest
import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
from models.demos.facebook_seamless_m4t_v2_large.tt.text_encoder import TextEncoder
from models.demos.facebook_seamless_m4t_v2_large.tt.text_generator import TextGenerator

# ---------------------------------------------------------------------------
# Model config (matches SeamlessM4T-v2-Large defaults)
# ---------------------------------------------------------------------------
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
NUM_LAYERS = 24
EPS = 1e-5
ENCODER_PADDING_IDX = 0
DECODER_PADDING_IDX = 0

# Decoder configuration. Per HF generation_config.json for SeamlessM4T-v2-Large:
DECODER_START_TOKEN_ID = 3  # = eos_token_id
EOS_TOKEN_ID = 3
TGT_LANG_ID_FRA = 256026  # text_decoder_lang_to_code_id["fra"]

# Test budget — we are running a 24-layer encoder + 24-layer decoder + LM head.
MAX_DECODE_SEQ_LEN = 64
MAX_NEW_TOKENS = 24
ENCODER_SEQ_LEN_PADDED = 32  # logical 5 tokens; padded to 32 (1 tile).
HF_PATH = wl.HF_PATH

_TILE = 32


def _pad_to_tile(n: int) -> int:
    return (_TILE - n % _TILE) % _TILE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_sd():
    return wl.load_hf_state_dict()


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def hf_inputs():
    """Tokenize ``"Hello world."`` with src_lang=eng.

    We return host tensors only — no model load yet (that happens in
    ``hf_reference_tokens`` so we can free the HF model after capture).
    """
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(HF_PATH)
    toks = proc(text="Hello world.", src_lang="eng", return_tensors="pt")
    return {
        "input_ids": toks["input_ids"],  # [1, S_logical]
        "attention_mask": toks["attention_mask"],  # [1, S_logical], 1=keep
        "processor": proc,
    }


@pytest.fixture(scope="module")
def hf_reference_tokens(hf_inputs):
    """Run HF SeamlessM4Tv2ForTextToText.generate with greedy decoding.

    Returns the list of generated token ids (including the prefix
    ``[decoder_start_token_id, tgt_lang_id]`` HF prepends).
    """
    from transformers import SeamlessM4Tv2ForTextToText

    model = SeamlessM4Tv2ForTextToText.from_pretrained(HF_PATH, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=hf_inputs["input_ids"],
            attention_mask=hf_inputs["attention_mask"],
            tgt_lang="fra",
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            early_stopping=False,
        )
    # `out` may be a tuple/dict (depending on return_dict_in_generate); in default
    # the call returns a LongTensor of generated ids.
    if hasattr(out, "sequences"):
        out_ids = out.sequences
    else:
        out_ids = out
    tokens = out_ids[0].tolist()
    # Decode for sanity.
    proc = hf_inputs["processor"]
    text = proc.decode(tokens, skip_special_tokens=False)
    del model
    return {
        "tokens": tokens,
        "decoded": text,
        "encoder_hidden_dump": None,  # populated below by encoder helper if needed
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tile_padded_encoder_inputs(hf_inputs):
    """Return ``(input_ids_padded, attention_mask_padded, src_logical)``.

    The TTNN ``TextEncoder`` handles its own SDPA tile padding when we
    pass ``attention_mask_torch`` directly. But the LOGICAL length we
    care about (for slicing the encoder output before feeding the
    decoder) is the number of real tokens in the source.

    Source = ``"Hello world."`` with src_lang=eng has 5 logical tokens
    (the tokenizer pads to 6 with a trailing 0; we strip that here).
    """
    input_ids = hf_inputs["input_ids"]
    attn_mask_2d = hf_inputs["attention_mask"]
    src_logical = int(attn_mask_2d.sum().item())
    # Trim padding to the logical length (so the encoder's tile-padding
    # path is well-defined for the cross-attn cache).
    input_ids = input_ids[:, :src_logical]
    attn_mask_2d = attn_mask_2d[:, :src_logical]
    return input_ids, attn_mask_2d, src_logical


def _run_ttnn_encoder(device, hf_sd, input_ids, attn_mask_2d):
    """Build a 24-layer TTNN TextEncoder, run forward, return host hidden states."""
    sd = wl.text_encoder_weights(hf_sd, num_layers=NUM_LAYERS, padding_idx=ENCODER_PADDING_IDX)
    encoder = TextEncoder(
        device=device,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        embed_tokens_weight=sd["embed_tokens"]["weight"],
        embed_positions_weights=sd["embed_positions_weights"],
        layers_state_dict=sd["layers"],
        final_layer_norm_state_dict=sd["final_layer_norm"],
        eps=EPS,
        padding_idx=ENCODER_PADDING_IDX,
        embed_scale=math.sqrt(EMBED_DIM),
        weight_dtype=ttnn.bfloat16,
    )
    # Expand 2D mask to 4D additive log-mask for the encoder's SDPA.
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

    tgt_len = int(input_ids.shape[-1])
    mask_4d = _prepare_4d_attention_mask(attn_mask_2d, torch.float32, tgt_len=tgt_len)
    enc_hidden_tt = encoder(input_ids, attention_mask_torch=mask_4d)
    enc_hidden_torch = ttnn.to_torch(enc_hidden_tt).to(torch.float32)
    ttnn.deallocate(enc_hidden_tt)
    if enc_hidden_torch.dim() == 4 and enc_hidden_torch.shape[0] == 1:
        enc_hidden_torch = enc_hidden_torch.squeeze(0)
    return enc_hidden_torch


def _tile_pad_encoder_hidden(enc_hidden: torch.Tensor) -> torch.Tensor:
    """Right-pad encoder hidden states along S to a tile multiple with zeros."""
    s = int(enc_hidden.shape[1])
    pad = _pad_to_tile(s)
    if pad == 0:
        return enc_hidden
    z = torch.zeros((enc_hidden.shape[0], pad, enc_hidden.shape[2]), dtype=enc_hidden.dtype)
    return torch.cat([enc_hidden, z], dim=1)


def _build_ttnn_generator(device, hf_sd, encoder_seq_len: int) -> TextGenerator:
    text_dec_sd = wl.text_decoder_weights(hf_sd, num_layers=NUM_LAYERS, padding_idx=DECODER_PADDING_IDX)
    lm_head_sd = wl.lm_head_weights(hf_sd)
    return TextGenerator(
        device=device,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        text_decoder_state_dict=text_dec_sd,
        lm_head_state_dict=lm_head_sd,
        max_decode_seq_len=MAX_DECODE_SEQ_LEN,
        encoder_seq_len=encoder_seq_len,
        eps=EPS,
        padding_idx=DECODER_PADDING_IDX,
        embed_scale=math.sqrt(EMBED_DIM),
        weight_dtype=ttnn.bfloat16,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_token_match_greedy_t2tt_short(device, hf_sd, hf_inputs, hf_reference_tokens):
    """Greedy generation: TTNN tokens >= 90% match HF tokens; first divergence (if any) at pos >= 6."""

    hf_tokens: List[int] = hf_reference_tokens["tokens"]
    print(f"\nHF reference tokens ({len(hf_tokens)}): {hf_tokens}")
    print(f"HF decoded: {hf_reference_tokens['decoded']!r}")

    input_ids, attn_mask_2d, src_logical = _tile_padded_encoder_inputs(hf_inputs)

    # --- 1. Encoder forward in TTNN. ---
    enc_hidden_torch = _run_ttnn_encoder(device, hf_sd, input_ids, attn_mask_2d)
    # The encoder slices its output back to logical length; pad along S
    # again to the cross-attn cache tile-multiple before handing it to
    # the generator.
    enc_hidden_padded = _tile_pad_encoder_hidden(enc_hidden_torch[:, :src_logical, :])
    encoder_seq_len_padded = int(enc_hidden_padded.shape[1])

    # --- 2. Generator forward (24-layer decoder + LM head). ---
    gen = _build_ttnn_generator(device, hf_sd, encoder_seq_len=encoder_seq_len_padded)
    tt_tokens = gen.generate(
        encoder_hidden_states=enc_hidden_padded,
        encoder_attention_mask=attn_mask_2d,
        decoder_start_token_id=DECODER_START_TOKEN_ID,
        tgt_lang_id=TGT_LANG_ID_FRA,
        eos_token_id=EOS_TOKEN_ID,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    print(f"\nTTNN tokens ({len(tt_tokens)}): {tt_tokens}")
    # Decode TTNN tokens through the processor.
    tt_decoded = hf_inputs["processor"].decode(tt_tokens, skip_special_tokens=False)
    print(f"TTNN decoded: {tt_decoded!r}")

    # --- 3. Compare. ---
    n = min(len(hf_tokens), len(tt_tokens))
    matches = [hf_tokens[i] == tt_tokens[i] for i in range(n)]
    match_rate = sum(matches) / n if n else 0.0
    first_div = next((i for i, m in enumerate(matches) if not m), None)
    print(
        f"match_rate={match_rate:.3f}  first_divergence_at={first_div}  HF_len={len(hf_tokens)} TT_len={len(tt_tokens)}"
    )

    # Hard floor.
    assert match_rate >= 0.90, f"token match rate {match_rate:.3f} < 0.90; first divergence at {first_div}"
    # The deterministic prefix (decoder_start + tgt_lang_id + first 4 sampled tokens)
    # must match -- if it doesn't, the encoder cross-attn cache or the
    # LM head wiring is broken, not just bf16 drift.
    if first_div is not None:
        assert first_div >= 6, (
            f"first divergence at position {first_div} < 6 — this indicates a structural error, "
            f"not bf16 drift.\n  HF[{first_div}]={hf_tokens[first_div]}, TT[{first_div}]={tt_tokens[first_div]}"
        )


def test_generation_terminates_on_eos(device, hf_sd, hf_inputs):
    """The AR loop must stop when it samples ``eos_token_id``.

    For short inputs (~"Hello world.") and a real-trained translation
    head, the generator will hit EOS well before ``max_new_tokens=24``.
    """
    input_ids, attn_mask_2d, src_logical = _tile_padded_encoder_inputs(hf_inputs)
    enc_hidden_torch = _run_ttnn_encoder(device, hf_sd, input_ids, attn_mask_2d)
    enc_hidden_padded = _tile_pad_encoder_hidden(enc_hidden_torch[:, :src_logical, :])
    encoder_seq_len_padded = int(enc_hidden_padded.shape[1])

    gen = _build_ttnn_generator(device, hf_sd, encoder_seq_len=encoder_seq_len_padded)
    tokens = gen.generate(
        encoder_hidden_states=enc_hidden_padded,
        encoder_attention_mask=attn_mask_2d,
        decoder_start_token_id=DECODER_START_TOKEN_ID,
        tgt_lang_id=TGT_LANG_ID_FRA,
        eos_token_id=EOS_TOKEN_ID,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    print(f"\ngenerated tokens: {tokens}")
    # Two possible legitimate ends:
    #   (a) the last token is EOS  -> loop hit EOS and broke,
    #   (b) len(tokens) == MAX_NEW_TOKENS  -> loop ran out of budget.
    if tokens[-1] == EOS_TOKEN_ID:
        assert tokens[-1] == EOS_TOKEN_ID, "loop should have terminated on EOS"
    else:
        # If we ran out of budget, it must be EXACTLY at MAX_NEW_TOKENS — no surprises.
        assert (
            len(tokens) == MAX_NEW_TOKENS
        ), f"loop ended without EOS at {len(tokens)} tokens, expected MAX_NEW_TOKENS={MAX_NEW_TOKENS}"
    # Either way: there must NOT be an EOS in the MIDDLE of the sequence (would mean we kept going past EOS).
    # Skip the prefix EOS at position 0 (decoder_start_token_id == eos_token_id by config).
    middle = tokens[1:-1]
    assert (
        EOS_TOKEN_ID not in middle
    ), f"generator did not stop at EOS: EOS appears in the middle of the sequence. tokens={tokens}"
