# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""ISL sweep for the XTTS-v2 GPT prefill chain, and a max_seq sweep for decode.

The GPT has two position tables (text 404, mel 608). Max text and max codes together are
unreachable (``text + codes ≤ 980``). This file sweeps the text/prefill axis to 384 (last tile
multiple under 404) and decode cache depth separately.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/pcc/test_gpt_isl_sweep.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.xtts.reference.xtts_conditioning import (
    load_reference_audio,
    reference_conditioning,
    wav_to_mel,
)
from models.experimental.xtts.reference.xtts_gpt_block import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
    build_causal_mask,
)
from models.experimental.xtts.reference.xtts_gpt_generate import wrap_text_ids
from models.experimental.xtts.reference.xtts_gpt_model import reference_gpt_model
from models.experimental.xtts.reference.xtts_gpt_stack import reference_gpt_stack
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel

TILE = 32

# Text/prefill axis: tile multiples up to the 404-row text position table (384 is the last one).
ISL_SWEEP = [32, 64, 96, 128, 192, 256, 320, 384]

# Cache-depth axis for decode: fixed 128-position prompt, cache sized for a range of code budgets.
# Every value MUST be a tile multiple — see the note in test_tt_gpt_decode_max_seq_sweep. 992 is the
# largest tile multiple inside the 1012 n_positions budget.
DECODE_PROMPT_TEXT_LEN = 96  # -> prompt_len 128 with the 32 conditioning latents
MAX_SEQ_SWEEP = [160, 384, 608, 992]

# Conditioning rows gated separately from text rows (fill_cache odd-tile bug; see xtts_gpt_model.py).
COND_ROW_PCC = 0.99


@pytest.fixture(scope="module")
def cond_latents(xtts_state_dict):
    """Real conditioning latents [1, 32, 1024] — independent of text length, so build once."""
    sd = xtts_state_dict
    wav = load_reference_audio(sample="en_sample.wav")
    mel = wav_to_mel(wav, sd["mel_stats"].cpu())
    with torch.no_grad():
        return reference_conditioning(sd)(mel).transpose(1, 2)  # [1, 1024, 32] -> [1, 32, 1024]


_SWEEP_TEXT = "the quick brown fox jumps over the lazy dog. " * 60


def _wrapped_text(text_len):
    """Real wrapped English text trimmed to exactly ``text_len`` ids.

    Not ``[STOP]``-padded: long STOP filler is degenerate and is not what inference feeds.
    """
    ids = wrap_text_ids(preprocess_text(_SWEEP_TEXT, lang="en"))
    assert ids.shape[1] >= text_len, f"_SWEEP_TEXT yields {ids.shape[1]} ids, need {text_len}"
    return ids[:, :text_len].contiguous()


def _split_heads(t):  # [1, S, hidden] -> [1, heads, S, head_dim]
    return t.view(1, -1, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()


def _reference_prompt_embeds(ref_model, wrapped, cond):
    """``[cond | text_emb]`` exactly as the model builds it (text positions 0..text_len-1)."""
    with torch.no_grad():
        text_pos = torch.arange(wrapped.shape[1])
        text_emb = ref_model.text_embedding(wrapped) + ref_model.text_pos_embedding(text_pos)
        return torch.cat([cond, text_emb], dim=1).float()  # [1, prompt_len, hidden]


def _reference_layer_kv(ref_stack, prompt_emb):
    """Per-layer (K, V) for the prompt, walking the reference stack layer by layer.

    K/V are per-position projections of each layer's INPUT hidden (``ln_1 -> c_attn``), so
    the prompt's K/V depend only on the prompt. Head split matches the TT path
    (``nlp_create_qkv_heads``, ``transpose_k_heads=False``): [q|k|v] blocks, head-major.
    """
    kvs = []
    with torch.no_grad():
        mask = build_causal_mask(prompt_emb.shape[1], prompt_emb.dtype)
        h = prompt_emb
        for blk in ref_stack.h:
            qkv = blk.attn.c_attn(blk.ln_1(h))
            _, k, v = qkv.split(HIDDEN_SIZE, dim=2)
            kvs.append((_split_heads(k), _split_heads(v)))
            out = blk(h, attention_mask=mask)
            h = out[0] if isinstance(out, tuple) else out
    return kvs


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_gpt_prefill_isl_sweep(device, xtts_state_dict, cond_latents, pcc, reset_seeds):
    """Full 30-layer prefill at each text ISL, gated on the seeded KV cache."""
    sd = xtts_state_dict
    cond = cond_latents
    n_cond = cond.shape[1]

    ref_model = reference_gpt_model(sd, num_layers=NUM_LAYERS)
    ref_stack = reference_gpt_stack(sd, num_layers=NUM_LAYERS)
    tt_model = TtXttsGptModel(sd, device, num_layers=NUM_LAYERS)
    cond_tt = ttnn.from_torch(cond.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    rows, failures = [], []
    for text_len in ISL_SWEEP:
        wrapped = _wrapped_text(text_len)
        prompt_len = n_cond + text_len
        assert prompt_len <= 1012, f"prompt_len {prompt_len} over the n_positions budget"

        prompt_emb = _reference_prompt_embeds(ref_model, wrapped, cond)
        ref_kv = _reference_layer_kv(ref_stack, prompt_emb)

        # Minimal tile-aligned cache: this test isolates PREFILL, so the cache only has to
        # hold the prompt (the decode cache-depth axis is the second test).
        max_seq = -(-(prompt_len + 1) // TILE) * TILE
        kv = tt_model.prefill(wrapped, cond_tt, max_seq)

        k_meds, v_meds, k_cond, v_cond = [], [], [], []
        for layer, (ref_k, ref_v) in enumerate(ref_kv):
            tt_k = ttnn.to_torch(kv[layer][0]).float()[:, :, :prompt_len, :]
            tt_v = ttnn.to_torch(kv[layer][1]).float()[:, :, :prompt_len, :]
            # Per-position PCC, then median. A flat PCC mixes the 32 uploaded conditioning rows
            # with GPT-computed text rows and is dominated by the former.
            kp = torch.tensor([float(comp_pcc(ref_k[:, :, p, :], tt_k[:, :, p, :], pcc)[1]) for p in range(prompt_len)])
            vp = torch.tensor([float(comp_pcc(ref_v[:, :, p, :], tt_v[:, :, p, :], pcc)[1]) for p in range(prompt_len)])
            k_meds.append(float(kp.median()))
            v_meds.append(float(vp.median()))
            k_cond.append(float(kp[:n_cond].min()))  # reported, not gated
            v_cond.append(float(vp[:n_cond].min()))

        k_med, v_med = min(k_meds), min(v_meds)  # worst layer's median

        # Decode latent off this prefill — what the vocoder consumes.
        with torch.no_grad():
            mel_emb = ref_model.mel_embedding(torch.tensor([[0]])) + ref_model.mel_pos_embedding(torch.arange(1))
            ref_latent = ref_model.final_norm(ref_stack(torch.cat([prompt_emb, mel_emb.float()], dim=1))[:, -1:, :])
        _, tt_latent = tt_model.decode_on_device(
            tt_model._pos_ids(0), tt_model._pos_ids(0), tt_model.cache_pos(prompt_len), kv, write_idx=prompt_len
        )
        lat_pcc = float(comp_pcc(ref_latent, ttnn.to_torch(tt_latent).float().reshape(1, 1, HIDDEN_SIZE), pcc)[1])

        rows.append((text_len, prompt_len, max_seq, lat_pcc, k_med, v_med, min(k_cond), min(v_cond)))
        logger.info(
            f"ISL text_len={text_len:>3} prompt_len={prompt_len:>3} max_seq={max_seq:>4}: "
            f"decode latent={lat_pcc:.6f} | cache K med={k_med:.6f} V med={v_med:.6f} "
            f"(worst layer {k_meds.index(k_med)}) | cond-row min K={min(k_cond):.4f}"
        )
        if lat_pcc < pcc:
            failures.append(f"text_len={text_len} decode latent PCC {lat_pcc:.6f} < {pcc}")
        if k_med < pcc:
            failures.append(f"text_len={text_len} cache K per-position median {k_med:.6f} < {pcc}")
        if v_med < pcc:
            failures.append(f"text_len={text_len} cache V per-position median {v_med:.6f} < {pcc}")
        # fill_cache odd-tile regression (see tt/xtts_gpt_model.py).
        if min(k_cond) < COND_ROW_PCC:
            failures.append(
                f"text_len={text_len} prompt_len={prompt_len} ({prompt_len // TILE} tiles) "
                f"conditioning-row K min {min(k_cond):.6f} < {COND_ROW_PCC}"
            )

    logger.info(
        f"\n{'text_len':>9}{'prompt':>8}{'tiles':>7}{'max_seq':>9}{'decode lat':>12}"
        f"{'K med':>10}{'V med':>10}{'K cond':>9}  (gated: decode lat, K cond)"
    )
    for text_len, plen, ms, lat, k_med, v_med, kc, vc in rows:
        logger.info(f"{text_len:>9}{plen:>8}{plen // TILE:>7}{ms:>9}{lat:>12.6f}{k_med:>10.6f}{v_med:>10.6f}{kc:>9.4f}")

    assert not failures, "GPT prefill ISL sweep failures:\n" + "\n".join(failures)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_gpt_decode_max_seq_sweep(device, xtts_state_dict, cond_latents, pcc, reset_seeds):
    """One decode step at a fixed fill level, over a range of cache sizes.

    Decode attends over the entire fixed cache regardless of position, so a wrong mask
    would let the zeroed tail leak in — and would do so *differently* at each ``max_seq``.
    Gating on both PCC-vs-reference and invariance across ``max_seq`` catches that.

    **``max_seq`` must be a multiple of 32.** Bisecting this sweep showed the decode output
    is bit-identical across every tile-aligned cache size (160 … 992, 1024) and silently
    WRONG at every unaligned one (500 → max delta 1.75, 990 → 1.0, 1000 → 3.25, 1008 →
    2.5) — it is a property of alignment, not of length. Production always tile-aligns
    (``-(-(prompt_len + max_new_tokens + 1) // 32) * 32``), so this is an unguarded
    precondition rather than a live bug, but nothing rejects an unaligned ``max_seq``.
    """
    sd = xtts_state_dict
    cond = cond_latents
    prompt_len = cond.shape[1] + DECODE_PROMPT_TEXT_LEN
    wrapped = _wrapped_text(DECODE_PROMPT_TEXT_LEN)

    # Reference: the decode step's hidden is the causal stack over [prompt | one mel token],
    # taken at the last position. Build the mel token's embedding the way the model does.
    ref_model = reference_gpt_model(sd, num_layers=NUM_LAYERS)
    ref_stack = reference_gpt_stack(sd, num_layers=NUM_LAYERS)
    prompt_emb = _reference_prompt_embeds(ref_model, wrapped, cond)
    with torch.no_grad():
        mel_id = torch.tensor([[0]], dtype=torch.long)
        mel_emb = ref_model.mel_embedding(mel_id) + ref_model.mel_pos_embedding(torch.arange(1))
        enc = ref_stack(torch.cat([prompt_emb, mel_emb.float()], dim=1))[:, -1:, :]  # ln_f applied
        ref_hidden = ref_model.final_norm(enc)  # TT's `latent` is final_norm(ln_f(hidden))

    tt_model = TtXttsGptModel(sd, device, num_layers=NUM_LAYERS)
    cond_tt = ttnn.from_torch(cond.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    outs, failures = {}, []
    for max_seq in MAX_SEQ_SWEEP:
        assert max_seq % TILE == 0, f"max_seq {max_seq} is not tile-aligned (see the docstring)"
        if max_seq < prompt_len + 1:
            continue
        kv = tt_model.prefill(wrapped, cond_tt, max_seq)
        _, latent = tt_model.decode_on_device(
            tt_model._pos_ids(0), tt_model._pos_ids(0), tt_model.cache_pos(prompt_len), kv, write_idx=prompt_len
        )
        out = ttnn.to_torch(latent).float().reshape(1, 1, HIDDEN_SIZE)
        outs[max_seq] = out

        step_pcc = float(comp_pcc(ref_hidden, out, pcc)[1])
        logger.info(
            f"max_seq={max_seq:>5} (prompt_len={prompt_len}, {max_seq - prompt_len} unfilled): PCC {step_pcc:.6f}"
        )
        if step_pcc < pcc:
            failures.append(f"max_seq={max_seq} decode PCC {step_pcc:.6f} < {pcc}")

    # Invariance: the unfilled tail is masked, so the answer must not depend on cache size.
    sizes = sorted(outs)
    base = outs[sizes[0]]
    for ms in sizes[1:]:
        delta = (outs[ms] - base).abs().max().item()
        logger.info(f"max_seq={ms:>5} vs {sizes[0]}: max abs delta {delta:.3e}")
        if delta != 0.0:
            failures.append(f"decode output changed with max_seq ({sizes[0]} vs {ms}): max delta {delta:.3e}")

    assert not failures, "GPT decode max_seq sweep failures:\n" + "\n".join(failures)
