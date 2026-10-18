# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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

ISL_SWEEP = [32, 64, 96, 128, 192, 256, 320, 384]

DECODE_PROMPT_TEXT_LEN = 96
MAX_SEQ_SWEEP = [160, 384, 608, 992]

# Conditioning rows gated separately from text rows (fill_cache odd-tile bug; see xtts_gpt_model.py).
COND_ROW_PCC = 0.99


@pytest.fixture(scope="module")
def cond_latents(xtts_state_dict):
    """Module fixture providing reference conditioning latents for ISL sweeps."""
    sd = xtts_state_dict
    wav = load_reference_audio(sample="en_sample.wav")
    mel = wav_to_mel(wav, sd["mel_stats"].cpu())
    with torch.no_grad():
        return reference_conditioning(sd)(mel).transpose(1, 2)


_SWEEP_TEXT = "the quick brown fox jumps over the lazy dog. " * 60


def _wrapped_text(text_len):
    """Return wrapped text token ids truncated to the requested length."""
    ids = wrap_text_ids(preprocess_text(_SWEEP_TEXT, lang="en"))
    assert ids.shape[1] >= text_len, f"_SWEEP_TEXT yields {ids.shape[1]} ids, need {text_len}"
    return ids[:, :text_len].contiguous()


def _split_heads(t):
    """Reshape a [B, T, H] tensor into multi-head [B, heads, T, head_dim]."""
    return t.view(1, -1, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()


def _reference_prompt_embeds(ref_model, wrapped, cond):
    """Build concatenated cond+text prompt embeddings from the reference GPT."""
    with torch.no_grad():
        text_pos = torch.arange(wrapped.shape[1])
        text_emb = ref_model.text_embedding(wrapped) + ref_model.text_pos_embedding(text_pos)
        return torch.cat([cond, text_emb], dim=1).float()


def _reference_layer_kv(ref_stack, prompt_emb):
    """Compute per-layer reference K/V tensors for a prompt embedding."""
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
    """Sweep prefill input lengths and gate KV/latent PCC against reference."""
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

        max_seq = -(-(prompt_len + 1) // TILE) * TILE
        kv = tt_model.prefill(wrapped, cond_tt, max_seq)

        k_meds, v_meds, k_cond, v_cond = [], [], [], []
        for layer, (ref_k, ref_v) in enumerate(ref_kv):
            tt_k = ttnn.to_torch(kv[layer][0]).float()[:, :, :prompt_len, :]
            tt_v = ttnn.to_torch(kv[layer][1]).float()[:, :, :prompt_len, :]
            kp = torch.tensor([float(comp_pcc(ref_k[:, :, p, :], tt_k[:, :, p, :], pcc)[1]) for p in range(prompt_len)])
            vp = torch.tensor([float(comp_pcc(ref_v[:, :, p, :], tt_v[:, :, p, :], pcc)[1]) for p in range(prompt_len)])
            k_meds.append(float(kp.median()))
            v_meds.append(float(vp.median()))
            k_cond.append(float(kp[:n_cond].min()))
            v_cond.append(float(vp[:n_cond].min()))

        k_med, v_med = min(k_meds), min(v_meds)

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
    """Sweep decode max_seq sizes and assert PCC and max_seq-invariance."""
    sd = xtts_state_dict
    cond = cond_latents
    prompt_len = cond.shape[1] + DECODE_PROMPT_TEXT_LEN
    wrapped = _wrapped_text(DECODE_PROMPT_TEXT_LEN)

    ref_model = reference_gpt_model(sd, num_layers=NUM_LAYERS)
    ref_stack = reference_gpt_stack(sd, num_layers=NUM_LAYERS)
    prompt_emb = _reference_prompt_embeds(ref_model, wrapped, cond)
    with torch.no_grad():
        mel_id = torch.tensor([[0]], dtype=torch.long)
        mel_emb = ref_model.mel_embedding(mel_id) + ref_model.mel_pos_embedding(torch.arange(1))
        enc = ref_stack(torch.cat([prompt_emb, mel_emb.float()], dim=1))[:, -1:, :]
        ref_hidden = ref_model.final_norm(enc)

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

    sizes = sorted(outs)
    base = outs[sizes[0]]
    for ms in sizes[1:]:
        delta = (outs[ms] - base).abs().max().item()
        logger.info(f"max_seq={ms:>5} vs {sizes[0]}: max abs delta {delta:.3e}")
        if delta != 0.0:
            failures.append(f"decode output changed with max_seq ({sizes[0]} vs {ms}): max delta {delta:.3e}")

    assert not failures, "GPT decode max_seq sweep failures:\n" + "\n".join(failures)
