# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full prompt->audio CFG fidelity: TT generate_song vs the genuine reference chain.

This closes the coverage gap between:
  - test_generate_song_api.py  -> shape/finite/non-silent smoke only (no numerical gate)
  - test_cfg_guidance.py       -> CFG denoise + VAE with RANDOM conditioning (not the real encoders)
  - test_full_pipeline_real_weights.py -> ConditionEncoder->DiT with RANDOM inputs, no CFG, no VAE

Here we exercise the ENTIRE customer path end to end on a real tokenized prompt+lyrics:
    tokenizer -> TT text encoder -> TT ConditionEncoder -> TT CFG denoise (APG) -> TT VAE decode
and compare the 48 kHz audio to the identical REFERENCE chain:
    tokenizer -> HF text encoder -> reference ConditionEncoder -> reference CFG denoise -> diffusers VAE.

Precision gate: same rationale as test_cfg_guidance — under CFG (gs=7) the bf16<->fp32 dtype gap is
~0.86 (the model's own bf16 mode can't beat it), so we measure the bf16 floor in-situ and require TT
to match the fp32 reference AT LEAST AS WELL AS the model's own bf16 does (with a small tolerance).
This guards the real encoder->CFG integration (which the other tests don't cover) without gating the
irreducible dtype gap.
"""

import pytest
import torch
import ttnn

from loguru import logger
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.common.utility_functions import comp_pcc
from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import (
    have_pipeline,
    load_module_weights,
    load_state_dict,
    vae_dir,
)
from models.experimental.acestep.tt.apg_guidance import MomentumBuffer, apg_forward
from models.experimental.acestep.tt.model_config import AceStepModelConfig, _build_qwen3_encoder
from models.experimental.acestep.tt.pipeline import AceStepPipeline, create_tt_pipeline
from models.experimental.acestep.tests.test_utils import require_single_device

NUM_DIT_LAYERS = 24
INFER_STEPS = 30
GUIDANCE_SCALE = 7.0
SHIFT = 1.0  # reference generate_audio default
SECONDS = 5.12  # -> seq_len 128 (t'=64), small + fast
PROMPT = "energetic synthwave, punchy drums, driving bass, bright synth lead"
LYRICS = "[verse]\nneon lights over the city tonight\n[chorus]\nwe are electric we are alive"


def _reference_condition(pipe, hf_te, ce, m, hf):
    """Build the reference DiT cross-attn context from the real tokenized prompt/lyrics via the HF
    text encoder + reference ConditionEncoder (timbre = silence[:, :512], CLS-sliced inside the ref)."""
    metas = AceStepPipeline._metadata_string(bpm=118, keyscale="C major", timesignature="4", audio_duration=SECONDS)
    ftext = AceStepPipeline._SFT_GEN_PROMPT.format(AceStepPipeline._DIT_INSTRUCTION, PROMPT, metas)
    flyr = f"# Languages\nen\n\n# Lyric\n{LYRICS}<|endoftext|>"
    tids = torch.tensor([pipe.tokenizer(ftext, truncation=True, max_length=256).input_ids])
    lids = torch.tensor([pipe.tokenizer(flyr, truncation=True, max_length=2048).input_ids])
    with torch.no_grad():
        th = hf_te(input_ids=tids).last_hidden_state.float()
        lh = hf_te.embed_tokens(lids).float()
        timbre_in = pipe._silence_latent[:, :512, :]
        order = torch.zeros(1, dtype=torch.long)
        ref_ctx, _ = ce(
            text_hidden_states=th,
            text_attention_mask=torch.ones(1, tids.shape[1]),
            lyric_hidden_states=lh,
            lyric_attention_mask=torch.ones(1, lids.shape[1]),
            refer_audio_acoustic_hidden_states_packed=timbre_in,
            refer_audio_order_mask=order,
        )
    return ref_ctx


def _ref_cfg_audio(ref_dit, ref_vae, ref_ctx, null_emb, noise, context, cast=None):
    """Reference CFG ODE denoise (APG, dims=[1]) on a real conditioning context -> VAE audio."""
    enc_unc = null_emb.reshape(1, 1, -1).expand(1, ref_ctx.shape[1], -1).contiguous()
    t = torch.linspace(1.0, 0.0, INFER_STEPS + 1)
    if SHIFT != 1.0:
        t = SHIFT * t / (1 + (SHIFT - 1) * t)
    mb = MomentumBuffer()
    c = (lambda x: x.to(cast)) if cast is not None else (lambda x: x)
    xt = noise
    with torch.no_grad():
        for i in range(INFER_STEPS):
            tc = t[i].reshape(1)
            vc = ref_dit(hidden_states=c(xt), timestep=c(tc), timestep_r=c(tc), attention_mask=None,
                         encoder_hidden_states=c(ref_ctx), encoder_attention_mask=None, context_latents=c(context))[0].float()
            vu = ref_dit(hidden_states=c(xt), timestep=c(tc), timestep_r=c(tc), attention_mask=None,
                         encoder_hidden_states=c(enc_unc), encoder_attention_mask=None, context_latents=c(context))[0].float()
            vt = apg_forward(vc, vu, GUIDANCE_SCALE, momentum_buffer=mb, dims=[1])
            xt = xt - vt * (t[i] - t[i + 1])
        return ref_vae.decode(xt.transpose(1, 2)).sample


@pytest.mark.slow
@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step pipeline bundle not downloaded")
def test_generate_song_cfg_fidelity(device):
    require_single_device(device)
    from diffusers import AutoencoderOobleck

    args = AceStepModelConfig.from_hf(num_hidden_layers=NUM_DIT_LAYERS)
    pipe = create_tt_pipeline(args, device)  # full text-to-music pipeline (encoders + VAE)

    m = load_modeling_module()
    hf = load_config()
    hf._attn_implementation = "eager"
    hf.num_hidden_layers = NUM_DIT_LAYERS
    ce = m.AceStepConditionEncoder(hf).eval()
    load_module_weights(ce, "encoder.")
    ref_dit = m.AceStepDiTModel(hf).eval()
    load_module_weights(ref_dit, "decoder.", allow_extra=True)
    ref_dit_bf16 = m.AceStepDiTModel(hf).eval()
    load_module_weights(ref_dit_bf16, "decoder.", allow_extra=True)
    ref_dit_bf16 = ref_dit_bf16.to(torch.bfloat16)
    ref_vae = AutoencoderOobleck.from_pretrained(vae_dir()).eval()
    _, hf_te = _build_qwen3_encoder(device, "Qwen3-Embedding-0.6B", dtype=ttnn.bfloat16)

    sd = load_state_dict()
    null_emb = sd.get("null_condition_emb", sd.get("decoder.null_condition_emb")).float()

    # Shared noise + text2music context (silence src + all-valid chunk mask), matching generate_song.
    seq_len = pipe._latent_len(SECONDS)
    gen = torch.Generator().manual_seed(0)
    noise = torch.randn(1, seq_len, args.audio_acoustic_hidden_dim, generator=gen)
    sl = pipe._silence_latent
    src = sl[:, :seq_len, :] if sl.shape[1] >= seq_len else sl.repeat(1, (seq_len + sl.shape[1] - 1) // sl.shape[1], 1)[:, :seq_len, :]
    context = torch.cat([src, torch.ones(1, seq_len, args.audio_acoustic_hidden_dim)], dim=-1)

    # Reference conditioning + fp32 golden audio + bf16-floor audio.
    ref_ctx = _reference_condition(pipe, hf_te, ce, m, hf)
    ref_wav = _ref_cfg_audio(ref_dit, ref_vae, ref_ctx, null_emb, noise, context)
    bf16_wav = _ref_cfg_audio(ref_dit_bf16, ref_vae, ref_ctx, null_emb, noise, context, cast=torch.bfloat16)
    nb = min(ref_wav.shape[-1], bf16_wav.shape[-1])
    _, bf16_floor = comp_pcc(ref_wav[..., :nb], bf16_wav[..., :nb], 0.0)
    print(f"SONG_BF16_FLOOR_PCC: {bf16_floor:.6f}")

    # TT full chain via generate_song (same prompt/lyrics/metadata/seed/shift/guidance).
    tt_wav = pipe.generate_song(
        PROMPT, lyrics=LYRICS, seconds=SECONDS, infer_steps=INFER_STEPS, seed=0,
        guidance_scale=GUIDANCE_SCALE, shift=SHIFT, bpm=118, keyscale="C major", timesignature="4",
    )
    n = min(ref_wav.shape[-1], tt_wav.shape[-1])
    _, tt_pcc = comp_pcc(ref_wav[..., :n], tt_wav[..., :n], 0.0)
    logger.info(f"generate_song full-chain CFG audio: ref={ref_wav.shape[-1]} tt={tt_wav.shape[-1]}")
    print(f"SONG_CFG_PCC: {tt_pcc:.6f}")

    # TT (bf16, real encoders) must match the fp32 reference at least as well as the model's own bf16
    # DiT does (minus tolerance). This guards the real encoder->CFG->VAE integration end to end.
    bar = bf16_floor - 0.05
    assert tt_pcc >= bar, (
        f"full-chain generate_song CFG audio PCC {tt_pcc:.4f} below the bf16 bar {bar:.4f} "
        f"(bf16 floor {bf16_floor:.4f}); the real encoder->CFG->VAE chain regressed"
    )
