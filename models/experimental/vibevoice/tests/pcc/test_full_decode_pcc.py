# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode PCC: pinned reference diffusion conditions vs TT post-diffusion chain.

``test_decode_ref_cond_frame_pcc`` pins reference LM conditions (pos/neg hidden) and shared
initial noise at each diffusion frame, then runs the TT post-diffusion chain (acoustic decode
→ semantic encode → connectors → LM) and gates fused embed and per-frame LM hidden vs the
reference-backed run under the same pinned diffusion inputs.
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import MODEL_PATH, TEXT_EXAMPLES_DIR, VOICES_DIR
from models.experimental.vibevoice.common.resource_utils import load_script
from models.experimental.vibevoice.tests.pcc.lm_pcc_common import PCC_THRESHOLD
from models.experimental.vibevoice.tt.ttnn_dpm_scheduler import (
    TTDPMSolverMultistepScheduler,
    sample_speech_latents,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_TEXT_PATH = TEXT_EXAMPLES_DIR / "2p_short.txt"
_VOICE_PATH = VOICES_DIR / "en-Alice_woman.wav"
CFG_SCALE = 1.3
NUM_DIFFUSION_STEPS = 10
# Post-prefill AR steps to generate for the teacher stream (kept small for CI; the
# reliable gate is the leading pre-diffusion step, so a short stream suffices).
MAX_NEW_TOKENS = 24


def _voice_path() -> str:
    if _VOICE_PATH.is_file():
        return str(_VOICE_PATH)
    wavs = list(VOICES_DIR.glob("*.wav"))
    assert wavs, f"No voice WAV in {VOICES_DIR}"
    return str(wavs[0])


def _build_processor_batch():
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    assert _TEXT_PATH.is_file(), f"Missing demo text: {_TEXT_PATH}"
    script = load_script(_TEXT_PATH)
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    inputs = processor(
        text=[script],
        voice_samples=[[_voice_path()]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return processor, inputs


def _hidden_to_flat(hidden: ttnn.Tensor) -> torch.Tensor:
    """Single decode-position LM hidden (ttnn [1,1,1,H] or [1,1,H]) → 1-D float32 [H]."""
    return ttnn.to_torch(hidden).to(torch.float32).reshape(-1)


def _fused_to_flat(fused: ttnn.Tensor) -> torch.Tensor:
    """Fused next-step embed (ttnn [1,1,1,H]) → 1-D float32 [H]."""
    return ttnn.to_torch(fused).to(torch.float32).reshape(-1).clone()


def _predraw_diffusion_noises(num_slots: int, latent_size: int = 64) -> list[torch.Tensor]:
    """Pre-draw ``[2, 1, 1, latent_size]`` bf16 noise tensors (fp32 draw → bf16 cast)."""
    rng = torch.Generator()
    rng.manual_seed(0)
    return [
        torch.randn(2, 1, 1, latent_size, dtype=torch.float32, generator=rng).to(torch.bfloat16).clone()
        for _ in range(num_slots)
    ]


def _latent_to_tt(lat: torch.Tensor, device, latent_size: int = 64) -> ttnn.Tensor:
    return ttnn.as_tensor(
        lat.view(1, 1, 1, latent_size).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _wrap_capture(generator):
    """Record the positive LM hidden state at prefill + every decode step.

    Wraps the generator's LM entry points (``_lm_prefill`` / ``_lm_step`` /
    ``_lm_decode_token``) so both backends are captured identically.  The negative-CFG
    LM passes go through ``_neg_lm_step`` and are intentionally *not* captured.
    """
    captured = {"prefill": None, "decode": []}
    orig_prefill = generator._lm_prefill
    orig_step = generator._lm_step
    orig_decode = generator._lm_decode_token

    def prefill(inputs_embeds, kv_cache):
        logits, hidden = orig_prefill(inputs_embeds, kv_cache)
        captured["prefill"] = _hidden_to_flat(hidden)
        return logits, hidden

    def step(inputs_embeds, start_pos, kv_cache):
        logits, hidden = orig_step(inputs_embeds, start_pos, kv_cache)
        captured["decode"].append(_hidden_to_flat(hidden))
        return logits, hidden

    def decode_token(token_id, start_pos, kv_cache):
        logits, hidden = orig_decode(token_id, start_pos, kv_cache)
        captured["decode"].append(_hidden_to_flat(hidden))
        return logits, hidden

    generator._lm_prefill = prefill
    generator._lm_step = step
    generator._lm_decode_token = decode_token
    return captured


def _run_backend(generator, *, input_ids, attention_mask, speech_input_mask, prefill_speech_embeds, forced):
    """Run one decode backend (seed-aligned) and return (captured_hiddens, sequences)."""
    captured = _wrap_capture(generator)
    torch.manual_seed(0)
    out = generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        speech_input_mask=speech_input_mask,
        prefill_speech_embeds=prefill_speech_embeds,
        max_new_tokens=None if forced is not None else MAX_NEW_TOKENS,
        forced_token_ids=forced,
    )
    return captured, out.sequences


def _condition_to_torch(condition: ttnn.Tensor) -> torch.Tensor:
    """LM hidden slice used as diffusion condition → float32 [H]."""
    return ttnn.to_torch(condition).to(torch.float32).reshape(-1).clone()


def _condition_to_tt(condition: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.as_tensor(
        condition.reshape(1, 1, 1, -1).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@torch.no_grad()
def _reference_sample_speech_latents(
    prediction_head,
    noise_scheduler,
    cond_pos: torch.Tensor,
    cond_neg: torch.Tensor,
    initial_noise: torch.Tensor,
    cfg_scale: float,
    num_steps: int,
) -> torch.Tensor:
    """Full DPM loop on the reference head — mirrors ``sample_speech_tokens`` with injected noise."""
    noise_scheduler.set_timesteps(num_steps)
    head_device = next(prediction_head.parameters()).device
    cond_pos = cond_pos.reshape(-1).to(device=head_device, dtype=torch.float32)
    cond_neg = cond_neg.reshape(-1).to(device=head_device, dtype=torch.float32)
    condition = torch.stack([cond_pos, cond_neg], dim=0)

    initial_noise = initial_noise.reshape(-1).to(device=head_device, dtype=torch.float32)
    latent_size = initial_noise.numel()
    speech = torch.empty(2, latent_size, device=head_device, dtype=torch.float32)
    speech[0] = initial_noise
    speech[1] = initial_noise

    for t in noise_scheduler.timesteps:
        half = speech[:1]
        combined = torch.cat([half, half], dim=0)
        # Match ``sample_speech_tokens``: cast timesteps to the latent dtype (float32).
        # ``TimestepEmbedder`` ends with ``embedding.to(t.dtype)`` — if t stays long, cos/sin
        # truncate to zero and the head sees all-zero timestep features.
        t_rep = t.repeat(combined.shape[0]).to(device=combined.device, dtype=combined.dtype)
        eps = prediction_head(combined, t_rep, condition=condition)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        speech = noise_scheduler.step(eps, t, speech).prev_sample

    return speech[:1].reshape(-1)


def _tt_sample_speech_latents(
    diffusion_head,
    device,
    cond_pos: torch.Tensor,
    cond_neg: torch.Tensor,
    initial_noise_bf16: torch.Tensor,
    cfg_scale: float,
    num_steps: int,
) -> ttnn.Tensor:
    """Full DPM loop on TT head + scheduler with the same conditions/noise as the reference path."""
    scheduler = TTDPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        solver_order=2,
        prediction_type="v_prediction",
    )
    cond_pos_tt = _condition_to_tt(cond_pos, device)
    cond_neg_tt = _condition_to_tt(cond_neg, device)
    initial_tt = ttnn.as_tensor(
        initial_noise_bf16.view(1, 1, 1, -1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return sample_speech_latents(
        diffusion_head,
        cond_pos_tt,
        cond_neg_tt,
        scheduler,
        initial_tt,
        cfg_scale=cfg_scale,
        num_steps=num_steps,
    )


def _capture_ref_diffusion_pipeline(ref_gen, *, common, forced, ref_model, pre_noises):
    """Reference-backed forced decode: per-frame pos/neg/noise/latent/fused/hidden."""
    frames = []
    after_fused = [False]
    noise_idx = [0]
    prediction_head = ref_model.model.prediction_head
    noise_scheduler = ref_model.model.noise_scheduler

    orig_post = ref_gen._post_diffusion_embeds
    orig_step = ref_gen._lm_step

    def diff(condition, neg_condition, latent_size=64, noise_2x=None, rng=None):
        i = noise_idx[0]
        noise_idx[0] += 1
        noise = pre_noises[i]
        pos = _condition_to_torch(condition)
        neg = _condition_to_torch(neg_condition)
        lat = _reference_sample_speech_latents(
            prediction_head,
            noise_scheduler,
            pos,
            neg,
            noise[:1].to(torch.float32).reshape(-1),
            ref_gen.cfg_scale,
            ref_gen.num_diffusion_steps,
        )
        frames.append({"pos": pos, "neg": neg, "noise": noise.clone(), "ref_latent": lat.clone()})
        return _latent_to_tt(lat, ref_gen.device, latent_size)

    def post(lat):
        fused, audio = orig_post(lat)
        frames[-1]["ref_fused"] = _fused_to_flat(fused)
        after_fused[0] = True
        return fused, audio

    def step(inputs_embeds, start_pos, kv_cache):
        logits, hidden = orig_step(inputs_embeds, start_pos, kv_cache)
        if after_fused[0]:
            frames[-1]["ref_hidden"] = _hidden_to_flat(hidden)
            after_fused[0] = False
        return logits, hidden

    ref_gen._run_speech_diffusion = diff
    ref_gen._post_diffusion_embeds = post
    ref_gen._lm_step = step
    torch.manual_seed(0)
    ref_gen.generate(forced_token_ids=forced, max_new_tokens=None, **common)
    return frames


def _capture_tt_pinned_ref_conditions(tt_gen, ref_frames, *, common, forced):
    """Pure-TT forced decode with diffusion inputs pinned to reference frames."""
    frames = []
    after_fused = [False]
    pin_idx = [0]

    orig_post = tt_gen._post_diffusion_embeds
    orig_step = tt_gen._lm_step

    def diff(condition, neg_condition, latent_size=64, noise_2x=None, rng=None):
        i = pin_idx[0]
        pin_idx[0] += 1
        fr = ref_frames[i]
        lat_tt = _tt_sample_speech_latents(
            tt_gen.diffusion_head,
            tt_gen.device,
            fr["pos"],
            fr["neg"],
            fr["noise"][:1],
            tt_gen.cfg_scale,
            tt_gen.num_diffusion_steps,
        )
        frames.append({})
        return lat_tt

    def post(lat):
        fused, audio = orig_post(lat)
        frames[-1]["tt_fused"] = _fused_to_flat(fused)
        after_fused[0] = True
        return fused, audio

    def step(inputs_embeds, start_pos, kv_cache):
        logits, hidden = orig_step(inputs_embeds, start_pos, kv_cache)
        if after_fused[0]:
            frames[-1]["tt_hidden"] = _hidden_to_flat(hidden)
            after_fused[0] = False
        return logits, hidden

    tt_gen._run_speech_diffusion = diff
    tt_gen._post_diffusion_embeds = post
    tt_gen._lm_step = step
    torch.manual_seed(0)
    tt_gen.generate(forced_token_ids=forced, max_new_tokens=None, **common)
    return frames


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_decode_ref_cond_frame_pcc(mesh_device):
    """Pinned reference diffusion conditions → TT post-diffusion fused embed / hidden PCC.

    For each ``speech_diffusion_id`` frame in a teacher-forced decode:
      1. Reference-backed run records the live ref pos/neg LM hidden states, shared initial
         noise, and the reference post-diffusion fused embed + LM hidden.
      2. Pure-TT run replays the same token stream but pins diffusion to those reference
         conditions/noise, then runs the TT acoustic decode → semantic encode → connectors
         → LM on the TT denoised latent.
      3. Gate per-frame fused-embed PCC (>= 0.99) and LM-hidden PCC (>= 0.99) — isolates
         post-diffusion drift while keeping diffusion inputs identical.
    """
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

    processor, inputs = _build_processor_batch()
    prefill_len = inputs["input_ids"].shape[1]

    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    ref_model.eval()
    ref_model.set_ddpm_inference_steps(num_steps=NUM_DIFFUSION_STEPS)
    ref_model.model.acoustic_tokenizer.std_dist_type = "none"

    with torch.no_grad():
        _, prefill_speech_embeds = ref_model._process_speech_inputs(
            inputs["speech_tensors"].to(ref_model.dtype),
            inputs["speech_masks"],
        )
    prefill_speech_embeds = prefill_speech_embeds.to(torch.float32)

    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device,
        MODEL_PATH,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
    )
    tt_model.set_speech_scale_bias(
        ref_model.model.speech_scaling_factor.item(),
        ref_model.model.speech_bias_factor.item(),
    )

    common = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        speech_input_mask=inputs["speech_input_mask"],
        prefill_speech_embeds=prefill_speech_embeds,
    )

    tt_gen = tt_model._make_generator(
        processor.tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    _, tt_sequences = _run_backend(tt_gen, forced=None, **common)
    forced = tt_sequences[0, prefill_len:].reshape(-1)
    assert forced.numel() > 0, "TT greedy decode produced no tokens"

    pre_noises = _predraw_diffusion_noises(MAX_NEW_TOKENS + 8)

    ref_gen = tt_model._make_generator(
        processor.tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=None,
        ref_inference=ref_model,
    )
    ref_frames = _capture_ref_diffusion_pipeline(
        ref_gen, common=common, forced=forced, ref_model=ref_model, pre_noises=pre_noises
    )

    tt_gen2 = tt_model._make_generator(
        processor.tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=None,
    )
    tt_frames = _capture_tt_pinned_ref_conditions(tt_gen2, ref_frames, common=common, forced=forced)

    n_frames = min(len(ref_frames), len(tt_frames))
    assert n_frames > 0, "no diffusion frames captured"
    assert len(ref_frames) == len(
        tt_frames
    ), f"ref/TT diffusion frame count mismatch: {len(ref_frames)} vs {len(tt_frames)}"

    fused_pccs = []
    hidden_pccs = []
    for i in range(n_frames):
        assert "ref_fused" in ref_frames[i] and "tt_fused" in tt_frames[i]
        fused_pccs.append(comp_pcc(ref_frames[i]["ref_fused"], tt_frames[i]["tt_fused"], pcc=0.0)[1])
        assert "ref_hidden" in ref_frames[i] and "tt_hidden" in tt_frames[i]
        hidden_pccs.append(comp_pcc(ref_frames[i]["ref_hidden"], tt_frames[i]["tt_hidden"], pcc=0.0)[1])

    min_fused = min(fused_pccs)
    min_hidden = min(hidden_pccs)
    mean_fused = sum(fused_pccs) / len(fused_pccs)
    mean_hidden = sum(hidden_pccs) / len(hidden_pccs)

    print(
        f"[test_decode_ref_cond_frame_pcc] frames={n_frames} steps={NUM_DIFFUSION_STEPS} "
        f"min_fused_PCC={min_fused:.6f} mean_fused_PCC={mean_fused:.6f} "
        f"min_hidden_PCC={min_hidden:.6f} mean_hidden_PCC={mean_hidden:.6f} threshold={PCC_THRESHOLD}"
    )
    print(
        "[test_decode_ref_cond_frame_pcc] per-frame fused PCC: "
        + " ".join(f"{i}:{p:.4f}" for i, p in enumerate(fused_pccs))
    )
    print(
        "[test_decode_ref_cond_frame_pcc] per-frame hidden PCC: "
        + " ".join(f"{i}:{p:.4f}" for i, p in enumerate(hidden_pccs))
    )

    assert min_fused >= PCC_THRESHOLD, (
        f"pinned-ref-cond fused-embed min PCC {min_fused:.6f} < {PCC_THRESHOLD} "
        f"(TT post-diffusion chain: acoustic decode / semantic encode / connectors)"
    )
    assert min_hidden >= PCC_THRESHOLD, (
        f"pinned-ref-cond LM hidden min PCC {min_hidden:.6f} < {PCC_THRESHOLD} "
        f"(TT LM on TT fused embed vs ref LM on ref fused embed, same pinned diffusion inputs)"
    )
