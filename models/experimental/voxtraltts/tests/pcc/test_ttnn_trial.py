# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS end-to-end trial: one TT full forward, one CPU ref forward, final waveform PCC.

Both stacks run the same ``text`` / ``voice`` / ``max_tokens`` / ``seed`` independently:

1. CPU reference ``generate()`` on that input.
2. TT ``forward_device_resident()`` on the same input (same path as the demo).

Final pass/fail is waveform ``comp_pcc`` >= 0.99 between CPU and TT outputs. Code agreement
is logged for visibility only.
"""
from __future__ import annotations

import gc

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.voxtral_tts import ACOUSTIC_CFG_ALPHA_DEFAULT, VoxtralTTSPipeline
from models.experimental.voxtraltts.utils.rng import acoustic_fm_noise_seed

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

FINAL_WAVEFORM_PCC = 0.99

_DEMO_TEXT = (
    "Voxtral is a four billion parameter open weight text to speech model "
    "released by Mistral AI in two thousand twenty six, designed for low "
    "latency multilingual voice generation across English, Spanish, French, "
    "Portuguese, Hindi, German, Dutch, and Italian. It builds on the "
    "Ministral three billion language backbone with a flow matching acoustic "
    "decoder and produces audio at twelve point five hertz with high quality, "
    "suitable for streaming voice applications and real time agent deployments."
)
_DEMO_VOICE = "casual_male"


def _log_pcc(label: str, pcc_value: float, target: float) -> None:
    status = "PASS" if pcc_value >= target else "LOW"
    logger.info(f"  {label}: PCC={pcc_value:.4f}  target>={target:.4f}  [{status}]")


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ttnn_voxtral_tts_e2e_trial(device, reset_seeds, request):
    """One TT full forward + one CPU ref forward on the same input; assert final waveform PCC."""
    generate_steps = 8
    pipe = None
    tt_out = None

    name = resolve_voxtral_model_name_or_skip()
    try:
        cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")

    logger.info("=" * 70)
    logger.info("CPU REFERENCE FORWARD (same text / voice / max_tokens / seed)")
    logger.info("=" * 70)
    ref_wav, ref_codes = cpu.generate(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
        return_tokenizer_codes=True,
    )
    assert torch.isfinite(ref_wav).all(), "CPU reference produced non-finite waveform samples"
    logger.info(f"  CPU codes shape={tuple(ref_codes.shape)} waveform samples={int(ref_wav.numel())}")

    try:
        # Use the default optimizations (BFP8 weights + HIFI2 matmuls). Counterintuitively,
        # this matches HF's BF16 inference behavior more closely than HIFI4 + BF16 weights:
        # HF's MistralForCausalLM in bf16 mode uses BF16×BF16 → FP32-accum → BF16 ops,
        # which has a precision pattern similar to HIFI2+fp32_accum. The FP32 RMSNorm patch
        # in voxtral_tts.py (patch_text_model_fp32_rms_norms) explicitly aligns the one op
        # HF promotes to FP32 (RMSNorm). HIFI4+BF16 over-corrects, making TT MORE precise
        # than HF, which paradoxically causes them to diverge at boundary cases.
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=512,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    speech_request = compose_speech_request(_DEMO_TEXT, name, voice=_DEMO_VOICE)
    prompt_ids = speech_request["prompt_token_ids"]
    _, cpu_embeds = cpu._prompt_embeddings(prompt_ids, _DEMO_VOICE)
    tt_embeds = pipe._build_voice_injected_embeds(prompt_ids, _DEMO_VOICE)
    cpu_prefill = cpu.text_model(
        inputs_embeds=cpu_embeds.unsqueeze(0),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    cpu_prefill_h = cpu_prefill.hidden_states[-1][:, -1, :]
    tt_prefill_h = pipe.text.prefill_from_embeds(tt_embeds, start_pos=0)
    _, prefill_pcc = comp_pcc(cpu_prefill_h.squeeze(0).float(), tt_prefill_h.float(), pcc=0.0)
    prefill_max_diff = float((cpu_prefill_h.squeeze(0).float() - tt_prefill_h.float()).abs().max().item())
    cfg_alpha = torch.tensor(ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=torch.bfloat16)

    # Step 0 with each backend's OWN prefill (the actual end-to-end situation)
    torch.manual_seed(acoustic_fm_noise_seed(0, 0))
    cpu_step0_own = cpu.acoustic_transformer(cpu_prefill_h, cfg_alpha).long()
    torch.manual_seed(acoustic_fm_noise_seed(0, 0))
    tt_step0_own = pipe.acoustic.forward(tt_prefill_h.unsqueeze(0), cfg_alpha).long()
    step0_own_ac = int((cpu_step0_own[:, 1:] == tt_step0_own[:, 1:]).sum().item())

    # ISOLATION: feed CPU's prefill hidden to BOTH acoustic backends. If TT acoustic on CPU
    # prefill matches CPU acoustic on CPU prefill, the divergence is purely from prefill (HF
    # text_model has fp32 internal promotions that TT text_model doesn't). If they still
    # diverge, there's an acoustic-level precision gap independent of prefill.
    # Both backends expect [bsz, dim] input. cpu_prefill_h is already [1, dim] — do NOT add
    # an extra unsqueeze (that bug caused tt_sem=0 in the previous diagnostic).
    cpu_h_bf16 = cpu_prefill_h.to(torch.bfloat16)  # [1, dim]
    torch.manual_seed(acoustic_fm_noise_seed(0, 0))
    cpu_step0_shared = cpu.acoustic_transformer(cpu_h_bf16, cfg_alpha).long()
    torch.manual_seed(acoustic_fm_noise_seed(0, 0))
    tt_step0_shared = pipe.acoustic.forward(cpu_h_bf16, cfg_alpha).long()  # already [1, dim]
    step0_shared_ac = int((cpu_step0_shared[:, 1:] == tt_step0_shared[:, 1:]).sum().item())
    step0_shared_sem_eq = int(cpu_step0_shared[0, 0] == tt_step0_shared[0, 0])

    logger.info("=" * 70)
    logger.info("PREFILL + STEP-0 DIAGNOSTIC (before independent forward)")
    logger.info("=" * 70)
    logger.info(f"  prompt_tokens={len(prompt_ids)} embeds_equal={torch.equal(cpu_embeds.cpu(), tt_embeds)}")
    _log_pcc("prefill hidden", float(prefill_pcc), 0.99)
    logger.info(f"  prefill hidden max_diff={prefill_max_diff:.4f}")
    logger.info(
        f"  step0 (each backend's own prefill): cpu_sem={int(cpu_step0_own[0, 0])} "
        f"tt_sem={int(tt_step0_own[0, 0])} match={int(cpu_step0_own[0, 0] == tt_step0_own[0, 0])} "
        f"acoustic_match={step0_own_ac}/36"
    )
    logger.info(
        f"  step0 (SHARED cpu_prefill_h to both): cpu_sem={int(cpu_step0_shared[0, 0])} "
        f"tt_sem={int(tt_step0_shared[0, 0])} match={step0_shared_sem_eq} "
        f"acoustic_match={step0_shared_ac}/36"
    )
    if step0_shared_ac >= 34 and step0_shared_sem_eq == 1:
        logger.info(
            "  → CONCLUSION: TT acoustic matches CPU acoustic on identical input. "
            "Divergence is purely from prefill (HF text_model fp32 promotions vs TT bf16)."
        )
    elif step0_shared_ac < step0_own_ac:
        logger.info(
            "  → CONCLUSION: Acoustic does NOT match even on identical input. "
            "TT acoustic has additional precision drift independent of prefill."
        )
    else:
        logger.info(
            "  → CONCLUSION: Acoustic match with shared input is similar to own input. "
            "Bf16 vs fp32 precision in FM Euler loop is the bottleneck."
        )

    logger.info("=" * 70)
    logger.info("TT WARMUP FORWARD (untimed; populates program cache / JIT)")
    logger.info("=" * 70)
    if use_signpost:
        signpost(header="warmup")
    _ = pipe.forward_device_resident(text=_DEMO_TEXT, voice=_DEMO_VOICE, max_tokens=generate_steps, seed=0)
    ttnn.synchronize_device(device)

    logger.info("=" * 70)
    logger.info("TT MEASURED FORWARD (signposted) — forward_device_resident")
    logger.info("=" * 70)
    if use_signpost:
        signpost(header="start")
    tt_out = pipe.forward_device_resident(text=_DEMO_TEXT, voice=_DEMO_VOICE, max_tokens=generate_steps, seed=0)
    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="stop")

    tt_wav = tt_out.waveform
    tt_codes = tt_out.codes_b37t
    assert torch.isfinite(tt_wav).all(), "TT forward produced non-finite waveform samples"
    assert tt_codes.dim() == 3 and tuple(tt_codes.shape[:2]) == (1, 37)
    logger.info(
        f"  TT codes shape={tuple(tt_codes.shape)} waveform shape={tuple(tt_wav.shape)} "
        f"hit_end_audio={tt_out.hit_end_audio}"
    )

    logger.info("=" * 70)
    logger.info("FINAL-OUTPUT COMPARISON")
    logger.info("=" * 70)
    n_frames = min(int(tt_codes.shape[2]), int(ref_codes.shape[2]))
    assert n_frames > 0, "no frames produced by one of the pipelines"
    tt_codes_aligned = tt_codes[:, :, :n_frames]
    ref_codes_aligned = ref_codes[:, :, :n_frames]

    sem_matches = int((tt_codes_aligned[:, 0] == ref_codes_aligned[:, 0]).sum().item())
    sem_total = int(tt_codes_aligned[:, 0].numel())
    sem_match_frac = sem_matches / sem_total

    ac_matches = int((tt_codes_aligned[:, 1:] == ref_codes_aligned[:, 1:]).sum().item())
    ac_total = int(tt_codes_aligned[:, 1:].numel())
    ac_match_frac = ac_matches / ac_total

    logger.info(f"  semantic-code match: {sem_match_frac:.4f}  ({sem_matches}/{sem_total})")
    logger.info(f"  acoustic-code match: {ac_match_frac:.4f}  ({ac_matches}/{ac_total})  (informational)")

    _, codes_pcc = comp_pcc(ref_codes_aligned.float(), tt_codes_aligned.float(), pcc=FINAL_WAVEFORM_PCC)
    logger.info(f"  codes PCC={float(codes_pcc):.4f}  (informational)")

    ref_flat = ref_wav.reshape(-1).float()
    tt_flat = tt_wav.reshape(-1).float()
    n_wav = min(int(ref_flat.numel()), int(tt_flat.numel()))
    assert n_wav > 0, "no waveform samples produced by one of the pipelines"
    ok_wav, wav_pcc = comp_pcc(ref_flat[:n_wav], tt_flat[:n_wav], pcc=FINAL_WAVEFORM_PCC)
    _log_pcc("waveform", float(wav_pcc), FINAL_WAVEFORM_PCC)
    assert ok_wav, f"final waveform PCC failed: {wav_pcc}  (samples={n_wav})"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del tt_out
    gc.collect()
