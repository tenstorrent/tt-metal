# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full Voxtral TTS pipeline PCC vs CPU reference (prefill, loop, waveform)."""

from __future__ import annotations

import gc

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_tokenizer_decode_reference
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

PREFILL_HIDDEN_PCC = 0.99
TEXT_DECODE_STEP_PCC = 0.99
ACOUSTIC_MATCH_FRAC = 0.88
WAVEFORM_PCC = 0.99

_DEMO_TEXT = "Hello from the Voxtral Tenstorrent demo."
_DEMO_VOICE = "casual_male"


def _log_stage_header(title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def _log_pcc(label: str, pcc_value: float, target: float) -> None:
    status = "PASS" if pcc_value >= target else "LOW"
    logger.info(f"  {label}: PCC={pcc_value:.4f}  target>={target:.4f}  [{status}]")


def _align_to_ref_shape(ref_t: torch.Tensor, tt_t: torch.Tensor) -> torch.Tensor:
    out = tt_t
    for dim, size in enumerate(ref_t.shape):
        if dim < out.dim() and out.shape[dim] > size:
            sl = [slice(None)] * out.dim()
            sl[dim] = slice(0, size)
            out = out[tuple(sl)]
    return out.reshape(ref_t.shape)


def _cpu_text_decode_step(
    cpu: VoxtralCPUReference,
    *,
    audio_codes_b37: torch.Tensor,
    past_key_values,
):
    next_input = cpu._audio_codes_to_input_embeds(audio_codes_b37)
    outputs = cpu.text_model(
        inputs_embeds=next_input,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
    return hidden, outputs.past_key_values


def _compare_acoustic_codes(
    pipe: VoxtralTTSPipeline,
    cpu: VoxtralCPUReference,
    *,
    hidden_bf16: torch.Tensor,
    cfg_alpha: torch.Tensor,
    rng_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    hidden_in = hidden_bf16.unsqueeze(0)
    torch.manual_seed(rng_seed)
    ref_codes = cpu.acoustic_transformer(hidden_in, cfg_alpha).long()
    torch.manual_seed(rng_seed)
    tt_codes = pipe.acoustic_codes_forward(hidden_in, cfg_alpha).long()
    tt_codes = _align_to_ref_shape(ref_codes, tt_codes)

    assert torch.equal(ref_codes[:, :1], tt_codes[:, :1]), "semantic token mismatch"
    acoustic_matches = int((ref_codes[:, 1:] == tt_codes[:, 1:]).sum().item())
    acoustic_total = int(ref_codes[:, 1:].numel())
    return ref_codes, tt_codes, acoustic_matches, acoustic_total


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ttnn_voxtral_tts_e2e_pcc(device, reset_seeds, request):
    """Full Voxtral TTS pipeline check: CPU reference vs TT stage outputs and final waveform."""
    generate_steps = 50
    tt_full = None
    tt_wav = None
    pipe = None
    name = resolve_voxtral_model_name_or_skip()
    try:
        pipe = VoxtralTTSPipeline.from_model_name(device, model_name_or_path=name, text_max_seq_len=512)
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    try:
        cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")

    _log_stage_header("0. FULL PIPELINE FORWARD SMOKE")
    tt_full = pipe.forward(text=_DEMO_TEXT, voice=_DEMO_VOICE, max_tokens=generate_steps, seed=0)
    ref_full_wav, ref_full_codes = cpu.generate(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
        return_tokenizer_codes=True,
    )
    assert tt_full.codes_b37t.dim() == 3 and tuple(tt_full.codes_b37t.shape[:2]) == (1, 37)
    assert torch.isfinite(tt_full.waveform).all(), "TT forward produced non-finite waveform samples"
    assert torch.isfinite(ref_full_wav).all(), "CPU reference produced non-finite waveform samples"
    logger.info(f"  CPU full codes shape={tuple(ref_full_codes.shape)} waveform samples={int(ref_full_wav.numel())}")
    logger.info(
        f"  TT forward codes shape={tuple(tt_full.codes_b37t.shape)} " f"waveform shape={tuple(tt_full.waveform.shape)}"
    )

    speech_request = compose_speech_request(_DEMO_TEXT, name, voice=_DEMO_VOICE, ref_audio=None)
    prompt_ids = speech_request["prompt_token_ids"]

    _log_stage_header("1. TEXT PREFILL HIDDEN PCC")
    _, cpu_embeds = cpu._prompt_embeddings(prompt_ids, _DEMO_VOICE)
    cpu_prefill = cpu.text_model(
        inputs_embeds=cpu_embeds.unsqueeze(0),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    cpu_hidden = cpu_prefill.hidden_states[-1][:, -1, :].squeeze(0).float()
    cpu_pkv = cpu_prefill.past_key_values

    tt_embeds = pipe._build_voice_injected_embeds(prompt_ids, _DEMO_VOICE)
    tt_hidden = pipe.text.prefill_from_embeds(tt_embeds, start_pos=0).float()
    ok, msg = comp_pcc(cpu_hidden, tt_hidden, pcc=PREFILL_HIDDEN_PCC)
    _log_pcc("prefill hidden", float(msg), PREFILL_HIDDEN_PCC)
    assert ok, f"prefill hidden PCC failed: {msg}"

    _log_stage_header("2. AUTOREGRESSIVE TEXT + ACOUSTIC LOOP")
    cfg_alpha = torch.tensor(cpu._acoustic_cfg_alpha, device=cpu_hidden.device, dtype=cpu.dtype)
    current_pos = len(prompt_ids)
    stacked_codes: list[torch.Tensor] = []
    acoustic_matches = 0
    acoustic_total = 0

    for step in range(generate_steps):
        _, tt_codes, step_matches, step_total = _compare_acoustic_codes(
            pipe,
            cpu,
            hidden_bf16=tt_hidden.to(torch.bfloat16),
            cfg_alpha=cfg_alpha,
            rng_seed=10_000 + step,
        )
        acoustic_matches += step_matches
        acoustic_total += step_total
        logger.info(
            f"  step={step} semantic={int(tt_codes[0, 0].item())} "
            f"acoustic_agreement={step_matches / step_total:.4f} ({step_matches}/{step_total})"
        )

        stacked_codes.append(tt_codes[0].detach().cpu())
        if int(tt_codes[0, 0].item()) == cpu.end_audio_id:
            break

        cpu_hidden, cpu_pkv = _cpu_text_decode_step(cpu, audio_codes_b37=tt_codes, past_key_values=cpu_pkv)
        mm_embed = pipe._audio_codes_to_mm_embed(tt_codes)
        tt_hidden = pipe.text.decode_step_from_embeds(mm_embed, current_pos).float()

        ok, msg = comp_pcc(cpu_hidden.float(), tt_hidden, pcc=TEXT_DECODE_STEP_PCC)
        _log_pcc(f"text hidden step={step} pos={current_pos}", float(msg), TEXT_DECODE_STEP_PCC)
        assert ok, f"text decode step={step} pos={current_pos} hidden PCC failed: {msg}"
        current_pos += 1

    assert stacked_codes, "pipeline produced no acoustic frames"
    match_frac = acoustic_matches / acoustic_total
    logger.info(
        f"  acoustic code agreement summary: {match_frac:.4f} "
        f"target>={ACOUSTIC_MATCH_FRAC:.4f} matched={acoustic_matches}/{acoustic_total}"
    )
    assert match_frac >= ACOUSTIC_MATCH_FRAC, f"acoustic code agreement {match_frac:.4f} < {ACOUSTIC_MATCH_FRAC}"

    _log_stage_header("3. FINAL WAVEFORM PCC FROM TT CODE STREAM")
    stacked = torch.stack(stacked_codes, dim=0)
    eoa = (stacked[:, 0] == cpu.end_audio_id).nonzero(as_tuple=False)
    cut = int(eoa[0].item()) if len(eoa) else stacked.shape[0]
    shifted = stacked[:cut]
    audio_tokens = shifted - 2
    codes_b37t = audio_tokens.T.unsqueeze(0).long()
    assert codes_b37t.shape[2] > 0, "pipeline reached end-audio before producing waveform codes"

    ref_wav = audio_tokenizer_decode_reference(
        codes_b37t, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args
    ).reshape(1, 1, -1)
    tt_wav = pipe.decode_waveform_from_codes_tt(codes_b37t).reshape(1, 1, -1)[:, :, : ref_wav.shape[-1]]
    ok, msg = comp_pcc(ref_wav.float(), tt_wav.float(), pcc=WAVEFORM_PCC)
    _log_pcc("waveform", float(msg), WAVEFORM_PCC)
    logger.info(f"  frames={codes_b37t.shape[2]} waveform shape={tuple(tt_wav.shape)}")
    assert ok, f"waveform PCC failed: {msg}"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del tt_full
    del tt_wav
    gc.collect()
