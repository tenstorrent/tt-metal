# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E PCC using the standard teacher-forced methodology.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_tokenizer_decode_reference
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
from models.experimental.voxtraltts.tests.common import (
    VOXTRAL_STANDARD_CHAR_TEXT,
    log_per_step_code_match,
    resolve_voxtral_model_name_or_skip,
)
from models.experimental.voxtraltts.demo.decode_trace_2cq import num_command_queues_for_decode
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_hf_aligned_optimizations
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline
from models.experimental.voxtraltts.utils.rng import acoustic_fm_noise_seed

os.environ.setdefault("VOXTRAL_DECODE_TRACE", "1")

WAVEFORM_PCC_TARGET = 0.99

_DEMO_TEXT = VOXTRAL_STANDARD_CHAR_TEXT
_DEMO_VOICE = "casual_male"


def _log_pcc(label: str, pcc_value: float, target: float) -> None:
    status = "PASS" if pcc_value >= target else "LOW"
    logger.info(f"  {label}: PCC={pcc_value:.4f}  target>={target:.4f}  [{status}]")


def _align_1d(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = min(int(a.numel()), int(b.numel()))
    return a[:n], b[:n]


def _trace_device_params() -> dict[str, int]:
    return {
        "trace_region_size": int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000))),
        "num_command_queues": num_command_queues_for_decode(),
    }


# remove during code cleanup
@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_trace_device_params()], indirect=True)
def test_ttnn_voxtral_tts_staged_pcc(device, reset_seeds, request):
    """Free-run E2E diagnostic: CPU reference rollout vs TT free-run (codes + waveform), logged not gated."""
    generate_steps = 1
    name = resolve_voxtral_model_name_or_skip()

    try:
        cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")

    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
            text_optimizations=voxtral_text_hf_aligned_optimizations,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    # ---------------------------------------------------------------------
    # One CPU generate produces the reference rollout (codes + waveform).
    # ---------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("CPU GENERATE (reference rollout)")
    logger.info("=" * 70)
    ref_wav_gen, ref_codes = cpu.generate(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
        return_tokenizer_codes=True,
    )
    assert torch.isfinite(ref_wav_gen).all(), "CPU reference produced non-finite waveform samples"
    assert int(ref_codes.shape[2]) > 0, "CPU reference produced no acoustic frames"

    # ---------------------------------------------------------------------
    # Informational only: free-running TT generation. Reads ~0.77 (fp32-softmax
    # aligned) because the discrete-code AR feedback diverges from the CPU rollout
    # once the first few FSQ-boundary flips occur. NOT asserted (see module docstring).
    # ---------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("FREE-RUN DIAGNOSTIC (TT generates its own codes; informational, not gated)")
    logger.info("=" * 70)
    tt_out = pipe.forward_device_resident(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
    )
    ttnn.synchronize_device(device)

    n_frames = min(int(tt_out.codes_b37t.shape[2]), int(ref_codes.shape[2]))
    tt_codes = tt_out.codes_b37t[:, :, :n_frames]
    ref_codes_aligned = ref_codes[:, :, :n_frames]
    log_per_step_code_match(ref_codes_aligned, tt_codes)

    sem_matches = int((tt_codes[:, 0] == ref_codes_aligned[:, 0]).sum().item())
    sem_total = int(tt_codes[:, 0].numel())
    ac_matches = int((tt_codes[:, 1:] == ref_codes_aligned[:, 1:]).sum().item())
    ac_total = int(tt_codes[:, 1:].numel())
    logger.info(f"  semantic-code match: {sem_matches / max(sem_total, 1):.4f}  ({sem_matches}/{sem_total})")
    logger.info(f"  acoustic-code match: {ac_matches / max(ac_total, 1):.4f}  ({ac_matches}/{ac_total})")

    assert torch.isfinite(tt_out.waveform).all(), "TT free-run produced non-finite waveform samples"
    free_ref, free_tt = _align_1d(ref_wav_gen, tt_out.waveform)
    _, free_pcc = comp_pcc(free_ref, free_tt, pcc=WAVEFORM_PCC_TARGET)
    _log_pcc("waveform (free-run, north-star, NOT gated)", float(free_pcc), WAVEFORM_PCC_TARGET)

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del tt_out
    gc.collect()


_GOLDEN_CODES_PATH = Path(__file__).resolve().parent.parent / "reference_outputs" / "voxtral_golden_codes.refpt"


def _load_golden_fixture() -> dict:
    """Load the committed golden fixture dict (env override: ``VOXTRAL_GOLDEN_CODES_PT``)."""
    path = Path(os.environ.get("VOXTRAL_GOLDEN_CODES_PT") or _GOLDEN_CODES_PATH)
    if not path.exists():
        pytest.skip(
            f"golden codes fixture missing: {path}. Generate once with "
            "models/experimental/voxtraltts/tests/generate_voxtral_golden_codes.py and commit it."
        )
    raw = torch.load(path, map_location="cpu")
    if not (isinstance(raw, dict) and "codes_b37t" in raw):
        raw = {"codes_b37t": raw}
    raw["codes_b37t"] = torch.as_tensor(raw["codes_b37t"], dtype=torch.long)
    assert raw["codes_b37t"].dim() == 3 and int(raw["codes_b37t"].shape[1]) == 37
    return raw


def _load_golden_codes() -> torch.Tensor:
    """Load the committed ``[1, 37, T]`` golden codes."""
    return _load_golden_fixture()["codes_b37t"]


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_trace_device_params()], indirect=True)
def test_ttnn_voxtral_tts_golden_codes_pcc(device, reset_seeds, request):
    """Audio-tokenizer PCC: decode the committed golden codes through the reference and TT
    tokenizers and compare the waveform. Fixed input, no reference run at test time."""
    name = resolve_voxtral_model_name_or_skip()
    golden_codes = _load_golden_codes()
    assert int(golden_codes.shape[2]) > 0, "golden codes have no frames"

    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
            text_optimizations=voxtral_text_hf_aligned_optimizations,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    logger.info("=" * 70)
    logger.info(f"GOLDEN-TRUTH TEACHER-FORCED (decode golden {tuple(golden_codes.shape)}: torch ref vs TT)")
    logger.info("=" * 70)
    ref_wav = audio_tokenizer_decode_reference(golden_codes, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args)
    tt_wav = pipe.decode_waveform_from_codes_tt(golden_codes)
    ttnn.synchronize_device(device)
    assert torch.isfinite(ref_wav).all(), "reference produced non-finite waveform samples"
    assert torch.isfinite(tt_wav).all(), "TT tokenizer produced non-finite waveform samples"

    ref_flat, tt_flat = _align_1d(ref_wav, tt_wav)
    _, wav_pcc = comp_pcc(ref_flat, tt_flat, pcc=WAVEFORM_PCC_TARGET)
    _log_pcc("waveform (golden-truth codes)", float(wav_pcc), WAVEFORM_PCC_TARGET)
    assert bool(wav_pcc >= WAVEFORM_PCC_TARGET), f"golden-truth waveform PCC below target: {wav_pcc}"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    gc.collect()


# Codes are shifted by 2 special tokens: tokenizer codes = model codes - 2.
_CODE_SHIFT = 2
# Acoustic-model waveform PCC (ref vs TT on the same realistic hidden). Below the tokenizer gate
# because the acoustic FM's bf16-vs-CPU rounding flips a few codes. Override with VOXTRAL_ACOUSTIC_PCC.
ACOUSTIC_PCC_TARGET = float(os.environ.get("VOXTRAL_ACOUSTIC_PCC", "0.97"))


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_trace_device_params()], indirect=True)
def test_ttnn_voxtral_tts_acoustic_pcc(device, reset_seeds, request):
    """Acoustic-model PCC in isolation: feed the SAME reference text hiddens (committed golden) +
    same noise to both the reference and TT acoustic models each step. No text model runs, so there
    is no text-model divergence — the only difference is the acoustic implementation. Compares
    per-step codes and the decoded waveform (both via the reference tokenizer, held constant).
    """
    name = resolve_voxtral_model_name_or_skip()
    fixture = _load_golden_fixture()
    if "text_hiddens" not in fixture:
        pytest.skip("golden fixture has no text_hiddens; regenerate with generate_voxtral_golden_codes.py")
    text_hiddens = fixture["text_hiddens"].to(torch.bfloat16)  # [T, dim] realistic reference hiddens
    n_steps = int(text_hiddens.shape[0])

    try:
        cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")
    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
            text_optimizations=voxtral_text_hf_aligned_optimizations,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    logger.info("=" * 70)
    logger.info(f"ACOUSTIC-MODEL PCC (same reference hidden to ref and TT, {n_steps} steps)")
    logger.info("=" * 70)

    cfg_alpha = torch.tensor(cpu._acoustic_cfg_alpha, dtype=cpu.dtype)
    ref_model_codes, tt_model_codes = [], []
    for i in range(n_steps):
        # SAME realistic hidden + SAME noise for both — only the acoustic implementation differs.
        hidden_i = text_hiddens[i : i + 1]  # [1, dim]
        noise_seed = acoustic_fm_noise_seed(0, i)
        torch.manual_seed(noise_seed)
        ref_code_i = cpu.acoustic_transformer(hidden_i, cfg_alpha).long().reshape(-1)
        tt_code_i = pipe.acoustic_codes_forward(hidden_i, cfg_alpha, noise_seed=noise_seed).long().reshape(-1)
        ref_model_codes.append(ref_code_i)
        tt_model_codes.append(tt_code_i)
    ttnn.synchronize_device(device)

    ref_codes_b37t = (torch.stack(ref_model_codes, dim=0).T.unsqueeze(0) - _CODE_SHIFT).clamp_min(0)  # [1,37,T]
    tt_codes_b37t = (torch.stack(tt_model_codes, dim=0).T.unsqueeze(0) - _CODE_SHIFT).clamp_min(0)

    # Diagnostic: per-step code agreement (TT vs ref).
    sem_match = int((tt_codes_b37t[:, 0] == ref_codes_b37t[:, 0]).sum()) / max(int(ref_codes_b37t[:, 0].numel()), 1)
    ac_match = int((tt_codes_b37t[:, 1:] == ref_codes_b37t[:, 1:]).sum()) / max(int(ref_codes_b37t[:, 1:].numel()), 1)
    logger.info(f"  per-step code match (TT vs ref): semantic={sem_match:.4f}  acoustic={ac_match:.4f}")

    # Both through the reference tokenizer so only the acoustic model differs.
    ref_wav = audio_tokenizer_decode_reference(
        ref_codes_b37t, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args
    )
    tt_wav = audio_tokenizer_decode_reference(tt_codes_b37t, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args)
    assert torch.isfinite(tt_wav).all(), "TT acoustic→tokenizer produced non-finite waveform samples"

    ref_flat, tt_flat = _align_1d(ref_wav, tt_wav)
    _, wav_pcc = comp_pcc(ref_flat, tt_flat, pcc=ACOUSTIC_PCC_TARGET)
    _log_pcc("waveform (acoustic model, ref vs TT)", float(wav_pcc), ACOUSTIC_PCC_TARGET)
    assert bool(wav_pcc >= ACOUSTIC_PCC_TARGET), f"acoustic waveform PCC below target: {wav_pcc}"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    gc.collect()


# Full-pipeline (text + acoustic + tokenizer) teacher-forced waveform PCC, ref vs TT.
PIPELINE_TF_PCC_TARGET = float(os.environ.get("VOXTRAL_PIPELINE_TF_PCC", "0.97"))


def _cpu_text_decode_step(cpu: VoxtralCPUReference, *, audio_codes_b37: torch.Tensor, past_key_values):
    """One reference text-decode step from an audio-code embedding → ``[1, dim]`` hidden."""
    next_input = cpu._audio_codes_to_input_embeds(audio_codes_b37)
    outputs = cpu.text_model(
        inputs_embeds=next_input,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states[-1][:, -1, :], outputs.past_key_values  # [1, dim]


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_trace_device_params()], indirect=True)
def test_ttnn_voxtral_tts_golden_acoustic_pcc(device, reset_seeds, request):
    """Full-pipeline teacher-forced PCC: ref vs TT, golden code fed to BOTH each step.

    Both models run live (text + acoustic); each step the golden code is fed to both text models
    (never their own output, so flips can't cascade), each produces its own hidden → its own code.
    Compares ref waveform (ref codes → ref tokenizer) vs TT waveform (TT codes → TT tokenizer).
    """
    name = resolve_voxtral_model_name_or_skip()
    fixture = _load_golden_fixture()
    golden_codes = fixture["codes_b37t"]  # [1,37,T] tokenizer codes (un-shifted)
    n_steps = int(golden_codes.shape[2])
    assert n_steps > 0, "golden fixture has no frames"
    golden_model_codes = golden_codes + _CODE_SHIFT  # model-space codes for text-model feedback

    try:
        cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")
    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
            text_optimizations=voxtral_text_hf_aligned_optimizations,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    logger.info("=" * 70)
    logger.info(f"FULL-PIPELINE TEACHER-FORCED (ref vs TT, golden fed to both for {n_steps} steps)")
    logger.info("=" * 70)

    prompt_ids = compose_speech_request(_DEMO_TEXT, name, voice=_DEMO_VOICE, ref_audio=None)["prompt_token_ids"]
    cfg_alpha = torch.tensor(cpu._acoustic_cfg_alpha, dtype=cpu.dtype)

    # Prefill both text models on the same prompt.
    _, cpu_embeds = cpu._prompt_embeddings(prompt_ids, _DEMO_VOICE)
    cpu_prefill = cpu.text_model(
        inputs_embeds=cpu_embeds.unsqueeze(0), use_cache=True, output_hidden_states=True, return_dict=True
    )
    cpu_hidden = cpu_prefill.hidden_states[-1][:, -1, :]  # [1, dim]
    cpu_pkv = cpu_prefill.past_key_values

    tt_embeds = pipe._build_voice_injected_embeds(prompt_ids, _DEMO_VOICE)
    tt_hidden_tt = pipe.text.prefill_from_embeds(tt_embeds, start_pos=0)
    tt_hidden = pipe.text.hidden_tt_to_torch(tt_hidden_tt)  # [dim]
    ttnn.deallocate(tt_hidden_tt)
    current_pos = len(prompt_ids)

    # Teacher-forced AR loop: feed golden to both, each emits its own acoustic code.
    ref_model_codes, tt_model_codes = [], []
    for i in range(n_steps):
        noise_seed = acoustic_fm_noise_seed(0, i)
        torch.manual_seed(noise_seed)
        ref_code_i = cpu.acoustic_transformer(cpu_hidden.to(torch.bfloat16), cfg_alpha).long().reshape(-1)
        tt_code_i = (
            pipe.acoustic_codes_forward(tt_hidden.to(torch.bfloat16).reshape(1, -1), cfg_alpha, noise_seed=noise_seed)
            .long()
            .reshape(-1)
        )
        ref_model_codes.append(ref_code_i)
        tt_model_codes.append(tt_code_i)

        # Advance both text models on the golden code, not their own output.
        golden_step = golden_model_codes[:, :, i].reshape(1, 37)
        cpu_hidden, cpu_pkv = _cpu_text_decode_step(cpu, audio_codes_b37=golden_step, past_key_values=cpu_pkv)
        mm_embed = pipe._audio_codes_to_mm_embed(golden_step)
        tt_hidden = pipe.text.decode_step_from_embeds(mm_embed, current_pos)
        current_pos += 1
    ttnn.synchronize_device(device)

    ref_codes_b37t = (torch.stack(ref_model_codes, dim=0).T.unsqueeze(0) - _CODE_SHIFT).clamp_min(0)  # [1,37,T]
    tt_codes_b37t = (torch.stack(tt_model_codes, dim=0).T.unsqueeze(0) - _CODE_SHIFT).clamp_min(0)

    # Diagnostic: per-step code agreement (TT vs ref).
    sem_match = int((tt_codes_b37t[:, 0] == ref_codes_b37t[:, 0]).sum()) / max(int(ref_codes_b37t[:, 0].numel()), 1)
    ac_match = int((tt_codes_b37t[:, 1:] == ref_codes_b37t[:, 1:]).sum()) / max(int(ref_codes_b37t[:, 1:].numel()), 1)
    logger.info(f"  per-step code match (TT vs ref): semantic={sem_match:.4f}  acoustic={ac_match:.4f}")

    # Each model through its own tokenizer.
    ref_wav = audio_tokenizer_decode_reference(
        ref_codes_b37t, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args
    )
    tt_wav = pipe.decode_waveform_from_codes_tt(tt_codes_b37t)
    ttnn.synchronize_device(device)
    assert torch.isfinite(ref_wav).all(), "reference produced non-finite waveform samples"
    assert torch.isfinite(tt_wav).all(), "TT produced non-finite waveform samples"

    ref_flat, tt_flat = _align_1d(ref_wav, tt_wav)
    _, wav_pcc = comp_pcc(ref_flat, tt_flat, pcc=PIPELINE_TF_PCC_TARGET)
    _log_pcc("waveform (full-pipeline teacher-forced: ref vs TT)", float(wav_pcc), PIPELINE_TF_PCC_TARGET)
    assert bool(wav_pcc >= PIPELINE_TF_PCC_TARGET), f"full-pipeline teacher-forced PCC below target: {wav_pcc}"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    gc.collect()
