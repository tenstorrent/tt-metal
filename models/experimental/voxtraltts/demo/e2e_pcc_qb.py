# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E component PCC for Voxtral TTS on QB (TP=4) — localizes audio noise.

Opens the 4-chip QB mesh (1x4, FABRIC_1D) and measures, against the CPU reference:

  1. TEACHER-FORCED waveform PCC: decode the SAME CPU-generated codes through the TT
     audio tokenizer/vocoder. Isolates the (replicated) tokenizer on QB. Target >= 0.99.
     If this drops on QB but is fine single-chip, the tokenizer has a multi-device bug.

  2. FREE-RUN code match + waveform PCC: TT generates its own codes on QB. Compares
     semantic/acoustic code agreement and waveform PCC against the CPU rollout. Free-run
     waveform PCC is structurally ~0.77 even single-chip (discrete FSQ feedback diverges),
     so this is informational — but a large QB-vs-single gap localizes acoustic/text drift.

    python models/experimental/voxtraltts/demo/e2e_pcc_qb.py

Set VOXTRAL_E2E_STEPS to change the number of generated frames (default 8).
"""
from __future__ import annotations

import os

os.environ.setdefault("VOXTRAL_DECODE_TRACE", "0")  # untraced path for clean host comparison

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_tokenizer_decode_reference
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_config import (
    DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
    DEFAULT_VOXTRAL_MODEL,
)
from models.experimental.voxtraltts.tests.common import log_per_step_code_match
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_hf_aligned_optimizations
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

MODEL = os.environ.get("VOXTRAL_PCC_MODEL", DEFAULT_VOXTRAL_MODEL)
VOICE = os.environ.get("VOXTRAL_PCC_VOICE", "casual_male")
STEPS = int(os.environ.get("VOXTRAL_E2E_STEPS", "8"))
TEXT = (
    "Voxtral is a four billion parameter open weight text to speech model "
    "released by Mistral AI, designed for low latency multilingual voice generation."
)
WAVEFORM_PCC_TARGET = 0.99


def _open_mesh():
    # Match the demo's device params: traced decode needs a trace region + a 2nd command queue.
    from models.experimental.voxtraltts.demo.decode_trace_2cq import (
        decode_trace_enabled,
        num_command_queues_for_decode,
    )
    from tests.scripts.common import get_updated_device_params

    params = {}
    if decode_trace_enabled():
        params["trace_region_size"] = int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000)))
        params["num_command_queues"] = num_command_queues_for_decode()
    updated = get_updated_device_params(params)

    n = ttnn.get_num_devices()
    if os.environ.get("VOXTRAL_PCC_NDEV") == "1":
        n = 1  # force single-chip for QB-vs-single comparison
    if n >= 4:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), physical_device_ids=[0, 1, 2, 3], **updated)
        logger.info("Opened 1x4 mesh (QB, TP=4) with FABRIC_1D")
    else:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], **updated)
        logger.info("Opened 1x1 mesh (single chip)")
    return mesh


def _align(a: torch.Tensor, b: torch.Tensor):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = min(int(a.numel()), int(b.numel()))
    return a[:n], b[:n]


def _pcc(label, ref, tt, target):
    rf, tf = _align(ref, tt)
    _, val = comp_pcc(rf, tf, pcc=target)
    try:
        v = float(val)
    except (TypeError, ValueError):
        v = float(str(val).strip().split()[-1])
    status = "PASS" if v >= target else "LOW"
    logger.info(f"  {label}: PCC={v:.5f}  target>={target:.3f}  [{status}]")
    return v


def main() -> None:
    mesh = _open_mesh()
    n_devices = mesh.get_num_devices()
    pipe = None
    try:
        logger.info(f"Loading CPU reference {MODEL!r} ...")
        cpu = VoxtralCPUReference(model_name_or_path=MODEL, dtype="bfloat16", device="cpu")
        logger.info(f"Loading TT pipeline (n_devices={n_devices}) ...")
        pipe = VoxtralTTSPipeline.from_model_name(
            mesh,
            model_name_or_path=MODEL,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
            text_optimizations=voxtral_text_hf_aligned_optimizations,
        )

        logger.info("=" * 72)
        logger.info(f"CPU GENERATE (shared codes, {STEPS} steps, seed=0)")
        ref_wav_gen, ref_codes = cpu.generate(
            text=TEXT, voice=VOICE, max_tokens=STEPS, seed=0, return_tokenizer_codes=True
        )
        assert int(ref_codes.shape[2]) > 0, "CPU reference produced no acoustic frames"

        # ── 1. Teacher-forced: tokenizer/vocoder isolation ────────────────────────────
        logger.info("=" * 72)
        logger.info("TEACHER-FORCED (decode shared CPU codes through TT tokenizer on QB)")
        ref_wav = audio_tokenizer_decode_reference(ref_codes, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args)
        tt_wav = pipe.decode_waveform_from_codes_tt(ref_codes)
        ttnn.synchronize_device(mesh)
        tf_pcc = _pcc("waveform (teacher-forced)", ref_wav, tt_wav, WAVEFORM_PCC_TARGET)

        # ── 2. Free-run: acoustic + text code generation on QB ────────────────────────
        logger.info("=" * 72)
        logger.info("FREE-RUN (TT generates its own codes on QB; informational)")
        tt_out = pipe.forward_device_resident(text=TEXT, voice=VOICE, max_tokens=STEPS, seed=0)
        ttnn.synchronize_device(mesh)

        n_frames = min(int(tt_out.codes_b37t.shape[2]), int(ref_codes.shape[2]))
        tt_codes = tt_out.codes_b37t[:, :, :n_frames]
        ref_aligned = ref_codes[:, :, :n_frames]
        log_per_step_code_match(ref_aligned, tt_codes)
        sem_m = int((tt_codes[:, 0] == ref_aligned[:, 0]).sum().item())
        sem_t = int(tt_codes[:, 0].numel())
        ac_m = int((tt_codes[:, 1:] == ref_aligned[:, 1:]).sum().item())
        ac_t = int(tt_codes[:, 1:].numel())
        logger.info(f"  semantic-code match: {sem_m / max(sem_t,1):.4f}  ({sem_m}/{sem_t})")
        logger.info(f"  acoustic-code match: {ac_m / max(ac_t,1):.4f}  ({ac_m}/{ac_t})")
        _pcc("waveform (free-run, NOT gated)", ref_wav_gen, tt_out.waveform, WAVEFORM_PCC_TARGET)

        logger.info("=" * 72)
        if tf_pcc >= WAVEFORM_PCC_TARGET:
            logger.info(
                "Teacher-forced waveform PCC PASS — TT tokenizer is correct on QB. "
                "Any residual audio noise is free-run AR code divergence, not a tokenizer bug."
            )
        else:
            logger.warning(
                f"Teacher-forced waveform PCC LOW ({tf_pcc:.5f}) — the TT tokenizer/vocoder "
                "diverges on QB. This is a multi-device tokenizer bug = the audio noise source."
            )
    finally:
        if pipe is not None:
            try:
                pipe.cleanup_all()
            except Exception as exc:
                logger.warning(f"pipe cleanup failed: {exc}")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
