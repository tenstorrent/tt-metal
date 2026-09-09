# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 06: the end-to-end MiniMax-Music3 pipeline (text -> stereo wav) on one Blackhole chip.

Gate tests (``-m "not slow"``):

1. golden replay - teacher-forced codes from ``sampled_codes.pt`` (+ frame 0 from ``sampled_raw.pt``) and the golden
   per-window noises: the two windows' latents match golden ``latents[k]`` at PCC >= 0.98 and the stitched wav is
   within ``LOGMEL_BAR_DB`` of the golden ``audio.wav`` in log-mel RMS distance;
2. free-running 10 s (seed 7): 44.1 kHz stereo, RMS > 1e-3, finite, duration within 1 s of ``frames / 25``; the wav
   is saved under ``generated/``;
3. ``audio_duration`` 0.4 s / 2 s / 10 s / 13 s (a 34-latent window; one window; two windows; three windows with a short
   tail) run with a reduced step count;
4. same-seed determinism (bit-identical audio and codes).

Log-mel bar: ``scripts/vocoder_control_cpu.py`` (host only, recorded in ``doc/pipeline/pcc/vocoder_control.json``)
measured the vendored vocoder in bf16 on the golden latents at 1.115 dB RMS vs the fp32 golden wav (the torch-vs-torch
bf16 control), 0.578 dB for latents perturbed to PCC 0.9996 (stage-05's per-window DiT PCC) and 3.839 dB for latents at
the 0.98 PCC bar. The wav bar is twice the bf16 control (2.0 dB): comfortably above the DiT-only contribution and below
what the latent PCC bar itself would allow.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from loguru import logger

from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.audio_metrics import audio_stats, code_stats, log_mel_distance
from models.autoports.minimaxai_minimax_music3.tt.constants import AUDIO_CODE_OFFSET, FRAME_RATE, NUM_CODEBOOKS
from models.autoports.minimaxai_minimax_music3.tt.denoiser import chunk_starts_for
from models.common.utility_functions import comp_pcc

PCC_LATENT = 0.98
LOGMEL_BAR_DB = 2.0
SAMPLING_RATE = 44100

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = MODEL_DIR / "doc" / "pipeline"
GENERATED_DIR = MODEL_DIR / "generated"
FAST_STEPS = 6  # steps for the duration sweep (the golden replay and the 10 s clip use the full 30)

GOLDEN_PROMPT = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)
GOLDEN_LYRICS = "[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n[chorus]\nSoftly the world begins to breathe"


# ----------------------------------------------------------------------------- evidence
def _run_meta() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=DOC_DIR.parent, text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    return {
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": commit,
        "loadavg_1m": os.getloadavg()[0],
        "log_hint": os.environ.get("MM3_RUN_LABEL", ""),
    }


def _record(name: str, **fields):
    if os.environ.get("TT_METAL_WATCHER") or os.environ.get("TT_METAL_DEVICE_PROFILER"):
        return
    out = DOC_DIR / "pcc"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    results = json.loads(path.read_text()) if path.is_file() else {}
    fields["_meta"] = _run_meta()
    results[name] = fields
    path.write_text(json.dumps(results, indent=2, sort_keys=True, default=float) + "\n")


def _pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    _, pcc = comp_pcc(golden.float(), actual.float(), 0.0)
    return float(pcc)


def _save_wav(name: str, audio: np.ndarray, sr: int) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / name
    sf.write(path, audio.T, sr, subtype="FLOAT")
    return path


def _check_audio(out: dict, min_seconds: float | None = None):
    audio = out["audio"]
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32, (type(audio), audio.dtype)
    assert audio.ndim == 2 and audio.shape[0] == 2, audio.shape
    assert out["sampling_rate"] == SAMPLING_RATE
    assert np.isfinite(audio).all(), "NaN / inf in the audio"
    assert float(np.abs(audio).max()) <= 1.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert rms > 1e-3, f"audio RMS {rms} (silent)"
    seconds = audio.shape[-1] / SAMPLING_RATE
    expected = out["frames"] / FRAME_RATE
    assert abs(seconds - expected) < 1.0, f"{seconds:.3f} s of audio for {out['frames']} frames ({expected:.3f} s)"
    if min_seconds is not None:
        assert seconds >= min_seconds, (seconds, min_seconds)
    return rms, seconds


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def pipeline(mm3_mesh_device):
    from models.autoports.minimaxai_minimax_music3.tt.pipeline import MiniMaxMusic3Pipeline

    torch.set_num_threads(max(8, (os.cpu_count() or 8) - 4))
    t0 = time.time()
    pipe = MiniMaxMusic3Pipeline.load(mm3_mesh_device)
    logger.info(f"pipeline loaded in {time.time() - t0:.0f}s: {json.dumps(pipe.load_log, default=str)}")
    _record("load", **{k: v for k, v in pipe.load_log.items()})
    yield pipe
    pipe.release()


@pytest.fixture(scope="session")
def golden():
    root = R.reference_dir()
    if not (root / "chunks.pt").is_file():
        pytest.skip(f"golden reference missing under {root}")
    manifest = json.loads((root / "manifest.json").read_text())
    codes = torch.load(root / "sampled_codes.pt")
    raw = torch.load(root / "sampled_raw.pt")
    groups = raw[: (raw.numel() // NUM_CODEBOOKS) * NUM_CODEBOOKS].reshape(-1, NUM_CODEBOOKS).clone()
    groups[:, 0] -= AUDIO_CODE_OFFSET
    assert torch.equal(groups[1 : codes.shape[0] + 1], codes), "sampled_raw / sampled_codes disagree"
    chunks = torch.load(root / "chunks.pt")
    wav, sr = sf.read(root / "audio.wav", dtype="float32")
    assert sr == SAMPLING_RATE
    return {
        "manifest": manifest,
        "text_ids": torch.load(root / "text_ids.pt"),
        "codes": codes,
        "frame0_codes": groups[0],
        "chunk_starts": list(chunks["chunk_starts"]),
        "noises": [n.float() for n in chunks["noises"]],
        "latents": [t.float() for t in chunks["latents"]],
        "frame_hiddens": torch.load(root / "frame_hiddens.pt"),
        "audio": wav.T.astype(np.float32),  # [2, S]
    }


# ----------------------------------------------------------------------------- tests
@pytest.mark.hardware
def test_golden_replay(pipeline, golden):
    """Teacher-forced codes + golden noises -> latents PCC >= 0.98 per window and wav within the log-mel bar."""
    frames = golden["codes"].shape[0]
    t0 = time.time()
    out = pipeline.generate(
        GOLDEN_PROMPT,
        GOLDEN_LYRICS,
        audio_duration=golden["manifest"]["audio_duration"],
        seed=golden["manifest"]["seed"],
        num_inference_steps=golden["manifest"]["num_inference_steps"],
        teacher_codes=golden["codes"],
        teacher_frame0_codes=golden["frame0_codes"],
        noises=golden["noises"],
    )
    wall = time.time() - t0
    assert torch.equal(out["text_ids"], golden["text_ids"])
    assert out["frames"] == frames and out["stopped_by"] == "teacher_codes"
    assert torch.equal(out["codes"], golden["codes"])
    assert out["chunk_starts"] == golden["chunk_starts"] == chunk_starts_for(frames)

    fh_pcc = _pcc(golden["frame_hiddens"], out["frame_hiddens"])
    latent_pccs = [_pcc(g, a) for g, a in zip(golden["latents"], out["latents"])]
    rms, seconds = _check_audio(out)
    assert out["audio"].shape == golden["audio"].shape, (out["audio"].shape, golden["audio"].shape)
    dist = log_mel_distance(out["audio"], golden["audio"], SAMPLING_RATE)
    # Path-matched qualitative control: the replay's spectral statistics next to the golden clip's.
    replay_full, golden_full = audio_stats(out["audio"], SAMPLING_RATE), audio_stats(golden["audio"], SAMPLING_RATE)
    replay_stats = {k: v for k, v in replay_full.items() if not isinstance(v, list)}
    golden_stats = {k: v for k, v in golden_full.items() if not isinstance(v, list)}
    replay_bands, golden_bands = replay_full["band_energy_mean"], golden_full["band_energy_mean"]
    # Per-second wav PCC across the whole clip (covers the window join at latent 431 = sample 220672).
    n = out["audio"].shape[-1] // SAMPLING_RATE
    per_second_pcc = [
        _pcc(
            torch.from_numpy(golden["audio"][:, i * SAMPLING_RATE : (i + 1) * SAMPLING_RATE]),
            torch.from_numpy(out["audio"][:, i * SAMPLING_RATE : (i + 1) * SAMPLING_RATE]),
        )
        for i in range(n)
    ]
    wav_pcc = _pcc(torch.from_numpy(golden["audio"]), torch.from_numpy(out["audio"]))
    path = _save_wav("golden_replay_seed7_10s.wav", out["audio"], out["sampling_rate"])
    logger.info(
        f"golden replay: frame_hiddens PCC {fh_pcc:.5f}, latent PCC {latent_pccs}, wav PCC {wav_pcc:.5f}, "
        f"log-mel rms {dist['rms_db']:.3f} dB (mean abs {dist['mean_abs_db']:.3f}), {seconds:.3f} s, {wall:.1f} s wall -> {path}"
    )
    _record(
        "golden_replay",
        frames=frames,
        frame_hiddens_pcc=fh_pcc,
        latent_pcc=latent_pccs,
        latent_shapes=[list(t.shape) for t in out["latents"]],
        wav_pcc=wav_pcc,
        log_mel=dist,
        log_mel_bar_db=LOGMEL_BAR_DB,
        per_second_wav_pcc=per_second_pcc,
        replay_audio_stats=replay_stats,
        golden_audio_stats=golden_stats,
        replay_band_energy_mean=replay_bands,
        golden_band_energy_mean=golden_bands,
        audio_seconds=seconds,
        audio_rms=rms,
        timings=out["timings"],
        wav=str(path.relative_to(MODEL_DIR)),
    )
    assert fh_pcc >= 0.99, fh_pcc
    for k, p in enumerate(latent_pccs):
        assert p >= PCC_LATENT, f"window {k}: latent PCC {p} < {PCC_LATENT}"
    assert dist["rms_db"] <= LOGMEL_BAR_DB, f"log-mel RMS distance {dist['rms_db']:.3f} dB > {LOGMEL_BAR_DB} dB"
    # The binding second is second 0 (the quiet intro, measured 0.9905); the clip is bit-deterministic on a fixed tree,
    # so a dip below the floor means a precision change, not flakiness.
    assert (
        min(per_second_pcc) >= 0.99
    ), f"per-second wav PCC dipped to {min(per_second_pcc):.4f} (window join / crop drift?)"
    for rb, gb in zip(replay_bands, golden_bands):
        assert abs(rb - gb) < 0.02, (replay_bands, golden_bands)


@pytest.mark.hardware
def test_free_running_10s(pipeline):
    """Seed 7, 10 s: a stereo 44.1 kHz wav with RMS > 1e-3, finite, duration within 1 s of frames / 25."""
    out = pipeline.generate(GOLDEN_PROMPT, GOLDEN_LYRICS, audio_duration=10.0, seed=7, num_inference_steps=30)
    rms, seconds = _check_audio(out)
    assert out["frames"] == 250 and out["stopped_by"] == "max_frames", (out["frames"], out["stopped_by"])
    assert out["chunk_starts"] == [0, 100]
    codes = code_stats(out["codes"])
    stats = audio_stats(out["audio"], out["sampling_rate"])
    path = _save_wav("free_running_seed7_10s.wav", out["audio"], out["sampling_rate"])
    logger.info(f"free running 10 s: {seconds:.3f} s, rms {rms:.4f}, codes {codes}, timings {out['timings']} -> {path}")
    _record(
        "free_running_10s",
        seed=7,
        frames=out["frames"],
        audio_seconds=seconds,
        audio_rms=rms,
        code_stats=codes,
        audio_stats={k: v for k, v in stats.items() if not isinstance(v, list)},
        timings=out["timings"],
        wav=str(path.relative_to(MODEL_DIR)),
    )
    assert codes["most_common_semantic_share"] <= 0.3, codes
    assert stats["silence_fraction_1s"] < 0.5, stats["silence_fraction_1s"]


@pytest.mark.hardware
@pytest.mark.parametrize(
    "audio_duration,expected_frames,expected_starts",
    [(0.4, 10, [0]), (2.0, 50, [0]), (10.0, 250, [0, 100]), (13.0, 325, [0, 100, 200])],
)
def test_audio_durations(pipeline, audio_duration, expected_frames, expected_starts):
    """A tiny window (10 frames -> 34 latents, the shortest realistic end-token song), one window (50 frames -> 172
    latents), two windows, three windows with a 125-frame tail; reduced step count."""
    out = pipeline.generate(
        GOLDEN_PROMPT, GOLDEN_LYRICS, audio_duration=audio_duration, seed=11, num_inference_steps=FAST_STEPS
    )
    assert out["frames"] == expected_frames and out["stopped_by"] == "max_frames", (out["frames"], out["stopped_by"])
    assert out["chunk_starts"] == expected_starts, out["chunk_starts"]
    rms, seconds = _check_audio(out)
    logger.info(
        f"duration {audio_duration} s: {out['frames']} frames, windows {out['chunk_starts']}, {seconds:.3f} s, rms {rms:.4f}, timings {out['timings']}"
    )
    _record(
        f"duration_{audio_duration:g}s",
        frames=out["frames"],
        chunk_starts=out["chunk_starts"],
        latent_shapes=[list(t.shape) for t in out["latents"]],
        audio_seconds=seconds,
        audio_rms=rms,
        steps=FAST_STEPS,
        timings=out["timings"],
    )


@pytest.mark.hardware
def test_same_seed_determinism(pipeline):
    """Two runs with the same seed give the same codes, latents and audio."""
    kwargs = dict(audio_duration=2.0, seed=3, num_inference_steps=FAST_STEPS)
    a = pipeline.generate(GOLDEN_PROMPT, GOLDEN_LYRICS, **kwargs)
    b = pipeline.generate(GOLDEN_PROMPT, GOLDEN_LYRICS, **kwargs)
    assert a["seed"] == b["seed"] == 3
    assert torch.equal(a["codes"], b["codes"]) and torch.equal(a["frame0_codes"], b["frame0_codes"])
    latent_diff = max(float((x - y).abs().max()) for x, y in zip(a["latents"], b["latents"]))
    audio_diff = float(np.abs(a["audio"] - b["audio"]).max())
    c = pipeline.generate(GOLDEN_PROMPT, GOLDEN_LYRICS, audio_duration=2.0, seed=4, num_inference_steps=FAST_STEPS)
    other_diff = float(np.abs(a["audio"][..., : c["audio"].shape[-1]] - c["audio"][..., : a["audio"].shape[-1]]).max())
    logger.info(
        f"determinism: max latent diff {latent_diff}, max audio diff {audio_diff}; different seed: {other_diff}, codes equal {torch.equal(a['codes'], c['codes'])}"
    )
    _record(
        "determinism",
        seed=3,
        frames=a["frames"],
        max_latent_diff=latent_diff,
        max_audio_diff=audio_diff,
        other_seed_max_audio_diff=other_diff,
    )
    assert latent_diff == 0.0 and audio_diff == 0.0, (latent_diff, audio_diff)
    assert not torch.equal(a["codes"], c["codes"]), "a different seed produced identical codes"


@pytest.mark.hardware
@pytest.mark.slow
def test_seed_none_draws_a_seed(pipeline):
    out = pipeline.generate(GOLDEN_PROMPT, GOLDEN_LYRICS, audio_duration=2.0, seed=None, num_inference_steps=2)
    assert isinstance(out["seed"], int)
    _check_audio(out)
