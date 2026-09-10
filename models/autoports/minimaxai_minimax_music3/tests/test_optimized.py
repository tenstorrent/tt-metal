# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 07: the optimized pipeline (``dtype_policy="optimized"``, the default) keeps the stage-06 accuracy bars and
is faster than the functional baseline measured with the same harness.

Gate tests (``-m "not slow"``):

1. policy report: the components really run the optimized dtypes (bfp8 backbone weights + KV cache, bfp8 depth and
   DiT transformer weights, traced DiT step);
2. golden replay (teacher-forced codes and noises, 30 steps): frame-hidden PCC >= 0.98 (stage prompt), latents
   PCC >= 0.98 per window, stitched wav within 2.0 dB log-mel RMS of the golden (stage-06 bar);
3. the traced DiT step reproduces the eager forward, replays deterministically and follows input changes;
4. the candidate-only multinomial (``sample_top_k_candidates``) has exactly the candidate set and probabilities of
   diffusers' ``_sample_top_k`` (host only);
5. ``doc/optimize/perf.json`` holds a ``before`` (functional) and an ``after`` (optimized) measurement of
   ``scripts/measure_perf.py`` and the optimized run is faster on both headline numbers;
6. a short free-running song (2 s, 6 steps) runs end to end and reports its timings.
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

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.ar_generator import (
    AR_SAMPLING_TOP_K,
    sample_top_k,
    sample_top_k_candidates,
)
from models.autoports.minimaxai_minimax_music3.tt.audio_metrics import log_mel_distance
from models.autoports.minimaxai_minimax_music3.tt.constants import AUDIO_CODE_OFFSET, NUM_CODEBOOKS
from models.autoports.minimaxai_minimax_music3.tt.flow_transformer import BATCH
from models.common.utility_functions import comp_pcc

PCC_FRAME_HIDDENS = 0.98  # stage prompt: teacher-forced frame_hiddens PCC with bfp8 weights / KV
PCC_LATENT = 0.98
LOGMEL_BAR_DB = 2.0
SAMPLING_RATE = 44100

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = MODEL_DIR / "doc" / "optimize"
GENERATED_DIR = MODEL_DIR / "generated"
GOLDEN_PROMPT = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)
GOLDEN_LYRICS = "[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n[chorus]\nSoftly the world begins to breathe"


def _run_meta() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=MODEL_DIR, text=True).strip()
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


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(comp_pcc(a.float(), b.float(), 0.0)[1])


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def pipeline(mm3_mesh_device):
    from models.autoports.minimaxai_minimax_music3.tt.pipeline import MiniMaxMusic3Pipeline

    torch.set_num_threads(max(8, (os.cpu_count() or 8) - 4))
    t0 = time.time()
    pipe = MiniMaxMusic3Pipeline.load(mm3_mesh_device)  # the stage-07 default preset
    logger.info(f"optimized pipeline loaded in {time.time() - t0:.0f}s: {json.dumps(pipe.load_log, default=str)}")
    _record("load", policy_report=pipe.policy_report(), **{k: v for k, v in pipe.load_log.items()})
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
    chunks = torch.load(root / "chunks.pt")
    wav, sr = sf.read(root / "audio.wav", dtype="float32")
    assert sr == SAMPLING_RATE
    return {
        "manifest": manifest,
        "text_ids": torch.load(root / "text_ids.pt"),
        "codes": codes,
        "frame0_codes": groups[0],
        "frame_hiddens": torch.load(root / "frame_hiddens.pt"),
        "chunk_starts": list(chunks["chunk_starts"]),
        "noises": [n.float() for n in chunks["noises"]],
        "latents": [t.float() for t in chunks["latents"]],
        "audio": torch.from_numpy(wav.T.copy()),
    }


# ----------------------------------------------------------------------------- host only
def test_candidate_multinomial_matches_reference_distribution():
    """``sample_top_k_candidates`` draws from exactly the candidate set / probabilities of ``_sample_top_k``."""
    g = torch.Generator().manual_seed(0)
    for width in (1024, 16389):
        logits = torch.randn(1, width, generator=g) * 3
        logits[0, torch.randint(0, width, (width // 3,), generator=g)] = -float("inf")  # masked ids
        logits[0, :3] = logits[0, 3]  # ties at the threshold are kept by both
        values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
        thr = torch.topk(values, AR_SAMPLING_TOP_K, dim=-1).values[..., -1, None]
        ref_probs = torch.softmax(values.masked_fill(values < thr, -float("inf")), dim=-1)[0]
        candidates = torch.nonzero(ref_probs > 0).reshape(-1)
        assert candidates.numel() >= AR_SAMPLING_TOP_K
        # Every draw of the candidate sampler is a candidate, and its empirical distribution matches ref_probs.
        draws = torch.tensor([sample_top_k_candidates(logits, torch.Generator().manual_seed(s)) for s in range(4000)])
        assert set(draws.tolist()) <= set(candidates.tolist())
        emp = torch.bincount(draws, minlength=width).float() / draws.numel()
        assert float((emp - ref_probs).abs().max()) < 0.03, float((emp - ref_probs).abs().max())
        # The literal reference sampler draws from the same set (only its RNG consumption differs).
        ref_draws = torch.tensor(
            [int(sample_top_k(logits, torch.Generator().manual_seed(s)).item()) for s in range(500)]
        )
        assert set(ref_draws.tolist()) <= set(candidates.tolist())


def test_perf_json_before_after():
    """The measured before/after evidence exists, was produced by the perf harness, and improved."""
    path = DOC_DIR / "perf.json"
    assert path.is_file(), "run scripts/measure_perf.py --write-perf-json before/after"
    perf = json.loads(path.read_text())
    before, after = perf["before"], perf["after"]
    assert before["load_kwargs"]["dtype_policy"] == "functional", before["load_kwargs"]
    assert after["load_kwargs"]["dtype_policy"] == "optimized", after["load_kwargs"]
    assert after["ar_frames_per_s"] > before["ar_frames_per_s"], (before["ar_frames_per_s"], after["ar_frames_per_s"])
    assert after["dit_chunk_s"] < before["dit_chunk_s"], (before["dit_chunk_s"], after["dit_chunk_s"])
    gr = after["golden_replay"]
    assert gr["frame_hiddens_pcc"] >= PCC_FRAME_HIDDENS and min(gr["latent_pcc"]) >= PCC_LATENT
    assert gr["log_mel_rms_db"] <= LOGMEL_BAR_DB
    logger.info(
        f"before {before['ar_frames_per_s']:.2f} -> after {after['ar_frames_per_s']:.2f} frames/s; "
        f"DiT chunk {before['dit_chunk_s']:.2f} -> {after['dit_chunk_s']:.2f} s; vocoder chunk "
        f"{before['vocoder_chunk_s']:.2f} -> {after['vocoder_chunk_s']:.2f} s"
    )


# ----------------------------------------------------------------------------- device
@pytest.mark.hardware
@pytest.mark.timeout(3600)
def test_policy_report(pipeline):
    rep = pipeline.policy_report()
    logger.info(json.dumps(rep, indent=1, default=str))
    assert rep["preset"] == "optimized"
    llm = rep["llm"]
    assert llm["policy"] == "optimized"
    assert llm["decoder_tensor_groups"]["wqkv"] == "DataType.BFLOAT8_B"
    assert llm["decoder_tensor_groups"]["kv_cache"] == "DataType.BFLOAT8_B"
    assert llm["kv_cache_tensor_dtype"] == "DataType.BFLOAT8_B"
    assert llm["lm_head_weight"] == "DataType.BFLOAT8_B"
    assert rep["depth_weight_dtype"] == "DataType.BFLOAT8_B"
    assert rep["dit_weight_dtype"] == "DataType.BFLOAT8_B"
    assert rep["dit_trace"] is True
    _record("policy_report", **rep)


@pytest.mark.hardware
@pytest.mark.timeout(3600)
def test_golden_replay_optimized(pipeline, golden):
    m = golden["manifest"]
    frames = golden["codes"].shape[0]
    t0 = time.perf_counter()
    out = pipeline.generate(
        m["prompt"],
        m["lyrics"],
        seed=m["seed"],
        num_inference_steps=30,
        max_frames=frames,
        teacher_codes=golden["codes"],
        teacher_frame0_codes=golden["frame0_codes"],
        noises=golden["noises"],
        text_ids=golden["text_ids"],
    )
    wall = time.perf_counter() - t0
    assert out["frames"] == frames and torch.equal(out["codes"], golden["codes"])
    fh, gh = out["frame_hiddens"], golden["frame_hiddens"]
    fh_pcc = _pcc(gh, fh)
    per_frame = [_pcc(gh[0, f], fh[0, f]) for f in range(frames)]
    latent_pcc = [_pcc(golden["latents"][k], out["latents"][k]) for k in range(len(golden["latents"]))]
    audio = torch.from_numpy(out["audio"])
    n = min(audio.shape[-1], golden["audio"].shape[-1])
    dist = log_mel_distance(audio[..., :n], golden["audio"][..., :n], SAMPLING_RATE)
    wav_pcc = _pcc(golden["audio"][..., :n], audio[..., :n])
    logger.info(
        f"optimized golden replay in {wall:.1f}s: frame_hiddens PCC {fh_pcc:.5f} (min {min(per_frame):.5f}), latents "
        f"{['%.5f' % p for p in latent_pcc]}, log-mel {dist['rms_db']:.3f} dB, wav PCC {wav_pcc:.5f}; "
        f"AR {out['timings']['ar_frames_per_s']:.2f} frames/s, DiT {out['timings']['dit_per_chunk']}, vocoder {out['timings']['vocoder_per_chunk']}"
    )
    _record(
        "golden_replay",
        bars={"frame_hiddens_pcc": PCC_FRAME_HIDDENS, "latent_pcc": PCC_LATENT, "log_mel_rms_db": LOGMEL_BAR_DB},
        frame_hiddens_pcc=fh_pcc,
        frame_hiddens_pcc_min=min(per_frame),
        latent_pcc=latent_pcc,
        log_mel=dist,
        wav_pcc=wav_pcc,
        wall_s=wall,
        timings={k: v for k, v in out["timings"].items()},
    )
    assert fh_pcc >= PCC_FRAME_HIDDENS, fh_pcc
    for k, p in enumerate(latent_pcc):
        assert p >= PCC_LATENT, f"window {k}: latent PCC {p} < {PCC_LATENT}"
    assert dist["rms_db"] <= LOGMEL_BAR_DB, dist


@pytest.mark.hardware
@pytest.mark.timeout(1800)
def test_dit_trace_matches_eager(pipeline):
    """The traced Euler step equals the eager forward, replays deterministically and follows its inputs."""
    tf = pipeline.transformer
    t = 689
    g = torch.Generator().manual_seed(5)
    latents = torch.randn(1, 128, t, generator=g).expand(BATCH, -1, -1).contiguous()
    cond = torch.cat([torch.randn(1, t, 2048, generator=g) * 0.4, torch.zeros(1, t, 2048)], dim=0)
    timestep = torch.full((BATCH,), 0.5)
    tr = tf.traced_step(t)
    assert tr.trace_id is not None and t in pipeline.denoiser.persistent_latents
    tr.set_condition(cond)
    v1 = tr.step(latents, timestep)
    v2 = tr.step(latents, timestep)
    assert torch.equal(v1, v2), "trace replay is not deterministic"
    cond_proj = tf.prepare_condition(cond)
    v_eager = tf(latents, timestep, cond_proj=cond_proj)
    ttnn.deallocate(cond_proj)
    pcc = _pcc(v_eager, v1)
    max_abs = float((v_eager - v1).abs().max())
    # Changed inputs must change the output (the persistent input buffers are really refreshed).
    v3 = tr.step(latents * 0.5, torch.full((BATCH,), 0.9))
    assert not torch.equal(v3, v1)
    logger.info(f"traced vs eager: PCC {pcc:.6f}, max abs {max_abs:.4f}; replays so far {tr.replays}")
    _record("dit_trace_vs_eager", pcc=pcc, max_abs_err=max_abs, replays=tr.replays)
    assert pcc >= 0.9999, pcc


@pytest.mark.hardware
@pytest.mark.timeout(1800)
def test_short_song_runs(pipeline):
    out = pipeline.generate(GOLDEN_PROMPT, GOLDEN_LYRICS, audio_duration=2.0, seed=11, num_inference_steps=6)
    audio = out["audio"]
    assert audio.shape[0] == 2 and np.isfinite(audio).all() and out["frames"] == 50
    assert abs(audio.shape[-1] / SAMPLING_RATE - 2.0) < 1.0
    logger.info(f"2 s song: {json.dumps({k: v for k, v in out['timings'].items() if k != 'ar_detail'}, default=float)}")
    _record("short_song", frames=out["frames"], timings={k: v for k, v in out["timings"].items()})
