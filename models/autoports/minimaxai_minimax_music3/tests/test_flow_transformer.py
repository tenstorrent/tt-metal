# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 05 tests: flow-matching DiT, condition encoder, Euler scheduler and the chunk loop.

References: the vendored diffusers modules in ``reference/flow_transformer_ref.py`` run in fp32 on the CPU
from the fp32 safetensors, the stage-01 golden tensors (fp32 diffusers run, seed 7, 30 steps) under
``~/mm3-bringup/reference`` (``chunks.pt``: ``conditions_raw`` / ``conditions`` / ``noises`` / ``latents`` per
window, ``frame_hiddens.pt``, ``sigmas.pt``) and ``doc/flow_dit/pcc/scheduler_triples.pt`` dumped from
diffusers' ``FlowMatchEulerDiscreteScheduler`` (``scripts/dump_scheduler_triples.py``).

Run (device, serialized):

    source ~/mm3-bringup/common.sh && cd $MM3_WT && \
    with_hw_lock timeout 3600 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_flow_transformer.py -m "not slow"

Every PCC / timing is written to ``doc/flow_dit/pcc/results.json``. If ``generated/ref_trajectory.pt``
exists (``scripts/dump_ref_trajectory.py``, CPU, ~10 min) the chunk tests log the per-step PCC against the
fp32 reference trajectory; otherwise they log the per-step PCC against the golden final latent.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import flow_transformer_ref as REF
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.condition_encoder import (
    ConditionEncoder,
    latent_length,
    nearest_index_map,
)
from models.autoports.minimaxai_minimax_music3.tt.denoiser import (
    CHUNK_FRAMES,
    OVERLAP_LATENT_LENGTH,
    ChunkDenoiser,
    chunk_starts_for,
)
from models.autoports.minimaxai_minimax_music3.tt.flow_transformer import BATCH, FlowTransformer, padded_seq_len
from models.autoports.minimaxai_minimax_music3.tt.scheduler import FlowMatchEulerScheduler
from models.common.utility_functions import comp_pcc

pytestmark = [pytest.mark.timeout(2400)]

PCC_FORWARD = 0.99  # one DiT forward vs the fp32 torch reference
PCC_CONDITION = 0.999  # condition encoder vs golden conditions_raw
PCC_CHUNK = 0.98  # full 30-step chunk vs golden latents
STEP_MAX_ABS_ERR = 1e-5
NUM_STEPS = 30

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = MODEL_DIR / "doc" / "flow_dit"
REF_TRAJECTORY = MODEL_DIR / "generated" / "ref_trajectory.pt"


def _record(name: str, **fields):
    if os.environ.get("TT_METAL_WATCHER") or os.environ.get("TT_METAL_DEVICE_PROFILER"):
        return
    out = DOC_DIR / "pcc"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    results = json.loads(path.read_text()) if path.is_file() else {}
    results[name] = fields
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def _pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    _, pcc = comp_pcc(golden.float(), actual.float(), 0.0)
    return float(pcc)


def _sync_time(mesh_device, fn, repeats: int) -> list:
    times = []
    for _ in range(repeats):
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        fn()
        ttnn.synchronize_device(mesh_device)
        times.append(time.time() - t0)
    return times


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def golden():
    root = R.reference_dir()
    if not (root / "chunks.pt").is_file():
        pytest.skip(f"golden reference missing under {root}")
    chunks = torch.load(root / "chunks.pt")
    return {
        **chunks,
        "frame_hiddens": torch.load(root / "frame_hiddens.pt"),
        "sigmas": torch.load(root / "sigmas.pt"),
    }


@pytest.fixture(scope="module")
def flow_transformer(mm3_mesh_device):
    model = FlowTransformer.from_pretrained(mm3_mesh_device)
    logger.info(f"FlowTransformer loaded in {model.load_seconds:.1f} s")
    yield model
    model.release()


@pytest.fixture(scope="module")
def condition_encoder(mm3_mesh_device):
    enc = ConditionEncoder.from_pretrained(mm3_mesh_device)
    yield enc
    enc.release()


@pytest.fixture(scope="module")
def ref_transformer():
    torch.set_num_threads(max(8, os.cpu_count() or 8))
    t0 = time.time()
    model = REF.load_transformer()
    logger.info(f"fp32 torch reference DiT loaded in {time.time() - t0:.1f} s")
    return model


@pytest.fixture(scope="module")
def ref_trajectory():
    if REF_TRAJECTORY.is_file():
        return torch.load(REF_TRAJECTORY)
    return None


def _both_rows(condition: torch.Tensor) -> torch.Tensor:
    """``[1, T, 2048]`` -> ``[2, T, 2048]`` with the zero unconditional row."""
    return torch.cat([condition, torch.zeros_like(condition)], dim=0)


# ----------------------------------------------------------------------------- scheduler (host only)
def test_scheduler_matches_diffusers(golden):
    triples = torch.load(DOC_DIR / "pcc" / "scheduler_triples.pt")
    assert triples["config"]["invert_sigmas"] is True and triples["config"]["num_train_timesteps"] == 1
    # the schedule the golden run used, per chunk
    sched = FlowMatchEulerScheduler(NUM_STEPS)
    for k, rec in enumerate(golden["sigmas"]):
        assert torch.equal(sched.sigmas, rec["sigmas"]), f"chunk {k} sigmas differ"
        assert torch.equal(sched.timesteps, rec["timesteps"]), f"chunk {k} timesteps differ"
    assert float(sched.timesteps[0]) == 0.0 and abs(float(sched.timesteps[-1]) - (1 - 1 / NUM_STEPS)) < 1e-6
    worst = 0.0
    checked = 0
    for rec in triples["records"]:
        n = rec["num_inference_steps"]
        s = FlowMatchEulerScheduler(n)
        assert torch.equal(s.sigmas, rec["sigmas"]) and torch.equal(s.timesteps, rec["timesteps"]), n
        for st in rec["steps"]:
            fresh = FlowMatchEulerScheduler(n)  # index_for_timestep path (first call)
            prev = fresh.step(st["velocity"], st["t"], st["sample"])
            assert prev.dtype == st["prev_sample"].dtype
            worst = max(worst, float((prev - st["prev_sample"]).abs().max()))
            checked += 1
        # sequential path: the dumped steps 0, 1, 2 are consecutive, so stepping one scheduler through them
        # (step index advancing) must reproduce each dumped prev_sample exactly as well
        seq = FlowMatchEulerScheduler(n)
        for st in rec["steps"][:3]:
            assert seq.step_index in (None, st["i"])
            prev = seq.step(st["velocity"], st["t"], st["sample"])
            worst = max(worst, float((prev - st["prev_sample"]).abs().max()))
        assert seq.step_index == min(3, n)
    logger.info(f"scheduler: {checked} dumped steps, max abs err {worst:.3e}")
    assert worst < STEP_MAX_ABS_ERR
    _record("scheduler", exact_schedule=True, dumped_steps_checked=checked, step_max_abs_err=worst)


def test_chunk_starts_match_reference(golden):
    assert chunk_starts_for(golden["frame_hiddens"].shape[1]) == list(golden["chunk_starts"])
    assert chunk_starts_for(200) == [0] and chunk_starts_for(201) == [0, 100] and chunk_starts_for(301) == [0, 100, 200]
    for k, start in enumerate(golden["chunk_starts"]):
        frames = min(start + CHUNK_FRAMES, golden["frame_hiddens"].shape[1]) - start
        assert latent_length(frames) == golden["conditions_raw"][k].shape[1]


# ----------------------------------------------------------------------------- condition encoder
@pytest.mark.hardware
def test_condition_encoder_golden(condition_encoder, golden):
    fh = golden["frame_hiddens"]
    for k, start in enumerate(golden["chunk_starts"]):
        end = min(start + CHUNK_FRAMES, fh.shape[1])
        ref = golden["conditions_raw"][k]
        t0 = time.time()
        out = condition_encoder(fh[:, start:end])
        dt = time.time() - t0
        assert out.shape == ref.shape, (out.shape, ref.shape)
        pcc = _pcc(ref, out)
        max_err = float((ref - out).abs().max())
        logger.info(
            f"condition encoder chunk {k}: {end - start} frames -> {out.shape[1]} latents, PCC {pcc:.6f}, max abs err {max_err:.4f}, {dt * 1e3:.0f} ms"
        )
        _record(
            f"condition_encoder_chunk{k}",
            frames=end - start,
            latents=out.shape[1],
            pcc=pcc,
            max_abs_err=max_err,
            seconds=dt,
        )
        assert pcc >= PCC_CONDITION
    # the nearest-neighbour map is torch's own (no off-by-one against the reference's F.interpolate)
    idx = nearest_index_map(200, latent_length(200))
    assert idx.shape == (689,) and int(idx[0]) == 0 and int(idx[-1]) == 199 and bool((idx[1:] >= idx[:-1]).all())


# ----------------------------------------------------------------------------- DiT forward
@pytest.mark.hardware
def test_flow_transformer_forward_golden_length(flow_transformer, ref_transformer, ref_trajectory, golden):
    """One forward at the golden T (689), both CFG rows, at a mid-trajectory latent (t = 0.5)."""
    k = 0
    noise, final, cond = golden["noises"][k], golden["latents"][k], golden["conditions"][k]
    t = 0.5
    if ref_trajectory is not None:
        latent = ref_trajectory[k]["steps"][14]  # the latent the reference feeds at t = 0.5 (timesteps[15])
        source = "ref_trajectory step 14"
    else:
        latent = (1.0 - t) * noise + t * final
        source = "linear blend of golden noise and final latent"
    latents = latent.expand(BATCH, -1, -1).contiguous()
    timestep = torch.full((BATCH,), t)
    condition = _both_rows(cond)
    with torch.no_grad():
        t0 = time.time()
        ref = ref_transformer(latents, timestep, condition)
        ref_s = time.time() - t0
    out = flow_transformer(latents, timestep, condition)
    assert out.shape == ref.shape == (BATCH, 128, noise.shape[-1])
    assert torch.isfinite(out).all()
    pcc_rows = [_pcc(ref[b], out[b]) for b in range(BATCH)]
    pcc_all = _pcc(ref, out)
    max_err = float((ref - out).abs().max())
    logger.info(
        f"DiT forward T={noise.shape[-1]} (S_pad {padded_seq_len(noise.shape[-1])}) at t={t} from {source}: PCC cond {pcc_rows[0]:.5f} uncond {pcc_rows[1]:.5f} all {pcc_all:.5f}, max abs err {max_err:.4f}, ref rms {float(ref.pow(2).mean().sqrt()):.4f}, cpu ref {ref_s:.1f} s"
    )
    _record(
        "dit_forward_T689",
        T=noise.shape[-1],
        S_pad=padded_seq_len(noise.shape[-1]),
        t=t,
        latent_source=source,
        pcc_conditional=pcc_rows[0],
        pcc_unconditional=pcc_rows[1],
        pcc_all=pcc_all,
        max_abs_err=max_err,
        ref_rms=float(ref.pow(2).mean().sqrt()),
    )
    assert min(pcc_rows) >= PCC_FORWARD


@pytest.mark.hardware
@pytest.mark.parametrize("num_latents", [100, 37, 127, 128])
def test_flow_transformer_short_and_unaligned(flow_transformer, ref_transformer, golden, num_latents):
    """T=100 (short), T=37 (S = 38, not a tile multiple), T=127 (S = 128 = S_pad: the no-mask branch; a 37-frame
    single-window song produces exactly this L) and T=128 (S = 129, one row past the boundary, 127 padded rows)."""
    noise, final, cond = golden["noises"][0], golden["latents"][0], golden["conditions"][0]
    t = float(FlowMatchEulerScheduler(NUM_STEPS).timesteps[7])
    latents = ((1.0 - t) * noise + t * final)[..., :num_latents].expand(BATCH, -1, -1).contiguous()
    condition = _both_rows(cond[:, :num_latents])
    timestep = torch.full((BATCH,), t)
    with torch.no_grad():
        ref = ref_transformer(latents, timestep, condition)
    out = flow_transformer(latents, timestep, condition)
    assert out.shape == (BATCH, 128, num_latents) and torch.isfinite(out).all()
    pcc = _pcc(ref, out)
    logger.info(f"DiT forward T={num_latents} (S_pad {padded_seq_len(num_latents)}): PCC {pcc:.5f}")
    _record(f"dit_forward_T{num_latents}", T=num_latents, S_pad=padded_seq_len(num_latents), t=t, pcc=pcc)
    assert pcc >= PCC_FORWARD


@pytest.mark.hardware
def test_flow_transformer_forward_timing(flow_transformer, mm3_mesh_device, golden):
    """Warmed eager wall time of one B=2 forward at the golden T with the condition projection precomputed."""
    noise, cond = golden["noises"][0], golden["conditions"][0]
    latents = noise.expand(BATCH, -1, -1).contiguous()
    timestep = torch.full((BATCH,), 0.5)
    cond_proj = flow_transformer.prepare_condition(_both_rows(cond))
    flow_transformer(latents, timestep, cond_proj=cond_proj)  # warm (program cache is already hot from earlier tests)
    times = _sync_time(mm3_mesh_device, lambda: flow_transformer(latents, timestep, cond_proj=cond_proj), 5)
    prep = _sync_time(mm3_mesh_device, lambda: ttnn.deallocate(flow_transformer.prepare_condition(_both_rows(cond))), 3)
    ttnn.deallocate(cond_proj)
    logger.info(
        f"DiT forward B=2 T={noise.shape[-1]} eager, warmed: median {statistics.median(times) * 1e3:.1f} ms (min {min(times) * 1e3:.1f}); prepare_condition {statistics.median(prep) * 1e3:.1f} ms"
    )
    _record(
        "dit_forward_timing",
        T=noise.shape[-1],
        batch=BATCH,
        mode="eager, host<->device latents/velocity per call",
        forward_seconds=times,
        forward_median_ms=statistics.median(times) * 1e3,
        prepare_condition_median_ms=statistics.median(prep) * 1e3,
    )


# ----------------------------------------------------------------------------- full chunk
@pytest.mark.hardware
@pytest.mark.parametrize("k", [0, 1])
def test_denoise_chunk_golden(flow_transformer, condition_encoder, ref_trajectory, golden, k):
    """Chunk ``k`` with the golden noise (chunk 1 with the golden carry from chunk 0) vs golden latents."""
    fh = golden["frame_hiddens"]
    starts = golden["chunk_starts"]
    start = starts[k]
    end = min(start + CHUNK_FRAMES, fh.shape[1])
    prev_lat = prev_cond = None
    if k > 0:
        prev = golden["latents"][k - 1]
        L0 = prev.shape[-1]
        os_, oe = max(0, L0 - 2 * OVERLAP_LATENT_LENGTH), max(
            max(0, L0 - 2 * OVERLAP_LATENT_LENGTH), L0 - OVERLAP_LATENT_LENGTH
        )
        prev_lat, prev_cond = prev[..., os_:oe], golden["conditions"][k - 1][:, os_:oe]
    golden_final = golden["latents"][k]
    steps_ref = ref_trajectory[k]["steps"] if ref_trajectory is not None else None
    drift = []

    def on_step(i, t, latents):
        if steps_ref is not None:
            drift.append(
                {
                    "step": i,
                    "t": t,
                    "pcc_vs_ref_step": _pcc(steps_ref[i], latents),
                    "max_abs_err_vs_ref_step": float((steps_ref[i] - latents).abs().max()),
                }
            )
        else:
            drift.append({"step": i, "t": t, "pcc_vs_golden_final": _pcc(golden_final, latents)})

    denoiser = ChunkDenoiser(flow_transformer, condition_encoder)
    t0 = time.time()
    result = denoiser.denoise_chunk(
        fh[:, start:end], prev_lat, prev_cond, golden["noises"][k], NUM_STEPS, on_step=on_step
    )
    seconds = time.time() - t0
    assert result.latents.shape == golden_final.shape
    assert torch.isfinite(result.latents).all()
    # the spliced condition must be the one the golden run fed the DiT
    cond_pcc = _pcc(golden["conditions"][k], result.condition)
    if k > 0:
        assert result.overlap == OVERLAP_LATENT_LENGTH
        assert torch.equal(result.condition[:, : result.overlap], golden["conditions"][k][:, : result.overlap])
        assert torch.equal(result.latents[..., : result.overlap], prev_lat[..., : result.overlap])
    pcc = _pcc(golden_final, result.latents)
    max_err = float((golden_final - result.latents).abs().max())
    body_pcc = _pcc(golden_final[..., result.overlap :], result.latents[..., result.overlap :])
    for d in drift:
        logger.info("  " + ", ".join(f"{kk}={v:.5f}" if isinstance(v, float) else f"{kk}={v}" for kk, v in d.items()))
    logger.info(
        f"chunk {k}: {end - start} frames, {golden_final.shape[-1]} latents, overlap {result.overlap}: latent PCC {pcc:.5f} (body {body_pcc:.5f}), max abs err {max_err:.3f}, condition PCC {cond_pcc:.6f}, {seconds:.1f} s for {NUM_STEPS} steps"
    )
    _record(
        f"denoise_chunk{k}",
        frames=end - start,
        latents=golden_final.shape[-1],
        overlap=result.overlap,
        steps=NUM_STEPS,
        pcc=pcc,
        pcc_excluding_overlap=body_pcc,
        max_abs_err=max_err,
        condition_pcc=cond_pcc,
        seconds=seconds,
        drift_reference="fp32 torch trajectory (generated/ref_trajectory.pt)"
        if steps_ref is not None
        else "golden final latent",
        drift=drift,
    )
    assert pcc >= PCC_CHUNK
    assert cond_pcc >= PCC_CONDITION


@pytest.mark.hardware
def test_denoise_three_windows_short_tail(flow_transformer, condition_encoder, golden):
    """Three windows with OUR carry and a 101-frame last window (L = 347: the carry window [3, 175) overlaps the
    172 restored latents) - the awkward chunking tail the golden clip does not reach. 2 Euler steps per window."""
    fh = golden["frame_hiddens"]
    frames = 301
    fh3 = fh.repeat(1, -(-frames // fh.shape[1]), 1)[:, :frames]
    starts = chunk_starts_for(frames)
    assert starts == [0, 100, 200]
    lengths = [latent_length(min(s + CHUNK_FRAMES, frames) - s) for s in starts]
    assert lengths == [689, 689, 347]
    g = torch.Generator().manual_seed(0)
    noises = [torch.randn(1, 128, L, generator=g) for L in lengths]
    denoiser = ChunkDenoiser(flow_transformer, condition_encoder)
    results = denoiser.denoise(fh3, noises, chunk_starts=starts, steps=2)
    assert [r.latents.shape[-1] for r in results] == lengths
    assert all(torch.isfinite(r.latents).all() for r in results)
    assert [r.overlap for r in results] == [0, OVERLAP_LATENT_LENGTH, OVERLAP_LATENT_LENGTH]
    for prev, cur in zip(results, results[1:]):
        assert torch.equal(cur.latents[..., :OVERLAP_LATENT_LENGTH], prev.previous_latent[..., :OVERLAP_LATENT_LENGTH])
        assert torch.equal(cur.condition[:, :OVERLAP_LATENT_LENGTH], prev.previous_condition[:, :OVERLAP_LATENT_LENGTH])
    assert results[2].previous_latent.shape[-1] == 175 - 3 and results[2].previous_condition.shape[1] == 175 - 3
    _record(
        "denoise_three_windows_short_tail",
        frames=frames,
        chunk_starts=starts,
        latents=lengths,
        steps=2,
        overlaps=[r.overlap for r in results],
    )


@pytest.mark.hardware
@pytest.mark.slow
def test_denoise_all_chunks_chained(flow_transformer, condition_encoder, golden):
    """Both windows with OUR carry (no golden splice), the way stage 06 will run them."""
    denoiser = ChunkDenoiser(flow_transformer, condition_encoder)
    results = denoiser.denoise(golden["frame_hiddens"], golden["noises"], chunk_starts=golden["chunk_starts"])
    pccs = [_pcc(golden["latents"][k], r.latents) for k, r in enumerate(results)]
    logger.info(f"chained chunks PCC {pccs}")
    _record("denoise_chained", pcc_per_chunk=pccs)
    assert min(pccs) >= PCC_CHUNK
