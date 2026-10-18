# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.experimental.xtts.config import CHUNKING, DEMO, GENERATION
from models.experimental.xtts.reference.xtts_conditioning import MEL_SR, NUM_LATENTS, load_coqui_test_audio
from models.experimental.xtts.reference.xtts_gpt_generate import MAX_AUDIO_TOKENS, STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_inference import XttsReference
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_inference import TtXtts

TILE = 32
SECONDS_PER_CODE = 4 * (24000 / 22050) * 256 / 24000

DEFAULT_SWEEP = [96, 128, 160, 192, 224, 256, 272]

FLOOR_TEXT_LEN = 96

OVERSHOOT = 1.2
CODES_PER_ID = CHUNKING.codes_per_id


def _budget_for(text_len):
    """Estimate a tile-aligned code budget for a given wrapped text length."""
    est = int(text_len * CODES_PER_ID * OVERSHOOT)
    return min(MAX_AUDIO_TOKENS, -(-est // TILE) * TILE)


SWEEP_TEXT = (
    "Voice synthesis has come a long way, and modern systems can already generate natural sounding "
    "speech with remarkable accuracy. "
) * 30

BUDGET_LADDER = [224, 256, 288, 320, 352, 384, 448, 512, 576, MAX_AUDIO_TOKENS]

L1_SMALL_SIZE = 65536
TRACE_REGION_SIZE = 52428800


def _budget_ladder():
    """Return the traced budget ladder from env or defaults."""
    raw = os.environ.get("XTTS_TRACED_BUDGETS")
    if not raw or not raw.strip():
        return list(BUDGET_LADDER)
    return [int(x) for x in raw.split(",") if x.strip()]


def _grid():
    """Build (text_len, budgets) pairs for the envelope map."""
    pairs = []
    for text_len in _sweep_points():
        need = -(-int(text_len * CODES_PER_ID) // TILE) * TILE
        pairs.append((text_len, [b for b in _budget_ladder() if b >= need]))
    return pairs


def _sweep_points():
    """Return traced text-length sweep points from env or defaults."""
    raw = os.environ.get("XTTS_TRACED_SWEEP")
    if not raw or not raw.strip():
        return list(DEFAULT_SWEEP)
    return [int(x) for x in raw.split(",") if x.strip()]


def _classify(exc):
    """Classify an allocation/runtime failure into a short failure tag."""
    msg = str(exc)
    if "clash with L1 buffers" in msg or "Statically allocated circular buffers" in msg:
        return "CB_CLASH"
    if "L1_SMALL" in msg or "l1_small" in msg:
        return "L1_SMALL"
    if "Out of Memory" in msg or "out of memory" in msg or "Failed to allocate" in msg:
        return "OOM"
    return f"OTHER({type(exc).__name__})"


def _inputs():
    """Load reference and speaker wav inputs for traced pass sweeps."""
    from scipy.signal import resample_poly

    wav = load_coqui_test_audio(samples=DEMO.ref_audio.split("+"), max_seconds=DEMO.ref_seconds)
    q = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(
        resample_poly(wav.reshape(-1).numpy().astype("float32"), SPK_SR // q, MEL_SR // q).astype("float32")
    ).unsqueeze(0)[:, : SPK_SR * DEMO.spk_seconds]
    return wav, spk_wav


def _text_at(text_len):
    """Build wrapped text ids padded or trimmed to the requested length."""
    ids = preprocess_text(SWEEP_TEXT, lang=DEMO.language)[:, : text_len - 2]
    wrapped = wrap_text_ids(ids)
    if wrapped.shape[1] < text_len:
        wrapped = F.pad(wrapped, (0, text_len - wrapped.shape[1]), value=STOP_TEXT_TOKEN)
    return wrapped


def _run_one(state_dict, ref_decoder_full, wav, spk_wav, wrapped, text_len, budget):
    """Run one fully-traced pass on a fresh device and record PASS/failure."""
    max_seq = -(-(NUM_LATENTS + text_len + budget + 2) // TILE) * TILE
    row = {"text_len": text_len, "budget": budget, "max_seq": max_seq}
    device = ttnn.open_device(device_id=0, l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE)
    try:
        tt = TtXtts(device, state_dict, ref_decoder_full)
        spk_dev = ttnn.from_torch(
            spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
        )
        wav_dev, codes, perf = tt.inference_fully_traced(
            wrapped,
            wav,
            spk_dev,
            max_seq,
            max_new_tokens=budget,
            temperature=GENERATION.temperature,
            top_k=GENERATION.top_k,
            top_p=GENERATION.top_p,
            repetition_penalty=GENERATION.repetition_penalty,
            # STOP left free so the pass ends with the text (not drone past it).
            min_new_tokens=0,
        )
        out = ttnn.to_torch(wav_dev).float().reshape(-1)
        row.update(
            result="PASS",
            codes=int(codes.shape[1]),
            audio_s=out.shape[0] / 24000.0,
            replay_s=float(perf["replay_s"]),
            compile_s=float(perf["compile_s"]),
            detail=(
                f"setup {perf['setup_replay_s']:.3f}s | decode {perf['decode_replay_s']:.3f}s "
                f"| vocoder {perf['vocoder_replay_s']:.3f}s"
            ),
        )
    except Exception as exc:
        row.update(result=_classify(exc), detail=str(exc).splitlines()[0][:150])
    finally:
        ttnn.close_device(device)
    return row


@pytest.mark.timeout(7200)
@pytest.mark.models_performance_bare_metal
def test_traced_pass_envelope_map(xtts_state_dict):
    """Map which (text_len, budget) pairs allocate for a single traced pass."""
    wav, spk_wav = _inputs()
    reference = XttsReference(xtts_state_dict)
    grid = _grid()
    total = sum(len(bs) for _, bs in grid)
    logger.info(f"envelope map: {total} (text_len, budget) points, fresh device each")

    results = {}
    for text_len, budgets in grid:
        wrapped = _text_at(text_len)
        for budget in budgets:
            row = _run_one(xtts_state_dict, reference.decoder_full, wav, spk_wav, wrapped, text_len, budget)
            results[(text_len, budget)] = row
            extra = f"{row['codes']} codes, {row['audio_s']:.2f}s" if row["result"] == "PASS" else row["result"]
            logger.info(f"[map] text_len {text_len:>4} budget {budget:>4} (max_seq {row['max_seq']:>4}) -> {extra}")

    ladder = _budget_ladder()
    logger.info("\nTRACED SINGLE-PASS ENVELOPE MAP  (row = text_len, col = budget; P = pass, . = CB clash)")
    logger.info("text_len | " + " ".join(f"{b:>4}" for b in ladder))
    for text_len, budgets in grid:
        cells = []
        for b in ladder:
            r = results.get((text_len, b))
            cells.append("   -" if r is None else ("   P" if r["result"] == "PASS" else "   ."))
        logger.info(f"{text_len:>8} | " + " ".join(cells))

    working = [(t, b, r) for (t, b), r in sorted(results.items()) if r["result"] == "PASS"]
    logger.info(f"\nworking pairs: {len(working)} of {total}")
    for t, b, r in working:
        logger.info(
            f"  text_len {t:>4} budget {b:>4} -> {r['codes']:>3} codes, {r['audio_s']:>6.2f}s audio, "
            f"replay {r['replay_s']:.3f}s"
        )
    if working:
        best = max(working, key=lambda w: w[2]["codes"])
        logger.info(
            f"longest single pass measured: text_len {best[0]}, budget {best[1]} -> {best[2]['codes']} codes, "
            f"{best[2]['audio_s']:.2f}s of audio (vs max_single_pass_codes {CHUNKING.max_single_pass_codes})"
        )

    assert working, (
        "no (text_len, budget) pair allocated at all — the traced path is broken, not envelope-limited. "
        f"Failure classes seen: {sorted({r['result'] for r in results.values()})}"
    )


@pytest.mark.timeout(7200)
@pytest.mark.models_performance_bare_metal
def test_traced_pass_len_sweep(xtts_state_dict):
    """Sweep traced single-pass text lengths and assert the demo floor still passes."""
    wav, spk_wav = _inputs()
    reference = XttsReference(xtts_state_dict)

    rows = []
    for text_len in _sweep_points():
        wrapped = _text_at(text_len)
        budget = _budget_for(text_len)
        row = _run_one(xtts_state_dict, reference.decoder_full, wav, spk_wav, wrapped, text_len, budget)
        logger.info(
            f"[traced] text_len {text_len} budget {budget} (max_seq {row['max_seq']}) -> "
            f"{row['result']} {row.get('detail', '')}"
        )
        rows.append(row)

    logger.info("\nTRACED SINGLE-PASS TEXT-LENGTH SWEEP — fresh device per point")
    logger.info(
        f"{'text_len':>9} {'budget':>7} {'max_seq':>8} {'codes':>6} {'audio_s':>8} "
        f"{'replay_s':>9} {'result':>10}  detail"
    )
    for r in rows:
        logger.info(
            f"{r['text_len']:>9} {r['budget']:>7} {r['max_seq']:>8} {r.get('codes', 0):>6} "
            f"{r.get('audio_s', 0):>8.2f} {r.get('replay_s', 0):>9.3f} {r['result']:>10}  {r.get('detail', '')}"
        )

    passed = [r for r in rows if r["result"] == "PASS"]
    failed = [r["text_len"] for r in rows if r["result"] != "PASS"]
    if passed:
        best = max(passed, key=lambda r: r["text_len"])
        logger.info(
            f"longest passing single pass: text_len {best['text_len']} -> {best['codes']} codes, "
            f"{best['audio_s']:.2f} s of audio in ONE pass"
        )
        logger.info(
            f"vs CHUNKING.max_single_pass_codes {CHUNKING.max_single_pass_codes} (text-split threshold), "
            f"max_chunk_codes {CHUNKING.max_chunk_codes}, GENERATION.max_tokens {GENERATION.max_tokens}"
        )
        span = max(r["audio_s"] for r in passed) - min(r["audio_s"] for r in passed)
        logger.info(f"audio_s span across passing rows: {span:.2f} s (must grow with text_len, else axis is inert)")
    else:
        logger.info("nothing passed")
    if failed:
        logger.info(f"first failing text_len: {min(failed)} | all failing: {failed}")

    floor = next((r for r in rows if r["text_len"] == FLOOR_TEXT_LEN), None)
    if floor is not None:
        assert floor["result"] == "PASS", (
            f"regression: the demo's default text length ({FLOOR_TEXT_LEN} ids) no longer runs "
            f"traced: {floor.get('detail')}"
        )
        assert passed, "no text length ran at all — the traced path is broken, not length-limited"
