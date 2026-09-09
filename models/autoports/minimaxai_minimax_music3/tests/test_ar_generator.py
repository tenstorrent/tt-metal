# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 04 tests for ``ARGenerator`` (global LLM + depth decoder + sampling on one Blackhole chip).

Reference: the stage-01 golden tensors (fp32 diffusers run of the 10 s clip, seed 7) under
``~/mm3-bringup/reference`` (``manifest.json``).

Run (device, serialized):

    source ~/mm3-bringup/common.sh && cd $MM3_WT && \
    with_hw_lock timeout 5400 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_ar_generator.py -m "not slow"

Evidence is written to ``doc/ar_generator/pcc/results.json``.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt import prompt as P
from models.autoports.minimaxai_minimax_music3.tt.constants import (
    AUDIO_CFG_TOKEN_ID,
    AUDIO_CODE_OFFSET,
    AUDIO_VOCAB_SIZE,
    FRAME_RATE,
    MAX_PROMPT_TOKENS,
    NUM_CODEBOOKS,
    SEMANTIC_VOCAB_SIZE,
)
from models.common.utility_functions import comp_pcc

PCC_FRAME = 0.99  # per-frame and overall frame_hiddens PCC vs the golden
DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "ar_generator"
MANIFEST = R.reference_dir() / "manifest.json"


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


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def manifest():
    if not MANIFEST.is_file():
        pytest.skip(f"golden manifest missing: {MANIFEST}")
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="session")
def golden_ar(manifest):
    root = R.reference_dir()
    raw = torch.load(root / "sampled_raw.pt")  # every top-k draw, frame 0 included
    codes = torch.load(root / "sampled_codes.pt")  # [F, 8] emitted frames
    groups = raw[: (raw.numel() // NUM_CODEBOOKS) * NUM_CODEBOOKS].reshape(-1, NUM_CODEBOOKS).clone()
    groups[:, 0] -= AUDIO_CODE_OFFSET
    assert torch.equal(groups[1 : codes.shape[0] + 1], codes), "sampled_raw / sampled_codes disagree"
    return {
        "text_ids": torch.load(root / "text_ids.pt"),
        "codes": codes,
        "frame0_codes": groups[0],
        "frame_hiddens": torch.load(root / "frame_hiddens.pt"),  # [1, F, 32768]
    }


@pytest.fixture(scope="session")
def prompt_encoder():
    return P.PromptEncoder()


@pytest.fixture(scope="session")
def depth_decoder(mm3_mesh_device):
    from models.autoports.minimaxai_minimax_music3.tt.depth_decoder import DepthDecoder

    dec = DepthDecoder.from_pretrained(mm3_mesh_device, R.weights_dir())
    logger.info(f"DepthDecoder weights on device in {dec.load_seconds:.1f}s")
    return dec


@pytest.fixture(scope="session")
def ar(music_llm, depth_decoder):
    from models.autoports.minimaxai_minimax_music3.tt.ar_generator import ARGenerator

    gen = ARGenerator(music_llm, depth_decoder)
    yield gen
    gen.release()


# ----------------------------------------------------------------------------- prompt contract (host only)
def test_build_text_ids_matches_golden(prompt_encoder, manifest, golden_ar):
    ids = prompt_encoder.encode(manifest["prompt"], manifest["lyrics"])
    assert ids.shape == tuple(manifest["text_ids_shape"]), ids.shape
    assert torch.equal(ids, golden_ar["text_ids"]), "text_ids differ from the golden"
    # The CFG row keeps <|im_start|> and the two trailing structure tokens only.
    assert torch.equal(ids[1, 1:-2], torch.full_like(ids[1, 1:-2], AUDIO_CFG_TOKEN_ID))
    assert torch.equal(ids[1, [0, -2, -1]], ids[0, [0, -2, -1]])
    _record("text_ids", exact_match=True, shape=list(ids.shape))


def test_prompt_contract_edge_cases(prompt_encoder, expect_error):
    # 1. Structure tags on the same line as lyric text: the text is dropped, consecutive tags kept.
    lyr = P.normalize_lyrics("[Verse] Morning light\n[chorus][bridge] la la\nplain line [intro] inline")
    # Expected strings below are the outputs of diffusers' _normalize_lyrics / _clean_caption (ref venv).
    assert lyr == "[start]\n[verse]\n[chorus][bridge]\nplain line\n[intro]\ninline", repr(lyr)
    ids = prompt_encoder.encode("Genre: pop.", "[verse] Morning light filtering through the pine\n[chorus]\nSoftly")
    assert ids.shape[0] == 2 and ids.shape[1] > 8
    # 2. Markdown in the caption.
    cap = P.clean_caption("# Title\n- **Genre:** *acoustic* pop\n---\n\n\nBPM: 96 <|key C major|> • warm    tone")
    assert cap == "Title\nGenre: acoustic pop\nBPM: 96 key is C major warmtone", repr(cap)
    md_ids = prompt_encoder.encode("# Title\n- **Genre:** *acoustic* pop\n---\nBPM: 96", "[verse]\nla la la")
    plain_ids = prompt_encoder.encode("Title\nGenre: acoustic pop\nBPM: 96", "[verse]\nla la la")
    assert torch.equal(md_ids, plain_ids)
    # 3. Exactly 5000 tokens tokenizes; one more raises the reference's ValueError.
    tok = prompt_encoder.tokenizer
    caption = "Genre: acoustic pop. BPM: 96."
    words = ["la"] * 4000
    while True:
        n = tok(P.assemble_prompt(caption, "[verse]\n" + " ".join(words)), return_tensors="pt")["input_ids"].shape[1]
        if n == MAX_PROMPT_TOKENS:
            break
        assert n < MAX_PROMPT_TOKENS, n
        words.extend(["la"] * max(1, (MAX_PROMPT_TOKENS - n) // 2))
    ids = prompt_encoder.encode(caption, "[verse]\n" + " ".join(words))
    assert ids.shape == (2, MAX_PROMPT_TOKENS), ids.shape
    with expect_error(ValueError, "maximum is 5000"):
        prompt_encoder.encode(caption, "[verse]\n" + " ".join(words + ["la"]))
    # 4. Empty inputs are rejected like the reference's check_inputs.
    with expect_error(ValueError, "must be a non-empty string"):
        prompt_encoder.encode("", "[verse]\nla")
    with expect_error(ValueError, "must be a non-empty string"):
        prompt_encoder.encode("pop", "   ")
    _record("prompt_edge_cases", tags_inline=True, markdown=True, max_prompt_tokens=MAX_PROMPT_TOKENS)


# ----------------------------------------------------------------------------- device
@pytest.mark.hardware
@pytest.mark.timeout(3600)
def test_teacher_forced_frame_hiddens_vs_golden(ar, manifest, golden_ar):
    """Teacher-forced run over the golden codes: frame_hiddens PCC >= 0.99 per frame and overall."""
    ids = ar.build_text_ids(manifest["prompt"], manifest["lyrics"])
    assert torch.equal(ids, golden_ar["text_ids"])
    golden = golden_ar["frame_hiddens"]
    F_ = golden.shape[1]
    t0 = time.perf_counter()
    out = ar.generate(
        manifest["prompt"],
        manifest["lyrics"],
        max_frames=F_,
        seed=manifest["seed"],
        teacher_codes=golden_ar["codes"],
        teacher_frame0_codes=golden_ar["frame0_codes"],
    )
    wall = time.perf_counter() - t0
    assert out["frames"] == F_ and out["stopped_by"] == "teacher_codes", (out["frames"], out["stopped_by"])
    assert torch.equal(out["codes"], golden_ar["codes"])
    assert torch.equal(out["frame0_codes"], golden_ar["frame0_codes"])
    fh = out["frame_hiddens"]
    assert fh.shape == golden.shape, (fh.shape, golden.shape)

    per_frame = [_pcc(golden[0, f], fh[0, f]) for f in range(F_)]
    # Split per frame into the backbone part (first 4096) and the seven depth parts.
    llm_part = [_pcc(golden[0, f, :4096], fh[0, f, :4096]) for f in range(F_)]
    depth_part = [_pcc(golden[0, f, 4096:], fh[0, f, 4096:]) for f in range(F_)]
    overall = _pcc(golden, fh)
    ranks = out["semantic_ranks"]
    top1 = sum(r["rank_guided"] == 0 for r in ranks) / len(ranks)
    top50 = sum(r["rank_conditional"] < 50 for r in ranks) / len(ranks)
    logger.info(
        f"teacher-forced {F_} frames in {wall:.1f}s: overall PCC {overall:.5f}, per-frame min {min(per_frame):.5f} "
        f"(frame {per_frame.index(min(per_frame))}), mean {sum(per_frame) / F_:.5f}; backbone min {min(llm_part):.5f}, "
        f"depth min {min(depth_part):.5f}; golden semantic code: top-1 of guided {top1:.3f}, in conditional top-50 {top50:.3f}"
    )
    for f in range(0, F_, 25):
        logger.info(f"  frame {f:3d}: PCC {per_frame[f]:.5f} (backbone {llm_part[f]:.5f}, depth {depth_part[f]:.5f})")
    _record(
        "teacher_forced_golden",
        bar=PCC_FRAME,
        frames=F_,
        overall_pcc=overall,
        per_frame_min=min(per_frame),
        per_frame_mean=sum(per_frame) / F_,
        per_frame_pcc=per_frame,
        backbone_part_min=min(llm_part),
        depth_part_min=min(depth_part),
        golden_code_top1_guided=top1,
        golden_code_in_conditional_top50=top50,
        wall_seconds=wall,
        timings={k: v for k, v in out["timings"].items() if k != "per_frame"},
        decode_stats=out["decode_stats"],
    )
    assert overall >= PCC_FRAME, overall
    assert min(per_frame) >= PCC_FRAME, (min(per_frame), per_frame.index(min(per_frame)))


def _check_codes(codes: torch.Tensor):
    assert codes.dim() == 2 and codes.shape[1] == NUM_CODEBOOKS, codes.shape
    assert int(codes[:, 0].min()) >= 0 and int(codes[:, 0].max()) < SEMANTIC_VOCAB_SIZE
    assert int(codes[:, 1:].min()) >= 0 and int(codes[:, 1:].max()) < AUDIO_VOCAB_SIZE


@pytest.mark.hardware
@pytest.mark.timeout(3600)
def test_free_running_generation(ar, manifest):
    """50 free-running frames (seed 7): valid codes, deterministic per seed, seed-dependent, non-degenerate, timed."""
    n = 50
    runs = {}
    for seed in (7, 7, 11):
        out = ar.generate(manifest["prompt"], manifest["lyrics"], max_frames=n, seed=seed)
        assert out["frames"] == n and out["stopped_by"] == "max_frames", (out["frames"], out["stopped_by"])
        assert out["frame_hiddens"].shape == (1, n, NUM_CODEBOOKS * 4096)
        assert torch.isfinite(out["frame_hiddens"]).all()
        _check_codes(out["codes"])
        runs.setdefault(seed, []).append(out)
    a, b = runs[7]
    assert torch.equal(a["codes"], b["codes"]), "same seed, different codes"
    assert torch.equal(a["frame0_codes"], b["frame0_codes"])
    assert torch.equal(a["frame_hiddens"], b["frame_hiddens"]), "same seed, different frame_hiddens"
    c = runs[11][0]
    assert not torch.equal(a["codes"], c["codes"]), "different seed, identical codes"

    # Qualitative: the semantic-code distribution is not degenerate.
    hist = Counter(a["codes"][:, 0].tolist())
    top_code, top_count = hist.most_common(1)[0]
    top_frac = top_count / n
    distinct = len(hist)
    logger.info(
        f"seed 7: {distinct} distinct semantic codes in {n} frames, most common {top_code} x{top_count} ({top_frac:.0%})"
    )
    assert top_frac <= 0.30, (top_code, top_frac)

    # Performance of the warmed loop (third run): host wall per frame, split by section.
    t = c["timings"]
    per_frame = t["per_frame"]
    warm = per_frame[5:]  # skip the first frames (first-frame host paths, caches)
    fps = len(warm) / sum(warm)
    total = t["llm_step"] + t["depth"] + t["host"]
    frames_run = len(per_frame)
    perf = {
        "frames": n,
        "prefill_s": t["prefill"],
        "frames_per_s_warm": fps,
        "ms_per_frame_warm": 1e3 / fps,
        "ms_per_frame_all": 1e3 * sum(per_frame) / frames_run,
        "llm_step_ms": 1e3 * t["llm_step"] / frames_run,
        "depth_loop_ms": 1e3 * t["depth"] / frames_run,
        "host_sampling_embed_ms": 1e3 * t["host"] / frames_run,
        "section_share": {k: t[k] / total for k in ("llm_step", "depth", "host")},
        "realtime_frames_per_s": FRAME_RATE,
        "realtime_gap_x": FRAME_RATE / fps,
        "decode_stats": c["decode_stats"],
    }
    logger.info(
        f"free-running: {fps:.2f} frames/s warm ({1e3 / fps:.1f} ms/frame: LLM {perf['llm_step_ms']:.1f} + depth "
        f"{perf['depth_loop_ms']:.1f} + host {perf['host_sampling_embed_ms']:.1f}); realtime needs {FRAME_RATE}, gap {perf['realtime_gap_x']:.1f}x"
    )
    _record(
        "free_running",
        seed7_codes_head=a["codes"][:8].tolist(),
        seed7_frame0_codes=a["frame0_codes"].tolist(),
        deterministic_same_seed=True,
        differs_other_seed=True,
        distinct_semantic_codes=distinct,
        most_common_semantic_code=[top_code, top_frac],
        perf=perf,
    )
    # Steady state must not refresh positions from the host: one refresh per generate() (after prefill).
    assert c["decode_stats"]["position_refreshes"] - a["decode_stats"]["position_refreshes"] <= 2


@pytest.mark.hardware
@pytest.mark.timeout(3600)
def test_end_token_for_short_lyric(ar):
    """A very short lyric with a generous max_frames ends by itself with the end token (qualitative check)."""
    max_frames = 1500  # 60 s of audio; a one-line lyric should end well before that
    out = ar.generate(
        "Genre: acoustic pop. BPM: 96. Key: C major. A short intimate vocal phrase over one guitar.",
        "[verse]\nMorning light through the pine",
        max_frames=max_frames,
        seed=7,
    )
    _check_codes(out["codes"])
    hist = Counter(out["codes"][:, 0].tolist())
    top_code, top_count = hist.most_common(1)[0]
    logger.info(
        f"short lyric: stopped_by={out['stopped_by']} after {out['frames']} frames ({out['frames'] / FRAME_RATE:.1f} s); "
        f"{len(hist)} distinct semantic codes, most common {top_code} x{top_count}"
    )
    _record(
        "end_token_short_lyric",
        stopped_by=out["stopped_by"],
        frames=out["frames"],
        seconds=out["frames"] / FRAME_RATE,
        max_frames=max_frames,
        distinct_semantic_codes=len(hist),
        most_common_semantic_code=[top_code, top_count / out["frames"]],
    )
    assert out["stopped_by"] == "end_token", out["stopped_by"]
    assert top_count / out["frames"] <= 0.30, (top_code, top_count)
