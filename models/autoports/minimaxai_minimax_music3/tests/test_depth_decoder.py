# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 03 tests for ``DepthDecoder`` (the RVQ depth decoder on one Blackhole chip).

Reference: the torch transcription in ``reference/depth_decoder_ref.py`` run in fp32 from the bf16
safetensors, plus the stage-01 golden tensors (fp32 diffusers run) under ``~/mm3-bringup/reference``.

Run (device, serialized):

    source ~/mm3-bringup/common.sh && cd $MM3_WT && \
    with_hw_lock timeout 3600 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_depth_decoder.py -m "not slow"

Every PCC / timing is written to ``doc/depth_decoder/pcc/pcc_results.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import depth_decoder_ref as REF
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.constants import AUDIO_CODE_OFFSET, AUDIO_VOCAB_SIZE, NUM_CODEBOOKS
from models.autoports.minimaxai_minimax_music3.tt.depth_decoder import DepthDecoder, DepthStepTrace
from models.common.utility_functions import comp_pcc

pytestmark = [pytest.mark.hardware, pytest.mark.timeout(1800)]

PCC_LAYER = 0.995  # per-step hidden and head logits vs the fp32 torch reference
PCC_GOLDEN = 0.99  # full 7-step loop vs the fp32 diffusers golden frame
DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "depth_decoder"


def _record(name: str, **fields):
    out = DOC_DIR / "pcc"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pcc_results.json"
    results = json.loads(path.read_text()) if path.is_file() else {}
    results[name] = fields
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def _pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    _, pcc = comp_pcc(golden.float(), actual.float(), 0.0)
    return float(pcc)


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def depth_decoder(mm3_mesh_device):
    dec = DepthDecoder.from_pretrained(mm3_mesh_device, R.weights_dir())
    logger.info(f"DepthDecoder weights on device in {dec.load_seconds:.1f}s")
    return dec


@pytest.fixture(scope="session")
def depth_ref():
    torch.set_num_threads(8)
    return REF.load_reference(R.weights_dir(), torch.float32)


@pytest.fixture(scope="session")
def golden_frame():
    """Golden frame 1: backbone hidden, semantic-code embedding, residual codes and the 7 depth hiddens (fp32)."""
    root = R.reference_dir()
    if not (root / "frame_hiddens.pt").is_file():
        pytest.skip(f"golden reference missing under {root}")
    frame_hiddens = torch.load(root / "frame_hiddens.pt")  # [1, frames, 8 * 4096]
    codes = torch.load(root / "sampled_codes.pt")  # [frames, 8]
    embed_weight = R.load_embed_weight()
    c0 = int(codes[0, 0])
    return {
        "global_hidden": frame_hiddens[0, 0, :4096].reshape(1, -1).repeat(2, 1).float(),
        "semantic_embed": embed_weight[c0 + AUDIO_CODE_OFFSET].reshape(1, -1).repeat(2, 1).float(),
        "residual_codes": codes[0, 1:].reshape(1, -1).repeat(2, 1),  # [2, 7]
        "depth_hiddens": frame_hiddens[0, 0, 4096:].reshape(NUM_CODEBOOKS - 1, 4096).float(),  # [7, 4096]
    }


@pytest.fixture(scope="session")
def ref_sequence(depth_ref, golden_frame):
    """The reference's full 9-step projected input sequence for golden frame 1 (steps: hidden, c0, c1..c7)."""
    g = golden_frame
    with torch.no_grad():
        rows = [depth_ref.projection(g["global_hidden"]), depth_ref.projection(g["semantic_embed"])]
        for k in range(1, NUM_CODEBOOKS):
            emb = depth_ref.audio_embeddings(g["residual_codes"][:, k - 1] + (k - 1) * AUDIO_VOCAB_SIZE)
            rows.append(depth_ref.projection(emb))
    return torch.stack(rows, dim=1)  # [2, 9, 4096]


# ----------------------------------------------------------------------------- 1. forward + heads per step length
@pytest.mark.parametrize("steps", list(range(2, 9)))
def test_forward_and_heads_vs_reference(depth_decoder, depth_ref, ref_sequence, steps):
    """Every step length the AR loop uses (2..8): last-step hidden and all 7 head logits, PCC >= 0.995."""
    x = ref_sequence[:, :steps]
    with torch.no_grad():
        ref_hidden = depth_ref(x)[:, -1]  # [2, 4096]
        ref_logits = [depth_ref.audio_heads[k](ref_hidden) for k in range(NUM_CODEBOOKS - 1)]

    tt_hidden_dev = depth_decoder.forward(x, num_steps=steps)
    tt_hidden = DepthDecoder.rows_to_host(tt_hidden_dev)
    pcc_hidden = _pcc(ref_hidden, tt_hidden)
    pcc_heads = []
    for k in range(1, NUM_CODEBOOKS):
        tt_logits = DepthDecoder.rows_to_host(depth_decoder.head(k, tt_hidden_dev))
        pcc_heads.append(_pcc(ref_logits[k - 1], tt_logits))
    # The fused all-heads matmul must agree with the per-head ones.
    fused = DepthDecoder.rows_to_host(depth_decoder.heads_all(tt_hidden_dev))
    pcc_fused = min(
        _pcc(ref_logits[k - 1], fused[:, (k - 1) * AUDIO_VOCAB_SIZE : k * AUDIO_VOCAB_SIZE])
        for k in range(1, NUM_CODEBOOKS)
    )
    logger.info(
        f"steps={steps}: hidden PCC {pcc_hidden:.5f}, heads min PCC {min(pcc_heads):.5f}, fused min {pcc_fused:.5f}"
    )
    _record(f"forward_steps_{steps}", hidden_pcc=pcc_hidden, head_pccs=pcc_heads, fused_heads_min_pcc=pcc_fused)
    assert pcc_hidden >= PCC_LAYER, pcc_hidden
    assert min(pcc_heads) >= PCC_LAYER, pcc_heads
    assert pcc_fused >= PCC_LAYER, pcc_fused


# ----------------------------------------------------------------------------- 2. golden frame, full depth loop
def test_golden_frame_depth_loop(depth_decoder, depth_ref, golden_frame):
    """Teacher-forced 7-step loop for golden frame 1 reproduces ``frame_hiddens[0, 0, 4096:]`` with PCC >= 0.99."""
    g = golden_frame
    # Control: the fp32 torch transcription itself vs the diffusers golden (checks the loop / code alignment).
    ref_hiddens, ref_logits = REF.teacher_forced_depth_loop(
        depth_ref, g["global_hidden"], g["semantic_embed"], g["residual_codes"]
    )
    ref_cat = torch.stack([h[0] for h in ref_hiddens])  # [7, 4096]
    pcc_ref = _pcc(g["depth_hiddens"], ref_cat)
    logger.info(f"torch reference vs golden: PCC {pcc_ref:.6f}")
    assert pcc_ref > 0.999, pcc_ref
    ref_argmax_hits = [int(l[0].argmax()) == int(g["residual_codes"][0, i]) for i, l in enumerate(ref_logits)]

    hiddens, logits = depth_decoder.teacher_forced_loop(g["global_hidden"], g["semantic_embed"], g["residual_codes"])
    tt_cat = torch.stack([DepthDecoder.rows_to_host(h)[0] for h in hiddens])  # [7, 4096], conditional row
    per_step = [_pcc(g["depth_hiddens"][i], tt_cat[i]) for i in range(NUM_CODEBOOKS - 1)]
    pcc_all = _pcc(g["depth_hiddens"].flatten(), tt_cat.flatten())
    # Teacher-forced argmax agreement with the golden codes is informative (not a gate: the pipeline samples top-k).
    argmax_hits = [
        int(DepthDecoder.rows_to_host(l)[0].argmax()) == int(g["residual_codes"][0, i]) for i, l in enumerate(logits)
    ]
    logger.info(
        f"golden frame loop: PCC {pcc_all:.5f}, per step {[f'{p:.4f}' for p in per_step]}, argmax hits {argmax_hits}"
    )
    _record(
        "golden_frame1_depth_loop",
        pcc_all_steps=pcc_all,
        pcc_per_step=per_step,
        torch_ref_vs_golden_pcc=pcc_ref,
        argmax_matches_golden_code=argmax_hits,
        torch_ref_argmax_matches_golden_code=ref_argmax_hits,
        tt_argmax_equals_ref_argmax=[
            int(DepthDecoder.rows_to_host(l)[0].argmax()) == int(r[0].argmax()) for l, r in zip(logits, ref_logits)
        ],
    )
    assert pcc_all >= PCC_GOLDEN, pcc_all
    assert min(per_step) >= PCC_GOLDEN, per_step


# ----------------------------------------------------------------------------- 3. determinism
def test_determinism(depth_decoder, golden_frame):
    g = golden_frame
    runs = []
    for _ in range(2):
        hiddens, logits = depth_decoder.teacher_forced_loop(
            g["global_hidden"], g["semantic_embed"], g["residual_codes"]
        )
        runs.append(
            (
                torch.stack([DepthDecoder.rows_to_host(h) for h in hiddens]),
                torch.stack([DepthDecoder.rows_to_host(l) for l in logits]),
            )
        )
    assert torch.equal(runs[0][0], runs[1][0]), "hidden states differ between identical runs"
    assert torch.equal(runs[0][1], runs[1][1]), "logits differ between identical runs"
    _record("determinism", identical=True)


# ----------------------------------------------------------------------------- 4. traced step vs eager + perf
def test_traced_step_matches_eager_and_perf(depth_decoder, golden_frame):
    """``DepthStepTrace`` (one trace per depth step) reproduces the eager loop and is timed per 7-step frame."""
    g = golden_frame
    eager_hiddens, eager_logits = depth_decoder.teacher_forced_loop(
        g["global_hidden"], g["semantic_embed"], g["residual_codes"]
    )
    eager_logits = [DepthDecoder.rows_to_host(l) for l in eager_logits]

    trace = DepthStepTrace(depth_decoder)
    try:

        def run_frame():
            trace.begin_frame(g["global_hidden"], g["semantic_embed"])
            out = []
            for index in range(1, NUM_CODEBOOKS):
                prev = None if index == 1 else g["residual_codes"][:, index - 2]
                trace.step(index, prev)
                out.append(trace.logits_for(index))  # read-back per step, as the sampling loop must
            return out

        traced = run_frame()
        pccs = [_pcc(e, t) for e, t in zip(eager_logits, traced)]
        max_abs = max(float((e - t).abs().max()) for e, t in zip(eager_logits, traced))
        logger.info(f"traced vs eager logits: min PCC {min(pccs):.6f}, max |diff| {max_abs:.4f}")
        assert min(pccs) >= 0.999, pccs
        # Second frame must not see state from the first one.
        traced2 = run_frame()
        assert all(torch.equal(a, b) for a, b in zip(traced, traced2))

        # Perf: warmed frames (7 traced steps + per-step logits read-back), host wall time.
        for _ in range(3):
            run_frame()
        ttnn.synchronize_device(depth_decoder.mesh_device)
        n = 20
        t0 = time.perf_counter()
        for _ in range(n):
            run_frame()
        ttnn.synchronize_device(depth_decoder.mesh_device)
        traced_ms = (time.perf_counter() - t0) / n * 1e3
    finally:
        trace.release()

    # Eager counterpart.
    for _ in range(3):
        depth_decoder.teacher_forced_loop(g["global_hidden"], g["semantic_embed"], g["residual_codes"])
    ttnn.synchronize_device(depth_decoder.mesh_device)
    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        hiddens, logits = depth_decoder.teacher_forced_loop(
            g["global_hidden"], g["semantic_embed"], g["residual_codes"]
        )
        for l in logits:
            DepthDecoder.rows_to_host(l)
    eager_ms = (time.perf_counter() - t0) / n * 1e3
    logger.info(f"7-step depth loop per frame: eager {eager_ms:.1f} ms, traced {traced_ms:.1f} ms")
    _record(
        "perf_depth_loop_per_frame",
        eager_ms=eager_ms,
        traced_ms=traced_ms,
        traced_vs_eager_min_pcc=min(pccs),
        note="host wall time per 7-step teacher-forced frame incl. per-step logits read-back; warmed",
    )
