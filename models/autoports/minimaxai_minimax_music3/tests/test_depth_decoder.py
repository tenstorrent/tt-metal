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
import os
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
    # Watcher / profiler runs inflate timings; they must not overwrite the committed evidence.
    if os.environ.get("TT_METAL_WATCHER") or os.environ.get("TT_METAL_DEVICE_PROFILER"):
        return
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


# ----------------------------------------------------------------------------- 2b. distinct batch rows
@pytest.fixture(scope="session")
def distinct_frames():
    """Row 0 = golden frame 1, row 1 = golden frame 2 (different hidden, semantic code and residual codes)."""
    root = R.reference_dir()
    if not (root / "frame_hiddens.pt").is_file():
        pytest.skip(f"golden reference missing under {root}")
    frame_hiddens = torch.load(root / "frame_hiddens.pt")
    codes = torch.load(root / "sampled_codes.pt")
    embed_weight = R.load_embed_weight()
    frames = [0, 1]
    return {
        "global_hidden": torch.stack([frame_hiddens[0, f, :4096] for f in frames]).float(),
        "semantic_embed": torch.stack([embed_weight[int(codes[f, 0]) + AUDIO_CODE_OFFSET] for f in frames]).float(),
        "residual_codes": torch.stack([codes[f, 1:] for f in frames]),  # [2, 7]
        "depth_hiddens": torch.stack(
            [frame_hiddens[0, f, 4096:].reshape(NUM_CODEBOOKS - 1, 4096) for f in frames]
        ).float(),  # [2, 7, 4096]
    }


def test_distinct_batch_rows(depth_decoder, depth_ref, distinct_frames):
    """Row 0 = golden frame 1, row 1 = golden frame 2: each row must reproduce ITS OWN golden depth hiddens.

    Every other test feeds identical rows, which would hide a row copy / swap in the tile-aligned
    row splits, the head split, or the one-hot selectors. Also checks per-row PCC vs the torch
    reference for hidden and all heads. The traced path is checked with the same rows at the end of
    ``test_traced_step_matches_eager_and_perf``.
    """
    d = distinct_frames
    g_hidden, s_embed, r_codes, golden = (
        d["global_hidden"],
        d["semantic_embed"],
        d["residual_codes"],
        d["depth_hiddens"],
    )

    ref_hiddens, ref_logits = REF.teacher_forced_depth_loop(depth_ref, g_hidden, s_embed, r_codes)
    hiddens, logits = depth_decoder.teacher_forced_loop(g_hidden, s_embed, r_codes)
    tt_h = torch.stack([DepthDecoder.rows_to_host(h) for h in hiddens], dim=1)  # [2, 7, 4096]
    tt_l = torch.stack([DepthDecoder.rows_to_host(l) for l in logits], dim=1)  # [2, 7, 1024]
    ref_h = torch.stack(ref_hiddens, dim=1)
    ref_l = torch.stack(ref_logits, dim=1)
    per_row_golden = [_pcc(golden[b].flatten(), tt_h[b].flatten()) for b in range(2)]
    per_row_ref_hidden = [_pcc(ref_h[b].flatten(), tt_h[b].flatten()) for b in range(2)]
    per_row_ref_logits = [min(_pcc(ref_l[b, i], tt_l[b, i]) for i in range(NUM_CODEBOOKS - 1)) for b in range(2)]
    # The two rows must actually differ (otherwise the test proves nothing) and must not be swapped.
    swapped = _pcc(golden[1].flatten(), tt_h[0].flatten())
    logger.info(
        f"distinct rows: per-row PCC vs own golden {per_row_golden}, vs ref hidden {per_row_ref_hidden}, "
        f"min head logits {per_row_ref_logits}, row0-vs-row1-golden {swapped:.4f}"
    )
    _record(
        "distinct_batch_rows_frames_1_2",
        per_row_pcc_vs_own_golden=per_row_golden,
        per_row_hidden_pcc_vs_ref=per_row_ref_hidden,
        per_row_min_head_logits_pcc_vs_ref=per_row_ref_logits,
        row0_vs_row1_golden_pcc=swapped,
    )
    assert swapped < 0.9, swapped
    assert min(per_row_golden) >= PCC_GOLDEN, per_row_golden
    assert min(per_row_ref_hidden) >= PCC_LAYER, per_row_ref_hidden
    assert min(per_row_ref_logits) >= PCC_LAYER, per_row_ref_logits


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
def test_traced_step_matches_eager_and_perf(depth_decoder, golden_frame, distinct_frames, capfd):
    """``DepthStepTrace`` (one trace per depth step) reproduces the eager loop and is timed per 7-step frame.

    Also asserts the frame is allocation-free after capture: tt-metal's "Allocating device buffers
    is unsafe due to the existence of an active trace" warning must not appear. tt-metal prints it
    once per thread per process (allocator.cpp, ``thread_local static bool warning_generated``), so
    the assertion is only meaningful if no earlier trace in this process already triggered it: keep
    this test the only trace-creating test in the module (the distinct-row traced check lives at
    the end of this test, after the assertion).
    """
    g = golden_frame
    eager_hiddens, eager_logits = depth_decoder.teacher_forced_loop(
        g["global_hidden"], g["semantic_embed"], g["residual_codes"]
    )
    eager_logits = [DepthDecoder.rows_to_host(l) for l in eager_logits]

    # Persistent device inputs exist BEFORE the traces are built (stage-04 pattern: the backbone's
    # hidden buffer outlives the depth trace); nothing may be allocated on device after this line
    # until the traces are released.
    dev_hidden = depth_decoder.rows_to_device(g["global_hidden"])
    dev_semantic = depth_decoder.rows_to_device(g["semantic_embed"])
    capfd.readouterr()  # drop everything logged so far
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
        # Seeding from persistent device row tensors (stage 04: the backbone hidden) is bit-identical.
        trace.begin_frame(dev_hidden, dev_semantic)
        traced3 = []
        for index in range(1, NUM_CODEBOOKS):
            trace.step(index, None if index == 1 else g["residual_codes"][:, index - 2])
            traced3.append(trace.logits_for(index))
        assert all(torch.equal(a, b) for a, b in zip(traced, traced3))

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
    captured = capfd.readouterr()
    unsafe = [l for l in (captured.err + captured.out).splitlines() if "Allocating device buffers is unsafe" in l]
    assert not unsafe, unsafe
    ttnn.deallocate(dev_hidden)
    ttnn.deallocate(dev_semantic)
    _record("traced_frame_allocation_free", no_unsafe_allocation_warning=True)

    # Distinct CFG rows through the traced path: the trace has its own row handling (seed padding /
    # copies, the code-id row), so run one host-seeded and one device-seeded traced frame with the
    # two different golden frames and require bit-identity per row with the eager loop, which
    # test_distinct_batch_rows validated per row against the golden and the torch reference.
    d = distinct_frames
    eager_h, eager_l = depth_decoder.teacher_forced_loop(d["global_hidden"], d["semantic_embed"], d["residual_codes"])
    eager_h = torch.stack([DepthDecoder.rows_to_host(h) for h in eager_h], dim=1)  # [2, 7, 4096]
    eager_l = torch.stack([DepthDecoder.rows_to_host(l) for l in eager_l], dim=1)  # [2, 7, 1024]
    dev_hidden = depth_decoder.rows_to_device(d["global_hidden"])
    dev_semantic = depth_decoder.rows_to_device(d["semantic_embed"])
    trace = DepthStepTrace(depth_decoder)
    try:
        for seeded_from, seeds in (
            ("host", (d["global_hidden"], d["semantic_embed"])),
            ("device", (dev_hidden, dev_semantic)),
        ):
            trace.begin_frame(*seeds)
            for index in range(1, NUM_CODEBOOKS):
                trace.step(index, None if index == 1 else d["residual_codes"][:, index - 2])
                assert torch.equal(trace.logits_for(index), eager_l[:, index - 1]), (seeded_from, index)
                assert torch.equal(trace.hidden_for(), eager_h[:, index - 1]), (seeded_from, index)
    finally:
        trace.release()
    ttnn.deallocate(dev_hidden)
    ttnn.deallocate(dev_semantic)
    _record("distinct_batch_rows_traced", bit_identical_to_eager_host_and_device_seeded=True)

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
