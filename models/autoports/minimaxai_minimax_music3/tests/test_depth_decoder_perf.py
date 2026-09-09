# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tracy-signposted performance evidence for ``DepthDecoder``: one warmed 7-step depth loop (frame).

Pattern (functional-decoder skill): compile/warm, synchronize, signpost start, run the warmed
measured window, synchronize once, signpost end. Marked ``slow`` (excluded from the stage gate);
driven by ``scripts/collect_depth_perf.sh`` under ``python -m tracy``. Without Tracy the tests still
record wall times in ``doc/depth_decoder/perf/*.json``.
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
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.constants import AUDIO_CODE_OFFSET, NUM_CODEBOOKS
from models.autoports.minimaxai_minimax_music3.tt.depth_decoder import DepthDecoder, DepthStepTrace

pytestmark = [pytest.mark.hardware, pytest.mark.slow, pytest.mark.timeout(1800)]
DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "depth_decoder"


def _profiled() -> bool:
    return bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))


def _drain_profiler(mesh_device) -> None:
    if _profiled():
        ttnn.ReadDeviceProfiler(mesh_device)


def _signpost(name: str) -> None:
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(name)


def _write(name: str, payload: dict) -> None:
    out = DOC_DIR / "perf"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def depth_decoder(mm3_mesh_device):
    return DepthDecoder.from_pretrained(mm3_mesh_device, R.weights_dir())


@pytest.fixture(scope="module")
def frame_inputs():
    root = R.reference_dir()
    if not (root / "frame_hiddens.pt").is_file():
        pytest.skip("golden reference missing")
    frame_hiddens = torch.load(root / "frame_hiddens.pt")
    codes = torch.load(root / "sampled_codes.pt")
    embed_weight = R.load_embed_weight()
    return (
        frame_hiddens[0, 0, :4096].reshape(1, -1).repeat(2, 1).float(),
        embed_weight[int(codes[0, 0]) + AUDIO_CODE_OFFSET].reshape(1, -1).repeat(2, 1).float(),
        codes[0, 1:].reshape(1, -1).repeat(2, 1),
    )


def test_eager_frame_perf(depth_decoder, frame_inputs):
    """Eager 7-step teacher-forced loop; the measured window is ONE warmed frame."""
    g_hidden, s_embed, r_codes = frame_inputs
    dev = depth_decoder.mesh_device

    def frame():
        _, logits = depth_decoder.teacher_forced_loop(g_hidden, s_embed, r_codes)
        for l in logits:
            DepthDecoder.rows_to_host(l)

    for _ in range(3):
        frame()
    ttnn.synchronize_device(dev)
    _drain_profiler(dev)
    _signpost("PERF_DEPTH_EAGER")
    t0 = time.perf_counter()
    frame()
    ttnn.synchronize_device(dev)
    wall_ms = (time.perf_counter() - t0) * 1e3
    _signpost("PERF_DEPTH_EAGER_END")
    _drain_profiler(dev)
    logger.info(f"eager frame: {wall_ms:.1f} ms (profiled={_profiled()})")
    _write("eager_frame", {"wall_ms_one_frame": wall_ms, "profiled": _profiled(), "steps": NUM_CODEBOOKS - 1})


def test_traced_frame_perf(depth_decoder, frame_inputs):
    """``DepthStepTrace``: 7 trace replays with per-step logits read-back; the measured window is ONE warmed frame."""
    g_hidden, s_embed, r_codes = frame_inputs
    dev = depth_decoder.mesh_device
    trace = DepthStepTrace(depth_decoder)
    try:

        def frame():
            trace.begin_frame(g_hidden, s_embed)
            for index in range(1, NUM_CODEBOOKS):
                trace.step(index, None if index == 1 else r_codes[:, index - 2])
                trace.logits_for(index)

        for _ in range(3):
            frame()
        ttnn.synchronize_device(dev)
        _drain_profiler(dev)
        _signpost("PERF_DEPTH_TRACED")
        t0 = time.perf_counter()
        frame()
        ttnn.synchronize_device(dev)
        wall_ms = (time.perf_counter() - t0) * 1e3
        _signpost("PERF_DEPTH_TRACED_END")
        _drain_profiler(dev)
    finally:
        trace.release()
    logger.info(f"traced frame: {wall_ms:.1f} ms (profiled={_profiled()})")
    _write("traced_frame", {"wall_ms_one_frame": wall_ms, "profiled": _profiled(), "steps": NUM_CODEBOOKS - 1})
