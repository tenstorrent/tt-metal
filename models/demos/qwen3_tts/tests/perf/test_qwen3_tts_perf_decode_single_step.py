# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy report for **one decode step**, block by block, in a single capture.

A decode step (one speech frame) is one Talker decode step plus 15 CodePredictor
passes — CP prefill at seq=2, then 14 CP decode steps — each sampling a code group.
Rather than profile the whole ~5,900-op frame, this captures **one of each distinct
block the step repeats** in a single Tracy run:

    start
      talker_decode_start  ── one Talker DecoderLayer, deployed decode step
      cp_prefill_start     ── one CodePredictor layer, CP prefill (seq=2)
      cp_decode_start      ── one CodePredictor layer, CP decode (seq=1)
      cp_sampling_start    ── one device sampling call (topk + gumbel + sampling)
    stop

One command:

    TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) python_env/bin/python3 -m pytest -s -q \\
      models/demos/qwen3_tts/tests/perf/test_qwen3_tts_perf_decode_single_step.py

Writes ``perf/reports/decode_single_step/``: one ``tt-perf-report.txt`` and
``ops_list.md`` over the whole ``start``/``stop`` window, plus
``ops_list_<block>.md`` / ``totals_<block>.json`` per block, all sliced from the
same CSV — so the ranked view and the per-block views describe one capture.

**Why sampling is in here.** It belongs to no layer, and one call costs more device
time than an entire CP layer; ``TopK`` alone is the single most expensive op in a
frame's non-layer work. A layers-only report is blind to it.

**These are the blocks, not the whole step.** A real step runs them 28 / 5 / 70 / 15
times respectively, and carries per-step work these windows exclude (the 15 LM heads,
15 CP final norms, the Talker codec_head, the accumulated codec embed, and host-side
D2H/H2D), so they cannot be scaled into a whole-step number.

Optional regression gate on total device kernel time for the whole window::

    QWEN3_TTS_PERF_BUDGET_US=1400 python_env/bin/python3 -m pytest -s -q <this file>
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.demos.qwen3_tts.tests.perf.qwen3_tts_perf_common import (
    build_code_predictor,
    build_talker_layer,
    capture_tracy_report,
    check_budget,
    close_perf_device,
    open_perf_device,
    profile_window,
    report_summary,
    run_cp_decode_layer_window,
    run_cp_prefill_layer_window,
    run_cp_sampling_window,
    run_talker_decode_layer_window,
    signpost,
)

_SCRIPT = Path(__file__).resolve()
_WINDOW = "decode_single_step"

# block -> (start signpost, stop signpost)
_BLOCKS = {
    "talker_decode": ("talker_decode_start", "talker_decode_stop"),
    "cp_prefill": ("cp_prefill_start", "cp_prefill_stop"),
    "cp_decode": ("cp_decode_start", "cp_decode_stop"),
    "cp_sampling": ("cp_sampling_start", "cp_sampling_stop"),
}


def main() -> None:
    """Profiled body: compile every block, then measure each inside one outer window."""
    device, mesh_shape = open_perf_device()
    try:
        layer = build_talker_layer(device)
        cp = build_code_predictor(device)

        def _run(block: str, warmup: bool) -> None:
            start, stop = _BLOCKS[block]
            # Throwaway names on the compile pass: it must not emit the real
            # signposts, or the window would open on the uncompiled run.
            names = (f"warm_{start}", f"warm_{stop}") if warmup else (start, stop)
            with profile_window(names[0], names[1], warmup=warmup):
                if block == "talker_decode":
                    run_talker_decode_layer_window(device, layer)
                elif block == "cp_prefill":
                    run_cp_prefill_layer_window(device, cp)
                elif block == "cp_decode":
                    run_cp_decode_layer_window(device, cp)
                else:
                    run_cp_sampling_window(device, *names, warmup=warmup)

        # Pass 1: compile everything OUTSIDE the outer window. _profile_forward's
        # compile run is not signposted, so leaving it inside `start`/`stop` would
        # put a second copy of every block in the outer report.
        for block in _BLOCKS:
            _run(block, warmup=True)

        # Pass 2: measured. One pass per block, all inside one outer window.
        signpost("start")
        for block in _BLOCKS:
            _run(block, warmup=False)
        signpost("stop")
    finally:
        close_perf_device(device, mesh_shape)


@pytest.mark.timeout(1800)
def test_decode_single_step_tracy_report():
    """Capture one Tracy report covering every block a decode frame repeats."""
    totals = capture_tracy_report(
        _WINDOW,
        _SCRIPT,
        # Talker decode ~30 + CP prefill ~30 + CP decode ~33 + sampling ~7 = ~100 on
        # N150. Well under that means a block's window never opened.
        min_ops=60,
        label=f"decode-frame blocks, one each (MESH_DEVICE={os.environ.get('MESH_DEVICE', 'default')})",
        sub_windows=_BLOCKS,
    )
    print(report_summary(_WINDOW, totals, blocks=len(_BLOCKS), mesh=os.environ.get("MESH_DEVICE", "default")))
    check_budget(_WINDOW, totals)


if __name__ == "__main__":
    main()
