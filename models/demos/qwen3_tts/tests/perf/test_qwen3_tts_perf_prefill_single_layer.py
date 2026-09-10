# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy per-op report for **one Talker decoder layer in prefill**.

One command, one report:

    export TT_VISIBLE_DEVICES=1
    export TT_METAL_CACHE=$HOME/.cache/tt_metal_n150_1
    export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto
    export MESH_DEVICE=N150

    TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) python_env/bin/python3 -m pytest -s -q \\
      models/demos/qwen3_tts/tests/perf/test_qwen3_tts_perf_prefill_single_layer.py

The test spawns its own Tracy capture of this file's ``main()`` and writes the
report to ``models/demos/qwen3_tts/tests/perf/reports/prefill_single_layer_<bucket>/``
(``ops_list.md``, ``tt-perf-report.txt``, ``ops.csv``, ``totals.json``,
``run.log``). Nothing else has to be run by hand.

The window is one untraced forward of one ``DecoderLayer`` at a demo prefill
TRACE bucket, between ``start`` / ``stop`` signposts — the graph
``qwen3_tts_perf_layers.py`` defines, called directly so the two cannot drift.

Bucket selection (the demo's buckets are 32 / 64 / 128; 64 is what the Japanese
sample pads to)::

    QWEN3_TTS_PERF_PREFILL_BUCKET=128 python_env/bin/python3 -m pytest -s -q <this file>

32 is a different QKV path (DRAM-sharded at seq<=32); 64 and 128 share that
family but differ in matmul M and RMSNorm/concat shard height.

Optional regression gate on device kernel time for the window::

    QWEN3_TTS_PERF_BUDGET_US=900 python_env/bin/python3 -m pytest -s -q <this file>
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.demos.qwen3_tts.tests.perf.qwen3_tts_perf_common import (
    build_talker_layer,
    capture_tracy_report,
    check_budget,
    close_perf_device,
    open_perf_device,
    prefill_buckets,
    report_summary,
    run_prefill_single_layer_window,
)

_SCRIPT = Path(__file__).resolve()
# The Japanese demo sample pads to 64; it is the bucket the deployed prefill runs.
_DEFAULT_BUCKET = 64


def bucket() -> int:
    return int(os.environ.get("QWEN3_TTS_PERF_PREFILL_BUCKET", _DEFAULT_BUCKET))


def main() -> None:
    """Profiled body: one prefill forward of one Talker layer between signposts."""
    seq_len = bucket()
    device, mesh_shape = open_perf_device()
    try:
        layer = build_talker_layer(device)
        run_prefill_single_layer_window(device, layer, seq_len)
    finally:
        close_perf_device(device, mesh_shape)


@pytest.mark.timeout(1200)
def test_prefill_single_layer_tracy_report():
    """Capture the Tracy report for one Talker decoder layer in prefill."""
    seq_len = bucket()
    assert seq_len in prefill_buckets(), f"bucket {seq_len} is not a demo TRACE bucket {prefill_buckets()}"
    window = f"prefill_single_layer_{seq_len}"

    totals = capture_tracy_report(
        window,
        _SCRIPT,
        # One Talker prefill layer is ~23 device ops per chip. A capture well under
        # that is truncated, not fast.
        min_ops=15,
        label=f"Talker DecoderLayer prefill, seq_len={seq_len} (MESH_DEVICE={os.environ.get('MESH_DEVICE', 'default')})",
    )
    print(report_summary(window, totals, seq_len=seq_len, mesh=os.environ.get("MESH_DEVICE", "default")))
    check_budget(window, totals)


if __name__ == "__main__":
    main()
