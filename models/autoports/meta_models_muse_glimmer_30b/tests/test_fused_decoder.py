# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run real-shape synthetic coverage with the official HF reference environment.

HF_REFERENCE_PYTHON may point to a compatible transformers>=5.15 environment
with this source TTNN installed. Normal tests require config and recorded weight
statistics, not checkpoint shards. Use run_fused.py --real for real evidence.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["sliding_rope", "full_nope"])
def test_fused_decoder(layer_idx, tmp_path):
    python = os.environ.get("HF_REFERENCE_PYTHON", sys.executable)
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    subprocess.run(
        [
            python,
            str(ROOT / "tests/run_fused.py"),
            "--layer",
            str(layer_idx),
            "--lengths",
            "31",
            "32",
            "33",
            "1023",
            "1024",
            "1025",
            "2047",
            "2048",
            "2049",
            "3073",
            "17",
            "--output",
            str(tmp_path / f"{layer_idx}.json"),
        ],
        env=env,
        check=True,
    )


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["sliding_rope", "full_nope"])
def test_fused_partial_core_row(layer_idx, tmp_path):
    python = os.environ.get("HF_REFERENCE_PYTHON", sys.executable)
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    subprocess.run(
        [
            python,
            str(ROOT / "tests/run_fused.py"),
            "--layer",
            str(layer_idx),
            "--batch",
            "9",
            "--lengths",
            "33",
            "--output",
            str(tmp_path / f"partial_{layer_idx}.json"),
        ],
        env=env,
        check=True,
    )
