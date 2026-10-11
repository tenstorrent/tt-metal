# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real checkpoint and recorded-HF activation coverage of OptimizedDecoder.

Run prepare_optimized_inputs once with HF_REFERENCE_PYTHON before these tests.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["sliding_rope", "full_nope"])
def test_optimized_decoder(layer_idx, tmp_path):
    python = os.environ.get("HF_REFERENCE_PYTHON", sys.executable)
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    subprocess.run(
        [
            python,
            str(ROOT / "tests/run_optimized.py"),
            "--real",
            "--layer",
            str(layer_idx),
            "--lengths",
            "1",
            "31",
            "32",
            "33",
            "63",
            "64",
            "65",
            "1023",
            "1024",
            "1025",
            "2047",
            "2048",
            "2049",
            "3073",
            "4095",
            "4096",
            "4097",
            "4159",
            "17",
            "--output",
            str(tmp_path / f"{layer_idx}.json"),
        ],
        env=env,
        check=True,
    )


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["sliding_rope", "full_nope"])
def test_optimized_partial_core_row(layer_idx, tmp_path):
    python = os.environ.get("HF_REFERENCE_PYTHON", sys.executable)
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    subprocess.run(
        [
            python,
            str(ROOT / "tests/run_optimized.py"),
            "--real",
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
