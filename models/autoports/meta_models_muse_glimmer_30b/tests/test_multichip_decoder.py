# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run serially on QB2 after activating the lane environment. No xdist."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("layer", [0, 3], ids=["sliding_rope", "full_nope"])
@pytest.mark.parametrize("batch", [1, 9, 17, 32])
def test_multichip_cache_trace_and_partial_rows(layer, batch, tmp_path):
    python = os.environ.get("HF_REFERENCE_PYTHON", sys.executable)
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    for mode in ("single", "replicated"):
        subprocess.run(
            [
                python,
                str(ROOT / "tests/run_multichip.py"),
                "--mode",
                mode,
                "--layer",
                str(layer),
                "--batch",
                str(batch),
                "--length",
                "33",
                "--pages",
                str(batch * 4),
                "--validate",
                "--stress",
                "20",
                "--output",
                str(tmp_path / f"{mode}_{layer}_{batch}.json"),
            ],
            env=env,
            check=True,
        )


def test_multichip_stacked_layer_kinds(tmp_path):
    python = os.environ.get("HF_REFERENCE_PYTHON", sys.executable)
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    for mode in ("single", "replicated"):
        subprocess.run(
            [
                python,
                str(ROOT / "tests/run_multichip.py"),
                "--mode",
                mode,
                "--length",
                "65",
                "--batch",
                "2",
                "--stack",
                "--validate",
                "--stress",
                "20",
                "--output",
                str(tmp_path / f"{mode}_stack.json"),
            ],
            env=env,
            check=True,
        )
