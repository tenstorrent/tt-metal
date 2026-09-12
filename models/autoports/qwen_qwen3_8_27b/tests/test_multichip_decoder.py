# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 real-weight regressions; each process owns and closes its mesh.

Run serially, without pytest-xdist. Failed hardware jobs require recovery before
continuing. Fixtures are recorded HF layer inputs used by the optimized stage.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = os.getenv("QWEN_MULTICHIP_STAGE", "multichip_decoder")
DOC = ROOT / "doc" / STAGE
WRAPPER = (
    "run_optimized_multichip_experiment.sh" if STAGE == "optimized_multichip_decoder" else "run_multichip_experiment.sh"
)


def baseline_modes(name, layer, batch, length, stack=False, continuation=False):
    """Optionally reuse a recorded fixture from the unchanged optimized source."""
    report = ROOT / "doc/multichip_decoder" / (name + "_baseline.json")
    fixture = Path("/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/multichip_compare") / (
        f"l{layer}_b{batch}_s{length}_stack{int(stack)}_c{int(continuation)}.pt"
    )
    if os.getenv("QWEN_REUSE_OPTIMIZED_BASELINE") == "1" and report.exists() and fixture.exists():
        baseline = json.loads(report.read_text())
        source_hash = hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest()
        if baseline["source_sha256"] == source_hash:
            return (False,)
    return (True, False)


@pytest.mark.parametrize("layer", [0, 3])
@pytest.mark.parametrize(
    "batch,length,continuation",
    [
        (1, 1, False),
        (1, 31, False),
        (1, 32, False),
        (1, 33, False),
        (1, 2047, False),
        (1, 2048, False),
        (1, 2049, False),
        (1, 4097, False),
        (1, 33, True),
        (1, 129, True),
        (3, 33, True),
        (8, 33, False),
        (32, 257, False),
    ],
)
def test_multichip_default(layer, batch, length, continuation):
    name = f"final_l{layer}_b{batch}_s{length}_c{int(continuation)}"
    options = ["--layer", str(layer), "--batch", str(batch), "--length", str(length), "--repeats", "5"]
    if continuation:
        options += ["--continuation"]
    for baseline in baseline_modes(name, layer, batch, length, continuation=continuation):
        run_name = name + ("_baseline" if baseline else "")
        subprocess.run(
            [
                "bash",
                str(ROOT / "tests" / WRAPPER),
                run_name,
                *options,
                *(["--baseline"] if baseline else []),
            ],
            check=True,
        )
        report = json.loads((DOC / f"{run_name}.json").read_text())
        assert report["trace_bitwise_equal"] and report["changed_input_replay_bitwise_equal"]
        if not baseline:
            assert min(report["pcc"].values()) >= 0.995
            assert min(v for values in report["per_user_pcc"].values() for v in values) >= 0.995


@pytest.mark.parametrize("batch", [1, 2, 3, 8, 16])
def test_stacked_layout(batch):
    name = f"final_stack_b{batch}"
    for baseline in baseline_modes(name, 0, batch, 33, stack=True):
        run_name = name + ("_baseline" if baseline else "")
        subprocess.run(
            [
                "bash",
                str(ROOT / "tests" / WRAPPER),
                run_name,
                "--layer",
                "0",
                "--batch",
                str(batch),
                "--length",
                "33",
                "--stack",
                "--repeats",
                "10",
                *(["--baseline"] if baseline else []),
            ],
            check=True,
        )
        report = json.loads((DOC / f"{run_name}.json").read_text())
        assert report["post_decode_state_bitwise_equal"]
        if not baseline:
            assert min(report["pcc"].values()) >= 0.995
