# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial real-weight CI smoke; dedicated runners add context, stress and watcher."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("layer", [0, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_multichip_decoder(layer, batch, tmp_path):
    reference = tmp_path / "reference.pt"
    environment = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    environment.pop("TT_VISIBLE_DEVICES", None)
    for baseline in [True, False]:
        output = tmp_path / ("baseline.json" if baseline else "multichip.json")
        command = [
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder",
            "--trace",
            "--repeats",
            "5",
            "--layer",
            str(layer),
            "--batch",
            str(batch),
            "--length",
            "33",
            "--reference",
            str(reference),
            "--output",
            str(output),
        ]
        if baseline:
            command.append("--baseline")
        subprocess.run(command, check=True, timeout=300, env=environment)
        result = json.loads(output.read_text())
        assert result["finite"] and result["trace"]["bit_equal_eager"]
        assert result["trace"]["positive_control_rejected"]
        if not baseline:
            assert result["pass"]
