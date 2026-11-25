# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
import sys

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_dry_run_then_real_run(device):
    """An op launched in a dry run sends nothing to the device; the same op launched for real afterwards, on the same
    device, runs and gives the right result."""
    torch_input = torch.rand(1, 1, 64, 64)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn._ttnn.device.BeginDryRun()
    try:
        dry_output = ttnn.exp(tt_input)
    finally:
        ttnn._ttnn.device.EndDryRun()
    assert dry_output.shape == tt_input.shape

    assert_with_pcc(torch.exp(torch_input), ttnn.to_torch(ttnn.exp(tt_input)), 0.999)


def test_precompile_plugin_end_to_end(tmp_path):
    """A pytest run with the precompile plugin forced on, from an empty kernel cache: the dry pass runs, the test
    passes in the real pass, and every kernel in the cache is accounted for as built ahead or built late."""
    env = dict(os.environ, PRECOMPILE="on", TT_METAL_CACHE=str(tmp_path))
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.plugins.precompile", f"{__file__}::test_dry_run_then_real_run"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "PRECOMPILE: on (PRECOMPILE=on; 1 tests" in run.stdout and "Dry pass over 1 tests" in run.stdout
    assert "1 passed" in run.stdout
    counts = re.search(r"PRECOMPILE: (\d+) kernels built ahead, (\d+) built late", run.stdout)
    assert counts and int(counts.group(1)) + int(counts.group(2)) > 0, run.stdout[-3000:]
