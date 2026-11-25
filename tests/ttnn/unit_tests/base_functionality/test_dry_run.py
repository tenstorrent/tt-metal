# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _dry(function, *args):
    ttnn._ttnn.device.BeginDryRun()
    try:
        return function(*args)
    finally:
        ttnn._ttnn.device.EndDryRun()


def test_dry_run_then_real_run(device):
    """An op launched in a dry run sends nothing to the device; the same op launched for real afterwards, on the same
    device, runs and gives the right result."""
    torch_input = torch.rand(1, 1, 64, 64)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    dry_output = _dry(ttnn.exp, tt_input)
    assert dry_output.shape == tt_input.shape

    assert_with_pcc(torch.exp(torch_input), ttnn.to_torch(ttnn.exp(tt_input)), 0.999)


def test_dry_run_queues_kernel_builds(device):
    """An op launched in a dry run has its kernel builds queued; they are built without the op being launched for
    real."""
    tt_input = ttnn.from_torch(torch.rand(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Forget which kernels this process has built already, so the queued builds are not skipped as known.
    ttnn.device.ClearKernelCache()
    built_before = ttnn._ttnn.device.FinishQueuedBuilds(drop_queued=False)

    _dry(ttnn.log, tt_input)

    assert ttnn._ttnn.device.FinishQueuedBuilds(drop_queued=False) > built_before
