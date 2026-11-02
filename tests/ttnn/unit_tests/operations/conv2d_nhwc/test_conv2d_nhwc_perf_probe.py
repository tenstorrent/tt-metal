# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe (not an acceptance test): BLOCK-sharded shapes for on-device A/B of R4 knobs.
Run under scripts/run_safe_pytest.sh --profile; read per-core kernel durations."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from test_conv2d_nhwc_sharded import ML, run_sharded  # noqa: E402


def test_perf_probe_block_28x28x256(device):
    run_sharded(device, (1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, True)


def test_perf_probe_block_sdxl_l1(device):
    run_sharded(
        device, (1, 64, 64, 640), 640, 3, 1, 1, ML.BLOCK_SHARDED, True, shard_shape=(410, 64), core_grid=(10, 10)
    )


def test_perf_probe_interleaved_sdxl_l1(device):
    run_sharded(device, (1, 64, 64, 640), 640, 3, 1, 1, ML.INTERLEAVED, True)
