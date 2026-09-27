# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Control prefill AG worker geometry through the unchanged stack harness."""

import os

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import test_multichip_stack

original = ttnn.experimental.all_gather_async


def controlled(value, *args, **kwargs):
    if value.shape[-2] > 1 and "PREFILL_AG_WORKERS" in os.environ:
        kwargs["num_workers_per_link"] = int(os.environ["PREFILL_AG_WORKERS"])
    return original(value, *args, **kwargs)


ttnn.experimental.all_gather_async = controlled
if os.environ.get("PROBE_BASELINE_PRECISION") or os.environ.get("PROBE_MOE_BF16"):
    factory = test_multichip_stack.MultichipDecoder.from_state_dict

    def baseline(cls, *args, **kwargs):
        if os.environ.get("PROBE_BASELINE_PRECISION"):
            kwargs["attention_precision"] = "baseline"
        if os.environ.get("PROBE_MOE_BF16"):
            kwargs["moe_ccl_bfp8"] = False
        return factory(*args, **kwargs)

    test_multichip_stack.MultichipDecoder.from_state_dict = classmethod(baseline)

if os.environ.get("PROBE_FULL_SEMAPHORE_GRID"):
    from models.demos.gpt_oss.tt.ccl import CCLManager

    def full_grid(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

    CCLManager._init_subdevice = full_grid
test_multichip_stack.main()
