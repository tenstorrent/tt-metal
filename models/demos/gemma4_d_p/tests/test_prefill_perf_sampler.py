# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.demo.prefill_perf import _device_samples


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_device_samples_with_traces_and_eager_programs(device):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("Requires the real-time device profiler")
    data = torch.linspace(-1, 1, 1024, dtype=torch.bfloat16).reshape(1, 1, 32, 32)
    source = ttnn.from_torch(data, layout=ttnn.TILE_LAYOUT, device=device)
    temporary, output = ttnn.empty_like(source), ttnn.empty_like(source)
    ttnn.neg(source, output_tensor=temporary)
    ttnn.relu(temporary, output_tensor=temporary)
    ttnn.neg(temporary, output_tensor=output)
    before = ttnn.begin_trace_capture(device, cq_id=0)
    ttnn.neg(source, output_tensor=temporary)
    ttnn.end_trace_capture(device, before, cq_id=0)
    after = ttnn.begin_trace_capture(device, cq_id=0)
    ttnn.neg(temporary, output_tensor=output)
    ttnn.end_trace_capture(device, after, cq_id=0)

    def run():
        ttnn.execute_trace(device, before, cq_id=0, blocking=False)
        ttnn.relu(temporary, output_tensor=temporary)
        ttnn.execute_trace(device, after, cq_id=0, blocking=False)

    try:
        samples = _device_samples(device, run=run, repeats=3)
        assert len(samples) == 3
        assert all(chip["programs"] == 3 for sample in samples for chip in sample["chips"].values())
        assert torch.equal(ttnn.to_torch(output), -(-data).relu())
    finally:
        ttnn.release_trace(device, before)
        ttnn.release_trace(device, after)
