# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1048576}], indirect=True)
@pytest.mark.parametrize("tensor_controls", [False, True], ids=["scalar_controls", "tensor_controls"])
@pytest.mark.parametrize("width", [96, 800], ids=["small", "uneven_work_split"])
def test_slice_tile_cache_argument_refresh(device, tensor_controls, width):
    torch.manual_seed(52651)
    # Retain every allocation. The second pass must reuse programs with different
    # input/output addresses and (for tensor controls) different control buffers.
    inputs, controls, goldens, outputs = [], [], [], []
    for iteration in range(6):
        value = torch.randn([1, 3, 128, width], dtype=torch.bfloat16)
        inputs.append(ttnn.from_torch(value, device=device, layout=ttnn.TILE_LAYOUT))
        row = (iteration % 2) * 64
        start, end = [0, 0, row, 0], [1, 3, row + 64, width]
        goldens.append(value[:, :, row : row + 64, :])
        if tensor_controls:
            controls.append(
                (
                    ttnn.from_torch(torch.tensor(start), device=device),
                    ttnn.from_torch(torch.tensor(end), device=device),
                )
            )
        else:
            controls.append((start, end))

    def run(index):
        kwargs = {"slice_dim": 2, "num_devices": 2} if tensor_controls else {}
        return ttnn.slice(inputs[index], *controls[index], **kwargs)

    def check(results):
        for actual, expected in zip(results, goldens):
            assert torch.equal(ttnn.to_torch(actual), expected)

    for index in range(len(inputs)):
        outputs.append(run(index))
        if index == 1:
            warmed_entries = device.num_program_cache_entries()
        if index > 1:
            assert device.num_program_cache_entries() == warmed_entries
    check(outputs)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_outputs = [run(index) for index in range(len(inputs))]
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        check(traced_outputs)
        assert device.num_program_cache_entries() == warmed_entries
    finally:
        ttnn.release_trace(device, trace_id)
