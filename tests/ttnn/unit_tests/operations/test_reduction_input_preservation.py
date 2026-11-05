# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("shape", [(2, 7, 128), (2, 33, 40), (2, 32, 128)])
@pytest.mark.parametrize("operation", ["topk", "max", "min", "sum", "mean"])
def test_reduction_preserves_input_bits(device, shape, operation):
    torch.manual_seed(0)
    values = torch.randn(shape, dtype=torch.bfloat16)
    # Negative zero, a subnormal, and a NaN payload distinguish a raw copy from
    # a floating-point unpack/pack round trip through fill_pad.
    values.view(torch.int16).flatten()[:3] = torch.tensor([-32768, 1, 0x7FC1], dtype=torch.int16)
    tensor = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    before = ttnn.to_torch(tensor).view(torch.int16).clone()
    if operation == "topk":
        result, _ = ttnn.topk(tensor, k=8, dim=-1, largest=False)
    else:
        result = getattr(ttnn, operation)(tensor, dim=-1)
    ttnn.to_torch(result)
    after = ttnn.to_torch(tensor).view(torch.int16)
    assert torch.equal(after, before)


@pytest.mark.parametrize("shape", [(2, 7, 128), (2, 33, 40)])
@pytest.mark.parametrize("operation", ["topk", "max", "min", "sum", "mean"])
def test_reduction_with_private_padding_matches_reference(device, shape, operation):
    torch.manual_seed(0)
    values = torch.rand(shape, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
    if operation == "topk":
        result, _ = ttnn.topk(tensor, k=8, dim=-1, largest=False)
        expected = torch.topk(values, k=8, dim=-1, largest=False).values
    else:
        result = getattr(ttnn, operation)(tensor, dim=-1, keepdim=True)
        expected = getattr(torch, operation)(values.float(), dim=-1, keepdim=True)
        if operation in ("max", "min"):
            expected = expected.values
    torch.testing.assert_close(ttnn.to_torch(result).float(), expected.float(), rtol=0.02, atol=0.01)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1000000}], indirect=True)
@pytest.mark.parametrize("shape, dim", [((1, 32, 8, 32), -2), ((2, 33, 40), -1)])
@pytest.mark.parametrize("operation", ["max", "min"])
def test_padded_reduction_trace_stays_on_worker_subdevice(device, shape, dim, operation):
    # Use a disjoint, non-origin grid: padding preparation must honor the same
    # worker cores as the producer and reduction, including during trace replay.
    worker_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(2, 1), ttnn.CoreCoord(3, 2)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 3), ttnn.CoreCoord(5, 3)),
        }
    )
    other_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    manager = device.create_sub_device_manager([ttnn.SubDevice([other_cores]), ttnn.SubDevice([worker_cores])], 65536)
    device.load_sub_device_manager(manager)
    device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])
    trace_id = None
    try:
        values = torch.randn(shape, dtype=torch.bfloat16)
        tensor = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)

        def reduce():
            produced = ttnn.neg(tensor, sub_core_grids=worker_cores)
            result = getattr(ttnn, operation)(produced, dim=dim, keepdim=True, sub_core_grids=worker_cores)
            ttnn.deallocate(produced)
            return result

        warmup = reduce()
        ttnn.synchronize_device(device)
        ttnn.deallocate(warmup)
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        result = reduce()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        for offset in (0, 3, -2):
            replay_values = values + offset
            host_input = ttnn.from_torch(replay_values, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_input, tensor)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            expected = getattr(torch, operation)(-replay_values, dim=dim, keepdim=True).values
            torch.testing.assert_close(ttnn.to_torch(result), expected, rtol=0, atol=0)
            torch.testing.assert_close(ttnn.to_torch(tensor), replay_values, rtol=0, atol=0)
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        device.reset_sub_device_stall_group()
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)
