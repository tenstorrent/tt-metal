# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A (1, 1) submesh on the last device of a two-device parent, with and without share_allocator."""

import json
import os
import statistics
import time

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch

# Four bf16 tiles: at most one page per DRAM bank, so each buffer spans [address, address + one page) in every bank.
SHAPE = (1, 1, 32, 128)
PAGE_BYTES = 32 * 32 * 2

pytestmark = [
    skip_for_slow_dispatch(),
    pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2")], indirect=True),
]


def to_device(torch_tensor, mesh):
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def per_device(tensor, mesh):
    """The tensor's shard on each device of the mesh, in mesh order."""
    stacked = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return list(stacked.split(SHAPE[0], dim=0))


def assert_disjoint(tensors):
    ranges = sorted((t.buffer_address(), t.buffer_address() + PAGE_BYTES) for t in tensors)
    for (start_a, end_a), (start_b, _) in zip(ranges, ranges[1:]):
        assert end_a <= start_b, f"buffers at {start_a:#x} and {start_b:#x} overlap"


def last_device_submesh(mesh_device, share_allocator):
    rows, cols = tuple(mesh_device.shape)
    return mesh_device.create_submesh(
        ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(rows - 1, cols - 1), share_allocator=share_allocator
    )


@pytest.mark.parametrize("share_allocator", [False, True], ids=["own_allocator", "shared_allocator"])
def test_submesh_write_and_parent_tensor(mesh_device, share_allocator):
    parent_torch = torch.full(SHAPE, 1.5, dtype=torch.bfloat16)
    parent_tensor = to_device(parent_torch, mesh_device)
    mesh_device.quiesce_devices()

    submesh = last_device_submesh(mesh_device, share_allocator)
    pattern = torch.full(SHAPE, -7.0, dtype=torch.bfloat16)
    # Enough submesh tensors to reach the parent tensor's address if the allocators are independent.
    submesh_tensors = [to_device(pattern, submesh) for _ in range(8)]
    hit = [t for t in submesh_tensors if abs(t.buffer_address() - parent_tensor.buffer_address()) < PAGE_BYTES]
    assert torch.equal(per_device(submesh_tensors[-1], submesh)[0], pattern)
    mesh_device.quiesce_devices()

    first, second = per_device(parent_tensor, mesh_device)
    assert torch.equal(first, parent_torch)
    if share_allocator:
        assert not hit
        assert_disjoint([parent_tensor] + submesh_tensors)
        assert torch.equal(second, parent_torch), "a submesh write reached the parent tensor"
    else:
        # The control: an independent allocator hands out the parent's address again, so the write lands in it.
        assert hit, "no submesh tensor landed on the parent tensor's address"
        assert torch.equal(second, pattern)
    mesh_device.quiesce_devices()


def test_alternating_ops_keep_both_meshes_correct(mesh_device):
    submesh = last_device_submesh(mesh_device, share_allocator=True)
    torch.manual_seed(0)
    a, b, c, d = (torch.rand(SHAPE, dtype=torch.bfloat16) for _ in range(4))
    parent_a, parent_b = to_device(a, mesh_device), to_device(b, mesh_device)
    mesh_device.quiesce_devices()
    sub_c, sub_d = to_device(c, submesh), to_device(d, submesh)
    mesh_device.quiesce_devices()

    live = [parent_a, parent_b, sub_c, sub_d]
    for round_index in range(4):
        # Parent, without reading back, so its program binaries stay in the dispatcher's cache.
        parent_sum = ttnn.add(parent_a, parent_b)
        mesh_device.quiesce_devices()

        # Submesh, with a different program on the shared device, and a readback.
        sub_product = ttnn.multiply(sub_c, sub_d)
        sub_matmul = ttnn.matmul(sub_c, ttnn.transpose(sub_d, -2, -1))
        torch.testing.assert_close(
            per_device(sub_product, submesh)[0], (c * d), rtol=1e-2, atol=1e-2, msg=f"round {round_index}"
        )
        torch.testing.assert_close(
            per_device(sub_matmul, submesh)[0].float(),
            c.float() @ d.float().transpose(-2, -1),
            rtol=5e-2,
            atol=5e-1,
        )
        mesh_device.quiesce_devices()

        # Parent again: the same program as before, then a read of both results.
        parent_sum_again = ttnn.add(parent_a, parent_b)
        expected = a + b
        for result in (parent_sum, parent_sum_again):
            for device_index, shard in enumerate(per_device(result, mesh_device)):
                torch.testing.assert_close(
                    shard, expected, rtol=1e-2, atol=1e-2, msg=f"round {round_index}, device {device_index}"
                )
        assert_disjoint(live + [parent_sum, parent_sum_again, sub_product])
        mesh_device.quiesce_devices()

    for t in live:
        ttnn.deallocate(t)


def test_switch_without_quiesce_raises(mesh_device, expect_error):
    submesh = last_device_submesh(mesh_device, share_allocator=True)
    a = torch.rand(SHAPE, dtype=torch.bfloat16)
    parent_a = to_device(a, mesh_device)
    ttnn.add(parent_a, parent_a)

    with expect_error(RuntimeError, "still has a command queue in use"):
        to_device(a, submesh)
    mesh_device.quiesce_devices()

    sub_a = to_device(a, submesh)
    with expect_error(RuntimeError, "still has a command queue in use"):
        ttnn.add(parent_a, parent_a)
    torch.testing.assert_close(per_device(ttnn.add(sub_a, sub_a), submesh)[0], a + a, rtol=1e-2, atol=1e-2)
    mesh_device.quiesce_devices()

    for shard in per_device(ttnn.add(parent_a, parent_a), mesh_device):
        torch.testing.assert_close(shard, a + a, rtol=1e-2, atol=1e-2)
    mesh_device.quiesce_devices()


def test_shared_submesh_rejects_traces(mesh_device, expect_error):
    submesh = last_device_submesh(mesh_device, share_allocator=True)
    with expect_error(RuntimeError, "Traces are not supported"):
        ttnn.begin_trace_capture(submesh, cq_id=0)


def _summary_us(samples):
    samples = sorted(samples)
    return {
        "n": len(samples),
        "median_us": round(statistics.median(samples) * 1e6, 1),
        "p90_us": round(samples[int(0.9 * (len(samples) - 1))] * 1e6, 1),
        "min_us": round(samples[0] * 1e6, 1),
    }


def test_switch_cost(mesh_device):
    """Time what a switch between the parent and an overlapping submesh costs over a plain finish."""
    iterations = 200
    submesh = last_device_submesh(mesh_device, share_allocator=True)
    small = torch.rand(SHAPE, dtype=torch.bfloat16)
    square = torch.rand((1, 1, 256, 256), dtype=torch.bfloat16)
    parent_small, parent_square = to_device(small, mesh_device), to_device(square, mesh_device)
    mesh_device.quiesce_devices()
    sub_small, sub_square = to_device(small, submesh), to_device(square, submesh)
    mesh_device.quiesce_devices()

    def phase(tensor):
        # A model-like step: two programs, eight times each.
        for _ in range(8):
            ttnn.add(ttnn.matmul(tensor, tensor), tensor)

    # Warm the program caches of both meshes.
    for mesh_tensors in ((parent_small, parent_square), (sub_small, sub_square)):
        ttnn.add(mesh_tensors[0], mesh_tensors[0])
        phase(mesh_tensors[1])
        mesh_device.quiesce_devices()

    timings = {name: [] for name in ("finish", "quiesce_idle", "quiesce_pending", "phase_steady", "phase_after_switch")}
    for _ in range(iterations):
        # The finish any phase boundary needs anyway, for comparison.
        ttnn.add(parent_small, parent_small)
        start = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        timings["finish"].append(time.perf_counter() - start)
        # Quiesce once the work has finished: the drain and reset alone.
        start = time.perf_counter()
        mesh_device.quiesce_devices()
        timings["quiesce_idle"].append(time.perf_counter() - start)
        # Quiesce right after enqueueing, as a serving loop would at a switch.
        ttnn.add(sub_small, sub_small)
        start = time.perf_counter()
        mesh_device.quiesce_devices()
        timings["quiesce_pending"].append(time.perf_counter() - start)

        # A parent step after another parent step, ending in a finish.
        phase(parent_square)
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        phase(parent_square)
        ttnn.synchronize_device(mesh_device)
        timings["phase_steady"].append(time.perf_counter() - start)
        mesh_device.quiesce_devices()
        # The same step right after a submesh step, ending in a quiesce: it also pays to resend the binaries.
        phase(sub_square)
        mesh_device.quiesce_devices()
        start = time.perf_counter()
        phase(parent_square)
        mesh_device.quiesce_devices()
        timings["phase_after_switch"].append(time.perf_counter() - start)

    # The timed loop left both meshes' data and results intact.
    expected = square.float() @ square.float() + square.float()
    for mesh, tensor in ((mesh_device, parent_square), (submesh, sub_square)):
        result = ttnn.add(ttnn.matmul(tensor, tensor), tensor)
        for shard in per_device(result, mesh):
            torch.testing.assert_close(shard.float(), expected, rtol=5e-2, atol=1.0)
        mesh_device.quiesce_devices()

    summary = {name: _summary_us(samples) for name, samples in timings.items()}
    summary["switch_overhead_median_us"] = round(
        summary["phase_after_switch"]["median_us"] - summary["phase_steady"]["median_us"], 1
    )
    print(json.dumps(summary, indent=2))
    out = os.environ.get("SHARED_SUBMESH_SWITCH_COST_JSON")
    if out:
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
