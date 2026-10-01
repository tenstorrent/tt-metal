# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Masked LLK router sort versus the current composed operations, with fresh inputs and trace reuse."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")


def _host(tensor):
    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0], other) for other in replicas[1:])
    return replicas[0]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
@pytest.mark.parametrize("rows", [1, 5, 32])
@pytest.mark.parametrize("exp_live", ["0", "1"])
def test_router_tail_current_llk_and_reuse(mesh_device, rows, exp_live, monkeypatch):
    monkeypatch.setenv(router_tail.EXP_LIVE_ENV, exp_live)
    cached = None
    for seed in (310, 311):
        generator = torch.Generator().manual_seed(seed)
        values = torch.randn((1, 1, rows, 512), generator=generator).to(torch.bfloat16).float()
        # Repeated maxima exercise the default unstabilized tie order, independent for each row.
        values[..., :16] = 3.0
        logits = ttnn.from_torch(
            values,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        expected = router_tail.router_tail_composed(logits)
        outputs = router_tail.router_tail(logits)
        for got, reference in zip(outputs, expected):
            assert torch.equal(_host(got), _host(reference))
        # Independent numeric control on probabilities; ties need not use PyTorch's index order.
        scores, indices = (_host(t) for t in outputs)
        probabilities = torch.softmax(values, dim=-1)
        selected = probabilities.gather(-1, indices.long())
        selected /= selected.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(scores.float(), selected, rtol=0.02, atol=0.001)
        current = mesh_device.num_program_cache_entries()
        if cached is not None:
            assert current == cached
        cached = current
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            router_tail.router_tail_into(logits, *outputs)
        finally:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            for _ in range(3):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                for got, reference in zip(outputs, expected):
                    assert torch.equal(_host(got), _host(reference))
            assert mesh_device.num_program_cache_entries() == cached
        finally:
            ttnn.release_trace(mesh_device, trace_id)
        for tensor in (logits, *outputs, *expected):
            ttnn.deallocate(tensor)
