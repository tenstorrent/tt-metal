# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qualify padded SP4/TP2 attention before running any non-stock video."""

import json
import math
from types import SimpleNamespace

import pytest
import torch
import ttnn

from attention import WanAttentionAdapter, kernel
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import from_torch, to_torch


@pytest.fixture
def device_params():
    return dict(fabric_config=ttnn.FabricConfig.FABRIC_1D, l1_small_size=65536, trace_region_size=32 * 1024**2)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("variant", list("DCBEFG"))
@pytest.mark.timeout(600)
def test_padded_sp4(mesh_device, variant):
    torch.set_num_threads(8)
    generator = torch.Generator().manual_seed(4282)
    values = [torch.randn((1, 4, 1024, 128), generator=generator).bfloat16() for _ in range(3)]
    # If the last eight keys are not masked, these values overwhelm the result.
    values[1][:, :, -8:] = 0
    values[2][:, :, -8:] = 128
    q, k, v = [x.double() for x in values]
    reference = (q[:, :, :1016] @ k[:, :, :1016].transpose(-1, -2) / math.sqrt(128)).softmax(-1) @ v[:, :, :1016]
    axes = [None, 0, 1, None]
    tensors = [from_torch(x, device=mesh_device, mesh_axes=axes) for x in values]
    attn = SimpleNamespace(
        mesh_device=mesh_device,
        parallel_config=SimpleNamespace(sequence_parallel=SimpleNamespace(factor=4, mesh_axis=1)),
        ccl_manager=CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear),
    )
    adapter = WanAttentionAdapter(variant)

    def invoke():
        return adapter.run("qualification", attn, *tensors, 1016)

    result = invoke()
    actual = to_torch(result, mesh_axes=axes)[:, :, :1016].double()
    assert torch.isfinite(actual).all()
    l2 = 100 * float(torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference))
    pcc = float(torch.corrcoef(torch.stack([actual.flatten(), reference.flatten()]))[0, 1])
    assert l2 < dict(D=0.5, C=1, B=8, E=8, F=8, G=30)[variant], (variant, l2)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = invoke()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            replay = to_torch(traced, mesh_axes=axes)[:, :, :1016].double()
            assert torch.equal(replay, actual)
    finally:
        ttnn.release_trace(mesh_device, trace)
    print(
        "WAN_MASK_QUALIFICATION",
        json.dumps(dict(variant=variant, l2_pct=l2, pcc=pcc, replay_exact=True, transport=adapter.transport)),
        flush=True,
    )
