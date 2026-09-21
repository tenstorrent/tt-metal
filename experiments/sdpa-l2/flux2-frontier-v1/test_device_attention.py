# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qualification before any frontier model image is generated."""

import json
import math
from types import SimpleNamespace

import pytest
import torch
import ttnn

import device_attention as kernel


def test_recipe_defines_match_frontier_records():
    files = {
        "A": "chain-h10-32768-main-v1.json",
        "B": "chain-h10-32768-fast-v1.json",
        "C": "chain-h10-32768-balanced-v1.json",
        "D": "chain-h10-32768-accurate-v1.json",
        "E": "chain-h10-32768-lofi_fast_b8-v1.json",
        "F": "native32k-lofi_fp32_b8-v1.json",
        "G": "chain-h10-32768-lofi_fast_b4-v1.json",
    }
    for variant, filename in files.items():
        recorded = json.loads((kernel.RESEARCH / filename).read_text())
        fp32, _, defines, fidelity = kernel.recipe(variant)
        assert defines == recorded["defines"], variant
        assert fp32 == recorded["fp32_dst"], variant
        assert str(fidelity) == recorded["fidelity"], variant


@pytest.fixture
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 32 * 1024**2}


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
@pytest.mark.parametrize("variant", list(kernel.VARIANTS))
@pytest.mark.timeout(600)
def test_rectangular_mesh(mesh_device, variant):
    generator = torch.Generator().manual_seed(1240)
    inputs = [torch.randn((1, 2, n, 128), generator=generator).bfloat16() for n in (512, 1024, 1024)]
    if variant == "G":
        for value in inputs[1:]:
            kernel.B4.validate_input(value)
    q, k, v = [x.double() for x in inputs]
    reference = (q @ k.transpose(-2, -1) / math.sqrt(128)).softmax(-1) @ v
    tensors = [
        ttnn.from_torch(
            x, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
        )
        for x in inputs
    ]

    def invoke():
        prepared = [kernel.prepare(mesh_device, x, variant, is_q=i == 0, cores=8) for i, x in enumerate(tensors)]
        result = kernel.attention(mesh_device, *prepared, variant, max_cores=4)
        return prepared, result

    prepared, output = invoke()
    shards = ttnn.get_device_tensors(output)
    assert len(shards) == 8
    expected_bits = ttnn.to_torch(shards[0]).clone()
    for shard in shards:
        assert torch.equal(ttnn.to_torch(shard), expected_bits)
    assert bool(torch.isfinite(expected_bits).all())
    actual = expected_bits.double()
    l2 = 100 * float(torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference))
    pcc = float(torch.corrcoef(torch.stack((actual.flatten(), reference.flatten())))[0, 1])
    # Loose implementation-failure gates, not new universal accuracy acceptance bands.
    assert l2 < {"A": 8, "B": 8, "C": 1, "D": 0.5, "E": 8, "F": 8, "G": 30}[variant]
    for i, value in enumerate(prepared):
        result = ttnn.to_torch(ttnn.get_device_tensors(value)[0]).float()
        if variant in "ABCD":
            oracle = inputs[i].float()
        elif variant == "G" and i:
            oracle = kernel.B4.host_rne_bfp4(inputs[i])
        else:
            oracle = kernel.PREP.MODEL.round_significand(inputs[i], 7 if i == 0 else 5)
            if i:
                oracle = kernel.PREP.MODEL.quantize(oracle, 7, "device")
        assert torch.equal(result, oracle), f"Preprocessing mismatch for input {i}"
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_inputs, traced_output = invoke()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            for shard in ttnn.get_device_tensors(traced_output):
                assert torch.equal(ttnn.to_torch(shard), expected_bits)
    finally:
        ttnn.release_trace(mesh_device, trace)
    print(
        "FRONTIER_QUALIFICATION",
        json.dumps(
            dict(
                variant=variant,
                l2_pct=l2,
                pcc=pcc,
                finite=True,
                exact_preprocessing=True,
                all_devices_equal=True,
                trace_equal=True,
            )
        ),
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
@pytest.mark.parametrize("variant", list(kernel.VARIANTS))
@pytest.mark.timeout(600)
def test_sharded_joint_attention(mesh_device, variant):
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import from_torch, to_torch

    from model_attention import FrontierAttention

    generator = torch.Generator().manual_seed(1241)
    spatial = [torch.randn((1, 8, 1024, 128), generator=generator).bfloat16() for _ in range(3)]
    prompt = [torch.randn((1, 8, 512, 128), generator=generator).bfloat16() for _ in range(3)]
    if variant == "G":
        for value in spatial[1:] + prompt[1:]:
            kernel.B4.validate_input(value)
    q, k, v = [torch.cat(parts, dim=2).double() for parts in zip(spatial, prompt)]
    reference = (q @ k.transpose(-2, -1) / math.sqrt(128)).softmax(-1) @ v
    axes = [None, 1, 0, None]
    tensors = [from_torch(x, device=mesh_device, mesh_axes=axes) for x in spatial + prompt]
    attn = SimpleNamespace(
        mesh_device=mesh_device,
        shard_prompt=True,
        parallel_config=SimpleNamespace(sequence_parallel=SimpleNamespace(factor=2, mesh_axis=0)),
        ccl_manager=CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear),
    )
    adapter = FrontierAttention(variant)

    def invoke():
        return adapter.run("joint.test", attn, *tensors, 1024, 512)

    outputs = invoke()
    actual = torch.cat([to_torch(x, mesh_axes=axes) for x in outputs], dim=2)
    assert actual.shape == reference.shape and bool(torch.isfinite(actual).all())
    l2 = 100 * float(torch.linalg.vector_norm(actual.double() - reference) / torch.linalg.vector_norm(reference))
    assert l2 < {"A": 8, "B": 8, "C": 1, "D": 0.5, "E": 8, "F": 8, "G": 30}[variant]
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = invoke()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            replay = torch.cat([to_torch(x, mesh_axes=axes) for x in traced], dim=2)
            assert torch.equal(replay, actual)
    finally:
        ttnn.release_trace(mesh_device, trace)
    print(
        "FRONTIER_JOINT_QUALIFICATION",
        json.dumps(dict(variant=variant, l2_pct=l2, trace_equal=True, transport=adapter.transport["joint.test"])),
    )
