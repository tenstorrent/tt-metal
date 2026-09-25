# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import VARIANTS, digest, make_inputs, metrics, prepare, reference
from .test_sdpa_joint_recipes import joint, options


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("distribution", ["replicated", "heads", "queries"])
def test_recipe_spmd_mesh(mesh_device, variant, distribution, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    mesh_device.enable_program_cache()
    host = make_inputs(1024, "normal", q_length=512, heads=4)
    replicated = ttnn.ReplicateTensorToMesh(mesh_device)
    if distribution == "heads":
        mappers = [ttnn.ShardTensorToMesh(mesh_device, dim=1)] * 3
        references = [reference(*(x.chunk(2, dim=1)[chip] for x in host)) for chip in range(2)]
    elif distribution == "queries":
        mappers = [ttnn.ShardTensorToMesh(mesh_device, dim=2), replicated, replicated]
        references = [reference(q, *host[1:]) for q in host[0].chunk(2, dim=2)]
    else:
        mappers = [replicated] * 3
        references = [reference(*host)] * 2

    def upload(values):
        return prepare(
            [
                ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                for x, mapper in zip(values, mappers)
            ],
            variant,
        )

    inputs = upload(host)
    grid = (4, 2)
    output = ttnn.transformer.scaled_dot_product_attention(*inputs, is_causal=False, **options(variant, grid))
    dense = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(output)]
    assert len(dense) == len(references)
    limits = {"A": 8, "B": 8, "C": 2, "D": 0.4, "E_bf16": 8, "E_bfp8": 8, "E_bfp4": 35}
    for chip, (actual, expected) in enumerate(zip(dense, references)):
        observed = metrics(actual, expected)
        assert observed["l2_pct"] < limits[variant]
        for key, value in observed.items():
            record_property(f"chip{chip}_{key}", value)

    # Sequence-sharded Q needs rank-local segment ordering; qualify joint on the
    # replicated and head-sharded layouts here, not a different global ordering.
    if distribution != "queries":
        segments = [
            upload([x[..., :split, :].contiguous() for x, split in zip(host, (256, 512, 512))]),
            upload([x[..., split:, :].contiguous() for x, split in zip(host, (256, 512, 512))]),
        ]
        outputs = joint(segments, variant, grid)
        per_segment = [[ttnn.to_torch(x) for x in ttnn.get_device_tensors(output)] for output in outputs]
        for chip in range(2):
            assert digest(torch.cat([segment[chip] for segment in per_segment], dim=2)) == digest(dense[chip])

    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = ttnn.transformer.scaled_dot_product_attention(*inputs, is_causal=False, **options(variant, grid))
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            actual = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(traced)]
            assert [digest(x) for x in actual] == [digest(x) for x in dense]
    finally:
        ttnn.release_trace(mesh_device, trace)
