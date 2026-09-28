# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exp ring joint SDPA with precision recipes on a 1x2 Blackhole mesh, against an FP64 reference.

Recipes and their numerics: tech_reports/FlashAttention/SDPAPrecisionRecipes.md. Ring joint SDPA recipes:
test_ring_joint_sdpa_recipes.py.
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa_recipes import (
    RING,
    close_ring_mesh,
    host_length,
    length_tensor,
    open_ring_mesh,
    per_chip,
    precision_inputs,
    randn,
)
from tests.ttnn.unit_tests.operations.sdpa.test_sdpa_recipes import L2_PCT_BOUND, VARIANTS, l2_pct, reference

pytestmark = pytest.mark.skipif(
    not is_blackhole() or os.environ.get("TT_METAL_SIMULATOR") is not None,
    reason="SDPA precision recipes run on Blackhole hardware",
)


@pytest.fixture(scope="module")
def exp_ring_mesh():
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    fabric = (
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router,
    )
    mesh, manager, cores, _ = open_ring_mesh(fabric)
    semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
    try:
        yield mesh, semaphores
    finally:
        close_ring_mesh(mesh, manager)


# heads, local rows, joint rows, logical_n (None = all rows), grid (SDPA columns + 1 MUX column, 4 rows), q_chunk.
# Heads x Q segments over the 4 grid rows set the passes per row (1-3).
# Recipes that pair Q tile rows and round an odd Q chunk up to the next even one.
PAIRED_RECIPES = (ttnn.SDPAPrecision.STANDARD, ttnn.SDPAPrecision.LOW_PRECISION)
EXP_RING_CASES = {
    "q256_aligned": (4, 1024, 0, None, (5, 4), 256),
    "joint": (2, 1024, 512, None, (4, 4), 256),
    "subtile_logical_n": (4, 1024, 0, 777, (5, 4), 256),
    "joint_tails": (2, 768, 768, 1300, (4, 4), 256),
    "two_pass": (8, 1024, 0, None, (5, 4), 256),
    "three_pass_joint_skip": (6, 1024, 512, 1536, (4, 4), 256),
    "q224_two_pass": (8, 1024, 0, None, (6, 4), 224),
    "q128": (2, 1024, 0, None, (5, 4), 128),
}


def run_exp_ring(mesh, semaphores, inputs, joints, backing, *, grid, q_chunk, logical_n, **options):
    return ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
        *inputs,
        *joints,
        persistent_output_buffer_k=backing[0],
        persistent_output_buffer_v=backing[1],
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=512
        ),
        dim=2,
        multi_device_global_semaphore=semaphores,
        num_links=2,
        cluster_axis=1,
        mesh_device=mesh,
        topology=ttnn.Topology.Ring,
        subdevice_id=ttnn.SubDeviceId(0),
        num_workers_per_link=grid[1] // 2,
        num_buffers_per_channel=16,
        **options,
    )


def exp_ring_case(mesh, variant, case):
    heads, local, joint, logical_n, grid, q_chunk = EXP_RING_CASES[case]
    q, k, v = (randn(1, heads, RING * local, 128, seed=10 + i) for i in range(3))
    logical_n = logical_n or RING * local
    if logical_n < RING * local:
        for x in (k, v):
            x[..., logical_n:, :] = 8 * torch.randn(x[..., logical_n:, :].shape).bfloat16()
    inputs = precision_inputs(mesh, variant, (q, k, v), ttnn.ShardTensorToMesh(mesh, dim=2))
    joint_host = tuple(randn(1, heads, joint, 128, seed=20 + i) for i in range(3)) if joint else None
    joints = precision_inputs(mesh, variant, joint_host, ttnn.ReplicateTensorToMesh(mesh)) if joint else [None] * 3
    backing = [
        ttnn.allocate_tensor_on_device(list(k.shape), x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]

    def expected(chip):
        queries = q[..., chip * local : (chip + 1) * local, :]
        keys, values = k[..., :logical_n, :], v[..., :logical_n, :]
        if joint:
            queries = torch.cat([queries, joint_host[0]], dim=2)
            keys, values = torch.cat([keys, joint_host[1]], dim=2), torch.cat([values, joint_host[2]], dim=2)
        return reference(queries, keys, values)

    return inputs, joints, backing, logical_n, dict(grid=grid, q_chunk=q_chunk), expected, joint > 0


@pytest.mark.parametrize("case", EXP_RING_CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_exp_ring_joint_sdpa_recipe(exp_ring_mesh, variant, case):
    if case == "q224_two_pass" and VARIANTS[variant][0] in PAIRED_RECIPES:
        pytest.skip("paired recipes round Q224 up to Q256: 4 Q chunks per head do not fill the case's 5 SDPA columns")
    mesh, semaphores = exp_ring_mesh
    inputs, joints, backing, logical_n, kwargs, expected, has_joint = exp_ring_case(mesh, variant, case)
    out = run_exp_ring(
        mesh, semaphores, inputs, joints, backing, logical_n=logical_n, precision=VARIANTS[variant][0], **kwargs
    )
    for chip in range(RING):
        got = per_chip(out[0])[chip]
        if has_joint:
            got = torch.cat([got, per_chip(out[1])[chip]], dim=2)
        assert l2_pct(got, expected(chip)) < L2_PCT_BOUND[variant], f"chip {chip}"


@pytest.mark.parametrize("variant", ["standard", "balanced", "low_precision_bfp4"])
def test_exp_ring_joint_sdpa_recipe_device_lengths(exp_ring_mesh, variant):
    mesh, semaphores = exp_ring_mesh
    inputs, joints, backing, _, kwargs, _, _ = exp_ring_case(mesh, variant, "subtile_logical_n")
    call = lambda n: run_exp_ring(
        mesh, semaphores, inputs, joints, backing, logical_n=n, precision=VARIANTS[variant][0], **kwargs
    )
    lengths = [777, 2048, 1536]
    scalar = [per_chip(call(n)[0]) for n in lengths]
    n_tensor = length_tensor(mesh, lengths[0])
    # The device-tensor path is its own program (worst-case placeholders); compile it before capture.
    assert all(torch.equal(a, b) for a, b in zip(per_chip(call(n_tensor)[0]), scalar[0]))
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        traced = call(n_tensor)
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        for i in [0, 1, 2, 0]:
            ttnn.copy_host_to_device_tensor(host_length(mesh, lengths[i]), n_tensor)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert all(torch.equal(a, b) for a, b in zip(per_chip(traced[0]), scalar[i])), f"logical_n={lengths[i]}"
    finally:
        ttnn.release_trace(mesh, trace)


@pytest.mark.parametrize("variant", ["fast", "standard", "balanced", "low_precision_bfp8"])
def test_exp_ring_joint_sdpa_recipe_op_selected_blocking(exp_ring_mesh, variant):
    """Q chunk of 0: the op chooses it (and may narrow the SDPA grid width)."""
    mesh, semaphores = exp_ring_mesh
    inputs, joints, backing, logical_n, kwargs, expected, _ = exp_ring_case(mesh, variant, "two_pass")
    kwargs.update(q_chunk=0, grid=(8, 4))
    out = run_exp_ring(
        mesh, semaphores, inputs, joints, backing, logical_n=logical_n, precision=VARIANTS[variant][0], **kwargs
    )
    for chip in range(RING):
        assert l2_pct(per_chip(out[0])[chip], expected(chip)) < L2_PCT_BOUND[variant], f"chip {chip}"
