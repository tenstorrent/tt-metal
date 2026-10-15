# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exp ring joint SDPA with precision recipes on a ring of four Blackholes, against an FP64 reference.

Recipes and their numerics: tech_reports/FlashAttention/SDPAPrecisionRecipes.md. Ring joint SDPA recipes:
test_ring_joint_sdpa_recipes.py.
"""

import pytest
import torch
import ttnn

from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa_recipes import (
    assert_per_chip,
    check_device_length_trace,
    close_ring_mesh,
    open_ring_mesh,
    per_chip,
    precision_inputs,
)
from tests.ttnn.unit_tests.operations.sdpa.sdpa_recipe_test_utils import (
    L2_PCT_BOUND,
    VARIANTS,
    blackhole_only,
    fp32_dest_config,
    l2_pct,
    randn,
    reference,
)

pytestmark = blackhole_only

# The op runs exactly two links, one per ring direction (one semaphore per link). Like the legacy exp ring tests it
# needs a ring of four devices (a BH QuietBox 2): the two chips of a single P300 share only two ethernet channels.
EXP_RING = 4
EXP_RING_LINKS = 2


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
    mesh, manager, cores, _ = open_ring_mesh(fabric, ring=EXP_RING)
    semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(EXP_RING_LINKS)]
    try:
        yield mesh, semaphores
    finally:
        close_ring_mesh(mesh, manager)


# heads, local rows, joint rows, logical_n (None = all rows), grid (SDPA columns + 1 MUX column, 4 rows), q_chunk.
# Heads x Q segments over the 4 grid rows set the passes per row (1-3).
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


def run_exp_ring(mesh, semaphores, inputs, joints, backing, *, grid, q_chunk, logical_n, k_chunk=512, **options):
    return ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
        *inputs,
        *joints,
        persistent_output_buffer_k=backing[0],
        persistent_output_buffer_v=backing[1],
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
        ),
        dim=2,
        multi_device_global_semaphore=semaphores,
        num_links=EXP_RING_LINKS,
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
    q, k, v = (randn(1, heads, EXP_RING * local, 128, seed=10 + i) for i in range(3))
    logical_n = logical_n or EXP_RING * local
    if logical_n < EXP_RING * local:
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
    mesh, semaphores = exp_ring_mesh
    inputs, joints, backing, logical_n, kwargs, expected, has_joint = exp_ring_case(mesh, variant, case)
    out = run_exp_ring(
        mesh, semaphores, inputs, joints, backing, logical_n=logical_n, precision=VARIANTS[variant][0], **kwargs
    )
    assert_per_chip(out, expected, variant, EXP_RING, has_joint)


@pytest.mark.parametrize("variant", ["standard", "balanced", "fast_bfp4"])
def test_exp_ring_joint_sdpa_recipe_device_lengths(exp_ring_mesh, variant):
    mesh, semaphores = exp_ring_mesh
    inputs, joints, backing, _, kwargs, _, _ = exp_ring_case(mesh, variant, "subtile_logical_n")
    call = lambda n: run_exp_ring(
        mesh, semaphores, inputs, joints, backing, logical_n=n, precision=VARIANTS[variant][0], **kwargs
    )
    check_device_length_trace(mesh, call, [777, 2048, 1536])


@pytest.mark.parametrize("variant", ["standard", "balanced", "fast_bfp8"])
def test_exp_ring_joint_sdpa_recipe_op_selected_blocking(exp_ring_mesh, variant):
    """Q chunk of 0: the op chooses it (and may narrow the SDPA grid width)."""
    mesh, semaphores = exp_ring_mesh
    inputs, joints, backing, logical_n, kwargs, expected, _ = exp_ring_case(mesh, variant, "two_pass")
    kwargs.update(q_chunk=0, grid=(8, 4))
    out = run_exp_ring(
        mesh, semaphores, inputs, joints, backing, logical_n=logical_n, precision=VARIANTS[variant][0], **kwargs
    )
    assert_per_chip(out, expected, variant, EXP_RING, has_joint=False)


def test_exp_ring_joint_sdpa_precision_routing(exp_ring_mesh):
    """Without precision, calls the legacy exp ring kernel cannot build run a recipe at op-chosen chunks: FP32 dest
    runs ACCURATE (bitwise the explicit call with op-chosen chunks), and a BF16-dest Q chunk of one tile row (a single
    Q subblock, outside the streaming kernel's rule; 8 serial passes on this grid, the op allows 3) runs STANDARD.
    Both used to fail to compile."""
    mesh, semaphores = exp_ring_mesh
    inputs, joints, backing, logical_n, kwargs, expected, _ = exp_ring_case(mesh, "accurate", "q256_aligned")
    call = lambda **extra: run_exp_ring(mesh, semaphores, inputs, joints, backing, logical_n=logical_n, **extra)
    routed = call(compute_kernel_config=fp32_dest_config(mesh), **kwargs)
    explicit = call(precision=ttnn.SDPAPrecision.ACCURATE, grid=kwargs["grid"], q_chunk=0, k_chunk=0)
    for chip in range(EXP_RING):
        assert torch.equal(per_chip(routed[0])[chip], per_chip(explicit[0])[chip]), f"chip {chip}"
        assert l2_pct(per_chip(routed[0])[chip], expected(chip)) < L2_PCT_BOUND["accurate"], f"chip {chip}"
    out = call(grid=kwargs["grid"], q_chunk=32)
    for chip in range(EXP_RING):
        assert l2_pct(per_chip(out[0])[chip], expected(chip)) < L2_PCT_BOUND["standard"], f"chip {chip}"
