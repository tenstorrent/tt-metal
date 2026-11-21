# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exp ring joint SDPA with the precision recipes on one Blackhole ring, against an FP64 reference.

The op runs exactly two links, one per ring direction (one semaphore per link), on a fabric ring: a BH QuietBox 2's four
chips (1x4), or one row of a Blackhole Galaxy (1x8). Dense recipe coverage: operations/sdpa/test_sdpa_recipes.py; this
file covers what the exp ring adds: K/V arriving shard by shard over the fused all-gather, head-serial passes, joint K/V
and logical lengths (host scalars or a device tensor replayed in a trace).
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_recipe_test_utils import (
    L2_PCT_BOUND,
    VARIANTS,
    blackhole_only,
    fast_input,
    fp32_dest_config,
    l2_pct,
    randn,
    reference,
)

pytestmark = blackhole_only

EXP_RING_LINKS = 2


def open_exp_ring_mesh():
    """One fabric ring and its parent mesh: a 1x8 row submesh of a Galaxy's 4x8 mesh (the fabric routers of a smaller
    opened mesh are not all built), or a QuietBox 2's four chips."""
    devices = ttnn.GetNumAvailableDevices()
    if devices >= 32:
        parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8), trace_region_size=16777216)
        return parent, parent.create_submesh(ttnn.MeshShape(1, 8)), 8
    if devices >= 4:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=16777216)
        return mesh, mesh, 4
    pytest.skip("Requires a fabric ring of 4 or 8 Blackhole devices")


def precision_inputs(mesh, variant, values, mapper):
    """Upload (Q, K, V); FAST rounds them for LoFi first (fast_input) and stores K/V in its K/V type."""
    precision, kv_dtype = VARIANTS[variant]
    if precision != ttnn.SDPAPrecision.FAST:
        return [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in values]
    return [
        ttnn.from_torch(
            fast_input(x, i == 0, kv_dtype),
            dtype=ttnn.bfloat16 if i == 0 else kv_dtype,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
        )
        for i, x in enumerate(values)
    ]


def per_chip(tensor):
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(tensor)]


def host_length(mesh, value):
    """A single-valued uint32 length, replicated over the mesh (host tensor)."""
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def assert_per_chip(out, expected, variant, chips, has_joint):
    """Each chip's output (followed by its joint output when has_joint) against expected(chip)."""
    outputs = per_chip(out[0])
    joint_outputs = per_chip(out[1]) if has_joint else None
    for chip in range(chips):
        got = torch.cat([outputs[chip], joint_outputs[chip]], dim=2) if has_joint else outputs[chip]
        assert l2_pct(got, expected(chip)) < L2_PCT_BOUND[variant], f"chip {chip}"


@pytest.fixture(scope="module")
def exp_ring_mesh():
    if ttnn.GetNumAvailableDevices() < 4:
        pytest.skip("Requires a fabric ring of 4 or 8 Blackhole devices")
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router,
    )
    parent, mesh, ring = open_exp_ring_mesh()
    mesh.enable_program_cache()
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
    mesh.load_sub_device_manager(manager)
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(EXP_RING_LINKS)]
    try:
        yield mesh, semaphores, ring
    finally:
        mesh.reset_sub_device_stall_group()
        mesh.clear_loaded_sub_device_manager()
        mesh.remove_sub_device_manager(manager)
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# heads, local rows, joint rows, logical_n (None = all rows), grid (SDPA columns + 1 MUX column, 4 rows), q_chunk.
# Heads x Q segments over the 4 grid rows set the passes per row (1-3); a small logical_n leaves later shards all pad.
EXP_RING_CASES = {
    "q256_aligned": (4, 1024, 0, None, (5, 4), 256),
    "joint": (2, 1024, 512, None, (4, 4), 256),
    "subtile_logical_n": (4, 1024, 0, 777, (5, 4), 256),
    "joint_tails": (2, 768, 768, 1300, (4, 4), 256),
    "two_pass": (8, 1024, 0, None, (5, 4), 256),
    "three_pass_joint_skip": (6, 1024, 512, 1536, (4, 4), 256),
    "q224_two_pass": (8, 1024, 0, None, (6, 4), 224),
    "q128": (2, 1024, 0, None, (5, 4), 128),
    "q160_odd": (4, 640, 0, None, (5, 4), 160),
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


def exp_ring_case(mesh, ring, variant, case):
    heads, local, joint, logical_n, grid, q_chunk = EXP_RING_CASES[case]
    q, k, v = (randn(1, heads, ring * local, 128, seed=10 + i) for i in range(3))
    logical_n = logical_n or ring * local
    if logical_n < ring * local:
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


def synchronized(mesh, out):
    # Back-to-back calls on one global semaphore set need a barrier (the models ping-pong two sets): a device that
    # starts the next call signals a ring neighbour that is still in this one, and the ring hangs.
    ttnn.synchronize_device(mesh)
    return out


@pytest.mark.parametrize("case", EXP_RING_CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_exp_ring_joint_sdpa_recipe(exp_ring_mesh, variant, case):
    mesh, semaphores, ring = exp_ring_mesh
    inputs, joints, backing, logical_n, kwargs, expected, has_joint = exp_ring_case(mesh, ring, variant, case)
    out = run_exp_ring(
        mesh, semaphores, inputs, joints, backing, logical_n=logical_n, precision=VARIANTS[variant][0], **kwargs
    )
    synchronized(mesh, out)
    assert_per_chip(out, expected, variant, ring, has_joint)


@pytest.mark.parametrize("variant", ["standard", "balanced", "fast_bfp4"])
def test_exp_ring_joint_sdpa_recipe_device_lengths(exp_ring_mesh, variant):
    """logical_n as a device tensor: one trace, replayed as the length changes, matches the host-scalar path."""
    mesh, semaphores, ring = exp_ring_mesh
    inputs, joints, backing, _, kwargs, _, _ = exp_ring_case(mesh, ring, variant, "subtile_logical_n")
    lengths = [777, 2048, 1536]

    def call(n):
        out = run_exp_ring(
            mesh, semaphores, inputs, joints, backing, logical_n=n, precision=VARIANTS[variant][0], **kwargs
        )
        return synchronized(mesh, out)

    scalar = [per_chip(call(n)[0]) for n in lengths]
    n_tensor = ttnn.to_device(host_length(mesh, lengths[0]), mesh)
    # The device-tensor path is its own program (worst-case placeholders); compile it before capture.
    assert all(torch.equal(a, b) for a, b in zip(per_chip(call(n_tensor)[0]), scalar[0]))
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        traced = run_exp_ring(
            mesh, semaphores, inputs, joints, backing, logical_n=n_tensor, precision=VARIANTS[variant][0], **kwargs
        )
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        for i in [0, 1, 2, 0]:
            ttnn.copy_host_to_device_tensor(host_length(mesh, lengths[i]), n_tensor)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert all(torch.equal(a, b) for a, b in zip(per_chip(traced[0]), scalar[i])), f"logical_n={lengths[i]}"
    finally:
        ttnn.release_trace(mesh, trace)


def test_exp_ring_joint_sdpa_ignores_compute_kernel_config(exp_ring_mesh):
    """The recipe owns the numerics: a compute_kernel_config (here FP32 dest) is ignored, so the call equals the default
    (STANDARD) one bit for bit."""
    mesh, semaphores, ring = exp_ring_mesh
    inputs, joints, backing, logical_n, kwargs, expected, _ = exp_ring_case(mesh, ring, "standard", "q256_aligned")
    default = synchronized(mesh, run_exp_ring(mesh, semaphores, inputs, joints, backing, logical_n=logical_n, **kwargs))
    ignored = synchronized(
        mesh,
        run_exp_ring(
            mesh,
            semaphores,
            inputs,
            joints,
            backing,
            logical_n=logical_n,
            compute_kernel_config=fp32_dest_config(mesh),
            **kwargs,
        ),
    )
    for chip, (a, b) in enumerate(zip(per_chip(default[0]), per_chip(ignored[0]))):
        assert torch.equal(a, b), f"chip {chip}"
        assert l2_pct(a, expected(chip)) < L2_PCT_BOUND["standard"], f"chip {chip}"
