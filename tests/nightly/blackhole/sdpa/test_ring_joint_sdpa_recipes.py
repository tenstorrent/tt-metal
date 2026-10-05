# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ring joint SDPA with precision recipes on a 1x2 Blackhole mesh, against an FP64 reference.

Recipes and their numerics: tech_reports/FlashAttention/SDPAPrecisionRecipes.md. Dense recipe coverage lives in
tests/ttnn/unit_tests/operations/sdpa/test_sdpa_recipes.py; this file covers what the ring adds: K/V arriving
shard by shard, joint K/V and logical lengths (host scalars or device tensors). Exp ring:
test_exp_ring_joint_sdpa_recipes.py.
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.sdpa.test_sdpa_recipes import L2_PCT_BOUND, VARIANTS, l2_pct, reference

RING = 2

pytestmark = pytest.mark.skipif(
    not is_blackhole() or os.environ.get("TT_METAL_SIMULATOR") is not None,
    reason="SDPA precision recipes run on Blackhole hardware",
)


def randn(*shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).bfloat16()


def precision_inputs(mesh, variant, values, mapper):
    """Upload values; LOW_PRECISION inputs are rounded with prepare_sdpa_input (Q first, then K and V)."""
    precision, kv_dtype = VARIANTS[variant]
    tensors = [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in values]
    if precision == ttnn.SDPAPrecision.LOW_PRECISION:
        tensors = [
            ttnn.transformer.prepare_sdpa_input(x, is_query=i == 0, dtype=ttnn.bfloat16 if i == 0 else kv_dtype)
            for i, x in enumerate(tensors)
        ]
    return tensors


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


def length_tensor(mesh, value):
    return ttnn.to_device(host_length(mesh, value), mesh)


def open_ring_mesh(fabric, **mesh_options):
    if ttnn.GetNumAvailableDevices() < RING:
        pytest.skip("Requires two connected Blackholes")
    ttnn.set_fabric_config(*fabric)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, RING), trace_region_size=16777216, **mesh_options)
    mesh.enable_program_cache()
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
    mesh.load_sub_device_manager(manager)
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    return mesh, manager, cores, grid


def close_ring_mesh(mesh, manager):
    mesh.reset_sub_device_stall_group()
    mesh.clear_loaded_sub_device_manager()
    mesh.remove_sub_device_manager(manager)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------------------------------------------
# Ring joint SDPA (linear topology)
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ring_mesh():
    fabric = (
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh, manager, cores, grid = open_ring_mesh(fabric)
    semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
    try:
        yield mesh, semaphores, grid.x - 1
    finally:
        close_ring_mesh(mesh, manager)


# batch, heads, kv_heads, local q rows, local k rows, head_dim, q_chunk, k_chunk, joint ("sharded"/"replicated"),
# logical_n (None = all rows), SDPA grid.
RING_CASES = {
    "q256_k512_d128": (1, 2, 2, 512, 1024, 128, 256, 512, None, None, (4, 2)),
    "joint_sharded": (1, 2, 2, 1024, 1024, 128, 256, 512, "sharded", None, (4, 2)),
    "joint_replicated": (1, 2, 2, 1024, 1024, 128, 256, 512, "replicated", None, (4, 2)),
    "logical_n_tail": (1, 2, 2, 1024, 1024, 128, 256, 512, None, 1300, (4, 2)),
    "gqa_batch2": (2, 4, 2, 512, 1024, 128, 256, 512, None, None, (8, 2)),
    "odd_q96_k160_d96": (1, 2, 2, 480, 800, 96, 96, 160, None, 1500, (4, 2)),
    "q128_k256_d256": (1, 1, 1, 256, 512, 256, 128, 256, None, None, (2, 1)),
    "odd_q288_k512_d128": (1, 2, 2, 576, 1024, 128, 288, 512, None, None, (4, 2)),
    # Four Q chunks per core: the recurrent state is checkpointed between them on every ring iteration.
    "multi_q_checkpoint": (1, 4, 4, 1024, 1024, 128, 256, 512, None, None, (2, 2)),
    "multi_q_checkpoint_wide": (1, 10, 10, 2368, 2368, 128, 288, 384, None, None, (8, 4)),
}


def run_ring(
    mesh,
    semaphores,
    ccl_column,
    inputs,
    joints,
    backing,
    *,
    grid,
    q_chunk,
    k_chunk,
    logical_n,
    logical_l,
    is_cross,
    **options,
):
    return ttnn.transformer.ring_joint_scaled_dot_product_attention(
        *inputs,
        *joints,
        persistent_output_buffer_k=backing[0],
        persistent_output_buffer_v=backing[1],
        joint_strategy="rear",
        logical_n=logical_n,
        logical_l=logical_l,
        is_causal=False,
        is_cross=is_cross,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
        ),
        dim=2,
        multi_device_global_semaphore=semaphores,
        num_links=1,
        cluster_axis=1,
        mesh_device=mesh,
        topology=ttnn.Topology.Linear,
        subdevice_id=ttnn.SubDeviceId(0),
        ccl_core_grid_offset=(ccl_column, 0),
        use_column_major_ccl=True,
        **options,
    )


def ring_case(mesh, variant, case):
    b, nh, nkv, q_local, k_local, d, q_chunk, k_chunk, joint, logical_n, grid = RING_CASES[case]
    q, k, v = (
        randn(b, nh, RING * q_local, d, seed=1),
        randn(b, nkv, RING * k_local, d, seed=2),
        randn(b, nkv, RING * k_local, d, seed=3),
    )
    logical_n = logical_n or RING * k_local
    if logical_n < RING * k_local:
        # Finite garbage past logical_n: the recipe must skip or mask it.
        for x in (k, v):
            x[..., logical_n:, :] = 8 * torch.randn(x[..., logical_n:, :].shape).bfloat16()
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    inputs = precision_inputs(mesh, variant, (q, k, v), shard)
    joint_host, joints, logical_l = None, [None] * 3, 0
    if joint:
        rows = 1024 if joint == "sharded" else 512
        joint_host = (randn(b, nh, rows, d, seed=4), randn(b, nkv, rows, d, seed=5), randn(b, nkv, rows, d, seed=6))
        mapper = shard if joint == "sharded" else ttnn.ReplicateTensorToMesh(mesh)
        joints, logical_l = precision_inputs(mesh, variant, joint_host, mapper), rows
    backing = [
        ttnn.allocate_tensor_on_device(list(k.shape), x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]
    kwargs = dict(grid=grid, q_chunk=q_chunk, k_chunk=k_chunk, logical_l=logical_l, is_cross=q_local != k_local)

    def expected(chip):
        queries = q.chunk(RING, dim=2)[chip]
        keys, values = k[..., :logical_n, :], v[..., :logical_n, :]
        if joint:
            jq = joint_host[0].chunk(RING, dim=2)[chip] if joint == "sharded" else joint_host[0]
            queries = torch.cat([queries, jq], dim=2)
            keys, values = torch.cat([keys, joint_host[1]], dim=2), torch.cat([values, joint_host[2]], dim=2)
        return reference(queries, keys, values)

    return inputs, joints, backing, logical_n, kwargs, expected, joint is not None


def legacy_ring_supports(q_chunk, k_chunk, head_dim):
    """FAST runs the legacy ring kernels, which support only these geometries."""
    return 128 <= q_chunk <= 320 and q_chunk % 32 == 0 and k_chunk in (256, 384, 512) and head_dim in (64, 128, 256)


@pytest.mark.parametrize("case", RING_CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_ring_joint_sdpa_recipe(ring_mesh, variant, case):
    mesh, semaphores, ccl_column = ring_mesh
    _, _, _, _, _, d, q_chunk, k_chunk, _, _, _ = RING_CASES[case]
    if variant == "fast" and not legacy_ring_supports(q_chunk, k_chunk, d):
        pytest.skip("FAST keeps the legacy ring kernels' geometry limits")
    inputs, joints, backing, logical_n, kwargs, expected, has_joint = ring_case(mesh, variant, case)
    out = run_ring(
        mesh,
        semaphores,
        ccl_column,
        inputs,
        joints,
        backing,
        logical_n=logical_n,
        precision=VARIANTS[variant][0],
        **kwargs,
    )
    for chip in range(RING):
        got = per_chip(out[0])[chip]
        if has_joint:
            got = torch.cat([got, per_chip(out[1])[chip]], dim=2)
        assert l2_pct(got, expected(chip)) < L2_PCT_BOUND[variant], f"chip {chip}"


@pytest.mark.parametrize("variant", ["fast", "standard", "accurate", "low_precision_bfp8"])
def test_ring_joint_sdpa_recipe_device_lengths(ring_mesh, variant):
    """logical_n as a device tensor: one trace, replayed as the length changes, matches the host-scalar path."""
    mesh, semaphores, ccl_column = ring_mesh
    inputs, joints, backing, _, kwargs, _, _ = ring_case(mesh, variant, "logical_n_tail")
    call = lambda n: run_ring(
        mesh, semaphores, ccl_column, inputs, joints, backing, logical_n=n, precision=VARIANTS[variant][0], **kwargs
    )
    lengths = [1300, 2048, 777]
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


@pytest.mark.parametrize("variant", ["fast", "standard", "accurate", "low_precision_bfp8"])
def test_ring_joint_sdpa_recipe_op_selected_blocking(ring_mesh, variant):
    """Chunk sizes of 0: the op chooses them."""
    mesh, semaphores, ccl_column = ring_mesh
    inputs, joints, backing, logical_n, kwargs, expected, has_joint = ring_case(mesh, variant, "joint_sharded")
    kwargs.update(q_chunk=0, k_chunk=0)
    out = run_ring(
        mesh,
        semaphores,
        ccl_column,
        inputs,
        joints,
        backing,
        logical_n=logical_n,
        precision=VARIANTS[variant][0],
        **kwargs,
    )
    for chip in range(RING):
        got = torch.cat([per_chip(out[0])[chip], per_chip(out[1])[chip]], dim=2)
        assert l2_pct(got, expected(chip)) < L2_PCT_BOUND[variant], f"chip {chip}"


def test_ring_joint_sdpa_fast_matches_legacy(ring_mesh):
    mesh, semaphores, ccl_column = ring_mesh
    inputs, joints, backing, logical_n, kwargs, _, _ = ring_case(mesh, "fast", "joint_sharded")
    fast = run_ring(
        mesh,
        semaphores,
        ccl_column,
        inputs,
        joints,
        backing,
        logical_n=logical_n,
        precision=ttnn.SDPAPrecision.FAST,
        **kwargs,
    )
    legacy = run_ring(mesh, semaphores, ccl_column, inputs, joints, backing, logical_n=logical_n, **kwargs)
    for index in (0, 1):
        assert all(torch.equal(a, b) for a, b in zip(per_chip(fast[index]), per_chip(legacy[index])))
