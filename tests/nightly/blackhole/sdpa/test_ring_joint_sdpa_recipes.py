# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ring joint SDPA with precision recipes on a 1x2 Blackhole mesh, against an FP64 reference.

Recipes and their numerics: tech_reports/FlashAttention/SDPAPrecisionRecipes.md. Dense recipe coverage lives in
operations/sdpa/test_sdpa_recipes.py under tests/ttnn/unit_tests/ (fast subset) and tests/ttnn/nightly/unit_tests/
(sweeps); this file covers what the ring adds: K/V arriving shard by shard, joint K/V and logical lengths (host
scalars or device tensors). Exp ring: test_exp_ring_joint_sdpa_recipes.py.
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sdpa_recipe_test_utils import (
    L2_PCT_BOUND,
    VARIANTS,
    fp32_dest_config,
    key_mask,
    l2_pct,
    reference,
    stored,
)

RING = 2

pytestmark = pytest.mark.skipif(
    not is_blackhole() or os.environ.get("TT_METAL_SIMULATOR") is not None,
    reason="SDPA precision recipes run on Blackhole hardware",
)


def randn(*shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).bfloat16()


def precision_inputs(mesh, variant, values, mapper):
    """Upload values; FAST inputs are rounded with prepare_sdpa_input (Q first, then K and V)."""
    precision, kv_dtype = VARIANTS[variant]
    tensors = [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in values]
    if precision == ttnn.SDPAPrecision.FAST:
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


def open_ring_mesh(fabric, ring=RING, **mesh_options):
    if ttnn.GetNumAvailableDevices() < ring:
        pytest.skip(f"Requires {ring} connected Blackholes")
    ttnn.set_fabric_config(*fabric)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, ring), trace_region_size=16777216, **mesh_options)
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
    is_causal=False,
    is_balanced=False,
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
        is_causal=is_causal,
        is_balanced=is_balanced,
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


@pytest.mark.parametrize("case", RING_CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_ring_joint_sdpa_recipe(ring_mesh, variant, case):
    mesh, semaphores, ccl_column = ring_mesh
    _, _, _, _, _, d, q_chunk, k_chunk, _, _, _ = RING_CASES[case]
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


@pytest.mark.parametrize("variant", ["standard", "accurate", "fast_bfp8"])
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


@pytest.mark.parametrize("variant", ["standard", "accurate", "fast_bfp8"])
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


@pytest.mark.parametrize("variant", ["standard", "balanced", "accurate"])
def test_ring_joint_sdpa_recipe_legacy_arguments(ring_mesh, variant):
    """BFP8 K/V on the BF16-input recipes, a custom scale, and an ignored compute config / exp_approx_mode."""
    mesh, semaphores, ccl_column = ring_mesh
    b, nh, nkv, q_local, k_local, d, q_chunk, k_chunk, _, _, grid = RING_CASES["gqa_batch2"]
    q = randn(b, nh, RING * q_local, d, seed=1)
    k, v = (stored(randn(b, nkv, RING * k_local, d, seed=s), ttnn.bfloat8_b) for s in (2, 3))
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    inputs = [
        ttnn.from_torch(x, dtype=dtype, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard)
        for x, dtype in ((q, ttnn.bfloat16), (k, ttnn.bfloat8_b), (v, ttnn.bfloat8_b))
    ]
    backing = [
        ttnn.allocate_tensor_on_device(list(k.shape), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for _ in range(2)
    ]
    out = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        *inputs,
        None,
        None,
        None,
        persistent_output_buffer_k=backing[0],
        persistent_output_buffer_v=backing[1],
        joint_strategy="rear",
        logical_n=RING * k_local,
        logical_l=0,
        is_causal=False,
        is_cross=q_local != k_local,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk, exp_approx_mode=False
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
        scale=0.0625,  # FP32-exact: the binding takes scale with noconvert
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        ),
        precision=VARIANTS[variant][0],
    )
    for chip in range(RING):
        expected = reference(q.chunk(RING, dim=2)[chip], k, v, scale=0.0625)
        assert l2_pct(per_chip(out[0])[chip], expected) < L2_PCT_BOUND[variant], f"chip {chip}"


def test_ring_joint_sdpa_precision_routing(ring_mesh):
    """Without precision, FP32 dest runs ACCURATE when the ring recipe has the call's features: bitwise the explicit
    ACCURATE call (sharded joint), and BFP8 Q/K/V (Q widened to BF16, outputs narrowed back), noncausal and causal."""
    mesh, semaphores, ccl_column = ring_mesh
    fp32 = fp32_dest_config(mesh)
    inputs, joints, backing, logical_n, kwargs, expected, _ = ring_case(mesh, "accurate", "joint_sharded")
    call = lambda **extra: run_ring(
        mesh, semaphores, ccl_column, inputs, joints, backing, logical_n=logical_n, **kwargs, **extra
    )
    routed, explicit = call(compute_kernel_config=fp32), call(precision=ttnn.SDPAPrecision.ACCURATE)
    for chip in range(RING):
        got = torch.cat([per_chip(routed[0])[chip], per_chip(routed[1])[chip]], dim=2)
        assert torch.equal(got, torch.cat([per_chip(explicit[0])[chip], per_chip(explicit[1])[chip]], dim=2))
        assert l2_pct(got, expected(chip)) < L2_PCT_BOUND["accurate"], f"chip {chip}"

    b, nh, nkv, q_local, k_local, d, q_chunk, k_chunk, _, _, grid = RING_CASES["gqa_batch2"]
    q = stored(randn(b, nh, RING * k_local, d, seed=20), ttnn.bfloat8_b)
    k, v = (stored(randn(b, nkv, RING * k_local, d, seed=s), ttnn.bfloat8_b) for s in (21, 22))
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    packed = [
        ttnn.from_torch(x, dtype=ttnn.bfloat8_b, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard)
        for x in (q, k, v)
    ]
    packed_backing = [
        ttnn.allocate_tensor_on_device(list(k.shape), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for _ in range(2)
    ]
    for causal in (False, True):
        out = run_ring(
            mesh,
            semaphores,
            ccl_column,
            packed,
            [None] * 3,
            packed_backing,
            grid=grid,
            q_chunk=q_chunk,
            k_chunk=k_chunk,
            logical_n=RING * k_local,
            logical_l=0,
            is_cross=False,
            is_causal=causal,
            compute_kernel_config=fp32,
        )
        assert out[0].dtype == ttnn.bfloat8_b
        for chip in range(RING):
            rows = RING * k_local
            mask = key_mask(k_local, rows, causal=True, q_offset=chip * k_local) if causal else None
            want = reference(q.chunk(RING, dim=2)[chip], k, v, mask)
            rounding = 1.5 * l2_pct(stored(want.bfloat16(), ttnn.bfloat8_b), want)
            bound = L2_PCT_BOUND["accurate"] + rounding
            assert l2_pct(per_chip(out[0])[chip], want) < bound, f"causal {causal} chip {chip}"


# Causal ring attention: batch, heads, kv_heads, local rows (Q = K), head_dim, q_chunk, k_chunk, SDPA grid. A balanced
# ring needs Q chunks dividing half the local rows; the K half may straddle a K chunk ("straddle": 480 / 192).
CAUSAL_CASES = {
    "q256_k512_d128": (1, 2, 2, 1024, 128, 256, 512, (4, 2)),
    "q512_k128": (1, 2, 2, 1024, 128, 512, 128, (2, 2)),
    "gqa_batch2": (2, 4, 2, 1024, 128, 128, 256, (8, 2)),
    "straddle_q96_k192_d64": (1, 2, 2, 960, 64, 96, 192, (4, 2)),
    # Several Q chunks per core: checkpointed state, and the balanced early half finishing before the last step.
    "multi_q_checkpoint": (1, 4, 4, 1024, 128, 128, 256, (2, 2)),
}


def causal_layout(x, balanced):
    """The global sequence as the ring holds it: device d gets chunk d (or, balanced, chunks d and 2R - 1 - d)."""
    if not balanced:
        return x, torch.arange(x.shape[2])
    chunks = torch.arange(x.shape[2]).chunk(2 * RING)
    order = torch.cat([torch.cat([chunks[d], chunks[2 * RING - 1 - d]]) for d in range(RING)])
    return x[:, :, order], order


@pytest.mark.parametrize("balanced", [False, True], ids=["causal", "balanced"])
@pytest.mark.parametrize("case", CAUSAL_CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_ring_joint_sdpa_recipe_causal(ring_mesh, variant, case, balanced):
    """is_causal (and is_balanced): the local step masks its diagonal, later shards are skipped (or, balanced, halved
    and the early Q half skipped), against an FP64 causal reference over the global sequence."""
    mesh, semaphores, ccl_column = ring_mesh
    b, nh, nkv, local, d, q_chunk, k_chunk, grid = CAUSAL_CASES[case]
    s = RING * local
    q, k, v = randn(b, nh, s, d, seed=21), randn(b, nkv, s, d, seed=22), randn(b, nkv, s, d, seed=23)
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    layout = [causal_layout(x, balanced) for x in (q, k, v)]
    inputs = precision_inputs(mesh, variant, [x for x, _ in layout], shard)
    if VARIANTS[variant][0] == ttnn.SDPAPrecision.FAST:
        # The reference takes the prepared values, back in sequence order.
        order = layout[0][1]
        q, k, v = (torch.cat(per_chip(x), dim=2)[:, :, torch.argsort(order)] for x in inputs)
    backing = [
        ttnn.allocate_tensor_on_device(list(k.shape), x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]
    out = run_ring(
        mesh,
        semaphores,
        ccl_column,
        inputs,
        [None] * 3,
        backing,
        grid=grid,
        q_chunk=q_chunk,
        k_chunk=k_chunk,
        logical_n=s,
        logical_l=0,
        is_cross=False,
        is_causal=True,
        is_balanced=balanced,
        precision=VARIANTS[variant][0],
    )
    expected = reference(q, k, v, key_mask(s, s, causal=True))
    rows = layout[0][1].chunk(RING)
    for chip in range(RING):
        got = per_chip(out[0])[chip]
        assert l2_pct(got, expected[:, :, rows[chip]]) < L2_PCT_BOUND[variant], f"chip {chip}"


@pytest.mark.parametrize("variant", ["standard", "accurate"])
def test_ring_joint_sdpa_recipe_rejects_sliding_window(ring_mesh, variant, expect_error):
    """The legacy FP32 ring kernel ignores sliding_window_size; the recipes reject it rather than ignore it."""
    mesh, semaphores, ccl_column = ring_mesh
    inputs, joints, backing, logical_n, kwargs, _, _ = ring_case(mesh, variant, "multi_q_checkpoint")
    with expect_error(RuntimeError, "do not support sliding_window_size"):
        run_ring(
            mesh,
            semaphores,
            ccl_column,
            inputs,
            joints,
            backing,
            logical_n=logical_n,
            is_causal=True,
            sliding_window_size=512,
            precision=VARIANTS[variant][0],
            **kwargs,
        )
