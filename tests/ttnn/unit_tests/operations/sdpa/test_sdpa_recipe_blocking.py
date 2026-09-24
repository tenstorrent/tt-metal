# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-selected blocking for the named SDPA recipes (docs/sdpa_precision.md, "Blocking").

Host tests exercise the chooser directly on representative DiT shapes (no device). Device tests run
every recipe variant on dense, joint, ring and exp ring attention with the chunks left to the op
(no program config / chunk size 0) and require the output to equal, bit for bit, an explicit call
with the chunks the op resolved.
"""

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, prepare

T = ttnn._ttnn.operations.transformer
KV_DTYPE = {"E_bf16": ttnn.bfloat16, "E_bfp8": ttnn.bfloat8_b, "E_bfp4": ttnn.bfloat4_b}
# Representative per-core CB budgets (bytes): a plain P150b device and the H3/ring pipeline's worker L1.
DEVICE_L1 = 1_440_000
PIPELINE_L1 = 1_344_544 - 191_360 + 191_360  # CBs end below the pipeline's live buffers
GRID = (11, 10)  # P150b compute grid
WORKER_GRID = (10, 10)  # ring: last column reserved for CCL


def precision_of(variant):
    return getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION"))


def choose(op, variant, heads, q_rows, k_rows, *, head_dim=128, grid=GRID, l1=DEVICE_L1, batch=1, **kwargs):
    return T._sdpa_recipe_blocking(
        op,
        precision_of(variant),
        KV_DTYPE.get(variant, ttnn.bfloat16),
        batch,
        heads,
        q_rows,
        k_rows,
        head_dim,
        ttnn.CoreCoord(*grid),
        l1,
        **kwargs,
    )


def supported(op, variant, q, k, head_dim=128):
    return (
        T._sdpa_recipe_geometry_rejection(op, precision_of(variant), KV_DTYPE.get(variant, ttnn.bfloat16), q, k, head_dim)
        is None
    )


# ------------------------------------------------------------------------------------------ host


@pytest.mark.parametrize("variant", VARIANTS)
def test_frozen_geometry_where_it_is_balanced(variant):
    # 10 heads x 8192 rows on 110 cores: Q256/K512 is as balanced as anything, so it is kept.
    q, k, *_ = choose("dense", variant, 10, 8192, 8192)
    assert (q, k) == (256, 512)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "name, op, heads, q_rows, k_rows, joint, head_dim",
    [
        ("flux1_1024_joint", "joint", 24, 4096, 4096, 512, 128),
        ("wan480_cross_1x2", "dense", 40, 16384, 512, 0, 128),
        ("wan480_self_1x1", "dense", 40, 32768, 32768, 0, 128),
        ("mochi_d128_joint", "joint", 24, 7920, 7920, 256, 128),
        ("ltx_audio_d64", "dense", 32, 1024, 1024, 0, 64),
        ("ideogram4_d256", "dense", 8, 4096, 4096, 0, 256),
        ("short_prompt", "joint", 8, 257, 257, 77, 128),
    ],
)
def test_dense_joint_choice(name, op, heads, q_rows, k_rows, joint, head_dim, variant):
    choice = choose(op, variant, heads, q_rows, k_rows, head_dim=head_dim, joint_q_rows=joint, joint_k_rows=joint)
    assert choice is not None, name
    q, k, gx, gy, cost, jobs, preferred, minimum = choice
    assert supported(op, variant, q, k, head_dim)
    assert minimum <= DEVICE_L1 and (gx, gy) == GRID
    # Never worse (in the model) than any candidate, and the qualified geometry when it is near-best.
    candidates = T._sdpa_recipe_blocking_candidates(
        op,
        precision_of(variant),
        KV_DTYPE.get(variant, ttnn.bfloat16),
        1,
        heads,
        q_rows,
        k_rows,
        head_dim,
        ttnn.CoreCoord(*GRID),
        DEVICE_L1,
        joint_q_rows=joint,
        joint_k_rows=joint,
    )
    best = min(c[4] for c in candidates)
    assert cost <= best * 1.03
    frozen = [c for c in candidates if c[:2] == (256, 512)]
    if frozen and frozen[0][4] <= best * 1.02:
        assert (q, k) == (256, 512)
    # A K chunk longer than the padded keys only adds padding.
    assert k <= max(512, math.ceil((k_rows + joint) / 32) * 32)


def test_small_q_chunks_avoided_when_balanced():
    # FLUX's tuned Q128 costs FAST ~1.37x per row; the chooser must not pick it at 4096+512 rows.
    q, *_ = choose("joint", "A", 24, 4096, 4096, joint_q_rows=512, joint_k_rows=512)
    assert q > 128


def test_l1_limits_chunks():
    # D256 at C/D: the frozen Q256/K512 does not fit; the choice must.
    for variant in ("C", "D"):
        preferred, minimum = T._sdpa_recipe_l1_bytes("dense", precision_of(variant), ttnn.bfloat16, 256, 512, 256)
        assert minimum > DEVICE_L1
        q, k, *_rest, l1 = choose("dense", variant, 8, 4096, 4096, head_dim=256)
        assert l1 <= DEVICE_L1


def test_explicit_chunks_are_honored():
    q, k, *_ = choose("dense", "B", 10, 8192, 8192, q_chunk_size=224, k_chunk_size=384)
    assert (q, k) == (224, 384)
    q, k, *_ = choose("dense", "B", 10, 8192, 8192, k_chunk_size=384)
    assert k == 384
    q, k, *_ = choose("ring", "A", 10, 4096, 4096, grid=WORKER_GRID, ring_size=8, q_chunk_size=288)
    assert (q, k) == (288, 512)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "name, heads, local, ring",
    [
        ("wan480_8x4", 10, 4096, 8),
        ("wan720_8x4", 10, 9472, 8),
        ("wan480_2x2", 20, 16384, 2),
        ("h3_5s", 14, 4768, 8),
        ("h3_10s", 14, 9216, 8),
        ("h3_15s", 14, 13632, 8),
    ],
)
def test_ring_choice(name, heads, local, ring, variant):
    for budget in (DEVICE_L1, PIPELINE_L1):
        choice = choose("ring", variant, heads, local, local, grid=WORKER_GRID, l1=budget, ring_size=ring)
        assert choice is not None, (name, budget)
        q, k, gx, gy, _cost, jobs, _preferred, minimum = choice
        assert supported("ring", variant, q, k) and (gx, gy) == WORKER_GRID and minimum <= budget
        assert jobs == math.ceil(heads * math.ceil(local / q) / (gx * gy))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "name, heads, local, ring",
    [("wan480_4x32", 10, 1024, 32), ("wan720_4x32", 10, 2368, 32), ("h3_4x32", 14, 1192, 32), ("1x2", 10, 4096, 2)],
)
def test_exp_ring_choice(name, heads, local, ring, variant):
    choice = choose("exp_ring", variant, heads, local, local, l1=PIPELINE_L1, ring_size=ring)
    assert choice is not None, name
    q, k, gx, gy, _cost, passes, _preferred, minimum = choice
    assert supported("exp_ring", variant, q, k) and k == 512 and minimum <= PIPELINE_L1
    cols = gx - 1
    chunks = math.ceil(local / q)
    assert gy == GRID[1] and 2 <= cols <= GRID[0] - 1 and chunks % cols == 0
    segments = heads * (chunks // cols)
    assert segments >= gy and passes == math.ceil(segments / gy) <= 3


def test_candidates_only_supported_geometry():
    for op in ("dense", "joint", "ring", "exp_ring"):
        for variant in VARIANTS:
            grid = WORKER_GRID if op == "ring" else GRID
            for q, k, *_ in T._sdpa_recipe_blocking_candidates(
                op,
                precision_of(variant),
                KV_DTYPE.get(variant, ttnn.bfloat16),
                1,
                10,
                2400,
                2400,
                128,
                ttnn.CoreCoord(*grid),
                PIPELINE_L1,
                ring_size=2,
            ):
                assert supported(op, variant, q, k), (op, variant, q, k)


# ---------------------------------------------------------------------------------------- device


@pytest.fixture(scope="module")
def blocking_mesh():
    if not is_blackhole() or ttnn.GetNumAvailableDevices() != 2:
        pytest.skip("requires two connected Blackholes")
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
    mesh = manager = None
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), worker_l1_size=1344544)
        mesh.enable_program_cache()
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        subdevice = ttnn.SubDeviceId(0)
        manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group([subdevice])
        semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
        yield mesh, subdevice, semaphores, grid
    finally:
        if mesh is not None:
            if manager is not None:
                mesh.reset_sub_device_stall_group()
                mesh.clear_loaded_sub_device_manager()
                mesh.remove_sub_device_manager(manager)
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def randn(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).bfloat16()


def upload(mesh, tensors, variant, mapper):
    inputs = [
        None if x is None else ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        for x in tensors
    ]
    if variant.startswith("E_"):
        # Prepare each Q/K/V triple (primary, then joint) independently.
        for start in range(0, len(inputs), 3):
            if inputs[start] is not None:
                inputs[start : start + 3] = prepare(inputs[start : start + 3], variant)
    return inputs


def to_host(tensor, mesh):
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def recipe_kwargs(variant):
    return {"precision": precision_of(variant), "inputs_prepared": variant.startswith("E_")}


def chunks(config):
    return config.q_chunk_size, config.k_chunk_size, config.compute_with_storage_grid_size.x


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "heads, q_rows, k_rows, joint, head_dim",
    [(4, 1000, 1500, 0, 128), (4, 900, 900, 77, 128), (2, 700, 700, 0, 64), (2, 512, 900, 0, 256)],
    ids=["dense", "joint", "dense_d64", "dense_d256"],
)
def test_dense_joint_auto_matches_explicit(blocking_mesh, heads, q_rows, k_rows, joint, head_dim, variant, record_property):
    mesh, _subdevice, _semaphores, _grid = blocking_mesh
    shapes = [(1, heads, n, head_dim) for n in (q_rows, k_rows, k_rows)]
    if joint:
        shapes += [(1, heads, joint, head_dim)] * 3
    inputs = upload(mesh, [randn(s, i) for i, s in enumerate(shapes)], variant, ttnn.ReplicateTensorToMesh(mesh))
    options = recipe_kwargs(variant)
    precision = options["precision"]
    if joint:
        auto_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=mesh.compute_with_storage_grid_size())
        resolved = T._sdpa_recipe_resolved_program_config(
            "joint", precision, inputs[0], inputs[1], joint_q=inputs[3], joint_k=inputs[4], program_config=auto_config
        )

        def run(config):
            out, joint_out = ttnn.transformer.joint_scaled_dot_product_attention(
                *inputs, joint_strategy="rear", program_config=config, **options
            )
            return torch.cat([to_host(out, mesh), to_host(joint_out, mesh)], dim=2)

        auto = run(auto_config)
    else:
        resolved = T._sdpa_recipe_resolved_program_config("dense", precision, inputs[0], inputs[1])

        def run(config):
            return to_host(ttnn.transformer.scaled_dot_product_attention(*inputs, is_causal=False, program_config=config, **options), mesh)

        auto = run(None)
    record_property("chosen", str(chunks(resolved)))
    assert resolved.q_chunk_size and resolved.k_chunk_size
    explicit = run(resolved)
    assert torch.isfinite(auto.float()).all()
    assert torch.equal(auto, explicit)


def test_partial_auto_and_legacy_rejection(blocking_mesh):
    mesh, *_ = blocking_mesh
    inputs = upload(mesh, [randn((1, 2, 640, 128), i) for i in range(3)], "B", ttnn.ReplicateTensorToMesh(mesh))
    grid = mesh.compute_with_storage_grid_size()
    config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, k_chunk_size=384)
    resolved = T._sdpa_recipe_resolved_program_config("dense", ttnn.SDPAPrecision.COMPENSATED, inputs[0], inputs[1], program_config=config)
    assert resolved.k_chunk_size == 384 and resolved.q_chunk_size > 0
    frozen = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=512)
    assert chunks(T._sdpa_recipe_resolved_program_config("dense", ttnn.SDPAPrecision.COMPENSATED, inputs[0], inputs[1], program_config=frozen)) == (256, 512, grid.x)
    with pytest.raises(RuntimeError, match="requires an explicit precision recipe"):
        ttnn.transformer.scaled_dot_product_attention(
            *inputs, is_causal=False, program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid)
        )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("heads, local, joint", [(10, 2048, 0), (4, 1536, 128)], ids=["self", "joint"])
def test_ring_auto_matches_explicit(blocking_mesh, heads, local, joint, variant, record_property):
    mesh, subdevice, semaphores, full = blocking_mesh
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    host = [randn((1, heads, 2 * local, 128), i) for i in range(3)]
    inputs = upload(mesh, host, variant, shard)
    joints = [None, None, None]
    if joint:
        joints = upload(mesh, [randn((1, heads, joint, 128), 10 + i) for i in range(3)], variant, ttnn.ReplicateTensorToMesh(mesh))
    backing = [
        ttnn.allocate_tensor_on_device([1, heads, 2 * local, 128], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]
    options = recipe_kwargs(variant)
    grid = (full.x - 1, full.y)
    auto_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid)
    resolved = T._sdpa_recipe_resolved_program_config(
        "ring", options["precision"], inputs[0], inputs[1], joint_q=joints[0], joint_k=joints[1], program_config=auto_config, ring_size=2
    )

    def run(config):
        out, joint_out, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            *inputs,
            *joints,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=2 * local,
            logical_l=joint,
            is_causal=False,
            dim=2,
            multi_device_global_semaphore=semaphores,
            num_links=1,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Linear,
            subdevice_id=subdevice,
            ccl_core_grid_offset=(full.x - 1, 0),
            use_column_major_ccl=True,
            program_config=config,
            **options,
        )
        result = to_host(out, mesh)
        return torch.cat([result, to_host(joint_out, mesh)], dim=2) if joint else result

    record_property("chosen", str(chunks(resolved)))
    auto = run(auto_config)
    explicit = run(resolved)
    assert torch.isfinite(auto.float()).all()
    assert torch.equal(auto, explicit)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("heads, local", [(10, 2368), (10, 1024), (20, 2560)], ids=["wan720_4x32", "wan480_4x32", "two_pass"])
def test_exp_ring_auto_matches_explicit(blocking_mesh, heads, local, variant, record_property):
    mesh, subdevice, semaphores, full = blocking_mesh
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    host = [randn((1, heads, 2 * local, 128), i) for i in range(3)]
    inputs = upload(mesh, host, variant, shard)
    backing = [
        ttnn.allocate_tensor_on_device([1, heads, 2 * local, 128], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]
    options = recipe_kwargs(variant)
    auto_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=full)
    resolved = T._sdpa_recipe_resolved_program_config(
        "exp_ring", options["precision"], inputs[0], inputs[1], program_config=auto_config, ring_size=2
    )
    record_property("chosen", str(chunks(resolved)))
    assert resolved.q_chunk_size and resolved.k_chunk_size == 512

    def run(config):
        out, _joint, _lse = ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
            *inputs,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=2 * local,
            program_config=config,
            dim=2,
            multi_device_global_semaphore=semaphores[:2],
            num_links=2,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            subdevice_id=subdevice,
            num_workers_per_link=full.y // 2,
            num_buffers_per_channel=16,
            **options,
        )
        return to_host(out, mesh)

    auto = run(auto_config)
    explicit = run(resolved)
    assert torch.isfinite(auto.float()).all()
    assert torch.equal(auto, explicit)
