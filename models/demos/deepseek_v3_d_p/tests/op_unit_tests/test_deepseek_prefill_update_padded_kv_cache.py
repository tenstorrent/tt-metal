# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device test for ttnn.experimental.deepseek_prefill.update_padded_kv_cache.

The op writes a per-chip input slab into a KV cache at a per-device start offset
derived from a single global token count `kv_actual_global`. When that count is
chunk-aligned every device writes at the same local offset; otherwise devices
around the boundary write at different offsets so new tokens overwrite the prior
cache's trailing pad cells before spilling into the next slab.
"""


from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    MlaKvCacheFormat,
    create_sequence_cache_mesh_composer,
    init_kvpe_cache,
    init_mla_kv_cache,
)

# MLA KVPE head dim (kv_lora_rank=512 + qk_rope_head_dim=64). The op is a pure page copy, so a
# gathered cache slot must byte-match the input we sent (read back through the same dtype
# encode/decode) -- the tests assert exact equality, not PCC.
KVPE_HEAD_DIM = 576


# (cache dtype, layout). bfloat8_b/bfloat4_b are block-float (TILE only); fp8_e4m3 is ROW_MAJOR only
# (Blackhole); bf16 covers the row-major page math in a lossless dtype. The tests assert bit-exact
# equality against the input read back, so no per-dtype tolerance is needed. Every case drives the
# per-element-tensor (metadata) path via the _update_kv helper below -- it is layout-agnostic, since the
# writer's page-row unit is a compile arg (TILE_HEIGHT for TILE, 1 for ROW_MAJOR).
DTYPE_LAYOUT_CASES = [
    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    (ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT),
]
DTYPE_LAYOUT_IDS = ["bfp8_tile", "bf16_rm", "fp8_rm"]


def _make_input(torch_chunk, dtype, layout, mesh_device, mesh_mapper):
    """Build a device input tensor. fp8_e4m3 cannot be constructed through the mesh-mapper path
    (it forces TILE), so build bf16 and typecast on device — typecast preserves ROW_MAJOR."""
    if dtype == ttnn.fp8_e4m3:
        tt = ttnn.from_torch(
            torch_chunk,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return ttnn.typecast(tt, ttnn.fp8_e4m3)
    return ttnn.from_torch(
        torch_chunk,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(
            (1, 1), marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="mesh-1x1"), id="1x1"
        ),
    ],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_scaled_fp8_packed_row(mesh_device):
    """The update op preserves the complete 656-byte mixed-format row as one FP8-typed stream."""
    if not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")

    head_dim = 656
    num_users, num_layers = 1, 2
    cache_tokens = 64
    chunk_tokens = 32
    sparse_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8,
        hf_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        mesh_device=mesh_device,
        seq_len=cache_tokens,
        mesh_shape=list(mesh_device.shape),
        sp_axis=0,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
    )
    cache = sparse_cache.storage

    torch.manual_seed(17)
    source = torch.randn(1, 1, chunk_tokens, head_dim, dtype=torch.bfloat16)
    tt_input = _make_input(
        source,
        ttnn.fp8_e4m3,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    expected = ttnn.to_torch(tt_input, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape(
        chunk_tokens, head_dim
    )

    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache,
        tt_input,
        slot_idx=0,
        layer_idx=1,
        num_layers=num_layers,
        kv_actual_global=0,
        cluster_axis=0,
    )
    result = ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape(
        num_users * num_layers, 1, cache_tokens, head_dim
    )

    assert torch.equal(result[1, 0, :chunk_tokens], expected)
    assert torch.count_nonzero(result[0].float()) == 0
    assert torch.count_nonzero(result[1, 0, chunk_tokens:].float()) == 0


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(
            (1, 1), marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="mesh-1x1"), id="1x1"
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUT_CASES, ids=DTYPE_LAYOUT_IDS)
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_single_device(mesh_device, dtype, layout):
    """Single-chip (1x1 mesh, sp=1) coverage that runs on a one-card box, so the op's per-dtype/layout
    copy can be validated without a 4-32 chip mesh. Uses the production init_kvpe_cache (ND-sharded);
    its DRAM-bank count is now device-derived, so it runs on harvested parts (e.g. 7 banks) too.

    sp=1 degenerates the per-chip offset math (boundary_chip=0, one slab == whole cache), so this is a
    plain per-slot KV fill: write a chunk-aligned slab per (user, layer) at offset 0.

    The op is a pure byte copy, so we assert bit-EXACT equality (not PCC) against the data we actually
    sent — i.e. the input read back, which has already been through the same dtype encode/decode. This
    isolates the op: any dtype quantization is identical on both sides, so a perfect copy is exactly
    equal. (Comparing against the original bf16 reference would instead measure the bfp8/fp8 round-trip.)"""
    if dtype == ttnn.fp8_e4m3 and not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")
    sp_axis = 0
    sp = mesh_device.shape[sp_axis]  # == 1
    tile = ttnn.TILE_SIZE

    num_users, num_layers = 2, 2
    chunk_local = 4 * tile  # 128 tokens/dev
    new_isl_global = chunk_local * sp
    cache_tokens = 512
    cache_global = cache_tokens * sp

    torch.manual_seed(0)
    sent = {
        (u, l): torch.randn(new_isl_global, KVPE_HEAD_DIM, dtype=torch.bfloat16)
        for u in range(num_users)
        for l in range(num_layers)
    }

    kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=KVPE_HEAD_DIM,
        mesh_device=mesh_device,
        seq_len=cache_global,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_users * num_layers,
        dtype=dtype,
        layout=layout,
    )

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2  # at sp=1 this keeps the whole chunk on the one chip

    mesh_device.enable_program_cache()

    # Capture each input read back to host (decoded the same way the cache will be) as the exact
    # reference for the bytes the op should copy.
    expected = {}
    for u in range(num_users):
        for l in range(num_layers):
            tt_input = _make_input(
                sent[(u, l)].reshape(1, 1, new_isl_global, KVPE_HEAD_DIM),
                dtype,
                layout,
                mesh_device,
                ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims),
            )
            expected[(u, l)] = (
                ttnn.to_torch(tt_input, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
                .to(torch.bfloat16)
                .reshape(new_isl_global, KVPE_HEAD_DIM)
            )
            _update_kv(
                kv_cache,
                tt_input,
                slot_idx=u,
                kv_actual_global=0,
                layer_idx=l,
                num_layers=num_layers,
                cluster_axis=sp_axis,
                layout=layout,
                mesh_device=mesh_device,
            )

    ttnn.synchronize_device(mesh_device)

    # NOTE: the op's program-cache-reuse contract (one program per layer) is asserted by the
    # multi-device tests; skipped here since the single-device from_torch/typecast path spins up
    # auxiliary device programs (tilize / layout conversion) that would inflate the count.
    cache_host = ttnn.to_torch(kv_cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(torch.bfloat16)
    # ConcatMeshToTensor on a 1x1 mesh stacks the single shard on dim 0; recover [slots, 1, seq, head].
    cache_host = cache_host.reshape(num_users * num_layers, 1, cache_tokens, KVPE_HEAD_DIM)

    for u in range(num_users):
        for l in range(num_layers):
            batch_idx = u * num_layers + l
            written = cache_host[batch_idx, 0, :chunk_local, :]
            assert torch.equal(written, expected[(u, l)]), (
                f"[{dtype}] user {u} layer {l}: cache slot does not byte-match the input sent "
                f"(max abs diff {(written.float() - expected[(u, l)].float()).abs().max().item()})"
            )
            logger.info(f"  [{dtype}] user {u} layer {l}: exact match")


def _make_scalar_tensor(mesh_device, value):
    """Build one 1-element uint32 DRAM tensor ([1,1,1,1], ROW_MAJOR), replicated across the mesh.
    The op's per-element-tensor (traceable) path reads element [0] of one such tensor for slot_idx
    and another for kv_actual_global on-device (no host scalars)."""
    payload = torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1)
    return ttnn.from_torch(
        payload,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _make_meta_tensors(mesh_device, kv_actual_global, slot_idx):
    """Build the two per-element tensors (slot_idx, kv_actual_global) the traceable path consumes."""
    return _make_scalar_tensor(mesh_device, slot_idx), _make_scalar_tensor(mesh_device, kv_actual_global)


def _update_kv(
    kv_cache, tt_input, *, slot_idx, kv_actual_global, layer_idx, num_layers, cluster_axis, layout, mesh_device
):
    """Drive update_padded_kv_cache through the per-element-tensor (metadata) path, which is the path
    this PR adds and the one traced prefill uses. It is layout-agnostic (the writer's page-row unit is
    a compile arg: TILE_HEIGHT for TILE, 1 for ROW_MAJOR), so every dtype/layout case runs on it. The
    scalar path is covered against it by test_update_padded_kv_cache_metadata_matches_scalar."""
    del layout  # both layouts drive the same (metadata) path; kept for call-site readability
    slot_t, kv_t = _make_meta_tensors(mesh_device, kv_actual_global=kv_actual_global, slot_idx=slot_idx)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        kv_cache, tt_input, slot_t, kv_t, layer_idx=layer_idx, num_layers=num_layers, cluster_axis=cluster_axis
    )
    ttnn.deallocate(slot_t)
    ttnn.deallocate(kv_t)


def _full_mesh_update_cases():
    num_devices = detect_num_devices()
    mesh = {4: (2, 2), 8: (2, 4), 32: (8, 4)}.get(num_devices)
    if mesh is None:
        return [
            pytest.param(
                (1, max(num_devices, 1)),
                marks=pytest.mark.skip(reason=f"no supported 2D full-mesh case for {num_devices} devices"),
                id=f"unsupported-{num_devices}dev",
            )
        ]
    return [pytest.param(mesh, id=f"{mesh[0]}x{mesh[1]}")]


def _stamp_full_mesh_sequence_topology(tensor, mesh_device):
    full_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    coords = [
        ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())]) for coord in ttnn.MeshCoordinateRange(full_shape)
    ]
    tensor.update_tensor_topology(
        ttnn.TensorTopology(full_shape, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
    )


@pytest.mark.parametrize("mesh_device", _full_mesh_update_cases(), indirect=True)
@pytest.mark.parametrize("use_metadata_tensor", [False, True], ids=["scalar", "metadata"])
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_full_mesh_rotated(mesh_device, use_metadata_tensor):
    """A rotated write preserves canonical row-major sequence order across the complete 2D mesh."""
    if not is_blackhole():
        pytest.skip("full-mesh cache update coverage is currently Blackhole-only")

    mesh_factor = mesh_device.get_num_devices()
    tile = ttnn.TILE_SIZE
    chunk_local = 2 * tile
    chunk_global = mesh_factor * chunk_local
    cache_tokens_local = 4 * chunk_local
    cache_global = mesh_factor * cache_tokens_local
    actual_start = chunk_local + tile  # chip 1, one tile into its local slab

    cache = init_kvpe_cache(
        kvpe_cache_head_dim=KVPE_HEAD_DIM,
        mesh_device=mesh_device,
        seq_len=cache_global,
        mesh_shape=list(mesh_device.shape),
        sp_axis=0,
        num_kvpe_cache_layers=1,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        full_mesh=True,
    )
    torch.manual_seed(31)
    source = torch.randn(1, 1, chunk_global, KVPE_HEAD_DIM, dtype=torch.bfloat16)
    tt_input = _make_input(
        source,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    _stamp_full_mesh_sequence_topology(tt_input, mesh_device)

    if use_metadata_tensor:
        slot_t, kv_t = _make_meta_tensors(mesh_device, kv_actual_global=actual_start, slot_idx=0)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache, tt_input, slot_t, kv_t, layer_idx=0, num_layers=1, cluster_axis=None
        )
        ttnn.deallocate(slot_t)
        ttnn.deallocate(kv_t)
    else:
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            tt_input,
            slot_idx=0,
            layer_idx=0,
            num_layers=1,
            kv_actual_global=actual_start,
            cluster_axis=None,
        )
    ttnn.synchronize_device(mesh_device)

    composer = create_sequence_cache_mesh_composer(mesh_device, full_mesh=True)
    input_host = ttnn.to_torch(tt_input, mesh_composer=composer).to(torch.bfloat16)[0, 0]
    expected = torch.zeros(cache_global, KVPE_HEAD_DIM, dtype=torch.bfloat16)
    boundary_slab = actual_start // chunk_global
    boundary_chip = (actual_start // chunk_local) % mesh_factor
    boundary_offset = actual_start % chunk_local
    for chip in range(mesh_factor):
        if chip < boundary_chip:
            local_start = (boundary_slab + 1) * chunk_local
        elif chip == boundary_chip:
            local_start = boundary_slab * chunk_local + boundary_offset
        else:
            local_start = boundary_slab * chunk_local
        cache_start = chip * cache_tokens_local + local_start
        input_start = chip * chunk_local
        expected[cache_start : cache_start + chunk_local] = input_host[input_start : input_start + chunk_local]

    cache_host = ttnn.to_torch(cache, mesh_composer=composer).to(torch.bfloat16)[0, 0]
    assert torch.equal(cache_host, expected)


@pytest.mark.parametrize("mesh_device", _full_mesh_update_cases(), indirect=True)
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_full_mesh_rejects_axis_topology(mesh_device, expect_error):
    """cluster_axis=None rejects the legacy cache topology instead of silently mis-addressing it."""
    if not is_blackhole():
        pytest.skip("full-mesh cache update coverage is currently Blackhole-only")

    mesh_factor = mesh_device.get_num_devices()
    chunk_local = 2 * ttnn.TILE_SIZE
    input_global = mesh_factor * chunk_local
    input_tensor = _make_input(
        torch.zeros(1, 1, input_global, KVPE_HEAD_DIM, dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    _stamp_full_mesh_sequence_topology(input_tensor, mesh_device)
    axis_cache = init_kvpe_cache(
        kvpe_cache_head_dim=KVPE_HEAD_DIM,
        mesh_device=mesh_device,
        seq_len=input_global * 4,
        mesh_shape=list(mesh_device.shape),
        sp_axis=0,
        num_kvpe_cache_layers=1,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    with expect_error(RuntimeError, "cluster_axis=None requires cache and input"):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            axis_cache,
            input_tensor,
            slot_idx=0,
            layer_idx=0,
            num_layers=1,
            kv_actual_global=0,
            cluster_axis=None,
        )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 4),
            torus_x_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="ring"),
            id="torus-x-1x4",
        ),
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-2x4",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUT_CASES, ids=DTYPE_LAYOUT_IDS)
@pytest.mark.parametrize(
    "config_name, num_users, num_layers, new_isl_tiles_per_dev, cache_tokens_per_dev",
    [
        ("small", 2, 3, 4, 512),  # small: 2 users x 3 layers, 4-tile chunk/dev, ~1k cache
        ("repr", 2, 3, 20, 6400),  # representative: 5k new isl + 50k cache on 8x4 (per-dev scaled)
    ],
    ids=["small", "repr"],
)
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_single_iteration_prefill(
    mesh_device,
    device_params,
    config_name,
    num_users,
    num_layers,
    new_isl_tiles_per_dev,
    cache_tokens_per_dev,
    dtype,
    layout,
    is_ci_env,
    is_ci_v2_env,
):
    """Single-iteration (non-padded) prefill: write one chunk-aligned slab per (user, layer)
    at offset 0, gather the whole cache, and assert each slot's valid data byte-matches what was sent.

    The op is a pure copy, so the reference is the input read back (already through the same dtype
    encode/decode), and the check is bit-EXACT equality -- not PCC against the bf16 source, which would
    only measure the bfp8/fp8 round-trip rather than the op."""
    if dtype == ttnn.fp8_e4m3 and not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")
    if is_ci_env or is_ci_v2_env:
        pytest.skip("CI runs only the small padded_partial case (multi-iteration); this is a subset of it")
    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    tile = ttnn.TILE_SIZE

    chunk_local = new_isl_tiles_per_dev * tile  # per-device new tokens
    new_isl_global = chunk_local * sp  # one global chunk = slab 0
    cache_global = cache_tokens_per_dev * sp

    torch.manual_seed(0)
    # Reference new tokens per (user, layer), in natural global order.
    sent = {
        (u, l): torch.randn(new_isl_global, KVPE_HEAD_DIM, dtype=torch.bfloat16)
        for u in range(num_users)
        for l in range(num_layers)
    }

    kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=KVPE_HEAD_DIM,
        mesh_device=mesh_device,
        seq_len=cache_global,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_users * num_layers,
        dtype=dtype,
        layout=layout,
    )

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2  # split the chunk across sp devices

    # Composer to read a tensor back in natural order: concat sp shards on the seq dim, take one
    # tp-replicated copy. Used both for the cache gather and to recover each input we sent.
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape)

    mesh_device.enable_program_cache()
    # init_kvpe_cache zeros the cache on device (DRAMZeroFill), which registers its own one-time
    # program; snapshot the post-init count so the assert below measures only what the OP adds.
    entries_after_init = mesh_device.num_program_cache_entries()

    expected = {}
    for u in range(num_users):
        for l in range(num_layers):
            tt_input = _make_input(
                sent[(u, l)].reshape(1, 1, new_isl_global, KVPE_HEAD_DIM),
                dtype,
                layout,
                mesh_device,
                ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims),
            )
            # Exact reference: the input read back in natural order (same encode/decode as the cache).
            expected[(u, l)] = ttnn.to_torch(tt_input, mesh_composer=composer).to(torch.bfloat16)[0, 0]
            _update_kv(
                kv_cache,
                tt_input,
                slot_idx=u,
                kv_actual_global=0,
                layer_idx=l,
                num_layers=num_layers,
                cluster_axis=sp_axis,
                layout=layout,
                mesh_device=mesh_device,
            )

    ttnn.synchronize_device(mesh_device)

    # slot_idx and kv_actual_global are device tensors (not in the program hash) and layer_idx is
    # hashed, so exactly one cached program per layer is reused across all users — the op adds
    # num_layers entries on top of init's fixed overhead (entries_after_init). Skip for fp8: its
    # tensors are built via ttnn.typecast, which adds its own cached programs and pollutes the global
    # count. The op's per-layer reuse is already covered by the bf16/bfp8 cases.
    if dtype != ttnn.fp8_e4m3:
        assert mesh_device.num_program_cache_entries() == entries_after_init + num_layers, (
            f"op must reuse one cached program per layer across users; expected "
            f"{entries_after_init + num_layers} entries, got {mesh_device.num_program_cache_entries()}"
        )

    # Gather the cache the same way (concat sp shards on seq, one tp copy).
    cache_host = ttnn.to_torch(kv_cache, mesh_composer=composer).to(torch.bfloat16)
    cache_host = cache_host[:, :1, :, :]  # [users*layers, 1, cache_global, KVPE_HEAD_DIM]

    for u in range(num_users):
        for l in range(num_layers):
            batch_idx = u * num_layers + l
            # Each chip's slab-0 prefix [0:chunk_local] holds its share of the chunk;
            # concat across chips to rebuild natural global order.
            written = torch.cat(
                [
                    cache_host[batch_idx, 0, c * cache_tokens_per_dev : c * cache_tokens_per_dev + chunk_local, :]
                    for c in range(sp)
                ],
                dim=0,
            )
            assert torch.equal(written, expected[(u, l)]), (
                f"user {u} layer {l}: cache valid data does not byte-match the input sent "
                f"(max abs diff {(written.float() - expected[(u, l)].float()).abs().max().item()})"
            )
            logger.info(f"  user {u} layer {l}: exact match")

    logger.info(f"program cache entries: {mesh_device.num_program_cache_entries()}")


def _rotated_chip_positions(kv_actual, sp, chunk_local):
    """Global token position carried by each chip-local input row after the op's KV-pad-aware
    rotation, mirroring the writer kernel's update_idxt math. positions[c][r] is the global
    position chip c's r-th input row will land at; rows whose position is >= the valid frontier
    are server pad. Slab-aware (handles kv_actual spanning multiple slabs)."""
    C = chunk_local
    chunk_global = sp * C
    boundary_slab = kv_actual // chunk_global
    boundary_chip = (kv_actual // C) % sp
    boundary_offset = kv_actual % C
    positions = [[0] * C for _ in range(sp)]
    for c in range(sp):
        if c < boundary_chip:
            update_idxt = (boundary_slab + 1) * C
        elif c == boundary_chip:
            update_idxt = boundary_slab * C + boundary_offset
        else:
            update_idxt = boundary_slab * C
        for r in range(C):
            lr = update_idxt + r  # local cache row this input row lands in
            positions[c][r] = (lr // C) * chunk_global + c * C + (lr % C)
    return positions


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(
            (2, 2), marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"), id="2x2"
        ),
        pytest.param(
            (2, 4), marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"), id="2x4"
        ),
        pytest.param(
            (8, 4), marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"), id="8x4"
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUT_CASES, ids=DTYPE_LAYOUT_IDS)
@pytest.mark.parametrize(
    "config_name, num_users, num_layers, new_isl_tiles_per_dev, cache_tokens_per_dev",
    [
        ("small", 2, 3, 4, 512),  # small
        ("repr", 2, 3, 20, 6400),  # representative
    ],
    ids=["small", "repr"],
)
@pytest.mark.parametrize("scenario", ["non_padded", "padded_partial"], ids=["non_padded", "padded_partial"])
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_multi_iteration_prefill(
    mesh_device,
    config_name,
    num_users,
    num_layers,
    new_isl_tiles_per_dev,
    cache_tokens_per_dev,
    scenario,
    dtype,
    layout,
    is_ci_env,
    is_ci_v2_env,
):
    """Multi-iteration prefill, multi-user / multi-layer.

    - non_padded: two full-chunk iterations (chunk-aligned, no rotation).
    - padded_partial: three iterations exercising whole-tile, non-zero pad offsets (the general
      case -- a whole-device pad boundary is just the offset == 0 special case). iter 0 fills the
      last device by one tile; iter 1 completes the last device (its write straddles a slab),
      fills the next device, and leaves device 1 partially filled by one tile; iter 2 is a full
      chunk that enters at device 1's tile offset (a second straddle).

    Each iteration sends server-rotated input; afterwards the cache is gathered, natural order is
    rebuilt, and every (user, layer) slot's valid prefix is checked for bit-exact equality against
    the inputs sent (read back through the same dtype encode/decode).
    """
    if dtype == ttnn.fp8_e4m3 and not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")
    if (is_ci_env or is_ci_v2_env) and not (config_name == "small" and scenario == "padded_partial"):
        pytest.skip("CI runs only the small padded_partial case; the others are subsets of it")
    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    tile = ttnn.TILE_SIZE
    C = new_isl_tiles_per_dev * tile  # per-device chunk (physical, fixed every iter)
    chunk_global = C * sp

    if scenario == "non_padded":
        new_actual_isls = [chunk_global, chunk_global]
    else:  # padded_partial: whole-tile boundary_offset != 0; boundary chip writes straddle slabs
        new_actual_isls = [(sp - 1) * C + tile, 2 * C, sp * C]
    # Each iteration writes exactly one chunk_global-token chunk, so the valid frontier can advance by
    # at most chunk_global per iteration. A larger advance would claim tokens valid that were never
    # written, leaving an unwritten hole -- e.g. a scenario tuned for sp>=2 (2*C) run at sp=1, where
    # chunk_global == C is smallest.
    assert all(
        isl <= chunk_global for isl in new_actual_isls
    ), f"each new_isl must be <= chunk_global ({chunk_global}); got {new_actual_isls}"
    cum_total = sum(new_actual_isls)
    cache_global = cache_tokens_per_dev * sp
    assert cum_total <= cache_global, f"valid tokens ({cum_total}) must fit the cache ({cache_global})"

    logger.info(
        f"sp={sp} chunk_local={C} chunk_global={chunk_global} cache_global={cache_global}; "
        f"new_isl per iter={new_actual_isls} (cum_total={cum_total})"
    )

    torch.manual_seed(0)
    sent = {
        (u, l): torch.randn(cum_total, KVPE_HEAD_DIM, dtype=torch.bfloat16)
        for u in range(num_users)
        for l in range(num_layers)
    }

    kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=KVPE_HEAD_DIM,
        mesh_device=mesh_device,
        seq_len=cache_global,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_users * num_layers,
        dtype=dtype,
        layout=layout,
    )

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2

    # Composer to read tensors back: concat sp shards on the seq dim (chip-concat order), one tp copy.
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape)

    mesh_device.enable_program_cache()
    # init_kvpe_cache zeros the cache on device (DRAMZeroFill), which registers its own one-time
    # program; snapshot the post-init count so the assert below measures only what the OP adds.
    entries_after_init = mesh_device.num_program_cache_entries()

    # Build the exact reference incrementally from the inputs we actually send (read back through the
    # same dtype encode/decode), placed at their natural global positions. The op is a pure copy, so
    # the gathered cache must byte-match this -- checked with exact equality, not PCC.
    expected = {
        (u, l): torch.zeros(cum_total, KVPE_HEAD_DIM, dtype=torch.bfloat16)
        for u in range(num_users)
        for l in range(num_layers)
    }

    kv_actual = 0
    for it, new_actual_isl in enumerate(new_actual_isls):
        positions = _rotated_chip_positions(kv_actual, sp, C)
        flat = [positions[c][r] for c in range(sp) for r in range(C)]  # chip-concat order
        valid_end = kv_actual + new_actual_isl
        logger.info(
            f"  iter {it}: kv_actual={kv_actual} new_isl={new_actual_isl} valid_end={valid_end} "
            f"pad_boundary_chip={(valid_end // C) % sp}"
        )
        gather_idx = torch.tensor([min(g, cum_total - 1) for g in flat], dtype=torch.long)
        pad_mask = torch.tensor([g >= valid_end for g in flat])
        valid_rows = (~pad_mask).nonzero(as_tuple=True)[0]
        flat_t = torch.tensor(flat, dtype=torch.long)
        for u in range(num_users):
            for l in range(num_layers):
                chunk = sent[(u, l)][gather_idx].clone()  # [chunk_global, KVPE_HEAD_DIM]
                chunk[pad_mask] = 0.0  # server pad rows
                tt_input = _make_input(
                    chunk.reshape(1, 1, chunk_global, KVPE_HEAD_DIM),
                    dtype,
                    layout,
                    mesh_device,
                    ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims),
                )
                # Read the input back in chip-concat order (row r carries global position flat[r]) and
                # scatter its valid rows into the natural-order reference.
                inp_rb = ttnn.to_torch(tt_input, mesh_composer=composer).to(torch.bfloat16)[0, 0]
                expected[(u, l)][flat_t[valid_rows]] = inp_rb[valid_rows]
                _update_kv(
                    kv_cache,
                    tt_input,
                    slot_idx=u,
                    kv_actual_global=kv_actual,
                    layer_idx=l,
                    num_layers=num_layers,
                    cluster_axis=sp_axis,
                    layout=layout,
                    mesh_device=mesh_device,
                )
        kv_actual = valid_end

    ttnn.synchronize_device(mesh_device)

    # kv_actual_global and slot_idx are device tensors (not in the program hash) and layer_idx is
    # hashed, so exactly one cached program per layer is reused across all iterations and users — the
    # op adds num_layers entries on top of init's fixed overhead (entries_after_init). Skip for fp8:
    # its tensors are built via ttnn.typecast, which adds its own cached programs and pollutes the
    # global count. Per-layer reuse is covered by the bf16/bfp8 cases.
    if dtype != ttnn.fp8_e4m3:
        assert mesh_device.num_program_cache_entries() == entries_after_init + num_layers, (
            f"op must reuse one cached program per layer across iterations/users; expected "
            f"{entries_after_init + num_layers} entries, got {mesh_device.num_program_cache_entries()}"
        )

    cache_host = ttnn.to_torch(kv_cache, mesh_composer=composer).to(torch.bfloat16)[
        :, :1, :, :
    ]  # [users*layers, 1, cache_global, KVPE_HEAD_DIM]

    # cache cell (chip c, local row lr) holds global position (lr//C)*chunk_global + c*C + (lr%C);
    # invert for every valid position to rebuild natural order from the chip-concatenated gather.
    p = torch.arange(cum_total)
    chip = (p % chunk_global) // C
    local_row = (p // chunk_global) * C + (p % C)
    dim2_idx = chip * cache_tokens_per_dev + local_row

    for u in range(num_users):
        for l in range(num_layers):
            batch_idx = u * num_layers + l
            recon = cache_host[batch_idx, 0, dim2_idx, :]  # [cum_total, KVPE_HEAD_DIM]
            assert torch.equal(recon, expected[(u, l)]), (
                f"user {u} layer {l}: cache valid prefix does not byte-match the inputs sent "
                f"(max abs diff {(recon.float() - expected[(u, l)].float()).abs().max().item()})"
            )
            logger.info(f"  user {u} layer {l}: exact match")

    logger.info(f"program cache entries: {mesh_device.num_program_cache_entries()}")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUT_CASES, ids=DTYPE_LAYOUT_IDS)
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_metadata_matches_scalar(mesh_device, dtype, layout):
    """The per-element-tensor (traceable) path and the scalar path must produce bit-identical caches.

    Drives the traceable path from two 1-element uint32 DRAM tensors (slot_idx, kv_actual_global)
    that the writer reads on-device, and compares the written cache against the same write done via
    the original scalar signature. Exact equality (the op is a pure copy and both paths run the
    identical writer math) over a couple of (slot, start) chunks."""
    if dtype == ttnn.fp8_e4m3 and not is_blackhole():
        pytest.skip("FP8_E4M3 is Blackhole-only")

    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    tile = ttnn.TILE_SIZE

    num_users, num_layers = 2, 2
    new_isl_tiles_per_dev = 4
    cache_tokens_per_dev = 512
    chunk_local = new_isl_tiles_per_dev * tile  # per-device new tokens
    chunk_global = chunk_local * sp  # one global chunk
    cache_global = cache_tokens_per_dev * sp

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2  # split the chunk across sp devices
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape)

    mesh_device.enable_program_cache()

    # (slot, layer, actual_start). Fix layer_idx=0 across cases so every metadata call hits — and must
    # REUSE — the same cached program; vary (slot, start): two chunk-aligned starts plus one NON-slab-
    # aligned start (chunk_global + one tile, so boundary chip 0 writes at a whole-tile offset while the
    # other chips stay on the slab base — the boundary-straddling write otherwise only exercised on the
    # scalar path). All fit the cache (4 chunks per slot). Program reuse is asserted after the loop.
    cases = [(0, 0, 0), (1, 0, chunk_global), (0, 0, chunk_global + tile)]
    entries_after_first = None

    torch.manual_seed(0)
    for slot_id, layer_idx, actual_start in cases:
        # 1) The two per-element metadata tensors the traceable path reads on-device.
        slot_t, kv_t = _make_meta_tensors(mesh_device, kv_actual_global=actual_start, slot_idx=slot_id)

        # 2) One shared KV input slab, and two identical freshly-zeroed caches.
        slab = torch.randn(chunk_global, KVPE_HEAD_DIM, dtype=torch.bfloat16).reshape(1, 1, chunk_global, KVPE_HEAD_DIM)
        tt_input = _make_input(
            slab,
            dtype,
            layout,
            mesh_device,
            ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims),
        )

        def _fresh_cache():
            return init_kvpe_cache(
                kvpe_cache_head_dim=KVPE_HEAD_DIM,
                mesh_device=mesh_device,
                seq_len=cache_global,
                mesh_shape=list(mesh_device.shape),
                sp_axis=sp_axis,
                num_kvpe_cache_layers=num_users * num_layers,
                dtype=dtype,
                layout=layout,
            )

        cache_meta = _fresh_cache()
        cache_scalar = _fresh_cache()

        # 3) Per-element-tensor path: slot_idx/kv_actual_global read on-device from slot_t/kv_t.
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache_meta,
            tt_input,
            slot_t,
            kv_t,
            layer_idx=layer_idx,
            num_layers=num_layers,
            cluster_axis=sp_axis,
        )
        # 4) Scalar path: original signature with host scalars.
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache_scalar,
            tt_input,
            slot_idx=slot_id,
            layer_idx=layer_idx,
            num_layers=num_layers,
            kv_actual_global=actual_start,
            cluster_axis=sp_axis,
        )
        ttnn.synchronize_device(mesh_device)

        # Compare only the VALID WRITTEN region (the chunk just written), not the whole cache: the
        # unwritten / aligned ROW_MAJOR page-padding cells are uninitialized and read back as
        # dtype-dependent garbage (e.g. NaN for bf16) that differs between the two separate cache
        # allocations — exactly what the existing single/multi-iteration tests sidestep. Both paths
        # run the identical writer with the same slot/offset, so the written region must byte-match.
        batch_idx = slot_id * num_layers + layer_idx
        # Per-chip write offset (tokens): chips before the boundary chip advance a full slab, the boundary
        # chip advances by its whole-tile pad offset, chips after it stay on the slab base. Chunk-aligned
        # starts make this uniform; the non-slab case makes it differ per chip (boundary chip straddles).
        # Mirrors the writer's update_idxt so we extract exactly the cells that were written.
        boundary_slab = actual_start // chunk_global
        boundary_chip = (actual_start // chunk_local) % sp
        boundary_offset = actual_start % chunk_local

        def _local_start_row(c):
            if c < boundary_chip:
                return (boundary_slab + 1) * chunk_local
            if c == boundary_chip:
                return boundary_slab * chunk_local + boundary_offset
            return boundary_slab * chunk_local

        def _written_slab(cache):
            host = ttnn.to_torch(cache, mesh_composer=composer).to(torch.float32)[:, :1, :, :]
            return torch.cat(
                [
                    host[
                        batch_idx,
                        0,
                        c * cache_tokens_per_dev
                        + _local_start_row(c) : c * cache_tokens_per_dev
                        + _local_start_row(c)
                        + chunk_local,
                        :,
                    ]
                    for c in range(sp)
                ],
                dim=0,
            )

        meta_slab = _written_slab(cache_meta)
        scalar_slab = _written_slab(cache_scalar)
        assert torch.equal(meta_slab, scalar_slab), (
            f"slot {slot_id} layer {layer_idx} start {actual_start}: per-element-tensor-path written slab differs "
            f"from scalar-path (max abs diff {(meta_slab - scalar_slab).abs().max().item()})"
        )
        logger.success(
            f"[{dtype}] slot {slot_id} layer {layer_idx} start {actual_start}: "
            f"per-element-tensor path == scalar path (bit-exact)"
        )
        # After the first case both programs (metadata + scalar, at layer 0) are compiled; capture the
        # count so we can assert no further growth across the remaining (varied slot/start, same layer)
        # cases — including the non-slab-aligned one.
        if entries_after_first is None:
            entries_after_first = mesh_device.num_program_cache_entries()
        ttnn.deallocate(slot_t)
        ttnn.deallocate(kv_t)
        ttnn.deallocate(tt_input)

    # The PR's core property: slot_idx / kv_actual_global are read on-device (never hashed), so successive
    # metadata-tensor calls at the same layer — including the non-slab-aligned one — reuse the one cached
    # metadata program (and the one scalar-path program) instead of recompiling per chunk.
    assert mesh_device.num_program_cache_entries() == entries_after_first, (
        f"program cache grew across metadata-tensor calls at a fixed layer — the metadata and scalar "
        f"programs should each compile once and be reused: {entries_after_first} -> "
        f"{mesh_device.num_program_cache_entries()}"
    )
    logger.info(f"program cache stable at {entries_after_first} entries across {len(cases)} metadata-path chunks")


def _natural_from_cache(cache_slot_rows, sp, chunk_local, cache_tokens_per_dev, chunk_global):
    """Un-rotate one cache slot read back in chip-concat order into natural token order.

    ``cache_slot_rows`` is [sp * cache_tokens_per_dev, head_dim] (the composer concatenates each chip's
    slab on the seq dim), and chip c's local row lr carries global position
    ``(lr // chunk_local) * chunk_global + c * chunk_local + (lr % chunk_local)`` -- the block-cyclic
    layout the writer targets."""
    nat = torch.empty_like(cache_slot_rows)
    for c in range(sp):
        for lr in range(cache_tokens_per_dev):
            pos = (lr // chunk_local) * chunk_global + c * chunk_local + (lr % chunk_local)
            nat[pos] = cache_slot_rows[c * cache_tokens_per_dev + lr]
    return nat


@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4)], ids=["2x2", "2x4"], indirect=True)
@pytest.mark.parametrize(
    "dtype, layout",
    [(ttnn.bfloat8_b, ttnn.TILE_LAYOUT), (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)],
    ids=["bfp8_tile", "bf16_rm"],
)
@pytest.mark.parametrize("path", ["metadata", "scalar"], ids=["metadata", "scalar"])
@pytest.mark.parametrize(
    "case", ["mid_chunk", "tail_overflow", "tail_chip_jump"], ids=["mid_chunk", "tail_overflow", "tail_chip_jump"]
)
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_valid_global_clamp(mesh_device, dtype, layout, path, case, expect_error):
    """`valid_global` keeps a chunk's PAD tail out of the cache.

    Three shapes: `mid_chunk` ends on a non-tile-aligned boundary (its 32-token block is written, the rest
    is not); `tail_overflow` pads one tile past the cache while its real tokens fit (the case the op
    rejects unclamped, asserted here too); `tail_chip_jump` has boundary_chip > 0, so the pre-boundary
    chips jump a whole slab past the cache end and must write NOTHING.

    Checked three ways and bit-exact against a sentinel-filled cache and poisoned pad rows: real rows
    match what was sent, rows past the real end still hold the sentinel, and the poison appears nowhere."""
    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    tile = ttnn.TILE_SIZE

    chunk_local = 4 * tile
    chunk_global = chunk_local * sp
    slabs = 2
    cache_tokens_per_dev = slabs * chunk_local
    cache_global = cache_tokens_per_dev * sp
    num_layers = 2  # slot 0 is written, slot 1 is the neighbour that must stay untouched
    slot_id, layer_idx = 0, 0
    batch_idx = slot_id * num_layers + layer_idx

    if case == "mid_chunk":
        # Ends 5 tokens into a page row, so ceil-to-32 keeps the boundary row and drops the rest.
        kv_actual, valid_global = 0, chunk_global - 2 * tile - 5
    elif case == "tail_overflow":
        # Starts one tile off the slab grid in the last slab: padded window ends at cache_global + tile.
        kv_actual, valid_global = chunk_global + tile, cache_global
    else:
        # boundary_chip = (kv_actual / chunk_local) % sp > 0, in the LAST slab: every chip before the
        # boundary jumps a full slab, landing exactly at the end of its cache -- nothing to write.
        kv_actual, valid_global = cache_global - chunk_local + tile, cache_global
        assert (kv_actual // chunk_local) % sp > 0, "case must put the boundary chip past chip 0"
    write_end = -(-valid_global // tile) * tile
    assert write_end < kv_actual + chunk_global, "the case must leave a pad tail to clamp away"
    if case != "mid_chunk":
        assert kv_actual + chunk_global > cache_global >= write_end, "the case must overflow only in its pad"

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims)

    def _to_device(rows):
        return _make_input(rows.reshape(1, 1, chunk_global, KVPE_HEAD_DIM), dtype, layout, mesh_device, mapper)

    def _read_back(tt):
        """Rows as the device holds them, in chip-concat order, through the cache's own dtype."""
        return ttnn.to_torch(tt, mesh_composer=composer).to(torch.float32)[0, 0]

    mesh_device.enable_program_cache()
    kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=KVPE_HEAD_DIM,
        mesh_device=mesh_device,
        seq_len=cache_global,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        dtype=dtype,
        layout=layout,
    )

    # Sentinel: fill every slot, slab by slab, with UNCLAMPED chunk-aligned writes (natural order in,
    # so slab s lands at positions [s*chunk_global, (s+1)*chunk_global)).
    torch.manual_seed(7)
    sentinel = {b: torch.randn(cache_global, KVPE_HEAD_DIM, dtype=torch.bfloat16) for b in range(num_layers)}
    sentinel_dev = {}
    for b in range(num_layers):
        slab_rows = []
        for s in range(slabs):
            tt_slab = _to_device(sentinel[b][s * chunk_global : (s + 1) * chunk_global])
            slab_rows.append(_read_back(tt_slab))  # what the dtype actually stored
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kv_cache,
                tt_slab,
                slot_idx=b // num_layers,
                layer_idx=b % num_layers,
                num_layers=num_layers,
                kv_actual_global=s * chunk_global,
                cluster_axis=sp_axis,
            )
            ttnn.deallocate(tt_slab)
        sentinel_dev[b] = torch.cat(slab_rows, dim=0)  # natural order, [cache_global, head_dim]
    ttnn.synchronize_device(mesh_device)

    # The chunk under test, in ROTATED order: chip c's row r carries positions[c][r]; rows past the real
    # end are pad and carry POISON.
    positions = _rotated_chip_positions(kv_actual, sp, chunk_local)
    poison = torch.full((KVPE_HEAD_DIM,), 8.0, dtype=torch.bfloat16)
    new_nat = torch.randn(chunk_global, KVPE_HEAD_DIM, dtype=torch.bfloat16)
    rotated = torch.stack(
        [
            new_nat[positions[c][r] - kv_actual] if positions[c][r] < write_end else poison
            for c in range(sp)
            for r in range(chunk_local)
        ]
    )
    tt_chunk = _to_device(rotated)
    sent_rows = _read_back(tt_chunk)  # rotated order, through the cache dtype
    sent_by_pos = {positions[c][r]: sent_rows[c * chunk_local + r] for c in range(sp) for r in range(chunk_local)}

    if case != "mid_chunk":  # unclamped, this write does not fit -- the op must still say so
        with expect_error(RuntimeError, "overflow global cache capacity"):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kv_cache,
                tt_chunk,
                slot_idx=slot_id,
                layer_idx=layer_idx,
                num_layers=num_layers,
                kv_actual_global=kv_actual,
                cluster_axis=sp_axis,
            )

    if path == "metadata":
        slot_t, kv_t = _make_meta_tensors(mesh_device, kv_actual_global=kv_actual, slot_idx=slot_id)
        valid_t = _make_scalar_tensor(mesh_device, valid_global)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            kv_cache,
            tt_chunk,
            slot_t,
            kv_t,
            layer_idx=layer_idx,
            num_layers=num_layers,
            cluster_axis=sp_axis,
            valid_global=valid_t,
        )
        for t in (slot_t, kv_t, valid_t):
            ttnn.deallocate(t)
    else:
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            kv_cache,
            tt_chunk,
            slot_idx=slot_id,
            layer_idx=layer_idx,
            num_layers=num_layers,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
            valid_global=valid_global,
        )
    ttnn.synchronize_device(mesh_device)

    cache_host = ttnn.to_torch(kv_cache, mesh_composer=composer).to(torch.float32)[:, :1, :, :]
    for b in range(num_layers):
        nat = _natural_from_cache(cache_host[b, 0], sp, chunk_local, cache_tokens_per_dev, chunk_global)
        for pos in range(cache_global):
            if b == batch_idx and kv_actual <= pos < write_end:
                want, what = sent_by_pos[pos], f"the chunk's row for position {pos}"
            else:
                want, what = sentinel_dev[b][pos], "the sentinel (this row must not have been written)"
            assert torch.equal(
                nat[pos], want.to(torch.float32)
            ), f"[{case}/{path}] slot {b} position {pos} does not hold {what}" + (
                " -- POISON LEAKED (a pad row was written)" if torch.equal(nat[pos], poison.to(torch.float32)) else ""
            )
    logger.success(
        f"[{case}/{path}] clamp held: wrote [{kv_actual}, {write_end}) of a chunk spanning "
        f"[{kv_actual}, {kv_actual + chunk_global}) into a {cache_global}-token cache"
    )


def _alloc_multihead_cache(mesh_device, *, batch, heads, seq_local, head_dim, dtype, layout):
    """Cache with a per-chip HEAD dim > 1, ND-sharded exactly like the model caches (32-token bank
    chunks, round-robin over the DRAM grid). ``init_kvpe_cache`` is fixed at one head, so this is its
    multi-head sibling -- the layout ``allocate_dflash_kv_cache`` produces when
    ``num_key_value_heads > tp``."""
    from models.demos.common.prefill.runners.migration import get_num_dram_banks

    grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0))
            for bank_id in range(get_num_dram_banks(mesh_device))
        ]
    )
    mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
            grid=grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )
    return ttnn.from_torch(
        torch.zeros(batch, heads, seq_local, head_dim, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param(
            (2, 2), marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"), id="2x2"
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("case", ["full", "clamped"], ids=["full", "clamped"])
@pytest.mark.timeout(0)
def test_update_padded_kv_cache_multihead_head_stride(mesh_device, case):
    """A per-chip head dim > 1 must not smear rows across heads.

    The cache's head stride exceeds the input's once the cache is deeper than one chunk, so a core holding
    blocks either side of a head boundary must address each from its own (head, row) -- 64 heads x 7
    page-rows over ~110 cores puts several cores across one. `clamped` repeats it with valid_global, so the
    row clamp is checked per head rather than per core."""
    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    tile = ttnn.TILE_SIZE

    heads, head_dim = 64, 64
    chunk_local = 7 * tile  # 7 page-rows/head: blocks (64*7=448) >> cores, boundaries every 7 blocks
    chunk_global = chunk_local * sp
    seq_local = 2 * chunk_local  # deeper than one chunk -> cache head stride != input head stride
    num_layers = 2
    layer_idx, slot_id = 0, 0
    valid_global = chunk_global - 2 * tile - 5 if case == "clamped" else None
    write_end = chunk_global if valid_global is None else -(-valid_global // tile) * tile

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1

    mesh_device.enable_program_cache()
    cache = _alloc_multihead_cache(
        mesh_device,
        batch=num_layers,
        heads=heads,
        seq_local=seq_local,
        head_dim=head_dim,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    torch.manual_seed(3)
    src = torch.randn(1, heads, chunk_global, head_dim, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        src,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims),
    )
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache,
        tt_input,
        slot_idx=slot_id,
        layer_idx=layer_idx,
        num_layers=num_layers,
        kv_actual_global=0,
        cluster_axis=sp_axis,
        valid_global=valid_global,
    )
    ttnn.synchronize_device(mesh_device)

    # Chunk 0 at offset 0: chip c holds global positions [c*chunk_local, (c+1)*chunk_local) in its first
    # chunk_local rows, so concatenating the chips' slabs on the seq dim gives natural order directly.
    host = ttnn.to_torch(
        cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[:num_layers, :, :, :head_dim]
    written = torch.cat([host[layer_idx, :, c * seq_local : c * seq_local + chunk_local, :] for c in range(sp)], dim=1)

    # Spill first: a misaddressed block shows up as a write past the chip's real rows, and that is
    # the more specific signal. It must read `host`, not `written` -- `written` holds exactly
    # chunk_local rows per chip, so unclamped (write_end == chunk_global) slicing it past write_end
    # yields nothing at all. Clamped, the per-head slice below is the dropped pad tail.
    for c in range(sp):
        tail = host[layer_idx, :, c * seq_local + chunk_local : (c + 1) * seq_local, :]
        assert torch.count_nonzero(tail) == 0, f"[{case}] chip {c}: wrote past its {chunk_local} real rows"
    assert torch.count_nonzero(host[1 - layer_idx]) == 0, "wrote into the neighbouring layer slot"
    for h in range(heads):
        assert torch.equal(written[h, :write_end], src[0, h, :write_end].to(torch.float32)), (
            f"[{case}] head {h}: rows [0, {write_end}) do not match what was sent -- a block landed in " f"another head"
        )
        assert torch.count_nonzero(written[h, write_end:]) == 0, f"[{case}] head {h}: wrote past {write_end}"
    logger.success(f"[{case}] {heads} heads x {chunk_local} rows placed exactly (write_end={write_end})")
