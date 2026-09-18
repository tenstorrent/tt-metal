# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device test for ttnn.experimental.deepseek_prefill.rotary_embedding_indexed.

The op applies rotary embedding to a per-chip input chunk, indexing into SP-sharded cos/sin caches
at a per-device offset derived on-device from a single global valid-KV length `kv_actual_global`.
The cos/sin caches are sharded in block-cyclic order keyed by the per-chip chunk size, so the
boundary chip's older-then-wrap token layout is read with a single contiguous `update_idxt` offset.

This mirrors test_deepseek_prefill_update_padded_kv_cache.py: the same `_rotated_chip_positions`
math gives, per (chip, local row), the global position that row carries; the test rotates a random
input chunk on device and PCCs against a torch RoPE reference applied at those global positions.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import rotate_half
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from tests.ttnn.utils_for_testing import assert_with_pcc

# RoPE is applied to the qk_rope_head_dim slice (64) in MLA. The op is dtype-agnostic; we use a
# self-consistent Meta-style cos/sin so the test validates the per-device indexing/offset logic
# (the new behavior) -- the rotary math itself is already covered by test_rope_prefill.py.
ROPE_HEAD_DIM = 64


@pytest.mark.parametrize("mesh_device", [(1, 4), (2, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 2 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("sp_axis", [0, 1])
def test_rotary_embedding_indexed_sequence_subshards(mesh_device, sp_axis, expect_error):
    """Early TP partition must commute with RoPE, including rotated starts and metadata replay.

    Cos/sin retain the original SP slabs. Compare against full-slab RoPE then partition, and retain
    one captured metadata program while changing chunk starts. Both mesh-axis orientations are used.
    """
    tp_axis = 1 - sp_axis
    sp, tp = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
    chunk_local = 128 * tp
    chunk_global = chunk_local * sp
    cos, sin = _make_cos_sin(4 * chunk_global, ROPE_HEAD_DIM)
    dims = [None, None]
    dims[sp_axis] = 2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)

    def upload(tensor, mesh_mapper=mapper):
        return ttnn.from_torch(
            tensor, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper
        )

    cos_tt = upload(block_cyclic_reorder(cos, chunk_local, sp, seq_dim=2))
    sin_tt = upload(block_cyclic_reorder(sin, chunk_local, sp, seq_dim=2))
    trans_tt = upload(get_rot_transformation_mat(), ttnn.ReplicateTensorToMesh(mesh_device))
    torch.manual_seed(42)
    full_q = upload(torch.randn(1, 8, chunk_global, ROPE_HEAD_DIM, dtype=torch.bfloat16))
    local_q = ttnn.mesh_partition(full_q, dim=2, cluster_axis=tp_axis)
    metadata = ttnn.from_torch(
        torch.zeros((1, 1, 1, 1), dtype=torch.int64),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rope = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed

    def local_rope(start):
        return rope(local_q, cos_tt, sin_tt, trans_tt, start, sp_axis, seq_subshard_axis=tp_axis)

    with expect_error(RuntimeError, "different mesh axis"):
        rope(local_q, cos_tt, sin_tt, trans_tt, 0, sp_axis, seq_subshard_axis=sp_axis)

    # Warm the exact metadata program before capturing. The captured output owns its allocation
    # through all replays; only the metadata value changes in place.
    warm = local_rope(metadata)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = local_rope(metadata)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    entries = None
    try:
        # Include a TP window crossing a block-cyclic slab boundary at a rotated chunk start.
        for start in (0, chunk_global, chunk_local + 32, 2 * chunk_global + chunk_local - 32):
            full_out = rope(full_q, cos_tt, sin_tt, trans_tt, start, sp_axis)
            expected = ttnn.mesh_partition(full_out, dim=2, cluster_axis=tp_axis)
            scalar_out = local_rope(start)
            host_start = ttnn.from_torch(
                torch.tensor([start], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_start, metadata)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            reference = ttnn.to_torch(expected, mesh_composer=composer)
            assert torch.equal(reference, ttnn.to_torch(scalar_out, mesh_composer=composer))
            assert torch.equal(reference, ttnn.to_torch(traced_out, mesh_composer=composer))
            if entries is None:
                entries = mesh_device.num_program_cache_entries()
            assert mesh_device.num_program_cache_entries() == entries
            for tensor in (full_out, expected, scalar_out):
                ttnn.deallocate(tensor)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(traced_out)


def _make_cos_sin(max_seq, head_dim):
    """Meta-style cos/sin [1, 1, max_seq, head_dim] = [c0,c0,c1,c1,...] -- same layout as
    get_cos_sin_matrix, so rotate_half(meta_style=True) is the matching reference."""
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)  # [max_seq, half]
    cos = torch.stack((angles.cos(), angles.cos()), dim=-1).flatten(-2)  # [max_seq, head_dim]
    sin = torch.stack((angles.sin(), angles.sin()), dim=-1).flatten(-2)
    return cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)


def _rotated_chip_positions(kv_actual, sp, chunk_local):
    """Global token position carried by each chip-local input row, mirroring the writer kernel's
    update_idxt math (identical to the kv-cache op test). positions[c][r] is the global position
    chip c's r-th input row maps to. Slab-aware (handles kv_actual spanning multiple slabs)."""
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


_MESHES = [(2, 2), (2, 4), (8, 4)]
_CONFIGS = [("small", 2, 4, 512), ("repr", 8, 20, 6400)]
_CASES = [
    pytest.param(
        mesh,
        *config,
        tensor_kind,
        scenario,
        ttnn.bfloat16,
        id=f"{mesh[0]}x{mesh[1]}-{config[0]}-{tensor_kind}-{scenario}-bf16",
    )
    for mesh in _MESHES
    for config in _CONFIGS
    for tensor_kind in ("Q", "KV")
    for scenario in ("non_padded", "padded_partial")
]
_CASES.append(
    pytest.param(
        (2, 4), "small", 2, 4, 512, "Q", "padded_partial", ttnn.bfloat8_b, id="2x4-small-Q-padded_partial-bfp8"
    )
)


@pytest.mark.parametrize(
    "mesh_device, config_name, num_heads_local, new_isl_tiles_per_dev, cache_tokens_per_dev, "
    "tensor_kind, scenario, input_dtype",
    _CASES,
    indirect=["mesh_device"],
)
@pytest.mark.timeout(0)
def test_rotary_embedding_indexed_multi_iteration_prefill(
    mesh_device,
    config_name,
    num_heads_local,
    new_isl_tiles_per_dev,
    cache_tokens_per_dev,
    tensor_kind,
    scenario,
    input_dtype,
    is_ci_env,
    is_ci_v2_env,
):
    """Multi-iteration prefill RoPE, for both MLA rope tensors.

    - tensor_kind="Q": multi-head input sharded on BOTH the TP axis (heads, like
      num_heads_local = num_heads // tp_factor in mla.py) and the SP axis (seq). Proves the op is
      TP-layout-agnostic -- the cos/sin offset depends only on the SP coordinate, and cos/sin are
      TP-replicated, so each device applies the right per-SP rope to whatever heads it holds.
    - tensor_kind="KV": single-head input, TP-replicated and SP-sharded (kv rope is reduced across
      TP before rotation in mla.py, so n_heads=1 and freq_per_head degenerates).

    - non_padded: two full-chunk iterations (chunk-aligned -> uniform per-device offset).
    - padded_partial: three iterations with whole-tile, non-zero pad offsets so the boundary chip's
      cos/sin read straddles a slab boundary (older-then-wrap), exactly as in the kv-cache op test.

    Each iteration rotates a random chunk on device and PCCs every (chip, row) against a torch RoPE
    reference applied at that row's true global position. Also asserts program-cache reuse, proving
    kv_actual_global is a runtime arg (not hashed). The additional BFP8 Q case uses the same BF16
    cos/sin and transformation matrix as the production path."""
    if (is_ci_env or is_ci_v2_env) and not (config_name == "small" and scenario == "padded_partial"):
        pytest.skip("CI runs only the small padded_partial case; the others are subsets of it")

    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    tp = mesh_device.shape[tp_axis]
    tile = ttnn.TILE_SIZE

    # Q: heads sharded across TP (n_heads = num_heads_local * tp), like mla.py tt_q_rope.
    # KV: single head, TP-replicated, like mla.py tt_kv_rope.
    if tensor_kind == "Q":
        n_heads = num_heads_local * tp
    else:
        n_heads = 1
    C = new_isl_tiles_per_dev * tile  # per-device chunk (tokens), fixed every iter
    chunk_global = C * sp

    if scenario == "non_padded":
        new_actual_isls = [chunk_global, chunk_global]
    else:  # padded_partial: whole-tile boundary_offset != 0; boundary chip read straddles slabs
        new_actual_isls = [(sp - 1) * C + tile, 2 * C, sp * C]
    cum_total = sum(new_actual_isls)
    cache_global = cache_tokens_per_dev * sp
    assert cum_total <= cache_global, f"valid tokens ({cum_total}) must fit the cache ({cache_global})"

    logger.info(
        f"tensor_kind={tensor_kind} dtype={input_dtype} n_heads={n_heads} sp={sp} tp={tp} chunk_local={C} "
        f"chunk_global={chunk_global} cache_global={cache_global}; "
        f"new_isl per iter={new_actual_isls} (cum_total={cum_total})"
    )

    torch.manual_seed(0)

    # Full cos/sin covering the whole cache, then block-cyclic-reorder keyed by the per-chip chunk
    # and SP-shard so each device's contiguous shard holds the rope values for every position it
    # will carry, in local-cache-row order.
    cos_full, sin_full = _make_cos_sin(cache_global, ROPE_HEAD_DIM)  # [1, 1, cache_global, head_dim]
    cos_re = block_cyclic_reorder(cos_full, C, sp, seq_dim=2)
    sin_re = block_cyclic_reorder(sin_full, C, sp, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2  # SP-shard the seq dim; replicate across TP
    from_torch_kwargs = dict(
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_tt = ttnn.from_torch(
        cos_re,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        **from_torch_kwargs,
    )
    sin_tt = ttnn.from_torch(
        sin_re,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        **from_torch_kwargs,
    )
    trans_tt = ttnn.from_torch(
        get_rot_transformation_mat(),
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        **from_torch_kwargs,
    )

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2  # split the global chunk across SP devices (contiguous)
    if tensor_kind == "Q":
        input_shard_dims[tp_axis] = 1  # shard heads across TP (like mla.py); KV replicates instead

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1

    mesh_device.enable_program_cache()
    entries_after_first = None

    kv_actual = 0
    for it, new_actual_isl in enumerate(new_actual_isls):
        positions = _rotated_chip_positions(kv_actual, sp, C)
        flat = [positions[c][r] for c in range(sp) for r in range(C)]  # chip-concat order, len chunk_global
        assert max(flat) < cache_global, f"position {max(flat)} exceeds cache ({cache_global})"
        logger.info(f"  iter {it}: kv_actual={kv_actual} new_isl={new_actual_isl} max_pos={max(flat)}")

        torch_input = torch.randn(1, n_heads, chunk_global, ROPE_HEAD_DIM, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=input_dtype,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims),
            **from_torch_kwargs,
        )

        # kv_actual_global is a per-call scalar held in a common runtime arg and patched on cache hits,
        # so its value stays out of the program hash and successive chunks reuse one cached program.
        tt_out = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            tt_input,
            cos_tt,
            sin_tt,
            trans_tt,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
        )
        assert tt_out.dtype == input_dtype
        if entries_after_first is None:
            ttnn.synchronize_device(mesh_device)
            entries_after_first = mesh_device.num_program_cache_entries()

        out_host = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)[
            :, :n_heads, :, :
        ]  # Q: heads reassembled across TP (no-op slice); KV: drop the TP-replicated copies

        # Reference: rotate each chip-concat row by cos/sin at its true global position.
        cos_sel = cos_full[0, 0, flat, :].unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_global, head_dim]
        sin_sel = sin_full[0, 0, flat, :].unsqueeze(0).unsqueeze(0)
        ref = (torch_input * cos_sel) + (rotate_half(torch_input, meta_style=True) * sin_sel)

        assert torch.isfinite(out_host).all()
        _, msg = assert_with_pcc(ref, out_host, 0.99)
        logger.info(f"  iter {it}: PCC {msg}")
        kv_actual += new_actual_isl

    ttnn.synchronize_device(mesh_device)

    # kv_actual_global is a runtime arg (not in the program hash), so every iteration must reuse the
    # program compiled on the first call.
    assert mesh_device.num_program_cache_entries() == entries_after_first, (
        f"op must reuse one cached program across iterations; entries grew from "
        f"{entries_after_first} to {mesh_device.num_program_cache_entries()}"
    )
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
@pytest.mark.timeout(0)
def test_rotary_embedding_indexed_metadata_matches_scalar(mesh_device):
    """The per-element-tensor (traceable) path and the scalar path must produce bit-identical outputs.

    Drives the traceable path from a 1-element uint32 DRAM tensor holding kv_actual_global (the reader
    reads its element [0] on-device), and compares the rotated output against the same call done via
    the original scalar kv_actual_global. Exact equality over chunk-0 and a mid-cache offset."""
    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    tile = ttnn.TILE_SIZE

    n_heads = 1  # KV-rope shape (single head, SP-sharded)
    new_isl_tiles_per_dev = 4
    cache_tokens_per_dev = 512
    C = new_isl_tiles_per_dev * tile  # per-device chunk (tokens)
    chunk_global = C * sp
    cache_global = cache_tokens_per_dev * sp

    torch.manual_seed(0)
    cos_full, sin_full = _make_cos_sin(cache_global, ROPE_HEAD_DIM)
    cos_re = block_cyclic_reorder(cos_full, C, sp, seq_dim=2)
    sin_re = block_cyclic_reorder(sin_full, C, sp, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2
    from_torch_kwargs = dict(
        device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cos_tt = ttnn.from_torch(
        cos_re,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        **from_torch_kwargs,
    )
    sin_tt = ttnn.from_torch(
        sin_re,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        **from_torch_kwargs,
    )
    trans_tt = ttnn.from_torch(
        get_rot_transformation_mat(), mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device), **from_torch_kwargs
    )

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape)

    # The traceable path reads kv_actual_global from element [0] of this 1-element uint32 DRAM tensor.
    def _make_scalar_tensor(value):
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    mesh_device.enable_program_cache()
    # Two slab-aligned offsets (boundary_chip == 0) plus one NON-slab-aligned offset (C + one tile:
    # boundary_chip == 1 with a whole-tile boundary_offset), so the boundary chip's cos/sin read straddles
    # a slab — the hard indexing case for this op — is exercised through the metadata path too, not just
    # slab-aligned cases. kv_actual_global is a runtime arg (not hashed), so every case reuses the same
    # cached program regardless of alignment (asserted after the loop).
    cases = [0, chunk_global, C + tile]
    entries_after_first = None

    for kv_actual in cases:
        torch_input = torch.randn(1, n_heads, chunk_global, ROPE_HEAD_DIM, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(
            torch_input,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims),
            **from_torch_kwargs,
        )

        kv_t = _make_scalar_tensor(kv_actual)

        out_scalar = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            tt_input, cos_tt, sin_tt, trans_tt, kv_actual_global=kv_actual, cluster_axis=sp_axis
        )
        out_meta = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            tt_input, cos_tt, sin_tt, trans_tt, kv_actual_global=kv_t, cluster_axis=sp_axis
        )
        ttnn.synchronize_device(mesh_device)

        scalar_host = ttnn.to_torch(out_scalar, mesh_composer=composer).to(torch.float32)[:, :n_heads, :, :]
        meta_host = ttnn.to_torch(out_meta, mesh_composer=composer).to(torch.float32)[:, :n_heads, :, :]
        assert torch.equal(meta_host, scalar_host), (
            f"kv_actual={kv_actual}: per-element-tensor-path output differs from scalar-path "
            f"(max abs diff {(meta_host - scalar_host).abs().max().item()})"
        )
        logger.success(f"kv_actual={kv_actual}: per-element-tensor path == scalar path (bit-exact)")
        # After the first chunk both programs (scalar + metadata, distinct by metadata.has_value()) are
        # compiled; capture the count so we can assert no further growth across the remaining chunks.
        if entries_after_first is None:
            entries_after_first = mesh_device.num_program_cache_entries()
        ttnn.deallocate(kv_t)
        ttnn.deallocate(tt_input)

    # The whole point of this path: kv_actual_global is a runtime arg (metadata address patched on cache
    # hits), NOT part of the program hash, so successive chunks — including the non-slab-aligned one —
    # must reuse the one cached metadata program rather than compile a new one each time.
    assert mesh_device.num_program_cache_entries() == entries_after_first, (
        f"program cache grew across chunks — the scalar+metadata programs should each compile once and be "
        f"reused (kv_actual_global is a runtime arg, not hashed): {entries_after_first} -> "
        f"{mesh_device.num_program_cache_entries()}"
    )
    logger.info(f"program cache stable at {entries_after_first} entries across {len(cases)} chunks")


@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4)], ids=["2x2", "2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 2 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("rotary_offset", [0, 32])
@pytest.mark.parametrize(
    "subshard, reuse_cos_sin", [(False, False), (True, False), (True, True)], ids=["keys", "queries", "queries-reuse"]
)
def test_rotary_embedding_indexed_partial(mesh_device, rotary_offset, subshard, reuse_cos_sin, expect_error):
    """Partial RoPE matches slice/rotate/concat and copies other channels exactly on replay."""
    sp, tp = mesh_device.shape
    # Keep each TP query shard tile-aligned on both meshes.
    chunk_local, width, rotary_dim = 160 * tp, 128, 64
    chunk_global = chunk_local * sp
    capacity = 4 * chunk_global
    positions = torch.arange(capacity).float()
    frequencies = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    angles = torch.outer(positions, frequencies).repeat_interleave(2, dim=-1)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None))

    def upload(x, mapper=mapper):
        return ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

    cos = upload(block_cyclic_reorder(angles.cos().reshape(1, 1, capacity, rotary_dim), chunk_local, sp, seq_dim=2))
    sin = upload(block_cyclic_reorder(angles.sin().reshape(1, 1, capacity, rotary_dim), chunk_local, sp, seq_dim=2))
    trans = upload(get_rot_transformation_mat(), ttnn.ReplicateTensorToMesh(mesh_device))
    torch.manual_seed(56)
    num_heads = 4 if subshard else 1
    if reuse_cos_sin:
        grid = mesh_device.compute_with_storage_grid_size()
        seq_tiles = chunk_local // tp // 32
        num_cores = grid.x * grid.y
        if num_cores < seq_tiles:
            pytest.skip("Cos/sin reuse coverage requires one core per sequence tile row")
        # Two heads per core and one row per head force RELOAD_IMPL=0 with shared cos/sin.
        # On an 80-core grid this matches GLM's 32 heads and five local sequence tile rows.
        num_heads = 2 * (num_cores // seq_tiles)
    x = upload(torch.randn(1, num_heads, chunk_global, width, dtype=torch.bfloat16))
    if subshard:
        full = x
        x = ttnn.mesh_partition(x, dim=2, cluster_axis=1)
        ttnn.deallocate(full)
    local_rows = chunk_local // tp if subshard else chunk_local
    heads = x.shape[1]
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(2, 1))
    input_host = ttnn.to_torch(x, mesh_composer=composer)
    metadata = ttnn.from_torch(
        torch.zeros(1, 1, 1, 1, dtype=torch.int64),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rope = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed
    opts = {"seq_subshard_axis": 1 if subshard else None}

    def partial(start):
        return rope(x, cos, sin, trans, start, 0, rotary_dim=rotary_dim, rotary_offset=rotary_offset, **opts)

    with expect_error(RuntimeError, "rotary"):
        rope(x, cos, sin, trans, 0, 0, rotary_dim=64, rotary_offset=16, **opts)
    with expect_error(RuntimeError, "rotary"):
        rope(x, cos, sin, trans, 0, 0, rotary_dim=64, rotary_offset=96, **opts)
    with expect_error(RuntimeError, "rotary"):
        rope(x, cos, sin, trans, 0, 0, rotary_dim=32, **opts)

    warmed = partial(metadata)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warmed)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = partial(metadata)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for start in (0, chunk_global, chunk_local + 32, 2 * chunk_global + chunk_local - 32):
            pe = ttnn.slice(x, [0, 0, 0, rotary_offset], [1, heads, local_rows, rotary_offset + rotary_dim])
            expected = rope(pe, cos, sin, trans, start, 0, **opts)
            expected_host = ttnn.to_torch(expected, mesh_composer=composer)
            scalar = partial(start)
            scalar_host = ttnn.to_torch(scalar, mesh_composer=composer)
            for t in (pe, expected, scalar):
                ttnn.deallocate(t)
            host_start = ttnn.from_torch(
                torch.tensor([start], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_start, metadata)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            replay_host = ttnn.to_torch(out, mesh_composer=composer)
            for result in (scalar_host, replay_host):
                assert_with_pcc(expected_host, result[..., rotary_offset : rotary_offset + rotary_dim], 0.9999)
                assert torch.equal(input_host[..., :rotary_offset], result[..., :rotary_offset])
                assert torch.equal(
                    input_host[..., rotary_offset + rotary_dim :], result[..., rotary_offset + rotary_dim :]
                )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(out)


@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4)], ids=["2x2", "2x4"], indirect=True)
@pytest.mark.parametrize("subshard", [False, True], ids=["keys", "queries"])
def test_indexer_deepseek_rope_fallback(mesh_device, subshard):
    """The unfused DeepSeek path preserves half-split rotary semantics and the nonrotary half."""
    from types import SimpleNamespace

    from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer
    from models.demos.deepseek_v3_d_p.tt.mla.rope import interleaved_perm_matrix

    sp, tp = mesh_device.shape
    # Keep each TP query shard tile-aligned on both meshes.
    rows, width = 160 * tp, 128
    capacity = 4 * rows * sp
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(2, None))

    def upload(x, mapper=mapper):
        return ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

    cos, sin = _make_cos_sin(capacity, 64)
    rope = {
        "cos_matrix": upload(block_cyclic_reorder(cos, rows, sp, seq_dim=2)),
        "sin_matrix": upload(block_cyclic_reorder(sin, rows, sp, seq_dim=2)),
        "trans_matrix": upload(get_rot_transformation_mat(), ttnn.ReplicateTensorToMesh(mesh_device)),
    }
    perm = interleaved_perm_matrix(64).to(torch.bfloat16)
    indexer = SimpleNamespace(
        sp_axis=0,
        index_args=SimpleNamespace(index_head_dim=width),
        _rope_perm=upload(perm, ttnn.ReplicateTensorToMesh(mesh_device)),
        hifi4_fp32_compute_kernel_config=ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    torch.manual_seed(57)
    x_host = torch.randn(1, 4 if subshard else 1, rows * sp, width, dtype=torch.bfloat16)
    x = upload(x_host)
    if subshard:
        full = x
        x = ttnn.mesh_partition(full, dim=2, cluster_axis=1)
        ttnn.deallocate(full)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(2, 1))
    for start in (0, rows + 32, 2 * rows * sp):
        positions = torch.tensor(_rotated_chip_positions(start, sp, rows)).flatten()
        # Rotate in DeepSeek's half-split basis, then permute to the stored interleaved basis.
        pe = x_host[..., :64].float()
        cs = cos[0, 0, positions, ::2].float().repeat(1, 2)
        sn = sin[0, 0, positions, ::2].float().repeat(1, 2)
        rotated = (pe * cs + rotate_half(pe) * sn) @ perm.float()
        expected = torch.cat((rotated, x_host[..., 64:].float()), dim=-1)
        expected = torch.cat(
            [
                torch.cat(
                    [
                        expected[
                            :,
                            :,
                            s * rows
                            + (t * rows // tp if subshard else 0) : s * rows
                            + ((t + 1) * rows // tp if subshard else rows),
                        ]
                        for t in range(tp)
                    ],
                    dim=1,
                )
                for s in range(sp)
            ],
            dim=2,
        )
        for use_metadata in (False, True):
            meta = None
            if use_metadata:
                actual_start = ttnn.from_torch(
                    torch.tensor([start], dtype=torch.int64).reshape(1, 1, 1, 1),
                    device=mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                meta = (None, actual_start)
            out = TtIndexer._bc_rope_pe(
                indexer, x, rope, start, metadata=meta, seq_subshard_axis=1 if subshard else None
            )
            actual = ttnn.to_torch(out, mesh_composer=composer).float()
            assert_with_pcc(expected[..., :64], actual[..., :64], 0.9999)
            assert torch.equal(expected[..., 64:], actual[..., 64:])
            ttnn.deallocate(out)
            if use_metadata:
                ttnn.deallocate(actual_start)


@pytest.mark.parametrize("use_metadata", [False, True], ids=["scalar", "metadata"])
def test_rotary_embedding_indexed_padded_default(device, use_metadata, expect_error):
    """Omission and None preserve padded full-width RoPE, including both Python overloads."""
    device.enable_program_cache()
    torch.manual_seed(0)

    def upload(x):
        return ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    source = torch.randn(1, 2, 32, 80, dtype=torch.bfloat16)
    cos = torch.randn(1, 1, 32, 80, dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    x, c, s = upload(source), upload(cos), upload(sin)
    trans = upload(get_rot_transformation_mat())
    start = 0
    if use_metadata:
        start = ttnn.from_torch(
            torch.zeros(1, 1, 1, 1, dtype=torch.int64),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    rope = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed
    # Compare to the same physical data with padding made explicit in the logical shape.
    padded = [upload(torch.nn.functional.pad(t, (0, 16))) for t in (source, cos, sin)]
    expected_out = rope(*padded, trans, start, 0, rotary_dim=96)
    expected = ttnn.to_torch(expected_out)[..., :80]
    ttnn.deallocate(expected_out)
    for options in ({}, {"rotary_dim": None}):
        out = rope(x, c, s, trans, start, 0, **options)
        actual = ttnn.to_torch(out)
        assert actual.shape == source.shape
        assert torch.equal(actual, expected)
        ttnn.deallocate(out)
    with expect_error(RuntimeError, "rotary region must fit"):
        rope(x, c, s, trans, start, 0, rotary_dim=96)


@pytest.mark.parametrize("cos_width, sin_width", [(33, 64), (64, 33), (33, 33)])
def test_rotary_embedding_indexed_logical_frequency_width(device, cos_width, sin_width, expect_error):
    """Explicit dimensions reject padded frequency columns even after warming the legacy program."""
    device.enable_program_cache()
    torch.manual_seed(0)

    def upload(x):
        return ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    x = upload(torch.randn(1, 2, 32, 64))
    cos = upload(torch.ones(1, 1, 32, cos_width))
    sin = upload(torch.zeros(1, 1, 32, sin_width))
    trans = upload(get_rot_transformation_mat())
    rope = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed
    warmed = rope(x, cos, sin, trans, 0, 0)
    ttnn.deallocate(warmed)
    with expect_error(RuntimeError, "rotary_dim must match logical cos and sin head dims"):
        rope(x, cos, sin, trans, 0, 0, rotary_dim=64)
