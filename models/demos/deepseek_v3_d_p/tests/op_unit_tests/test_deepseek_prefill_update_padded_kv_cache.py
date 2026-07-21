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

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache

# MLA KVPE head dim (kv_lora_rank=512 + qk_rope_head_dim=64). The op is a pure page copy, so a
# gathered cache slot must byte-match the input we sent (read back through the same dtype
# encode/decode) -- the tests assert exact equality, not PCC.
KVPE_HEAD_DIM = 576


# (cache dtype, layout). bfloat8_b/bfloat4_b are block-float (TILE only); fp8_e4m3 is ROW_MAJOR only
# (Blackhole); bf16 covers the row-major page math in a lossless dtype. The tests assert bit-exact
# equality against the input read back, so no per-dtype tolerance is needed.
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


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
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


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
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
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kv_cache,
                tt_input,
                slot_idx=u,
                layer_idx=l,
                num_layers=num_layers,
                kv_actual_global=0,
                cluster_axis=sp_axis,
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


@pytest.mark.parametrize("mesh_device", [(1, 4), (2, 4), (8, 4)], ids=["1x4", "2x4", "8x4"], indirect=True)
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
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kv_cache,
                tt_input,
                slot_idx=u,
                layer_idx=l,
                num_layers=num_layers,
                kv_actual_global=0,
                cluster_axis=sp_axis,
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


@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4), (8, 4)], ids=["2x2", "2x4", "8x4"], indirect=True)
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
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    kv_cache,
                    tt_input,
                    slot_idx=u,
                    layer_idx=l,
                    num_layers=num_layers,
                    kv_actual_global=kv_actual,
                    cluster_axis=sp_axis,
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
