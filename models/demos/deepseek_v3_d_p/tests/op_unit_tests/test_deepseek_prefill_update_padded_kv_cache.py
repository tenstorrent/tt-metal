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
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from tests.ttnn.utils_for_testing import assert_with_pcc

# MLA KVPE head dim (kv_lora_rank=512 + qk_rope_head_dim=64). The op is a pure tile
# copy, so a gathered cache slot should match the data we sent (PCC, allowing for the
# bfloat8_b cache round-trip).
KVPE_HEAD_DIM = 576


@pytest.mark.parametrize("mesh_device", [(1, 4), (2, 4), (8, 4)], ids=["1x4", "2x4", "8x4"], indirect=True)
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
    is_ci_env,
    is_ci_v2_env,
):
    """Single-iteration (non-padded) prefill: write one chunk-aligned slab per (user, layer)
    at offset 0, gather the whole cache, and PCC each slot's valid data against what was sent."""
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
    )

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2  # split the chunk across sp devices

    mesh_device.enable_program_cache()

    for u in range(num_users):
        for l in range(num_layers):
            tt_input = ttnn.from_torch(
                sent[(u, l)].reshape(1, 1, new_isl_global, KVPE_HEAD_DIM),
                device=mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims
                ),
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

    # slot_idx and kv_actual_global are device tensors (not in the program hash) and layer_idx is
    # hashed, so exactly one cached program per layer is reused across all users — entries == num_layers.
    assert mesh_device.num_program_cache_entries() == num_layers, (
        f"op must reuse one cached program per layer across users; expected {num_layers} entries, "
        f"got {mesh_device.num_program_cache_entries()}"
    )

    # Gather: concat sp shards on the seq dim, drop the tp-replicated head copies.
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    cache_host = ttnn.to_torch(
        kv_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
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
            _, msg = assert_with_pcc(sent[(u, l)], written, 0.99)
            logger.info(f"  user {u} layer {l}: valid-data PCC {msg}")

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
    rebuilt, and every (user, layer) slot's valid prefix is PCC'd against the data sent.
    """
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
    )

    input_shard_dims = [None, None]
    input_shard_dims[sp_axis] = 2

    mesh_device.enable_program_cache()

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
        for u in range(num_users):
            for l in range(num_layers):
                chunk = sent[(u, l)][gather_idx].clone()  # [chunk_global, KVPE_HEAD_DIM]
                chunk[pad_mask] = 0.0  # server pad rows
                tt_input = ttnn.from_torch(
                    chunk.reshape(1, 1, chunk_global, KVPE_HEAD_DIM),
                    device=mesh_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_shard_dims
                    ),
                )
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
    # hashed, so exactly one cached program per layer is reused across all iterations and users —
    # entries == num_layers.
    assert mesh_device.num_program_cache_entries() == num_layers, (
        f"op must reuse one cached program per layer across iterations/users; expected {num_layers} "
        f"entries, got {mesh_device.num_program_cache_entries()}"
    )

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    cache_host = ttnn.to_torch(
        kv_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)[
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
            _, msg = assert_with_pcc(sent[(u, l)], recon, 0.99)
            logger.info(f"  user {u} layer {l}: valid-prefix PCC {msg}")

    logger.info(f"program cache entries: {mesh_device.num_program_cache_entries()}")
