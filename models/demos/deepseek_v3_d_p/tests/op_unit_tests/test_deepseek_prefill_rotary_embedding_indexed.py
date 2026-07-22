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
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder
from tests.ttnn.utils_for_testing import assert_with_pcc

# RoPE is applied to the qk_rope_head_dim slice (64) in MLA. The op is dtype-agnostic; we use a
# self-consistent Meta-style cos/sin so the test validates the per-device indexing/offset logic
# (the new behavior) -- the rotary math itself is already covered by test_rope_prefill.py.
ROPE_HEAD_DIM = 64


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


@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4), (8, 4)], ids=["2x2", "2x4", "8x4"], indirect=True)
@pytest.mark.parametrize(
    "config_name, num_heads_local, new_isl_tiles_per_dev, cache_tokens_per_dev",
    [
        ("small", 2, 4, 512),  # small: 2 heads/dev, 4-tile chunk/dev
        ("repr", 8, 20, 6400),  # representative: 8 heads/dev, 5k new isl + 50k cache on 8x4 (per-dev scaled)
    ],
    ids=["small", "repr"],
)
@pytest.mark.parametrize("tensor_kind", ["Q", "KV"], ids=["Q", "KV"])
@pytest.mark.parametrize("scenario", ["non_padded", "padded_partial"], ids=["non_padded", "padded_partial"])
@pytest.mark.timeout(0)
def test_rotary_embedding_indexed_multi_iteration_prefill(
    mesh_device,
    config_name,
    num_heads_local,
    new_isl_tiles_per_dev,
    cache_tokens_per_dev,
    tensor_kind,
    scenario,
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
    kv_actual_global is a runtime arg (not hashed)."""
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
        f"tensor_kind={tensor_kind} n_heads={n_heads} sp={sp} tp={tp} chunk_local={C} "
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
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
