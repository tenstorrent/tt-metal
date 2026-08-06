# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for mesh-distributed VAE mid-block attention.

The mid-block is a single-head block-causal attention over the whole
spatiotemporal extent, so keys and values genuinely need every H/W rank's
tokens.  Queries do not: ``RMSNorm`` reduces only over channels and ``to_q`` is
a 1x1x1 convolution, so the query for one spatial position depends on that
position alone.  A rank can therefore compute exactly the output rows it
already stores, which removes the post-attention ``mesh_partition`` and divides
attention, projection, and residual work by the rank count.

Everything here is pure torch and runs without a device.  The device cases at
the bottom follow ``test_vae_attention_chunking.py`` and need a real mesh.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.hunyuanvideo_1_5.tt.vae_decoder import AttnBlock
from models.demos.hf_eager.hunyuanvideo_1_5.tt.vae_spatial import (
    BF16_TILE_BYTES,
    SDPA_L1_BUDGET_BYTES,
    SpatialShardPlan,
    attention_distributed_from_env,
    attention_sdpa_from_env,
    block_causal_chunk_plan,
    chunk_plan_peak_score_elements,
    largest_sdpa_k_chunk,
    plan_supports_rank_local_edge_fill,
    replicate_pad_to_plan,
    sdpa_cb_l1_bytes,
    sdpa_cb_tiles,
    sdpa_chunks_fit_l1,
)

# (latent_frames, latent_h, latent_w) actually decoded on Blackhole Galaxy.
PRODUCTION_LATENTS = {
    "480p": (31, 30, 53),
    "720p": (31, 45, 80),
}
GALAXY_MESH = (8, 4)
MID_BLOCK_CHANNELS = 1024
WAN_VAE_MID_CHANNELS = 96 * 4  # base_dim 96 x dim_mult[-1] 4, num_heads 1


# ---------------------------------------------------------------------------
# Torch mirrors of the TTNN ops the block is built from
# ---------------------------------------------------------------------------


def rms_norm(x_bthwc, gamma, eps=1e-12):
    """Mirror of ``RMSNorm``: mean-square over the channel axis only."""
    mean_square = (x_bthwc * x_bthwc).mean(dim=-1, keepdim=True)
    return x_bthwc * torch.rsqrt(mean_square + eps) * gamma


def pointwise_conv(x_bthwc, weight, bias):
    """Mirror of a kernel-1 ``CausalConv3d``: a per-position affine map.

    ``weight`` is the torch ``(Cout, Cin, 1, 1, 1)`` state-dict tensor.  With
    ``kt == 1`` the block's ``t_front`` and ``pad_hw`` are both zero, so the
    device op does no replicate padding and no neighbor exchange at all.
    """
    return x_bthwc @ weight.reshape(weight.shape[0], weight.shape[1]).transpose(0, 1) + bias


def chunked_attention(q_seq, k_seq, v_seq, plan, scale):
    """Mirror of ``AttnBlock._attend_blocks`` over ``(B, seq, C)`` operands."""
    blocks = []
    for chunk in plan:
        q_block = q_seq[:, chunk.q_start : chunk.q_stop, :]
        scores = torch.matmul(q_block, k_seq[:, : chunk.kv_stop, :].transpose(-2, -1)) * scale
        blocks.append(torch.matmul(torch.softmax(scores, dim=-1), v_seq[:, : chunk.kv_stop, :]))
    return torch.cat(blocks, dim=1)


def make_weights(channels, seed, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    weights = {"gamma": torch.rand(channels, generator=generator, dtype=dtype) + 0.5}
    for name in ("to_q", "to_k", "to_v", "proj_out"):
        weights[f"{name}.weight"] = (
            torch.randn(channels, channels, 1, 1, 1, generator=generator, dtype=dtype) / channels**0.5
        )
        weights[f"{name}.bias"] = torch.randn(channels, generator=generator, dtype=dtype) * 0.01
    return weights


# ---------------------------------------------------------------------------
# Replicated reference: what the current sharded path computes
# ---------------------------------------------------------------------------


def replicated_attn_block(x_bthwc, weights, q_chunk_tokens=0):
    """Every rank gathers H/W, computes the whole grid, then repartitions."""
    B, T, H, W, C = x_bthwc.shape
    n_hw = H * W
    h = rms_norm(x_bthwc, weights["gamma"])
    q = pointwise_conv(h, weights["to_q.weight"], weights["to_q.bias"])
    k = pointwise_conv(h, weights["to_k.weight"], weights["to_k.bias"])
    v = pointwise_conv(h, weights["to_v.weight"], weights["to_v.bias"])
    plan = block_causal_chunk_plan(T, n_hw, q_chunk_tokens)
    out = chunked_attention(
        q.reshape(B, T * n_hw, C), k.reshape(B, T * n_hw, C), v.reshape(B, T * n_hw, C), plan, C**-0.5
    )
    out = pointwise_conv(out.reshape(B, T, H, W, C), weights["proj_out.weight"], weights["proj_out.bias"])
    return out + x_bthwc


# ---------------------------------------------------------------------------
# Distributed simulation: one rank's storage, one rank's work
# ---------------------------------------------------------------------------


def canonicalize_local(local_bthwc, plan, rank_h, rank_w):
    """Torch mirror of ``canonicalize_replicated_shard_edges`` in BTHWC.

    The device version builds the same rank-local candidate everywhere and
    selects it through an H/W-fractured mask, so only storage-only tail cells
    on the final H/W ranks change.  H is repaired before W, which is what makes
    the bottom-right corner replicate the true logical corner.
    """
    if not plan_supports_rank_local_edge_fill(plan):
        raise ValueError("rank-local edge fill requires at least one logical row and column per rank")
    out = local_bthwc.clone()
    shard = plan.shard(rank_h, rank_w)
    if shard.logical_height < plan.local_height:
        out[:, :, shard.logical_height :, :, :] = out[:, :, shard.logical_height - 1 : shard.logical_height, :, :]
    if shard.logical_width < plan.local_width:
        out[:, :, :, shard.logical_width :, :] = out[:, :, :, shard.logical_width - 1 : shard.logical_width, :]
    return out


def stitch_bthwc(shards, plan):
    """Join rank-major equal-storage BTHWC shards into the padded grid."""
    rows = []
    for rank_h in range(plan.height_factor):
        start = rank_h * plan.width_factor
        rows.append(torch.cat(shards[start : start + plan.width_factor], dim=3))
    return torch.cat(rows, dim=2)


def distributed_attn_block(padded_bthwc, plan, weights, q_chunk_tokens=0):
    """Simulate every rank of the H/W-fractured formulation.

    Returns the rank-major list of local outputs, each still fractured, so a
    caller can check both the logical result and the storage-only tail cells.
    """
    B, T = padded_bthwc.shape[0], padded_bthwc.shape[1]
    C = padded_bthwc.shape[-1]
    scale = C**-0.5

    locals_in = []
    for rank_h in range(plan.height_factor):
        for rank_w in range(plan.width_factor):
            shard = plan.shard(rank_h, rank_w)
            local = padded_bthwc[:, :, shard.h_start : shard.h_stop, shard.w_start : shard.w_stop, :]
            locals_in.append(canonicalize_local(local, plan, rank_h, rank_w))

    # Every rank normalizes and projects only its own rows; K/V are then
    # all-gathered and cropped back to the logical grid, exactly as the device
    # path does, so the padded duplicate keys never enter any softmax.
    normed = [rms_norm(local, weights["gamma"]) for local in locals_in]
    k_full = stitch_bthwc([pointwise_conv(h, weights["to_k.weight"], weights["to_k.bias"]) for h in normed], plan)[
        :, :, : plan.logical_height, : plan.logical_width, :
    ]
    v_full = stitch_bthwc([pointwise_conv(h, weights["to_v.weight"], weights["to_v.bias"]) for h in normed], plan)[
        :, :, : plan.logical_height, : plan.logical_width, :
    ]
    kv_hw = plan.logical_height * plan.logical_width
    k_seq = k_full.reshape(B, T * kv_hw, C)
    v_seq = v_full.reshape(B, T * kv_hw, C)

    local_hw = plan.local_height * plan.local_width
    rank_plan = block_causal_chunk_plan(T, local_hw, q_chunk_tokens, kv_hw=kv_hw)
    outputs = []
    for local_in, h in zip(locals_in, normed):
        q_local = pointwise_conv(h, weights["to_q.weight"], weights["to_q.bias"])
        attended = chunked_attention(q_local.reshape(B, T * local_hw, C), k_seq, v_seq, rank_plan, scale)
        attended = attended.reshape(B, T, plan.local_height, plan.local_width, C)
        projected = pointwise_conv(attended, weights["proj_out.weight"], weights["proj_out.bias"])
        outputs.append(projected + local_in)
    return outputs


# ---------------------------------------------------------------------------
# The claim under test: Q never needs the all-gather
# ---------------------------------------------------------------------------


# Every partition here keeps at least one logical row and column on every
# rank, which is what the decoder's rank-local edge fill requires.
SUPPORTED_PARTITIONS = [
    ((3, 4, 4), (2, 2)),  # even in both axes
    ((4, 5, 7), (2, 2)),  # uneven H and W
    ((2, 10, 3), (4, 1)),  # H-only fracture, uneven
    ((3, 3, 10), (1, 4)),  # W-only fracture, uneven
    ((2, 7, 9), (4, 2)),  # uneven both, rectangular mesh
    ((2, 30, 53), (8, 4)),  # the real 480p latent grid on the real Galaxy mesh
]


@pytest.mark.parametrize("latent,mesh", SUPPORTED_PARTITIONS)
@pytest.mark.parametrize("q_chunk_tokens", [0, 1, 3, 1024])
def test_distributed_attention_matches_the_replicated_result(latent, mesh, q_chunk_tokens):
    """The rank decomposition is a rearrangement, not an approximation."""
    T, H, W = latent
    channels = 16
    weights = make_weights(channels, seed=T * 1000 + H * 31 + W)
    generator = torch.Generator().manual_seed(H * 97 + W)
    x = torch.randn(1, T, H, W, channels, generator=generator, dtype=torch.float64)

    expected = replicated_attn_block(x, weights, q_chunk_tokens=0)

    plan = SpatialShardPlan(H, W, mesh[0], mesh[1])
    padded = replicate_pad_to_plan(x, plan, h_dim=2, w_dim=3)
    actual = stitch_bthwc(distributed_attn_block(padded, plan, weights, q_chunk_tokens), plan)

    torch.testing.assert_close(actual[:, :, :H, :W, :], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("latent,mesh", [((4, 5, 7), (2, 2)), ((2, 7, 9), (4, 2)), ((2, 30, 53), (8, 4))])
def test_padded_storage_rows_come_out_as_replicas_of_the_last_logical_row(latent, mesh):
    """Replicate semantics survive without any explicit output edge repair.

    Given the shared keys and values, every remaining stage is a function of a
    single spatial position, so a padded row holding a copy of the last logical
    row necessarily produces a copy of that row's output.  This is the property
    the decoder's convolution halos rely on.

    The identity is exact in real arithmetic.  Host BLAS still blocks a matmul
    differently depending on which row an operand lands in, so bit-identical
    inputs can land a few ULP apart in float64; the tolerance here is four
    orders of magnitude tighter than any semantic difference would be, and the
    device path runs in bf16 where this is far below the noise floor.
    """
    T, H, W = latent
    channels = 16
    weights = make_weights(channels, seed=5)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, T, H, W, channels, generator=generator, dtype=torch.float64)

    plan = SpatialShardPlan(H, W, mesh[0], mesh[1])
    padded = replicate_pad_to_plan(x, plan, h_dim=2, w_dim=3)
    stitched = stitch_bthwc(distributed_attn_block(padded, plan, weights), plan)

    assert stitched.shape[2] == plan.padded_height
    assert stitched.shape[3] == plan.padded_width
    if plan.padded_height > H:
        torch.testing.assert_close(
            stitched[:, :, H:, :, :],
            stitched[:, :, H - 1 : H, :, :].expand(-1, -1, plan.padded_height - H, -1, -1),
            rtol=1e-12,
            atol=1e-12,
        )
    if plan.padded_width > W:
        torch.testing.assert_close(
            stitched[:, :, :, W:, :],
            stitched[:, :, :, W - 1 : W, :].expand(-1, -1, -1, plan.padded_width - W, -1),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("latent,mesh", [((4, 5, 7), (2, 2)), ((2, 7, 9), (4, 2))])
def test_corrupted_storage_cells_are_repaired_before_they_can_reach_a_key(latent, mesh):
    """Entry canonicalization makes the block independent of upstream hygiene.

    Padded storage cells are cropped out of K/V, but they are still this rank's
    own query and residual rows, so a block that trusted them would emit wrong
    padding into the next convolution's halo.
    """
    T, H, W = latent
    channels = 16
    weights = make_weights(channels, seed=8)
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(1, T, H, W, channels, generator=generator, dtype=torch.float64)

    plan = SpatialShardPlan(H, W, mesh[0], mesh[1])
    padded = replicate_pad_to_plan(x, plan, h_dim=2, w_dim=3)
    assert plan.padded_height > H or plan.padded_width > W, "case must actually have padding"

    corrupted = padded.clone()
    corrupted[:, :, H:, :, :] = 1e3
    corrupted[:, :, :, W:, :] = -1e3

    clean = stitch_bthwc(distributed_attn_block(padded, plan, weights), plan)
    repaired = stitch_bthwc(distributed_attn_block(corrupted, plan, weights), plan)
    torch.testing.assert_close(repaired, clean, rtol=0, atol=0)


def test_only_keys_and_values_are_gathered():
    """Perturbing another rank's input must not move this rank's query.

    If Q needed the all-gather, changing a remote row would change this rank's
    output through the query path as well as through K/V.  Zeroing the two
    key/value projections isolates the query path: the output must then be
    unaffected by remote rows entirely.
    """
    T, H, W = 3, 4, 4
    channels = 16
    weights = make_weights(channels, seed=12)
    plan = SpatialShardPlan(H, W, 2, 2)
    generator = torch.Generator().manual_seed(21)
    x = torch.randn(1, T, H, W, channels, generator=generator, dtype=torch.float64)

    perturbed = x.clone()
    perturbed[:, :, 2:, 2:, :] += 5.0  # rank (1, 1) only

    baseline = distributed_attn_block(replicate_pad_to_plan(x, plan, h_dim=2, w_dim=3), plan, weights)
    moved = distributed_attn_block(replicate_pad_to_plan(perturbed, plan, h_dim=2, w_dim=3), plan, weights)
    # With real K/V the remote change legitimately reaches every rank.
    assert not torch.allclose(baseline[0], moved[0])

    # Constant K/V make attention a constant map, leaving only the query path.
    frozen = dict(weights)
    frozen["to_k.weight"] = torch.zeros_like(weights["to_k.weight"])
    frozen["to_v.weight"] = torch.zeros_like(weights["to_v.weight"])
    frozen_baseline = distributed_attn_block(replicate_pad_to_plan(x, plan, h_dim=2, w_dim=3), plan, frozen)
    frozen_moved = distributed_attn_block(replicate_pad_to_plan(perturbed, plan, h_dim=2, w_dim=3), plan, frozen)
    torch.testing.assert_close(frozen_baseline[0], frozen_moved[0], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Partition legality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent,mesh", SUPPORTED_PARTITIONS)
def test_every_exercised_partition_keeps_a_logical_row_on_every_rank(latent, mesh):
    _, height, width = latent
    assert plan_supports_rank_local_edge_fill(SpatialShardPlan(height, width, mesh[0], mesh[1]))


@pytest.mark.parametrize("resolution", sorted(PRODUCTION_LATENTS))
def test_both_production_grids_partition_legally_on_the_galaxy_mesh(resolution):
    """The 480p and 720p grids are the shapes this actually has to serve."""
    _, height, width = PRODUCTION_LATENTS[resolution]
    plan = SpatialShardPlan(height, width, *GALAXY_MESH)
    assert plan_supports_rank_local_edge_fill(plan)
    print(
        f"[vae dist attn] {resolution} {height}x{width} on {GALAXY_MESH} -> "
        f"local {plan.local_height}x{plan.local_width}, storage tail "
        f"{plan.padded_height - height}x{plan.padded_width - width}",
        flush=True,
    )


@pytest.mark.parametrize(
    "height,width,mesh",
    [
        (5, 3, (4, 1)),
        (6, 9, (4, 2)),
        (7, 5, (8, 4)),
        # The geometry the device cases used to request: 5 rows over 4 H ranks
        # is 2-row shards with a 3-row tail, so rank 3 is entirely padding.
        (5, 7, (4, 2)),
    ],
)
def test_a_rank_made_entirely_of_padding_is_rejected(height, width, mesh):
    """These partitions have no valid row to replicate from, on host or device.

    The device path raises the same way inside
    ``canonicalize_replicated_shard_edges``; distribution does not change which
    partitions are legal, so they are excluded rather than special-cased.
    """
    plan = SpatialShardPlan(height, width, mesh[0], mesh[1])
    assert not plan_supports_rank_local_edge_fill(plan)
    x = torch.zeros(1, 1, height, width, 4, dtype=torch.float64)
    with pytest.raises(ValueError):
        distributed_attn_block(replicate_pad_to_plan(x, plan, h_dim=2, w_dim=3), plan, make_weights(4, seed=1))


# ---------------------------------------------------------------------------
# Plan structure and work division
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_frame,local_hw,kv_hw", [(1, 1, 4), (3, 8, 32), (31, 120, 3600), (31, 28, 1590)])
@pytest.mark.parametrize("q_chunk_tokens", [0, 1, 7, 1024])
def test_distributed_plan_partitions_only_the_local_query_sequence(n_frame, local_hw, kv_hw, q_chunk_tokens):
    plan = block_causal_chunk_plan(n_frame, local_hw, q_chunk_tokens, kv_hw=kv_hw)

    assert plan[0].q_start == 0
    assert plan[-1].q_stop == n_frame * local_hw
    for previous, current in zip(plan, plan[1:]):
        assert current.q_start == previous.q_stop
    for chunk in plan:
        assert chunk.q_start // local_hw == chunk.frame
        assert (chunk.q_stop - 1) // local_hw == chunk.frame
        assert chunk.kv_stop == (chunk.frame + 1) * kv_hw


def test_kv_hw_defaults_to_the_query_stride_so_existing_plans_are_unchanged():
    for n_frame, n_hw, q_chunk in ((5, 24, 0), (31, 1590, 1024), (3, 8, 7)):
        assert block_causal_chunk_plan(n_frame, n_hw, q_chunk) == block_causal_chunk_plan(
            n_frame, n_hw, q_chunk, kv_hw=n_hw
        )


def test_distributed_plan_rejects_a_non_positive_key_stride():
    with pytest.raises(ValueError):
        block_causal_chunk_plan(4, 8, 0, kv_hw=0)
    with pytest.raises(ValueError):
        block_causal_chunk_plan(4, 8, 0, kv_hw=-3)


@pytest.mark.parametrize("resolution", sorted(PRODUCTION_LATENTS))
def test_distribution_divides_attention_work_by_the_rank_count(resolution):
    """Total score elements fall by the mesh size, up to storage padding."""
    n_frame, height, width = PRODUCTION_LATENTS[resolution]
    n_hw = height * width
    plan = SpatialShardPlan(height, width, *GALAXY_MESH)
    ranks = GALAXY_MESH[0] * GALAXY_MESH[1]
    local_hw = plan.local_height * plan.local_width

    replicated = sum(chunk.q_len * chunk.kv_stop for chunk in block_causal_chunk_plan(n_frame, n_hw, 0))
    per_rank = sum(chunk.q_len * chunk.kv_stop for chunk in block_causal_chunk_plan(n_frame, local_hw, 0, kv_hw=n_hw))

    # Equal-storage padding is the only reason this is not exactly 1 / ranks.
    storage_overhead = (plan.padded_height * plan.padded_width) / n_hw
    assert per_rank * ranks == pytest.approx(replicated * storage_overhead, rel=1e-9)
    assert per_rank < replicated / (ranks * 0.75)
    print(
        f"[vae dist attn] {resolution} ranks={ranks} local_hw={local_hw} of {n_hw} "
        f"replicated_score_elems={replicated:,} per_rank={per_rank:,} "
        f"speedup={replicated / per_rank:.1f}x storage_overhead={storage_overhead:.3f}",
        flush=True,
    )


@pytest.mark.parametrize("resolution", sorted(PRODUCTION_LATENTS))
def test_distributed_peak_score_block_is_small_without_any_chunk_request(resolution):
    """One frame of rank-local queries is already a tiny block."""
    n_frame, height, width = PRODUCTION_LATENTS[resolution]
    n_hw = height * width
    plan = SpatialShardPlan(height, width, *GALAXY_MESH)
    local_hw = plan.local_height * plan.local_width
    peak = chunk_plan_peak_score_elements(block_causal_chunk_plan(n_frame, local_hw, 0, kv_hw=n_hw)) * 2
    assert peak < 64 * 1024**2
    print(
        f"[vae dist attn] {resolution} unchunked per-rank peak score block {peak / 1024**2:.1f} MiB",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Flash SDPA geometry
# ---------------------------------------------------------------------------


def test_sdpa_cb_model_matches_the_program_factory_arithmetic():
    """Hand-check one geometry against the C++ CB sizing block."""
    # Sq_chunk_t = 1, Sk_chunk_t = 8, DHt = 32.
    expected = (
        1 * 32 * 2  # q_tiles, double buffered
        + 8 * 32 * 2  # k_tiles
        + 8 * 32 * 2  # v_tiles
        + 2  # lightweight mask palette
        + 1 * 8  # qk_tiles
        + 1 * 32  # out_im_tiles
        + 1 * 32  # out0_t upper bound
        + 1  # statistics
        + 1  # scale
    )
    assert sdpa_cb_tiles(32, 256, 1024) == expected
    assert sdpa_cb_l1_bytes(32, 256, 1024) == expected * BF16_TILE_BYTES


def test_wan_key_chunk_does_not_transfer_to_the_hunyuan_head_dim():
    """Wan's q=32/k=256 preset fits at 384 channels and not at 1024."""
    assert sdpa_chunks_fit_l1(32, 256, WAN_VAE_MID_CHANNELS)
    assert not sdpa_chunks_fit_l1(32, 256, MID_BLOCK_CHANNELS)
    print(
        f"[vae sdpa geometry] q=32 k=256 wan(head_dim={WAN_VAE_MID_CHANNELS})="
        f"{sdpa_cb_l1_bytes(32, 256, WAN_VAE_MID_CHANNELS) / 1024:.0f} KiB "
        f"hunyuan(head_dim={MID_BLOCK_CHANNELS})="
        f"{sdpa_cb_l1_bytes(32, 256, MID_BLOCK_CHANNELS) / 1024:.0f} KiB "
        f"budget={SDPA_L1_BUDGET_BYTES / 1024:.0f} KiB",
        flush=True,
    )


def test_a_legal_key_chunk_exists_at_the_hunyuan_head_dim():
    """The geometry is usable; only the chunk size has to be derived."""
    k_chunk = largest_sdpa_k_chunk(MID_BLOCK_CHANNELS, q_chunk=32)
    assert k_chunk >= 32
    assert k_chunk % 32 == 0
    assert sdpa_chunks_fit_l1(32, k_chunk, MID_BLOCK_CHANNELS)
    assert not sdpa_chunks_fit_l1(32, k_chunk + 32, MID_BLOCK_CHANNELS)
    print(
        f"[vae sdpa geometry] head_dim={MID_BLOCK_CHANNELS} q=32 largest k_chunk={k_chunk} "
        f"({sdpa_cb_l1_bytes(32, k_chunk, MID_BLOCK_CHANNELS) / 1024:.0f} KiB)",
        flush=True,
    )


def test_the_head_dim_is_tile_aligned_so_sdpa_padding_rules_are_satisfied():
    """SDPA forbids padding on batch, num_heads, and head_dim.

    ``head_dim = 1024`` is 32 tiles exactly, and ``num_heads = 1`` trivially
    satisfies ``nqh >= nkv and nqh % nkv == 0``, so only the sequence axis is
    padded and that axis is the one SDPA explicitly allows to be padded.
    """
    assert MID_BLOCK_CHANNELS % 32 == 0
    assert MID_BLOCK_CHANNELS // 32 == 32


def test_sdpa_chunk_helpers_reject_non_tile_geometry():
    for bad in ((31, 256, 1024), (32, 200, 1024), (32, 256, 1000), (0, 256, 1024)):
        with pytest.raises(ValueError):
            sdpa_cb_tiles(*bad)


# ---------------------------------------------------------------------------
# Environment gates
# ---------------------------------------------------------------------------


def test_new_attention_gates_are_off_by_default():
    assert attention_distributed_from_env({}) is False
    assert attention_sdpa_from_env({}) is False
    assert attention_distributed_from_env({"HY_VAE_ATTN_DIST": "0"}) is False
    assert attention_sdpa_from_env({"HY_VAE_ATTN_SDPA": "0"}) is False


def test_new_attention_gates_opt_in_explicitly():
    assert attention_distributed_from_env({"HY_VAE_ATTN_DIST": "1"}) is True
    assert attention_sdpa_from_env({"HY_VAE_ATTN_SDPA": "1"}) is True


@pytest.mark.parametrize("raw", ["", "yes", "true", "2", "-1", "01"])
def test_new_attention_gates_fail_closed_on_bad_values(raw):
    with pytest.raises(ValueError):
        attention_distributed_from_env({"HY_VAE_ATTN_DIST": raw})
    with pytest.raises(ValueError):
        attention_sdpa_from_env({"HY_VAE_ATTN_SDPA": raw})


# ---------------------------------------------------------------------------
# Device cases (hardware required; not run during host validation)
# ---------------------------------------------------------------------------


def _attention_node(channels, generator):
    state = {"norm.gamma": torch.ones(channels)}
    for name in ("to_q", "to_k", "to_v", "proj_out"):
        weight = torch.randn(channels, channels, 1, 1, 1, generator=generator) / channels**0.5
        state[f"{name}.weight"] = weight
        state[f"{name}.bias"] = torch.randn(channels, generator=generator) * 0.01
    return SimpleNamespace(state_dict=lambda: state, in_channels=channels)


def _parallel_kwargs(mesh_device):
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager

    shape = tuple(mesh_device.shape)
    config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(shape[0], 0),
        width_parallel=ParallelFactor(shape[1], 1),
    )
    return config, CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)


def _replicated_bthwc(mesh_device, host_bthwc):
    return ttnn.from_torch(
        host_bthwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _dram_allocated_bytes(mesh_device):
    ttnn.synchronize_device(mesh_device)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.num_banks * view.total_bytes_allocated_per_bank


_FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}]

# Latent geometry paired with the mesh that partitions it. Every entry keeps at
# least one logical row and column on the final rank of each axis, which is what
# ``canonicalize_replicated_shard_edges`` needs to rebuild the replicate tail
# without a collective; ``test_every_hardware_case_partitions_legally`` enforces
# that on host so an illegal request cannot reach hardware again. The frame count
# is small on purpose: T only lengthens the causal prefix, while what these
# cases exercise is the H/W partition, and the replicated reference they are
# compared against is O((T*H*W)^2).
DEVICE_PARTITIONS = [
    ((4, 5, 7), (1, 2)),  # W-only fracture, uneven W
    ((3, 4, 8), (2, 2)),  # even in both axes
    ((3, 5, 8), (2, 2)),  # H uneven, W even
    ((3, 4, 7), (2, 2)),  # H even, W uneven
    ((4, 5, 7), (2, 2)),  # uneven in both axes
    # 7 rows over 4 ranks gives 2-row shards with a 1-row tail. 5 rows (the
    # geometry the other cases use) would give 2-row shards with a 3-row tail,
    # leaving rank 3 holding nothing but padding, which is an illegal partition
    # on host and on device alike.
    ((3, 7, 7), (4, 2)),  # uneven in both, four H ranks
]

# The grids actually decoded in production, on the real mesh. 30 rows over 8
# ranks gives 4-row shards with a 2-row tail and 45 gives 6-row shards with a
# 3-row tail, so both are legal; W divides evenly at 720p and not at 480p.
DEVICE_PRODUCTION_PARTITIONS = [
    ((2, 30, 53), (8, 4)),
    ((2, 45, 80), (8, 4)),
]


@pytest.mark.parametrize("latent,mesh", DEVICE_PARTITIONS + DEVICE_PRODUCTION_PARTITIONS)
def test_every_hardware_case_partitions_legally(latent, mesh):
    """Host guard on the device parameterization itself.

    The device cases previously asked for 5 rows on a 4-rank H axis, which left
    the final rank holding only padding; that is rejected by design, so the
    tests failed on an illegal request rather than on anything about the
    hardware. Keeping the parameter lists under a host assertion means the next
    such edit fails in a second, without a device.
    """
    _, height, width = latent
    plan = SpatialShardPlan(height, width, mesh[0], mesh[1])
    assert plan_supports_rank_local_edge_fill(plan), (
        f"{height}x{width} on {mesh} stores {plan.local_height}x{plan.local_width} per rank "
        f"({plan.padded_height}x{plan.padded_width} padded): a rank holds only padding"
    )


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("latent,mesh_device", DEVICE_PARTITIONS, indirect=["mesh_device"])
@pytest.mark.parametrize("q_chunk_tokens", [0, 4])
def test_device_distributed_attention_matches_the_replicated_path(mesh_device, latent, q_chunk_tokens):
    """Same weights and input, gather-Q versus gather-KV, on real silicon."""
    from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard

    generator = torch.Generator().manual_seed(1)
    channels = 32
    latent_t, latent_h, latent_w = latent
    source = torch.randn(1, latent_t, latent_h, latent_w, channels, generator=generator)
    torch_attn = _attention_node(channels, generator)

    reference = AttnBlock(torch_attn, device=mesh_device, attention_chunk_tokens=0, attention_distributed=False)
    reference_out = ttnn.to_torch(ttnn.get_device_tensors(reference(_replicated_bthwc(mesh_device, source)))[0]).float()

    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(latent_h, latent_w, config.height_parallel.factor, config.width_parallel.factor)
    distributed = AttnBlock(
        torch_attn,
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
        attention_chunk_tokens=q_chunk_tokens,
        attention_distributed=True,
    )
    sharded_input = typed_tensor_2dshard(
        replicate_pad_to_plan(source, plan, h_dim=2, w_dim=3),
        mesh_device,
        shard_mapping={config.height_parallel.mesh_axis: 2, config.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    dims = [None, None]
    dims[config.height_parallel.mesh_axis] = 2
    dims[config.width_parallel.mesh_axis] = 3
    out = distributed(sharded_input, latent_h, latent_w)
    host = fast_device_to_host(out, mesh_device, dims, ccl_manager=manager)

    ok, pcc = comp_pcc(reference_out, host[:, :, :latent_h, :latent_w, :], 0.999)
    print(
        f"[vae dist attn {latent_t}x{latent_h}x{latent_w} q={q_chunk_tokens} "
        f"mesh={tuple(mesh_device.shape)}] PCC={pcc}",
        flush=True,
    )
    assert ok, f"distributed attention PCC {pcc} < 0.999"

    # Replicate semantics on the storage-only tail, which the next halo reads.
    if plan.padded_height > latent_h:
        edge = host[:, :, latent_h - 1 : latent_h, :, :]
        tail = host[:, :, latent_h:, :, :]
        assert comp_pcc(edge.expand_as(tail), tail, 0.999)[0]


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("latent,mesh_device", DEVICE_PRODUCTION_PARTITIONS, indirect=["mesh_device"])
def test_device_distributed_attention_on_the_production_grids(mesh_device, latent):
    """The 480p 30x53 and 720p 45x80 partitions, on the real 8x4 mesh.

    Same comparison as above at the H/W geometry the decoder actually runs, so
    the uneven 8-rank H tail (2 rows at 480p, 3 at 720p) and the uneven 4-rank W
    tail (3 columns at 480p, none at 720p) are exercised on real ranks rather
    than only in the host simulation. The frame count is 2 rather than the
    production 31 because the replicated reference this is checked against is
    quadratic in `T*H*W`; the partition, which is what distribution changes, is
    independent of T. The channel count is likewise 32 rather than 1024 --
    `test_device_720p_distributed_attention_allocation_budget` covers the real
    width.
    """
    from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard

    generator = torch.Generator().manual_seed(11)
    channels = 32
    latent_t, latent_h, latent_w = latent
    source = torch.randn(1, latent_t, latent_h, latent_w, channels, generator=generator)
    torch_attn = _attention_node(channels, generator)

    reference = AttnBlock(torch_attn, device=mesh_device, attention_chunk_tokens=0, attention_distributed=False)
    reference_out = ttnn.to_torch(ttnn.get_device_tensors(reference(_replicated_bthwc(mesh_device, source)))[0]).float()

    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(latent_h, latent_w, config.height_parallel.factor, config.width_parallel.factor)
    distributed = AttnBlock(
        torch_attn,
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
        attention_chunk_tokens=0,
        attention_distributed=True,
    )
    sharded_input = typed_tensor_2dshard(
        replicate_pad_to_plan(source, plan, h_dim=2, w_dim=3),
        mesh_device,
        shard_mapping={config.height_parallel.mesh_axis: 2, config.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    dims = [None, None]
    dims[config.height_parallel.mesh_axis] = 2
    dims[config.width_parallel.mesh_axis] = 3
    host = fast_device_to_host(distributed(sharded_input, latent_h, latent_w), mesh_device, dims, ccl_manager=manager)

    ok, pcc = comp_pcc(reference_out, host[:, :, :latent_h, :latent_w, :], 0.999)
    print(
        f"[vae dist attn production {latent_h}x{latent_w} mesh={tuple(mesh_device.shape)} "
        f"local={plan.local_height}x{plan.local_width}] PCC={pcc}",
        flush=True,
    )
    assert ok, f"distributed attention PCC {pcc} < 0.999 at {latent_h}x{latent_w}"


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_device_distributed_attention_issues_no_mesh_partition(mesh_device):
    """The distributed output is already fractured, so nothing repartitions."""
    from models.tt_dit.utils.tensor import typed_tensor_2dshard

    generator = torch.Generator().manual_seed(2)
    channels = 32
    latent_h, latent_w = 5, 7
    source = torch.randn(1, 3, latent_h, latent_w, channels, generator=generator)
    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(latent_h, latent_w, config.height_parallel.factor, config.width_parallel.factor)
    block = AttnBlock(
        _attention_node(channels, generator),
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
        attention_distributed=True,
    )
    sharded_input = typed_tensor_2dshard(
        replicate_pad_to_plan(source, plan, h_dim=2, w_dim=3),
        mesh_device,
        shard_mapping={config.height_parallel.mesh_axis: 2, config.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    out = block(sharded_input, latent_h, latent_w)
    assert tuple(out.shape[2:4]) == (plan.local_height, plan.local_width)


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_device_720p_distributed_attention_allocation_budget(mesh_device):
    """The gate that previously OOMed, now with rank-local queries.

    Opt in with ``HY_VAE_ATTN_720P_GATE=1``; it allocates the real mid-block
    activations for ``(T, H, W) = (31, 45, 80)`` at 1024 channels.
    """
    from models.tt_dit.utils.tensor import typed_tensor_2dshard

    if os.environ.get("HY_VAE_ATTN_720P_GATE", "0") != "1":
        pytest.skip("set HY_VAE_ATTN_720P_GATE=1 to run the 720p attention budget gate")

    generator = torch.Generator().manual_seed(17)
    channels = MID_BLOCK_CHANNELS
    latent_t, latent_h, latent_w = PRODUCTION_LATENTS["720p"]
    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(latent_h, latent_w, config.height_parallel.factor, config.width_parallel.factor)
    source = torch.randn(1, latent_t, latent_h, latent_w, channels, generator=generator)

    block = AttnBlock(
        _attention_node(channels, generator),
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
        attention_chunk_tokens=int(os.environ.get("HY_VAE_ATTN_CHUNK", "0")),
        attention_distributed=True,
        attention_sdpa=os.environ.get("HY_VAE_ATTN_SDPA", "0") == "1",
    )
    sharded_input = typed_tensor_2dshard(
        replicate_pad_to_plan(source, plan, h_dim=2, w_dim=3),
        mesh_device,
        shard_mapping={config.height_parallel.mesh_axis: 2, config.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    before = _dram_allocated_bytes(mesh_device)
    out = block(sharded_input, latent_h, latent_w)
    ttnn.synchronize_device(mesh_device)
    delta = _dram_allocated_bytes(mesh_device) - before

    print(f"[vae 720p distributed attn] allocated delta {delta / 1024**3:.2f} GiB", flush=True)
    assert out.shape[1] == latent_t
    assert tuple(out.shape[2:4]) == (plan.local_height, plan.local_width)
    assert delta < 24_916_262_912


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("channels", [64, MID_BLOCK_CHANNELS])
def test_device_flash_sdpa_matches_the_matmul_blocks(mesh_device, channels):
    """Probe the open question: does the flash kernel accept 1 x head_dim?

    This is the test the device-owning agent should run first.  A failure here
    is expected to be a circular-buffer or kernel-geometry error rather than a
    numerical one, which is why the two paths are compared at a low PCC bar.
    """
    generator = torch.Generator().manual_seed(5)
    latent = (1, 5, 4, 6, channels)
    source = torch.randn(*latent, generator=generator)
    torch_attn = _attention_node(channels, generator)

    reference = AttnBlock(torch_attn, device=mesh_device, attention_chunk_tokens=0, attention_sdpa=False)
    reference_out = ttnn.to_torch(ttnn.get_device_tensors(reference(_replicated_bthwc(mesh_device, source)))[0]).float()

    flash = AttnBlock(torch_attn, device=mesh_device, attention_chunk_tokens=0, attention_sdpa=True)
    flash_out = ttnn.to_torch(ttnn.get_device_tensors(flash(_replicated_bthwc(mesh_device, source)))[0]).float()

    ok, pcc = comp_pcc(reference_out, flash_out, 0.999)
    print(f"[vae flash sdpa C={channels} k_chunk={flash._sdpa_k_chunk}] PCC={pcc}", flush=True)
    assert ok, f"flash SDPA PCC {pcc} < 0.999"
