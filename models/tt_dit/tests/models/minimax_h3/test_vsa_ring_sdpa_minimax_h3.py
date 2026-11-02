# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The ring-joint SDPA in Sol gather mode computes exactly the blocks the VSA pre-pass selects.

Synthetic Q/K/V laid out in a 4x8 ladder rung (pad rows repeat a real row of their tile, as in the model), the
device VSAPrepass mask, and the ring op without pooled K/V (exact step only), against the torch block-sparse
reference fed the same mask plus the op's block-0 fallback. The ragged geometry runs in a rung whose shards end in a
32-row straggler, and also without `sol_block_counts`, which must miss the bar: that proves the counts reach the
kernel.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.vsa_prepass import VSAPrepass, compute_topk
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MINIMAX_H3_BUCKET_LADDER_4X8
from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, build_vsa_geometry, vsa_min_padded_len
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING
from .vsa_oracle import VSA_INDEX_SENTINEL, fine_attention_batched

MIN_PCC = 0.999
SPARSITY = 0.9
NUM_HEADS = 4
HEAD_DIM = 128
K_CHUNK = 256
# A request the in-place check swaps in: 128 candidates, so a different top-k k than either case.
SWAPPED_IN = ((300, 0, 200), (8, 32, 32))


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def mask_to_kept(mask: torch.Tensor, geometry, nb_pad: int) -> torch.Tensor:
    """[H, S * nb_pad, S * nb_pad] 0/1 (local Q block rows per shard, padded global columns) -> [H, n_blocks,
    n_blocks] bool in global block ids."""
    slot = torch.arange(geometry.sp_factor * nb_pad)
    real_slot = (slot % nb_pad) < geometry.blocks_per_shard
    assert bool((mask[:, :, ~real_slot] == 0).all()), "the mask selects a padding column"
    kept = mask[:, real_slot][:, :, real_slot] != 0
    pad = geometry.valid_counts == 0
    assert not bool(kept[:, :, pad].any()), "the mask selects a pad block"
    assert not bool(kept[:, pad].any()), "a pad block's query row keeps a block"
    return kept


def with_shard_fallback(kept: torch.Tensor, geometry) -> torch.Tensor:
    """The op computes block 0 of a shard for a query block that keeps nothing there (one exact block per ring
    iteration); shards holding only pad blocks are skipped. [..., n_blocks] bool -> the same with that rule."""
    sp, tiles = geometry.sp_factor, geometry.blocks_per_shard
    per_shard = kept.reshape(*kept.shape[:-1], sp, tiles).clone()
    live = geometry.valid_counts.reshape(sp, tiles).sum(dim=-1) > 0
    per_shard[..., 0] |= ~per_shard.any(dim=-1) & live
    return per_shard.reshape(kept.shape)


def kept_to_indices(kept: torch.Tensor) -> torch.Tensor:
    """[H, n_blocks, n_blocks] bool -> oracle index rows [1, H, n_blocks, n_blocks], sentinel-tailed."""
    heads, n_tiles, _ = kept.shape
    indices = torch.full((1, heads, n_tiles, n_tiles), VSA_INDEX_SENTINEL, dtype=torch.int64)
    for h in range(heads):
        for r in range(n_tiles):
            cols = torch.nonzero(kept[h, r]).reshape(-1)
            indices[0, h, r, : cols.numel()] = cols
    return indices


@GALAXY_RING
@pytest.mark.parametrize(
    ("prefix_segments", "grid", "rung"),
    [
        pytest.param((512, 0, 0), (8, 16, 16), 18944, id="quick-aligned"),
        pytest.param((70, 5, 130), (9, 10, 13), 20224, id="quick-ragged-straggler"),
    ],
)
def test_vsa_ring_sdpa(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, prefix_segments, grid, rung, reset_seeds, expect_error
) -> None:
    sp = tuple(mesh_device.shape)[sp_axis]
    assert rung in MINIMAX_H3_BUCKET_LADDER_4X8 and rung >= vsa_min_padded_len(prefix_segments, grid, sp)
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=sp, padded_len=rung)
    ragged = bool((geometry.valid_counts[geometry.valid_counts > 0] < VSA_TILE_TOKENS).any())
    logger.info(
        f"{geometry.n_real_tiles} tiles in {geometry.n_blocks} blocks at rung {rung} "
        f"({geometry.rows_per_shard} rows/device), ragged={ragged}"
    )

    torch.manual_seed(0)
    packed = [torch.randn(geometry.seq_len, NUM_HEADS, HEAD_DIM).bfloat16().float() for _ in range(3)]
    # Packed [S, H, D] -> [1, H, rung, D] in device row order.
    q, k, v = (geometry.permute_metadata(x).permute(1, 0, 2).unsqueeze(0) for x in packed)

    rows = geometry.rows_per_shard
    to_shards = lambda x: x[0].reshape(NUM_HEADS, sp, rows, HEAD_DIM).permute(1, 0, 2, 3).contiguous()
    upload = lambda x: from_torch(
        to_shards(x), device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None]
    )
    tq, tk, tv = upload(q), upload(k), upload(v)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    prepass = VSAPrepass(
        rung=rung,
        sp_factor=sp,
        sparsity=SPARSITY,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        sp_axis=sp_axis,
    )
    prepass.update(geometry, traced=False)
    routing = prepass(tq, tk, tv, coarse=False)

    grid_size = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_size.x - 1, grid_size.y),
        q_chunk_size=VSA_TILE_TOKENS,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    dummy_joint = bf16_tensor(torch.zeros((1, NUM_HEADS, 0, HEAD_DIM)), device=mesh_device)
    dims = [None, None]
    dims[sp_axis] = 2
    dims[tp_axis] = 1
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))

    def run(with_counts: bool, **sol_extra) -> torch.Tensor:
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            dummy_joint,
            dummy_joint,
            dummy_joint,
            persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(tk.shape, 2, sp_axis, dtype=ttnn.bfloat16),
            persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(tv.shape, 2, sp_axis, dtype=ttnn.bfloat16),
            joint_strategy="rear",
            logical_n=geometry.logical_n,
            program_config=program_config,
            compute_kernel_config=compute_config,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(grid_size.x - 1, 0),
            use_column_major_ccl=True,
            sol_block_mask=routing["block_mask"],
            sol_block_counts=routing["block_counts"] if with_counts else None,
            sol_mode="gather",
            **sol_extra,
        )
        ttnn.synchronize_device(mesh_device)
        return ttnn.to_torch(out, mesh_composer=composer)[:, :NUM_HEADS].float()

    # The TP replicas of the mask are identical; keep the first.
    mask = ttnn.to_torch(routing["block_mask"], mesh_composer=composer)[0, :NUM_HEADS]
    indices = kept_to_indices(with_shard_fallback(mask_to_kept(mask, geometry, prepass.nb_pad), geometry))
    reference = fine_attention_batched(q, k, v, indices, geometry)

    real_rows = geometry.gather_index >= 0
    out = run(with_counts=True)
    assert torch.equal(out, run(with_counts=True)), "the cached program (rebound buffers) must reproduce the first call"
    pcc = _pcc(out[:, :, real_rows], reference[:, :, real_rows])
    density = (indices[0] != VSA_INDEX_SENTINEL).sum().item() / (NUM_HEADS * geometry.n_blocks**2)
    k = compute_topk(SPARSITY, int(geometry.is_candidate.sum()))
    logger.info(f"kept density {density:.3f}, k={k} of top-k width {prepass.k_max}: PCC with block counts {pcc:.6f}")
    assert pcc >= MIN_PCC, pcc

    if ragged:
        pcc_without = _pcc(run(with_counts=False)[:, :, real_rows], reference[:, :, real_rows])
        logger.info(f"PCC without block counts {pcc_without:.6f}")
        assert pcc_without < MIN_PCC, "pad rows of ragged tiles are attended without counts, so this must miss"

    with expect_error(RuntimeError, "do not pass sol_bias"):
        run(with_counts=True, sol_bias=bf16_tensor(torch.zeros((1, NUM_HEADS, 32, 32)), device=mesh_device))

    # The pipeline rewrites a rung's pre-pass in place for each request; it must route exactly as a fresh one.
    def routed_mask(p: VSAPrepass, qkv: tuple) -> torch.Tensor:
        r = p(*qkv, coarse=False)
        out = ttnn.to_torch(r["block_mask"], mesh_composer=composer)
        ttnn.deallocate(r["block_mask"])
        return out

    first = routed_mask(prepass, (tq, tk, tv))
    other = build_vsa_geometry(*SWAPPED_IN, sp_factor=sp, padded_len=rung)
    other_k = compute_topk(SPARSITY, int(other.is_candidate.sum()))
    assert other_k != k, "the swapped-in request must change k"
    # Its own Q/K/V in its own layout, as a real request has: distinct pooled keys, so no ties at the threshold.
    other_qkv = tuple(
        upload(other.permute_metadata(torch.randn(other.seq_len, NUM_HEADS, HEAD_DIM)).permute(1, 0, 2).unsqueeze(0))
        for _ in range(3)
    )
    fresh = VSAPrepass(
        rung=rung,
        sp_factor=sp,
        sparsity=SPARSITY,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        sp_axis=sp_axis,
    )
    fresh.update(other, traced=False)
    prepass.update(other, traced=True)
    other_mask = routed_mask(prepass, other_qkv)
    assert torch.equal(other_mask, routed_mask(fresh, other_qkv)), "an in-place update must route as a fresh pre-pass"
    assert not torch.equal(other_mask, first), "the other geometry must change the routing"
    other_kept = mask_to_kept(other_mask[0, :NUM_HEADS], other, prepass.nb_pad)
    scored_rows = (other.valid_counts > 0) & ~other.is_exempt
    kept_candidates = other_kept[:, scored_rows][:, :, other.is_candidate].sum(dim=-1)
    assert bool((kept_candidates == other_k).all()), "each scored row keeps exactly the request's k candidates"
    prepass.update(geometry, traced=True)
    assert torch.equal(routed_mask(prepass, (tq, tk, tv)), first), "updating back in place must restore the routing"
