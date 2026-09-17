# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Row-split sampling top-k: give the per-device top-k more rows so it lands on more cores.

On Blackhole the sampler's ``ttnn.topk`` over the per-device logits routes to the
``topk_large_indices`` composite, which parallelises by ROW: one full logits row per
core. A 32-user block over a 32064-wide vocab shard is therefore 32 active cores out of
110, each walking 63 serial 512-element chunks (the op's Classic body, since more than 32
chunks do not fit its fused end-to-end body) -- ~138 us per token on a ~0.14 ms op.

The logits block is TILE [1, 1, 32, W] and ``W`` is a multiple of ``N * 32``, so the SAME
buffer can be re-described, with no data movement, as ``[1, 1, 32 * N, W / N]``: tile-row
``c`` of the new shape is columns ``[c * W/N, (c+1) * W/N)`` of the old one for every user.
The top-k then sees ``32 * N`` rows (``32 * N`` cores) that are ``N`` times narrower, and
for N = 2 the 32-chunk rows also take the fused body. Its ``[32 * N, 32]`` value and index
outputs are ``N`` tiles stacked vertically, which is byte-for-byte the ``[32, 32 * N]``
tile row the downstream sampler wants -- again a view. The only real change is the global
index offsets: candidate block ``c`` of shard ``d`` starts at ``d * W + c * W/N``.

The per-user candidate set becomes the union of the top-32 of each column chunk instead of
the top-32 of the row -- a strict SUPERSET of the row's top-32, so greedy and top-k <= 32
sampling see exactly the same candidates they did before. Only where the batch is one tile
tall and k one tile wide (the tile-view identities above), and only for the mesh-sharded
path; every other configuration keeps the stock sampler unchanged.

Installed by swapping the live sampler's class (see ``enable_row_split_topk``) so the
``SamplingGenerator`` / ``SeedManager`` references to the instance stay valid.
"""

import torch
from loguru import logger

import ttnn
from models.common.sampling._utils import topk_would_route_to_large_indices
from models.common.sampling.tt_sampling import TTSampling

TILE = 32


class RowSplitTTSampling(TTSampling):
    """``TTSampling`` whose mesh-sharded top-k runs on ``row_split`` x more rows (cores)."""

    _row_split = 0
    _row_split_offsets = None

    def _init_row_split(self, row_split: int) -> bool:
        self._row_split = 0
        self._row_split_offsets = None
        if row_split < 2:
            return False
        if self.multi_step_reduction or self.sub_core_grid_topk is not None or self._sampling_dp != 1:
            return False
        # The two tile-view identities need exactly one tile row of users and a one-tile k.
        if self.max_batch_size != TILE or self.max_top_k != TILE:
            return False
        num_shards = self._get_num_sampling_shards()
        per_device = self.padded_vocab_size // num_shards
        if per_device % (row_split * TILE) != 0:
            return False
        # ttnn.sampling needs a power-of-two number of candidate tiles.
        cand_tiles = num_shards * row_split
        if cand_tiles & (cand_tiles - 1):
            return False

        chunk = per_device // row_split
        offsets = torch.zeros(1, 1, self.max_batch_size, self.max_top_k * cand_tiles, dtype=torch.int64)
        for d in range(num_shards):
            for c in range(row_split):
                col = (d * row_split + c) * self.max_top_k
                offsets[:, :, :, col : col + self.max_top_k] = d * per_device + c * chunk
        self._row_split_offsets = ttnn.from_torch(
            offsets,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._row_split = row_split
        logger.info(f"Sampling top-k row split enabled: {row_split} x [{TILE}, {chunk}] rows per device")
        return True

    def forward(self, x: ttnn.Tensor, tt_out_tok: ttnn.Tensor = None):
        if not self._row_split or self._force_argmax_sampling:
            return super().forward(x, tt_out_tok)
        n = self._row_split
        k = self.max_top_k

        x_bf16 = (
            x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, dtype=ttnn.bfloat16, sub_core_grids=self.sub_core_grids)
        )
        x_bf16 = self._mask_invalid_vocab_logits(x_bf16)
        users, width = x_bf16.shape[-2], x_bf16.shape[-1]
        if users != TILE or width % (n * TILE) != 0:
            return super().forward(x, tt_out_tok)

        # Same buffer, N x the rows: tile-row c is column chunk c of every user.
        stacked = ttnn.experimental.view(x_bf16, (1, 1, users * n, width // n))
        routed = topk_would_route_to_large_indices(stacked, k)
        topk_values, topk_indices = ttnn.topk(
            stacked,
            k=k,
            dim=-1,
            stable=False if routed else self._topk_stable,
        )
        # [users*n, k] is n tiles stacked vertically == the [users, n*k] tile row, byte for byte.
        topk_values = ttnn.experimental.view(topk_values, (1, 1, users, n * k))
        topk_indices = ttnn.experimental.view(topk_indices, (1, 1, users, n * k))

        sampling_cluster_axis = self._get_sampling_cluster_axis()
        topk_values_gathered = self._perform_all_gather(
            topk_values,
            dim=3,
            cluster_axis=sampling_cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=self.num_gather_links,
            buffer_key="SAMPLING_VALUES",
        )
        ttnn.deallocate(topk_values)
        if self.sampling_memory_config != ttnn.DRAM_MEMORY_CONFIG:
            topk_values_gathered_bf16 = ttnn.to_memory_config(
                topk_values_gathered, memory_config=self.sampling_memory_config, dtype=ttnn.bfloat16
            )
            topk_values_gathered_bf16_interleaved = ttnn.to_memory_config(
                topk_values_gathered_bf16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(topk_values_gathered_bf16)
        else:
            topk_values_gathered_bf16_interleaved = topk_values_gathered

        topk_indices_gathered = self._perform_all_gather(
            topk_indices,
            dim=3,
            cluster_axis=sampling_cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=self.num_gather_links,
            buffer_key="SAMPLING_INDICES",
        )
        ttnn.deallocate(topk_indices)

        topk_indices_gathered_int32 = ttnn.typecast(
            topk_indices_gathered, dtype=ttnn.int32, sub_core_grids=self.sub_core_grids
        )
        if self.sampling_memory_config != ttnn.DRAM_MEMORY_CONFIG:
            topk_indices_gathered_int32_sharded = ttnn.to_memory_config(
                topk_indices_gathered_int32, self.sampling_memory_config
            )
            ttnn.deallocate(topk_indices_gathered_int32)
        else:
            topk_indices_gathered_int32_sharded = topk_indices_gathered_int32
        # Shard AND chunk offsets -> global vocab positions.
        topk_global_indices = ttnn.add(
            self._row_split_offsets,
            topk_indices_gathered_int32_sharded,
            dtype=ttnn.uint32,
            memory_config=self.sampling_memory_config,
        )
        ttnn.deallocate(topk_indices_gathered_int32_sharded)
        topk_global_indices_interleaved = ttnn.to_memory_config(topk_global_indices, ttnn.DRAM_MEMORY_CONFIG)

        topk_global_indices_interleaved_untilised = ttnn.untilize(
            topk_global_indices_interleaved, use_multicore=True, sub_core_grids=self.sub_core_grids
        )
        sampling_values = self._adjust_values_for_tiebreak(
            topk_values_gathered_bf16_interleaved, topk_global_indices_interleaved
        )
        ttnn.manual_seed(
            seeds=self.seeds_tt_tensor,
            user_ids=self.user_ids_tt_tensor,
            sub_core_grids=self._sampling_sub_core_grids,
        )
        tt_out_tok = ttnn.sampling(
            sampling_values,
            topk_global_indices_interleaved_untilised,
            k=self.k_tensor,
            p=self.p_tensor,
            temp=self.temp_tensor,
            sub_core_grids=self._sampling_sub_core_grids,
            output_tensor=tt_out_tok,
        )

        if self.log_probs_calculator.enable_log_probs and self.log_probs_calculator._use_topk_logprobs:
            self.tt_log_probs = self.log_probs_calculator.calculate_topk_log_probs(
                logits_tensor=x,
                topk_values=topk_values_gathered_bf16_interleaved,
                topk_global_indices=topk_global_indices_interleaved,
                sub_core_grid_topk=self.sub_core_grid_topk,
            )
        elif self.log_probs_calculator.enable_log_probs:
            self.tt_log_probs = self.log_probs_calculator.calculate_log_probs(x, tt_out_tok)
        else:
            self.tt_log_probs = None

        ttnn.deallocate(sampling_values)
        ttnn.deallocate(topk_values_gathered_bf16_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved_untilised)
        return tt_out_tok, self.tt_log_probs


def enable_row_split_topk(tt_sampling: TTSampling, row_split: int = 2) -> bool:
    """Upgrade a live ``TTSampling`` in place (its owners keep their references). Returns
    whether the split is active; when it is not, the instance behaves exactly as before."""
    if type(tt_sampling) is TTSampling:
        tt_sampling.__class__ = RowSplitTTSampling
    if not isinstance(tt_sampling, RowSplitTTSampling):
        return False
    return tt_sampling._init_row_split(row_split)
