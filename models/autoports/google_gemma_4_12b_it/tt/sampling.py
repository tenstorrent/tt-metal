# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device split sampling for the vocab-sharded Gemma 4 12B logits on the QB2 ring.

``SplitTopKSampling`` is ``models.common.sampling.TTSampling`` (persistent k / p / 1/T / seed tensors, the greedy
lowest-index tie-break, ``ttnn.sampling`` with ``tt_out_tok``) with a faster per-chip candidate stage for this
model's ``[1, 1, 32, 65536]`` BF16 shard:

* A one-tile-row row reduction runs on one core per tile row (``ttnn.topk`` over the 65536-wide row: 338 us).
  ``ttnn.experimental.view`` reinterprets the interleaved tile buffer as ``[1, 16, 32, 4096]`` (the same tiles in
  the same order, no data movement), so the level-1 top-32 runs over 16 tile rows at once.
* Level 1 keeps the top 64 of each chunk, so level 2 takes the top 32 of 16 x 64 = 1024 survivors (the multi-core
  top-k; 512 columns would take the single-core one) and carries the level-1 indices through ``indices_tensor``
  after adding the constant per-chip and per-chunk offsets, so its indices are already global token ids
  (``chip * 65536 + chunk * 4096 + column``): no offset add after the gather.
* Then the TTSampling tail: gather the 4 x 32 candidates (values, ids; async ring all-gathers with the barrier
  semaphore), lowest-global-id tie-break for greedy rows, ``manual_seed`` and ``ttnn.sampling`` into the persistent
  token buffer.

* Suppressed ids (``generation_config.suppress_tokens``) are masked among the level-1 candidates (int32 id compare,
  values to -inf): a chunk keeps its top 64, so at most 32 suppressed ids still leave its top 32 allowed tokens and
  the result equals masking the full logits, without a 4 MB mask pass per step.

Both levels keep every candidate that can reach a row's global top 32: a chunk's top 64 contains every element of
the chunk that is in the chip's top 32, and the chip's top 32 contains every element in the global top 32. Greedy
ties resolve to the lowest global id whenever a chip holds at most 32 logits tied at the row maximum, the guarantee
of TTSampling's tie-break (``_adjust_values_for_tiebreak``, KNOWN LIMITATION; doc/optimized_full_model README).
"""

import torch

import ttnn
from models.common.sampling.tt_sampling import TTSampling

LEVEL1_CHUNKS = 16
# Level-2 width 1024: at 512 columns (16 x top-32) ttnn.topk takes its single-core program (73 us); padding the 512
# survivors to 1024 costs 32 us of pads; keeping the top 64 per chunk at level 1 gives 1024 directly
# (sampler 354.8 / 347.9 / 331.8 us for 512 / pad / level-1 k = 64; doc/optimized_full_model work log 11)
LEVEL2_WIDTH = 1024


class SplitTopKSampling(TTSampling):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        chunks=LEVEL1_CHUNKS,
        gather_mode="async",
        level2_width=LEVEL2_WIDTH,
        level2_fill="level1_k",
        suppress_ids=(),
    ):
        super().__init__(mesh_device=mesh_device, tt_ccl=tt_ccl, args=args)
        num_shards = self._get_num_sampling_shards()
        self.local_vocab = self.padded_vocab_size // num_shards
        assert self.local_vocab % (chunks * ttnn.TILE_SIZE) == 0 and self.max_top_k == ttnn.TILE_SIZE
        self.chunks = chunks
        # level-2 input width: the 16 x 32 survivors, optionally padded (-inf values) to a wider power of two
        self.level2_width = level2_width or chunks * self.max_top_k
        # "pad": level 1 keeps the top 32 per chunk and the 512 survivors are padded; "level1_k": level 1 keeps the
        # top level2_width / chunks per chunk, so the level-2 input is already level2_width wide
        self.level1_k = self.level2_width // chunks if level2_fill == "level1_k" else self.max_top_k
        # suppressed token ids (generation_config.suppress_tokens) are masked among the level-1 candidates: a chunk
        # keeps its top level1_k, so after removing at most level1_k - 32 suppressed ids it still holds its top 32
        # allowed tokens, and the result equals masking the full logits (no 4 MB mask pass per step)
        self.suppress_ids = [int(t) for t in suppress_ids]
        assert len(self.suppress_ids) <= self.level1_k - self.max_top_k, "too many suppressed ids for level 1"
        self.chunk_width = self.local_vocab // chunks
        rows = self.max_batch_size
        off = torch.stack(
            [
                (d * self.local_vocab + torch.arange(chunks) * self.chunk_width)
                .reshape(1, chunks, 1, 1)
                .expand(1, chunks, rows, self.level1_k)
                for d in range(num_shards)
            ]
        ).reshape(num_shards, chunks, rows, self.level1_k)
        self.suppress_id_tensors = [
            ttnn.from_torch(
                torch.full((1, chunks, rows, self.level1_k), t, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for t in self.suppress_ids
        ]
        self.level1_offsets = ttnn.from_torch(
            off.to(torch.int32).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        # candidate gathers: "stock" (TTSampling's ttnn.all_gather), "async" (ring all_gather_async, fresh output,
        # barrier semaphore) or "persistent" (all_gather_async into persistent DRAM outputs; the op then skips its
        # barrier, all_gather_async_default_program_factory.cpp, so cross-invocation skew is unbounded: diagnostic)
        assert gather_mode in ("stock", "async", "persistent")
        self.gather_mode = gather_mode
        self.gather_bufs = None
        if gather_mode == "persistent":
            width = num_shards * self.max_top_k
            self.gather_bufs = {
                dt: ttnn.from_torch(
                    torch.zeros(1, 1, rows, width),
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                for dt in (ttnn.bfloat16, ttnn.int32)
            }

    def local_candidates(self, x):
        """Per-chip top-32 of the ``[1, 1, rows, 65536]`` shard: (BF16 values, INT32 global token ids), each
        ``[1, 1, rows, 32]``."""
        rows = x.shape[-2]
        v = ttnn.experimental.view(x, [1, self.chunks, rows, self.chunk_width])
        vals, idx = ttnn.topk(v, k=self.level1_k, dim=-1, largest=True, sorted=True)
        idx32 = ttnn.typecast(idx, ttnn.int32)
        ttnn.deallocate(idx)
        gidx = ttnn.add(idx32, self.level1_offsets, dtype=ttnn.int32)
        ttnn.deallocate(idx32)
        for sup in self.suppress_id_tensors:  # exact int32 compare of global ids, then -inf the suppressed values
            hit = ttnn.eq(gidx, sup, dtype=ttnn.bfloat16)
            masked = ttnn.where(hit, -1e30, vals)
            ttnn.deallocate(hit)
            ttnn.deallocate(vals)
            vals = masked
        width = self.chunks * self.level1_k
        l2_v = ttnn.experimental.view(vals, [1, 1, rows, width])
        l2_i = ttnn.experimental.view(gidx, [1, 1, rows, width])
        if self.level2_width != width:
            pad = [(0, 0), (0, 0), (0, 0), (0, self.level2_width - width)]
            l2_v, l2_i = ttnn.pad(l2_v, pad, value=-1e30), ttnn.pad(l2_i, pad, value=0)
            ttnn.deallocate(vals)
            ttnn.deallocate(gidx)
            vals, gidx = l2_v, l2_i
        top_v, top_i = ttnn.topk(l2_v, k=self.max_top_k, dim=-1, largest=True, sorted=True, indices_tensor=l2_i)
        ttnn.deallocate(vals)
        ttnn.deallocate(gidx)
        return top_v, top_i

    def _gather(self, t):
        if self.gather_mode == "async":
            return self.tt_ccl.all_gather(t, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.gather_mode == "stock":
            return self._perform_all_gather(
                t,
                dim=3,
                cluster_axis=self._get_sampling_cluster_axis(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=self.num_gather_links,
            )
        return self.tt_ccl.all_gather(
            t, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, persistent_output_buffer=self.gather_bufs[t.dtype]
        )

    def forward(self, x, tt_out_tok=None):
        assert not self._force_argmax_sampling and not self.log_probs_calculator.enable_log_probs
        assert (
            x.dtype == ttnn.bfloat16 and self.tt_invalid_vocab_mask is None and self.tt_invalid_vocab_tail_mask is None
        )
        top_v, top_i = self.local_candidates(x)
        values, ids = (self._gather(t) for t in (top_v, top_i))
        ttnn.deallocate(top_v)
        ttnn.deallocate(top_i)
        ids_u32 = ttnn.typecast(ids, ttnn.uint32)
        if self.gather_bufs is None:
            ttnn.deallocate(ids)
        ids_rm = ttnn.untilize(ids_u32, use_multicore=True)
        sampling_values = self._adjust_values_for_tiebreak(values, ids_u32)
        ttnn.manual_seed(seeds=self.seeds_tt_tensor, user_ids=self.user_ids_tt_tensor)
        tt_out_tok = ttnn.sampling(
            sampling_values, ids_rm, k=self.k_tensor, p=self.p_tensor, temp=self.temp_tensor, output_tensor=tt_out_tok
        )
        for t in (sampling_values, ids_u32, ids_rm) + (() if self.gather_bufs else (values,)):
            ttnn.deallocate(t)
        self.tt_log_probs = None
        return tt_out_tok, None
