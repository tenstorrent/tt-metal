# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Run DeepSeek V3 parallel Q/KV RMS norm test multiple times with different
seeds to verify build cache correctness across varying inputs.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.ops.descriptors.fusion import Parallel
from models.experimental.ops.descriptors.fusion.fusion import _BUILD_CACHE
from models.experimental.ops.descriptors.normalization import rms_norm


def torch_rms_norm(x, weight, eps=1e-5):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x / rms
    if weight is not None:
        out = out * weight
    return out


def cores(x1, y1, x2=None, y2=None):
    if x2 is None:
        x2, y2 = x1, y1
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x1, y1), ttnn.CoreCoord(x2, y2))})


class TestDeepSeekV3Cache:
    """Run the DeepSeek V3 parallel Q/KV RMS norm with different seeds to test cache."""

    NUM_ITERATIONS = 10
    PCC_THRESHOLD = 0.98

    def test_cache_across_seeds(self, device):
        """Build once, then re-run with different input data each iteration.

        Iteration 0 is the cold build. Iterations 1+ must hit the cache.
        PCC must be high on every iteration.
        """
        _BUILD_CACHE.clear()

        q_cores = cores(0, 0, 3, 3)  # 4x4 = 16 cores
        q_shard_w = 96
        q_total_w = 16 * q_shard_w  # 1536

        kv_cores = cores(5, 0, 6, 7)  # 2x8 = 16 cores
        kv_shard_w = 32
        kv_total_w = 16 * kv_shard_w  # 512

        q_shard_spec = ttnn.ShardSpec(q_cores, [32, q_shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        q_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=q_shard_spec,
        )
        q_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 4),
            subblock_w=q_shard_w // 32,
            block_h=1,
            block_w=q_shard_w // 32,
            inplace=False,
        )

        kv_shard_spec = ttnn.ShardSpec(kv_cores, [32, kv_shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        kv_mem = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=kv_shard_spec,
        )
        kv_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(2, 8),
            subblock_w=kv_shard_w // 32,
            block_h=1,
            block_w=kv_shard_w // 32,
            inplace=False,
        )

        cache_size_before_build = len(_BUILD_CACHE)

        for i in range(self.NUM_ITERATIONS):
            seed = 1000 + i * 37  # Distinct seed each iteration
            torch.manual_seed(seed)

            # Generate fresh random inputs
            torch_q_input = torch.rand(1, 1, 32, q_total_w, dtype=torch.bfloat16)
            torch_q_weight = torch.rand(1, 1, 1, q_total_w, dtype=torch.bfloat16)
            torch_kv_input = torch.rand(1, 1, 32, kv_total_w, dtype=torch.bfloat16)
            torch_kv_weight = torch.rand(1, 1, 1, kv_total_w, dtype=torch.bfloat16)

            # Move to device
            tt_q_input = ttnn.from_torch(torch_q_input, device=device, layout=ttnn.TILE_LAYOUT, memory_config=q_mem)
            tt_q_weight = ttnn.from_torch(torch_q_weight, device=device, layout=ttnn.TILE_LAYOUT)
            tt_kv_input = ttnn.from_torch(torch_kv_input, device=device, layout=ttnn.TILE_LAYOUT, memory_config=kv_mem)
            tt_kv_weight = ttnn.from_torch(torch_kv_weight, device=device, layout=ttnn.TILE_LAYOUT)

            # Build descriptors
            q_branch = rms_norm.rms_norm(
                tt_q_input,
                epsilon=1e-5,
                weight=tt_q_weight,
                memory_config=q_mem,
                core_range_set=q_cores,
                program_config=q_pc,
            )
            kv_branch = rms_norm.rms_norm(
                tt_kv_input,
                epsilon=1e-5,
                weight=tt_kv_weight,
                memory_config=kv_mem,
                core_range_set=kv_cores,
                program_config=kv_pc,
            )

            # Build + launch
            fused = Parallel(q_branch, kv_branch).build()
            fused.launch()

            # Check cache behavior
            if i == 0:
                # First iteration: cache miss → new entry
                assert (
                    len(_BUILD_CACHE) == cache_size_before_build + 1
                ), f"Iteration 0 should add one cache entry, got {len(_BUILD_CACHE)} (expected {cache_size_before_build + 1})"
                prev_desc_id = id(fused.descriptor)
            else:
                # Subsequent iterations: cache hit → no new entries
                assert (
                    len(_BUILD_CACHE) == cache_size_before_build + 1
                ), f"Iteration {i} should hit cache (still {cache_size_before_build + 1} entries), got {len(_BUILD_CACHE)}"
                # Isolation: each cache hit must return a fresh descriptor copy
                curr_desc_id = id(fused.descriptor)
                assert (
                    curr_desc_id != prev_desc_id
                ), f"[iter {i}] Cache hit returned same descriptor object (id={curr_desc_id}), expected fresh copy"
                prev_desc_id = curr_desc_id

            # Verify Q norm PCC
            q_golden = torch_rms_norm(torch_q_input.float(), torch_q_weight.float())
            q_result = ttnn.to_torch(q_branch.output_tensors[0])
            passing_q, pcc_q = comp_pcc(q_golden, q_result, pcc=self.PCC_THRESHOLD)
            assert passing_q, f"[iter {i}, seed {seed}] Q norm PCC too low: {pcc_q}"

            # Verify KV norm PCC
            kv_golden = torch_rms_norm(torch_kv_input.float(), torch_kv_weight.float())
            kv_result = ttnn.to_torch(kv_branch.output_tensors[0])
            passing_kv, pcc_kv = comp_pcc(kv_golden, kv_result, pcc=self.PCC_THRESHOLD)
            assert passing_kv, f"[iter {i}, seed {seed}] KV norm PCC too low: {pcc_kv}"

            print(f"  iter {i} seed={seed}: Q PCC={pcc_q:.6f}, KV PCC={pcc_kv:.6f}, cache_size={len(_BUILD_CACHE)}")

        print(f"\nAll {self.NUM_ITERATIONS} iterations passed with PCC >= {self.PCC_THRESHOLD}")
