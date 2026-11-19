# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic packed-versus-separate active-expert gate/up decode comparison.

Only the packed gate/up sparse matmul is replaced. Routing, selected indices,
down projections, grouped prefill, precision and all cache behavior are inherited.
This single-process diagnostic temporarily patches sparse_matmul while _moe runs;
it is not intended for concurrent calls or the production runtime.
"""

import argparse
from unittest.mock import patch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder


class ExpertSplitCandidateDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        split_experts=True,
        split_cores=16,
        split_block_w=16,
        split_subblock_w=1,
        **kwargs,
    ):
        if split_cores not in (8, 16):
            raise ValueError("split_cores must be 8 or 16")
        per_n = 512 // (32 * split_cores)
        if split_block_w <= 0 or 64 % split_block_w:
            raise ValueError("split_block_w must be a positive divisor of K/32=64")
        if split_subblock_w <= 0 or per_n % split_subblock_w:
            raise ValueError("split_subblock_w must divide per-core N")
        if kwargs.get("expert_dtype", "bfp8") != "bfp8":
            raise ValueError("this gate/up topology comparison requires BFP8 expert weights")
        self = super().from_state_dict(state_dict, **kwargs)
        self.split_experts = split_experts
        self.precision_policy["expert_split_candidate"] = dict(
            enabled=split_experts,
            cores=split_cores,
            block_w=split_block_w,
            subblock_w=split_subblock_w,
            per_core_n=per_n,
            scope="indexed gate/up sparse matmul only; grouped prefill unchanged",
        )
        if split_experts:
            if tuple(self.egu.shape) != (1, 256, 2048, 1024):
                raise ValueError(f"unexpected packed expert shape {tuple(self.egu.shape)}")
            # Setup-only device slicing preserves exactly the quantized packed
            # weights; no re-quantization or CPU transfer is part of the trial.
            self.split_gate_weight = ttnn.slice(
                self.egu, (0, 0, 0, 0), (1, 256, 2048, 512), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.split_up_weight = ttnn.slice(
                self.egu, (0, 0, 0, 512), (1, 256, 2048, 1024), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.split_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, split_cores // 8),
                in0_block_w=split_block_w,
                out_subblock_h=1,
                out_subblock_w=split_subblock_w,
                out_block_h=1,
                out_block_w=per_n,
                per_core_M=1,
                per_core_N=per_n,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
        return self

    def _moe(self, x, *, ordered_router=False):
        if not self.split_experts or x.shape[1] != 1:
            return super()._moe(x, ordered_router=ordered_router)
        original = ttnn.sparse_matmul

        def split_sparse(a, w, **kwargs):
            if w is not self.egu:
                return original(a, w, **kwargs)
            settings = dict(kwargs)
            settings["program_config"] = self.split_program
            settings["memory_config"] = ttnn.L1_MEMORY_CONFIG
            gate = original(a, self.split_gate_weight, **settings)
            up = original(a, self.split_up_weight, **settings)
            return ttnn.concat([gate, up], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        with patch.object(ttnn, "sparse_matmul", split_sparse):
            return super()._moe(x, ordered_router=ordered_router)


def main():
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import run_decoder

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-kwargs", default="{}")
    p.add_argument("--activation-file")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output")
    p.add_argument("--compare-functional", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--prefill-repeats", type=int, default=3)
    p.add_argument("--decode-repeats", type=int, default=30)
    run_decoder.FunctionalDecoder = ExpertSplitCandidateDecoder
    run_decoder.run(p.parse_args())


if __name__ == "__main__":
    main()
