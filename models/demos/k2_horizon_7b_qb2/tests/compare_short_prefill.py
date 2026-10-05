"""Integrate measured short-prefill geometry in the unchanged full-model harness."""

import hashlib
import math
import runpy
import sys
from pathlib import Path

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_full_model_policy import OptimizedFullModelDecoder


def install(mode):
    original = MultichipDecoder._linear

    def linear(self, x, w):
        if not 32 < x.shape[2] < 256:
            return original(self, x, w)
        if w is self.wqkv:
            cores, block = (64, 32) if mode == "fast" else (16, 2)
        elif w is self.wo:
            cores, block = (32, 8) if mode == "fast" else (32, 2)
        elif w is self.wdown:
            cores, block = (32, 8) if mode == "fast" else (32, 2)
        elif w is self.wgate or w is self.wup:
            if w.dtype == ttnn.bfloat4_b:
                cores, block = 64, 32
            else:
                cores, block = (64, 8) if mode == "fast" else (32, 2)
        else:
            return original(self, x, w)
        m = math.ceil(x.shape[2] / 32)
        subblock = max(i for i in range(1, 5) if m % i == 0)
        activation = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG) if mode == "exact" and w is self.wdown else x
        return ttnn.linear(
            activation,
            w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._weight_compute(w),
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, cores // 8),
                in0_block_w=block,
                out_subblock_h=subblock,
                out_subblock_w=1,
                per_core_M=m,
                per_core_N=math.ceil(w.shape[-1] / 32 / cores),
                fuse_batch=True,
                mcast_in0=True,
            ),
        )

    MultichipDecoder._linear = linear
    OptimizedFullModelDecoder._linear = linear


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    assert mode in ("fast", "exact")
    print("SHORT_PREFILL_CANDIDATE", mode, hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    install(mode)
    runpy.run_module("models.demos.k2_horizon_7b_qb2.tests.run_full_model", run_name="__main__")
