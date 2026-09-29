"""Stage7 working geometry; all stage6 numerical and residual policies retained."""

import math
from dataclasses import replace

import ttnn

from .full_model_policy import stage6_precision_policy
from .multichip_decoder import MultichipDecoder
from .optimized_decoder import MatmulGeometry


def optimized_full_model_policy(layer_index):
    base = stage6_precision_policy(layer_index)
    return replace(
        base,
        qkv_geometry=MatmulGeometry(8, 8, 2, True),
        o_geometry=MatmulGeometry(8, 4, 2, True),
        down_geometry=MatmulGeometry(8, 4, 2, True),
        mlp_geometry=replace(base.mlp_geometry, block_w=4) if base.mlp == "bfloat8_b" else base.mlp_geometry,
    )


class OptimizedFullModelDecoder(MultichipDecoder):
    """Measured prefill geometry, preserving all numerical policies."""

    def _prefill_fused_matmul_config(self, x, *, swiglu):
        if swiglu and (x.shape[2] <= 512 or self.policy.mlp == "bfloat4_b"):
            return ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=16,
                N_block_size=8,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            )
        return super()._prefill_fused_matmul_config(x, swiglu=swiglu)

    def _linear(self, x, w):
        if not 32 < x.shape[2] < 256:
            return super()._linear(x, w)
        if w is self.wqkv:
            cores, block = 64, 32
        elif w is self.wo or w is self.wdown:
            cores, block = 32, 8
        elif w is self.wgate or w is self.wup:
            cores, block = (64, 32) if w.dtype == ttnn.bfloat4_b else (64, 8)
        else:
            return super()._linear(x, w)
        m = math.ceil(x.shape[2] / 32)
        subblock = max(i for i in range(1, 5) if m % i == 0)
        return ttnn.linear(
            x,
            w,
            dtype=getattr(self, "matmul_output_dtype", ttnn.bfloat16),
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
