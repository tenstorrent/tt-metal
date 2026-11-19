# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapt whole-layer prefill geometry to larger chunks with a broader grid."""
import math
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_long_prefill_geometry_retry as original

class LargeChunkGeometry(original.WholeLayerGeometry):
    large_k_block = None
    def linear(self, x, name):
        group = "input" if name == "packed_input" else "down" if name == "mlp.down_proj" else "output"
        n = x.shape[2]
        if n <= 1024 or group not in self.selected:
            return super().linear(x, name)
        weight = self.weights[name + ".weight"]
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11,10), in0_block_w=self.large_k_block or (16 if group == "input" else 8),
            out_subblock_h=1, out_subblock_w=2, per_core_M=math.ceil(n/320),
            per_core_N=math.ceil(weight.shape[-1]/704)*2, transpose_mcast=False, fused_activation=None)
        return ttnn.linear(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG),weight,
            program_config=cfg,compute_kernel_config=self.matmul_compute(name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,dtype=ttnn.bfloat16)

def main():
    # Frozen geometry reference predates the verified CCL semaphore-grid fix.
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_sdpa_reference as reference
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import CCLManager
    reference.CCLManager=CCLManager
    original.WholeLayerGeometry=LargeChunkGeometry
    original.main()

if __name__ == "__main__":
    main()
