# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _selector(self, x, w):
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        per_m = math.ceil(x.shape[-2] / 32 / 8)
        return ttnn.linear(
            x,
            w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=w.shape[-2] // 32,
                out_subblock_h=2 if per_m % 2 == 0 else 1,
                out_subblock_w=1,
                per_core_M=per_m,
                per_core_N=1,
                fuse_batch=True,
                mcast_in0=False,
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True
            ),
        )
