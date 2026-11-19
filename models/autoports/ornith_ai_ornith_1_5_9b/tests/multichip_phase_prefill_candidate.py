# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Apply precision-matched S1024 sweep winners to complete decoder prefill."""
import argparse
import math
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder

class PhasePrefill(MultichipDecoder):
    phase_roles = 'both'
    def linear(self, x, name):
        if not 512 <= x.shape[2] <= 1024 or self.fused_prefill:
            return super().linear(x,name)
        is_input = name == 'packed_input'
        is_output = name in ('linear_attn.out_proj','self_attn.o_proj','mlp.down_proj')
        if not ((is_input and self.phase_roles in ('input','both')) or
                (is_output and self.phase_roles in ('output','both'))):
            return super().linear(x,name)
        gx,gy,sb = (11,10,1) if is_input and self.kind=='linear_attention' else (11,8,2) if is_input else (8,8,2)
        weight = self.weights[name+'.weight']
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx,gy), in0_block_w=16,
            out_subblock_h=1, out_subblock_w=sb,
            per_core_M=math.ceil(x.shape[2]/(32*gy)),
            per_core_N=math.ceil(weight.shape[-1]/(32*gx*sb))*sb,
            fused_activation=None,transpose_mcast=False)
        return ttnn.linear(ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG), weight,
            program_config=config,compute_kernel_config=self.matmul_compute(name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,dtype=ttnn.bfloat16)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    ap.add_argument('--phase-roles',choices=['control','input','output','both'],required=True)
    args,rest=ap.parse_known_args();PhasePrefill.phase_roles=args.phase_roles
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_probe as probe
    probe.MultichipDecoder=PhasePrefill
    sys.argv=[sys.argv[0],*rest];probe.main()
