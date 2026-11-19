# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-prefill recheck of phase-dtype kernel sweep winners."""
import argparse
import math
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_sdpa_reference import MultichipDecoder

class LongPrefillGeometry(MultichipDecoder):
    selected=()
    def linear(self,x,name):
        group='input' if name=='packed_input' else 'down' if name=='mlp.down_proj' else 'output'
        n=x.shape[2]
        if n<512 or group not in self.selected:
            return super().linear(x,name)
        weight=self.weights[name+'.weight']
        gx,gy,kb=(11,10,16) if group=='input' else (8,8,16 if self.kind=='linear_attention' and group=='output' else 8)
        cfg=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(gx,gy),in0_block_w=kb,
            out_subblock_h=1,out_subblock_w=2,per_core_M=math.ceil(n/(32*gy)),
            per_core_N=math.ceil(weight.shape[-1]/(32*gx*2))*2,transpose_mcast=False,fused_activation=None)
        return ttnn.linear(ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG),weight,program_config=cfg,
               compute_kernel_config=self.matmul_compute(name),memory_config=ttnn.DRAM_MEMORY_CONFIG,dtype=ttnn.bfloat16)

def main():
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--roles',required=True);args,rest=ap.parse_known_args()
    LongPrefillGeometry.selected=args.roles.split(',')
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_probe as probe
    probe.MultichipDecoder=LongPrefillGeometry
    sys.argv=[sys.argv[0],*rest];probe.main()
if __name__=='__main__':main()
