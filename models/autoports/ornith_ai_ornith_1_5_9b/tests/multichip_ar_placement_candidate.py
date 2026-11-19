# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same residual/buffer family; compare AR link count and NoC/worker placement."""
import argparse
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder

class ARPlacement(MultichipDecoder):
    links=2
    noc1=False
    custom_workers=False
    def decode_reduce(self,x):
        i=self.ar_index
        self.ar_index=1-i
        out=ttnn.experimental.all_reduce_async(
            x if self.movement_optimized and x.is_sharded() else ttnn.to_memory_config(x,self.ar_mem),
            self.ar_buffers[i],cluster_axis=1,mesh_device=self.device,
            multi_device_global_semaphore=self.ar_semaphores[i],
            memory_config=self.ar_mem,dtype=ttnn.bfloat16,
            topology=self.ccl.topology,num_links=self.links,
            use_noc1_only=self.noc1,use_optimal_ccl_for_llama=self.custom_workers)
        return out if self.native_local else ttnn.to_memory_config(out,ttnn.DRAM_MEMORY_CONFIG)

if __name__=='__main__':
    ap=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    ap.add_argument('--ar-links',type=int,choices=(1,2),default=2)
    ap.add_argument('--noc1-only',action='store_true')
    ap.add_argument('--custom-workers',action='store_true')
    args,rest=ap.parse_known_args()
    ARPlacement.links=args.ar_links;ARPlacement.noc1=args.noc1_only;ARPlacement.custom_workers=args.custom_workers
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder=ARPlacement
    sys.argv=[sys.argv[0],*rest]
    stack.main()
