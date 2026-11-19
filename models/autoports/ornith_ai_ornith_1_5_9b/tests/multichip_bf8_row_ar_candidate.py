# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Coherent BF8 row-projection outputs and AR payload, BF16 residual result."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_bf8_ar_candidate import BF8AR, BF8Collectives

class BF8RowAR(BF8AR):
    def decode_linear(self,x,name):
        if name not in ('linear_attn.out_proj','self_attn.o_proj','mlp.down_proj'):
            return super().decode_linear(x,name)
        im,om,cfg=self.dram_layouts[name]
        return ttnn.linear(ttnn.to_memory_config(x,im),self.dram_weights[name],
            program_config=cfg,compute_kernel_config=self.matmul_compute(name),
            memory_config=om,dtype=ttnn.bfloat8_b)

if __name__=='__main__':
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipCollectives=BF8Collectives
    stack.MultichipDecoder=BF8RowAR
    stack.main()
