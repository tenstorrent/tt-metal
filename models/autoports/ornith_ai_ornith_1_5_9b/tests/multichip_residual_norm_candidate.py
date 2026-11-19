# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure fused add+norm while preserving the residual needed after the MLP."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder

class ResidualNorm(MultichipDecoder):
    def decode_finish(self,x,mixed):
        residual=ttnn.add(x,mixed)
        h=ttnn.rms_norm(x,residual_input_tensor=mixed,
            weight=self.weights["post_attention_layernorm.weight"],
            epsilon=self.config.rms_norm_eps,program_config=self.norm_program,
            compute_kernel_config=self.compute,memory_config=self.ar_mem)
        gate=self.decode_linear(h,"mlp.gate_proj")
        up=self.decode_linear(h,"mlp.up_proj")
        h=ttnn.multiply(gate,up,input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(residual,self.decode_reduce(self.decode_linear(h,"mlp.down_proj")))

if __name__=="__main__":
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder=ResidualNorm
    stack.main()
