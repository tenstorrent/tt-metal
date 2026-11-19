# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BF8 async all-reduce payload with persistent BF8 receive workspace."""
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as production

class BF8Collectives(production.MultichipCollectives):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ar_buffers = [ttnn.typecast(x, ttnn.bfloat8_b) for x in self.ar_buffers]

class BF8AR(production.MultichipDecoder):
    output_bf8 = False
    def decode_reduce(self, x):
        i = self.ar_index
        self.ar_index = 1-i
        x = ttnn.typecast(x, ttnn.bfloat8_b, memory_config=self.ar_mem)
        out = ttnn.experimental.all_reduce_async(
            x, self.ar_buffers[i], cluster_axis=1, mesh_device=self.device,
            multi_device_global_semaphore=self.ar_semaphores[i],
            memory_config=self.ar_mem,
            dtype=ttnn.bfloat8_b if self.output_bf8 else ttnn.bfloat16,
            topology=self.ccl.topology, num_links=self.ccl.num_links)
        if self.output_bf8:
            out = ttnn.typecast(out, ttnn.bfloat16)
        return out if self.native_local else ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

if __name__ == '__main__':
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    if '--output-bf8' in sys.argv:
        sys.argv.remove('--output-bf8')
        BF8AR.output_bf8 = True
    stack.MultichipCollectives = BF8Collectives
    stack.MultichipDecoder = BF8AR
    stack.main()
