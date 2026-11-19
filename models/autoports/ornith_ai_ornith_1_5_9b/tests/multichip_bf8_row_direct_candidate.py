# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Feed BF8 row projection shards directly to AR, preserving the producer grid."""
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_bf8_row_ar_candidate import BF8RowAR, BF8Collectives

class BF8RowDirect(BF8RowAR):
    def decode_reduce(self, x):
        i = self.ar_index
        self.ar_index = 1-i
        # A typecast cannot also change a sharded tensor's core count. The row
        # projection already produces BF8; AR supports its producer shard grid.
        if x.dtype != ttnn.bfloat8_b:
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        out = ttnn.experimental.all_reduce_async(
            x, self.ar_buffers[i], cluster_axis=1, mesh_device=self.device,
            multi_device_global_semaphore=self.ar_semaphores[i],
            memory_config=self.ar_mem, dtype=ttnn.bfloat16,
            topology=self.ccl.topology, num_links=self.ccl.num_links)
        return out if self.native_local else ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

if __name__ == '__main__':
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipCollectives = BF8Collectives
    stack.MultichipDecoder = BF8RowDirect
    stack.main()
