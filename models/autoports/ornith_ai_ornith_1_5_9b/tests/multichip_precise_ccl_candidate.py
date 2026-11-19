# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Discriminating all-reduce accumulator precision experiment."""

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder


class PreciseCCLCandidate(MultichipDecoder):
    def decode_reduce(self, x):
        """Two-link asynchronous AR using the shared setup-owned receive buffers."""
        i = self.ar_index
        self.ar_index = 1 - i
        out = ttnn.experimental.all_reduce_async(
            x if self.movement_optimized and x.is_sharded() else ttnn.to_memory_config(x, self.ar_mem),
            self.ar_buffers[i],
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=self.ar_semaphores[i],
            memory_config=self.ar_mem,
            dtype=ttnn.bfloat16,
            topology=self.ccl.topology,
            num_links=self.ccl.num_links,
            fp32_dest_acc=True,
        )
        return out if self.native_local else ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
