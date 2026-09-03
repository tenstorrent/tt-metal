# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
DistributedLayerNorm: vision counterpart of `tt_transformers.tt.distributed_norm.DistributedNorm`.

The vision tower's hidden dim (1152) is small enough that
`is_distributed_norm` returns False on T3K/QB2 (the LLM uses 4k as the cutoff).
For that regime the LLM's DistributedNorm just all-gathers the fractured
input back to replicated and runs a regular norm locally. We mirror that
pattern here, but with `LayerNorm` (mean + variance + scale + bias) instead
of `RMSNorm`, since the Qwen3.5 vision tower uses LayerNorm.

I/O contract (TP mode):
  in:  fractured along dim=-1 (1/TP of hidden on each device)
  out: replicated full-hidden tensor on every device
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

from .vision_layernorm import LayerNorm


class DistributedLayerNorm(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        state_dict,
        state_dict_prefix,
        tt_ccl,
        weight_cache_path=None,
        weight_dtype=ttnn.bfloat8_b,
        eps: float = 1e-05,
        ccl_topology=ttnn.Topology.Linear,
    ):
        super().__init__()
        self.tt_ccl = tt_ccl
        self.ccl_topology = ccl_topology
        self.is_multichip = device.__class__.__name__ == "MeshDevice" and device.get_num_devices() > 1

        # Use the existing replicated-weight LayerNorm under the hood.
        self.norm = LayerNorm(
            device=device,
            dim=dim,
            eps=eps,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            weight_dtype=weight_dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # If we're not multi-chip there is nothing to gather; keep the
        # behaviour identical to the existing replicated LayerNorm.
        if not self.is_multichip:
            return self.norm(x)

        # Gather the fractured hidden dim back into a replicated tensor.
        # Mirrors `DistributedNorm.forward` (non-TG, non-distributed-norm path):
        # all_gather along dim=3, then run a regular norm.
        #
        # NOTE: we do NOT deallocate `x` here. The caller (e.g. VisionBlock)
        # still needs the input tensor for the residual add and is responsible
        # for its lifetime, exactly the way `tt_transformers.tt.distributed_norm`
        # does it.
        gathered = ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.tt_ccl.get_num_links(1),
            topology=self.ccl_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        # Regular replicated LayerNorm on full hidden dim. The gathered buffer
        # is an intermediate we own; free it once the norm has produced its
        # own output buffer.
        out = self.norm(gathered)
        if out is not gathered:
            ttnn.deallocate(gathered)
        return out
