# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gather local attention heads, fused WO to sharded residual; next norm consumes it.

Compare to row-WO+RS. Global WO is column-sharded instead of row-sharded.
The consumer normalizes local residual shards before gathering normalized activations.
"""

import math

import torch

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_multichip_distributed_norm import (
    MultichipDecoder as Distributed,
)
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_stage5_fused_distributed import (
    MultichipDecoder as Decode,
)
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_stage5_multicore_combined import (
    MultichipDecoder as Multicore,
)


class MultichipDecoder(Distributed, Multicore):
    decode_forward = Decode.decode_forward

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        if not self.residual_sharded:
            raise ValueError("AGMM candidate requires sharded residual")
        prefix = f"model.layers.{self.layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        self.wo_column = ttnn.from_torch(
            sd["self_attn.o_proj.weight"].T.reshape(1, 1, 4096, 2560).contiguous(),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=3),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.ag_buffer = ttnn.from_torch(
            torch.zeros(1, 1, 32, 4096),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=getattr(ttnn, self.options.get("ag_dtype", "bfloat16")),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self

    def _finish(self, residual, gate_logits, attention, *, valid_tokens=None):
        if attention.shape[-2] != 32:
            return super()._finish(residual, gate_logits, attention, valid_tokens=valid_tokens)
        gated = ttnn.mul(
            attention,
            gate_logits,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, math.log(2), 20.0)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gated = ttnn.typecast(gated, getattr(ttnn, self.options.get("ag_dtype", "bfloat16")))
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(4, 4),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=5,
            transpose_mcast=False,
            fuse_batch=False,
            fused_activation=None,
        )
        gathered, projected = ttnn.experimental.all_gather_matmul_async(
            gated,
            self.wo_column,
            persistent_output_buffer=self.ag_buffer,
            dim=3,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
            all_gather_core_grid_offset=(0, 4),
            num_links=self.options["num_links"],
            memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            dtype=ttnn.bfloat16,
            program_config=program,
            compute_kernel_config=self.matmul_compute["attention"],
        )
        x = ttnn.add(residual, projected)
        normed = self._norm(x, "post_attention_layernorm")
        return ttnn.add(x, self._mlp(normed, valid_tokens=valid_tokens))
