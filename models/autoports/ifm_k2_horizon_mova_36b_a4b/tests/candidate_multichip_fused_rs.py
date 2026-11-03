# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shape-faithful fused WO/down + reduce-scatter with next-consumer layouts."""

import math

import torch

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.multichip_candidate_base import MultichipDecoder as Base


class MultichipDecoder(Base):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        self.fused_buffers = {}
        for name in ["self_attn.o_proj"] + ([] if self.sparse else ["mlp.down_proj"]):
            self.fused_buffers[name] = [
                ttnn.from_torch(
                    torch.zeros(1, 1, 32, width),
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    dtype=getattr(ttnn, self.options.get("fused_dtype", "bfloat16")),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for width in (2560, 640)
            ]
        return self

    def _row_project(self, x, name):
        if x.shape[-2] != 32:
            return self._reduce(self._linear(x, name))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        intermediate, output = self.fused_buffers[name]
        k = x.shape[-1] // 32
        block = self.options.get("fused_k", 8)
        if k % block:
            raise ValueError("Fused block must divide K")
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=block,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=10,
            out_block_w=5,
            transpose_mcast=False,
            fuse_batch=False,
            fused_activation=None,
        )
        mm, rs = ttnn.experimental.matmul_reduce_scatter_async(
            x,
            self.weights[name + ".weight"],
            persistent_intermediate_buffer=intermediate,
            persistent_output_buffer=output,
            dim=3,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
            reduce_scatter_core_grid_offset=(0, 4),
            num_links=self.options["num_links"],
            memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            dtype=getattr(ttnn, self.options.get("fused_dtype", "bfloat16")),
            program_config=program,
            compute_kernel_config=self.matmul_compute[self._weight_group(name)],
        )
        return rs if self.residual_sharded else self._gather(rs)

    def _finish(self, residual, gate_logits, attention, *, valid_tokens=None):
        gated = ttnn.mul(
            attention,
            gate_logits,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, math.log(2), 20.0)],
        )
        projected = self._row_project(gated, "self_attn.o_proj")
        x = ttnn.add(residual, projected)
        normed = self._norm(self._gather(x) if self.residual_sharded else x, "post_attention_layernorm")
        return ttnn.add(x, self._mlp(normed, valid_tokens=valid_tokens))

    def _mlp(self, x, *, valid_tokens=None):
        if self.sparse:
            return super()._mlp(x, valid_tokens=valid_tokens)
        gu = self._linear(x, "mlp.gate_up")
        hidden = ttnn.mul(gu[..., :1536], gu[..., 1536:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return self._row_project(hidden, "mlp.down_proj")
