# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase-specific value expert ownership: active EP banks, then scatter to KV owners."""

import math

import torch

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        if not self.sparse:
            return self
        prefix = f"model.layers.{self.layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        banks = []
        selectors = []
        for rank in range(4):
            ids = [local * 4 + rank for local in range(16)]
            banks.append(torch.stack([sd[f"self_attn.v_experts.{i}.weight"].T for i in ids]).unsqueeze(0))
            selector = torch.zeros(1, 1, 64, 32, dtype=torch.bfloat16)
            for local, i in enumerate(ids):
                selector[0, 0, i, local] = 1
            selectors.append(selector)
        for name, parts, dtype in [
            ("value_ep", banks, getattr(ttnn, self.options["value_dtype"])),
            ("value_ep_selector", selectors, ttnn.bfloat16),
        ]:
            self.weights[name] = ttnn.from_torch(
                torch.cat(parts, dim=0),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.weight_manifest[name] = dict(
                local_shape=list(parts[0].shape), dtype=str(dtype), mapping="EP4:16 real value experts"
            )
        return self

    def _experts(self, x, *, logits, value=False, valid_tokens=None):
        if not value or x.shape[-2] == 32 or valid_tokens is not None:
            return super()._experts(x, logits=logits, value=value, valid_tokens=valid_tokens)
        groups = x.shape[-2] // 32
        routes = self._route(logits, "self_attn.v_router", 4, 64)
        local = ttnn.linear(
            routes, self.weights["value_ep_selector"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute
        )[..., :16]
        local = ttnn.reshape(local, (1, groups, 32, 16))
        sparsity = ttnn.to_layout(ttnn.sum(local, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        grid = self.options.get("ep_value_grid", [8, 4])
        sub = self.options.get("ep_value_subblock", 1)
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=tuple(grid),
            in0_block_w=self.options.get("ep_value_block", 40),
            out_subblock_h=1,
            out_subblock_w=sub,
            out_block_h=1,
            out_block_w=sub,
            per_core_M=1,
            per_core_N=math.ceil(32 / math.prod(grid)),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        out = ttnn.sparse_matmul(
            ttnn.reshape(x, (1, groups, 32, 2560)),
            self.weights["value_ep"],
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            program_config=program,
            compute_kernel_config=self.matmul_compute["value"],
            dtype=getattr(ttnn, self.options.get("ep_value_activation", "bfloat8_b")),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.reshape(out, (groups, 16, 32, 1024))
        scores = ttnn.reshape(ttnn.permute(local, (0, 1, 3, 2)), (groups, 16, 32, 1))
        out = ttnn.experimental.fast_reduce_nc(
            ttnn.mul(out, scores, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), dims=[1]
        )
        out = ttnn.typecast(ttnn.reshape(out, (1, 1, groups * 32, 1024)), ttnn.bfloat16)
        return ttnn.experimental.reduce_scatter_minimal_async(
            out,
            dim=3,
            cluster_axis=1,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
            topology=ttnn.Topology.Ring,
            num_links=self.options["num_links"],
        )
