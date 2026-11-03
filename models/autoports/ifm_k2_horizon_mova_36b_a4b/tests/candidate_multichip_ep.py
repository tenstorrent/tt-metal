# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TP4 attention/shared MLP and EP4 routed MLP, no token dispatch.

Replicated residual/router gives every rank the same inputs and routing scores.
Each rank owns25 experts, padded to32; sparse matmuls only execute selected local
experts. Local expert scores/reduction precede the existing shared+expert AR.
"""

import math

import torch

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_multichip_allreduce import MultichipDecoder as Base


class MultichipDecoder(Base):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        if not self.sparse:
            return self
        prefix = f"model.layers.{self.layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        banks = {"experts.gate_up": [], "experts.down_proj": []}
        selectors = []
        for rank in range(4):
            gu = torch.zeros(1, 32, 2560, 1536, dtype=torch.bfloat16)
            down = torch.zeros(1, 32, 768, 2560, dtype=torch.bfloat16)
            for e in range(25):
                expert = rank * 25 + e
                gu[0, e] = torch.cat(
                    [sd[f"mlp.experts.{expert}.gate_proj.weight"].T, sd[f"mlp.experts.{expert}.up_proj.weight"].T],
                    dim=-1,
                )
                down[0, e] = sd[f"mlp.experts.{expert}.down_proj.weight"].T
            banks["experts.gate_up"].append(gu)
            banks["experts.down_proj"].append(down)
            select = torch.zeros(1, 1, 128, 32, dtype=torch.bfloat16)
            select[0, 0, torch.arange(rank * 25, (rank + 1) * 25), torch.arange(25)] = 1
            selectors.append(select)
        for name, bank in banks.items():
            self.weights[name].deallocate(True)
            self.weights[name] = ttnn.from_torch(
                torch.cat(bank, dim=0),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
            )
            self.weight_manifest[name] = {
                "local_shape": list(bank[0].shape),
                "dtype": str(ttnn.bfloat4_b),
                "mapping": "EP4 25 owners +7 inert",
            }
        self.ep_selector = ttnn.from_torch(
            torch.cat(selectors, dim=0),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        return self

    def _experts(self, x, *, logits, value=False, valid_tokens=None):
        if value:
            return super()._experts(x, logits=logits, value=True, valid_tokens=valid_tokens)
        tokens = x.shape[-2]
        groups = tokens // 32
        routes = self._route(logits, "mlp.gate", 8, 100)
        if valid_tokens is not None and valid_tokens < tokens:
            routes = ttnn.pad(routes[:, :, :valid_tokens, :], [(0, 0), (0, 0), (0, tokens - valid_tokens), (0, 0)], 0)
        routes = ttnn.pad(routes, [(0, 0), (0, 0), (0, 0), (0, 28)], 0)
        local_routes = ttnn.linear(routes, self.ep_selector, dtype=ttnn.bfloat16, compute_kernel_config=self.compute)
        local_routes = ttnn.reshape(local_routes, (1, groups, 32, 32))
        sparsity = ttnn.to_layout(ttnn.sum(local_routes, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)

        def mm(inp, w, sparse, down=False):
            grid = self.options.get("ep_down_grid" if down else "ep_up_grid", [8, 5] if down else [8, 6])
            n = w.shape[-1]
            sub = 2 if down else 1
            prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=tuple(grid),
                in0_block_w=self.options.get("ep_down_block" if down else "ep_up_block", 24 if down else 40),
                out_subblock_h=1,
                out_subblock_w=sub,
                out_block_h=1,
                out_block_w=sub,
                per_core_M=1,
                per_core_N=math.ceil(n / 32 / math.prod(grid)),
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
            return ttnn.sparse_matmul(
                inp,
                w,
                sparsity=sparse,
                nnz=None,
                is_input_a_sparse=down,
                is_input_b_sparse=not down,
                program_config=prog,
                compute_kernel_config=self.matmul_compute["experts"],
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG if groups == 1 else ttnn.DRAM_MEMORY_CONFIG,
            )

        x = ttnn.reshape(x, (1, groups, 32, 2560))
        gu = mm(x, self.weights["experts.gate_up"], sparsity)
        product = ttnn.reshape(
            ttnn.mul(gu[..., :768], gu[..., 768:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            (groups, 32, 32, 768),
        )
        down = mm(product, self.weights["experts.down_proj"], ttnn.reshape(sparsity, (1, 1, groups, 32)), True)
        down = ttnn.reshape(down, (groups, 32, 32, 2560))
        scores = ttnn.reshape(ttnn.permute(local_routes, (0, 1, 3, 2)), (groups, 32, 32, 1))
        result = ttnn.experimental.fast_reduce_nc(ttnn.mul(down, scores), dims=[1])
        return ttnn.reshape(result, (1, 1, tokens, 2560))
