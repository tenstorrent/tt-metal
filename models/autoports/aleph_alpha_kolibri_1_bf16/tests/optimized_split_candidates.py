# SPDX-License-Identifier: Apache-2.0
"""Tuned separate same-input projections, compared against packed projections."""
import json
import math
import os

import torch

import ttnn

from ..tt.optimized_decoder import DRAM, OptimizedDecoder


class SplitCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        self.split_options = json.loads(os.environ.get("OPT_SPLIT", "{}"))
        w = {k.removeprefix(f"model.layers.{self.layer_idx}."): v for k, v in state_dict.items()}
        self.separate = {}
        dg = self.device.dram_grid_size()
        banks = dg.x * dg.y
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))})
        for role in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]:
            attention = role in ["q_proj", "k_proj", "v_proj"]
            value = w[("self_attn." if attention else "mlp.shared_experts.") + role + ".weight"].T.contiguous()
            opts = dict(dtype=self.policy.attention_dtype if attention else self.policy.shared_dtype)
            role_index = 0 if attention else 2
            k, n = value.shape
            weight = ttnn.from_torch(
                value,
                device=self.device,
                dtype=getattr(ttnn, opts["dtype"]),
                layout=ttnn.TILE_LAYOUT,
                memory_config=DRAM,
            )
            readers = self.split_options.get("readers", opts.get("readers", 1))
            shard_n = math.ceil(n / 32 / banks / readers) * 32 * readers
            physical_n = shard_n * banks
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, (k, shard_n), ttnn.ShardOrientation.ROW_MAJOR),
            )
            sharded = ttnn.from_torch(
                torch.nn.functional.pad(value, (0, physical_n - n)),
                device=self.device,
                dtype=weight.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory,
            )
            cores = self.split_options.get("cores", self.policy.projection_cores[role_index])
            kblock = self.split_options.get("kblock", self.policy.projection_k[role_index])
            grid = ttnn.num_cores_to_corerangeset(cores, self.device.compute_with_storage_grid_size(), row_wise=True)
            self.projection_info[id(weight)] = dict(
                role=role,
                weight=sharded,
                logical_n=n,
                options=opts,
                compute=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=self.policy.projection_fp32[role_index],
                    packer_l1_acc=True,
                ),
                input_mem=ttnn.create_sharded_memory_config(
                    (32, k // cores), grid, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
                ),
                program=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=kblock,
                    per_core_M=1,
                    per_core_N=math.ceil(physical_n / 32 / cores),
                    num_workers_per_dram_bank=readers,
                ),
            )
            self.separate[role] = weight
        if self.split_options.get("experts"):
            for role in ["gate_proj", "up_proj"]:
                value = torch.stack([w[f"mlp.experts.{e}.{role}.weight"].T for e in range(384)])[None]
                self.experts[role] = ttnn.from_torch(
                    value.contiguous(),
                    device=self.device,
                    dtype=getattr(ttnn, self.policy.expert_gate_up_dtype),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=DRAM,
                )
        return self

    def _linear(self, x, w, *, dtype=ttnn.bfloat16, activation=None):
        if self.split_options.get("qkv") and w is self.projections["qkv"] and x.shape[-2] <= 32:
            a = ttnn.to_memory_config(x, self.projection_info[id(self.separate["q_proj"])]["input_mem"])
            parts = [
                super(SplitCandidate, self)._linear(a, self.separate[role]) for role in ["q_proj", "k_proj", "v_proj"]
            ]
            parts = [ttnn.to_memory_config(part, ttnn.L1_MEMORY_CONFIG) for part in parts]
            packed = ttnn.concat(parts, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            grid = ttnn.num_cores_to_corerangeset(8, self.device.compute_with_storage_grid_size(), row_wise=True)
            mem = ttnn.create_sharded_memory_config(
                (32, 896), grid, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
            )
            return ttnn.to_memory_config(packed, mem)
        return super()._linear(x, w, dtype=dtype, activation=activation)

    def _indexed_moe(self, x, logits, ids):
        mem = ttnn.L1_MEMORY_CONFIG
        indices = ttnn.to_layout(
            ids if ids.dtype == ttnn.uint16 else ttnn.typecast(ids, ttnn.uint16),
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=mem,
        )
        sparsity = ttnn.slice(self.mask_zero_rm, (0, 0, 0, 0), (1, 1, 1, 384))

        def sparse(a, role, n, active=False):
            program = self._sparse_config(1, n)
            if role in ("gate_proj", "up_proj"):
                cores = self.split_options.get("expert_cores", 16)
                gx = min(8, cores)
                gy = cores // gx
                pn = math.ceil(n / 32 / cores)
                program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=self.split_options.get("gate_k", 40),
                    out_subblock_h=1,
                    out_subblock_w=pn,
                    out_block_h=1,
                    out_block_w=pn,
                    per_core_M=1,
                    per_core_N=pn,
                    fuse_batch=False,
                    mcast_in0=True,
                )
            return ttnn.reshape(
                ttnn.sparse_matmul(
                    a,
                    self.experts[role],
                    sparsity=sparsity,
                    indices=indices,
                    is_input_a_sparse=active,
                    is_input_b_sparse=True,
                    program_config=program,
                    compute_kernel_config=self.sparse_compute,
                    dtype=ttnn.bfloat16,
                    memory_config=mem,
                ),
                (1, 6, 1, n),
            )

        if self.split_options.get("experts"):
            gate = sparse(x, "gate_proj", 512)
            up = sparse(x, "up_proj", 512)
        else:
            both = sparse(x, "gate_up", 1024)
            gate = ttnn.slice(both, (0, 0, 0, 0), (1, 6, 1, 512), memory_config=mem)
            up = ttnn.slice(both, (0, 0, 0, 512), (1, 6, 1, 1024), memory_config=mem)
        middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        down = sparse(middle, "down_proj", 2560, True)
        if self.policy.router_score_bf16:
            logits = ttnn.typecast(logits, ttnn.bfloat16, memory_config=mem)
        scores = ttnn.sigmoid(ttnn.gather(logits, -1, index=ids, memory_config=mem), memory_config=mem)
        scores = ttnn.reshape(ttnn.permute(scores, (0, 3, 2, 1)), (1, 6, 1, 1))
        routed = ttnn.sum(ttnn.multiply(down, scores, memory_config=mem), dim=1, keepdim=True, memory_config=mem)
        if self.split_options.get("shared"):
            a = ttnn.to_memory_config(x, self.projection_info[id(self.separate["gate_proj"])]["input_mem"])
            gate = self._linear(a, self.separate["gate_proj"])
            up = self._linear(a, self.separate["up_proj"])
        else:
            both = self._linear(x, self.shared["gate_up"])
            gate = ttnn.slice(both, (0, 0, 0, 0), (1, 1, 1, 512))
            up = ttnn.slice(both, (0, 0, 0, 512), (1, 1, 1, 1024))
        middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(routed, self._linear(middle, self.shared["down_proj"]), dtype=ttnn.bfloat16)
