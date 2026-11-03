# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental DRAM weight sharding and phase-specific projection programs."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_optimized_base import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        import torch

        sd = args[0]
        prefix = f"model.layers.{self.layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in sd.items()}

        def put(name, value, group):
            self.weights[name] = ttnn.from_torch(
                value.contiguous(),
                device=self.mesh_device,
                dtype=getattr(ttnn, self.options.get(group + "_dtype", "bfloat8_b")),
                layout=ttnn.TILE_LAYOUT,
            )

        if self.options.get("split_attention"):
            self.attention_names = (
                ["self_attn.q_proj", "self_attn.k_proj"]
                + ([] if self.sparse else ["self_attn.v_proj"])
                + ["self_attn.gate_proj"]
                + (["self_attn.v_router"] if self.sparse else [])
            )
            for name in self.attention_names:
                put(name + ".weight", sd[name + ".weight"].T, "attention")
        if self.options.get("packed_dense") and not self.sparse:
            put(
                "mlp.gate_up.weight",
                torch.cat([sd["mlp.gate_proj.weight"].T, sd["mlp.up_proj.weight"].T], dim=-1),
                "mlp",
            )
        if self.sparse and self.options.get("split_shared"):
            put(
                "mlp.shared_gate_up.weight",
                torch.cat(
                    [sd["mlp.shared_experts.gate_proj.weight"].T, sd["mlp.shared_experts.up_proj.weight"].T], dim=-1
                ),
                "mlp",
            )
            router = torch.zeros(2560, 128, dtype=torch.bfloat16)
            router[:, :100] = sd["mlp.gate.weight"].T
            self.weights["mlp.gate.weight"] = ttnn.from_torch(
                router, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        grid = self.mesh_device.dram_grid_size()
        self.banks = grid.x * grid.y
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.decode_weights = {}
        for name, w in self.weights.items():
            if name.endswith(".weight") and w.shape[-2] > 1 and "swiglu" not in name:
                k, n = tuple(w.shape)[-2:]
                role = self._role(name.removesuffix(".weight"))
                readers = self.options.get(role + "_readers", self.options.get("readers", 1))
                shard_n = math.ceil(n / (32 * self.banks * readers)) * 32 * readers
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(dram_grid, [k, shard_n], ttnn.ShardOrientation.ROW_MAJOR),
                )
                if self.options.get("pad_dram") and shard_n * self.banks != n:
                    original_dtype = w.dtype
                    if original_dtype == ttnn.bfloat4_b:
                        w = ttnn.typecast(w, ttnn.bfloat16)
                    w = ttnn.pad(w, [(0, 0)] * (len(w.shape) - 1) + [(0, shard_n * self.banks - n)], 0)
                    if w.dtype != original_dtype:
                        w = ttnn.typecast(w, original_dtype)
                self.decode_weights[name] = ttnn.to_memory_config(w, memory)
        return self

    @staticmethod
    def _role(name):
        return (
            "attn"
            if name == "attention_packed"
            or name
            in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj", "self_attn.v_router")
            else ("output" if name == "self_attn.o_proj" else ("down" if "down_proj" in name else "up"))
        )

    def _linear(self, x, name, activation=None):
        if name == "attention_packed" and self.options.get("split_attention"):
            return ttnn.concat([self._linear(x, part) for part in self.attention_names], dim=-1)
        w = self.weights[name + ".weight"]
        k, n = tuple(w.shape)[-2:]
        compute = self.matmul_compute[self._weight_group(name)]
        if self.options.get("output_fidelity") and name == "self_attn.o_proj":
            compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, self.options["output_fidelity"]),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        if name == "mlp.gate" and self.options.get("router_interleaved", True):
            return ttnn.linear(
                x,
                w,
                compute_kernel_config=self.compute,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG if x.shape[-2] == 32 else ttnn.DRAM_MEMORY_CONFIG,
            )
        if x.shape[-2] == 32:
            cores = self.options.get("dense_cores", 8)
            grid = ttnn.num_cores_to_corerangeset(cores, self.mesh_device.compute_with_storage_grid_size(), True)
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, k // cores], ttnn.ShardOrientation.ROW_MAJOR),
            )
            x = ttnn.to_memory_config(x, mem)
            role = self._role(name)
            block = self.options.get("dram_" + role + "_block", k // 32 // cores)
            prog = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=math.ceil(self.decode_weights[name + ".weight"].shape[-1] / 32 / cores),
                num_workers_per_dram_bank=self.options.get(role + "_readers", self.options.get("readers", 1)),
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
            )
            out = ttnn.linear(
                x,
                self.decode_weights[name + ".weight"],
                program_config=prog,
                compute_kernel_config=compute,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            if (
                self.options.get("keep_sharded")
                and out.shape[-1] == n
                and (
                    name == "self_attn.o_proj"
                    or (not self.sparse and name.startswith("mlp.") and not self.options.get("packed_dense"))
                )
            ):
                return out
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
            return out[..., :n] if out.shape[-1] != n else out
        if self.options.get("prefill_2d"):
            grid = self.options.get("prefill_grid", [8, 8])
            m = math.ceil(x.shape[-2] / 32 / grid[1])
            nt = math.ceil(n / 32 / grid[0])
            sub = next(v for v in range(8, 0, -1) if nt % v == 0)
            prog = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=tuple(grid),
                in0_block_w=self.options.get("prefill_block", 8),
                out_subblock_h=1,
                out_subblock_w=sub,
                per_core_M=m,
                per_core_N=nt,
                transpose_mcast=False,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
            )
            return ttnn.linear(
                x,
                w,
                program_config=prog,
                compute_kernel_config=compute,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.linear(
            x,
            w,
            compute_kernel_config=compute,
            dtype=ttnn.bfloat16,
            activation=activation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mlp(self, x, *, valid_tokens=None):
        if self.sparse and self.options.get("split_shared"):
            gu = self._linear(x, "mlp.shared_gate_up")
            logits = self._linear(x, "mlp.gate")
            hidden = ttnn.mul(gu[..., :768], gu[..., 768:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            dense = self._linear(hidden, "mlp.shared_experts.down_proj")
            return ttnn.add(self._experts(x, valid_tokens=valid_tokens, logits=logits), dense)
        if self.sparse or x.shape[-2] > 32:
            return super()._mlp(x, valid_tokens=valid_tokens)
        if self.options.get("packed_dense"):
            packed = self._linear(x, "mlp.gate_up")
            hidden = ttnn.mul(
                packed[..., :6144], packed[..., 6144:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
        else:
            gate = self._linear(x, "mlp.gate_proj", activation="silu")
            hidden = ttnn.mul(gate, self._linear(x, "mlp.up_proj"))
        return self._linear(hidden, "mlp.down_proj")
