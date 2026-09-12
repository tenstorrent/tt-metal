# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized single-mesh Qwen3.8-27B text decoder, with explicit caller-owned request state.

Inputs/outputs are TILE BF16 [batch, sequence, 5120]. Prefill accepts logical
lengths; it executes bounded chunks internally. Full attention uses paged BFP8
K/V with 32-token pages; linear attention uses persistent FP32 recurrent state
and a three-token convolution history. Decode accepts one token and device
INT32 current positions. Callers refresh input/position/RoPE/page-table tensors
before trace replay and restore state after warmup/capture.

Weight conversion and state allocation are setup boundaries. Forward methods
and their helpers contain only TTNN device operations and shape orchestration.
"""

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule

# Measured Blackhole 11x10 / eight-bank policy. Overrides are full experiment policies.
DEFAULT_POLICY = {
    "adaptive_sdpa": True,
    "attention_block": 2,
    "attention_cores": 80,
    "attention_dtype": "bfloat4_b",
    "attention_dram_batch": 8,
    "attention_fidelity": "LoFi",
    "attention_readers": 3,
    "carry_input": True,
    "carry_output": True,
    "carry_residual": True,
    "chunk_size": 2048,
    "down_block": 17,
    "down_cores": 32,
    "down_dtype": "bfloat4_b",
    "down_fidelity": "LoFi",
    "down_readers": 3,
    "dram": True,
    "dram_prefill_max": 32,
    "gate_block": 2,
    "gate_cores": 80,
    "gate_dtype": "bfloat4_b",
    "gate_epilogue": False,
    "gate_fidelity": "LoFi",
    "gate_readers": 3,
    "kv_dtype": "bfloat8_b",
    "minimal_n": 16,
    "minimal_prefill": True,
    "minimal_prefill_roles": ["output", "down"],
    "minimal_role_min": 128,
    "output_block": 4,
    "output_cores": 48,
    "output_dtype": "bfloat4_b",
    "output_fidelity": "LoFi",
    "output_readers": 3,
    "prefill_sdpa_k": 128,
    "prefill_sdpa_q": 128,
    "rectangular_working": True,
    "residual_cores": 80,
    "sdpa_k": 64,
    "sdpa_short_grid": [8, 2],
    "sharded_norm": True,
    "up_block": 2,
    "up_cores": 80,
    "up_dtype": "bfloat4_b",
    "up_fidelity": "LoFi",
    "up_readers": 3,
}


@dataclass
class DecoderState:
    key: object = None
    value: object = None
    recurrent: object = None
    conv: object = None


class OptimizedDecoder(LightweightModule):
    PAGE_SIZE = 32
    CHUNK_SIZE = DEFAULT_POLICY["chunk_size"]

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None):
        """Load an HF layer-local state dict (keys match Qwen3_5DecoderLayer)."""
        import torch

        self = cls()
        self.policy = dict(policy or DEFAULT_POLICY)
        for field, default in (("dtype", "bfloat8_b"), ("fidelity", "HiFi2"), ("readers", 1)):
            self.policy.setdefault("output_" + field, self.policy.get("attention_" + field, default))
        self.CHUNK_SIZE = self.policy.get("chunk_size", 128)
        self.config = hf_config
        self.device = mesh_device
        self.layer_idx = layer_idx
        self.kind = hf_config.layer_types[layer_idx]
        self.eps = hf_config.rms_norm_eps
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        if mesh_device.get_num_devices() != 1:
            raise ValueError("Optimized decoder requires a single-device mesh")
        if (hf_config.hidden_size, hf_config.intermediate_size) != (5120, 17408):
            raise ValueError("Expected the real Qwen3.8-27B text config")

        self.projection_configs = {}
        for role in ("attention", "output", "gate", "up", "down"):
            fidelity = self.policy.get(role + "_fidelity", "HiFi2")
            self.projection_configs[role] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

        def upload(tensor, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                tensor.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.weights = {}
        for name, tensor in state_dict.items():
            if tensor.ndim == 2:
                tensor = tensor.T
            elif name.endswith("layernorm.weight") or name in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                tensor = (tensor.float() + 1).reshape(1, 1, -1)
            elif name == "linear_attn.norm.weight":
                tensor = tensor.reshape(-1)
            elif tensor.ndim == 1:
                tensor = tensor.reshape(1, 1, -1)
            if name != "linear_attn.conv1d.weight":
                self.weights[name] = upload(tensor, self._weight_dtype(name) if tensor.ndim == 2 else ttnn.bfloat16)
        if self.kind == "linear_attention":
            conv = state_dict["linear_attn.conv1d.weight"]
            self.conv_taps = [upload(conv[:, 0, i].reshape(1, 1, -1)) for i in range(4)]
            self.a_neg = upload(-state_dict["linear_attn.A_log"].float().exp().reshape(1, 1, -1), ttnn.float32)
            self.dt_bias = upload(state_dict["linear_attn.dt_bias"].float().reshape(1, 1, -1), ttnn.float32)
            # Constants supplied explicitly: the native op's default builds them on host.
            c = 32
            masks = torch.zeros(1, 1, 32, 96)
            masks[:, :, :16, :16] = 1
            masks[:, :, 16:, 48:64] = 1
            masks[:, :, 16:, 64:80] = 1
            self.delta_constants = {
                "eye": upload(torch.eye(c).reshape(1, 1, c, c), ttnn.float32),
                "tril": upload(torch.ones(c, c).tril().reshape(1, 1, c, c), ttnn.float32),
                "ones": upload(torch.ones(1, 1, c, c), ttnn.float32),
                "masks": upload(masks, ttnn.float32),
            }
        if self.kind == "full_attention":
            c = hf_config
            qg = state_dict["self_attn.q_proj.weight"].reshape(c.num_attention_heads, 2, c.head_dim, c.hidden_size)
            packed = torch.cat(
                [
                    qg[:, 0].reshape(-1, c.hidden_size),
                    state_dict["self_attn.k_proj.weight"],
                    state_dict["self_attn.v_proj.weight"],
                    qg[:, 1].reshape(-1, c.hidden_size),
                ],
                dim=0,
            )
            self.weights["self_attn.qkvg.weight"] = upload(packed.T, self._weight_dtype("self_attn.qkvg.weight"))
            for name in ("q", "k", "v"):
                ttnn.deallocate(self.weights.pop(f"self_attn.{name}_proj.weight"))
        if self.kind == "linear_attention":
            names = ("qkv", "z", "b", "a")
            pieces = []
            for name in names:
                w = state_dict[f"linear_attn.in_proj_{name}.weight"]
                pieces.append(torch.nn.functional.pad(w, (0, 0, 0, (-w.shape[0]) % 32)))
            self.weights["linear_attn.packed.weight"] = upload(
                torch.cat(pieces, dim=0).T, self._weight_dtype("linear_attn.packed.weight")
            )
            for name in names:
                ttnn.deallocate(self.weights.pop(f"linear_attn.in_proj_{name}.weight"))
        if self.policy.get("packed_mlp", False):
            packed = torch.cat([state_dict["mlp.gate_proj.weight"], state_dict["mlp.up_proj.weight"]], dim=0)
            self.weights["mlp.gate_up.weight"] = upload(packed.T, self._weight_dtype("mlp.gate_proj.weight"))
        if self.policy.get("split_attention", False):
            if self.kind == "full_attention":
                c = hf_config
                qg = state_dict["self_attn.q_proj.weight"].reshape(c.num_attention_heads, 2, c.head_dim, c.hidden_size)
                pieces = {
                    "qg": torch.cat([qg[:, 0].reshape(-1, c.hidden_size), qg[:, 1].reshape(-1, c.hidden_size)], dim=0),
                    "k": state_dict["self_attn.k_proj.weight"],
                    "v": state_dict["self_attn.v_proj.weight"],
                }
                prefix = "self_attn"
            else:
                pieces = {
                    name: torch.nn.functional.pad(
                        state_dict[f"linear_attn.in_proj_{name}.weight"],
                        (0, 0, 0, (-state_dict[f"linear_attn.in_proj_{name}.weight"].shape[0]) % 32),
                    )
                    for name in ("qkv", "z", "b", "a")
                }
                prefix = "linear_attn"
            for name, tensor in pieces.items():
                self.weights[f"{prefix}.split_{name}.weight"] = upload(
                    tensor.T, self._weight_dtype(f"{prefix}.split_{name}.weight")
                )
        if self.policy.get("minimal_mlp", False):
            # Native fused SwiGLU consumes alternating gate/up column tiles.
            packed = torch.stack(
                [
                    state_dict["mlp.gate_proj.weight"].T.reshape(5120, -1, 32),
                    state_dict["mlp.up_proj.weight"].T.reshape(5120, -1, 32),
                ],
                dim=2,
            ).reshape(5120, -1)
            self.weights["mlp.interleaved_gate_up.weight"] = upload(packed, self._weight_dtype("mlp.gate_proj.weight"))
        self.dram_weights = {}
        if self.policy.get("dram", False):
            banks = mesh_device.dram_grid_size().x
            bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            for name, tensor in self.weights.items():
                if len(tensor.shape) != 2:
                    continue
                role = self._role(name)
                readers = self.policy.get(role + "_readers", 1)
                k, n = tensor.shape
                shard_width = ((n + 32 * banks * readers - 1) // (32 * banks * readers)) * 32 * readers
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(bank_grid, [k, shard_width], ttnn.ShardOrientation.ROW_MAJOR),
                )
                self.dram_weights[name] = ttnn.to_memory_config(tensor, memory)
        return self

    def allocate_state(self, *, batch_size, num_pages=None):
        """Setup only. Page ownership and page-table construction belong to caller."""

        def zeros(shape, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.zeros(
                shape, dtype=dtype, layout=layout, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        if self.kind == "full_attention":
            if num_pages is None or num_pages < 1:
                raise ValueError("Full attention requires num_pages")
            shape = [num_pages, self.config.num_key_value_heads, self.PAGE_SIZE, self.config.head_dim]
            return DecoderState(
                key=zeros(shape, getattr(ttnn, self.policy.get("kv_dtype", "bfloat16"))),
                value=zeros(shape, getattr(ttnn, self.policy.get("kv_dtype", "bfloat16"))),
            )
        c = self.config
        width = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
        return DecoderState(
            recurrent=zeros(
                [batch_size, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim], ttnn.float32
            ),
            conv=zeros([batch_size, 3, width], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        )

    def _role(self, name):
        return (
            (
                "gate"
                if name == "mlp.gate_up" or name == "mlp.gate_up.weight"
                else name.split(".")[1].removesuffix("_proj")
            )
            if name.startswith("mlp.")
            else ("output" if ".o_proj" in name or ".out_proj" in name else "attention")
        )

    def _weight_dtype(self, name):
        return getattr(ttnn, self.policy.get(self._role(name) + "_dtype", "bfloat8_b"))

    def _linear(self, x, name, activation=None, keep_sharded=False):
        if self.policy.get("split_attention", False):
            if name == "self_attn.qkvg":
                qg = self._linear(x, "self_attn.split_qg")
                width = self.config.num_attention_heads * self.config.head_dim
                return ttnn.concat(
                    [
                        qg[:, :, :width],
                        self._linear(x, "self_attn.split_k"),
                        self._linear(x, "self_attn.split_v"),
                        qg[:, :, width:],
                    ],
                    dim=-1,
                )
            if name == "linear_attn.packed":
                return ttnn.concat(
                    [self._linear(x, "linear_attn.split_" + part) for part in ("qkv", "z", "b", "a")], dim=-1
                )
        group = "mlp" if name.startswith("mlp.") else "attention"
        if self.policy.get(group + "_activation", "bfloat16") != "bfloat16":
            x = ttnn.typecast(x, getattr(ttnn, self.policy[group + "_activation"]))
        if self.policy.get("prefill_l1", False) and x.shape[1] > 1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if (
            self.policy.get("minimal_prefill", False) and x.shape[1] >= self.policy.get("minimal_prefill_min", 512)
        ) or (
            x.shape[1] >= self.policy.get("minimal_role_min", 128)
            and self._role(name) in self.policy.get("minimal_prefill_roles", [])
        ):
            return self._minimal(
                x, self.weights[name + ".weight"], self.projection_configs[self._role(name)], activation=activation
            )
        grid = self.device.compute_with_storage_grid_size()
        dram_prefill = (
            len(x.shape) == 3 and x.shape[1] > 1 and x.shape[0] * x.shape[1] <= self.policy.get("dram_prefill_max", 0)
        )
        if dram_prefill and x.shape[0] * x.shape[1] > 32:
            step = 32 // x.shape[0]
            return ttnn.concat(
                [self._linear(x[:, i : i + step, :], name, activation=activation) for i in range(0, x.shape[1], step)],
                dim=1,
            )
        if self.policy.get("dram", False) and (x.shape[1] == 1 or dram_prefill):
            public_shape = x.shape
            role = self._role(name)
            k, n = self.weights[name + ".weight"].shape
            cores = self.policy.get(role + "_cores", 8 if k == 17408 else 10 if k == 5120 else 12)
            shard_tiles = (k // 32 + cores - 1) // cores
            block = self.policy.get(role + "_block", shard_tiles)
            batch = x.shape[0] * x.shape[1] if len(x.shape) == 3 else x.shape[-2]
            memory = self._width_memory(cores, ((batch + 31) // 32) * 32, shard_tiles * 32)
            x = ttnn.reshape(x, [1, 1, batch, k])
            x = ttnn.to_memory_config(x, memory)
            program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=(batch + 31) // 32,
                per_core_N=(
                    self.dram_weights[name + ".weight"].memory_config().shard_spec.shape[1]
                    // 32
                    * self.device.dram_grid_size().x
                    + cores
                    - 1
                )
                // cores,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
                num_workers_per_dram_bank=self.policy.get(role + "_readers", 1),
            )
            output = ttnn.linear(
                x,
                self.dram_weights[name + ".weight"],
                compute_kernel_config=self.projection_configs[role],
                program_config=program,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            if dram_prefill:
                return ttnn.reshape(
                    ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG), [public_shape[0], public_shape[1], n]
                )
            if keep_sharded or (
                role == "output" and self.policy.get("carry_output", False) and self.policy.get("carry_residual", False)
            ):
                return output
            return self._public_rows(
                output,
                batch,
                n,
                keep_sharded=False,
                force_dram=role == "attention" and batch >= self.policy.get("attention_dram_batch", 10**9),
            )
        if self.policy.get("prefill_2d", False) and x.shape[1] >= self.policy.get("prefill_2d_min", 256):
            gx, gy = self.policy.get("prefill_grid", [8, 8])
            k, n = self.weights[name + ".weight"].shape
            per_m = ((x.shape[1] + 31) // 32 + gy - 1) // gy
            per_n = (n // 32 + gx - 1) // gx
            sub_w = next(v for v in (4, 2, 1) if per_n % v == 0)
            program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=self.policy.get("prefill_block", 4),
                per_core_M=per_m,
                per_core_N=per_n,
                out_subblock_h=1,
                out_subblock_w=sub_w,
                out_block_h=self.policy.get("prefill_out_block_h"),
                out_block_w=self.policy.get("prefill_out_block_w"),
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
            )
            return ttnn.linear(
                x,
                self.weights[name + ".weight"],
                program_config=program,
                compute_kernel_config=self.projection_configs[self._role(name)],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        return ttnn.linear(
            x,
            self.weights[name + ".weight"],
            compute_kernel_config=self.projection_configs[self._role(name)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            **({"activation": activation, "core_grid": ttnn.CoreGrid(x=grid.x, y=grid.y)} if activation else {}),
        )

    def _minimal(self, x, weight, compute, activation=None, fuse_swiglu=False):
        grid = self.device.compute_with_storage_grid_size()
        target = (
            ttnn.L1_MEMORY_CONFIG
            if self.policy.get("prefill_l1", False) and x.shape[1] > 1
            else ttnn.DRAM_MEMORY_CONFIG
        )
        x = ttnn.to_memory_config(x, target)
        config = ttnn.MinimalMatmulConfig(
            M_block_size=1 if x.shape[1] == 1 else self.policy.get("minimal_m", 4),
            K_block_size=self.policy.get("minimal_k", 8),
            N_block_size=self.policy.get("minimal_n", 8),
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=(grid.x, grid.y),
        )
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            config=config,
            compute_kernel_config=compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
            fuse_swiglu=fuse_swiglu,
        )

    def _width_memory(self, cores, height, width):
        if self.policy.get("rectangular_working", False) and cores % 10 == 0:
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, cores // 10 - 1))})
        else:
            grid = ttnn.num_cores_to_corerangeset(cores, self.device.compute_with_storage_grid_size(), row_wise=True)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [height, width], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _public_rows(self, tensor, batch, width, keep_sharded=True, force_dram=False):
        # Packed decode has ceil(B/32)*32 rows; public [B,1,H] has B*32.
        # Repack rows on device through an interleaved boundary for B>1.
        if batch > 1 or not keep_sharded:
            target = (
                ttnn.DRAM_MEMORY_CONFIG
                if force_dram or batch >= self.policy.get("public_dram_batch", 10**9)
                else ttnn.L1_MEMORY_CONFIG
            )
            tensor = ttnn.to_memory_config(tensor, target)
        return ttnn.reshape(tensor, [batch, 1, width])

    def _residual_memory(self, batch):
        cores = self.policy.get("residual_cores", 40)
        grid = (10, cores // 10)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))}),
                [((batch + 31) // 32) * 32, 5120 // cores],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _norm(self, x, name):
        if (
            self.policy.get("prefill_sharded_norm", False)
            and len(x.shape) == 3
            and x.shape[1] > 1
            and name.endswith("layernorm")
        ):
            batch, length, width = x.shape
            rows = batch * length
            cores = self.policy.get("residual_cores", 80)
            memory = self._residual_memory(rows)
            packed = ttnn.to_memory_config(ttnn.reshape(x, [1, 1, rows, width]), memory)
            shard_width = 160 // cores
            program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(10, cores // 10),
                subblock_w=min(shard_width, 4),
                block_h=(rows + 31) // 32,
                block_w=shard_width,
                inplace=False,
            )
            result = ttnn.rms_norm(
                packed,
                weight=self.weights[name + ".weight"],
                epsilon=self.eps,
                program_config=program,
                compute_kernel_config=self.ckc,
                memory_config=memory,
            )
            return ttnn.reshape(ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG), [batch, length, width])
        if self.policy.get("sharded_norm", False) and x.shape[1] == 1 and name.endswith("layernorm"):
            batch = x.shape[0] if len(x.shape) == 3 else x.shape[-2]
            public = len(x.shape) == 3
            memory = self._residual_memory(batch)
            cores = self.policy.get("residual_cores", 40)
            grid = (10, cores // 10)
            x = ttnn.to_memory_config(ttnn.reshape(x, [1, 1, batch, 5120]), memory)
            width = 160 // cores
            program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid,
                subblock_w=min(width, 4),
                block_h=(batch + 31) // 32,
                block_w=width,
                inplace=False,
            )
            output = ttnn.rms_norm(
                x,
                weight=self.weights[name + ".weight"],
                epsilon=self.eps,
                program_config=program,
                compute_kernel_config=self.ckc,
                memory_config=memory,
            )
            if public:
                return self._public_rows(output, batch, 5120, keep_sharded=self.policy.get("carry_residual", False))
            return output
        return ttnn.rms_norm(
            x,
            weight=self.weights[name + ".weight"],
            epsilon=self.eps,
            compute_kernel_config=self.ckc if self.kind == "linear_attention" else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _finish(self, x, attention):
        if self.policy.get("carry_residual", False) and self.policy.get("dram", False) and x.shape[1] == 1:
            memory = self._residual_memory(x.shape[0])
            shape = x.shape
            x = ttnn.to_memory_config(ttnn.reshape(x, [1, 1, shape[0], 5120]), memory)
            attention = ttnn.to_memory_config(ttnn.reshape(attention, [1, 1, shape[0], 5120]), memory)
            h = ttnn.add(x, attention, memory_config=memory)
            n = self._norm(h, "post_attention_layernorm")
            if self.policy.get("minimal_mlp", False):
                product = self._minimal(
                    n, self.weights["mlp.interleaved_gate_up.weight"], self.projection_configs["gate"], fuse_swiglu=True
                )
            elif self.policy.get("packed_mlp", False):
                packed = self._linear(n, "mlp.gate_up")
                width = self.config.intermediate_size
                product = ttnn.mul(
                    packed[:, :, :width],
                    packed[:, :, width:],
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                gate = self._linear(
                    n,
                    "mlp.gate_proj",
                    activation="silu" if self.policy.get("gate_epilogue", True) else None,
                    keep_sharded=True,
                )
                up = self._linear(n, "mlp.up_proj", keep_sharded=True)
                product = ttnn.mul(
                    gate,
                    up,
                    memory_config=gate.memory_config(),
                    input_tensor_a_activations=(
                        [] if self.policy.get("gate_epilogue", True) else [ttnn.UnaryOpType.SILU]
                    ),
                )
            down = self._linear(product, "mlp.down_proj", keep_sharded=True)
            down = ttnn.to_memory_config(down, memory)
            return self._public_rows(ttnn.add(h, down, memory_config=memory), shape[0], 5120)
        h = ttnn.add(x, attention)
        n = self._norm(h, "post_attention_layernorm")
        if self.policy.get("minimal_mlp", False):
            product = self._minimal(
                n, self.weights["mlp.interleaved_gate_up.weight"], self.projection_configs["gate"], fuse_swiglu=True
            )
        elif self.policy.get("packed_mlp", False):
            packed = self._linear(n, "mlp.gate_up")
            width = self.config.intermediate_size
            product = ttnn.mul(
                packed[:, :, :width], packed[:, :, width:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
        else:
            gate = self._linear(
                n, "mlp.gate_proj", activation="silu" if self.policy.get("gate_epilogue", True) else None
            )
            up = self._linear(n, "mlp.up_proj")
            product = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[] if self.policy.get("gate_epilogue", True) else [ttnn.UnaryOpType.SILU],
            )
        return ttnn.add(h, self._linear(product, "mlp.down_proj"))

    def _qkv(self, x, cos, sin):
        b, t, _ = x.shape
        c = self.config
        packed = self._linear(x, "self_attn.qkvg")
        q_width, kv_width = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
        if t == 1:
            grid = self.device.compute_with_storage_grid_size()
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.num_cores_to_corerangeset(b, grid, row_wise=True),
                    [32, c.head_dim],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.reshape(packed[:, :, : q_width + 2 * kv_width], [1, 1, b, q_width + 2 * kv_width]),
                num_heads=c.num_attention_heads,
                num_kv_heads=c.num_key_value_heads,
                memory_config=memory,
            )
            q, k, v = [ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG) for a in (q, k, v)]
            q = self._norm(q, "self_attn.q_norm")
            k = self._norm(k, "self_attn.k_norm")
            q, k = self._rope_decode(q, cos, sin), self._rope_decode(k, cos, sin)
            q = ttnn.reshape(q, [b, c.num_attention_heads, 1, c.head_dim])
            k = ttnn.reshape(k, [b, c.num_key_value_heads, 1, c.head_dim])
            v = ttnn.reshape(v, [b, c.num_key_value_heads, 1, c.head_dim])
        else:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                packed[:, :, : q_width + 2 * kv_width],
                num_heads=c.num_attention_heads,
                num_kv_heads=c.num_key_value_heads,
                transpose_key=False,
            )
            q = self._norm(q, "self_attn.q_norm")
            k = self._norm(k, "self_attn.k_norm")
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        gate = packed[:, :, q_width + 2 * kv_width :]
        return q, k, v, ttnn.reshape(gate, [b, t, c.num_attention_heads * c.head_dim])

    def _rope(self, x, cos, sin):
        b, _, length, _ = x.shape
        rotary_width = cos.shape[-1]
        outputs = []
        for user in range(b):
            part = x[user : user + 1, :, :, :rotary_width]
            cc = ttnn.reshape(cos[user : user + 1], [1, 1, length, rotary_width])
            ss = ttnn.reshape(sin[user : user + 1], [1, 1, length, rotary_width])
            rotated = ttnn.experimental.rotary_embedding(part, cc, ss)
            rotated = ttnn.reshape(rotated, part.shape, part.padded_shape)
            outputs.append(ttnn.concat([rotated, x[user : user + 1, :, :, rotary_width:]], dim=-1))
        return outputs[0] if b == 1 else ttnn.concat(outputs, dim=0)

    def _rope_decode(self, x, cos, sin):
        # Keep [1,B,H,D] from decode head creation through paged attention.
        b, rotary_width = x.shape[1], cos.shape[-1]
        outputs = []
        for user in range(b):
            part = x[:, user : user + 1, :, :rotary_width]
            cc = ttnn.reshape(cos[user : user + 1], [1, 1, 1, rotary_width])
            ss = ttnn.reshape(sin[user : user + 1], [1, 1, 1, rotary_width])
            rotated = ttnn.experimental.rotary_embedding(part, cc, ss, token_index=0)
            rotated = ttnn.reshape(rotated, part.shape, part.padded_shape)
            outputs.append(ttnn.concat([rotated, x[:, user : user + 1, :, rotary_width:]], dim=-1))
        return outputs[0] if b == 1 else ttnn.concat(outputs, dim=1)

    def _attention_output(self, attention, gate):
        attention = ttnn.transformer.concatenate_heads(attention)
        return self._linear(
            ttnn.mul(attention, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]), "self_attn.o_proj"
        )

    def _full_decode(self, x, state, page_table, current_pos, cos, sin):
        b = x.shape[0]
        c = self.config
        grid = self.device.compute_with_storage_grid_size()
        cores = ttnn.num_cores_to_corerangeset(b, grid, row_wise=True)
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [32, c.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        packed = self._linear(x, "self_attn.qkvg")
        qkv_width = (c.num_attention_heads + 2 * c.num_key_value_heads) * c.head_dim
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            ttnn.reshape(packed[:, :, :qkv_width], [1, 1, b, qkv_width]),
            num_heads=c.num_attention_heads,
            num_kv_heads=c.num_key_value_heads,
            memory_config=memory,
        )
        if b == 1:
            norm_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [32, c.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
            )
            q, k = [ttnn.to_memory_config(a, norm_memory) for a in (q, k)]
            norm_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[grid.x, grid.y], subblock_w=8, block_h=1, block_w=8, inplace=False
            )
            q = ttnn.rms_norm(
                q,
                weight=self.weights["self_attn.q_norm.weight"],
                epsilon=self.eps,
                memory_config=norm_memory,
                program_config=norm_config,
            )
            k = ttnn.rms_norm(
                k,
                weight=self.weights["self_attn.k_norm.weight"],
                epsilon=self.eps,
                memory_config=norm_memory,
                program_config=norm_config,
            )
        else:
            # Batch shards span height; RMSNorm rejects height sharding and a
            # multi-row batch does not fit the single-row block-sharded grid.
            q = self._norm(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), "self_attn.q_norm")
            k = self._norm(ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG), "self_attn.k_norm")
        rope_memory = ttnn.L1_MEMORY_CONFIG if b == 1 else ttnn.DRAM_MEMORY_CONFIG
        q = ttnn.to_memory_config(q, rope_memory)
        k = ttnn.to_memory_config(k, rope_memory)
        q, k = self._rope_decode(q, cos, sin), self._rope_decode(k, cos, sin)
        gate = packed[:, :, qkv_width:]
        key_cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x))
                for i in range(b, 2 * b)
            }
        )
        key_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(key_cores, [32, c.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        k = ttnn.to_memory_config(ttnn.reshape(k, [1, b, c.num_key_value_heads, c.head_dim]), key_memory)
        ttnn.experimental.paged_fused_update_cache(
            state.key, k, state.value, v, update_idxs_tensor=current_pos, page_table=page_table
        )
        decode_k = (
            32
            if self.policy.get("adaptive_sdpa", False) and page_table.shape[-1] < 16
            else self.policy.get("sdpa_k", 32)
        )
        decode_k = decode_k or 32
        while (page_table.shape[-1] * self.PAGE_SIZE) % decode_k:
            decode_k //= 2
        decode_grid = self.policy.get("sdpa_grid", [grid.x, grid.y])
        if b == 1 and page_table.shape[-1] < 16:
            # Short caches do not benefit from sequence parallelism across the
            # full grid. Larger batches/contexts retain the measured wide grid.
            decode_grid = self.policy.get("sdpa_short_grid", decode_grid)
        result = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            state.key,
            state.value,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=c.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=decode_grid,
                q_chunk_size=self.policy.get("sdpa_q", 32),
                k_chunk_size=decode_k,
            ),
        )
        result = ttnn.reshape(result, [b, 1, c.num_attention_heads * c.head_dim])
        return self._linear(
            ttnn.mul(result, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]), "self_attn.o_proj"
        )

    def _full_prefill(self, x, state, page_table, start_pos, cos, sin):
        # Per-request page tables are disjoint. Padded writes touch only unused
        # future rows of this request's final page; causal SDPA hides those rows.
        b, t, _ = x.shape
        q, k, v, gate = self._qkv(x, cos, sin)
        # The composite requires absolute prefix alignment as well as page
        # alignment. Keep that internal when a continuation starts mid-block.
        q_chunk = self.policy.get("prefill_sdpa_q", 32) if t >= self.policy.get("prefill_sdpa_q", 32) else 32
        k_chunk = self.policy.get("prefill_sdpa_k", 32) or 32
        while start_pos % q_chunk:
            q_chunk //= 2
        capacity = page_table.shape[-1] * self.PAGE_SIZE
        if start_pos + t > capacity:
            raise ValueError("Page table does not cover the requested prefix and continuation")
        while start_pos % k_chunk or ((start_pos + t + k_chunk - 1) // k_chunk) * k_chunk > capacity:
            k_chunk //= 2
        outputs = []
        for user in range(b):
            table = page_table[user : user + 1, :]
            chunk_table = table[:, start_pos // self.PAGE_SIZE : (start_pos + t + self.PAGE_SIZE - 1) // self.PAGE_SIZE]
            for cache, update in ((state.key, k), (state.value, v)):
                part = update[user : user + 1, :, :, :]
                if part.dtype != cache.dtype:
                    part = ttnn.typecast(part, cache.dtype)
                ttnn.experimental.paged_fill_cache(cache, part, chunk_table, batch_idx=0)
            outputs.append(
                ttnn.transformer.chunked_scaled_dot_product_attention(
                    q[user : user + 1, :, :, :],
                    state.key,
                    state.value,
                    table,
                    start_pos,
                    scale=self.config.head_dim**-0.5,
                    program_config=ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                        q_chunk_size=q_chunk,
                        k_chunk_size=k_chunk,
                    ),
                )
            )
        attention = outputs[0] if b == 1 else ttnn.concat(outputs, dim=0)
        return self._attention_output(attention, gate)

    def _delta(self, x, state):
        b, t, _ = x.shape
        c = self.config
        h, hv, d = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim
        packed = self._linear(x, "linear_attn.packed")
        conv_width = (2 * h + hv) * d
        z_width = hv * d
        gate_width = (hv + 31) // 32 * 32
        qkv = packed[:, :, :conv_width]
        z = packed[:, :, conv_width : conv_width + z_width]
        beta = ttnn.sigmoid(packed[:, :, conv_width + z_width : conv_width + z_width + hv])
        a = ttnn.typecast(
            packed[:, :, conv_width + z_width + gate_width : conv_width + z_width + gate_width + hv], ttnn.float32
        )
        padded_t = (t + 31) // 32 * 32
        padded_qkv = qkv if padded_t == t else ttnn.pad(qkv, [(0, 0), (0, padded_t - t), (0, 0)], 0.0)
        row_qkv = ttnn.to_layout(padded_qkv, ttnn.ROW_MAJOR_LAYOUT)
        chunks = []
        for user in range(b):
            chunks.append(
                ttnn.experimental.kda.qkv_causal_conv1d_silu(
                    row_qkv[user : user + 1],
                    state.conv[user : user + 1],
                    *self.conv_taps,
                    h * d,
                    h * d,
                    hv * d,
                    program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
                )
            )
        q, k, v = [parts[0] if b == 1 else ttnn.concat(parts, dim=0) for parts in zip(*chunks)]
        history_tail = (
            row_qkv[:, t - 3 : t, :] if t >= 3 else ttnn.concat([state.conv[:, t:, :], row_qkv[:, :t, :]], dim=1)
        )
        ttnn.copy(history_tail, state.conv)

        g = ttnn.mul(
            self.a_neg,
            ttnn.add(a, self.dt_bias),
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        if padded_t != t:

            def pad_time(a):
                return ttnn.pad(a, [(0, 0), (0, padded_t - t), (0, 0)], 0.0)

            # Keep the convolution outputs physically padded. Zero beta and
            # log-decay make every padded recurrence step an identity, even
            # when its convolution Q/K/V values are nonzero.
            g, beta = [pad_time(a) for a in (g, beta)]
        # The native scan assigns one value head to each core. Split only
        # its independent batch axis, preserving the public batch contract.
        grid = self.device.compute_with_storage_grid_size()
        scan_batch = grid.x * grid.y // hv
        outputs, states = [], []
        for start in range(0, b, scan_batch):
            end = min(start + scan_batch, b)
            output_part, state_part = ttnn.transformer.chunk_gated_delta_rule(
                q[start:end],
                k[start:end],
                v[start:end],
                g[start:end],
                beta[start:end],
                initial_state=state.recurrent[start:end],
                output_final_state=True,
                output_head_major=True,
                chunk_size=32,
                **self.delta_constants,
            )
            outputs.append(output_part)
            states.append(state_part)
        output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=0)
        new_state = states[0] if len(states) == 1 else ttnn.concat(states, dim=0)
        ttnn.copy(new_state, state.recurrent)
        if padded_t != t:
            z = ttnn.pad(z, [(0, 0), (0, padded_t - t), (0, 0)], 0.0)
        output = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            output, z, self.weights["linear_attn.norm.weight"], hv, epsilon=self.eps, output_dtype=ttnn.bfloat16
        )
        output = ttnn.mul(output, z)
        # Hide only trailing rows; retain the identical physical tile geometry.
        # The following projection and norms act independently on each row.
        output = ttnn.reshape(output, [b, t, hv * d], output.padded_shape)
        return self._linear(output, "linear_attn.out_proj")

    def decode_forward(self, x, *, state, current_pos, page_table=None, cos=None, sin=None):
        """One token [B,1,5120]; positions/page table/RoPE are device tensors."""
        if x.shape[1] != 1:
            raise ValueError("Decode requires one token per request")
        if self.policy.get("carry_input", False) and x.shape[0] == 1:
            x = ttnn.to_memory_config(x, self._residual_memory(1))
        n = self._norm(x, "input_layernorm")
        attention = (
            self._full_decode(n, state, page_table, current_pos, cos, sin)
            if self.kind == "full_attention"
            else self._delta(n, state)
        )
        return self._finish(x, attention)

    def prefill_forward(self, x, *, state, start_pos=0, page_table=None, cos=None, sin=None, positions=None):
        """Logical [B,S,5120] prompt/continuation; start_pos is absolute prefix length.

        For full attention, positions is an INT32 device tensor [S,B]. It is
        needed only for a continuation beginning inside a page. All requests in
        this prefill batch share the same logical length and prefix position.
        Linear-attention request state must correspond to that prefix; fresh
        requests receive newly allocated or explicitly restored zero state.
        """
        length = x.shape[1]
        if length < 1 or start_pos < 0 or start_pos + length > self.config.max_position_embeddings:
            raise ValueError("Prefill lies outside the advertised context")
        outputs = []
        offset = 0
        while offset < length:
            absolute = start_pos + offset
            inside_page = self.kind == "full_attention" and absolute % self.PAGE_SIZE != 0
            count = 1 if inside_page else min(self.CHUNK_SIZE, length - offset)
            chunk = ttnn.to_layout(x[:, offset : offset + count, :], ttnn.TILE_LAYOUT)
            n = self._norm(chunk, "input_layernorm")
            if self.kind == "linear_attention":
                attention = self._delta(n, state)
            else:
                cc = ttnn.to_layout(cos[:, offset : offset + count, :], ttnn.TILE_LAYOUT)
                ss = ttnn.to_layout(sin[:, offset : offset + count, :], ttnn.TILE_LAYOUT)
                if inside_page:
                    if positions is None:
                        raise ValueError("Unaligned prefix continuation requires positions [S,B]")
                    pos = ttnn.reshape(positions[offset : offset + 1, :], [x.shape[0]])
                    attention = self._full_decode(n, state, page_table, pos, cc, ss)
                else:
                    attention = self._full_prefill(n, state, page_table, absolute, cc, ss)
            finished = self._finish(chunk, attention)
            if length > count:
                finished = ttnn.to_memory_config(finished, ttnn.DRAM_MEMORY_CONFIG)
            outputs.append(finished)
            offset += count
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)
