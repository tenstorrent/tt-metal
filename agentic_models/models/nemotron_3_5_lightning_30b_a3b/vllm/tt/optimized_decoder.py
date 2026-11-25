# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized single-chip Nemotron-H blocks.

Inputs and outputs are device tensors [1, 1, tokens, 2688]. Each row in
``prefill_forward`` is an independent request with explicit start/length metadata.
Attention uses caller-owned paged K/V and device current positions. Mamba uses
caller-owned recurrent/conv state: continuation requires the exact prefix state.
No positional rotation is applied, matching NemotronHAttention.forward.
Weight conversion and state construction are setup operations only.
"""

import math
from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class DecoderState:
    key: object = None
    value: object = None
    conv: object = None
    recurrent: object = None


class OptimizedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        weight_dtype="bfp8",
        fidelity="LoFi",
        projection_overrides=None,
        **kwargs,
    ):
        import torch

        if kwargs:
            raise TypeError(f"Unsupported decoder setup options: {sorted(kwargs)}")
        expected = {
            "hidden_size": 2688,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "mamba_num_heads": 64,
            "mamba_head_dim": 64,
            "n_groups": 8,
            "ssm_state_size": 128,
            "conv_kernel": 4,
            "n_routed_experts": 128,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1856,
            "moe_shared_expert_intermediate_size": 3712,
            "n_group": 1,
            "topk_group": 1,
            "max_position_embeddings": 262144,
        }
        for name, value in expected.items():
            if getattr(hf_config, name) != value:
                raise ValueError(f"Target config requires {name}={value}")
        if hf_config.moe_latent_size is not None or hf_config.mlp_hidden_act != "relu2":
            raise ValueError("Target requires non-latent relu2 experts")
        self = cls()
        self.fusions = frozenset(
            {
                "folded_norm_scales",
                "direct_sharded_norm",
                "dram_qkv",
                "indexed_active_experts",
                "grouped_request_decode",
                "fp32_state_mac",
                "row_vector_state_readout",
                "metadata_views",
                "padded_logical_tails",
                "accurate_fp32_router_reduce",
                "grouped_expert_prefill",
                "tile_packed_batch_projections",
            }
        )
        self.precision_policy = {
            "weight_dtype": weight_dtype,
            "fidelity": fidelity,
            "projections": {},
            "activations": {"attention_prefill_qkv": "bfp8", "default": "bf16", "router_state": "fp32"},
        }
        self.extra_expert_weight_bytes = 0
        projection_overrides = projection_overrides or {}
        self.projection_configs = {}
        self.config = hf_config
        self.layer_idx = layer_idx
        self.kind = hf_config.layers_block_type[layer_idx]
        self.fusions |= {"explicit_batch_readout"} if self.kind == "mamba" else {"packed_request_io"}
        self.device = mesh_device
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        prefix = f"backbone.layers.{layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items()}

        # Fold learned column scales before quantizing projection weights at setup.
        if self.kind == "attention":
            for name in ("q", "k", "v"):
                key = f"mixer.{name}_proj.weight"
                sd[key] = sd[key] * sd["norm.weight"].reshape(1, -1)
        elif self.kind == "mamba":
            sd["mixer.in_proj.weight"] = sd["mixer.in_proj.weight"] * sd["norm.weight"].reshape(1, -1)
            sd["mixer.out_proj.weight"] = sd["mixer.out_proj.weight"] * sd["mixer.norm.weight"].reshape(1, -1)

        def tensor(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def projection(role, values):
            policy = {"weight_dtype": weight_dtype, "fidelity": fidelity}
            policy.update(projection_overrides.get(role, {}))
            if set(policy) != {"weight_dtype", "fidelity"}:
                raise ValueError(f"Invalid projection policy for {role}: {policy}")
            self.precision_policy["projections"][role] = policy
            self.projection_configs[role] = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, policy["fidelity"]),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[policy["weight_dtype"]]
            return tensor(values, dtype)

        def weight(name):
            role = name.split(".")[-2].removesuffix("_proj")
            if "shared_experts" in name:
                role = "shared_" + role
            elif role in ("in", "out"):
                role += "_proj"
            return projection(role, sd[name].T.unsqueeze(0).unsqueeze(0))

        if self.kind == "moe":
            self.norm = tensor(sd["norm.weight"].reshape(1, 1, 1, -1))
        if self.kind == "attention":
            self.o = weight("mixer.o_proj.weight")
            self.qkv = projection(
                "qkv",
                torch.cat([sd[f"mixer.{name}_proj.weight"] for name in ("q", "k", "v")]).T.unsqueeze(0).unsqueeze(0),
            )
            self.shard = ttnn.create_sharded_memory_config(
                (32, 128),
                ttnn.CoreGrid(y=1, x=1),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            # Direct width-sharded QKV lets head creation place K and V on
            # disjoint cores, as required by the native fused cache update.
            self.qkv_decode_memory = ttnn.create_sharded_memory_config(
                (32, 64),
                ttnn.CoreGrid(x=9, y=8),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.qkv_scalar_memory = ttnn.create_sharded_memory_config(
                (32, 128),
                ttnn.CoreGrid(x=6, y=6),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        elif self.kind == "moe":
            self.route_expand_zero = tensor(torch.zeros(1, 1, 32, 128), ttnn.float32)
            self.norm_expand_zero = tensor(torch.zeros(1, 1, 32, 2688))
            mask = torch.zeros(1, 1, 32, 32)
            mask[:, :, :, :6] = 1
            self.route_score_mask = tensor(mask, ttnn.float32)
            self.reduce_mapping = ttnn.from_torch(
                torch.zeros(1, 128, dtype=torch.int32),
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            )
            self.reduce_indices = {
                n: ttnn.from_torch(
                    torch.arange(128, dtype=torch.int32).reshape(1, 1, 1, 128).repeat(n, 1, 1, 1),
                    dtype=ttnn.uint16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                )
                for n in range(1, 33)
            }
            self.dummy_sparsity = ttnn.from_torch(
                torch.ones(1, 1, 1, 128), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
            )
            self.router_rows = tensor(sd["mixer.gate.weight"].reshape(1, 1, 128, 2688), ttnn.float32)
            self.bias = tensor(sd["mixer.gate.e_score_correction_bias"].reshape(1, 1, 1, 128), ttnn.float32)
            for name in ["up", "down"]:
                key = f"mixer.experts.{name}_proj"
                w = (
                    sd[key]
                    if key in sd
                    else torch.stack([sd[f"mixer.experts.{e}.{name}_proj.weight"] for e in range(128)])
                )
                setattr(self, name, projection(name, w.transpose(-1, -2).unsqueeze(0)))
            self.shared_up = weight("mixer.shared_experts.up_proj.weight")
            self.shared_down = weight("mixer.shared_experts.down_proj.weight")
            self.grouped_weights = []
            self.grouped_minimum = 128
            self.grouped_l1_max_rows = 512
            self.grouped_down_width = 2816
            # The native grouped API supports BF16/BFP8. Explicit BFP4 policy
            # probes retain this file's tuned sparse path for both phases.
            grouped_legal = all(
                self.precision_policy["projections"][r]["weight_dtype"] != "bfp4" for r in ("up", "down")
            )
            if grouped_legal:
                raw_weights = {}
                for role in ("up", "down"):
                    key = f"mixer.experts.{role}_proj"
                    raw_weights[role] = (
                        sd[key]
                        if key in sd
                        else torch.stack([sd[f"mixer.experts.{e}.{role}_proj.weight"] for e in range(128)])
                    )
                for start in (0, 64):
                    packed = []
                    for role in ("up", "down"):
                        values = raw_weights[role][start : start + 64]
                        if role == "down":
                            # 88 output tiles divide over 11 core columns for the native row-major epilogue.
                            values = torch.nn.functional.pad(values, (0, 0, 0, 128))
                        subsets = torch.chunk(values, 2, dim=1) if role == "up" else [values]
                        tensors = []
                        policy_dtype = self.precision_policy["projections"][role]["weight_dtype"]
                        storage_dtype = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[policy_dtype]
                        for subset in subsets:
                            columns = subset.transpose(-1, -2).permute(1, 0, 2).contiguous()
                            columns = columns.reshape(1, 1, columns.shape[0], -1)
                            tensors.append(tensor(columns, storage_dtype))
                            self.extra_expert_weight_bytes += columns.numel() * (
                                2 if policy_dtype == "bf16" else 1088 / 1024
                            )
                        packed.append(tensors if role == "up" else tensors[0])
                    self.grouped_weights.append(packed)
                self.combine_zero = tensor(torch.zeros(1, 1, 1, self.grouped_down_width))
            self.extra_expert_weight_bytes = int(self.extra_expert_weight_bytes)

        elif self.kind == "mamba":
            self.in_proj = weight("mixer.in_proj.weight")
            self.out_proj = weight("mixer.out_proj.weight")
            self.conv_weights = [tensor(sd["mixer.conv1d.weight"][:, 0, i].reshape(1, 1, 1, -1)) for i in range(4)]
            self.conv_bias = tensor(sd["mixer.conv1d.bias"].reshape(1, 1, 1, -1))
            self.dt_bias = tensor(sd["mixer.dt_bias"].reshape(1, 1, 1, 64), ttnn.float32)
            self.A = tensor(-sd["mixer.A_log"].float().exp().reshape(1, 64, 1, 1), ttnn.float32)
            self.D = tensor(sd["mixer.D"].reshape(1, 64, 1, 1), ttnn.float32)
            self.causal_masks = {
                n: tensor(torch.tril(torch.ones(1, 1, n, n)), ttnn.float32) for n in range(32, 257, 32)
            }
            self.tail_masks = {}
            for length in range(1, 32):
                mask = torch.zeros(1, 1, 32, 1, dtype=torch.float32)
                mask[:, :, :length, :] = 1
                self.tail_masks[length] = tensor(mask, ttnn.float32)
        else:
            raise ValueError(f"Unsupported target block kind {self.kind}")
        unknown = set(projection_overrides) - set(self.projection_configs)
        if unknown:
            raise ValueError(f"Unknown projection roles: {sorted(unknown)}")
        self._inside_prefill = False
        self.prefill_chunk = {"mamba": 256, "moe": 8192, "attention": 2048}[self.kind]
        if self.kind == "moe" and not self.grouped_weights:
            self.prefill_chunk = 128
        self.decode_batch_chunk = {"mamba": 8, "moe": 32, "attention": 32}[self.kind]
        self.dense_roles = {id(getattr(self, role)): role for role in self.projection_configs}
        self.dense_programs = {}
        self.prefill_programs = {}
        self.sparse_programs = {}
        self.router_chunk = 256
        if self.kind in ("mamba", "attention"):
            grid = ttnn.CoreGrid(x=4, y=1)
            self.residual_memory = ttnn.create_sharded_memory_config(
                (32, 672),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(4, 1), subblock_w=3, block_h=1, block_w=21, inplace=False
            )
        role_configs = {
            "mamba": {"in_proj": (11, 8, 3), "out_proj": (11, 8, 8)},
            "moe": {"shared_up": (11, 8, 4), "shared_down": (11, 10, 29)},
            "attention": {},
        }[self.kind]
        for role, (gx, gy, block) in role_configs.items():
            n = getattr(self, role).shape[-1]
            pn = math.ceil(n / (32 * gx * gy))
            self.dense_programs[role] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=max(v for v in (1, 2, 3, 4) if pn % v == 0),
                per_core_M=1,
                per_core_N=pn,
                fuse_batch=True,
                mcast_in0=True,
                fused_activation=None,
            )
        if self.kind == "mamba":
            self.batch_input_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(11, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
                fuse_batch=True,
                mcast_in0=True,
                fused_activation=None,
            )
            # Measured native-batch FP32 readout: retain full state precision.
            self.batch_readout_program = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(11, 10),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,
                per_core_N=2,
            )
            for role in ("in_proj", "out_proj"):
                pn = math.ceil(getattr(self, role).shape[-1] / (32 * 11))
                sub = max(v for v in (1, 2, 3, 4) if pn % v == 0)
                for m in range(64, 257, 32):
                    self.prefill_programs[role, m] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(11, 8),
                        in0_block_w=4,
                        out_subblock_h=1,
                        out_subblock_w=sub,
                        per_core_M=math.ceil(m / 256),
                        per_core_N=pn,
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=True,
                    )
        if self.kind == "attention":
            old_id = id(self.qkv)
            banks = self.device.dram_grid_size().x
            bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            weight_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(bank_grid, (2688, 4608 // banks), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.qkv = ttnn.to_memory_config(self.qkv, weight_memory)
            self.dense_roles.pop(old_id)
            self.dense_roles[id(self.qkv)] = "qkv"
            self.qkv_dram_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=7, per_core_M=1, per_core_N=36, num_workers_per_dram_bank=1
            )
            for m in range(32, self.prefill_chunk + 1, 32):
                self.prefill_programs[m] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(11, 8),
                    in0_block_w=7,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=math.ceil(m / 256),
                    per_core_N=14,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                )
            self.sdpa_decode_programs = {
                k: ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=k)
                for k in (32, 64, 128, 256)
            }
            self.sdpa_prefill_program = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=512, exp_approx_mode=False
            )
            self.batch_sdpa_decode_programs = {
                k: ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(11, 8), q_chunk_size=32, k_chunk_size=k)
                for k in (32, 64, 128, 256)
            }
            # Measured physical-page capacities where a smaller chunk beats
            # read-table padding. Other capacities keep the preferred chunk.
            self.sdpa_capacity_chunks = {1: 32, 2: 64, 6: 64, 10: 64, 14: 64, 18: 64, 66: 64, 260: 128, 516: 128}
            grid_size = ttnn.CoreCoord(11, 8)
            self.concat_subgrid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
            )
            self.batch_heads_memory, self.batch_concat_memory = {}, {}
            for batch in range(1, 33):
                for count, target in ((2 * batch, self.batch_heads_memory), (batch, self.batch_concat_memory)):
                    grid = ttnn.num_cores_to_corerangeset(
                        count, ttnn.CoreCoord(8, 8) if batch == 1 else grid_size, row_wise=True
                    )
                    target[batch] = ttnn.create_sharded_memory_config(
                        (32, 128),
                        grid,
                        ttnn.ShardStrategy.HEIGHT,
                        ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
        if self.kind == "moe":
            for rows in range(64, 129, 32):
                self.reduce_indices[rows] = ttnn.from_torch(
                    torch.arange(128, dtype=torch.int32).reshape(1, 1, 1, 128).repeat(rows, 1, 1, 1),
                    dtype=ttnn.uint16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                )
            for m in range(1, 129):
                for n, (pn, block, sub) in ((1856, (3, 7, 3)), (2688, (2, 29, 2))):
                    workers = math.ceil(n / (32 * pn))
                    grid = max((x, workers // x) for x in range(1, 12) if workers % x == 0 and workers // x <= 10)
                    self.sparse_programs[m, n] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=grid,
                        in0_block_w=block,
                        out_subblock_h=1,
                        out_subblock_w=sub,
                        per_core_M=math.ceil(m / 32),
                        per_core_N=pn,
                        fuse_batch=False,
                        mcast_in0=True,
                        fused_activation=None,
                    )
        return self

    def allocate_state(self, *, context_length=262144, block_size=32, num_slots=1):
        import torch

        def zeros(shape, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                torch.zeros(shape),
                device=self.device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if not 1 <= context_length <= self.config.max_position_embeddings:
            raise ValueError("Cache context must lie within the advertised capability")
        if self.kind == "attention":
            if block_size != 32:
                raise ValueError("The paged-cache contract uses 32-token blocks")
            blocks = (context_length + block_size - 1) // block_size * num_slots
            return DecoderState(
                key=zeros((blocks, 2, block_size, 128), ttnn.bfloat8_b),
                value=zeros((blocks, 2, block_size, 128), ttnn.bfloat8_b),
            )
        if self.kind == "mamba":
            return DecoderState(
                conv=[zeros((1, 1, 1, 6144)) for _ in range(3)], recurrent=zeros((1, 64, 64, 128), ttnn.float32)
            )
        return DecoderState()

    def _linear(self, x, w, dtype=ttnn.bfloat16, **kwargs):
        role = self.dense_roles.get(id(w))
        decode_qkv = kwargs.pop("decode_qkv", False)
        if dtype == ttnn.float32 or role is None:
            if self.kind == "mamba" and x.shape[0] > 1 and dtype == ttnn.float32:
                kwargs.setdefault("memory_config", ttnn.L1_MEMORY_CONFIG)
                if x.shape[2] == 1:
                    kwargs.setdefault("program_config", self.batch_readout_program)
            return ttnn.linear(x, w, dtype=dtype, compute_kernel_config=self.compute, **kwargs)
        compute = self.projection_configs[role]
        m = x.shape[-2]
        bias = kwargs.get("bias")
        separate_bias = bias is not None and (x.shape[0] > 1 or bias.shape[-2] > 32)
        if separate_bias:
            kwargs.pop("bias")
        if role == "qkv":
            if m <= 32 and (not self._inside_prefill or decode_qkv):
                requested = kwargs.pop("memory_config", ttnn.DRAM_MEMORY_CONFIG)
                kwargs.pop("program_config", None)
                x = ttnn.to_memory_config(x, self.residual_memory)
                out = ttnn.linear(
                    x,
                    w,
                    dtype=dtype,
                    compute_kernel_config=compute,
                    program_config=self.qkv_dram_program,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    **kwargs,
                )
                out = ttnn.to_memory_config(out, requested)
            else:
                kwargs.pop("program_config", None)
                out = ttnn.linear(
                    x, w, dtype=dtype, compute_kernel_config=compute, program_config=self.prefill_programs[m], **kwargs
                )
        elif self.kind == "mamba" and x.shape[0] > 1:
            kwargs.pop("program_config", None)
            kwargs.setdefault("memory_config", ttnn.L1_MEMORY_CONFIG)
            batch = x.shape[0]
            packed = ttnn.reshape(x, (1, 1, batch, x.shape[-1]))
            program = self.batch_input_program if role == "in_proj" else self.dense_programs[role]
            out = ttnn.linear(packed, w, dtype=dtype, compute_kernel_config=compute, program_config=program, **kwargs)
            out = ttnn.reshape(out, (batch, 1, 1, out.shape[-1]))
        elif self.kind == "mamba" and m > 32:
            out = ttnn.linear(
                x,
                w,
                dtype=dtype,
                compute_kernel_config=compute,
                program_config=self.prefill_programs[role, m],
                **kwargs,
            )
        elif role in self.dense_programs and m <= 32:
            requested = kwargs.pop(
                "memory_config",
                (
                    ttnn.L1_MEMORY_CONFIG
                    if not self._inside_prefill and (self.kind == "mamba" or role == "shared_down")
                    else ttnn.DRAM_MEMORY_CONFIG
                ),
            )
            kwargs.pop("program_config", None)
            if x.is_sharded() and role != "in_proj":
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            out = ttnn.linear(
                x,
                w,
                dtype=dtype,
                compute_kernel_config=compute,
                program_config=self.dense_programs[role],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                **kwargs,
            )
            out = ttnn.to_memory_config(out, requested)
        else:
            out = ttnn.linear(x, w, dtype=dtype, compute_kernel_config=compute, **kwargs)
        return ttnn.add(out, bias) if separate_bias else out

    def _norm(self, x):
        if self.kind != "moe" and x.shape[0] == 1 and x.shape[-2] <= 32 and not self._inside_prefill:
            x = ttnn.to_memory_config(x, self.residual_memory)
            return ttnn.rms_norm(
                x,
                epsilon=self.config.layer_norm_epsilon,
                program_config=self.norm_program,
                compute_kernel_config=self.compute,
                memory_config=self.residual_memory,
            )
        return self._norm_unsharded(x)

    def _norm_unsharded(self, x):
        rows = x.shape[2]
        if self.kind == "moe":
            # Keep the accurate FP32 reduction, avoiding its implicit-row fill.
            # Decode rows must share an addition order: width reduction can
            # differ by two FP32 ULP across rows, crossing the BF16 boundary and
            # changing MoE routing. Preserve the prepared prefill arithmetic.
            if rows == 1:
                x = ttnn.add(x, self.norm_expand_zero)
            xf = ttnn.typecast(x, ttnn.float32)
            square = ttnn.square(xf)
            uniform_decode_order = not self._inside_prefill and rows <= 32
            if uniform_decode_order:
                square = ttnn.transpose(square, -1, -2)
            variance = ttnn.mean(
                square,
                dim=2 if uniform_decode_order else 3,
                keepdim=True,
                compute_kernel_config=self.compute,
                fast_and_approximate_mode=False,
            )
            if uniform_decode_order:
                variance = ttnn.transpose(variance, -1, -2)
            normalized = ttnn.typecast(
                ttnn.multiply(
                    xf,
                    ttnn.add(variance, self.config.layer_norm_epsilon),
                    input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0)],
                ),
                ttnn.bfloat16,
            )
        else:
            normalized = ttnn.rms_norm(x, epsilon=self.config.layer_norm_epsilon, compute_kernel_config=self.compute)
        result = ttnn.multiply(normalized, self.norm) if self.kind == "moe" else normalized
        return result[:, :, :rows, :] if self.kind == "moe" and rows == 1 else result

    def _attention_decode(self, x, *, state, page_table, current_pos, residual):
        batch = x.shape[2]
        qkv = self._linear(
            x, self.qkv, memory_config=self.qkv_scalar_memory if batch == 1 else self.qkv_decode_memory, decode_qkv=True
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=32, num_kv_heads=2, overlap_qk_coregrid=False, memory_config=self.batch_heads_memory[batch]
        )
        ttnn.experimental.paged_fused_update_cache(
            state.key, k, state.value, v, update_idxs_tensor=current_pos, page_table=page_table
        )
        # Capacity is host-known metadata; current logical positions remain on device.
        pages = page_table.shape[-1]
        key_chunk = self.sdpa_capacity_chunks.get(pages, 128 if pages <= 256 else 256)
        # SDPA reads full chunks before masking. Initialize every lookahead
        # mapping as an alias of this request's last valid physical page.
        # Cache updates above retain the original logical table and capacity.
        read_table = page_table
        padding = -pages % (key_chunk // 32)
        if padding:
            read_table = ttnn.concat([page_table, ttnn.repeat(page_table[:, -1:], (1, padding))], dim=1)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            state.key,
            state.value,
            read_table,
            cur_pos_tensor=current_pos,
            program_config=(self.sdpa_decode_programs if batch == 1 else self.batch_sdpa_decode_programs)[key_chunk],
            compute_kernel_config=self.compute,
        )
        out = ttnn.to_memory_config(out, self.batch_concat_memory[batch])
        out = ttnn.experimental.nlp_concat_heads_decode(
            out, num_heads=32, **({"sub_core_grids": self.concat_subgrid} if batch > 11 and batch % 11 != 0 else {})
        )
        out = out[:, :, :batch, :]
        return self._linear(out, self.o, bias=residual)

    def _sparse_config(self, m, n):
        return self.sparse_programs[m, n]

    def _router_logits(self, x):
        # Hard top-k membership needs the preserved accurate FP32 reduction.
        # Native HiFi4 matmul changes expert membership on the seed1000 gate.
        # Bound expanded products independently of the larger expert chunk.
        parts = []
        for start in range(0, x.shape[2], self.router_chunk):
            rows = (
                x if x.shape[2] <= self.router_chunk else x[:, :, start : min(start + self.router_chunk, x.shape[2]), :]
            )
            tokens = ttnn.permute(ttnn.typecast(rows, ttnn.float32), (0, 2, 1, 3))
            products = ttnn.multiply(tokens, self.router_rows)
            logits = ttnn.sum(
                products, dim=3, keepdim=True, compute_kernel_config=self.compute, fast_and_approximate_mode=False
            )
            parts.append(ttnn.permute(logits, (0, 3, 1, 2)))
            del rows, tokens, products, logits
        while len(parts) > 32:
            parts = [ttnn.concat(parts[i : i + 32], dim=2) for i in range(0, len(parts), 32)]
        return parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)

    def _route(self, x):
        scores = ttnn.sigmoid(self._router_logits(x))
        rows = x.shape[2]
        if rows in (1, 32):
            work_scores = ttnn.add(scores, self.route_expand_zero) if rows == 1 else scores
            choice = ttnn.add(work_scores, self.bias)
            _, indices = ttnn.topk(choice, k=32, dim=-1, sorted=True)
            selected = ttnn.gather(work_scores, dim=3, index=indices)
            indices = indices[:, :, :rows, :6]
            selected = ttnn.multiply(selected, self.route_score_mask)
        else:
            choice = ttnn.add(scores, self.bias)
            _, indices = ttnn.topk(choice, k=8, dim=-1, sorted=True)
            indices = indices[:, :, :, :6]
            selected = ttnn.gather(scores, dim=3, index=indices)
        if not self._inside_prefill and rows <= 32:
            # The FP32 width reduction associates additions differently across
            # tile row groups. Even a two-ULP denominator difference can change
            # the BF16 expert result and later sampled token for identical rows.
            # Reduce transposed columns with one association for every request.
            denominator = ttnn.transpose(
                ttnn.sum(
                    ttnn.transpose(selected, -1, -2),
                    dim=2,
                    keepdim=True,
                    compute_kernel_config=self.compute,
                    fast_and_approximate_mode=False,
                ),
                -1,
                -2,
            )
        else:
            denominator = ttnn.sum(selected, dim=3, keepdim=True)
        selected = ttnn.multiply(
            selected,
            denominator,
            input_tensor_b_activations=[
                ttnn.UnaryOpType.RECIP,
                ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 2.5),
            ],
        )
        if rows in (1, 32):
            selected = selected[:, :, :rows, :6]
        return indices, selected, scores

    def _moe(self, x):
        memory = ttnn.DRAM_MEMORY_CONFIG if x.shape[2] > 32 else ttnn.L1_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, memory)
        indices, selected, scores = self._route(x)
        if self._inside_prefill and self.grouped_weights and x.shape[2] >= self.grouped_minimum:
            return self._grouped_moe(x, indices, selected)
        if x.shape[2] == 1:
            ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT)
            up = ttnn.sparse_matmul(
                x,
                self.up,
                sparsity=self.dummy_sparsity,
                indices=ids,
                program_config=self._sparse_config(1, 1856),
                compute_kernel_config=self.projection_configs["up"],
                memory_config=memory,
            )
            act = ttnn.unary_chain(up, [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE])
            down = ttnn.sparse_matmul(
                act,
                self.down,
                sparsity=self.dummy_sparsity,
                indices=ids,
                is_input_a_sparse=True,
                program_config=self._sparse_config(1, 2688),
                compute_kernel_config=self.projection_configs["down"],
                memory_config=memory,
            )
            selected = ttnn.typecast(selected, ttnn.bfloat16)
            routed = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                ttnn.reshape(down, (6, 1, 1, 2688)),
                ids,
                self.reduce_mapping,
                0,
                num_shared_experts=0,
                split_size=2688,
                cluster_axis=0,
                scores_tensor=ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT),
                compute_kernel_config=self.compute,
            )[0]
            shared_up = self._linear(x, self.shared_up)
            shared = self._linear(
                ttnn.unary_chain(shared_up, [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE]),
                self.shared_down,
                bias=routed,
            )
            return shared
        routing = ttnn.scatter(
            ttnn.zeros_like(ttnn.typecast(scores, ttnn.bfloat16)),
            dim=3,
            index=indices,
            src=ttnn.typecast(selected, ttnn.bfloat16),
        )
        routed_scores = routing
        sparsity = ttnn.to_layout(
            ttnn.typecast(ttnn.sum(routing, dim=2, keepdim=True), ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT
        )
        up = ttnn.sparse_matmul(
            x,
            self.up,
            sparsity=sparsity,
            nnz=None,
            program_config=self._sparse_config(x.shape[2], 1856),
            compute_kernel_config=self.projection_configs["up"],
            memory_config=memory,
        )
        act = ttnn.unary_chain(up, [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE])
        down = ttnn.sparse_matmul(
            act,
            self.down,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=True,
            program_config=self._sparse_config(x.shape[2], 2688),
            compute_kernel_config=self.projection_configs["down"],
            memory_config=memory,
        )
        scores_rm = ttnn.reshape(ttnn.to_layout(routed_scores, ttnn.ROW_MAJOR_LAYOUT), (x.shape[2], 1, 1, 128))
        routed = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            ttnn.reshape(down, (128, 1, x.shape[2], 2688)),
            self.reduce_indices[x.shape[2]],
            self.reduce_mapping,
            0,
            num_shared_experts=0,
            split_size=2688,
            cluster_axis=0,
            scores_tensor=scores_rm,
            compute_kernel_config=self.compute,
        )[0]
        shared_up = self._linear(x, self.shared_up)
        shared = self._linear(
            ttnn.unary_chain(shared_up, [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE]), self.shared_down, bias=routed
        )
        return shared

    def _grouped_moe(self, x, indices, selected):
        rows = x.shape[2]
        activation_memory = ttnn.L1_MEMORY_CONFIG if rows <= self.grouped_l1_max_rows else ttnn.DRAM_MEMORY_CONFIG
        ids_float = ttnn.typecast(indices, ttnn.float32)
        table = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        combined = None
        for half, (up_weight, down_weight) in enumerate(self.grouped_weights):
            start = half * 64
            valid = ttnn.logical_and(ttnn.ge(ids_float, start), ttnn.lt(ids_float, start + 64))
            local_ids = ttnn.typecast(ttnn.where(valid, ttnn.subtract(ids_float, start), 63.0), ttnn.uint32)
            local_scores = ttnn.multiply(selected, valid)
            gather, combine, weights, counts = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
                local_ids, local_scores, num_experts=64
            )
            dispatched = ttnn.embedding(gather, table, layout=ttnn.TILE_LAYOUT, memory_config=activation_memory)
            dispatched = ttnn.reshape(dispatched, (1, 1, dispatched.shape[-2], 2688))
            up_parts = [
                ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                    dispatched,
                    weight,
                    counts,
                    dtype=ttnn.bfloat16,
                    memory_config=activation_memory,
                    compute_kernel_config=self.projection_configs["up"],
                )
                for weight in up_weight
            ]
            # Dispatch has reached its last consumer; free it before allocating activation storage.
            del dispatched
            up = up_parts[0] if len(up_parts) == 1 else ttnn.concat(up_parts, dim=3, memory_config=activation_memory)
            del up_parts
            # The up result has no later consumer: preserve its placement and avoid a copy.
            activation = ttnn.unary_chain(
                up,
                [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE],
                memory_config=activation_memory,
                output_tensor=up,
            )
            del up
            down_rows = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                activation,
                down_weight,
                counts,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.projection_configs["down"],
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            del activation
            out = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                down_rows,
                combine,
                weights,
                self.combine_zero,
                num_tokens=rows,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute,
            )
            combined = out if combined is None else ttnn.add(combined, out)
            del down_rows, gather, combine, weights, counts
        # Combine padded expert rows first, then trim once at the shared-path boundary.
        combined = combined[..., :2688]
        shared_up = self._linear(x, self.shared_up)
        return self._linear(
            ttnn.unary_chain(shared_up, [ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE]),
            self.shared_down,
            bias=combined,
        )

    def _mamba_decode(self, x, *, state, residual, use_mac=True):
        batch = x.shape[0]
        projected = self._linear(x, self.in_proj)
        gate = projected[:, :, :, :4096]
        conv_in = projected[:, :, :, 4096:10240]
        dt = ttnn.typecast(projected[:, :, :, 10240:10304], ttnn.float32)
        conv = ttnn.multiply(conv_in, self.conv_weights[3])
        for history, weight in zip(state.conv, self.conv_weights[:3]):
            conv = ttnn.mac(history, weight, conv) if use_mac else ttnn.add(conv, ttnn.multiply(history, weight))
        for i in range(2):
            ttnn.copy(state.conv[i + 1], state.conv[i])
        ttnn.copy(conv_in, state.conv[2])
        conv = ttnn.add(conv, self.conv_bias, activations=[ttnn.UnaryOpType.SILU])
        x_flat = ttnn.typecast(conv[:, :, :, :4096], ttnn.float32)
        x_ssm = ttnn.reshape(x_flat, (batch, 64, 64, 1))
        vector_view = ttnn.experimental.view
        B = vector_view(ttnn.typecast(conv[:, :, :, 4096:5120], ttnn.float32), (batch, 8, 1, 128))
        C = vector_view(ttnn.typecast(conv[:, :, :, 5120:6144], ttnn.float32), (batch, 8, 1, 128))
        B = ttnn.repeat_interleave(B, 8, dim=1)
        C = ttnn.repeat_interleave(C, 8, dim=1)
        dt = ttnn.add(
            dt,
            self.dt_bias,
            activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0),
                ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 0.001),
            ],
        )
        dt = ttnn.reshape(dt, (batch, 64, 1, 1))
        decay = ttnn.multiply(dt, self.A, activations=[ttnn.UnaryOpType.EXP])
        update = ttnn.multiply(ttnn.multiply(x_ssm, dt), B)
        new_state = (
            ttnn.mac(state.recurrent, decay, update, output_tensor=state.recurrent)
            if use_mac
            else ttnn.add(ttnn.multiply(state.recurrent, decay), update, output_tensor=state.recurrent)
        )
        if batch > 1:
            del update, decay, dt, B, conv, conv_in, projected, x_ssm
        y = self._linear(C, new_state, ttnn.float32, transpose_b=True)
        x_out = ttnn.experimental.view(x_flat, (batch, 64, 1, 64))
        y = ttnn.mac(x_out, self.D, y) if use_mac else ttnn.add(y, ttnn.multiply(x_out, self.D))
        y = ttnn.experimental.view(y, (batch, 1, 1, 4096))
        gated = ttnn.multiply(y, ttnn.typecast(gate, ttnn.float32), input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        view = ttnn.experimental.view
        grouped = view(gated, (batch, 8, 1, 512))
        grouped = ttnn.rms_norm(grouped, epsilon=self.config.layer_norm_epsilon, compute_kernel_config=self.compute)
        normalized = ttnn.typecast(view(grouped, (batch, 1, 1, 4096)), ttnn.bfloat16)
        return self._linear(normalized, self.out_proj, bias=residual)

    def _mamba_prefill_chunk(self, x, *, state, valid_length=None):
        length = x.shape[2]
        projected = self._linear(x, self.in_proj)
        gate = projected[:, :, :, :4096]
        conv_in = projected[:, :, :, 4096:10240]
        dt = ttnn.typecast(projected[:, :, :, 10240:10304], ttnn.float32)
        history = ttnn.concat([*state.conv, conv_in], dim=2)
        conv = ttnn.multiply(history[:, :, :length, :], self.conv_weights[0])
        for i in range(1, 4):
            conv = ttnn.add(conv, ttnn.multiply(history[:, :, i : i + length, :], self.conv_weights[i]))
        for i in range(3):
            if valid_length is None:
                ttnn.copy(conv_in[:, :, length - 3 + i : length - 2 + i, :], state.conv[i])
            else:
                ttnn.copy(history[:, :, valid_length + i : valid_length + i + 1, :], state.conv[i])
        conv = ttnn.add(conv, self.conv_bias, activations=[ttnn.UnaryOpType.SILU])
        xs = ttnn.permute(
            ttnn.reshape(ttnn.typecast(conv[:, :, :, :4096], ttnn.float32), (1, length, 64, 64)), (0, 2, 1, 3)
        )
        B = ttnn.permute(
            ttnn.reshape(ttnn.typecast(conv[:, :, :, 4096:5120], ttnn.float32), (1, length, 8, 128)), (0, 2, 1, 3)
        )
        C = ttnn.permute(
            ttnn.reshape(ttnn.typecast(conv[:, :, :, 5120:6144], ttnn.float32), (1, length, 8, 128)), (0, 2, 1, 3)
        )
        B = ttnn.repeat_interleave(B, 8, dim=1)
        C = ttnn.repeat_interleave(C, 8, dim=1)
        dt = ttnn.add(
            dt,
            self.dt_bias,
            activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0),
                ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 0.001),
            ],
        )
        dt = ttnn.permute(ttnn.reshape(dt, (1, length, 64, 1)), (0, 2, 1, 3))
        if valid_length is not None:
            dt = ttnn.multiply(dt, self.tail_masks[valid_length])
        adt = ttnn.multiply(dt, self.A)
        accumulated = ttnn.cumsum(adt, dim=2)
        differences = ttnn.subtract(accumulated, ttnn.transpose(accumulated, 2, 3))
        decay = ttnn.where(self.causal_masks[length], differences, -1e30)
        xd = ttnn.multiply(xs, dt)
        CB = self._linear(C, B, ttnn.float32, transpose_b=True)
        local = self._linear(
            ttnn.multiply(CB, decay, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)]),
            xd,
            ttnn.float32,
        )
        from_prefix = self._linear(C, state.recurrent, ttnn.float32, transpose_b=True)
        y = ttnn.add(
            local,
            ttnn.multiply(
                from_prefix, accumulated, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)]
            ),
        )
        y = ttnn.add(y, ttnn.multiply(xs, self.D))
        last = accumulated[:, :, length - 1 : length, :]
        Bdecay = ttnn.multiply(B, ttnn.subtract(last, accumulated, activations=[ttnn.UnaryOpType.EXP]))
        update = self._linear(xd, Bdecay, ttnn.float32, transpose_a=True)
        ttnn.add(
            ttnn.multiply(
                state.recurrent, last, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)]
            ),
            update,
            output_tensor=state.recurrent,
        )
        y = ttnn.reshape(ttnn.permute(y, (0, 2, 1, 3)), (1, 1, length, 4096))
        gated = ttnn.multiply(y, ttnn.typecast(gate, ttnn.float32), input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        grouped = ttnn.reshape(gated, (1, length, 8, 512))
        grouped = ttnn.rms_norm(grouped, epsilon=self.config.layer_norm_epsilon, compute_kernel_config=self.compute)
        normalized = ttnn.typecast(ttnn.reshape(grouped, (1, 1, length, 4096)), ttnn.bfloat16)
        out = self._linear(normalized, self.out_proj)
        return out if valid_length is None else out[:, :, :valid_length, :]

    def decode_forward(self, hidden_states, *, state, page_table=None, current_pos=None):
        """Decode independent request rows; position and route values stay on device."""
        if isinstance(hidden_states, (tuple, list)):
            requests = list(zip(hidden_states, state, page_table, current_pos, strict=True))
            if not requests:
                return []
            native = True
            if self.kind == "attention":
                native = all(
                    st.key is state[0].key
                    and st.value is state[0].value
                    and tuple(p.shape) == tuple(page_table[0].shape)
                    for _, st, p, _ in requests
                )
            elif self.kind == "mamba":
                buffers = [id(t) for st in state for t in (*st.conv, st.recurrent)]
                native = len(set(buffers)) == len(buffers)
            if not native:
                return [self.decode_forward(x, state=st, page_table=p, current_pos=c) for x, st, p, c in requests]
            outputs = []
            for start in range(0, len(requests), self.decode_batch_chunk):
                group = requests[start : start + self.decode_batch_chunk]
                if len(group) == 1:
                    x, st, p, c = group[0]
                    outputs.append(self.decode_forward(x, state=st, page_table=p, current_pos=c))
                    continue
                if self.kind == "mamba":
                    inp = ttnn.concat([r[0] for r in group], dim=0)
                    combined = DecoderState(
                        conv=[
                            ttnn.concat([r[1].conv[i] for r in group], dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)
                            for i in range(3)
                        ],
                        recurrent=ttnn.concat(
                            [r[1].recurrent for r in group], dim=0, memory_config=ttnn.L1_MEMORY_CONFIG
                        ),
                    )
                    out = self._mamba_decode(self._norm_unsharded(inp), state=combined, residual=inp)
                    for i, (_, destination, _, _) in enumerate(group):
                        for new, old in zip(combined.conv, destination.conv):
                            ttnn.copy(new[i : i + 1], old)
                        ttnn.copy(combined.recurrent[i : i + 1], destination.recurrent)
                        outputs.append(out[i : i + 1])
                    del combined, inp, out
                else:
                    # Convert the complete batch once instead of untilizing each request.
                    inp = ttnn.reshape(ttnn.concat([r[0] for r in group], dim=0), (1, 1, len(group), 2688))
                    if self.kind == "moe":
                        out = ttnn.add(inp, self._moe(self._norm(inp)))
                    else:
                        pages = ttnn.concat([r[2] for r in group], dim=0)
                        positions = ttnn.concat([r[3] for r in group], dim=0)
                        out = self._attention_decode(
                            self._norm(inp), state=group[0][1], page_table=pages, current_pos=positions, residual=inp
                        )
                    # Batch-axis slices preserve the request API without per-row tilization.
                    packed = ttnn.reshape(out, (len(group), 1, 1, 2688))
                    outputs.extend(packed[i : i + 1] for i in range(len(group)))
            return outputs
        x = self._norm(hidden_states)
        if self.kind == "attention":
            return self._attention_decode(
                x, state=state, page_table=page_table, current_pos=current_pos, residual=hidden_states
            )
        if self.kind == "mamba":
            return self._mamba_decode(x, state=state, residual=hidden_states)
        return ttnn.add(hidden_states, self._moe(x))

    def _attention_prefill_chunk(self, x, *, state, page_table, start_pos, residual=None):
        length = x.shape[2]
        qkv = self._linear(x, self.qkv, dtype=ttnn.bfloat8_b)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(qkv, (1, length, 4608)), num_heads=32, num_kv_heads=2, transpose_key=False
        )
        if state.key.dtype != k.dtype:
            k = ttnn.typecast(k, state.key.dtype)
            v = ttnn.typecast(v, state.value.dtype)
        block = state.key.shape[2]
        fill_table = page_table[:, start_pos // block : (start_pos + length) // block]
        ttnn.experimental.paged_fill_cache(state.key, k, fill_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(state.value, v, fill_table, batch_idx=0)
        read_table = page_table
        padding_pages = -page_table.shape[-1] % 16
        if padding_pages:
            read_table = ttnn.concat([page_table, ttnn.repeat(page_table[:, -1:], (1, padding_pages))], dim=1)
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            state.key,
            state.value,
            read_table,
            chunk_start_idx=start_pos,
            program_config=self.sdpa_prefill_program,
            compute_kernel_config=self.compute,
        )
        out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, length, 4096))
        return self._linear(out, self.o, bias=residual)

    def _moe_prefill_requests(self, hidden_states, *, state, page_table, start_pos, positions):
        """Pack at most 32 stateless requests and 8192 logical tokens per group.

        Validate each public span before packing; grouped token positions are
        irrelevant to MoE. Larger individual requests retain scalar chunking.
        """
        requests = list(zip(hidden_states, state, page_table, start_pos, strict=True))
        for x, _, _, start in requests:
            if not 0 <= start < start + x.shape[2] <= self.config.max_position_embeddings:
                raise ValueError("Prefill span must lie within the advertised context")
        outputs = []
        index = 0
        while index < len(requests):
            end = index + 1
            tokens = requests[index][0].shape[2]
            while end < len(requests) and end - index < 32 and tokens + requests[end][0].shape[2] <= 8192:
                tokens += requests[end][0].shape[2]
                end += 1
            group = requests[index:end]
            if len(group) == 1:
                x, st, page, start = group[0]
                outputs.append(self.prefill_forward(x, state=st, page_table=page, start_pos=start, positions=positions))
            else:
                packed = ttnn.concat([x for x, _, _, _ in group], dim=2)
                result = self.prefill_forward(
                    packed, state=group[0][1], page_table=group[0][2], start_pos=0, positions=positions
                )
                offset = 0
                for x, _, _, _ in group:
                    length = x.shape[2]
                    outputs.append(result[:, :, offset : offset + length, :])
                    offset += length
            index = end
        return outputs

    def prefill_forward(self, hidden_states, *, state, page_table=None, start_pos=0, positions=None):
        """Preserve arbitrary logical lengths and continuation in bounded physical chunks.

        Attention positions is a caller-owned int32 [context,1] device bank.
        Padding touches only the request's allocated final page; causal outputs
        and the returned tensor retain the exact logical length.
        """
        if isinstance(hidden_states, (tuple, list)):
            if self.kind == "moe":
                return self._moe_prefill_requests(
                    hidden_states, state=state, page_table=page_table, start_pos=start_pos, positions=positions
                )
            return [
                self.prefill_forward(x, state=st, page_table=p, start_pos=a, positions=positions)
                for x, st, p, a in zip(hidden_states, state, page_table, start_pos, strict=True)
            ]
        length = hidden_states.shape[2]
        if not 0 <= start_pos < start_pos + length <= self.config.max_position_embeddings:
            raise ValueError("Prefill span must lie within the advertised context")
        previous = self._inside_prefill
        self._inside_prefill = True
        outputs = []
        offset = 0
        try:
            while offset < length:
                absolute, remaining = start_pos + offset, length - offset
                if self.kind == "attention" and absolute % 512 == 0:
                    count = min(self.prefill_chunk, remaining)
                    inp = hidden_states[:, :, offset : offset + count, :]
                    physical = math.ceil(count / 32) * 32
                    padded = (
                        inp
                        if physical == count
                        else ttnn.pad(inp, [(0, 0), (0, 0), (0, physical - count), (0, 0)], value=0.0)
                    )
                    out = self._attention_prefill_chunk(
                        self._norm(padded),
                        state=state,
                        page_table=page_table,
                        start_pos=absolute,
                        residual=padded if physical == 32 else None,
                    )
                    out = out[:, :, :count, :]
                    if physical != 32:
                        out = ttnn.add(inp, out)
                elif self.kind == "mamba" and remaining >= 32:
                    count = min(self.prefill_chunk, remaining // 32 * 32)
                    inp = hidden_states[:, :, offset : offset + count, :]
                    out = ttnn.add(inp, self._mamba_prefill_chunk(self._norm(inp), state=state))
                elif self.kind == "mamba" and remaining >= 2:
                    count = remaining
                    inp = hidden_states[:, :, offset : offset + count, :]
                    padded = ttnn.pad(inp, [(0, 0), (0, 0), (0, 32 - count), (0, 0)], value=0.0)
                    out = ttnn.add(inp, self._mamba_prefill_chunk(self._norm(padded), state=state, valid_length=count))
                elif self.kind == "moe":
                    count = min(self.prefill_chunk, remaining)
                    if count < 64:
                        count = count // 32 * 32 if count >= 32 else count
                    inp = hidden_states[:, :, offset : offset + count, :]
                    physical = math.ceil(count / 32) * 32 if count >= 64 else count
                    work = (
                        inp
                        if physical == count
                        else ttnn.pad(inp, [(0, 0), (0, 0), (0, physical - count), (0, 0)], value=0.0)
                    )
                    out = ttnn.add(work, self._moe(self._norm(work)))[:, :, :count, :]
                else:
                    count = 1
                    inp = hidden_states[:, :, offset : offset + 1, :]
                    if self.kind == "mamba":
                        out = self._mamba_decode(self._norm(inp), state=state, residual=inp, use_mac=False)
                    else:
                        pos = positions[absolute : absolute + 1, :]
                        out = self.decode_forward(inp, state=state, page_table=page_table, current_pos=pos)
                outputs.append(out)
                offset += count
            while len(outputs) > 32:
                outputs = [ttnn.concat(outputs[i : i + 32], dim=2) for i in range(0, len(outputs), 32)]
            return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        finally:
            self._inside_prefill = previous
