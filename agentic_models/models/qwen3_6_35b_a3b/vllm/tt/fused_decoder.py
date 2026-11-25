# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B fused decoder; setup owns all host tensor conversion.

Inputs/outputs are device tensors [1, S, 2048]. Each request owns its recurrent
state or paged KV pool. Runtime methods perform only TTNN tensor operations.
"""
import ttnn
from models.common.lightweightmodule import LightweightModule

DRAM = ttnn.DRAM_MEMORY_CONFIG


def reshape(x, shape):
    """Repack logical rows explicitly across padded tile boundaries."""
    return ttnn.to_layout(ttnn.reshape(x, shape), ttnn.TILE_LAYOUT)


class FusedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device):
        if hf_config.layer_types[layer_idx] == "full_attention":
            import torch

            self = cls()
            self.config, self.mesh_device = (hf_config, mesh_device)
            self.layer_idx, self.kind = (layer_idx, hf_config.layer_types[layer_idx])
            expected = dict(
                hidden_size=2048,
                num_experts=256,
                num_experts_per_tok=8,
                moe_intermediate_size=512,
                shared_expert_intermediate_size=512,
                num_attention_heads=16,
                num_key_value_heads=2,
                head_dim=256,
                linear_num_key_heads=16,
                linear_num_value_heads=32,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
            )
            for name, value in expected.items():
                if getattr(hf_config, name) != value:
                    raise ValueError(f"target config requires {name}={value}")
            if self.kind not in ("linear_attention", "full_attention"):
                raise ValueError(f"unsupported layer kind {self.kind}")
            if (
                hf_config.rope_parameters["rope_type"] != "default"
                or hf_config.rope_parameters["partial_rotary_factor"] != 0.25
            ):
                raise ValueError("target text contract requires default 64-dimensional partial RoPE")
            self.compute = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self.w = {}
            round_op = ttnn.UnaryWithParam(
                ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value)
            )
            widen_op = ttnn.UnaryWithParam(
                ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value)
            )
            self.round_widen_chain = [round_op, widen_op]
            self.conv_post_chain = [round_op, widen_op, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), round_op]
            self.beta_chain = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID), round_op, widen_op]
            self.rsqrt_round_chain = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0), round_op, widen_op]

            def upload(x, dtype=ttnn.bfloat16):
                return ttnn.from_torch(
                    x.contiguous(), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
                )

            for name, value in state_dict.items():
                if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                    continue
                if name.endswith("conv1d.weight"):
                    self.taps_fp32 = [upload(value[:, 0, j].reshape(1, 1, -1), ttnn.float32) for j in range(4)]
                elif name.endswith(("A_log", "dt_bias")):
                    continue
                elif name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                    v = value.float() if name == "linear_attn.norm.weight" else value.float() + 1
                    self.w[name] = upload(v.reshape(1, 1, -1), ttnn.float32)
                else:
                    self.w[name] = upload(value.T, ttnn.float32 if name == "mlp.gate.weight" else ttnn.bfloat16)
            if self.kind == "linear_attention":
                keys = ["linear_attn.in_proj_" + suffix + ".weight" for suffix in ("qkv", "z", "a", "b")]
            else:
                keys = ["self_attn." + suffix + "_proj.weight" for suffix in ("q", "k", "v")]
            pieces = [state_dict[key] for key in keys]
            if self.kind == "full_attention":
                qg = pieces[0].reshape(16, 2, 256, 2048)
                pieces = [qg[:, 0].reshape(4096, 2048), pieces[1], pieces[2], qg[:, 1].reshape(4096, 2048)]
            if self.kind == "full_attention":
                self.w["attention_packed"] = upload(torch.cat(pieces, dim=0).T)
            if self.kind == "linear_attention":
                order = torch.arange(2048).reshape(1024, 2).flip(1).reshape(2048)
                self.attention_k_indices = upload(order.to(torch.int32).reshape(1, 1, 2048), ttnn.uint16)
                pair32 = torch.arange(32).reshape(16, 2).flip(1).reshape(1, 1, 32).expand(1, 64, 32)
                self.attention_pair32_indices = ttnn.from_torch(
                    pair32.to(torch.int32).contiguous(),
                    device=mesh_device,
                    dtype=ttnn.uint16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=DRAM,
                )
                self.w["output_only_packed"] = upload(torch.cat((pieces[0][:2048], pieces[1]), dim=0).T)
                state_weight = torch.cat((pieces[0][2048:], pieces[2], pieces[3]), dim=0).T[order]
                self.scan_attention_weight = upload(state_weight.reshape(1, 1, 2048, 6208), ttnn.bfloat16)
                codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float()
                gate_values = -state_dict["linear_attn.A_log"].float().exp().reshape(
                    32, 1
                ) * torch.nn.functional.softplus(
                    codes.reshape(1, -1) + state_dict["linear_attn.dt_bias"].float().reshape(32, 1)
                )
                decay_values = gate_values.exp()
                self.gate_table, self.decay_table = [
                    ttnn.from_torch(
                        value.contiguous(),
                        device=mesh_device,
                        dtype=ttnn.float32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=DRAM,
                    )
                    for value in (gate_values, decay_values)
                ]
                decay_bits = decay_values.contiguous().view(torch.int32).reshape(32, 2, 32768)
                changes = (decay_bits[:, :, 1:32641] != decay_bits[:, :, :32640]).any(
                    dim=(0, 1)
                ).nonzero().flatten() + 1
                lo, hi = (int(changes.min()) - 1, int(changes.max())) if changes.numel() else (0, 0)
                window = hi - lo + 1
                self.compact_decay_table = None
                if window + 127 <= 8192:
                    compact = torch.zeros((32, 16384), dtype=torch.float32)
                    for sign in (0, 1):
                        compact[:, sign * 8192 : sign * 8192 + window] = decay_values[
                            :, sign * 32768 + lo : sign * 32768 + hi + 1
                        ]
                        compact[:, sign * 8192 + window : sign * 8192 + window + 127] = decay_values[
                            :, sign * 32768 + 32641 : (sign + 1) * 32768
                        ]
                    raw = torch.arange(65536, dtype=torch.int64)
                    mag = raw & 32767
                    mapped = mag.clamp(lo, hi) - lo + mag.clamp(min=32640) - 32640 + ((raw & 32768) >> 2)
                    assert torch.equal(
                        compact[:, mapped].contiguous().view(torch.int32), decay_values.view(torch.int32)
                    )
                    self.compact_decay_table = ttnn.from_torch(
                        compact,
                        device=mesh_device,
                        dtype=ttnn.float32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=DRAM,
                    )
                    U = ttnn.UnaryOpType
                    self.decay_finite_index_chain = [
                        ttnn.UnaryWithParam(op, float(value))
                        for op, value in (
                            (U.BITWISE_AND, 32767),
                            (U.MAXIMUM, lo),
                            (U.MINIMUM, hi),
                            (U.SUB_UNARY_SFPU, lo),
                        )
                    ]
                    self.decay_nan_index_chain = [
                        ttnn.UnaryWithParam(op, float(value))
                        for op, value in ((U.BITWISE_AND, 32767), (U.MAXIMUM, 32640), (U.SUB_UNARY_SFPU, 32640))
                    ]
                    self.decay_sign_index_chain = [
                        ttnn.UnaryWithParam(op, float(value))
                        for op, value in ((U.BITWISE_AND, 32768), (U.RIGHT_SHIFT, 2))
                    ]
            for key in keys:
                del self.w[key]
            shared_keys = (
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert_gate.weight",
            )
            shared_weights = [state_dict[key] for key in shared_keys]
            shared_weights[-1] = torch.nn.functional.pad(shared_weights[-1], (0, 0, 0, 31))
            self.w["shared_packed"] = upload(torch.cat(shared_weights, dim=0).T)
            for key in shared_keys:
                del self.w[key]
            gate, up = state_dict["mlp.experts.gate_up_proj"].chunk(2, dim=1)
            self.egu = upload(torch.cat((gate, up), dim=1).transpose(-1, -2).unsqueeze(0))
            self.ed = upload(state_dict["mlp.experts.down_proj"].transpose(-1, -2).unsqueeze(0))
            self.grouped_gu, self.grouped_down = ([], [])
            for bank in range(4):
                gu = state_dict["mlp.experts.gate_up_proj"][bank * 64 : (bank + 1) * 64]
                down = state_dict["mlp.experts.down_proj"][bank * 64 : (bank + 1) * 64]
                self.grouped_gu.append(upload(gu.permute(2, 0, 1).reshape(1, 1, 2048, 64 * 1024)))
                if self.kind == "full_attention":
                    down = torch.nn.functional.pad(down, (0, 0, 0, 64))
                    self.grouped_down.append(upload(down.permute(2, 0, 1).reshape(1, 1, 512, 64 * 2112)))
                else:
                    self.grouped_down.append(upload(down.permute(2, 0, 1).reshape(1, 1, 512, 64 * 2048)))
            self.combine_bias = upload(torch.zeros(1, 1, 1, 2048))
            eye, ones = (torch.eye(32), torch.ones(32, 32))
            i, j = (torch.arange(32)[:, None], torch.arange(32)[None, :])
            masks = torch.cat(
                [((i < 16) & (j < 16)).float(), ((i >= 16) & (j >= 16)).float(), ((i >= 16) & (j < 16)).float()], dim=1
            )
            self.constants = dict(
                zip(
                    ("eye", "tril", "ones", "masks"),
                    [upload(t.reshape(1, 1, *t.shape), ttnn.float32) for t in (eye, ones.tril(), ones, masks)],
                )
            )
            self.decode_linear = {}
            banks = mesh_device.dram_grid_size().x
            bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            cores = ttnn.num_cores_to_corerangeset(32, mesh_device.compute_with_storage_grid_size(), row_wise=True)
            self.norm_memory = ttnn.create_sharded_memory_config(
                (32, 64),
                cores,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            projection_keys = (
                ("linear_attn.out_proj.weight", "output_only_packed")
                if self.kind == "linear_attention"
                else ("attention_packed", "self_attn.o_proj.weight")
            )
            for key in projection_keys:
                w = self.w[key]
                k, n = (w.shape[-2], w.shape[-1])
                wm = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        bank_grid, (k, (n + 96 * banks - 1) // (96 * banks) * 96), ttnn.ShardOrientation.ROW_MAJOR
                    ),
                )
                input_cores = 16 if key == "self_attn.o_proj.weight" else 32
                im = ttnn.create_sharded_memory_config(
                    (32, k // input_cores),
                    ttnn.num_cores_to_corerangeset(
                        input_cores, mesh_device.compute_with_storage_grid_size(), row_wise=True
                    ),
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                per_n = (n + 2047) // 2048
                out_cores = ttnn.num_cores_to_corerangeset(
                    (n + per_n * 32 - 1) // (per_n * 32), mesh_device.compute_with_storage_grid_size(), row_wise=True
                )
                om = ttnn.create_sharded_memory_config(
                    (32, per_n * 32),
                    out_cores,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=2,
                    per_core_M=1,
                    per_core_N=per_n,
                    fused_activation=None,
                    num_workers_per_dram_bank=3,
                    use_multicore_pipeline=True,
                )
                self.decode_linear[key] = (ttnn.to_memory_config(w, wm), im, om, pc)
            self.router_shard = ttnn.create_sharded_memory_config(
                (32, 32),
                ttnn.CoreGrid(y=1, x=1),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            def router_upload(t, dtype, layout):
                return ttnn.from_torch(
                    t, device=mesh_device, dtype=dtype, layout=layout, memory_config=self.router_shard
                )

            self.router_bias = router_upload(torch.zeros(1, 16, 16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            self.router_indices = router_upload(
                torch.arange(256).reshape(1, 16, 16).transpose(1, 2), ttnn.uint16, ttnn.TILE_LAYOUT
            )
            self.router_out = router_upload(torch.zeros(1, 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            self.router_out_ids = router_upload(
                torch.zeros(1, 32, 32, dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
            )
            self.indexed_sparsity = ttnn.from_torch(
                torch.ones(1, 1, 1, 256), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            if self.kind == "linear_attention":
                self.output_norm_bf16 = upload(state_dict["linear_attn.norm.weight"].reshape(128))
            return self
        import torch

        self = cls()
        self.config, self.mesh_device = (hf_config, mesh_device)
        self.layer_idx, self.kind = (layer_idx, hf_config.layer_types[layer_idx])
        expected = dict(
            hidden_size=2048,
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        for name, value in expected.items():
            if getattr(hf_config, name) != value:
                raise ValueError(f"target config requires {name}={value}")
        if self.kind not in ("linear_attention", "full_attention"):
            raise ValueError(f"unsupported layer kind {self.kind}")
        if (
            hf_config.rope_parameters["rope_type"] != "default"
            or hf_config.rope_parameters["partial_rotary_factor"] != 0.25
        ):
            raise ValueError("target text contract requires default 64-dimensional partial RoPE")
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.w = {}
        round_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value))
        widen_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value))
        self.round_widen_chain = [round_op, widen_op]
        self.conv_post_chain = [round_op, widen_op, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), round_op]
        self.beta_chain = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID), round_op, widen_op]
        self.rsqrt_round_chain = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0), round_op, widen_op]

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
            )

        for name, value in state_dict.items():
            if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                continue
            if name.endswith("conv1d.weight"):
                self.taps_fp32 = [upload(value[:, 0, j].reshape(1, 1, -1), ttnn.float32) for j in range(4)]
            elif name.endswith(("A_log", "dt_bias")):
                continue
            elif name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                v = value.float() if name == "linear_attn.norm.weight" else value.float() + 1
                self.w[name] = upload(v.reshape(1, 1, -1), ttnn.float32)
            else:
                self.w[name] = upload(value.T, ttnn.float32 if name == "mlp.gate.weight" else ttnn.bfloat16)
        if self.kind == "linear_attention":
            keys = ["linear_attn.in_proj_" + suffix + ".weight" for suffix in ("qkv", "z", "a", "b")]
        else:
            keys = ["self_attn." + suffix + "_proj.weight" for suffix in ("q", "k", "v")]
        pieces = [state_dict[key] for key in keys]
        if self.kind == "full_attention":
            qg = pieces[0].reshape(16, 2, 256, 2048)
            pieces = [qg[:, 0].reshape(4096, 2048), pieces[1], pieces[2], qg[:, 1].reshape(4096, 2048)]
        if self.kind == "full_attention":
            self.w["attention_packed"] = upload(torch.cat(pieces, dim=0).T)
        if self.kind == "linear_attention":
            order = torch.arange(2048).reshape(1024, 2).flip(1).reshape(2048)
            self.attention_k_indices = upload(order.to(torch.int32).reshape(1, 1, 2048), ttnn.uint16)
            pair32 = torch.arange(32).reshape(16, 2).flip(1).reshape(1, 1, 32).expand(1, 64, 32)
            self.attention_pair32_indices = ttnn.from_torch(
                pair32.to(torch.int32).contiguous(),
                device=mesh_device,
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=DRAM,
            )
            self.w["output_only_packed"] = upload(torch.cat((pieces[0][:2048], pieces[1]), dim=0).T)
            state_weight = torch.cat((pieces[0][2048:], pieces[2], pieces[3]), dim=0).T[order]
            self.scan_attention_weight = upload(state_weight.reshape(1, 1, 2048, 6208), ttnn.bfloat16)
            self.scan_output_weight = upload(
                torch.cat((pieces[0][:2048], pieces[1]), dim=0).T[order].reshape(1, 1, 2048, 6144), ttnn.bfloat16
            )
            codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float()
            gate_values = -state_dict["linear_attn.A_log"].float().exp().reshape(32, 1) * torch.nn.functional.softplus(
                codes.reshape(1, -1) + state_dict["linear_attn.dt_bias"].float().reshape(32, 1)
            )
            decay_values = gate_values.exp()
            self.gate_table, self.decay_table = [
                ttnn.from_torch(
                    value.contiguous(),
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=DRAM,
                )
                for value in (gate_values, decay_values)
            ]
            decay_bits = decay_values.contiguous().view(torch.int32).reshape(32, 2, 32768)
            changes = (decay_bits[:, :, 1:32641] != decay_bits[:, :, :32640]).any(dim=(0, 1)).nonzero().flatten() + 1
            lo, hi = (int(changes.min()) - 1, int(changes.max())) if changes.numel() else (0, 0)
            window = hi - lo + 1
            self.compact_decay_table = None
            if window + 127 <= 8192:
                compact = torch.zeros((32, 16384), dtype=torch.float32)
                for sign in (0, 1):
                    compact[:, sign * 8192 : sign * 8192 + window] = decay_values[
                        :, sign * 32768 + lo : sign * 32768 + hi + 1
                    ]
                    compact[:, sign * 8192 + window : sign * 8192 + window + 127] = decay_values[
                        :, sign * 32768 + 32641 : (sign + 1) * 32768
                    ]
                raw = torch.arange(65536, dtype=torch.int64)
                mag = raw & 32767
                mapped = mag.clamp(lo, hi) - lo + mag.clamp(min=32640) - 32640 + ((raw & 32768) >> 2)
                assert torch.equal(compact[:, mapped].contiguous().view(torch.int32), decay_values.view(torch.int32))
                self.compact_decay_table = ttnn.from_torch(
                    compact, device=mesh_device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM
                )
                U = ttnn.UnaryOpType
                self.decay_finite_index_chain = [
                    ttnn.UnaryWithParam(op, float(value))
                    for op, value in ((U.BITWISE_AND, 32767), (U.MAXIMUM, lo), (U.MINIMUM, hi), (U.SUB_UNARY_SFPU, lo))
                ]
                self.decay_nan_index_chain = [
                    ttnn.UnaryWithParam(op, float(value))
                    for op, value in ((U.BITWISE_AND, 32767), (U.MAXIMUM, 32640), (U.SUB_UNARY_SFPU, 32640))
                ]
                self.decay_sign_index_chain = [
                    ttnn.UnaryWithParam(op, float(value)) for op, value in ((U.BITWISE_AND, 32768), (U.RIGHT_SHIFT, 2))
                ]
        for key in keys:
            del self.w[key]
        shared_keys = (
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert_gate.weight",
        )
        shared_weights = [state_dict[key] for key in shared_keys]
        shared_weights[-1] = torch.nn.functional.pad(shared_weights[-1], (0, 0, 0, 31))
        self.w["shared_packed"] = upload(torch.cat(shared_weights, dim=0).T)
        for key in shared_keys:
            del self.w[key]
        gate, up = state_dict["mlp.experts.gate_up_proj"].chunk(2, dim=1)
        self.egu = upload(torch.cat((gate, up), dim=1).transpose(-1, -2).unsqueeze(0))
        self.ed = upload(state_dict["mlp.experts.down_proj"].transpose(-1, -2).unsqueeze(0))
        self.grouped_gu, self.grouped_down = ([], [])
        for bank in range(4):
            gu = state_dict["mlp.experts.gate_up_proj"][bank * 64 : (bank + 1) * 64]
            down = state_dict["mlp.experts.down_proj"][bank * 64 : (bank + 1) * 64]
            self.grouped_gu.append(upload(gu.permute(2, 0, 1).reshape(1, 1, 2048, 64 * 1024)))
            if self.kind == "full_attention":
                down = torch.nn.functional.pad(down, (0, 0, 0, 64))
                self.grouped_down.append(upload(down.permute(2, 0, 1).reshape(1, 1, 512, 64 * 2112)))
            else:
                self.grouped_down.append(upload(down.permute(2, 0, 1).reshape(1, 1, 512, 64 * 2048)))
        self.combine_bias = upload(torch.zeros(1, 1, 1, 2048))
        eye, ones = (torch.eye(32), torch.ones(32, 32))
        i, j = (torch.arange(32)[:, None], torch.arange(32)[None, :])
        masks = torch.cat(
            [((i < 16) & (j < 16)).float(), ((i >= 16) & (j >= 16)).float(), ((i >= 16) & (j < 16)).float()], dim=1
        )
        self.constants = dict(
            zip(
                ("eye", "tril", "ones", "masks"),
                [upload(t.reshape(1, 1, *t.shape), ttnn.float32) for t in (eye, ones.tril(), ones, masks)],
            )
        )
        self.decode_linear = {}
        banks = mesh_device.dram_grid_size().x
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        cores = ttnn.num_cores_to_corerangeset(32, mesh_device.compute_with_storage_grid_size(), row_wise=True)
        self.norm_memory = ttnn.create_sharded_memory_config(
            (32, 64),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        projection_keys = (
            ("linear_attn.out_proj.weight", "output_only_packed")
            if self.kind == "linear_attention"
            else ("attention_packed", "self_attn.o_proj.weight")
        )
        for key in projection_keys:
            w = self.w[key]
            k, n = (w.shape[-2], w.shape[-1])
            wm = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    bank_grid, (k, (n + 96 * banks - 1) // (96 * banks) * 96), ttnn.ShardOrientation.ROW_MAJOR
                ),
            )
            input_cores = 16 if key == "self_attn.o_proj.weight" else 32
            im = ttnn.create_sharded_memory_config(
                (32, k // input_cores),
                ttnn.num_cores_to_corerangeset(
                    input_cores, mesh_device.compute_with_storage_grid_size(), row_wise=True
                ),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            per_n = (n + 2047) // 2048
            out_cores = ttnn.num_cores_to_corerangeset(
                (n + per_n * 32 - 1) // (per_n * 32), mesh_device.compute_with_storage_grid_size(), row_wise=True
            )
            om = ttnn.create_sharded_memory_config(
                (32, per_n * 32),
                out_cores,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=2,
                per_core_M=1,
                per_core_N=per_n,
                fused_activation=None,
                num_workers_per_dram_bank=3,
                use_multicore_pipeline=True,
            )
            self.decode_linear[key] = (ttnn.to_memory_config(w, wm), im, om, pc)
        self.router_shard = ttnn.create_sharded_memory_config(
            (32, 32),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        def router_upload(t, dtype, layout):
            return ttnn.from_torch(t, device=mesh_device, dtype=dtype, layout=layout, memory_config=self.router_shard)

        self.router_bias = router_upload(torch.zeros(1, 16, 16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.router_indices = router_upload(
            torch.arange(256).reshape(1, 16, 16).transpose(1, 2), ttnn.uint16, ttnn.TILE_LAYOUT
        )
        self.router_out = router_upload(torch.zeros(1, 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.router_out_ids = router_upload(
            torch.zeros(1, 32, 32, dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
        )
        self.indexed_sparsity = ttnn.from_torch(
            torch.ones(1, 1, 1, 256), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        if self.kind == "linear_attention":
            self.output_norm_bf16 = upload(state_dict["linear_attn.norm.weight"].reshape(128))
            output_order = torch.arange(4096).reshape(2048, 2).flip(1).reshape(4096)
            self.scan_attention_out_weight = upload(
                state_dict["linear_attn.out_proj.weight"].T[output_order].reshape(1, 1, 4096, 2048)
            )
            output_pair32 = torch.arange(32).reshape(16, 2).flip(1).reshape(1, 1, 32).expand(1, 128, 32)
            self.output_pair32_indices = ttnn.from_torch(
                output_pair32.to(torch.int32).contiguous(),
                device=mesh_device,
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=DRAM,
            )
        router_order = torch.arange(2048).reshape(1024, 2).flip(1).reshape(2048)
        self.scan_router_weight = upload(state_dict["mlp.gate.weight"].T[router_order].reshape(1, 1, 2048, 256))
        router_pair32 = torch.arange(32).reshape(16, 2).flip(1).reshape(1, 1, 32).expand(1, 64, 32)
        self.router_pair32_indices = ttnn.from_torch(
            router_pair32.to(torch.int32).contiguous(),
            device=mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        self.inline_selector = InlineSelector(ttnn, mesh_device, PORTABLE_CPP)
        silu_codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        self.silu_table = ttnn.from_torch(
            torch.nn.functional.silu(silu_codes.float()).reshape(1, 65536),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        if self.kind == "linear_attention":

            def upload_chunk_constant(x):
                return ttnn.from_torch(
                    x, device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
                )

            self.chunk64_state = Chunk64OutputGraph(
                ttnn, mesh_device, upload_chunk_constant, variant="sfpu_fma", cumsum="compensated"
            )
        if self.kind == "linear_attention":
            assert self.config.rms_norm_eps == 1e-06, "Measured norm-tail epsilon contract"
            self.decode_weighted_norm_tail = WeightedNormTail(
                ttnn, mesh_device, state_dict["linear_attn.norm.weight"], debug=False
            )
            self.decode_local_silu = LocalSiluMultiply(ttnn, mesh_device, debug=False)
        if self.kind == "linear_attention":
            self.streaming_projection_scan = StreamingProjectionScan(ttnn, mesh_device)
            self.direct_coefficient_scan = DirectCoefficientStreamingScan(ttnn, mesh_device)
        if self.kind == "linear_attention":
            self.decode_convolution = DecodeConvolution(ttnn, mesh_device, self.taps_fp32)
        if self.kind == "linear_attention":
            self.norm_terminal_scan = NormTerminalScan(ttnn, mesh_device)
        if self.kind == "linear_attention":
            self.decode_decay_lookup = DecodeDecayLookup(
                ttnn,
                mesh_device,
                self.compact_decay_table if self.compact_decay_table is not None else self.decay_table,
                lo if self.compact_decay_table is not None else None,
                hi if self.compact_decay_table is not None else None,
            )
        return self

    def linear(
        self,
        x,
        key,
        dtype=ttnn.bfloat16,
        *,
        projection_k=512,
        ordered_out=False,
        ordered_router=False,
        terminal_scan=False,
    ):
        if self.kind == "full_attention":
            if x.shape[-2] == 1 and key == "attention_packed" and (self.kind == "linear_attention"):
                x = ttnn.to_layout(ttnn.to_memory_config(x, DRAM), ttnn.ROW_MAJOR_LAYOUT)
                x = ttnn.reshape(x, (1, 64, 32))
                x = ttnn.gather(x, -1, self.attention_pair32_indices)
                blocks = 2048 // projection_k
                coefficients = ttnn.to_layout(ttnn.reshape(x, (blocks, 1, projection_k, 1)), ttnn.TILE_LAYOUT)
                weight = ttnn.reshape(self.scan_attention_weight, (blocks, 1, projection_k, 6208))
                products = ttnn.multiply(
                    weight,
                    coefficients,
                    dtype=ttnn.float32,
                    fast_and_approximate_mode=False,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                prefixes = ttnn.cumsum(
                    products, -2, dtype=ttnn.float32, disable_compensated_sum=True, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                del products
                tail = prefixes[:, :, projection_k - 32 : projection_k, :]
                del prefixes
                partial = tail[:, :, 31:32, :]
                del tail
                result = ttnn.empty(
                    (1, 1, 1, 6208),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                result = ttnn.experimental.fast_reduce_nc(
                    partial,
                    dims=[0],
                    output=result,
                    compute_kernel_config=self.compute,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                return ttnn.reshape(ttnn.typecast(result, dtype), (1, 1, 6208))
            if key == "attention_packed" and self.kind == "linear_attention":
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
                outputs = []
                for row in range(x.shape[-2]):
                    projected = self.linear(x[:, row : row + 1, :], key, dtype, projection_k=1024)
                    outputs.append(ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM))
                    del projected
                return ttnn.to_layout(
                    ttnn.concat(outputs, dim=1, memory_config=DRAM), ttnn.TILE_LAYOUT, memory_config=DRAM
                )
            if x.shape[-2] == 1 and key in self.decode_linear:
                w, im, om, pc = self.decode_linear[key]
                result = ttnn.linear(
                    ttnn.to_memory_config(x, im),
                    w,
                    bias=None,
                    dtype=dtype,
                    compute_kernel_config=self.compute,
                    memory_config=om,
                    program_config=pc,
                )
                result = ttnn.to_memory_config(result, DRAM)
                return result
            return ttnn.linear(x, self.w[key], dtype=dtype, compute_kernel_config=self.compute, memory_config=DRAM)
        if ordered_router and x.shape[-2] == 1 and (key == "mlp.gate.weight"):
            if terminal_scan:
                projection_k = 512
                blocks = 2048 // projection_k
                width = 256
                weight = ttnn.reshape(self.scan_router_weight, (blocks, 1, projection_k, width))
                partial = self.direct_coefficient_scan(weight, x)
                result = ttnn.empty(
                    (1, 1, 1, width),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                result = ttnn.experimental.fast_reduce_nc(
                    partial,
                    dims=[0],
                    output=result,
                    compute_kernel_config=self.compute,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                return ttnn.reshape(ttnn.typecast(result, dtype, memory_config=DRAM), (1, 1, width))
            projection_k = 512
            x = ttnn.to_layout(ttnn.to_memory_config(x, DRAM), ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(x, (1, 64, 32))
            x = ttnn.gather(x, -1, self.router_pair32_indices)
            blocks = 2048 // projection_k
            width = 256
            coefficients = ttnn.to_layout(ttnn.reshape(x, (blocks, 1, projection_k, 1)), ttnn.TILE_LAYOUT)
            weight = ttnn.reshape(self.scan_router_weight, (blocks, 1, projection_k, width))
            if terminal_scan:
                partial = self.streaming_projection_scan(weight, coefficients)
            else:
                products = ttnn.multiply(
                    weight,
                    coefficients,
                    dtype=ttnn.float32,
                    fast_and_approximate_mode=False,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                prefixes = ttnn.cumsum(
                    products, -2, dtype=ttnn.float32, disable_compensated_sum=True, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                del products
                tail = prefixes[:, :, projection_k - 32 : projection_k, :]
                del prefixes
                partial = tail[:, :, 31:32, :]
                del tail
            result = ttnn.empty(
                (1, 1, 1, width),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            result = ttnn.experimental.fast_reduce_nc(
                partial,
                dims=[0],
                output=result,
                compute_kernel_config=self.compute,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            return ttnn.to_memory_config(ttnn.reshape(ttnn.typecast(result, dtype), (1, 1, width)), DRAM)
        if (
            ordered_out
            and x.shape[-2] == 1
            and (key == "linear_attn.out_proj.weight")
            and (self.kind == "linear_attention")
        ):
            if terminal_scan:
                projection_k = 512
                blocks = 4096 // projection_k
                width = 2048
                weight = ttnn.reshape(self.scan_attention_out_weight, (blocks, 1, projection_k, width))
                partial = self.direct_coefficient_scan(weight, x)
                result = ttnn.empty(
                    (1, 1, 1, width),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                result = ttnn.experimental.fast_reduce_nc(
                    partial,
                    dims=[0],
                    output=result,
                    compute_kernel_config=self.compute,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                return ttnn.reshape(ttnn.typecast(result, dtype, memory_config=DRAM), (1, 1, width))
            projection_k = 512
            x = ttnn.to_layout(ttnn.to_memory_config(x, DRAM), ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(x, (1, 128, 32))
            x = ttnn.gather(x, -1, self.output_pair32_indices)
            blocks = 4096 // projection_k
            width = 2048
            coefficients = ttnn.to_layout(ttnn.reshape(x, (blocks, 1, projection_k, 1)), ttnn.TILE_LAYOUT)
            weight = ttnn.reshape(self.scan_attention_out_weight, (blocks, 1, projection_k, width))
            if terminal_scan:
                partial = self.streaming_projection_scan(weight, coefficients)
            else:
                products = ttnn.multiply(
                    weight,
                    coefficients,
                    dtype=ttnn.float32,
                    fast_and_approximate_mode=False,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                prefixes = ttnn.cumsum(
                    products, -2, dtype=ttnn.float32, disable_compensated_sum=True, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                del products
                tail = prefixes[:, :, projection_k - 32 : projection_k, :]
                del prefixes
                partial = tail[:, :, 31:32, :]
                del tail
            result = ttnn.empty(
                (1, 1, 1, width),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            result = ttnn.experimental.fast_reduce_nc(
                partial,
                dims=[0],
                output=result,
                compute_kernel_config=self.compute,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            return ttnn.to_memory_config(ttnn.reshape(ttnn.typecast(result, dtype), (1, 1, width)), DRAM)
        if x.shape[-2] == 1 and key in ("attention_packed", "output_only_packed") and (self.kind == "linear_attention"):
            if terminal_scan:
                blocks = 2048 // projection_k
                width = 6208 if key == "attention_packed" else 6144
                weight = ttnn.reshape(
                    self.scan_attention_weight if key == "attention_packed" else self.scan_output_weight,
                    (blocks, 1, projection_k, width),
                )
                partial = self.direct_coefficient_scan(weight, x)
                result = ttnn.empty(
                    (1, 1, 1, width),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                result = ttnn.experimental.fast_reduce_nc(
                    partial,
                    dims=[0],
                    output=result,
                    compute_kernel_config=self.compute,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                return ttnn.reshape(ttnn.typecast(result, dtype), (1, 1, width))
            x = ttnn.to_layout(ttnn.to_memory_config(x, DRAM), ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(x, (1, 64, 32))
            x = ttnn.gather(x, -1, self.attention_pair32_indices)
            blocks = 2048 // projection_k
            width = 6208 if key == "attention_packed" else 6144
            coefficients = ttnn.to_layout(ttnn.reshape(x, (blocks, 1, projection_k, 1)), ttnn.TILE_LAYOUT)
            weight = ttnn.reshape(
                self.scan_attention_weight if key == "attention_packed" else self.scan_output_weight,
                (blocks, 1, projection_k, width),
            )
            if terminal_scan:
                partial = self.streaming_projection_scan(weight, coefficients)
            else:
                products = ttnn.multiply(
                    weight,
                    coefficients,
                    dtype=ttnn.float32,
                    fast_and_approximate_mode=False,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                prefixes = ttnn.cumsum(
                    products, -2, dtype=ttnn.float32, disable_compensated_sum=True, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                del products
                tail = prefixes[:, :, projection_k - 32 : projection_k, :]
                del prefixes
                partial = tail[:, :, 31:32, :]
                del tail
            result = ttnn.empty(
                (1, 1, 1, width),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            result = ttnn.experimental.fast_reduce_nc(
                partial,
                dims=[0],
                output=result,
                compute_kernel_config=self.compute,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            return ttnn.reshape(ttnn.typecast(result, dtype), (1, 1, width))
        if key in ("attention_packed", "output_only_packed") and self.kind == "linear_attention":
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            outputs = []
            for row in range(x.shape[-2]):
                projected = self.linear(x[:, row : row + 1, :], key, dtype, projection_k=1024)
                outputs.append(ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM))
                del projected
            return ttnn.to_layout(ttnn.concat(outputs, dim=1, memory_config=DRAM), ttnn.TILE_LAYOUT, memory_config=DRAM)
        if x.shape[-2] == 1 and key in self.decode_linear:
            w, im, om, pc = self.decode_linear[key]
            result = ttnn.linear(
                ttnn.to_memory_config(x, im),
                w,
                bias=None,
                dtype=dtype,
                compute_kernel_config=self.compute,
                memory_config=om,
                program_config=pc,
            )
            result = ttnn.to_memory_config(result, DRAM)
            return result
        return ttnn.linear(x, self.w[key], dtype=dtype, compute_kernel_config=self.compute, memory_config=DRAM)

    def norm(self, x, key):
        if self.kind == "full_attention":
            if key == "input_layernorm.weight" and self.kind == "linear_attention":
                rows = x.shape[-2]
                original_memory = x.memory_config()
                wide = ttnn.to_memory_config(x, DRAM)
                if rows == 1:
                    squares = ttnn.reshape(
                        ttnn.multiply(wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False), (4, 16, 1, 32)
                    )
                    total = ttnn.empty(
                        (1, 1, 1, 32),
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=DRAM,
                    )
                    total = ttnn.experimental.fast_reduce_nc(
                        squares, dims=[0, 1], output=total, compute_kernel_config=self.compute
                    )
                else:
                    squares = ttnn.reshape(
                        ttnn.multiply(wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False),
                        (1, rows, 64, 32),
                    )
                    squares = ttnn.permute(squares, (0, 2, 1, 3))
                    squares = ttnn.reshape(squares, (4, 16, rows, 32))
                    total = ttnn.empty(
                        (1, 1, rows, 32),
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=DRAM,
                    )
                    total = ttnn.experimental.fast_reduce_nc(
                        squares, dims=[0, 1], output=total, compute_kernel_config=self.compute
                    )
                groups = ttnn.reshape(total, (rows, 1, 4, 8))
                lanes = ttnn.cumsum(groups, -2, dtype=ttnn.float32, disable_compensated_sum=True)[:, :, 3:4, :]
                total = ttnn.cumsum(lanes, -1, dtype=ttnn.float32, disable_compensated_sum=True)[:, :, :, 7:8]
                shifted = ttnn.reshape(
                    ttnn.add(
                        total,
                        self.config.rms_norm_eps,
                        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1 / 2048)],
                    ),
                    (1, rows, 1),
                )

                def split(value):
                    high = ttnn.unary_chain(value, self.round_widen_chain)
                    return (high, ttnn.subtract(value, high))

                root = ttnn.sqrt(shifted, fast_and_approximate_mode=False)
                high, low = split(root)
                residual = ttnn.subtract(
                    ttnn.subtract(
                        ttnn.subtract(shifted, high, input_tensor_b_activations=[ttnn.UnaryOpType.SQUARE]),
                        ttnn.multiply(
                            high,
                            low,
                            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 2.0)],
                        ),
                    ),
                    low,
                    input_tensor_b_activations=[ttnn.UnaryOpType.SQUARE],
                )
                root = ttnn.add(
                    root,
                    ttnn.multiply(
                        residual,
                        root,
                        input_tensor_b_activations=[
                            ttnn.UnaryOpType.RECIP,
                            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 0.5),
                        ],
                    ),
                )
                inverse = ttnn.reciprocal(root)
                rh, rl = split(root)
                ih, il = split(inverse)
                residual = ttnn.subtract(
                    ttnn.subtract(
                        ttnn.subtract(
                            ttnn.multiply(rh, ih, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSUB, 1.0)]),
                            ttnn.multiply(rh, il),
                        ),
                        ttnn.multiply(rl, ih),
                    ),
                    ttnn.multiply(rl, il),
                )
                inverse = ttnn.add(inverse, ttnn.multiply(inverse, residual))
                normalized = ttnn.multiply(wide, inverse, dtype=ttnn.float32, fast_and_approximate_mode=False)
                result = ttnn.multiply(normalized, self.w[key], dtype=ttnn.bfloat16, fast_and_approximate_mode=False)
                return ttnn.to_memory_config(result, original_memory)
            result = ttnn.rms_norm(
                x,
                weight=self.w[key],
                memory_config=ttnn.L1_MEMORY_CONFIG
                if key == "post_attention_layernorm.weight" and 1 < x.shape[-2] <= 2048
                else x.memory_config(),
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.compute,
            )
            return result if result.dtype == ttnn.bfloat16 else ttnn.typecast(result, ttnn.bfloat16)
        if key in ("input_layernorm.weight", "post_attention_layernorm.weight") and self.kind == "linear_attention":
            rows = x.shape[-2]
            original_memory = x.memory_config()
            wide = ttnn.to_memory_config(x, DRAM)
            if rows == 1:
                squares = ttnn.reshape(
                    ttnn.multiply(wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False), (4, 16, 1, 32)
                )
                total = ttnn.empty(
                    (1, 1, 1, 32),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=DRAM,
                )
                total = ttnn.experimental.fast_reduce_nc(
                    squares, dims=[0, 1], output=total, compute_kernel_config=self.compute
                )
            else:
                squares = ttnn.reshape(
                    ttnn.multiply(wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False), (1, rows, 64, 32)
                )
                squares = ttnn.permute(squares, (0, 2, 1, 3))
                squares = ttnn.reshape(squares, (4, 16, rows, 32))
                total = ttnn.empty(
                    (1, 1, rows, 32),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=DRAM,
                )
                total = ttnn.experimental.fast_reduce_nc(
                    squares, dims=[0, 1], output=total, compute_kernel_config=self.compute
                )
            groups = ttnn.reshape(total, (rows, 1, 4, 8))
            total = self.norm_terminal_scan(groups)
            shifted = ttnn.reshape(
                ttnn.add(
                    total,
                    self.config.rms_norm_eps,
                    input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1 / 2048)],
                ),
                (1, rows, 1),
            )

            def split(value):
                high = ttnn.unary_chain(value, self.round_widen_chain)
                return (high, ttnn.subtract(value, high))

            root = ttnn.sqrt(shifted, fast_and_approximate_mode=False)
            high, low = split(root)
            residual = ttnn.subtract(
                ttnn.subtract(
                    ttnn.subtract(shifted, high, input_tensor_b_activations=[ttnn.UnaryOpType.SQUARE]),
                    ttnn.multiply(
                        high,
                        low,
                        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 2.0)],
                    ),
                ),
                low,
                input_tensor_b_activations=[ttnn.UnaryOpType.SQUARE],
            )
            root = ttnn.add(
                root,
                ttnn.multiply(
                    residual,
                    root,
                    input_tensor_b_activations=[
                        ttnn.UnaryOpType.RECIP,
                        ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 0.5),
                    ],
                ),
            )
            inverse = ttnn.reciprocal(root)
            rh, rl = split(root)
            ih, il = split(inverse)
            residual = ttnn.subtract(
                ttnn.subtract(
                    ttnn.subtract(
                        ttnn.multiply(rh, ih, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSUB, 1.0)]),
                        ttnn.multiply(rh, il),
                    ),
                    ttnn.multiply(rl, ih),
                ),
                ttnn.multiply(rl, il),
            )
            inverse = ttnn.add(inverse, ttnn.multiply(inverse, residual))
            normalized = ttnn.multiply(wide, inverse, dtype=ttnn.float32, fast_and_approximate_mode=False)
            result = ttnn.multiply(normalized, self.w[key], dtype=ttnn.bfloat16, fast_and_approximate_mode=False)
            return ttnn.to_memory_config(result, original_memory)
        result = ttnn.rms_norm(
            x,
            weight=self.w[key],
            memory_config=ttnn.L1_MEMORY_CONFIG
            if key == "post_attention_layernorm.weight" and 1 < x.shape[-2] <= 2048
            else x.memory_config(),
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
        )
        return result if result.dtype == ttnn.bfloat16 else ttnn.typecast(result, ttnn.bfloat16)

    def new_recurrent_state(self):
        """Setup boundary, called before capture."""
        import torch

        def zeros(shape, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                torch.zeros(shape), device=self.mesh_device, dtype=dtype, layout=layout, memory_config=DRAM
            )

        return {
            "conv": zeros((1, 3, 8192), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            "recurrent": zeros((1, 32, 128, 128), ttnn.float32),
        }

    def _recurrent_sum(self, products):
        head_rows = ttnn.permute(products, (0, 2, 1, 3))
        chunks = ttnn.reshape(head_rows, (8, 16, 32, 128))
        total = ttnn.experimental.fast_reduce_nc(
            chunks, dims=[0, 1], compute_kernel_config=self.compute, memory_config=products.memory_config()
        )
        return ttnn.permute(total, (0, 2, 1, 3))

    def _recurrent_step(self, q, k, v, g, beta, state, *, decay_value=None, round_state=False, output_state=None):
        """FP32 SFPU state update; native FDR narrows FP32 state through TF32 FPU inputs."""
        q = ttnn.reshape(ttnn.permute(q, (0, 2, 3, 1)), (1, 16, 1, 128, 1))
        k = ttnn.reshape(ttnn.permute(k, (0, 2, 3, 1)), (1, 16, 1, 128, 1))
        v = ttnn.permute(ttnn.typecast(v, ttnn.float32), (0, 2, 1, 3))
        beta = reshape(beta, (1, 32, 1, 1))
        state_activations = self.round_widen_chain if round_state else []
        if decay_value is None:
            decay = reshape(g, (1, 32, 1, 1))
            decayed = ttnn.multiply(
                state,
                decay,
                input_tensor_a_activations=state_activations,
                input_tensor_b_activations=[ttnn.UnaryOpType.EXP],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            decayed = ttnn.multiply(
                state,
                reshape(decay_value, (1, 32, 1, 1)),
                input_tensor_a_activations=state_activations,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        decayed_heads = ttnn.reshape(decayed, (1, 16, 2, 128, 128))
        memory = self._recurrent_sum(ttnn.reshape(ttnn.multiply(decayed_heads, k), (1, 32, 128, 128)))
        delta = ttnn.multiply(ttnn.subtract(v, memory, memory_config=ttnn.L1_MEMORY_CONFIG), beta)
        update = ttnn.multiply(k, ttnn.reshape(delta, (1, 16, 2, 1, 128)), memory_config=ttnn.L1_MEMORY_CONFIG)
        new = ttnn.add(decayed, ttnn.reshape(update, (1, 32, 128, 128)), output_tensor=output_state)
        output = self._recurrent_sum(
            ttnn.reshape(
                ttnn.multiply(
                    ttnn.reshape(new, (1, 16, 2, 128, 128)),
                    q,
                    input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 128 ** (-0.5))],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                ),
                (1, 32, 128, 128),
            )
        )
        return (ttnn.permute(output, (0, 2, 1, 3)), new)

    def _linear_attention(self, x, state, *, decode, cache_boundary=True, projection_k=512):
        s = x.shape[1]
        state_projected = self.linear(
            x, "attention_packed", ttnn.bfloat16, projection_k=projection_k, terminal_scan=decode and cache_boundary
        )
        output_projected = self.linear(
            x, "output_only_packed", projection_k=projection_k, terminal_scan=decode and cache_boundary
        )
        qkv = ttnn.concat([output_projected[:, :, :2048], state_projected[:, :, :6144]], dim=-1)
        projection_a = state_projected[:, :, 6144:6176]
        projection_b = state_projected[:, :, 6176:6208]
        projection_z = output_projected[:, :, 2048:]
        if decode and cache_boundary:
            conv, _conv_partial = self.decode_convolution(qkv, state["conv"], debug=False)
            q, k, v = (conv[:, :, :2048], conv[:, :, 2048:4096], conv[:, :, 4096:])
        else:
            history = ttnn.concat([state["conv"], ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)], dim=1)
            tap_input = ttnn.to_layout(history[:, :s, :], ttnn.TILE_LAYOUT)
            accum = ttnn.multiply(tap_input, self.taps_fp32[0], dtype=ttnn.float32, fast_and_approximate_mode=False)
            history_rm = history
            for j in range(1, 4):
                tap_input = ttnn.to_layout(history_rm[:, j : j + s, :], ttnn.TILE_LAYOUT, dtype=ttnn.float32)
                accum = ttnn.addcmul(accum, tap_input, self.taps_fp32[j])
            conv = ttnn.unary_chain(accum, self.conv_post_chain)
            q, k, v = (conv[:, :, :2048], conv[:, :, 2048:4096], conv[:, :, 4096:])
            ttnn.slice(history_rm, (0, s, 0), (1, s + 3, 8192), output_tensor=state["conv"])
        v = reshape(v[:, :s, :], (1, s, 32, 128)) if decode else v

        def l2(a):
            def rounded(x):
                return ttnn.unary_chain(x, self.round_widen_chain)

            wide = ttnn.typecast(a, ttnn.float32)
            squares = ttnn.multiply(wide, wide, activations=self.round_widen_chain)
            sums = rounded(ttnn.sum(squares, dim=-1, keepdim=True, compute_kernel_config=self.compute))
            shifted = ttnn.add(sums, 1e-06, activations=self.round_widen_chain)
            inverse = ttnn.unary_chain(shifted, self.rsqrt_round_chain)
            return ttnn.multiply(wide, inverse, activations=self.round_widen_chain)

        qk = reshape(ttnn.concat([q[:, :s, :], k[:, :s, :]], dim=0), (2, s, 16, 128))
        qk = l2(qk)
        q, k = (qk[:1, :, :, :], qk[1:, :, :, :])
        b = ttnn.typecast(projection_b, ttnn.float32)
        if decode and cache_boundary:
            g = None
            decay_value = self.decode_decay_lookup(projection_a)
        else:
            raw_indices = ttnn.bitcast(projection_a, ttnn.uint16)
            if decode and self.compact_decay_table is not None:
                raw_indices = ttnn.typecast(raw_indices, ttnn.uint32)
                magnitude_indices = ttnn.add(
                    raw_indices,
                    raw_indices,
                    dtype=ttnn.uint32,
                    input_tensor_a_activations=self.decay_finite_index_chain,
                    input_tensor_b_activations=self.decay_nan_index_chain,
                )
                raw_indices = ttnn.add(
                    magnitude_indices,
                    raw_indices,
                    dtype=ttnn.uint32,
                    input_tensor_b_activations=self.decay_sign_index_chain,
                )
            indices = ttnn.reshape(ttnn.transpose(raw_indices, -1, -2), (32, s))
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)

            def lookup(table):
                values = ttnn.transpose(ttnn.gather(table, 1, indices), -1, -2)
                return ttnn.to_layout(ttnn.reshape(values, (1, s, 32)), ttnn.TILE_LAYOUT)

            g = None if decode else lookup(self.gate_table)
            decay_value = None
            if decode:
                decay_value = lookup(
                    self.compact_decay_table if self.compact_decay_table is not None else self.decay_table
                )
        beta = ttnn.unary_chain(b, self.beta_chain)
        if decode:
            initial = state["recurrent"]
            o, new = self._recurrent_step(
                q,
                k,
                v,
                g,
                beta,
                initial,
                decay_value=decay_value,
                round_state=cache_boundary,
                output_state=state["recurrent"],
            )
        else:
            qq = ttnn.permute(ttnn.repeat_interleave(q, 2, dim=2), (0, 2, 1, 3))
            kk = ttnn.permute(ttnn.repeat_interleave(k, 2, dim=2), (0, 2, 1, 3))
            vv = ttnn.permute(ttnn.typecast(reshape(v, (1, s, 32, 128)), ttnn.float32), (0, 2, 1, 3))
            bb = ttnn.permute(reshape(beta, (1, s, 32, 1)), (0, 2, 1, 3))
            gg = ttnn.permute(reshape(g, (1, s, 32, 1)), (0, 2, 1, 3))
            o, new, _ = self.chunk64_state(qq, kk, vv, gg, bb, state["recurrent"])
        if not decode:
            ttnn.copy(reshape(new, (1, 32, 128, 128)), state["recurrent"])
        if not decode:
            core = reshape(ttnn.typecast(o, ttnn.bfloat16), (1, s * 32, 128))
            wide = ttnn.typecast(core, ttnn.float32, memory_config=DRAM)
            squares = ttnn.multiply(wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=DRAM)
            sums = ttnn.sum(squares, dim=-1, keepdim=True, compute_kernel_config=self.compute, memory_config=DRAM)
            variance = ttnn.multiply(
                sums, 1 / 128, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=DRAM
            )
            shifted = ttnn.add(variance, self.config.rms_norm_eps, memory_config=DRAM)
            inverse = ttnn.rsqrt(shifted, fast_and_approximate_mode=False, memory_config=DRAM)
            normalized_fp32 = ttnn.multiply(
                wide, inverse, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=DRAM
            )
            normalized = ttnn.typecast(normalized_fp32, ttnn.bfloat16, memory_config=DRAM)
            weighted = ttnn.multiply(
                normalized,
                self.w["linear_attn.norm.weight"],
                dtype=ttnn.float32,
                fast_and_approximate_mode=False,
                activations=self.round_widen_chain,
                memory_config=DRAM,
            )
            silu_indices = ttnn.bitcast(projection_z, ttnn.uint16)
            silu_indices = ttnn.reshape(ttnn.to_layout(silu_indices, ttnn.ROW_MAJOR_LAYOUT), (1, s * 4096))
            gate_rows = ttnn.gather(self.silu_table, 1, silu_indices)
            gate_rows = ttnn.to_layout(ttnn.reshape(gate_rows, (1, s * 32, 128)), ttnn.TILE_LAYOUT)
            gated = ttnn.multiply(
                weighted, gate_rows, dtype=ttnn.bfloat16, fast_and_approximate_mode=False, memory_config=DRAM
            )
            return self.linear(reshape(gated, (1, s, 4096)), "linear_attn.out_proj.weight")
        core_bf16 = ttnn.typecast(o, ttnn.bfloat16)
        core_rows = reshape(core_bf16, (1, 32, 128))
        wide = ttnn.typecast(core_rows, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        squares = ttnn.multiply(
            wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        sums = ttnn.sum(
            squares, dim=-1, keepdim=True, compute_kernel_config=self.compute, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        if cache_boundary:
            weighted, _ = self.decode_weighted_norm_tail(wide, sums)
            merged, _ = self.decode_local_silu(weighted, projection_z)
            return self.linear(
                merged,
                "linear_attn.out_proj.weight",
                ordered_out=cache_boundary,
                terminal_scan=decode and cache_boundary,
            )
        else:
            variance = ttnn.multiply(
                sums, 1 / 128, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            shifted = ttnn.add(variance, self.config.rms_norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG)
            inverse = ttnn.rsqrt(shifted, fast_and_approximate_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG)
            normalized_fp32 = ttnn.multiply(
                wide, inverse, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            normalized = ttnn.typecast(normalized_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            weighted = ttnn.multiply(
                normalized,
                self.w["linear_attn.norm.weight"],
                dtype=ttnn.float32,
                fast_and_approximate_mode=False,
                activations=self.round_widen_chain,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            gate_rows = reshape(ttnn.typecast(projection_z, ttnn.float32), (1, 32, 128))
            gated = ttnn.multiply(
                weighted,
                gate_rows,
                input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                fast_and_approximate_mode=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        head_shard = ttnn.create_sharded_memory_config(
            (32, 128),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        gated = ttnn.to_memory_config(ttnn.reshape(gated, (1, 1, 32, 128)), head_shard)
        merged = ttnn.experimental.nlp_concat_heads_decode(gated, num_heads=32)
        merged = ttnn.reshape(merged, (1, 1, 4096), (1, 32, 4096))
        return self.linear(
            merged, "linear_attn.out_proj.weight", ordered_out=cache_boundary, terminal_scan=decode and cache_boundary
        )

    def _moe(self, x, *, ordered_router=False):
        if self.kind == "full_attention":
            s = x.shape[1]
            logits = reshape(self.linear(x, "mlp.gate.weight", ttnn.bfloat16 if s == 1 else ttnn.float32), (s, 256))
            if s == 1:
                face = reshape(logits, (1, 16, 16))
                face = ttnn.to_memory_config(face, self.router_shard)
                gate_scores, gate_ids = ttnn.experimental.deepseek.moe.generalized_moe_gate(
                    face,
                    bias_tensor=self.router_bias,
                    input_indices_tensor=self.router_indices,
                    output_tensor=self.router_out,
                    output_indices_tensor=self.router_out_ids,
                    scaling_factor=1.0,
                    enable_sigmoid=False,
                    topk=8,
                    output_softmax=True,
                )
                scores = reshape(ttnn.slice(gate_scores, (0, 0, 0), (1, 1, 8), memory_config=DRAM), (1, 8))
                indices = ttnn.reshape(ttnn.slice(gate_ids, (0, 0, 0), (1, 1, 8), memory_config=DRAM), (1, 8))
            else:
                values, indices = ttnn.topk(logits, k=8, dim=-1, sorted=True)
                scores = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
            if s == 1:
                xx = reshape(x, (1, 1, 1, 2048))
                ids = ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, 8))

                def sparse(a, w, sparse_a=False):
                    n = w.shape[-1]
                    grid = (8, 4) if n == 1024 else (8, 8)
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                        in0_block_w=8,
                        out_subblock_h=1,
                        out_subblock_w=1,
                        out_block_h=1,
                        out_block_w=1,
                        per_core_M=1,
                        per_core_N=1,
                        fuse_batch=False,
                        fused_activation=None,
                        mcast_in0=True,
                    )
                    return ttnn.sparse_matmul(
                        a,
                        w,
                        sparsity=self.indexed_sparsity,
                        indices=ids,
                        is_input_a_sparse=sparse_a,
                        is_input_b_sparse=True,
                        program_config=pc,
                        compute_kernel_config=self.compute,
                        memory_config=DRAM,
                    )

                gate_up = ttnn.reshape(sparse(xx, self.egu), (1, 8, 1, 1024))
                gate, up = (gate_up[:, :, :, :512], gate_up[:, :, :, 512:])
                down = sparse(
                    ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), self.ed, True
                )
                down = ttnn.reshape(down, (1, 8, 1, 2048))
                routing = reshape(scores, (1, 8, 1, 1))
                weighted = ttnn.multiply(down, routing)
                total = ttnn.experimental.fast_reduce_nc(
                    weighted, dims=[1], memory_config=DRAM, compute_kernel_config=self.compute
                )
                routed = ttnn.reshape(total, (1, 1, 2048), (1, 32, 2048))
            else:
                group_memory = ttnn.L1_MEMORY_CONFIG if s <= 256 else DRAM
                x_rows = ttnn.to_layout(reshape(x, (1, 1, s, 2048)), ttnn.ROW_MAJOR_LAYOUT)
                indices32 = reshape(ttnn.typecast(indices, ttnn.uint32), (1, 1, s, 8))
                weights = reshape(scores, (1, 1, s, 8))
                routed = None
                for bank in range(4):
                    valid = ttnn.logical_and(ttnn.ge(indices32, bank * 64), ttnn.lt(indices32, (bank + 1) * 64))
                    local_ids = ttnn.subtract(indices32, bank * 64)
                    local_weights = ttnn.where(valid, weights, 0.0)
                    (
                        gather,
                        combine,
                        combine_weights,
                        counts,
                    ) = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
                        local_ids, local_weights, num_experts=64, token_start=0, num_tokens=s
                    )
                    dispatched = reshape(
                        ttnn.embedding(gather, x_rows, layout=ttnn.TILE_LAYOUT, memory_config=group_memory),
                        (1, 1, gather.shape[-1], 2048),
                    )
                    gu = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                        dispatched,
                        self.grouped_gu[bank],
                        counts,
                        dtype=ttnn.bfloat16,
                        memory_config=group_memory,
                        compute_kernel_config=self.compute,
                    )
                    del dispatched
                    activated = ttnn.multiply(
                        gu[:, :, :, :512], gu[:, :, :, 512:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
                    )
                    del gu
                    down = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                        activated,
                        self.grouped_down[bank],
                        counts,
                        dtype=ttnn.bfloat16,
                        output_layout=ttnn.ROW_MAJOR_LAYOUT if self.kind == "full_attention" else ttnn.TILE_LAYOUT,
                        memory_config=group_memory,
                        compute_kernel_config=self.compute,
                    )
                    del activated
                    part = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                        down[:, :, :, :2048]
                        if self.kind == "full_attention"
                        else ttnn.to_layout(down, ttnn.ROW_MAJOR_LAYOUT),
                        combine,
                        combine_weights,
                        self.combine_bias,
                        num_tokens=s,
                        memory_config=DRAM,
                        compute_kernel_config=self.compute,
                    )
                    del down
                    routed = part if routed is None else ttnn.add(routed, part)
                routed = reshape(routed, (1, s, 2048))
            shared_all = self.linear(x, "shared_packed")
            shared = ttnn.multiply(
                shared_all[:, :, :512], shared_all[:, :, 512:1024], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
            shared = self.linear(shared, "mlp.shared_expert.down_proj.weight")
            shared = ttnn.multiply(
                shared, shared_all[:, :, 1024:1025], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
            )
            return ttnn.add(routed, shared)
        s = x.shape[1]
        logits = reshape(
            self.linear(
                x,
                "mlp.gate.weight",
                ttnn.bfloat16 if s == 1 else ttnn.float32,
                ordered_router=ordered_router,
                terminal_scan=ordered_router,
            ),
            (s, 256),
        )
        if s == 1:
            routing, ids = self.inline_selector(logits)
            scores = reshape(routing, (1, 8))
            indices = ttnn.reshape(ids, (1, 8))
        else:
            values, indices = ttnn.topk(logits, k=8, dim=-1, sorted=True)
            scores = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        if s == 1:
            xx = reshape(x, (1, 1, 1, 2048))
            ids = ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, 8))

            def sparse(a, w, sparse_a=False):
                n = w.shape[-1]
                grid = (8, 4) if n == 1024 else (8, 8)
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    out_block_h=1,
                    out_block_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    fuse_batch=False,
                    fused_activation=None,
                    mcast_in0=True,
                )
                return ttnn.sparse_matmul(
                    a,
                    w,
                    sparsity=self.indexed_sparsity,
                    indices=ids,
                    is_input_a_sparse=sparse_a,
                    is_input_b_sparse=True,
                    program_config=pc,
                    compute_kernel_config=self.compute,
                    memory_config=DRAM,
                )

            gate_up = ttnn.reshape(sparse(xx, self.egu), (1, 8, 1, 1024))
            gate, up = (gate_up[:, :, :, :512], gate_up[:, :, :, 512:])
            down = sparse(ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), self.ed, True)
            down = ttnn.reshape(down, (1, 8, 1, 2048))
            routing = reshape(scores, (1, 8, 1, 1))
            weighted = ttnn.multiply(down, routing)
            total = ttnn.experimental.fast_reduce_nc(
                weighted, dims=[1], memory_config=DRAM, compute_kernel_config=self.compute
            )
            routed = ttnn.reshape(total, (1, 1, 2048), (1, 32, 2048))
        else:
            group_memory = ttnn.L1_MEMORY_CONFIG if s <= 256 else DRAM
            x_rows = ttnn.to_layout(reshape(x, (1, 1, s, 2048)), ttnn.ROW_MAJOR_LAYOUT)
            indices32 = reshape(ttnn.typecast(indices, ttnn.uint32), (1, 1, s, 8))
            weights = reshape(scores, (1, 1, s, 8))
            routed = None
            for bank in range(4):
                valid = ttnn.logical_and(ttnn.ge(indices32, bank * 64), ttnn.lt(indices32, (bank + 1) * 64))
                local_ids = ttnn.subtract(indices32, bank * 64)
                local_weights = ttnn.where(valid, weights, 0.0)
                gather, combine, combine_weights, counts = ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata(
                    local_ids, local_weights, num_experts=64, token_start=0, num_tokens=s
                )
                dispatched = reshape(
                    ttnn.embedding(gather, x_rows, layout=ttnn.TILE_LAYOUT, memory_config=group_memory),
                    (1, 1, gather.shape[-1], 2048),
                )
                gu = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                    dispatched,
                    self.grouped_gu[bank],
                    counts,
                    dtype=ttnn.bfloat16,
                    memory_config=group_memory,
                    compute_kernel_config=self.compute,
                )
                del dispatched
                activated = ttnn.multiply(
                    gu[:, :, :, :512], gu[:, :, :, 512:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
                )
                del gu
                down = ttnn.experimental.nomic_embed_text_v2_moe.grouped_matmul(
                    activated,
                    self.grouped_down[bank],
                    counts,
                    dtype=ttnn.bfloat16,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT if self.kind == "full_attention" else ttnn.TILE_LAYOUT,
                    memory_config=group_memory,
                    compute_kernel_config=self.compute,
                )
                del activated
                part = ttnn.experimental.nomic_embed_text_v2_moe.routed_combine(
                    down[:, :, :, :2048]
                    if self.kind == "full_attention"
                    else ttnn.to_layout(down, ttnn.ROW_MAJOR_LAYOUT),
                    combine,
                    combine_weights,
                    self.combine_bias,
                    num_tokens=s,
                    memory_config=DRAM,
                    compute_kernel_config=self.compute,
                )
                del down
                routed = part if routed is None else ttnn.add(routed, part)
            routed = reshape(routed, (1, s, 2048))
        shared_all = self.linear(x, "shared_packed")
        shared = ttnn.multiply(
            shared_all[:, :, :512], shared_all[:, :, 512:1024], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
        )
        shared = self.linear(shared, "mlp.shared_expert.down_proj.weight")
        shared = ttnn.multiply(
            shared, shared_all[:, :, 1024:1025], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]
        )
        return ttnn.add(routed, shared)

    def new_kv_state(self, *, context_length=262144, page_order=None, num_physical_pages=None, kv_cache=None):
        """Allocate finite paged storage. Page size is 32; order names owned pages.

        State allocation and page-table upload are setup, before any live trace.
        kv_cache can share a physical pool across rows with disjoint owned pages.
        The table is padded to eight entries by aliasing its final owned page.
        These entries are read-only causal padding, never writable logical pages.
        """
        import torch

        if not 1 <= context_length <= self.config.max_position_embeddings:
            raise ValueError("context_length is outside the HF contract")
        blocks = (context_length + 31) // 32
        padded_blocks = (blocks + 7) // 8 * 8
        pages = torch.arange(blocks, dtype=torch.int32) if page_order is None else page_order.to(torch.int32)
        physical = padded_blocks if num_physical_pages is None else num_physical_pages
        if physical < padded_blocks:
            raise ValueError(
                "physical pool must contain at least round_up(logical_pages, 8) pages for TTNN table validation"
            )
        if pages.shape != (blocks,) or len(pages.unique()) != blocks or pages.min() < 0 or (pages.max() >= physical):
            raise ValueError("page_order must name distinct owned pages within the physical pool")
        pages = torch.cat((pages, pages[-1:].expand(padded_blocks - blocks)))
        table = ttnn.from_torch(pages.reshape(1, -1), dtype=ttnn.int32, device=self.mesh_device)
        shape = (physical, 2, 32, 256)
        if kv_cache is None:
            kv_cache = tuple(
                (
                    ttnn.zeros(shape, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                    for _ in range(2)
                )
            )
        if len(kv_cache) != 2 or any(
            (tuple(t.shape) != shape or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT for t in kv_cache)
        ):
            raise ValueError("shared KV pool must contain two BF16 tiled tensors with the declared geometry")
        return {"k": kv_cache[0], "v": kv_cache[1], "page_table": table, "context_length": context_length}

    def prepare_positions(self, *, start_pos, seq_len):
        """Host setup boundary: absolute positions and text-only partial RoPE.

        Caller refreshes these same device tensors before trace replay. For text,
        the three HF mRoPE axes have identical positions.
        """
        import torch

        if not 0 <= start_pos < start_pos + seq_len <= self.config.max_position_embeddings:
            raise ValueError("positions are outside the HF contract")
        inv = 1.0 / self.config.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64)
        angles = torch.arange(start_pos, start_pos + seq_len).float()[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), dim=-1).reshape(1, seq_len, 1, 64)

        def upload(x, dtype, layout):
            return ttnn.from_torch(x, device=self.mesh_device, dtype=dtype, layout=layout)

        return {
            "cos": upload(angles.cos(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "sin": upload(angles.sin(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "current_pos": upload(torch.tensor([start_pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        }

    def _full_attention(self, x, state, positions, *, decode, start_pos):
        n = x.shape[1]
        projected = self.linear(x, "attention_packed")
        packed_qkv = projected[:, :, :5120]
        gate = reshape(projected[:, :, 5120:], (1, n, 16, 256))
        shard = ttnn.create_sharded_memory_config(
            (32, 256),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if decode:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.reshape(packed_qkv, (1, 1, 1, 5120)), num_heads=16, num_kv_heads=2, memory_config=shard
            )
            norm_shard = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard.shard_spec)
            q, k = [ttnn.to_memory_config(t, norm_shard) for t in (q, k)]
        else:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                packed_qkv, num_heads=16, num_kv_heads=2, transpose_key=False
            )
            gate = ttnn.permute(gate, (0, 2, 1, 3))
        q, k = (self.norm(q, "self_attn.q_norm.weight"), self.norm(k, "self_attn.k_norm.weight"))
        if decode:
            q, k = [ttnn.to_memory_config(t, shard) for t in (q, k)]
        cos = ttnn.permute(positions["cos"], (0, 2, 1, 3))
        sin = ttnn.permute(positions["sin"], (0, 2, 1, 3))

        def rope(a):
            r = ttnn.experimental.rotary_embedding(a[:, :, :, :64], cos, sin, token_index=0 if decode else None)
            r = r[:, : a.shape[1], : a.shape[2], :]
            return ttnn.concat([r, a[:, :, :, 64:]], dim=-1)

        q, k = (rope(q), rope(k))
        if decode:
            k, v = [ttnn.to_memory_config(a, shard) for a in (k, v)]
            for dest, src in ((state["k"], k), (state["v"], v)):
                ttnn.experimental.paged_update_cache(
                    dest, src, update_idxs_tensor=positions["current_pos"], page_table=state["page_table"]
                )
            q = ttnn.to_memory_config(q, shard)
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                state["k"],
                state["v"],
                cur_pos_tensor=positions["current_pos"],
                page_table_tensor=state["page_table"],
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256
                ),
                compute_kernel_config=self.compute,
                memory_config=DRAM,
            )
        else:
            pages = state["page_table"][:, start_pos // 32 :]
            for dest, src in ((state["k"], k), (state["v"], v)):
                ttnn.experimental.paged_fill_cache(dest, src, pages, batch_idx=0)
            o = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                state["k"],
                state["v"],
                state["page_table"],
                chunk_start_idx=start_pos,
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256
                ),
                compute_kernel_config=self.compute,
            )
        if decode:
            o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            o = ttnn.to_memory_config(o, shard)
            merged = ttnn.experimental.nlp_concat_heads_decode(o, num_heads=16)
            merged = ttnn.reshape(merged, (1, 1, 4096), (1, 32, 4096))
        else:
            o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            merged = ttnn.transformer.concatenate_heads(o)
        return self.linear(merged, "self_attn.o_proj.weight")

    def _forward(
        self, hidden_states, *, state, decode, positions=None, start_pos=0, cache_boundary=True, projection_k=512
    ):
        if self.kind == "full_attention":
            norm_input = (
                ttnn.to_memory_config(hidden_states, self.norm_memory)
                if decode and self.kind == "full_attention"
                else hidden_states
            )
            x = self.norm(norm_input, "input_layernorm.weight")
            if self.kind == "linear_attention":
                x = self._linear_attention(
                    x, state, decode=decode, cache_boundary=cache_boundary, projection_k=projection_k
                )
            else:
                x = self._full_attention(x, state, positions, decode=decode, start_pos=start_pos)
            x = ttnn.add(hidden_states, x)
            mlp_input = ttnn.to_memory_config(x, self.norm_memory) if decode else x
            mlp_input = self.norm(mlp_input, "post_attention_layernorm.weight")
            if decode:
                mlp_input = ttnn.to_memory_config(mlp_input, DRAM)
            return ttnn.add(x, self._moe(mlp_input))
        norm_input = (
            ttnn.to_memory_config(hidden_states, self.norm_memory)
            if decode and self.kind == "full_attention"
            else hidden_states
        )
        x = self.norm(norm_input, "input_layernorm.weight")
        if self.kind == "linear_attention":
            x = self._linear_attention(
                x, state, decode=decode, cache_boundary=cache_boundary, projection_k=projection_k
            )
        else:
            x = self._full_attention(x, state, positions, decode=decode, start_pos=start_pos)
        x = ttnn.add(hidden_states, x, fast_and_approximate_mode=False)
        mlp_input = x
        mlp_input = self.norm(mlp_input, "post_attention_layernorm.weight")
        return ttnn.add(x, self._moe(mlp_input, ordered_router=decode and cache_boundary))

    def prepare_prefill(self, *, seq_len, start_pos=0, chunk_size=4096):
        """Prepare a host plan outside measured execution. Linear prefill uses chunks64 relative to this public call, including a logical final tail. Full-attention retains its aligned interior and singleton boundaries. All positive logical lengths and valid starts remain supported."""
        if self.kind == "full_attention":
            "Setup a bounded host plan; boundary tokens use the exact decode kernel.\n\n        Arbitrary logical starts/lengths are accepted. Aligned interiors use chunks\n        of at most chunk_size. A tail never writes padded cache or recurrent state.\n        Prepare plans and warm all their kernels before recording a live trace.\n"
            if chunk_size < 32 or chunk_size % 32:
                raise ValueError("physical chunk size must be a positive multiple of 32")
            if not 0 <= start_pos < start_pos + seq_len <= self.config.max_position_embeddings:
                raise ValueError("logical sequence is outside HF context")
            plan, offset = ([], 0)
            while offset < seq_len:
                absolute = start_pos + offset
                remaining = seq_len - offset
                alignment = 256 if self.kind == "full_attention" else 32
                physical_chunk = min(chunk_size, 256) if self.kind == "full_attention" else chunk_size
                n = min(physical_chunk, remaining // 32 * 32) if absolute % alignment == 0 and remaining >= 32 else 1
                positions = (
                    self.prepare_positions(start_pos=absolute, seq_len=n) if self.kind == "full_attention" else None
                )
                plan.append((offset, n, absolute, positions))
                offset += n
            return plan
        if self.kind == "linear_attention":
            if not 0 <= start_pos < start_pos + seq_len <= self.config.max_position_embeddings:
                raise ValueError("logical sequence is outside HF context")
            if chunk_size <= 0:
                raise ValueError("physical chunk bound must be positive")
            return [(offset, min(64, seq_len - offset), start_pos + offset, None) for offset in range(0, seq_len, 64)]
        if chunk_size < 32 or chunk_size % 32:
            raise ValueError("physical chunk size must be a positive multiple of 32")
        if not 0 <= start_pos < start_pos + seq_len <= self.config.max_position_embeddings:
            raise ValueError("logical sequence is outside HF context")
        plan, offset = ([], 0)
        while offset < seq_len:
            absolute = start_pos + offset
            remaining = seq_len - offset
            alignment = 256 if self.kind == "full_attention" else 32
            physical_chunk = min(chunk_size, 256) if self.kind == "full_attention" else chunk_size
            n = min(physical_chunk, remaining // 32 * 32) if absolute % alignment == 0 and remaining >= 32 else 1
            positions = self.prepare_positions(start_pos=absolute, seq_len=n) if self.kind == "full_attention" else None
            plan.append((offset, n, absolute, positions))
            offset += n
        return plan

    def prefill_forward(self, hidden_states, *, state, plan):
        """Run one row [1,S,2048], preserving all logical outputs and prior state.

        Plan is produced by prepare_prefill outside the measured pass.
        For heterogeneous rows use prefill_batch.
        """
        if self.kind == "full_attention":
            "Run one row [1,S,2048], preserving all logical outputs and prior state.\n\n        Plan is produced by prepare_prefill outside the measured pass.\n        For heterogeneous rows use prefill_batch.\n"
            if plan[-1][0] + plan[-1][1] != hidden_states.shape[1]:
                raise ValueError("plan length does not match input")
            if self.kind == "full_attention" and plan[-1][2] + plan[-1][1] > state["context_length"]:
                raise ValueError("prefill exceeds the allocated context")
            if self.kind == "linear_attention":
                initial = ttnn.typecast(ttnn.typecast(state["recurrent"], ttnn.bfloat16), ttnn.float32)
                ttnn.copy(initial, state["recurrent"])
            outputs = []
            for offset, n, absolute, positions in plan:
                outputs.append(
                    self._forward(
                        hidden_states[:, offset : offset + n, :],
                        state=state,
                        decode=n == 1,
                        positions=positions,
                        start_pos=absolute,
                        cache_boundary=False,
                        projection_k=1024 if hidden_states.shape[1] > 1 else 512,
                    )
                )
            return ttnn.concat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        if plan[-1][0] + plan[-1][1] != hidden_states.shape[1]:
            raise ValueError("plan length does not match input")
        if self.kind == "full_attention" and plan[-1][2] + plan[-1][1] > state["context_length"]:
            raise ValueError("prefill exceeds the allocated context")
        if self.kind == "linear_attention":
            initial = ttnn.typecast(ttnn.typecast(state["recurrent"], ttnn.bfloat16), ttnn.float32)
            ttnn.copy(initial, state["recurrent"])
        outputs = []
        for offset, n, absolute, positions in plan:
            outputs.append(
                self._forward(
                    hidden_states[:, offset : offset + n, :],
                    state=state,
                    decode=n == 1 and (self.kind == "full_attention" or hidden_states.shape[1] == 1),
                    positions=positions,
                    start_pos=absolute,
                    cache_boundary=False,
                    projection_k=1024 if hidden_states.shape[1] > 1 else 512,
                )
            )
        return ttnn.concat(outputs, dim=1) if len(outputs) > 1 else outputs[0]

    def prefill_batch(self, hidden_states, *, states, plans):
        """Heterogeneous rows; row-owned state and plans, shared layer weights."""
        if not len(hidden_states) == len(states) == len(plans):
            raise ValueError("row counts differ")
        return tuple((self.prefill_forward(x, state=s, plan=p) for x, s, p in zip(hidden_states, states, plans)))

    def decode_forward(self, hidden_states, *, state, positions=None):
        """Single-row decode [1,1,2048]; tensor current_pos and stable state.

        Every operation is on device and this entire call can be captured.
        """
        if tuple(hidden_states.shape) != (1, 1, 2048):
            raise ValueError("decode row must have shape [1, 1, 2048]; use decode_batch for several rows")
        return self._forward(hidden_states, state=state, decode=True, positions=positions)

    def decode_batch(self, hidden_states, *, states, positions):
        """Row-independent device decode; the complete batch loop is traceable."""
        if not len(hidden_states) == len(states) == len(positions):
            raise ValueError("row counts differ")
        return tuple(
            (self.decode_forward(x, state=s, positions=p) for x, s, p in zip(hidden_states, states, positions))
        )


PORTABLE_CPP = "\ninline bool v6_greater(uint32_t a, uint32_t b) {\n    uint32_t x = a >> 16, y = b >> 16;\n    bool xn = (x & 0x7fff) > 0x7f80, yn = (y & 0x7fff) > 0x7f80;\n    if (xn || yn) return xn && !yn;\n    if ((x & 0x7fff) == 0) x = 0;\n    if ((y & 0x7fff) == 0) y = 0;\n    uint32_t kx = (x & 0x8000) ? ((~x) & 0xffff) : (x ^ 0x8000);\n    uint32_t ky = (y & 0x8000) ? ((~y) & 0xffff) : (y ^ 0x8000);\n    return kx > ky;\n}\ninline void v6_swap(uint32_t* q, int a, int b) {\n    uint32_t value=q[a]; q[a]=q[b]; q[b]=value;\n}\ninline void v6_insertion(uint32_t* q, int first, int last) {\n    for (int pos=first+1; pos<last; ++pos) {\n        uint32_t value=q[pos]; int hole=pos;\n        while (hole>first && v6_greater(value,q[hole-1])) {\n            q[hole]=q[hole-1]; --hole;\n        }\n        q[hole]=value;\n    }\n}\ninline void v6_push_heap(uint32_t* q, int first, int hole, int top, uint32_t value) {\n    while (hole>top) {\n        int parent=(hole-1)/2;\n        if (!v6_greater(q[first+parent],value)) break;\n        q[first+hole]=q[first+parent]; hole=parent;\n    }\n    q[first+hole]=value;\n}\ninline void v6_adjust_heap(uint32_t* q, int first, int hole, int length, uint32_t value) {\n    int top=hole, second_child=hole;\n    while (second_child<(length-1)/2) {\n        second_child=2*(second_child+1);\n        if (v6_greater(q[first+second_child],q[first+second_child-1])) --second_child;\n        q[first+hole]=q[first+second_child]; hole=second_child;\n    }\n    if ((length&1)==0 && second_child==(length-2)/2) {\n        second_child=2*(second_child+1);\n        q[first+hole]=q[first+second_child-1]; hole=second_child-1;\n    }\n    v6_push_heap(q,first,hole,top,value);\n}\ninline void v6_heap_select(uint32_t* q, int first, int middle, int last) {\n    int length=middle-first;\n    if (length>=2) {\n        for (int parent=(length-2)/2; parent>=0; --parent)\n            v6_adjust_heap(q,first,parent,length,q[first+parent]);\n    }\n    for (int pos=middle; pos<last; ++pos) {\n        if (v6_greater(q[pos],q[first])) {\n            uint32_t value=q[pos]; q[pos]=q[first];\n            v6_adjust_heap(q,first,0,length,value);\n        }\n    }\n}\ninline void v6_select_top8(uint32_t* q, int depth=16) {\n    int first=0,last=256,nth=7;\n    while (last-first>3) {\n        if (depth==0) {\n            v6_heap_select(q,first,nth+1,last); v6_swap(q,first,nth);\n            v6_insertion(q,0,7); return;\n        }\n        --depth;\n        int a=first+1,b=first+(last-first)/2,c=last-1,median;\n        if (v6_greater(q[a],q[b])) {\n            if (v6_greater(q[b],q[c])) median=b;\n            else if (v6_greater(q[a],q[c])) median=c;\n            else median=a;\n        } else if (v6_greater(q[a],q[c])) median=a;\n        else if (v6_greater(q[b],q[c])) median=c;\n        else median=b;\n        v6_swap(q,first,median);\n        int left=first+1,right=last; uint32_t pivot=q[first];\n        while (true) {\n            while (v6_greater(q[left],pivot)) ++left;\n            --right;\n            while (v6_greater(pivot,q[right])) --right;\n            if (left>=right) break;\n            v6_swap(q,left,right); ++left;\n        }\n        if (left<=nth) first=left;\n        else last=left;\n    }\n    v6_insertion(q,first,last); v6_insertion(q,0,7);\n}\n"


class InlineSelector:
    """Scratch and outputs are setup-owned; scalar work executes on BRISC."""

    def __init__(self, ttnn, mesh, selector_cpp):
        import torch

        self.ttnn = ttnn
        self.mesh = mesh
        core = ttnn.CoreCoord(0, 0)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        self.shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        ids_shard = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, (1, 16), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.values = ttnn.from_torch(
            torch.zeros(1, 8), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=self.shard
        )
        self.ids = ttnn.from_torch(
            torch.zeros(1, 8, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ids_shard,
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.kernel_source = (
            '#include "api/dataflow/dataflow_api.h"\n'
            + selector_cpp
            + "\nvoid kernel_main() {\n    auto* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_arg_val<uint32_t>(0));\n    auto* values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_arg_val<uint32_t>(1));\n    auto* indices = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_arg_val<uint32_t>(2));\n    cb_reserve_back(0, 1);\n    auto* scratch = reinterpret_cast<uint32_t*>(get_write_ptr(0));\n    for (uint32_t i = 0; i < 256; ++i) scratch[i] = (uint32_t(src[i]) << 16) | i;\n    v6_select_top8(scratch);\n    for (uint32_t i = 0; i < 1024; ++i) values[i] = 0;\n    for (uint32_t i = 0; i < 16; ++i) indices[i] = 0;\n    for (uint32_t i = 0; i < 8; ++i) {\n        values[i] = scratch[i] >> 16;\n        indices[i] = scratch[i] & 65535;\n    }\n    cb_push_back(0, 1);\n    cb_wait_front(0, 1);\n    cb_pop_front(0, 1);\n}\n"
        )
        runtime = ttnn.RuntimeArgs()
        runtime[0][0] = [0, self.values.buffer_address(), self.ids.buffer_address()]
        self.kernel = ttnn.KernelDescriptor(
            kernel_source=self.kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self.cores,
            runtime_args=runtime,
            config=ttnn.WriterConfigDescriptor(),
        )
        self.program = ttnn.ProgramDescriptor(
            kernels=[self.kernel],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=4096,
                    core_ranges=self.cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, 4096)],
                )
            ],
        )

    def __call__(self, logits):
        ttnn = self.ttnn
        face = ttnn.reshape(logits, (1, 16, 16))
        face = ttnn.to_memory_config(face, self.shard)
        self.program.kernels[0].runtime_args[0][0] = [
            face.buffer_address(),
            self.values.buffer_address(),
            self.ids.buffer_address(),
        ]
        ttnn.generic_op([face, self.values, self.ids], self.program)
        values = ttnn.to_memory_config(self.values, ttnn.DRAM_MEMORY_CONFIG)
        values = ttnn.typecast(values, ttnn.float32)
        scores = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        scores = ttnn.typecast(scores, ttnn.bfloat16)
        routing = ttnn.reshape(scores, (1, 8, 1, 1))
        ids = ttnn.reshape(ttnn.to_memory_config(self.ids, ttnn.DRAM_MEMORY_CONFIG), (1, 1, 1, 8))
        return (routing, ids)


class Chunk64StateGraph:
    sum_order = "canonical"
    "Device-only state graph after setup, optionally instrumented by observer.\n\n    Native mode uses HiFi4 FP32-destination matmul: full FP32 operand precision\n    is NOT claimed. sfpu_reduce mode forms FP32 products explicitly and reduces\n    them; reduction order still need not equal Torch CPU matrix multiplication.\n    sfpu_fma is a sequential addcmul control. Despite its SFPMAD instruction,\n    the native full-mantissa tests document that it is not IEEE single-rounding\n    FP32 FMA. Its CLI name identifies the proposed candidate, not an exactness claim.\n    "

    def __init__(self, ttnn, mesh, upload, variant="native", exp_approx=False, cumsum="plain"):
        import torch

        self.t = ttnn
        self.mesh = mesh
        self.variant = variant
        self.exp_approx = exp_approx
        assert cumsum in ("plain", "compensated")
        self.cumsum = cumsum
        self.mem = ttnn.DRAM_MEMORY_CONFIG
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.lower = upload(torch.ones(1, 1, 64, 64).tril())
        self.strict = upload(torch.ones(1, 1, 64, 64).tril(-1))
        self.eye = upload(torch.eye(64).reshape(1, 1, 64, 64))
        self.mkl_negative_exp = Chunk64MKLNegativeExp(ttnn, mesh, upload, self.mem)
        self.corrected_mm = FusedCorrectedMatmul(ttnn, mesh, max_cores=64)
        self.triangular_recurrence = Chunk64TriangularRecurrence(ttnn, mesh)
        self.transpose_mm = TransposeCorrectedMatmul(ttnn, mesh)

    def add(self, a, b):
        return self.t.add(a, b, dtype=self.t.float32, memory_config=self.mem)

    def sub(self, a, b):
        return self.t.subtract(a, b, dtype=self.t.float32, memory_config=self.mem)

    def mul(self, a, b):
        return self.t.multiply(a, b, dtype=self.t.float32, fast_and_approximate_mode=False, memory_config=self.mem)

    def exp(self, a):
        return self.mkl_negative_exp(a)

    def mm(self, a, b):
        return self.corrected_mm(a, b)

    def __call__(self, key, value, g, beta, initial, observer=None):
        """Support every positive logical S, padding only its final chunk."""
        t = self.t
        n = key.shape[-2]
        assert n > 0 and all((x.shape[-2] == n for x in (value, g, beta)))
        pad = -n % 64
        if pad:
            key, value, g, beta = [
                t.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0, memory_config=self.mem) for x in (key, value, g, beta)
            ]
        state = initial
        for start in range(0, n, 64):
            prefix = f"chunk{start // 64}/"
            k, v, gg, bb = [x[..., start : start + 64, :] for x in (key, value, g, beta)]
            kb, vb = (self.mul(k, bb), self.mul(v, bb))
            cumulative = ttnn_double_single_chunk_cumsum(t, gg, mesh_device=self.mesh, memory_config=self.mem)
            differences = self.sub(cumulative, t.transpose(cumulative, -1, -2))
            masked_diff = t.where(self.lower, differences, 0.0, memory_config=self.mem)
            decay = self.mul(self.exp(masked_diff), self.lower)
            a = t.where(
                self.strict, self.mul(self.transpose_mm(kb, k, transpose_b=True), decay), 0.0, memory_config=self.mem
            )
            a = self.mul(a, -1.0)
            if observer:
                for name, tensor in (
                    ("k_beta", kb),
                    ("v_beta", vb),
                    ("cumulative_g", cumulative),
                    ("decay_mask", decay),
                    ("A_initial", a),
                ):
                    observer(prefix + name, tensor)
            a = self.triangular_recurrence(a)
            if observer:
                observer(prefix + "A_row63", a)
            a = self.add(a, self.eye)
            transformed_v = self.mm(a, vb)
            transformed_k = self.mm(a, self.mul(kb, self.exp(cumulative)))
            v_new = self.sub(transformed_v, self.mm(transformed_k, state))
            last = cumulative[..., 63:64, :]
            weighted_k = self.mul(k, self.exp(self.sub(last, cumulative)))
            state = self.add(self.mul(state, self.exp(last)), self.transpose_mm(weighted_k, v_new, transpose_a=True))
            if observer:
                for name, tensor in (
                    ("A_final", a),
                    ("transformed_v", transformed_v),
                    ("transformed_k", transformed_k),
                    ("v_new", v_new),
                    ("weighted_k", weighted_k),
                    ("state", state),
                ):
                    observer(prefix + name, tensor)
        return state

    def triangular_sum(self, products):
        """Sum [..., i, i] to [..., 1, i], matching proven CPU add scheduling."""
        t = self.t
        shape = tuple(products.shape)
        rows, columns = shape[-2:]
        assert len(shape) == 4 and rows == columns and (1 <= rows <= 63)
        if self.sum_order == "native":
            return t.sum(products, dim=-2, keepdim=True, compute_kernel_config=self.compute, memory_config=self.mem)
        assert self.sum_order == "canonical"

        def zeros(width):
            return t.zeros(
                (shape[0], shape[1], 1, width),
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                device=self.mesh,
                memory_config=self.mem,
            )

        def row(index, width):
            return t.slice(products, (0, 0, index, 0), (shape[0], shape[1], index + 1, width), memory_config=self.mem)

        partials = [zeros(columns) for _ in range(4)]
        end = rows // 4 * 4
        for index in range(end):
            slot = index % 4
            partials[slot] = self.add(partials[slot], row(index, columns))
        for index in range(end, rows):
            partials[0] = self.add(partials[0], row(index, columns))
        for index in range(1, 4):
            partials[0] = self.add(partials[0], partials[index])
        result = self.add(zeros(columns), partials[0])
        cascade_columns = columns // 32 * 32 if columns >= 8 else columns // 4 * 4
        if cascade_columns:
            completed = zeros(cascade_columns)
            end = rows // 16 * 16
            for start in range(0, end, 16):
                partial = zeros(cascade_columns)
                for index in range(start, start + 16):
                    partial = self.add(partial, row(index, cascade_columns))
                completed = self.add(completed, partial)
            tail = zeros(cascade_columns)
            for index in range(end, rows):
                tail = self.add(tail, row(index, cascade_columns))
            prefix = self.add(zeros(cascade_columns), self.add(tail, completed))
            if cascade_columns == columns:
                result = prefix
            else:
                suffix = t.slice(
                    result, (0, 0, 0, cascade_columns), (shape[0], shape[1], 1, columns), memory_config=self.mem
                )
                result = t.concat([prefix, suffix], dim=-1, memory_config=self.mem)
        return result


def ttnn_double_single_chunk_cumsum(t, values, *, mesh_device, memory_config):
    """Device-only candidate, restarting at each internal chunk64 boundary.

    No Torch/host conversion and no unsupported FP32 ADD/SUB flags. Returns the
    internally padded length so callers can inspect the final chunk's g_last.
    CPU finite-fixture parity is not a universal floating-point proof.
    """
    n = values.shape[-2]
    assert n > 0 and values.dtype == t.float32
    pad = -n % 64
    if pad:
        values = t.pad(values, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0, memory_config=memory_config)
    shape = (*tuple(values.shape)[:-2], 1, values.shape[-1])
    zero = t.zeros(shape, dtype=t.float32, layout=t.TILE_LAYOUT, device=mesh_device, memory_config=memory_config)
    output = []

    def add(a, b):
        return t.add(a, b, dtype=t.float32, memory_config=memory_config)

    def sub(a, b):
        return t.subtract(a, b, dtype=t.float32, memory_config=memory_config)

    for start in range(0, n, 64):
        hi = lo = zero
        for i in range(start, start + 64):
            x = values[..., i : i + 1, :]
            s = add(hi, x)
            b = sub(s, hi)
            e = add(sub(hi, sub(s, b)), sub(x, b))
            lo = add(lo, e)
            output.append(add(s, lo))
            hi = s
    return t.concat(output, dim=-2, memory_config=memory_config)


CHUNK64_EXP_SAFE_ABS = 87.33654022216797
ROW_SNAPSHOTS = (1, 2, 31, 63)


def chunk64_mathematical_exp_table():
    """Generic 2**(j/128) split into nearest FP32 high and residual low."""
    import struct
    from decimal import Decimal, localcontext

    import torch

    def f32(value):
        return struct.unpack("f", struct.pack("f", float(value)))[0]

    rows = []
    with localcontext() as context:
        context.prec = 100
        log2 = Decimal(2).ln()
        for index in range(128):
            exact = (log2 * Decimal(index) / Decimal(128)).exp()
            high = f32(exact)
            low = f32(exact - Decimal.from_float(high))
            rows.append((high, low))
    return torch.tensor(rows, dtype=torch.float32)


class Chunk64MKLNegativeExp:
    """TTNN-only runtime, with independent rounded multiply/add/sub operations.

    Tables are replicated at setup to avoid reshaping sub-tile logical indices
    or changing layouts inside the diagnostic path. Default table allocation is
    2 MiB total (two FP32 [1,32,64,128] tables), containing only 256 unique values.
    Input is FP32 [B,H,R,C], B<=1,H<=32,R<=64. Width C is not constrained by 128.
    No device exactness is claimed until this component has been measured.
    """

    def __init__(self, t, mesh, upload, memory_config, max_shape=(1, 32, 64)):
        import torch

        self.t, self.mesh, self.mem = (t, mesh, memory_config)
        self.max_shape = tuple(max_shape)
        table = chunk64_mathematical_exp_table()
        self.high = upload(table[:, 0].reshape(1, 1, 1, 128).expand(*self.max_shape, 128).contiguous())
        self.low = upload(table[:, 1].reshape(1, 1, 1, 128).expand(*self.max_shape, 128).contiguous())
        self.one_bits = t.from_torch(
            torch.tensor([[[[1065353216]]]], dtype=torch.int64),
            dtype=t.uint32,
            layout=t.TILE_LAYOUT,
            device=mesh,
            memory_config=memory_config,
        )
        self.exponent_bias = t.from_torch(
            torch.tensor([[[[16 << 23]]]], dtype=torch.int64),
            dtype=t.uint32,
            layout=t.TILE_LAYOUT,
            device=mesh,
            memory_config=memory_config,
        )

    def __call__(self, x, observer=None):
        t, mem = (self.t, self.mem)
        shape = tuple(x.shape)
        assert len(shape) == 4 and x.dtype == t.float32
        assert all((got <= maximum for got, maximum in zip(shape[:3], self.max_shape)))

        def add(a, b):
            return t.add(a, b, dtype=t.float32, memory_config=mem)

        def sub(a, b):
            return t.subtract(a, b, dtype=t.float32, memory_config=mem)

        def mul(a, b):
            return t.multiply(a, b, dtype=t.float32, fast_and_approximate_mode=False, memory_config=mem)

        safe = t.ge(x, -CHUNK64_EXP_SAFE_ABS, memory_config=mem)
        bounded = t.where(safe, x, -CHUNK64_EXP_SAFE_ABS, memory_config=mem)
        z = add(mul(bounded, 184.6649627685547), 12582912.0)
        raw = t.bitcast(z, t.uint32, memory_config=mem)
        index = t.bitwise_and(raw, 127, memory_config=mem)
        high = t.gather(self.high, dim=-1, index=index, memory_config=mem)
        low = t.gather(self.low, dim=-1, index=index, memory_config=mem)
        n = add(z, -12582912.0)
        r = sub(bounded, mul(n, 0.0054168701171875))
        r = add(r, mul(n, 1.6577690757912933e-06))
        polynomial = add(mul(mul(r, r), 0.5), r)
        reduced = add(add(mul(polynomial, high), low), high)
        shifted = t.bitwise_left_shift(raw, 16, memory_config=mem)
        exponent_bits = t.add(
            t.bitwise_and(
                t.add(shifted, self.one_bits, dtype=t.uint32, memory_config=mem), 2139095040, memory_config=mem
            ),
            self.exponent_bias,
            dtype=t.uint32,
            memory_config=mem,
        )
        exponent = t.bitcast(exponent_bits, t.float32, memory_config=mem)
        value = mul(mul(reduced, 1.52587890625e-05), exponent)
        result = t.where(safe, value, 0.0, memory_config=mem)
        if observer:
            for name, tensor in (
                ("bounded", bounded),
                ("z", z),
                ("raw", raw),
                ("index", index),
                ("high", high),
                ("low", low),
                ("n", n),
                ("r", r),
                ("polynomial", polynomial),
                ("reduced", reduced),
                ("exponent_bits", exponent_bits),
                ("exponent", exponent),
                ("result", result),
            ):
                observer(name, tensor)
        return result


class Chunk64OutputGraph(Chunk64StateGraph):
    """Normalized FP32 [B,H,S,D] inputs; returns BF16 output, state, FP32 output.

    Outputs use [B,S,H,Dv]; state remains FP32 [B,H,Dk,Dv]. The state method is
    the unchanged combined diagnostic. Internal observers retain device handles
    only; user observers are optional diagnostic instrumentation.
    """

    state_call = Chunk64StateGraph.__call__

    def __call__(self, query, key, value, g, beta, initial, observer=None):
        t = self.t
        n = key.shape[-2]
        assert n > 0 and tuple(query.shape) == tuple(key.shape)
        assert query.dtype == key.dtype == value.dtype == t.float32
        query = self.mul(query, 1 / query.shape[-1] ** 0.5)
        output_key = key
        pad = -n % 64
        if pad:
            query, output_key = [
                t.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0, memory_config=self.mem) for x in (query, output_key)
            ]
        outputs = []
        matrices = {}
        carry = {"state": initial}

        def collect(name, tensor):
            prefix, stage = name.rsplit("/", 1)
            if observer:
                observer(name, tensor)
            if stage in ("cumulative_g", "decay_mask", "v_new"):
                matrices[stage] = tensor
            if stage == "state":
                start = len(outputs) * 64
                q = query[..., start : start + 64, :]
                k = output_key[..., start : start + 64, :]
                attn = self.mul(self.transpose_mm(q, k, transpose_b=True), matrices["decay_mask"])
                inter = self.mm(self.mul(q, self.exp(matrices["cumulative_g"])), carry["state"])
                output = self.add(inter, self.mm(attn, matrices["v_new"]))
                outputs.append(output)
                if observer:
                    for label, item in (
                        ("query_scaled", q),
                        ("output_attn", attn),
                        ("attn_inter", inter),
                        ("chunk_output_fp32", output),
                    ):
                        observer(prefix + "/" + label, item)
                carry["state"] = tensor
                matrices.clear()

        state = self.state_call(key, value, g, beta, initial, observer=collect)
        padded_output = t.concat(outputs, dim=-2, memory_config=self.mem) if len(outputs) > 1 else outputs[0]
        shape = tuple(padded_output.shape)
        cropped = t.slice(padded_output, (0, 0, 0, 0), (shape[0], shape[1], n, shape[3]), memory_config=self.mem)
        output_fp32 = t.permute(cropped, (0, 2, 1, 3), memory_config=self.mem)
        output = t.typecast(output_fp32, t.bfloat16, memory_config=self.mem)
        return (output, state, output_fp32)


import math

WEIGHTED_NORM_COMPUTE_SOURCE = '\n#include "api/compute/common.h"\n#include "api/compute/compute_kernel_api.h"\n#include "api/compute/compute_kernel_hw_startup.h"\n#include "api/compute/experimental/reg_api.h"\n#include "api/compute/pack.h"\n#include "api/compute/tile_move_copy.h"\n#include "api/compute/eltwise_unary/rsqrt.h"\n#include "api/compute/sfpu_binary_bcast.h"\n#include "api/dataflow/circular_buffer.h"\n#ifdef TRISC_MATH\n#include "llk_math_eltwise_ternary_sfpu.h"\nsfpi_inline sfpi::vFloat round_bf16(sfpi::vFloat x) {\n sfpi::vUInt bits=sfpi::as<sfpi::vUInt>(x);\n sfpi::vUInt tie=(bits>>16)&1;\n bits=bits+0x7fff; bits=bits+tie; bits=bits&0xffff0000;\n return sfpi::as<sfpi::vFloat>(bits);\n}\n#endif\nvoid kernel_main() {\n constexpr bool debug=get_compile_time_arg_val(0)!=0;\n CircularBuffer core(0), sums(1), gamma(2), output(16);\n CircularBuffer norm_debug(17), weight_debug(18), inverse_debug(19);\n compute_kernel_hw_startup(0,16); copy_init(0);\n gamma.reserve_back(4); gamma.push_back(4);\n core.wait_front(4); sums.wait_front(1); gamma.wait_front(4);\n for(uint32_t tile=0;tile<4;++tile) {\n  tile_regs_acquire_math_clear();\n  copy_tile(0,tile,0); copy_tile(1,0,1); copy_tile(2,tile,2);\n#ifdef TRISC_MATH\n  sfpu::_init_sfpu_config_reg();\n  addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);\n  math::reset_counters(p_setrwc::SET_ABD_F);\n  _llk_math_eltwise_sfpu_start_(1);\n  for(uint32_t lane=0;lane<32;++lane) {\n   // Explicit DST store/load separates multiply and add, preserving rounding.\n   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[0]=x*0x1p-7f; }\n   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[0]=x+0x1.0c6f7a0b5ed8dp-20f; }\n   sfpi::dst_reg++;\n  }\n  _llk_math_eltwise_sfpu_done_();\n#endif\n  rsqrt_tile_init(); rsqrt_tile<RsqrtMode::Default,true>(1);\n  sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col<true>(0,1);\n#ifdef TRISC_MATH\n  math::reset_counters(p_setrwc::SET_ABD_F);\n  _llk_math_eltwise_sfpu_start_(0);\n  for(uint32_t lane=0;lane<32;++lane) {\n   { sfpi::vFloat x=sfpi::dst_reg[0]; if constexpr(debug) sfpi::dst_reg[128]=x; sfpi::dst_reg[0]=round_bf16(x); }\n   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::vFloat w=sfpi::dst_reg[64]; sfpi::dst_reg[0]=x*w; }\n   { sfpi::vFloat x=sfpi::dst_reg[0]; x=round_bf16(x); sfpi::dst_reg[0]=x; if constexpr(debug) sfpi::dst_reg[160]=x; }\n   sfpi::dst_reg++;\n  }\n  _llk_math_eltwise_sfpu_done_();\n#endif\n  tile_regs_commit(); tile_regs_wait();\n  pack_reconfig_data_format(16);\n  output.reserve_back(1); pack_tile(0,16); output.push_back(1);\n  if constexpr(debug) {\n   pack_reconfig_data_format(17);\n   norm_debug.reserve_back(1); pack_tile(4,17); norm_debug.push_back(1);\n   weight_debug.reserve_back(1); pack_tile(5,18); weight_debug.push_back(1);\n   inverse_debug.reserve_back(1); pack_tile(1,19); inverse_debug.push_back(1);\n  }\n  tile_regs_release_math_clear();\n }\n core.pop_front(4); sums.pop_front(1); gamma.pop_front(4);\n}\n'


class WeightedNormTail:
    """Preserve explicit post-sum arithmetic, returning BF16-rounded FP32 weights."""

    def __init__(self, t, mesh, gamma, debug=False):
        self.t, self.mesh, self.debug = (t, mesh, debug)
        self.cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(0, 0), t.CoreCoord(0, 0))])
        self.memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 128), t.ShardOrientation.ROW_MAJOR),
        )
        self.sum_memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 32), t.ShardOrientation.ROW_MAJOR),
        )
        expanded_gamma = gamma.float().reshape(1, 1, 128).expand(1, 32, 128).contiguous()
        self.gamma = t.from_torch(
            expanded_gamma, dtype=t.float32, layout=t.TILE_LAYOUT, device=mesh, memory_config=self.memory
        )
        self.config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        self.config.unpack_to_dest_mode = modes
        self.source = WEIGHTED_NORM_COMPUTE_SOURCE

    def __call__(self, core, sums):
        t = self.t
        assert tuple(core.shape) == (1, 32, 128) and tuple(sums.shape) == (1, 32, 1)
        assert core.dtype == sums.dtype == t.float32 and core.layout == sums.layout == t.TILE_LAYOUT
        assert core.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
        assert sums.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
        output = t.empty(
            (1, 32, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.memory
        )
        bindings = [(2, self.gamma), (16, output)]
        debug_outputs = {}
        if self.debug:
            for index, name in ((17, "normalized_fp32"), (18, "weighted_fp32_carrier"), (19, "inverse_repeated")):
                value = t.empty(
                    (1, 32, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.memory
                )
                bindings.append((index, value))
                debug_outputs[name] = value
        ac = list(t.TensorAccessorArgs(core).get_compile_time_args())
        bc = list(t.TensorAccessorArgs(sums).get_compile_time_args())
        runtime = t.RuntimeArgs()
        runtime[0][0] = [core.buffer_address(), sums.buffer_address()]
        read = t.KernelDescriptor(
            kernel_source=WEIGHTED_NORM_INTERLEAVED_READER,
            source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self.cores,
            compile_time_args=[1 + len(ac)] + ac + bc,
            runtime_args=runtime,
            config=t.ReaderConfigDescriptor(),
        )
        compute = t.KernelDescriptor(
            kernel_source=self.source,
            source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self.cores,
            compile_time_args=[int(self.debug)],
            runtime_args=[],
            config=self.config,
        )
        cbs = [t.cb_descriptor_from_sharded_tensor(i, value) for i, value in bindings]
        for index, pages in ((0, 4), (1, 1)):
            cbs.append(
                t.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=self.cores,
                    format_descriptors=[t.CBFormatDescriptor(index, t.float32, 4096)],
                )
            )
        program = t.ProgramDescriptor(kernels=[read, compute], semaphores=[], cbs=cbs)
        t.generic_op([core, sums] + [value for _, value in bindings], program)
        return (output, debug_outputs)


TERMINAL_SCAN_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n const auto input=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));\n const uint32_t start=get_arg_val<uint32_t>(1), runs=get_arg_val<uint32_t>(4);\n const uint32_t ht=get_arg_val<uint32_t>(2), wt=get_arg_val<uint32_t>(3);\n Noc noc; DataflowBuffer buffer(0);\n for(uint32_t run=start;run<start+runs;++run) { for(uint32_t h=0;h<ht;++h) {\n  buffer.reserve_back(1);\n  const uint32_t id=(run/wt)*ht*wt+h*wt+run%wt;\n  noc.async_read(input,buffer,4096,{.page_id=id},{.offset_bytes=0});\n  noc.async_read_barrier();buffer.push_back(1);\n }}\n}\n'
TERMINAL_SCAN_COMPUTE = '\n#include "api/compute/compute_kernel_api.h"\n#include "api/compute/common.h"\n#include "api/compute/compute_kernel_hw_startup.h"\n#include "api/compute/cumsum.h"\n#include "api/compute/pack.h"\n#include "api/compute/tile_move_copy.h"\n#include "api/dataflow/dataflow_buffer.h"\nvoid kernel_main() {\n DataflowBuffer input(0), output(16);\n compute_kernel_hw_startup(0,16);\n constexpr uint32_t count=get_compile_time_arg_val(0);\n const uint32_t runs=get_arg_val<uint32_t>(0);\n for(uint32_t run=0;run<runs;++run) { for(uint32_t j=0;j<count;++j) {\n  input.wait_front(1);tile_regs_acquire();\n  copy_init(0);copy_tile(0,0,0);\n  cumsum_tile_init<CumsumMode::Default>();\n  cumsum_tile<CumsumMode::Default>(0,j==0);\n  tile_regs_commit();input.pop_front(1);tile_regs_wait();\n  if(j+1==count) {\n   output.reserve_back(1);pack_tile(0,16);output.push_back(1);\n  }\n  tile_regs_release();\n }}\n}\n'
TERMINAL_SCAN_WRITER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n const auto output=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));\n const uint32_t start=get_arg_val<uint32_t>(1), runs=get_arg_val<uint32_t>(2);\n Noc noc;DataflowBuffer final_tile(16), row(17);\n for(uint32_t id=start;id<start+runs;++id) {\n final_tile.wait_front(1);row.reserve_back(1);\n auto* src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(16));\n auto* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(17));\n for(uint32_t i=0;i<1024;++i) dst[i]=0;\n // Float32 tile faces: copy physical row31 to logical row0 without arithmetic.\n for(uint32_t i=0;i<16;++i) { dst[i]=src[752+i];dst[256+i]=src[1008+i]; }\n row.push_back(1);row.wait_front(1);\n noc.async_write(row,output,4096,{}, {.page_id=id});\n noc.async_write_barrier();row.pop_front(1);final_tile.pop_front(1);\n }\n}\n'


def terminal_projection_schedule(total, grid_x, grid_y):
    """Balanced contiguous run partition, with no inactive CB consumers."""
    active = min(total, grid_x * grid_y)
    base, extra = divmod(total, active)
    start, result = (0, [])
    for i in range(active):
        count = base + (i < extra)
        result.append((i % grid_x, i // grid_x, start, count))
        start += count
    assert start == total
    return result


class TerminalProjectionScan:
    """Keep native Default cumsum arithmetic and emit one row per leading batch."""

    def __init__(self, ttnn, mesh):
        self.ttnn, self.mesh = (ttnn, mesh)
        self.grid = mesh.compute_with_storage_grid_size()

    def __call__(self, products):
        ttnn = self.ttnn
        shape = tuple(products.shape)
        assert len(shape) >= 2 and shape[-2] in (512, 1024)
        assert shape[-1] > 0 and shape[-1] % 32 == 0
        assert products.dtype == ttnn.float32 and products.layout == ttnn.TILE_LAYOUT
        assert products.memory_config() in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
        ht, wt = (shape[-2] // 32, shape[-1] // 32)
        runs = math.prod(shape[:-2]) * wt
        partition = terminal_projection_schedule(runs, self.grid.x, self.grid.y)
        cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y, _, _ in partition]
        )
        output = ttnn.empty(
            shape[:-2] + (1, shape[-1]),
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=products.memory_config(),
        )
        rd, wr, cr = (ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs())
        for x, y, start, count in partition:
            rd[x][y] = [products.buffer_address(), start, ht, wt, count]
            wr[x][y] = [output.buffer_address(), start, count]
            cr[x][y] = [count]
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes

        def kernel(source, conf, compile_args, runtime):
            return ttnn.KernelDescriptor(
                kernel_source=source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=compile_args,
                runtime_args=runtime,
            )

        program = ttnn.ProgramDescriptor(
            kernels=[
                kernel(
                    TERMINAL_SCAN_READER,
                    ttnn.ReaderConfigDescriptor(),
                    ttnn.TensorAccessorArgs(products).get_compile_time_args(),
                    rd,
                ),
                kernel(TERMINAL_SCAN_COMPUTE, config, [ht], cr),
                kernel(
                    TERMINAL_SCAN_WRITER,
                    ttnn.WriterConfigDescriptor(),
                    ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                    wr,
                ),
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.float32, 4096)],
                )
                for index, pages in [(0, 2), (16, 1), (17, 1)]
            ],
        )
        ttnn.generic_op([products, output], program)
        return output


STREAMING_SCAN_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n constexpr uint32_t coeff_offset=get_compile_time_arg_val(0);\n const auto weight=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));\n const auto coeff=TensorAccessor(TensorAccessorArgs<coeff_offset>(),get_arg_val<uint32_t>(1));\n const uint32_t start=get_arg_val<uint32_t>(2), runs=get_arg_val<uint32_t>(3);\n const uint32_t ht=get_arg_val<uint32_t>(4), wt=get_arg_val<uint32_t>(5);\n Noc noc;DataflowBuffer a(0),b(1);\n for(uint32_t run=start;run<start+runs;++run) {\n  for(uint32_t h=0;h<ht;++h) {\n   a.reserve_back(1);b.reserve_back(1);\n   const uint32_t tile=(run/wt)*ht*wt+h*wt+run%wt;\n   const uint32_t column=(run/wt)*ht+h;\n   noc.async_read(weight,a,2048,{.page_id=tile},{.offset_bytes=0});\n   noc.async_read(coeff,b,2048,{.page_id=column},{.offset_bytes=0});\n   noc.async_read_barrier();a.push_back(1);b.push_back(1);\n  }\n }\n}\n'
STREAMING_SCAN_COMPUTE = '\n#include "api/compute/compute_kernel_api.h"\n#include "api/compute/common.h"\n#include "api/compute/compute_kernel_hw_startup.h"\n#include "api/compute/bcast.h"\n#include "api/compute/cumsum.h"\n#include "api/compute/pack.h"\n#include "api/dataflow/dataflow_buffer.h"\nvoid kernel_main() {\n DataflowBuffer a(0),b(1),out(16);\n compute_kernel_hw_startup(0,1,16);\n constexpr uint32_t ht=get_compile_time_arg_val(0);\n constexpr bool products_only=get_compile_time_arg_val(1)!=0;\n const uint32_t runs=get_arg_val<uint32_t>(0);\n for(uint32_t run=0;run<runs;++run) {\n  for(uint32_t h=0;h<ht;++h) {\n   a.wait_front(1);b.wait_front(1);tile_regs_acquire();\n   // FPU multiplication leaves the native cumsum SFPU carry registers intact.\n   mul_bcast_cols_init(0,1);mul_tiles_bcast_cols(0,1,0,0,0);\n   if constexpr(!products_only) {\n    cumsum_tile_init<CumsumMode::Default>();\n    cumsum_tile<CumsumMode::Default>(0,h==0);\n   }\n   tile_regs_commit();a.pop_front(1);b.pop_front(1);tile_regs_wait();\n   if(products_only || h+1==ht) {\n    out.reserve_back(1);pack_tile(0,16);out.push_back(1);\n   }\n   tile_regs_release();\n  }\n }\n}\n'
STREAMING_SCAN_PRODUCT_WRITER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n const auto output=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));\n const uint32_t start=get_arg_val<uint32_t>(1), runs=get_arg_val<uint32_t>(2);\n const uint32_t ht=get_arg_val<uint32_t>(3),wt=get_arg_val<uint32_t>(4);\n Noc noc;DataflowBuffer out(16);\n for(uint32_t run=start;run<start+runs;++run) {\n  for(uint32_t h=0;h<ht;++h) {\n   const uint32_t tile=(run/wt)*ht*wt+h*wt+run%wt;\n   out.wait_front(1);\n   noc.async_write(out,output,4096,{}, {.page_id=tile});\n   noc.async_write_barrier();out.pop_front(1);\n  }\n }\n}\n'
STREAMING_SCAN_WRITER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n const auto output=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));\n const uint32_t start=get_arg_val<uint32_t>(1), runs=get_arg_val<uint32_t>(2);\n Noc noc;DataflowBuffer final_tile(16), row(17);\n for(uint32_t id=start;id<start+runs;++id) {\n final_tile.wait_front(1);row.reserve_back(1);\n auto* src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(16));\n auto* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(17));\n for(uint32_t i=0;i<1024;++i) dst[i]=0;\n // Float32 tile faces: copy physical row31 to logical row0 without arithmetic.\n for(uint32_t i=0;i<16;++i) { dst[i]=src[752+i];dst[256+i]=src[1008+i]; }\n row.push_back(1);row.wait_front(1);\n noc.async_write(row,output,4096,{}, {.page_id=id});\n noc.async_write_barrier();row.pop_front(1);final_tile.pop_front(1);\n }\n}\n'


class StreamingProjectionScan:
    """BF16 column-broadcast product into FP32 DST followed by native cumsum."""

    def __init__(self, ttnn, mesh):
        self.ttnn, self.mesh = (ttnn, mesh)
        self.grid = mesh.compute_with_storage_grid_size()

    def __call__(self, weight, coefficients, *, products_only=False):
        ttnn = self.ttnn
        shape = tuple(weight.shape)
        assert len(shape) >= 2 and shape[-2] in (512, 1024) and (shape[-1] % 32 == 0)
        assert tuple(coefficients.shape) == shape[:-1] + (1,)
        for tensor in (weight, coefficients):
            assert tensor.dtype == ttnn.bfloat16 and tensor.layout == ttnn.TILE_LAYOUT
            assert tensor.memory_config() in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
        ht, wt = (shape[-2] // 32, shape[-1] // 32)
        partition = terminal_projection_schedule(math.prod(shape[:-2]) * wt, self.grid.x, self.grid.y)
        cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y, _, _ in partition]
        )
        output = ttnn.empty(
            shape if products_only else shape[:-2] + (1, shape[-1]),
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if products_only else ttnn.L1_MEMORY_CONFIG,
        )
        rd, wr, cr = (ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs())
        for x, y, start, count in partition:
            rd[x][y] = [weight.buffer_address(), coefficients.buffer_address(), start, count, ht, wt]
            wr[x][y] = [output.buffer_address(), start, count, ht, wt]
            cr[x][y] = [count]
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )

        def kernel(source, conf, args, runtime):
            return ttnn.KernelDescriptor(
                kernel_source=source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=args,
                runtime_args=runtime,
            )

        wa = ttnn.TensorAccessorArgs(weight).get_compile_time_args()
        ca = ttnn.TensorAccessorArgs(coefficients).get_compile_time_args()
        program = ttnn.ProgramDescriptor(
            kernels=[
                kernel(STREAMING_SCAN_READER, ttnn.ReaderConfigDescriptor(), [1 + len(wa)] + wa + ca, rd),
                kernel(STREAMING_SCAN_COMPUTE, config, [ht, int(products_only)], cr),
                kernel(
                    STREAMING_SCAN_PRODUCT_WRITER if products_only else STREAMING_SCAN_WRITER,
                    ttnn.WriterConfigDescriptor(),
                    ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                    wr,
                ),
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=pages * page,
                    core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page)],
                )
                for index, pages, page, dtype in [
                    (0, 2, 2048, ttnn.bfloat16),
                    (1, 2, 2048, ttnn.bfloat16),
                    (16, 1, 4096, ttnn.float32),
                    (17, 1, 4096, ttnn.float32),
                ]
            ],
        )
        ttnn.generic_op([weight, coefficients, output], program)
        return output


FUSED_CORRECTED_MM_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nconstexpr uint32_t face_offset(uint32_t r,uint32_t c) {\n return (r/16)*512+(c/16)*256+(r%16)*16+c%16;\n}\nvoid kernel_main() {\n constexpr uint32_t b_accessor_offset=get_compile_time_arg_val(0);\n const auto at=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));\n const auto bt=TensorAccessor(TensorAccessorArgs<b_accessor_offset>(),get_arg_val<uint32_t>(1));\n const uint32_t start=get_arg_val<uint32_t>(2), count=get_arg_val<uint32_t>(3);\n const uint32_t M=get_arg_val<uint32_t>(4), N=get_arg_val<uint32_t>(5), K=get_arg_val<uint32_t>(6);\n const uint32_t rank=get_arg_val<uint32_t>(7), Mt=(M+31)/32, Nt=(N+31)/32, Kt=(K+31)/32;\n DataflowBuffer a(0), b(1), zero(2), ascratch(3), bscratch(4);\n Noc noc;\n ascratch.reserve_back(1); bscratch.reserve_back(1);\n auto* sa=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3));\n auto* sb=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4));\n for(uint32_t tile=start;tile<start+count;++tile) {\n  const uint32_t matrix_tile=tile%(Mt*Nt), mr=matrix_tile/Nt, nc=matrix_tile%Nt;\n  uint32_t batch=tile/(Mt*Nt), abatch=0, bbatch=0;\n  for(uint32_t i=rank;i>0;--i) {\n   const uint32_t j=i-1, d=get_arg_val<uint32_t>(8+j), coordinate=batch%d;\n   batch/=d;\n   abatch+=coordinate*get_arg_val<uint32_t>(8+rank+j);\n   bbatch+=coordinate*get_arg_val<uint32_t>(8+2*rank+j);\n  }\n  zero.reserve_back(1);\n  auto* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));\n  for(uint32_t i=0;i<1024;++i) z[i]=0;\n  zero.push_back(1);\n  for(uint32_t k=0;k<K;++k) {\n   if(k%32==0) {\n    noc.async_read(at,ascratch,4096,{.page_id=(abatch*Mt+mr)*Kt+k/32},{.offset_bytes=0});\n    noc.async_read(bt,bscratch,4096,{.page_id=(bbatch*Kt+k/32)*Nt+nc},{.offset_bytes=0});\n    noc.async_read_barrier();\n   }\n   a.reserve_back(1); b.reserve_back(1);\n   auto* aa=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));\n   auto* bb=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n   for(uint32_t r=0;r<32;++r) {\n    const uint32_t av=mr*32+r<M ? sa[face_offset(r,k%32)] : 0;\n    for(uint32_t c=0;c<32;++c) {\n     const uint32_t offset=face_offset(r,c);\n     aa[offset]=av;\n     bb[offset]=nc*32+c<N ? sb[face_offset(k%32,c)] : 0;\n    }\n   }\n   a.push_back(1); b.push_back(1);\n  }\n }\n}\n'
FUSED_CORRECTED_MM_COMPUTE = '\n#include "api/compute/common.h"\n#include "api/compute/compute_kernel_api.h"\n#include "api/compute/compute_kernel_hw_startup.h"\n#include "api/compute/experimental/reg_api.h"\n#include "api/compute/pack.h"\n#include "api/compute/tile_move_copy.h"\n#include "api/dataflow/circular_buffer.h"\n#ifdef TRISC_MATH\n#include "llk_math_eltwise_ternary_sfpu.h"\n#endif\nvoid kernel_main() {\n CircularBuffer a(0),b(1),zero(2),feedback(24),output(16);\n compute_kernel_hw_startup(0,16); copy_init(0);\n constexpr uint32_t K=get_compile_time_arg_val(0);\n const uint32_t count=get_arg_val<uint32_t>(0);\n for(uint32_t tile=0;tile<count;++tile) {\n  for(uint32_t k=0;k<K;++k) {\n   tile_regs_acquire_math_clear();\n   a.wait_front(1); b.wait_front(1);\n   if(k==0) { zero.wait_front(1); } else { feedback.wait_front(1); }\n   copy_tile(0,0,0); copy_tile(1,0,1);\n   if(k==0) { copy_tile(2,0,2); } else { copy_tile(24,0,2); }\n#ifdef TRISC_MATH\n   sfpu::_init_sfpu_config_reg();\n   addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);\n   math::reset_counters(p_setrwc::SET_ABD_F);\n   _llk_math_eltwise_sfpu_start_(0);\n   for(uint32_t lane=0;lane<32;++lane) {\n    sfpi::dst_reg[96] = sfpi::vFloat(0x1.0010000000000p+12f);\nsfpi::dst_reg[97] = sfpi::vFloat(0x0.0p+0f);\nsfpi::dst_reg[98] = sfpi::vFloat(0x1.0000000000000p-1f);\n{ sfpi::vFloat a=sfpi::dst_reg[96]; sfpi::vFloat b=sfpi::dst_reg[0]; sfpi::dst_reg[99]=a * b; }\n{ sfpi::vFloat a=sfpi::dst_reg[99]; sfpi::vFloat b=sfpi::dst_reg[0]; sfpi::dst_reg[100]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[99]; sfpi::vFloat b=sfpi::dst_reg[100]; sfpi::dst_reg[101]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[0]; sfpi::vFloat b=sfpi::dst_reg[101]; sfpi::dst_reg[102]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[96]; sfpi::vFloat b=sfpi::dst_reg[32]; sfpi::dst_reg[103]=a * b; }\n{ sfpi::vFloat a=sfpi::dst_reg[103]; sfpi::vFloat b=sfpi::dst_reg[32]; sfpi::dst_reg[104]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[103]; sfpi::vFloat b=sfpi::dst_reg[104]; sfpi::dst_reg[105]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[32]; sfpi::vFloat b=sfpi::dst_reg[105]; sfpi::dst_reg[106]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[0]; sfpi::vFloat b=sfpi::dst_reg[32]; sfpi::dst_reg[107]=a * b; }\n{ sfpi::vFloat a=sfpi::dst_reg[101]; sfpi::vFloat b=sfpi::dst_reg[105]; sfpi::dst_reg[108]=a * b; }\n{ sfpi::vFloat a=sfpi::dst_reg[107]; sfpi::vFloat b=sfpi::dst_reg[108]; sfpi::dst_reg[109]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[102]; sfpi::vFloat b=sfpi::dst_reg[105]; sfpi::dst_reg[110]=a * b; }\n{ sfpi::vFloat a=sfpi::dst_reg[109]; sfpi::vFloat b=sfpi::dst_reg[110]; sfpi::dst_reg[111]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[101]; sfpi::vFloat b=sfpi::dst_reg[106]; sfpi::dst_reg[112]=a * b; }\n{ sfpi::vFloat a=sfpi::dst_reg[111]; sfpi::vFloat b=sfpi::dst_reg[112]; sfpi::dst_reg[113]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[102]; sfpi::vFloat b=sfpi::dst_reg[106]; sfpi::dst_reg[114]=a * b; }\n{ sfpi::vFloat a=sfpi::dst_reg[114]; sfpi::vFloat b=sfpi::dst_reg[113]; sfpi::dst_reg[115]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[107]; sfpi::vFloat b=sfpi::dst_reg[64]; sfpi::dst_reg[116]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[116]; sfpi::vFloat b=sfpi::dst_reg[107]; sfpi::dst_reg[117]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[116]; sfpi::vFloat b=sfpi::dst_reg[117]; sfpi::dst_reg[118]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[107]; sfpi::vFloat b=sfpi::dst_reg[118]; sfpi::dst_reg[119]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[64]; sfpi::vFloat b=sfpi::dst_reg[117]; sfpi::dst_reg[120]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[119]; sfpi::vFloat b=sfpi::dst_reg[120]; sfpi::dst_reg[121]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[121]; sfpi::vFloat b=sfpi::dst_reg[115]; sfpi::dst_reg[122]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[116]; sfpi::vFloat b=sfpi::dst_reg[122]; sfpi::dst_reg[123]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[64]; sfpi::vFloat b=sfpi::dst_reg[115]; sfpi::dst_reg[124]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[124]; sfpi::vFloat b=sfpi::dst_reg[64]; sfpi::dst_reg[125]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[124]; sfpi::vFloat b=sfpi::dst_reg[125]; sfpi::dst_reg[126]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[64]; sfpi::vFloat b=sfpi::dst_reg[126]; sfpi::dst_reg[127]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[115]; sfpi::vFloat b=sfpi::dst_reg[125]; sfpi::dst_reg[128]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[127]; sfpi::vFloat b=sfpi::dst_reg[128]; sfpi::dst_reg[129]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[124]; sfpi::vFloat b=sfpi::dst_reg[107]; sfpi::dst_reg[130]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[130]; sfpi::vFloat b=sfpi::dst_reg[124]; sfpi::dst_reg[131]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[130]; sfpi::vFloat b=sfpi::dst_reg[131]; sfpi::dst_reg[132]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[124]; sfpi::vFloat b=sfpi::dst_reg[132]; sfpi::dst_reg[133]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[107]; sfpi::vFloat b=sfpi::dst_reg[131]; sfpi::dst_reg[134]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[133]; sfpi::vFloat b=sfpi::dst_reg[134]; sfpi::dst_reg[135]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[97]; sfpi::vFloat b=sfpi::dst_reg[123]; sfpi::dst_reg[136]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[136]; sfpi::vFloat b=sfpi::dst_reg[129]; sfpi::dst_reg[137]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[137]; sfpi::vFloat b=sfpi::dst_reg[136]; sfpi::dst_reg[138]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[137]; sfpi::vFloat b=sfpi::dst_reg[138]; sfpi::dst_reg[139]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[136]; sfpi::vFloat b=sfpi::dst_reg[139]; sfpi::dst_reg[140]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[129]; sfpi::vFloat b=sfpi::dst_reg[138]; sfpi::dst_reg[141]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[140]; sfpi::vFloat b=sfpi::dst_reg[141]; sfpi::dst_reg[142]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[137]; sfpi::vFloat b=sfpi::dst_reg[135]; sfpi::dst_reg[143]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[143]; sfpi::vFloat b=sfpi::dst_reg[137]; sfpi::dst_reg[144]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[143]; sfpi::vFloat b=sfpi::dst_reg[144]; sfpi::dst_reg[145]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[137]; sfpi::vFloat b=sfpi::dst_reg[145]; sfpi::dst_reg[146]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[135]; sfpi::vFloat b=sfpi::dst_reg[144]; sfpi::dst_reg[147]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[146]; sfpi::vFloat b=sfpi::dst_reg[147]; sfpi::dst_reg[148]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[143]; sfpi::vFloat b=sfpi::dst_reg[130]; sfpi::dst_reg[149]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[149]; sfpi::vFloat b=sfpi::dst_reg[143]; sfpi::dst_reg[150]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[149]; sfpi::vFloat b=sfpi::dst_reg[150]; sfpi::dst_reg[151]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[143]; sfpi::vFloat b=sfpi::dst_reg[151]; sfpi::dst_reg[152]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[130]; sfpi::vFloat b=sfpi::dst_reg[150]; sfpi::dst_reg[153]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[152]; sfpi::vFloat b=sfpi::dst_reg[153]; sfpi::dst_reg[154]=a + b; }\n{ sfpi::vFloat direction=0.0f; { sfpi::vFloat x=sfpi::dst_reg[142]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) { direction=1.0f; } v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) { direction=-1.0f; } v_endif; } { sfpi::vFloat x=sfpi::dst_reg[148]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) { direction=1.0f; } v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) { direction=-1.0f; } v_endif; } { sfpi::vFloat x=sfpi::dst_reg[154]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) { direction=1.0f; } v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) { direction=-1.0f; } v_endif; } { sfpi::vFloat x=sfpi::dst_reg[149]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) { direction=1.0f; } v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) { direction=-1.0f; } v_endif; } sfpi::dst_reg[155]=direction; }\n{\n sfpi::vFloat r=sfpi::dst_reg[123]; sfpi::vFloat direction=sfpi::dst_reg[155];\n sfpi::vInt bits=sfpi::as<sfpi::vInt>(r); sfpi::vInt step=0;\n v_if(direction>0.0f && bits>=0) { step=1; } v_endif;\n v_if(direction>0.0f && bits<0) { step=-1; } v_endif;\n v_if(direction<0.0f && bits>=0) { step=-1; } v_endif;\n v_if(direction<0.0f && bits<0) { step=1; } v_endif;\n sfpi::vInt nextbits=bits+step;\n v_if((bits & 0x7fffffff)==0 && direction>0.0f) { nextbits=1; } v_endif;\n v_if((bits & 0x7fffffff)==0 && direction<0.0f) { nextbits=sfpi::vInt(-2147483647); } v_endif;\n sfpi::dst_reg[156]=sfpi::as<sfpi::vFloat>(nextbits);\n}\n{ sfpi::vFloat a=sfpi::dst_reg[156]; sfpi::vFloat b=sfpi::dst_reg[123]; sfpi::dst_reg[157]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[157]; sfpi::vFloat b=sfpi::dst_reg[98]; sfpi::dst_reg[158]=a * b; }\n{ sfpi::vFloat a=sfpi::dst_reg[97]; sfpi::vFloat b=sfpi::dst_reg[158]; sfpi::dst_reg[159]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[159]; sfpi::vFloat b=sfpi::dst_reg[142]; sfpi::dst_reg[160]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[160]; sfpi::vFloat b=sfpi::dst_reg[159]; sfpi::dst_reg[161]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[160]; sfpi::vFloat b=sfpi::dst_reg[161]; sfpi::dst_reg[162]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[159]; sfpi::vFloat b=sfpi::dst_reg[162]; sfpi::dst_reg[163]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[142]; sfpi::vFloat b=sfpi::dst_reg[161]; sfpi::dst_reg[164]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[163]; sfpi::vFloat b=sfpi::dst_reg[164]; sfpi::dst_reg[165]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[160]; sfpi::vFloat b=sfpi::dst_reg[148]; sfpi::dst_reg[166]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[166]; sfpi::vFloat b=sfpi::dst_reg[160]; sfpi::dst_reg[167]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[166]; sfpi::vFloat b=sfpi::dst_reg[167]; sfpi::dst_reg[168]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[160]; sfpi::vFloat b=sfpi::dst_reg[168]; sfpi::dst_reg[169]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[148]; sfpi::vFloat b=sfpi::dst_reg[167]; sfpi::dst_reg[170]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[169]; sfpi::vFloat b=sfpi::dst_reg[170]; sfpi::dst_reg[171]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[166]; sfpi::vFloat b=sfpi::dst_reg[154]; sfpi::dst_reg[172]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[172]; sfpi::vFloat b=sfpi::dst_reg[166]; sfpi::dst_reg[173]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[172]; sfpi::vFloat b=sfpi::dst_reg[173]; sfpi::dst_reg[174]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[166]; sfpi::vFloat b=sfpi::dst_reg[174]; sfpi::dst_reg[175]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[154]; sfpi::vFloat b=sfpi::dst_reg[173]; sfpi::dst_reg[176]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[175]; sfpi::vFloat b=sfpi::dst_reg[176]; sfpi::dst_reg[177]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[172]; sfpi::vFloat b=sfpi::dst_reg[149]; sfpi::dst_reg[178]=a + b; }\n{ sfpi::vFloat a=sfpi::dst_reg[178]; sfpi::vFloat b=sfpi::dst_reg[172]; sfpi::dst_reg[179]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[178]; sfpi::vFloat b=sfpi::dst_reg[179]; sfpi::dst_reg[180]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[172]; sfpi::vFloat b=sfpi::dst_reg[180]; sfpi::dst_reg[181]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[149]; sfpi::vFloat b=sfpi::dst_reg[179]; sfpi::dst_reg[182]=a - b; }\n{ sfpi::vFloat a=sfpi::dst_reg[181]; sfpi::vFloat b=sfpi::dst_reg[182]; sfpi::dst_reg[183]=a + b; }\n{ sfpi::vFloat direction=0.0f; { sfpi::vFloat x=sfpi::dst_reg[165]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) { direction=1.0f; } v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) { direction=-1.0f; } v_endif; } { sfpi::vFloat x=sfpi::dst_reg[171]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) { direction=1.0f; } v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) { direction=-1.0f; } v_endif; } { sfpi::vFloat x=sfpi::dst_reg[177]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) { direction=1.0f; } v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) { direction=-1.0f; } v_endif; } { sfpi::vFloat x=sfpi::dst_reg[183]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) { direction=1.0f; } v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) { direction=-1.0f; } v_endif; } { sfpi::vFloat x=sfpi::dst_reg[178]; sfpi::vInt bits=sfpi::as<sfpi::vInt>(x); v_if((bits & 0x7fffffff) != 0) { direction=1.0f; } v_endif; v_if((bits & 0x7fffffff) != 0 && bits < 0) { direction=-1.0f; } v_endif; } sfpi::dst_reg[184]=direction; }\n{\n sfpi::vFloat r=sfpi::dst_reg[123]; sfpi::vFloat neighbor=sfpi::dst_reg[156];\n sfpi::vFloat direction=sfpi::dst_reg[155]; sfpi::vFloat cmp_sign=sfpi::dst_reg[184];\n sfpi::vFloat cmp=direction*cmp_sign; sfpi::vFloat selected=r; sfpi::vInt rb=sfpi::as<sfpi::vInt>(r);\n v_if(cmp>0.0f) { selected=neighbor; } v_endif;\n v_if(cmp==0.0f && direction!=0.0f && (rb & 1)!=0) { selected=neighbor; } v_endif;\n v_if(direction==0.0f && (rb & 0x7fffffff)==0) { selected=0.0f; } v_endif;\n sfpi::dst_reg[185]=selected;\n}\n{\n sfpi::vFloat selected=sfpi::dst_reg[185]; sfpi::vFloat direction=sfpi::dst_reg[155];\n sfpi::vInt rb=sfpi::as<sfpi::vInt>(selected);\n sfpi::vFloat a=sfpi::dst_reg[0]; sfpi::vInt ab=sfpi::as<sfpi::vInt>(a);\n sfpi::vFloat b=sfpi::dst_reg[32]; sfpi::vInt bb=sfpi::as<sfpi::vInt>(b);\n sfpi::vFloat c=sfpi::dst_reg[64]; sfpi::vInt cb=sfpi::as<sfpi::vInt>(c);\n // Flatten the signed-zero predicate: SFPI 7.95 rvtt_cc ICEs on the mixed\n // six-way conjunction/disjunction above a predicated assignment.\n sfpi::vInt eligible=1; sfpi::vInt zero_operand=0;\n v_if((ab & 0x7fffffff)==0) { zero_operand=1; } v_endif;\n v_if((bb & 0x7fffffff)==0) { zero_operand=1; } v_endif;\n v_if(direction!=0.0f) { eligible=0; } v_endif;\n v_if((rb & 0x7fffffff)!=0) { eligible=0; } v_endif;\n v_if(zero_operand==0) { eligible=0; } v_endif;\n v_if(cb!=sfpi::vInt(-2147483647-1)) { eligible=0; } v_endif;\n v_if((ab ^ bb)>=0) { eligible=0; } v_endif;\n v_if(eligible!=0) { selected=sfpi::as<sfpi::vFloat>(sfpi::vInt(-2147483647-1)); } v_endif;\n sfpi::dst_reg[185]=selected;\n}\n    { sfpi::vFloat result=sfpi::dst_reg[185]; sfpi::dst_reg[224]=result; }\n    sfpi::dst_reg++;\n   }\n   _llk_math_eltwise_sfpu_done_();\n#endif\n   // Release the consumed feedback page before PACK reserves its successor.\n   a.pop_front(1); b.pop_front(1);\n   if(k==0) { zero.pop_front(1); } else { feedback.pop_front(1); }\n   tile_regs_commit(); tile_regs_wait();\n   if(k+1==K) { output.reserve_back(1); pack_tile(7,16); output.push_back(1); }\n   else { feedback.reserve_back(1); pack_tile(7,24); feedback.push_back(1); }\n   tile_regs_release_math_clear();\n  }\n }\n}\n'
FUSED_CORRECTED_MM_WRITER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n const auto output=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));\n const uint32_t start=get_arg_val<uint32_t>(1), count=get_arg_val<uint32_t>(2);\n DataflowBuffer result(16); Noc noc;\n for(uint32_t tile=start;tile<start+count;++tile) {\n  result.wait_front(1);\n  noc.async_write(result,output,4096,{}, {.page_id=tile});\n  noc.async_write_barrier(); result.pop_front(1);\n }\n}\n'


def fused_corrected_mm_shape_plan(ashape, bshape):
    ashape, bshape = (tuple(ashape), tuple(bshape))
    assert len(ashape) >= 2 and len(bshape) >= 2
    assert all((x > 0 for x in ashape + bshape))
    assert ashape[-1] == bshape[-2]
    rank = max(len(ashape), len(bshape)) - 2
    al = (1,) * (rank - len(ashape) + 2) + ashape[:-2]
    bl = (1,) * (rank - len(bshape) + 2) + bshape[:-2]
    leading = []
    for a, b in zip(al, bl):
        assert a == b or a == 1 or b == 1, (ashape, bshape)
        leading.append(max(a, b))

    def strides(shape):
        return [math.prod(shape[i + 1 :]) if d != 1 else 0 for i, d in enumerate(shape)]

    m, k, n = (ashape[-2], ashape[-1], bshape[-1])
    return dict(
        shape=tuple(leading) + (m, n),
        leading=tuple(leading),
        a_strides=strides(al),
        b_strides=strides(bl),
        M=m,
        N=n,
        K=k,
        Mt=(m + 31) // 32,
        Nt=(n + 31) // 32,
        Kt=(k + 31) // 32,
        total_tiles=math.prod(leading) * ((m + 31) // 32) * ((n + 31) // 32),
    )


def fused_corrected_mm_schedule(total, gx, gy, max_cores):
    active = min(total, gx * gy, max_cores)
    assert active > 0
    base, extra = divmod(total, active)
    result = []
    start = 0
    for i in range(active):
        count = base + (i < extra)
        result.append((i % gx, i // gx, start, count))
        start += count
    assert start == total
    return result


class FusedCorrectedMatmul:
    """Device-only streaming matrix recurrence; arbitrary broadcast leading axes."""

    def __init__(self, t, mesh, max_cores=64):
        self.t, self.mesh, self.max_cores = (t, mesh, max_cores)
        self.source = FUSED_CORRECTED_MM_COMPUTE
        self.grid = mesh.compute_with_storage_grid_size()
        self.last_plan = None

    def __call__(self, a, b):
        t = self.t
        assert a.dtype == b.dtype == t.float32
        assert a.layout == b.layout == t.TILE_LAYOUT
        assert a.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        assert b.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        plan = fused_corrected_mm_shape_plan(tuple(a.shape), tuple(b.shape))
        partitions = fused_corrected_mm_schedule(plan["total_tiles"], self.grid.x, self.grid.y, self.max_cores)
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        out = t.empty(
            plan["shape"], device=self.mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        rd, wr, cr = (t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs())
        for x, y, start, count in partitions:
            rd[x][y] = [
                a.buffer_address(),
                b.buffer_address(),
                start,
                count,
                plan["M"],
                plan["N"],
                plan["K"],
                len(plan["leading"]),
                *plan["leading"],
                *plan["a_strides"],
                *plan["b_strides"],
            ]
            wr[x][y] = [out.buffer_address(), start, count]
            cr[x][y] = [count]
        ac = list(t.TensorAccessorArgs(a).get_compile_time_args())
        bc = list(t.TensorAccessorArgs(b).get_compile_time_args())
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2, 24):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

        def kernel(source, config, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=config,
                compile_time_args=args,
                runtime_args=runtime,
            )

        program = t.ProgramDescriptor(
            kernels=[
                kernel(FUSED_CORRECTED_MM_READER, t.ReaderConfigDescriptor(), [1 + len(ac)] + ac + bc, rd),
                kernel(self.source, cfg, [plan["K"]], cr),
                kernel(
                    FUSED_CORRECTED_MM_WRITER,
                    t.WriterConfigDescriptor(),
                    t.TensorAccessorArgs(out).get_compile_time_args(),
                    wr,
                ),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(i, t.float32, 4096)],
                )
                for i, pages in ((0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (16, 1), (24, 2))
            ],
        )
        t.generic_op([a, b, out], program)
        self.last_plan = {
            **plan,
            "core_count": len(partitions),
            "tiles_per_core": [p[3] for p in partitions],
            "cb_bytes_per_core": 40960,
        }
        return out


DIRECT_COEFFICIENT_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n constexpr uint32_t activation_offset=get_compile_time_arg_val(0);\n constexpr uint32_t output_offset=get_compile_time_arg_val(1);\n constexpr bool coefficients_only=get_compile_time_arg_val(2)!=0;\n const auto weight=TensorAccessor(TensorAccessorArgs<3>(),get_arg_val<uint32_t>(0));\n const auto activation=TensorAccessor(TensorAccessorArgs<activation_offset>(),get_arg_val<uint32_t>(1));\n const auto coefficient_output=TensorAccessor(TensorAccessorArgs<output_offset>(),get_arg_val<uint32_t>(6));\n const uint32_t start=get_arg_val<uint32_t>(2),runs=get_arg_val<uint32_t>(3);\n const uint32_t ht=get_arg_val<uint32_t>(4),wt=get_arg_val<uint32_t>(5);\n Noc noc;DataflowBuffer a(0),b(1),raw(2);\n // Initialize both contiguous coefficient CB pages before publishing either.\n // Only column0 changes afterward; padded columns stay zero on every reuse.\n b.reserve_back(2);\n auto* initial=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n for(uint32_t i=0;i<1024;++i) initial[i]=0;\n for(uint32_t run=start;run<start+runs;++run) {\n  if(coefficients_only && run%wt!=0) continue;\n  for(uint32_t h=0;h<ht;++h) {\n   const uint32_t tile=(run/wt)*ht*wt+h*wt+run%wt;\n   const uint32_t column=(run/wt)*ht+h;\n   raw.reserve_back(1);b.reserve_back(1);\n   if constexpr(!coefficients_only) {\n    a.reserve_back(1);\n    noc.async_read(weight,a,2048,{.page_id=tile},{.offset_bytes=0});\n   }\n   noc.async_read(activation,raw,2048,{.page_id=column},{.offset_bytes=0});\n   noc.async_read_barrier();raw.push_back(1);raw.wait_front(1);\n   auto* source=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(2));\n   auto* target=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(1));\n   // Original activation TILE row0, pair-flipped inside each 32-lane tile.\n   // Coefficient TILE column0: face0 rows0..15, face2 rows16..31.\n   for(uint32_t r=0;r<32;++r) {\n    const uint32_t paired=r^1;\n    const uint32_t src=(paired/16)*256+paired%16;\n    const uint32_t dst=(r/16)*512+(r%16)*16;\n    target[dst]=source[src];\n   }\n   raw.pop_front(1);b.push_back(1);\n   if constexpr(coefficients_only) {\n    b.wait_front(1);\n    noc.async_write(b,coefficient_output,2048,{}, {.page_id=column});\n    noc.async_write_barrier();b.pop_front(1);\n   } else { a.push_back(1); }\n  }\n }\n}\n'


class DirectCoefficientStreamingScan:
    """Same streaming FPU/cumsum, with raw BF16 original-input coefficient construction."""

    def __init__(self, ttnn, mesh):
        self.ttnn, self.mesh = (ttnn, mesh)
        self.grid = mesh.compute_with_storage_grid_size()

    def __call__(self, weight, activation, *, products_only=False, coefficients_only=False):
        t = self.ttnn
        shape = tuple(weight.shape)
        assert shape[-2] in (512, 1024) and shape[-1] % 32 == 0 and (len(shape) == 4) and (shape[1] == 1)
        total_k = math.prod(shape[:-2]) * shape[-2]
        assert tuple(activation.shape) == (1, 1, total_k)
        assert not (products_only and coefficients_only)
        assert weight.dtype == activation.dtype == t.bfloat16 and weight.layout == activation.layout == t.TILE_LAYOUT
        assert weight.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
        ht, wt = (shape[-2] // 32, shape[-1] // 32)
        partition = terminal_projection_schedule(math.prod(shape[:-2]) * wt, self.grid.x, self.grid.y)
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partition])
        output_shape = (
            shape[:-1] + (1,) if coefficients_only else shape if products_only else shape[:-2] + (1, shape[-1])
        )
        output = t.empty(
            output_shape,
            device=self.mesh,
            dtype=t.bfloat16 if coefficients_only else t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=t.DRAM_MEMORY_CONFIG if products_only or coefficients_only else t.L1_MEMORY_CONFIG,
        )
        rd, wr, cr = (t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs())
        for x, y, start, count in partition:
            rd[x][y] = [
                weight.buffer_address(),
                activation.buffer_address(),
                start,
                count,
                ht,
                wt,
                output.buffer_address(),
            ]
            wr[x][y] = [output.buffer_address(), start, count, ht, wt]
            cr[x][y] = [count]
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )

        def kernel(source, conf, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=args,
                runtime_args=runtime,
            )

        accessors = [t.TensorAccessorArgs(value) for value in (weight, activation, output)]
        assert all((not a.get_common_runtime_args() for a in accessors))
        wa, xa, oa = [a.get_compile_time_args() for a in accessors]
        kernels = [
            kernel(
                DIRECT_COEFFICIENT_READER,
                t.ReaderConfigDescriptor(),
                [3 + len(wa), 3 + len(wa) + len(xa), int(coefficients_only)] + wa + xa + oa,
                rd,
            )
        ]
        if not coefficients_only:
            kernels += [
                kernel(STREAMING_SCAN_COMPUTE, config, [ht, int(products_only)], cr),
                kernel(
                    STREAMING_SCAN_PRODUCT_WRITER if products_only else STREAMING_SCAN_WRITER,
                    t.WriterConfigDescriptor(),
                    oa,
                    wr,
                ),
            ]
        program = t.ProgramDescriptor(
            kernels=kernels,
            cbs=[
                t.CBDescriptor(
                    total_size=pages * page,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(index, dtype, page)],
                )
                for index, pages, page, dtype in [
                    (0, 2, 2048, t.bfloat16),
                    (1, 2, 2048, t.bfloat16),
                    (2, 1, 2048, t.bfloat16),
                    (16, 1, 4096, t.float32),
                    (17, 1, 4096, t.float32),
                ]
            ],
        )
        t.generic_op([weight, activation, output], program)
        return output


SILU_POS_LO, SILU_POS_HI, SILU_NEG_LO, SILU_NEG_HI = (13249, 16773, 46144, 49841)
SILU_POS_COUNT, SILU_NEG_COUNT = (3525, 3698)
SILU_COMPACT_ENTRIES = 7488
LOCAL_SILU_READER_SOURCE = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\n\ninline uint32_t silu_lookup_bits(uint32_t code, const volatile tt_l1_ptr uint32_t* table) {\n if(code<0x8000) {\n  if(code>=0x33c1 && code<=0x4185) return table[code-0x33c1];\n  if(code>0x4185 && code<0x7f80) return code<<16;\n  if(code>=0x7f80) return table[7223+code-0x7f80];\n } else {\n  if(code>=0xb440 && code<=0xc2b1) return table[3525+code-0xb440];\n  if(code>0xc2b1 && code<0xff80) return 0x80000000;\n  if(code>=0xff80) return table[7351+code-0xff80];\n }\n // Exactly representable FP32 x/2 for BF16 x, including FP32 subnormals.\n const uint32_t magnitude=(code&0x7fff)<<16, sign=(code&0x8000)<<16;\n return sign | (magnitude>=0x01000000 ? magnitude-0x00800000 : magnitude>>1);\n}\n\nvoid kernel_main() {\n auto w_args=TensorAccessorArgs<0,0>();\n auto z_args=TensorAccessorArgs<w_args.next_compile_time_args_offset(),w_args.next_common_runtime_args_offset()>();\n const auto weighted=TensorAccessor(w_args,get_arg_val<uint32_t>(0));\n const auto z=TensorAccessor(z_args,get_arg_val<uint32_t>(1));\n const auto* table=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));\n const uint32_t head=get_arg_val<uint32_t>(3);\n Noc noc; DataflowBuffer weights(0), gates(1), codes(2);\n // Blackhole DRAM reads require matching source/destination address bits 0..5.\n // Use two 64-byte-aligned slots for the 32-byte BF16 half rows. Do not assume\n // that an arbitrary circular-buffer base has stricter than L1 alignment.\n codes.reserve_back(1);\n const uint32_t code_bias=(64-(get_write_ptr(2)&63))&63;\n for(uint32_t tile=0;tile<4;++tile) {\n  weights.reserve_back(1);gates.reserve_back(1);\n  auto* w=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));\n  auto* g=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n  const auto* c=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(2)+code_bias);\n  for(uint32_t j=0;j<1024;++j) {w[j]=0;g[j]=0;}\n  const uint32_t source_row=((head/16)*512+(head%16)*16)*4;\n  for(uint32_t half=0;half<2;++half) {\n   noc.async_read(weighted,weights,64,\n      {.page_id=tile,.offset_bytes=source_row+half*1024},{.offset_bytes=half*1024});\n   noc.async_read(z,codes,32,\n      {.page_id=head*4+tile,.offset_bytes=half*512},{.offset_bytes=code_bias+half*64});\n  }\n  noc.async_read_barrier();\n  for(uint32_t j=0;j<32;++j) g[(j/16)*256+j%16]=silu_lookup_bits(c[(j/16)*32+j%16],table);\n  weights.push_back(1);gates.push_back(1);\n }\n}\n'
LOCAL_SILU_COMPUTE_SOURCE = '\n#include "api/compute/common.h"\n#include "api/compute/compute_kernel_api.h"\n#include "api/compute/compute_kernel_hw_startup.h"\n#include "api/compute/experimental/reg_api.h"\n#include "api/compute/pack.h"\n#include "api/compute/tile_move_copy.h"\n#include "api/dataflow/circular_buffer.h"\n#ifdef TRISC_MATH\n#include "llk_math_eltwise_ternary_sfpu.h"\n#endif\nvoid kernel_main() {\n constexpr bool debug=get_compile_time_arg_val(0)!=0;\n CircularBuffer weighted(0),gate(1),output(16),debug_gate(17);\n compute_kernel_hw_startup(0,16);copy_init(0);\n for(uint32_t tile=0;tile<4;++tile) {\n  weighted.wait_front(1);gate.wait_front(1);\n  tile_regs_acquire_math_clear();copy_tile(0,0,0);copy_tile(1,0,1);\n#ifdef TRISC_MATH\n  sfpu::_init_sfpu_config_reg();\n  addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);\n  math::reset_counters(p_setrwc::SET_ABD_F);_llk_math_eltwise_sfpu_start_(0);\n  for(uint32_t lane=0;lane<32;++lane) {\n   {sfpi::vFloat w=sfpi::dst_reg[0],g=sfpi::dst_reg[32];sfpi::dst_reg[0]=w*g;}\n   {sfpi::vFloat product=sfpi::dst_reg[0];sfpi::vUInt bits=sfpi::as<sfpi::vUInt>(product);\n    sfpi::vUInt tie=(bits>>16)&1;bits=bits+0x7fff;bits=bits+tie;\n    sfpi::dst_reg[0]=sfpi::as<sfpi::vFloat>(bits&0xffff0000);}\n   sfpi::dst_reg++;\n  }\n  _llk_math_eltwise_sfpu_done_();\n#endif\n  tile_regs_commit();tile_regs_wait();\n  pack_reconfig_data_format(16);output.reserve_back(1);pack_tile(0,16);output.push_back(1);\n  if constexpr(debug) {\n   pack_reconfig_data_format(17);debug_gate.reserve_back(1);pack_tile(1,17);debug_gate.push_back(1);\n  }\n  weighted.pop_front(1);gate.pop_front(1);tile_regs_release_math_clear();\n }\n}\n'


def local_silu_math_table():
    import torch

    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return torch.nn.functional.silu(codes.float()).reshape(1, 65536)


def compact_silu_math_table(full):
    import torch

    raw = torch.arange(65536, dtype=torch.int64)
    values = full.flatten().contiguous().view(torch.int32).long() & 4294967295
    magnitude, sign = ((raw & 32767) << 16, (raw & 32768) << 16)
    half = sign | torch.where(magnitude >= 16777216, magnitude - 8388608, magnitude >> 1)
    derived = []
    for start in (0, 32768):
        ids = raw[start : start + 32640]
        far = ids << 16 if start == 0 else torch.full_like(ids, 2147483648)
        lo = int(ids[values[ids] != half[ids]].min())
        hi = int(ids[values[ids] != far].max())
        derived.extend((lo, hi))
    assert derived == [SILU_POS_LO, SILU_POS_HI, SILU_NEG_LO, SILU_NEG_HI], derived
    parts = [
        values[SILU_POS_LO : SILU_POS_HI + 1],
        values[SILU_NEG_LO : SILU_NEG_HI + 1],
        values[32640:32768],
        values[65408:],
    ]
    packed = torch.cat(parts)
    assert packed.numel() == 7479
    padded = torch.cat((packed, torch.zeros(SILU_COMPACT_ENTRIES - packed.numel(), dtype=torch.int64)))
    reconstructed = half.clone()
    reconstructed[SILU_POS_LO : SILU_POS_HI + 1] = padded[:SILU_POS_COUNT]
    reconstructed[SILU_POS_HI + 1 : 32640] = raw[SILU_POS_HI + 1 : 32640] << 16
    reconstructed[SILU_NEG_LO : SILU_NEG_HI + 1] = padded[SILU_POS_COUNT : SILU_POS_COUNT + SILU_NEG_COUNT]
    reconstructed[SILU_NEG_HI + 1 : 65408] = 2147483648
    reconstructed[32640:32768] = padded[7223:7351]
    reconstructed[65408:] = padded[7351:7479]
    assert torch.equal(reconstructed, values), int((reconstructed != values).sum())
    return padded.to(torch.int32).view(torch.float32).reshape(1, SILU_COMPACT_ENTRIES)


class LocalSiluMultiply:
    """32 workers, compact immutable LUT, no runtime tensor conversions.

    Weighted FP32 carriers [1,32,128] have already been rounded to BF16.
    Z is BF16 TILE [1,1,4096]. Output directly matches native head concat:
    BF16 TILE [1,1,4096], padded [1,32,4096], 32 width shards [32,128].
    """

    def __init__(self, t, mesh, debug=False):
        self.t, self.mesh, self.debug = (t, mesh, debug)
        grid = mesh.compute_with_storage_grid_size()
        assert grid.x * grid.y >= 32
        self.coordinates = [(i % grid.x, i // grid.x) for i in range(32)]
        self.cores = t.num_cores_to_corerangeset(32, grid, row_wise=True)
        self.output_memory = t.MemoryConfig(
            t.TensorMemoryLayout.WIDTH_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (32, 128), t.ShardOrientation.ROW_MAJOR),
        )
        table_memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(self.cores, (1, SILU_COMPACT_ENTRIES), t.ShardOrientation.ROW_MAJOR),
        )
        self.table = t.from_torch(
            compact_silu_math_table(local_silu_math_table()).expand(32, SILU_COMPACT_ENTRIES).contiguous(),
            dtype=t.float32,
            layout=t.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=table_memory,
        )
        self.config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = t.UnpackToDestMode.UnpackToDestFp32
        self.config.unpack_to_dest_mode = modes

    def __call__(self, weighted, z):
        t = self.t
        assert tuple(weighted.shape) == (1, 32, 128) and weighted.dtype == t.float32
        assert tuple(z.shape) == (1, 1, 4096) and z.dtype == t.bfloat16
        assert weighted.layout == z.layout == t.TILE_LAYOUT
        output = t.empty(
            (1, 1, 4096), dtype=t.bfloat16, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.output_memory
        )
        bindings = [(16, output)]
        gate = None
        if self.debug:
            gate = t.empty(
                (1, 1, 4096), dtype=t.float32, layout=t.TILE_LAYOUT, device=self.mesh, memory_config=self.output_memory
            )
            bindings.append((17, gate))
        runtime = t.RuntimeArgs()
        for head, (x, y) in enumerate(self.coordinates):
            runtime[x][y] = [weighted.buffer_address(), z.buffer_address(), self.table.buffer_address(), head]
        accessor_args = t.TensorAccessorArgs(weighted).get_compile_time_args()
        accessor_args += t.TensorAccessorArgs(z).get_compile_time_args()

        def kernel(source, config, compile_args, runtime_args):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.cores,
                config=config,
                compile_time_args=compile_args,
                runtime_args=runtime_args,
            )

        cbs = [t.cb_descriptor_from_sharded_tensor(i, value) for i, value in bindings]
        for index, dtype, size, pages in ((0, t.float32, 4096, 2), (1, t.float32, 4096, 2), (2, t.bfloat16, 2048, 1)):
            cbs.append(
                t.CBDescriptor(
                    total_size=pages * size,
                    core_ranges=self.cores,
                    format_descriptors=[t.CBFormatDescriptor(index, dtype, size)],
                )
            )
        program = t.ProgramDescriptor(
            kernels=[
                kernel(LOCAL_SILU_READER_SOURCE, t.ReaderConfigDescriptor(), accessor_args, runtime),
                kernel(LOCAL_SILU_COMPUTE_SOURCE, self.config, [int(self.debug)], []),
            ],
            cbs=cbs,
        )
        t.generic_op([weighted, z, self.table, *[v for _, v in bindings]], program)
        return (output, gate)


CHUNK64_TRIANGULAR_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nconstexpr uint32_t face(uint32_t r,uint32_t c) {\n return (r/16)*512+(c/16)*256+(r%16)*16+c%16;\n}\nconstexpr uint32_t index(uint32_t r,uint32_t c) {\n return ((r/32)*2+c/32)*1024+face(r%32,c%32);\n}\nvoid kernel_main() {\n constexpr uint32_t output_offset=get_compile_time_arg_val(0);\n const auto input=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));\n const auto output=TensorAccessor(TensorAccessorArgs<output_offset>(),get_arg_val<uint32_t>(1));\n const uint32_t head=get_arg_val<uint32_t>(2),stop=get_arg_val<uint32_t>(3);\n DataflowBuffer coeff(0),previous(1),oldrow(2),matrix(3),feedback(16); Noc noc;\n matrix.reserve_back(4);\n auto* a=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3));\n for(uint32_t tile=0;tile<4;++tile)\n  noc.async_read(input,matrix,4096,{.page_id=head*4+tile},{.offset_bytes=tile*4096});\n noc.async_read_barrier();\n // Each producer ring has two contiguous pages; all unused tile rows remain zero.\n coeff.reserve_back(2);previous.reserve_back(2);oldrow.reserve_back(1);\n auto* cp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));\n auto* pp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n auto* op=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));\n for(uint32_t n=0;n<2048;++n) {cp[n]=0;pp[n]=0;}\n for(uint32_t n=0;n<1024;++n) op[n]=0;\n for(uint32_t i=1;i<=stop;++i) {\n  uint32_t saved[64];\n  for(uint32_t k=0;k<64;++k) saved[k]=a[index(i,k)];\n  const uint32_t passes=(i>=4&&i<8)?2:(i+31)/32;\n  for(uint32_t pass=0;pass<passes;++pass) {\n   const bool mixed=i>=4&&i<8;\n   const uint32_t column=mixed?0:pass*32;\n   const bool cascade=mixed?(pass==0):(i>=32&&pass==0);\n   const uint32_t begin=mixed?(pass==0?0:4):column;\n   const uint32_t end=mixed?(pass==0?4:i):((column+32<i)?column+32:i);\n   oldrow.reserve_back(1);\n   auto* old=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));\n   for(uint32_t c=0;c<32;++c) old[face(0,c)]=saved[column+c];\n   oldrow.push_back(1);\n   for(uint32_t k=0;k<i;++k) {\n    coeff.reserve_back(1);previous.reserve_back(1);\n    auto* cc=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));\n    auto* pv=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n    for(uint32_t c=0;c<32;++c) {\n     cc[face(0,c)]=saved[k];pv[face(0,c)]=a[index(k,column+c)];\n    }\n    coeff.push_back(1);previous.push_back(1);\n   }\n   feedback.wait_front(1);\n   auto* f=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(16));\n   for(uint32_t j=begin;j<end;++j) a[index(i,j)]=f[face(0,j-column)];\n   feedback.pop_front(1);\n  }\n }\n for(uint32_t tile=0;tile<4;++tile)\n  noc.async_write(matrix,output,4096,{.offset_bytes=tile*4096},{.page_id=head*4+tile});\n noc.async_write_barrier();\n}\n'
CHUNK64_TRIANGULAR_COMPUTE = '\n#include "api/compute/common.h"\n#include "api/compute/compute_kernel_api.h"\n#include "api/compute/compute_kernel_hw_startup.h"\n#include "api/compute/experimental/reg_api.h"\n#include "api/compute/pack.h"\n#include "api/compute/tile_move_copy.h"\n#include "api/dataflow/circular_buffer.h"\n#ifdef TRISC_MATH\n#include "llk_math_eltwise_ternary_sfpu.h"\n// Native FP32 product/add output boundary: preserve the sign when flushing.\ninline sfpi::vFloat flush_result(sfpi::vFloat value) {\n sfpi::vUInt bits=sfpi::as<sfpi::vUInt>(value);\n v_if((bits & 0x7f800000u)==0u) { bits=bits & 0x80000000u; } v_endif;\n return sfpi::as<sfpi::vFloat>(bits);\n}\ninline sfpi::vFloat add_round(sfpi::vFloat a,sfpi::vFloat b) {\n sfpi::vFloat sum=a+b;return flush_result(sum);\n}\ninline void begin_sfpu() {\n sfpu::_init_sfpu_config_reg();\n addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);\n math::reset_counters(p_setrwc::SET_ABD_F);_llk_math_eltwise_sfpu_start_(0);\n}\ntemplate<uint32_t SLOT> inline void product_add() {\n begin_sfpu();\n for(uint32_t lane=0;lane<32;++lane) {\n  sfpi::vFloat a=sfpi::dst_reg[128],b=sfpi::dst_reg[160];\n  sfpi::dst_reg[128]=flush_result(a*b);\n  sfpi::vFloat product=sfpi::dst_reg[128],acc=sfpi::dst_reg[SLOT];\n  sfpi::dst_reg[SLOT]=add_round(acc,product);\n  sfpi::dst_reg++;\n }\n _llk_math_eltwise_sfpu_done_();\n}\n#endif\nvoid kernel_main() {\n CircularBuffer coeff(0),previous(1),oldrow(2),feedback(16);\n compute_kernel_hw_startup(0,16);copy_init(0);\n const uint32_t stop=get_arg_val<uint32_t>(0);\n for(uint32_t i=1;i<=stop;++i) {\n  const bool mixed=i>=4&&i<8;\n  const uint32_t passes=mixed?2:(i+31)/32;\n  for(uint32_t pass=0;pass<passes;++pass) {\n   const bool cascade=mixed?(pass==0):(i>=32&&pass==0);\n   tile_regs_acquire_math_clear();oldrow.wait_front(1);copy_tile(2,0,6);oldrow.pop_front(1);\n#ifdef TRISC_MATH\n   begin_sfpu();\n   for(uint32_t lane=0;lane<32;++lane) {\n    sfpi::dst_reg[0]=0.0f;sfpi::dst_reg[32]=0.0f;\n    sfpi::dst_reg[64]=0.0f;sfpi::dst_reg[96]=0.0f;sfpi::dst_reg++;\n   }\n   _llk_math_eltwise_sfpu_done_();\n#endif\n   for(uint32_t k=0;k<i;++k) {\n    coeff.wait_front(1);previous.wait_front(1);copy_tile(0,0,4);copy_tile(1,0,5);\n#ifdef TRISC_MATH\n    const uint32_t slot=cascade?0:((k<(i/4)*4)?k%4:0);\n    if(slot==0) product_add<0>();\n    else if(slot==1) product_add<32>();\n    else if(slot==2) product_add<64>();\n    else product_add<96>();\n    if(cascade&&(k+1)%16==0) {\n     begin_sfpu();\n     for(uint32_t lane=0;lane<32;++lane) {\n      sfpi::vFloat done=sfpi::dst_reg[32],part=sfpi::dst_reg[0];\n      sfpi::dst_reg[32]=add_round(done,part);sfpi::dst_reg[0]=0.0f;sfpi::dst_reg++;\n     }\n     _llk_math_eltwise_sfpu_done_();\n    }\n#endif\n    coeff.pop_front(1);previous.pop_front(1);\n   }\n#ifdef TRISC_MATH\n   begin_sfpu();\n   for(uint32_t lane=0;lane<32;++lane) {\n    sfpi::vFloat a=sfpi::dst_reg[0],b=sfpi::dst_reg[32];\n    sfpi::dst_reg[0]=add_round(a,b);\n    if(!cascade) {\n     sfpi::vFloat p=sfpi::dst_reg[0],c=sfpi::dst_reg[64];\n     sfpi::dst_reg[0]=add_round(p,c);\n     sfpi::vFloat q=sfpi::dst_reg[0],d=sfpi::dst_reg[96];\n     sfpi::dst_reg[0]=add_round(q,d);\n    }\n    sfpi::vFloat z=0.0f,correction=sfpi::dst_reg[0];\n    sfpi::dst_reg[0]=add_round(z,correction);\n    sfpi::vFloat corrected=sfpi::dst_reg[0],old=sfpi::dst_reg[192];\n    sfpi::dst_reg[224]=add_round(old,corrected);sfpi::dst_reg++;\n   }\n   _llk_math_eltwise_sfpu_done_();\n#endif\n   tile_regs_commit();tile_regs_wait();feedback.reserve_back(1);\n   pack_tile(7,16);feedback.push_back(1);tile_regs_release_math_clear();\n  }\n }\n}\n'
CHUNK64_TRIANGULAR_EMPTY_WRITER = "void kernel_main() {}"


class Chunk64TriangularRecurrence:
    def __init__(self, t, mesh):
        self.t, self.mesh = (t, mesh)

    def __call__(self, matrix, stop=63):
        t = self.t
        shape = tuple(matrix.shape)
        assert shape[0] == 1 and shape[-2:] == (64, 64)
        assert matrix.dtype == t.float32 and matrix.layout == t.TILE_LAYOUT
        assert matrix.memory_config() == t.DRAM_MEMORY_CONFIG
        assert 1 <= stop <= 63
        heads = shape[1]
        grid = self.mesh.compute_with_storage_grid_size()
        assert heads <= grid.x * grid.y
        positions = [(h % grid.x, h // grid.x, h) for h in range(heads)]
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _ in positions])
        output = t.empty(
            shape, device=self.mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        rd, cr, wr = (t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs())
        for x, y, h in positions:
            rd[x][y] = [matrix.buffer_address(), output.buffer_address(), h, stop]
            cr[x][y] = [stop]
            wr[x][y] = []
        ac = list(t.TensorAccessorArgs(matrix).get_compile_time_args())
        bc = list(t.TensorAccessorArgs(output).get_compile_time_args())
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

        def kernel(source, config, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=config,
                compile_time_args=args,
                runtime_args=runtime,
            )

        program = t.ProgramDescriptor(
            kernels=[
                kernel(CHUNK64_TRIANGULAR_READER, t.ReaderConfigDescriptor(), [1 + len(ac)] + ac + bc, rd),
                kernel(CHUNK64_TRIANGULAR_COMPUTE, cfg, [], cr),
                kernel(CHUNK64_TRIANGULAR_EMPTY_WRITER, t.WriterConfigDescriptor(), [], wr),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(i, t.float32, 4096)],
                )
                for i, pages in ((0, 2), (1, 2), (2, 1), (3, 4), (16, 1))
            ],
        )
        t.generic_op([matrix, output], program)
        return output


DECODE_CONV_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nconstexpr uint32_t face(uint32_t c) {return (c/16)*256+c%16;}\nvoid kernel_main() {\n constexpr uint32_t so=get_compile_time_arg_val(0),w0o=get_compile_time_arg_val(1);\n constexpr uint32_t w1o=get_compile_time_arg_val(2),w2o=get_compile_time_arg_val(3),w3o=get_compile_time_arg_val(4);\n const auto q=TensorAccessor(TensorAccessorArgs<5>(),get_arg_val<uint32_t>(0));\n const auto state=TensorAccessor(TensorAccessorArgs<so>(),get_arg_val<uint32_t>(1));\n const auto w0=TensorAccessor(TensorAccessorArgs<w0o>(),get_arg_val<uint32_t>(2));\n const auto w1=TensorAccessor(TensorAccessorArgs<w1o>(),get_arg_val<uint32_t>(3));\n const auto w2=TensorAccessor(TensorAccessorArgs<w2o>(),get_arg_val<uint32_t>(4));\n const auto w3=TensorAccessor(TensorAccessorArgs<w3o>(),get_arg_val<uint32_t>(5));\n const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7);\n DataflowBuffer input(0),weight(1),first(2),scratch(3);Noc noc;\n scratch.reserve_back(1);\n const uint32_t bias=(64-(get_write_ptr(3)&63))&63;\n auto* raw=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(3)+bias);\n auto* wide=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3)+bias);\n input.reserve_back(2);weight.reserve_back(2);\n auto* iz=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));\n auto* wz=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n for(uint32_t n=0;n<2048;++n){iz[n]=0;wz[n]=0;}\n first.reserve_back(1);\n auto* fz=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(2));\n for(uint32_t n=0;n<1024;++n)fz[n]=0;\n for(uint32_t tile=start;tile<start+count;++tile) {\n  // DRAM reads retain matching64-byte offsets; source half-row blocks include padding.\n  noc.async_read(q,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias});\n  noc.async_read(q,scratch,64,{.page_id=tile,.offset_bytes=512},{.offset_bytes=bias+64});\n  for(uint32_t row=0;row<3;++row)\n   noc.async_read(state,scratch,64,{.page_id=row,.offset_bytes=tile*64},{.offset_bytes=bias+128+row*64});\n  noc.async_read(w0,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias+384});\n  noc.async_read(w0,scratch,64,{.page_id=tile,.offset_bytes=1024},{.offset_bytes=bias+448});\n  noc.async_read(w1,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias+512});\n  noc.async_read(w1,scratch,64,{.page_id=tile,.offset_bytes=1024},{.offset_bytes=bias+576});\n  noc.async_read(w2,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias+640});\n  noc.async_read(w2,scratch,64,{.page_id=tile,.offset_bytes=1024},{.offset_bytes=bias+704});\n  noc.async_read(w3,scratch,64,{.page_id=tile,.offset_bytes=0},{.offset_bytes=bias+768});\n  noc.async_read(w3,scratch,64,{.page_id=tile,.offset_bytes=1024},{.offset_bytes=bias+832});\n  noc.async_read_barrier();\n  for(uint32_t c=0;c<32;++c) {\n   const uint16_t code=raw[c<16?c:32+c-16];\n   // Selected native TILE->RM transport flushes exponent-zero BF16 codes to+0.\n   raw[160+c]=(code&0x7f80u)?code:0u;\n  }\n  for(uint32_t j=0;j<4;++j) {\n   if(j==0)first.reserve_back(1);else input.reserve_back(1);weight.reserve_back(1);\n   auto* in=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));\n   auto* wt=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n   auto* bf=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(2));\n   for(uint32_t c=0;c<32;++c) {\n    const uint32_t code=raw[(j<3?64+j*32:160)+c];\n    // Current folded RM->FP32 TILE mapping: all finite exponent-zero codes map to+0.\n    if(j==0)bf[face(c)]=(code&0x7f80u)?code:0u;\n    else in[face(c)]=(code&0x7f80u)?code<<16:0u;\n    wt[face(c)]=wide[96+j*32+c];\n   }\n   if(j==0)first.push_back(1);else input.push_back(1);weight.push_back(1);\n  }\n  // All source history words were fetched before any in-place carry writes.\n  noc.async_write(scratch,state,64,{.offset_bytes=bias+192},{.page_id=0,.offset_bytes=tile*64});\n  noc.async_write(scratch,state,64,{.offset_bytes=bias+256},{.page_id=1,.offset_bytes=tile*64});\n  noc.async_write(scratch,state,64,{.offset_bytes=bias+320},{.page_id=2,.offset_bytes=tile*64});\n  noc.async_write_barrier();\n }\n}\n'
DECODE_CONV_COMPUTE = '\n#include "api/compute/common.h"\n#include "api/compute/compute_kernel_api.h"\n#include "api/compute/compute_kernel_hw_startup.h"\n#include "api/compute/experimental/reg_api.h"\n#include "api/compute/pack.h"\n#include "api/compute/tile_move_copy.h"\n#include "api/compute/reconfig_data_format.h"\n#include "api/compute/eltwise_binary_sfpu.h"\n#include "api/compute/eltwise_unary/addcmul.h"\n#include "api/compute/eltwise_unary/typecast.h"\n#include "api/dataflow/circular_buffer.h"\nvoid kernel_main() {\n constexpr bool debug=get_compile_time_arg_val(0)!=0;\n const uint32_t count=get_arg_val<uint32_t>(0);\n CircularBuffer input(0),weight(1),first(2),feedback(24),output(16),partials(17);\n compute_kernel_hw_startup(0,24);copy_init(0);\n for(uint32_t tile=0;tile<count;++tile) {\n  for(uint32_t j=0;j<4;++j) {\n   tile_regs_acquire_math_clear();weight.wait_front(1);\n   if(j==0){\n    first.wait_front(1);\n    reconfig_data_format_srca(0,2);copy_init(2);copy_tile(2,0,1);\n    reconfig_data_format_srca(2,1);copy_init(1);copy_tile(1,0,2);\n    reconfig_data_format_srca(1,0);copy_init(0);\n   }else{input.wait_front(1);copy_tile(0,0,1);copy_tile(1,0,2);}\n   if(j==0){mul_binary_tile_init();mul_binary_tile<true>(1,2,0);}\n   else {\n    feedback.wait_front(1);copy_tile(24,0,0);feedback.pop_front(1);\n    addcmul_tile_init();addcmul_tile<DataFormat::Float32,true>(0,1,2,0,0x3f800000);\n   }\n   if(j==0)first.pop_front(1);else input.pop_front(1);weight.pop_front(1);tile_regs_commit();tile_regs_wait();\n   pack_reconfig_data_format(24);feedback.reserve_back(1);pack_tile(0,24);feedback.push_back(1);\n   if constexpr(debug){partials.reserve_back(1);pack_tile(0,17);partials.push_back(1);}\n   tile_regs_release_math_clear();\n  }\n  // Preserve final accumulator\'s native FP32 pack/unpack before unary chain.\n  tile_regs_acquire_math_clear();feedback.wait_front(1);copy_tile(24,0,0);feedback.pop_front(1);\n  typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();\n  typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b),true>(0);\n  typecast_tile<static_cast<uint32_t>(DataFormat::Float16_b),static_cast<uint32_t>(DataFormat::Float32),true>(0);\n  silu_tile_init();silu_tile<true>(0);\n  typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b),true>(0);\n  tile_regs_commit();tile_regs_wait();pack_reconfig_data_format(16);\n  output.reserve_back(1);pack_tile(0,16);output.push_back(1);tile_regs_release_math_clear();\n }\n}\n'
DECODE_CONV_WRITER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main(){\n constexpr bool debug=get_compile_time_arg_val(0)!=0;\n constexpr uint32_t po=get_compile_time_arg_val(1);\n const auto output=TensorAccessor(TensorAccessorArgs<2>(),get_arg_val<uint32_t>(0));\n const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);\n DataflowBuffer out(16),partials(17);Noc noc;\n for(uint32_t tile=start;tile<start+count;++tile){\n  if constexpr(debug){\n   const auto p=TensorAccessor(TensorAccessorArgs<po>(),get_arg_val<uint32_t>(1));\n   for(uint32_t j=0;j<4;++j){partials.wait_front(1);noc.async_write(partials,p,4096,{}, {.page_id=j*256+tile});noc.async_write_barrier();partials.pop_front(1);}\n  }\n  out.wait_front(1);noc.async_write(out,output,2048,{}, {.page_id=tile});noc.async_write_barrier();out.pop_front(1);\n }\n}\n'


class DecodeConvolution:
    def __init__(self, t, mesh, taps, max_cores=64):
        self.t, self.mesh, self.taps, self.max_cores = (t, mesh, taps, max_cores)

    def __call__(self, qkv, state, debug=False):
        t = self.t
        assert tuple(qkv.shape) == (1, 1, 8192) and qkv.dtype == t.bfloat16 and (qkv.layout == t.TILE_LAYOUT)
        assert tuple(state.shape) == (1, 3, 8192) and state.dtype == t.bfloat16 and (state.layout == t.ROW_MAJOR_LAYOUT)
        assert state.memory_config() == t.DRAM_MEMORY_CONFIG
        assert qkv.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        assert all(
            (tuple(w.shape) == (1, 1, 8192) and w.dtype == t.float32 and (w.layout == t.TILE_LAYOUT) for w in self.taps)
        )
        grid = self.mesh.compute_with_storage_grid_size()
        active = min(256, grid.x * grid.y, self.max_cores)
        partitions = []
        start = 0
        for h in range(active):
            count = 256 // active + (h < 256 % active)
            partitions.append((h % grid.x, h // grid.x, start, count))
            start += count
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        out = t.empty(
            (1, 1, 8192), device=self.mesh, dtype=t.bfloat16, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        partial = (
            t.empty(
                (4, 1, 1, 8192),
                device=self.mesh,
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                memory_config=t.DRAM_MEMORY_CONFIG,
            )
            if debug
            else None
        )
        rd, cr, wr = (t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs())
        for x, y, start, count in partitions:
            rd[x][y] = [
                qkv.buffer_address(),
                state.buffer_address(),
                *[w.buffer_address() for w in self.taps],
                start,
                count,
            ]
            cr[x][y] = [count]
            wr[x][y] = [out.buffer_address(), partial.buffer_address() if debug else 0, start, count]
        accessors = [list(t.TensorAccessorArgs(x).get_compile_time_args()) for x in (qkv, state, *self.taps)]
        offsets = []
        cursor = 5
        for a in accessors:
            offsets.append(cursor)
            cursor += len(a)
        rdargs = offsets[1:] + sum(accessors, [])
        oa = list(t.TensorAccessorArgs(out).get_compile_time_args())
        pa = list(t.TensorAccessorArgs(partial if debug else out).get_compile_time_args())
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2, 24):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

        def kernel(source, config, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=config,
                compile_time_args=args,
                runtime_args=runtime,
            )

        cb_specs = [
            (0, 2, 4096, t.float32),
            (1, 2, 4096, t.float32),
            (2, 1, 2048, t.bfloat16),
            (3, 1, 4096, t.float32),
            (24, 2, 4096, t.float32),
            (16, 1, 2048, t.bfloat16),
        ]
        if debug:
            cb_specs.append((17, 1, 4096, t.float32))
        program = t.ProgramDescriptor(
            kernels=[
                kernel(DECODE_CONV_READER, t.ReaderConfigDescriptor(), rdargs, rd),
                kernel(DECODE_CONV_COMPUTE, cfg, [int(debug)], cr),
                kernel(DECODE_CONV_WRITER, t.WriterConfigDescriptor(), [int(debug), 2 + len(oa)] + oa + pa, wr),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=pages * size,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(i, dtype, size)],
                )
                for i, pages, size, dtype in cb_specs
            ],
        )
        t.generic_op([qkv, state, *self.taps, out] + ([partial] if debug else []), program)
        return (out, partial)


NORM_TERMINAL_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n const auto input=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));\n const uint32_t start=get_arg_val<uint32_t>(1), count=get_arg_val<uint32_t>(2);\n Noc noc;DataflowBuffer in(0);\n for(uint32_t row=start;row<start+count;++row) {\n  in.reserve_back(1);\n  noc.async_read(input,in,4096,{.page_id=row},{});\n  noc.async_read_barrier();in.push_back(1);\n }\n}\n'
NORM_TERMINAL_COMPUTE = '\n#include "api/compute/compute_kernel_api.h"\n#include "api/compute/common.h"\n#include "api/compute/compute_kernel_hw_startup.h"\n#include "api/compute/cumsum.h"\n#include "api/compute/pack.h"\n#include "api/compute/tile_move_copy.h"\n#include "api/dataflow/dataflow_buffer.h"\nvoid kernel_main() {\n const uint32_t count=get_arg_val<uint32_t>(0);\n DataflowBuffer groups(0), transposed_lanes(1), height_scan(16), width_scan(17);\n compute_kernel_hw_startup(0,16);\n for(uint32_t row=0;row<count;++row) {\n  groups.wait_front(1);tile_regs_acquire();\n  copy_init(0);copy_tile(0,0,0);\n  cumsum_tile_init<CumsumMode::Default>();\n  cumsum_tile<CumsumMode::Default>(0,true);\n  tile_regs_commit();groups.pop_front(1);tile_regs_wait();\n  height_scan.reserve_back(1);pack_tile(0,16);height_scan.push_back(1);\n  tile_regs_release();\n  // BRISC copies the exact packed row3 bits into a transposed lane tile.\n  // Native width scan also transposes before scanning; no arithmetic is moved.\n  transposed_lanes.wait_front(1);tile_regs_acquire();\n  copy_init(1);copy_tile(1,0,0);\n  cumsum_tile_init<CumsumMode::Default>();\n  cumsum_tile<CumsumMode::Default>(0,true);\n  tile_regs_commit();transposed_lanes.pop_front(1);tile_regs_wait();\n  width_scan.reserve_back(1);pack_tile(0,17);width_scan.push_back(1);\n  tile_regs_release();\n }\n}\n'
NORM_TERMINAL_WRITER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n constexpr bool debug=get_compile_time_arg_val(0)!=0;\n auto output_args=TensorAccessorArgs<1,0>();\n auto lanes_args=TensorAccessorArgs<output_args.next_compile_time_args_offset(),output_args.next_common_runtime_args_offset()>();\n const auto output=TensorAccessor(output_args,get_arg_val<uint32_t>(0));\n const auto lanes_output=TensorAccessor(lanes_args,get_arg_val<uint32_t>(1));\n const uint32_t start=get_arg_val<uint32_t>(2), count=get_arg_val<uint32_t>(3);\n Noc noc;DataflowBuffer height_scan(16), transposed_lanes(1), width_scan(17), scalar(18), lanes(19);\n for(uint32_t row=start;row<start+count;++row) {\n  height_scan.wait_front(1);transposed_lanes.reserve_back(1);\n  const auto* h=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(16));\n  auto* t=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n  for(uint32_t j=0;j<1024;++j)t[j]=0;\n  // FP32 physical face0 row3 colj -> transposed physical rowj col0.\n  // Later padded columns/rows cannot affect either logical terminal.\n  for(uint32_t j=0;j<8;++j)t[j*16]=h[3*16+j];\n  if constexpr(debug) {\n   lanes.reserve_back(1);\n   auto* l=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(19));\n   for(uint32_t j=0;j<1024;++j)l[j]=0;\n   for(uint32_t j=0;j<8;++j)l[j]=h[3*16+j];\n   lanes.push_back(1);lanes.wait_front(1);\n   noc.async_write(lanes,lanes_output,4096,{}, {.page_id=row});\n   noc.async_write_barrier();lanes.pop_front(1);\n  }\n  transposed_lanes.push_back(1);height_scan.pop_front(1);\n  width_scan.wait_front(1);scalar.reserve_back(1);\n  const auto* w=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(17));\n  auto* o=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(18));\n  for(uint32_t j=0;j<1024;++j)o[j]=0;\n  o[0]=w[7*16];\n  scalar.push_back(1);scalar.wait_front(1);\n  noc.async_write(scalar,output,4096,{}, {.page_id=row});\n  noc.async_write_barrier();scalar.pop_front(1);width_scan.pop_front(1);\n }\n}\n'


class NormTerminalScan:
    """Two native Default FP32 scans and exact packed-bit selection in one dispatch."""

    def __init__(self, ttnn, mesh):
        self.t, self.mesh = (ttnn, mesh)
        self.grid = mesh.compute_with_storage_grid_size()

    def __call__(self, groups, *, debug=False):
        t = self.t
        shape = tuple(groups.shape)
        assert len(shape) == 4 and shape[1:] == (1, 4, 8)
        assert groups.dtype == t.float32 and groups.layout == t.TILE_LAYOUT
        assert groups.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        rows = shape[0]
        active = min(rows, self.grid.x * self.grid.y)
        cores = t.CoreRangeSet(
            [
                t.CoreRange(
                    t.CoreCoord(i % self.grid.x, i // self.grid.x), t.CoreCoord(i % self.grid.x, i // self.grid.x)
                )
                for i in range(active)
            ]
        )
        output = t.empty(
            (rows, 1, 1, 1),
            device=self.mesh,
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=groups.memory_config(),
        )
        lanes = (
            t.empty(
                (rows, 1, 1, 8),
                device=self.mesh,
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                memory_config=groups.memory_config(),
            )
            if debug
            else output
        )
        rd, wr, cr = (t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs())
        start = 0
        for i in range(active):
            count = rows // active + (i < rows % active)
            x, y = (i % self.grid.x, i // self.grid.x)
            rd[x][y] = [groups.buffer_address(), start, count]
            wr[x][y] = [output.buffer_address(), lanes.buffer_address(), start, count]
            cr[x][y] = [count]
            start += count
        assert start == rows
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
        modes = [t.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = t.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes

        def kernel(source, conf, compile_args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=compile_args,
                runtime_args=runtime,
            )

        program = t.ProgramDescriptor(
            kernels=[
                kernel(
                    NORM_TERMINAL_READER,
                    t.ReaderConfigDescriptor(),
                    t.TensorAccessorArgs(groups).get_compile_time_args(),
                    rd,
                ),
                kernel(NORM_TERMINAL_COMPUTE, config, [], cr),
                kernel(
                    NORM_TERMINAL_WRITER,
                    t.WriterConfigDescriptor(),
                    [int(debug)]
                    + t.TensorAccessorArgs(output).get_compile_time_args()
                    + t.TensorAccessorArgs(lanes).get_compile_time_args(),
                    wr,
                ),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=4096 * pages,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(index, t.float32, 4096)],
                )
                for index, pages in [(0, 2), (1, 1), (16, 1), (17, 1), (18, 1), (19, 1)]
            ],
        )
        t.generic_op([groups, output, lanes] if debug else [groups, output], program)
        return (output, lanes) if debug else output


DECAY_LOOKUP_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nconstexpr uint32_t face(uint32_t r,uint32_t c){return (r/16)*512+(c/16)*256+(r%16)*16+c%16;}\nvoid kernel_main(){\n constexpr uint32_t ta=get_compile_time_arg_val(0),oa=get_compile_time_arg_val(1);\n constexpr bool compact=get_compile_time_arg_val(2)!=0;\n constexpr uint32_t lo=get_compile_time_arg_val(3),hi=get_compile_time_arg_val(4);\n const auto input=TensorAccessor(TensorAccessorArgs<5>(),get_arg_val<uint32_t>(0));\n const auto table=TensorAccessor(TensorAccessorArgs<ta>(),get_arg_val<uint32_t>(1));\n const auto output=TensorAccessor(TensorAccessorArgs<oa>(),get_arg_val<uint32_t>(2));\n const uint32_t start=get_arg_val<uint32_t>(3),count=get_arg_val<uint32_t>(4),rows=get_arg_val<uint32_t>(5);\n DataflowBuffer scratch(0);Noc noc;scratch.reserve_back(3);\n const uint32_t bias=(64-(get_write_ptr(0)&63))&63;\n auto* codes=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(0)+bias);\n auto* blocks=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+2048);\n auto* result=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+4096);\n for(uint32_t tile=start;tile<start+count;++tile){\n  noc.async_read(input,scratch,2048,{.page_id=tile},{.offset_bytes=bias});noc.async_read_barrier();\n  for(uint32_t i=0;i<1024;++i)result[i]=0;\n  const uint32_t valid=rows-tile*32<32?rows-tile*32:32;\n  for(uint32_t r=0;r<valid;++r){\n   uint32_t mapped[32];\n   for(uint32_t h=0;h<32;++h){\n    const uint32_t code=codes[face(r,h)],mag=code&32767u;\n    if constexpr(compact){\n     const uint32_t clamped=mag<lo?lo:(mag>hi?hi:mag);\n     mapped[h]=clamped-lo+(mag>32640?mag-32640:0)+((code&32768u)>>2);\n    }else mapped[h]=code;\n    noc.async_read(table,scratch,64,{.page_id=h,.offset_bytes=(mapped[h]>>4)*64},\n      {.offset_bytes=bias+2048+h*64});\n   }\n   noc.async_read_barrier();\n   for(uint32_t h=0;h<32;++h)result[face(r,h)]=blocks[h*16+(mapped[h]&15)];\n  }\n  noc.async_write(scratch,output,4096,{.offset_bytes=bias+4096},{.page_id=tile});noc.async_write_barrier();\n }\n}\n'
DECAY_LOOKUP_EMPTY = "void kernel_main() {}\n"


class DecodeDecayLookup:
    def __init__(self, t, mesh, table, lo=None, hi=None, max_cores=32):
        self.t, self.mesh, self.table = (t, mesh, table)
        self.lo, self.hi, self.max_cores = (lo, hi, max_cores)
        assert tuple(table.shape) in ((32, 16384), (32, 65536))
        assert table.dtype == t.float32 and table.layout == t.ROW_MAJOR_LAYOUT
        assert table.memory_config() == t.DRAM_MEMORY_CONFIG
        assert (lo is not None and hi is not None) == (table.shape[1] == 16384)

    def __call__(self, projection_a):
        t, table = (self.t, self.table)
        assert projection_a.dtype == t.bfloat16 and projection_a.layout == t.TILE_LAYOUT
        assert len(projection_a.shape) == 3 and projection_a.shape[0] == 1 and (projection_a.shape[2] == 32)
        assert projection_a.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        rows = projection_a.shape[1]
        tiles = (rows + 31) // 32
        grid = self.mesh.compute_with_storage_grid_size()
        active = min(tiles, grid.x * grid.y, self.max_cores)
        partitions, start = ([], 0)
        for i in range(active):
            count = tiles // active + (i < tiles % active)
            partitions.append((i % grid.x, i // grid.x, start, count))
            start += count
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        output = t.empty(
            tuple(projection_a.shape),
            device=self.mesh,
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
        ia, ta, oa = [list(t.TensorAccessorArgs(x).get_compile_time_args()) for x in (projection_a, table, output)]
        args = [5 + len(ia), 5 + len(ia) + len(ta), int(self.lo is not None), self.lo or 0, self.hi or 0] + ia + ta + oa
        runtime = t.RuntimeArgs()
        for x, y, start, count in partitions:
            runtime[x][y] = [
                projection_a.buffer_address(),
                table.buffer_address(),
                output.buffer_address(),
                start,
                count,
                rows,
            ]
        program = t.ProgramDescriptor(
            kernels=[
                t.KernelDescriptor(
                    kernel_source=DECAY_LOOKUP_READER,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=t.ReaderConfigDescriptor(),
                    compile_time_args=args,
                    runtime_args=runtime,
                ),
                t.KernelDescriptor(
                    kernel_source=DECAY_LOOKUP_EMPTY,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=t.WriterConfigDescriptor(),
                ),
                t.KernelDescriptor(
                    kernel_source=DECAY_LOOKUP_EMPTY,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=t.ComputeConfigDescriptor(),
                ),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=12288, core_ranges=cores, format_descriptors=[t.CBFormatDescriptor(0, t.float32, 4096)]
                )
            ],
        )
        t.generic_op([projection_a, table, output], program)
        return output


TRANSPOSE_CORRECTED_MM_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nconstexpr uint32_t face_offset(uint32_t r,uint32_t c) {\n return (r/16)*512+(c/16)*256+(r%16)*16+c%16;\n}\nvoid kernel_main() {\n constexpr uint32_t b_accessor_offset=get_compile_time_arg_val(0);\n constexpr bool transpose_a=get_compile_time_arg_val(1)!=0;\n constexpr bool transpose_b=get_compile_time_arg_val(2)!=0;\n const auto at=TensorAccessor(TensorAccessorArgs<3>(),get_arg_val<uint32_t>(0));\n const auto bt=TensorAccessor(TensorAccessorArgs<b_accessor_offset>(),get_arg_val<uint32_t>(1));\n const uint32_t start=get_arg_val<uint32_t>(2), count=get_arg_val<uint32_t>(3);\n const uint32_t M=get_arg_val<uint32_t>(4), N=get_arg_val<uint32_t>(5), K=get_arg_val<uint32_t>(6);\n const uint32_t rank=get_arg_val<uint32_t>(7), Mt=(M+31)/32, Nt=(N+31)/32, Kt=(K+31)/32;\n DataflowBuffer a(0), b(1), zero(2), ascratch(3), bscratch(4);\n Noc noc;\n ascratch.reserve_back(1); bscratch.reserve_back(1);\n auto* sa=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3));\n auto* sb=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4));\n for(uint32_t tile=start;tile<start+count;++tile) {\n  const uint32_t matrix_tile=tile%(Mt*Nt), mr=matrix_tile/Nt, nc=matrix_tile%Nt;\n  uint32_t batch=tile/(Mt*Nt), abatch=0, bbatch=0;\n  for(uint32_t i=rank;i>0;--i) {\n   const uint32_t j=i-1, d=get_arg_val<uint32_t>(8+j), coordinate=batch%d;\n   batch/=d;\n   abatch+=coordinate*get_arg_val<uint32_t>(8+rank+j);\n   bbatch+=coordinate*get_arg_val<uint32_t>(8+2*rank+j);\n  }\n  zero.reserve_back(1);\n  auto* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));\n  for(uint32_t i=0;i<1024;++i) z[i]=0;\n  zero.push_back(1);\n  for(uint32_t k=0;k<K;++k) {\n   if(k%32==0) {\n    noc.async_read(at,ascratch,4096,{.page_id=transpose_a ? (abatch*Kt+k/32)*Mt+mr : (abatch*Mt+mr)*Kt+k/32},{.offset_bytes=0});\n    noc.async_read(bt,bscratch,4096,{.page_id=transpose_b ? (bbatch*Nt+nc)*Kt+k/32 : (bbatch*Kt+k/32)*Nt+nc},{.offset_bytes=0});\n    noc.async_read_barrier();\n   }\n   a.reserve_back(1); b.reserve_back(1);\n   auto* aa=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));\n   auto* bb=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));\n   for(uint32_t r=0;r<32;++r) {\n    const uint32_t av=mr*32+r<M ? sa[transpose_a ? face_offset(k%32,r) : face_offset(r,k%32)] : 0;\n    for(uint32_t c=0;c<32;++c) {\n     const uint32_t offset=face_offset(r,c);\n     aa[offset]=av;\n     bb[offset]=nc*32+c<N ? sb[transpose_b ? face_offset(c,k%32) : face_offset(k%32,c)] : 0;\n    }\n   }\n   a.push_back(1); b.push_back(1);\n  }\n }\n}\n'


class TransposeCorrectedMatmul:
    """Device-only streaming matrix recurrence; arbitrary broadcast leading axes."""

    def __init__(self, t, mesh, max_cores=64):
        self.t, self.mesh, self.max_cores = (t, mesh, max_cores)
        self.source = FUSED_CORRECTED_MM_COMPUTE
        self.grid = mesh.compute_with_storage_grid_size()
        self.last_plan = None

    def __call__(self, a, b, *, transpose_a=False, transpose_b=False):
        t = self.t
        assert a.dtype == b.dtype == t.float32
        assert a.layout == b.layout == t.TILE_LAYOUT
        assert a.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        assert b.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        ashape, bshape = (tuple(a.shape), tuple(b.shape))
        if transpose_a:
            ashape = ashape[:-2] + (ashape[-1], ashape[-2])
        if transpose_b:
            bshape = bshape[:-2] + (bshape[-1], bshape[-2])
        plan = fused_corrected_mm_shape_plan(ashape, bshape)
        partitions = fused_corrected_mm_schedule(plan["total_tiles"], self.grid.x, self.grid.y, self.max_cores)
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        out = t.empty(
            plan["shape"], device=self.mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        rd, wr, cr = (t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs())
        for x, y, start, count in partitions:
            rd[x][y] = [
                a.buffer_address(),
                b.buffer_address(),
                start,
                count,
                plan["M"],
                plan["N"],
                plan["K"],
                len(plan["leading"]),
                *plan["leading"],
                *plan["a_strides"],
                *plan["b_strides"],
            ]
            wr[x][y] = [out.buffer_address(), start, count]
            cr[x][y] = [count]
        ac = list(t.TensorAccessorArgs(a).get_compile_time_args())
        bc = list(t.TensorAccessorArgs(b).get_compile_time_args())
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2, 24):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

        def kernel(source, config, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=config,
                compile_time_args=args,
                runtime_args=runtime,
            )

        program = t.ProgramDescriptor(
            kernels=[
                kernel(
                    TRANSPOSE_CORRECTED_MM_READER,
                    t.ReaderConfigDescriptor(),
                    [3 + len(ac), int(transpose_a), int(transpose_b)] + ac + bc,
                    rd,
                ),
                kernel(self.source, cfg, [plan["K"]], cr),
                kernel(
                    FUSED_CORRECTED_MM_WRITER,
                    t.WriterConfigDescriptor(),
                    t.TensorAccessorArgs(out).get_compile_time_args(),
                    wr,
                ),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(i, t.float32, 4096)],
                )
                for i, pages in ((0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (16, 1), (24, 2))
            ],
        )
        t.generic_op([a, b, out], program)
        self.last_plan = {
            **plan,
            "core_count": len(partitions),
            "tiles_per_core": [p[3] for p in partitions],
            "cb_bytes_per_core": 40960,
        }
        return out


WEIGHTED_NORM_INTERLEAVED_READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/dataflow_buffer.h"\n#include "api/tensor/noc_traits.h"\nvoid kernel_main() {\n constexpr uint32_t sum_offset=get_compile_time_arg_val(0);\n const auto core_input=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));\n const auto sum_input=TensorAccessor(TensorAccessorArgs<sum_offset>(),get_arg_val<uint32_t>(1));\n DataflowBuffer core(0),sums(1);Noc noc;\n core.reserve_back(4);sums.reserve_back(1);\n for(uint32_t tile=0;tile<4;++tile) {\n  noc.async_read(core_input,core,4096,{.page_id=tile},{.offset_bytes=tile*4096});\n }\n noc.async_read(sum_input,sums,4096,{.page_id=0},{.offset_bytes=0});\n noc.async_read_barrier();\n core.push_back(4);sums.push_back(1);\n}\n'
