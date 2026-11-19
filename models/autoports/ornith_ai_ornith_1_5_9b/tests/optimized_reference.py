# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ornith Qwen3.5 text decoder, single-device optimized implementation.

Inputs/outputs are device BF16 [1, 1, logical_tokens, 4096]. Each request owns
its state; full attention uses 32-token paged KV, linear attention uses FP32
recurrent state and three convolution history rows. Setup performs conversion;
forward methods contain only TTNN operations and shape/chunk planning.
"""

import math
from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class DecoderState:
    conv: object = None
    recurrent: object = None
    key: object = None
    value: object = None


class OptimizedDecoder(LightweightModule):
    # Experiment subclasses can compare the same fused graph without DRAM sharding.
    use_dram_decode = True
    attention_dtype = ttnn.bfloat8_b
    mlp_dtype = ttnn.bfloat8_b
    attention_fidelity = ttnn.MathFidelity.HiFi2
    mlp_fidelity = ttnn.MathFidelity.HiFi2
    decode_readers = 4
    decode_cores = 64
    decode_block = 1
    role_configs = {}

    def weight_dtype(self, name):
        return self.mlp_dtype if name.startswith(("mlp.", "gate_up", "packed_mlp")) else self.attention_dtype

    def matmul_compute(self, name):
        return self.mlp_compute if name.startswith(("mlp.", "gate_up", "packed_mlp")) else self.attention_compute

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, chunk_size=1024):
        import torch

        self = cls()
        c = getattr(hf_config, "text_config", hf_config)
        assert (c.hidden_size, c.intermediate_size, c.head_dim) == (4096, 12288, 256)
        assert chunk_size > 0 and chunk_size % 32 == 0
        self.config, self.device, self.chunk_size = c, mesh_device, chunk_size
        self.kind = c.layer_types[layer_idx]
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.attention_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=self.attention_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.mlp_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=self.mlp_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        prefix = f"model.language_model.layers.{layer_idx}."
        sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not sd:
            sd = state_dict

        def tensor(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(x.contiguous(), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        self.weights = {}
        host_weights = {}
        for name, value in sd.items():
            if value.ndim == 2:
                host_weights[name] = value.T
                self.weights[name] = tensor(value.T, self.weight_dtype(name))
            elif name.endswith("norm.weight"):
                offset = not name.endswith("linear_attn.norm.weight")
                self.weights[name] = tensor((value.float() + int(offset)).reshape(1, 1, 1, -1))
        projection_names = (
            ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"]
            if self.kind == "linear_attention"
            else ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        )
        projection_weights = [sd[name + ".weight"] for name in projection_names]
        if self.kind == "full_attention":
            projection_weights[0] = (
                projection_weights[0].reshape(16, 2, 256, 4096).permute(1, 0, 2, 3).reshape(8192, 4096)
            )
        host_weights["packed_input.weight"] = torch.cat(projection_weights, dim=0).T
        self.weights["packed_input.weight"] = tensor(host_weights["packed_input.weight"], self.attention_dtype)
        for name in projection_names:
            del self.weights[name + ".weight"]
        mlp_weight = torch.cat([sd["mlp.gate_proj.weight"], sd["mlp.up_proj.weight"]], dim=0).T
        host_weights["gate_up.weight"] = mlp_weight
        self.weights["gate_up.weight"] = tensor(mlp_weight, self.mlp_dtype)
        mlp_weight = mlp_weight.reshape(4096, 2, 384, 32).permute(0, 2, 1, 3).reshape(4096, 24576)
        self.weights["packed_mlp.weight"] = tensor(mlp_weight, self.mlp_dtype)
        del self.weights["mlp.gate_proj.weight"], self.weights["mlp.up_proj.weight"]
        if self.kind == "linear_attention":
            self.taps = [tensor(sd["linear_attn.conv1d.weight"][:, 0, i].reshape(1, 1, 1, -1)) for i in range(4)]
            self.decay = tensor(-sd["linear_attn.A_log"].float().exp().reshape(1, 1, 32), ttnn.float32)
            ii = torch.arange(32)[:, None] < 16
            jj = torch.arange(32)[None, :] < 16
            const = [
                torch.eye(32),
                torch.ones(32, 32).tril(),
                torch.ones(32, 32),
                torch.cat([(ii & jj).float(), (~ii & ~jj).float(), (~ii & jj).float()], dim=1),
            ]
            self.gdn_constants = dict(
                zip(("eye", "tril", "ones", "masks"), [tensor(v.reshape(1, 1, *v.shape), ttnn.float32) for v in const])
            )
            self.conv_compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            self.conv_start = ttnn.from_torch(
                torch.tensor([0], dtype=torch.int32),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self.dt_bias = tensor(sd["linear_attn.dt_bias"].float().reshape(1, 1, 32), ttnn.float32)
        else:
            inv = 1.0 / (c.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64))
            angle = torch.arange(c.max_position_embeddings).float()[:, None] * inv[None, :]
            self.cos = tensor(torch.cat([angle, angle], -1).cos())
            self.sin = tensor(torch.cat([angle, angle], -1).sin())
            self.rope_decode = ttnn.to_layout(ttnn.concat([self.cos, self.sin], dim=-1), ttnn.ROW_MAJOR_LAYOUT)
            self.positions = ttnn.from_torch(
                torch.arange(c.max_position_embeddings, dtype=torch.int32),
                device=mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self.shard = ttnn.create_sharded_memory_config(
                (32, 256), ttnn.CoreGrid(x=1, y=1), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
        self.dram_weights, self.dram_layouts = {}, {}
        if self.use_dram_decode:
            self._prepare_decode_projections(host_weights)
            del self.weights["gate_up.weight"]  # Prefill consumes tile-interleaved packed_mlp.
        return self

    def _prepare_decode_projections(self, host_weights):
        """Prepare the measured BF16/HiFi4 packed projections once, before capture."""
        banks = self.device.dram_grid_size().x
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        names = [
            "packed_input",
            "gate_up",
            "mlp.down_proj",
            "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj",
        ]
        for name in names:
            role = self.role_configs.get(name, {})
            cores = role.get("cores", self.decode_cores)
            readers = role.get("readers", self.decode_readers)
            block = role.get("block", self.decode_block)
            weight = host_weights[name + ".weight"]
            k, n = weight.shape
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.dram_weights[name] = ttnn.from_torch(
                weight.contiguous(),
                device=self.device,
                dtype=self.weight_dtype(name),
                layout=ttnn.TILE_LAYOUT,
                memory_config=mc,
            )

            def layout(width):
                sw = math.ceil(width / (32 * cores))
                nc = math.ceil(width / (32 * sw))
                core_range = ttnn.num_cores_to_corerangeset(nc, self.device.compute_with_storage_grid_size(), True)
                return (
                    ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(core_range, (32, sw * 32), ttnn.ShardOrientation.ROW_MAJOR),
                    ),
                    sw,
                )

            im, _ = layout(k)
            om, ow = layout(n)
            cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=ow,
                num_workers_per_dram_bank=readers,
                use_multicore_pipeline=readers > 1,
            )
            self.dram_layouts[name] = im, om, cfg

    def allocate_state(self, *, context=None):
        context = context or self.config.max_position_embeddings
        assert 0 < context <= self.config.max_position_embeddings and context % 32 == 0

        def zero(shape, dtype=ttnn.bfloat16):
            return ttnn.zeros(
                shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        if self.kind == "linear_attention":
            return DecoderState(conv=zero((1, 1, 3, 8192)), recurrent=zero((1, 32, 128, 128), ttnn.float32))
        return DecoderState(key=zero((context // 32, 4, 32, 256)), value=zero((context // 32, 4, 32, 256)))

    def linear(self, x, name):
        if x.shape[2] == 1 and name in self.dram_weights:
            im, om, cfg = self.dram_layouts[name]
            out = ttnn.linear(
                ttnn.to_memory_config(x, im),
                self.dram_weights[name],
                program_config=cfg,
                compute_kernel_config=self.matmul_compute(name),
                memory_config=om,
            )
            return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            x,
            self.weights[name + ".weight"],
            compute_kernel_config=self.matmul_compute(name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def norm(self, x, name):
        return ttnn.rms_norm(
            x,
            weight=self.weights[name + ".weight"],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
        )

    def finish(self, x, mixed):
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        if x.shape[2] == 1:
            gu = self.linear(h, "gate_up")
            h = ttnn.multiply(
                gu[:, :, :, :12288], gu[:, :, :, 12288:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
        else:
            h = ttnn.experimental.minimal_matmul(
                h,
                self.weights["packed_mlp.weight"],
                fuse_swiglu=True,
                compute_kernel_config=self.mlp_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.add(residual, self.linear(h, "mlp.down_proj"))

    def linear_attention(self, x, state, *, decode):
        n = x.shape[2]
        packed = self.linear(x, "packed_input")
        z = packed[:, :, :, 8192:12288]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 12288:12320]), (1, n, 32))
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 12320:12352], (1, n, 32)), ttnn.float32)
        g = ttnn.multiply(
            ttnn.add(a, self.dt_bias),
            self.decay,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        old_history = ttnn.reshape(ttnn.to_layout(state.conv, ttnn.ROW_MAJOR_LAYOUT), (1, 3, 8192))
        padded = packed
        if n % 32:
            padded = ttnn.pad(packed, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            ttnn.reshape(padded, (1, padded.shape[2], 12352)),
            old_history,
            *self.taps,
            2048,
            2048,
            4096,
            actual_start=self.conv_start,
            predecessor_carry=old_history,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
            compute_kernel_config=self.conv_compute,
        )
        if n >= 3:
            ttnn.copy(packed[:, :, n - 3 : n, :8192], state.conv)
        else:
            history = ttnn.concat([state.conv, packed[:, :, :, :8192]], dim=2)
            ttnn.copy(history[:, :, n : n + 3, :], state.conv)

        if n % 32:
            # Inactive tail rows leave recurrence unchanged: exp(g)=1 and beta=0.
            g = ttnn.pad(g, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
            beta = ttnn.pad(beta, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        out, new_state = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state.recurrent,
            output_final_state=True,
            compute_kernel_config=self.compute,
            chunk_size=32,
            num_key_heads=16,
            output_head_major=True,
            **self.gdn_constants,
        )
        ttnn.copy(new_state, state.recurrent)
        out = ttnn.typecast(out, ttnn.bfloat16)
        out = self.norm(out, "linear_attn.norm")
        out = ttnn.transformer.concatenate_heads(ttnn.reshape(out, (1, 32, padded.shape[2], 128)))
        out = ttnn.reshape(out, (1, 1, padded.shape[2], 4096))[:, :, :n, :]
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return self.linear(out, "linear_attn.out_proj")

    def rope(self, x, cos, sin):
        rot = x[:, :, :, :64]
        rotated = ttnn.experimental.rotary_embedding(
            rot,
            cos,
            sin,
            token_index=0 if cos.shape[2] == 1 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )
        return ttnn.concat([rotated[:, :, : x.shape[2], :], x[:, :, :, 64:]], dim=-1)

    def full_attention(self, x, state, *, page_table, position, decode):
        n = x.shape[2]
        packed = self.linear(x, "packed_input")
        if decode:
            q = self.norm(ttnn.reshape(packed[:, :, :, :4096], (1, n, 16, 256)), "self_attn.q_norm")
            gate = packed[:, :, :, 4096:8192]
            k = self.norm(ttnn.reshape(packed[:, :, :, 8192:9216], (1, n, 4, 256)), "self_attn.k_norm")
            v = ttnn.reshape(packed[:, :, :, 9216:10240], (1, n, 4, 256))
        else:
            qg, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn.reshape(packed, (1, n, 10240)),
                num_heads=32,
                num_kv_heads=4,
                transpose_key=False,
            )
            q = self.norm(qg[:, :16, :, :], "self_attn.q_norm")
            k = self.norm(k, "self_attn.k_norm")
            gate = ttnn.reshape(ttnn.permute(qg[:, 16:32, :, :], (0, 2, 1, 3)), (1, 1, n, 4096))
        if decode:
            rope_position = ttnn.typecast(position, ttnn.uint32)
            rope_values = ttnn.reshape(
                ttnn.embedding(rope_position, self.rope_decode, layout=ttnn.TILE_LAYOUT), (1, 1, 1, 128)
            )
            cos, sin = rope_values[:, :, :, :64], rope_values[:, :, :, 64:]
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            q = ttnn.to_memory_config(q, self.shard)
            k = ttnn.to_memory_config(k, self.shard)
            v = ttnn.to_memory_config(v, self.shard)
            ttnn.experimental.paged_update_cache(state.key, k, update_idxs_tensor=position, page_table=page_table)
            ttnn.experimental.paged_update_cache(state.value, v, update_idxs_tensor=position, page_table=page_table)
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                state.key,
                state.value,
                page_table,
                cur_pos_tensor=position,
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=32
                ),
                compute_kernel_config=self.compute,
            )
            out = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(out, self.shard), num_heads=16)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)[:, :, :1, :]
        else:
            cos = ttnn.reshape(self.cos[position : position + n, :], (1, 1, n, 64))
            sin = ttnn.reshape(self.sin[position : position + n, :], (1, 1, n, 64))
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            local_table = page_table[:, position // 32 : (position + n) // 32]
            ttnn.experimental.paged_fill_cache(state.key, k, local_table)
            ttnn.experimental.paged_fill_cache(state.value, v, local_table)
            attention_table = page_table
            if page_table.shape[1] % 8:
                attention_table = ttnn.pad(page_table, [(0, 0), (0, 8 - page_table.shape[1] % 8)], 0)
            out = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                state.key,
                state.value,
                attention_table,
                chunk_start_idx_tensor=self.positions[position : position + 1],
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=32
                ),
                compute_kernel_config=self.compute,
            )
            out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, n, 4096))
        return self.linear(
            ttnn.multiply(gate, out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]), "self_attn.o_proj"
        )

    def decode_forward(self, x, *, state, page_table=None, current_position=None):
        """One token per state; current_position is a persistent int32 device tensor.

        The caller owns position advancement and state restoration at request boundaries.
        This function is the entire trace-captured layer pass.
        """
        h = self.norm(x, "input_layernorm")
        mixed = (
            self.linear_attention(h, state, decode=True)
            if self.kind == "linear_attention"
            else self.full_attention(h, state, page_table=page_table, position=current_position, decode=True)
        )
        return self.finish(x, mixed)

    def prefill_forward(self, x, *, state, start_position=0, page_table=None):
        """Consume a logical [1,1,S,4096] segment and update its request state.

        For continuation the caller supplies state from exactly start_position.
        Padded rows are never part of the logical recurrent sequence.
        """
        n = x.shape[2]
        assert 0 < n and start_position + n <= self.config.max_position_embeddings
        outputs = []
        offset = 0
        while offset < n:
            absolute = start_position + offset
            if self.kind == "full_attention" and absolute % 32:
                chunk = x[:, :, offset : offset + 1, :]
                outputs.append(
                    self.decode_forward(
                        chunk,
                        state=state,
                        page_table=page_table,
                        current_position=self.positions[absolute : absolute + 1],
                    )
                )
                offset += 1
                continue
            count = min(n - offset, self.chunk_size)
            chunk = x[:, :, offset : offset + count, :]
            if self.kind == "full_attention" and count % 32:
                chunk = ttnn.pad(chunk, [(0, 0), (0, 0), (0, 32 - count % 32), (0, 0)], 0.0)
            h = self.norm(chunk, "input_layernorm")
            mixed = (
                self.linear_attention(h, state, decode=False)
                if self.kind == "linear_attention"
                else self.full_attention(h, state, page_table=page_table, position=absolute, decode=False)
            )
            outputs.append(self.finish(chunk, mixed)[:, :, :count, :])
            offset += count
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def prefill_batch(self, rows, *, states, start_positions, page_tables):
        """Ragged functional baseline: independent device rows, starts and page maps."""
        assert len(rows) == len(states) == len(start_positions) == len(page_tables)
        return [
            self.prefill_forward(x, state=st, start_position=start, page_table=pt)
            for x, st, start, pt in zip(rows, states, start_positions, page_tables)
        ]

    def decode_batch(self, rows, *, states, current_positions, page_tables):
        """Fixed row-count traceable baseline; rows may use distinct current positions."""
        assert len(rows) == len(states) == len(current_positions) == len(page_tables)
        return [
            self.decode_forward(x, state=st, current_position=pos, page_table=pt)
            for x, st, pos, pt in zip(rows, states, current_positions, page_tables)
        ]
