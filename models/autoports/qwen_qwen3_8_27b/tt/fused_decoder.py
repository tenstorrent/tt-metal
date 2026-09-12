# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused single-mesh Qwen3.8-27B text decoder, with explicit caller-owned request state.

Inputs/outputs are TILE BF16 [batch, sequence, 5120]. Prefill accepts logical
lengths; it executes bounded chunks internally. Full attention uses paged BF16
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


@dataclass
class DecoderState:
    key: object = None
    value: object = None
    recurrent: object = None
    conv: object = None


class FusedDecoder(LightweightModule):
    PAGE_SIZE = 32
    CHUNK_SIZE = 128

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device):
        """Load an HF layer-local state dict (keys match Qwen3_5DecoderLayer)."""
        import torch

        self = cls()
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
            raise ValueError("Fused decoder requires a single-device mesh")
        if (hf_config.hidden_size, hf_config.intermediate_size) != (5120, 17408):
            raise ValueError("Expected the real Qwen3.8-27B text config")

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
                self.weights[name] = upload(tensor)
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
            self.weights["self_attn.qkvg.weight"] = upload(packed.T)
            for name in ("q", "k", "v"):
                ttnn.deallocate(self.weights.pop(f"self_attn.{name}_proj.weight"))
        if self.kind == "linear_attention":
            names = ("qkv", "z", "b", "a")
            pieces = []
            for name in names:
                w = state_dict[f"linear_attn.in_proj_{name}.weight"]
                pieces.append(torch.nn.functional.pad(w, (0, 0, 0, (-w.shape[0]) % 32)))
            self.weights["linear_attn.packed.weight"] = upload(torch.cat(pieces, dim=0).T)
            for name in names:
                ttnn.deallocate(self.weights.pop(f"linear_attn.in_proj_{name}.weight"))
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
            return DecoderState(key=zeros(shape, ttnn.bfloat16), value=zeros(shape, ttnn.bfloat16))
        c = self.config
        width = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
        return DecoderState(
            recurrent=zeros(
                [batch_size, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim], ttnn.float32
            ),
            conv=zeros([batch_size, 3, width], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        )

    def _linear(self, x, name):
        return ttnn.linear(
            x, self.weights[name + ".weight"], compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _norm(self, x, name):
        return ttnn.rms_norm(
            x,
            weight=self.weights[name + ".weight"],
            epsilon=self.eps,
            compute_kernel_config=self.ckc if self.kind == "linear_attention" else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _finish(self, x, attention):
        h = ttnn.add(x, attention)
        n = self._norm(h, "post_attention_layernorm")
        grid = self.device.compute_with_storage_grid_size()
        gate = ttnn.linear(
            n,
            self.weights["mlp.gate_proj.weight"],
            activation="silu",
            core_grid=ttnn.CoreGrid(x=grid.x, y=grid.y),
            compute_kernel_config=self.ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = self._linear(n, "mlp.up_proj")
        return ttnn.add(h, self._linear(ttnn.mul(gate, up), "mlp.down_proj"))

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
        result = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            state.key,
            state.value,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=c.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=32
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
        outputs = []
        for user in range(b):
            table = page_table[user : user + 1, :]
            chunk_table = table[:, start_pos // self.PAGE_SIZE : (start_pos + t + self.PAGE_SIZE - 1) // self.PAGE_SIZE]
            for cache, update in ((state.key, k), (state.value, v)):
                ttnn.experimental.paged_fill_cache(cache, update[user : user + 1, :, :, :], chunk_table, batch_idx=0)
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
                        q_chunk_size=32,
                        k_chunk_size=32,
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
            outputs.append(self._finish(chunk, attention))
            offset += count
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)
