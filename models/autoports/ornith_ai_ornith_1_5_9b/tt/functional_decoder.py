# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ornith Qwen3.5 text decoder, single-device functional implementation.

Inputs/outputs are device BF16 [1, 1, logical_tokens, 4096]. Each request owns
its state; full attention uses 32-token paged KV, linear attention uses FP32
recurrent state and three convolution history rows. Setup performs conversion;
forward methods contain only TTNN operations and shape/chunk planning.
"""

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


def reshape(x, shape):
    return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT)


@dataclass
class DecoderState:
    conv: object = None
    recurrent: object = None
    key: object = None
    value: object = None


class FunctionalDecoder(LightweightModule):
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
        prefix = f"model.language_model.layers.{layer_idx}."
        sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not sd:
            sd = state_dict

        def tensor(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(x.contiguous(), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        self.weights = {}
        for name, value in sd.items():
            if value.ndim == 2:
                self.weights[name] = tensor(value.T)
            elif name.endswith("norm.weight"):
                offset = not name.endswith("linear_attn.norm.weight")
                self.weights[name] = tensor((value.float() + int(offset)).reshape(1, 1, 1, -1))
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
            self.dt_bias = tensor(sd["linear_attn.dt_bias"].float().reshape(1, 1, 32), ttnn.float32)
        else:
            inv = 1.0 / (c.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64))
            angle = torch.arange(c.max_position_embeddings).float()[:, None] * inv[None, :]
            self.cos = tensor(torch.cat([angle, angle], -1).cos())
            self.sin = tensor(torch.cat([angle, angle], -1).sin())
            self.positions = ttnn.from_torch(
                torch.arange(c.max_position_embeddings, dtype=torch.int32),
                device=mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self.shard = ttnn.create_sharded_memory_config(
                (32, 256), ttnn.CoreGrid(x=1, y=1), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
        return self

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
        return ttnn.linear(
            x, self.weights[name + ".weight"], compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
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
        h = ttnn.multiply(ttnn.silu(self.linear(h, "mlp.gate_proj")), self.linear(h, "mlp.up_proj"))
        return ttnn.add(residual, self.linear(h, "mlp.down_proj"))

    def linear_attention(self, x, state, *, decode):
        n = x.shape[2]
        z = reshape(self.linear(x, "linear_attn.in_proj_z"), (1, n, 32, 128))
        beta = reshape(ttnn.sigmoid(self.linear(x, "linear_attn.in_proj_b")), (1, n, 32))
        a = ttnn.typecast(reshape(self.linear(x, "linear_attn.in_proj_a"), (1, n, 32)), ttnn.float32)
        g = ttnn.multiply(self.decay, ttnn.softplus(ttnn.add(a, self.dt_bias)))
        qkv = self.linear(x, "linear_attn.in_proj_qkv")
        history = ttnn.concat([state.conv, qkv], dim=2)
        conv = ttnn.multiply(ttnn.typecast(history[:, :, :n, :], ttnn.float32), self.taps[0])
        for tap in range(1, 4):
            conv = ttnn.add(
                conv, ttnn.multiply(ttnn.typecast(history[:, :, tap : tap + n, :], ttnn.float32), self.taps[tap])
            )
        ttnn.copy(history[:, :, n : n + 3, :], state.conv)
        conv = ttnn.silu(ttnn.typecast(conv, ttnn.bfloat16))
        q = reshape(conv[:, :, :, :2048], (1, n, 16, 128))
        k = reshape(conv[:, :, :, 2048:4096], (1, n, 16, 128))
        v = reshape(conv[:, :, :, 4096:], (1, n, 32, 128))

        def l2(x):
            xf = ttnn.typecast(x, ttnn.float32)
            return ttnn.typecast(
                ttnn.multiply(xf, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(xf, xf), dim=-1, keepdim=True), 1e-6))),
                ttnn.bfloat16,
            )

        q, k = l2(q), l2(k)
        op = ttnn.transformer.fused_recurrent_gated_delta_rule if decode else ttnn.transformer.chunk_gated_delta_rule
        out, new_state = op(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state.recurrent,
            output_final_state=True,
            compute_kernel_config=self.compute,
            **({} if decode else dict(chunk_size=32, **self.gdn_constants)),
        )
        ttnn.copy(new_state, state.recurrent)
        out = ttnn.to_layout(ttnn.typecast(out, ttnn.bfloat16), ttnn.TILE_LAYOUT)
        out = ttnn.multiply(self.norm(out, "linear_attn.norm"), ttnn.silu(z))
        return self.linear(reshape(out, (1, 1, n, 4096)), "linear_attn.out_proj")

    def rope(self, x, cos, sin):
        rot = x[:, :, :, :64]
        half = ttnn.concat([ttnn.neg(rot[:, :, :, 32:]), rot[:, :, :, :32]], dim=-1)
        return ttnn.concat([ttnn.add(ttnn.multiply(rot, cos), ttnn.multiply(half, sin)), x[:, :, :, 64:]], dim=-1)

    def full_attention(self, x, state, *, page_table, position, decode):
        n = x.shape[2]
        qg = reshape(self.linear(x, "self_attn.q_proj"), (1, n, 16, 512))
        q = self.norm(qg[:, :, :, :256], "self_attn.q_norm")
        gate = reshape(qg[:, :, :, 256:], (1, 1, n, 4096))
        k = self.norm(reshape(self.linear(x, "self_attn.k_proj"), (1, n, 4, 256)), "self_attn.k_norm")
        v = reshape(self.linear(x, "self_attn.v_proj"), (1, n, 4, 256))
        if decode:
            rope_position = ttnn.typecast(position, ttnn.uint32)
            cos = reshape(ttnn.embedding(rope_position, self.cos, layout=ttnn.TILE_LAYOUT), (1, 1, 1, 64))
            sin = reshape(ttnn.embedding(rope_position, self.sin, layout=ttnn.TILE_LAYOUT), (1, 1, 1, 64))
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
            out = reshape(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG), (1, 1, 1, 4096))
        else:
            q = ttnn.permute(q, (0, 2, 1, 3))
            k = ttnn.permute(k, (0, 2, 1, 3))
            v = ttnn.permute(v, (0, 2, 1, 3))
            cos = reshape(self.cos[position : position + n, :], (1, 1, n, 64))
            sin = reshape(self.sin[position : position + n, :], (1, 1, n, 64))
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
            out = reshape(ttnn.permute(out, (0, 2, 1, 3)), (1, 1, n, 4096))
        return self.linear(ttnn.multiply(out, ttnn.sigmoid(gate)), "self_attn.o_proj")

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
