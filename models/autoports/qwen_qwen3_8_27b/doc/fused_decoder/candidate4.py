# SPDX-License-Identifier: Apache-2.0
"""Single-device Qwen3.8 text decoder; weights and request state are explicit.

Runtime tensors stay on device. Setup and reference comparisons belong to callers.
Prefill accepts [B,S,5120], decode [B,1,5120]. Full-attention callers supply
RoPE cos/sin for the absolute positions and a paged cache; linear attention
callers supply convolution and FP32 recurrent state. State is updated in place.
Validation evidence and limitations are in doc/fused_decoder/README.md.
"""

from dataclasses import dataclass

import ttnn

from models.common.lightweightmodule import LightweightModule
from .fp32_recurrence import fp32_recurrence


@dataclass
class LinearState:
    conv: object  # [B,3,10240], BF16
    recurrent: object  # [B,48,128,128], FP32


@dataclass
class PagedState:
    key: object  # [physical_pages,4,32,256], BF16
    value: object


def reshape(x, shape):
    """Preserve logical ordering across tile boundaries."""
    return ttnn.reshape(x, shape)


class FusedDecoder(LightweightModule):
    swiglu_k = 16
    swiglu_n = 10
    chunk_size = 128
    page_size = 32

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device):
        import torch

        config = getattr(hf_config, "text_config", hf_config)
        assert config.hidden_size == 5120 and config.intermediate_size == 17408
        assert config.num_attention_heads == 24 and config.num_key_value_heads == 4
        assert config.linear_num_key_heads == 16 and config.linear_num_value_heads == 48
        assert mesh_device.get_num_devices() == 1
        self = cls()
        self.config, self.mesh_device = config, mesh_device
        self.layer_idx, self.layer_type = layer_idx, config.layer_types[layer_idx]
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        prefix = f"model.language_model.layers.{layer_idx}."
        if not any(k.startswith(prefix) for k in state_dict):
            prefix = ""  # layer-only HF state dict

        def weight(name):
            return state_dict[prefix + name].detach().float()

        def put(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                   device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def linear(name):
            return put(weight(name + ".weight").T)

        self.in_norm = put(1 + weight("input_layernorm.weight"))
        self.post_norm = put(1 + weight("post_attention_layernorm.weight"))
        gate_up = torch.stack([weight("mlp.gate_proj.weight").T.reshape(5120, -1, 32),
                               weight("mlp.up_proj.weight").T.reshape(5120, -1, 32)], dim=2)
        self.gate_up = put(gate_up.reshape(5120, 34816))
        self.down = linear("mlp.down_proj")
        if self.layer_type == "linear_attention":
            eye = torch.eye(32).reshape(1, 1, 32, 32)
            masks = torch.zeros(1, 1, 32, 96)
            masks[..., :16, :16] = 1
            masks[..., 16:, 48:64] = 1
            masks[..., 16:, 64:80] = 1
            self.gdn_constants = dict(eye=put(eye, ttnn.float32),
                tril=put(torch.ones_like(eye).tril(), ttnn.float32),
                ones=put(torch.ones_like(eye), ttnn.float32), masks=put(masks, ttnn.float32))
            self.mix = put(torch.cat([weight("linear_attn.in_proj_qkv.weight").T,
                weight("linear_attn.in_proj_z.weight").T,
                torch.nn.functional.pad(weight("linear_attn.in_proj_a.weight").T, (0, 16)),
                torch.nn.functional.pad(weight("linear_attn.in_proj_b.weight").T, (0, 16))], dim=-1))
            self.out = linear("linear_attn.out_proj")
            conv = weight("linear_attn.conv1d.weight")[:, 0, :]
            self.taps = [put(conv[:, i].reshape(1, 1, -1)) for i in range(4)]
            self.neg_A = put(-weight("linear_attn.A_log").exp().reshape(1, 1, 48), ttnn.float32)
            self.dt = put(weight("linear_attn.dt_bias").reshape(1, 1, 48), ttnn.float32)
            self.out_norm = put(weight("linear_attn.norm.weight"))
        else:
            qg = weight("self_attn.q_proj.weight").reshape(24, 2, 256, 5120)
            self.mix = put(torch.cat([qg[:, 0].reshape(6144, 5120).T,
                weight("self_attn.k_proj.weight").T, weight("self_attn.v_proj.weight").T,
                qg[:, 1].reshape(6144, 5120).T], dim=-1))
            self.out = linear("self_attn.o_proj")
            self.q_norm = put(1 + weight("self_attn.q_norm.weight"))
            self.k_norm = put(1 + weight("self_attn.k_norm.weight"))
            self.chunk_positions = [
                ttnn.from_torch(torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
                for position in range(0, config.max_position_embeddings, self.chunk_size)
            ]
        return self

    def allocate_state(self, *, batch_size, context_length=None):
        """Setup only. Owners release state after all traces referencing it retire."""
        def zeros(shape, dtype):
            return ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                              device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.layer_type == "linear_attention":
            return LinearState(zeros([batch_size, 3, 10240], ttnn.bfloat16),
                               zeros([batch_size, 48, 128, 128], ttnn.float32))
        if context_length is None or not 1 <= context_length <= self.config.max_position_embeddings:
            raise ValueError("Supply a valid context_length for the paged pool")
        # Flexible chunked attention reads page-table rows in 32-byte sticks:
        # round physical capacity to eight INT32 page IDs, not logical context.
        pages = batch_size * ((context_length + 255) // 256) * 8
        return PagedState(zeros([pages, 4, 32, 256], ttnn.bfloat16),
                          zeros([pages, 4, 32, 256], ttnn.bfloat16))

    def snapshot(self, state):
        """Setup/boundary operation: retain an independent resume point on device."""
        if isinstance(state, LinearState):
            return LinearState(ttnn.clone(state.conv), ttnn.clone(state.recurrent))
        return PagedState(ttnn.clone(state.key), ttnn.clone(state.value))

    def _linear(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _norm(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=self.config.rms_norm_eps,
                             compute_kernel_config=self.ckc)

    def _finish(self, residual, mixer):
        x = ttnn.add(residual, mixer)
        h = self._norm(x, self.post_norm)
        mlp = self._linear(ttnn.experimental.minimal_matmul(h, self.gate_up, fuse_swiglu=True,
            config=ttnn.MinimalMatmulConfig(M_block_size=min(4, (h.shape[-2]+31)//32),
                K_block_size=self.swiglu_k, N_block_size=self.swiglu_n, subblock_h=1, subblock_w=2,
                compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size()),
            compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG), self.down)
        return ttnn.add(x, mlp)

    def _delta(self, x, state):
        batch, length, _ = x.shape
        projected = self._linear(x, self.mix)
        qkv = projected[..., :10240]
        history = ttnn.concat([state.conv, qkv], dim=1)
        conv = ttnn.multiply(history[:, :length, :], self.taps[0])
        for i in range(1, 4):
            conv = ttnn.add(conv, ttnn.multiply(history[:, i:i + length, :], self.taps[i]))
        ttnn.copy(ttnn.to_layout(history[:, length:length + 3, :], ttnn.TILE_LAYOUT), state.conv)
        conv = ttnn.silu(conv)
        q = reshape(conv[:, :, :2048], [batch, length, 16, 128])
        k = reshape(conv[:, :, 2048:4096], [batch, length, 16, 128])
        v = reshape(conv[:, :, 4096:], [batch, length, 48, 128])

        def l2(t):
            t = ttnn.typecast(t, ttnn.float32)
            return ttnn.multiply(t, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(t, t), dim=-1, keepdim=True), 1e-6)))

        a = ttnn.typecast(projected[..., 16384:16432], ttnn.float32)
        decay = ttnn.multiply(self.neg_A, ttnn.softplus(ttnn.add(a, self.dt)))
        beta = ttnn.sigmoid(ttnn.typecast(projected[..., 16448:16496], ttnn.float32))
        if length > 1:
            physical = ((length + 31) // 32) * 32
            def padded(t):
                return ttnn.pad(t, [(0, 0), (0, physical-length), (0, 0)], value=0) if physical != length else t
            y, new_state = ttnn.transformer.chunk_gated_delta_rule(
                padded(conv[:, :, :2048]), padded(conv[:, :, 2048:4096]), padded(conv[:, :, 4096:]),
                padded(decay), padded(beta), initial_state=state.recurrent, output_final_state=True,
                chunk_size=32, output_head_major=True, compute_kernel_config=self.ckc, **self.gdn_constants)
            y = ttnn.permute(ttnn.reshape(y, [batch, 48, physical, 128]), (0, 2, 1, 3))[:, :length]
        else:
            q, k, v = l2(q), l2(k), ttnn.typecast(v, ttnn.float32)
            # This kernel assigns one value head to each core. Split its batch axis,
            # retaining batched projections and an explicit state for every request.
            grid = self.mesh_device.compute_with_storage_grid_size()
            kernel_batch = (grid.x * grid.y) // 48
            ys, states = [], []
            for begin in range(0, batch, kernel_batch):
                end = min(begin + kernel_batch, batch)
                y_part, state_part = fp32_recurrence(
                    q[begin:end], k[begin:end], v[begin:end], decay[begin:end], beta[begin:end],
                    initial_state=state.recurrent[begin:end], output_final_state=True, compute_kernel_config=self.ckc,
                )
                ys.append(y_part)
                states.append(state_part)
            y = ttnn.concat(ys, dim=0) if len(ys) > 1 else ys[0]
            new_state = ttnn.concat(states, dim=0) if len(states) > 1 else states[0]
        ttnn.copy(new_state, state.recurrent)
        y = self._norm(ttnn.typecast(y, ttnn.bfloat16), self.out_norm)
        z = reshape(projected[..., 10240:16384], [batch, length, 48, 128])
        return self._linear(reshape(ttnn.multiply(z, y, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), [batch, length, 6144]), self.out)

    def _rope(self, x, cos, sin):
        rot, rest = x[..., :64], x[..., 64:]
        pieces = [ttnn.experimental.rotary_embedding(rot[b:b+1], cos[b:b+1], sin[b:b+1],
                    compute_kernel_config=self.ckc) for b in range(x.shape[0])]
        rotated = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=0)
        return ttnn.concat([rotated[..., :x.shape[-2], :], rest], dim=-1)

    def _attention(self, x, state, *, page_table, cos, sin, current_pos=None, chunk_start=0):
        batch, length, _ = x.shape
        projected = self._linear(x, self.mix)
        q = self._norm(reshape(projected[..., :6144], [batch, length, 24, 256]), self.q_norm)
        k = self._norm(reshape(projected[..., 6144:7168], [batch, length, 4, 256]), self.k_norm)
        v = reshape(projected[..., 7168:8192], [batch, length, 4, 256])
        q = self._rope(ttnn.permute(q, (0, 2, 1, 3)), cos, sin)
        k = self._rope(ttnn.permute(k, (0, 2, 1, 3)), cos, sin)
        v = ttnn.permute(v, (0, 2, 1, 3))
        if current_pos is not None:
            # Decode kernel layout is [1,B,H,D]. Cache update requires height sharding.
            q = ttnn.permute(q, (2, 0, 1, 3))
            k, v = ttnn.permute(k, (2, 0, 1, 3)), ttnn.permute(v, (2, 0, 1, 3))
            mem = ttnn.create_sharded_memory_config(
                [32, 256], core_grid=ttnn.num_cores_to_corerangeset(batch, self.mesh_device.compute_with_storage_grid_size(), row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            for cache, update in ((state.key, k), (state.value, v)):
                ttnn.experimental.paged_update_cache(cache, ttnn.to_memory_config(update, mem),
                                                     update_idxs_tensor=current_pos, page_table=page_table)
            y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, state.key, state.value, page_table_tensor=page_table, cur_pos_tensor=current_pos,
                scale=256**-0.5, compute_kernel_config=self.ckc,
                # The default is 32 tokens and one core/head: 8192 serial BF16
                # softmax-stat updates at full context. Bound the accumulation
                # chain with larger chunks and sequence-parallel cores.
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32, k_chunk_size=256, max_cores_per_head_batch=16,
                    exp_approx_mode=False,
                ),
            )
            y = reshape(y, [batch, 1, 6144])
        else:
            fill_table = page_table[:, chunk_start // 32:]
            for user in range(batch):
                ttnn.experimental.paged_fill_cache(state.key, k[user:user + 1], fill_table, batch_idx=user)
                ttnn.experimental.paged_fill_cache(state.value, v[user:user + 1], fill_table, batch_idx=user)
            y = ttnn.transformer.chunked_scaled_dot_product_attention(
                q, state.key, state.value, page_table_tensor=page_table, chunk_start_idx=None,
                chunk_start_idx_tensor=self.chunk_positions[chunk_start // self.chunk_size],
                compute_kernel_config=self.ckc,
            )
            y = ttnn.transformer.concatenate_heads(y)
        # Matches the installed HF reference. Gate semantics are independently audited.
        y = ttnn.multiply(projected[..., 8192:], y, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return self._linear(y, self.out)

    def decode_forward(self, hidden_states, *, state, page_table=None, current_pos=None, cos=None, sin=None):
        """One token per request; current_pos is a device INT32 tensor, never a host list."""
        assert hidden_states.shape[1] == 1
        h = self._norm(hidden_states, self.in_norm)
        if self.layer_type == "linear_attention":
            y = self._delta(h, state)
        else:
            y = self._attention(h, state, page_table=page_table, current_pos=current_pos, cos=cos, sin=sin)
        return self._finish(hidden_states, y)

    def prefill_forward(self, hidden_states, *, state, start_position=0, page_table=None, cos=None, sin=None):
        """Chunked [B,S,H] prefill, all logical outputs returned; resume at a chunk boundary.

        cos/sin: [B,1,S,64] for absolute positions start_position..start_position+S.
        Linear state summarizes precisely the prefix; paged state owns the prefix pages.
        """
        batch, length, _ = hidden_states.shape
        if not 0 < length or start_position + length > self.config.max_position_embeddings:
            raise ValueError("Invalid logical sequence extent")
        if start_position % self.chunk_size:
            raise ValueError("Prefill continuation starts at a chunk boundary; decode handles individual tokens")
        outputs = []
        for offset in range(0, length, self.chunk_size):
            n = min(self.chunk_size, length - offset)
            x = ttnn.to_layout(hidden_states[:, offset:offset + n, :], ttnn.TILE_LAYOUT)
            h = self._norm(x, self.in_norm)
            if self.layer_type == "linear_attention":
                y = self._delta(h, state)
            else:
                c, s = cos[:, :, offset:offset + n, :], sin[:, :, offset:offset + n, :]
                physical = ((n + 31) // 32) * 32
                if physical != n:
                    h = ttnn.pad(h, [(0, 0), (0, physical - n), (0, 0)], value=0)
                    c = ttnn.pad(c, [(0, 0), (0, 0), (0, physical - n), (0, 0)], value=0)
                    s = ttnn.pad(s, [(0, 0), (0, 0), (0, physical - n), (0, 0)], value=0)
                y = self._attention(h, state, page_table=page_table, cos=c, sin=s, chunk_start=start_position + offset)
                y = y[:, :n, :]
            outputs.append(self._finish(x, y))
        # Bound each concatenation's fan-in for advertised-context prompts.
        while len(outputs) > 1:
            outputs = [ttnn.concat(outputs[i:i + 16], dim=1) if len(outputs[i:i + 16]) > 1 else outputs[i]
                       for i in range(0, len(outputs), 16)]
        return outputs[0]
