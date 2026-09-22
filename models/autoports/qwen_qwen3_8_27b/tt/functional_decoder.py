# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-mesh Qwen3.8-27B text decoder, with explicit caller-owned request state.

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
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import apply_rotary_pos_emb_ttnn


@dataclass
class DecoderState:
    key: object = None
    value: object = None
    recurrent: object = None
    conv: object = None


class FunctionalDecoder(LightweightModule):
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
            raise ValueError("Functional decoder requires a single-device mesh")
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
        return self

    def allocate_state(self, *, batch_size, num_pages=None):
        """Setup only. Page ownership and page-table construction belong to caller."""

        def zeros(shape, dtype):
            return ttnn.zeros(
                shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
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
            conv=zeros([batch_size, 3, width], ttnn.bfloat16),
        )

    def _linear(self, x, name):
        return ttnn.linear(
            x, self.weights[name + ".weight"], compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _norm(self, x, name):
        return ttnn.rms_norm(
            x, weight=self.weights[name + ".weight"], epsilon=self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _finish(self, x, attention):
        h = ttnn.add(x, attention)
        n = self._norm(h, "post_attention_layernorm")
        gate = ttnn.silu(self._linear(n, "mlp.gate_proj"))
        up = self._linear(n, "mlp.up_proj")
        return ttnn.add(h, self._linear(ttnn.mul(gate, up), "mlp.down_proj"))

    def _qkv(self, x, cos, sin):
        b, t, _ = x.shape
        c = self.config
        qg = ttnn.reshape(self._linear(x, "self_attn.q_proj"), [b, t, c.num_attention_heads, 2 * c.head_dim])
        q, gate = ttnn.chunk(qg, 2, dim=-1)
        q = ttnn.transpose(self._norm(q, "self_attn.q_norm"), 1, 2)
        k = ttnn.reshape(self._linear(x, "self_attn.k_proj"), [b, t, c.num_key_value_heads, c.head_dim])
        k = ttnn.transpose(self._norm(k, "self_attn.k_norm"), 1, 2)
        v = ttnn.transpose(
            ttnn.reshape(self._linear(x, "self_attn.v_proj"), [b, t, c.num_key_value_heads, c.head_dim]), 1, 2
        )
        q, k = apply_rotary_pos_emb_ttnn(q, k, cos, sin)
        return q, k, v, ttnn.reshape(gate, [b, t, c.num_attention_heads * c.head_dim])

    def _attention_output(self, attention, gate):
        b, _, t, _ = attention.shape
        attention = ttnn.reshape(ttnn.transpose(attention, 1, 2), [b, t, -1])
        return self._linear(ttnn.mul(attention, ttnn.sigmoid(gate)), "self_attn.o_proj")

    def _full_decode(self, x, state, page_table, current_pos, cos, sin):
        b = x.shape[0]
        c = self.config
        q, k, v, gate = self._qkv(x, cos, sin)
        grid = self.device.compute_with_storage_grid_size()
        cores = ttnn.num_cores_to_corerangeset(b, grid, row_wise=True)
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [32, c.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        for cache, update in ((state.key, k), (state.value, v)):
            update = ttnn.to_memory_config(ttnn.reshape(update, [1, b, c.num_key_value_heads, c.head_dim]), memory)
            ttnn.experimental.paged_update_cache(cache, update, update_idxs_tensor=current_pos, page_table=page_table)
        q = ttnn.reshape(ttnn.transpose(q, 1, 2), [1, b, c.num_attention_heads, c.head_dim])
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
        result = ttnn.reshape(result, [b, 1, c.num_attention_heads, c.head_dim])
        return self._attention_output(ttnn.transpose(result, 1, 2), gate)

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

    def _delta(self, x, state, *, decode):
        b, t, _ = x.shape
        c = self.config
        h, hv, d = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim
        qkv = self._linear(x, "linear_attn.in_proj_qkv")
        history = ttnn.concat([state.conv, qkv], dim=1)
        conv = ttnn.mul(ttnn.to_layout(history[:, :t, :], ttnn.TILE_LAYOUT), self.conv_taps[0])
        for tap in range(1, 4):
            shifted = ttnn.to_layout(history[:, tap : tap + t, :], ttnn.TILE_LAYOUT)
            conv = ttnn.add(conv, ttnn.mul(shifted, self.conv_taps[tap]))
        ttnn.copy(ttnn.to_layout(history[:, t : t + 3, :], ttnn.TILE_LAYOUT), state.conv)
        conv = ttnn.silu(conv)
        q = ttnn.reshape(ttnn.to_layout(conv[:, :, : h * d], ttnn.TILE_LAYOUT), [b, t, h, d])
        k = ttnn.reshape(ttnn.to_layout(conv[:, :, h * d : 2 * h * d], ttnn.TILE_LAYOUT), [b, t, h, d])
        v = ttnn.reshape(ttnn.to_layout(conv[:, :, 2 * h * d :], ttnn.TILE_LAYOUT), [b, t, hv, d])

        # HF normalizes q/k before promoting the recurrence to FP32.
        def l2(a):
            af = ttnn.typecast(a, ttnn.float32)
            return ttnn.typecast(
                ttnn.mul(af, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.mul(af, af), dim=-1, keepdim=True), 1e-6))),
                ttnn.bfloat16,
            )

        q, k = l2(q), l2(k)
        beta = ttnn.sigmoid(self._linear(x, "linear_attn.in_proj_b"))
        a = ttnn.typecast(self._linear(x, "linear_attn.in_proj_a"), ttnn.float32)
        g = ttnn.mul(self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias)))
        if decode:
            q = ttnn.typecast(ttnn.repeat_interleave(q, hv // h, dim=2), ttnn.float32)
            k = ttnn.typecast(ttnn.repeat_interleave(k, hv // h, dim=2), ttnn.float32)
            qr = ttnn.reshape(ttnn.mul(q, d**-0.5), [b, hv, 1, d])
            kr = ttnn.reshape(k, [b, hv, 1, d])
            decayed = ttnn.mul(state.recurrent, ttnn.reshape(ttnn.exp(g), [b, hv, 1, 1]))
            read = ttnn.matmul(kr, decayed, compute_kernel_config=self.ckc)
            delta = ttnn.mul(
                ttnn.subtract(ttnn.reshape(ttnn.typecast(v, ttnn.float32), [b, hv, 1, d]), read),
                ttnn.reshape(ttnn.typecast(beta, ttnn.float32), [b, hv, 1, 1]),
            )
            new_state = ttnn.add(decayed, ttnn.matmul(ttnn.transpose(kr, 2, 3), delta, compute_kernel_config=self.ckc))
            output = ttnn.reshape(ttnn.matmul(qr, new_state, compute_kernel_config=self.ckc), [b, 1, hv, d])
        else:
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
                    chunk_size=32,
                    **self.delta_constants,
                )
                outputs.append(output_part)
                states.append(state_part)
            output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=0)
            new_state = states[0] if len(states) == 1 else ttnn.concat(states, dim=0)
        ttnn.copy(new_state, state.recurrent)
        output = ttnn.typecast(ttnn.to_layout(output, ttnn.TILE_LAYOUT), ttnn.bfloat16)
        output = self._norm(output, "linear_attn.norm")
        z = ttnn.reshape(self._linear(x, "linear_attn.in_proj_z"), [b, t, hv, d])
        output = ttnn.mul(output, ttnn.silu(z))
        return self._linear(ttnn.reshape(output, [b, t, hv * d]), "linear_attn.out_proj")

    def decode_forward(self, x, *, state, current_pos, page_table=None, cos=None, sin=None):
        """One token [B,1,5120]; positions/page table/RoPE are device tensors."""
        if x.shape[1] != 1:
            raise ValueError("Decode requires one token per request")
        n = self._norm(x, "input_layernorm")
        attention = (
            self._full_decode(n, state, page_table, current_pos, cos, sin)
            if self.kind == "full_attention"
            else self._delta(n, state, decode=True)
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
                attention = self._delta(n, state, decode=count == 1)
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
