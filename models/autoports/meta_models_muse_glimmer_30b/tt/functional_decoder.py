# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer text decoder, BF16 single-chip functional implementation.

Setup loads an HF layer-local state dict (or the canonical checkpoint prefix).
Prefill takes [batch, 1, logical_sequence, 6656] tiled device activations and
returns the same shape. Each batch row names a page-table slot via ``slots``.
``start_pos`` is the absolute cached prefix length (append-only); fresh requests
use zero. Arbitrary logical lengths are padded internally and processed in
chunks. The caller owns page allocation; pages must be disjoint across requests.

Decode takes [1, 1, batch, 6656], a device int32 current-position vector [batch],
a page table [batch, context/32], and optional RoPE tensors [1,batch,1,128].
Positions are the slots to WRITE, and attention includes those slots. The entire
method is device-only and can be captured/replayed. Refresh these persistent
inputs before replay. The caller prepares all eager/trace variants before capture.

Caches are BF16 [physical_pages,2,32,128] for both layer kinds. Sliding attention
masks to the HF window without reducing the advertised absolute context.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class FunctionalDecoder(LightweightModule):
    PAGE_SIZE = 32
    CHUNK_SIZE = 1024

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device):
        import torch  # Setup boundary only.

        self = cls()
        c = getattr(hf_config, "text_config", hf_config)
        if (c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim) != (
            6656,
            19968,
            32,
            2,
            128,
        ):
            raise ValueError("This implementation requires the real Muse-Glimmer-30B text config")
        if c.attention_bias or c.hidden_activation != "silu":
            raise ValueError("Unsupported attention bias / MLP activation")
        self.config, self.device, self.layer_idx = c, mesh_device, layer_idx
        self.kind = c.layer_types[layer_idx]
        self.window = c.sliding_window if self.kind == "sliding_attention" else None
        self.use_rope = c.layer_rope_theta[layer_idx] != 0
        self.ck = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=32, k_chunk_size=32, exp_approx_mode=False
        )
        self.weights = {}
        prefix = f"model.language_model.layers.{layer_idx}."
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.gate_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            key = name + ".weight"
            w = state_dict[key] if key in state_dict else state_dict[prefix + key]
            if w.ndim == 1:
                w = (1 + w.float()).reshape(1, 1, 1, -1)
            else:
                w = w.t().contiguous()
            self.weights[name] = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self.positions = ttnn.from_torch(
            torch.arange(c.max_position_embeddings, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        )
        if self.use_rope:
            inv = 1 / (c.layer_rope_theta[layer_idx] ** (torch.arange(0, 128, 2).float() / 128))
            freq = torch.arange(c.max_position_embeddings).float()[:, None] * inv
            freq = torch.cat((freq, freq), dim=-1)[None, None]
            self.cos = ttnn.from_torch(freq.cos(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
            self.sin = ttnn.from_torch(freq.sin(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        return self

    def allocate_cache(self, *, batch_size=1):
        """Setup only: full-context, disjoint pages for up to batch_size requests."""
        shape = (batch_size * self.config.max_position_embeddings // self.PAGE_SIZE, 2, self.PAGE_SIZE, 128)
        return tuple(
            ttnn.zeros(
                shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )

    def _linear(self, x, name):
        return ttnn.linear(x, self.weights[name], compute_kernel_config=self.ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _norm(self, x, name=None, *, eps=None):
        return ttnn.rms_norm(
            x,
            epsilon=eps or self.config.rms_norm_eps,
            weight=self.weights[name] if name else None,
            compute_kernel_config=self.ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _rope(x, cos, sin):
        a = ttnn.slice(x, [0, 0, 0, 0], [*list(x.shape)[:3], 64])
        b = ttnn.slice(x, [0, 0, 0, 64], [*list(x.shape)[:3], 128])
        rot = ttnn.concat([ttnn.neg(b), a], dim=3)
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))

    def _finish(self, residual, normalized, attn):
        gate = ttnn.sigmoid(self._linear(normalized, "self_attn.gate_proj"))
        attn = self._linear(ttnn.multiply(attn, gate), "self_attn.o_proj")
        x = ttnn.add(residual, self._norm(attn, "post_attention_layernorm", eps=self.config.post_norm_eps))
        normalized = self._norm(x, "pre_feedforward_layernorm")
        gate = ttnn.silu(self._linear(normalized, "mlp.gate_proj"))
        up = self._linear(normalized, "mlp.up_proj")
        mlp = self._linear(ttnn.multiply(gate, up), "mlp.down_proj")
        return ttnn.add(x, self._norm(mlp, "post_feedforward_layernorm", eps=self.config.post_norm_eps))

    def decode_forward(self, x, *, kv_cache, page_table, current_pos, rope_cos=None, rope_sin=None):
        """One on-device step; position/page/RoPE inputs can change between replays."""
        batch = x.shape[2]
        normalized = self._norm(x, "input_layernorm")
        q = ttnn.reshape(self._linear(normalized, "self_attn.q_proj"), (1, batch, 32, 128))
        k = ttnn.reshape(self._linear(normalized, "self_attn.k_proj"), (1, batch, 2, 128))
        v = ttnn.reshape(self._linear(normalized, "self_attn.v_proj"), (1, batch, 2, 128))
        q = ttnn.multiply(self._norm(q), self.config.qk_scale_factor)
        k = self._norm(k)
        if self.use_rope:
            if rope_cos is None or rope_sin is None:
                raise ValueError("RoPE device tensors are required for sliding layers")
            q, k = self._rope(q, rope_cos, rope_sin), self._rope(k, rope_cos, rope_sin)
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        k, v = ttnn.to_memory_config(k, mem), ttnn.to_memory_config(v, mem)
        ttnn.experimental.paged_update_cache(kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            *kv_cache,
            page_table,
            cur_pos_tensor=current_pos,
            sliding_window_size=self.window,
            scale=128**-0.5,
            program_config=self.sdpa,
            compute_kernel_config=self.ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn = ttnn.reshape(attn, (1, 1, batch, 4096))
        return self._finish(x, normalized, attn)

    def _prefill_chunk(self, x, *, start_pos, kv_cache, page_table, slot):
        length = x.shape[2]
        normalized = self._norm(x, "input_layernorm")

        def heads(name, count):
            return ttnn.permute(ttnn.reshape(self._linear(normalized, name), (1, length, count, 128)), (0, 2, 1, 3))

        q, k, v = heads("self_attn.q_proj", 32), heads("self_attn.k_proj", 2), heads("self_attn.v_proj", 2)
        q, k = ttnn.multiply(self._norm(q), self.config.qk_scale_factor), self._norm(k)
        if self.use_rope:
            cos = ttnn.slice(self.cos, [0, 0, start_pos, 0], [1, 1, start_pos + length, 128])
            sin = ttnn.slice(self.sin, [0, 0, start_pos, 0], [1, 1, start_pos + length, 128])
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        pages = ttnn.slice(page_table, [slot, start_pos // 32], [slot + 1, (start_pos + length) // 32])
        for cache, new in zip(kv_cache, (k, v)):
            ttnn.experimental.paged_fill_cache(cache, new, pages, batch_idx=0)
        row = ttnn.slice(page_table, [slot, 0], [slot + 1, page_table.shape[1]])
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            *kv_cache,
            row,
            chunk_start_idx=start_pos,
            sliding_window_size=self.window,
            program_config=self.sdpa,
            compute_kernel_config=self.ck,
        )
        attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), (1, 1, length, 4096))
        return self._finish(x, normalized, attn)

    def prefill_forward(self, x, *, kv_cache, page_table, start_pos=0, slots=None):
        """Chunked, append-only prefill; logical S may be any positive length.

        Initial unaligned prefix positions use single-row cache updates until the
        next page boundary. Padded final rows occupy only unused request-owned
        cache slots, are causally invisible, and are replaced on continuation.
        """
        batch, _, length, hidden = x.shape
        if not (
            0 <= start_pos < self.config.max_position_embeddings
            and 0 < length <= self.config.max_position_embeddings - start_pos
        ):
            raise ValueError("Prefill exceeds the advertised context or is empty")
        slots = tuple(range(batch)) if slots is None else tuple(slots)
        if len(slots) != batch or len(set(slots)) != batch:
            raise ValueError("Each batch row needs a distinct cache slot")
        results = []
        for b, slot in enumerate(slots):
            pieces, offset = [], 0
            while offset < length:
                absolute = start_pos + offset
                if absolute % 32:
                    token = ttnn.slice(x, [b, 0, offset, 0], [b + 1, 1, offset + 1, hidden])
                    row = ttnn.slice(page_table, [slot, 0], [slot + 1, page_table.shape[1]])
                    pos = ttnn.slice(self.positions, [absolute], [absolute + 1])
                    cos = (
                        ttnn.slice(self.cos, [0, 0, absolute, 0], [1, 1, absolute + 1, 128]) if self.use_rope else None
                    )
                    sin = (
                        ttnn.slice(self.sin, [0, 0, absolute, 0], [1, 1, absolute + 1, 128]) if self.use_rope else None
                    )
                    pieces.append(
                        self.decode_forward(
                            token, kv_cache=kv_cache, page_table=row, current_pos=pos, rope_cos=cos, rope_sin=sin
                        )
                    )
                    offset += 1
                    continue
                valid = min(self.CHUNK_SIZE, length - offset)
                physical = (valid + 31) // 32 * 32
                chunk = ttnn.slice(x, [b, 0, offset, 0], [b + 1, 1, offset + valid, hidden])
                if physical != valid:
                    chunk = ttnn.pad(chunk, [(0, 0), (0, 0), (0, physical - valid), (0, 0)], 0)
                out = self._prefill_chunk(
                    chunk, start_pos=absolute, kv_cache=kv_cache, page_table=page_table, slot=slot
                )
                pieces.append(ttnn.slice(out, [0, 0, 0, 0], [1, 1, valid, hidden]))
                offset += valid
            results.append(ttnn.concat(pieces, dim=2) if len(pieces) > 1 else pieces[0])
        return ttnn.concat(results, dim=0) if batch > 1 else results[0]
