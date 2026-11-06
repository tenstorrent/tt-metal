# SPDX-License-Identifier: Apache-2.0
"""Text-only Gemma 4 31B decoder, derived from the pinned HF reference.

Setup owns weight conversion and cache allocation. Runtime takes only TTNN tensors.
Prefill consumes a page-aligned chunk, with arbitrary logical length (padding is
internal to the caller's physical bucket); chunk_page_table maps its first page to
absolute chunk_start. Causal attention never sees padded future rows. Decode uses
device current positions and stable page tables, suitable for TTNN capture.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

HF_MODEL = "google/gemma-4-31B-it"
HF_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"


class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        obj = cls()
        c = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        if kwargs:
            raise TypeError(f"Unsupported decoder options: {tuple(kwargs)}")
        if (
            c.hidden_size,
            c.intermediate_size,
            c.num_attention_heads,
            c.head_dim,
            c.global_head_dim,
            c.num_key_value_heads,
            c.num_global_key_value_heads,
        ) != (5376, 21504, 32, 256, 512, 16, 4):
            raise ValueError("Expected pinned Gemma 4 31B text geometry")
        if c.enable_moe_block or c.num_kv_shared_layers or c.hidden_size_per_layer_input:
            raise ValueError("Pinned target has no MoE, KV sharing or per-layer input")
        if c.max_position_embeddings != 262144 or not c.attention_k_eq_v or c.attention_bias:
            raise ValueError("Expected pinned context and attention projection contract")
        if c.hidden_activation != "gelu_pytorch_tanh" or c.sliding_window != 1024:
            raise ValueError("Expected pinned activation and sliding-window contract")
        obj.config, obj.mesh_device = c, mesh_device
        obj.layer_type = c.layer_types[layer_idx]
        if obj.layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError("Unknown target layer kind")
        obj.sliding_window = c.sliding_window if obj.layer_type == "sliding_attention" else None
        obj.head_dim = c.head_dim if obj.sliding_window else c.global_head_dim
        obj.nkv = c.num_key_value_heads if obj.sliding_window else c.num_global_key_value_heads
        obj.nq, obj.hidden = c.num_attention_heads, c.hidden_size
        obj.kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # The default one-core/K32 recurrence loses BF16 state precision at 262K.
        # Splitting across 16 cores with K128 bounds the local recurrence to 128
        # chunks at the advertised batch-1 context. See AUTOFIX.md and the probe.
        obj.decode_program = (
            None
            if obj.sliding_window
            else ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=32,
                k_chunk_size=128,
                max_cores_per_head_batch=16,
                exp_approx_mode=False,
            )
        )
        obj.weights = {}
        prefix = f"model.language_model.layers.{layer_idx}."
        if any(name.startswith(prefix) for name in state_dict):
            state_dict = {
                name.removeprefix(prefix): tensor for name, tensor in state_dict.items() if name.startswith(prefix)
            }
        expected = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "layer_scalar",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        }
        if obj.sliding_window:
            expected.add("self_attn.v_proj.weight")
        if set(state_dict) != expected:
            raise ValueError(
                f"Layer weight keys mismatch: missing={expected-set(state_dict)}, extra={set(state_dict)-expected}"
            )
        for name, tensor in state_dict.items():
            if name.endswith("proj.weight"):
                tensor = tensor.T.contiguous()
            elif name == "layer_scalar":
                obj.scalar = float(tensor.item())
                continue
            else:
                tensor = tensor.reshape(1, 1, 1, -1)
            obj.weights[name] = ttnn.from_torch(
                tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return obj

    def allocate_cache(self, *, num_blocks, page_size=32):
        if page_size != 32 or num_blocks < 1:
            raise ValueError("This decoder uses positive cache capacity with 32-token pages")
        shape = (num_blocks, self.nkv, page_size, self.head_dim)
        return tuple(
            ttnn.zeros(
                shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )

    def _norm(self, x, name=None):
        return ttnn.rms_norm(
            x,
            epsilon=self.config.rms_norm_eps,
            weight=self.weights[name + ".weight"] if name else None,
            compute_kernel_config=self.kernel,
        )

    def _linear(self, x, name):
        return ttnn.linear(x, self.weights[name + ".weight"], compute_kernel_config=self.kernel)

    def _rope(self, x, cos, sin):
        d = self.head_dim
        a = ttnn.slice(x, (0, 0, 0, 0), (*tuple(x.shape)[:-1], d // 2))
        b = ttnn.slice(x, (0, 0, 0, d // 2), tuple(x.shape))
        rotated = ttnn.concat([ttnn.neg(b), a], dim=-1)
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))

    def _qkv(self, x, cos, sin, *, decode):
        z = self._norm(x, "input_layernorm")

        def heads(y, n):
            if decode:
                return ttnn.reshape(y, (1, x.shape[2], n, self.head_dim))
            return ttnn.permute(ttnn.reshape(y, (x.shape[0], x.shape[2], n, self.head_dim)), (0, 2, 1, 3))

        q = heads(self._linear(z, "self_attn.q_proj"), self.nq)
        kraw = heads(self._linear(z, "self_attn.k_proj"), self.nkv)
        vraw = heads(self._linear(z, "self_attn.v_proj"), self.nkv) if self.sliding_window else kraw
        q = self._rope(self._norm(q, "self_attn.q_norm"), cos, sin)
        k = self._rope(self._norm(kraw, "self_attn.k_norm"), cos, sin)
        v = self._norm(vraw)
        return q, k, v

    def _finish(self, x, attention, *, decode):
        if not decode:
            attention = ttnn.permute(attention, (0, 2, 1, 3))
        attention = ttnn.reshape(attention, (*tuple(x.shape)[:-1], self.nq * self.head_dim))
        x = ttnn.add(x, self._norm(self._linear(attention, "self_attn.o_proj"), "post_attention_layernorm"))
        z = self._norm(x, "pre_feedforward_layernorm")
        gate = ttnn.gelu(self._linear(z, "mlp.gate_proj"), fast_and_approximate_mode=True)
        mlp = self._linear(ttnn.multiply(gate, self._linear(z, "mlp.up_proj")), "mlp.down_proj")
        return ttnn.multiply(ttnn.add(x, self._norm(mlp, "post_feedforward_layernorm")), self.scalar)

    @staticmethod
    def _sequence_view(x, length):
        # Only the logical row count changes; backing storage and tile geometry
        # stay fixed. No FillPad program or new persistent buffer is needed when
        # a request changes logical length inside its 32-row physical bucket.
        logical = list(x.shape)
        padded = list(x.padded_shape)
        if (length + 31) // 32 * 32 != padded[-2]:
            raise ValueError("Sequence view must preserve physical tile storage")
        logical[-2] = length
        return ttnn.reshape(x, ttnn.Shape(logical), ttnn.Shape(padded), skip_padding_fill=True)

    def prefill_forward(self, x, *, cos, sin, kv_cache, page_table, chunk_start=0):
        """Prefill arbitrary logical [B,1,S,H] in bounded 1024-token chunks.

        RoPE inputs are [B,1,S,D], already evaluated at absolute positions.
        Explicit continuation starts at a page boundary and retains prefix pages.
        All chunk planning uses shape metadata; tensor work stays on device.
        """
        b, _, length, _ = tuple(x.shape)
        page_size = kv_cache[0].shape[2]
        if length < 1 or chunk_start % page_size:
            raise ValueError("Empty prefill or continuation outside a page boundary")
        if chunk_start + length > self.config.max_position_embeddings:
            raise ValueError("Context exceeds pinned HF contract")
        outputs = []
        for offset in range(0, length, 1024):
            valid = min(1024, length - offset)
            physical = (valid + 31) // 32 * 32

            def chunk(t):
                physical_input = self._sequence_view(t, t.padded_shape[2])
                return ttnn.slice(physical_input, (0, 0, offset, 0), (b, 1, offset + physical, t.shape[3]))

            first_page = (chunk_start + offset) // page_size
            table = ttnn.slice(page_table, (0, first_page), (b, first_page + (physical + page_size - 1) // page_size))
            outputs.append(
                self.prefill_chunk_forward(
                    chunk(x),
                    cos=chunk(cos),
                    sin=chunk(sin),
                    kv_cache=kv_cache,
                    page_table=page_table,
                    chunk_page_table=table,
                    chunk_start=chunk_start + offset,
                    valid_tokens=valid,
                )
            )
        physical_outputs = [self._sequence_view(y, y.padded_shape[2]) for y in outputs]
        combined = physical_outputs[0] if len(physical_outputs) == 1 else ttnn.concat(physical_outputs, dim=2)
        return self._sequence_view(combined, length)

    def prefill_chunk_forward(self, x, *, cos, sin, kv_cache, page_table, chunk_page_table, chunk_start, valid_tokens):
        """x [B,1,S,H]; RoPE [B,1,S,D]; returns [B,1,valid_tokens,H].

        S is the padded physical length; 0 < valid_tokens <= S. Each request
        owns distinct cache pages. Resume retains all prefix pages and starts at
        a page boundary. The final partial page is private to this request.
        """
        if not 0 < valid_tokens <= x.shape[2] or chunk_start % kv_cache[0].shape[2]:
            raise ValueError("Invalid logical length or non-page-aligned chunk start")
        if chunk_start + valid_tokens > self.config.max_position_embeddings:
            raise ValueError("Context exceeds pinned HF contract")
        q, k, v = self._qkv(x, cos, sin, decode=False)
        for b in range(x.shape[0]):
            kb = ttnn.slice(k, (b, 0, 0, 0), (b + 1, self.nkv, x.shape[2], self.head_dim))
            vb = ttnn.slice(v, (b, 0, 0, 0), (b + 1, self.nkv, x.shape[2], self.head_dim))
            ttnn.experimental.paged_fill_cache(kv_cache[0], kb, chunk_page_table, batch_idx=b)
            ttnn.experimental.paged_fill_cache(kv_cache[1], vb, chunk_page_table, batch_idx=b)
        a = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            *kv_cache,
            page_table,
            chunk_start_idx=chunk_start,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=32
            ),
            compute_kernel_config=self.kernel,
        )
        out = self._finish(x, a, decode=False)
        return self._sequence_view(out, valid_tokens)

    def decode_forward(self, x, *, cos, sin, kv_cache, page_table, current_pos):
        """x [1,1,B,H], RoPE [1,B,1,D], int32 positions [B], returns x shape.

        Positions are absolute token indices. Updates include the current token;
        the sliding mask is applied in absolute position space. No host reads.
        """
        q, k, v = self._qkv(x, cos, sin, decode=True)
        b = x.shape[2]
        cores = ttnn.num_cores_to_corerangeset(b, self.mesh_device.compute_with_storage_grid_size(), row_wise=True)
        mem = ttnn.create_sharded_memory_config(
            (32, self.head_dim),
            cores,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        for cache, value in zip(kv_cache, (k, v)):
            sharded = ttnn.to_memory_config(value, mem)
            ttnn.experimental.paged_update_cache(cache, sharded, update_idxs_tensor=current_pos, page_table=page_table)
        a = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            *kv_cache,
            page_table,
            cur_pos_tensor=current_pos,
            program_config=self.decode_program,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            compute_kernel_config=self.kernel,
        )
        return self._finish(x, a, decode=True)
