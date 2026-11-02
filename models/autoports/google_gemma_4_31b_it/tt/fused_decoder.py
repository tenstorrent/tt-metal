# SPDX-License-Identifier: Apache-2.0
"""Fused text-only Gemma 4 31B decoder, derived from the functional stage.

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


class FusedDecoder(LightweightModule):
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
        if obj.sliding_window:
            # At B32, packed prefill is faster while separate decode projections
            # remain faster. The additional setup-only copy costs 176160768 bytes;
            # its measured benefit and unchanged context capacity are documented.
            obj.weights["self_attn.prefill_qkv.weight"] = ttnn.concat(
                [obj.weights["self_attn." + name + "_proj.weight"] for name in ("q", "k", "v")], dim=-1
            )
        else:
            # Full attention shares raw K/V. Pack only Q and K; discard the
            # individual device weights so persistent weight bytes do not grow.
            obj.weights["self_attn.qk_proj.weight"] = ttnn.concat(
                [obj.weights["self_attn.q_proj.weight"], obj.weights["self_attn.k_proj.weight"]], dim=-1
            )
            del obj.weights["self_attn.q_proj.weight"], obj.weights["self_attn.k_proj.weight"]
        grid = mesh_device.compute_with_storage_grid_size()
        # Prepare row identities once. Prefill takes a device slice for its
        # logical batch; no host index construction occurs in either forward.
        obj.prefill_batch_indices = ttnn.arange(
            0, grid.x * grid.y, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
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

    def _decode_memory(self, batch, *, offset=0):
        grid = self.mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x))
                for i in range(offset, offset + batch)
            ]
        )
        return ttnn.create_sharded_memory_config(
            (32, self.head_dim),
            cores,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _rope(self, x, cos, sin, *, decode):
        if decode:
            return ttnn.experimental.rotary_embedding_hf(
                ttnn.to_memory_config(x, self._decode_memory(x.shape[1])),
                cos,
                sin,
                is_decode_mode=True,
                compute_kernel_config=self.kernel,
            )
        batch, heads, length, dim = tuple(x.shape)
        # RoPE's shared-cache kernel requires [1,1,S,D]. Flatten batch into
        # sequence, preserving each request's own absolute-position cos/sin.
        flat = ttnn.reshape(ttnn.permute(x, (1, 0, 2, 3)), (1, heads, batch * length, dim)) if batch > 1 else x
        out = ttnn.experimental.rotary_embedding(
            flat,
            ttnn.reshape(cos, (1, 1, batch * length, dim)),
            ttnn.reshape(sin, (1, 1, batch * length, dim)),
            compute_kernel_config=self.kernel,
        )
        return ttnn.permute(ttnn.reshape(out, (heads, batch, length, dim)), (1, 0, 2, 3)) if batch > 1 else out

    def _qkv(self, x, cos, sin, *, decode):
        if decode:
            mem = self._decode_memory(x.shape[2])
            cos, sin = ttnn.to_memory_config(cos, mem), ttnn.to_memory_config(sin, mem)
        z = self._norm(x, "input_layernorm")
        if self.sliding_window and not decode and x.shape[0] >= 32:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                self._linear(z, "self_attn.prefill_qkv"),
                num_heads=self.nq,
                num_kv_heads=self.nkv,
                transpose_k_heads=False,
            )
        elif self.sliding_window:
            qp = self._linear(z, "self_attn.q_proj")
            kp = self._linear(z, "self_attn.k_proj")
            vp = self._linear(z, "self_attn.v_proj")
            if decode and 16 <= x.shape[2] <= 32:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    ttnn.concat([qp, kp, vp], dim=-1),
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                # RMSNorm cannot consume height-sharded heads in this runtime.
                q, k, v = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v)]
            elif decode:
                q = ttnn.reshape(qp, (1, x.shape[2], self.nq, self.head_dim))
                k = ttnn.reshape(kp, (1, x.shape[2], self.nkv, self.head_dim))
                v = ttnn.reshape(vp, (1, x.shape[2], self.nkv, self.head_dim))
            else:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    qp,
                    ttnn.concat([kp, vp], dim=-1),
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    transpose_k_heads=False,
                )
        else:
            qk = self._linear(z, "self_attn.qk_proj")
            if decode and 16 <= x.shape[2] <= 32:
                width = self.nq * self.head_dim
                kp = ttnn.slice(qk, (0, 0, 0, width), tuple(qk.shape))
                q, unused_k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    ttnn.concat([qk, kp], dim=-1),
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                # The decode splitter has no tied-KV mode. V carries the shared
                # raw projection; only Q and V need interleaved norm inputs.
                q, v = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, v)]
                k = v
                del unused_k
            elif decode:
                width = self.nq * self.head_dim
                qp = ttnn.slice(qk, (0, 0, 0, 0), (*tuple(qk.shape)[:-1], width))
                kp = ttnn.slice(qk, (0, 0, 0, width), tuple(qk.shape))
                q = ttnn.reshape(qp, (1, x.shape[2], self.nq, self.head_dim))
                k = v = ttnn.reshape(kp, (1, x.shape[2], self.nkv, self.head_dim))
            else:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    qk,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    transpose_k_heads=False,
                    kv_tied=True,
                )
        q = self._rope(self._norm(q, "self_attn.q_norm"), cos, sin, decode=decode)
        v = self._norm(v)
        # Full attention normalizes the same raw K/V vector. Reuse its reduction;
        # the extra BF16 rounding is covered by real-weight and context PCC tests.
        k = (
            self._norm(k, "self_attn.k_norm")
            if self.sliding_window
            else ttnn.multiply(v, self.weights["self_attn.k_norm.weight"])
        )
        k = self._rope(k, cos, sin, decode=decode)
        if decode and not self.sliding_window:
            # Long-context SDPA's static buffers collide with resident sharded Q
            # at D512, even after cache-update temporaries are released.
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        return q, k, v

    def _finish(self, x, attention, *, decode):
        if not decode:
            if self.head_dim == 512:
                # A single 32-head concatenation needs >2MiB L1 at D512.
                # Two 16-head groups stay within the hardware's per-core L1.
                attention = ttnn.concat(
                    [
                        ttnn.transformer.concatenate_heads(
                            ttnn.slice(
                                attention,
                                (0, start, 0, 0),
                                (attention.shape[0], start + 16, attention.shape[2], self.head_dim),
                            )
                        )
                        for start in (0, 16)
                    ],
                    dim=-1,
                )
            else:
                attention = ttnn.transformer.concatenate_heads(attention)
        attention = ttnn.reshape(attention, (*tuple(x.shape)[:-1], self.nq * self.head_dim))
        x = ttnn.add(x, self._norm(self._linear(attention, "self_attn.o_proj"), "post_attention_layernorm"))
        z = self._norm(x, "pre_feedforward_layernorm")
        gate, up = self._linear(z, "mlp.gate_proj"), self._linear(z, "mlp.up_proj")
        mlp = self._linear(
            ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.GELU_TANH]), "mlp.down_proj"
        )
        return ttnn.add(
            x,
            self._norm(mlp, "post_feedforward_layernorm"),
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.scalar)],
        )

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

    def _prefill_indices(self, batch):
        if batch == 1 or batch > self.prefill_batch_indices.shape[0]:
            return None
        return ttnn.slice(self.prefill_batch_indices, (0,), (batch,))

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
        batch_idx_tensor = self._prefill_indices(b)
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
                    batch_idx_tensor=batch_idx_tensor,
                )
            )
        physical_outputs = [self._sequence_view(y, y.padded_shape[2]) for y in outputs]
        combined = physical_outputs[0] if len(physical_outputs) == 1 else ttnn.concat(physical_outputs, dim=2)
        return self._sequence_view(combined, length)

    def prefill_chunk_forward(
        self, x, *, cos, sin, kv_cache, page_table, chunk_page_table, chunk_start, valid_tokens, batch_idx_tensor=None
    ):
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
        if batch_idx_tensor is None:
            batch_idx_tensor = self._prefill_indices(x.shape[0])
        if batch_idx_tensor is not None:
            ttnn.experimental.paged_fill_cache(kv_cache[0], k, chunk_page_table, batch_idx_tensor=batch_idx_tensor)
            ttnn.experimental.paged_fill_cache(kv_cache[1], v, chunk_page_table, batch_idx_tensor=batch_idx_tensor)
        else:
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
        batch = x.shape[2]
        if not self.sliding_window and batch >= 34:
            # RoPE places K below its then-live Q/cos/sin shards. At D512/B34+
            # the cache op's B*16 output tiles collide with that low placement.
            # Reclaim K before resharing so it occupies the now-free top of L1.
            # S33 controls pass at B36; both paths exceed available L1 at B37.
            k_dram = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(k)
            k = ttnn.to_memory_config(k_dram, self._decode_memory(batch))
            ttnn.deallocate(k_dram)
        # Fused cache update requires disjoint core sets. K already occupies
        # the first batch cores after RoPE; V uses the following batch cores.
        grid = self.mesh_device.compute_with_storage_grid_size()
        if 2 * batch <= grid.x * grid.y:
            v_sharded = ttnn.to_memory_config(v, self._decode_memory(batch, offset=batch))
            ttnn.experimental.paged_fused_update_cache(
                kv_cache[0],
                k,
                kv_cache[1],
                v_sharded,
                update_idxs_tensor=current_pos,
                page_table=page_table,
            )
        else:
            # Preserve the functional interface beyond the tuned batch set when
            # two disjoint cache-update grids cannot fit on this device.
            ttnn.experimental.paged_update_cache(kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table)
            v_sharded = ttnn.to_memory_config(v, self._decode_memory(batch))
            ttnn.experimental.paged_update_cache(
                kv_cache[1], v_sharded, update_idxs_tensor=current_pos, page_table=page_table
            )
        del k, v, v_sharded
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
