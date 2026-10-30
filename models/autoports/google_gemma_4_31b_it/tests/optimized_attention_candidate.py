"""Cache precision and explicit attention-program experiments."""

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_packing_candidate import PackingCandidate


class AttentionCandidate(PackingCandidate):
    @classmethod
    def from_state_dict(cls, state_dict, **kw):
        obj = super().from_state_dict(state_dict, **kw)
        obj.sdpa_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, obj.geometry.get("sdpa_fidelity", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        obj.cache_dtype = ttnn.bfloat8_b if obj.geometry.get("kv8", False) else ttnn.bfloat16
        if "decode_k" in obj.geometry:
            obj.decode_program = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=tuple(obj.geometry.get("decode_grid", [8, 8])),
                q_chunk_size=0,
                k_chunk_size=obj.geometry["decode_k"],
                max_cores_per_head_batch=obj.geometry.get("decode_cores", 16),
                exp_approx_mode=False,
            )
        return obj

    def allocate_cache(self, *, num_blocks, page_size=32):
        if page_size != 32 or num_blocks < 1:
            raise ValueError("Expected positive 32-token paged capacity")
        return tuple(
            ttnn.zeros(
                (num_blocks, self.nkv, page_size, self.head_dim),
                dtype=self.cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )

    def _qkv(self, x, cos, sin, *, decode):
        q, k, v = super()._qkv(x, cos, sin, decode=decode)
        if not decode and self.cache_dtype != ttnn.bfloat16:
            k, v = ttnn.typecast(k, self.cache_dtype), ttnn.typecast(v, self.cache_dtype)
        return q, k, v

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
        if self.geometry.get("prefill_l1_inplace"):
            x = ttnn.clone(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        elif self.geometry.get("prefill_l1_carry"):
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
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
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=self.geometry.get("prefill_q", 32),
                k_chunk_size=self.geometry.get("prefill_k", 32),
            ),
            compute_kernel_config=self.sdpa_kernel,
        )
        out = self._finish(x, a, decode=False)
        if self.geometry.get("prefill_l1_carry"):
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        return self._sequence_view(out, valid_tokens)

    def decode_forward(self, x, **kwargs):
        self.decoding = True
        if self.carry:
            x = ttnn.to_memory_config(x, self._working_memory(x, self.geometry.get("residual_cores", 8)))
        try:
            return self._decode_attention_forward(x, **kwargs)
        finally:
            self.decoding = False

    def _decode_attention_forward(self, x, *, cos, sin, kv_cache, page_table, current_pos):
        """x [1,1,B,H], RoPE [1,B,1,D], int32 positions [B], returns x shape.

        Positions are absolute token indices. Updates include the current token;
        the sliding mask is applied in absolute position space. No host reads.
        """
        q, k, v = self._qkv(x, cos, sin, decode=True)
        batch = x.shape[2]
        if not self.sliding_window and batch >= 34 and not self.geometry.get("skip_relocation"):
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
            compute_kernel_config=self.sdpa_kernel,
        )
        return self._finish(x, a, decode=True)
