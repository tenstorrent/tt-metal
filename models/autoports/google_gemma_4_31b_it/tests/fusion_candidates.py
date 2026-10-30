"""Isolated fusion experiments; runtime overrides use only generic TTNN ops."""
import ttnn
from models.autoports.google_gemma_4_31b_it.tests.fusion_baseline import FusedDecoder


class Candidate(FusedDecoder):
    options = frozenset()

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        obj = super().from_state_dict(state_dict, **kwargs)
        # Packing belongs to setup. Device concat keeps the original BF16 values.
        if "pack_qkv" in cls.options:
            names = ["self_attn.q_proj.weight", "self_attn.k_proj.weight"]
            if obj.sliding_window:
                names.append("self_attn.v_proj.weight")
            obj.weights["packed_qkv.weight"] = ttnn.concat([obj.weights[n] for n in names], dim=-1)
            for n in names:
                del obj.weights[n]
        if "pack_mlp" in cls.options:
            obj.weights["packed_mlp.weight"] = ttnn.concat(
                [obj.weights["mlp.gate_proj.weight"], obj.weights["mlp.up_proj.weight"]], dim=-1
            )
            del obj.weights["mlp.gate_proj.weight"], obj.weights["mlp.up_proj.weight"]
        return obj

    def _rope(self, x, cos, sin):
        if "keep_rope_sharded" not in self.options or not (cos.shape[2] == 1 and x.shape[0] == 1):
            return super()._rope(x, cos, sin)
        batch = x.shape[1]
        cores = ttnn.num_cores_to_corerangeset(batch, self.mesh_device.compute_with_storage_grid_size(), row_wise=True)
        mem = ttnn.create_sharded_memory_config(
            (32, self.head_dim),
            cores,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.experimental.rotary_embedding_hf(
            ttnn.to_memory_config(x, mem),
            ttnn.to_memory_config(cos, mem),
            ttnn.to_memory_config(sin, mem),
            is_decode_mode=True,
            compute_kernel_config=self.kernel,
        )

    def _qkv(self, x, cos, sin, *, decode):
        if "pack_qkv" not in self.options:
            return super()._qkv(x, cos, sin, decode=decode)
        packed = self._linear(self._norm(x, "input_layernorm"), "packed_qkv")
        qwidth, kwidth = self.nq * self.head_dim, self.nkv * self.head_dim

        def part(start, width, n):
            y = ttnn.slice(packed, (0, 0, 0, start), (*tuple(packed.shape)[:-1], start + width))
            if decode:
                return ttnn.reshape(y, (1, x.shape[2], n, self.head_dim))
            return ttnn.permute(ttnn.reshape(y, (x.shape[0], x.shape[2], n, self.head_dim)), (0, 2, 1, 3))

        if "split_heads" in self.options:
            if decode:
                if not self.sliding_window:
                    kt = ttnn.slice(packed, (0, 0, 0, qwidth), tuple(packed.shape))
                    packed = ttnn.concat([packed, kt], dim=-1)
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    packed,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                q, k, v = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v)]
            else:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    packed,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    transpose_k_heads=False,
                    kv_tied=not self.sliding_window,
                )
        else:
            q, k = part(0, qwidth, self.nq), part(qwidth, kwidth, self.nkv)
            v = part(qwidth + kwidth, kwidth, self.nkv) if self.sliding_window else k
        q = self._rope(self._norm(q, "self_attn.q_norm"), cos, sin)
        k = self._rope(self._norm(k, "self_attn.k_norm"), cos, sin)
        return q, k, self._norm(v)

    def _finish(self, x, attention, *, decode):
        if "concat_heads" in self.options and not decode:
            if self.head_dim == 512:
                halves = [
                    ttnn.transformer.concatenate_heads(
                        ttnn.slice(
                            attention, (0, h, 0, 0), (attention.shape[0], h + 16, attention.shape[2], self.head_dim)
                        )
                    )
                    for h in (0, 16)
                ]
                attention = ttnn.concat(halves, dim=-1)
            else:
                attention = ttnn.transformer.concatenate_heads(attention)
        elif not decode:
            attention = ttnn.permute(attention, (0, 2, 1, 3))
        attention = ttnn.reshape(attention, (*tuple(x.shape)[:-1], self.nq * self.head_dim))
        x = ttnn.add(x, self._norm(self._linear(attention, "self_attn.o_proj"), "post_attention_layernorm"))
        z = self._norm(x, "pre_feedforward_layernorm")
        if "pack_mlp" in self.options:
            gu = self._linear(z, "packed_mlp")
            width = self.config.intermediate_size
            gate = ttnn.slice(gu, (0, 0, 0, 0), (*tuple(gu.shape)[:-1], width))
            up = ttnn.slice(gu, (0, 0, 0, width), tuple(gu.shape))
        else:
            up = self._linear(z, "mlp.up_proj")
            if "matmul_gelu" in self.options:
                gate = ttnn.linear(
                    z,
                    self.weights["mlp.gate_proj.weight"],
                    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH),
                    compute_kernel_config=self.kernel,
                )
            else:
                gate = self._linear(z, "mlp.gate_proj")
        acts = [] if "matmul_gelu" in self.options else [ttnn.UnaryOpType.GELU_TANH]
        mlp = self._linear(ttnn.multiply(gate, up, input_tensor_a_activations=acts), "mlp.down_proj")
        norm = self._norm(mlp, "post_feedforward_layernorm")
        if "scalar_merge" in self.options:
            return ttnn.add(x, norm, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.scalar)])
        return ttnn.multiply(ttnn.add(x, norm), self.scalar)


class FlatRopeCandidate(Candidate):
    def _rope(self, x, cos, sin):
        if cos.shape[2] == 1 and x.shape[0] == 1:
            return super()._rope(x, cos, sin)
        b, h, s, d = tuple(x.shape)
        flat = ttnn.reshape(ttnn.permute(x, (1, 0, 2, 3)), (1, h, b * s, d)) if b > 1 else x
        c = ttnn.reshape(cos, (1, 1, b * s, d))
        sn = ttnn.reshape(sin, (1, 1, b * s, d))
        out = ttnn.experimental.rotary_embedding(flat, c, sn, compute_kernel_config=self.kernel)
        return ttnn.permute(ttnn.reshape(out, (h, b, s, d)), (1, 0, 2, 3)) if b > 1 else out


class ExtendedCandidate(FlatRopeCandidate):
    def decode_forward(self, x, *, cos, sin, kv_cache, page_table, current_pos):
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
        if "fused_cache" in self.options:
            grid = self.mesh_device.compute_with_storage_grid_size()
            vc = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x))
                    for i in range(b, 2 * b)
                ]
            )
            vm = ttnn.create_sharded_memory_config(
                (32, self.head_dim),
                vc,
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            ks, vs = ttnn.to_memory_config(k, mem), ttnn.to_memory_config(v, vm)
            ttnn.experimental.paged_fused_update_cache(
                kv_cache[0], ks, kv_cache[1], vs, update_idxs_tensor=current_pos, page_table=page_table
            )
            del ks, vs, k, v
        else:
            for cache, value in zip(kv_cache, (k, v)):
                sharded = ttnn.to_memory_config(value, mem)
                ttnn.experimental.paged_update_cache(
                    cache, sharded, update_idxs_tensor=current_pos, page_table=page_table
                )
            del k, v, value, sharded
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

    def _qkv(self, x, cos, sin, *, decode):
        z = self._norm(x, "input_layernorm")

        def heads(y, n):
            if decode:
                return ttnn.reshape(y, (1, x.shape[2], n, self.head_dim))
            return ttnn.permute(ttnn.reshape(y, (x.shape[0], x.shape[2], n, self.head_dim)), (0, 2, 1, 3))

        if "pack_qkv" in self.options:
            packed = self._linear(z, "packed_qkv")
            qw, kw = self.nq * self.head_dim, self.nkv * self.head_dim

            def part(start, width):
                return ttnn.slice(packed, (0, 0, 0, start), (*tuple(packed.shape)[:-1], start + width))

            if "split_heads" in self.options and decode:
                if not self.sliding_window:
                    packed = ttnn.concat([packed, part(qw, kw)], dim=-1)
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    packed,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                if "keep_split_sharded" not in self.options:
                    q, k, v = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v)]
            elif ("split_heads" in self.options or "split_unpacked" in self.options) and not decode:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    packed,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    transpose_k_heads=False,
                    kv_tied=not self.sliding_window,
                )
            else:
                q, k = heads(part(0, qw), self.nq), heads(part(qw, kw), self.nkv)
                v = heads(part(qw + kw, kw), self.nkv) if self.sliding_window else k
        else:
            qp, kp = self._linear(z, "self_attn.q_proj"), self._linear(z, "self_attn.k_proj")
            vp = self._linear(z, "self_attn.v_proj") if self.sliding_window else kp
            if "split_unpacked" in self.options and not decode:
                kv = ttnn.concat([kp, vp], dim=-1) if self.sliding_window else kp
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    qp,
                    kv,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    transpose_k_heads=False,
                    kv_tied=not self.sliding_window,
                )
            elif "split_heads" in self.options and decode:
                packed = ttnn.concat([qp, kp, vp], dim=-1)
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    packed,
                    num_heads=self.nq,
                    num_kv_heads=self.nkv,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    overlap_qk_coregrid=True,
                )
                if "keep_split_sharded" not in self.options:
                    q, k, v = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v)]
            else:
                q, k, v = heads(qp, self.nq), heads(kp, self.nkv), heads(vp, self.nkv)
        q = self._rope(self._norm(q, "self_attn.q_norm"), cos, sin)
        vn = self._norm(v)
        kn = (
            ttnn.multiply(vn, self.weights["self_attn.k_norm.weight"])
            if "shared_kv_norm" in self.options and not self.sliding_window
            else self._norm(k, "self_attn.k_norm")
        )
        k = self._rope(kn, cos, sin)
        if decode and not self.sliding_window:
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        return q, k, vn

    def _finish(self, x, attention, *, decode):
        if "concat_decode" in self.options and decode:
            b = x.shape[2]
            cores = ttnn.num_cores_to_corerangeset(b, self.mesh_device.compute_with_storage_grid_size(), row_wise=True)
            mem = ttnn.create_sharded_memory_config(
                (32, self.head_dim),
                cores,
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            attention = ttnn.experimental.nlp_concat_heads_decode(
                ttnn.to_memory_config(attention, mem), num_heads=self.nq
            )
            attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
            attention = self._sequence_view(attention, b)
        return super()._finish(x, attention, decode=decode)


class SharedRopeCandidate(ExtendedCandidate):
    def _qkv(self, x, cos, sin, *, decode):
        if decode:
            b = x.shape[2]
            cores = ttnn.num_cores_to_corerangeset(b, self.mesh_device.compute_with_storage_grid_size(), row_wise=True)
            mem = ttnn.create_sharded_memory_config(
                (32, self.head_dim),
                cores,
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            cos, sin = ttnn.to_memory_config(cos, mem), ttnn.to_memory_config(sin, mem)
        return super()._qkv(x, cos, sin, decode=decode)
