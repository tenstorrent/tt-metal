# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI U.S. Corp., for the TTNN port. Reference code © Meta Platforms, Inc. (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0


import ttnn


def _tile(x, mc=ttnn.DRAM_MEMORY_CONFIG):
    if x.layout == ttnn.TILE_LAYOUT:
        return x
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=mc)


class TtAttention:
    def __init__(self, parameters, device):
        self.p = parameters
        self.device = device
        self.num_heads = parameters.num_heads
        self.semantic_head_dim = parameters.semantic_head_dim
        self.head_dim = parameters.padded_head_dim or self.semantic_head_dim
        self.internal_dim = self.num_heads * self.head_dim
        self.position_correction_cache = None
        self.qkv_position_correction_cache = None
        self.token_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.spatial_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.sdpa_program_config = None
        self.formatter_memory_configs = None

    def _proj(self, x, w, b):
        x = _tile(x)
        compute_kernel_config = (
            self.token_compute_kernel_config if x.shape[-2] <= 32 else self.spatial_compute_kernel_config
        )
        return ttnn.linear(
            x,
            w,
            bias=b,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def __call__(self, q, k, v, position_delta=None, *, cache_position_delta=False):
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        if (
            q is k
            and self.p.qkv_proj is not None
            and (k is v or (position_delta is not None and self.p.qkv_position_proj is not None))
        ):
            qkv = self._proj(q, self.p.qkv_proj.weight, self.p.qkv_proj.bias)
            if k is not v:
                correction = self.qkv_position_correction_cache if cache_position_delta else None
                if correction is None:
                    correction = self._proj(position_delta, self.p.qkv_position_proj.weight, None)
                    if cache_position_delta:
                        self.qkv_position_correction_cache = correction
                corrected_qkv = ttnn.subtract(qkv, correction, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(qkv)
                if correction is not self.qkv_position_correction_cache:
                    ttnn.deallocate(correction)
                qkv = corrected_qkv
            q_heads, k_heads, v_heads = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv,
                num_heads=self.num_heads,
                transpose_key=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)
        else:
            q_projection = self._proj(q, self.p.q_proj.weight, self.p.q_proj.bias)
            kv_projection = self._proj(k, self.p.kv_proj.weight, self.p.kv_proj.bias)
            if position_delta is not None:
                correction = self.position_correction_cache
                if correction is None:
                    correction = self._proj(position_delta, self.p.kv_position_proj.weight, None)
                    if Nk >= 1024:
                        self.position_correction_cache = correction
                corrected_kv = ttnn.subtract(kv_projection, correction, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(kv_projection)
                if correction is not self.position_correction_cache:
                    ttnn.deallocate(correction)
                kv_projection = corrected_kv
            q_projection = ttnn.reshape(q_projection, (B, 1, Nq, self.internal_dim))
            formatter_q_tokens = max(Nq, self.num_heads * 32)
            if formatter_q_tokens != Nq:
                padded_q_projection = ttnn.pad(
                    q_projection, [B, 1, formatter_q_tokens, self.internal_dim], [0, 0, 0, 0], 0
                )
                ttnn.deallocate(q_projection)
                q_projection = padded_q_projection
            kv_projection = ttnn.reshape(kv_projection, (B, 1, Nk, 2 * self.internal_dim))
            if self.formatter_memory_configs is None:
                shard_grid = ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.num_heads - 1, 0))}
                )
                self.formatter_memory_configs = (
                    ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(
                            shard_grid,
                            [B * formatter_q_tokens, self.internal_dim // self.num_heads],
                            ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                    ),
                    ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(
                            shard_grid,
                            [((B * Nk + 31) // 32) * 32, 2 * self.internal_dim // self.num_heads],
                            ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                    ),
                )
            q_input_memory_config, kv_input_memory_config = self.formatter_memory_configs
            q_projection = ttnn.to_memory_config(q_projection, q_input_memory_config)
            kv_projection = ttnn.to_memory_config(kv_projection, kv_input_memory_config)
            q_heads, k_heads, v_heads = ttnn.experimental.create_qkv_heads_from_separate_tensors(
                q_projection,
                kv_projection,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(q_projection)
            ttnn.deallocate(kv_projection)
            if formatter_q_tokens != Nq:
                padded_q_heads = q_heads
                q_heads = ttnn.slice(
                    padded_q_heads,
                    [0, 0, 0, 0],
                    [B, self.num_heads, Nq, self.head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(padded_q_heads)
            if Nk % 32:
                padded_k_heads, padded_v_heads = k_heads, v_heads
                k_heads = ttnn.slice(
                    padded_k_heads,
                    [0, 0, 0, 0],
                    [B, self.num_heads, Nk, self.head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                v_heads = ttnn.slice(
                    padded_v_heads,
                    [0, 0, 0, 0],
                    [B, self.num_heads, Nk, self.head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(padded_k_heads)
                ttnn.deallocate(padded_v_heads)
            q_heads = ttnn.to_memory_config(q_heads, ttnn.DRAM_MEMORY_CONFIG)
            k_heads = ttnn.to_memory_config(k_heads, ttnn.DRAM_MEMORY_CONFIG)
            v_heads = ttnn.to_memory_config(v_heads, ttnn.DRAM_MEMORY_CONFIG)
        if self.sdpa_program_config is None:
            self.sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=32 if Nq <= 32 else 64,
                k_chunk_size=32 if Nk <= 32 else 256,
                exp_approx_mode=False,
            )
        out = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            scale=1.0 / (self.semantic_head_dim**0.5),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)
        merged = ttnn.transformer.concatenate_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        merged = ttnn.reshape(merged, (B, Nq, self.internal_dim))
        ttnn.deallocate(out)
        return self._proj(_tile(merged), self.p.out_proj.weight, self.p.out_proj.bias)


class TtTwoWayAttentionBlock:
    def __init__(self, parameters, device, skip_first_layer_pe):
        self.p = parameters
        self.skip_first_layer_pe = skip_first_layer_pe
        self.self_attn = TtAttention(parameters.self_attn, device)
        self.cross_attn_token_to_image = TtAttention(parameters.cross_attn_token_to_image, device)
        self.cross_attn_image_to_token = TtAttention(parameters.cross_attn_image_to_token, device)
        self.token_compute_kernel_config = self.self_attn.token_compute_kernel_config

    def _ln(self, x, norm):
        return ttnn.layer_norm(
            _tile(x), weight=norm.weight, bias=norm.bias, epsilon=1e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _add_ln(self, x, residual, norm):
        return ttnn.layer_norm(
            _tile(x),
            weight=norm.weight,
            bias=norm.bias,
            residual_input_tensor=_tile(residual),
            epsilon=1e-6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mlp(self, x):
        x = _tile(x)
        first, second = self.p.mlp.layers["0"], self.p.mlp.layers["1"]
        x = ttnn.linear(
            x,
            first.weight,
            bias=first.bias,
            activation="relu",
            compute_kernel_config=self.token_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        return ttnn.linear(
            x,
            second.weight,
            bias=second.bias,
            compute_kernel_config=self.token_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def _self_attention_queries(self, queries, query_pe, *, cache_query_pe=False):
        if self.skip_first_layer_pe:
            return self._ln(self.self_attn(q=queries, k=queries, v=queries), self.p.norm1)
        q = ttnn.add(queries, query_pe, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out = self.self_attn(
            q=q,
            k=q,
            v=queries,
            position_delta=query_pe,
            cache_position_delta=cache_query_pe,
        )
        return self._add_ln(queries, attn_out, self.p.norm1)

    def __call__(
        self,
        queries,
        keys,
        query_pe,
        key_pe,
        *,
        cache_query_pe=False,
        self_attended_queries=None,
    ):
        queries = (
            self._self_attention_queries(queries, query_pe, cache_query_pe=cache_query_pe)
            if self_attended_queries is None
            else self_attended_queries
        )
        q = ttnn.add(queries, query_pe, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.add(keys, key_pe, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, position_delta=key_pe)
        queries = self._add_ln(queries, attn_out, self.p.norm2)
        queries = self._add_ln(queries, self._mlp(queries), self.p.norm3)
        q = ttnn.add(queries, query_pe, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.add(keys, key_pe, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries, position_delta=query_pe)
        keys = self._add_ln(keys, attn_out, self.p.norm4)
        return queries, keys


class TtTwoWayTransformer:
    def __init__(self, parameters, device):
        self.layers = [
            TtTwoWayAttentionBlock(
                parameters.layers[index],
                device,
                skip_first_layer_pe=index == 0,
            )
            for index in range(2)
        ]
        self.final_attn = TtAttention(
            parameters.final_attn_token_to_image,
            device,
        )
        self.norm_final = parameters.norm_final_attn
        self.cached_first_layer_queries = None

    def __call__(self, image_embedding, image_pe, point_embedding, *, cache_point_embedding=False):
        pe = image_pe
        queries = point_embedding
        keys = image_embedding
        for layer_index, layer in enumerate(self.layers):
            self_attended_queries = None
            if layer_index == 0 and cache_point_embedding:
                self_attended_queries = self.cached_first_layer_queries
                if self_attended_queries is None:
                    self_attended_queries = layer._self_attention_queries(
                        queries,
                        point_embedding,
                        cache_query_pe=True,
                    )
                    self.cached_first_layer_queries = self_attended_queries
            queries, keys = layer(
                queries,
                keys,
                point_embedding,
                pe,
                cache_query_pe=cache_point_embedding,
                self_attended_queries=self_attended_queries,
            )
        q = ttnn.add(queries, point_embedding, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.add(keys, pe, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out = self.final_attn(q=q, k=k, v=keys, position_delta=pe)
        queries = ttnn.layer_norm(
            queries,
            weight=self.norm_final.weight,
            bias=self.norm_final.bias,
            residual_input_tensor=attn_out,
            epsilon=1e-6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return queries, keys


class TtMaskDecoder:
    def __init__(self, parameters, device):
        self.device = device
        self.p = parameters
        self.num_mask_tokens = 4
        self.head_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            output_layout=ttnn.TILE_LAYOUT,
            act_block_h_override=32,
        )
        self.output_tokens = parameters.output_tokens
        self.transformer = TtTwoWayTransformer(parameters.transformer, device)
        self._conv_cache = {}
        self._mask_indices = None
        self.cached_point_embeddings = parameters.no_prompt_decoder_tokens

    def _linear(self, x, w, b, act=None):
        x = _tile(x)
        x = ttnn.linear(
            x,
            w,
            bias=b,
            activation="relu" if act == "relu" else None,
            compute_kernel_config=self.head_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        if act == "sigmoid":
            x = ttnn.sigmoid(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def _mlp_forward(self, mlp_spec, x, final_act=None):
        x = self._linear(x, mlp_spec.layers["0"].weight, mlp_spec.layers["0"].bias, act="relu")
        x = self._linear(x, mlp_spec.layers["1"].weight, mlp_spec.layers["1"].bias, act="relu")
        x = self._linear(x, mlp_spec.layers["2"].weight, mlp_spec.layers["2"].bias, act=final_act)
        return x

    def _mask_index_row(self):
        if self._mask_indices is None:
            indices = ttnn.arange(
                0,
                self.num_mask_tokens - 1,
                1,
                dtype=ttnn.uint32,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self._mask_indices = ttnn.reshape(indices, (1, self.num_mask_tokens - 1))
        return self._mask_indices

    def _dynamic_multimask_via_stability(self, masks, iou_pred):
        batch, _, height, width = masks.shape
        single_mask = ttnn.slice(masks, [0, 0, 0, 0], [batch, 1, height, width])
        single_iou = ttnn.slice(iou_pred, [0, 0], [batch, 1])
        multimasks = ttnn.slice(masks, [0, 1, 0, 0], [batch, self.num_mask_tokens, height, width])
        multi_ious = ttnn.slice(iou_pred, [0, 1], [batch, self.num_mask_tokens])

        best_idx = ttnn.argmax(multi_ious, dim=-1, keepdim=True)
        onehot = ttnn.typecast(ttnn.eq(self._mask_index_row(), best_idx), ttnn.bfloat16)
        onehot = _tile(onehot)
        onehot = ttnn.reshape(onehot, (batch, 1, self.num_mask_tokens - 1))
        tiled_multimasks = _tile(multimasks)
        best_multimask = ttnn.matmul(
            onehot,
            ttnn.reshape(tiled_multimasks, (batch, self.num_mask_tokens - 1, height * width)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        best_multimask = ttnn.reshape(best_multimask, (batch, 1, height, width))
        best_iou = ttnn.matmul(
            onehot,
            ttnn.reshape(multi_ious, (batch, self.num_mask_tokens - 1, 1)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        best_iou = ttnn.reshape(best_iou, (batch, 1))

        delta = self.p.dynamic_multimask_stability_delta
        tiled_single_mask = _tile(single_mask)
        inner = ttnn.typecast(ttnn.gt(tiled_single_mask, delta), ttnn.float32)
        union = ttnn.typecast(ttnn.gt(tiled_single_mask, -delta), ttnn.float32)
        inner_area = ttnn.sum(inner, [2, 3], True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        union_area = ttnn.sum(union, [2, 3], True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stable = ttnn.ge(
            inner_area,
            union_area * self.p.dynamic_multimask_stability_thresh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        stable_weight = ttnn.typecast(ttnn.reshape(stable, (batch, 1)), ttnn.bfloat16)
        selection_weights = ttnn.concat([stable_weight, ttnn.rsub(stable_weight, 1.0)], dim=1)
        selection_weights = ttnn.reshape(_tile(selection_weights), (batch, 1, 2))
        best_multimask_rm = ttnn.to_layout(
            best_multimask,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask_candidates = ttnn.concat([single_mask, best_multimask_rm], dim=1)
        mask_candidates = ttnn.reshape(_tile(mask_candidates), (batch, 2, height * width))
        selected_mask = ttnn.matmul(
            selection_weights,
            mask_candidates,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        selected_mask = ttnn.reshape(selected_mask, (batch, 1, height, width))
        iou_candidates = ttnn.concat([single_iou, best_iou], dim=1)
        iou_candidates = ttnn.reshape(_tile(iou_candidates), (batch, 2, 1))
        selected_iou = ttnn.matmul(selection_weights, iou_candidates, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        selected_iou = ttnn.reshape(selected_iou, (batch, 1))
        return selected_mask, selected_iou

    def _conv_transpose(self, x, w, b, in_c, out_c):
        B, H, W, _ = x.shape
        x_nhwc = _tile(x)
        conv_config = self.conv_config
        compute_config = self.compute_kernel_config
        cache_key = ("transpose", id(w), B, H, W, in_c, out_c)
        if cache_key not in self._conv_cache:
            # Prepare once outside trace capture. The hot path reuses device-side
            # prepared tensors and therefore does not read weights back to host.
            w_host = ttnn.from_device(w) if ttnn.is_tensor_storage_on_device(w) else w
            b_host = ttnn.from_device(b) if ttnn.is_tensor_storage_on_device(b) else b
            prepared_w = ttnn.prepare_conv_transpose2d_weights(
                weight_tensor=w_host,
                input_memory_config=x_nhwc.memory_config(),
                input_layout=x_nhwc.layout,
                weights_format="IOHW",
                in_channels=in_c,
                out_channels=out_c,
                batch_size=B,
                input_height=H,
                input_width=W,
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=(0, 0),
                output_padding=(0, 0),
                dilation=(1, 1),
                has_bias=b is not None,
                groups=1,
                device=self.device,
                conv_config=conv_config,
                compute_config=compute_config,
                mirror_kernel=True,
                input_dtype=ttnn.bfloat16,
            )
            prepared_b = ttnn.prepare_conv_transpose2d_bias(
                bias_tensor=b_host,
                input_memory_config=x_nhwc.memory_config(),
                input_layout=x_nhwc.layout,
                in_channels=in_c,
                out_channels=out_c,
                batch_size=B,
                input_height=H,
                input_width=W,
                device=self.device,
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                conv_config=conv_config,
                compute_config=compute_config,
                input_dtype=ttnn.bfloat16,
            )
            self._conv_cache[cache_key] = (prepared_w, prepared_b)
        w_prepared, b_prepared = self._conv_cache[cache_key]
        out, [oh, ow] = ttnn.conv_transpose2d(
            input_tensor=x_nhwc,
            weight_tensor=w_prepared,
            bias_tensor=b_prepared,
            in_channels=in_c,
            out_channels=out_c,
            batch_size=B,
            input_height=H,
            input_width=W,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            conv_config=conv_config,
            compute_config=compute_config,
            device=self.device,
            return_output_dim=True,
            return_weights_and_bias=False,
        )
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, (B, oh, ow, out_c))

    def __call__(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        high_res_features,
    ):
        owns_tokens_t = sparse_prompt_embeddings is not None
        if owns_tokens_t:
            row_major = ttnn.concat([self.output_tokens, sparse_prompt_embeddings], dim=1)
            tokens_t = _tile(row_major)
            if tokens_t is not row_major:
                ttnn.deallocate(row_major)
        else:
            tokens_t = self.cached_point_embeddings
        src = ttnn.add(image_embeddings, dense_prompt_embeddings, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        B, _, c = src.shape
        hs, src = self.transformer(
            src,
            image_pe,
            tokens_t,
            cache_point_embedding=tokens_t is self.cached_point_embeddings,
        )
        if owns_tokens_t:
            ttnn.deallocate(tokens_t)
        iou_token_out = ttnn.slice(hs, [0, 1, 0], [1, 2, c])
        mask_tokens_out = ttnn.slice(hs, [0, 2, 0], [1, 2 + self.num_mask_tokens, c])
        src = ttnn.reshape(src, (B, 64, 64, c))
        feat_s0, feat_s1 = high_res_features
        dc1, ln1, dc2 = self.p.output_upscaling
        dc1_out = self._conv_transpose(src, dc1.weight, dc1.bias, c, c // 4)
        dc1_out = ttnn.layer_norm(
            dc1_out,
            weight=ln1.weight,
            bias=ln1.bias,
            residual_input_tensor=_tile(feat_s1),
            epsilon=1e-6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        upscaled = ttnn.gelu(dc1_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        upscaled = self._conv_transpose(upscaled, dc2.weight, dc2.bias, c // 4, c // 8)
        added = ttnn.add(upscaled, _tile(feat_s0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        upscaled = ttnn.gelu(added, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dynamic_fallback = not multimask_output and self.p.dynamic_multimask_via_stability
        if not multimask_output:
            mask_tok = ttnn.slice(mask_tokens_out, [0, 0, 0], [B, 1, c])
            mask_tok = ttnn.reshape(mask_tok, (B, c))
            single_hyper = self._mlp_forward(self.p.single_mask_output_hypernetwork_mlp, mask_tok)
            single_hyper = ttnn.reshape(single_hyper, (B, 1, c // 8))
            ttnn.deallocate(mask_tok)
        if multimask_output or dynamic_fallback:
            multimask_tokens = ttnn.slice(mask_tokens_out, [0, 1, 0], [B, self.num_mask_tokens, c])
            multimask_tokens = ttnn.reshape(multimask_tokens, (self.num_mask_tokens - 1, 1, c))
            multi_hyper = self._mlp_forward(self.p.multimask_output_hypernetwork_mlp, multimask_tokens)
            multi_hyper = ttnn.reshape(multi_hyper, (B, self.num_mask_tokens - 1, c // 8))
        if dynamic_fallback:
            hyper_in = ttnn.concat([single_hyper, multi_hyper], dim=1)
            hyper_token_count = self.num_mask_tokens
        elif multimask_output:
            hyper_in = multi_hyper
            hyper_token_count = self.num_mask_tokens - 1
        else:
            hyper_in = single_hyper
            hyper_token_count = 1

        obj_tok = ttnn.slice(hs, [0, 0, 0], [B, 1, c])
        obj_tok = ttnn.reshape(obj_tok, (B, c))
        object_score_logits = self._mlp_forward(self.p.pred_obj_score_head, obj_tok)
        ttnn.deallocate(obj_tok)
        ub, uh, uw, uc = upscaled.shape
        ups_flat = ttnn.reshape(upscaled, (ub, uh * uw, uc))
        ups_flat = ttnn.transpose(ups_flat, -2, -1)
        masks = ttnn.matmul(
            hyper_in,
            ups_flat,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if masks.layout != ttnn.ROW_MAJOR_LAYOUT:
            masks = ttnn.to_layout(masks, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        masks = ttnn.reshape(masks, (ub, hyper_token_count, uh, uw))
        iou_pred = self._mlp_forward(
            self.p.iou_prediction_head,
            ttnn.reshape(iou_token_out, (B, c)),
            final_act="sigmoid",
        )
        if multimask_output:
            iou_pred = ttnn.slice(iou_pred, [0, 1], [B, self.num_mask_tokens])
            sam_tokens_out = ttnn.slice(
                mask_tokens_out,
                [0, 1, 0],
                [B, self.num_mask_tokens, mask_tokens_out.shape[2]],
            )
        else:
            sam_tokens_out = ttnn.slice(mask_tokens_out, [0, 0, 0], [B, 1, mask_tokens_out.shape[2]])
            if dynamic_fallback:
                masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
            else:
                iou_pred = ttnn.slice(iou_pred, [0, 0], [B, 1])
        if masks.layout != ttnn.ROW_MAJOR_LAYOUT:
            masks = ttnn.to_layout(masks, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return masks, iou_pred, sam_tokens_out, object_score_logits
