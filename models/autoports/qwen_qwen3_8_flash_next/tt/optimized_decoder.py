# SPDX-License-Identifier: Apache-2.0
"""Qualified single-device decoder policy for Qwen3.8-Flash-Next on Blackhole.

Setup owns weight packing and immutable constants. Forward paths consume only
TT tensors and static shapes; request positions, slots and page tables stay on
device. See doc/optimized_decoder for the real-weight policy and measurements.
"""

import math
import ttnn
from .fused_decoder import (
    FusedDecoder,
    FusedGatedDeltaNet,
    FusedQSA,
    FusedPLE,
    FusedMoE,
    FusedHyperConnection,
    _PackedMoE,
    _PrefixView,
    _shared_rope_tables,
    _shared_apply_rope,
    _PRECISE_GDN_READER,
    _PRECISE_PREFILL_READER,
    _PRECISE_GDN_COMPUTE,
    ngram_indices,
    _conv_taps,
    _mask_padding,
)


class OptimizedDecoder(FusedDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch_size=1,
        max_requests=None,
        chunk_size=512,
        cache_dir=None,
    ):
        import torch  # Setup only.

        obj = cls()
        cfg = getattr(hf_config, "text_config", hf_config)
        if (cfg.hidden_size, cfg.hc_count) != (2560, 4):
            raise ValueError("Expected the target Qwen3.8-Flash-Next configuration")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        prefix = f"model.language_model.layers.{layer_idx}."
        state = _PrefixView(state_dict, prefix) if prefix + "mlp.gate.weight" in state_dict else state_dict
        obj.batch_size, obj.chunk_size = batch_size, chunk_size
        obj.context = cfg.max_position_embeddings
        obj.mesh = mesh_device
        obj.kind = cfg.layer_types[layer_idx]
        obj.attn_hyper = OptimizedHyperConnection.from_state_dict(
            state, prefix="attn_hyper_connection", hf_config=cfg, mesh_device=mesh_device
        )
        obj.mlp_hyper = OptimizedHyperConnection.from_state_dict(
            state, prefix="mlp_hyper_connection", hf_config=cfg, mesh_device=mesh_device
        )
        obj.mlp = OptimizedMoE.from_state_dict(state, hf_config=cfg, mesh_device=mesh_device)
        requests = max_requests or batch_size
        if obj.kind == "linear_attention":
            obj.attention = OptimizedGatedDeltaNet.from_state_dict(
                _PrefixView(state, "linear_attn."),
                hf_config=cfg,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch_size=batch_size,
                max_requests=requests,
                chunk_size=chunk_size,
            )
        else:
            obj.attention = OptimizedQSA.from_state_dict(
                _PrefixView(state, "self_attn."),
                hf_config=cfg,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch_size=batch_size,
                max_requests=requests,
            )
        obj.ple = None
        if layer_idx + 1 in cfg.ple_layer_ids:
            obj.ple = OptimizedPLE.from_state_dict(
                state, hf_config=cfg, layer_idx=layer_idx, prefix="ple.", mesh_device=mesh_device, cache_dir=cache_dir
            )
            obj.ple_cache = obj.ple.make_cache(max_requests=requests)
        obj.lengths = {}
        obj.prepare([1, chunk_size])
        return obj


class _ProjectionPolicy:
    projection_dtype = ttnn.bfloat8_b
    projection_fidelity = ttnn.MathFidelity.HiFi2
    output_dtype = None
    output_fidelity = None

    def _prepare_projections(self, names):
        self.projection_compute = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=self.projection_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        if self.projection_dtype == ttnn.bfloat16 and self.projection_fidelity == ttnn.MathFidelity.HiFi4:
            self.projection_compute = self.compute
        self.projection_computes = {}
        for name in names:
            output = name in ("out_weight", "wo")
            dtype = self.output_dtype if output and self.output_dtype is not None else self.projection_dtype
            value = getattr(self, name)
            if value.dtype != dtype:
                value = ttnn.typecast(value, dtype)
                setattr(self, name, value)
            compute = self.projection_compute
            if output and self.output_fidelity is not None:
                compute = ttnn.init_device_compute_kernel_config(
                    self.device.arch(),
                    math_fidelity=self.output_fidelity,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
            self.projection_computes[value.buffer_address()] = compute

    def _linear(self, x, weight):
        return _dram_linear(self, x, weight)


class OptimizedGatedDeltaNet(_ProjectionPolicy, FusedGatedDeltaNet):
    state_workers = 32

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj._prepare_projections(("packed_weight", "out_weight"))
        import torch  # Setup only.

        _configure_dram(obj, ("packed_weight", "out_weight"), cores=(10, 8), blocks=(8, 24), readers=(2, 2))
        obj.dram_result_memory = "l1"
        obj.qscale = ttnn.from_torch(
            torch.full((1, 1, 1, 128), 1 / 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=obj.device
        )
        obj.kscale = ttnn.from_torch(
            torch.full((1, 1, 1, 128), 128**-0.5), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=obj.device
        )
        return obj

    def _route_state(self, slots, positions, *, store):
        """Gather/scatter complete raw tiles using device request IDs.

        Unique slot IDs within each batch are a caller precondition. Position
        zero initializes the selected lane on gather; scatter updates only the
        selected slot. Banks and work buffers are persistent across replay.
        """
        for bank, work, tiles, tile_bytes in (
            (self.recurrent_bank, self.recurrent_state, 48 * 4 * 4, 4096),
            (self.conv_bank, self.conv_state, 320, 2048),
        ):
            grid = self.device.compute_with_storage_grid_size()
            coords = [
                ttnn.CoreCoord(i % grid.x, i // grid.x)
                for i in range(min(grid.x * grid.y, self.batch_size * self.state_workers))
            ]
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
            tensors = [bank, work, slots, positions]
            workers = max(1, len(coords) // self.batch_size)
            coords = coords[: workers * self.batch_size]
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
            args = [tiles, tile_bytes, self.max_requests, int(store), workers]
            for tensor in tensors:
                args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
            cb = ttnn.CBDescriptor(
                total_size=tile_bytes + 256,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=tile_bytes + 256)
                ],
            )
            kernel = ttnn.KernelDescriptor(
                kernel_source=_PARALLEL_STATE_SOURCE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=args,
                runtime_args=[
                    ((c.x, c.y), [*[t.buffer_address() for t in tensors], i // workers, i % workers])
                    for i, c in enumerate(coords)
                ],
                config=ttnn.ReaderConfigDescriptor(),
            )
            ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))

    def _chunk(self, x):
        b, s, _ = x.shape
        padded = (s + 31) // 32 * 32
        projections = self._linear(x, self.packed_weight)
        projected = projections[..., :10240]
        proj = ttnn.pad(projected, [(0, 0), (0, padded - s), (0, 0)], 0) if padded != s else projected
        lanes = []
        for lane in range(b):
            incoming = ttnn.to_layout(self.conv_state[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT)
            lanes.append(
                ttnn.experimental.kda.qkv_causal_conv1d_silu(
                    ttnn.to_layout(proj[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT),
                    incoming,
                    *self.conv_taps,
                    2048,
                    2048,
                    6144,
                    actual_start=self.conv_zero,
                    predecessor_carry=incoming,
                    program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
                    compute_kernel_config=self.compute,
                )
            )
        q, k, v = [ttnn.concat([row[i] for row in lanes], dim=0) if b > 1 else lanes[0][i] for i in range(3)]
        q, k, v = [ttnn.to_layout(t[:, :s, :], ttnn.TILE_LAYOUT) for t in (q, k, v)]
        self._roll_conv(projected)
        a = ttnn.typecast(ttnn.to_layout(projections[..., 16384:16432], ttnn.TILE_LAYOUT), ttnn.float32)
        g = ttnn.multiply(self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias)), activations=[ttnn.UnaryOpType.EXP])
        beta = ttnn.typecast(
            ttnn.sigmoid(ttnn.to_layout(projections[..., 16448:16496], ttnn.TILE_LAYOUT)), ttnn.float32
        )
        o, new_state = self._recurrence(q, k, v, g, beta)
        o = ttnn.typecast(o, ttnn.bfloat16)
        gate = projections[..., 10240:16384]
        if padded != s:
            o = ttnn.pad(o, [(0, 0), (0, 0), (0, padded - s), (0, 0)], 0)
            gate = ttnn.pad(gate, [(0, 0), (0, padded - s), (0, 0)], 0)
        o = ttnn.reshape(o, (b * 48, padded, 128))
        gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o,
            gate,
            self.norm_weight_flat,
            48,
            epsilon=self.norm_eps,
            compute_kernel_config=self.compute,
            output_dtype=ttnn.bfloat16,
        )
        gated = ttnn.to_layout(gated[:, :s, :], ttnn.TILE_LAYOUT)
        return self._linear(gated, self.out_weight)

    def _recurrence(self, q, k, v, g, beta):
        b, s, _ = q.shape
        q = self._heads(q, b, s, 16)
        k = self._heads(k, b, s, 16)
        q = ttnn.rms_norm(
            ttnn.typecast(q, ttnn.float32), weight=self.qscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        k = ttnn.rms_norm(
            ttnn.typecast(k, ttnn.float32), weight=self.kscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        v = ttnn.typecast(self._heads(v, b, s, 48), ttnn.float32)
        return _inplace_recurrence(self, q, k, v, g, beta)

    def _roll_conv(self, projected):
        b, s, _ = projected.shape
        workers = 80
        coords = [ttnn.CoreCoord(i % 10, i // 10) for i in range(workers)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        tensors = [self.conv_state, projected]
        args = [b, s, workers]
        for t in tensors:
            args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        k = ttnn.KernelDescriptor(
            kernel_source=_ROLL_CONV_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=6144,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=6144)],
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[k], cbs=[cb]))


class OptimizedQSA(_ProjectionPolicy, FusedQSA):
    cache_dtype = ttnn.bfloat16
    sparse_fidelity = ttnn.MathFidelity.HiFi4
    sparse_chunk = 128
    decode_chunk = 32
    index_width = 2176
    indexer_chunk = 128

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj._prepare_projections(("packed_projection", "wo"))
        _configure_dram(obj, ("packed_projection", "wo"), cores=(10, 8), blocks=(8, 24), readers=(2, 2))
        obj.dram_result_memory = "l1"
        import torch  # Setup only.

        obj.chunk_size = 512 if obj.batch_size == 1 else 128
        obj.chunk_offsets = ttnn.from_torch(
            torch.arange(obj.chunk_size).float().reshape(1, 1, -1, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=obj.device,
        )
        obj.pool_tokens = obj.k_cache.shape[0] * obj.page_size
        del obj.k_cache, obj.v_cache
        if obj.batch_size == 1:
            del obj._physical_update_indices, obj._index_core
        obj.packed_cache = ttnn.empty(
            (1, 1, obj.pool_tokens * 2, 512),
            dtype=cls.cache_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=obj.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        obj.sparse_compute = ttnn.init_device_compute_kernel_config(
            obj.device.arch(),
            math_fidelity=cls.sparse_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        obj.indexer_prefix = ttnn.zeros(
            (obj.batch_size, 4, 31, 128), device=obj.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        obj.indexer_weights = ttnn.ones(
            (obj.batch_size, 1, obj.chunk_size + 32, 4), device=obj.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        obj.indexer_compute = ttnn.init_device_compute_kernel_config(
            obj.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        return obj

    def prefill_forward(self, hidden_states, *, current_pos, page_table, chunk_size=None, slots=None):
        size = self.chunk_size if chunk_size is None else chunk_size
        if not 1 <= size <= self.chunk_size:
            raise ValueError(f"chunk_size must lie in [1,{self.chunk_size}]")
        return super().prefill_forward(
            hidden_states, current_pos=current_pos, page_table=page_table, chunk_size=size, slots=slots
        )

    def _adapter(self, source, output, positions, page_table, mode, seq, index_width=None):
        width = self.index_width if index_width is None else index_width
        work = self.batch_size * seq * (2 if mode == 0 else 1)
        grid = self.device.compute_with_storage_grid_size()
        _, cores, g1, _, n1, n2 = ttnn.split_work_to_cores(grid, work, True)
        tensors = [source, output, positions, page_table]
        args = [
            self.batch_size,
            seq,
            self.page_size,
            self.max_context,
            self.pool_tokens,
            width,
            mode,
            1024 if self.cache_dtype == ttnn.bfloat16 else 512,
        ]
        for t in tensors:
            args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt = []
        offset = 0
        for i, c in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
            n = n1 if i < g1.num_cores() else n2
            rt.append(((c.x, c.y), [*[t.buffer_address() for t in tensors], offset, n]))
            offset += n
        scratch = 128 + self.max_context // self.page_size * 4 + 2048 + width * 4
        scratch = (scratch + 63) // 64 * 64
        cb = ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)],
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=_QSA_ADAPTER_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))

    def _write_packed(self, k, v, positions, page_table):
        packed = ttnn.to_layout(ttnn.concat((v, k), dim=-1), ttnn.ROW_MAJOR_LAYOUT)
        if packed.dtype != self.cache_dtype:
            packed = ttnn.typecast(packed, self.cache_dtype)
        self._adapter(packed, self.packed_cache, positions, page_table, 0, k.shape[2])

    def _sparse_attention(self, q, selected, positions, page_table):
        b, _, s, _ = tuple(q.shape)
        k_chunk = self.decode_chunk if s == 1 and self.decode_chunk else self.sparse_chunk
        width = (2051 + k_chunk - 1) // k_chunk * k_chunk
        indices = ttnn.empty(
            (1, 1, b * 2 * s, width),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        selected = ttnn.to_layout(ttnn.typecast(selected, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        self._adapter(selected, indices, positions, page_table, 1, s, index_width=width)
        groups = []
        for batch in range(b):
            for g in range(2):
                chunk = q[batch : batch + 1, g * 12 : (g + 1) * 12, :, :]
                chunk = ttnn.pad(chunk, [(0, 0), (0, 20), (0, 0), (0, 0)], 0)
                groups.append(ttnn.concat((ttnn.zeros_like(chunk), chunk), dim=-1))
        folded = ttnn.to_layout(ttnn.concat(groups, dim=2), ttnn.ROW_MAJOR_LAYOUT)
        fmt = (
            ttnn.transformer.SparseKVFormat.BF16
            if self.cache_dtype == ttnn.bfloat16
            else ttnn.transformer.SparseKVFormat.FP8_E4M3
        )
        out = ttnn.transformer.sparse_sdpa(
            folded,
            self.packed_cache,
            indices,
            256,
            kv_format=fmt,
            scale=0.0625,
            k_chunk_size=k_chunk,
            compute_kernel_config=self.sparse_compute,
        )
        rows = []
        for batch in range(b):
            heads = [out[:, :12, (batch * 2 + g) * s : (batch * 2 + g + 1) * s, :] for g in range(2)]
            rows.append(ttnn.concat(heads, dim=1))
        out = rows[0] if b == 1 else ttnn.concat(rows, dim=0)
        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (b, 1, s, 6144))
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT)

    def _chunk_forward(self, hidden_states, *, current_pos, page_table):
        """Vectorized projections/selection with bounded selected-KV attention."""
        batch, seq = (self.batch_size, hidden_states.shape[-2])
        x = self._reshape_logical(hidden_states, (batch, 1, seq, 2560))
        positions_packed = self._packed_position(current_pos)
        start = ttnn.permute(self._position_float(self._canonical_position(current_pos)), (1, 0, 2, 3))
        positions = ttnn.add(start, self.chunk_offsets[:, :, :seq, :])

        def heads(value, count, width):
            value = self._reshape_logical(value, (batch, seq, count, width))
            return ttnn.permute(value, (0, 2, 1, 3))

        projections = self._linear(x, self.packed_projection)
        qkv = ttnn.reshape(projections[..., :7168], (batch, seq, 7168))
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=24, num_kv_heads=2, transpose_key=False
        )
        rope_cos, rope_sin = _shared_rope_tables(self, positions, batch, seq)
        q = _shared_apply_rope(self, self._norm(q, self.nq), rope_cos, rope_sin)
        k = _shared_apply_rope(self, self._norm(k, self.nk), rope_cos, rope_sin)
        gate = projections[..., 7168:13312]
        index = projections[..., 13312:]
        iq = _shared_apply_rope(self, self._norm(heads(index[..., :512], 4, 128), self.niq), rope_cos, rope_sin)
        raw = index[..., 512:640]
        self._write_packed(k, v, positions_packed, page_table)
        self._transfer(raw, self.raw_key_cache, positions_packed, page_table, mode=0, sequence=seq, heads=1)
        groups = self._transfer(
            self.raw_key_cache,
            self._allocate((batch * seq, 1, 4, 128)),
            positions_packed,
            page_table,
            mode=1,
            sequence=seq,
            heads=1,
        )
        pooled = ttnn.typecast(ttnn.mean(ttnn.typecast(groups, ttnn.float32), dim=2, keepdim=True), ttnn.bfloat16)
        pooled = self._reshape_logical(pooled, (batch, 1, seq, 128))
        block_start = ttnn.multiply(ttnn.floor(ttnn.multiply(positions, 0.25)), 4.0)
        pooled = self._chunk_rope(self._norm(pooled, self.nik), block_start)
        self._transfer(pooled, self.pooled_keys, positions_packed, page_table, mode=2, sequence=seq, heads=1)
        qq = ttnn.to_layout(ttnn.concat((self.indexer_prefix, iq), dim=2), ttnn.TILE_LAYOUT)
        padded = (seq + 31 + 31) // 32 * 32
        qq = ttnn.pad(qq, [(0, 0), (0, 0), (0, padded - seq - 31), (0, 0)], 0)
        score_rows = []
        for lane in range(batch):
            score_rows.append(
                ttnn.experimental.indexer_score_dsa(
                    qq[lane : lane + 1],
                    self.pooled_keys,
                    self.indexer_weights[lane : lane + 1, :, :padded, :],
                    chunk_start_idx=self.blocks - 32,
                    key_compression_ratio=1,
                    cache_batch_idx=lane,
                    program_config=ttnn.IndexerScoreProgramConfig(
                        q_chunk_size=32, k_chunk_size=self.indexer_chunk, head_group_size=4
                    ),
                    compute_kernel_config=self.indexer_compute,
                )
            )
        scores = score_rows[0] if batch == 1 else ttnn.concat(score_rows, dim=0)
        scores = scores[:, :, 31 : 31 + seq, :]
        scores = ttnn.to_layout(ttnn.typecast(scores, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        bound = self._mask_scores(scores, positions_packed, seq)
        selected = ttnn.experimental.topk_large_indices(scores, k=512, valid_length_tensor=bound)
        out = self._sparse_attention(q, selected, positions_packed, page_table)
        out = self._linear(ttnn.multiply(out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]), self.wo)
        return self._reshape_logical(out, (1, batch, seq, 2560))

    def _mask_scores(self, scores, positions, seq):
        bound = ttnn.empty((1,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        grid = self.device.compute_with_storage_grid_size()
        workers = grid.x * grid.y
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        tensors = [scores, positions, bound]
        args = [self.batch_size, seq, self.blocks, workers]
        for t in tensors:
            args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        k = ttnn.KernelDescriptor(
            kernel_source=_QSA_MASK_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=768,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=768)],
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[k], cbs=[cb]))
        return bound


class OptimizedMoE(FusedMoE):
    dedicated_weight_dtype = ttnn.bfloat4_b
    down_dtype = ttnn.bfloat4_b
    sparse_output_dtype = ttnn.bfloat8_b

    @classmethod
    def from_state_dict(cls, state, **kwargs):
        import torch

        obj = _PackedMoE.from_state_dict.__func__(cls, state, expert_dtype=cls.dedicated_weight_dtype, **kwargs)
        mesh = kwargs["mesh_device"]

        def upload(x, dtype=None, layout=ttnn.TILE_LAYOUT):
            dtype = cls.dedicated_weight_dtype if dtype is None else dtype
            return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=layout, device=mesh)

        gu = state["mlp.experts.gate_up_proj"]
        down = state["mlp.experts.down_proj"]
        obj.expert_gates = [upload(gu[e, : obj.intermediate].T) for e in range(obj.experts)]
        obj.expert_ups = [upload(gu[e, obj.intermediate :].T) for e in range(obj.experts)]
        obj.expert_downs = [upload(down[e].T) for e in range(obj.experts)]
        obj.expert_ids = upload(torch.arange(obj.experts, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        obj.expert_offsets = upload(
            torch.arange(obj.experts, dtype=torch.int32) * 32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        obj.capacity_input = ttnn.zeros(
            (1, 1, obj.experts * 32, obj.hidden), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        obj._dispatch_coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(8)]
        obj._dispatch_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in obj._dispatch_coords])
        if mesh.get_num_devices() != 1 or (obj.experts, obj.hidden) != (512, 2560):
            raise ValueError("Weighted expert reduction is qualified for single-device E512/H2560")
        obj._reduce_mapping = upload(
            torch.zeros((1, obj.experts), dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
        )
        obj.mesh = mesh
        obj.compact_indices = upload(
            torch.arange(10, dtype=torch.int32).reshape(1, 1, 1, 10), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
        )
        obj.sparse_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        obj.active_nnz = None
        for width, cores, block in ((1280, 20, 40), (2560, 40, 20)):
            n = math.ceil(width / 32 / cores)
            obj.programs[width] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(5, cores // 5),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=min(n, 4),
                out_block_h=1,
                out_block_w=n,
                per_core_M=1,
                per_core_N=n,
                fuse_batch=False,
                mcast_in0=True,
            )
        obj.shared_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        obj.shared_programs = {}
        for width, k in ((1824, 2560), (2560, 640)):
            obj.shared_programs[width] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(10, 8),
                in0_block_w=k // 32,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=1,
                per_core_N=1,
                fuse_batch=False,
                mcast_in0=True,
            )
        return obj

    def _shared_linear(self, x, weight):
        return ttnn.linear(
            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG),
            weight,
            program_config=self.shared_programs[weight.shape[-1]],
            compute_kernel_config=self.shared_compute,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _sparse(self, x, w, sparsity, width, *, input_sparse=False):
        return ttnn.sparse_matmul(
            x,
            w,
            sparsity=sparsity,
            nnz=self.active_nnz,
            is_input_a_sparse=input_sparse,
            is_input_b_sparse=True,
            program_config=self.programs[width],
            compute_kernel_config=self.sparse_compute,
            dtype=self.sparse_output_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _group(self, x):
        router_shared = self._shared_linear(x, self.router_shared_packed)
        logits = router_shared[..., :512]
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        sparsity = ttnn.to_layout(ttnn.max(routing, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        packed = ttnn.reshape(
            self._sparse(x, self.gate_up, sparsity, self.intermediate * 2), (1, self.experts, 32, self.intermediate * 2)
        )
        gate, up = (packed[..., : self.intermediate], packed[..., self.intermediate :])
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        down = self._sparse(hidden, self.down, sparsity, self.hidden, input_sparse=True)
        del hidden, gate, up
        del packed
        routing = ttnn.permute(routing, (0, 3, 2, 1))
        weighted = ttnn.multiply(down, routing, output_tensor=down)
        result = ttnn.sum(weighted, dim=1, keepdim=True)
        sg, su = (router_shared[..., 512:1152], router_shared[..., 1152:1792])
        scale = router_shared[..., 1792:1793]
        shared = self._shared_linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))

    def _dispatch_input(self, x, counts):
        source = ttnn.to_layout(x[..., :1, :], ttnn.ROW_MAJOR_LAYOUT)
        args = (
            ttnn.TensorAccessorArgs(source).get_compile_time_args()
            + ttnn.TensorAccessorArgs(self.capacity_input).get_compile_time_args()
            + ttnn.TensorAccessorArgs(counts).get_compile_time_args()
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=_ONE_ROW_DISPATCH_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self._dispatch_cores,
            compile_time_args=args,
            runtime_args=[
                (
                    (c.x, c.y),
                    [source.buffer_address(), self.capacity_input.buffer_address(), counts.buffer_address(), i],
                )
                for i, c in enumerate(self._dispatch_coords)
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=self._dispatch_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
            )
            for index, size, dtype in ((0, 5120, ttnn.bfloat16), (1, 2048, ttnn.uint32))
        ]
        ttnn.generic_op([source, self.capacity_input, counts], ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs))
        return self.capacity_input

    def _counts(self, indices, scores):
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.empty((512,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh)
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        tensors = [ids, scores, counts]
        args = []
        for t in tensors:
            args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        k = ttnn.KernelDescriptor(
            kernel_source=_ROUTING_COUNTS_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core,
            compile_time_args=args,
            runtime_args=[((0, 0), [t.buffer_address() for t in tensors])],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=2304,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=2304)],
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[k], cbs=[cb]))
        return (counts, ids, scores)

    def _compact(self, down, indices, selected):
        scores = selected
        compact = ttnn.empty(
            (1, 10, 32, self.hidden),
            dtype=down.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        coords = [ttnn.CoreCoord(i % 10, i // 10) for i in range(80)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        tensors = [down, indices, scores, compact]
        args = []
        for t in tensors:
            args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        kernel = ttnn.KernelDescriptor(
            kernel_source=_COMPACT_EXPERT_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=1280,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=1280)],
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
        scores = scores[:, :, :, :10]
        reduced = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            compact,
            self.compact_indices,
            self._reduce_mapping,
            reduce_dim=1,
            split_size=self.hidden,
            cluster_axis=0,
            scores_tensor=scores,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )[0]
        return ttnn.typecast(reduced, ttnn.bfloat16)

    def _group_unified(self, x):
        router_input = x[:, :, :1, :]
        router_shared = self._shared_linear(router_input, self.router_shared_packed)
        logits = router_shared[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        counts, indices, selected = self._counts(indices, selected)
        dispatched = self._dispatch_input(x, counts)
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            dispatched,
            self.expert_offsets,
            counts,
            self.expert_ids,
            self.expert_gates,
            self.expert_ups,
            self.expert_downs,
            32,
        )
        result = self._compact(down, indices, selected)
        sg, su = (router_shared[..., 512:1152], router_shared[..., 1152:1792])
        scale = router_shared[..., 1792:1793]
        shared = self._shared_linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))


class OptimizedHyperConnection(FusedHyperConnection):
    weight_dtype = ttnn.bfloat8_b
    fidelity = ttnn.MathFidelity.LoFi
    norm_cores = 20
    dram_readers = 1
    dram_cores = (5, 2)
    dram_blocks = (64, 5)
    dram_result_memory = "l1"

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj.mesh = kwargs["mesh_device"]
        for name in ("read_packed", "up"):
            w = getattr(obj, name)
            if w.dtype != cls.weight_dtype:
                setattr(obj, name, ttnn.typecast(w, cls.weight_dtype))
        obj.hc_compute = ttnn.init_device_compute_kernel_config(
            obj.mesh.arch(),
            math_fidelity=cls.fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        if cls.dram_readers:
            obj.device = obj.mesh
            obj.projection_compute = obj.hc_compute
            names = ("read_packed", "up")
            cores = cls.dram_cores
            blocks = cls.dram_blocks
            _configure_dram(obj, names, cores=cores, blocks=blocks, readers=cls.dram_readers, activations=None)
        if cls.norm_cores:
            n = cls.norm_cores
            gx = min(n, 10)
            gy = n // gx
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, obj.width // n], ttnn.ShardOrientation.ROW_MAJOR),
            )
            obj.norm_sharded = ttnn.to_memory_config(obj.norm_packed, mem)
        return obj

    def _project(self, x, w, activation=None):
        if self.dram_readers and x.shape[1] == 1 and (x.shape[2] == 1):
            y = _dram_linear(self, x, w)
            return ttnn.sigmoid(y) if activation else y
        if x.shape[-2] > 4 and math.prod(tuple(x.shape)[:-2]) == 1:
            # Explicit 2D prefill geometry; keep this small HC intermediate in L1.
            k, n = tuple(w.shape)[-2:]
            block = min(4, k // 32)
            while k // 32 % block:
                block -= 1
            gx, gy = min(10, math.ceil(n / 32)), min(10, math.ceil(x.shape[-2] / 32))
            pn, pm = math.ceil(n / 32 / gx), math.ceil(x.shape[-2] / 32 / gy)
            sub = min(pn, 4)
            while pn % sub:
                sub -= 1
            program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=sub,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
            )
            return ttnn.linear(
                ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG),
                w,
                activation=activation,
                program_config=program,
                compute_kernel_config=self.hc_compute,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        return ttnn.linear(x, w, activation=activation, compute_kernel_config=self.hc_compute)

    def _norm_candidate(self, x):
        # The width-sharded tile contains one four-stream row group. Larger
        # batches use the native interleaved norm without two layout copies.
        if not self.norm_cores or x.shape[1] != 1:
            return ttnn.rms_norm(x, epsilon=self.eps, compute_kernel_config=self.compute)
        n = self.norm_cores
        gx = min(n, 10)
        gy = n // gx
        k = self.width // 32 // n
        height = 1
        for size in tuple(x.padded_shape)[:-1]:
            height *= size
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [height, self.width // n], ttnn.ShardOrientation.ROW_MAJOR),
        )
        prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy],
            subblock_w=2 if k % 2 == 0 else 1,
            block_h=height // 32,
            block_w=k,
            inplace=False,
        )
        y = ttnn.rms_norm(
            ttnn.to_memory_config(x, mem),
            epsilon=self.eps,
            compute_kernel_config=self.compute,
            program_config=prog,
            memory_config=mem,
        )
        return y if x.shape[1] == 1 else ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

    def _read_prefill(self, x):
        streams = [
            ttnn.rms_norm(self._part(x, i), weight=self.norm[i], epsilon=self.eps, compute_kernel_config=self.compute)
            for i in range(self.count)
        ]
        normalized = ttnn.concat(streams, dim=-1)
        packed = self._project(normalized, self.read_packed)
        mix = ttnn.silu(packed[..., :320])
        mix = self._project(mix, self.up, activation="sigmoid")
        mixed = ttnn.multiply(self._part(mix, 0), streams[0])
        for i in range(1, self.count):
            mixed = ttnn.add(mixed, ttnn.multiply(self._part(mix, i), streams[i]))
        inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return (ttnn.multiply(mixed, 1 / self.count), inject)

    def read(self, x):
        if x.shape[2] > 4:
            return self._read_prefill(x)
        b, seq = tuple(x.shape)[1:3]
        retain = b * seq == 1
        streams = ttnn.reshape(x, (1, b * seq, self.count, self.width))
        streams = ttnn.multiply(self._norm_candidate(streams), self.norm_sharded if retain else self.norm_packed)
        flat_source = ttnn.to_memory_config(streams, ttnn.L1_MEMORY_CONFIG) if retain else streams
        normalized = ttnn.reshape(flat_source, (1, b, seq, self.count * self.width))
        packed = self._project(normalized, self.read_packed)
        mix = ttnn.silu(packed[..., :320])
        mix = self._project(mix, self.up, activation="sigmoid")
        mix = ttnn.reshape(mix, (1, b * seq, self.count, self.width))
        if retain:
            mix = ttnn.to_memory_config(mix, streams.memory_config())
        mixed = ttnn.multiply(mix, streams)
        if retain:
            mixed = ttnn.to_memory_config(mixed, ttnn.L1_MEMORY_CONFIG)
        mixed = ttnn.mean(mixed, dim=2, keepdim=True)
        mixed = ttnn.reshape(mixed, (1, b, seq, self.width))
        inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return (mixed, inject)


class OptimizedPLE(FusedPLE):
    norm_cores = 40

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj.device = kwargs["mesh_device"]
        obj.packed_projection = ttnn.typecast(obj.packed_projection, ttnn.bfloat8_b)
        obj.projection_compute = ttnn.init_device_compute_kernel_config(
            obj.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        return obj

    def _project(self, x):
        if x.shape[-2] == 1 and math.prod(tuple(x.shape)[:-2]) == 1:
            return ttnn.linear(
                ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG),
                self.packed_projection,
                compute_kernel_config=self.projection_compute,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        return _prefill_linear(x, self.packed_projection, self.projection_compute)

    def forward(self, x, *, tokens, positions, slots, valid_lengths, cache):
        batch, seq = tokens.shape
        if tuple(x.shape) != (1, batch, seq, 10240):
            raise ValueError("PLE x must have shape [1,B,S,10240]")
        ws = self.workspaces[(batch, seq)]
        ngram_indices(
            tokens,
            positions,
            slots,
            cache.token_history,
            valid_lengths=valid_lengths,
            output=ws.indices,
            multipliers=self.multipliers,
            primes=self.primes,
            offsets=self.offsets,
            eos=self.eos,
        )
        _lookup_ple_rows(self.tables, ws.indices, output=ws.embeddings)
        n = batch * seq
        emb = ttnn.reshape(ttnn.to_layout(ws.embeddings, ttnn.TILE_LAYOUT), (1, 1, n, 2560))
        projections = self._project(emb)
        key = self._norm(projections[..., :10240], self.norm_key, n)
        query = self._norm(x, self.norm_query, n)
        # Gate definition: sign(g) * sqrt(max(abs(g), 1e-6)).
        # The fused unary chain removes intermediate BF16 packs; matched
        # correctness checks qualify this rounding change.
        dot = ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True)
        gate = ttnn.multiply(dot, 2560**-0.5)
        gate = ttnn.multiply(
            gate,
            gate,
            input_tensor_a_activations=[
                ttnn.UnaryOpType.ABS,
                ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 1e-6),
                ttnn.UnaryOpType.SQRT,
            ],
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGN],
        )
        value = projections[..., 10240:]
        value = ttnn.reshape(value, (1, n, 1, 2560))
        gated = ttnn.reshape(
            ttnn.multiply(gate, value, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]), (1, 1, n, 10240)
        )
        normalized = self._norm(gated, self.norm_conv, n)
        normalized = ttnn.to_layout(ttnn.reshape(normalized, (n, 10240)), ttnn.ROW_MAJOR_LAYOUT)
        # The gather zeros padding, resets only position-zero requests, and
        # updates all nine historical vectors after reading all old taps.
        _conv_taps(
            normalized, positions, slots, valid_lengths, cache.conv_history, output=ws.taps, batch=batch, seq=seq
        )
        taps = ttnn.to_layout(ws.taps, ttnn.TILE_LAYOUT)
        # Reduce the four tap planes directly; generic sum transposes and
        # tile-pads this non-spatial dimension, causing a large DRAM peak.
        convolved = ttnn.experimental.fast_reduce_nc(
            ttnn.multiply(ttnn.typecast(taps, ttnn.float32), self.conv),
            dims=[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )
        convolved = ttnn.typecast(convolved[..., :n, :], ttnn.bfloat16)
        # Mask gated values via the same device controls; tap index 3 is the
        # normalized current token, so cannot be used as a numerical mask.
        gated_rm = ttnn.to_layout(ttnn.reshape(gated, (n, 10240)), ttnn.ROW_MAJOR_LAYOUT)
        _mask_padding(gated_rm, valid_lengths, batch=batch, seq=seq)
        result = ttnn.add(
            ttnn.to_layout(gated_rm, ttnn.TILE_LAYOUT),
            ttnn.reshape(convolved, (n, 10240)),
            input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
        )
        return ttnn.reshape(result, (1, batch, seq, 10240))

    def _norm(self, x, weight, n):
        x = ttnn.typecast(ttnn.reshape(x, (1, n, 4, 2560)), ttnn.float32)
        if not self.norm_cores or n != 1:
            normed = ttnn.rms_norm(x, epsilon=self.eps, compute_kernel_config=self.compute)
            return ttnn.multiply(normed, weight, dtype=ttnn.bfloat16)
        count = self.norm_cores
        gx = min(10, count)
        gy = count // gx
        kw = 80 // count
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 2560 // count], ttnn.ShardOrientation.ROW_MAJOR),
        )
        prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy],
            subblock_w=2 if kw % 2 == 0 else 1,
            block_h=1,
            block_w=kw,
            inplace=False,
        )
        normed = ttnn.rms_norm(
            ttnn.to_memory_config(x, mem),
            epsilon=self.eps,
            program_config=prog,
            compute_kernel_config=self.compute,
            memory_config=mem,
        )
        normed = ttnn.multiply(normed, weight, dtype=ttnn.bfloat16, memory_config=mem)
        return ttnn.to_memory_config(normed, ttnn.L1_MEMORY_CONFIG)


def _configure_dram(obj, names, *, cores, blocks, readers, activations=None):
    mesh = obj.device
    banks = mesh.dram_grid_size().x
    dg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))])
    obj.dram_projections = {}
    for index, (name, count, block) in enumerate(zip(names, cores, blocks)):
        role_readers = readers[index] if isinstance(readers, (tuple, list)) else readers
        w = getattr(obj, name)
        k, n = tuple(w.shape)[-2:]
        assert k // 32 % count == 0 and k // 32 // count % block == 0, (name, k, count, block)
        shard_n = math.ceil(n / (32 * banks * role_readers)) * 32 * role_readers
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dg, [k, shard_n], ttnn.ShardOrientation.ROW_MAJOR),
        )
        # Setup-only repacking; the kernel keeps logical N and zero padding internal.
        wdram = ttnn.from_torch(
            ttnn.to_torch(w), dtype=w.dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mem
        )
        coords = [ttnn.CoreCoord(i % min(8, count), i // min(8, count)) for i in range(count)]
        ag = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        imem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(ag, [32, k // count], ttnn.ShardOrientation.ROW_MAJOR),
        )
        program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=block,
            per_core_M=1,
            per_core_N=shard_n // 32 // role_readers,
            num_workers_per_dram_bank=role_readers,
            fused_activation=activations.get(name) if activations else None,
        )
        obj.dram_projections[w.buffer_address()] = (wdram, imem, program)


def _dram_linear(obj, x, weight):
    compute = getattr(obj, "projection_computes", {}).get(weight.buffer_address(), obj.projection_compute)
    if x.shape[-2] != 1 or math.prod(tuple(x.shape)[:-2]) != 1:
        return _prefill_linear(x, weight, compute)
    w, mem, program = obj.dram_projections[weight.buffer_address()]
    original = tuple(x.shape)
    x = ttnn.reshape(x, (1, 1, 1, original[-1]))
    x = ttnn.to_memory_config(x, mem)
    y = ttnn.linear(
        x,
        w,
        program_config=program,
        compute_kernel_config=compute,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    result_memory = getattr(obj, "dram_result_memory", "dram")
    if result_memory != "sharded":
        y = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG if result_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(y, tuple(original[:-1]) + (weight.shape[-1],))


def _prefill_linear(x, weight, compute):
    """Explicit 2D geometry for large prefill; tile padding remains internal."""
    if x.shape[-2] < 256 or math.prod(tuple(x.shape)[:-2]) != 1:
        return ttnn.linear(x, weight, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    m = math.ceil(x.shape[-2] / 32)
    n = math.ceil(weight.shape[-1] / 32)
    gy, gx = min(m, 10), min(n, 10)
    pm, pn = math.ceil(m / gy), math.ceil(n / gx)
    subblock = min(4, pn)
    while pn % subblock:
        subblock -= 1
    program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=subblock,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
    )
    return ttnn.linear(
        x, weight, program_config=program, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _inplace_recurrence(self, q, k, v, g, beta):
    """Exact token recurrence; prefill keeps FP32 state in aliased local buffers."""
    batch, length = q.shape[0], q.shape[1]
    # Query normalization already applied the scale as its output activation.
    decay = g  # Already exponentiated by the multiply output activation.
    output = ttnn.empty(
        (batch, 48, length, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final = self.recurrent_state
    tensors = [q, k, v, decay, beta, self.recurrent_state, output, final]
    grid = self.device.compute_with_storage_grid_size()
    workers = min(batch * 48, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    assignments = [
        (i * (batch * 48) // workers, (i + 1) * (batch * 48) // workers - i * (batch * 48) // workers)
        for i in range(workers)
    ]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
    )
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_PRECISE_GDN_READER if length == 1 else _PRECISE_PREFILL_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            config=ttnn.ReaderConfigDescriptor(),
            runtime_args=[
                ((c.x, c.y), [*[t.buffer_address() for t in tensors], *job, length])
                for c, job in zip(coords, assignments)
            ],
        ),
        ttnn.KernelDescriptor(
            kernel_source=_PRECISE_GDN_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            config=config,
            runtime_args=[((c.x, c.y), [job[1] * length]) for c, job in zip(coords, assignments)],
        ),
    ]
    sizes = [16, 4, 4, 4, 1, 1, 1, 16, 16, 4, 1, 4, 16, 4, 1]
    if length > 1:
        sizes.append(4)
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i, count in enumerate(sizes)
        if not (length > 1 and i in (0, 12))
    ]
    if length > 1:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=16 * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096) for i in (0, 12)
                ],
            )
        )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    return output, final


_PARALLEL_STATE_SOURCE = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Request-slot gather/scatter of complete interleaved state tiles. No numeric
// conversion: FP32 recurrence and BF16 convolution state retain their bits.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t pages = get_compile_time_arg_val(0);
    constexpr uint32_t bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr bool store = get_compile_time_arg_val(3);
    constexpr uint32_t workers = get_compile_time_arg_val(4);
    constexpr auto ba = TensorAccessorArgs<5>();
    constexpr auto wa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto bank = TensorAccessor(ba, get_arg_val<uint32_t>(0));
    const auto work = TensorAccessor(wa, get_arg_val<uint32_t>(1));
    const auto slot = TensorAccessor(sa, get_arg_val<uint32_t>(2));
    const auto pos = TensorAccessor(pa, get_arg_val<uint32_t>(3));
    const uint32_t lane = get_arg_val<uint32_t>(4);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    Noc noc;
    noc.async_read(slot, CoreLocalMem<uint32_t>(base + bytes), 128, {.page_id = lane}, {});
    noc.async_read(pos, CoreLocalMem<uint32_t>(base + bytes + 128), 128, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t request = words[bytes / 4];
    const uint32_t position = words[bytes / 4 + 32];
    ASSERT(request < slots);
    if (!store && position == 0) {
        for (uint32_t j = 0; j < bytes / 4; ++j) {
            words[j] = 0;
        }
    }
    for (uint32_t tile = get_arg_val<uint32_t>(5); tile < pages; tile += workers) {
        if constexpr (store) {
            noc.async_read(work, CoreLocalMem<uint32_t>(base), bytes, {.page_id = lane * pages + tile}, {});
            noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base), bank, bytes, {}, {.page_id = request * pages + tile});
        } else {
            if (position != 0) {
                noc.async_read(bank, CoreLocalMem<uint32_t>(base), bytes, {.page_id = request * pages + tile}, {});
                noc.async_read_barrier();
            }
            noc.async_write(CoreLocalMem<uint32_t>(base), work, bytes, {}, {.page_id = lane * pages + tile});
        }
        noc.async_write_barrier();
    }
}
"""


_COMPACT_EXPERT_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto da=TensorAccessorArgs<0>();
    constexpr auto ia=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto down=TensorAccessor(da,get_arg_val<uint32_t>(0));
    const auto indices=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto out=TensorAccessor(oa,get_arg_val<uint32_t>(3));
    const uint32_t col=get_arg_val<uint32_t>(4);
    CircularBuffer cb(0); cb.reserve_back(1);
    const uint32_t base=cb.get_write_ptr();
    Noc noc;
    noc.async_read(indices,CoreLocalMem<uint32_t>(base+1088),128,{.page_id=0},{});
    noc.async_read(scores,CoreLocalMem<uint32_t>(base+1216),64,{.page_id=0},{});
    noc.async_read_barrier();
    auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+1088);
    auto weights=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+1216);
    for(uint32_t e=0;e<10;++e) {
        ASSERT(ids[e]<512);
        if (weights[e]!=0) {
            noc.async_read(down,CoreLocalMem<uint32_t>(base),1088,{.page_id=ids[e]*80+col},{});
            noc.async_read_barrier();
        } else {
            auto words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
            for(uint32_t j=0;j<272;++j) words[j]=0;
        }
        noc.async_write(CoreLocalMem<uint32_t>(base),out,1088,{}, {.page_id=e*80+col});
        noc.async_write_barrier();
    }
}
"""


_ROUTING_COUNTS_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ia=TensorAccessorArgs<0>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(0));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(1));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(2));
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    noc.async_read(ids,CoreLocalMem<uint32_t>(base+2048),128,{.page_id=0},{});
    noc.async_read(scores,CoreLocalMem<uint32_t>(base+2176),64,{.page_id=0},{});
    auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    for(uint32_t i=0;i<512;++i)out[i]=0;
    noc.async_read_barrier();
    auto idx=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+2048);
    auto sc=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+2176);
    for(uint32_t i=0;i<10;++i) {ASSERT(idx[i]<512);if(sc[i]!=0)out[idx[i]]=1;}
    noc.async_write(CoreLocalMem<uint32_t>(base),counts,2048,{}, {.page_id=0});
    noc.async_write_barrier();
}
"""


_ROLL_CONV_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0), S=get_compile_time_arg_val(1), W=get_compile_time_arg_val(2);
    constexpr auto ca=TensorAccessorArgs<3>();
    constexpr auto pa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto cache=TensorAccessor(ca,get_arg_val<uint32_t>(0));
    const auto proj=TensorAccessor(pa,get_arg_val<uint32_t>(1));
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    auto old=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    auto src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+2048);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+4096);
    for(uint32_t page=get_arg_val<uint32_t>(2);page<B*320;page+=W) {
        const uint32_t b=page/320,c=page%320;
        for(uint32_t i=0;i<1024;++i)out[i]=0;
        if(S<3){noc.async_read(cache,CoreLocalMem<uint32_t>(base),2048,{.page_id=page},{});noc.async_read_barrier();}
        uint32_t loaded=0xffffffff;
        for(uint32_t r=0;r<3;++r) {
            const uint32_t history=S+r;
            const bool from_old=history<3;
            const uint32_t row=from_old?history:history-3;
            if(!from_old && loaded!=row/32) {
                noc.async_read(proj,CoreLocalMem<uint32_t>(base+2048),2048,{.page_id=(b*((S+31)/32)+row/32)*320+c},{});
                noc.async_read_barrier();loaded=row/32;
            }
            for(uint32_t side=0;side<2;++side)for(uint32_t j=0;j<16;++j) {
                const uint32_t offset=(row%32/16)*512+side*256+(row%16)*16+j;
                out[side*256+r*16+j]=from_old?old[offset]:src[offset];
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base+4096),cache,2048,{}, {.page_id=page});noc.async_write_barrier();
    }
}
"""


_QSA_ADAPTER_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0), S=get_compile_time_arg_val(1);
    constexpr uint32_t P=get_compile_time_arg_val(2), C=get_compile_time_arg_val(3);
    constexpr uint32_t T=get_compile_time_arg_val(4), L=get_compile_time_arg_val(5);
    constexpr uint32_t mode=get_compile_time_arg_val(6), rowbytes=get_compile_time_arg_val(7);
    constexpr auto aa=TensorAccessorArgs<8>();
    constexpr auto oa=TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    constexpr auto pa=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ta=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto src=TensorAccessor(aa,get_arg_val<uint32_t>(0));
    const auto dst=TensorAccessor(oa,get_arg_val<uint32_t>(1));
    const auto pos=TensorAccessor(pa,get_arg_val<uint32_t>(2));
    const auto table=TensorAccessor(ta,get_arg_val<uint32_t>(3));
    const uint32_t begin=get_arg_val<uint32_t>(4), count=get_arg_val<uint32_t>(5);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();
    constexpr uint32_t tablebytes=C/P*4;
    const uint32_t paddr=base, taddr=base+128, iaddr=taddr+tablebytes, oaddr=iaddr+2048;
    auto pp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(paddr);
    auto pages=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(taddr);
    auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iaddr);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(oaddr);
    Noc noc;
    for(uint32_t job=begin;job<begin+count;++job) {
        const uint32_t b=job/(S*(mode==0?2:1));
        const uint32_t s=job%S;
        ASSERT(b<B);
        noc.async_read(pos,CoreLocalMem<uint32_t>(paddr),128,{.page_id=b},{});
        noc.async_read(table,CoreLocalMem<uint32_t>(taddr),tablebytes,{.page_id=b},{});
        noc.async_read_barrier();
        const uint32_t absolute=pp[0]+s;ASSERT(absolute<C);
        if constexpr(mode==0) {
            const uint32_t g=(job/S)%2;
            const uint32_t physical=pages[absolute/P]*P+absolute%P;ASSERT(physical<T);
            noc.async_read(src,CoreLocalMem<uint32_t>(iaddr),rowbytes,{.page_id=job},{});
            noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(iaddr),dst,rowbytes,{}, {.page_id=2*physical+g});
            noc.async_write_barrier();
        } else {
            noc.async_read(src,CoreLocalMem<uint32_t>(iaddr),2048,{.page_id=b*S+s},{});
            noc.async_read_barrier();
            const uint32_t complete=(absolute+1)/4;
            uint32_t cursor=0;
            for(uint32_t e=0;e<512;++e) {
                const uint32_t block=ids[e];
                if(block>=complete) continue;
                for(uint32_t j=0;j<4;++j) {
                    const uint32_t logical=block*4+j;
                    const uint32_t physical=pages[logical/P]*P+logical%P;ASSERT(physical<T);
                    out[cursor++]=physical*2;
                }
            }
            for(uint32_t logical=complete*4;logical<=absolute;++logical) {
                const uint32_t physical=pages[logical/P]*P+logical%P;ASSERT(physical<T);
                out[cursor++]=physical*2;
            }
            ASSERT(cursor>0 && cursor<=2051);
            for(uint32_t j=cursor;j<L;++j) out[j]=0xffffffff;
            noc.async_write(CoreLocalMem<uint32_t>(oaddr),dst,L*4,{}, {.page_id=(b*2)*S+s});
            noc.async_write_barrier();
            for(uint32_t j=0;j<cursor;++j) out[j]+=1;
            noc.async_write(CoreLocalMem<uint32_t>(oaddr),dst,L*4,{}, {.page_id=(b*2+1)*S+s});
            noc.async_write_barrier();
        }
    }
}
"""


_QSA_MASK_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0),S=get_compile_time_arg_val(1),K=get_compile_time_arg_val(2),W=get_compile_time_arg_val(3);
    constexpr auto sa=TensorAccessorArgs<4>();
    constexpr auto pa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ba=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(0));
    const auto positions=TensorAccessor(pa,get_arg_val<uint32_t>(1));
    const auto bound=TensorAccessor(ba,get_arg_val<uint32_t>(2));
    const uint32_t worker=get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    auto pos=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto values=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+128);
    if(worker==0) {
        uint32_t last=512;
        for(uint32_t b=0;b<B;++b) {
            noc.async_read(positions,CoreLocalMem<uint32_t>(base),128,{.page_id=b},{});noc.async_read_barrier();
            const uint32_t end=(pos[0]+S)/4;if(end>last)last=end;
        }
        ASSERT(last<=K);
        auto result=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+640);
        for(uint32_t i=0;i<8;++i) { result[i]=0; }
        result[0]=last;
        noc.async_write(CoreLocalMem<uint32_t>(base+640),bound,32,{}, {.page_id=0});noc.async_write_barrier();
    }
    constexpr uint32_t chunks=(K+255)/256;
    uint32_t loaded=0xffffffff,start=0;
    for(uint32_t job=worker;job<B*S*chunks;job+=W) {
        const uint32_t row=job/chunks,b=row/S,col=(job%chunks)*256;
        if(loaded!=b) {noc.async_read(positions,CoreLocalMem<uint32_t>(base),128,{.page_id=b},{});noc.async_read_barrier();start=pos[0];loaded=b;}
        const uint32_t complete=(start+row%S+1)/4;
        const uint32_t count=K-col<256?K-col:256;
        if(col+count<=complete)continue;
        uint32_t first=0;
        if(col<complete) {
            noc.async_read(scores,CoreLocalMem<uint32_t>(base+128),count*2,{.page_id=row,.offset_bytes=col*2},{});
            noc.async_read_barrier();first=complete-col;
        }
        for(uint32_t i=first;i<count;++i)values[i]=0xff80;
        noc.async_write(CoreLocalMem<uint32_t>(base+128),scores,count*2,{}, {.page_id=row,.offset_bytes=col*2});noc.async_write_barrier();
    }
}
"""

_ONE_ROW_DISPATCH_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto ya=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<ya.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto y=TensorAccessor(ya,get_arg_val<uint32_t>(1));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(2));
    const uint32_t worker=get_arg_val<uint32_t>(3);
    CircularBuffer row_cb(0),count_cb(1);row_cb.reserve_back(1);count_cb.reserve_back(1);Noc noc;
    noc.async_read(x,CoreLocalMem<uint32_t>(row_cb.get_write_ptr()),5120,{.page_id=0},{});
    noc.async_read(counts,CoreLocalMem<uint32_t>(count_cb.get_write_ptr()),2048,{.page_id=0},{});
    noc.async_read_barrier();
    const volatile uint32_t* active=reinterpret_cast<volatile uint32_t*>(count_cb.get_write_ptr());
    for(uint32_t expert=worker;expert<512;expert+=8) {
        if(active[expert]==0)continue;
        noc.async_write(CoreLocalMem<uint32_t>(row_cb.get_write_ptr()),y,5120,{}, {.page_id=expert*32});
    }
    noc.async_write_barrier();
}
"""


def _lookup_ple_rows(tables, indices, *, output):
    rows = tables[0].shape[-2]
    n = indices.shape[0]
    mesh = indices.device()
    grid = mesh.compute_with_storage_grid_size()
    _, cores, group1, _, work1, work2 = ttnn.split_work_to_cores(grid, n * 16, True)
    work = []
    start = 0
    for i, core in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
        count = work1 if i < group1.num_cores() else work2
        work.append(
            (
                (core.x, core.y),
                [
                    indices.buffer_address(),
                    output.buffer_address(),
                    start,
                    count,
                    *[t.buffer_address() for t in tables],
                ],
            )
        )
        start += count
    args = [
        rows,
        len(tables),
        *ttnn.TensorAccessorArgs(indices).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(tables[0]).get_compile_time_args(),
    ]
    cb = ttnn.CBDescriptor(
        total_size=1024,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=1024)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=_PLE_LOOKUP_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=work,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([*tables, indices, output], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
    return output


_PLE_LOOKUP_SOURCE = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Decode BFP4_b exactly as blockfloat_common.cpp, directly into BF16 bits.
static uint16_t decode(uint8_t nibble, uint8_t exponent) {
    uint32_t mantissa = nibble & 7;
    if (!mantissa) {
        return 0;
    }
    uint32_t shifts = 0;
    while (!(mantissa & 4)) {
        mantissa <<= 1;
        ++shifts;
    }
    mantissa = (mantissa << 1) & 7;
    return ((nibble >> 3) << 15) | ((exponent - shifts) << 7) | (mantissa << 4);
}

void kernel_main() {
    Noc noc;
    constexpr uint32_t rows_per_shard = get_compile_time_arg_val(0);
    constexpr uint32_t num_shards = get_compile_time_arg_val(1);
    constexpr auto indices_args = TensorAccessorArgs<2>();
    constexpr auto output_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto indices = TensorAccessor(indices_args, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1));
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t scratch = cb.get_write_ptr();
    auto ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    auto tile = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch + 64);
    auto out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 640);
    for (uint32_t work = start; work < start + count; ++work) {
        const uint32_t token=work/16, head=work%16;
        noc.async_read(indices, CoreLocalMem<uint32_t>(scratch), 64, {.page_id = token}, {});
        noc.async_read_barrier();
        {
            const uint32_t index = ids[head];
            const uint32_t shard = index / rows_per_shard;
            ASSERT(shard < num_shards);
            const uint32_t row = index % rows_per_shard;
            const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(4 + shard));
            for (uint32_t col_tile = 0; col_tile < 5; ++col_tile) {
                noc.async_read(
                    table, CoreLocalMem<uint32_t>(scratch + 64), 576, {.page_id = (row / 32) * 5 + col_tile}, {});
                noc.async_read_barrier();
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t face = (row % 32 / 16) * 2 + c / 16;
                    const uint32_t face_row = row % 16;
                    const uint32_t elem = face * 256 + face_row * 16 + c % 16;
                    const uint8_t packed = tile[64 + elem / 2];
                    const uint8_t nibble = (packed >> (4 * (elem % 2))) & 15;
                    out[col_tile * 32 + c] = decode(nibble, tile[face * 16 + face_row]);
                }
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(scratch + 640), output, 320, {}, {.page_id = token, .offset_bytes = head * 320});
        noc.async_write_barrier();
    }
}

"""
