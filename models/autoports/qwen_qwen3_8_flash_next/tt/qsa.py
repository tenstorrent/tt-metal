# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 indexed attention, with device-resident QSA selection.

Public activations are [1, batch, sequence, 2560]. Decode consumes one token.
current_pos is an INT32 ROW_MAJOR [batch] device tensor (absolute token indices),
page_table is INT32 ROW_MAJOR [batch, max_context/page_size]. Pages can be
arbitrarily permuted but must belong exclusively to their request. Per-request
indexer state belongs to the batch slot; moving a request must move its indexer
state as well. Sequential continuation is required. Position zero automatically
starts a fresh request; stale future cache entries are masked on device.

Bounded prefill chunks vectorize projections and index selection; a model-local
dataflow kernel gathers selected K/V through device page tables. There are no
host tensor conversions in forward. See qsa_notes.md for measured coverage.
"""

from pathlib import Path

import ttnn


class QSA:
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch_size=1,
        max_context=None,
        page_size=32,
        max_requests=None,
        allocate_kv_cache=True,
        allocate_block_ids=True,
        attention_pool_tokens=None,
    ):
        """Load canonical HF attention keys; all torch work is setup-only."""
        import torch

        config = getattr(hf_config, "text_config", hf_config)
        expected = dict(
            hidden_size=2560,
            num_attention_heads=24,
            num_key_value_heads=2,
            head_dim=256,
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
        )
        for name, value in expected.items():
            if getattr(config, name) != value:
                raise ValueError(f"Unsupported target {name}={getattr(config, name)}; expected {value}")
        if config.rope_parameters["rope_type"] != "default":
            raise ValueError("Only the target checkpoint's default RoPE is implemented")
        if config.attention_bias:
            raise ValueError("The target checkpoint has no attention biases")
        obj = cls()
        obj.device, obj.batch_size, obj.eps = mesh_device, batch_size, config.rms_norm_eps
        obj.max_context = max_context if max_context is not None else config.max_position_embeddings
        if not 1 <= obj.max_context <= config.max_position_embeddings:
            raise ValueError("max_context must lie within the HF context contract")
        if page_size % 32 or obj.max_context % page_size or obj.max_context < 2048:
            raise ValueError("Physical capacity must be >=2048 and divisible by tile-aligned page size")
        obj.paged_pooled = attention_pool_tokens is not None
        if obj.paged_pooled and (allocate_kv_cache or page_size != 32):
            raise ValueError("Independent attention pool requires packed cache and 32-token pages")
        if obj.paged_pooled and (attention_pool_tokens <= 0 or attention_pool_tokens % page_size):
            raise ValueError("attention_pool_tokens must be a positive page-aligned physical capacity")
        obj.page_size = page_size
        obj.chunk_size = 128
        obj.blocks = obj.max_context // 4
        prefix = f"model.language_model.layers.{layer_idx}.self_attn."
        # A caller can also pass a canonical attention-only state_dict.
        if prefix + "q_proj.weight" not in state_dict:
            prefix = ""

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def weight(name):
            return state_dict[prefix + name].float()

        def linear(name):
            return upload(weight(name).T)

        # q_proj interleaves query/gate within each 512-wide head, not globally.
        qg = weight("q_proj.weight").reshape(24, 512, 2560)
        obj.wq = upload(qg[:, :256].reshape(6144, 2560).T)
        obj.wgate = upload(qg[:, 256:].reshape(6144, 2560).T)
        obj.wk, obj.wv, obj.wo = [linear(n + ".weight") for n in ("k_proj", "v_proj", "o_proj")]
        obj.windex = linear("indexer.index_qk_proj.weight")
        for attr, name in (
            ("nq", "q_norm"),
            ("nk", "k_norm"),
            ("niq", "indexer.q_layernorm"),
            ("nik", "indexer.k_layernorm"),
        ):
            setattr(obj, attr, upload((1.0 + weight(name + ".weight")).reshape(1, 1, 1, -1), ttnn.float32))
        obj.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        obj.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
        inv = 1.0 / (config.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64))
        angle = torch.arange(obj.max_context).float()[:, None] * inv[None, :]
        angle = torch.cat((angle, angle), -1)
        obj.cos = upload(angle.cos().bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT)
        obj.sin = upload(angle.sin().bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT)
        pages = (attention_pool_tokens if obj.paged_pooled else batch_size * obj.max_context) // page_size
        # ParallelQSA supplies its own packed storage and never consumes these
        # legacy caches. Avoid their full-context construction peak in that path.
        if allocate_kv_cache:
            obj.k_cache = upload(torch.zeros(pages, 2, page_size, 256, dtype=torch.bfloat16))
            obj.v_cache = upload(torch.zeros(pages, 2, page_size, 256, dtype=torch.bfloat16))
        obj.pooled_keys = upload(
            torch.zeros(1 if obj.paged_pooled else batch_size, 1, obj.blocks, 128, dtype=torch.bfloat16)
        )
        if obj.paged_pooled:
            obj.pooled_cache = upload(
                torch.zeros(1, 1, attention_pool_tokens // 4, 128, dtype=torch.bfloat16),
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        obj.raw_key_cache = upload(torch.zeros(pages, 1, page_size, 128, dtype=torch.bfloat16))
        obj.chunk_offsets = upload(torch.arange(obj.chunk_size).float().reshape(1, 1, -1, 1), ttnn.float32)
        obj.tail_offsets = upload(torch.arange(32).float().reshape(1, 1, 1, 32), ttnn.float32)
        obj.max_requests = max_requests
        obj.pooled_bank = (
            upload(torch.zeros(max_requests, 1, obj.blocks, 128, dtype=torch.bfloat16))
            if max_requests is not None and not obj.paged_pooled
            else None
        )
        # ParallelQSA's native selection mask has no block-id tensor input.
        obj.block_ids = (
            upload(torch.arange(obj.blocks).float().reshape(1, 1, 1, -1), ttnn.float32) if allocate_block_ids else None
        )
        return obj

    def _linear(self, x, weight):
        return ttnn.linear(x, weight, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _norm(self, x, weight):
        x = ttnn.typecast(x, ttnn.float32)
        variance = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        normalized = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(variance, self.eps)))
        return ttnn.typecast(ttnn.multiply(normalized, weight), ttnn.bfloat16)

    def _position_float(self, position):
        position = ttnn.typecast(position, ttnn.float32)
        return ttnn.to_layout(ttnn.reshape(position, (1, self.batch_size, 1, 1)), ttnn.TILE_LAYOUT)

    def _position_int(self, position):
        position = ttnn.to_layout(position, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(ttnn.typecast(position, ttnn.int32), (self.batch_size,))

    def _canonical_position(self, position):
        # Shared decoder harness packs one position per row in [B,32].
        if tuple(position.shape) == (self.batch_size, 32):
            position = position[:, :1]
        if tuple(position.shape) not in ((self.batch_size,), (self.batch_size, 1)):
            raise ValueError("current_pos must be [B], [B,1], or packed [B,32]")
        position = ttnn.to_layout(position, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(ttnn.typecast(position, ttnn.int32), (self.batch_size,))

    def decode_forward(self, hidden_states, *, current_pos, page_table, slots=None):
        """One trace-compatible sequential state transition; inputs remain on device."""
        if tuple(hidden_states.shape) != (1, self.batch_size, 1, 2560):
            raise ValueError("decode hidden_states must have shape [1,B,1,2560]")
        self._route_pooled(slots, current_pos, store=False)
        out = self._chunk_forward(hidden_states, current_pos=current_pos, page_table=page_table)
        self._route_pooled(slots, current_pos, store=True)
        return out

    @staticmethod
    def _reshape_logical(x, shape):
        # Real row-major reshapes keep logical tails separate from tile padding.
        return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT)

    def _packed_position(self, position):
        position = self._canonical_position(position)
        position = ttnn.reshape(ttnn.typecast(position, ttnn.uint32), (self.batch_size, 1))
        return ttnn.pad(position, ((0, 0), (0, 31)), 0)

    def _route_pooled(self, slots, position, *, store):
        if slots is None:
            return
        if self.pooled_bank is None:
            raise ValueError("Set max_requests at setup to use device request-slot IDs")
        if tuple(slots.shape) != (self.batch_size, 32) or slots.dtype != ttnn.uint32:
            raise ValueError("slots must be UINT32 ROW_MAJOR [B,32], first column request ID")
        position = self._packed_position(position)
        grid = self.device.compute_with_storage_grid_size()
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(self.batch_size)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        tensors = [self.pooled_bank, self.pooled_keys, slots, position]
        args = [self.blocks // 32 * 4, 2048, self.max_requests, int(store)]
        for tensor in tensors:
            args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        cb = ttnn.CBDescriptor(
            total_size=2304,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=2304)],
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(Path(__file__).with_name("kernels") / "gdn_state.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))

    def _transfer(self, source, output, positions, page_table, *, mode, sequence, heads, indices=None):
        """Raw BF16 tile-row transfer; scalar contents never reach Python."""
        indices = positions if indices is None else indices
        width_tiles = source.shape[-1] // 32
        work = self.batch_size * heads * width_tiles
        if mode in (1, 3):
            work *= sequence
        grid = self.device.compute_with_storage_grid_size()
        _, cores, group1, _, work1, work2 = ttnn.split_work_to_cores(grid, work, True)
        tensors = [source, output, positions, page_table, indices]
        args = [
            mode,
            self.batch_size,
            sequence,
            heads,
            width_tiles,
            self.page_size,
            self.max_context,
            2 if indices.dtype == ttnn.uint16 else 4,
            int(indices.layout == ttnn.TILE_LAYOUT),
            0x51534101,
        ]
        if self.paged_pooled:
            args.append(self.pool_tokens // self.page_size)
        for tensor in tensors:
            args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        runtime = []
        offset = 0
        for i, core in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
            count = work1 if i < group1.num_cores() else work2
            runtime.append(((core.x, core.y), [*[t.buffer_address() for t in tensors], offset, count]))
            offset += count
        scratch_size = 4224 + self.max_context // self.page_size * 4 + 2048 + 4096
        scratch_size = (scratch_size + 31) // 32 * 32
        cb = ttnn.CBDescriptor(
            total_size=scratch_size,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch_size)
            ],
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(
                Path(__file__).with_name("kernels")
                / ("qsa_gather_bounded.cpp" if self.paged_pooled else "qsa_gather.cpp")
            ),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=runtime,
            config=ttnn.ReaderConfigDescriptor(),
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
        return output

    def _allocate(self, shape):
        return ttnn.allocate_tensor_on_device(
            shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )

    def _chunk_rope(self, x, positions):
        batch, _, seq, _ = x.shape
        indices = ttnn.reshape(
            ttnn.typecast(ttnn.to_layout(positions, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32), (batch, seq)
        )
        cos = ttnn.reshape(ttnn.embedding(indices, self.cos, layout=ttnn.TILE_LAYOUT), (batch, 1, seq, 64))
        sin = ttnn.reshape(ttnn.embedding(indices, self.sin, layout=ttnn.TILE_LAYOUT), (batch, 1, seq, 64))
        rotary = ttnn.add(
            ttnn.multiply(x[..., :64], cos),
            ttnn.multiply(ttnn.concat((ttnn.neg(x[..., 32:64]), x[..., :32]), dim=-1), sin),
        )
        return ttnn.concat((rotary, x[..., 64:]), dim=-1)

    def _chunk_forward(self, hidden_states, *, current_pos, page_table):
        """Vectorized projections/selection with bounded selected-KV attention."""
        batch, seq = self.batch_size, hidden_states.shape[-2]
        x = self._reshape_logical(hidden_states, (batch, 1, seq, 2560))
        positions_packed = self._packed_position(current_pos)
        start = ttnn.permute(self._position_float(self._canonical_position(current_pos)), (1, 0, 2, 3))
        positions = ttnn.add(start, self.chunk_offsets[:, :, :seq, :])

        def heads(value, count, width):
            value = self._reshape_logical(value, (batch, seq, count, width))
            return ttnn.permute(value, (0, 2, 1, 3))

        q = self._chunk_rope(self._norm(heads(self._linear(x, self.wq), 24, 256), self.nq), positions)
        k = self._chunk_rope(self._norm(heads(self._linear(x, self.wk), 2, 256), self.nk), positions)
        v = heads(self._linear(x, self.wv), 2, 256)
        gate = self._linear(x, self.wgate)
        index = self._linear(x, self.windex)
        iq = self._chunk_rope(self._norm(heads(index[..., :512], 4, 128), self.niq), positions)
        raw = index[..., 512:640]
        for value, cache, count in ((k, self.k_cache, 2), (v, self.v_cache, 2), (raw, self.raw_key_cache, 1)):
            self._transfer(value, cache, positions_packed, page_table, mode=0, sequence=seq, heads=count)
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
        # TTNN batched matmul cannot broadcast only the head axis when B>1.
        # Fold the four query heads into M; keep each request's K bank separate.
        flat_iq = self._reshape_logical(iq, (batch, 1, 4 * seq, 128))
        scores = ttnn.matmul(flat_iq, ttnn.transpose(self.pooled_keys, -1, -2), compute_kernel_config=self.compute)
        scores = self._reshape_logical(scores, (batch, 4, seq, self.blocks))
        scores = ttnn.sum(ttnn.relu(scores), dim=1, keepdim=True)
        complete = ttnn.floor(ttnn.multiply(ttnn.add(positions, 1.0), 0.25))
        scores = ttnn.where(ttnn.typecast(ttnn.lt(self.block_ids, complete), ttnn.bfloat16), scores, float("-inf"))
        _, selected = ttnn.topk(ttnn.typecast(scores, ttnn.bfloat16), k=512, dim=-1)
        valid = ttnn.lt(ttnn.typecast(selected, ttnn.float32), complete)
        selected_tokens = ttnn.repeat_interleave(valid, 4, dim=3)
        tail_count = ttnn.subtract(ttnn.add(positions, 1.0), ttnn.multiply(complete, 4.0))
        tail = ttnn.lt(self.tail_offsets, tail_count)
        mask = ttnn.typecast(ttnn.where(ttnn.concat((selected_tokens, tail), dim=3), 0.0, float("-inf")), ttnn.bfloat16)
        mask = self._reshape_logical(mask, (batch * seq, 1, 1, 2080))
        gathered = []
        for cache in (self.k_cache, self.v_cache):
            gathered.append(
                self._transfer(
                    cache,
                    self._allocate((batch * seq, 2, 2080, 256)),
                    positions_packed,
                    page_table,
                    mode=3,
                    sequence=seq,
                    heads=2,
                    indices=selected,
                )
            )
        q = self._reshape_logical(ttnn.permute(q, (0, 2, 1, 3)), (batch * seq, 24, 1, 256))
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            gathered[0],
            gathered[1],
            is_causal=False,
            attn_mask=mask,
            scale=0.0625,
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute,
        )
        out = self._reshape_logical(out, (batch, 1, seq, 6144))
        out = self._linear(ttnn.multiply(out, ttnn.sigmoid(gate)), self.wo)
        return self._reshape_logical(out, (1, batch, seq, 2560))

    def prefill_forward(self, hidden_states, *, current_pos, page_table, chunk_size=128, slots=None):
        """Bounded chunks, arbitrary logical S, absolute device-resident prefix position.

        No padding token is executed, so tails cannot overwrite a subsequent live
        cache entry. Host loop indices describe input slices only; absolute cache
        positions are calculated on device. Return every logical output token.
        """
        if len(hidden_states.shape) != 4 or hidden_states.shape[0] != 1 or hidden_states.shape[1] != self.batch_size:
            raise ValueError("prefill hidden_states must have shape [1,B,S,2560]")
        if hidden_states.shape[-1] != 2560 or not 0 < hidden_states.shape[-2] <= self.max_context:
            raise ValueError("Invalid hidden width or logical sequence length")
        if not 1 <= chunk_size <= self.chunk_size:
            raise ValueError("chunk_size must lie in [1,128]")
        self._route_pooled(slots, current_pos, store=False)
        outputs = []
        initial_position = self._position_float(self._canonical_position(current_pos))
        for chunk_start in range(0, hidden_states.shape[-2], chunk_size):
            position = self._position_int(ttnn.add(initial_position, float(chunk_start)))
            outputs.append(
                self._chunk_forward(
                    hidden_states[..., chunk_start : min(chunk_start + chunk_size, hidden_states.shape[-2]), :],
                    current_pos=position,
                    page_table=page_table,
                )
            )
        self._route_pooled(slots, current_pos, store=True)
        return ttnn.concat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
