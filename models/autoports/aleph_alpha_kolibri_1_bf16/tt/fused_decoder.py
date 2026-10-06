# SPDX-License-Identifier: Apache-2.0
"""Kolibri fused decoder: sandwich RMSNorm, GQA/RNoPE or sliding RoPE, sigmoid MoE.

All forward inputs, outputs and KV state are device tensors. Weight conversion
is restricted to from_state_dict. Logical prefill uses a setup-time plan and
performs chunking/padding on device. The streaming chunk entrypoint accepts
page-aligned physical chunks. Decode consumes tensor positions. See the stage
README for complete shapes, cache ownership and trace-lifetime contracts.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

DRAM = ttnn.DRAM_MEMORY_CONFIG


class FusedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        import torch

        self = cls()
        self.config = hf_config
        self.layer_idx = layer_idx
        self.device = mesh_device
        if kwargs:
            raise TypeError(f"Unsupported decoder options: {sorted(kwargs)}")
        if (
            hf_config.hidden_size,
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
            hf_config.num_experts,
            hf_config.num_experts_per_tok,
        ) != (2560, 48, 4, 128, 384, 6):
            raise ValueError("This decoder requires the released Kolibri-1 dimensions")
        if hf_config.norm_topk_prob or hf_config.hidden_act != "silu":
            raise ValueError("Kolibri requires unnormalized sigmoid routing and SwiGLU")
        if hf_config.layer_types[layer_idx] not in ("sliding_attention", "full_attention"):
            raise ValueError("Unknown Kolibri layer kind")
        if hf_config.sliding_window != 513:
            raise ValueError("The released Kolibri sliding window is 513 tokens")
        self.sliding = hf_config.layer_types[layer_idx] == "sliding_attention"
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

        def convert(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
            )

        prefix = f"model.layers.{layer_idx}."
        w = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        self.norms = {
            k: convert(w[k + ".weight"].reshape(1, 1, 1, -1))
            for k in [
                "input_layernorm",
                "post_attn_norm",
                "post_attention_layernorm",
                "post_ffn_norm",
                "self_attn.q_norm",
                "self_attn.k_norm",
            ]
        }
        self.projections = {k: convert(w["self_attn." + k + ".weight"].T) for k in ["o_proj"]}
        self.projections["qkv"] = convert(
            torch.cat([w["self_attn." + k + ".weight"].T for k in ["q_proj", "k_proj", "v_proj"]], dim=-1)
        )
        self.router = convert(w["mlp.gate.weight"].T, ttnn.float32)
        self.expert_bias = convert(w["moe.router.expert_bias"].reshape(1, 1, 1, 384), ttnn.float32)
        self.shared = {k: convert(w["mlp.shared_experts." + k + ".weight"].T) for k in ["down_proj"]}
        self.experts = {}
        for k in ["down_proj"]:
            packed = torch.stack([w[f"mlp.experts.{e}.{k}.weight"].T for e in range(384)])[None]
            self.experts[k] = convert(packed)
        packed = torch.stack(
            [
                torch.cat([w[f"mlp.experts.{e}.{k}.weight"].T for k in ["gate_proj", "up_proj"]], dim=-1)
                for e in range(384)
            ]
        )[None]
        self.experts["gate_up"] = convert(packed)
        self.shared["gate_up"] = convert(
            torch.cat([w["mlp.shared_experts." + k + ".weight"].T for k in ["gate_proj", "up_proj"]], dim=-1)
        )
        # Scatter produces a fresh output, so these setup constants are safe
        # across changing requests and trace replays. Keep row-major variants
        # for the common buckets to avoid converting fixed masks every call.
        self.mask_zero = convert(torch.zeros(1, 1, 128, 384))
        self.mask_one = convert(torch.ones(1, 1, 128, 6))
        self.mask_zero_rm = ttnn.to_layout(self.mask_zero, ttnn.ROW_MAJOR_LAYOUT)
        self.mask_one_rm = ttnn.to_layout(self.mask_one, ttnn.ROW_MAJOR_LAYOUT)
        self.reduce_indices = ttnn.from_torch(
            torch.arange(384, dtype=torch.int32).repeat(32, 1),
            device=self.device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        self.reduce_mapping = ttnn.from_torch(
            torch.zeros(1, 384, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        # Dedicated prefill experts and packed sparse decode experts share HF values.
        self.prefill_experts = {
            role: [convert(w[f"mlp.experts.{e}.{role}.weight"].T) for e in range(384)]
            for role in ["gate_proj", "up_proj", "down_proj"]
        }
        self.expert_ids = ttnn.from_torch(
            torch.arange(384, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        self.expert_sequence = convert(torch.arange(384, dtype=torch.float32).reshape(1, 1, 1, 384), ttnn.float32)
        self.expert_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        return self

    def _norm(self, x, name):
        # Preserve the BF16 rounding boundary before gamma and expert selection.
        normalized = ttnn.rms_norm(
            x, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.compute, memory_config=DRAM
        )
        return ttnn.multiply(normalized, self.norms[name])

    def _linear(self, x, w, *, dtype=ttnn.bfloat16, activation=None):
        return ttnn.linear(
            x,
            w,
            activation=activation,
            dtype=dtype,
            memory_config=DRAM,
            compute_kernel_config=self.compute,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

    def _sparse_config(self, m, n):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 2),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=(m + 31) // 32,
            per_core_N=(n + 511) // 512,
            fuse_batch=False,
            mcast_in0=True,
        )

    def _moe(self, x, *, prefill=False):
        tokens = x.shape[-2]
        logits = self._linear(x, self.router, dtype=ttnn.float32)
        choice = ttnn.add(logits, self.expert_bias)
        _, ids = ttnn.topk(choice, k=6, dim=-1)
        scores = logits
        if tokens <= 128:
            zero, one = self.mask_zero_rm, self.mask_one_rm
        else:
            # Larger public chunk choices remain supported entirely on device.
            repeats = (tokens + 127) // 128
            zero = ttnn.repeat(self.mask_zero, (1, 1, repeats, 1))
            one = ttnn.repeat(self.mask_one, (1, 1, repeats, 1))
        zero = ttnn.slice(zero, (0, 0, 0, 0), (1, 1, tokens, 384))
        one = ttnn.slice(one, (0, 0, 0, 0), (1, 1, tokens, 6))
        selected = ttnn.scatter(zero, dim=-1, index=ids, src=one)
        selected = ttnn.to_layout(selected, ttnn.TILE_LAYOUT)
        routing = ttnn.multiply(scores, selected, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        active = ttnn.max(selected, dim=2, keepdim=True)

        if prefill and tokens >= 64:
            counts = ttnn.reshape(
                ttnn.to_layout(
                    ttnn.typecast(ttnn.multiply(ttnn.typecast(active, ttnn.float32), float(tokens)), ttnn.uint32),
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                (384,),
            )
            offsets = ttnn.reshape(
                ttnn.to_layout(
                    ttnn.typecast(ttnn.multiply(self.expert_sequence, float(tokens)), ttnn.uint32),
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                (384,),
            )
            # Current dedicated op bounds output-only offsets against input capacity.
            # Padding avoids the defect without changing shared code; only the first
            # tokens rows are read (read_x_at_offset=False).
            # Convert the small live prefix before capacity padding; untilizing
            # the expanded allocation would move 384 times as many rows.
            rows = ttnn.to_layout(ttnn.reshape(x, (tokens, 2560)), ttnn.ROW_MAJOR_LAYOUT)
            expanded = ttnn.pad(rows, ((0, 383 * tokens), (0, 0)), 0.0)
            output = ttnn.empty(
                (384 * tokens, 2560),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=DRAM,
            )
            down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                expanded,
                self.prefill_experts["gate_proj"],
                self.prefill_experts["up_proj"],
                self.prefill_experts["down_proj"],
                counts,
                self.expert_ids,
                input_m_tiles=tokens // 32,
                compute_kernel_config=self.expert_compute,
                core_grid=ttnn.CoreCoord(8, 8),
                output=output,
                expert_region_offsets=offsets,
                read_x_at_offset=False,
            )
            down = ttnn.reshape(down, (1, 384, tokens, 2560))
        else:
            sparsity = ttnn.to_layout(active, ttnn.ROW_MAJOR_LAYOUT)

            def sparse(a, w, n, active=False):
                return ttnn.sparse_matmul(
                    a,
                    w,
                    sparsity=sparsity,
                    nnz=None,
                    is_input_a_sparse=active,
                    is_input_b_sparse=True,
                    program_config=self._sparse_config(tokens, n),
                    compute_kernel_config=self.compute,
                    dtype=ttnn.bfloat16,
                    memory_config=DRAM,
                )

            gate_up = ttnn.reshape(sparse(x, self.experts["gate_up"], 1024), (1, 384, tokens, 1024))
            gate = ttnn.slice(gate_up, (0, 0, 0, 0), (1, 384, tokens, 512))
            up = ttnn.slice(gate_up, (0, 0, 0, 512), (1, 384, tokens, 1024))
            middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            down = ttnn.reshape(sparse(middle, self.experts["down_proj"], 2560, True), (1, 384, tokens, 2560))
        # The fused reducer allocates one score tile per expert and row tile.
        # Limit its internal row group to 32 so score tiles fit in L1. Public
        # sequence lengths stay arbitrary; prefill planning pads physical chunks.
        routed_parts = []
        for start in range(0, tokens, 32):
            count = min(32, tokens - start)
            scores_part = ttnn.slice(routing, (0, 0, start, 0), (1, 1, start + count, 384))
            scores_bf16 = ttnn.typecast(scores_part, ttnn.bfloat16)
            scores_rm = ttnn.reshape(ttnn.to_layout(scores_bf16, ttnn.ROW_MAJOR_LAYOUT), (count, 1, 1, 384))
            mask = ttnn.reshape(ttnn.permute(scores_bf16, (0, 3, 2, 1)), (1, 384, count, 1))
            piece = ttnn.slice(down, (0, 0, start, 0), (1, 384, start + count, 2560))
            clean = ttnn.where(mask, piece, 0.0)
            part = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                clean,
                self.reduce_indices,
                self.reduce_mapping,
                1,
                split_size=2560,
                cluster_axis=1,
                scores_tensor=scores_rm,
                compute_kernel_config=self.compute,
            )[0]
            routed_parts.append(part)
        routed = ttnn.concat(routed_parts, dim=2) if len(routed_parts) > 1 else routed_parts[0]
        both = self._linear(x, self.shared["gate_up"])
        width = both.shape[-1] // 2
        sg = ttnn.slice(both, (0, 0, 0, 0), (1, 1, tokens, width))
        su = ttnn.slice(both, (0, 0, 0, width), (1, 1, tokens, 2 * width))
        middle_shared = ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        shared = self._linear(middle_shared, self.shared["down_proj"])
        return ttnn.add(routed, shared, dtype=ttnn.bfloat16)

    def _finish(self, residual, attention, *, prefill=False):
        x = ttnn.add(residual, self._norm(attention, "post_attn_norm"))
        return ttnn.add(
            x, self._norm(self._moe(self._norm(x, "post_attention_layernorm"), prefill=prefill), "post_ffn_norm")
        )

    def _qkv(self, x, cos, sin):
        s = x.shape[-2]
        n = self._norm(x, "input_layernorm")
        packed = self._linear(n, self.projections["qkv"])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(packed, num_heads=48, num_kv_heads=4, transpose_k_heads=False)
        q, k, v = [ttnn.slice(t, (0, 0, 0, 0), (1, h, s, 128)) for t, h in [(q, 48), (k, 4), (v, 4)]]
        q = self._norm(q, "self_attn.q_norm")
        k = self._norm(k, "self_attn.k_norm")
        if self.sliding:
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
        return (q, k, v)

    def _rope(self, x, cos, sin):
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, compute_kernel_config=self.compute)

    def prefill_chunk_forward(
        self, hidden_states, *, kv_cache, page_table, chunk_page_table, chunk_start, cos=None, sin=None
    ):
        """One padded physical chunk [1,1,S,2560], S a multiple of 32.

        chunk_start: INT32 device scalar absolute offset; chunk_page_table:
        pages starting at that offset. Caller retains only valid logical rows.
        page_table maps the entire request, including its cached prefix.
        """
        q, k, v = self._qkv(hidden_states, cos, sin)
        ttnn.experimental.paged_fill_cache(kv_cache[0], k, chunk_page_table)
        ttnn.experimental.paged_fill_cache(kv_cache[1], v, chunk_page_table)
        y = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table,
            chunk_start_idx_tensor=chunk_start,
            sliding_window_size=513 if self.sliding else None,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=32, exp_approx_mode=False
            ),
            compute_kernel_config=self.compute,
        )
        y = ttnn.reshape(ttnn.transformer.concatenate_heads(y), (1, 1, hidden_states.shape[-2], 6144))
        return self._finish(hidden_states, self._linear(y, self.projections["o_proj"]), prefill=True)

    def decode_forward(self, hidden_states, *, kv_cache, page_table, current_pos, cos=None, sin=None):
        """One token per request: [1,1,B,2560], tensor positions [B], B table rows.

        Every op is device-only and can be captured in a TTNN execution trace.
        Refresh stable input/position/RoPE/page-table buffers before replay.
        """
        batch = hidden_states.shape[-2]
        if tuple(hidden_states.shape) != (1, 1, batch, 2560) or not 1 <= batch <= 32:
            raise ValueError("Decode expects [1,1,B,2560], 1 <= B <= 32")
        if tuple(current_pos.shape) != (batch,) or page_table.shape[0] != batch:
            raise ValueError("Decode positions and page-table rows must match batch")
        grid = self.device.compute_with_storage_grid_size()
        # Sharded SDPA assigns batch i to (i % grid.x, i // grid.x).
        # Q/K/V and RoPE must use that exact physical row-major prefix.
        shard_grid = ttnn.num_cores_to_corerangeset(batch, grid, row_wise=True)
        mem = ttnn.create_sharded_memory_config(
            (32, 128), shard_grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )
        n = self._norm(hidden_states, "input_layernorm")
        packed = self._linear(n, self.projections["qkv"])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(packed, num_heads=48, num_kv_heads=4, memory_config=mem)
        q = self._norm(ttnn.to_memory_config(q, DRAM), "self_attn.q_norm")
        k = self._norm(ttnn.to_memory_config(k, DRAM), "self_attn.k_norm")
        if self.sliding:
            qmem = ttnn.create_sharded_memory_config(
                (64, 128), shard_grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
            q = ttnn.to_memory_config(q, qmem)
            k = ttnn.to_memory_config(k, mem)
            c = ttnn.to_memory_config(ttnn.permute(cos, (0, 2, 1, 3)), mem)
            s = ttnn.to_memory_config(ttnn.permute(sin, (0, 2, 1, 3)), mem)
            q = ttnn.experimental.rotary_embedding_hf(q, c, s, is_decode_mode=True, compute_kernel_config=self.compute)
            k = ttnn.experimental.rotary_embedding_hf(k, c, s, is_decode_mode=True, compute_kernel_config=self.compute)
        k = ttnn.to_memory_config(k, mem)
        v = ttnn.to_memory_config(v, mem)
        ttnn.experimental.paged_update_cache(kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table)
        y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table,
            cur_pos_tensor=current_pos,
            sliding_window_size=513 if self.sliding else None,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                q_chunk_size=32,
                k_chunk_size=256,
                max_cores_per_head_batch=16,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.compute,
        )
        y = ttnn.reshape(y, (1, 1, batch, 6144))
        return self._finish(hidden_states, self._linear(y, self.projections["o_proj"]))

    def prepare_prefill(self, *, page_table_host, seq_len, start_pos=0, chunk_size=128):
        """Setup-only plan: bind request page mappings, absolute RoPE and offsets.

        No weights or KV are rebuilt. Prepare plans and their physical programs
        before the first trace. A generator may instead reuse stable tensors with
        prefill_chunk_forward, refreshing their contents at request boundaries.
        """
        import torch

        if not 0 < seq_len or start_pos < 0 or start_pos + seq_len > self.config.max_position_embeddings:
            raise ValueError("Prefill extent is outside the configured context")
        if chunk_size <= 0 or chunk_size % 32:
            raise ValueError("Physical chunk size must be a positive multiple of 32")

        def tt(x, dtype, layout):
            return ttnn.from_torch(x.contiguous(), device=self.device, dtype=dtype, layout=layout, memory_config=DRAM)

        if page_table_host.ndim != 2 or page_table_host.shape[0] < 1:
            raise ValueError("Page table must have at least one request row")
        if page_table_host.dtype != torch.int32 or (page_table_host < 0).any():
            raise ValueError("Page table must contain nonnegative INT32 physical page IDs")
        if page_table_host.shape[1] * 32 < start_pos + seq_len:
            raise ValueError("Page table does not cover the logical prefill extent")
        plans = []
        for slot in range(page_table_host.shape[0]):
            pages = page_table_host[slot : slot + 1]
            table = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            chunks = []
            offset = 0
            while offset < seq_len:
                pos = start_pos + offset
                decode = pos % 32 != 0
                count = 1 if decode else min(chunk_size, seq_len - offset)
                physical = 1 if decode else (count + 31) // 32 * 32
                positions = torch.arange(pos, pos + physical).float()
                phase = positions[:, None] / 10000.0 ** (torch.arange(0, 128, 2).float() / 128)
                phase = torch.cat([phase, phase], -1)[None, None]
                kw = dict(
                    page_table=table,
                    cos=tt(phase.cos().bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                    sin=tt(phase.sin().bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                )
                position = tt(torch.tensor([pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                if decode:
                    kw["current_pos"] = position
                else:
                    kw["chunk_start"] = position
                    kw["chunk_page_table"] = tt(
                        pages[:, pos // 32 : (pos + physical) // 32], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                    )
                chunks.append((offset, count, physical, decode, kw))
                offset += count
            plans.append(chunks)
        return dict(seq_len=seq_len, batch=len(plans), slots=plans)

    def prefill_forward(self, hidden_states, *, kv_cache, plan):
        """Logical prefill [1,B,S,2560] -> [1,B,S,2560].

        Accepts every S>=1 within context, including continuation at arbitrary
        positions. A setup-time plan supplies device page mappings and positions.
        Chunking/padding/slicing stay on device. KV entries before start_pos are
        preserved. Padding only occupies future rows in this request's own pages;
        causal attention cannot see it, and later append overwrites those rows.
        """
        seq_len = plan["seq_len"]
        batch = plan["batch"]
        if tuple(hidden_states.shape) != (1, batch, seq_len, 2560):
            raise ValueError("Prefill input shape does not match its plan")
        results = []
        for slot, chunks in enumerate(plan["slots"]):
            outputs = []
            for offset, count, physical, decode, kw in chunks:
                x = ttnn.slice(hidden_states, (0, slot, offset, 0), (1, slot + 1, offset + count, 2560))
                if physical != count:
                    x = ttnn.pad(x, ((0, 0), (0, 0), (0, physical - count), (0, 0)), value=0.0)
                y = (self.decode_forward if decode else self.prefill_chunk_forward)(x, kv_cache=kv_cache, **kw)
                outputs.append(ttnn.slice(y, (0, 0, 0, 0), (1, 1, count, 2560)))
            results.append(ttnn.concat(outputs, dim=2) if len(outputs) > 1 else outputs[0])
        return ttnn.concat(results, dim=1) if batch > 1 else results[0]
