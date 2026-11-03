# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip K2 Horizon decoder: grouped RMSNorm, gated GQA, MoVA and MoE.

All runtime arguments are device tensors. Weight conversion belongs exclusively
to ``from_state_dict``. Cache ownership and RoPE input preparation belong to the
caller. Prefill accepts [1, B, S, 2560]; decode accepts [1, 1, B, 2560].
Decode positions are INT32 [B], page tables INT32 [B, pages], and caches are
BF16 [physical_pages, 8, page_size, 128]. RoPE cos/sin use HF split-half order.
"""

import math

import ttnn
from models.common.lightweightmodule import LightweightModule


class FusedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, chunk_size=256):
        import torch

        self = cls()
        self.config = c = hf_config
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.chunk_size = chunk_size
        if (c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim) != (2560, 32, 8, 128):
            raise ValueError("Expected the real IFM/K2-Horizon-MoVA-36B-A4B configuration")
        if c.query_key_norm or c.sliding_window is not None or c.rope_head_dim != 128:
            raise ValueError("Unsupported target configuration")
        if not 0 <= layer_idx < c.num_hidden_layers or chunk_size <= 0 or chunk_size % 32:
            raise ValueError("Invalid layer index or physical chunk size")
        if (
            c.mova_num_experts,
            c.mova_num_experts_per_tok,
            c.num_experts,
            c.num_experts_per_tok,
            c.moe_intermediate_size,
            c.num_shared_experts,
            c.layernorm_num_groups,
            c.intermediate_size,
        ) != (64, 4, 100, 8, 768, 1, 2, 6144):
            raise ValueError("Expert and normalization shapes must match the target checkpoint")
        if (
            c.attention_bias
            or c.attention_gate_func != "softplus"
            or c.hidden_act != "silu"
            or c.router_score_func != "sigmoid"
            or c.router_scaling_factor != 2.5
            or not c.norm_topk_prob
            or not c.moe_gate_bias
            or c.decoder_sparse_step != 1
        ):
            raise ValueError("Unsupported attention or router semantics")
        self.sparse = layer_idx not in c.mlp_only_layers
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        prefix = f"model.layers.{layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
        if not sd:
            sd = state_dict
        self.weights = {}

        def put(name, value, dtype=ttnn.bfloat16):
            self.weights[name] = ttnn.from_torch(
                value.contiguous(),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        for name, value in sd.items():
            if ".experts." in name or ".v_experts." in name:
                continue
            if name.endswith("weight") and value.ndim == 2:
                value = value.T
            if value.ndim == 1:
                value = value.reshape(1, 1, 1, -1)
            put(name, value, ttnn.float32 if name.endswith("bias") else ttnn.bfloat16)
        if self.sparse:
            put(
                "value_experts",
                torch.stack([sd[f"self_attn.v_experts.{i}.weight"].T for i in range(c.mova_num_experts)]).unsqueeze(0),
            )
            for proj in ("gate_proj", "up_proj", "down_proj"):
                put(
                    "experts." + proj,
                    torch.stack([sd[f"mlp.experts.{i}.{proj}.weight"].T for i in range(c.num_experts)]).unsqueeze(0),
                )
            # Padded experts must never win top-k.
            for name, experts in (("self_attn.v_router", c.mova_num_experts), ("mlp.gate", c.num_experts)):
                padded = math.ceil(experts / 32) * 32
                bias = torch.full((1, 1, 1, padded), -float("inf"))
                bias[..., :experts] = sd[name + ".bias"].float()
                put(name + ".bias", bias, ttnn.float32)
                w = torch.zeros(c.hidden_size, padded, dtype=torch.bfloat16)
                w[:, :experts] = sd[name + ".weight"].T
                put(name + ".weight", w)
        prefix = "mlp.shared_experts" if self.sparse else "mlp"
        put(
            prefix + ".gate_up.weight",
            torch.cat([sd[prefix + ".gate_proj.weight"].T, sd[prefix + ".up_proj.weight"].T], dim=-1),
        )

        if self.sparse:
            put(
                "experts.gate_up",
                torch.stack(
                    [
                        torch.cat(
                            [sd[f"mlp.experts.{i}.gate_proj.weight"].T, sd[f"mlp.experts.{i}.up_proj.weight"].T], dim=-1
                        )
                        for i in range(c.num_experts)
                    ]
                ).unsqueeze(0),
            )
            for proj in ("gate_proj", "up_proj"):
                self.weights.pop("experts." + proj).deallocate(True)
        gate = sd[prefix + ".gate_proj.weight"].T
        up = sd[prefix + ".up_proj.weight"].T
        width = gate.shape[-1]
        interleaved = torch.stack(
            [gate.reshape(2560, width // 32, 32), up.reshape(2560, width // 32, 32)], dim=2
        ).reshape(2560, 2 * width)
        put(prefix + ".swiglu.weight", interleaved)
        names = ["self_attn.q_proj", "self_attn.k_proj"]
        if not self.sparse:
            names.append("self_attn.v_proj")
        names.append("self_attn.gate_proj")
        put("attention_packed.weight", torch.cat([sd[n + ".weight"].T for n in names], dim=-1))
        for name in names:
            self.weights.pop(name + ".weight").deallocate(True)
        if self.sparse:
            grid = ttnn.num_cores_to_corerangeset(32, mesh_device.compute_with_storage_grid_size(), True)
            self.router_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.router_buffers = {}
            for name, experts in (("self_attn.v_router", 64), ("mlp.gate", 100)):
                bias = torch.full((32, 256), -float("inf"), dtype=torch.bfloat16)
                bias[:, :experts] = sd[name + ".bias"]
                tensors = [
                    bias.reshape(32, 16, 16).transpose(-1, -2),
                    torch.arange(256).int().reshape(1, 16, 16).expand(32, -1, -1).transpose(-1, -2),
                    torch.zeros(32, 32, 32),
                    torch.zeros(32, 32, 32).int(),
                ]
                self.router_buffers[name] = [
                    ttnn.from_torch(
                        t.contiguous(),
                        device=mesh_device,
                        dtype=d,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=self.router_mem,
                    )
                    for t, d in zip(tensors, (ttnn.bfloat16, ttnn.uint16, ttnn.bfloat16, ttnn.uint16))
                ]
        self.positions = ttnn.from_torch(
            torch.arange(c.max_position_embeddings, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return self

    def _linear(self, x, name):
        return ttnn.linear(x, self.weights[name + ".weight"], compute_kernel_config=self.compute)

    def _norm(self, x, name):
        shape = tuple(x.shape)
        grouped = ttnn.reshape(ttnn.typecast(x, ttnn.float32), (1, 1, math.prod(shape) // 1280, 1280))
        y = ttnn.rms_norm(grouped, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.compute)
        y = ttnn.reshape(y, shape)
        return ttnn.mul(y, self.weights[name + ".weight"], dtype=ttnn.bfloat16)

    def _route(self, x, name, top_k, experts):
        logits = self._linear(x, name)
        parts = []
        bias, ids, output, output_ids = self.router_buffers[name]
        for start in range(0, x.shape[-2], 32):
            tile = logits[:, :, start : start + 32, :]
            face = ttnn.pad(tile, [(0, 0), (0, 0), (0, 0), (0, 256 - tile.shape[-1])], 0)
            face = ttnn.to_memory_config(ttnn.reshape(face, (32, 16, 16)), self.router_mem)
            scores, indices = ttnn.experimental.deepseek.moe.generalized_moe_gate(
                face,
                bias_tensor=bias,
                input_indices_tensor=ids,
                output_tensor=output,
                output_indices_tensor=output_ids,
                enable_sigmoid=True,
                topk=top_k,
                scaling_factor=2.5,
                output_softmax=False,
                grouped=False,
            )
            scores = ttnn.to_memory_config(scores, ttnn.DRAM_MEMORY_CONFIG)[:, 0:1, :top_k]
            indices = ttnn.to_memory_config(indices, ttnn.DRAM_MEMORY_CONFIG)[:, 0:1, :top_k]
            scores = ttnn.reshape(scores, (1, 1, 32, top_k))
            indices = ttnn.reshape(ttnn.typecast(indices, ttnn.uint32), (1, 1, 32, top_k))
            parts.append(ttnn.scatter(ttnn.zeros_like(tile), dim=3, index=indices, src=scores)[..., :experts])
        return ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0]

    def _sparse_mm(self, x, w, sparsity, *, sparse_input=False):
        n = w.shape[-1]
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=x.shape[-2] // 32,
            per_core_N=math.ceil(n / 256),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        return ttnn.sparse_matmul(
            x,
            w,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=sparse_input,
            is_input_b_sparse=not sparse_input,
            program_config=program,
            compute_kernel_config=self.compute,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _experts(self, x, *, value=False):
        # Tiles form independent sparse batches. An expert is active for a tile
        # iff at least one token selects it; per-token weights mask the result.
        tokens = x.shape[-2]
        e = self.config.mova_num_experts if value else self.config.num_experts
        k = self.config.mova_num_experts_per_tok if value else self.config.num_experts_per_tok
        name = "self_attn.v_router" if value else "mlp.gate"
        routes = self._route(x, name, k, e)
        groups = tokens // 32
        routes = ttnn.reshape(routes, (1, groups, 32, e))
        sparsity = ttnn.to_layout(ttnn.sum(routes, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (1, groups, 32, 2560))
        if value:
            out = self._sparse_mm(x, self.weights["value_experts"], sparsity)
            out = ttnn.reshape(out, (groups, e, 32, 1024))
        else:
            gate_up = self._sparse_mm(x, self.weights["experts.gate_up"], sparsity)
            gate, up = gate_up[..., :768], gate_up[..., 768:]
            product = ttnn.reshape(
                ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), (groups, e, 32, 768)
            )
            down_sparsity = ttnn.reshape(sparsity, (1, 1, groups, e))
            out = self._sparse_mm(product, self.weights["experts.down_proj"], down_sparsity, sparse_input=True)
            out = ttnn.reshape(out, (groups, e, 32, 2560))
        routes = ttnn.reshape(ttnn.permute(routes, (0, 1, 3, 2)), (groups, e, 32, 1))
        out = ttnn.experimental.fast_reduce_nc(
            ttnn.mul(out, routes, input_tensor_a_activations=[ttnn.UnaryOpType.SILU] if value else []), dims=[1]
        )
        return ttnn.reshape(out, (1, 1, tokens, 1024 if value else 2560))

    def _mlp(self, x):
        prefix = "mlp.shared_experts" if self.sparse else "mlp"
        if x.shape[-2] > 32:
            hidden = ttnn.experimental.minimal_matmul(
                x, self.weights[prefix + ".swiglu.weight"], fuse_swiglu=True, compute_kernel_config=self.compute
            )
        else:
            gate = ttnn.linear(
                x, self.weights[prefix + ".gate_proj.weight"], activation="silu", compute_kernel_config=self.compute
            )
            hidden = ttnn.mul(gate, self._linear(x, prefix + ".up_proj"))
        dense = self._linear(hidden, prefix + ".down_proj")
        return ttnn.add(self._experts(x), dense) if self.sparse else dense

    def _rope(self, x, cos, sin, *, decode=False):
        if decode:
            b = x.shape[1]
            grid = ttnn.num_cores_to_corerangeset(b, self.mesh_device.compute_with_storage_grid_size(), True)
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
            )
            x, cos, sin = [ttnn.to_memory_config(t, mem) for t in (x, cos, sin)]
        return ttnn.experimental.rotary_embedding_hf(
            x, cos, sin, is_decode_mode=decode, compute_kernel_config=self.compute
        )

    def _finish(self, residual, gate_logits, attention):
        gate = ttnn.softplus(gate_logits, beta=math.log(2), threshold=20)
        x = ttnn.add(residual, self._linear(ttnn.mul(attention, gate), "self_attn.o_proj"))
        return ttnn.add(x, self._mlp(self._norm(x, "post_attention_layernorm")))

    def _attention_pages(self, page_table, page_size):
        # Decode reads complete 1024-token K blocks before causal masking.
        # Extend with valid addresses, never -1 or uninitialized table entries.
        # Aliases lie beyond the logical capacity and therefore every legal
        # current_pos. Repeating the first page also avoids an unwritten last
        # page where possible. Concat is captured and refreshes with the table.
        columns = max(8, 1024 // page_size)  # Also satisfy prefill's 32-byte row.
        extra = -page_table.shape[1] % columns
        if extra:
            first = page_table[:, :1]
            return ttnn.concat([page_table] + [first] * extra, dim=1)
        return page_table

    def prefill_forward(self, hidden_states, *, cos, sin, page_table, kv_cache, start_pos=0):
        """Paged, chunked prefill. Logical sequence length is returned unchanged.

        ``start_pos`` is an absolute cached-prefix offset. ``cos/sin`` describe
        the new input only, [B,1,S,128]. Request pages must be exclusively owned.
        """
        b, s = hidden_states.shape[1], hidden_states.shape[2]
        if hidden_states.shape[0] != 1 or hidden_states.shape[3] != 2560 or not 1 <= b <= 32:
            raise ValueError("Prefill expects [1,B,S,2560], 1<=B<=32")
        if s < 1 or start_pos < 0 or start_pos + s > self.config.max_position_embeddings:
            raise ValueError("Prefill exceeds the advertised context or has an invalid length")
        page_size = kv_cache[0].shape[2]
        if page_size % 32 or self.chunk_size % page_size:
            raise ValueError("Page size must be tile-aligned and divide the physical chunk size")
        if page_table.shape[0] != b or page_table.shape[1] * page_size < start_pos + s:
            raise ValueError("Page table does not cover the requested batch and context")
        results = []
        for user in range(b):
            chunks = []
            offset = 0
            # Preserve partial prefix pages with individual paged updates until
            # a page-aligned fill is legal. Position slicing stays on device.
            while offset < s and (start_pos + offset) % kv_cache[0].shape[2]:
                chunks.append(
                    self.decode_forward(
                        hidden_states[:, user : user + 1, offset : offset + 1, :],
                        cos=cos[user : user + 1, :, offset : offset + 1, :],
                        sin=sin[user : user + 1, :, offset : offset + 1, :],
                        page_table=page_table[user : user + 1, :],
                        kv_cache=kv_cache,
                        current_pos=self.positions[start_pos + offset : start_pos + offset + 1],
                    )
                )
                offset += 1
            for offset in range(offset, s, self.chunk_size):
                count = min(s - offset, self.chunk_size)
                chunks.append(
                    self._prefill_chunk(
                        hidden_states[:, user : user + 1, offset : offset + count, :],
                        cos=cos[user : user + 1, :, offset : offset + count, :],
                        sin=sin[user : user + 1, :, offset : offset + count, :],
                        page_table=page_table[user : user + 1, :],
                        kv_cache=kv_cache,
                        start_pos=start_pos + offset,
                    )
                )
            results.append(ttnn.concat(chunks, dim=2) if len(chunks) > 1 else chunks[0])
        return ttnn.concat(results, dim=1) if b > 1 else results[0]

    def _prefill_chunk(self, x, *, cos, sin, page_table, kv_cache, start_pos):
        logical = x.shape[2]
        physical = math.ceil(logical / 32) * 32
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        cos = ttnn.pad(cos, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        sin = ttnn.pad(sin, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        normed = self._norm(x, "input_layernorm")
        packed = self._linear(normed, "attention_packed")
        qkv_width = 5120 if self.sparse else 6144
        gate_logits = packed[..., qkv_width:]
        qkv = packed[..., :qkv_width]
        if self.sparse:
            qkv = ttnn.concat([qkv, self._experts(normed, value=True)], dim=-1)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(qkv, (1, physical, 6144)), num_heads=32, num_kv_heads=8, transpose_key=False
        )
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        first_page = start_pos // kv_cache[0].shape[2]
        pages = page_table[:, first_page : first_page + math.ceil(physical / kv_cache[0].shape[2])]
        full_tiles = logical // 32 * 32
        if full_tiles:
            ttnn.experimental.paged_fill_cache(kv_cache[0], k[:, :, :full_tiles, :], pages, batch_idx=0)
            ttnn.experimental.paged_fill_cache(kv_cache[1], v[:, :, :full_tiles, :], pages, batch_idx=0)
        # Fill writes whole tiles. Tail updates touch only valid rows, preserving
        # all other cache entries even when the physical compute input is padded.
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(1, self.mesh_device.compute_with_storage_grid_size(), True),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        for token in range(full_tiles, logical):
            position = self.positions[start_pos + token : start_pos + token + 1]
            for cache, heads in zip(kv_cache, (k, v)):
                row = ttnn.permute(heads[:, :, token : token + 1, :], (0, 2, 1, 3))
                ttnn.experimental.paged_update_cache(
                    cache, ttnn.to_memory_config(row, mem), update_idxs_tensor=position, page_table=page_table
                )
        # Bound BF16 online accumulation to at most 512 K-block merges at
        # the advertised 524288 context. Smaller blocks fail long-context PCC.
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            kv_cache[0],
            kv_cache[1],
            self._attention_pages(page_table, kv_cache[0].shape[2]),
            chunk_start_idx=None,
            chunk_start_idx_tensor=self.positions[start_pos : start_pos + 1],
            compute_kernel_config=self.compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=1024, exp_approx_mode=False
            ),
        )
        out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, physical, 4096))
        return self._finish(x, gate_logits, out)[:, :, :logical, :]

    def decode_forward(self, hidden_states, *, cos, sin, page_table, current_pos, kv_cache):
        """Device-only traceable decode; current_pos and page_table are mutable inputs.

        cos/sin: [1,B,1,128]. No host reads, conversions, cache construction or
        state reset occurs in this call. The output is [1,1,B,2560].
        """
        b = hidden_states.shape[2]
        if tuple(hidden_states.shape) != (1, 1, b, 2560) or not 1 <= b <= 32:
            raise ValueError("Decode expects [1,1,B,2560], 1<=B<=32")
        padded = math.ceil(b / 32) * 32
        x = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, padded - b), (0, 0)], 0)
        normed = self._norm(x, "input_layernorm")
        packed = self._linear(normed, "attention_packed")
        qkv_width = 5120 if self.sparse else 6144
        gate_logits = packed[..., qkv_width:]
        qkv = packed[..., :qkv_width]
        if self.sparse:
            qkv = ttnn.concat([qkv, self._experts(normed, value=True)], dim=-1)
        grid = ttnn.num_cores_to_corerangeset(b, self.mesh_device.compute_with_storage_grid_size(), True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv[:, :, :b, :], num_heads=32, num_kv_heads=8, memory_config=mem
        )
        q, k = self._rope(q, cos, sin, decode=True), self._rope(k, cos, sin, decode=True)
        grid = ttnn.num_cores_to_corerangeset(b, self.mesh_device.compute_with_storage_grid_size(), True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        k, v = ttnn.to_memory_config(k, mem), ttnn.to_memory_config(v, mem)
        ttnn.experimental.paged_update_cache(kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            self._attention_pages(page_table, kv_cache[0].shape[2]),
            cur_pos_tensor=current_pos,
            compute_kernel_config=self.compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=1024, exp_approx_mode=False
            ),
        )
        out = ttnn.reshape(out, (1, 1, b, 4096))
        out = ttnn.pad(out, [(0, 0), (0, 0), (0, padded - b), (0, 0)], 0)
        return self._finish(x, gate_logits, out)[:, :, :b, :]
