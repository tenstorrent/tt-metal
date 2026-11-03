# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Carry a 640-wide residual through local adds and distributed grouped RMSNorm."""

import math

import torch

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        assert self.residual_sharded
        prefix = f"model.layers.{self.layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        self.local_gamma = {}
        for name in ("input_layernorm", "post_attention_layernorm"):
            self.local_gamma[name] = ttnn.from_torch(
                sd[name + ".weight"].reshape(1, 1, 1, 2560),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=3),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        selectors = []
        for rank in range(4):
            sel = torch.zeros(1, 1, 128, 32)
            for peer in (2 * (rank // 2), 2 * (rank // 2) + 1):
                sel[0, 0, peer * 32, 0] = 0.5
            selectors.append(sel)
        self.group_selector = ttnn.from_torch(
            torch.cat(selectors, dim=0),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        return self

    def _norm(self, x, name):
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[-2] == 32 else ttnn.DRAM_MEMORY_CONFIG
        xf = ttnn.typecast(ttnn.to_memory_config(x, mem), ttnn.float32)
        mean = ttnn.mean(ttnn.square(xf), dim=-1, keepdim=True)
        mean = ttnn.pad(mean, [(0, 0), (0, 0), (0, 0), (0, 31)], 0)
        stats = self._gather(ttnn.to_memory_config(mean, ttnn.DRAM_MEMORY_CONFIG))
        mean = ttnn.linear(
            stats, self.group_selector, compute_kernel_config=self.compute, dtype=ttnn.float32, memory_config=mem
        )[..., :1]
        inv = ttnn.rsqrt(ttnn.add(mean, self.config.rms_norm_eps))
        normalized = ttnn.mul(
            ttnn.mul(xf, inv),
            self.local_gamma[name],
            dtype=getattr(
                ttnn, self.options["attention_activation" if name == "input_layernorm" else "mlp_activation"]
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        full = self._gather(normalized)
        return ttnn.to_memory_config(full, self._residual_memory()) if x.shape[-2] == 32 else full

    def _prefill_chunk(self, x, *, cos, sin, page_table, kv_cache, start_pos):
        logical = x.shape[2]
        physical = math.ceil(logical / 32) * 32
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        cos = ttnn.pad(cos, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        sin = ttnn.pad(sin, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        normed = self._norm(x, "input_layernorm")
        packed = self._linear(normed, "attention_packed")
        qkv_width = 1280 if self.sparse else 1536
        gate_logits = packed[..., qkv_width : qkv_width + 1024]
        qkv = packed[..., :qkv_width]
        if self.sparse:
            qkv = ttnn.concat([qkv, self._experts(normed, value=True, logits=packed[..., qkv_width + 1024 :])], dim=-1)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(qkv, (1, physical, 1536)), num_heads=8, num_kv_heads=2, transpose_key=False
        )
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        first_page = start_pos // kv_cache[0].shape[2]
        pages = page_table[:, first_page : first_page + math.ceil(physical / kv_cache[0].shape[2])]
        full_tiles = logical // 32 * 32
        if full_tiles:
            ttnn.experimental.paged_fill_cache(
                kv_cache[0], ttnn.typecast(k[:, :, :full_tiles, :], kv_cache[0].dtype), pages, batch_idx=0
            )
            ttnn.experimental.paged_fill_cache(
                kv_cache[1], ttnn.typecast(v[:, :, :full_tiles, :], kv_cache[1].dtype), pages, batch_idx=0
            )
        # Fill writes whole tiles. Tail updates touch only valid rows, preserving
        # all other cache entries even when the physical compute input is padded.
        for token in range(full_tiles, logical):
            position = self.positions[start_pos + token : start_pos + token + 1]
            rows = [ttnn.permute(heads[:, :, token : token + 1, :], (0, 2, 1, 3)) for heads in (k, v)]
            self._update_cache(*rows, position, page_table, kv_cache)
        # Bound BF16 online accumulation to at most 512 K-block merges at
        # the advertised 524288 context. Smaller blocks fail long-context PCC.
        full_grid = self.mesh_device.compute_with_storage_grid_size()
        attention_grid = (full_grid.x, full_grid.y) if start_pos + physical >= 65536 else (8, 8)
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            kv_cache[0],
            kv_cache[1],
            self._attention_pages(page_table, kv_cache[0].shape[2]),
            chunk_start_idx=None,
            chunk_start_idx_tensor=self.positions[start_pos : start_pos + 1],
            compute_kernel_config=self.sdpa_compute["prefill"],
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=attention_grid, q_chunk_size=32, k_chunk_size=1024, exp_approx_mode=False
            ),
        )
        out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, 1, physical, 1024))
        return self._finish(x, gate_logits, out)[:, :, :logical, :]

    def decode_forward(self, hidden_states, *, cos, sin, page_table, current_pos, kv_cache):
        """Device-only traceable decode; current_pos and page_table are mutable inputs.

        cos/sin: [1,B,1,128]. No host reads, conversions, cache construction or
        state reset occurs in this call. The output is [1,1,B,2560].
        """
        b = hidden_states.shape[2]
        if tuple(hidden_states.shape) != (1, 1, b, 640 if self.residual_sharded else 2560) or not 1 <= b <= 32:
            raise ValueError("Decode expects [1,1,B,2560], 1<=B<=32")
        padded = math.ceil(b / 32) * 32
        x = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, padded - b), (0, 0)], 0)
        normed = self._norm(x, "input_layernorm")
        packed = self._linear(normed, "attention_packed")
        qkv_width = 1280 if self.sparse else 1536
        gate_logits = packed[..., qkv_width : qkv_width + 1024]
        qkv = packed[..., :qkv_width]
        if self.sparse:
            qkv = ttnn.concat(
                [qkv, self._experts(normed, value=True, valid_tokens=b, logits=packed[..., qkv_width + 1024 :])], dim=-1
            )
        # SDPA's sharded reader uses its 8-wide reducer grid, not the physical
        # device width. Producer and consumer must agree on each user's core.
        grid = ttnn.num_cores_to_corerangeset(b, ttnn.CoreCoord(8, 8), True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv[:, :, :b, :], num_heads=8, num_kv_heads=2, memory_config=mem
        )
        if b > 8 and b % 8:
            # Native rotary launches the grid's bounding rectangle. Back every
            # launched core, including the inactive final-row slots.
            rectangle = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, (b - 1) // 8))})
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(rectangle, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
            )
            q, k = ttnn.to_memory_config(q, mem), ttnn.to_memory_config(k, mem)
        cos, sin = ttnn.to_memory_config(cos, mem), ttnn.to_memory_config(sin, mem)
        q, k = self._rope(q, cos, sin, decode=True), self._rope(k, cos, sin, decode=True)
        self._update_cache(k, v, current_pos, page_table, kv_cache)
        del k, v, cos, sin
        if page_table.shape[1] * kv_cache[0].shape[2] >= 262144:
            # The long page-table CB and k_chunk1024 nearly fill L1. Release
            # the transient query shard before SDPA; capacity stays unchanged.
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        attention_compute, attention_program = self._decode_attention_config(b, page_table, kv_cache[0])
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            self._attention_pages(page_table, kv_cache[0].shape[2]),
            cur_pos_tensor=current_pos,
            compute_kernel_config=attention_compute,
            program_config=attention_program,
        )
        concat_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(out, concat_mem), num_heads=8)
        return self._finish(x, gate_logits, out, valid_tokens=b)[:, :, :b, :]

    def _finish(self, residual, gate_logits, attention, *, valid_tokens=None):
        gated = ttnn.mul(
            attention,
            gate_logits,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, math.log(2), 20.0)],
        )
        projected = self._reduce(self._linear(gated, "self_attn.o_proj"))
        x = ttnn.add(residual, projected)
        normed = self._norm(x, "post_attention_layernorm")
        return ttnn.add(x, self._mlp(normed, valid_tokens=valid_tokens))
