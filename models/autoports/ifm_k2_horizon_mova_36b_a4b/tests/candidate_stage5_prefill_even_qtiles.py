# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _prefill_chunk(self, x, *, cos, sin, page_table, kv_cache, start_pos):
        logical = x.shape[2]
        physical = math.ceil(logical / 64) * 64
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        cos = ttnn.pad(cos, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        sin = ttnn.pad(sin, [(0, 0), (0, 0), (0, physical - logical), (0, 0)], 0)
        normed = self._norm(self._gather(x) if self.residual_sharded else x, "input_layernorm")
        packed = self._linear(normed, "attention_packed")
        qkv_width = 1280 if self.sparse else 1536
        gate_logits = packed[..., qkv_width : qkv_width + 1024]
        qkv = packed[..., :qkv_width]
        if self.sparse:
            qkv = ttnn.concat([qkv, self._experts(normed, value=True, logits=packed[..., qkv_width + 1024 :])], dim=-1)
        # Projection and expert consumers are finished; release the L1 norm
        # before SDPA reserves its large BF16-cache circular buffers.
        del normed
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
