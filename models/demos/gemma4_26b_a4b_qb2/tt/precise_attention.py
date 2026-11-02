# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP32 decode attention over a device-selected range of paged K/V."""

import torch

import ttnn


class PrecisePagedAttention:
    def __init__(self, mesh, config, max_context, output_dtype=ttnn.float32):
        self.config = config
        self.output_dtype = output_dtype
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def constant(value):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        self.page_offsets = constant(torch.arange(max_context // 32 + 1).float()[None])
        self.row_offsets = ttnn.from_torch(
            torch.arange(config.num_key_value_heads * 32, dtype=torch.int32).reshape(1, 1, -1),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
        )
        self.token_offsets = constant(torch.arange(max_context + 32).float().reshape(1, 1, 1, -1))

    def cache_row_indices(self, physical):
        """Keep physical addresses integer, including row indices above 2**24."""
        stride = self.config.num_key_value_heads * 32
        selected = physical.shape[-1]
        physical = ttnn.to_layout(ttnn.typecast(physical, ttnn.uint32), ttnn.TILE_LAYOUT)
        base = ttnn.bitwise_left_shift(physical, stride.bit_length() - 1)
        rows = ttnn.add(ttnn.reshape(base, (1, selected, 1)), self.row_offsets)
        return ttnn.to_layout(ttnn.reshape(rows, (1, selected * stride)), ttnn.ROW_MAJOR_LAYOUT)

    def __call__(self, q, k, v, *, cur_pos_tensor, page_table_tensor, **kwargs):
        cfg = self.config
        block = k.shape[-2]
        if block != 32:
            raise ValueError("Functional paged attention requires 32-token pages")
        pages = page_table_tensor.shape[-1]
        position = ttnn.reshape(ttnn.to_layout(ttnn.typecast(cur_pos_tensor, ttnn.float32), ttnn.TILE_LAYOUT), (1, 1))
        if cfg.is_sliding:
            selected = min(pages, cfg.sliding_window // block + 1)
            start = ttnn.floor(ttnn.mul(ttnn.maximum(ttnn.add(position, 1 - cfg.sliding_window), 0), 1 / block))
            page_ids = ttnn.add(start, self.page_offsets[:, :selected])
            page_ids = ttnn.typecast(ttnn.minimum(page_ids, pages - 1), ttnn.uint32)
            page_ids = ttnn.to_layout(page_ids, ttnn.ROW_MAJOR_LAYOUT)
            physical = ttnn.gather(page_table_tensor, dim=1, index=page_ids)
        else:
            selected = pages
            start = ttnn.mul(position, 0.0)
            # Full attention reads all logical pages in table order.
            physical = page_table_tensor
        rows = self.cache_row_indices(physical)

        def gather(cache):
            flat = ttnn.reshape(cache, (cache.shape[0] * cfg.num_key_value_heads * block, cfg.head_dim))
            gathered = ttnn.embedding(rows, flat, layout=ttnn.TILE_LAYOUT)
            gathered = ttnn.reshape(gathered, (selected, cfg.num_key_value_heads, block, cfg.head_dim))
            gathered = ttnn.permute(gathered, (1, 0, 2, 3))
            return ttnn.reshape(gathered, (1, cfg.num_key_value_heads, selected * block, cfg.head_dim))

        keys, values = gather(k), gather(v)
        absolute = ttnn.add(
            ttnn.reshape(ttnn.mul(start, block), (1, 1, 1, 1)), self.token_offsets[:, :, :, : selected * block]
        )
        pos = ttnn.reshape(position, (1, 1, 1, 1))
        allowed = ttnn.le(absolute, pos)
        if cfg.is_sliding:
            allowed = ttnn.logical_and(allowed, ttnn.gt(absolute, ttnn.subtract(pos, cfg.sliding_window)))
        group = cfg.num_attention_heads // cfg.num_key_value_heads
        outputs = []
        for head in range(cfg.num_key_value_heads):
            query = ttnn.typecast(q[:, :, head * group : (head + 1) * group, :], ttnn.float32)
            key = keys[:, head : head + 1, :, :]
            value = values[:, head : head + 1, :, :]
            scores = ttnn.matmul(query, key, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.compute)
            scores = ttnn.where(allowed, scores, -1.0e30)
            shifted = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True))
            probabilities = ttnn.exp(shifted, fast_and_approximate_mode=False)
            probabilities = ttnn.div(probabilities, ttnn.sum(probabilities, dim=-1, keepdim=True))
            outputs.append(ttnn.matmul(probabilities, value, dtype=ttnn.float32, compute_kernel_config=self.compute))
        return ttnn.concat(outputs, dim=2)
