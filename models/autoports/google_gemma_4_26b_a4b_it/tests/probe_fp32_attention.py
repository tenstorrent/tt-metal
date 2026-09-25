# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic device FP32 attention over dynamically selected paged K/V."""

import torch

import ttnn


class PrecisePagedAttention:
    def __init__(self, mesh, config, max_context, output_dtype=ttnn.bfloat16, split_inputs=False):
        self.config = config
        self.output_dtype = output_dtype
        self.split_inputs = split_inputs
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
        self.row_offsets = constant(torch.arange(config.num_key_value_heads * 32).float().reshape(1, 1, -1))
        self.token_offsets = constant(torch.arange(max_context + 32).float().reshape(1, 1, 1, -1))

    def matmul(self, value, weight, **kw):
        if not self.split_inputs:
            return ttnn.matmul(value, weight, dtype=ttnn.float32, compute_kernel_config=self.compute, **kw)
        high = ttnn.typecast(value, ttnn.bfloat16)
        low = ttnn.typecast(ttnn.subtract(value, ttnn.typecast(high, ttnn.float32)), ttnn.bfloat16)
        first = ttnn.matmul(high, weight, dtype=ttnn.float32, compute_kernel_config=self.compute, **kw)
        second = ttnn.matmul(low, weight, dtype=ttnn.float32, compute_kernel_config=self.compute, **kw)
        return ttnn.add(first, second)

    def __call__(self, q, k, v, *, cur_pos_tensor, page_table_tensor, **kwargs):
        cfg = self.config
        block = k.shape[-2]
        if block != 32:
            raise ValueError("Diagnostic uses the stage's 32-token pages")
        pages = page_table_tensor.shape[-1]
        position = ttnn.reshape(ttnn.to_layout(ttnn.typecast(cur_pos_tensor, ttnn.float32), ttnn.TILE_LAYOUT), (1, 1))
        if cfg.is_sliding:
            selected = min(pages, cfg.sliding_window // block + 1)
            start = ttnn.floor(ttnn.mul(ttnn.maximum(ttnn.add(position, 1 - cfg.sliding_window), 0), 1 / block))
        else:
            selected = pages
            start = ttnn.mul(position, 0.0)
        page_ids = ttnn.add(start, self.page_offsets[:, :selected])
        page_ids = ttnn.typecast(ttnn.minimum(page_ids, pages - 1), ttnn.uint32)
        page_ids = ttnn.to_layout(page_ids, ttnn.ROW_MAJOR_LAYOUT)
        physical = ttnn.gather(page_table_tensor, dim=1, index=page_ids)
        physical = ttnn.to_layout(ttnn.typecast(physical, ttnn.float32), ttnn.TILE_LAYOUT)
        base_rows = ttnn.reshape(ttnn.mul(physical, cfg.num_key_value_heads * block), (1, selected, 1))
        rows = ttnn.add(base_rows, self.row_offsets)
        rows = ttnn.to_layout(
            ttnn.typecast(ttnn.reshape(rows, (1, selected * cfg.num_key_value_heads * block)), ttnn.uint32),
            ttnn.ROW_MAJOR_LAYOUT,
        )

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
            scores = self.matmul(query, key, transpose_b=True)
            scores = ttnn.where(allowed, scores, -1.0e30)
            shifted = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True))
            probabilities = ttnn.exp(shifted, fast_and_approximate_mode=False)
            probabilities = ttnn.div(probabilities, ttnn.sum(probabilities, dim=-1, keepdim=True))
            outputs.append(self.matmul(probabilities, value))
        output = ttnn.typecast(ttnn.concat(outputs, dim=2), self.output_dtype)
        if hasattr(self, "observer"):
            replacement = self.observer(q, keys, values, output)
            if replacement is not None:
                return replacement
        return output
