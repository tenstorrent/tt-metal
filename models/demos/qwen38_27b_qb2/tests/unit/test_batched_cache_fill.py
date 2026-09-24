# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch contract for experimental batched cache fill; silicon checked separately."""

import unittest
from types import MethodType, SimpleNamespace

import torch

from models.demos.qwen38_27b_qb2.tests.unit.test_serving_prefill_trace import load_methods


class BatchedCacheFillTests(unittest.TestCase):
    def test_grouped_fill_keeps_page_rows_and_continuation_offset(self):
        fills, allocations = [], []

        def arange(start, end, step, **kwargs):
            allocations.append(end)
            return torch.arange(start, end, step, dtype=torch.int32)

        ops = SimpleNamespace(
            int32=torch.int32,
            ROW_MAJOR_LAYOUT="row",
            DRAM_MEMORY_CONFIG="dram",
            arange=arange,
            reshape=torch.reshape,
            concat=lambda xs, dim: torch.cat(xs, dim=dim),
            typecast=lambda tensor, dtype: tensor.to(dtype),
            SDPAProgramConfig=lambda **kwargs: kwargs,
            experimental=SimpleNamespace(paged_fill_cache=lambda *a, **kw: fills.append((a, kw))),
            transformer=SimpleNamespace(chunked_scaled_dot_product_attention=lambda q, *a, **kw: q),
        )
        method = load_methods("decoder.py", "Qwen38Decoder", ["_full_prefill"], ops)
        q = torch.arange(2 * 64 * 32).reshape(2, 1, 64, 32).float()
        key, value = q + 100, q + 200
        layer = SimpleNamespace(
            PAGE_SIZE=32,
            policy={"batched_prefill_cache_fill": True},
            config=SimpleNamespace(head_dim=32),
            device=SimpleNamespace(compute_with_storage_grid_size=lambda: (11, 10)),
            _qkv=lambda *args: (q, key, value, None),
            _attention_output=lambda output, gate: output,
        )
        state = SimpleNamespace(key=torch.zeros(1, dtype=torch.bfloat16), value=torch.zeros(1, dtype=torch.bfloat16))
        table = torch.tensor([[9, 3, 7, 1], [8, 2, 6, 0]], dtype=torch.int32)
        call = MethodType(method["_full_prefill"], layer)
        for _ in range(2):
            actual = call(torch.zeros(2, 64, 32), state, table, 32, None, None)
            self.assertTrue(torch.equal(actual, q))
        self.assertEqual(allocations, [2])
        self.assertEqual(len(fills), 4)
        for i, (args, kwargs) in enumerate(fills):
            self.assertIs(args[0], state.key if i % 2 == 0 else state.value)
            self.assertTrue(torch.equal(args[1], (key if i % 2 == 0 else value).to(torch.bfloat16)))
            self.assertTrue(torch.equal(args[2], table[:, 1:3]))
            self.assertEqual(kwargs["batch_idx_tensor"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
