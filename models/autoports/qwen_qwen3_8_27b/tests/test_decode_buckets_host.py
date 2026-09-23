# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of bucket selection and scheduler-state gather/scatter."""

import unittest
from types import MethodType, SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_27b.tests.test_serving_prefill_trace_host import load_methods


class DecodeBucketsTests(unittest.TestCase):
    def make_model(self):
        def pad(tensor, padding, *, value):
            return torch.nn.functional.pad(tensor, tuple(v for pair in reversed(padding) for v in pair), value=value)

        ops = SimpleNamespace(
            reshape=torch.reshape,
            concat=lambda rows, dim: torch.cat(rows, dim=dim),
            full_like=lambda x, fill_value: torch.full_like(x, fill_value),
            zeros_like=torch.zeros_like,
            copy=lambda source, target: target.copy_(source),
            pad=pad,
        )
        methods = load_methods("model.py", "QwenModel", ["decode", "_decode_bucket"], ops)
        methods["DecoderState"] = SimpleNamespace
        methods["ModelCache"] = lambda layers, batch, capacity, pages: SimpleNamespace(
            layers=layers, batch_size=batch, capacity=capacity, num_pages=pages
        )
        model = SimpleNamespace(decode_buckets=True, calls=[])
        model.decode = MethodType(methods["decode"], model)
        model._decode_bucket = MethodType(methods["_decode_bucket"], model)

        def fixed(tokens, positions, *, cache, page_table, rope_indices=None, active_slots=None):
            model.calls.append((cache.batch_size, positions.clone(), page_table.clone(), rope_indices.clone()))
            for state in cache.layers:
                if state.conv is not None:
                    state.conv.add_(1)
                    state.recurrent.add_(2)
            # Token identity lets us verify logits are restored to scheduler order.
            return tokens.reshape(-1)[: cache.batch_size].reshape(1, 1, cache.batch_size, 1).float()

        model._decode_fixed = fixed
        return model

    def cache(self, batch=16):
        state = SimpleNamespace(
            key=None,
            value=None,
            conv=torch.arange(batch).reshape(batch, 1, 1).float(),
            recurrent=torch.arange(batch).reshape(batch, 1, 1).float() * 10,
        )
        return SimpleNamespace(layers=[state], batch_size=batch, capacity=1024, num_pages=100)

    def run_decode(self, model, cache, slots):
        b = cache.batch_size
        return model.decode(
            torch.arange(32).reshape(1, 1, 1, 32),
            torch.arange(b, dtype=torch.int32) + 20,
            cache=cache,
            page_table=torch.arange(b * 2).reshape(b, 2),
            rope_indices=torch.arange(b, dtype=torch.int32) + 20,
            active_slots=slots,
        )

    def test_all_occupancies_and_state_ownership(self):
        for count in range(1, 17):
            with self.subTest(count=count):
                model, cache = self.make_model(), self.cache()
                # Include non-prefix slots and reversed scheduler order.
                slots = tuple(reversed(range(16 - count, 16)))
                old_conv = cache.layers[0].conv.clone()
                old_recurrent = cache.layers[0].recurrent.clone()
                logits = self.run_decode(model, cache, slots)
                bucket = 1 if count == 1 else 8 if count <= 8 else 16
                actual, positions, table, rope = model.calls[-1]
                self.assertEqual(actual, bucket)
                self.assertEqual(positions[:count].tolist(), [20 + i for i in slots])
                self.assertEqual(positions[count:].tolist(), [-1] * (bucket - count))
                self.assertEqual(rope[:count].tolist(), [20 + i for i in slots])
                self.assertEqual(table[:count].tolist(), [[2 * i, 2 * i + 1] for i in slots])
                for i in range(16):
                    self.assertEqual(cache.layers[0].conv[i].item(), old_conv[i].item() + (i in slots))
                    self.assertEqual(cache.layers[0].recurrent[i].item(), old_recurrent[i].item() + 2 * (i in slots))
                    self.assertEqual(logits[0, 0, i, 0].item(), i if i in slots else 0)

    def test_transitions_preserve_accumulated_state(self):
        model, cache = self.make_model(), self.cache()
        expected = cache.layers[0].conv.clone()
        for slots in ((15,), tuple(range(5)), tuple(range(12)), (7,), tuple(range(16)), (2, 9)):
            self.run_decode(model, cache, slots)
            expected[list(slots)] += 1
            self.assertTrue(torch.equal(cache.layers[0].conv, expected))
        self.assertEqual([call[0] for call in model.calls], [1, 8, 16, 1, 16, 8])

    def test_rejects_invalid_slots_and_capacity(self):
        for slots in ((), (0, 0), (-1,), (16,)):
            with self.assertRaises(ValueError):
                self.run_decode(self.make_model(), self.cache(), slots)
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.run_decode(self.make_model(), self.cache(32), (0,))


if __name__ == "__main__":
    unittest.main()
