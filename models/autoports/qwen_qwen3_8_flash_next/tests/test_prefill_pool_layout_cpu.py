# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU setup regression: actual reduced/full48 cache geometry; no TTNN import."""
import unittest
from types import SimpleNamespace

import torch

from .shared_pool_layout import indexed_attention_layout, shared_pool_layout, validate_pool_aliases


def cache_for(indices, pairs=None):
    layers = []
    attention_index = 0
    for index in indices:
        state = None
        if index % 4 == 3:
            pair = pairs[attention_index] if pairs is not None else (None, None)
            state = SimpleNamespace(k_cache=pair[0], v_cache=pair[1])
            attention_index += 1
        layers.append(SimpleNamespace(attention=state))
    return SimpleNamespace(layers=layers, shared_kv=True, kv_bound=True)


class PoolLayoutTest(unittest.TestCase):
    def test_reduced_and_full48_specs_from_actual_cache(self):
        for indices, count in (([0, 1, 3], 1), (list(range(48)), 12)):
            with self.subTest(layers=len(indices)):
                specs, tables = indexed_attention_layout(cache_for(indices), 2, 384)
                self.assertEqual(len(specs), count)
                self.assertEqual([entry[2] for entry in specs], list(range(count)))
                self.assertTrue(all(shape == (24, 1, 32, 256) and dtype == torch.bfloat16 for shape, dtype, _ in specs))
                self.assertEqual(len(tables), count)
                for table in tables:
                    self.assertEqual(tuple(table.shape), (2, 12))
                    self.assertEqual(len(set(table.flatten().tolist())), 24)
                pairs = [(object(), object()) for _ in specs]
                pool = SimpleNamespace(
                    pairs=dict(enumerate(pairs)), pair_for_attention_layer=lambda index: pairs[index]
                )
                caches = [cache_for(indices, pairs) for _ in range(3)]
                validate_pool_aliases(caches, pool, specs)
                # An unbound last layer in scratch must be caught, including layer47.
                caches[-1].layers[-1].attention.v_cache = object()
                with self.assertRaises(AssertionError):
                    validate_pool_aliases(caches, pool, specs)

    def test_missing_full48_spec_is_rejected(self):
        pairs = [(object(), object()) for _ in range(12)]
        pool = SimpleNamespace(pairs=dict(enumerate(pairs)), pair_for_attention_layer=lambda i: pairs[i])
        specs, _ = shared_pool_layout(2, 384, 1)
        with self.assertRaises(AssertionError):
            validate_pool_aliases([cache_for(range(48), pairs)], pool, specs)

    def test_b1_and_non_aligned_capacity(self):
        specs, tables = indexed_attention_layout(cache_for([0, 1, 3]), 1, 385)
        self.assertEqual(specs[0][0], (16, 1, 32, 256))
        self.assertEqual(tuple(tables[0].shape), (1, 16))


if __name__ == "__main__":
    unittest.main()
