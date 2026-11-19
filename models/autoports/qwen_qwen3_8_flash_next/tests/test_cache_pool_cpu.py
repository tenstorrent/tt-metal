# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU ownership, alias, scheduler-table and no-partial-binding checks."""

import unittest
from types import SimpleNamespace

from models.autoports.qwen_qwen3_8_flash_next.tt.cache_pool import SharedKVPool


def tensor(shape=(4, 1, 32, 256), dtype="bfloat16"):
    return SimpleNamespace(shape=shape, dtype=dtype, layout="TILE")


def cache():
    def attention():
        return SimpleNamespace(
            capacity=128,
            page_size=32,
            page_table=SimpleNamespace(shape=(1, 4)),
            position=object(),
            raw_history=object(),
            compressed_keys=object(),
            k_cache=tensor(),
            v_cache=tensor(),
        )

    return SimpleNamespace(
        capacity=125,
        position=0,
        layers=[
            SimpleNamespace(attention=attention(), ple_conv=object()),
            SimpleNamespace(attention=None, recurrent=object()),
            SimpleNamespace(attention=attention()),
        ],
    )


def pool(shared=False):
    specs = [((16, 1, 32, 256), "bfloat16", 3), ((16, 1, 32, 256), "bfloat16", 3 if shared else 7)]
    return SharedKVPool(specs, allocate_pair=lambda mesh, spec: (tensor(spec.shape), tensor(spec.shape)))


class SharedPoolTest(unittest.TestCase):
    def test_bfp8_pool_bytes_include_exponents_and_unique_tensor_ownership(self):
        for dtype, tile_bytes in (("bfloat16", 2048), ("bfloat8_b", 1088)):
            with self.subTest(dtype=dtype):
                specs = [((8192, 1, 32, 256), dtype, index) for index in range(12)]
                owner = SharedKVPool(
                    specs + [specs[0]],
                    allocate_pair=lambda mesh, spec: (tensor(spec.shape, spec.dtype), tensor(spec.shape, spec.dtype)),
                )
                self.assertEqual(owner.bytes_per_rank, 12 * 8192 * 2 * 8 * tile_bytes)
                self.assertEqual(owner.bytes_per_rank, 3221225472 if dtype == "bfloat16" else 1711276032)
                self.assertIs(owner.pair_for_attention_layer(0), owner.pair_for_attention_layer(12))
                self.assertEqual(len(owner.shared_tensor_ids), 24)

    def test_shared_identity_cannot_mix_storage_dtypes(self):
        calls = []
        specs = [((4, 1, 32, 256), dtype, 0) for dtype in ("bfloat16", "bfloat8_b")]
        with self.assertRaisesRegex(ValueError, "identical shape and dtype"):
            SharedKVPool(specs, allocate_pair=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_allocator_cannot_substitute_wrong_storage_dtype(self):
        for requested, allocated in (("bfloat8_b", "bfloat16"), ("bfloat16", "bfloat8_b")):
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(ValueError, "incompatible KV tensor"):
                    SharedKVPool(
                        [((4, 1, 32, 256), requested, 0)],
                        allocate_pair=lambda mesh, spec: (tensor(spec.shape, allocated), tensor(spec.shape, allocated)),
                    )

    def test_bind_preserves_all_non_kv_objects_and_tensor_idx_sharing(self):
        for shared in (False, True):
            owner, first, second, scratch = pool(shared), cache(), cache(), cache()
            retained = [
                (item.layers[0].attention.page_table, item.layers[0].attention.raw_history, item.layers[1].recurrent)
                for item in (first, second, scratch)
            ]
            displaced = owner.bind([first, second, scratch])
            self.assertEqual(len(displaced), 12)
            for item, previous in zip((first, second, scratch), retained):
                self.assertTrue(item.shared_kv)
                self.assertIs(item.shared_kv_pool, owner)
                self.assertIs(item.layers[0].attention.page_table, previous[0])
                self.assertIs(item.layers[0].attention.raw_history, previous[1])
                self.assertIs(item.layers[1].recurrent, previous[2])
                self.assertIs(item.layers[0].attention.k_cache, first.layers[0].attention.k_cache)
                self.assertEqual(item.layers[0].attention.capacity, 128)
                self.assertEqual(item.layers[0].attention.k_cache.shape[0], 16)
            self.assertEqual(first.layers[0].attention.k_cache is first.layers[2].attention.k_cache, shared)
            self.assertEqual(owner.bytes_per_rank, 16 * 32 * 256 * 2 * 2 * (1 if shared else 2))
            self.assertEqual(len(owner.shared_tensor_ids), 2 if shared else 4)
            self.assertEqual(owner.bind([first]), ())

    def test_validation_failure_cannot_partially_replace_cache(self):
        owner, first, second = pool(), cache(), cache()
        original = first.layers[0].attention.k_cache
        second.layers[2].attention.page_size = 64
        with self.assertRaisesRegex(ValueError, "page32"):
            owner.bind([first, second])
        self.assertIs(first.layers[0].attention.k_cache, original)
        self.assertFalse(hasattr(first, "shared_kv"))

    def test_rebinding_after_prepare_or_nonempty_cache_rejected(self):
        owner, first = pool(), cache()
        first.position = 1
        with self.assertRaisesRegex(ValueError, "empty"):
            owner.bind([first])
        owner.freeze()
        with self.assertRaisesRegex(RuntimeError, "freeze"):
            owner.bind([cache()])

    def test_bad_plan_rejected_before_any_allocation(self):
        calls = []
        good = ((16, 1, 32, 256), "bfloat16", 0)
        for bad in (
            ((16, 1, 32, 256), "bfloat4_b", 1),
            ((16, 2, 32, 256), "bfloat16", 1),
            ((16, 1, 64, 256), "bfloat16", 1),
            ((8, 1, 32, 256), "bfloat16", 0),
            ((16, 1, 32, 256), "bfloat16", -1),
        ):
            with self.assertRaises(ValueError):
                SharedKVPool([good, bad], allocate_pair=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_global_ids_beyond_private_context_and_padding(self):
        owner = pool()
        tables = owner.normalize_tables(
            batch_size=2, capacity=125, page_table=[[15, 11, -1], [12, -1, -1]], lengths=[33, 1]
        )
        self.assertEqual([item.tolist() for item in tables], [[[15, 11, 0, 0], [12, 0, 0, 0]]] * 2)
        reserved = owner.normalize_tables(batch_size=1, capacity=125, page_table=[[15, 11, 10, -1]], lengths=[1])
        self.assertEqual(reserved[0].tolist(), [[15, 11, 10, 0]])

    def test_shared_index_layers_require_disjoint_tables(self):
        owner = pool(shared=True)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            owner.normalize_tables(batch_size=1, capacity=128, page_table=[[15]], lengths=[1])
        tables = owner.normalize_tables(
            batch_size=2, capacity=128, page_tables_by_layer=[[[15], [14]], [[13], [12]]], lengths=[1, 1]
        )
        self.assertTrue(owner.validate_active_pages(tables, [1, 1]))

    def test_inactive_padding_not_claimed_and_duplicate_live_pages_rejected(self):
        owner = pool()
        tables = owner.normalize_tables(batch_size=2, capacity=128, page_table=[[15], [-1]], lengths=[1, 0])
        self.assertEqual(tables[0].tolist(), [[15, 0, 0, 0], [0, 0, 0, 0]])
        for rows, lengths in (([[15], [15]], [1, 1]), ([[1, 1], [0]], [33, 0])):
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                owner.normalize_tables(batch_size=2, capacity=128, page_table=rows, lengths=lengths)

    def test_invalid_tables_cannot_truncate_or_wrap_ids(self):
        owner = pool()
        for table, length in (([[16]], 1), ([[-1]], 1), ([[1.5]], 1), ([[]], 1), ([[1] * 5], 1)):
            with self.assertRaises(ValueError):
                owner.normalize_tables(batch_size=1, capacity=128, page_table=table, lengths=[length])

    def test_warmup_tables_reserve_disjoint_shared_group_pages(self):
        owner = pool(shared=True)
        tables = owner.warmup_tables(batch_size=3, capacity=128)
        self.assertEqual(tables[0].tolist(), [[0, 1, 0, 0], [2, 3, 0, 0], [4, 5, 0, 0]])
        self.assertEqual(tables[1].tolist(), [[6, 7, 0, 0], [8, 9, 0, 0], [10, 11, 0, 0]])
        with self.assertRaisesRegex(ValueError, "warmup pages"):
            owner.warmup_tables(batch_size=5, capacity=128)


if __name__ == "__main__":
    unittest.main()
