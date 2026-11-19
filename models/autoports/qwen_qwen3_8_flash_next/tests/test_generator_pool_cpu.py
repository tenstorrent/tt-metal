# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""AST-extracted generator pool lifecycle and scheduler checks; no TTNN import."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests import test_decode_validation_cpu as decode_validation
from models.autoports.qwen_qwen3_8_flash_next.tt.cache_budget import KNOWN_STATIC_BYTES, plan_shared_kv_cache
from models.autoports.qwen_qwen3_8_flash_next.tt.cache_pool import SharedKVPool, normalize_specs
from models.autoports.qwen_qwen3_8_flash_next.tt.precision import load_policy, static_weight_delta


class GeneratorPoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        decode_validation.DecodeValidationTest.setUpClass.__func__(cls)
        source = Path(__file__).resolve().parents[1] / "tt/generator.py"
        owner = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator"
        )
        names = {
            "allocate_kv_cache_per_layer",
            "allocate_kv_cache",
            "_bind",
            "warmup_model_prefill",
            "warmup_model_decode",
            "get_max_tokens_all_users",
        }
        nodes = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), cls.namespace)
        for name in names:
            setattr(cls.gen_type, name, cls.namespace[name])
        cls.namespace["normalize_specs"] = normalize_specs
        cls.namespace["plan_shared_kv_cache"] = plan_shared_kv_cache
        cls.namespace.update(
            load_policy=load_policy,
            static_weight_delta=static_weight_delta,
            KNOWN_STATIC_BYTES=KNOWN_STATIC_BYTES,
            Path=Path,
            __file__=str(source),
        )

    def setUp(self):
        decode_validation.DecodeValidationTest.setUp(self)
        self.g.shared_pool = SharedKVPool(
            [((16, 1, 32, 256), "bfloat16", 0)],
            allocate_pair=lambda mesh, spec: tuple(
                SimpleNamespace(shape=spec.shape, dtype="bfloat16", layout="tile") for _ in range(2)
            ),
        )
        self.g.page_tables = {}
        for cache in self.g.caches:
            cache.position = 0

    def table(self, *ids):
        return torch.tensor([list(ids) + [0] * (4 - len(ids))], dtype=torch.int32)

    def test_dormant_request_pages_cannot_be_overwritten(self):
        g = self.g
        g.caches[0].position = 3
        g.page_tables[0] = (self.table(15),)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            g._table_plan(self.table(15), [g.caches[1]], (1,), [3])
        valid = g._table_plan(self.table(14), [g.caches[1]], (1,), [3])
        self.assertEqual(valid[0].tolist(), [[14, 0, 0, 0]])
        self.assertEqual(self.events, [])

    def test_remap_validation_uses_source_ownership_before_copy(self):
        g = self.g
        for slot, block in enumerate((15, 14)):
            g.caches[slot].position = 3
            g.page_tables[slot] = (self.table(block),)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            g._table_plan(self.table(15), [g.caches[0]], (0,), [4], remap=(1, 0))
        g._table_plan(self.table(14), [g.caches[0]], (0,), [4], remap=(1, 0))
        self.assertEqual(self.events, [])

    def test_boundary_checks_new_page_but_unchanged_replay_does_not_rebuild(self):
        g = self.g
        g.caches[0].position, g.caches[1].position = 32, 3
        g.page_tables = {0: (self.table(15, 14),), 1: (self.table(14),)}
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            g._table_plan(None, [g.caches[0]], (0,), [33])
        g.caches[0].position = 30
        g.shared_pool.validate_active_pages = lambda *args: self.fail("Unchanged within-page replay rebuilt tables")
        self.assertIsNone(g._table_plan(None, [g.caches[0]], (0,), [31]))

    def test_per_layer_padded_rows_override_group_zero_and_use_submission_order(self):
        g = self.g
        table = torch.cat([self.table(15), self.table(14)])
        normalized = g._table_plan(
            self.table(99), [g.caches[1]], (1,), [3], page_tables_per_layer=[table], table_rows=[0], submission_rows=1
        )
        self.assertEqual(normalized[0][0, 0].item(), 15)
        with self.assertRaisesRegex(ValueError, "explicit scheduler"):
            g._table_plan(None, [g.caches[0]], (0,), [3])

    def allocation_fixture(self):
        g = self.g
        g.shared_pool, g.cache_setup_error, g.owns_cache_storage = None, None, True
        g.max_batch_size, g.caches = 1, g.caches[:1]
        g.prepared = False
        g.model.precision = {"kv_cache_dtype": "bfloat16"}
        g.traces, g.prefill_traces, g.sampling_traces = {}, {}, {}
        g.slot_mover = SimpleNamespace(scratch=None)
        state = g.caches[0].layers[0].attention
        state.capacity, state.page_size = 128, 32
        state.v_cache = SimpleNamespace(shape=(4, 1, 32, 256))
        self.namespace["ttnn"].deallocate = lambda tensor: self.events.append("free")
        self.namespace["ttnn"].synchronize_device = lambda mesh: self.events.append("sync")
        self.namespace["SlotStateMover"] = lambda caches, scratch: SimpleNamespace(caches=caches, scratch=scratch)
        return g

    def test_allocation_frees_owned_private_before_global_allocation(self):
        g = self.allocation_fixture()
        events = self.events

        class CpuPool(SharedKVPool):
            def __init__(self, specs, mesh):
                events.append("allocate")
                super().__init__(
                    specs,
                    allocate_pair=lambda _, spec: tuple(
                        SimpleNamespace(shape=spec.shape, dtype="bfloat16", layout="tile") for _ in range(2)
                    ),
                )

        original = self.namespace["SharedKVPool"]
        self.namespace["SharedKVPool"] = CpuPool
        try:
            pool = g.allocate_kv_cache_per_layer([((16, 1, 32, 256), "bfloat16", 0)])
            self.assertEqual(events[:4], ["sync", "free", "free", "allocate"])
            self.assertIs(g._resolve_caches(pool, (0,))[0], g.caches[0])
            self.assertTrue(g.caches[0].shared_kv)
            self.assertIsNone(g.cache_setup_error)
        finally:
            self.namespace["SharedKVPool"] = original

    def test_allocator_failure_poison_prevents_old_cache_fallback(self):
        g = self.allocation_fixture()
        original = self.namespace["SharedKVPool"]

        class FailedPool(SharedKVPool):
            def __init__(self, *args):
                raise MemoryError("injected allocation failure")

        self.namespace["SharedKVPool"] = FailedPool
        try:
            with self.assertRaises(MemoryError):
                g.allocate_kv_cache_per_layer([((16, 1, 32, 256), "bfloat16", 0)])
            with self.assertRaisesRegex(RuntimeError, "reconstruct"):
                g._resolve_caches(None, (0,))
            self.assertIsNone(g.caches[0].layers[0].attention.k_cache)
        finally:
            self.namespace["SharedKVPool"] = original

    def test_bad_specs_borrowed_or_prepared_storage_reject_without_free(self):
        g = self.allocation_fixture()
        with self.assertRaises(ValueError):
            g.allocate_kv_cache_per_layer([((16, 1, 32, 256), "bfloat8_b", 0)])
        g.owns_cache_storage = False
        with self.assertRaisesRegex(ValueError, "externally bound"):
            g.allocate_kv_cache_per_layer([((16, 1, 32, 256), "bfloat16", 0)])
        g.prepared = True
        with self.assertRaisesRegex(RuntimeError, "preparation"):
            g.allocate_kv_cache_per_layer([((16, 1, 32, 256), "bfloat16", 0)])
        self.assertEqual(self.events, [])

    def test_deferred_storage_cannot_execute_and_has_no_private_kv_to_free(self):
        g = self.allocation_fixture()
        g.cache_binding_pending = True
        state = g.caches[0].layers[0].attention
        state.k_cache = state.v_cache = None
        with self.assertRaisesRegex(RuntimeError, "Deferred"):
            g._check_cache_setup()
        original = self.namespace["SharedKVPool"]

        class CpuPool(SharedKVPool):
            def __init__(self, specs, mesh):
                super().__init__(
                    specs,
                    allocate_pair=lambda _, spec: tuple(
                        SimpleNamespace(shape=spec.shape, dtype="bfloat16", layout="tile") for _ in range(2)
                    ),
                )

        self.namespace["SharedKVPool"] = CpuPool
        try:
            g.allocate_kv_cache_per_layer([((16, 1, 32, 256), "bfloat16", 0)])
            g._check_cache_setup()
            self.assertNotIn("free", self.events)
            self.assertTrue(g.caches[0].kv_bound)
        finally:
            self.namespace["SharedKVPool"] = original

    def test_deferred_decoder_factory_omits_only_kv_allocations(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "tt/multichip_decoder.py").read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultichipDecoder")
        node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_create_base_state")
        allocations = []

        def zeros(shape, **kwargs):
            allocations.append(tuple(shape))
            return SimpleNamespace(shape=shape, **kwargs)

        namespace = dict(
            QSAState=SimpleNamespace,
            LinearState=SimpleNamespace,
            ttnn=SimpleNamespace(
                zeros=zeros,
                from_torch=lambda value, **kw: value.clone(),
                int32="int32",
                float32="float32",
                bfloat16="bfloat16",
                TILE_LAYOUT="tile",
            ),
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "decoder", "exec"), namespace)
        decoder = SimpleNamespace(
            kind="indexed_attention",
            kv_dtype="bfloat16",
            config=SimpleNamespace(max_position_embeddings=128),
            mesh_device=object(),
            workspace=SimpleNamespace(index_arrays={128: (object(), object())}),
        )
        deferred = namespace[node.name](decoder, capacity=128, defer_kv_cache=True).attention
        self.assertEqual(allocations, [(1, 1, 3, 128), (1, 1, 32, 128)])
        self.assertIsNone(deferred.k_cache)
        self.assertIsNone(deferred.v_cache)
        allocations.clear()
        default = namespace[node.name](decoder, capacity=128).attention
        self.assertEqual(allocations[:2], [(4, 1, 32, 256)] * 2)
        self.assertTrue(torch.equal(default.page_table, deferred.page_table))
        self.assertTrue(torch.equal(default.position, deferred.position))

    def test_unbound_model_rejects_before_embedding_or_device_work(self):
        source = Path(__file__).resolve().parents[1] / "tt/model.py"
        owner = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenModel"
        )
        nodes = [
            n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in ("forward_chunk", "decode_batch")
        ]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        model = SimpleNamespace(embed=lambda *args: self.fail("Unbound model issued embedding"))
        cache = SimpleNamespace(kv_bound=False)
        with self.assertRaisesRegex(RuntimeError, "bound"):
            namespace["forward_chunk"](model, None, None, cache=cache)
        with self.assertRaisesRegex(RuntimeError, "bound"):
            namespace["decode_batch"](model, [None], [None], caches=[cache])

    def test_pinned_sizing_getter_accounts_for_worker_padding_and_preserves_context(self):
        policy = load_policy(Path(__file__).resolve().parents[1])
        expected = plan_shared_kv_cache(
            262144,
            1,
            known_static_bytes=KNOWN_STATIC_BYTES + static_weight_delta(policy),
            kv_cache_dtype=policy["kv_cache_dtype"],
        ).plugin_max_tokens_all_users()
        self.assertEqual(self.gen_type.get_max_tokens_all_users(max_model_len=262144, max_num_seqs=1), expected)
        self.assertGreaterEqual(expected, 262144)
        with self.assertRaisesRegex(ValueError, "TP4"):
            self.gen_type.get_max_tokens_all_users(num_devices=1)
        with self.assertRaisesRegex(ValueError, "cannot admit"):
            self.gen_type.get_max_tokens_all_users(max_model_len=262144, max_num_seqs=32)


if __name__ == "__main__":
    unittest.main()
