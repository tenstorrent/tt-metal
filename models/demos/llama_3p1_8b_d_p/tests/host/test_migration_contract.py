# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only migration contracts. These tests do not validate device writes."""

import importlib
import sys
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

PREFIX = "models.demos.llama_3p1_8b_d_p.tt."


class MigrationContractTests(unittest.TestCase):
    def layout(self, **kwargs):
        try:
            module = importlib.import_module(PREFIX + "runners.kv_layout")
        except ModuleNotFoundError:
            self.fail("portable KV layout module has not been implemented")
        return module.PrefillKVLayout(**{"num_banks": 8, **kwargs})

    # Every physical NdShard page must map once to the matching slot, layer, head, and SP row.
    def test_complete_two_slot_table_matches_independent_bank_walk(self):
        layout = self.layout(max_seq_len=2048, num_slots=2, num_layers=32, num_banks=8)
        seen = set()
        for config in range(16):
            for row in range(4):
                page = 0
                for slot in range(2):
                    for layer in range(32):
                        for block in range(2):
                            for local in range(0, 256, 32):
                                position = block * 1024 + row * 256 + local
                                coord, bank, offset = layout.locate(config, layer, position, slot, 65536)
                                self.assertEqual(coord, (row, config % 8))
                                self.assertEqual((bank, offset), (page % 8, 65536 + page // 8 * 4352))
                                key = (config // 8, coord, bank, offset)
                                self.assertNotIn(key, seen)
                                seen.add(key)
                                page += 1
        self.assertEqual(len(seen), 16 * 32 * 64 * 2)

    # Long-context metadata changes the slot stride; no 2K constant may leak into address calculation.
    def test_long_context_boundaries_and_config_order(self):
        layout = self.layout(max_seq_len=131072, num_slots=2, num_layers=32, num_banks=8)
        self.assertEqual(layout.config_names, tuple(f"{kv}_h{h}" for kv in ("k", "v") for h in range(8)))
        self.assertEqual(sorted(layout.config_names), list(layout.config_names))
        for length in (2048, 4096, 8192, 16384, 32768, 65536, 131072):
            candidate = self.layout(max_seq_len=length, num_slots=2, num_layers=32, num_banks=8)
            for position in (0, 224, 256, 992, 1024, length - 32):
                first = candidate.locate(15, 31, position, 0, 65536)
                second = candidate.locate(15, 31, position, 1, 65536)
                self.assertNotEqual(first, second)
                self.assertEqual(first[0], ((position % 1024) // 256, 7))
        self.assertEqual(layout.chunk_size_bytes, 4352)

    # Bank count is a device property and must never be guessed by the generic helper.
    def test_bank_count_is_required_and_generic_counts_work(self):
        module = importlib.import_module(PREFIX + "runners.kv_layout")
        with self.assertRaises(TypeError):
            module.PrefillKVLayout()
        generic = module.PrefillKVLayout(num_banks=12)
        self.assertEqual(generic.locate(0, 0, 1024, 0, 65536)[1:], (8, 65536))

    # Invalid metadata must fail before it can manufacture an address outside the cache.
    def test_invalid_layout_and_lookup(self):
        for kwargs in (dict(max_seq_len=2016), dict(num_slots=0), dict(num_banks=0)):
            with self.assertRaises((ValueError, TypeError)):
                self.layout(**kwargs)
        layout = self.layout()
        for args in ((16, 0, 0, 0, 0), (0, 32, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 2048, 0, 0), (0, 0, 0, 2, 0)):
            with self.assertRaises((ValueError, TypeError)):
                layout.locate(*args)


class RuntimeContractTests(unittest.TestCase):
    def make_runtime(self, *, failure=None):
        try:
            mod = importlib.import_module(PREFIX + "tt_prefill_runtime")
        except ModuleNotFoundError:
            self.fail("portable prefill runtime has not been implemented")
        self.events = []
        self.output = SimpleNamespace(deallocate=lambda force: self.events.append(("free", force)))

        def forward(tokens, cache, **kwargs):
            self.events.append(("forward", tokens, kwargs))
            if failure == "forward":
                raise RuntimeError("forward failed")
            return self.output

        def synchronize(mesh):
            self.events.append(("sync", mesh))
            if failure == "sync":
                raise RuntimeError("sync failed")

        model = SimpleNamespace(prefill_chunk=forward, num_layers=32, max_seq_len=2048)
        config = mod.TtPrefillRuntimeConfig(max_seq_len=2048, chunk_size=1024, num_users=2)
        runtime = mod.TtPrefillRuntime(
            "mesh", config=config, model=model, synchronize=synchronize, upload=lambda tokens, **kw: tokens
        )
        runtime.compiled = True
        self.cache = SimpleNamespace(num_users=2, num_layers=32, max_seq_len=2048, sp=4)
        return runtime

    def call(self, runtime, *, request=7, slot=1, start=32, end=65, tokens="already-shuffled"):
        return runtime.prefill_chunk(
            tokens,
            self.cache,
            slot_id=slot,
            actual_start=start,
            actual_end=end,
            request_id=request,
            metadata_msg="borrowed",
            d2h_service=None,
        )

    # The factory selects the persistent migration dtype explicitly instead of relying on a model default.
    def test_factory_explicitly_selects_bfp8_cache(self):
        mod = importlib.import_module(PREFIX + "tt_prefill_runtime")
        params = SimpleNamespace(
            max_seq_len=2048,
            chunk_size=1024,
            num_users=2,
            num_layers=32,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            mesh_shape=(4, 8),
            sp_axis=0,
            tp_axis=1,
            use_trace=False,
            dflash_enabled=False,
            tp_shard_kv=False,
        )
        seen = {}

        def model(*args, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(num_layers=32, max_seq_len=kwargs["max_seq_len"])

        dtype = object()
        with patch.dict(
            sys.modules,
            {
                "ttnn": SimpleNamespace(bfloat8_b=dtype, synchronize_device=lambda mesh: None),
                PREFIX + "model": SimpleNamespace(PrefillModel=model),
                PREFIX + "input": SimpleNamespace(upload_token_chunk=lambda *args, **kw: None),
            },
        ):
            mod.build_runtime("mesh", params=params, checkpoint_path="checkpoint")
        self.assertIs(seen.get("cache_dtype"), dtype)
        self.assertIs(seen["enable_lm_head"], False)

    # The common runner reads use_trace directly even on the single-rank eager path.
    def test_runtime_declares_eager_mode(self):
        runtime = self.make_runtime()
        self.assertIs(getattr(runtime.config, "use_trace", None), False)

    # Warmup owns its uploaded tokens, emits no readiness, and compiles both initial and continuation calls.
    def test_compile_warms_two_chunks_without_acknowledgments(self):
        runtime = self.make_runtime()
        runtime.compiled = False
        uploads = []

        def upload(tokens, **bounds):
            uploads.append((len(tokens), bounds))
            return SimpleNamespace(deallocate=lambda force: self.events.append(("free-input", force)))

        runtime._upload = upload
        runtime.compile(self.cache)
        self.assertTrue(runtime.compiled)
        self.assertEqual(
            uploads, [(1024, dict(actual_start=0, actual_end=1024)), (1024, dict(actual_start=1024, actual_end=2048))]
        )
        self.assertEqual(sum(e[0] == "sync" for e in self.events), 2)
        self.assertEqual(sum(e[0] == "free-input" for e in self.events), 2)
        self.assertFalse(any(e[0] == "ack" for e in self.events))
        count = len(self.events)
        runtime.compile(self.cache)
        self.assertEqual(count, len(self.events))

    # The common multicache gather and builder must agree on all 16 stage bases and real keyword names.
    def test_migration_hooks_match_common_runner_surface(self):
        runtime = self.make_runtime()
        self.cache.k = SimpleNamespace(buffer_address=lambda: 65536)
        self.cache.v = SimpleNamespace(buffer_address=lambda: 1048576)
        stage_type = namedtuple("KvCacheStage", "base_addr first_layer count")
        shared = "models.demos.common.prefill.runners.migration"
        builder = PREFIX + "runners.kv_chunk_table"
        captured = {}

        def serialize(**kwargs):
            captured.update(kwargs)
            return kwargs["path"]

        with patch.dict(
            sys.modules,
            {
                shared: SimpleNamespace(KvCacheStage=stage_type),
                builder: SimpleNamespace(build_and_serialize_kv_chunk_table=serialize),
            },
        ):
            stages = runtime.kv_migration_stages(self.cache, 0, 32)
            self.assertEqual([s.base_addr for s in stages], [65536] * 8 + [1048576] * 8)
            layouts = [[dict(base_addr=s.base_addr, first_layer=0, count=32)] for s in stages]
            self.assertEqual(
                runtime.build_kv_chunk_table(
                    self.cache, "table.pb", first_layer_idx=0, num_my_layers=32, stage_layouts=layouts
                ),
                "table.pb",
            )
            self.assertIs(captured["kv_cache"], self.cache)
            self.assertEqual(captured["chunk_size"], 1024)
            with self.assertRaises(ValueError):
                runtime.build_kv_chunk_table(self.cache, "table.pb", stage_layouts=layouts[:8])

    # Acknowledgments identify the actual call and occur only after forward and synchronization finish.
    def test_two_distinct_slots_and_request_ids(self):
        runtime = self.make_runtime()
        runtime.set_layer_completion_sink(lambda layer, req: self.events.append(("ack", layer, req)))
        runtime._upload = lambda *args, **kwargs: self.fail("H2D input must not be packed again")
        for request, slot, start, end in ((7, 1, 32, 65), (8, 0, 1024, 1033)):
            begin = len(self.events)
            tokens = ["prompt-A" if slot else "prompt-B", request]
            self.assertIsNone(self.call(runtime, request=request, slot=slot, start=start, end=end, tokens=tokens))
            events = self.events[begin:]
            self.assertEqual(events[0][0], "forward")
            self.assertIs(events[0][1], tokens)
            self.assertEqual(events[0][2], dict(slot_idx=slot, actual_start=start, actual_end=end, skip_lm_head=True))
            self.assertEqual(events[1], ("sync", "mesh"))
            self.assertEqual([e for e in events if e[0] == "ack"], [("ack", layer, request) for layer in range(32)])
            self.assertIn(("free", True), events)

    # A failed write or wait cannot certify any layer; the failed runtime cannot be reused.
    def test_forward_and_sync_failure_never_ack(self):
        for failure in ("forward", "sync"):
            runtime = self.make_runtime(failure=failure)
            runtime.set_layer_completion_sink(lambda *args: self.events.append(("ack", *args)))
            with self.assertRaisesRegex(RuntimeError, "failed"):
                self.call(runtime)
            self.assertFalse(any(event[0] == "ack" for event in self.events))
            with self.assertRaisesRegex(RuntimeError, "failed"):
                self.call(runtime, request=8)

    # A partly emitted completion sequence cannot be replayed after a sink exception.
    def test_sink_failure_propagates_and_poison_runtime(self):
        runtime = self.make_runtime()

        def sink(layer, request):
            self.events.append(("ack", layer, request))
            if layer == 3:
                raise RuntimeError("sink failed")

        runtime.set_layer_completion_sink(sink)
        with self.assertRaisesRegex(RuntimeError, "sink failed"):
            self.call(runtime)
        self.assertEqual(len([e for e in self.events if e[0] == "ack"]), 4)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.call(runtime, request=8)

    # Replacing a sink between chunks is supported; replacing it inside a call is rejected.
    def test_sink_replacement_is_explicit(self):
        runtime = self.make_runtime()
        old, new = [], []
        runtime.set_layer_completion_sink(lambda *args: old.append(args))
        runtime.set_layer_completion_sink(lambda *args: new.append(args))
        self.call(runtime)
        self.assertEqual(old, [])
        self.assertEqual(len(new), 32)

        def sink(*args):
            with self.assertRaisesRegex(RuntimeError, "active"):
                runtime.set_layer_completion_sink(None)

        runtime.set_layer_completion_sink(sink)
        self.call(runtime, request=8)
        runtime.set_layer_completion_sink(None)

    # A stale chunk counter cannot emit a duplicate acknowledgment sequence.
    def test_duplicate_request_rejected(self):
        runtime = self.make_runtime()
        runtime.set_layer_completion_sink(lambda *args: None)
        self.call(runtime)
        count = len(self.events)
        with self.assertRaisesRegex(ValueError, "request_id"):
            self.call(runtime)
        self.assertEqual(len(self.events), count)

    # Empty intervals are rejected before dispatch; no chunk may silently omit its 32 acknowledgments.
    def test_empty_invalid_and_unsupported_requests(self):
        runtime = self.make_runtime()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.call(runtime, start=32, end=32)
        self.assertEqual(self.events, [])
        for changes in (dict(slot=2), dict(start=1), dict(end=2049), dict(request=-1)):
            with self.assertRaises((ValueError, TypeError)):
                self.call(runtime, **changes)
        with self.assertRaises(NotImplementedError):
            runtime.prefill_chunk("x", self.cache, slot_id=0, actual_start=0, actual_end=32, d2h_service=object())
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
