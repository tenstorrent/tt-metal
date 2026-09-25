# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual serving prefill methods with CPU tensors and fake TTNN effects."""

import ast
import unittest
import weakref
from collections import Counter
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


def load_methods(filename, class_name, names, ops):
    source = Path(__file__).resolve().parents[2] / "tt" / filename
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    found = {node.name for node in methods}
    if found != set(names):
        raise AssertionError(f"Missing methods: {set(names) - found}")
    namespace = {"ttnn": ops, "torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class FakeLogits:
    shape = (1, 1, 1, 62080)
    dtype = "bfloat16"
    layout = "TILE"


class ServingPrefillTraceTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.owned = object()
        self.ops = SimpleNamespace(
            uint32="uint32",
            int32="int32",
            Tensor=FakeLogits,
            ROW_MAJOR_LAYOUT="ROW_MAJOR",
            execute_trace=lambda mesh, trace, **kw: self.events.append(("execute", trace, kw)),
            release_trace=lambda mesh, trace: self.events.append(("release", trace)),
            clone=lambda tensor: object(),
            copy=lambda source, target: self.events.append(("copy", source, target)),
            concat=lambda outputs, **kwargs: self.events.append(("concat", tuple(outputs))) or FakeLogits(),
            pad=lambda tensor, *args, **kwargs: self.events.append(("pad",)) or FakeLogits(),
            synchronize_device=lambda mesh: self.events.append(("sync",)),
        )
        self.methods = load_methods(
            "generator.py",
            "Qwen38Generator",
            [
                "serving_prefill_tokens",
                "_prepare_serving_prefill_sampling",
                "_release_traces",
                "_prefill_for_generate",
                "_capture",
                "_ensure_cache",
                "bind_cache",
                "_prefill_trace_logits",
                "reset_recurrent_slots",
            ],
            self.ops,
        )
        self.gen = SimpleNamespace(
            mesh=object(),
            cache=SimpleNamespace(batch_size=4, capacity=8192, num_pages=1024, layers=[]),
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=250000)),
            page_table=SimpleNamespace(shape=(4, 256)),
            tokens=object(),
            positions=object(),
            rope_indices=object(),
            sampler=SimpleNamespace(seeds_tt_tensor=object()),
            prefill_prepared=None,
            prefill_signatures=set(),
            trace=None,
            sample_trace=None,
            prefill_trace=None,
            prefill_sample_trace=None,
            prefill_sample_input=None,
            trace_records_history=None,
            counters=Counter(),
        )
        self.gen._refresh_table = lambda table: self.events.append(("table", table))
        self.gen._sampling_step = lambda output: self.events.append(("sample", output))
        self.gen._release_traces = MethodType(self.methods["_release_traces"], self.gen)
        self.gen._prepare_serving_prefill_sampling = MethodType(
            self.methods["_prepare_serving_prefill_sampling"], self.gen
        )

        def owned(tokens, *, trace_sampling=False):
            self.events.append(("owned", tokens.clone(), trace_sampling))
            self.gen.prefill_prepared = {"output": self.owned, "trace_sampling": trace_sampling}
            return self.owned

        def public(tokens, **kwargs):
            output = FakeLogits()
            self.events.append(("public", tokens.clone(), kwargs, output))
            return [output]

        self.gen._prefill_for_generate = owned
        self.gen.prefill_forward = public
        self.serving = MethodType(self.methods["serving_prefill_tokens"], self.gen)

    def request(self, tokens=None, *, ends=(3,), starts=(0,), slots=(0,), cache=None):
        tokens = torch.arange(max(ends)).repeat(len(ends), 1) if tokens is None else tokens
        return self.serving(
            tokens,
            page_table=self.gen.page_table,
            kv_cache=self.gen.cache if cache is None else cache,
            prompt_lens=ends,
            start_pos=starts,
            slots=slots,
        )

    def test_single_fresh_row_refreshes_table_before_owned_prefill_and_samples_once(self):
        result = self.request(torch.tensor([[11, 12, 13, 99]]))
        self.assertIs(result, self.gen.tokens)
        self.assertEqual([event[0] for event in self.events], ["table", "owned", "sample"])
        self.assertEqual(self.events[1][1].tolist(), [[11, 12, 13]])
        self.assertTrue(self.events[1][2])
        self.assertIs(self.events[2][1], self.owned)
        self.assertEqual(self.gen.counters["prefill_sampling_eager_calls"], 1)

    def test_opt_in_groups_equal_prompts_before_sampling(self):
        self.gen.batched_prefill = True
        calls = []

        def public(tokens, **kwargs):
            calls.append((tokens.clone(), kwargs))
            return [FakeLogits() for _ in kwargs["slots"]]

        self.gen.prefill_forward = public
        self.request(ends=(64, 64), starts=(32, 32), slots=(1, 2))
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0][0].shape), (2, 32))
        self.assertEqual(calls[0][1]["prompt_lens"], [32, 32])
        self.assertEqual(calls[0][1]["start_pos"], [32, 32])
        self.assertEqual(calls[0][1]["slots"], [1, 2])
        self.assertEqual(sum(event[0] == "sample" for event in self.events), 1)

    def test_warm_sampling_replays_the_prefill_sampler_nonblocking(self):
        self.gen.prefill_sample_trace = "prefill-sampler"
        self.gen.sample_trace = "decode-sampler"
        result = self.request()
        self.assertIs(result, self.gen.tokens)
        self.assertEqual([event[0] for event in self.events], ["table", "owned", "execute"])
        self.assertEqual(self.events[-1], ("execute", "prefill-sampler", {"cq_id": 0, "blocking": False}))
        self.assertEqual(self.gen.counters["prefill_sampling_replays"], 1)
        self.assertEqual(self.gen.counters["prefill_sampling_eager_calls"], 0)

    def test_multirow_continuation_keeps_exact_old_slicing_and_packed_order(self):
        tokens = torch.arange(80).reshape(2, 40)
        self.request(tokens, ends=(3, 35), starts=(0, 32), slots=(3, 1))
        self.assertEqual([event[0] for event in self.events], ["public", "public", "concat", "pad", "copy", "sample"])
        for row, (start, end, slot) in enumerate(((0, 3, 3), (32, 35, 1))):
            event = self.events[row]
            self.assertTrue(torch.equal(event[1], tokens[row : row + 1, start:end]))
            self.assertEqual(event[2]["prompt_lens"], [end - start])
            self.assertEqual(event[2]["start_pos"], [start])
            self.assertEqual(event[2]["slots"], [slot])
            self.assertIs(event[2]["kv_cache"], self.gen.cache)
        self.assertEqual(self.events[2][1], (self.events[0][3], self.events[1][3]))
        self.assertIsNot(self.events[0][3], self.events[1][3])
        self.assertEqual(self.gen.counters["prefill_sampling_eager_calls"], 1)

    def test_long_other_slot_and_continuation_use_public_fallback(self):
        for ends, starts, slots in (((4097,), (0,), (0,)), ((3,), (0,), (2,)), ((35,), (32,), (0,))):
            with self.subTest(ends=ends, starts=starts, slots=slots):
                self.events.clear()
                self.request(ends=ends, starts=starts, slots=slots)
                self.assertEqual([event[0] for event in self.events], ["public", "pad", "copy", "sample"])

    def test_boundary_lengths_are_eligible(self):
        for length in (1, 4095, 4096):
            with self.subTest(length=length):
                self.events.clear()
                self.request(ends=(length,))
                self.assertEqual([event[0] for event in self.events], ["table", "owned", "sample"])

    def test_wrong_cache_and_invalid_geometry_fail_before_device_effects(self):
        cases = [
            dict(cache=object()),
            dict(ends=(0,), starts=(0,)),
            dict(ends=(3,), starts=(-1,)),
            dict(ends=(3,), starts=(3,)),
            dict(ends=(3,), slots=(4,)),
            dict(ends=(3, 4), starts=(0, 0), slots=(0, 0)),
            dict(ends=(8193,)),
            dict(tokens=torch.tensor([[1, 2]]), ends=(3,)),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.events.clear()
                with self.assertRaises(ValueError):
                    self.request(**kwargs)
                self.assertEqual(self.events, [])

    def test_release_drops_four_handles_but_can_keep_preparation(self):
        self.gen.trace, self.gen.sample_trace = "decode", "decode-sample"
        self.gen.prefill_trace, self.gen.prefill_sample_trace = "prefill", "prefill-sample"
        prepared = self.gen.prefill_prepared = {"output": self.owned, "trace_sampling": True}
        staging = self.gen.prefill_sample_input = object()
        self.gen._release_traces(keep_prefill=True)
        self.assertEqual(
            self.events, [("release", name) for name in ("decode", "decode-sample", "prefill", "prefill-sample")]
        )
        self.assertIs(self.gen.prefill_prepared, prepared)
        self.assertIs(self.gen.prefill_sample_input, staging)
        for name in ("trace", "sample_trace", "prefill_trace", "prefill_sample_trace"):
            self.assertIsNone(getattr(self.gen, name))
        self.gen._release_traces()
        self.assertIsNone(self.gen.prefill_prepared)
        self.assertIs(self.gen.prefill_sample_input, staging)

    def test_preparation_mode_change_releases_before_new_persistent_allocations(self):
        def upload(tensor, **kwargs):
            self.events.append(("upload",))
            return SimpleNamespace(shape=tuple(tensor.shape))

        self.gen.model.upload = upload
        self.gen._prefill_trace_logits = lambda: self.owned
        self.gen._prefill_trace_step = lambda: None
        self.gen._copy = lambda *args: None
        prepare = MethodType(self.methods["_prefill_for_generate"], self.gen)
        tokens = torch.tensor([[11, 12, 13]])
        prepare(tokens)
        standalone = self.gen.prefill_prepared
        self.assertFalse(standalone["trace_sampling"])
        self.gen.trace = "old-decode"
        self.events.clear()
        prepare(tokens, trace_sampling=True)
        serving = self.gen.prefill_prepared
        self.assertIsNot(serving, standalone)
        self.assertTrue(serving["trace_sampling"])
        self.assertEqual(self.events[0], ("release", "old-decode"))
        self.assertEqual(self.events[1], ("upload",))
        prepare(tokens, trace_sampling=True)
        self.assertIs(self.gen.prefill_prepared, serving)
        prepare(tokens)
        self.assertFalse(self.gen.prefill_prepared["trace_sampling"])
        self.assertIsNot(self.gen.prefill_prepared, serving)

    def capture(self, *, serving, fail_sampling=False, fallback=False):
        next_trace = 0

        def begin(mesh, *, cq_id):
            nonlocal next_trace
            next_trace += 1
            self.events.append(("begin", next_trace))
            return next_trace

        def sample(output):
            self.events.append(("sample", output))
            if fail_sampling and output is self.owned:
                raise RuntimeError("Controlled fourth-capture failure")

        self.ops.begin_trace_capture = begin
        self.ops.end_trace_capture = lambda mesh, trace, **kw: self.events.append(("end", trace))
        self.gen.prefill_prepared = {"output": self.owned, "trace_sampling": serving}
        if fallback:
            self.gen.prefill_prepared = None
            self.gen.prefill_sample_input = self.owned
        self.gen._model_step = lambda: "decode-logits"
        self.gen._sampling_step = sample
        self.gen._prefill_trace_step = lambda: self.events.append(("prefill-copy",))
        return self.methods["_capture"](self.gen)

    def test_serving_captures_fourth_sampler_over_owned_output(self):
        self.capture(serving=True)
        self.assertEqual(
            (self.gen.trace, self.gen.sample_trace, self.gen.prefill_trace, self.gen.prefill_sample_trace), (1, 2, 3, 4)
        )
        fourth = self.events.index(("begin", 4))
        self.assertEqual(self.events[fourth : fourth + 3], [("begin", 4), ("sample", self.owned), ("end", 4)])
        self.assertEqual(self.gen.counters["prefill_sampling_trace_captures"], 1)

    def test_release_publishes_resident_state_after_releasing_traces(self):
        self.gen.trace = "decode"
        self.gen.sample_trace = "sample"
        self.gen.model._resident_decode_bucket = object()
        self.gen.model.flush_decode_bucket = lambda: self.events.append(("flush",))
        self.gen._release_traces()
        self.assertEqual(self.events, [("release", "decode"), ("release", "sample"), ("flush",)])

    def test_capture_restores_resident_state_not_stale_scheduler_state(self):
        conv, recurrent = object(), object()
        resident = SimpleNamespace(layers=[SimpleNamespace(conv=conv, recurrent=recurrent)])
        self.gen.active_slots = (0,)
        self.gen.model.decode_buckets = True
        self.gen.model.prepare_decode_bucket = lambda cache, slots: resident
        self.capture(serving=False)
        restored = [event[2] for event in self.events if event[0] == "copy"]
        self.assertIn(conv, restored)
        self.assertIn(recurrent, restored)

    def test_single_slot_prefill_uses_resident_state(self):
        resident = SimpleNamespace(batch_size=1)
        self.gen.model._resident_decode_bucket = (self.gen.cache, (0,), resident)
        self.gen.model._resident_decode_valid = True
        self.gen.prefill_prepared = dict(tokens=object(), positions=object(), length=128)
        calls = []
        self.gen.model.prefill = lambda tokens, **kw: calls.append(kw) or FakeLogits()
        self.methods["_prefill_trace_logits"](self.gen)
        self.assertIs(calls[0]["cache"], resident)

    def test_single_slot_reset_zeros_resident_state_in_place(self):
        state = SimpleNamespace(conv=torch.ones(1, 3, 4), recurrent=torch.ones(1, 2, 4, 4))
        resident = SimpleNamespace(layers=[state], batch_size=1)
        self.gen.model._resident_decode_bucket = (self.gen.cache, (0,), resident)
        self.gen.model._resident_decode_valid = True
        self.ops.zeros_like = torch.zeros_like
        self.ops.copy = lambda source, target: target.copy_(source)
        conv, recurrent = state.conv, state.recurrent
        self.methods["reset_recurrent_slots"](self.gen, [0])
        self.assertIs(state.conv, conv)
        self.assertIs(state.recurrent, recurrent)
        self.assertEqual(conv.count_nonzero().item(), 0)
        self.assertEqual(recurrent.count_nonzero().item(), 0)
        self.assertTrue(self.gen.model._resident_decode_valid)

    def test_standalone_keeps_three_trace_contract(self):
        self.capture(serving=False)
        self.assertEqual(
            (self.gen.trace, self.gen.sample_trace, self.gen.prefill_trace, self.gen.prefill_sample_trace),
            (1, 2, 3, None),
        )
        self.assertNotIn(("sample", self.owned), self.events)

    def test_failed_fourth_capture_closes_and_releases_every_partial_handle(self):
        with self.assertRaisesRegex(RuntimeError, "fourth-capture failure"):
            self.capture(serving=True, fail_sampling=True)
        self.assertEqual(self.events[-5:], [("end", 4)] + [("release", i) for i in range(1, 5)])
        for name in ("trace", "sample_trace", "prefill_trace", "prefill_sample_trace"):
            self.assertIsNone(getattr(self.gen, name))

    def test_fallback_captures_sampler_as_third_trace_without_prefill_trace(self):
        self.capture(serving=False, fallback=True)
        self.assertEqual(
            (self.gen.trace, self.gen.sample_trace, self.gen.prefill_trace, self.gen.prefill_sample_trace),
            (1, 2, None, 3),
        )
        third = self.events.index(("begin", 3))
        self.assertEqual(self.events[third : third + 3], [("begin", 3), ("sample", self.owned), ("end", 3)])
        self.assertEqual(self.gen.counters["prefill_trace_captures"], 0)
        self.assertEqual(self.gen.counters["prefill_sampling_trace_captures"], 1)

    def test_failed_fallback_capture_releases_all_three_handles(self):
        with self.assertRaisesRegex(RuntimeError, "fourth-capture failure"):
            self.capture(serving=False, fallback=True, fail_sampling=True)
        self.assertEqual(self.events[-4:], [("end", 3)] + [("release", i) for i in range(1, 4)])
        for name in ("trace", "sample_trace", "prefill_trace", "prefill_sample_trace"):
            self.assertIsNone(getattr(self.gen, name))

    def test_fallback_releases_before_first_pack_and_persistent_allocation(self):
        self.gen.trace, self.gen.sample_trace = "decode", "decode-sample"
        self.gen.prefill_sample_trace = "old-prefill-sample"
        self.ops.clone = lambda tensor: self.events.append(("clone",)) or FakeLogits()
        self.request(ends=(3, 4), starts=(0, 0), slots=(1, 2))
        self.assertEqual(
            [event[0] for event in self.events],
            ["public", "public", "release", "release", "release", "concat", "pad", "clone", "copy", "sample"],
        )
        self.assertEqual(self.gen.counters["prefill_sampling_input_allocations"], 1)
        staging = self.gen.prefill_sample_input
        self.events.clear()
        self.gen.prefill_sample_trace = "fallback-sample"
        self.request(ends=(3, 4), starts=(0, 0), slots=(1, 2))
        self.assertEqual([event[0] for event in self.events], ["public", "public", "concat", "pad", "copy", "execute"])
        self.assertIs(self.gen.prefill_sample_input, staging)
        self.assertEqual(self.gen.counters["prefill_sampling_input_allocations"], 1)
        self.assertEqual(self.gen.counters["prefill_sampling_eager_calls"], 1)
        self.assertEqual(self.gen.counters["prefill_sampling_replays"], 1)

    def test_unseen_fallback_count_invalidates_before_pack_but_retains_staging(self):
        self.request(ends=(3,), slots=(1,))
        staging = self.gen.prefill_sample_input
        self.gen.prefill_sample_trace = "old-prefill-sample"
        self.events.clear()
        self.request(ends=(3, 4), starts=(0, 0), slots=(1, 2))
        self.assertEqual(
            [event[0] for event in self.events], ["public", "public", "release", "concat", "pad", "copy", "sample"]
        )
        self.assertIs(self.gen.prefill_sample_input, staging)

    def test_failed_fallback_sampling_leaves_signature_unwarmed(self):
        self.gen._sampling_step = lambda output: (_ for _ in ()).throw(RuntimeError("Controlled sample failure"))
        with self.assertRaisesRegex(RuntimeError, "Controlled sample failure"):
            self.request(slots=(1,))
        self.assertEqual(self.gen._prefill_sampling_signatures, set())
        self.assertEqual(self.gen.counters["prefill_sampling_eager_calls"], 0)

    def test_fallback_temporaries_are_dead_before_trace_replay(self):
        transient_refs = []

        def transient():
            tensor = FakeLogits()
            transient_refs.append(weakref.ref(tensor))
            return tensor

        self.gen.prefill_forward = lambda *args, **kwargs: [transient()]
        self.ops.concat = lambda *args, **kwargs: transient()
        self.ops.pad = lambda *args, **kwargs: transient()
        self.ops.clone = lambda tensor: FakeLogits()
        self.ops.copy = lambda source, target: None
        self.gen._sampling_step = lambda output: None
        self.request(ends=(3, 4), starts=(0, 0), slots=(1, 2))
        transient_refs.clear()
        self.gen.prefill_sample_trace = "fallback-sample"
        replays = []

        def execute(mesh, trace, **kwargs):
            self.assertTrue(transient_refs)
            self.assertTrue(
                all(ref() is None for ref in transient_refs), "A new public/packed tensor survives into replay"
            )
            replays.append(trace)

        self.ops.execute_trace = execute
        self.request(ends=(3, 4), starts=(0, 0), slots=(1, 2))
        self.assertEqual(replays, ["fallback-sample"])

    def test_owned_and_external_cache_replacement_discard_staging_after_release(self):
        self.gen.model.context = 8192
        self.gen.model.layers = []
        self.gen.model.upload = lambda *args, **kwargs: self.events.append(("upload",)) or object()
        self.gen._reset_history = lambda: None
        self.gen.owns_cache = False
        replacement = SimpleNamespace(batch_size=4, capacity=8192, num_pages=1024, layers=[])
        self.gen.model.allocate_cache = lambda **kwargs: replacement
        for method in ("_ensure_cache", "bind_cache"):
            with self.subTest(method=method):
                self.gen.prefill_sample_input = object()
                self.gen.prefill_sample_trace = "old-sample"
                self.events.clear()
                if method == "_ensure_cache":
                    self.methods[method](self.gen, 4, 8192)
                else:
                    self.methods[method](self.gen, replacement, torch.zeros(4, 256, dtype=torch.int32))
                self.assertIsNone(self.gen.prefill_sample_input)
                self.assertEqual(self.events[0], ("release", "old-sample"))
                self.assertIs(self.gen.cache, replacement)


class ServingPrefillAdapterTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.cache, self.table = object(), object()
        self.device_sampling = True
        self.token_output = torch.arange(32).reshape(32, 1)

        def check(cache):
            self.assertIs(cache, self.cache)
            self.events.append(("cache",))

        def sample(params, **kwargs):
            self.events.append(("params", kwargs))
            return self.device_sampling

        def serving(tokens, **kwargs):
            self.events.append(("serving", tokens.clone(), kwargs))
            return self.token_output

        def public(tokens, **kwargs):
            self.events.append(("public", tokens.clone(), kwargs))
            return [torch.full((1, 1, 1, 5), int(tokens[0, -1]))]

        generator = SimpleNamespace(
            reset_recurrent_slots=lambda slots: self.events.append(("reset", slots)),
            serving_prefill_tokens=serving,
            prefill_forward=public,
            _host_logits=lambda output: output,
        )
        self.adapter = SimpleNamespace(
            generator=generator,
            _cache=check,
            _table=lambda table, slots: self.table,
            _sampling=sample,
            _decode_bound=True,
            read_decode_output=lambda output: output,
            process_decode_output_host=lambda output, **kwargs: output,
        )
        methods = load_methods("generator_vllm.py", "Qwen38ForCausalLM", ["prefill_forward"], SimpleNamespace())
        self.prefill = MethodType(methods["prefill_forward"], self.adapter)

    def request(self):
        return self.prefill(
            torch.arange(80).reshape(2, 40),
            page_table=object(),
            kv_cache=self.cache,
            prompt_lens=[3, 35],
            start_pos=[0, 32],
            sampling_params=object(),
            empty_slots=[3, 1],
        )

    def test_device_branch_passes_full_rows_and_absolute_ends_once(self):
        tokens, deltas = self.request()
        self.assertEqual([event[0] for event in self.events], ["cache", "reset", "params", "serving"])
        self.assertEqual(self.events[1][1], [3])
        self.assertEqual(self.events[2][1], {"reset": True, "output_positions": [3, 35]})
        self.assertTrue(torch.equal(self.events[3][1], torch.arange(80).reshape(2, 40)))
        kwargs = self.events[3][2]
        self.assertEqual(kwargs["prompt_lens"], [3, 35])
        self.assertEqual(kwargs["start_pos"], [0, 32])
        self.assertEqual(kwargs["slots"], [3, 1])
        self.assertIs(kwargs["page_table"], self.table)
        self.assertIs(kwargs["kv_cache"], self.cache)
        self.assertEqual(tokens.tolist(), [[0], [1]])
        self.assertEqual(deltas.tolist(), [0, 0])
        self.assertEqual(deltas.dtype, torch.int64)
        self.assertFalse(self.adapter._decode_bound)

    def test_host_branch_keeps_public_slicing_and_skips_token_helper(self):
        self.device_sampling = False
        logits, deltas = self.request()
        self.assertEqual([event[0] for event in self.events], ["cache", "reset", "params", "public", "public"])
        self.assertEqual(self.events[3][1].tolist(), [[0, 1, 2]])
        self.assertEqual(self.events[4][1].tolist(), [[72, 73, 74]])
        self.assertEqual(tuple(logits.shape), (2, 1, 5))
        self.assertEqual(logits[:, 0, 0].tolist(), [2, 74])
        self.assertEqual(deltas.tolist(), [0, 0])
        self.assertFalse(self.adapter._decode_bound)


if __name__ == "__main__":
    unittest.main(verbosity=2)
