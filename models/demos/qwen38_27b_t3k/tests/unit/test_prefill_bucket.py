# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rounding a prompt to its allocated pages so the stack is compiled per bucket, not per prompt.

Device programs are keyed on logical length, so an unrounded prompt compiles all 64 layers the
first time that exact token count is served: measured at 21 s for 129 tokens against 0.095 s for a
repeated 128. Rounding up to the page size collapses 32 lengths onto one program set.

The ceiling is the page size and not something larger because the scheduler allocates
ceil(prompt/32) blocks and pads the rest of the table with zero. A prefill reaching past its own
blocks would write physical page 0, which belongs to another request.
"""

import unittest
from collections import Counter
from types import MethodType, SimpleNamespace

import torch

from models.demos.qwen38_27b_t3k.tests.unit.test_serving_prefill_trace import FakeGenerator, load_methods
from models.demos.qwen38_27b_t3k.tt.generator import prefill_bucket

CAPACITY = 8192


class BucketArithmeticTests(unittest.TestCase):
    def test_a_prompt_is_rounded_up_to_whole_pages(self):
        for valid, expected in ((3, 32), (31, 32), (32, 32), (33, 64), (129, 160), (4096, 4096)):
            with self.subTest(valid=valid):
                self.assertEqual(prefill_bucket(valid, CAPACITY), expected)

    def test_a_prompt_shorter_than_one_convolution_window_is_left_exact(self):
        # The selectors address a three-row history; a shorter prompt has no window to point at.
        for valid in (1, 2):
            with self.subTest(valid=valid):
                self.assertEqual(prefill_bucket(valid, CAPACITY), valid)

    def test_every_length_in_one_page_shares_a_bucket(self):
        self.assertEqual({prefill_bucket(n, CAPACITY) for n in range(97, 129)}, {128})

    def test_a_prompt_whose_rounding_would_leave_the_cache_is_left_exact(self):
        # Rounding is only safe while it stays on pages this request owns.
        self.assertEqual(prefill_bucket(CAPACITY - 5, CAPACITY), CAPACITY)
        self.assertEqual(prefill_bucket(CAPACITY - 5, CAPACITY, start=32), CAPACITY - 5)
        self.assertEqual(prefill_bucket(100, CAPACITY, start=CAPACITY - 110), 100)
        self.assertEqual(prefill_bucket(100, CAPACITY, start=CAPACITY - 128), 128)


class OwnedPrefillFixture(unittest.TestCase):
    """The slot-zero serving path, which is what a single short request takes."""

    def setUp(self):
        self.calls = []
        self.ops = SimpleNamespace(
            uint32="uint32",
            ROW_MAJOR_LAYOUT="ROW_MAJOR",
            clone=lambda tensor: object(),
            copy=lambda source, target: None,
            pad=lambda tensor, *a, **kw: tensor,
            execute_trace=lambda mesh, trace, **kw: None,
            release_trace=lambda mesh, trace: None,
        )
        self.methods = load_methods(
            "generator.py",
            "Qwen38Generator",
            [
                "_prefill_for_generate",
                "_release_traces",
                "_prefill_trace_logits",
                "_prefill_trace_step",
                "_aim_prefill_selectors",
                "_bucket_selectors",
            ],
            self.ops,
        )

        def prefill(tokens, **kwargs):
            self.calls.append(dict(tokens=tokens, **kwargs))
            return object()

        self.gen = FakeGenerator(
            mesh=object(),
            cache=SimpleNamespace(batch_size=4, capacity=CAPACITY, num_pages=1024, layers=[]),
            model=SimpleNamespace(
                config=SimpleNamespace(vocab_size=250000),
                prefill=prefill,
                upload=lambda tensor, **kw: tensor,
                prefill_selectors=lambda *, length: SimpleNamespace(
                    length=length, valid=None, update=lambda valid: None
                ),
            ),
            page_table=SimpleNamespace(shape=(4, 256)),
            prefill_prepared=None,
            prefill_resident={},
            prefill_signatures=set(),
            trace=None,
            sample_trace=None,
            _prefill_sample_only_trace=None,
            trace_records_history=None,
            counters=Counter(),
        )
        self.gen._release_traces = MethodType(self.methods["_release_traces"], self.gen)
        self.gen._prefill_trace_logits = MethodType(self.methods["_prefill_trace_logits"], self.gen)
        self.gen._aim_prefill_selectors = MethodType(self.methods["_aim_prefill_selectors"], self.gen)
        self.gen._bucket_selectors = MethodType(self.methods["_bucket_selectors"], self.gen)
        self.gen._prefill_selectors = {}
        self.gen._prefill_trace_step = MethodType(self.methods["_prefill_trace_step"], self.gen)
        self.gen._copy = lambda source, target, name: self.gen.counters.update([name])
        self.owned = MethodType(self.methods["_prefill_for_generate"], self.gen)

    def run_prompt(self, valid):
        self.gen.prefill_prepared = None
        self.gen.prefill_resident.clear()
        self.calls.clear()
        self.owned(torch.arange(valid, dtype=torch.int64).reshape(1, valid))
        return self.calls[0]


class OwnedPrefillBucketTests(OwnedPrefillFixture):
    def test_a_short_prompt_is_padded_to_its_page_and_declares_what_is_real(self):
        call = self.run_prompt(100)
        self.assertEqual(call["length"], 128)
        self.assertEqual(call["valid"], 100)
        self.assertEqual(tuple(call["tokens"].shape), (1, 128))
        # Padding must be inert tokens, not a repeat that the recurrent layers would absorb.
        self.assertTrue(torch.equal(call["tokens"][0, 100:], torch.zeros(28, dtype=torch.int64)))
        self.assertTrue(torch.equal(call["tokens"][0, :100], torch.arange(100)))

    def test_an_aligned_prompt_is_passed_through_untouched(self):
        call = self.run_prompt(128)
        self.assertEqual((call["length"], call["valid"]), (128, 128))
        self.assertEqual(tuple(call["tokens"].shape), (1, 128))

    def test_prompts_a_token_apart_compile_against_one_shape(self):
        # The whole point: 97..128 tokens must reach the device as one program set.
        self.assertEqual({self.run_prompt(n)["length"] for n in (97, 100, 113, 128)}, {128})

    def test_the_positions_cover_the_bucket_so_padded_rows_carry_real_rope(self):
        self.run_prompt(100)
        self.assertEqual(tuple(self.gen.prefill_prepared["positions"].shape), (1, 128))

    def test_the_prepared_buffers_are_keyed_on_the_bucket_alone(self):
        # The prompt length lives in the selectors, so one trace serves every length in the bucket.
        self.run_prompt(100)
        key, prepared = self.gen.prefill_prepared["key"], self.gen.prefill_prepared
        self.assertEqual((prepared["length"], prepared["valid"]), (128, 100))
        self.assertNotIn(100, key)
        buffers = prepared["tokens"]
        self.owned(torch.arange(113, dtype=torch.int64).reshape(1, 113))
        self.assertEqual(self.gen.prefill_prepared["key"], key)
        self.assertEqual(self.gen.prefill_prepared["valid"], 113)
        # The second prompt refreshed the bucket's buffers rather than reallocating them, which is
        # what lets it reach the trace the first one captured.
        self.assertIs(self.gen.prefill_prepared["tokens"], buffers)
        self.assertEqual(self.gen.counters["prefill_token_refreshes"], 1)
        self.assertEqual(self.gen.counters["prefill_selector_refreshes"], 2)


class ResidentBucketTests(OwnedPrefillFixture):
    """Each bucket keeps its own buffers and traces, so changing bucket replays.

    One prepared slot meant alternating buckets paid an eager prefill and a decode recapture every
    time, about 2.1 s measured, which is every growing chat session.
    """

    def test_each_bucket_keeps_its_own_buffers(self):
        self.run_prompt(100)
        small = self.gen.prefill_prepared
        self.owned(torch.arange(300, dtype=torch.int64).reshape(1, 300))
        large = self.gen.prefill_prepared
        self.assertEqual([prepared["length"] for prepared in self.gen.prefill_resident.values()], [128, 320])
        self.assertIsNot(small["tokens"], large["tokens"])

    def test_returning_to_a_resident_bucket_reuses_its_buffers(self):
        self.run_prompt(100)
        small = self.gen.prefill_prepared
        self.owned(torch.arange(300, dtype=torch.int64).reshape(1, 300))
        self.owned(torch.arange(110, dtype=torch.int64).reshape(1, 110))
        self.assertIs(self.gen.prefill_prepared, small)
        self.assertEqual(self.gen.counters["prefill_bucket_allocations"], 2)

    def test_a_resident_trace_survives_a_detour_through_another_bucket(self):
        self.run_prompt(100)
        self.gen.prefill_prepared["trace"] = "small-trace"
        self.owned(torch.arange(300, dtype=torch.int64).reshape(1, 300))
        # Allocating the second bucket releases every trace, because its buffers must not land on
        # an address a live trace still reads.
        self.assertIsNone(self.gen.prefill_resident[self.gen.prefill_prepared["key"]]["trace"])
        self.assertEqual([prepared["trace"] for prepared in self.gen.prefill_resident.values()], [None, None])

    def test_the_coldest_bucket_is_evicted_once_the_set_is_full(self):
        from models.demos.qwen38_27b_t3k.tt.generator import MAX_RESIDENT_PREFILL

        lengths = [32 * (i + 1) for i in range(MAX_RESIDENT_PREFILL)]
        self.run_prompt(lengths[0])
        for length in lengths[1:]:
            self.owned(torch.arange(length, dtype=torch.int64).reshape(1, length))
        # Touching the oldest again makes it the newest, so the next allocation evicts the second.
        self.owned(torch.arange(lengths[0], dtype=torch.int64).reshape(1, lengths[0]))
        self.owned(torch.arange(4096, dtype=torch.int64).reshape(1, 4096))
        resident = [prepared["length"] for prepared in self.gen.prefill_resident.values()]
        self.assertEqual(len(resident), MAX_RESIDENT_PREFILL)
        self.assertIn(lengths[0], resident)
        self.assertNotIn(lengths[1], resident)
        self.assertEqual(self.gen.counters["prefill_bucket_evictions"], 1)


class ReadOnlyTraceViewTests(unittest.TestCase):
    """The trace views read the active bucket, so nothing may assign to them.

    Every unit test here drives a stand-in namespace rather than the generator, so an assignment
    that the property refuses would otherwise only surface on device.
    """

    def test_nothing_assigns_to_the_trace_views(self):
        import ast
        from pathlib import Path

        source = (Path(__file__).parents[2] / "tt" / "generator.py").read_text()
        writes = [
            node.lineno
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and node.attr in ("prefill_trace", "prefill_sample_trace")
        ]
        self.assertEqual(writes, [], f"generator.py assigns a read-only trace view at {writes}")


class ValidLengthRefusalTests(unittest.TestCase):
    """A bucket the recurrent layers cannot mask is refused before it reaches the device."""

    def setUp(self):
        self.prefill = load_methods("model.py", "Qwen38Model", ["prefill"], None)["prefill"]
        self.layer = load_methods("decoder.py", "Qwen38Decoder", ["prefill_forward"], None)["prefill_forward"]

    def model(self):
        return SimpleNamespace(config=SimpleNamespace(max_position_embeddings=262144))

    def test_a_prompt_longer_than_its_bucket_is_refused(self):
        with self.assertRaisesRegex(ValueError, "valid length"):
            self.prefill(self.model(), object(), cache=None, page_table=None, length=128, valid=129)

    def test_an_empty_prompt_is_refused(self):
        with self.assertRaisesRegex(ValueError, "valid length"):
            self.prefill(self.model(), object(), cache=None, page_table=None, length=128, valid=0)

    def test_a_bucket_that_is_not_whole_pages_is_refused(self):
        # Partial pages would put the mask and the paged write at different granularities.
        with self.assertRaisesRegex(ValueError, "whole number of pages"):
            self.prefill(self.model(), object(), cache=None, page_table=None, length=100, valid=90)

    def test_a_layer_refuses_a_valid_length_outside_its_input(self):
        layer = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=262144), kind="linear_attention")
        with self.assertRaisesRegex(ValueError, "valid length"):
            self.layer(layer, SimpleNamespace(shape=(1, 128, 5120)), state=None, valid=200)


if __name__ == "__main__":
    unittest.main()
