# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check the real prefill sampling guard without importing TTNN or opening devices."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_sample_prefill(ops):
    source = Path(__file__).resolve().parents[2] / "tt/generator.py"
    tree = ast.parse(source.read_text())
    generator = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38Generator")
    method = next(
        node for node in generator.body if isinstance(node, ast.FunctionDef) and node.name == "sample_prefill"
    )
    namespace = {"ttnn": ops}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["sample_prefill"]


class PrefillSamplingTraceTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.fail_sampling = False
        self.gen = SimpleNamespace(cache=object(), tokens=object(), trace=object())

        def release(*, keep_prefill):
            self.assertTrue(keep_prefill)
            self.events.append("release")
            self.gen.trace = None

        def concat(logits, *, dim):
            self.assertEqual(dim, 2)
            self.events.append("concat")
            return object()

        def pad(logits, padding, *, value):
            self.assertEqual(value, 0.0)
            self.events.append("pad")
            return object()

        def sample(logits):
            self.events.append("sample")
            if self.fail_sampling:
                raise RuntimeError("Controlled sampling failure")

        self.gen._release_traces = release
        self.gen._sampling_step = sample
        self.sample = load_sample_prefill(SimpleNamespace(concat=concat, pad=pad))

    @staticmethod
    def logits(count, *, dtype="bfloat16"):
        return [SimpleNamespace(shape=(1, 1, 1, 62080), dtype=dtype, layout="TILE") for _ in range(count)]

    def test_unseen_signature_releases_before_pack_and_sampling(self):
        result = self.sample(self.gen, self.logits(2))
        self.assertIs(result, self.gen.tokens)
        self.assertEqual(self.events, ["release", "concat", "pad", "sample"])
        self.assertEqual(len(self.gen._prefill_sampling_signatures), 1)

    def test_repeated_signature_retains_a_new_decode_trace(self):
        self.sample(self.gen, self.logits(2))
        self.events.clear()
        live_trace = self.gen.trace = object()
        self.sample(self.gen, self.logits(2))
        self.assertIs(self.gen.trace, live_trace)
        self.assertEqual(self.events, ["concat", "pad", "sample"])
        self.assertEqual(len(self.gen._prefill_sampling_signatures), 1)

    def test_new_count_and_dtype_release_before_new_programs(self):
        self.sample(self.gen, self.logits(1))
        for logits in (self.logits(2), self.logits(2, dtype="float32")):
            with self.subTest(signature=logits[0].dtype, count=len(logits)):
                self.gen.trace = object()
                self.events.clear()
                self.sample(self.gen, logits)
                self.assertEqual(self.events, ["release", "concat", "pad", "sample"])
        self.assertEqual(len(self.gen._prefill_sampling_signatures), 3)

    def test_new_cache_discards_old_signature_history(self):
        self.sample(self.gen, self.logits(1))
        self.sample(self.gen, self.logits(2))
        self.gen.cache = object()
        self.gen.trace = object()
        self.events.clear()
        self.sample(self.gen, self.logits(2))
        self.assertEqual(self.events, ["release", "concat", "pad", "sample"])
        self.assertIs(self.gen._prefill_sampling_cache, self.gen.cache)
        self.assertEqual(len(self.gen._prefill_sampling_signatures), 1)

    def test_failed_sampling_does_not_mark_new_signature_warm(self):
        self.sample(self.gen, self.logits(1))
        warmed = self.gen._prefill_sampling_signatures.copy()
        self.fail_sampling = True
        self.events.clear()
        with self.assertRaisesRegex(RuntimeError, "Controlled sampling failure"):
            self.sample(self.gen, self.logits(2))
        self.assertEqual(self.gen._prefill_sampling_signatures, warmed)
        self.fail_sampling = False
        self.gen.trace = object()
        self.events.clear()
        self.sample(self.gen, self.logits(2))
        self.assertEqual(self.events, ["release", "concat", "pad", "sample"])
        self.assertEqual(len(self.gen._prefill_sampling_signatures), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
