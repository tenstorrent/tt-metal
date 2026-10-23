# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU mapping and policy checks; the device test is never imported."""

import unittest
from unittest.mock import patch

import torch

from models.demos.llama_3p1_8b_d_p.tests.model import reference
from models.demos.llama_3p1_8b_d_p.tests.utils import (
    FULL_LIMITS,
    assemble_head,
    check_logits_rows,
    check_metric,
    full_hidden,
    hidden_limits,
    join_hidden_tp_replicas,
    layer_input,
    local_limits,
    validate_observer_order,
    windows,
)


class LocalMappingTests(unittest.TestCase):
    # Independent integer token labels prove SP sequence assembly and all eight TP replicas.
    def test_hidden_coordinates_and_replica_corruption(self):
        shards = [(torch.arange(256) + (chip // 8) * 256).reshape(1, 1, 256, 1) for chip in range(32)]
        self.assertTrue(torch.equal(join_hidden_tp_replicas(shards)[:, 0], torch.arange(1024)))
        bad = [x.clone() for x in shards]
        bad[17][0, 0, 7, 0] += 1
        with self.assertRaisesRegex(AssertionError, "TP replica"):
            join_hidden_tp_replicas(bad)

    # The second chunk stays at positions 1024..2047; layer input comes from native predecessor.
    def test_actual_input_lineage_and_chunk_order(self):
        embedding = {0: torch.arange(1024)[:, None], 1024: torch.arange(1024, 2048)[:, None]}
        layers = {
            layer: {start: value + (layer + 1) * 10000 for start, value in embedding.items()} for layer in range(32)
        }
        for layer in range(32):
            expected = torch.arange(2048) + layer * 10000
            self.assertTrue(torch.equal(layer_input(embedding, layers, layer)[:, 0], expected))
        swapped = {0: embedding[1024], 1024: embedding[0]}
        self.assertFalse(torch.equal(full_hidden(swapped), full_hidden(embedding)))

    # Full K/V reconstruction preserves user-major plane, head, SP stripe, and both chunks.
    def test_all_cache_coordinates(self):
        shards = []
        for chip in range(32):
            sp, head = divmod(chip, 8)
            local = torch.arange(512)
            position = (local // 256) * 1024 + sp * 256 + local % 256
            shards.append((torch.arange(64)[:, None] * 100000 + head * 10000 + position)[..., None].unsqueeze(1))
        for slot in (0, 1):
            for layer in range(32):
                for head in range(8):
                    expected = (slot * 32 + layer) * 100000 + head * 10000 + torch.arange(2048)
                    self.assertTrue(torch.equal(assemble_head(shards, slot * 32 + layer, head)[:, 0], expected))
        self.assertFalse(torch.equal(assemble_head(shards, 31, 0), assemble_head(shards, 32, 0)))
        self.assertFalse(torch.equal(assemble_head(shards, 0, 0), assemble_head(shards, 0, 1)))

    # Omitted, duplicated, and reordered layer invocations cannot pass the structural layer check.
    def test_layer_invocation_mutations_are_rejected(self):
        validate_observer_order(list(range(32)))
        cases = [list(range(31)), [0] + list(range(31)), [1, 0] + list(range(2, 32))]
        for case in cases:
            with self.assertRaises(AssertionError):
                validate_observer_order(case)
        with self.assertRaises(KeyError):
            layer_input({}, {}, 1)

    # Local bounds remain the existing strict decoder bounds, with no depth-based widening.
    def test_dtype_gates_and_complete_windows(self):
        self.assertEqual(local_limits("bfloat16", "hidden"), (0.999, 0.025))
        self.assertEqual(local_limits("bfloat8_b", "hidden"), (0.999, 0.05))
        for kind in ("k", "v"):
            self.assertEqual(local_limits("bfloat16", kind), (0.9999, 0.01))
            self.assertEqual(local_limits("bfloat8_b", kind), (0.999, 0.02))
        self.assertEqual([p for _, _, begin, end in windows() for p in range(begin, end)], list(range(2048)))

    # Malformed geometry cannot masquerade as a shortened or reordered valid capture.
    def test_invalid_shapes_and_missing_chunks_are_rejected(self):
        for shards in ([], [torch.zeros(1, 1, 255, 1)] * 32):
            with self.assertRaises(ValueError):
                join_hidden_tp_replicas(shards)
        for chunks in ({0: torch.zeros(1024, 1)}, {0: torch.zeros(1024, 1), 1024: torch.zeros(1023, 1)}):
            with self.assertRaises(ValueError):
                full_hidden(chunks)

    # Nonfinite values or metric results must fail before they can corrupt report JSON.
    def test_nonfinite_statistics_are_hard(self):
        from models.demos.llama_3p1_8b_d_p.tests.utils import score_row

        finite = torch.arange(4, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            score_row(lambda e, a: (1.0, 0.0), finite, finite * float("nan"), (0.999, 0.025))
        with self.assertRaises(AssertionError):
            score_row(lambda e, a: (float("nan"), 0.0), finite, finite, (0.999, 0.025))


class ExplicitMetricPolicyTests(unittest.TestCase):
    # A scaled layer-zero output must fail the actual strict metric check used on device readbacks.
    def test_corrupted_layer0_hidden_is_hard(self):
        expected = torch.linspace(-1, 1, 4096)
        records = []
        with self.assertRaises(AssertionError):
            check_metric(expected, expected * 2, hidden_limits(1, True), "layer0", records)
        self.assertTrue(records[0]["enforced"])
        self.assertFalse(records[0]["within_limits"])
        self.assertEqual(records[0]["limits"], (0.999, 0.025))

    # Diagnostic drift is recorded with the same limits, without turning a diagnostic into a gate.
    def test_global_raw_hidden_miss_is_recorded(self):
        expected = torch.linspace(-1, 1, 4096)
        records = []
        check_metric(expected, expected * 2, FULL_LIMITS, "accumulated raw hidden", records, enforce=False)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["enforced"])
        self.assertFalse(records[0]["within_limits"])
        self.assertEqual(records[0]["limits"], FULL_LIMITS)
        self.assertAlmostEqual(records[0]["nl2"], 1.0)

    # Scaling preserves top-1/top-5, but must fail the actual final-logit metric gate before ranking.
    def test_corrupted_final_logits_are_hard_even_with_same_top1(self):
        expected = torch.linspace(-1, 1, 32).reshape(1, -1)
        actual = expected * 2
        self.assertEqual(int(expected.argmax()), int(actual.argmax()))
        records = []
        with self.assertRaises(AssertionError) as caught:
            check_logits_rows(
                expected, actual, torch.tensor([0]), start=0, end=1024, limits=FULL_LIMITS, records=records
            )
        self.assertEqual(caught.exception.args[0][0], "logits range=[0,1024)")
        self.assertTrue(records[0]["enforced"])
        self.assertFalse(records[0]["within_limits"])


class TokenizerStub:
    def __init__(self, length):
        self.length = length
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return list(range(self.length))

    def decode(self, ids, **kwargs):
        return "decoded fixture prefix"


class PromptOverrideTests(unittest.TestCase):
    # Held-out text enters one user message and one template application, then truncates exactly once.
    def test_override_has_one_template_and_exact_prefix(self):
        tokenizer = TokenizerStub(2056)
        text = "Held-out prose, Python, JSON and numeric rows."
        with patch.object(reference.AutoTokenizer, "from_pretrained", return_value=tokenizer):
            ids, metadata = reference.chat_tokens("unused-local-path", slot=0, length=2048, user_text=text)
        self.assertEqual(ids.tolist(), list(range(2048)))
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(tokenizer.calls[0][0], [{"role": "user", "content": text}])
        self.assertEqual(tokenizer.calls[0][1], dict(tokenize=True, add_generation_prompt=True, return_dict=False))
        self.assertTrue(metadata["user_text_override"])
        self.assertTrue(metadata["prefix_of_rendered_chat"])
        self.assertEqual(metadata["token_ids"], ids.tolist())

    # A short held-out rendering fails; it must not silently repeat text or add more chat templates.
    def test_short_override_is_rejected(self):
        tokenizer = TokenizerStub(2047)
        with patch.object(reference.AutoTokenizer, "from_pretrained", return_value=tokenizer):
            with self.assertRaisesRegex(ValueError, "did not reach"):
                reference.chat_tokens("unused-local-path", slot=0, length=2048, user_text="fixed text")
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertEqual(tokenizer.calls[0][0], [{"role": "user", "content": "fixed text"}])

    # The unchanged second slot still uses the original tea fixture when no override is supplied.
    def test_default_slot_fixture_is_preserved(self):
        tokenizer = TokenizerStub(2056)
        with patch.object(reference.AutoTokenizer, "from_pretrained", return_value=tokenizer):
            _, metadata = reference.chat_tokens("unused-local-path", slot=1, length=2048)
        expected = "List the steps to make a cup of tea. Explain each step in simple words. " * 2048
        self.assertEqual(tokenizer.calls[0][0], [{"role": "user", "content": expected}])
        self.assertFalse(metadata["user_text_override"])
