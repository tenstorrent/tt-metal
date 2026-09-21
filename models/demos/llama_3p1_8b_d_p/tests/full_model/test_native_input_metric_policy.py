# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for explicit numerical enforcement and single-template prompt overrides."""

import ast
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.demos.llama_3p1_8b_d_p.tests.full_model import reference


class Logger:
    def info(self, message):
        pass


def canonical_checks():
    """Load only the actual host check functions; importing the device test would import TTNN."""
    path = Path(__file__).with_name("test_prefill_model_vs_ref.py")
    wanted = {"_check_metric", "_hidden_limits", "_positions", "_check_hidden", "_check_logits"}
    definitions = [
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    fake_ttnn = types.SimpleNamespace(
        bfloat16="bf16",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
        get_device_tensors=lambda tensor: tensor.shards,
        to_torch=lambda tensor: tensor,
    )
    namespace = dict(torch=torch, metrics=reference.metrics, logger=Logger(), ttnn=fake_ttnn, FULL_LIMITS=(0.99, 0.15))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ExplicitMetricPolicyTests(unittest.TestCase):
    # First-layer output corruption must still raise with the existing strict BF16 decoder limit.
    def test_corrupted_layer0_hidden_is_hard(self):
        checks = canonical_checks()
        expected = torch.linspace(-1, 1, 4096).expand(2048, 4096)
        shard = (expected[:256] * 2).reshape(1, 1, 256, 4096)
        hidden = types.SimpleNamespace(
            shape=(1, 1, 256, 4096), dtype="bf16", layout="tile", memory_config=lambda: "dram", shards=[shard]
        )
        records = []
        with self.assertRaises(AssertionError):
            checks["_check_hidden"](
                hidden,
                expected,
                start=0,
                end=256,
                limits=checks["_hidden_limits"](1, "bf16"),
                label="layer0",
                records=records,
            )
        self.assertTrue(records[0]["enforced"])
        self.assertFalse(records[0]["within_limits"])
        self.assertEqual(records[0]["limits"], (0.999, 0.025))

    # Accumulated global drift remains visible with the original limits, even when it is not enforced.
    def test_global_raw_hidden_miss_is_recorded(self):
        checks = canonical_checks()
        expected = torch.linspace(-1, 1, 4096).expand(2048, 4096)
        shard = (expected[:256] * 2).reshape(1, 1, 256, 4096)
        hidden = types.SimpleNamespace(
            shape=(1, 1, 256, 4096), dtype="bf16", layout="tile", memory_config=lambda: "dram", shards=[shard]
        )
        records = []
        checks["_check_hidden"](
            hidden,
            expected,
            start=0,
            end=256,
            limits=(0.99, 0.15),
            label="accumulated raw hidden",
            records=records,
            enforce=False,
        )
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["enforced"])
        self.assertFalse(records[0]["within_limits"])
        self.assertEqual(records[0]["limits"], (0.99, 0.15))
        self.assertAlmostEqual(records[0]["nl2"], 1.0)

    # Scaling logits preserves every token rank; the final-logit NL2 gate must still reject corruption.
    def test_corrupted_final_logits_are_hard_even_with_same_top1(self):
        checks = canonical_checks()
        expected = torch.linspace(-1, 1, 128256).reshape(1, -1)
        actual = expected * 2
        self.assertEqual(int(expected.argmax()), int(actual.argmax()))
        shards = [part.reshape(1, 1, 1, 16032).expand(1, 1, 256, 16032) for part in actual.split(16032, dim=-1)]
        logits = types.SimpleNamespace(shape=(1, 1, 256, 16032), shards=shards)
        records = []
        with self.assertRaises(AssertionError) as caught:
            checks["_check_logits"](
                logits,
                dict(logit_positions=torch.tensor([0]), logits=expected),
                start=0,
                end=1024,
                num_layers=32,
                dtype="bf16",
                records=records,
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
