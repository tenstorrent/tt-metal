# SPDX-License-Identifier: Apache-2.0
"""CPU-only request identity regressions; no model inference or device imports."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_stage.subsets import digest
from lm_eval.api.instance import Instance
from lm_eval.models.api_models import JsonChatStr, TemplateAPI
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.models.utils import Collator

SOURCE = Path(__file__).resolve().parents[1] / "tools/benchmark_client_hooks/request_identity.py"
SPEC = importlib.util.spec_from_file_location("duplicate_identity_host_test", SOURCE)
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


class DuplicateIdentityTests(unittest.TestCase):
    def request(self, doc_id, context=None, generation=None, task="triviaqa", repeats=1):
        context = context or JsonChatStr(json.dumps([{"role": "user", "content": "Same question?"}]))
        generation = (
            generation
            if generation is not None
            else {
                "max_gen_toks": 8192,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "do_sample": True,
                "until": [],
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": True,
                    "reasoning_effort": "xhigh",
                },
            }
        )
        return Instance(
            "generate_until",
            {"question": "Same question?", "answer": {"aliases": ["synthetic"]}},
            (context, generation),
            0,
            metadata=(task, doc_id, repeats),
        )

    def payload(self, request):
        model = SimpleNamespace(_batch_size=1, _max_gen_toks=8192, model="synthetic-no-inference")
        messages = TemplateAPI.create_message(model, [request.args[0]], generate=True)
        return LocalChatCompletion._create_payload(
            model, messages, generate=True, gen_kwargs=copy.deepcopy(request.args[1]), seed=None
        )

    def assert_rejected(self, requests):
        try:
            IDENTITY.disambiguate_requests(requests)
        except (ValueError, TypeError):
            return
        self.fail("Unsupported identity inputs were accepted")

    def test_duplicates_preserve_real_wire_payload_and_scored_argument_hashes(self):
        requests = [self.request(i) for i in range(4)]
        original = copy.deepcopy(requests)
        objects = [id(r) for r in requests]
        kwargs_objects = [id(r.args[1]) for r in requests]
        expected = [self.payload(r) for r in requests]
        records = IDENTITY.disambiguate_requests(requests)
        self.assertTrue(records)
        self.assertEqual([id(r) for r in requests], objects)
        self.assertEqual([id(r.args[1]) for r in requests], kwargs_objects)
        self.assertEqual([self.payload(r) for r in requests], expected)
        hashes = [digest(r.args) for r in requests]
        self.assertEqual(len(set(hashes)), 4)
        for before, after, key in zip(original, requests, hashes):
            self.assertEqual(after.doc, before.doc)
            self.assertEqual(after.metadata, before.metadata)
            self.assertEqual(after.repeats, 1)
            self.assertEqual(after.args[1], before.args[1])
            self.assertEqual(json.loads(after.args[0].prompt), json.loads(before.args[0].prompt))
            # Evaluator logs these same Instance.args; JSON turns tuple wrappers into arrays.
            scored = json.loads(json.dumps({"arguments": [after.args]}))
            self.assertEqual(digest(scored["arguments"][0]), key)
            raw = after.args[0].prompt
            parsed, end = json.JSONDecoder().raw_decode(raw.lstrip())
            self.assertEqual(parsed, json.loads(before.args[0].prompt))
            self.assertTrue(set(raw.lstrip()[end:]) <= set(" \t\r\n"))

    def test_unique_contexts_and_distinct_generation_policies_are_unchanged(self):
        requests = [
            self.request(0),
            self.request(1, JsonChatStr('[{"role":"user","content":"Other?"}]')),
            self.request(2, generation={"temperature": 0.0, "max_gen_toks": 8192}),
        ]
        args = [r.arguments for r in requests]
        IDENTITY.disambiguate_requests(requests)
        for r, original in zip(requests, args):
            self.assertIs(r.arguments, original)

    def test_collision_with_reserved_original_identity_and_new_identities(self):
        trial = [self.request(0), self.request(1)]
        IDENTITY.disambiguate_requests(trial)
        collision_context = trial[1].args[0]
        self.assertNotEqual(collision_context, trial[0].args[0])
        requests = [self.request(0), self.request(1), self.request(2, collision_context), self.request(3)]
        reserved = requests[2].arguments
        expected = [self.payload(r) for r in requests]
        IDENTITY.disambiguate_requests(requests)
        self.assertIs(requests[2].arguments, reserved)
        self.assertEqual(len({digest(r.args) for r in requests}), 4)
        self.assertEqual([self.payload(r) for r in requests], expected)

    def test_collator_does_not_merge_duplicate_semantic_prompts(self):
        requests = [self.request(i) for i in range(4)]
        IDENTITY.disambiguate_requests(requests)
        items = [(r.args[0], r.args[1], None) for r in requests]
        collator = Collator(items, sort_fn=lambda x: -len(x[0]), group_by="gen_kwargs")
        reordered = [item for batch in collator.get_batched(n=0) for item in batch]
        self.assertEqual(len(reordered), 4)
        self.assertEqual(
            collator.get_original([digest((x[0], x[1])) for x in reordered]), [digest(r.args) for r in requests]
        )

    def test_invalid_occurrences_and_repeats_fail_closed(self):
        same = self.request(0)
        for requests in (
            [same, same],
            [self.request(0), self.request(0)],
            [self.request(0, repeats=2)],
            [self.request(0, repeats=0)],
        ):
            with self.subTest(requests=requests):
                self.assert_rejected(requests)

    def test_plain_text_and_multimodal_argument_shapes_fail_closed(self):
        plain = self.request(0)
        plain.arguments = ("not a JsonChatStr", plain.args[1])
        multi = self.request(1)
        multi.arguments = (*multi.args, {"visual": []})
        self.assert_rejected([plain])
        self.assert_rejected([multi])

    def test_evaluator_wrapper_restores_method_after_failure_and_guards_tokenization(self):
        from lm_eval import evaluator

        requests = [self.request(0), self.request(1)]

        def generate(model, received):
            self.assertEqual(len({digest(r.args) for r in received}), 2)
            raise RuntimeError("synthetic evaluator failure; no inference")

        recorded_class = type(
            "RecordedChat",
            (),
            {
                "__module__": "benchmark_stage.evaluate",
                "generate_until": generate,
                "tokenized_requests": False,
                "_batch_size": 1,
                "_concurrent": 32,
            },
        )
        model = recorded_class()

        def evaluate(*, model):
            return model.generate_until(requests)

        with tempfile.TemporaryDirectory() as directory, patch.object(evaluator, "simple_evaluate", evaluate):
            IDENTITY.install_evaluator_hook(Path(directory))
            try:
                evaluator.simple_evaluate(model=model)
            except RuntimeError as error:
                self.assertIn("synthetic evaluator failure", str(error))
            else:
                self.fail("Synthetic failure was suppressed")
            self.assertNotIn("generate_until", model.__dict__)
            self.assertIs(model.generate_until.__func__, generate)
            records = [
                json.loads(line) for line in (Path(directory) / "request_identity.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum(row["serialization_changed"] for row in records), 1)
            model.tokenized_requests = True
            try:
                evaluator.simple_evaluate(model=model)
            except ValueError:
                pass
            else:
                self.fail("Tokenized backend was accepted")
            self.assertNotIn("generate_until", model.__dict__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
