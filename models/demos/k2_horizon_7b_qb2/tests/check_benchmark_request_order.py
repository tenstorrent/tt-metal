"""Host-only scheduling checks: upstream transport runs against an in-memory session."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from lm_eval.models import api_models
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion

from models.demos.k2_horizon_7b_qb2.tests import benchmark_request_order as order

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "doc/benchmark/attempts/20260929-task-order-incomplete/run"


class SchedulingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.manifest, cls.entries = order.preserved_requests(
            EVIDENCE / "run_config.json", EVIDENCE / "manifest.json", EVIDENCE / "benchmark-inputs.jsonl"
        )
        cls.plan = order.make_plan(cls.config, cls.manifest, cls.entries, EVIDENCE)
        cls.inputs = order.read_jsonl(EVIDENCE / "benchmark-inputs.jsonl")

    def backend(self):
        backend = LocalChatCompletion(
            model=self.config["model"],
            base_url="http://not-used.invalid/v1/chat/completions",
            tokenizer_backend=None,
            tokenized_requests=False,
            num_concurrent=32,
            max_retries=0,
            max_gen_toks=2048,
        )
        backend.request_metadata = {row["request_sha256"]: row for row in self.entries}
        backend.cache_hook = SimpleNamespace(add_partial=lambda *args: None)
        requests = [JsonChatStr(*row["arguments"][0]) for row in self.inputs]
        keys = [(request, row["arguments"][1]) for request, row in zip(requests, self.inputs, strict=True)]
        return backend, requests, keys

    def test_manifest_full_mapping_and_no_score_fields(self):
        self.assertEqual(len(self.entries), 512)
        self.assertEqual(
            [row["doc_id"] for row in self.plan["entries"][:12]],
            [52, 243, 355, 393, 407, 412, 425, 481, 482, 491, 529, 534],
        )
        self.assertEqual(sum(row["unfinished"] for row in self.plan["entries"]), 12)
        self.assertEqual(self.plan["entries"][12]["observed_output_tokens"], 13884)
        self.assertTrue(
            all(not any("score" in field or "answer" in field for field in row) for row in self.plan["entries"])
        )

    def test_physical_json_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text(json.dumps({"content": "a\u2028b\u2029c"}, ensure_ascii=False) + "\n")
            self.assertEqual(order.read_jsonl(path), [{"content": "a\u2028b\u2029c"}])

    def test_permutation_tuple_identity_and_inverse(self):
        backend, requests, keys = self.backend()
        context_lengths = list(range(512))
        seen = []

        async def original(self, submitted, cache_keys, *, ctxlens, **kwargs):
            for request, key, length in zip(submitted, cache_keys, ctxlens, strict=True):
                self_index = requests.index(request)
                seen.append(order.digest(key))
                assert request is requests[self_index] and key is keys[self_index] and length == self_index
            assert kwargs == {"generate": True, "opaque": "unchanged"}
            return [[order.digest(key)] for key in cache_keys]

        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "audit.json"
            result = asyncio.run(
                order.ordering_wrapper(original, self.plan, audit)(
                    backend, requests, keys, ctxlens=context_lengths, opaque="unchanged"
                )
            )
            self.assertEqual(result, [[order.digest(key)] for key in keys])
            self.assertEqual(seen, [row["request_sha256"] for row in self.plan["entries"]])
            proof = json.loads(audit.read_text())
            self.assertTrue(proof["results_restored_to_upstream_order"])
            self.assertTrue(
                all(
                    proof["original_to_scheduled"][source] == position
                    for position, source in enumerate(proof["scheduled_to_original"])
                )
            )

    def test_duplicates_missing_and_payload_change_fail_before_transport(self):
        for kind in ("duplicate", "missing", "payload"):
            with self.subTest(kind=kind):
                backend, requests, keys = self.backend()
                if kind == "duplicate":
                    keys[-1] = keys[0]
                elif kind == "missing":
                    requests.pop()
                    keys.pop()
                else:
                    backend._create_payload = lambda *args, **kwargs: {"wrong": True}
                original = AsyncMock()
                with self.assertRaises(ValueError):
                    asyncio.run(order.ordering_wrapper(original, self.plan)(backend, requests, keys))
                original.assert_not_awaited()

    def test_full_upstream_pool_cache_and_final_proof_without_network(self):
        backend, requests, keys = self.backend()
        payload_to_key = {row["wire_payload_sha256"]: row["request_sha256"] for row in self.entries}
        cache = []
        backend.cache_hook = SimpleNamespace(
            add_partial=lambda method, key, answer: cache.append((method, order.digest(key), answer))
        )
        posted = []
        owner = self

        class Response:
            ok = True

            def __init__(self, payload):
                self.key = payload_to_key[order.digest(payload)]

            async def __aenter__(self):
                await asyncio.sleep(0)
                return self

            async def __aexit__(self, *args):
                return False

            def raise_for_status(self):
                pass

            async def json(self):
                return {
                    "id": "host-test-" + self.key,
                    "created": 12345,
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": self.key}}],
                }

        class Session:
            def __init__(self, **kwargs):
                owner.assertEqual(kwargs["connector"].limit, 32)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def post(self, url, *, json, headers):
                posted.append(payload_to_key[order.digest(json)])
                return Response(json)

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "manifest.json").write_text(json.dumps(self.manifest))
            for name in ("run_config.json", "benchmark-inputs.jsonl"):
                shutil.copyfile(EVIDENCE / name, run / name)
            plan_path = run / "frozen-plan.json"
            plan_path.write_text(json.dumps(self.plan))
            originals = {
                name: getattr(LocalChatCompletion, name)
                for name in ("amodel_call", "parse_generations", "get_batched_requests")
            }
            original_parse = backend.parse_generations

            def recorded_parse(outputs, **kwargs):
                key = outputs["choices"][0]["message"]["content"]
                metadata = backend.request_metadata[key]
                group = run / metadata["group"]
                group.mkdir(exist_ok=True)
                with (group / "responses.jsonl").open("a") as target:
                    target.write(json.dumps(outputs) + "\n")
                with (group / "request_links.jsonl").open("a") as target:
                    target.write(json.dumps(dict(metadata, response_id=outputs["id"])) + "\n")
                return LocalChatCompletion.parse_generations(backend, outputs, **kwargs)

            backend.parse_generations = recorded_parse
            try:
                order.install(
                    plan_path,
                    run / "request-order-observed.json",
                    activation={
                        "orig_argv": [sys.executable, "-m", "benchmark_stage", "evaluate"],
                        "evaluate_child": True,
                    },
                )
                with (
                    patch.object(api_models, "ClientSession", Session),
                    patch.object(api_models, "TCPConnector", lambda **kwargs: SimpleNamespace(**kwargs)),
                ):
                    result = backend.generate_until([SimpleNamespace(args=key) for key in keys])
                self.assertEqual(result, [order.digest(key) for key in keys])
                self.assertEqual(posted, [row["request_sha256"] for row in self.plan["entries"]])
                self.assertEqual(len(cache), 512)
                self.assertTrue(all(method == "generate_until" and key == answer for method, key, answer in cache))
                for name, method in originals.items():
                    setattr(LocalChatCompletion, name, method)
                for row in self.inputs:
                    key = row["request_sha256"]
                    group = backend.request_metadata[key]["group"]
                    filters = ["strict-match", "flexible-extract"] if group == "gsm8k_cot" else ["none"]
                    for filter_name in filters:
                        sample = dict(
                            doc_id=row["doc_id"],
                            doc=row["doc"],
                            arguments=[row["arguments"]],
                            resps=[[key]],
                            filter=filter_name,
                        )
                        with (run / group / f"samples_{row['task']}.jsonl").open("a") as target:
                            target.write(json.dumps(sample) + "\n")
                proof = order.verify_run(run, plan_path)
                self.assertTrue(proof["valid"])
                self.assertEqual(proof["count"], 512)
                self.assertTrue((run / "request-order-plan.json").is_file())
                sample_path = run / "gsm8k_cot/samples_gsm8k_cot.jsonl"
                original_samples = sample_path.read_text()
                samples = order.read_jsonl(sample_path)
                samples[0]["resps"] = [["wrong-document-response"]]
                sample_path.write_text("".join(json.dumps(row) + "\n" for row in samples))
                with self.assertRaisesRegex(ValueError, "different request"):
                    order.verify_run(run, plan_path)
                sample_path.write_text(original_samples)
                journal = run / "request-order-api-links.jsonl"
                rows = order.read_jsonl(journal)
                journal.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
                with self.assertRaises(ValueError):
                    order.verify_run(run, plan_path)
            finally:
                backend.parse_generations = original_parse
                for name, method in originals.items():
                    setattr(LocalChatCompletion, name, method)

    def test_sitecustomize_scope_and_fail_closed(self):
        environment = dict(os.environ, K2_BENCHMARK_REQUEST_ORDER="/definitely-missing-k2-order-plan.json")
        environment["PYTHONPATH"] = (
            str(ROOT / "tests/benchmark_order_hook") + os.pathsep + environment.get("PYTHONPATH", "")
        )
        outside = subprocess.run(
            [sys.executable, "-c", "print('outside-evaluate-ok')"], env=environment, capture_output=True, text=True
        )
        self.assertEqual(outside.returncode, 0, outside.stderr)
        self.assertIn("outside-evaluate-ok", outside.stdout)
        inside = subprocess.run(
            [sys.executable, "-m", "benchmark_stage", "evaluate", "--output", "/tmp/never-used-k2-order"],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(inside.returncode, 0)
        self.assertIn("K2 request-order hook failed", inside.stderr)
        self.assertIn("Failed to import the site module", inside.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
