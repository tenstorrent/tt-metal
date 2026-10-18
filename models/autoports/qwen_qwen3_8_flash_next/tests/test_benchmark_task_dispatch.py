# SPDX-License-Identifier: Apache-2.0
"""Host-only task dispatch and inverse scorer-order regression checks."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "tools/benchmark_client_hooks/task_dispatch.py"
SPEC = importlib.util.spec_from_file_location("task_dispatch_host", SOURCE)
DISPATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPATCH)


class TaskDispatchTests(unittest.TestCase):
    def frozen_fixture(self, directory):
        requests, jobs = [], []
        for i in range(384):
            task, local = ("math", i) if i < 256 else ("trivia", i - 256)
            # Duplicate content remains distinct by local occurrence identity.
            doc = {"question": str(local % 72)}
            messages = [{"role": "user", "content": doc["question"]}]
            generation = {"max_gen_toks": 8192, "temperature": 1}
            requests.append(
                SimpleNamespace(
                    task_name=task,
                    doc_id=local,
                    idx=0,
                    repeats=1,
                    doc=doc,
                    args=(SimpleNamespace(prompt=json.dumps(messages)), generation),
                )
            )
            jobs.append(
                dict(
                    task=task,
                    local_doc_id=local,
                    idx=0,
                    original_index=i,
                    document_sha256=DISPATCH.digest(doc),
                    structured_messages_sha256=DISPATCH.digest(messages),
                    generation_sha256=DISPATCH.digest(generation),
                    estimated_completion_tokens=8192 if i >= 380 else (i * 37) % 1000 + 1,
                )
            )
        path = Path(directory) / "estimates.json"
        data = json.dumps(dict(algorithm=DISPATCH.ALGORITHM, jobs=jobs)).encode()
        path.write_bytes(data)
        return requests, jobs, path, hashlib.sha256(data).hexdigest()

    def test_frozen_full384_inverse_mapping_and_exact_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            requests, jobs, path, sha = self.frozen_fixture(directory)
            expected = sorted(range(384), key=lambda i: (-jobs[i]["estimated_completion_tokens"], i))
            ids = {id(r): i for i, r in enumerate(requests)}
            seen = []

            def generate(ordered):
                seen.extend(ids[id(r)] for r in ordered)
                return [ids[id(r)] for r in ordered]

            with patch.object(DISPATCH, "ESTIMATES", path), patch.object(DISPATCH, "ESTIMATES_SHA256", sha):
                self.assertEqual(DISPATCH.dispatch(requests, generate, directory), list(range(384)))
            self.assertEqual(seen, expected)
            self.assertEqual((Path(directory) / "runtime_estimates.json").read_bytes(), path.read_bytes())
            records = [json.loads(s) for s in (Path(directory) / "request_dispatch.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["dispatch_to_original"], expected)
            self.assertEqual(records[0]["runtime_estimates_sha256"], sha)
            self.assertTrue(records[1]["original_order_restored"])

    def test_frozen_dispatch_rejects_changed_inputs_and_missing_occurrences(self):
        for change in ("drop", "duplicate", "document", "messages", "generation", "repeats", "checksum"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                requests, _, path, sha = self.frozen_fixture(directory)
                if change == "drop":
                    requests.pop()
                elif change == "duplicate":
                    requests[-1] = requests[0]
                elif change == "document":
                    requests[0].doc = {"question": "changed"}
                elif change == "messages":
                    requests[0].args[0].prompt = "[]"
                elif change == "generation":
                    requests[0].args[1]["temperature"] = 0
                elif change == "repeats":
                    requests[0].repeats = 2
                elif change == "checksum":
                    path.write_bytes(path.read_bytes() + b" ")
                with patch.object(DISPATCH, "ESTIMATES", path), patch.object(DISPATCH, "ESTIMATES_SHA256", sha):
                    with self.assertRaises(ValueError):
                        DISPATCH.runtime_permutation(requests)

    def requests(self):
        return [
            SimpleNamespace(task_name=task, doc_id=index, idx=0, args=object())
            for index, task in enumerate(["math"] * 4 + ["trivia"] * 2 + ["third"])
        ]

    def test_round_robin_uses_only_task_names_and_preserves_task_order(self):
        requests = self.requests()
        self.assertEqual(DISPATCH.task_permutation(requests), [0, 4, 6, 1, 5, 2, 3])
        for request in requests:
            request.doc_id = object()
            request.idx = object()
            request.args = object()
        self.assertEqual(DISPATCH.task_permutation(requests), [0, 4, 6, 1, 5, 2, 3])
        self.assertEqual(DISPATCH.task_permutation(requests[:4]), [0, 1, 2, 3])

    def test_inverse_order_preserves_instances_arguments_and_scorer_pairing(self):
        requests = self.requests()
        original_ids = [id(r) for r in requests]
        args = [r.args for r in requests]
        seen = []

        def generate(ordered):
            seen.extend(id(r) for r in ordered)
            return [f"answer-for-{r.doc_id}" for r in ordered]

        with tempfile.TemporaryDirectory() as directory:
            outputs = DISPATCH.dispatch(requests, generate, directory, algorithm=DISPATCH.ROUND_ROBIN)
            records = [
                json.loads(line) for line in (Path(directory) / "request_dispatch.jsonl").read_text().splitlines()
            ]
        self.assertEqual(seen, [original_ids[i] for i in [0, 4, 6, 1, 5, 2, 3]])
        self.assertEqual(outputs, [f"answer-for-{r.doc_id}" for r in requests])
        self.assertEqual([id(r) for r in requests], original_ids)
        self.assertEqual([r.args for r in requests], args)
        self.assertEqual(records[0]["task_counts"], dict(math=4, trivia=2, third=1))
        self.assertEqual(records[1]["event"], "restored")

    def test_failure_preserves_dispatch_journal_without_false_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                DISPATCH.dispatch(self.requests(), lambda requests: [], directory, algorithm=DISPATCH.ROUND_ROBIN)
            records = [
                json.loads(line) for line in (Path(directory) / "request_dispatch.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["event"] for r in records], ["dispatch"])
        with self.assertRaises(ValueError):
            DISPATCH.task_permutation([SimpleNamespace(task_name=None)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
