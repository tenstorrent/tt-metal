# SPDX-License-Identifier: Apache-2.0
"""CPU-only cumulative scheduling provenance and fail-closed input checks."""

import copy
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tools/benchmark_freeze_runtime_estimates.py"
SPEC = importlib.util.spec_from_file_location("runtime_freezer_host", SOURCE)
FREEZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZE)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class FreezeTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        tasks = {
            task: dict(indices=list(range(n)), document_sha256=[f"doc-{task}-{i}" for i in range(n)])
            for task, n in FREEZE.COUNTS.items()
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(dict(tasks=tasks)))
        identities, inputs = [], []
        for task, spec in tasks.items():
            for i in spec["indices"]:
                args = [
                    [json.dumps([dict(role="user", content=f"{task}-{i}")])],
                    dict(max_gen_toks=8192, temperature=1),
                ]
                identities.append(
                    dict(
                        task=task,
                        doc_id=i,
                        idx=0,
                        original_request_sha256=FREEZE.digest(args),
                        effective_request_sha256=FREEZE.digest(args),
                        structured_messages_sha256=FREEZE.digest(json.loads(args[0][0])),
                        generation_sha256=FREEZE.digest(args[1]),
                    )
                )
                inputs.append(dict(task=task, doc_id=i, repeats=1, args=args))
        attempts = []
        for attempt_idx in range(2):
            a = root / f"attempt{attempt_idx}"
            a.mkdir()
            (a / "manifest.json").write_bytes(manifest.read_bytes())
            write_rows(a / "request_identity.jsonl", identities)
            write_rows(a / "upstream_inputs_before_identity.jsonl", inputs)
            write_rows(a / "upstream_inputs.jsonl", inputs)
            for task, spec in tasks.items():
                links, responses = [], []
                for i in spec["indices"]:
                    if attempt_idx == 0 and task == "triviaqa" and i == 127:
                        continue
                    ident = next(r for r in identities if r["task"] == task and r["doc_id"] == i)
                    rid = f"{task}-{attempt_idx}-{i}"
                    links.append(
                        dict(
                            task=task,
                            group=task,
                            doc_id=i,
                            response_id=rid,
                            request_sha256=ident["effective_request_sha256"],
                        )
                    )
                    responses.append(
                        dict(
                            id=rid,
                            usage=dict(completion_tokens=100 + attempt_idx * (i + 1)),
                            choices=[dict(message=dict(content="IGNORED"))],
                            score="IGNORED",
                        )
                    )
                write_rows(a / task / "request_links.jsonl", links)
                write_rows(a / task / "responses.jsonl", responses)
            attempts.append(a)
        return attempts, manifest

    def mutate(self, path, fn):
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        fn(rows)
        write_rows(path, rows)

    def test_maximum_all384_missing_cap_and_deterministic_order(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts, manifest = self.fixture(directory)
            result = FREEZE.build(attempts, manifest)
            self.assertEqual(len(result["jobs"]), 384)
            self.assertEqual(result, FREEZE.build(list(reversed(attempts)), manifest))
            self.assertEqual(result["jobs"][0]["estimated_completion_tokens"], 101)
            self.assertEqual(result["jobs"][-1]["estimated_completion_tokens"], 8192)
            self.assertEqual(result["jobs"][-1]["observations"][0]["estimate_source"], "8192_cap_unfinished")
            self.assertEqual(len(result["jobs"][-1]["observations"]), 2)
            self.assertEqual(len(result["sources"]), 17)
            self.assertEqual(result["algorithm"], "stable_longest_estimated_completion_first_v1")
            self.assertEqual(result["schema_version"], 1)

    def test_priority_ignores_answer_reasoning_score_and_finish_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts, manifest = self.fixture(directory)
            before = FREEZE.build(attempts, manifest)["jobs"]
            self.mutate(
                attempts[1] / "triviaqa/responses.jsonl",
                lambda rows: rows[0].update(
                    choices=None, score={"correct": 999}, finish_reason="length", reasoning="changed"
                ),
            )
            self.assertEqual(before, FREEZE.build(attempts, manifest)["jobs"])

    def test_archived_stream_parts_can_split_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts, manifest = self.fixture(directory)
            before = FREEZE.build(attempts, manifest)["jobs"]
            path = attempts[0] / "request_identity.jsonl"
            data = path.read_bytes()
            cut = len(data) // 2 + 3
            for i, part in enumerate((data[:cut], data[cut:])):
                with gzip.open(str(path) + f".hash.stream.part{i:04d}.gz", "wb") as stream:
                    stream.write(part)
            path.unlink()
            self.assertEqual(before, FREEZE.build(attempts, manifest)["jobs"])

    def test_bad_provenance_and_counts_fail_closed(self):
        changes = [
            "identity",
            "drop_input",
            "repeat",
            "messages",
            "generation",
            "link",
            "extra_doc",
            "duplicate_link",
            "duplicate_response",
            "count_bool",
            "count_over_cap",
            "manifest",
            "cross_identity",
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                attempts, manifest = self.fixture(directory)
                a = attempts[1]
                if change == "identity":
                    self.mutate(a / "request_identity.jsonl", lambda r: r[0].update(generation_sha256="bad"))
                elif change == "drop_input":
                    self.mutate(a / "upstream_inputs.jsonl", lambda r: r.pop())
                elif change == "repeat":
                    self.mutate(a / "upstream_inputs.jsonl", lambda r: r[0].update(repeats=2))
                elif change in ("messages", "generation"):

                    def change_args(r):
                        if change == "messages":
                            r[0]["args"][0][0] = "[]"
                        else:
                            r[0]["args"][1]["temperature"] = 0

                    self.mutate(a / "upstream_inputs.jsonl", change_args)
                elif change in ("link", "extra_doc", "duplicate_link"):
                    self.mutate(
                        a / "triviaqa/request_links.jsonl",
                        lambda r: r[0].update(request_sha256="bad")
                        if change == "link"
                        else r[0].update(doc_id=999)
                        if change == "extra_doc"
                        else r.append(copy.deepcopy(r[0])),
                    )
                elif change in ("duplicate_response", "count_bool", "count_over_cap"):
                    self.mutate(
                        a / "triviaqa/responses.jsonl",
                        lambda r: r.append(copy.deepcopy(r[0]))
                        if change == "duplicate_response"
                        else r[0]["usage"].update(completion_tokens=True if change == "count_bool" else 8193),
                    )
                elif change == "manifest":
                    d = json.loads((a / "manifest.json").read_text())
                    d["tasks"]["triviaqa"]["document_sha256"][0] = "bad"
                    (a / "manifest.json").write_text(json.dumps(d))
                else:
                    # A self-consistent changed native prompt still fails cross-attempt identity.
                    for name in ("upstream_inputs.jsonl", "upstream_inputs_before_identity.jsonl"):
                        self.mutate(
                            a / name, lambda r: r[0]["args"][0].__setitem__(0, '[{"role":"user","content":"changed"}]')
                        )
                    row = json.loads((a / "upstream_inputs.jsonl").read_text().splitlines()[0])
                    new = FREEZE.digest(row["args"])
                    self.mutate(
                        a / "request_identity.jsonl",
                        lambda r: r[0].update(
                            original_request_sha256=new,
                            effective_request_sha256=new,
                            structured_messages_sha256=FREEZE.digest(json.loads(row["args"][0][0])),
                        ),
                    )
                    self.mutate(a / "gsm8k_cot_zeroshot/request_links.jsonl", lambda r: r[0].update(request_sha256=new))
                with self.assertRaises(ValueError):
                    FREEZE.build(attempts, manifest)

    def test_duplicate_attempt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts, manifest = self.fixture(directory)
            with self.assertRaises(ValueError):
                FREEZE.build([attempts[0], attempts[0]], manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
