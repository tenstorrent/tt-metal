"""Host-only checks against actual upstream GSM8K construction and IFM template."""

import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODEL_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(
    "/mnt/models/huggingface/hub/models--IFM--K2-Horizon-7B/snapshots/" "036114ce8d46c32b24c15423211069abb9c5d25e"
)
EVIDENCE = MODEL_DIR / "doc/benchmark/setup/gsm8k-native-chat-contract.json"
PLUGIN = Path(os.environ["TT_MODEL_BRINGUP_ROOT"])


def capture_requests(destination):
    """Run unchanged upstream construction, intercept before any HTTP call."""
    from benchmark_stage.evaluate import evaluate_groups
    from benchmark_stage.subsets import digest
    from lm_eval.models.openai_completions import LocalChatCompletion

    manifest_path = PLUGIN / "runtime/benchmark_stage/profiles/ci-v1.json"
    manifest = json.loads(manifest_path.read_text())

    class Captured(Exception):
        pass

    def capture(self, requests, **kwargs):
        rows = [
            {
                "doc_id": manifest["tasks"][request.task_name]["indices"][request.doc_id],
                "doc_sha256": digest(request.doc),
                "arguments": request.args,
                "messages": self.create_message([request.args[0]]),
                "request_sha256": digest([request.args[0], request.args[1]]),
            }
            for request in requests
        ]
        Path(destination).write_text(json.dumps(rows, ensure_ascii=False) + "\n")
        raise Captured

    LocalChatCompletion.generate_until = capture
    with tempfile.TemporaryDirectory() as directory:
        try:
            evaluate_groups(
                model="IFM/K2-Horizon-7B",
                base_url="http://127.0.0.1:1",
                manifest_path=manifest_path,
                groups=["gsm8k_cot"],
                output=Path(directory) / "requests",
                generation={
                    "max_gen_toks": 32768,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": -1,
                    "do_sample": True,
                    "until": [],
                    "chat_template_kwargs": {"reasoning_effort": "high"},
                },
                shared=True,
            )
        except Captured:
            return
    raise AssertionError("Upstream construction did not reach the interception point")


class NativeChatMiddlewareContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from vllm.entrypoints.chat_utils import parse_chat_messages
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

        from models.autoports.ifm_k2_horizon_7b.tests.benchmark_chat_middleware import (
            K2BenchmarkChatMiddleware,
            normalize_assistant_history,
        )

        cls.middleware = K2BenchmarkChatMiddleware
        cls.normalize = staticmethod(normalize_assistant_history)
        cls.parse_chat_messages = staticmethod(parse_chat_messages)
        cls.request_type = ChatCompletionRequest
        cls.tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
        cls.manifest = json.loads((PLUGIN / "runtime/benchmark_stage/profiles/ci-v1.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            captured = Path(directory) / "requests.json"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(PLUGIN / "runtime") + os.pathsep + environment.get("PYTHONPATH", "")
            subprocess.run(
                [
                    "/home/vkovacevic/k2-horizon/lm-eval-env/bin/python",
                    str(Path(__file__).resolve()),
                    "--capture",
                    str(captured),
                ],
                env=environment,
                check=True,
                timeout=120,
            )
            cls.rows = json.loads(captured.read_text())

    def payload(self, messages):
        return {
            "model": "IFM/K2-Horizon-7B",
            "messages": messages,
            "chat_template_kwargs": {"reasoning_effort": "high"},
        }

    def render(self, payload):
        request = self.request_type(**payload)
        conversation, mm_data, mm_ids = self.parse_chat_messages(
            request.messages,
            SimpleNamespace(multimodal_config=None, allowed_local_media_path="", allowed_media_domains=None),
            "string",
        )
        self.assertIsNone(mm_data)
        self.assertIsNone(mm_ids)
        return conversation, self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, reasoning_effort="high"
        )

    async def through_asgi(self, payload, path="/v1/chat/completions", method="POST", raw=None):
        body = raw if raw is not None else json.dumps(payload, ensure_ascii=False).encode()
        original_scope = {
            "type": "http",
            "path": path,
            "method": method,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"x-test", b"kept"),
            ],
        }
        events = [
            {"type": "http.request", "body": body[:7], "more_body": True},
            {"type": "http.request", "body": body[7:], "more_body": False},
        ]
        expected_events = copy.deepcopy(events)
        consumed, sent = [], []

        async def receive():
            return events.pop(0) if events else {"type": "http.disconnect"}

        async def send(event):
            sent.append(event)

        async def app(scope, receive, send):
            while True:
                event = await receive()
                consumed.append(event)
                if not event.get("more_body", False):
                    break
            consumed.append(await receive())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            return scope

        observed_scope = await self.middleware(app)(original_scope, receive, send)
        return original_scope, observed_scope, expected_events, consumed, sent

    def test_all_256_packaged_questions_preserve_eight_examples_and_one_template(self):
        self.assertEqual(len(self.rows), 256)
        frozen = self.manifest["tasks"]["gsm8k_cot"]
        self.assertEqual(frozen["num_fewshot"], 8)
        expected_docs = dict(zip(frozen["indices"], frozen["document_sha256"]))
        evidence_rows = []
        for row in self.rows:
            self.assertEqual(row["doc_sha256"], expected_docs[row["doc_id"]])
            messages = row["messages"]
            original = self.payload(messages)
            before = copy.deepcopy(original)
            normalized = self.normalize(original)
            _, _, _, events, _ = asyncio.run(self.through_asgi(original))
            self.assertEqual(json.loads(events[0]["body"]), normalized)
            self.assertEqual(original, before)
            self.assertEqual(len(messages), 17)
            self.assertEqual(sum(message["role"] == "assistant" for message in messages), 8)
            conversation, prompt = self.render(normalized)
            self.assertEqual([m["content"] for m in conversation], [m["content"] for m in messages])
            self.assertEqual([m["role"] for m in conversation], [m["role"] for m in messages])
            # The native template writes an empty think pair for each of the
            # eight assistant exemplars, then one unmatched generation opener.
            self.assertEqual(prompt.count("<ifm|think>"), 9)
            self.assertEqual(prompt.count("</ifm|think>"), 8)
            self.assertEqual(prompt.count("<|ifm|im_start|>user"), 9)
            self.assertEqual(prompt.count("<|ifm|im_start|>assistant"), 9)
            self.assertTrue(prompt.endswith("<ifm|think>\n"))
            for message in messages:
                self.assertIn(message["content"], prompt)
            evidence_rows.append(
                {
                    **row,
                    "original_messages": messages,
                    "normalized_messages": normalized["messages"],
                    "server_normalized_messages": conversation,
                    "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
        EVIDENCE.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "inference_calls": 0,
                    "construction": "Pinned benchmark_stage evaluate_groups/lm-eval 0.4.13, intercepted before HTTP",
                    "normalization": "ASGI -> ChatCompletionRequest -> actual chat_utils -> pinned HF template once",
                    "manifest_sha256": self.manifest["manifest_sha256"],
                    "native_template_sha256": hashlib.sha256(
                        (SNAPSHOT / "chat_template.jinja").read_bytes()
                    ).hexdigest(),
                    "task": "gsm8k_cot",
                    "samples": 256,
                    "population": 1319,
                    "num_fewshot": 8,
                    "fewshot_sha256": frozen["fewshot_sha256"],
                    "requests": evidence_rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

    def test_unadapted_upstream_exemplars_fail_native_template(self):
        import jinja2

        with self.assertRaisesRegex(jinja2.TemplateError, "missing a thinking field"):
            self.render(self.payload(self.rows[0]["messages"]))

    def test_valid_supplied_reasoning_is_preserved(self):
        for fields in (
            {"reasoning": "existing"},
            {"reasoning_content": "existing"},
            {"reasoning": "first", "reasoning_content": "second"},
        ):
            original = self.payload(
                [{"role": "assistant", "content": "answer", **fields}, {"role": "user", "content": "next"}]
            )
            before = copy.deepcopy(original)
            normalized = self.normalize(original)
            conversation, prompt = self.render(normalized)
            expected = fields.get("reasoning", fields.get("reasoning_content"))
            self.assertEqual(conversation[0]["reasoning"], expected)
            self.assertIn(expected, prompt)
            self.assertEqual(original, before)
            for key, value in fields.items():
                self.assertEqual(normalized["messages"][0][key], value)

    def test_invalid_supplied_reasoning_remains_unchanged(self):
        for key in ("reasoning", "reasoning_content"):
            for value in (None, 3, [], {}):
                original = self.payload([{"role": "assistant", "content": "answer", key: value}])
                self.assertIs(self.normalize(original), original)

    def test_zero_shot_and_other_model_unchanged(self):
        original = self.payload([{"role": "user", "content": "unaltered é question"}])
        self.assertIs(self.normalize(original), original)
        self.assertEqual(self.render(original), self.render(self.normalize(original)))
        original = self.payload([{"role": "assistant", "content": "answer"}])
        original["model"] = "other/model"
        self.assertIs(self.normalize(original), original)

    def test_chunked_body_and_content_length(self):
        original = self.payload(
            [{"role": "assistant", "content": "example é"}, {"role": "user", "content": "question"}]
        )
        before_scope, scope, _, events, sent = asyncio.run(self.through_asgi(original))
        body = events[0]["body"]
        self.assertEqual(json.loads(body), self.normalize(original))
        self.assertEqual(dict(scope["headers"])[b"content-length"], str(len(body)).encode())
        self.assertEqual(dict(scope["headers"])[b"x-test"], b"kept")
        self.assertNotEqual(dict(before_scope["headers"])[b"content-length"], str(len(body)).encode())
        self.assertEqual(events[-1], {"type": "http.disconnect"})
        self.assertEqual(sent[0]["status"], 200)

    def test_actual_import_identity_uses_only_configured_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transport.json"
            with patch.dict(os.environ, {"K2_BENCHMARK_CHAT_TRANSPORT_PATH": str(path)}):
                self.middleware(None)
            identity = json.loads(path.read_text())
            source = Path(identity["file"])
            self.assertEqual(source, Path(__file__).with_name("benchmark_chat_middleware.py"))
            self.assertEqual(identity["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(identity["class"], self.middleware.__module__ + ".K2BenchmarkChatMiddleware")
            self.assertFalse(identity["template_rendered_by_middleware"])

    def test_nonmatching_requests_preserve_exact_body_chunks_and_headers(self):
        cases = [
            (self.payload([{"role": "user", "content": "question"}]), "/v1/chat/completions", "POST", None),
            (self.payload([{"role": "assistant", "content": "answer"}]), "/v1/completions", "POST", None),
            (
                {"model": "other/model", "messages": [{"role": "assistant", "content": "answer"}]},
                "/v1/chat/completions",
                "POST",
                None,
            ),
            ({}, "/v1/chat/completions", "GET", None),
            ({}, "/v1/chat/completions", "POST", b"not valid JSON"),
        ]
        for payload, path, method, raw in cases:
            before_scope, scope, original_events, events, _ = asyncio.run(self.through_asgi(payload, path, method, raw))
            self.assertIs(scope, before_scope)
            self.assertEqual(events[:-1], original_events)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--capture":
        capture_requests(sys.argv[2])
    else:
        unittest.main(verbosity=2)
