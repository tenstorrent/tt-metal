"""Host-only native tokenizer/parser contract checks; run as a Python script."""

import importlib.util
import json
import os
import unittest
from pathlib import Path

from transformers import AutoTokenizer
from vllm.reasoning import ReasoningParserManager

SNAPSHOT = Path(
    "/mnt/models/huggingface/hub/models--IFM--K2-Horizon-7B/snapshots/" "036114ce8d46c32b24c15423211069abb9c5d25e"
)
START = "<ifm|think>"
END = "</ifm|think>"


class NativeReasoningParserContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
        cls.plugin = Path(__file__).with_name("benchmark_reasoning_parser.py")
        ReasoningParserManager.import_reasoning_parser(str(cls.plugin))
        cls.parser_type = ReasoningParserManager.get_reasoning_parser("k2_horizon_benchmark")
        responses_path = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/benchmark_stage/responses.py"
        spec = importlib.util.spec_from_file_location("benchmark_stage_responses", responses_path)
        cls.responses = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.responses)

    def setUp(self):
        self.parser = self.parser_type(self.tokenizer)

    def stream(self, chunks):
        previous_text, previous_ids = "", []
        reasoning, content = [], []
        for delta_text in chunks:
            delta_ids = self.tokenizer.encode(delta_text, add_special_tokens=False)
            current_text, current_ids = previous_text + delta_text, previous_ids + delta_ids
            delta = self.parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_ids,
                current_ids,
                delta_ids,
            )
            if delta is not None:
                if delta.reasoning is not None:
                    reasoning.append(delta.reasoning)
                if delta.content is not None:
                    content.append(delta.content)
            previous_text, previous_ids = current_text, current_ids
        return "".join(reasoning), "".join(content)

    def test_registered_plugin_and_native_marker_ids(self):
        self.assertEqual(Path(__import__(self.parser_type.__module__).__file__), self.plugin)
        self.assertEqual(self.tokenizer.encode(START, add_special_tokens=False), [250029])
        self.assertEqual(self.tokenizer.encode(END, add_special_tokens=False), [250030])
        self.assertEqual(self.parser.start_token_id, 250029)
        self.assertEqual(self.parser.end_token_id, 250030)

    def test_native_template_supplies_exactly_one_implicit_opener(self):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 + 2?"}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="high",
        )
        self.assertEqual(prompt.count(START), 1)
        self.assertTrue(prompt.endswith(START + "\n"), repr(prompt[-100:]))
        self.assertEqual(self.stream(["Two plus two is four.", END, "4"]), ("Two plus two is four.", "4"))

    def test_explicit_opener(self):
        self.assertEqual(self.stream([START, "Reasoning", END, "Answer"]), ("Reasoning", "Answer"))
        self.assertEqual(self.stream([START + "Reasoning" + END + "Answer"]), ("Reasoning", "Answer"))

    def test_close_marker_alone(self):
        self.assertEqual(self.stream([END]), ("", ""))
        self.assertEqual(self.stream([END, "Answer"]), ("", "Answer"))

    def test_close_and_answer_in_same_chunk(self):
        self.assertEqual(self.stream(["Reasoning", END + "Answer"]), ("Reasoning", "Answer"))
        self.assertEqual(self.stream(["Reasoning" + END + "Answer"]), ("Reasoning", "Answer"))

    def test_split_text_chunks(self):
        self.assertEqual(self.stream(["Rea", "son", "ing", END, "Ans", "wer"]), ("Reasoning", "Answer"))

    def test_reasoning_only_length_preserves_empty_final_answer(self):
        self.assertEqual(self.stream(["Unfinished", " reasoning"]), ("Unfinished reasoning", ""))
        self.assertEqual(self.parser.extract_reasoning("Unfinished reasoning", None), ("Unfinished reasoning", None))

    def test_full_extraction(self):
        for opener in ("", START):
            with self.subTest(opener=opener):
                self.assertEqual(
                    self.parser.extract_reasoning(opener + "Reasoning" + END + "Answer", None), ("Reasoning", "Answer")
                )
                self.assertEqual(self.parser.extract_reasoning(opener + "Reasoning" + END, None), ("Reasoning", None))
        self.assertEqual(self.parser.extract_reasoning(END + "Answer", None), ("", "Answer"))

    def test_benchmark_scoring_preserves_raw_reasoning_and_question(self):
        reasoning, content = self.parser.extract_reasoning("Unfinished reasoning", None)
        raw = {
            "choices": [
                {"index": 0, "finish_reason": "length", "message": {"content": content, "reasoning": reasoning}}
            ],
            "usage": {"completion_tokens": 2},
        }
        normalized, exhausted = self.responses.scoring_response(raw)
        self.assertTrue(exhausted)
        self.assertIsNone(raw["choices"][0]["message"]["content"])
        self.assertEqual(normalized["choices"][0]["message"]["content"], "")
        self.assertEqual(normalized["choices"][0]["message"]["reasoning"], reasoning)

    def test_full_eight_shot_prompt_keeps_new_generation_in_reasoning(self):
        evidence_path = Path(__file__).resolve().parents[1] / "doc/benchmark/setup/gsm8k-native-chat-contract.json"
        evidence = json.loads(evidence_path.read_text())
        self.assertEqual(len(evidence["requests"]), 256)
        for row in evidence["requests"]:
            prompt = self.tokenizer.apply_chat_template(
                row["server_normalized_messages"],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="high",
            )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            self.assertEqual(ids.count(self.parser.start_token_id), 9)
            self.assertEqual(ids.count(self.parser.end_token_id), 8)
            # This is the actual non-streaming server initialization call.
            self.assertFalse(self.parser.is_reasoning_end(ids))
            self.assertEqual(
                self.parser.extract_reasoning("fresh thought" + END + "answer", None), ("fresh thought", "answer")
            )

    def test_returned_token_metadata_does_not_change_sampling(self):
        import msgspec
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

        payload = {
            "model": "IFM/K2-Horizon-7B",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "max_tokens": 32768,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "seed": 1234,
            "stop": [],
            "chat_template_kwargs": {"reasoning_effort": "high"},
        }
        original = ChatCompletionRequest(**payload, return_token_ids=False)
        recorded = ChatCompletionRequest(**payload, return_token_ids=True)
        self.assertFalse(original.stream)
        self.assertIsNone(original.structured_outputs)
        self.assertEqual(
            msgspec.to_builtins(original.to_sampling_params(32768, {})),
            msgspec.to_builtins(recorded.to_sampling_params(32768, {})),
        )

    def test_initial_close_control_reproduces_observed_leak_without_prompt_state(self):
        evidence_path = Path(__file__).resolve().parents[1] / "doc/benchmark/setup/fewshot-parser-faulty-responses.json"
        observed = json.loads(evidence_path.read_text())["examples"][0]["choices"][0]["message"]
        # A discriminating hypothetical control, not recovered raw generation.
        # The real generated token IDs are needed before changing semantics.
        possible_generation = END + observed["content"]
        self.assertEqual(
            self.parser.extract_reasoning(possible_generation, None),
            (observed["reasoning"], observed["content"]),
        )

    def test_actual_generated_repeated_closers_remain_literal_content(self):
        directory = Path(__file__).resolve().parents[1] / "doc/benchmark/setup/parser-token-control"
        paths = sorted(directory.glob("response-*.json"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            response = json.loads(path.read_text())["response"]
            choice = response["choices"][0]
            ids = choice["token_ids"]
            self.assertFalse(self.parser.is_reasoning_end(response["prompt_token_ids"]))
            self.assertEqual(ids[0], self.parser.end_token_id)
            self.assertEqual(ids.count(self.parser.start_token_id), 0)
            self.assertGreaterEqual(ids.count(self.parser.end_token_id), 2)
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            message = choice["message"]
            self.assertEqual(self.parser.extract_reasoning(decoded, None), (message["reasoning"], message["content"]))
            self.assertEqual(message["reasoning"], "")
            self.assertIn(END, message["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
