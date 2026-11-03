# SPDX-License-Identifier: Apache-2.0
"""Host-only parser checks; run directly to avoid hardware pytest conftest."""
import unittest
from pathlib import Path

from transformers import AutoTokenizer
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

PLUGIN = Path(__file__).with_name("benchmark_reasoning.py")
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--IFM--K2-Horizon-MoVA-36B-A4B/snapshots/e9ce717642b2074082875be233c02c729ba47960"
)


class ReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ReasoningParserManager.import_reasoning_parser(str(PLUGIN))
        cls.tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=True)
        cls.parser_type = ReasoningParserManager.get_reasoning_parser("k2_horizon")

    def setUp(self):
        self.parser = self.parser_type(self.tokenizer)

    def tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def stream(self, chunks):
        previous_text, previous_ids = "", []
        reasoning, content = "", ""
        for chunk in chunks:
            delta_ids = self.tokens(chunk)
            current_text = previous_text + chunk
            current_ids = previous_ids + delta_ids
            delta = self.parser.extract_reasoning_streaming(
                previous_text, current_text, chunk, previous_ids, current_ids, delta_ids
            )
            if delta is not None:
                reasoning += delta.reasoning or ""
                content += delta.content or ""
            previous_text, previous_ids = current_text, current_ids
        return reasoning, content or None

    def test_nonstream_prompt_open(self):
        self.assertEqual(self.parser.extract_reasoning("Analysis.</ifm|think>Final.", None), ("Analysis.", "Final."))

    def test_nonstream_generated_open(self):
        self.assertEqual(
            self.parser.extract_reasoning("<ifm|think>Analysis.</ifm|think>Final.", None), ("Analysis.", "Final.")
        )

    def test_nonstream_truncated_reasoning(self):
        self.assertEqual(self.parser.extract_reasoning("Incomplete reasoning", None), ("Incomplete reasoning", None))

    def test_stream_prompt_open(self):
        self.assertEqual(self.stream(["Analysis", ".", "</ifm|think>", "Final", "."]), ("Analysis.", "Final."))

    def test_stream_generated_open(self):
        self.assertEqual(self.stream(["<ifm|think>", "Analysis.", "</ifm|think>", "Final."]), ("Analysis.", "Final."))

    def test_stream_batched_markers(self):
        self.assertEqual(self.stream(["<ifm|think>Analysis.", "</ifm|think>Final."]), ("Analysis.", "Final."))
        self.assertEqual(self.stream(["<ifm|think>Analysis.</ifm|think>Final."]), ("Analysis.", "Final."))

    def test_stream_truncated_reasoning(self):
        self.assertEqual(self.stream(["Incomplete", " reasoning"]), ("Incomplete reasoning", None))

    def test_reasoning_token_count(self):
        analysis = self.tokens("Analysis.")
        final = self.tokens("Final.")
        self.assertEqual(self.parser.count_reasoning_tokens(analysis), len(analysis))
        for opening in ([], [self.parser.start_token_id]):
            ids = opening + analysis + [self.parser.end_token_id] + final
            self.assertEqual(self.parser.count_reasoning_tokens(ids), len(analysis))
            self.assertEqual(self.parser.extract_content_ids(ids), final)
            self.assertTrue(self.parser.is_reasoning_end(ids))

    def test_native_template_once_zero_shot(self):
        messages = [{"role": "user", "content": "A benchmark question."}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, reasoning_effort="high"
        )
        self.assertEqual(rendered.count("<|ifm|im_start|>user"), 1)
        self.assertEqual(rendered.count("<|ifm|im_start|>assistant"), 1)
        self.assertTrue(rendered.endswith("<ifm|think>\n"))
        for marker in (self.parser.start_token, self.parser.end_token):
            self.assertEqual(len(self.tokens(marker)), 1)

    def test_native_template_fewshot_requires_thinking_field(self):
        from jinja2.exceptions import TemplateError

        messages = [
            {"role": "user", "content": "Example?"},
            {"role": "assistant", "content": "Example answer."},
            {"role": "user", "content": "Question?"},
        ]
        with self.assertRaisesRegex(TemplateError, "missing a thinking field"):
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        messages[1]["reasoning"] = ""
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertIn("</ifm|think>Example answer.", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
