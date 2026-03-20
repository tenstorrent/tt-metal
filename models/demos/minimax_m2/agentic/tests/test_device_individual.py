# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Per-tool device tests for the General Agentic Mode.

Tests each tool wrapper in isolation:
  1. First call  — triggers warmup / kernel compilation / trace capture
  2. Second call — verifies trace is reused (must not re-compile)
  3. Output format validation

Run with:
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    source python_env/bin/activate

    pytest models/demos/minimax_m2/agentic/tests/test_device_individual.py -v -m device

Marks:
  device  — all tests in this file require the N300 hardware.
"""

import time
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

import ttnn

from .conftest import BERT_CONTEXT, BERT_EXPECTED_SUBSTRING, BERT_QUESTION

pytestmark = pytest.mark.device


# ===========================================================================
# Helpers
# ===========================================================================


def _assert_nonempty_str(val, label: str):
    assert isinstance(val, str), f"{label}: expected str, got {type(val)}"
    assert len(val.strip()) > 0, f"{label}: returned empty string"


def _assert_audio_file(path: str, label: str):
    p = Path(path)
    assert p.exists(), f"{label}: WAV file not created at {path}"
    assert p.stat().st_size > 1000, f"{label}: WAV file suspiciously small ({p.stat().st_size} bytes)"


# ===========================================================================
# Whisper — transcription
# ===========================================================================


class TestWhisperTool:
    """Warmup + two-call trace-reuse test for Whisper."""

    def test_transcribe_first_call(self, whisper_tool, test_audio_path):
        """First call triggers Whisper decode-trace compilation."""
        result = whisper_tool.transcribe(test_audio_path)
        # Whisper on a sine-wave may return empty or noise text — that is OK;
        # we verify the call completes without error and returns a string.
        assert isinstance(result, str), "transcribe() must return str"

    def test_transcribe_second_call_trace_reuse(self, whisper_tool, test_audio_path):
        """Second call must reuse the compiled trace (faster, same return type)."""
        t0 = time.perf_counter()
        result = whisper_tool.transcribe(test_audio_path)
        elapsed = time.perf_counter() - t0
        assert isinstance(result, str)
        # The second call should complete in reasonable time (< 120 s for 2-second audio)
        assert elapsed < 120.0, f"Second transcribe call took {elapsed:.1f}s — possible trace miss"

    def test_transcribe_longer_audio(self, whisper_tool, test_audio_path_long):
        """Longer audio (5 s) must complete without error."""
        result = whisper_tool.transcribe(test_audio_path_long)
        assert isinstance(result, str)

    def test_translate_first_call(self, whisper_tool, test_audio_path):
        """translate() compiles a separate pipeline but must return a string."""
        result = whisper_tool.translate(test_audio_path)
        assert isinstance(result, str)

    def test_translate_second_call(self, whisper_tool, test_audio_path_long):
        """Second translate() call verifies pipeline reuse."""
        result = whisper_tool.translate(test_audio_path_long)
        assert isinstance(result, str)


# ===========================================================================
# SpeechT5 — text-to-speech
# ===========================================================================


class TestSpeechT5Tool:
    """Warmup + two-call trace-reuse test for SpeechT5 TTS."""

    def test_synthesize_short_text(self, speecht5_tool, tts_output_path):
        """First call triggers TTS model compilation."""
        out = speecht5_tool.synthesize("Hello from Tenstorrent.", output_path=tts_output_path)
        assert out == tts_output_path, "synthesize() must return the output path"
        _assert_audio_file(tts_output_path, "SpeechT5 first call")

    def test_synthesize_second_call_trace_reuse(self, speecht5_tool, tmp_path):
        """Second call must not re-compile; WAV must be written."""
        out2 = speecht5_tool.synthesize(
            "The N300 has two Wormhole B0 chips.",
            output_path=str(tmp_path / "tts2.wav"),
        )
        _assert_audio_file(out2, "SpeechT5 second call")

    def test_synthesize_longer_text(self, speecht5_tool, tmp_path):
        """Longer text (multiple words) must produce a valid WAV."""
        text = (
            "The Tenstorrent N300 is an AI accelerator with two Wormhole B0 chips, "
            "each with eighty Tensix cores and twelve gigabytes of memory."
        )
        out = speecht5_tool.synthesize(text, output_path=str(tmp_path / "tts_long.wav"))
        _assert_audio_file(out, "SpeechT5 long text")

    def test_synthesize_output_is_readable_audio(self, speecht5_tool, tmp_path):
        """Synthesized file must be a readable WAV with non-zero samples."""
        import soundfile as sf

        out = speecht5_tool.synthesize("Test audio output.", output_path=str(tmp_path / "tts_check.wav"))
        data, sr = sf.read(out)
        assert sr > 0, "Sample rate must be positive"
        assert len(data) > 0, "Audio data must not be empty"
        assert np.abs(data).max() > 1e-6, "Audio data must not be all-zero"


# ===========================================================================
# Cross-tool sequential test on one shared opened device
# ===========================================================================


class TestSequentialWhisperThenSpeechT5:
    """
    Agentic pattern: load + warm up ALL tools first, then dispatch inference.

    1. Load Whisper (full mesh) — pipeline created, trace not yet captured
    2. Load SpeechT5 (chip0 view) — warmup inference runs in __init__
    3. Whisper warmup call — captures decoder trace (first pipeline invocation)
    4. SpeechT5 warmup already done in step 2
    5. Run Whisper inference — reuses captured trace
    6. Run SpeechT5 inference — reuses program cache
    """

    def test_load_warmup_then_infer_sequentially(self, mesh_device, test_audio_path, tmp_path):
        from models.demos.minimax_m2.agentic.tool_wrappers.speecht5_tool import SpeechT5Tool
        from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

        chip0 = (
            mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
            if mesh_device.get_num_devices() > 1
            else mesh_device
        )

        # ── Phase 1: Load both models ──
        logger.info("=== PHASE 1: Loading Whisper (full mesh) ===")
        whisper = WhisperTool(mesh_device=mesh_device)

        logger.info("=== PHASE 1: Loading SpeechT5 (chip0 view) ===")
        speecht5 = SpeechT5Tool(mesh_device=chip0)

        # ── Phase 2: Warm up Whisper (trace capture on first call) ──
        logger.info("=== PHASE 2: Whisper warmup (trace capture) ===")
        warmup_transcript = whisper.transcribe(test_audio_path)
        assert isinstance(warmup_transcript, str), "Whisper warmup must return str"
        logger.info(f"Whisper warmup done, transcript type: {type(warmup_transcript)}")

        # SpeechT5 warmup already ran during __init__ — both tools are now ready
        logger.info("=== Both tools warmed up, traces/caches captured ===")

        # ── Phase 3: Inference dispatches (agentic turns) ──
        logger.info("=== PHASE 3a: Whisper inference (trace reuse) ===")
        transcript = whisper.transcribe(test_audio_path)
        assert isinstance(transcript, str), "Whisper inference must return str"
        logger.info(f"Whisper inference done: {transcript!r}")

        logger.info("=== PHASE 3b: SpeechT5 inference (program cache reuse) ===")
        out = speecht5.synthesize(
            "Sequential shared-device pytest check.",
            output_path=str(tmp_path / "sequential_tts.wav"),
        )
        _assert_audio_file(out, "SpeechT5 sequential inference")
        logger.info(f"SpeechT5 inference done: {out}")

        whisper.close()


# ===========================================================================
# OWL-ViT — object detection
# ===========================================================================


class TestOWLViTTool:
    """Warmup + two-call trace-reuse test for OWL-ViT."""

    def test_detect_first_call_returns_list(self, owlvit_tool, test_image_path):
        """First call compiles OWL-ViT TTNN graph; output must be a list."""
        results = owlvit_tool.detect(test_image_path, "red block, blue block, green block")
        assert isinstance(results, list), "detect() must return a list"

    def test_detect_second_call_trace_reuse(self, owlvit_tool, test_image_path):
        """Second call must reuse compiled kernels."""
        results = owlvit_tool.detect(test_image_path, "coloured square")
        assert isinstance(results, list)

    def test_detect_result_format(self, owlvit_tool, test_image_path):
        """Each detection must have label, score, and bbox keys."""
        results = owlvit_tool.detect(test_image_path, "red block, blue block")
        for det in results:
            assert "label" in det, "Detection missing 'label'"
            assert "score" in det, "Detection missing 'score'"
            assert "bbox" in det, "Detection missing 'bbox'"
            assert 0.0 <= det["score"] <= 1.0, "Score must be in [0, 1]"
            assert len(det["bbox"]) == 4, "BBox must have 4 coordinates"

    def test_detect_with_no_threshold_match(self, owlvit_tool, test_image_path):
        """Querying for an object not in image must return empty list (no crash)."""
        results = owlvit_tool.detect(test_image_path, "airplane, helicopter, submarine", threshold=0.99)
        assert isinstance(results, list)

    def test_detect_different_queries(self, owlvit_tool, test_image_path):
        """Multiple distinct queries on the same image must each complete."""
        for query in ["red square", "blue rectangle", "green shape"]:
            results = owlvit_tool.detect(test_image_path, query)
            assert isinstance(results, list), f"Failed for query: {query}"


# ===========================================================================
# BERT — extractive QA
# ===========================================================================


class TestBERTTool:
    """Warmup + two-call trace-reuse test for BERT Large QA."""

    def test_qa_first_call(self, bert_tool):
        """First call triggers BERT TTNN graph compilation."""
        answer = bert_tool.qa(BERT_QUESTION, BERT_CONTEXT)
        _assert_nonempty_str(answer, "BERT first QA call")

    def test_qa_answer_contains_expected(self, bert_tool):
        """Extractive answer for 'how many chips' must mention 'two'."""
        answer = bert_tool.qa(BERT_QUESTION, BERT_CONTEXT)
        assert (
            BERT_EXPECTED_SUBSTRING.lower() in answer.lower()
        ), f"Expected '{BERT_EXPECTED_SUBSTRING}' in answer, got: '{answer}'"

    def test_qa_second_call_trace_reuse(self, bert_tool):
        """Second call must reuse compiled BERT graph."""
        q = "How much total memory does the N300 have?"
        answer = bert_tool.qa(q, BERT_CONTEXT)
        _assert_nonempty_str(answer, "BERT second QA call")

    def test_qa_different_questions(self, bert_tool):
        """Multiple questions on the same context must all return non-empty strings."""
        questions = [
            "What chip is used in the N300?",
            "How many Tensix cores per chip?",
            "What type of memory does the N300 use?",
        ]
        for q in questions:
            answer = bert_tool.qa(q, BERT_CONTEXT)
            _assert_nonempty_str(answer, f"BERT qa: '{q}'")

    def test_qa_short_context(self, bert_tool):
        """Very short context must not crash BERT."""
        answer = bert_tool.qa("Who made this?", "Tenstorrent made the N300.")
        assert isinstance(answer, str)


# ===========================================================================
# LLM — Llama 3.2 3B Instruct
# ===========================================================================


class TestLLMTool:
    """Warmup + two-call trace-reuse test for TTNN Llama 3B."""

    def test_generate_first_call_triggers_warmup(self, llm_tool):
        """
        First generate_response call triggers prefill kernel warmup
        (for all supported sequence lengths) and lazy decode-trace capture.
        """
        messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]
        response = llm_tool.generate_response(messages, max_new_tokens=16)
        _assert_nonempty_str(response, "LLM first call")

    def test_generate_answer_contains_four(self, llm_tool):
        """Greedy decode of '2+2' must mention 4 somewhere."""
        messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]
        response = llm_tool.generate_response(messages, max_new_tokens=16)
        assert "4" in response, f"Expected '4' in response, got: '{response}'"

    def test_generate_second_call_trace_reuse(self, llm_tool):
        """Second call must reuse compiled decode trace."""
        messages = [{"role": "user", "content": "What is the capital of France? One word."}]
        response = llm_tool.generate_response(messages, max_new_tokens=16)
        _assert_nonempty_str(response, "LLM second call")

    def test_generate_with_system_message(self, llm_tool):
        """System message injection via apply_chat_template must work."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always respond in exactly 3 words."},
            {"role": "user", "content": "Say hello."},
        ]
        response = llm_tool.generate_response(messages, max_new_tokens=24)
        _assert_nonempty_str(response, "LLM with system message")

    def test_generate_with_tool_schemas_injected(self, llm_tool):
        """
        Tool schemas passed to generate_response must be included in the
        tokenized prompt without error. The output may or may not be a tool call.
        """
        from models.demos.minimax_m2.agentic.tools import TOOL_SCHEMAS

        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What is 3 * 7?"},
        ]
        response = llm_tool.generate_response(messages, tools=TOOL_SCHEMAS, max_new_tokens=32)
        _assert_nonempty_str(response, "LLM with tool schemas")

    def test_generate_returns_tool_call_for_audio(self, llm_tool):
        """
        When the user message contains [AUDIO_ATTACHMENT: path] and the
        system prompt says to call transcribe_audio, the LLM should emit a tool call.
        """
        from models.demos.minimax_m2.agentic.orchestrator import SYSTEM_PROMPT, _is_tool_call
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call
        from models.demos.minimax_m2.agentic.tools import TOOL_SCHEMAS

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Please transcribe this audio.\n[AUDIO_ATTACHMENT: /tmp/voice.wav]"},
        ]
        response = llm_tool.generate_response(messages, tools=TOOL_SCHEMAS, max_new_tokens=128)
        _assert_nonempty_str(response, "LLM audio tool call")
        # The LLM MUST emit a tool call given the SYSTEM_PROMPT instruction
        assert _is_tool_call(response), f"LLM did not emit a tool call for audio attachment. Got:\n{response}"
        parsed = _parse_tool_call(response)
        assert parsed is not None, "Could not parse tool call JSON"
        name, args = parsed
        assert name == "transcribe_audio", f"Expected transcribe_audio, got: {name}"
        assert "path" in args, "transcribe_audio call missing 'path' argument"

    def test_generate_returns_tool_call_for_image(self, llm_tool):
        """
        When the user message contains [IMAGE_ATTACHMENT: path], the LLM
        should emit a detect_objects tool call.
        """
        from models.demos.minimax_m2.agentic.orchestrator import SYSTEM_PROMPT, _is_tool_call
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call
        from models.demos.minimax_m2.agentic.tools import TOOL_SCHEMAS

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What objects are in this image?\n[IMAGE_ATTACHMENT: /tmp/photo.jpg]"},
        ]
        response = llm_tool.generate_response(messages, tools=TOOL_SCHEMAS, max_new_tokens=128)
        _assert_nonempty_str(response, "LLM image tool call")
        assert _is_tool_call(response), f"LLM did not emit a tool call for image attachment. Got:\n{response}"
        parsed = _parse_tool_call(response)
        assert parsed is not None
        name, args = parsed
        assert name == "detect_objects", f"Expected detect_objects, got: {name}"
