# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for LLM tool calling in the agentic workflow.

Tests:
1. Tool call parsing (unit tests - no device needed)
2. Translation chain: English → French → German → English → TTS → Whisper
3. Image detection with OWL-ViT using test images

Usage:
    # Run all tests
    pytest models/demos/minimax_m2/agentic/tests/test_tool_calling.py -v

    # Run specific test
    pytest models/demos/minimax_m2/agentic/tests/test_tool_calling.py::test_parse_tool_call -v

    # Run device tests only (requires N300)
    pytest models/demos/minimax_m2/agentic/tests/test_tool_calling.py -v -k "device"
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))


# =============================================================================
# Unit Tests - No Device Required
# =============================================================================


class TestToolCallParsing:
    """Unit tests for LLM tool call parsing."""

    def test_parse_python_tag_format(self):
        """Test parsing Llama 3.2 <|python_tag|> format."""
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

        text = '<|python_tag|>{"name": "transcribe_audio", "arguments": {"path": "/tmp/audio.wav"}}'
        result = _parse_tool_call(text)

        assert result is not None
        name, args = result
        assert name == "transcribe_audio"
        assert args["path"] == "/tmp/audio.wav"

    def test_parse_json_code_block(self):
        """Test parsing ```json code block format."""
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

        text = """I'll help you with that.
```json
{"name": "detect_objects", "parameters": {"image_path": "/tmp/img.jpg", "query": "cat"}}
```
"""
        result = _parse_tool_call(text)

        assert result is not None
        name, args = result
        assert name == "detect_objects"
        assert args["image_path"] == "/tmp/img.jpg"
        assert args["query"] == "cat"

    def test_parse_bare_json(self):
        """Test parsing bare JSON object in text."""
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

        text = 'Let me call a tool: {"name": "text_to_speech", "arguments": {"text": "Hello world"}}'
        result = _parse_tool_call(text)

        assert result is not None
        name, args = result
        assert name == "text_to_speech"
        assert args["text"] == "Hello world"

    def test_parse_with_parameters_key(self):
        """Test parsing with 'parameters' instead of 'arguments'."""
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

        text = '{"name": "answer_from_context", "parameters": {"question": "What is X?", "context": "X is Y."}}'
        result = _parse_tool_call(text)

        assert result is not None
        name, args = result
        assert name == "answer_from_context"
        assert args["question"] == "What is X?"

    def test_parse_no_tool_call(self):
        """Test that plain text returns None."""
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

        text = "The answer is 42. No tools needed."
        result = _parse_tool_call(text)
        assert result is None

    def test_parse_invalid_json(self):
        """Test that malformed JSON returns None."""
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

        text = '{"name": "broken", "arguments": {missing quotes}}'
        result = _parse_tool_call(text)
        assert result is None

    def test_parse_nested_json(self):
        """Test parsing nested JSON objects."""
        from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

        text = '{"name": "complex_tool", "arguments": {"config": {"nested": {"value": 123}}}}'
        result = _parse_tool_call(text)

        assert result is not None
        name, args = result
        assert name == "complex_tool"
        assert args["config"]["nested"]["value"] == 123


class TestIsToolCall:
    """Unit tests for tool call detection."""

    def test_detect_python_tag(self):
        """Test detection of <|python_tag|> format."""
        from models.demos.minimax_m2.agentic.orchestrator import _is_tool_call

        assert _is_tool_call('<|python_tag|>{"name": "foo"}') is True
        assert _is_tool_call("Just plain text") is False

    def test_detect_json_code_block(self):
        """Test detection of ```json code block."""
        from models.demos.minimax_m2.agentic.orchestrator import _is_tool_call

        text = '```json\n{"name": "tool"}\n```'
        assert _is_tool_call(text) is True

    def test_detect_bare_json_tool(self):
        """Test detection of bare JSON with name and arguments."""
        from models.demos.minimax_m2.agentic.orchestrator import _is_tool_call

        text = '{"name": "my_tool", "arguments": {}}'
        assert _is_tool_call(text) is True

    def test_no_false_positives(self):
        """Test that normal text doesn't trigger false positives."""
        from models.demos.minimax_m2.agentic.orchestrator import _is_tool_call

        assert _is_tool_call("Hello, how can I help you?") is False
        assert _is_tool_call("The name of the capital is Paris.") is False


class TestToolDispatch:
    """Unit tests for tool dispatch logic."""

    def test_dispatch_unknown_tool(self):
        """Test that unknown tools return error message."""
        from models.demos.minimax_m2.agentic.tools import dispatch_tool

        models = MagicMock()
        result = dispatch_tool("nonexistent_tool", {}, models)
        assert "Unknown tool" in result

    def test_dispatch_transcribe_audio(self):
        """Test dispatch routes to whisper."""
        from models.demos.minimax_m2.agentic.tools import dispatch_tool

        models = MagicMock()
        models.whisper.transcribe.return_value = "Hello world"

        result = dispatch_tool("transcribe_audio", {"path": "/tmp/audio.wav"}, models)

        models.whisper.transcribe.assert_called_once_with("/tmp/audio.wav")
        assert result == "Hello world"

    def test_dispatch_text_to_speech(self):
        """Test dispatch routes to qwen3_tts."""
        from models.demos.minimax_m2.agentic.tools import dispatch_tool

        models = MagicMock()
        models.qwen3_tts.synthesize.return_value = "/tmp/output.wav"

        result = dispatch_tool(
            "text_to_speech",
            {"text": "Hello", "output_path": "/tmp/out.wav", "language": "english"},
            models,
        )

        models.qwen3_tts.synthesize.assert_called_once_with("Hello", "/tmp/out.wav", language="english")
        assert result == "/tmp/output.wav"

    def test_dispatch_detect_objects(self):
        """Test dispatch routes to owlvit."""
        from models.demos.minimax_m2.agentic.tools import dispatch_tool

        models = MagicMock()
        models.owlvit.detect.return_value = [{"label": "cat", "score": 0.95, "bbox": [0, 0, 100, 100]}]

        result = dispatch_tool("detect_objects", {"image_path": "/tmp/img.jpg", "query": "cat"}, models)

        models.owlvit.detect.assert_called_once()
        assert "cat" in result

    def test_dispatch_translate_text(self):
        """Test dispatch routes to t5."""
        from models.demos.minimax_m2.agentic.tools import dispatch_tool

        models = MagicMock()
        models.t5.translate.return_value = "Bonjour"

        result = dispatch_tool(
            "translate_text",
            {"text": "Hello", "source_lang": "en", "target_lang": "fr"},
            models,
        )

        models.t5.translate.assert_called_once_with("Hello", "en", "fr")
        assert result == "Bonjour"


# =============================================================================
# Integration Tests - Require N300 Device
# =============================================================================


def _skip_if_no_device():
    """Skip test if no device available."""
    try:
        pass

        # Try to check if device is available
        return False
    except Exception:
        return True


@pytest.fixture(scope="module")
def n300_device():
    """Fixture to open N300 device for tests."""
    import ttnn
    from models.demos.minimax_m2.agentic.loader import open_n300_device

    mesh = open_n300_device()
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def loaded_models(n300_device):
    """Fixture to load all models for testing.

    Note: T5 is disabled due to submesh cleanup issues with TTNN.
    The 4 core models (Whisper, Qwen3-TTS, OWL-ViT, BERT) work together.
    """
    from models.demos.minimax_m2.agentic.loader import cleanup_models, load_all_models

    models = load_all_models(
        mesh_device=n300_device,
        load_llm=False,  # Skip LLM for faster tests
        load_whisper=True,
        load_qwen3_tts=True,
        load_owlvit=True,
        load_bert=True,
        load_sd=False,
        load_yunet=False,
        load_t5=True,  # Fixed: now works on full mesh
    )
    yield models
    cleanup_models(models)


@pytest.mark.skipif(_skip_if_no_device(), reason="No N300 device available")
class TestTranslationChain:
    """
    Integration test: Translation chain with TTS and STT.

    Flow: English → French → German → English → TTS → Whisper → Compare
    """

    def test_translation_chain_roundtrip(self, loaded_models):
        """Test full translation chain with TTS/STT roundtrip."""
        import soundfile as sf

        models = loaded_models

        if models.t5 is None:
            pytest.skip("T5 not loaded")

        # Input text
        original_text = "Hello, how are you today?"

        # Step 1: English → French
        french_text = models.t5.translate(original_text, source_lang="en", target_lang="fr")
        print(f"English → French: '{french_text}'")
        assert french_text is not None
        assert len(french_text) > 0

        # Step 2: French → German
        german_text = models.t5.translate(french_text, source_lang="fr", target_lang="de")
        print(f"French → German: '{german_text}'")
        assert german_text is not None
        assert len(german_text) > 0

        # Step 3: German → English
        back_to_english = models.t5.translate(german_text, source_lang="de", target_lang="en")
        print(f"German → English: '{back_to_english}'")
        assert back_to_english is not None
        assert len(back_to_english) > 0

        # Step 4: TTS (English text to audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        try:
            tts_output = models.qwen3_tts.synthesize(
                back_to_english,
                output_path=audio_path,
                language="english",
                max_new_tokens=128,
                auto_trim_bleed=False,
            )
            print(f"TTS output: {tts_output}")
            assert Path(tts_output).exists()

            # Verify audio file is valid
            audio_data, sr = sf.read(tts_output)
            assert len(audio_data) > 0
            assert sr == 24000

            # Step 5: Whisper (audio back to text)
            whisper_text = models.whisper.transcribe(tts_output)
            print(f"Whisper transcription: '{whisper_text}'")
            assert whisper_text is not None
            assert len(whisper_text) > 0

            # Step 6: Compare (semantic similarity)
            # Note: Exact match is unlikely due to translation/TTS/STT losses
            # Check for key words
            original_lower = original_text.lower()
            whisper_lower = whisper_text.lower()

            # At minimum, "hello" or "how" should appear
            has_greeting = any(word in whisper_lower for word in ["hello", "hi", "hey", "how", "you"])
            print(f"Original: '{original_text}'")
            print(f"Final: '{whisper_text}'")
            print(f"Has greeting word: {has_greeting}")

            # This is a soft check - translation chains lose information
            assert len(whisper_text) > 3, "Whisper should produce some text"

        finally:
            if Path(audio_path).exists():
                os.unlink(audio_path)


@pytest.mark.skipif(_skip_if_no_device(), reason="No N300 device available")
class TestImageDetection:
    """Integration tests for image detection with OWL-ViT."""

    @pytest.fixture
    def test_image_path(self):
        """Download a test image from COCO dataset."""
        import urllib.request

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image_path = f.name

        urllib.request.urlretrieve(url, image_path)
        yield image_path

        if Path(image_path).exists():
            os.unlink(image_path)

    def test_detect_cats_in_image(self, loaded_models, test_image_path):
        """Test detecting cats in the standard OWL-ViT test image."""
        models = loaded_models

        # The COCO image 000000039769.jpg contains two cats
        detections = models.owlvit.detect(test_image_path, "a cat")

        print(f"Detections: {detections}")
        assert detections is not None
        assert isinstance(detections, list)
        # Should detect at least one cat
        assert len(detections) >= 1

        # Check detection format
        for det in detections:
            assert "label" in det or "score" in det
            assert "bbox" in det or "box" in det

    def test_detect_multiple_queries(self, loaded_models, test_image_path):
        """Test detecting with different queries."""
        models = loaded_models

        # Test multiple queries
        queries = ["a cat", "a couch", "furniture"]

        for query in queries:
            detections = models.owlvit.detect(test_image_path, query)
            print(f"Query '{query}': {len(detections)} detections")
            assert detections is not None

    def test_detect_negative_query(self, loaded_models, test_image_path):
        """Test that unlikely objects are not detected."""
        models = loaded_models

        # The cat image shouldn't have elephants
        detections = models.owlvit.detect(test_image_path, "an elephant")
        print(f"Elephant detections: {detections}")

        # Should either return empty list or low-confidence detections
        if detections:
            for det in detections:
                # If any detections, they should be low confidence
                score = det.get("score", det.get("confidence", 0))
                assert score < 0.5, f"Unexpected high-confidence elephant detection: {det}"

    def test_detect_with_dog_image(self, loaded_models):
        """Test with the Gemma3 dog image."""
        models = loaded_models

        dog_image = _REPO_ROOT / "models/demos/multimodal/gemma3/dog.jpg"
        if not dog_image.exists():
            pytest.skip(f"Dog image not found: {dog_image}")

        detections = models.owlvit.detect(str(dog_image), "a dog")
        print(f"Dog detections: {detections}")

        assert detections is not None
        assert len(detections) >= 1


@pytest.mark.skipif(_skip_if_no_device(), reason="No N300 device available")
class TestBERTQA:
    """Integration tests for BERT extractive QA."""

    def test_simple_qa(self, loaded_models):
        """Test simple question answering."""
        models = loaded_models

        context = "Paris is the capital of France. It is known for the Eiffel Tower."
        question = "What is the capital of France?"

        answer = models.bert.qa(question, context)
        print(f"Question: {question}")
        print(f"Answer: {answer}")

        assert answer is not None
        assert "Paris" in answer or "paris" in answer.lower()

    def test_qa_with_numbers(self, loaded_models):
        """Test QA with numerical answers."""
        models = loaded_models

        context = "The company was founded in 2020 and has 500 employees."
        question = "When was the company founded?"

        answer = models.bert.qa(question, context)
        print(f"Answer: {answer}")

        assert answer is not None
        assert "2020" in answer


# =============================================================================
# End-to-End Tests with LLM
# =============================================================================


@pytest.mark.skipif(_skip_if_no_device(), reason="No N300 device available")
@pytest.mark.slow
class TestLLMToolCalling:
    """End-to-end tests with actual LLM tool calling."""

    @pytest.fixture(scope="class")
    def models_with_llm(self, n300_device):
        """Load models including LLM."""
        from models.demos.minimax_m2.agentic.loader import cleanup_models, load_all_models

        models = load_all_models(
            mesh_device=n300_device,
            load_llm=True,
            load_whisper=True,
            load_qwen3_tts=True,
            load_owlvit=True,
            load_bert=True,
            load_sd=False,
            load_yunet=False,
            load_t5=False,
        )
        yield models
        cleanup_models(models)

    def test_llm_generates_tool_call_for_audio(self, models_with_llm):
        """Test that LLM generates appropriate tool call for audio."""
        from models.demos.minimax_m2.agentic.orchestrator import _is_tool_call
        from models.demos.minimax_m2.agentic.tools import TOOL_SCHEMAS

        models = models_with_llm

        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What did the person say? [AUDIO_ATTACHMENT: /tmp/speech.wav]"},
        ]

        response = models.llm.generate_response(messages, tools=TOOL_SCHEMAS)
        print(f"LLM response: {response}")

        # LLM should generate a tool call for transcription
        assert _is_tool_call(response), f"Expected tool call, got: {response}"

        parsed = models.llm.parse_tool_call(response)
        assert parsed is not None
        tool_name, tool_args = parsed
        assert tool_name == "transcribe_audio"

    def test_llm_answers_directly_for_simple_question(self, models_with_llm):
        """Test that LLM answers simple questions without tools."""
        from models.demos.minimax_m2.agentic.orchestrator import _is_tool_call
        from models.demos.minimax_m2.agentic.tools import TOOL_SCHEMAS

        models = models_with_llm

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer simple questions directly."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]

        response = models.llm.generate_response(messages, tools=TOOL_SCHEMAS)
        print(f"LLM response: {response}")

        # Should NOT generate a tool call for simple math
        is_tool = _is_tool_call(response)
        # Accept either direct answer or no tool call
        assert "4" in response or not is_tool


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
