# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
pytest tests for the General Agentic Mode.

Tests are structured to be fast and not require the full model stack.
Each test verifies:
  1. The component loads correctly (or gracefully skips if unavailable).
  2. The output format is correct.
  3. The agentic loop routes tool calls properly.

Run with:
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    source python_env/bin/activate

    # All tests (requires N300 device)
    pytest models/demos/minimax_m2/agentic/tests/test_agentic.py -v

    # Offline tests only (no device, no model downloads)
    pytest models/demos/minimax_m2/agentic/tests/test_agentic.py -v -m offline
"""

from unittest.mock import MagicMock

import pytest

from models.demos.minimax_m2.agentic.orchestrator import (
    SYSTEM_PROMPT,
    _build_user_message,
    _classify_attachment,
    _is_tool_call,
    process_single_query,
)
from models.demos.minimax_m2.agentic.tools import TOOL_SCHEMAS, _format_detections, dispatch_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_models(**kwargs):
    """Return a mock ModelBundle with overrideable attributes."""
    bundle = MagicMock()
    for k, v in kwargs.items():
        setattr(bundle, k, v)
    return bundle


# ---------------------------------------------------------------------------
# Tool schema tests (offline)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_tool_schemas_structure():
    """All TOOL_SCHEMAS entries must have 'type' and 'function' keys."""
    assert len(TOOL_SCHEMAS) == 5, f"Expected 5 tools, got {len(TOOL_SCHEMAS)}"
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        params = fn["parameters"]
        assert "type" in params
        assert "properties" in params


@pytest.mark.offline
def test_tool_schema_names():
    """Tool names must match those expected by dispatch_tool."""
    expected = {
        "transcribe_audio",
        "translate_audio",
        "text_to_speech",
        "detect_objects",
        "answer_from_context",
    }
    actual = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert actual == expected


# ---------------------------------------------------------------------------
# Attachment detection tests (offline)
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.parametrize(
    "path,expected",
    [
        ("/tmp/audio.wav", "audio"),
        ("/tmp/clip.mp3", "audio"),
        ("/tmp/recording.flac", "audio"),
        ("/tmp/photo.jpg", "image"),
        ("/tmp/image.png", "image"),
        ("/tmp/document.pdf", "unknown"),
        ("/tmp/data.csv", "unknown"),
    ],
)
def test_classify_attachment(path, expected):
    assert _classify_attachment(path) == expected


@pytest.mark.offline
def test_build_user_message_no_attachments():
    msg = _build_user_message("Hello!", [])
    assert msg == "Hello!"


@pytest.mark.offline
def test_build_user_message_with_audio():
    msg = _build_user_message("What did I say?", ["/tmp/voice.wav"])
    assert "[AUDIO_ATTACHMENT: /tmp/voice.wav]" in msg
    assert "What did I say?" in msg


@pytest.mark.offline
def test_build_user_message_with_image():
    msg = _build_user_message("What is this?", ["/tmp/photo.jpg"])
    assert "[IMAGE_ATTACHMENT: /tmp/photo.jpg]" in msg


@pytest.mark.offline
def test_build_user_message_multiple_attachments():
    msg = _build_user_message("Analyze these", ["/tmp/a.wav", "/tmp/b.png"])
    assert "[AUDIO_ATTACHMENT: /tmp/a.wav]" in msg
    assert "[IMAGE_ATTACHMENT: /tmp/b.png]" in msg


# ---------------------------------------------------------------------------
# Tool call detection tests (offline)
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.parametrize(
    "text,expected",
    [
        # python_tag format
        ('<|python_tag|>{"name": "transcribe_audio", "parameters": {"path": "/tmp/a.wav"}}', True),
        # JSON code block
        (
            'Sure!\n```json\n{"name": "detect_objects", "arguments": {"image_path": "/tmp/x.jpg", "query": "cat"}}\n```',
            True,
        ),
        # Bare JSON
        ('{"name": "answer_from_context", "arguments": {"question": "What?", "context": "ctx"}}', True),
        # Plain text — no tool call
        ("The capital of France is Paris.", False),
        ("I can help with that!", False),
    ],
)
def test_is_tool_call(text, expected):
    assert _is_tool_call(text) == expected


# ---------------------------------------------------------------------------
# Tool call parsing tests (offline)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_parse_tool_call_python_tag():
    from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

    text = '<|python_tag|>{"name": "transcribe_audio", "parameters": {"path": "/tmp/voice.wav"}}'
    result = _parse_tool_call(text)
    assert result is not None
    name, args = result
    assert name == "transcribe_audio"
    assert args["path"] == "/tmp/voice.wav"


@pytest.mark.offline
def test_parse_tool_call_json_block():
    from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

    text = '```json\n{"name": "answer_from_context", "arguments": {"question": "Who?", "context": "Alice."}}\n```'
    result = _parse_tool_call(text)
    assert result is not None
    name, args = result
    assert name == "answer_from_context"
    assert args["question"] == "Who?"


@pytest.mark.offline
def test_parse_tool_call_no_match():
    from models.demos.minimax_m2.agentic.tool_wrappers.llm_tool import _parse_tool_call

    result = _parse_tool_call("The answer is 42.")
    assert result is None


# ---------------------------------------------------------------------------
# dispatch_tool tests with mocked models (offline)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_dispatch_transcribe_audio():
    whisper = MagicMock()
    whisper.transcribe.return_value = "Hello world"
    models = _make_mock_models(whisper=whisper)
    result = dispatch_tool("transcribe_audio", {"path": "/tmp/a.wav"}, models)
    assert result == "Hello world"
    whisper.transcribe.assert_called_once_with("/tmp/a.wav")


@pytest.mark.offline
def test_dispatch_translate_audio():
    whisper = MagicMock()
    whisper.translate.return_value = "Bonjour le monde"
    models = _make_mock_models(whisper=whisper)
    result = dispatch_tool("translate_audio", {"path": "/tmp/fr.wav", "source_language": "fr"}, models)
    assert result == "Bonjour le monde"
    whisper.translate.assert_called_once_with("/tmp/fr.wav")


@pytest.mark.offline
def test_dispatch_text_to_speech():
    speecht5 = MagicMock()
    speecht5.synthesize.return_value = "/tmp/out.wav"
    models = _make_mock_models(speecht5=speecht5)
    result = dispatch_tool("text_to_speech", {"text": "Hello!", "output_path": "/tmp/out.wav"}, models)
    assert result == "/tmp/out.wav"
    speecht5.synthesize.assert_called_once_with("Hello!", "/tmp/out.wav")


@pytest.mark.offline
def test_dispatch_text_to_speech_default_path():
    speecht5 = MagicMock()
    speecht5.synthesize.return_value = "/tmp/response.wav"
    models = _make_mock_models(speecht5=speecht5)
    dispatch_tool("text_to_speech", {"text": "Hi"}, models)
    speecht5.synthesize.assert_called_once_with("Hi", "/tmp/response.wav")


@pytest.mark.offline
def test_dispatch_detect_objects():
    owlvit = MagicMock()
    owlvit.detect.return_value = [
        {"label": "cat", "score": 0.91, "bbox": [0.1, 0.2, 0.5, 0.8]},
    ]
    models = _make_mock_models(owlvit=owlvit)
    result = dispatch_tool("detect_objects", {"image_path": "/tmp/x.jpg", "query": "cat"}, models)
    assert "cat" in result
    assert "0.910" in result


@pytest.mark.offline
def test_dispatch_answer_from_context():
    bert = MagicMock()
    bert.qa.return_value = "Paris"
    models = _make_mock_models(bert=bert)
    result = dispatch_tool(
        "answer_from_context",
        {"question": "Capital?", "context": "France's capital is Paris."},
        models,
    )
    assert result == "Paris"


@pytest.mark.offline
def test_dispatch_unknown_tool():
    models = _make_mock_models()
    result = dispatch_tool("nonexistent_tool", {}, models)
    assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# Format helpers tests (offline)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_format_detections_empty():
    result = _format_detections([])
    assert "No objects" in result


@pytest.mark.offline
def test_format_detections_with_items():
    detections = [
        {"label": "dog", "score": 0.95, "bbox": [0.1, 0.2, 0.3, 0.4]},
        {"label": "cat", "score": 0.87, "bbox": [0.5, 0.6, 0.7, 0.8]},
    ]
    result = _format_detections(detections)
    assert "dog" in result
    assert "0.950" in result
    assert "cat" in result


# ---------------------------------------------------------------------------
# Agentic loop tests with mocked LLM (offline)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_run_one_turn_no_tool_call():
    """When LLM returns plain text, run_one_turn returns it directly."""
    llm = MagicMock()
    llm.generate_response.return_value = "The answer is 42."
    llm.parse_tool_call.return_value = None
    models = _make_mock_models(llm=llm)

    messages = [{"role": "user", "content": "What is 6*7?"}]
    from models.demos.minimax_m2.agentic.orchestrator import run_one_turn

    response = run_one_turn(messages, models)
    assert response == "The answer is 42."


@pytest.mark.offline
def test_run_one_turn_single_tool_call():
    """Verify the tool-call → result → final-answer flow."""
    llm = MagicMock()
    tool_json = '<|python_tag|>{"name": "transcribe_audio", "parameters": {"path": "/tmp/a.wav"}}'
    llm.generate_response.side_effect = [
        tool_json,
        "You said: Hello world.",
    ]
    llm.parse_tool_call.return_value = ("transcribe_audio", {"path": "/tmp/a.wav"})

    whisper = MagicMock()
    whisper.transcribe.return_value = "Hello world"

    models = _make_mock_models(llm=llm, whisper=whisper)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What did I say? [AUDIO_ATTACHMENT: /tmp/a.wav]"},
    ]
    from models.demos.minimax_m2.agentic.orchestrator import run_one_turn

    response = run_one_turn(messages, models)

    assert "Hello world" in response or response == "You said: Hello world."
    whisper.transcribe.assert_called_once_with("/tmp/a.wav")


@pytest.mark.offline
def test_run_one_turn_tool_error_handled():
    """Tool errors are caught and returned as an error message."""
    llm = MagicMock()
    tool_json = '<|python_tag|>{"name": "answer_from_context", "parameters": {"question": "?", "context": ""}}'
    llm.generate_response.side_effect = [
        tool_json,
        "I could not find the answer.",
    ]
    llm.parse_tool_call.return_value = ("answer_from_context", {"question": "?", "context": ""})

    bert = MagicMock()
    bert.qa.side_effect = ValueError("Empty context")

    models = _make_mock_models(llm=llm, bert=bert)
    messages = [{"role": "user", "content": "Answer from empty context"}]

    from models.demos.minimax_m2.agentic.orchestrator import run_one_turn

    response = run_one_turn(messages, models)
    # Should not raise; should return some response
    assert isinstance(response, str)


@pytest.mark.offline
def test_process_single_query():
    """process_single_query builds conversation history correctly."""
    llm = MagicMock()
    llm.generate_response.return_value = "Paris is the capital of France."
    llm.parse_tool_call.return_value = None
    models = _make_mock_models(llm=llm)

    response, history = process_single_query("What is the capital of France?", models)
    assert response == "Paris is the capital of France."
    # history: system + user + assistant
    assert len(history) == 3
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == response
