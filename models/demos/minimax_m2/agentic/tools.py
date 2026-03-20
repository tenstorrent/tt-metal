# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tool schemas (for apply_chat_template) and the central dispatcher.

TOOL_SCHEMAS follows the OpenAI/Llama-3 tool-definition format.
dispatch_tool() routes tool names to the appropriate model wrapper.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Tool schemas (passed to tokenizer.apply_chat_template(tools=TOOL_SCHEMAS))
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": (
                "Converts an audio file (wav/mp3/flac) to text using Whisper STT. "
                "Use when the user attaches an audio file or asks about spoken content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_audio",
            "description": (
                "Transcribes and translates audio to English. " "Use when the user attaches non-English audio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file",
                    },
                    "source_language": {
                        "type": "string",
                        "description": "ISO 639-1 language code, e.g. 'fr', 'de'",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": (
                "Converts text to speech and saves as a wav file. "
                "Use when the user explicitly wants an audio response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to synthesize",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output .wav file path",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_objects",
            "description": (
                "Detects objects in an image based on a text query. "
                "Use when the user attaches an image and asks what is in it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "query": {
                        "type": "string",
                        "description": "What to look for, e.g. 'a person', 'cars'",
                    },
                },
                "required": ["image_path", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_from_context",
            "description": (
                "Extracts an answer from a provided text passage. "
                "Use when the user asks a specific question about a document or long text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                    },
                    "context": {
                        "type": "string",
                        "description": "The passage to search for the answer",
                    },
                },
                "required": ["question", "context"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def dispatch_tool(name: str, args: Dict[str, Any], models) -> Any:
    """
    Route a tool call to the appropriate model wrapper.

    Args:
        name:   Tool name (must match a key in TOOL_SCHEMAS).
        args:   Parsed arguments dict from the LLM output.
        models: Loaded models bundle (see loader.py).

    Returns:
        Tool result (str or list).  Always serialisable to str.
    """
    if name == "transcribe_audio":
        return models.whisper.transcribe(args["path"])

    elif name == "translate_audio":
        return models.whisper.translate(args["path"])

    elif name == "text_to_speech":
        output_path = args.get("output_path", "/tmp/response.wav")
        return models.speecht5.synthesize(args["text"], output_path)

    elif name == "detect_objects":
        detections = models.owlvit.detect(args["image_path"], args["query"])
        return _format_detections(detections)

    elif name == "answer_from_context":
        return models.bert.qa(args["question"], args["context"])

    else:
        return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_detections(detections: List[Dict]) -> str:
    if not detections:
        return "No objects detected above threshold."
    lines = []
    for d in detections:
        bbox = d.get("bbox", [])
        bbox_str = ", ".join(f"{v:.3f}" for v in bbox) if bbox else "unknown"
        lines.append(f"- {d['label']} (score={d['score']:.3f}, bbox=[{bbox_str}])")
    return "\n".join(lines)
