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
                "Converts an audio file (wav/mp3/flac) to English text using Whisper STT. "
                "English audio only. Use when the user attaches an audio file or asks about spoken content."
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
            "name": "text_to_speech",
            "description": (
                "Converts text to speech using Qwen3-TTS with voice cloning. "
                "Supported languages: english, chinese, japanese, korean, german, french, spanish, italian, portuguese, russian. "
                "High-quality 24kHz audio. Use when the user explicitly wants an audio response."
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
                    "language": {
                        "type": "string",
                        "description": "Language: english, chinese, japanese, korean, german, french, spanish, italian, portuguese, russian (default: english)",
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
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generates an image from a text description using Stable Diffusion. "
                "Use when the user asks to create, draw, or generate an image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output image path (default: /tmp/generated.png)",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_faces",
            "description": (
                "Detects faces in an image and returns bounding boxes with facial keypoints. "
                "Use when the user asks about faces or people in an image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": (
                "Translates text between languages using T5. "
                "Supported languages: en (English), de (German), fr (French), ro (Romanian). "
                "Use when the user asks to translate text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to translate",
                    },
                    "source_lang": {
                        "type": "string",
                        "description": "Source language code: 'en', 'de', 'fr', or 'ro' (default: en)",
                    },
                    "target_lang": {
                        "type": "string",
                        "description": "Target language code: 'en', 'de', 'fr', or 'ro'",
                    },
                },
                "required": ["text", "target_lang"],
            },
        },
    },
    # TrOCR (OCR) - DISABLED: experimental model segfaults on transpose op during generation
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "read_text_from_image",
    #         "description": "Reads and extracts text from an image using OCR (TrOCR).",
    #         "parameters": {"type": "object", "properties": {"image_path": {"type": "string"}}, "required": ["image_path"]},
    #     },
    # },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Searches the knowledge base for relevant information using semantic search. "
                "Use when the user asks a question that might be answered by documents in the knowledge base. "
                "Returns relevant context passages from uploaded documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant passages to return (default: 3)",
                    },
                },
                "required": ["query"],
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

    elif name == "text_to_speech":
        output_path = args.get("output_path", "/tmp/response.wav")
        # Use SpeechT5 if available (fast, English only), else Qwen3-TTS (slow, multi-language)
        if models.speecht5 is not None:
            return models.speecht5.synthesize(args["text"], output_path)
        elif models.qwen3_tts is not None:
            language = args.get("language", "english")
            return models.qwen3_tts.synthesize(args["text"], output_path, language=language)
        else:
            return "Error: No TTS model loaded"

    elif name == "detect_objects":
        detections = models.owlvit.detect(args["image_path"], args["query"])
        return _format_detections(detections)

    elif name == "answer_from_context":
        return models.bert.qa(args["question"], args["context"])

    elif name == "generate_image":
        output_path = args.get("output_path", "/tmp/generated.png")
        return models.sd.generate(args["prompt"], output_path)

    elif name == "detect_faces":
        # Generate output path for annotated image
        import uuid

        output_path = f"/tmp/web_demo_outputs/faces_{uuid.uuid4().hex[:8]}.jpg"
        detections = models.yunet.detect(args["image_path"], output_image_path=output_path)
        return _format_face_detections(detections)

    elif name == "translate_text":
        source_lang = args.get("source_lang", "en")
        target_lang = args.get("target_lang", "de")
        return models.t5.translate(args["text"], source_lang, target_lang)

    elif name == "search_knowledge_base":
        if models.rag is None:
            return "Error: RAG/knowledge base not loaded"
        top_k = args.get("top_k", 3)
        result = models.rag.query_with_context(args["query"], top_k=top_k)
        return _format_rag_result(result)

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


def _format_face_detections(detections: List[Dict]) -> str:
    if not detections:
        return "No faces detected."

    # Check if we have actual face detections or just the output image
    real_detections = [d for d in detections if d.get("box") is not None]
    output_image = detections[0].get("output_image") if detections else None

    if not real_detections:
        if output_image:
            return f"No faces detected.\nAnnotated image saved to: {output_image}"
        return "No faces detected."

    lines = [f"Found {len(real_detections)} face(s):"]
    for i, d in enumerate(real_detections, 1):
        box = d.get("box", (0, 0, 0, 0))
        conf = d.get("confidence", 0)
        lines.append(f"- Face {i}: bbox=({box[0]}, {box[1]}, {box[2]}, {box[3]}), confidence={conf:.3f}")

    if output_image:
        lines.append(f"\nAnnotated image saved to: {output_image}")

    return "\n".join(lines)


def _format_rag_result(result: Dict) -> str:
    """Format RAG search result for LLM consumption."""
    if result["num_results"] == 0:
        return "No relevant documents found in the knowledge base."

    lines = [
        f"Found {result['num_results']} relevant passage(s) from: {', '.join(result['sources'])}",
        "",
        "=== CONTEXT ===",
        result["context"],
        "=== END CONTEXT ===",
    ]
    return "\n".join(lines)
