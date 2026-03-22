# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
orchestrator.py — agentic conversation loop for the General Agentic Mode.

Architecture:

  User input (text / audio path / image path)
       │
       ▼
  ┌──────────────────────────────────────────┐
  │  ORCHESTRATOR                            │
  │  1. Detect attachment type               │
  │  2. Inject [AUDIO/IMAGE_ATTACHMENT] tag  │
  │  3. Build messages list                  │
  └─────────────────┬────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │  LLM (Llama 3B)    │  ◄── TOOL_SCHEMAS via apply_chat_template
          └─────────┬──────────┘
                    │ emits tool_call JSON  OR  final text
                    │
       ┌────────────▼───────────────────┐
       │  TOOL DISPATCHER               │
       │  (see tools.py)                │
       └────────────┬───────────────────┘
                    │ tool result → messages.append → loop back to LLM
                    │
          ┌─────────▼────────────────┐
          │  Deliver final response   │
          └───────────────────────────┘
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from models.demos.minimax_m2.agentic.tools import TOOL_SCHEMAS, dispatch_tool

SYSTEM_PROMPT = """\
You are a helpful AI assistant on Tenstorrent N300 with access to specialized AI tools.

AVAILABLE TOOLS:
- transcribe_audio(path): Convert audio to text using Whisper STT (English only)
- text_to_speech(text, output_path, language): Convert text to audio using Qwen3-TTS
  Supported languages: english, chinese, japanese, korean, german, french, spanish, italian, portuguese, russian
- detect_objects(image_path, query): Find objects in image - cats, dogs, cars, people, etc. (OWL-ViT)
- detect_faces(image_path): Count human faces and draw boxes around them (YUNet)
- answer_from_context(question, context): Extract answer from text passage (BERT QA)
- translate_text(text, source_lang, target_lang): Translate text between languages (T5)
  Supported languages: en (English), de (German), fr (French), ro (Romanian)
- generate_image(prompt, output_path): Generate an image from text description (Stable Diffusion)

CRITICAL RULES FOR IMAGE ATTACHMENTS:

When you see [IMAGE_ATTACHMENT: path], check if the user mentions "face" or "faces":
- If YES (user says "face", "faces", "how many people", "count people") → detect_faces(path)
- If NO (anything else: "what's in this", "describe", "what do you see") → detect_objects(path, query)

The DEFAULT is ALWAYS detect_objects. Only use detect_faces when user specifically mentions faces.

RULES FOR IMAGE GENERATION:
- "draw", "create", "generate an image of", "make a picture" → generate_image(prompt)
- This is for creating NEW images, not analyzing existing ones

RULES FOR AUDIO [AUDIO_ATTACHMENT: path]:
- FIRST: Always call transcribe_audio to get the transcript
- THEN: Based on user request, translate or speak the result

RULES FOR TEXT-ONLY (no attachments):
- Math, facts, general knowledge → answer directly
- "say this aloud" → call text_to_speech
- Context paragraph + question → call answer_from_context
- "translate X to German/French" → call translate_text

AFTER receiving tool results, provide your FINAL ANSWER as plain text.

Tool call format: {"name": "tool_name", "parameters": {...}}
"""

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
_MAX_TOOL_TURNS = 3  # prevent infinite tool-call loops


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _classify_attachment(path: str) -> str:
    """Return "audio", "image", or "unknown"."""
    ext = Path(path).suffix.lower()
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _IMAGE_EXTS:
        return "image"
    return "unknown"


def _build_user_message(user_text: str, attachments: List[str]) -> str:
    """Append attachment tags to the user text."""
    content = user_text
    for path in attachments:
        kind = _classify_attachment(path)
        if kind == "audio":
            content += f"\n[AUDIO_ATTACHMENT: {path}]"
        elif kind == "image":
            content += f"\n[IMAGE_ATTACHMENT: {path}]"
        else:
            content += f"\n[ATTACHMENT: {path}]"
    return content


def _is_tool_call(text: str) -> bool:
    """Return True if the LLM output contains a tool call marker."""
    return (
        "<|python_tag|>" in text
        or ("```" in text and ('"name"' in text or "'name'" in text))
        or ('"name"' in text and ('"arguments"' in text or '"parameters"' in text))
    )


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------


def run_agentic_loop(models, device=None) -> None:
    """
    Run an interactive CLI agentic loop.

    Reads user input from stdin; attachments can be given as space-separated
    file paths following the text on the same line, e.g.:

        What did I say? /tmp/voice.wav
        Describe this image. /home/user/photo.jpg

    Type "exit" or "quit" to stop.
    """
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n" + "=" * 70)
    print("  Tenstorrent N300 Agentic Assistant")
    print("  Type your message (optionally append file paths)")
    print("  Type 'exit' to quit")
    print("=" * 70 + "\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            print("Bye!")
            break

        # Split off any file paths from the end of the input
        parts = raw.split()
        attachments = [p for p in parts if Path(p).suffix.lower() in (_AUDIO_EXTS | _IMAGE_EXTS)]
        user_text_parts = [p for p in parts if p not in attachments]
        user_text = " ".join(user_text_parts) if user_text_parts else raw

        user_content = _build_user_message(user_text, attachments)
        messages.append({"role": "user", "content": user_content})

        response = run_one_turn(messages, models)
        messages.append({"role": "assistant", "content": response})
        print(f"\nAssistant: {response}\n")


def run_one_turn(
    messages: List[Dict],
    models,
    max_tool_turns: int = _MAX_TOOL_TURNS,
) -> str:
    """
    Run one agentic turn: call the LLM, handle tool calls, return final answer.

    This function mutates *messages* by appending assistant + tool results,
    but does NOT append the final assistant response — the caller does that.

    Args:
        messages:       Conversation history so far.
        models:         Loaded ModelBundle.
        max_tool_turns: Maximum number of tool calls before forcing a final answer.

    Returns:
        Final assistant response text.
    """
    called_tools = set()  # Track which tools have been called to prevent loops

    for turn in range(max_tool_turns):
        output_text = models.llm.generate_response(
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        logger.debug(f"LLM output (turn {turn + 1}): {output_text[:200]}...")

        if _is_tool_call(output_text):
            parsed = models.llm.parse_tool_call(output_text)
            if parsed is None:
                logger.warning("Could not parse tool call; treating as final answer.")
                return output_text

            tool_name, tool_args = parsed
            logger.info(f"Tool call: {tool_name}({json.dumps(tool_args)})")

            # Prevent calling the same tool twice
            if tool_name in called_tools:
                logger.warning(f"Tool {tool_name} already called; forcing final answer.")
                return f"Based on the previous tool results, here is my answer: {output_text}"
            called_tools.add(tool_name)

            messages.append({"role": "assistant", "content": output_text})

            try:
                tool_result = dispatch_tool(tool_name, tool_args, models)
            except Exception as exc:
                tool_result = f"Tool error: {exc}"
                logger.error(f"Tool {tool_name} raised: {exc}")

            result_str = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": result_str,
                }
            )
            logger.info(f"Tool result: {result_str[:200]}")

        else:
            return output_text

    # Exceeded max tool turns — return last LLM output
    logger.warning(f"Exceeded max tool turns ({max_tool_turns}); returning last output.")
    return output_text


def process_single_query(
    query: str,
    models,
    attachments: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict]] = None,
) -> Tuple[str, List[Dict]]:
    """
    Process a single query and return the response + updated history.

    Convenience function for programmatic use (tests, API integration).

    Args:
        query:                Text of the user's question.
        models:               Loaded ModelBundle.
        attachments:          Optional list of file paths.
        conversation_history: Existing message history (modified in-place).

    Returns:
        (response_text, updated_messages)
    """
    if conversation_history is None:
        conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_content = _build_user_message(query, attachments or [])
    conversation_history.append({"role": "user", "content": user_content})

    response = run_one_turn(conversation_history, models)
    conversation_history.append({"role": "assistant", "content": response})

    return response, conversation_history
