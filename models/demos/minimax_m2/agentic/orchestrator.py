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
You are a helpful AI assistant running on Tenstorrent N300 hardware.
You have access to specialist AI tools for audio, vision, and text processing.

ATTACHMENT HANDLING RULES:
- If you see [AUDIO_ATTACHMENT: path], you MUST call transcribe_audio(path) before answering.
- If you see [IMAGE_ATTACHMENT: path] and the user asks what's in it, call detect_objects.
- If the user wants an audio response, call text_to_speech with your final answer text.
- If the user provides a long document and asks a specific question, call answer_from_context.

Always explain what you are doing before calling a tool. After receiving tool results,
use them to construct your final answer — do not call the same tool twice for the same input.
"""

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
_MAX_TOOL_TURNS = 10  # prevent infinite tool-call loops


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
