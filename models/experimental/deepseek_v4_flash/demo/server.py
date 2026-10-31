# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP server on the full ttnn ``DeepSeekV4Model``.

Serves the same decode engine as ``demo/chat_cli.py`` (whose ``ChatEngine`` is
reused unmodified) over HTTP in the shape of the OpenAI API, so existing SDKs,
``curl`` and notebook clients can talk to the model without touching tt-metal
themselves:

* ``GET    /v1/models``           -- the served model
* ``POST   /v1/chat/completions`` -- chat completions (``stream=true`` for SSE)
* ``POST   /v1/completions``      -- legacy text completions
* ``POST   /v1/chat/completions/<id>/cancel`` -- stop one completion (extension; also ``/v1/completions/<id>/cancel``)
* ``GET    /v1/sessions``         -- active KV-cache sessions / usage (extension)
* ``DELETE /v1/sessions/<user>``  -- reset one session (extension)
* ``POST   /v1/sessions/<user>/cancel`` -- stop every completion of ``user`` (extension)
* ``GET    /health``              -- liveness

**Decode over the device sockets.** The model is loaded once and kept resident;
every turn is a handful of traced decode steps. Each step pushes the token to the
model through the **host->device socket** (``DeepSeekV4Model.write_step_packet``
-- the only host->device traffic of a traced step) and receives the logits back
off the **device->host socket** (``DeepSeekV4Model.read_decoded_output``).

**Prefill.** A prompt that starts a session (a first turn, or a re-prefill from the
request) is prefilled the way ``tests/prefill/test_prefill_decode_demo.py`` does it: on
a 32-chip Galaxy the decode model and a :class:`DeepSeekV4PrefillModel` share eight
``1 x 4`` TP4 stages (one per mesh row), the prompt's 128-aligned prefix (up to
``--prefill-max-len``, by default the whole context) runs through the traced, pipelined
prefill (one trace per stage and position tier, ``--prefill-trace-chunk`` tokens per chunk,
the stages on consecutive chunks at once; the sliding layers run mask-free),
its attention state is committed into the session's decode buffers on device, and the
ragged tail is replayed through decode. A follow-up turn feeds only the tokens it adds
through decode, unless rewinding and prefilling the whole conversation is estimated to
be faster (a long new turn); short prompts skip the prefill model for the same reason.
Decode keeps the DRISC prefetcher on this path: a prefill replay overwrites decode's L1
tensors, which are snapshot and written back, and the GCB ring data, which is empty
between decode steps; the GCB config pages sit in L1_SMALL, out of prefill's reach.
``--no-prefill`` replays every prompt token through decode instead, on the profile's
own decode layout.

**Users & KV cache.** ``--num-users`` paged KV-cache sessions (8 by default) are claimed
at startup -- one captured trace, one block pool holding ``--total-context`` tokens
across everyone -- and then handed out a turn at a time: ``--num-users`` is the number of
slot ids, i.e. conversations kept warm on the device. Raising it costs little memory beyond
the shared budget (a session's sliding ring is bounded by ``sliding_window``). Of those, at
most ``--max-running`` (8 by default) turns generate at once; a request beyond that keeps its
slot and waits in a first-come-first-served run queue. The users' steps are pipelined through
the stages rather than batched, so up to one user per stage adds throughput at little cost to
each reply; past that the stages are full and the users share their rate. A slot is *running*,
*queued*, *parked* (the turn is done, its KV stays in the slot's device DRAM) or *empty*; a
conversation can also be in *host* RAM (``--host-users`` of them in at most ``--host-cache-gb``),
copied there when its slot is emptied. A request is given the slot or host copy whose stored
conversation its ``messages`` continue -- then only the new turn's tokens are fed, after a host
copy is written back into a slot -- falling back to an empty slot. A parked slot keeps its
conversation for as long as it can: only when no slot is empty is the one idle longest taken
for a new request, and only once it has been idle ``--slot-idle-timeout`` seconds (300 by
default) since its last request; its conversation then moves to host RAM. Otherwise the request
gets a 429. Clients that send one ``user`` identifier for all their traffic, or none, still run
in parallel.

**Concurrency.** Up to ``--max-running`` turns generate at once, their steps in the
stage pipeline together, as ``tests/decode/test_multi_user_paged_decode_demo.py`` runs
its users: every decoding turn has one step in flight, and its next step is seated,
posted and packed as soon as the previous one's token is back, so with as many users as
stages each stage works on a different user's step. Prompt tokens replayed through decode
keep up to ``--prefill-chunk`` steps in flight per turn. A turn decoding alone has its next step's
traces posted one step early, so their dispatch overlaps the step before rather than idling the device
(``--no-decode-ahead`` turns it off), and the reader thread samples its token and writes that step's packet
itself the moment the output is off the socket, so the scheduler is off the token-to-packet path
(``_OutputReader.arm``). Outputs are read off the socket
by a reader thread, in dispatch order. A traced prefill waits for every step in flight
to come back and then holds the device for that prompt. One scheduler thread does all
the dispatch, which is required, not incidental: the trace replays and the paged session
state have to be driven from one thread, and each step's output must be matched in
dispatch order. HTTP threads never touch the device -- they submit a turn and relay its
reply -- so total throughput scales with the users while each reply streams
independently. A request that arrives with every slot busy waits for one to free.

**Stopping.** A turn stops when its client disconnects (checked every ``_CLIENT_POLL_SECONDS`` while the
request waits on its reply, streamed or not) or on a cancel request, which names the completion's ``id`` (the
first streamed chunk and the ``X-Request-Id`` header carry it) or the ``user``. A decoding turn ends once its
step in flight is back, keeping the reply so far. A turn still on its prompt ends once its replayed steps are
back, or, in the traced prefill, its chunks in flight; its reply is empty. The session's cache then ends
part-way through that prompt, so the next request on the slot resumes from an earlier turn or re-prefills.

**Live console.** On a terminal the server runs a split view (``demo/tui.py``): a status
header with uptime, active turns, throughput and the KV block pool, a row per cache slot
showing who holds it and how fast it is decoding, and under it the log with the time in a
left column and the message on the right. The mouse wheel, the arrow keys and
page-up/page-down scroll that log (home for the oldest line, end to follow the newest
again); shift-drag still selects text. The live view owns the alternate screen, so the
terminal's own scrollback cannot reach it. ``d`` toggles the
debug lines (each request's sender, options and messages, admissions, slot assignments,
page allocations, prefill lengths, per-user tok/s), ``v`` switches those logged prompts
between the full text and a 200-character snippet (``--prompt-snippet`` starts on the
snippet), ``p`` pauses the scroll,
``c`` clears it, ``k`` stops every running turn (see "Stopping") and ``q`` quits. ``--debug`` starts with those lines on and ``--no-tui``
(or a redirected stdout) falls back to plain logging.

**OpenAI compatibility notes.**

* ``messages`` are OpenAI-shaped ``{role, content}`` dicts, where ``content`` is
  either a string or an array of ``{"type": "text", "text": ...}`` parts (this model
  is text-only, so an image or audio part is refused). If the incoming
  history is a strict extension of the session's stored conversation only the new
  user turn is fed; anything else (a first turn, a foreign client, an edited
  history) rewinds the session and re-prefills from the request, so a client that
  keeps its own full history always stays consistent with the server.
* ``max_tokens`` caps the reply (default: the engine-wide ``--max-new-tokens``);
  ``temperature`` / ``top_p`` sample from the per-step logits (default: the
  server's ``--temperature``, unset = greedy argmax). ``thinking`` and ``reasoning_effort`` are DeepSeek extensions
  that switch that session to the thinking template. ``thinking`` is ``true`` / ``false`` or
  ``{"type": "enabled"}`` / ``{"type": "disabled"}``. ``reasoning_effort`` is
  ``low``, ``medium``, ``high``, or ``max`` (``medium`` uses the same template
  as ``high``). In thinking mode the reply
  carries the reasoning block in the ``reasoning_content`` field, exactly like the
  DeepSeek reasoner API, while ``content`` holds the answer. The prompt already
  opens ``<think>``, so a completion is split on ``</think>`` even when the
  generated text never repeats the opening tag. ``stop`` sequences
  are not supported.
* Tool calling works the OpenAI way round-trip. ``tools`` on the request are rendered
  into the system turn so the model knows what it may call; when it calls one it writes
  a DSML block (DeepSeek Markup Language, the XML-ish ``<｜DSML｜tool_calls>`` form its
  own encoder defines), which is markup for a tool runner rather than prose and so never
  appears in ``content``: it comes back as OpenAI ``tool_calls`` with ``finish_reason``
  ``tool_calls``, streamed as one delta when the block closes. Results are sent back as
  ordinary ``role: "tool"`` messages; the model has no such role, so they are folded into
  the next user turn as ``<tool_result>`` blocks the way the checkpoint's encoder does,
  ordered by the calls that asked for them. The assistant's calls are kept in the
  session, so the whole loop continues on the warm cache instead of re-prefilling.
* Errors use the OpenAI envelope ``{"error": {"message", "type", "code"}}``; for
  ``stream=true`` the error arrives as an SSE event before ``[DONE]``.

Run it (ttnn venv, from the repo root; 32-chip Galaxy for the prefill model)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    python models/experimental/deepseek_v4_flash/demo/server.py \\
        --num-users 8 --max-context 131072 --host 0.0.0.0 --port 8000 --debug

Then, from any other shell (see ``demo/client.py`` for the full client)::

    curl -s --no-buffer -X POST http://127.0.0.1:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import socket
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.demo import tui
from models.experimental.deepseek_v4_flash.demo.chat_cli import (
    ChatEngine,
    ChatFormat,
    ContextFull,
    UserSession,
    decoded_token,
    open_mesh_device,
)
from models.experimental.deepseek_v4_flash.encoding_dsv4 import REASONING_EFFORTS
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _TP_SIZE,
)
from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import PagedCacheFull
from models.experimental.deepseek_v4_flash.tt.model import prefill_bias_slots
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT
from models.experimental.deepseek_v4_flash.tt.prefill.weights import checkpoint_weights
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config

_VENDOR = "tenstorrent"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_DEFAULT_USER = "default"

# DeepSeek-V4 emits tool calls as DSML (DeepSeek Markup Language), an XML-like block the
# checkpoint's own encoder defines in ``encoding/encoding_dsv4.py``. Note the delimiter is
# U+FF5C FULLWIDTH VERTICAL LINE, as in the model's other special tokens, not an ASCII pipe::
#
#     <｜DSML｜tool_calls>
#     <｜DSML｜invoke name="get_weather">
#     <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
#     <｜DSML｜parameter name="days" string="false">5</｜DSML｜parameter>
#     </｜DSML｜invoke>
#     </｜DSML｜tool_calls>
#
# ``string="true"`` marks a raw string; ``string="false"`` marks a JSON value (number,
# boolean, array, object). The block follows the answer text, separated by a blank line.
# The tag names are the engine's :class:`ChatFormat`'s: V4.1 writes ``<｜DSML｜ calls>`` etc.


class RequestError(Exception):
    """A client-side error mapped onto an OpenAI error envelope."""

    def __init__(self, status: int, message: str, err_type: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type or ("invalid_request_error" if status < 500 else "server_error")


def _thinking_requested(value) -> bool:
    """Whether a request's ``thinking`` field turns the thinking template on.

    A boolean is the server's own form. DeepSeek and DeepSeek Harness send
    ``{"type": "enabled"}`` or ``{"type": "disabled"}``; both objects are truthy,
    so the ``type`` field is what decides.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and value.get("type") in ("enabled", "disabled"):
        return value["type"] == "enabled"
    raise RequestError(400, 'thinking must be a boolean or {"type": "enabled"|"disabled"}')


class _ClientGone(Exception):
    """The SSE client hung up mid-stream; stop generating and drop the reply."""


class _PromptStopped(Exception):
    """A stopped turn's traced prefill, raised between its chunks."""


# How often a request waiting on its reply checks that its client is still connected.
_CLIENT_POLL_SECONDS = 0.5
# sys.setswitchinterval while serving. A thread waiting for the GIL gets it only once the holder has run this
# long (default 5 ms), and the output reader takes it back after every socket read and torch op on a token's
# way to the next packet: an HTTP, console or scheduler thread busy meanwhile would add up to that per token.
_GIL_SWITCH_SECONDS = 0.0002


def _error_parts(exc: Exception) -> tuple[int, str, str]:
    """Map an engine exception onto an OpenAI ``(status, message, type)``."""
    if isinstance(exc, RequestError):
        return exc.status, exc.message, exc.err_type
    if isinstance(exc, ContextFull):
        return 413, str(exc), "context_length_exceeded"
    if isinstance(exc, PagedCacheFull):
        return 429, f"shared KV-cache pool is full: {exc}", "insufficient_quota"
    return 500, f"internal server error: {exc}", "server_error"


def _content_text(content, index: int) -> str:
    """The text of one message's ``content``.

    OpenAI accepts either a plain string or an array of typed parts, and SDKs emit the
    array form freely. The parts are concatenated; this model is text-only, so any other
    part type (an image, audio) is refused rather than silently dropped from the prompt.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError(400, f"messages[{index}].content must be a string or an array of content parts")
    text = []
    for part in content:
        if isinstance(part, str):
            text.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text" and isinstance(part.get("text"), str):
            text.append(part["text"])
        else:
            kind = part.get("type", "<no type>") if isinstance(part, dict) else type(part).__name__
            raise RequestError(
                400, f"messages[{index}] carries a {kind!r} content part, but this model takes text only"
            )
    return "".join(text)


_PROMPT_SNIPPET_CHARS = 200
# Whether a logged message is shown whole or cut to its first ``_PROMPT_SNIPPET_CHARS``
# characters. A one-element list so the console's 'v' key can flip it from its own thread;
# it applies to requests logged after the flip, not to lines already in the log.
_LOG_FULL_PROMPT = [True]


def _toggle_full_prompt() -> bool:
    """Flip between full prompts and snippets in the log; returns whether it is now full."""
    _LOG_FULL_PROMPT[0] = not _LOG_FULL_PROMPT[0]
    return _LOG_FULL_PROMPT[0]


def _one_line(text: str) -> str:
    """``text`` with its whitespace collapsed, so a message logs as a single record.

    Whole by default, since the log is what a prompt gets debugged with and the console folds
    a long record across as many screen lines as it needs. With snippets on, a message longer
    than ``_PROMPT_SNIPPET_CHARS`` is cut there and says how much was left out, so a long
    prompt does not push everything else out of the log.
    """
    line = " ".join(text.split())
    if _LOG_FULL_PROMPT[0] or len(line) <= _PROMPT_SNIPPET_CHARS:
        return line
    return f"{line[:_PROMPT_SNIPPET_CHARS]}... [+{len(line) - _PROMPT_SNIPPET_CHARS} chars, 'v' shows full]"


def _describe_request(completion_id: str, client: str, user_key: str, body: dict) -> str:
    """A debug rendering of one request: who sent it, the options, and the messages.

    Multi-line on purpose -- the console stamps the first line and indents the rest --
    and every message is abbreviated, since a full history runs to tens of kilobytes and
    would push everything else out of the log.
    """
    options = {
        key: body[key]
        for key in ("model", "stream", "max_tokens", "temperature", "top_p", "thinking", "reasoning_effort")
        if body.get(key) is not None
    }
    messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]

    def shown(content) -> str:
        # An array's non-text parts (an image's base64 runs to megabytes) are named, not printed.
        if not isinstance(content, list):
            return str(content if content is not None else "")
        return "".join(
            (
                part
                if isinstance(part, str)
                else (
                    part["text"]
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                    else f"[{part.get('type', '?') if isinstance(part, dict) else type(part).__name__}]"
                )
            )
            for part in content
        )

    texts = [shown(m.get("content")) for m in messages]
    head = (
        f"{completion_id} from {client} user={user_key!r} "
        f"{' '.join(f'{k}={v!r}' for k, v in options.items())}\n"
        f"  {len(messages)} messages, {sum(len(t) for t in texts)} chars"
    )
    return "\n".join([head] + [f"    [{m.get('role')}] {_one_line(t)}" for m, t in zip(messages, texts)])


_ROLES = ("system", "developer", "user", "assistant", "tool")


def _normalized_tool_calls(tool_calls, index: int) -> list[dict]:
    """A request's echoed ``tool_calls``, in the shape this server also emits."""
    if not isinstance(tool_calls, list):
        raise RequestError(400, f"messages[{index}].tool_calls must be an array")
    out = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise RequestError(400, f"messages[{index}].tool_calls entries need a function name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):  # some clients send the object rather than its JSON
            arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
        out.append(
            {
                "id": str(call.get("id") or ""),
                "type": "function",
                "function": {"name": function["name"], "arguments": arguments},
            }
        )
    return out


def _normalized_tools(tools) -> list[dict]:
    """The request's ``tools``, checked for the fields the prompt renderer reads."""
    if not isinstance(tools, list) or not tools:
        raise RequestError(400, "tools must be a non-empty array")
    out = []
    for index, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise RequestError(400, f"tools[{index}] must be an object with a function name")
        out.append({"type": "function", "function": function})
    return out


def _text_content(role: str, content, index: int) -> dict:
    """A message's content fields for the renderer: ``{"content": text}`` (see :func:`_content_text`)."""
    return {"content": _content_text(content, index)}


def _normalized_messages(messages, tools, chat_format: ChatFormat, content_fields=_text_content) -> list[dict]:
    """Validate a request's ``messages`` and put them in the form the renderer wants.

    ``content_fields(role, content, index)`` turns one message's ``content`` into the entry's content fields
    (:func:`_text_content` by default; a model that reads images also returns ``content_blocks``).

    Two shapes are canonicalised here. OpenAI's ``content`` may be a string or an array
    of typed parts, and the extra fields a client echoes back with an assistant reply
    (``refusal``, a null ``tool_calls``) are dropped, since downstream compares whole
    messages to recognise a continued conversation.

    The model also has no ``tool`` role: it reads results as ``<tool_result>`` blocks
    inside a user message, so a tool turn is folded in by the checkpoint's own
    ``merge_tool_messages``. That leaves the last message a user message again, which is
    what the rest of the server expects.
    """
    if not isinstance(messages, list) or not messages:
        raise RequestError(400, "messages must be a non-empty array")
    out = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise RequestError(400, f"messages[{index}] must be an object with a role")
        role = message["role"]
        if role not in _ROLES:
            raise RequestError(400, f"messages[{index}] has role {role!r}, which this model has no place for")
        entry = {"role": role, **content_fields(role, message.get("content"), index)}
        if role == "assistant" and message.get("tool_calls"):
            entry["tool_calls"] = _normalized_tool_calls(message["tool_calls"], index)
        if role == "tool":
            # Carries which call this answers, so several results can be put back in the
            # order the assistant asked for them.
            entry["tool_call_id"] = str(message.get("tool_call_id") or "")
        out.append(entry)

    if out[-1]["role"] not in ("user", "tool"):
        raise RequestError(
            400, f"the final message must come from the user or a tool, but its role is {out[-1]['role']!r}"
        )

    if tools:
        # The schemas are rendered into the system turn; without one the model is never
        # told what it may call.
        system = next((m for m in out if m["role"] == "system"), None)
        if system is None:
            system = {"role": "system", "content": ""}
            out.insert(0, system)
        system["tools"] = _normalized_tools(tools)

    if any(m["role"] == "tool" for m in out):
        out = chat_format.sort_tool_results_by_call_order(chat_format.merge_tool_messages(out))
    return out


def _conversation_key(message: dict) -> str:
    """What identifies a message when matching a request against a warm cache.

    ``reasoning_content`` is left out: earlier turns are re-rendered without their
    thinking anyway (``drop_thinking``), and clients seldom echo it back, so counting it
    would re-prefill conversations that do in fact continue.

    A user message that is only its own text is also written both ways -- plain
    ``content``, and the ``content_blocks`` mirror that folding in a tool result gives
    every user turn. They mean the same thing, so the mirror is dropped: otherwise the
    first tool result of a session would appear to rewrite all the turns before it and
    re-prefill the whole conversation.

    An assistant message is matched on what it says, not on how the client echoes it: a
    continuation never re-renders it (its tokens are the model's own, already cached), and
    harnesses routinely rename tool-call ids, re-serialize the arguments or trim the text.
    So its content is compared stripped and its tool calls by name and parsed arguments.
    """
    message = {k: v for k, v in message.items() if k != "reasoning_content"}
    if message.get("content_blocks") == [{"type": "text", "text": message.get("content", "")}]:
        del message["content_blocks"]
    if message.get("role") == "assistant":
        message["content"] = (message.get("content") or "").strip()
        calls = []
        for call in message.pop("tool_calls", None) or []:
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                arguments = function.get("arguments")
            calls.append([function.get("name"), arguments])
        if calls:
            message["tool_calls"] = calls
    return json.dumps(message, sort_keys=True, default=str)


def _continues(messages: list[dict], stored: list[dict]) -> bool:
    """Whether ``messages`` extends the conversation a slot already holds."""
    if not stored or len(messages) <= len(stored):
        return False
    return [_conversation_key(m) for m in messages[: len(stored)]] == [_conversation_key(m) for m in stored]


def _parameter_value(value: str, is_string: bool):
    """One DSML parameter, decoded by its ``string`` flag.

    A value the model flagged as JSON but did not write as JSON is kept as text: the
    call is still worth forwarding, and the tool's own validation is a better place to
    reject it than a 500 from here.
    """
    if is_string:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_tool_calls(block: str, chat_format: ChatFormat) -> list[dict]:
    """The OpenAI ``tool_calls`` described by a DSML block in ``chat_format``'s tags.

    Deliberately tolerant where the checkpoint's reference parser raises: a serving loop
    should hand back whatever the model managed to express rather than fail the request
    over a malformed tag. Both argument forms are read -- the ``parameter`` tags, and the
    bare JSON object that the model sometimes writes inside ``invoke`` instead.
    """
    calls = []
    for invoke in chat_format.invoke_re.finditer(block):
        parameters = list(chat_format.param_re.finditer(invoke["body"]))
        arguments: dict = {}
        if parameters:
            for parameter in parameters:
                arguments[parameter["name"]] = _parameter_value(parameter["value"], parameter["string"] == "true")
        elif invoke["body"].strip():
            with contextlib.suppress(json.JSONDecodeError):
                loaded = json.loads(invoke["body"].strip())
                if isinstance(loaded, dict):
                    arguments = loaded
        calls.append(
            {
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": invoke["name"], "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        )
    return calls


SAMPLER_TOP_K = 64

_REPLAY_RATE_SEED = 130.0
# Prompt tokens a replay must span before its rate is trusted: a few rounds' latency dominates shorter ones.
_REPLAY_RATE_MIN_TOKENS = 64

# Generated tokens the live decode rate is averaged over, as
# ``tests/decode/test_full_model_decode_demo.py`` reports throughput over a rolling window
# rather than a whole run. A session's first ``sliding_window`` positions replay the
# masked SDPA variant, whose cost is set by ``--max-context`` instead of by the
# position, so a turn that starts on a cold cache is slow for that prefix and then
# speeds up mid-reply when the causal variant takes over. Averaged over the whole
# reply that prefix hides the rate the turn actually settled at.
DECODE_RATE_WINDOW = 64


def _make_sampler(temperature, top_p):
    """A logits->token sampler honouring OpenAI's ``temperature``/``top_p``, or
    ``None`` for greedy argmax when the request asks for neither.

    Sampling narrows to the ``SAMPLER_TOP_K`` highest logits before normalising.
    A softmax over the full 129k vocabulary costs ~3ms per token on the host, and
    ~15ms once ``top_p`` sorts it, against a ~10ms decode step: the scheduler
    thread would spend more time sampling than the device spends decoding.
    """
    try:
        temperature = float(temperature) if temperature is not None else 0.0
        top_p = float(top_p) if top_p is not None else None
    except (TypeError, ValueError):
        raise RequestError(400, "temperature and top_p must be numbers")
    if not temperature >= 0:
        raise RequestError(400, "temperature must be a non-negative number")
    if top_p is not None and not 0 < top_p <= 1:
        raise RequestError(400, "top_p must be in (0, 1]")
    if temperature <= 0 and top_p is None:
        return None

    def sample(logits: torch.Tensor) -> int:
        t = temperature if temperature > 0 else 1.0
        values, index = torch.topk(logits[0], min(SAMPLER_TOP_K, logits.shape[-1]))
        probs = torch.softmax(values / t, dim=-1)
        if top_p is not None and top_p < 1.0:
            cumulative = torch.cumsum(probs, dim=-1)
            probs = torch.where(cumulative - probs <= top_p, probs, torch.zeros_like(probs))
            probs = probs / probs.sum()
        return int(index[torch.multinomial(probs, 1)].item())

    return sample


class _Streamer:
    """Incrementally detokenize a reply, separating the ``<think>`` reasoning block
    from the answer and reporting only newly resolved text via
    ``on_chunk(reasoning, content)``.

    A token can carry half a UTF-8 sequence, so the ids are re-decoded cumulatively
    and a trailing U+FFFD is held back; trailing text that could still turn into a
    think tag is likewise withheld until it resolves, so deltas never carry half a
    tag. The reasoning block and the answer are emitted as tag-free strings, ready
    for the ``reasoning_content`` / ``content`` fields of the OpenAI response.

    The thinking template already ends the prompt with ``<think>``, so the completion
    starts inside the block and the opening tag is not generated again. With
    ``thinking``, text up to ``</think>`` is reasoning; an opening tag in the
    completion itself still wins, which is the same split as a reply that repeats it.

    A DSML tool-call block is held back the same way and never streamed as content: it
    is markup addressed to the client's tool runner, not prose for the reader. What it
    describes is parsed into OpenAI ``tool_calls`` and left on :attr:`tool_calls`.
    """

    def __init__(self, tokenizer, on_chunk, chat_format: ChatFormat, thinking: bool = False):
        self.tokenizer = tokenizer
        self.on_chunk = on_chunk
        self.chat_format = chat_format
        self.thinking = thinking
        self.text = ""  # raw decoded reply, tags included, no escapes
        self.reasoning = ""
        self.content = ""
        self.tool_calls: list[dict] = []
        self._sent_r = 0
        self._sent_c = 0

    def push(self, token_ids) -> None:
        full = self.tokenizer.decode(token_ids, skip_special_tokens=False).rstrip("\ufffd")
        self.text = full
        held = self._held_back(full)
        reasoning, content, _ = self._split(full[: len(full) - held])
        # A re-decode can revise the tail (split UTF-8); resync to the common prefix.
        if not reasoning.startswith(self.reasoning):
            self._sent_r = min(self._sent_r, len(os.path.commonprefix([reasoning, self.reasoning])))
        if not content.startswith(self.content):
            self._sent_c = min(self._sent_c, len(os.path.commonprefix([content, self.content])))
        if len(reasoning) > self._sent_r:
            self.on_chunk(reasoning[self._sent_r :], "")
            self._sent_r = len(reasoning)
        if len(content) > self._sent_c:
            self.on_chunk("", content[self._sent_c :])
            self._sent_c = len(content)
        self.reasoning, self.content = reasoning, content

    def close(self) -> None:
        """Flush the tail held back for a possible partial tag or character.

        Generation is over, so nothing more can complete the tail: it is emitted
        as-is and :attr:`reasoning` / :attr:`content` hold the complete reply."""
        reasoning, content, tool_block = self._split(self.text)
        self.tool_calls = _parse_tool_calls(tool_block, self.chat_format) if tool_block else []
        if len(reasoning) > self._sent_r:
            self.on_chunk(reasoning[self._sent_r :], "")
            self._sent_r = len(reasoning)
        if len(content) > self._sent_c:
            self.on_chunk("", content[self._sent_c :])
            self._sent_c = len(content)
        self.reasoning, self.content = reasoning, content

    def reasoning_tokens(self, token_ids) -> int:
        """How many of the reply's ``token_ids`` are its reasoning block, tags included, split as
        :meth:`_split` splits the text (``<think>`` and ``</think>`` are single tokens)."""
        ids = list(token_ids)
        open_id, close_id = self.tokenizer.convert_tokens_to_ids([_THINK_OPEN, _THINK_CLOSE])
        if open_id in ids:
            start = ids.index(open_id)
        elif self.thinking:
            start = 0
        else:
            return 0
        end = ids.index(close_id, start) + 1 if close_id in ids[start:] else len(ids)
        return end - start

    def _split(self, text: str) -> tuple[str, str, str]:
        """``text`` as (reasoning, answer, tool-call block), each without its tags."""
        start = text.find(_THINK_OPEN)
        if start == -1:
            end = text.find(_THINK_CLOSE)
            # The prompt already opened the block, so the completion begins with reasoning
            # and the first ``</think>`` is the boundary. Without that mode, a stray close
            # tag is ordinary answer text.
            if self.thinking and end != -1:
                reasoning, answer = text[:end], text[end + len(_THINK_CLOSE) :]
            elif self.thinking:
                reasoning, answer = text, ""
            else:
                reasoning, answer = "", text
        else:
            after = text[start + len(_THINK_OPEN) :]
            end = after.find(_THINK_CLOSE)
            if end == -1:
                reasoning, answer = after, text[:start]
            else:
                reasoning, answer = after[:end], text[:start] + after[end + len(_THINK_CLOSE) :]

        calls_open, calls_close = self.chat_format.calls_open, self.chat_format.calls_close
        opened = answer.find(calls_open)
        if opened == -1:
            return reasoning, answer, ""
        closed = answer.find(calls_close, opened)
        block = answer[opened : closed + len(calls_close)] if closed != -1 else answer[opened:]
        # The encoder separates the answer from the block with a blank line, which would
        # otherwise trail the reply as whitespace.
        return reasoning, answer[:opened].rstrip("\n"), block

    def _held_back(self, text: str) -> int:
        """Characters at the end that could still turn into a tag once more arrive.

        The newlines before a tool-call block count: they are the block's separator, and
        streaming them the moment they arrive would leave a trailing blank line on a
        reply whose visible text has actually ended."""
        for tag in (_THINK_CLOSE, _THINK_OPEN, self.chat_format.calls_open):
            for n in range(len(tag) - 1, 0, -1):
                if text.endswith(tag[:n]):
                    return n + len(text[: len(text) - n]) - len(text[: len(text) - n].rstrip("\n"))
        return len(text) - len(text.rstrip("\n"))


@dataclass(eq=False)
class _HostConversation:
    """A conversation whose device slot another request needed, kept in host RAM: what its
    :class:`UserSession` held, plus ``snapshot``, the model's host copy of its KV
    (``DeepSeekV41Model.export_session``). A request that continues it gets it copied back into a slot."""

    owner: str | None
    messages: list[dict]
    tokens: list[int]
    marks: list[tuple[int, int, int | None]]
    pending_id: int | None
    next_render: int
    thinking_mode: str
    reasoning_effort: str | None
    system_prompt: str | None
    snapshot: object
    stored_at: float


@dataclass(eq=False)
class _Lease:
    """A slot claimed for one turn by :meth:`_SlotPool.acquire`. ``previous`` held the slot before; ``restore`` is a
    host conversation the turn continues, which the scheduler copies into the slot before admitting it."""

    slot: int
    previous: str | None
    restore: _HostConversation | None


class _SlotPool:
    """The KV-cache slots (``--num-users`` of them, each a slot id with its own KV session) and the
    conversations they hold, in these states:

    * **running**: the slot's turn holds one of the ``max_running`` seats and is generating;
    * **queued**: the slot is claimed for a turn waiting, first come first served, for a seat
      (:meth:`wait_to_run`);
    * **parked**: the slot holds a finished turn's conversation, its KV still in device DRAM, so a
      request continuing it feeds only its new turn. It keeps it until a request needs a slot when none
      is empty and it has been idle for ``idle_timeout`` seconds (see below);
    * **empty**: the slot holds no conversation;
    * **host**: a conversation whose slot was emptied, its KV copied to host RAM
      (:class:`_HostConversation`). A request continuing it is given an empty slot and the KV copied
      back rather than re-prefilled. At most ``host_users`` conversations and ``host_bytes`` bytes are
      kept, the least recently stored dropped past either; ``host_users=0`` turns the tier off.

    A slot is claimed for the duration of a turn and released afterwards, and any request can use
    any slot that is not claimed. :meth:`acquire` prefers, in order:

    1. the longest stored conversation that is a prefix of this request's history,
       parked or in host RAM (parked on a tie) -- the follow-up case, where the cache
       already holds everything but the new turn and only the new tokens are fed;
    2. an empty slot, which costs nobody their cache;
    3. with no empty slot, the parked slot idle longest (since its last request finished), once that
       is at least ``idle_timeout`` seconds (0: never).

    Otherwise :meth:`acquire` raises a 429, and every parked conversation stays where it is. A slot
    taken over is emptied by the scheduler, its conversation offloaded to host RAM first
    (:meth:`_Scheduler._offload`), as are parked slots when the shared block pool cannot fit a turn
    (:meth:`_Scheduler._make_room`).
    """

    def __init__(
        self,
        engine: ChatEngine,
        host_users: int = 0,
        host_bytes: int = 0,
        max_running: int = 0,
        idle_timeout: float = 0.0,
    ):
        self.engine = engine
        self.idle_timeout = idle_timeout
        self._cv = threading.Condition()
        self._busy = [False] * len(engine.users)
        self.owner: list[str | None] = [None] * len(engine.users)
        self._used_at = [0.0] * len(engine.users)
        self._closing = False
        self.max_running = min(max_running, len(engine.users)) if max_running > 0 else len(engine.users)
        self._queue: deque[int] = deque()  # slots waiting for a seat, oldest first
        self._running: set[int] = set()
        self.host_users = host_users
        self.host_bytes = host_bytes
        self.host: list[_HostConversation] = []

    def __len__(self) -> int:
        return len(self._busy)

    # -- claiming --------------------------------------------------------------- #
    def acquire(self, user_key: str, messages: list[dict]) -> _Lease:
        """Claim a slot for one turn; a 429 when every slot is claimed or holds another user's conversation."""
        with self._cv:
            if self._closing:
                raise RequestError(503, "server is shutting down", "server_error")
            picked = self._pick(messages)
            if picked is None:
                free = [i for i, busy in enumerate(self._busy) if not busy]
                when = ""
                if free and self.idle_timeout > 0:
                    wait = self.idle_timeout - (time.time() - min(self._used_at[i] for i in free))
                    when = f"; the longest-idle parked slot can be taken in {max(wait, 0):.0f}s"
                raise RequestError(
                    429,
                    f"all {len(self._busy)} KV slots are in use ({len(self._running)} running, "
                    f"{len(self._queue)} queued, {len(free)} parked{when}); retry later, or free a "
                    f"slot with DELETE /v1/sessions/<user>",
                    "rate_limit_error",
                )
            slot, restore = picked
            if restore is not None:
                self.host.remove(restore)
            user = self.engine.users[slot]
            if user.pos and not _continues(messages, user.messages):
                logger.info(
                    f"user {user_key!r}: no empty slot, taking slot {slot} from {self.owner[slot]!r} "
                    f"(idle {time.time() - self._used_at[slot]:.0f}s, {user.pos} tokens)"
                )
            self._busy[slot] = True
            previous, self.owner[slot] = self.owner[slot], user_key
            self._used_at[slot] = time.time()
        # logger.debug(
        #     f"user {user_key!r} -> slot {slot} (sid {self.engine.users[slot].sid}, "
        #     f"{self._reason(slot, previous, user_key, messages)})"
        # )
        return _Lease(slot, previous, restore)

    def wait_to_run(self, slot: int, give_up) -> bool:
        """Queue claimed ``slot`` for one of the ``max_running`` seats and wait, first come first served;
        returns whether it got one (``False`` once ``give_up()`` is true). Hand the seat back with
        :meth:`finish_run`."""
        with self._cv:
            self._queue.append(slot)
            try:
                while True:
                    if self._closing:
                        raise RequestError(503, "server is shutting down", "server_error")
                    if self._queue[0] == slot and len(self._running) < self.max_running:
                        self._running.add(slot)
                        return True
                    if give_up():
                        return False
                    self._cv.wait(timeout=0.5)
            finally:
                self._queue.remove(slot)
                self._cv.notify_all()

    def finish_run(self, slot: int) -> None:
        with self._cv:
            self._running.discard(slot)
            self._cv.notify_all()

    def claim_parked(self, exclude: int) -> tuple[int, str | None] | None:
        """Claim the least recently used parked slot other than ``exclude``, for the scheduler to offload;
        ``(slot, owner)``, or ``None`` when there is none. Give it back with :meth:`release`."""
        with self._cv:
            parked = [
                i
                for i, user in enumerate(self.engine.users)
                if not self._busy[i] and i != exclude and user.pos and not self._closing
            ]
            if not parked:
                return None
            slot = min(parked, key=lambda i: self._used_at[i])
            self._busy[slot] = True
            return slot, self.owner[slot]

    def reserve(self, slot: int) -> None:
        """Claim one specific slot, waiting for its turn to finish (used by a reset)."""
        with self._cv:
            while self._busy[slot] and not self._closing:
                self._cv.wait(timeout=1.0)
            if self._closing:
                raise RequestError(503, "server is shutting down", "server_error")
            self._busy[slot] = True

    def release(self, slot: int, vacated: bool = False) -> None:
        """Give back a claimed slot; ``vacated``: its conversation has gone (offloaded and reset), so it has no owner."""
        with self._cv:
            self._busy[slot] = False
            self._used_at[slot] = time.time()
            if vacated:
                self.owner[slot] = None
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closing = True
            self._cv.notify_all()

    def _pick(self, messages: list[dict]) -> tuple[int, _HostConversation | None] | None:
        """``(slot, host conversation to copy into it or None)``, or ``None`` when no slot may be taken."""
        free = [i for i, busy in enumerate(self._busy) if not busy]
        if not free:
            return None
        # 1. The longest stored conversation this request continues, parked or in host RAM.
        best, best_len = None, 0
        for i in free:
            stored = self.engine.users[i].messages
            if len(stored) > best_len and _continues(messages, stored):
                best, best_len = i, len(stored)
        restore = None
        for conv in self.host:
            if len(conv.messages) > best_len and _continues(messages, conv.messages):
                restore, best_len = conv, len(conv.messages)
        if best is not None:
            return best, restore  # a parked prefix of the host conversation is no loss to overwrite
        # 2. A slot holding nothing.
        for i in free:
            if not self.engine.users[i].pos:
                return i, restore
        # 3. No slot left: the parked slot idle longest, once it has been idle for idle_timeout.
        if self.idle_timeout > 0:
            oldest = min(free, key=lambda i: self._used_at[i])
            if time.time() - self._used_at[oldest] >= self.idle_timeout:
                return oldest, restore
        return None

    # -- host tier -------------------------------------------------------------- #
    def store(self, conv: _HostConversation) -> None:
        """Keep ``conv`` in host RAM and drop the least recently stored conversations past ``host_users`` /
        ``host_bytes`` (``conv`` itself when it alone is over the bytes)."""
        dropped = []
        with self._cv:
            self.host.append(conv)
            while self.host and (
                len(self.host) > self.host_users or sum(c.snapshot.nbytes for c in self.host) > self.host_bytes
            ):
                oldest = min(self.host, key=lambda c: c.stored_at)
                self.host.remove(oldest)
                dropped.append(oldest)
        for conv in dropped:
            logger.info(
                f"user {conv.owner!r}: dropped its {conv.snapshot.pos}-token conversation from host RAM "
                f"({self.host_summary()}); continuing it re-prefills"
            )

    def drop_host(self, user_key: str) -> int:
        """Drop every host conversation ``user_key`` held; returns how many there were."""
        with self._cv:
            kept = [c for c in self.host if c.owner != user_key]
            dropped, self.host = len(self.host) - len(kept), kept
        return dropped

    def counts(self) -> dict:
        """How many slots are ``running`` / ``queued`` / ``parked`` / ``empty``, plus ``max_running``."""
        with self._cv:
            running, queued = len(self._running), len(self._queue)
            free = [i for i, busy in enumerate(self._busy) if not busy]
            parked = sum(1 for i in free if self.engine.users[i].pos)
            return {
                "running": running,
                "queued": queued,
                "parked": parked,
                "empty": len(free) - parked,
                "max_running": self.max_running,
            }

    def host_usage(self) -> tuple[int, int]:
        """``(conversations, bytes)`` in host RAM."""
        with self._cv:
            return len(self.host), sum(c.snapshot.nbytes for c in self.host)

    def host_summary(self) -> str:
        count, nbytes = self.host_usage()
        return f"{count}/{self.host_users} conversations, {nbytes / 2**30:.2f}/{self.host_bytes / 2**30:.0f} GiB"

    def _reason(self, slot: int, previous: str | None, user_key: str, messages: list[dict]) -> str:
        user = self.engine.users[slot]
        stored = user.messages
        if _continues(messages, stored):
            return f"continuing {len(stored)} messages at {user.pos} tokens"
        if not stored:
            return "empty slot"
        if previous not in (None, user_key):
            return f"reusing {previous!r}'s idle slot, dropping {user.pos} cached tokens"
        return f"re-prefill, dropping {user.pos} cached tokens"

    # -- reporting -------------------------------------------------------------- #
    def rows(self) -> list[dict]:
        """One row per device slot, then one per host conversation (``index`` ``None``, ``host`` its
        position), for ``/v1/sessions`` and the status pane. ``tier`` is ``running``, ``queued``, ``busy``
        (claimed for a reset or to be emptied, or about to queue), ``parked``, ``empty`` or ``host``; ``queue``
        is a queued slot's place in the run queue (0 is next)."""
        with self._cv:
            busy, owner, host = list(self._busy), list(self.owner), list(self.host)
            running, queue = set(self._running), list(self._queue)
            queued = set(queue)

        def tier(index: int, user) -> str:
            if index in running:
                return "running"
            if index in queued:
                return "queued"
            return "busy" if busy[index] else "parked" if user.pos else "empty"

        rows = []
        for index, user in enumerate(self.engine.users):
            rows.append(
                {
                    "id": owner[index] or "",
                    "index": index,
                    "tier": tier(index, user),
                    "queue": queue.index(index) if index in queued else None,
                    "tokens": user.pos,
                    "max_context": self.engine.max_seq,
                    "messages": len(user.messages),
                    "thinking": user.thinking_mode == "thinking",
                    "busy": busy[index],
                }
            )
        for i, conv in enumerate(sorted(host, key=lambda c: c.stored_at, reverse=True)):
            rows.append(
                {
                    "id": conv.owner or "",
                    "index": None,
                    "host": i,
                    "tier": "host",
                    "tokens": conv.snapshot.pos,
                    "max_context": self.engine.max_seq,
                    "messages": len(conv.messages),
                    "thinking": conv.thinking_mode == "thinking",
                    "busy": False,
                    "bytes": conv.snapshot.nbytes,
                }
            )
        return rows

    def slots_of(self, user_key: str) -> list[int]:
        with self._cv:
            return [i for i, owner in enumerate(self.owner) if owner == user_key]


class _Prefiller:
    """Traced, pipelined prefill on the decode model's own stages, as
    ``tests/prefill/test_prefill_decode_demo.py`` runs it.

    The prefill model is built by :meth:`DeepSeekV4Model.build_prefill`, one layer on each
    decode layer's submesh, sharing decode's routed experts, embedding table and
    ``lm_head``. A prompt's 128-aligned prefix is prefilled by replaying one trace per
    stage per chunk (the stages work on consecutive chunks at once), its attention state
    is committed into the session's decode buffers on device, and the ragged tail goes
    through decode as before.

    Construction is the test's steps 2-4 up to the decode capture (decode compile, then the
    prefill build, compile and capture with decode's GCBs and L1 tensors released, then
    those restored); :meth:`snapshot` is the rest of step 4, after :meth:`ChatEngine.warmup`
    has captured the decode traces. With the prefetcher on, the device needs an
    ``l1_small_size`` for the GCB config pages.
    """

    def __init__(self, engine: ChatEngine, args, prefetcher: contextlib.ExitStack):
        model = engine.model
        self.engine = engine
        self.chunk_size = args.prefill_trace_chunk
        # Sizes the persistent buffers and the position tiers (the prefill model reads its RoPE rows from
        # ``engine.rope``, which spans ``engine.max_seq``, so ``max_len`` must stay below it). A chunk's cost follows its
        # position, not ``max_len`` (DEEPSEEK_V4_PREFILL_TIER_FACTOR), so this is a capacity knob: a larger value costs
        # DRAM for the buffers and a trace per stage for each further tier, not speed.
        self.max_len = min(args.prefill_max_len or engine.max_seq, engine.max_seq - 1) // ALIGNMENT * ALIGNMENT
        if self.max_len < ALIGNMENT:
            raise ValueError(f"traced prefill needs a context of at least {ALIGNMENT + 1} tokens")
        self.num_stages = len(dict.fromkeys(model.layer_submesh_ids))
        # What a prefill costs, for choosing it over replaying the prompt through decode: the first chunk
        # crosses every stage before its logits arrive (plus the write-back and the commit), each later one
        # lands a stage behind it. Seeded from test_prefill_decode_demo.py on 8 stages with 1024-token
        # chunks; every prefill folds its own timings in. Chunks of later position tiers cost more than early
        # ones, so the per-chunk figure is a mean over the prompts seen, good enough to tell prefill from replay.
        self.fill_seconds = 1.6
        self.chunk_seconds = 0.23
        self._loud = True
        self.t_prefill_out = 0.0  # when the latest run()'s logits came back
        tokenizer = engine.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else engine.eos_id

        # Decode's ops keep buffers of their own (global semaphores, cached constants) that no snapshot can
        # reach; compiling first allocates them before prefill lays out its memory, so its replays keep off them.
        t0 = time.perf_counter()
        model.compile_traces(pad_id, 0)
        logger.info(f"decode compile: {time.perf_counter() - t0:.1f}s")
        # Stops the prefetcher and frees the GCBs; allowed because no decode trace exists yet.
        model.release_prefetch_buffers()
        t0 = time.perf_counter()
        self.model = self._build_model(engine)
        self.model.synchronize("uploads")
        logger.info(f"prefill model built in {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        self.model.compile_traced_prefill(self.max_len, self.chunk_size)
        logger.info(
            f"traced prefill compiled in {time.perf_counter() - t0:.1f}s: prompts of up to {self.max_len} "
            f"tokens in chunks of {self.chunk_size}, {self.num_stages} pipelined stage(s)"
        )
        # Unwinds before the decode model's shutdown (LIFO).
        prefetcher.callback(self.model.release_traced_prefill)
        self.bias_slots = self._bias_slots()
        # Captured before decode: stopping the prefetcher once the decode traces exist would drop the
        # prefetch requests they replay.
        t0 = time.perf_counter()
        self.model.capture_traced_prefill()
        logger.info(f"traced prefill captured in {time.perf_counter() - t0:.1f}s")
        # The GCBs and L1 tensors come back under prefill's buffers; the prefetcher then runs until teardown.
        model.restore_prefetch_buffers()

    def _build_model(self, engine: ChatEngine):
        """The prefill model on the decode stages (decode's GCBs and L1 tensors already released)."""
        indexer = os.environ.get("DEEPSEEK_V4_PREFILL_INDEXER", "1") == "1"
        logger.info(f"prefill lightning indexer {'on' if indexer else 'off'}")
        return engine.model.build_prefill(
            engine.rope,
            checkpoint_weights(engine.loader, engine.config, engine.model.num_layers),
            lm_head=engine.lm_head,
            cache=engine.weight_cache,
            weight_dtype=_ATTENTION_WEIGHT_DTYPE,
            dense_csa=True,
            lightning_indexer=indexer,
            # Sliding layers attend without an additive mask (CSA / HCA stay masked), as in the test.
            maskless=True,
            progress=self._note,
        )

    def _bias_slots(self):
        """What :meth:`run` hands ``commit_prefill_state`` as ``bias_slots``."""
        return prefill_bias_slots(self.model)

    def _note(self, message: str, important: bool = False) -> None:
        if important:
            (logger.info if self._loud else logger.debug)(f"[prefill] {message}")

    def snapshot(self) -> None:
        """Copy decode's resident tensors to the host, after :meth:`ChatEngine.warmup` has captured the
        decode traces: they sit under prefill's buffers, so :meth:`run` writes them back after every replay."""
        self.engine.model.snapshot_resident_state()
        self._loud = False

    # A prefill continues a session (``user.pos > 0``, see :meth:`aligned`) only from multiples of this;
    # ``None``: the prefill model only starts sessions.
    continue_alignment: int | None = None
    # A session rewound to an earlier position (see _Scheduler._resume) is continued by a bounded replay: the
    # prefill runs this many cached tokens before that position again, only to rebuild the sliding window, and
    # reuses the cached compressed KV. ``None``: the prefill model has no bounded replay.
    replay_window: int | None = None

    def aligned(self, n_tokens: int, start: int = 0, bounded: bool = False) -> int:
        """How many of ``n_tokens`` prompt tokens at positions ``start ..`` the prefill takes (0 for none);
        decode replays the rest. ``bounded``: as a bounded replay from ``start`` (the first
        :attr:`replay_window` of them the cached window). V4's prefill model only starts a session."""
        if start or bounded:
            return 0
        return min(n_tokens // ALIGNMENT * ALIGNMENT, self.max_len)

    def _prefill(self, user: UserSession, ids: torch.Tensor, on_chunk, replayed: int = 0):
        """``(logits, states)`` of the traced prefill of ``ids`` ``[1, n]`` at ``user.pos``, or, ``replayed``
        (:attr:`replay_window`), as a bounded replay whose first ``replayed`` ids precede ``user.pos``."""
        return self.model.prefill_traced(ids, on_chunk=on_chunk)

    def _commit(self, user: UserSession, states) -> int:
        """Write ``states`` into ``user``'s session; returns its new position."""
        model = self.engine.model
        # The previous conversation (or the capture step) must not leak into this one.
        model.reset_session(user.sid)
        model.reset_static_caches()
        return model.commit_prefill_state(states, user.sid, bias_slots=self.bias_slots)

    def seconds(self, n_tokens: int) -> float:
        """Expected wall time of :meth:`run` over ``n_tokens`` (a multiple of 128)."""
        return self.fill_seconds + (-(-n_tokens // self.chunk_size) - 1) * self.chunk_seconds

    def run(
        self,
        user: UserSession,
        ids: list[int],
        on_progress=None,
        drain_seconds: float = 0.0,
        drained: int = 0,
        replayed: int = 0,
    ) -> torch.Tensor | None:
        """Prefill ``ids`` (as many tokens as :meth:`aligned` allows at ``user.pos``) into ``user``'s
        session, empty or continued from ``user.pos``; returns the next-token logits ``[V]`` (host fp32), or
        ``None`` from a prefill model without a head (its :meth:`aligned` then always leaves tokens for decode), and
        advances ``user.pos`` by ``len(ids) - replayed``. A continued prefill that fails rewinds the session, so the
        next turn re-prefills the conversation whole. ``on_progress`` gets the count of new ``ids`` done.

        ``replayed`` (:attr:`replay_window`) makes it a bounded replay: the first ``replayed`` of ``ids`` are the
        cached tokens just before ``user.pos``, run again to rebuild the sliding window over the cached
        compressed KV (``aligned(len(ids), user.pos - replayed, bounded=True)`` of them).

        Must run with no decode step in flight: the replays share decode's command queue and
        overwrite its L1 tensors, which are written back before anything else runs.
        ``drain_seconds`` / ``drained``: how long the caller waited for its ``drained`` decode
        steps in flight to come back first, reported with the rest of the decode <-> prefill switch.
        Sets :attr:`t_prefill_out`, when the prefill's logits came back."""
        engine, model, prefill = self.engine, self.engine.model, self.model
        if engine._replay_q:
            raise RuntimeError("prefill with decode traces still posted")
        # Seated first, so the previous occupant's window state is saved to its own (DRAM)
        # block before the replay overwrites the resident L1 copies.
        t_seat = time.perf_counter()
        user.activate()
        seat_seconds = time.perf_counter() - t_seat
        n, base = len(ids), user.pos
        num_chunks = -(-n // self.chunk_size)
        chunk_times: list[tuple[int, float]] = []
        t0 = time.perf_counter()

        def on_chunk(_i, start, end, seconds):
            chunk_times.append((end - start, seconds))
            if on_progress is not None:
                on_progress(max(0, end - base))  # the status pane reads this from another thread

        try:
            try:
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
                logits, states = self._prefill(user, ids_t, on_chunk, replayed)
            finally:
                self.t_prefill_out = time.perf_counter()
                model.reload_resident_state()
        except Exception:
            if base:
                with contextlib.suppress(Exception):
                    user.reset()
            raise
        prefill_seconds = self.t_prefill_out - t0
        t1 = time.perf_counter()
        reload_seconds = t1 - self.t_prefill_out
        try:
            committed = self._commit(user, states)
        except Exception:
            with contextlib.suppress(Exception):
                user.reset()
            raise
        finally:
            prefill.free_traced_states(states)
        user.pos = committed
        commit_seconds = time.perf_counter() - t1
        total_seconds = time.perf_counter() - t0
        steady = chunk_times[1:] or chunk_times
        steady_seconds = sum(seconds for _, seconds in steady)
        piped = f"{sum(t for t, _ in steady) / steady_seconds:.1f}" if steady_seconds else "-"
        if len(chunk_times) > 1:
            self.chunk_seconds = 0.5 * self.chunk_seconds + 0.5 * steady_seconds / len(steady)
        fill_seconds = max(total_seconds - (num_chunks - 1) * self.chunk_seconds, 0.0)
        self.fill_seconds = 0.5 * self.fill_seconds + 0.5 * fill_seconds
        how = f" continuing from {base}" if base else ""
        if replayed:
            how += f" after replaying the {replayed}-token window before it"
        logger.info(
            f"prefill model (user {user.index}): {n - replayed} tokens{how} in "
            f"{prefill_seconds:.2f}s = "
            f"{n / prefill_seconds:.1f} tok/s, {num_chunks} chunk(s) of up to {self.chunk_size} through "
            f"{self.num_stages} stage(s), {piped} tok/s pipelined"
        )
        to_prefill = drain_seconds + seat_seconds
        to_decode = reload_seconds + commit_seconds
        logger.info(
            f"prefill <-> decode switch (user {user.index}): decode -> prefill {1e3 * to_prefill:.1f} ms "
            f"(waiting for {drained} decode step(s) in flight {1e3 * drain_seconds:.1f} ms, seating the user "
            f"{1e3 * seat_seconds:.1f} ms); prefill -> decode {1e3 * to_decode:.1f} ms (decode's L1 written back "
            f"+ prefetcher fenced {1e3 * reload_seconds:.1f} ms, state commit {1e3 * commit_seconds:.1f} ms); "
            f"host time, the device's share shows in the first decode output"
        )
        if logits is None:  # a KV-only prefill model has no head: decode replays the prompt's tail
            return None
        row = logits.reshape(-1)
        if not torch.isfinite(row).all():
            raise RuntimeError("prefill produced non-finite logits")
        return row


class _Turn:
    """One in-flight chat turn: the request's decode state plus its output channel.

    Created by the HTTP thread, then owned by the :class:`_Scheduler` thread, which is
    the only one allowed to touch the model or the :class:`UserSession`. The two
    threads meet at :attr:`events` (the reply, delta by delta) and :attr:`cancelled`
    (the client hung up or asked to stop), both of which are thread-safe.

    A turn starts in the ``prefill`` phase, feeding its prompt tokens, and moves to
    ``decode`` once the last of them has been fed. A long prompt has :attr:`device_prefill`
    of its tokens run by the traced prefill model in one go and only the rest replayed
    through decode: from position 0 for a fresh session, or, continuing one, after
    :attr:`prefill_from` tokens replayed to bring the cache to a position the prefill model
    continues from.
    :attr:`pending` records the steps
    this turn has in flight, in dispatch order, so the scheduler knows how many
    outputs to collect for it and whether each one is a prompt token's (discarded) or
    a generated token's.
    """

    PREFILL = "prefill"
    DECODE = "decode"

    def __init__(
        self,
        user_key: str,
        slot: int,
        body: dict,
        sampler,
        max_tokens: int,
        rate_window: int = DECODE_RATE_WINDOW,
    ):
        self.user_key = user_key
        self.slot = slot
        self.previous_owner: str | None = None  # who held the slot before this turn (see _Lease)
        self.restore: _HostConversation | None = None  # copied into the slot at admission, then None
        self.body = body
        self.sampler = sampler
        self.max_tokens = max_tokens
        # The turn this request adds. After normalisation it is always a user message,
        # though it may carry tool results as content blocks rather than plain text.
        self.message = body["messages"][-1]

        self.events: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()

        self.phase = self.PREFILL
        self.ids: list[int] = []  # the turn's prompt tokens
        self.cached = 0  # the prompt's tokens before ``ids``, already in the slot's cache at admission
        self.fed = 0  # how many of them have been dispatched
        self.device_prefill = 0  # prompt tokens still to go through the traced prefill model
        self.prefill_from = 0  # ... from this many fed (replayed through decode up to an aligned position)
        self.prefilled_on_device = 0  # ... and how many it took, once it has run
        # A conversation resumed from an earlier turn (see _Scheduler._resume): the cached tokens before the
        # rewound position that the prefill model replays first, to rebuild the sliding window.
        self.replay_ids: list[int] = []
        self.resumed_from = 0  # ... the position it rewound to (0: not resumed)
        self.rebuilt = False  # a continued conversation rewound and re-prefilled whole, since that was faster
        self.t_replay = 0.0  # when this turn's prompt tokens started going through decode
        self.next_id: int | None = None  # produced but not yet fed back
        self.t_read = 0.0  # when this turn's latest output came off the socket
        self.t_prefill_out = 0.0  # when its traced prefill's logits came back, until its first decode output does
        self.generated: list[int] = []
        self.pending: deque[bool] = deque()  # per in-flight step: is it the last prompt token?
        self.stream: _Streamer | None = None
        self.hit_cap = False
        # Set when a dispatch fails. The turn stays in the round-robin until its steps
        # already in flight have been read back, because the outputs of every other turn
        # queued behind them can only be collected in order.
        self.error: Exception | None = None

        # Timings, for the status pane and the debug log. Prefill and decode are measured
        # separately because they cost very differently per token: a prompt's tokens
        # pipeline ``prefill_chunk`` at a time, a reply's are one per round.
        self.t_submit = time.perf_counter()
        self.t_admit = 0.0
        self.t_prefill_done = 0.0
        self.t_done = 0.0
        # Rate of the latest prefill chunk only (the TUI shows this, not the cumulative mean).
        self.prefill_chunk_rate = 0.0
        self._chunk_t = 0.0
        # One mark per generated token, for the trailing-window rate. Bounded, so the
        # window slides rather than growing into a whole-reply average.
        self._marks: deque[float] = deque(maxlen=max(2, rate_window) + 1)

    @property
    def prompt_left(self) -> int:
        return len(self.ids) - self.fed

    @property
    def prefill_seconds(self) -> float:
        end = self.t_prefill_done or time.perf_counter()
        return max(end - self.t_admit, 0.0) if self.t_admit else 0.0

    @property
    def decode_seconds(self) -> float:
        if not self.t_prefill_done:
            return 0.0
        return max((self.t_done or time.perf_counter()) - self.t_prefill_done, 0.0)

    def mark_prefill_chunk(self, done: int) -> None:
        """Record that ``done`` prompt tokens are prefilled; sets :attr:`prefill_chunk_rate`
        from the chunk just finished (not the cumulative average)."""
        now = time.perf_counter()
        start = self._chunk_t or self.t_admit or now
        if done > self.fed and now > start:
            self.prefill_chunk_rate = (done - self.fed) / (now - start)
        self._chunk_t = now
        self.fed = done

    def mark_token(self) -> None:
        """Record that a generated token was dispatched, for :attr:`decode_rate`."""
        self._marks.append(time.perf_counter())

    @property
    def decode_rate(self) -> float:
        """Tokens per second over the last few generated tokens, for this user alone.

        The trailing window is what the turn is decoding at *now*, which is the useful
        number both for the status pane and for judging the effect of a cold cache: a
        turn admitted at position 0 climbs to its steady rate part-way through the
        reply (see :data:`DECODE_RATE_WINDOW`), and this shows that instead of
        flattening it. Falls back to the mean until two tokens have been marked.
        """
        if len(self._marks) < 2:
            return self.mean_decode_rate
        span = self._marks[-1] - self._marks[0]
        return (len(self._marks) - 1) / span if span > 0 else 0.0

    @property
    def mean_decode_rate(self) -> float:
        """Tokens per second averaged over the whole reply, for this user alone."""
        seconds = self.decode_seconds
        return len(self.generated) / seconds if seconds > 0 else 0.0

    def status(self) -> dict:
        """A snapshot for the status pane (read from another thread, so keep it cheap)."""
        return {
            "user": self.user_key,
            "slot": self.slot,
            "phase": self.phase,
            "prompt_tokens": len(self.ids),
            "prefilled": self.fed,
            "generated": len(self.generated),
            "max_tokens": self.max_tokens,
            "prefill_seconds": self.prefill_seconds,
            "prefill_rate": self.prefill_chunk_rate,
            "decode_rate": self.decode_rate,
            "cancelled": self.cancelled.is_set(),
        }


class _OutputReader:
    """Reads traced decode outputs off the device->host socket on a thread of its own.

    The socket FIFO holds a single output page, so with several steps in flight the last
    stage can only move on once the host has taken the previous output -- and the host may
    meanwhile be blocked writing the next step's packet. Reading on a separate thread keeps
    the device draining whatever the scheduler is doing, as ``_OutputReader`` in
    ``tests/decode/test_multi_user_paged_decode_demo.py`` does. :meth:`expect` is called once
    per step, in dispatch order; :attr:`results` yields ``(ticket, out)`` -- fp32 logits ``[1, V]``, or under
    device sampling the int32 token page ``[1, words]`` -- in that
    order, or ``(ticket, exception)`` once a read fails, after which it stops; each item also carries the
    ``time.perf_counter()`` at which the read returned, and the token :meth:`arm` had it feed (else None) with
    the time its packet was written.

    :meth:`arm` takes the scheduler off a lone decoding turn's token path: that turn's next step is posted
    before its token is known (:meth:`_Scheduler._post_ahead`), and whatever the turn does next -- go on, or
    end -- that step is fed the token sampled from the newest step's output, at its position. So this thread
    samples it and writes the packet itself, as soon as the output is off the socket, the way
    ``test_full_model_decode_demo.py`` keeps its whole loop on one thread; the scheduler only does the
    book-keeping afterwards. The step it fed is expected here under :attr:`PREFED`, whose owner is whoever
    the scheduler puts at the head of its in-flight order.
    """

    _STOP = object()
    PREFED = object()

    def __init__(self, model, packet_lock):
        self._model = model
        self._packet_lock = packet_lock
        self._cv = threading.Condition()
        self._tickets: deque[list] = deque()  # [ticket, (sampler, pos) to feed, or None]
        self._reading: list | None = None  # the entry whose read is in progress
        self.results: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="decode-reader", daemon=True)
        self._thread.start()

    def expect(self, ticket) -> None:
        with self._cv:
            self._tickets.append([ticket, None])
            self._cv.notify()

    def arm(self, sampler, pos: int) -> bool:
        """Have the newest expected step's output feed the step posted at ``pos`` its token (``sampler``, or
        argmax when None). False if that output is already read, and the scheduler feeds the step itself."""
        with self._cv:
            entry = self._tickets[-1] if self._tickets else self._reading
            if entry is None:
                return False
            entry[1] = (sampler, pos)
            return True

    def close(self) -> None:
        self.expect(self._STOP)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._tickets:
                    self._cv.wait()
                entry = self._reading = self._tickets.popleft()
            ticket = entry[0]
            if ticket is self._STOP:
                return
            try:
                out = self._model.read_decoded_output().reshape(1, -1)
                # A device-sampled step sends an int32 page with the token first; casting it would turn
                # it into "logits" whose argmax is column 0, i.e. token 0.
                if out.dtype != torch.int32:
                    out = out.float()
            except BaseException as e:  # noqa: BLE001 - handed to the scheduler, which aborts
                self.results.put((ticket, e, time.perf_counter(), None, 0.0))
                return
            t_read = time.perf_counter()
            with self._cv:
                self._reading = None
                feed = entry[1]
            if feed is None:
                self.results.put((ticket, out, t_read, None, 0.0))
                continue
            sampler, pos = feed
            try:
                # A device-sampled step's output is already the token: ``sampler`` (the turn's temperature)
                # cannot apply, so it is the one case where a turn's own sampling is not honoured.
                token = decoded_token(out) if sampler is None or out.dtype == torch.int32 else sampler(out)
                with self._packet_lock:
                    self._model.write_step_packet(token, pos)
            except BaseException as e:  # noqa: BLE001 - handed to the scheduler, which aborts
                self.results.put((ticket, e, t_read, None, 0.0))
                return
            t_fed = time.perf_counter()
            with self._cv:
                self._tickets.append([self.PREFED, None])
            self.results.put((ticket, out, t_read, token, t_fed))


class _Scheduler:
    """The one thread that drives the device, generating for every active turn at once.

    Traced decode has two hard constraints: the trace replays and the paged session
    state must be driven from a single thread, and ``read_decoded_output`` returns the
    *oldest* in-flight step, so outputs have to be collected in dispatch order. Both
    are met by keeping all model work here and tracking dispatch order explicitly in
    :attr:`_inflight`; the reads themselves happen on an :class:`_OutputReader`.

    Every active turn keeps its steps in the stage pipeline at once, the way
    ``tests/decode/test_multi_user_paged_decode_demo.py`` runs its batches: a decoding turn
    has one step in flight, and its next is dispatched as soon as that step's token comes
    back, while the other turns' steps are still crossing the stages. A step is seated
    (``activate_sessions`` swaps that user's window state and page tables in, queued on
    the command queue behind the previous user's traces), its traces are posted, then its
    packet is written. With as many decoding users as stages every stage stays busy.

    Prompt tokens replayed through decode have no step-to-step dependency, so a prompt
    turn keeps up to ``prefill_chunk`` of them in flight, topped up as they come back. A
    traced prefill needs the device to itself: it waits until every step in flight has
    been read back, then runs between steps.
    """

    def __init__(self, server: "GenerationServer", prefill_chunk: int = 16, decode_ahead: bool = True):
        self.server = server
        self.engine = server.engine
        self.prefill_chunk = max(1, prefill_chunk)
        self.decode_ahead = decode_ahead
        # The lone decoding turn whose next step's traces are posted ahead of its packet (see _post_ahead).
        self._ahead: _Turn | None = None
        # (token, when written) of the posted step whose packet the reader already wrote (_OutputReader.arm).
        self._prefed: tuple[int, float] | None = None
        self._flushing = False  # in _flush_ahead: post nothing more ahead
        self._new: deque[_Turn] = deque()
        self._chores: deque[tuple] = deque()  # (callable, done event, result box)
        self._active: list[_Turn] = []
        self._inflight: deque[_Turn | None] = deque()  # dispatch order across all turns; None is discarded
        self._reader: _OutputReader | None = None  # started with the scheduler thread
        self._wake = threading.Condition()
        self._stop = False
        self._broken: Exception | None = None  # a failed readback; see _abort_all
        # Slots whose cache ends part-way through a stopped turn's prompt (see _stop_prompt), which no request continues.
        self._cut_short: set[int] = set()
        # Prompt tokens/s replayed through decode, for :meth:`_choose_feed`. Seeded with what the
        # 8 x TP4 layout replays with the prefetcher off; each long enough replay refines it.
        self.replay_rate = _REPLAY_RATE_SEED
        # Counters for the status pane and the periodic debug lines.
        self._pool_seen: dict = {}
        self.rounds = 0
        self.steps = 0
        self._window_t0 = time.perf_counter()
        self._window_steps = 0
        self._window_dispatch = 0.0  # host seconds spent seating, posting and packing steps
        self._window_decode_steps = 0  # generated-token steps dispatched
        self._window_ahead = 0  # ... of which _post_ahead had already posted the traces
        self._window_prefed = 0  # ... of which the output reader had already written the packet
        self._window_gap = 0.0  # host seconds from a turn's output coming back to its next packet
        self.step_rate = 0.0  # steps/s across all users, over the last window
        self._t_started = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="decode-scheduler", daemon=True)

    # -- lifecycle -------------------------------------------------------------- #
    def start(self) -> None:
        self._reader = _OutputReader(self.engine.model, self.engine.packet_lock)
        self._thread.start()

    def stop(self) -> None:
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        self._thread.join(timeout=30)

    def submit(self, turn: _Turn) -> None:
        """Hand a turn to the scheduler; it is admitted at the next round boundary."""
        with self._wake:
            if self._stop:
                raise RequestError(503, "server is shutting down", "server_error")
            if self._broken is not None:
                raise RequestError(503, f"decode is wedged, restart the server: {self._broken}", "server_error")
            self._new.append(turn)
            self._wake.notify_all()

    def run_on_scheduler(self, fn):
        """Run ``fn`` on the scheduler thread, between rounds, and return its result.

        For work that touches the device or a session outside a turn -- a ``/v1/sessions``
        reset frees KV blocks and rewrites page tables -- which must not race with the
        rounds nor land between a step's dispatch and its readback."""
        done = threading.Event()
        box: list = []
        with self._wake:
            if self._stop:
                raise RequestError(503, "server is shutting down", "server_error")
            self._chores.append((fn, done, box))
            self._wake.notify_all()
        done.wait()
        if isinstance(box[0], BaseException):
            raise box[0]
        return box[0]

    @property
    def active_turns(self) -> int:
        return len(self._active)

    def status(self) -> dict:
        """A snapshot of the scheduler for the status pane.

        Read from the TUI thread without a lock: every field is a plain int/float or a
        copy, and a status pane that is one round stale is harmless, whereas locking the
        scheduler to paint a screen would not be.
        """
        active = [turn.status() for turn in list(self._active)]
        return {
            "active": active,
            "rounds": self.rounds,
            "steps": self.steps,
            "step_rate": self.step_rate,
            "per_user_rate": self.step_rate / len(active) if active else 0.0,
            "inflight": len(self._inflight),
            "prefill_chunk": self.prefill_chunk,
            "pool": self.pool_usage(),
            "tokens_left": self.engine.tokens_left(),
            "broken": None if self._broken is None else str(self._broken),
        }

    def _loop(self) -> None:
        try:
            self._serve()
        except BaseException as e:  # noqa: BLE001 - nobody is left to report this to
            logger.exception(f"decode scheduler died: {e}")
            self._broken = e if isinstance(e, Exception) else RuntimeError(str(e))
            self._ahead = self._prefed = None  # nothing more is written to a pipeline in an unknown state
            self._drain(RequestError(503, f"decode scheduler died: {e}", "server_error"))
            raise
        finally:
            self._reader.close()

    def _serve(self) -> None:
        while True:
            with self._wake:
                while not self._stop and not self._new and not self._chores and not self._active:
                    self._wake.wait()
                if self._stop:
                    # No trace may be left parked on in-trace recv when the device closes.
                    self._settle()
                    self._drain(RequestError(503, "server is shutting down", "server_error"))
                    break
                if self._broken is not None:
                    # Nothing more can be generated (submit() rejects new turns too), but a
                    # turn that slipped in before the failure still needs an answer.
                    self._drain(self._broken)
                    continue
                admitting, self._new = list(self._new), deque()
                chores, self._chores = list(self._chores), deque()
            if admitting or chores:
                # Both may rewind a session (free its blocks, rewrite page tables), which
                # must not land between another step's dispatch and its readback.
                self._settle()
            for fn, done, box in chores:
                try:
                    box.append(fn())
                except BaseException as e:  # noqa: BLE001 - handed back to the caller's thread
                    box.append(e)
                finally:
                    done.set()
            for turn in admitting:
                self._admit(turn)
            if self._active:
                self._round()

    # -- admission -------------------------------------------------------------- #
    def _admit(self, turn: _Turn) -> None:
        """Bring a submitted turn into the round-robin: sync its conversation against
        the request and render the tokens this turn adds.

        Runs here rather than on the HTTP thread because a rewind
        (``UserSession.reset``) frees KV blocks and rewrites page tables on the
        device."""
        if turn.cancelled.is_set():
            self._stop_prompt(turn)
            return
        user = self.engine.users[turn.slot]
        turn.t_admit = time.perf_counter()
        continuing = False
        cut_short = turn.slot in self._cut_short
        self._cut_short.discard(turn.slot)
        logger.debug(
            f"admitting user {turn.user_key!r} (slot {turn.slot}, sid {user.sid}): "
            f"queued {turn.t_admit - turn.t_submit:.3f}s, at pos {user.pos}/{self.engine.max_seq}, "
            f"{len(user.messages)} stored messages, {len(self._active)} turns already active"
        )
        try:
            # Before the request's options touch the slot: what it holds is the previous conversation's.
            if not cut_short and not _continues(turn.body["messages"], user.messages):
                self._offload(turn.slot, turn.previous_owner)
            if turn.restore is not None:
                if self._restore(user, turn):
                    cut_short = False
            self.server._apply_options(user, turn.body)
            if cut_short and not self._resume(user, turn):
                user.reset()
            if not turn.resumed_from and (
                cut_short or _continues(turn.body["messages"], user.messages) or not self._resume(user, turn)
            ):
                continuing = self.server._sync_messages(user, turn.body["messages"])
                if continuing:
                    user.messages.append(turn.message)
                    turn.ids = self.server._render_ids(user, include_assistant=False)
                else:
                    user.messages = user.messages + [turn.message]
                    user._next_render = 0
                    turn.ids = self.server._render_ids(user, include_assistant=True)
                self._choose_feed(user, turn, continuing)
            self._make_room(len(turn.ids), turn.slot)
            self._check_capacity(user, turn.ids)
        except Exception as e:  # noqa: BLE001 - reported to the client, the server stays up
            if cut_short:
                self._cut_short.add(turn.slot)
            if turn.rebuilt or turn.resumed_from:
                user.reset()  # the cache was already rewound: start the conversation afresh next time
            elif continuing:
                user.messages.pop()  # nothing was fed: leave the conversation as it was
            logger.debug(f"user {turn.user_key!r} rejected at admission: {type(e).__name__}: {e}")
            turn.events.put(("error", e))
            return
        turn.cached = user.pos
        user._next_render = len(user.messages)
        turn.t_replay = turn.t_admit
        turn.stream = _Streamer(
            self.engine.tokenizer,
            lambda r, c: turn.events.put(("delta", r, c)),
            self.engine.chat_format,
            thinking=user.thinking_mode == "thinking",
        )
        self._active.append(turn)
        if turn.resumed_from:
            how = f"resumed from the cache at {turn.resumed_from}"
        else:
            how = "re-prefill whole conversation" if turn.rebuilt else "continuing" if continuing else "fresh session"
        logger.debug(
            f"user {turn.user_key!r} admitted: prefill {len(turn.ids)} tokens ({how}"
            f"{f', {turn.device_prefill} by the prefill model' if turn.device_prefill else ', all through decode'}), "
            f"cap {turn.max_tokens} new tokens, pool {self._pool_summary()}"
        )

    def _choose_feed(self, user: UserSession, turn: _Turn, continuing: bool) -> None:
        """Pick the faster way to get this turn's prompt into the cache.

        The prefill model takes 128-aligned runs of tokens, and a single chunk already costs
        a pass through every stage, so replaying a short prompt through decode is quicker.
        For a fresh session the choice is how much the prefill model takes
        (:attr:`_Turn.device_prefill`). A continued conversation replays a short new turn
        onto the warm cache. A long one goes through the prefill model on top of the cache
        when the model can continue a session (:attr:`_Prefiller.continue_alignment`): decode
        replays the few tokens up to the next position it continues from
        (:attr:`_Turn.prefill_from`), the prefill model the bulk, decode the ragged rest.
        Otherwise the whole conversation is rewound and re-prefilled, once that is faster.
        """
        prefiller = self.server.prefiller
        if prefiller is None:
            return
        rate = self.replay_rate

        def with_prefill(n_tokens: int) -> tuple[int, float]:
            n = prefiller.aligned(n_tokens)
            return n, (prefiller.seconds(n) + (n_tokens - n) / rate if n else float("inf"))

        def note_overflow(n_tokens: int) -> None:
            if n_tokens > prefiller.max_len:
                logger.info(
                    f"user {turn.user_key!r}: {n_tokens}-token prompt is longer than the {prefiller.max_len} "
                    f"the prefill model was compiled for (--prefill-max-len); replaying the last "
                    f"{n_tokens - prefiller.max_len} through decode at ~{rate:.0f} tok/s"
                )

        replay_seconds = len(turn.ids) / rate
        if not continuing:
            n, seconds = with_prefill(len(turn.ids))
            if seconds < replay_seconds:
                turn.device_prefill = n
                note_overflow(len(turn.ids))
            return
        if replay_seconds <= prefiller.fill_seconds:
            return  # no prefill can beat appending the new turn to the warm cache
        step = prefiller.continue_alignment
        lead = -user.pos % step if step else 0
        n = prefiller.aligned(len(turn.ids) - lead, user.pos + lead) if step and lead < len(turn.ids) else 0
        if n:
            seconds = prefiller.seconds(n) + (len(turn.ids) - n) / rate
            if seconds < replay_seconds:
                logger.debug(
                    f"user {turn.user_key!r}: continuing the cache at {user.pos} with the prefill model: "
                    f"{lead} tokens through decode, {n} by the prefill model, {len(turn.ids) - lead - n} through "
                    f"decode (~{seconds:.1f}s instead of ~{replay_seconds:.1f}s replaying all {len(turn.ids)})"
                )
                turn.device_prefill, turn.prefill_from = n, lead
            return
        next_render, user._next_render = user._next_render, 0
        try:
            full = self.server._render_ids(user, include_assistant=True)
        finally:
            user._next_render = next_render
        n, seconds = with_prefill(len(full))
        if seconds >= replay_seconds or len(full) >= self.engine.max_seq:
            return
        logger.debug(
            f"user {turn.user_key!r}: re-prefilling the {len(full)}-token conversation (~{seconds:.1f}s) "
            f"instead of replaying its {len(turn.ids)} new tokens (~{replay_seconds:.1f}s)"
        )
        conversation = list(user.messages)
        user.reset()
        user.messages = conversation
        user._next_render = 0
        turn.ids = full
        turn.device_prefill = n
        turn.rebuilt = True
        note_overflow(len(full))

    def _resume(self, user: UserSession, turn: _Turn) -> bool:
        """Rewind ``user``'s cache to the latest turn the request still agrees with, instead of to position 0,
        when the request is not a continuation (an edited, truncated or forked history); returns whether it did.

        The resume point is a mark (:attr:`UserSession.marks`): the end of an assistant turn whose messages the
        request repeats and which a non-assistant message follows, so the cache there holds exactly the
        continuation render's prefix. The cache rewinds to ``Q`` at or before the mark, with ``Q - replay_window``
        on the grid the prefill model continues from; the compressed KV before ``Q`` is kept, and the prefill model rebuilds
        the sliding window by replaying the ``replay_window`` cached tokens before ``Q`` (SWA bounded replay),
        then takes the cached tokens from ``Q`` to the mark and the newly rendered messages. Only when that beats
        re-prefilling the whole conversation.
        """
        prefiller = self.server.prefiller
        if prefiller is None or not prefiller.replay_window or not prefiller.continue_alignment:
            return False
        if not user.marks or len(user.tokens) != user.pos:
            return False
        window, step, rate = prefiller.replay_window, prefiller.continue_alignment, self.replay_rate
        messages = turn.body["messages"]
        keys = [_conversation_key(m) for m in messages]
        common = next(
            (i for i, old in enumerate(user.messages[: len(messages)]) if _conversation_key(old) != keys[i]),
            min(len(messages), len(user.messages)),
        )
        mark = next(
            (
                m
                for m in reversed(user.marks)
                if m[0] <= common and m[0] < len(messages) and messages[m[0]].get("role") != "assistant"
            ),
            None,
        )
        if mark is None:
            return False
        n_messages, pos, pending = mark
        start = (pos - window) // step * step  # the replay starts on the prefill's grid, not just Q
        resume_at = start + window
        if start <= 0:
            return False

        stored = (user.messages, user._next_render, user.pending_id)
        user.messages, user._next_render, user.pending_id = list(messages), n_messages, pending
        try:
            rendered = self.server._render_ids(user, include_assistant=True, closing=True)
        finally:
            user.messages, user._next_render, user.pending_id = stored
        ids = user.tokens[resume_at:pos] + rendered
        n = prefiller.aligned(window + len(ids), start, bounded=True)
        if not n:
            return False
        n -= window
        seconds = prefiller.seconds(window + n) + (len(ids) - n) / rate
        whole = pos + len(rendered)  # about what re-prefilling the conversation from 0 renders
        n_whole = prefiller.aligned(whole)
        whole_seconds = prefiller.seconds(n_whole) + (whole - n_whole) / rate if n_whole else whole / rate
        if seconds >= whole_seconds:
            return False

        logger.debug(
            f"user {turn.user_key!r}: resuming the cache at {resume_at} (message {n_messages} of {len(messages)}, "
            f"cached to {user.pos}): {window} cached tokens replayed for the window, {n} by the prefill model, "
            f"{len(ids) - n} through decode (~{seconds:.1f}s instead of ~{whole_seconds:.1f}s re-prefilling)"
        )
        self.engine.model.truncate_session(user.sid, resume_at)
        turn.resumed_from = resume_at
        user.pos = resume_at
        user.pending_id = None
        user.marks = [m for m in user.marks if m[1] <= resume_at and m[0] <= n_messages]
        turn.replay_ids = user.tokens[start:resume_at]
        del user.tokens[resume_at:]
        user.messages = list(messages)
        turn.ids = ids
        turn.device_prefill = n
        turn.prefill_from = 0
        return True

    # -- KV pool reporting ------------------------------------------------------- #
    def pool_usage(self) -> dict:
        """Per-group ``(blocks used, blocks total)`` of the shared page pool, or ``{}``
        before the pool exists. Host-side bookkeeping, so it is cheap enough to poll for
        the status pane."""
        if not getattr(self.engine.model, "paged", None):
            return {}
        try:
            return dict(self.engine.model.session_usage())
        except Exception:  # noqa: BLE001 - status reporting must never break a turn
            return {}

    def _pool_summary(self) -> str:
        usage = self.pool_usage()
        if not usage:
            return "no page pool yet"
        groups = ", ".join(f"{name} {used}/{total} blocks" for name, (used, total) in usage.items())
        return f"{groups}; ~{self.engine.tokens_left()} tokens free"

    def _log_pool_change(self) -> None:
        """Log the pool whenever a group's block count moves, i.e. when a session grew
        its KV pages (a step allocates as its sliding window or a compressor closes)."""
        usage = self.pool_usage()
        if usage != self._pool_seen:
            grew = [
                f"{name} {self._pool_seen.get(name, (0, 0))[0]}->{used}"
                for name, (used, _total) in usage.items()
                if self._pool_seen.get(name, (None, None))[0] != used
            ]
            if grew and self._pool_seen:
                logger.debug(f"KV pages allocated:  {', '.join(grew)} ({self._pool_summary()})")
            self._pool_seen = usage

    # -- host KV tier ------------------------------------------------------------ #
    def _offload(self, slot: int, owner: str | None) -> None:
        """Copy the conversation parked in ``slot`` to host RAM (:meth:`_SlotPool.store`), so a request
        continuing it later restores it instead of re-prefilling. Leaves the slot as it was."""
        pool, user = self.server.pool, self.engine.users[slot]
        # A cut-short slot's cache ends part-way through a prompt, and without every token fed there is no conversation to rebuild.
        if not pool.host_users or not user.pos or len(user.tokens) != user.pos or slot in self._cut_short:
            return
        t0 = time.perf_counter()
        try:
            snapshot = self.engine.model.export_session(user.sid)
        except Exception as e:  # noqa: BLE001 - only the host copy is lost; the conversation re-prefills later
            logger.warning(f"user {owner!r} (slot {slot}): could not copy the session to host RAM: {e}")
            return
        pool.store(
            _HostConversation(
                owner=owner,
                messages=list(user.messages),
                tokens=list(user.tokens),
                marks=list(user.marks),
                pending_id=user.pending_id,
                next_render=user._next_render,
                thinking_mode=user.thinking_mode,
                reasoning_effort=user.reasoning_effort,
                system_prompt=user.system_prompt,
                snapshot=snapshot,
                stored_at=time.time(),
            )
        )
        logger.info(
            f"user {owner!r} (slot {slot}): {user.pos}-token conversation copied to host RAM in "
            f"{time.perf_counter() - t0:.2f}s ({snapshot.nbytes / 2**20:.0f} MiB; {pool.host_summary()})"
        )

    def _restore(self, user: UserSession, turn: _Turn) -> bool:
        """Copy the host conversation ``turn`` continues into its slot (whose own conversation
        :meth:`_offload` has already saved); returns whether it did. On failure the slot is left empty
        and the conversation goes back to host RAM, so the turn re-prefills from its messages."""
        conv, turn.restore = turn.restore, None
        t0 = time.perf_counter()
        user.reset()
        self._make_room(conv.snapshot.pos, turn.slot)
        try:
            self.engine.model.import_session(user.sid, conv.snapshot)
        except Exception as e:  # noqa: BLE001 - the turn falls back to a re-prefill
            logger.warning(f"user {turn.user_key!r} (slot {turn.slot}): could not restore from host RAM: {e}")
            user.reset()
            self.server.pool.store(conv)
            return False
        user.messages = list(conv.messages)
        user.tokens = list(conv.tokens)
        user.marks = list(conv.marks)
        user.pending_id = conv.pending_id
        user._next_render = conv.next_render
        user.thinking_mode = conv.thinking_mode
        user.reasoning_effort = conv.reasoning_effort
        user.system_prompt = conv.system_prompt
        user.pos = conv.snapshot.pos
        logger.info(
            f"user {turn.user_key!r} (slot {turn.slot}): {user.pos}-token conversation restored from host RAM in "
            f"{time.perf_counter() - t0:.2f}s ({self.server.pool.host_summary()})"
        )
        return True

    def _make_room(self, n_tokens: int, keep: int) -> None:
        """Free parked slots other than ``keep``, least recently used first (offloading each to host RAM),
        until the shared block pool has room for ``n_tokens`` more tokens or no parked slot is left."""
        pool = self.server.pool
        while n_tokens > self.engine.tokens_left():
            claimed = pool.claim_parked(exclude=keep)
            if claimed is None:
                return
            slot, owner = claimed
            try:
                self._offload(slot, owner)
                self.engine.users[slot].reset()
                self._cut_short.discard(slot)
            finally:
                pool.release(slot, vacated=True)
            logger.debug(f"freed parked slot {slot} (user {owner!r}) for {n_tokens} tokens: {self._pool_summary()}")

    def _check_capacity(self, user: UserSession, ids: list[int]) -> None:
        engine = self.engine
        if user.pos + len(ids) >= engine.max_seq:
            raise ContextFull(f"user {user.index} needs {user.pos + len(ids)} of {engine.max_seq} tokens")
        if len(ids) > engine.tokens_left():
            raise ContextFull(
                f"the shared cache pool has room for about {engine.tokens_left()} more tokens, "
                f"this turn needs {len(ids)}"
            )

    # -- one round -------------------------------------------------------------- #
    def _round(self) -> None:
        self.rounds += 1
        before = self.steps
        self._round_turns()
        self._account(self.steps - before)
        self._log_pool_change()

    def _account(self, steps: int) -> None:
        """Roll the step counters and, every few seconds, log the aggregate rate.

        ``step_rate`` is steps (tokens fed, prompt or generated) per second across all
        users; dividing by the turns that were active gives the per-user rate."""
        self._window_steps += steps
        elapsed = time.perf_counter() - self._window_t0
        if elapsed < 5.0:
            return
        self.step_rate = self._window_steps / elapsed if elapsed else 0.0
        if self._window_steps:
            active = self._active
            decoding = [t for t in active if t.phase == _Turn.DECODE]
            per_user = f"{self.step_rate / len(active):.2f}" if active else "-"
            logger.debug(
                f"round {self.rounds}: {self.step_rate:.2f} steps/s total, {per_user} steps/s/user "
                f"over {len(active)} active turns ({len(decoding)} decoding), "
                f"{self._window_steps} steps in {elapsed:.1f}s, "
                f"host dispatch {1e3 * self._window_dispatch / self._window_steps:.1f} ms/step"
                + (
                    f"; decode steps: {self._window_ahead}/{self._window_decode_steps} posted ahead, "
                    f"{self._window_prefed} fed by the output reader, "
                    f"token back -> next packet {1e3 * self._window_gap / self._window_decode_steps:.2f} ms"
                    if self._window_decode_steps
                    else ""
                )
            )
        self._window_t0 = time.perf_counter()
        self._window_steps = 0
        self._window_dispatch = 0.0
        self._window_decode_steps = self._window_ahead = self._window_prefed = 0
        self._window_gap = 0.0

    def _round_turns(self) -> None:
        """Dispatch the next steps of every turn that can take one, then read back at least one output.

        A decoding turn takes its next step once the previous one's token is back; a prompt
        turn tops its in-flight steps back up to ``prefill_chunk`` once half of them have
        returned. The read at the end waits for the oldest step in flight only, so the other
        turns' steps stay on the stages while that turn moves on.
        """
        for turn in list(self._active):
            if turn not in self._active:
                continue
            if turn.error is not None:
                if not turn.pending:
                    self._fail(turn, turn.error)  # its in-flight steps have now been read back
                continue
            try:
                if (
                    turn.phase == _Turn.PREFILL
                    and (turn.device_prefill or turn.replay_ids)
                    and turn.fed == turn.prefill_from
                    and not turn.cancelled.is_set()
                ):
                    t_drain = time.perf_counter()
                    drained = len(self._inflight) + len(self.engine._replay_q)
                    if not self._settle():
                        return
                    self._prefill_on_device(turn, time.perf_counter() - t_drain, drained)
                steps = self._plan(turn)
            except _PromptStopped:
                self._stop_prompt(turn)
                continue
            except Exception as e:  # noqa: BLE001 - one bad turn must not stop the others
                if turn.pending:
                    turn.error = e
                else:
                    self._fail(turn, e)
                continue
            if steps:
                self._dispatch(turn, steps)
        if self._inflight:
            try:
                self._collect(block=True)
            except Exception as e:  # noqa: BLE001 - see _abort_all: ordering is lost
                self._abort_all(e)

    def _settle(self) -> bool:
        """Read back every step in flight, e.g. before a traced prefill takes the device, once a step
        posted ahead for a lone decoding turn has been taken (:meth:`_flush_ahead`).

        Returns ``False`` if a read failed, in which case every turn has been aborted."""
        try:
            self._flush_ahead()
            while self._inflight:
                self._collect(block=True)
        except Exception as e:  # noqa: BLE001 - see _abort_all: ordering is lost
            self._abort_all(e)
            return False
        return True

    def _collect(self, block: bool) -> None:
        """Take every output that has come back, first waiting for the oldest one if ``block``.

        Outputs arrive in dispatch order, so each belongs to the turn at the head of
        :attr:`_inflight`. Raises if a read failed.

        Stops after an output whose reader fed the posted step (:attr:`_prefed`): the next output is that
        step's, which is in :attr:`_inflight` only once :meth:`_feed` has taken it."""
        results = self._reader.results
        try:
            item = results.get() if block else results.get_nowait()
        except queue.Empty:
            return
        while True:
            self._take(*item)
            if self._prefed is not None:
                return
            try:
                item = results.get_nowait()
            except queue.Empty:
                return

    def _take(self, turn: _Turn | None, out, t_read: float, fed: int | None, t_fed: float) -> None:
        """Hand one step's logits ``[1, V]``, read off the socket at ``t_read``, to the turn that dispatched it.
        ``fed``: the token the reader sampled from them and fed the posted step at ``t_fed`` (else None)."""
        if turn is _OutputReader.PREFED and self._inflight:
            turn = self._inflight[0]
        assert self._inflight and self._inflight[0] is turn, "decode outputs read out of dispatch order"
        self._inflight.popleft()
        if isinstance(out, BaseException):
            raise RuntimeError(f"reading a decode output failed: {out}") from out
        if fed is not None:
            self._prefed = (fed, t_fed)
        if turn is None:
            return  # a posted trace fed a dummy packet after a failed dispatch
        turn.t_read = t_read
        if turn.t_prefill_out:
            logger.info(
                f"user {turn.user_key!r}: first decode output {1e3 * (t_read - turn.t_prefill_out):.1f} ms after "
                f"the prefill's logits (the prefill -> decode switch, host and device, plus one decode step)"
            )
            turn.t_prefill_out = 0.0
        last_prompt_token = turn.pending.popleft()
        if fed is not None:
            turn.next_id = fed  # what _feed_posted feeds the posted step, even for a failed turn
        if turn.error is not None:
            return
        if fed is None and (turn.phase == _Turn.DECODE or last_prompt_token):
            if turn.sampler is not None and out.dtype != torch.int32:
                turn.next_id = turn.sampler(out)
            else:
                turn.next_id = decoded_token(out)
        if last_prompt_token:
            self._prompt_done(turn)

    def _prompt_done(self, turn: _Turn) -> None:
        """Move a turn whose last prompt token has produced its logits on to decode."""
        turn.phase = _Turn.DECODE
        turn.t_prefill_done = time.perf_counter()
        replayed = len(turn.ids) - turn.prefilled_on_device
        replay_seconds = turn.t_prefill_done - turn.t_replay
        if replayed >= _REPLAY_RATE_MIN_TOKENS and replay_seconds > 0:
            self.replay_rate = 0.5 * self.replay_rate + 0.5 * replayed / replay_seconds
        seconds = turn.prefill_seconds
        rate = f"{len(turn.ids) / seconds:.1f}" if seconds > 0 else "-"
        logger.debug(
            f"user {turn.user_key!r}: prefill of {len(turn.ids)} tokens done in "
            f"{seconds:.2f}s ({rate} tok/s), cache at {self.engine.users[turn.slot].pos} tokens"
        )

    def _prefill_on_device(self, turn: _Turn, drain_seconds: float = 0.0, drained: int = 0) -> None:
        """Run the turn's aligned prompt prefix through the traced prefill model.

        Blocks the scheduler for the whole prefill: every stage of the device is busy with
        this prompt meanwhile, so the other turns' decode resumes once it is done.
        ``drain_seconds`` / ``drained``: the wait for the decode steps in flight before it."""
        user = self.engine.users[turn.slot]
        n, turn.device_prefill = turn.device_prefill, 0
        replay, turn.replay_ids = turn.replay_ids, []
        base = turn.fed
        assert base == turn.prefill_from and not turn.pending, "device prefill before its lead tokens are back"
        logger.debug(
            f"user {turn.user_key!r}: prefilling {n} of {len(turn.ids)} tokens with the prefill model "
            f"from pos {user.pos}{f' after a {len(replay)}-token window replay' if replay else ''} "
            f"({len(self._active) - 1} other turn(s) wait)"
        )
        # The turn's own clock starts here, so the pane's prefill tok/s covers this prefill.
        turn.t_admit = turn.t_admit or time.perf_counter()

        def progress(done: int) -> None:
            turn.mark_prefill_chunk(base + done)
            if turn.cancelled.is_set():
                raise _PromptStopped()

        row = self.server.prefiller.run(
            user,
            replay + turn.ids[base : base + n],
            on_progress=progress,
            drain_seconds=drain_seconds,
            drained=drained,
            replayed=len(replay),
        )
        user.tokens.extend(turn.ids[base : base + n])
        turn.t_prefill_out = self.server.prefiller.t_prefill_out
        turn.fed = base + n
        turn.prefilled_on_device = n
        turn.t_replay = time.perf_counter()
        self.steps += n
        if turn.fed == len(turn.ids):
            # The prefill model has a head of its own, so these are always logits here.
            out = row.reshape(1, -1)
            turn.next_id = turn.sampler(out) if turn.sampler is not None else int(out[0].argmax().item())
            self._prompt_done(turn)

    def _plan(self, turn: _Turn) -> list[tuple[int, int, bool]]:
        """The steps this turn dispatches now, as ``(token_id, pos, last_prompt_token)``.

        Touches no device state, but finishes a turn that is done (EOS, its ``max_tokens``,
        the context, or stopped).
        """
        engine = self.engine
        user = engine.users[turn.slot]
        if turn.phase == _Turn.PREFILL:
            if turn.cancelled.is_set():
                if not turn.pending:
                    self._stop_prompt(turn)
                return []
            room = self.prefill_chunk - len(turn.pending)
            if not turn.prompt_left or room < (self.prefill_chunk + 1) // 2:
                return []
            # Tokens up to a pending device prefill (its lead to an aligned position) go through decode first.
            left = turn.prefill_from - turn.fed if turn.device_prefill or turn.replay_ids else turn.prompt_left
            if left <= 0:
                return []
            if turn.fed in (0, turn.prefill_from + turn.prefilled_on_device):
                logger.debug(
                    f"user {turn.user_key!r}: replaying {left} prompt tokens through decode "
                    f"from pos {user.pos}, up to {self.prefill_chunk} in flight"
                )
            n = min(room, left)
            return [(int(turn.ids[turn.fed + i]), user.pos + i, turn.fed + i + 1 == len(turn.ids)) for i in range(n)]
        if turn.pending:
            return []
        if turn.cancelled.is_set() or turn.next_id == engine.eos_id:
            self._finish(turn)
            return []
        if len(turn.generated) >= turn.max_tokens or user.pos >= engine.max_seq - 1:
            turn.hit_cap = True
            self._finish(turn)
            return []
        turn.generated.append(turn.next_id)
        turn.mark_token()
        if len(turn.generated) % 32 == 0:
            logger.debug(
                f"user {turn.user_key!r}: {len(turn.generated)}/{turn.max_tokens} tokens "
                f"at {turn.decode_rate:.2f} tok/s over the last {DECODE_RATE_WINDOW} "
                f"({turn.mean_decode_rate:.2f} tok/s for the reply so far), "
                f"cache at {user.pos}/{engine.max_seq}"
            )
        return [(int(turn.next_id), user.pos, False)]

    def _dispatch(self, turn: _Turn, steps: list[tuple[int, int, bool]]) -> None:
        """Seat the turn's user, post its steps' traces, then write their packets.

        The seat swap is queued on the command queue behind the previous user's traces,
        so it lands on each stage only once that stage has finished with them. A failure
        marks the turn (see :attr:`_Turn.error`); any trace already posted is fed a dummy
        packet so it does not sit on in-trace recv in front of the next user's packet.
        """
        engine = self.engine
        user = engine.users[turn.slot]
        t0 = time.perf_counter()
        try:
            positions = [pos for _, pos, _ in steps]
            posted_ahead = self._ahead is turn and list(engine._replay_q) == positions
            if posted_ahead:
                self._ahead = None  # posted by the previous step's _post_ahead
            elif engine._replay_q:
                raise RuntimeError(f"steps {list(engine._replay_q)} are posted ahead of this turn's {positions}")
            else:
                engine.post_traced(user, positions)
            for token_id, pos, last_prompt_token in steps:
                t_fed = self._feed(turn, token_id, pos)
                turn.pending.append(last_prompt_token)
                user.tokens.append(token_id)
                user.pos += 1
                if turn.phase == _Turn.PREFILL:
                    turn.fed += 1
                self.steps += 1
            if turn.phase == _Turn.DECODE:
                self._window_decode_steps += 1
                self._window_ahead += posted_ahead
                if turn.t_read:
                    self._window_gap += t_fed - turn.t_read
            # After the packet: the next step's dispatch then runs while the device works on this one.
            self._post_ahead(turn, positions[-1])
        except Exception as e:  # noqa: BLE001 - one bad turn must not stop the others
            turn.error = e
            self._ahead = None
            with contextlib.suppress(Exception):
                while engine._replay_q:
                    self._feed(None, None, engine._replay_q[0])
        finally:
            self._window_dispatch += time.perf_counter() - t0
        if turn.error is None and turn.phase == _Turn.DECODE:
            # Detokenizing re-decodes the whole reply, so it is kept off the path from a token to its packet.
            turn.stream.push(turn.generated)

    def _feed(self, ticket: _Turn | None, token_id: int | None, pos: int) -> float:
        """Write the packet of the oldest posted step, at ``pos``, and expect its output for ``ticket``; returns
        when the packet was written. ``token_id`` None feeds a dummy 0.

        If the reader already wrote it (:attr:`_prefed`) only the step is taken off the posted queue: it was
        fed the token the turn produced last, which is what every caller feeds a posted step."""
        engine = self.engine
        if self._prefed is None:
            engine.write_traced(0 if token_id is None else token_id, pos)
            self._reader.expect(ticket)
            t_fed = time.perf_counter()
        else:
            (fed, t_fed), self._prefed = self._prefed, None
            queued = engine._replay_q.popleft()
            if queued != pos or token_id not in (None, fed):
                raise RuntimeError(f"step ({token_id}, {pos}) is not the ({fed}, {queued}) the output reader fed")
            self._window_prefed += 1
        self._inflight.append(ticket)
        return t_fed

    def _post_ahead(self, turn: _Turn, pos: int) -> None:
        """Post the traces of a lone decoding turn's step after ``pos`` before that step's token is known.

        With one user nothing else is on the stages while a step's traces are dispatched, so posting each one
        just before its packet (as for several users) leaves the device idle for that dispatch on every token;
        posted a step early, the dispatch overlaps the step before. Only for a lone turn: packets are written
        in posting order, so another turn's step would wait behind it.

        A posted step is never fed a dummy packet: a CSA window-closing step retires its window into
        ``prev_*``, so running one twice at a position corrupts the overlap. :meth:`_flush_ahead` lets the
        turn take that step for real before anything else needs the device, and a turn that ends first
        feeds it the token it produced last (:meth:`_feed_posted`), the one the next turn would start with.
        """
        if (
            not self.decode_ahead
            or self._flushing
            or turn.phase != _Turn.DECODE
            or self._active != [turn]
            or self._new
            or self._chores
            or turn.cancelled.is_set()
        ):
            return
        engine = self.engine
        nxt = pos + 1
        # Nothing is posted unpacked, and _plan will give the turn the next step: a token left, and not the
        # last context row.
        if engine._replay_q or len(turn.generated) >= turn.max_tokens or nxt >= engine.max_seq - 1:
            return
        try:
            engine.model.ensure_session_capacity(nxt)
        except Exception as e:  # noqa: BLE001 - a full pool fails the real step, not a speculative one
            logger.debug(f"user {turn.user_key!r}: not posting the next decode step ahead: {e}")
            return
        engine.post_traced(engine.users[turn.slot], [nxt])
        self._ahead = turn
        self._reader.arm(turn.sampler, nxt)

    def _flush_ahead(self) -> None:
        """Let the turn :meth:`_post_ahead` posted a step for take it (or end) without posting another."""
        owner = self._ahead
        if owner is None:
            return
        self._flushing = True
        try:
            while self._ahead is owner and owner.error is None:
                while owner.pending:  # the token the posted step is fed comes back with these
                    self._collect(block=True)
                steps = self._plan(owner)  # a turn that is done feeds the posted step in _retire
                if steps:
                    self._dispatch(owner, steps)
        finally:
            self._flushing = False

    def _feed_posted(self, turn: _Turn) -> None:
        """Feed the step posted ahead for ``turn``, which has ended, its last produced token.

        That token is ``pending_id``, which the user's next turn would feed first at this position, so it
        is fed now instead (its logits discarded) and the pending prefix shrinks by it: an EOS needs nothing
        more, any other token still needs the EOS that closes the reply.
        """
        engine = self.engine
        user = engine.users[turn.slot]
        token = turn.next_id if turn.next_id is not None else engine.eos_id
        if engine._replay_q:
            self._feed(None, token, user.pos)
            user.tokens.append(token)
            user.pos += 1
            self.steps += 1
            user.pending_id = None if token == engine.eos_id else engine.eos_id

    # -- completion ------------------------------------------------------------- #
    def _finish(self, turn: _Turn) -> None:
        """Close out a finished turn: store the assistant message and report the stats.

        Only called with no steps of this turn in flight, so its session is quiescent
        and the next turn for that user can be admitted safely."""
        user = self.engine.users[turn.slot]
        turn.t_done = time.perf_counter()
        turn.stream.close()
        user.pending_id = turn.next_id  # produced but never fed; the next turn starts with it
        assistant: dict = {"role": "assistant", "content": turn.stream.content}
        if turn.stream.reasoning:
            assistant["reasoning_content"] = turn.stream.reasoning
        if turn.stream.tool_calls:
            # Kept in OpenAI shape, which is what both the renderer and a client echoing
            # this turn back use, so the conversation can continue on the warm cache.
            assistant["tool_calls"] = turn.stream.tool_calls
        user.messages.append(assistant)
        user._next_render = len(user.messages)
        self._retire(turn)
        if len(user.tokens) == user.pos:
            user.marks.append((len(user.messages), user.pos, user.pending_id))
        replayed = len(turn.ids) - turn.prefilled_on_device
        if turn.prefilled_on_device:
            how = f"{turn.prefilled_on_device} by the prefill model + {replayed} through decode"
        else:
            how = "all through decode"
        if turn.rebuilt:
            how += ", whole conversation re-prefilled"
        if turn.resumed_from:
            how += f", resumed from the cache at {turn.resumed_from}"
        prefill_rate = len(turn.ids) / turn.prefill_seconds if turn.prefill_seconds > 0 else 0.0
        logger.info(
            f"user {turn.user_key!r} (slot {user.index}): prefill {len(turn.ids)} tokens ({how}) in "
            f"{turn.prefill_seconds:.2f}s = {prefill_rate:.1f} tok/s, decoded {len(turn.generated)} at "
            f"{turn.mean_decode_rate:.2f} tok/s mean, {turn.decode_rate:.2f} tok/s at the end, "
            f"context {user.pos}/{self.engine.max_seq}, finish={'length' if turn.hit_cap else 'stop'}"
            f"{', stopped' if turn.cancelled.is_set() else ''}, {len(self._active)} turns still active"
        )
        logger.debug(f"user {turn.user_key!r} finished: pool {self._pool_summary()}")
        turn.events.put(
            (
                "done",
                {
                    "prompt_tokens": len(turn.ids),
                    "cached_tokens": turn.cached,
                    "completion_tokens": len(turn.generated),
                    "reasoning_tokens": turn.stream.reasoning_tokens(turn.generated),
                    "content": turn.stream.content,
                    "reasoning_content": turn.stream.reasoning,
                    "tool_calls": turn.stream.tool_calls,
                    "finish_reason": ("length" if turn.hit_cap else "tool_calls" if turn.stream.tool_calls else "stop"),
                },
            )
        )

    def _stop_prompt(self, turn: _Turn) -> None:
        """End a turn stopped before its prompt was in, with an empty reply.

        Only called with no steps of this turn in flight. An admitted turn's cache ends part-way through its
        prompt (or, after a stopped traced prefill, wherever the rewind of :meth:`_Prefiller.run` left it), and
        its stored conversation already holds the prompt's message, so the slot goes into :attr:`_cut_short`."""
        if turn in self._active:
            self._cut_short.add(turn.slot)
        self._retire(turn)
        turn.t_done = time.perf_counter()
        logger.info(
            f"user {turn.user_key!r} (slot {turn.slot}): stopped after {turn.fed}/{len(turn.ids)} prompt tokens, "
            f"{len(self._active)} turns still active"
        )
        turn.events.put(
            (
                "done",
                {
                    "prompt_tokens": len(turn.ids),
                    "cached_tokens": turn.cached,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [],
                    "finish_reason": "stop",
                },
            )
        )

    def _fail(self, turn: _Turn, exc: Exception) -> None:
        """Drop a turn whose step raised. Its session keeps whatever was written; the
        client is told, and the remaining turns keep generating."""
        logger.warning(f"turn for user {turn.user_key!r} failed: {exc}")
        self._retire(turn)
        turn.events.put(("error", exc))

    def _abort_all(self, exc: Exception) -> None:
        """Give up on every active turn after a failed readback.

        A read that fails leaves an unknown number of steps in flight, so which output
        belongs to which turn can no longer be established and no turn can be trusted to
        continue. Generation stops here rather than serving one user another user's
        logits; the process must be restarted (a wedged device would not recover anyway).
        """
        logger.error(f"decode output readback failed, abandoning {len(self._active)} turns: {exc}")
        self._broken = exc
        self._ahead = self._prefed = None  # nothing more is written to a pipeline whose outputs are lost
        for turn in list(self._active):
            self._fail(turn, exc)
        self._inflight.clear()

    def _drain(self, exc: Exception) -> None:
        """Release everyone still waiting on a reply, on shutdown."""
        with self._wake:
            pending, self._new = list(self._new), deque()
        for turn in list(self._active) + pending:
            self._retire(turn)
            turn.events.put(("error", exc))

    def _retire(self, turn: _Turn) -> None:
        if self._ahead is turn:
            self._ahead = None
            self._feed_posted(turn)
        if turn in self._active:
            self._active.remove(turn)


class GenerationServer:
    """The resident model plus the user/session bookkeeping behind the HTTP layer.

    One :class:`ChatEngine` (weights, RoPE tables, decode traces) serves every request.
    The engine claimed one paged KV session per ``--num-users`` at startup, and
    :class:`_SlotPool` hands those out a turn at a time, preferring the slot that already
    holds the conversation the request continues.

    Turns run concurrently: an HTTP thread validates the request, claims a slot, waits in the
    run queue for one of the ``--max-running`` seats, hands a :class:`_Turn` to the
    :class:`_Scheduler` and then does nothing but relay the reply as it arrives, while the
    scheduler thread interleaves every active turn's decode steps on the device.
    """

    def __init__(
        self,
        engine: ChatEngine,
        model_id: str,
        max_body_bytes: int = 16 << 20,
        prefill_chunk: int = 16,
        prefiller: _Prefiller | None = None,
        decode_ahead: bool = True,
        temperature: float | None = None,
        host_users: int = 0,
        host_bytes: int = 0,
        max_running: int = 0,
        slot_idle_timeout: float = 0.0,
    ):
        self.engine = engine
        self.model_id = model_id
        self.temperature = temperature  # for requests that send no ``temperature``
        self.max_body_bytes = max_body_bytes
        self.prefiller = prefiller
        if host_users and not hasattr(engine.model, "export_session"):
            logger.info(f"{type(engine.model).__name__} cannot copy a session to host RAM: host KV tier off")
            host_users = 0
        self.pool = _SlotPool(engine, host_users, host_bytes, max_running, slot_idle_timeout)
        self._created = int(time.time())
        self.scheduler = _Scheduler(self, prefill_chunk, decode_ahead)
        self._turns: dict[str, _Turn] = {}  # running turns by request id, for cancel()
        self._turns_lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------------- #
    def start(self) -> None:
        self.scheduler.start()

    def stop(self) -> None:
        self.pool.close()  # release anyone waiting for a seat before the threads go
        self.scheduler.stop()

    # -- status ----------------------------------------------------------------- #
    def stats(self) -> dict:
        """Everything the status pane shows, in one snapshot (see :mod:`demo.tui`)."""
        stats = self.scheduler.status()
        stats.update(
            {
                "model_id": self.model_id,
                "uptime": time.time() - self._created,
                "slots": len(self.engine.users),
                "max_seq": self.engine.max_seq,
                "users": self.pool.rows(),
                "slot_states": self.pool.counts(),
                "slot_idle_timeout": self.pool.idle_timeout,
                "host": dict(zip(("users", "bytes"), self.pool.host_usage())),
                "host_limits": {"users": self.pool.host_users, "bytes": self.pool.host_bytes},
            }
        )
        return stats

    # -- users / sessions ------------------------------------------------------- #
    def reset_user(self, user_key: str) -> None:
        """Rewind the conversations ``user_key`` last held, and their KV pages.

        Reserves each slot first, so it waits for a turn in flight rather than pulling
        the pages out from under it. Its conversations in host RAM are dropped too."""
        dropped = self.pool.drop_host(user_key)
        if dropped:
            logger.debug(f"reset for {user_key!r}: dropped {dropped} conversation(s) from host RAM")
        slots = self.pool.slots_of(user_key)
        if not slots:
            logger.debug(f"reset for {user_key!r}: it holds no slot")
            return
        for slot in slots:
            self.pool.reserve(slot)
            try:
                before = self.engine.users[slot].pos
                self.scheduler.run_on_scheduler(self.engine.users[slot].reset)
                logger.debug(
                    f"user {user_key!r} (slot {slot}) reset: {before} tokens released, "
                    f"pool {self.scheduler._pool_summary()}"
                )
            finally:
                self.pool.release(slot)

    def session_rows(self) -> list[dict]:
        return self.pool.rows()

    def cancel(self, request_id: str) -> bool:
        """Stop the turn generating ``request_id``; returns whether one was running."""
        with self._turns_lock:
            turn = self._turns.get(request_id)
        if turn is None:
            return False
        logger.info(f"user {turn.user_key!r}: stop requested for {request_id}")
        turn.cancelled.set()
        return True

    def cancel_user(self, user_key: str) -> int:
        """Stop every turn generating for ``user_key``; returns how many there were."""
        with self._turns_lock:
            turns = [turn for turn in self._turns.values() if turn.user_key == user_key]
        if turns:
            logger.info(f"user {user_key!r}: stop requested for {len(turns)} turn(s)")
        for turn in turns:
            turn.cancelled.set()
        return len(turns)

    def cancel_all(self) -> int:
        """Stop every running turn (the console's ``k`` key); returns how many there were."""
        with self._turns_lock:
            turns = list(self._turns.values())
        logger.info(f"stop requested for all {len(turns)} running turn(s)")
        for turn in turns:
            turn.cancelled.set()
        return len(turns)

    # -- conversation sync ------------------------------------------------------ #
    def _sync_messages(self, user: UserSession, messages: list[dict]) -> bool:
        """Align the session's stored conversation with the request's history.

        Returns whether the request is a continuation. When the stored conversation
        is a strict prefix of the incoming messages the request continues it: only
        the new user turn will be fed -- the KV pages already hold the past.
        Anything else (a first turn, a foreign client, an edited history) rewinds
        the session so the next turn re-prefills from the request, keeping the
        session in step with whatever the client sends.
        """
        if _continues(messages, user.messages):
            user.messages.extend(messages[len(user.messages) : -1])
            return True
        if user.messages:
            differs = next(
                (
                    i
                    for i, (new, old) in enumerate(zip(messages, user.messages))
                    if _conversation_key(new) != _conversation_key(old)
                ),
                None,
            )
            why = (
                f"message {differs} ({messages[differs].get('role')}) differs from the cached one"
                if differs is not None
                else f"the request has {len(messages)} messages, the cache holds {len(user.messages)}"
            )
            logger.debug(f"user {user.index}: not a continuation ({why}); rewinding the cache")
        user.reset()
        user.messages = list(messages[:-1])
        user._next_render = 0
        return False

    # -- generation ------------------------------------------------------------- #
    def normalized_messages(self, body: dict) -> list[dict]:
        """The request's ``messages`` validated and in the renderer's form (see :func:`_normalized_messages`).
        Runs on the HTTP thread, so it must not touch the device."""
        return _normalized_messages(body.get("messages"), body.get("tools"), self.engine.chat_format)

    def generate(self, user_key: str, body: dict, on_chunk, request_id: str | None = None, client_gone=None) -> dict:
        """Run one chat-completion turn for ``user_key``.

        ``on_chunk(reasoning, content)`` receives the reply's new text as it resolves
        (either argument can be empty); raising ``_ClientGone`` aborts the turn, keeping
        the reply generated so far. So does :meth:`cancel` of ``request_id``, or
        ``client_gone()`` returning true, which is polled while the turn runs.

        Blocks until the reply is complete, but holds nothing except this user's own
        lock: the decode steps run on the scheduler thread, interleaved with the other
        users' turns, and this thread only relays what they produce."""
        messages = self.normalized_messages(body)
        body = {**body, "messages": messages}

        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.engine.max_new_tokens
        else:
            try:
                max_tokens = int(max_tokens)
            except (TypeError, ValueError):
                raise RequestError(400, "max_tokens must be a positive integer")
            if max_tokens < 1:
                raise RequestError(400, "max_tokens must be a positive integer")

        # A request may override the server-wide default, unless the device is sampling (see below).
        request_temperature, request_top_p = body.get("temperature"), body.get("top_p")
        temperature = self.temperature if request_temperature is None else request_temperature
        sampler = _make_sampler(temperature, request_top_p)
        if self.engine.model.device_sampler is not None:
            if request_temperature is not None or request_top_p is not None:
                # The sampler lives in the captured decode trace, so it has one temperature for every
                # turn; a request cannot bring its own.
                raise RequestError(
                    400,
                    "this server samples on the device (--device-sampling), so a request cannot set "
                    "temperature/top_p; set --temperature for the whole server instead",
                )
            # The engine already pointed the device sampler at the server's temperature, and its PRNG
            # carries from step to step, so nothing is sampled or re-seeded on the host.
            sampler = None
        # A slot is claimed per turn, not per user, so two requests run concurrently even
        # when they carry the same ``user`` (or none). Held until the reply is complete:
        # the turn decodes into this slot's KV session.
        lease = self.pool.acquire(user_key, messages)
        request_id = request_id or uuid.uuid4().hex
        turn = None
        seated = False
        try:
            turn = _Turn(user_key, lease.slot, body, sampler, max_tokens)
            turn.previous_owner, turn.restore = lease.previous, lease.restore
            with self._turns_lock:
                self._turns[request_id] = turn
            counts = self.pool.counts()
            if counts["running"] >= counts["max_running"] or counts["queued"]:
                logger.info(
                    f"user {user_key!r} (slot {lease.slot}) queued: {counts['running']}/{counts['max_running']} "
                    f"running, {counts['queued']} ahead"
                )
            seated = self.pool.wait_to_run(
                lease.slot, lambda: turn.cancelled.is_set() or (client_gone is not None and client_gone())
            )
            if not seated:
                # Given up in the queue: the scheduler retires a cancelled turn at admission without running it.
                turn.cancelled.set()
            self.scheduler.submit(turn)
            return self._relay(turn, on_chunk, client_gone)
        finally:
            if seated:
                self.pool.finish_run(lease.slot)
            with self._turns_lock:
                self._turns.pop(request_id, None)
            # A host conversation the turn never got to restore (rejected or cancelled before admission) stays in host RAM.
            restore = lease.restore if turn is None else turn.restore
            if restore is not None:
                self.pool.store(restore)
            self.pool.release(lease.slot)

    @staticmethod
    def _relay(turn: _Turn, on_chunk, client_gone=None) -> dict:
        """Forward a turn's events to ``on_chunk`` until it finishes.

        A client that hangs up (``_ClientGone`` out of ``on_chunk``, or ``client_gone()``,
        polled every ``_CLIENT_POLL_SECONDS``) cancels the turn but does not abandon it:
        the loop keeps draining, silently, until the scheduler reports the turn done, so
        the session is left consistent and its steps in flight are still read off the
        socket in order."""
        polled = time.monotonic()
        while True:
            try:
                kind, *payload = turn.events.get(timeout=_CLIENT_POLL_SECONDS)
            except queue.Empty:
                kind = None
            now = time.monotonic()
            if client_gone is not None and not turn.cancelled.is_set() and now - polled >= _CLIENT_POLL_SECONDS:
                polled = now
                if client_gone():
                    logger.info(f"user {turn.user_key!r}: client disconnected, stopping its turn")
                    turn.cancelled.set()
            if kind is None:
                continue
            if kind == "delta":
                if turn.cancelled.is_set():
                    continue
                try:
                    on_chunk(*payload)
                except _ClientGone:
                    turn.cancelled.set()
            elif kind == "error":
                raise payload[0]
            else:  # done
                return payload[0]

    @staticmethod
    def _apply_options(user: UserSession, body: dict) -> None:
        if "thinking" in body:
            user.thinking_mode = "thinking" if _thinking_requested(body.get("thinking")) else "chat"
        effort = body.get("reasoning_effort")
        if effort is not None:
            efforts = user.engine.chat_format.reasoning_efforts
            if effort not in efforts:
                raise RequestError(400, f"reasoning_effort must be one of {list(efforts)}, but got {effort!r}")
            user.reasoning_effort = effort

    def _render_ids(self, user: UserSession, include_assistant: bool, closing: bool | None = None) -> list[int]:
        """Tokens for the not-yet-encoded part of ``user.messages``.

        With ``include_assistant`` (a fresh re-prefill) every message is rendered,
        past assistant turns included. Otherwise (a continuation) assistant turns
        are skipped -- the model's own tokens are in the cache already -- and the
        turn starts with the pending token that closed the previous one. ``closing``
        overrides whether it does (a resumed conversation renders the assistant turns
        after its resume point and still closes the cached one). A render from the
        first message starts with BOS."""
        engine = user.engine
        first_turn = user._next_render == 0
        rendered = "".join(
            engine.chat_format.render(i, user.messages, user.thinking_mode, user.reasoning_effort)
            for i in range(user._next_render, len(user.messages))
            if include_assistant or user.messages[i]["role"] != "assistant"
        )
        ids = list(engine.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if first_turn:
            # The tokenizer does not add BOS (V4.1's add_bos_token is false); encode_messages prepends it.
            ids.insert(0, engine.tokenizer.bos_token_id)
        if closing is None:
            closing = not include_assistant
        if not closing:
            return ids
        prefix: list[int] = []
        if user.pending_id is not None:
            prefix.append(user.pending_id)
            if user.pending_id != engine.eos_id:
                prefix.append(engine.eos_id)  # close an assistant turn that hit the token cap
        return prefix + ids


class _Handler(BaseHTTPRequestHandler):
    """One HTTP connection, served on its own thread by :class:`_ModelHTTPServer`.

    The thread does no device work: it validates the request, hands a turn to the
    scheduler and relays the reply, so requests from different users generate
    concurrently (see :class:`_Scheduler`).
    """

    protocol_version = "HTTP/1.1"
    server_version = "DeepSeekV4Flash/1.0"
    # Whether a stream's last chunk carries :meth:`_usage`; with ``stream_options.include_usage`` every earlier
    # chunk then has ``"usage": null``.
    stream_usage = False

    @property
    def api(self) -> GenerationServer:
        return self.server.api  # type: ignore[attr-defined]

    def _usage(self, stats: dict) -> dict:
        """The response's ``usage``, from the stats :meth:`GenerationServer.generate` returns."""
        return {
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
        }

    def log_message(self, fmt: str, *args) -> None:
        logger.info(f"{self.address_string()} {fmt % args}")

    # -- response plumbing ------------------------------------------------------ #
    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, exc: Exception) -> None:
        status, message, err_type = _error_parts(exc)
        logger.warning(f"{self.command} {self.path} -> {status} {err_type}: {message}")
        self._send_json({"error": {"message": message, "type": err_type, "code": status}}, status)

    def _client_gone(self) -> bool:
        """Whether the client has closed the connection: a peek that reads EOF, or a dead socket."""
        try:
            return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except BlockingIOError:
            return False
        except OSError:
            return True

    def _sse_start(self, request_id: str) -> None:
        self.send_response(200)
        self.send_header("X-Request-Id", request_id)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        # The client reads the stream to EOF, so the connection must close rather
        # than be reused for another request.
        self.close_connection = True

    def _sse_event(self, obj: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def _sse_done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # -- request parsing -------------------------------------------------------- #
    def _read_json(self) -> dict:
        length = self.headers.get("Content-Length")
        n = int(length) if length and length.isdigit() else 0
        if n > self.api.max_body_bytes:
            raise RequestError(413, "request body too large")
        raw = self.rfile.read(n) if n else b""
        if not raw.strip():
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise RequestError(400, f"malformed JSON body: {e}") from e

    # -- completions ------------------------------------------------------------ #
    def _completion(self, body: dict, *, chat: bool) -> None:
        """Serve one ``/v1/chat/completions`` (``chat=True``) or ``/v1/completions``
        (legacy, ``chat=False``) request, streaming via SSE when ``stream`` is set."""
        if chat:
            # Validated before a streaming reply commits to a 200 and its SSE header, so
            # a malformed request still gets a real status code. The result is discarded:
            # ``generate`` normalises the body itself, and folding tool turns into user
            # messages is not something to do twice.
            self.api.normalized_messages(body)
            completion_id = "chatcmpl-" + uuid.uuid4().hex
            object_kind, chunk_kind = "chat.completion", "chat.completion.chunk"
        else:
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise RequestError(400, "prompt must be a non-empty string")
            body = {**body, "messages": [{"role": "user", "content": prompt}]}
            completion_id = "cmpl-" + uuid.uuid4().hex
            object_kind, chunk_kind = "text_completion", "text_completion"

        stream = bool(body.get("stream"))
        stream_options = body.get("stream_options") or {}
        if not isinstance(stream_options, dict):
            raise RequestError(400, "stream_options must be an object")
        include_usage = self.stream_usage and bool(stream_options.get("include_usage"))
        created = int(time.time())
        model = str(body.get("model") or self.api.model_id)
        user_key = str(body.get("user") or _DEFAULT_USER)
        client = f"{self.client_address[0]}:{self.client_address[1]}" if self.client_address else "?"
        logger.debug(_describe_request(completion_id, client, user_key, body))

        def frame(delta: dict, finish) -> dict:
            chunk = {
                "id": completion_id,
                "object": chunk_kind,
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if include_usage:
                chunk["usage"] = None
            return chunk

        if stream:
            self._sse_start(completion_id)
            if chat:
                # Sent at once rather than with the first text, so a client has the id to cancel during the prefill.
                try:
                    self._sse_event(frame({"role": "assistant", "content": ""}, None))
                except (BrokenPipeError, ConnectionResetError):
                    return
            stopped = [False]

            def on_chunk(reasoning: str, content: str) -> None:
                if stopped[0] or (not reasoning and not content):
                    return
                if chat:
                    delta: dict = {}
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                    if content:
                        delta["content"] = content
                else:
                    delta = {"text": reasoning + content}
                try:
                    self._sse_event(frame(delta, None))
                except (BrokenPipeError, ConnectionResetError):
                    stopped[0] = True
                    raise _ClientGone()

            try:
                stats = self.api.generate(user_key, body, on_chunk, completion_id, self._client_gone)
            except _ClientGone:
                return
            except Exception as e:
                status, message, err_type = _error_parts(e)
                logger.warning(f"{self.command} {self.path} -> {status} {err_type}: {message}")
                try:
                    self._sse_event({"error": {"message": message, "type": err_type, "code": status}})
                    self._sse_done()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            try:
                # The calls are known only once the block has closed, so they go out as
                # one delta rather than being built up piece by piece across chunks.
                if chat and stats.get("tool_calls"):
                    calls = [{"index": i, **call} for i, call in enumerate(stats["tool_calls"])]
                    self._sse_event(frame({"tool_calls": calls}, None))
                last = frame({}, stats["finish_reason"])
                if self.stream_usage:
                    last["usage"] = self._usage(stats)
                self._sse_event(last)
                self._sse_done()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        try:
            stats = self.api.generate(user_key, body, lambda r, c: None, completion_id, self._client_gone)
        except Exception as e:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self._send_error_json(e)
            return
        if chat:
            message = {"role": "assistant", "content": stats["content"]}
            if stats["reasoning_content"]:
                message["reasoning_content"] = stats["reasoning_content"]
            if stats.get("tool_calls"):
                message["tool_calls"] = stats["tool_calls"]
            choices = [{"index": 0, "message": message, "finish_reason": stats["finish_reason"]}]
        else:
            choices = [
                {"index": 0, "text": stats["content"], "logprobs": None, "finish_reason": stats["finish_reason"]}
            ]
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):  # a client that hung up stopped the turn
            self._send_json(
                {
                    "id": completion_id,
                    "object": object_kind,
                    "created": created,
                    "model": model,
                    "choices": choices,
                    "usage": self._usage(stats),
                }
            )

    # -- routing ---------------------------------------------------------------- #
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in ("/health", "/healthz"):
                self._send_json(
                    {
                        "status": "ok",
                        "model": self.api.model_id,
                        "sessions": len(self.api.engine.users),
                        "slot_states": self.api.pool.counts(),
                        "host_sessions": self.api.pool.host_usage()[0],
                        "active_turns": self.api.scheduler.active_turns,
                        "uptime": time.time() - self.api._created,
                    }
                )
            elif path == "/v1/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": self.api.model_id,
                                "object": "model",
                                "created": self.api._created,
                                "owned_by": _VENDOR,
                            }
                        ],
                    }
                )
            elif path == "/v1/sessions":
                self._send_json({"object": "list", "data": self.api.session_rows()})
            else:
                raise RequestError(404, f"unknown endpoint: GET {path}")
        except Exception as e:
            self._send_error_json(e)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        try:
            if not path.startswith("/v1/sessions/"):
                raise RequestError(404, f"unknown endpoint: DELETE {path}")
            user_key = unquote(path[len("/v1/sessions/") :])
            if not user_key:
                raise RequestError(400, "missing user in path")
            self.api.reset_user(user_key)
            self._send_json({"object": "session", "id": user_key, "deleted": True})
        except Exception as e:
            self._send_error_json(e)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/v1/chat/completions":
                self._completion(body, chat=True)
            elif path == "/v1/completions":
                self._completion(body, chat=False)
            elif path.startswith(("/v1/chat/completions/", "/v1/completions/")) and path.endswith("/cancel"):
                request_id = unquote(path.rsplit("/", 2)[1])
                if not self.api.cancel(request_id):
                    raise RequestError(404, f"no running completion {request_id!r}")
                self._send_json({"id": request_id, "object": "completion.cancel", "cancelled": True})
            elif path.startswith("/v1/sessions/") and path.endswith("/cancel"):
                user_key = unquote(path[len("/v1/sessions/") : -len("/cancel")])
                if not user_key:
                    raise RequestError(400, "missing user in path")
                self._send_json({"object": "session", "id": user_key, "cancelled": self.api.cancel_user(user_key)})
            else:
                raise RequestError(404, f"unknown endpoint: POST {path}")
        except Exception as e:
            self._send_error_json(e)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()


class _ModelHTTPServer(ThreadingHTTPServer):
    """A threaded HTTP server carrying the :class:`GenerationServer`."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, api: GenerationServer):
        super().__init__(address, handler)
        self.api = api


def _add_model_args(p: argparse.ArgumentParser, sys_cfg) -> None:
    """The model/engine flags, identical to ``chat_cli.parse_args``.

    The sizing defaults come from ``sys_cfg``, the machine's system profile (its
    ``server`` variant -- see :mod:`...tt.system_config`), so they follow the hardware
    instead of being restated here.
    """
    from models.experimental.deepseek_v4_flash.tests.decode.test_full_model_decode_demo import _DEFAULT_MODEL_DIR

    decode = sys_cfg.decode
    p.add_argument("--model-dir", default=_DEFAULT_MODEL_DIR, help="HF snapshot (or hub cache) of the checkpoint")
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("DEEPSEEK_V4_CACHE_DIR", "../cache"),
        help="converted ttnn weight-tile cache, reused across runs ('' disables it)",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=decode.num_layers or None,
        help="cap the decoder stack (the full 43 layers do not fit one Blackhole)",
    )
    p.add_argument(
        "--num-users",
        type=int,
        default=decode.num_users,
        help="turns that can generate at once, each with its own KV session (fixed at "
        "startup: their cache blocks cannot be allocated once the traces exist). Their "
        "steps are pipelined through the stages rather than batched: up to one user per "
        "stage adds throughput, past that the users share the stages' rate",
    )
    p.add_argument(
        "--max-context",
        type=int,
        default=decode.max_context,
        help="tokens (all turns) one user's caches are addressed for; rounded up. The "
        "model handles 524288, but this is a throughput knob as much as a capacity one: "
        "a session's opening steps attend through a mask that the kernel walks over the "
        "whole addressed axis, so a reply decoded on a cold cache costs in proportion to "
        "this value until it passes the sliding window (see the profile's comment). "
        "Raise it for long conversations and expect slower first replies",
    )
    p.add_argument(
        "--total-context",
        type=int,
        default=decode.total_context or None,
        help="total tokens the shared block pool holds across all users "
        "(default: --num-users x --max-context, i.e. every user can fill its context)",
    )
    p.add_argument(
        "--max-running",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_MAX_RUNNING", 8)),
        help="turns generating at once (one per pipeline stage keeps every stage busy); further requests "
        "keep their slot and wait in a first-come-first-served queue. Capped at --num-users",
    )
    p.add_argument(
        "--slot-idle-timeout",
        type=float,
        default=float(os.environ.get("DEEPSEEK_V4_SLOT_IDLE_TIMEOUT", 300)),
        help="seconds a parked slot must have been idle since its last request before a new request may take "
        "it, and only when no slot is empty; its conversation then moves to host RAM (see --host-users). "
        "Until then, or with 0 (never take a parked slot), a request finding no empty slot gets a 429",
    )
    p.add_argument(
        "--host-users",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_HOST_USERS", 32)),
        help="conversations kept in host RAM once their device slot is needed by another request "
        "(or its blocks by a longer turn); continuing one copies its KV back instead of re-prefilling. "
        "0 turns the host tier off (models without session export always have it off)",
    )
    p.add_argument(
        "--host-cache-gb",
        type=float,
        default=float(os.environ.get("DEEPSEEK_V4_HOST_CACHE_GB", 32)),
        help="host RAM (GiB) the --host-users conversations may take; the least recently stored go first",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=decode.max_new_tokens,
        help="cap on the tokens generated per reply",
    )
    p.add_argument(
        "--system-prompt",
        default=os.environ.get("DEEPSEEK_V4_SYSTEM_PROMPT", ""),
        help="system prompt prefixed to every user's conversation",
    )
    p.add_argument(
        "--system-prompt-file",
        help="read the system prompt from this file instead of --system-prompt",
    )
    p.add_argument(
        "--think",
        action="store_true",
        help="thinking mode: the reply is preceded by a <think> reasoning block",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=None,
        help="reasoning-effort hint, only meaningful with --think " "(medium uses the same template as high)",
    )
    p.add_argument(
        "--trace-region-size",
        type=int,
        default=sys_cfg.device.trace_region_size,
        help="bytes reserved for the captured traces; with the prefill model on that is decode's plus one prefill "
        "trace per stage and position tier, so raise it if a capture reports the trace region too small",
    )
    p.add_argument("--quiet", action="store_true", help="only warnings and above from the model logs")


def build_parser(argv: list[str] | None = None, description: str = __doc__):
    """The server's argument parser and the system profile its defaults come from.

    A model variant's server (``deepseek_v41_flash/demo/server.py``) changes the defaults on it with
    ``set_defaults`` before parsing, then validates with :func:`check_args`.
    """
    # See ``chat_cli.parse_args``: the profile has to be resolved before the parser is
    # built, since it supplies the sizing defaults. ``variant="server"`` picks the
    # server flavour of whichever machine profile applies.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--system-profile", default=None)
    pre.add_argument("--system-config-file", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    sys_cfg = load_system_config(profile=pre_args.system_profile, path=pre_args.system_config_file, variant="server")

    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--system-profile",
        default=pre_args.system_profile,
        help=f"machine tuning profile to load (default: {sys_cfg.name}); "
        f"see configs/system_configs.yaml, or $DEEPSEEK_V4_SYSTEM_PROFILE",
    )
    p.add_argument(
        "--system-config-file",
        default=pre_args.system_config_file,
        help="alternate profile file (default: configs/system_configs.yaml)",
    )
    _add_model_args(p, sys_cfg)
    p.add_argument(
        "--host", default=os.environ.get("DEEPSEEK_V4_HOST", "0.0.0.0"), help="interface to bind (default: all)"
    )
    p.add_argument(
        "--port", type=int, default=int(os.environ.get("DEEPSEEK_V4_PORT", "8000")), help="TCP port (default: 8000)"
    )
    p.add_argument(
        "--model-id",
        default=os.environ.get("DEEPSEEK_V4_MODEL_ID", "deepseek-v4-flash"),
        help="model id advertised by /v1/models and echoed in completions",
    )
    p.add_argument(
        "--prefill-chunk",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_PREFILL_CHUNK", "16")),
        help="prompt tokens a turn keeps in flight while replaying them through decode (a continued "
        "conversation's new turn, or the ragged tail after the prefill model); higher "
        "prefills faster, lower keeps replies smoother for users already generating",
    )
    p.add_argument(
        "--no-decode-ahead",
        dest="decode_ahead",
        action="store_false",
        help="post each decode step's traces just before its packet even when only one turn is decoding "
        "(by default a lone turn's next step is posted a step early, so its dispatch overlaps the step "
        "before instead of leaving the device idle)",
    )
    p.add_argument(
        "--no-prefill",
        dest="prefill",
        action="store_false",
        help="replay every prompt token through decode instead of running the traced prefill "
        "model (keeps the profile's decode layout and the DRISC prefetcher on)",
    )
    p.add_argument(
        "--prefill-max-len",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_PREFILL_MAX_LEN", "0")),
        help="longest prompt prefix the prefill model takes (rounded down to 128, capped by the "
        "context; 0, the default, = the whole --max-context); the rest is replayed through "
        "decode, ~30x slower. It sizes the prefill buffers and position tiers (one trace per "
        "stage and tier, DEEPSEEK_V4_PREFILL_TIER_FACTOR); a chunk's cost follows its position, "
        "not this value, so a smaller one saves DRAM and traces but replays the excess of "
        "longer prompts through decode",
    )
    p.add_argument(
        "--prefill-trace-chunk",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_E2E_CHUNK", "1024")),
        help="tokens per traced prefill chunk (a multiple of 128); the stages pipeline consecutive chunks",
    )
    p.add_argument(
        "--l1-small-size",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", "8192")),
        help="bytes of L1_SMALL reserved for the CCL semaphores and the decode prefetcher's GCB config "
        "pages when the prefill model is on (must not be 0)",
    )
    p.add_argument(
        "--max-body-bytes",
        type=int,
        default=16 << 20,
        help="largest accepted request body in bytes (default: 16 MiB)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature for requests that send none (default: greedy argmax); "
        "a request's own temperature overrides it, and 0 asks for greedy. With --device-sampling it "
        "is the one temperature every turn samples at, and requests cannot set their own",
    )
    p.add_argument(
        "--device-sampling",
        dest="device_sampling",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DEEPSEEK_V4_DEVICE_SAMPLING", "0") not in ("0", "", "false", "False"),
        help="pick each token on the output stage instead of reading the whole 129k-wide logits row back "
        "per token: a greedy step then sends one int page over PCIe, and sampling costs nothing on the "
        "host. One temperature for the whole server (--temperature), so requests cannot set their own; "
        "--no-device-sampling samples on the host",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V4_SEED", "1234")),
        help="seed of the device sampler's PRNGs (--device-sampling); it still advances from step to step, "
        "so a reply is reproducible but not one token repeated",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="log every scheduler event: admissions, slot assignments, KV page allocations, "
        "prefill lengths and per-user decode rates (in the live console, the 'd' key toggles it)",
    )
    p.add_argument(
        "--prompt-snippet",
        action="store_true",
        help=f"log only the first {_PROMPT_SNIPPET_CHARS} characters of each message instead of the "
        "whole prompt (in the live console, the 'v' key toggles it)",
    )
    p.add_argument(
        "--no-tui",
        dest="tui",
        action="store_false",
        help="plain line-by-line logging instead of the live split console (implied when "
        "stdout is not a terminal, e.g. under nohup or in CI)",
    )
    return p, sys_cfg


def check_args(p: argparse.ArgumentParser, args: argparse.Namespace, sys_cfg) -> argparse.Namespace:
    """Validate and complete the parsed ``args`` (``p.error`` on a bad combination)."""
    args.system_config = sys_cfg
    # ``ChatEngine`` re-resolves the profile once the mesh is open; keep it on the
    # server flavour when it does.
    args.system_variant = "server"
    if args.system_prompt_file:
        args.system_prompt = Path(args.system_prompt_file).expanduser().read_text().strip()
    if args.reasoning_effort and not args.think:
        p.error("--reasoning-effort requires --think")
    if args.num_users < 1:
        p.error("--num-users must be at least 1")
    if args.total_context is None:
        # Every user able to fill its own context, rather than all of them sharing one
        # context's worth of blocks (which made a busy server hand out 429s early).
        args.total_context = args.num_users * args.max_context
    elif args.total_context < args.max_context:
        p.error(
            f"--total-context {args.total_context} is below --max-context {args.max_context}: "
            "no user could ever fill its context"
        )
    if args.max_running < 1:
        p.error("--max-running must be at least 1")
    if args.slot_idle_timeout < 0:
        p.error("--slot-idle-timeout must be non-negative")
    if args.host_users < 0:
        p.error("--host-users must be non-negative")
    if args.host_cache_gb < 0:
        p.error("--host-cache-gb must be non-negative")
    if not 0 < args.port < 65536:
        p.error("--port must be a TCP port number")
    if args.prefill_chunk < 1:
        p.error("--prefill-chunk must be at least 1")
    if args.prefill_trace_chunk <= 0 or args.prefill_trace_chunk % ALIGNMENT:
        p.error(f"--prefill-trace-chunk must be a positive multiple of {ALIGNMENT}")
    if args.prefill_max_len and args.prefill_max_len < ALIGNMENT:
        p.error(f"--prefill-max-len must be 0 (the whole context) or at least {ALIGNMENT}")
    if args.quiet and args.debug:
        p.error("--quiet and --debug ask for opposite things")
    if args.temperature is not None and not args.temperature >= 0:
        p.error("--temperature must be non-negative")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p, sys_cfg = build_parser(argv)
    return check_args(p, p.parse_args(argv), sys_cfg)


def main(argv: list[str] | None = None) -> int:
    return serve(parse_args(argv))


@torch.no_grad()
def serve(
    args: argparse.Namespace,
    engine_cls=ChatEngine,
    prefiller_cls=_Prefiller,
    num_stages: int | None = None,
    handler_cls=_Handler,
    server_cls=GenerationServer,
) -> int:
    """Build the engine (``engine_cls``, a :class:`ChatEngine`) and the prefill (``prefiller_cls``, a
    :class:`_Prefiller`) on the opened mesh and serve until interrupted.

    ``num_stages`` is how many ``1 x TP`` submeshes to carve out of the full mesh. Unset, prefill
    takes every row. The mesh itself stays the whole system either way. The V4.1 server passes
    ``DEEPSEEK_V4_NUM_STAGES``, its own ``handler_cls`` (DeepSeek ``GET /models``) and ``server_cls``
    (a :class:`GenerationServer` that takes images).
    """
    # The model emits plenty of non-ASCII (CJK, emoji, typographic punctuation), which
    # an ASCII/POSIX locale would turn into a UnicodeEncodeError mid-reply.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # loguru's default sink is DEBUG, which would make the scheduler's per-round lines the
    # normal output; they are opt-in via --debug (or the console's 'd' key) instead.
    logger.remove()
    logger.add(sys.stderr, level="WARNING" if args.quiet else "DEBUG" if args.debug else "INFO")
    if not Path(args.model_dir).expanduser().exists():
        print(f"checkpoint not found: {args.model_dir}", file=sys.stderr)
        return 1
    torch.manual_seed(0)
    _LOG_FULL_PROMPT[0] = not args.prompt_snippet
    logger.info(
        f"KV budget: {args.num_users} concurrent users x {args.max_context} tokens, "
        f"shared pool {args.total_context} tokens. The users' steps are pipelined through the "
        f"stages rather than batched."
    )
    mesh_shape = l1_small_size = None
    if args.prefill or num_stages is not None:
        # The layout of tests/prefill/test_prefill_decode_demo.py: the whole mesh opened as
        # ``rows x 4`` (8 x 4 = all 32 chips of a Galaxy). ``num_stages`` carves that many 1x4
        # submeshes off the front; the rest of the mesh stays open and idle.
        chips = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
        if chips % _TP_SIZE:
            print(
                f"TP{_TP_SIZE} stages need a multiple of {_TP_SIZE} chips, but the system has {chips}", file=sys.stderr
            )
            return 1
        rows = chips // _TP_SIZE
        stages = rows if num_stages is None else num_stages
        if stages < 1 or stages > rows:
            print(f"{stages} pipeline stage(s) do not fit the {rows} mesh rows", file=sys.stderr)
            return 1
        args.tp_size, args.num_stages = _TP_SIZE, stages
        mesh_shape = (rows, _TP_SIZE)
        if args.prefill:
            l1_small_size = args.l1_small_size
            # MTP disabled (read by DeepSeekV4Model.__init__): every row is a pipeline stage. Its
            # prefetches could also outlive a step, which a prefill between steps does not allow.
            os.environ["DEEPSEEK_V4_DSPARK"] = "0"
            # Room above the GCB rings for decode's own buffers, so releasing the rings frees one block of L1 for prefill.
            os.environ.setdefault("DEEPSEEK_V4_GCB_HEADROOM", "16384")
            logger.info(
                f"traced prefill on: {args.num_stages} x TP{_TP_SIZE} stages "
                f"({args.num_stages * _TP_SIZE} of {chips} chips) shared with decode"
            )
        else:
            logger.info(
                f"decode on: {args.num_stages} x TP{_TP_SIZE} stages ({args.num_stages * _TP_SIZE} of {chips} chips)"
            )
    with open_mesh_device(
        args.trace_region_size, system_config=args.system_config, mesh_shape=mesh_shape, l1_small_size=l1_small_size
    ) as mesh_device:
        if args.prefill:
            args.submeshes = [
                mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0))
                for i in range(args.num_stages)
            ]
        # The prefetcher session spans the trace capture, the warmup and every served
        # turn, so it is opened inside ``ChatEngine`` (once the model exists) against
        # this stack, and the senders are stopped before the mesh device is closed.
        with contextlib.ExitStack() as prefetcher:
            engine = engine_cls(mesh_device, args, prefetcher)
            # Decode compile, then prefill build, compile and capture (decode's GCBs and L1 released).
            # The warmup then captures decode, and the snapshot copies decode's resident tensors.
            prefiller = prefiller_cls(engine, args, prefetcher) if args.prefill else None
            engine.warmup()
            if prefiller is not None:
                prefiller.snapshot()
            api = server_cls(
                engine,
                args.model_id,
                args.max_body_bytes,
                args.prefill_chunk,
                prefiller=prefiller,
                decode_ahead=args.decode_ahead,
                temperature=args.temperature,
                host_users=args.host_users,
                host_bytes=int(args.host_cache_gb * 2**30),
                max_running=args.max_running,
                slot_idle_timeout=args.slot_idle_timeout,
            )
            # The scheduler is the only thread that touches the device from here on; the
            # traces it replays were captured by the warmup above, which a pipelined
            # caller cannot do mid-flight.
            sys.setswitchinterval(_GIL_SWITCH_SECONDS)
            api.start()
            httpd = _ModelHTTPServer((args.host, args.port), handler_cls, api)
            # The live console owns the logger from here on, so it is opened after the
            # (long, chatty) model build and warmup have printed normally. ``shutdown``
            # has to come from another thread than ``serve_forever``, which the console's
            # key thread is.
            with tui.console(
                logger,
                api.stats,
                on_quit=lambda: threading.Thread(target=httpd.shutdown).start(),
                enabled=args.tui,
                on_toggle_prompt=_toggle_full_prompt,
                full_prompt=_LOG_FULL_PROMPT[0],
                debug=args.debug,
                on_stop_all=api.cancel_all,
            ):
                prefill_mode = (
                    f"traced prefill up to {prefiller.max_len} tokens in chunks of {prefiller.chunk_size}"
                    if prefiller is not None
                    else "prompts replayed through decode"
                )
                logger.info(
                    f"serving {args.model_id!r} on http://{args.host}:{args.port} "
                    f"({len(engine.users)} concurrent sessions, context {engine.max_seq}/user, "
                    f"{prefill_mode}, decode replay chunk {args.prefill_chunk})"
                )
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    logger.info("interrupted, shutting down")
                finally:
                    httpd.server_close()
                    api.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
