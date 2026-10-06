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
* ``GET    /v1/sessions``         -- active KV-cache sessions / usage (extension)
* ``DELETE /v1/sessions/<user>``  -- reset one session (extension)
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
prefill (one trace per stage, ``--prefill-trace-chunk`` tokens per chunk, the stages on
consecutive chunks at once),
its attention state is committed into the session's decode buffers on device, and the
ragged tail is replayed through decode. A follow-up turn feeds only the tokens it adds
through decode, unless rewinding and prefilling the whole conversation is estimated to
be faster (a long new turn); short prompts skip the prefill model for the same reason.
Decode runs with the DRISC prefetcher off on this path (a prefill replay
overwrites decode's L1, which is snapshot and written back; a GCB cannot be).
``--no-prefill`` replays every prompt token through decode instead, on the profile's
own decode layout with the prefetcher on.

**Users & KV cache.** ``--num-users`` paged KV-cache sessions (8 by default) are claimed
at startup -- one captured trace, one block pool holding ``--total-context`` tokens
across everyone -- and then handed out a turn at a time, so ``--num-users`` is the number
of requests that can generate at once. Raising it costs little memory (a session's
sliding ring is bounded by ``sliding_window``, and the compressed blocks come from the
shared budget). The users' steps are pipelined through the stages rather than batched, so
up to one user per stage adds throughput at little cost to each reply; past that the
stages are full and the users share their rate. A slot holds a warm cache, so a request is given the slot whose stored
conversation its ``messages`` continue -- then only the new turn's tokens are fed --
falling back to an empty slot, or to the least recently used one, which is re-prefilled
from the request. The OpenAI ``user`` field only labels who last held a slot: clients
that send one identifier for all their traffic, or none, still run in parallel.

**Concurrency.** Up to ``--num-users`` turns generate at once, their steps in the
stage pipeline together, as ``tests/decode/test_multi_user_paged_decode_demo.py`` runs
its users: every decoding turn has one step in flight, and its next step is seated,
posted and packed as soon as the previous one's token is back, so with as many users as
stages each stage works on a different user's step. Prompt tokens replayed through decode
keep up to ``--prefill-chunk`` steps in flight per turn. Outputs are read off the socket
by a reader thread, in dispatch order. A traced prefill waits for every step in flight
to come back and then holds the device for that prompt. One scheduler thread does all
the dispatch, which is required, not incidental: the trace replays and the paged session
state have to be driven from one thread, and each step's output must be matched in
dispatch order. HTTP threads never touch the device -- they submit a turn and relay its
reply -- so total throughput scales with the users while each reply streams
independently. A request that arrives with every slot busy waits for one to free.

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
``c`` clears it and ``q`` quits. ``--debug`` starts with those lines on and ``--no-tui``
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
  ``temperature`` / ``top_p`` sample from the per-step logits (default greedy
  argmax). ``thinking: true`` and ``reasoning_effort`` are DeepSeek-V4 extensions
  that switch that session to the thinking template. In thinking mode the reply
  carries the reasoning block in the ``reasoning_content`` field, exactly like the
  DeepSeek reasoner API, while ``content`` holds the answer. ``stop`` sequences
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
import re
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.demo import tui
from models.experimental.deepseek_v4_flash.demo.chat_cli import (
    ChatEngine,
    ContextFull,
    UserSession,
    open_mesh_device,
)
from models.experimental.deepseek_v4_flash.encoding_dsv4 import (
    merge_tool_messages,
    render_message,
    sort_tool_results_by_call_order,
)
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
_DSML = "｜DSML｜"
_TOOL_CALLS_OPEN = f"<{_DSML}tool_calls>"
_TOOL_CALLS_CLOSE = f"</{_DSML}tool_calls>"
_INVOKE_RE = re.compile(rf'<{_DSML}invoke name="(?P<name>[^"]*)">(?P<body>.*?)</{_DSML}invoke>', re.DOTALL)
_PARAM_RE = re.compile(
    rf'<{_DSML}parameter name="(?P<name>[^"]*)" string="(?P<string>true|false)">(?P<value>.*?)</{_DSML}parameter>',
    re.DOTALL,
)


class RequestError(Exception):
    """A client-side error mapped onto an OpenAI error envelope."""

    def __init__(self, status: int, message: str, err_type: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type or ("invalid_request_error" if status < 500 else "server_error")


class _ClientGone(Exception):
    """The SSE client hung up mid-stream; stop generating and drop the reply."""


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
    messages = body.get("messages") or []
    total = sum(len(str(m.get("content", ""))) for m in messages)
    head = (
        f"{completion_id} from {client} user={user_key!r} "
        f"{' '.join(f'{k}={v!r}' for k, v in options.items())}\n"
        f"  {len(messages)} messages, {total} chars"
    )
    return "\n".join([head] + [f"    [{m.get('role')}] {_one_line(str(m.get('content', '')))}" for m in messages])


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


def _normalized_messages(messages, tools=None) -> list[dict]:
    """Validate a request's ``messages`` and put them in the form the renderer wants.

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
        entry = {"role": role, "content": _content_text(message.get("content"), index)}
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
        out = sort_tool_results_by_call_order(merge_tool_messages(out))
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
    """
    message = {k: v for k, v in message.items() if k != "reasoning_content"}
    if message.get("content_blocks") == [{"type": "text", "text": message.get("content", "")}]:
        del message["content_blocks"]
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


def _parse_tool_calls(block: str) -> list[dict]:
    """The OpenAI ``tool_calls`` described by a DSML block.

    Deliberately tolerant where the checkpoint's reference parser raises: a serving loop
    should hand back whatever the model managed to express rather than fail the request
    over a malformed tag. Both argument forms are read -- the ``parameter`` tags, and the
    bare JSON object that the model sometimes writes inside ``invoke`` instead.
    """
    calls = []
    for invoke in _INVOKE_RE.finditer(block):
        parameters = list(_PARAM_RE.finditer(invoke["body"]))
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
    temperature = float(temperature) if temperature is not None else 0.0
    top_p = float(top_p) if top_p is not None else None
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

    A DSML tool-call block is held back the same way and never streamed as content: it
    is markup addressed to the client's tool runner, not prose for the reader. What it
    describes is parsed into OpenAI ``tool_calls`` and left on :attr:`tool_calls`.
    """

    def __init__(self, tokenizer, on_chunk):
        self.tokenizer = tokenizer
        self.on_chunk = on_chunk
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
        self.tool_calls = _parse_tool_calls(tool_block) if tool_block else []
        if len(reasoning) > self._sent_r:
            self.on_chunk(reasoning[self._sent_r :], "")
            self._sent_r = len(reasoning)
        if len(content) > self._sent_c:
            self.on_chunk("", content[self._sent_c :])
            self._sent_c = len(content)
        self.reasoning, self.content = reasoning, content

    def _split(self, text: str) -> tuple[str, str, str]:
        """``text`` as (reasoning, answer, tool-call block), each without its tags."""
        start = text.find(_THINK_OPEN)
        if start == -1:
            reasoning, answer = "", text
        else:
            after = text[start + len(_THINK_OPEN) :]
            end = after.find(_THINK_CLOSE)
            if end == -1:
                reasoning, answer = after, text[:start]
            else:
                reasoning, answer = after[:end], text[:start] + after[end + len(_THINK_CLOSE) :]

        opened = answer.find(_TOOL_CALLS_OPEN)
        if opened == -1:
            return reasoning, answer, ""
        closed = answer.find(_TOOL_CALLS_CLOSE, opened)
        block = answer[opened : closed + len(_TOOL_CALLS_CLOSE)] if closed != -1 else answer[opened:]
        # The encoder separates the answer from the block with a blank line, which would
        # otherwise trail the reply as whitespace.
        return reasoning, answer[:opened].rstrip("\n"), block

    @staticmethod
    def _held_back(text: str) -> int:
        """Characters at the end that could still turn into a tag once more arrive.

        The newlines before a tool-call block count: they are the block's separator, and
        streaming them the moment they arrive would leave a trailing blank line on a
        reply whose visible text has actually ended."""
        for tag in (_THINK_CLOSE, _THINK_OPEN, _TOOL_CALLS_OPEN):
            for n in range(len(tag) - 1, 0, -1):
                if text.endswith(tag[:n]):
                    return n + len(text[: len(text) - n]) - len(text[: len(text) - n].rstrip("\n"))
        return len(text) - len(text.rstrip("\n"))


class _SlotPool:
    """The KV-cache slots, handed out per *request* rather than per ``user`` key.

    ``user`` in the OpenAI API is a stable end-user identifier: a client may send one
    value for all of its traffic, or none at all. Tying a slot to it would make every
    request from such a client queue behind the previous one, so a slot is instead
    claimed for the duration of a turn and released afterwards, and any request can use
    any slot that is not busy.

    Which free slot matters, because a slot holds a warm KV cache. :meth:`acquire`
    prefers, in order:

    1. a slot whose stored conversation is a prefix of this request's history -- the
       follow-up case, where the cache already holds everything but the new turn and
       only the new tokens are fed;
    2. a slot with no conversation at all, which costs nobody their cache;
    3. the least recently used slot, whose conversation is then dropped and the request
       re-prefilled from its own messages.

    When every slot is busy the caller waits: the turns in flight are bounded by
    ``--max-new-tokens``, so a slot always comes back.
    """

    def __init__(self, engine: ChatEngine):
        self.engine = engine
        self._cv = threading.Condition()
        self._busy = [False] * len(engine.users)
        self.owner: list[str | None] = [None] * len(engine.users)
        self._used_at = [0.0] * len(engine.users)
        self._closing = False

    def __len__(self) -> int:
        return len(self._busy)

    # -- claiming --------------------------------------------------------------- #
    def acquire(self, user_key: str, messages: list[dict]) -> int:
        """Claim a slot for one turn, waiting if they are all busy."""
        t0 = time.perf_counter()
        with self._cv:
            while True:
                if self._closing:
                    raise RequestError(503, "server is shutting down", "server_error")
                slot = self._pick(messages)
                if slot is not None:
                    break
                # logger.debug(
                #     f"user {user_key!r} waiting for a KV slot: all {len(self._busy)} busy "
                #     f"({', '.join(str(o) for o in self.owner)})"
                # )
                self._cv.wait(timeout=1.0)
            self._busy[slot] = True
            previous, self.owner[slot] = self.owner[slot], user_key
            self._used_at[slot] = time.time()
        waited = time.perf_counter() - t0
        # logger.debug(
        #     f"user {user_key!r} -> slot {slot} (sid {self.engine.users[slot].sid}, "
        #     f"{self._reason(slot, previous, user_key, messages)}"
        #     f"{f', waited {waited:.2f}s' if waited > 0.01 else ''})"
        # )
        return slot

    def reserve(self, slot: int) -> None:
        """Claim one specific slot, waiting for its turn to finish (used by a reset)."""
        with self._cv:
            while self._busy[slot] and not self._closing:
                self._cv.wait(timeout=1.0)
            if self._closing:
                raise RequestError(503, "server is shutting down", "server_error")
            self._busy[slot] = True

    def release(self, slot: int) -> None:
        with self._cv:
            self._busy[slot] = False
            self._used_at[slot] = time.time()
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closing = True
            self._cv.notify_all()

    def _pick(self, messages: list[dict]) -> int | None:
        free = [i for i, busy in enumerate(self._busy) if not busy]
        if not free:
            return None
        # 1. The longest stored conversation this request continues.
        best, best_len = None, 0
        for i in free:
            stored = self.engine.users[i].messages
            if len(stored) > best_len and _continues(messages, stored):
                best, best_len = i, len(stored)
        if best is not None:
            return best
        # 2. A slot holding nothing.
        for i in free:
            if not self.engine.users[i].messages:
                return i
        # 3. The one whose cache has been idle longest.
        return min(free, key=lambda i: self._used_at[i])

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
        """One row per slot, for ``/v1/sessions`` and the status pane."""
        with self._cv:
            busy, owner = list(self._busy), list(self.owner)
        rows = []
        for index, user in enumerate(self.engine.users):
            rows.append(
                {
                    "id": owner[index] or "",
                    "index": index,
                    "tokens": user.pos,
                    "max_context": self.engine.max_seq,
                    "messages": len(user.messages),
                    "thinking": user.thinking_mode == "thinking",
                    "busy": busy[index],
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

    Construction is the test's steps 2-3 (decode compile, then the prefill build and
    compile with decode's L1 tensors parked); :meth:`capture` is step 4, after
    :meth:`ChatEngine.warmup` has captured the decode traces. Requires a model built with
    the prefetcher off: a prefill replay would also overwrite the GCB config pages, which
    cannot be snapshot.
    """

    def __init__(self, engine: ChatEngine, args, prefetcher: contextlib.ExitStack):
        model = engine.model
        self.engine = engine
        self.chunk_size = args.prefill_trace_chunk
        # Every chunk attends the whole FIFO sized for ``max_len``, so this is a speed knob too.
        self.max_len = min(args.prefill_max_len or engine.max_seq, engine.max_seq - 1) // ALIGNMENT * ALIGNMENT
        if self.max_len < ALIGNMENT:
            raise ValueError(f"traced prefill needs a context of at least {ALIGNMENT + 1} tokens")
        self.num_stages = len(dict.fromkeys(model.layer_submesh_ids))
        # What a prefill costs, for choosing it over replaying the prompt through decode: the first chunk
        # crosses every stage before its logits arrive (plus the write-back and the commit), each later one
        # lands a stage behind it. Seeded from test_prefill_decode_demo.py on 8 stages with 1024-token
        # chunks; every prefill folds its own timings in.
        self.fill_seconds = 1.6
        self.chunk_seconds = 0.23
        self._loud = True
        tokenizer = engine.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else engine.eos_id

        # Decode's ops keep buffers of their own (global semaphores, cached constants) that no snapshot can
        # reach; compiling first allocates them before prefill lays out its memory, so its replays keep off them.
        t0 = time.perf_counter()
        model.compile_traces(pad_id, 0)
        logger.info(f"decode compile: {time.perf_counter() - t0:.1f}s")
        model.release_prefetch_buffers()
        indexer = os.environ.get("DEEPSEEK_V4_PREFILL_INDEXER", "1") == "1"
        t0 = time.perf_counter()
        self.model = model.build_prefill(
            engine.rope,
            checkpoint_weights(engine.loader, engine.config, model.num_layers),
            lm_head=engine.lm_head,
            cache=engine.weight_cache,
            weight_dtype=_ATTENTION_WEIGHT_DTYPE,
            dense_csa=True,
            lightning_indexer=indexer,
            progress=self._note,
        )
        self.model.synchronize("uploads")
        logger.info(f"prefill model built in {time.perf_counter() - t0:.1f}s (lightning indexer {indexer})")
        t0 = time.perf_counter()
        self.model.compile_traced_prefill(self.max_len, self.chunk_size)
        logger.info(
            f"traced prefill compiled in {time.perf_counter() - t0:.1f}s: prompts of up to {self.max_len} "
            f"tokens in chunks of {self.chunk_size}, {self.num_stages} pipelined stage(s)"
        )
        # Unwinds before the decode model's shutdown (LIFO).
        prefetcher.callback(self.model.release_traced_prefill)
        self.bias_slots = prefill_bias_slots(self.model)
        model.restore_prefetch_buffers()

    def _note(self, message: str, important: bool = False) -> None:
        if important:
            (logger.info if self._loud else logger.debug)(f"[prefill] {message}")

    def capture(self) -> None:
        """Capture one prefill trace per stage, after the decode traces exist.

        Decode's L1 tensors are parked meanwhile and put back at the addresses the decode
        traces recorded; they then sit under prefill's buffers, so they are copied to the
        host for :meth:`run` to write back after every replay."""
        model = self.engine.model
        t0 = time.perf_counter()
        model.release_prefetch_buffers()
        self.model.capture_traced_prefill()
        model.restore_prefetch_buffers()
        model.snapshot_resident_state()
        self._loud = False
        logger.info(f"traced prefill captured in {time.perf_counter() - t0:.1f}s")

    def aligned(self, n_tokens: int) -> int:
        """How many of a fresh ``n_tokens``-token prompt the prefill takes; decode replays the rest."""
        return min(n_tokens // ALIGNMENT * ALIGNMENT, self.max_len)

    def seconds(self, n_tokens: int) -> float:
        """Expected wall time of :meth:`run` over ``n_tokens`` (a multiple of 128)."""
        return self.fill_seconds + (-(-n_tokens // self.chunk_size) - 1) * self.chunk_seconds

    def run(self, user: UserSession, ids: list[int], on_progress=None) -> torch.Tensor:
        """Prefill ``ids`` (a multiple of 128 tokens) into ``user``'s empty session; returns
        the next-token logits ``[V]`` (host fp32) and leaves ``user.pos`` at ``len(ids)``.

        Must run with no decode step in flight: the replays share decode's command queue and
        overwrite its L1 tensors, which are written back before anything else runs."""
        engine, model, prefill = self.engine, self.engine.model, self.model
        if engine._replay_q:
            raise RuntimeError("prefill with decode traces still posted")
        # Seated first, so the previous occupant's window state is saved to its own (DRAM)
        # block before the replay overwrites the resident L1 copies.
        user.activate()
        n = len(ids)
        num_chunks = -(-n // self.chunk_size)
        chunk_times: list[tuple[int, float]] = []
        t0 = time.perf_counter()

        def on_chunk(_i, start, end, seconds):
            chunk_times.append((end - start, seconds))
            if on_progress is not None:
                on_progress(end)  # the status pane reads this from another thread

        try:
            logits, states = prefill.prefill_traced(torch.tensor(ids, dtype=torch.long).unsqueeze(0), on_chunk=on_chunk)
        finally:
            model.reload_resident_state()
        prefill_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        try:
            # The previous conversation (or the capture step) must not leak into this one.
            model.reset_session(user.sid)
            model.reset_static_caches()
            committed = model.commit_prefill_state(states, user.sid, bias_slots=self.bias_slots)
        except Exception:
            with contextlib.suppress(Exception):
                user.reset()
            raise
        finally:
            prefill.free_traced_states(states)
        user.pos = committed
        total_seconds = time.perf_counter() - t0
        steady = chunk_times[1:] or chunk_times
        steady_seconds = sum(seconds for _, seconds in steady)
        piped = f"{sum(t for t, _ in steady) / steady_seconds:.1f}" if steady_seconds else "-"
        if len(chunk_times) > 1:
            self.chunk_seconds = 0.5 * self.chunk_seconds + 0.5 * steady_seconds / len(steady)
        fill_seconds = max(total_seconds - (num_chunks - 1) * self.chunk_seconds, 0.0)
        self.fill_seconds = 0.5 * self.fill_seconds + 0.5 * fill_seconds
        logger.info(
            f"prefill model (user {user.index}): {n} tokens in {prefill_seconds:.2f}s = "
            f"{n / prefill_seconds:.1f} tok/s, {num_chunks} chunk(s) of up to {self.chunk_size} through "
            f"{self.num_stages} stage(s), {piped} tok/s pipelined; commit {time.perf_counter() - t1:.2f}s"
        )
        row = logits.reshape(-1)
        if not torch.isfinite(row).all():
            raise RuntimeError("prefill produced non-finite logits")
        return row


class _Turn:
    """One in-flight chat turn: the request's decode state plus its output channel.

    Created by the HTTP thread, then owned by the :class:`_Scheduler` thread, which is
    the only one allowed to touch the model or the :class:`UserSession`. The two
    threads meet at :attr:`events` (the reply, delta by delta) and :attr:`cancelled`
    (the client hung up), both of which are thread-safe.

    A turn starts in the ``prefill`` phase, feeding its prompt tokens, and moves to
    ``decode`` once the last of them has been fed. A turn that starts a session from
    position 0 has its 128-aligned prefix (:attr:`device_prefill` tokens) run by the
    traced prefill model in one go, and only the rest replayed through decode.
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
        self.fed = 0  # how many of them have been dispatched
        self.device_prefill = 0  # leading prompt tokens still to go through the traced prefill model
        self.prefilled_on_device = 0  # ... and how many it took, once it has run
        self.rebuilt = False  # a continued conversation rewound and re-prefilled whole, since that was faster
        self.t_replay = 0.0  # when this turn's prompt tokens started going through decode
        self.next_id: int | None = None  # produced but not yet fed back
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
    per step, in dispatch order; :attr:`results` yields ``(ticket, logits [1, V])`` in that
    order, or ``(ticket, exception)`` once a read fails, after which it stops.
    """

    _STOP = object()

    def __init__(self, model):
        self._model = model
        self._tickets: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="decode-reader", daemon=True)
        self._thread.start()

    def expect(self, ticket) -> None:
        self._tickets.put(ticket)

    def close(self) -> None:
        self._tickets.put(self._STOP)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        for ticket in iter(self._tickets.get, self._STOP):
            try:
                out = self._model.read_decoded_output().reshape(1, -1).float()
            except BaseException as e:  # noqa: BLE001 - handed to the scheduler, which aborts
                self.results.put((ticket, e))
                return
            self.results.put((ticket, out))


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

    def __init__(self, server: "GenerationServer", prefill_chunk: int = 16):
        self.server = server
        self.engine = server.engine
        self.prefill_chunk = max(1, prefill_chunk)
        self._new: deque[_Turn] = deque()
        self._chores: deque[tuple] = deque()  # (callable, done event, result box)
        self._active: list[_Turn] = []
        self._inflight: deque[_Turn | None] = deque()  # dispatch order across all turns; None is discarded
        self._reader: _OutputReader | None = None  # started with the scheduler thread
        self._wake = threading.Condition()
        self._stop = False
        self._broken: Exception | None = None  # a failed readback; see _abort_all
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
        self.step_rate = 0.0  # steps/s across all users, over the last window
        self._t_started = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="decode-scheduler", daemon=True)

    # -- lifecycle -------------------------------------------------------------- #
    def start(self) -> None:
        self._reader = _OutputReader(self.engine.model)
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
        user = self.engine.users[turn.slot]
        turn.t_admit = time.perf_counter()
        continuing = False
        logger.debug(
            f"admitting user {turn.user_key!r} (slot {turn.slot}, sid {user.sid}): "
            f"queued {turn.t_admit - turn.t_submit:.3f}s, at pos {user.pos}/{self.engine.max_seq}, "
            f"{len(user.messages)} stored messages, {len(self._active)} turns already active"
        )
        try:
            self.server._apply_options(user, turn.body)
            continuing = self.server._sync_messages(user, turn.body["messages"])
            if continuing:
                user.messages.append(turn.message)
                turn.ids = self.server._render_ids(user, include_assistant=False)
            else:
                user.messages = user.messages + [turn.message]
                user._next_render = 0
                turn.ids = self.server._render_ids(user, include_assistant=True)
            self._choose_feed(user, turn, continuing)
            self._check_capacity(user, turn.ids)
        except Exception as e:  # noqa: BLE001 - reported to the client, the server stays up
            if turn.rebuilt:
                user.reset()  # the cache was already rewound: start the conversation afresh next time
            elif continuing:
                user.messages.pop()  # nothing was fed: leave the conversation as it was
            logger.debug(f"user {turn.user_key!r} rejected at admission: {type(e).__name__}: {e}")
            turn.events.put(("error", e))
            return
        user._next_render = len(user.messages)
        turn.t_replay = turn.t_admit
        turn.stream = _Streamer(self.engine.tokenizer, lambda r, c: turn.events.put(("delta", r, c)))
        self._active.append(turn)
        how = "re-prefill whole conversation" if turn.rebuilt else "continuing" if continuing else "fresh session"
        logger.debug(
            f"user {turn.user_key!r} admitted: prefill {len(turn.ids)} tokens ({how}"
            f"{f', {turn.device_prefill} by the prefill model' if turn.device_prefill else ', all through decode'}), "
            f"cap {turn.max_tokens} new tokens, pool {self._pool_summary()}"
        )

    def _choose_feed(self, user: UserSession, turn: _Turn, continuing: bool) -> None:
        """Pick the faster way to get this turn's prompt into the cache.

        The prefill model only starts an empty session and takes 128-aligned prefixes, and a
        single chunk already costs a pass through every stage, so replaying a short prompt
        through decode is quicker. For a fresh session the choice is how much the prefill
        model takes (:attr:`_Turn.device_prefill`). A continued conversation can either
        replay just its new turn onto the warm cache or rewind and re-prefill the whole
        conversation with the prefill model; the latter wins once the new turn is long.
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
            )
        self._window_t0 = time.perf_counter()
        self._window_steps = 0
        self._window_dispatch = 0.0

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
                if turn.phase == _Turn.PREFILL and turn.device_prefill:
                    if not self._settle():
                        return
                    self._prefill_on_device(turn)
                steps = self._plan(turn)
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
        """Read back every step in flight, e.g. before a traced prefill takes the device.

        Returns ``False`` if a read failed, in which case every turn has been aborted."""
        try:
            while self._inflight:
                self._collect(block=True)
        except Exception as e:  # noqa: BLE001 - see _abort_all: ordering is lost
            self._abort_all(e)
            return False
        return True

    def _collect(self, block: bool) -> None:
        """Take every output that has come back, first waiting for the oldest one if ``block``.

        Outputs arrive in dispatch order, so each belongs to the turn at the head of
        :attr:`_inflight`. Raises if a read failed."""
        results = self._reader.results
        try:
            item = results.get() if block else results.get_nowait()
        except queue.Empty:
            return
        while True:
            self._take(*item)
            try:
                item = results.get_nowait()
            except queue.Empty:
                return

    def _take(self, turn: _Turn | None, out) -> None:
        """Hand one step's logits ``[1, V]`` to the turn that dispatched it."""
        assert self._inflight and self._inflight[0] is turn, "decode outputs read out of dispatch order"
        self._inflight.popleft()
        if isinstance(out, BaseException):
            raise RuntimeError(f"reading a decode output failed: {out}") from out
        if turn is None:
            return  # a posted trace fed a dummy packet after a failed dispatch
        last_prompt_token = turn.pending.popleft()
        if turn.error is not None:
            return
        if turn.phase == _Turn.DECODE or last_prompt_token:
            turn.next_id = turn.sampler(out) if turn.sampler is not None else int(out[0].argmax().item())
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

    def _prefill_on_device(self, turn: _Turn) -> None:
        """Run the turn's aligned prompt prefix through the traced prefill model.

        Blocks the scheduler for the whole prefill: every stage of the device is busy with
        this prompt meanwhile, so the other turns' decode resumes once it is done."""
        user = self.engine.users[turn.slot]
        n, turn.device_prefill = turn.device_prefill, 0
        assert user.pos == 0 and turn.fed == 0, "the prefill model only starts a session"
        logger.debug(
            f"user {turn.user_key!r}: prefilling {n} of {len(turn.ids)} tokens with the prefill model "
            f"({len(self._active) - 1} other turn(s) wait)"
        )
        # The turn's own clock starts here, so the pane's prefill tok/s covers this prefill.
        turn.t_admit = turn.t_admit or time.perf_counter()

        def progress(done: int) -> None:
            turn.mark_prefill_chunk(done)

        row = self.server.prefiller.run(user, turn.ids[:n], on_progress=progress)
        turn.fed = turn.prefilled_on_device = n
        turn.t_replay = time.perf_counter()
        self.steps += n
        if n == len(turn.ids):
            out = row.reshape(1, -1)
            turn.next_id = turn.sampler(out) if turn.sampler is not None else int(out[0].argmax().item())
            self._prompt_done(turn)

    def _plan(self, turn: _Turn) -> list[tuple[int, int, bool]]:
        """The steps this turn dispatches now, as ``(token_id, pos, last_prompt_token)``.

        Touches no device state, but finishes a decoding turn that is done (EOS, its
        ``max_tokens``, the context, or a client that hung up).
        """
        engine = self.engine
        user = engine.users[turn.slot]
        if turn.phase == _Turn.PREFILL:
            # A client that hangs up mid-prompt is not abandoned here: the prompt is fed
            # to the end so the KV cache matches the stored conversation, and the turn
            # then finishes with an empty reply.
            room = self.prefill_chunk - len(turn.pending)
            if not turn.prompt_left or room < (self.prefill_chunk + 1) // 2:
                return []
            if turn.fed == turn.prefilled_on_device:
                logger.debug(
                    f"user {turn.user_key!r}: replaying {turn.prompt_left} prompt tokens through decode "
                    f"from pos {user.pos}, up to {self.prefill_chunk} in flight"
                )
            n = min(room, turn.prompt_left)
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
        turn.stream.push(turn.generated)
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
            engine.post_traced(user, [pos for _, pos, _ in steps])
            for token_id, pos, last_prompt_token in steps:
                engine.write_traced(token_id, pos)
                self._reader.expect(turn)
                self._inflight.append(turn)
                turn.pending.append(last_prompt_token)
                user.pos += 1
                if turn.phase == _Turn.PREFILL:
                    turn.fed += 1
                self.steps += 1
        except Exception as e:  # noqa: BLE001 - one bad turn must not stop the others
            turn.error = e
            with contextlib.suppress(Exception):
                while engine._replay_q:
                    engine.write_traced(0, engine._replay_q[0])
                    self._reader.expect(None)
                    self._inflight.append(None)
        finally:
            self._window_dispatch += time.perf_counter() - t0

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
        replayed = len(turn.ids) - turn.prefilled_on_device
        if turn.prefilled_on_device:
            how = f"{turn.prefilled_on_device} by the prefill model + {replayed} through decode"
        else:
            how = "all through decode"
        if turn.rebuilt:
            how += ", whole conversation re-prefilled"
        prefill_rate = len(turn.ids) / turn.prefill_seconds if turn.prefill_seconds > 0 else 0.0
        logger.info(
            f"user {turn.user_key!r} (slot {user.index}): prefill {len(turn.ids)} tokens ({how}) in "
            f"{turn.prefill_seconds:.2f}s = {prefill_rate:.1f} tok/s, decoded {len(turn.generated)} at "
            f"{turn.mean_decode_rate:.2f} tok/s mean, {turn.decode_rate:.2f} tok/s at the end, "
            f"context {user.pos}/{self.engine.max_seq}, finish={'length' if turn.hit_cap else 'stop'}"
            f"{', client gone' if turn.cancelled.is_set() else ''}, {len(self._active)} turns still active"
        )
        logger.debug(f"user {turn.user_key!r} finished: pool {self._pool_summary()}")
        turn.events.put(
            (
                "done",
                {
                    "prompt_tokens": len(turn.ids),
                    "completion_tokens": len(turn.generated),
                    "content": turn.stream.content,
                    "reasoning_content": turn.stream.reasoning,
                    "tool_calls": turn.stream.tool_calls,
                    "finish_reason": ("length" if turn.hit_cap else "tool_calls" if turn.stream.tool_calls else "stop"),
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
        if turn in self._active:
            self._active.remove(turn)


class GenerationServer:
    """The resident model plus the user/session bookkeeping behind the HTTP layer.

    One :class:`ChatEngine` (weights, RoPE tables, decode traces) serves every request.
    The engine claimed one paged KV session per ``--num-users`` at startup, and
    :class:`_SlotPool` hands those out a turn at a time, preferring the slot that already
    holds the conversation the request continues.

    Turns run concurrently: an HTTP thread validates the request, claims a slot, hands a
    :class:`_Turn` to the :class:`_Scheduler` and then does nothing but relay the reply as
    it arrives, while the scheduler thread interleaves every active turn's decode steps on
    the device.
    """

    def __init__(
        self,
        engine: ChatEngine,
        model_id: str,
        max_body_bytes: int = 16 << 20,
        prefill_chunk: int = 16,
        prefiller: _Prefiller | None = None,
    ):
        self.engine = engine
        self.model_id = model_id
        self.max_body_bytes = max_body_bytes
        self.prefiller = prefiller
        self.pool = _SlotPool(engine)
        self._created = int(time.time())
        self.scheduler = _Scheduler(self, prefill_chunk)

    # -- lifecycle -------------------------------------------------------------- #
    def start(self) -> None:
        self.scheduler.start()

    def stop(self) -> None:
        self.pool.close()  # release anyone waiting for a slot before the threads go
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
            }
        )
        return stats

    # -- users / sessions ------------------------------------------------------- #
    def reset_user(self, user_key: str) -> None:
        """Rewind the conversations ``user_key`` last held, and their KV pages.

        Reserves each slot first, so it waits for a turn in flight rather than pulling
        the pages out from under it."""
        slots = self.pool.slots_of(user_key)
        if not slots:
            logger.debug(f"reset for {user_key!r}: it holds no slot, nothing to do")
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
        user.reset()
        user.messages = list(messages[:-1])
        user._next_render = 0
        return False

    # -- generation ------------------------------------------------------------- #
    def generate(self, user_key: str, body: dict, on_chunk) -> dict:
        """Run one chat-completion turn for ``user_key``.

        ``on_chunk(reasoning, content)`` receives the reply's new text as it resolves
        (either argument can be empty); raising ``_ClientGone`` aborts the turn, keeping
        the reply generated so far.

        Blocks until the reply is complete, but holds nothing except this user's own
        lock: the decode steps run on the scheduler thread, interleaved with the other
        users' turns, and this thread only relays what they produce."""
        messages = _normalized_messages(body.get("messages"), body.get("tools"))
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

        sampler = _make_sampler(body.get("temperature"), body.get("top_p"))
        # A slot is claimed per turn, not per user, so two requests run concurrently even
        # when they carry the same ``user`` (or none). Held until the reply is complete:
        # the turn decodes into this slot's KV session.
        slot = self.pool.acquire(user_key, messages)
        try:
            turn = _Turn(user_key, slot, body, sampler, max_tokens)
            self.scheduler.submit(turn)
            return self._relay(turn, on_chunk)
        finally:
            self.pool.release(slot)

    @staticmethod
    def _relay(turn: _Turn, on_chunk) -> dict:
        """Forward a turn's events to ``on_chunk`` until it finishes.

        A client that hangs up (``_ClientGone`` out of ``on_chunk``) cancels the turn but
        does not abandon it: the loop keeps draining, silently, until the scheduler
        reports the turn done, so the session is left consistent and its steps in flight
        are still read off the socket in order."""
        while True:
            kind, *payload = turn.events.get()
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
            user.thinking_mode = "thinking" if body.get("thinking") else "chat"
        effort = body.get("reasoning_effort")
        if effort is not None:
            if effort not in ("high", "max"):
                raise RequestError(400, "reasoning_effort must be 'high' or 'max'")
            user.reasoning_effort = effort

    def _render_ids(self, user: UserSession, include_assistant: bool) -> list[int]:
        """Tokens for the not-yet-encoded part of ``user.messages``.

        With ``include_assistant`` (a fresh re-prefill) every message is rendered,
        past assistant turns included. Otherwise (a continuation) assistant turns
        are skipped -- the model's own tokens are in the cache already -- and the
        turn starts with the pending token that closed the previous one."""
        engine = user.engine
        first_turn = user._next_render == 0
        rendered = "".join(
            render_message(i, user.messages, user.thinking_mode, reasoning_effort=user.reasoning_effort)
            for i in range(user._next_render, len(user.messages))
            if include_assistant or user.messages[i]["role"] != "assistant"
        )
        ids = list(engine.tokenizer(rendered, add_special_tokens=first_turn)["input_ids"])
        if include_assistant:
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

    @property
    def api(self) -> GenerationServer:
        return self.server.api  # type: ignore[attr-defined]

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

    def _sse_start(self) -> None:
        self.send_response(200)
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
            _normalized_messages(body.get("messages"), body.get("tools"))
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
        created = int(time.time())
        model = str(body.get("model") or self.api.model_id)
        user_key = str(body.get("user") or _DEFAULT_USER)
        client = f"{self.client_address[0]}:{self.client_address[1]}" if self.client_address else "?"
        logger.debug(_describe_request(completion_id, client, user_key, body))

        def frame(delta: dict, finish) -> dict:
            return {
                "id": completion_id,
                "object": chunk_kind,
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        if stream:
            self._sse_start()
            first = [True]
            stopped = [False]

            def on_chunk(reasoning: str, content: str) -> None:
                if stopped[0] or (not reasoning and not content):
                    return
                if chat:
                    delta: dict = {}
                    if first[0]:
                        delta["role"] = "assistant"
                        first[0] = False
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
                stats = self.api.generate(user_key, body, on_chunk)
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
                self._sse_event(frame({}, stats["finish_reason"]))
                self._sse_done()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        try:
            stats = self.api.generate(user_key, body, lambda r, c: None)
        except Exception as e:
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
        self._send_json(
            {
                "id": completion_id,
                "object": object_kind,
                "created": created,
                "model": model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
                },
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
        choices=("high", "max"),
        default=None,
        help="reasoning-effort hint, only meaningful with --think",
    )
    p.add_argument("--trace-region-size", type=int, default=sys_cfg.device.trace_region_size)
    p.add_argument("--quiet", action="store_true", help="only warnings and above from the model logs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # See ``chat_cli.parse_args``: the profile has to be resolved before the parser is
    # built, since it supplies the sizing defaults. ``variant="server"`` picks the
    # server flavour of whichever machine profile applies.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--system-profile", default=None)
    pre.add_argument("--system-config-file", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    sys_cfg = load_system_config(profile=pre_args.system_profile, path=pre_args.system_config_file, variant="server")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        "decode, ~30x slower. Every chunk attends buffers sized for it, so a smaller value "
        "prefills shorter prompts faster but replays the excess of longer ones",
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
        default=int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", "4096")),
        help="bytes of L1_SMALL reserved for the CCL semaphores when the prefill model is on",
    )
    p.add_argument(
        "--max-body-bytes",
        type=int,
        default=16 << 20,
        help="largest accepted request body in bytes (default: 16 MiB)",
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
    args = p.parse_args(argv)
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
    return args


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    if args.prefill:
        # The layout of tests/prefill/test_prefill_decode_demo.py: ``1 x 4`` TP4 stages, one per
        # mesh row (8 x 4 = all 32 chips of a Galaxy), shared by the decode and the prefill model.
        chips = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
        if chips % _TP_SIZE:
            print(f"the prefill model needs TP{_TP_SIZE} stages, but the system has {chips} chips", file=sys.stderr)
            return 1
        args.tp_size, args.num_stages = _TP_SIZE, chips // _TP_SIZE
        # Prefill replays overwrite decode's L1, which is snapshot and written back; a GCB's
        # config pages cannot be, so decode copies its weights DRAM -> L1 per call instead.
        args.use_prefetcher = False
        mesh_shape, l1_small_size = (args.num_stages, _TP_SIZE), args.l1_small_size
        # MTP disabled (read by DeepSeekV4Model.__init__): every row is a pipeline stage.
        os.environ["DEEPSEEK_V4_DSPARK"] = "0"
        logger.info(
            f"traced prefill on: {args.num_stages} x TP{_TP_SIZE} stages ({chips} chips) shared with decode, "
            f"decode prefetcher off"
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
            engine = ChatEngine(mesh_device, args, prefetcher)
            # Decode compile, then prefill build + compile (decode's L1 parked): both before any
            # trace exists. The warmup then captures decode, and prefill is captured after it.
            prefiller = _Prefiller(engine, args, prefetcher) if args.prefill else None
            engine.warmup()
            if prefiller is not None:
                prefiller.capture()
            api = GenerationServer(engine, args.model_id, args.max_body_bytes, args.prefill_chunk, prefiller=prefiller)
            # The scheduler is the only thread that touches the device from here on; the
            # traces it replays were captured by the warmup above, which a pipelined
            # caller cannot do mid-flight.
            api.start()
            httpd = _ModelHTTPServer((args.host, args.port), _Handler, api)
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
