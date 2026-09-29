# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible chat server over the single-trace decode chain (one device, requests queued in arrival order).

``POST /v1/chat/completions`` (streaming SSE or one JSON document; tools,
``reasoning_content``, stop strings, a per-request thinking budget; with
``--sampling`` also sampling with the OpenAI fields plus ``top_k``, ``min_p``,
``repetition_penalty`` and ``greedy``, ``seed`` echoed, ``logprobs`` from the
candidate row), ``GET /v1/models`` and ``GET /v1/models/<id>``, ``GET /health``.  On a sampling server a
request naming no sampling field, ``temperature 0`` or ``greedy`` is the bitwise
greedy loop (the argmax the TAIL resolves; the candidate row is never read); a
request with ``temperature > 0`` samples with it and one naming another sampling
field without a temperature takes the model card's profile for its thinking
mode; the draw runs on the device unless ``--host-sampler``.  Without ``--sampling`` (``--no-sampling``, the argparse default) TAIL
captures no candidate row, the loop is the greedy one at its measured period and
explicit sampling fields are refused; the launchers pass ``--sampling``.  The
device prompt of every request is the reference render of the client's
messages: no system prompt is added, and a follow-up turn holds the served reply
as the template re-renders it, never the recorded reasoning.  Runs under
``tools/run_qwen38_chat_server.sh`` (or a development launcher): the server admits the runtime it
imports (the ttnn built from this checkout, ``runtime_admission``), prepares the
model inputs on the CPU, opens the mesh, converts the routed experts on the
first start, replays the CPU acceptance records after the captures
(``--sampling-discriminator`` then runs the sampling chain arms and stops;
``--agreement-reference`` teacher-forces the reference corpus and writes the
agreement records, the device column and its score against the HF reference),
writes READY, serves until SIGTERM, then releases the chain and the mesh in the
timing runner's order.  ``--host`` is loopback unless the profile serves the
LAN (the QuietBox, the p150 line) or ``--allow-lan`` is given.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import http.server
import itertools
import json
import os
import queue
import secrets
import select
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import ttnn
from models.demos.blackhole.qwen38_flash_next import mrope, vision_splice
from models.demos.blackhole.qwen38_flash_next.chat import (
    EOS_TOKEN_IDS,
    PINNED_TOKENIZER_ARTIFACTS,
    VOCAB_SIZE,
    Qwen38OfficialChatTemplate,
)
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_protocol as protocol
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_reference_corpus as reference_corpus
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as sampling_step
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vision_inputs as vision_inputs
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode, runtime_admission
from models.demos.blackhole.qwen38_flash_next.tools.checkpoint_budget import vision_resident_layout
from models.demos.blackhole.qwen38_flash_next.tools.evidence_records import (
    append_marker,
    append_phase_record,
    utc_now,
    write_result,
)
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import (
    construct_live_decode_diagnostic,
    missing_bf4_layers,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import Qwen38ChatRequestRejected
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    CHUNK_PREFILL_MIN_ROWS,
    DEFAULT_PREFILL_MODE,
    LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES,
    MAX_TOKENS_BOUND,
    MTP_DRAFTS,
    MTP_GDN_ANCHORS,
    PREFILL_MODES,
    RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND,
    Qwen38ChatChainError,
    Qwen38ChatSession,
    construct_chain,
    lanes_capacity_admission,
    mtp_capacity_admission,
    mtp_verify_forms,
    open_partition_b_mesh,
    resolve_route,
    template_decoder,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler import (
    DEFAULT_STALL_BUDGET_SECONDS,
    Qwen38LaneScheduler,
    Qwen38LaneSchedulerBusy,
    Qwen38LaneTicket,
    lane_geometry,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_mtp_device_accept import SWITCH as DEVICE_ACCEPT_SWITCH
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_mtp_device_accept import device_accept_switch
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    RESIDENT_MAX_QSA_CACHE_CAPACITY,
    RESIDENT_QSA_CACHE_CAPACITIES,
    Qwen38ResidentContext,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import is_slab_rows
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_scan as gdn_rows_scan_module
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import (
    admit_slab_moe_switches,
    moe_local_output_enabled,
    moe_rows_form,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.vision_residency import (
    PEAK_ACTIVATION_BYTES_PER_ROW_PER_DIE,
    VISION_ROW_BUCKETS,
    compose_warm_hooks,
)

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
ACCEPTANCE_CONTINUATION = 96
ACCEPTANCE_GATE_PROMPT = "json"
ACCEPTANCE_HANDOFF_SPLIT = 40  # MTP chains: the gate record replayed as 40 tokens in one mode then 56 in the other
DISCRIMINATOR_PROMPT = "story"
MAX_REQUEST_BYTES = 16 << 20
# Decision 6 (2026-09-03): thinking ON for Hermes, medium when the client sends no effort.
ENABLE_THINKING_DEFAULT = True
REASONING_EFFORT_DEFAULT = "medium"
# A request without max_tokens gets the remaining context (context limit less the prompt) and an explicit one is
# bounded by it: the session's require_budget, once the prompt is rendered.  /health and /v1/models say so.
MAX_TOKENS_RULE = {"default_max_tokens": "remaining context", "max_tokens_limit": "remaining context"}
QUEUE_LIMIT = 4
QUEUE_POLL_SECONDS = 0.5  # a queued request looks at its client's socket this often: a hang-up gives up its place
RETRY_AFTER_SECONDS = 5
HEARTBEAT_SECONDS = 30.0
# A socket write that makes no progress for this long (a streaming reader that stopped reading) raises, so the device
# loop ends the request as "disconnected" instead of blocking in sendall with the device held.
SOCKET_TIMEOUT_SECONDS = 60.0
# The stop signal waits this long for the request in flight to end at its next poll and release the device before
# the chain is released; the launchers' timeout --kill-after is 60 s.
DRAIN_SECONDS = 30.0
ALLOWED_METHODS = "GET, HEAD, OPTIONS, POST"
# The template flags a client may send under chat_template_kwargs (the vLLM/SGLang convention the model card uses).
TEMPLATE_KWARGS = ("enable_thinking", "reasoning_effort", "preserve_thinking")
# Every field parse_chat_request reads; any other top-level key is logged as ignored.
KNOWN_REQUEST_FIELDS = frozenset(
    (
        "model",
        "messages",
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "response_format",
        "parallel_tool_calls",
        "logit_bias",
        "user",
        "chat_template_kwargs",
        "enable_thinking",
        "reasoning_effort",
        "thinking_budget",
        "ignore_eos",
        "prefill_mode",
        "mtp_drafts",
    )
    + sampling_step.SAMPLING_REQUEST_FIELDS
)
# A request's seed when it sends none: 63 random bits, echoed in the response.
SEED_BITS = 63
PROFILER_VARIABLES = ("TT_METAL_DEVICE_PROFILER", "TT_METAL_PROFILER_CPP_POST_PROCESS", "TTNN_OP_PROFILER")


class Qwen38ChatServerStop(RuntimeError):
    """SIGTERM or SIGINT: the launcher's timeout or the user; a clean shutdown."""


def _handle_stop_signal(signum: int, _frame: Any) -> None:
    raise Qwen38ChatServerStop(f"chat server received signal {signum}")


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"utc": utc_now(), "event": event, **fields}, sort_keys=True), flush=True)


# MTP drafting for sampled requests: on by default on an --mtp --sampling server (the split verify is captured beside
# the fused one: greedy requests run the fused verify, the pass loop drafts for sampled requests through the split form
# by exact speculative sampling); QWEN38_MTP_SAMPLED=0 restores the plain sampled path (the fused verify alone, sampled
# requests on the 1-row loop).  A server without --mtp or without --sampling has no drafting for sampled requests: unset
# resolves to off there and an explicit 1 is refused at start.  Any other value is refused.
MTP_SAMPLED_VARIABLE = "QWEN38_MTP_SAMPLED"


def effective_device_sampler(args: Any) -> bool:
    """The server's sampler: the on-device sampler is the ``--sampling`` default on a one-stream server; an ``--mtp``
    server samples on the host (its TAIL resolves the greedy token for the MTP row and the pass loop's point-mass
    decision is the host's: ``ttnn/speculative_sampling.py``), so ``--device-sampler`` with ``--mtp`` is refused and
    ``--host-sampler`` is its default.  Without ``--sampling`` there is no sampler at all."""

    requested = args.device_sampler  # None: the default; True / False: --device-sampler / --host-sampler
    if requested and not args.sampling:
        raise SystemExit("--device-sampler needs --sampling (the sampler reads the candidate row)")
    if requested and args.mtp is not None:
        raise SystemExit(
            "--device-sampler is not served with --mtp: the MTP chain samples on the host (its tail resolves the greedy "
            "token for the draft row; the pass loop's point-mass decision is the host's); drop one of the two"
        )
    if requested is None:
        return bool(args.sampling) and args.mtp is None
    return bool(requested)


MTP_DRAFTS_PER_REQUEST_VARIABLE = "QWEN38_MTP_DRAFTS_PER_REQUEST"
# The pair of chains a per-request server captures: the default (--mtp K) and the other member; a request picks with
# extra_body.mtp_drafts (docs/SERVER.md).  The measured pair (2026-09-26): k = 5 on the 6-row verify MoE form pays on
# structured output (json, code), k = 4 stays the chat / prose default.
MTP_DRAFTS_PER_REQUEST_PAIR = (4, 5)


def mtp_drafts_per_request_switch(environment: Mapping[str, str], *, drafts: int | None) -> tuple[int, ...]:
    """``QWEN38_MTP_DRAFTS_PER_REQUEST``: 1 captures the other chain of :data:`MTP_DRAFTS_PER_REQUEST_PAIR` beside the
    server's ``--mtp`` chain and admits ``mtp_drafts`` in requests; the result is the admitted draft counts, the default
    first (empty = one chain, the field refused).  Unset or 0 leaves one chain; 1 needs ``--mtp`` in the pair."""

    value = environment.get(MTP_DRAFTS_PER_REQUEST_VARIABLE)
    if value is None or value == "0":
        return ()
    if value != "1":
        raise SystemExit(f"{MTP_DRAFTS_PER_REQUEST_VARIABLE} must be 0 or 1, got {value!r}")
    if drafts not in MTP_DRAFTS_PER_REQUEST_PAIR:
        raise SystemExit(
            f"{MTP_DRAFTS_PER_REQUEST_VARIABLE}=1 needs --mtp in {MTP_DRAFTS_PER_REQUEST_PAIR} (the chains it captures), "
            f"got --mtp {drafts}"
        )
    return (drafts,) + tuple(count for count in MTP_DRAFTS_PER_REQUEST_PAIR if count != drafts)


def mtp_sampled_switch(environment: Mapping[str, str], *, applicable: bool = True) -> bool:
    """The switch's resolution; ``applicable``: the server runs with ``--mtp`` and ``--sampling``."""

    value = environment.get(MTP_SAMPLED_VARIABLE)
    if value is None:
        return applicable
    if value not in ("0", "1"):
        raise SystemExit(f"{MTP_SAMPLED_VARIABLE} must be 0 or 1, got {value!r}")
    return value == "1"


# The device acceptance (qwen38_mtp_device_accept.SWITCH, QWEN38_MTP_DEVICE_ACCEPT, default off): with it on beside the
# split verify the chain warms and captures the device-decided sampled form too (the head, fused.mtp_accept and the
# tail in one trace); which sampled requests run it is the acceptance's admission.  An explicit 1 without the split
# verify is refused at start.  QWEN38_MTP_DEVICE_ACCEPT_DUMP=<dir> (dev) makes the session record every device-decided
# pass's candidate rows and write one JSON per request for the acceptance's gate tool (dev).
DEVICE_ACCEPT_DUMP_VARIABLE = "QWEN38_MTP_DEVICE_ACCEPT_DUMP"

# The MTP pass's early PLE rows (QWEN38_MTP_PLE_EARLY; on by default with --mtp since 2026-09-26, =0 restores the
# one-queue open and the whole-row lookup after the pass row): the mesh opens with a second command queue and the fused /
# device-decided pass loops read the verify lanes there while the draft runs, looking the next pass's PLE rows 0-1 up
# under the draft (mtp_v2.Qwen38TTNNEarlyRowsReader); the second queue carries nothing but that buffer read.  The
# host-decided split form keeps its order either way.  An explicit 1 without --mtp is refused at start.
PLE_EARLY_VARIABLE = "QWEN38_MTP_PLE_EARLY"


def ple_early_switch(environment: Mapping[str, str], *, applicable: bool = True) -> bool:
    """``QWEN38_MTP_PLE_EARLY``: ``1`` on, ``0`` off, unset = ``applicable`` (the server runs with ``--mtp``); anything
    else refused."""

    value = environment.get(PLE_EARLY_VARIABLE)
    if value is None:
        return applicable
    if value not in ("0", "1"):
        raise SystemExit(f"{PLE_EARLY_VARIABLE} must be 0 or 1, got {value!r}")
    return value == "1"


# -- requests and responses ----------------------------------------------------------------------


def lanes_switches(environment: Mapping[str, str], args: Any) -> dict[str, Any] | None:
    """``--lanes B``'s admission at start (None without it): B in 2..8 with B x (k + 1) <= 32 rows, ``--mtp``
    required, greedy only (``--sampling``, ``--sampling-discriminator`` and ``--agreement-reference`` refused), the
    second command queue's early rows read (``QWEN38_MTP_PLE_EARLY=1``) and the per-request drafting chains
    (``QWEN38_MTP_DRAFTS_PER_REQUEST=1``) refused (neither has a lanes form).  The verify-rows fold serves the lanes
    through its lanes form (``ttnn/fused/gdn_rows_scan``), so a ``--lanes`` process runs the same fused set as the
    single-stream server.  Returns the geometry."""

    lanes = int(args.lanes)
    if lanes == 0:
        return None
    if args.mtp is None:
        raise SystemExit("--lanes needs --mtp (the lanes are the MTP lane chain: B x (k + 1) rows in one tile)")
    try:
        lanes, rows = lane_geometry(lanes, int(args.mtp))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.sampling:
        raise SystemExit(
            "--lanes serves greedy requests only (the lane body resolves by argmax; the sampled lanes form is a later "
            "wave): drop --sampling (the argparse default, --no-sampling, refuses sampling fields with HTTP 400)"
        )
    if args.sampling_discriminator or args.agreement_reference is not None:
        raise SystemExit("--lanes does not combine with --sampling-discriminator or --agreement-reference")
    if environment.get(PLE_EARLY_VARIABLE, "").strip() == "1":
        raise SystemExit(
            f"{PLE_EARLY_VARIABLE}=1 has no lanes form: unset it under --lanes (the lanes run the one-queue form)"
        )
    if environment.get(MTP_DRAFTS_PER_REQUEST_VARIABLE, "").strip() == "1":
        raise SystemExit(f"{MTP_DRAFTS_PER_REQUEST_VARIABLE}=1 has no lanes form: unset it under --lanes")
    return {"lanes": lanes, "rows": rows, "drafts": int(args.mtp)}


def parse_chat_request(
    document: Any, *, seed: int | None = None, sampling_available: bool = True, mtp_drafts_admitted: Sequence[int] = ()
) -> dict[str, Any]:
    """The fields the server honours, validated with actual-vs-expected messages (one completion; the sampling
    fields per ``qwen38_sampling_step``, ``extra_body`` merged under the top level, JSON null an absent field,
    ``chat_template_kwargs`` the thinking flags' other spelling).  ``seed`` is the server's draw for a request that
    sends none.  A greedy-only server (``sampling_available`` false) refuses explicit sampling fields and runs a
    request without them greedily.  A field the server cannot honour (structured output, logit_bias, serial tool
    calls) is refused, never dropped.  Raises ``ValueError`` subclasses carrying ``param`` and ``code``."""

    if not isinstance(document, Mapping):
        raise Qwen38ChatRequestRejected(f"request body must be a JSON object, got {type(document).__name__}")
    extra = document.get("extra_body")
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise Qwen38ChatRequestRejected(
                f"extra_body must be an object, got {type(extra).__name__}", param="extra_body"
            )
        document = {**extra, **{key: value for key, value in document.items() if key != "extra_body"}}
    document = {key: value for key, value in document.items() if value is not None}
    template_kwargs = document.get("chat_template_kwargs", {})
    if not isinstance(template_kwargs, Mapping):
        raise Qwen38ChatRequestRejected(
            f"chat_template_kwargs must be an object, got {type(template_kwargs).__name__}",
            param="chat_template_kwargs",
        )
    for key, value in template_kwargs.items():
        if key not in TEMPLATE_KWARGS:
            raise Qwen38ChatRequestRejected(
                f"chat_template_kwargs.{key} is not a template flag of this server (the flags are {TEMPLATE_KWARGS})",
                param=f"chat_template_kwargs.{key}",
            )
        if key == "preserve_thinking" and value is not True:
            raise Qwen38ChatRequestRejected(
                f"chat_template_kwargs.preserve_thinking must be true (the server keeps the reasoning), got {value!r}",
                param="chat_template_kwargs.preserve_thinking",
            )
        if key in document and document[key] != value:
            raise Qwen38ChatRequestRejected(
                f"chat_template_kwargs.{key} {value!r} contradicts {key} {document[key]!r}",
                param=f"chat_template_kwargs.{key}",
            )
    document = {**{key: template_kwargs[key] for key in TEMPLATE_KWARGS[:2] if key in template_kwargs}, **document}
    # The user messages' image_url parts (data: URLs; video refused by the protocol), decoded here so a bad image
    # is a 400 naming its part; the tower runs on the device thread later, the grid decides the prompt's pads.
    image_parts: list[protocol.Qwen38ImagePart] = []
    messages = protocol.normalize_messages(document.get("messages"), images=image_parts)
    images: list[vision_inputs.Qwen38DecodedImage] = []
    for part in image_parts:
        where = f"messages[{part.message_index}].content[{part.part_index}].image_url"
        try:
            images.append(vision_inputs.decode_image(part.data, part.detail))
        except vision_inputs.Qwen38ImageError as error:
            raise Qwen38ChatRequestRejected(f"{where}: {error}", param=where, code="invalid_image") from error
    tool_choice = document.get("tool_choice", "auto")
    if tool_choice not in protocol.TOOL_CHOICES:
        raise Qwen38ChatRequestRejected(
            f"tool_choice must be one of {protocol.TOOL_CHOICES} (greedy server), got {tool_choice!r}",
            param="tool_choice",
        )
    tools = protocol.validate_tools(document.get("tools")) if tool_choice == "auto" else []
    stream = document.get("stream", False)
    if type(stream) is not bool:
        raise Qwen38ChatRequestRejected(f"stream must be a boolean, got {stream!r}", param="stream")
    stream_options = document.get("stream_options", {})
    if not isinstance(stream_options, Mapping) or set(stream_options) - {"include_usage"}:
        raise Qwen38ChatRequestRejected(
            f"stream_options may only hold include_usage (usage rides on the final chunk), got {stream_options!r}",
            param="stream_options",
        )
    response_format = document.get("response_format", {"type": "text"})
    kind = response_format.get("type") if isinstance(response_format, Mapping) else response_format
    if kind != "text":
        raise Qwen38ChatRequestRejected(
            f"response_format.type must be 'text' (this server has no constrained decoding), got {kind!r}",
            param="response_format",
        )
    mtp_drafts = document.get("mtp_drafts")
    if mtp_drafts is not None:
        # The chain a request drafts with (extra_body.mtp_drafts): one of the draft counts this server captured at
        # open (QWEN38_MTP_DRAFTS_PER_REQUEST=1: the --mtp chain and the pair's other member); anything else is
        # refused with the admitted list, and a one-chain server refuses the field rather than dropping it.
        admitted = tuple(mtp_drafts_admitted)
        if not admitted:
            raise Qwen38ChatRequestRejected(
                f"mtp_drafts is not admitted by this server (one drafting chain; {MTP_DRAFTS_PER_REQUEST_VARIABLE}=1 "
                f"opens the pair): drop it, got {mtp_drafts!r}",
                param="mtp_drafts",
            )
        if isinstance(mtp_drafts, bool) or type(mtp_drafts) is not int or mtp_drafts not in admitted:
            raise Qwen38ChatRequestRejected(
                f"mtp_drafts must be one of {list(admitted)} (the chains this server captured), got {mtp_drafts!r}",
                param="mtp_drafts",
            )
    if document.get("logit_bias", {}) != {}:
        raise Qwen38ChatRequestRejected(
            f"logit_bias is not applied by this server: drop it, got {document.get('logit_bias')!r}", param="logit_bias"
        )
    if document.get("parallel_tool_calls", True) is not True:
        raise Qwen38ChatRequestRejected(
            f"parallel_tool_calls must be true (every call the model emits is returned), "
            f"got {document.get('parallel_tool_calls')!r}",
            param="parallel_tool_calls",
        )
    if not isinstance(document.get("user", ""), str):
        raise Qwen38ChatRequestRejected(f"user must be a string, got {document.get('user')!r}", param="user")
    if document.get("n", 1) != 1:
        raise Qwen38ChatRequestRejected(f"n must be 1 (one greedy completion), got {document.get('n')!r}", param="n")
    # None (neither name sent) stays None: the session resolves the remaining context once the prompt is known.
    max_tokens = document.get("max_tokens", document.get("max_completion_tokens"))
    if max_tokens is not None and (type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS_BOUND):
        raise Qwen38ChatRequestRejected(
            f"max_tokens must be an integer in [1, {MAX_TOKENS_BOUND}] when given, got {max_tokens!r}",
            param="max_tokens",
        )
    enable_thinking = document.get("enable_thinking", ENABLE_THINKING_DEFAULT)
    if type(enable_thinking) is not bool:
        raise Qwen38ChatRequestRejected(
            f"enable_thinking must be a boolean, got {enable_thinking!r}", param="enable_thinking"
        )
    reasoning_effort = protocol.effort_level(document.get("reasoning_effort", REASONING_EFFORT_DEFAULT))
    requested_budget = document.get("thinking_budget")
    if requested_budget is not None and (type(requested_budget) is not int or requested_budget < 0):
        raise Qwen38ChatRequestRejected(
            f"thinking_budget must be a non-negative integer, got {requested_budget!r}", param="thinking_budget"
        )
    ignore_eos = document.get("ignore_eos", False)
    if type(ignore_eos) is not bool:
        raise Qwen38ChatRequestRejected(f"ignore_eos must be a boolean, got {ignore_eos!r}", param="ignore_eos")
    prefill_mode = document.get("prefill_mode")
    if prefill_mode is not None and prefill_mode not in PREFILL_MODES:
        raise Qwen38ChatRequestRejected(
            f"prefill_mode must be one of {PREFILL_MODES} when given, got {prefill_mode!r}", param="prefill_mode"
        )
    try:
        sampling = sampling_step.parameters_from_request(
            document, enable_thinking=enable_thinking, seed=secrets.randbits(SEED_BITS) if seed is None else seed
        )
        logprobs, top_logprobs = sampling_step.logprobs_from_request(document)
    except sampling_step.Qwen38SamplingRequestError as error:
        raise Qwen38ChatRequestRejected(str(error), param=str(error).split(" ", 1)[0]) from error
    if logprobs and sampling is None:
        raise Qwen38ChatRequestRejected(
            "logprobs need a sampled request (temperature > 0): the greedy loop reads no candidate row",
            param="logprobs",
        )
    if sampling is not None and not sampling_available:
        explicit = [
            name
            for name in sampling_step.SAMPLING_REQUEST_FIELDS
            if name not in ("n", "greedy") and document.get(name) is not None
        ]
        if explicit:
            raise Qwen38ChatRequestRejected(
                f"sampling is unavailable on this server (greedy only): drop {explicit} or send temperature 0",
                param=explicit[0],
            )
        sampling = None
    return {
        "messages": messages,
        "raw_messages": document.get("messages"),
        "images": images,
        "tools": tools,
        "stream": stream,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "thinking_budget": requested_budget,
        "stop": protocol.validate_stop(document.get("stop")),
        "ignore_eos": ignore_eos,
        "prefill_mode": prefill_mode,
        "mtp_drafts": mtp_drafts,
        "sampling": sampling,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "ignored": sorted(set(document) - KNOWN_REQUEST_FIELDS),
    }


def _usage(prompt_tokens: int, completion_tokens: int, queue_wait: float) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "queue_wait_seconds": round(queue_wait, 4),
    }


def _extension(
    completion: Any,
    assembler: protocol.Qwen38ReplyAssembler,
    *,
    queue_wait: float,
    think_budget: int | None,
    sampling: sampling_step.Qwen38SamplingRequest | None,
) -> dict[str, Any]:
    """The ``qwen38`` object of a response (and the ledger's per-request fields).  ``decode_loop`` records which
    loop produced the tokens: ``greedy`` (the TAIL's argmax, bitwise the reference; a request without sampling
    fields on any server) or ``sampled`` (the candidate-row sampler).  ``served_reasoning_tokens`` is the reasoning
    of earlier turns held in the device prompt beyond the client's history: always 0, the prompt is the reference
    render; kept as a field."""

    return {
        **completion.as_dict(),
        "finish": completion.finish_reason,
        "queue_wait_seconds": round(queue_wait, 4),
        "decode_loop": "greedy" if sampling is None else "sampled",
        "served_reasoning_tokens": 0,
        "sampling": None if sampling is None else sampling.as_dict(),
        "seed": None if sampling is None else sampling.parameters.seed,
        "reasoning_tokens": assembler.reasoning_tokens,
        "thinking_forced": protocol.THINK_END_ID in completion.token_ids
        and think_budget is not None
        and assembler.reasoning_tokens >= think_budget,
        "stop_string_hit": assembler.stop_hit,
        "tool_calls": len(assembler.calls),
        "tool_parse_errors": assembler.parse_errors,
        "truncated_tool_call": assembler.truncated_tool_call,
    }


def _completion_id() -> str:
    return f"chatcmpl-{secrets.token_hex(8)}"


def _vmrss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


# OpenAI finish_reason from the session's reason and the assembled reply.
def _finish_reason(session_finish: str, assembler: protocol.Qwen38ReplyAssembler) -> str:
    if assembler.stop_hit:
        return "stop"
    if assembler.calls:
        return "tool_calls"
    return {"deadline": "length", "shutdown": "length", "disconnected": "stop"}.get(session_finish, session_finish)


class Qwen38ServerBusy(RuntimeError):
    """The bounded request queue is full: HTTP 503 with Retry-After."""


class Qwen38ClientGone(RuntimeError):
    """The client hung up while its request waited for the device: the ticket is dropped, nothing is generated."""


def _vision_health(residency: Any) -> dict[str, Any]:
    """The ``/health.vision`` form: the residency decision and the buckets (READY carries the whole record)."""

    summary = residency.vision_summary()
    return {
        key: summary[key]
        for key in ("resident", "shortfall", "shortfall_bytes_per_bank", "buckets", "resident_bytes_per_bank")
    }


class Qwen38ChatHTTPServer(http.server.ThreadingHTTPServer):
    """Threaded accept, parse and validate; one device at a time through a bounded FIFO turnstile.

    A request that fails validation is answered at once even while the device
    is busy; a valid one waits its turn behind at most ``queue_limit`` others
    (``queue_wait_seconds`` is reported) or is refused with 503 above the bound.
    A queued request whose client hangs up gives up its place.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        session: Qwen38ChatSession | None,
        *,
        ledger: Path,
        queue_limit: int = QUEUE_LIMIT,
        request_deadline_seconds: float | None = None,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        socket_timeout_seconds: float = SOCKET_TIMEOUT_SECONDS,
        stall_seconds: float | None = None,
        system_fingerprint: str | None = None,
        dram_after_captures: Mapping[str, Any] | None = None,
        mtp_drafts_admitted: Sequence[int] = (),
        listen: bool = True,
        lanes: Qwen38LaneScheduler | None = None,
    ) -> None:
        # The address is bound here, so a taken port or a host the address family cannot carry fails at once (main
        # constructs the server before the mesh opens, with the session attached after the captures); connections
        # are accepted only from server_activate (``listen`` false: main's, right before serve_forever).
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, Qwen38ChatHandler, bind_and_activate=False)
        try:
            self.server_bind()
            if listen:
                self.server_activate()
        except BaseException:
            self.server_close()
            raise
        self.session = session
        self.vision = None  # the tower's residency (ttnn.vision_residency), attached after the chain's warm hook
        self.ledger = ledger
        self.queue_limit = queue_limit
        self.request_deadline_seconds = request_deadline_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.socket_timeout_seconds = socket_timeout_seconds
        # A request whose device made no progress (no should_stop poll: a step, a prefill event) for this long is a
        # wedge, not a long prompt: the server ends as fatal so the launcher restarts it.  None: no watchdog.
        self.stall_seconds = stall_seconds
        self.progress: dict[str, Any] | None = None  # the request holding the device: start, last poll, polls
        self.system_fingerprint = system_fingerprint  # source head + runtime .so: what a seed reproduces against
        self.mtp_drafts_admitted = tuple(mtp_drafts_admitted)  # the chains a request may pick (extra_body.mtp_drafts)
        # The mesh allocator read after the traces were captured (hardware_profiles.symmetric_mesh_dram_memory: one observation
        # that applies to every card): free_bytes_per_bank is the build's headroom, reported as read, never updated.
        self.dram_after_captures = dram_after_captures
        self.created = int(time.time())  # /v1/models created: when this server came up
        # ``--lanes B``: the scheduler over the lane chain replaces the turnstile (its driver thread owns the device;
        # handler threads submit tickets and read their queues); None is the single-stream server.
        self.lanes = lanes
        self.lanes_thread: threading.Thread | None = None
        self.turnstile = threading.Condition()
        self.waiting: collections.deque[int] = collections.deque()
        self.tickets = itertools.count()
        self.busy = False
        self.stopping = False  # the stop signal came: new and queued requests get 503, the in-flight one ends
        self.last_prefill_ms_per_token: float | None = None
        self.fatal: BaseException | None = None

    def vision_refusal(self, images: Sequence[Any]) -> str | None:
        """Why an image request cannot be served now (None = it can): a process whose tower is not resident (the
        admission's shortfall), or an image above the largest prewarmed row bucket.  The HTTP 400 text.  The lanes
        server serves images like the single stream (the tower runs inside the lane admission)."""

        if self.vision is None:
            return "the vision tower is not resident in this process"
        for image in images:
            grid = image.grid
            reason = self.vision.refusal_reason(grid.t * grid.h * grid.w)
            if reason is not None:
                return reason
        return None

    def vision_prompt_for(
        self, prompt_ids: Sequence[int], positions: Any, images: Sequence[Any]
    ) -> tuple[Any, dict[str, Any]]:
        """On the device thread: every image through the resident tower in its row bucket (the processor's pixel
        patches, the merged feature rows in prompt order) and the splice's prompt object; the request's record."""

        import torch

        rows: list[torch.Tensor] = []
        per_image: list[dict[str, Any]] = []
        started = time.perf_counter()
        for image in images:
            patches = vision_inputs.pixel_patches(image)
            grid_thw = torch.tensor([list(image.grid.as_tuple())], dtype=torch.long)
            image_started = time.perf_counter()
            output = self.vision.run_image_in_bucket(patches, grid_thw)
            features = self.vision.tower.features_to_torch(output).to(torch.bfloat16)
            ttnn.deallocate(output.features)
            rows.append(features)
            per_image.append(
                {
                    "width": image.width,
                    "height": image.height,
                    "detail": image.detail,
                    "grid_thw": list(image.grid.as_tuple()),
                    "patches": output.patches,
                    "rows": output.rows,
                    "tokens": output.tokens,
                    "tower_seconds": round(time.perf_counter() - image_started, 4),
                    "device_seconds": round(output.timing["device_s"], 4),
                }
            )
        prompt = vision_splice.Qwen38VisionPrompt(
            positions,
            torch.cat(rows),
            digest=vision_inputs.request_digest(images),
            image_digests=vision_inputs.image_digests(images),
        )
        prompt.validate_prompt(prompt_ids)
        record = {
            "images": per_image,
            "image_tokens": sum(row["tokens"] for row in per_image),
            "tower_seconds": round(time.perf_counter() - started, 4),
            "rope_delta": positions.delta,
            "digest": prompt.digest,
        }
        _log(
            "vision_request", **{key: value for key, value in record.items() if key != "images"}, images=len(per_image)
        )
        return prompt, record

    @property
    def queue_depth(self) -> int:
        return len(self.waiting) if self.lanes is None else self.lanes.waiting_count

    @property
    def device_busy(self) -> bool:
        """A request holds the device: the turnstile's, or any lane active."""

        return self.busy if self.lanes is None else self.lanes.active_count > 0

    def current_request(self) -> dict[str, Any] | None:
        """The request holding the device: when it started and how long since the device last completed a step
        (should_stop is polled after every one), so a long prefill can be told from a wedge."""

        progress = self.progress
        if progress is None:
            return None
        now = time.perf_counter()
        return {
            "id": progress["id"],
            "started_utc": progress["started_utc"],
            "elapsed_seconds": round(now - progress["started"], 3),
            "seconds_since_progress": round(now - progress["last_progress"], 3),
            "polls": progress["polls"],
        }

    def enqueue(self) -> int:
        """A place in the FIFO (the ticket), or ``Qwen38ServerBusy`` when ``queue_limit`` requests already wait or
        the server is stopping."""

        with self.turnstile:
            if self.stopping:
                raise Qwen38ServerBusy("the server is stopping")
            if len(self.waiting) >= self.queue_limit:
                raise Qwen38ServerBusy(f"{len(self.waiting)} requests are queued, the limit is {self.queue_limit}")
            ticket = next(self.tickets)
            self.waiting.append(ticket)
            return ticket

    def await_turn(self, ticket: int, abandoned: Callable[[], bool]) -> float:
        """Wait for the device in arrival order; returns the seconds waited.  ``abandoned`` is asked every
        ``QUEUE_POLL_SECONDS`` whether the client is still there: a hang-up drops the ticket (``Qwen38ClientGone``);
        a stop while waiting drops it with ``Qwen38ServerBusy``."""

        started = time.perf_counter()
        with self.turnstile:
            while not self.turnstile.wait_for(
                lambda: self.stopping or (not self.busy and self.waiting[0] == ticket), timeout=QUEUE_POLL_SECONDS
            ):
                if abandoned():
                    self.waiting.remove(ticket)
                    self.turnstile.notify_all()
                    raise Qwen38ClientGone(
                        f"the client hung up after {time.perf_counter() - started:.1f} s in the queue"
                    )
            if self.stopping:
                self.waiting.remove(ticket)
                self.turnstile.notify_all()
                raise Qwen38ServerBusy("the server is stopping")
            self.waiting.popleft()
            self.busy = True
            return time.perf_counter() - started

    def drain(self, seconds: float) -> bool:
        """The stop: no more connections (the listening socket closes), queued requests are refused, the request in
        flight ends at its next poll with finish ``shutdown``; waits up to ``seconds`` for the device to be released
        and says whether it was (False: the chain's ownership is uncertain)."""

        with self.turnstile:
            self.stopping = True
            self.turnstile.notify_all()
        self.server_close()
        if self.lanes is not None:
            # The lanes: the driver ends every active lane's request with ``shutdown`` at its next pass boundary and
            # leaves its loop; the device is free once the thread has ended.
            self.lanes.stop()
            if self.lanes_thread is not None:
                self.lanes_thread.join(timeout=seconds)
                return not self.lanes_thread.is_alive() and self.lanes.fatal is None
            return True
        with self.turnstile:
            return self.turnstile.wait_for(lambda: not self.busy, timeout=seconds)

    def withdraw(self, ticket: int) -> None:
        """A queued request that will not be served after all gives up its place."""

        with self.turnstile:
            self.waiting.remove(ticket)
            self.turnstile.notify_all()

    def release_device(self) -> None:
        with self.turnstile:
            self.busy = False
            self.turnstile.notify_all()

    def service_actions(self) -> None:
        # Called once per serve_forever iteration: a device failure inside a
        # request ends the process (a poisoned model owner cannot serve).
        if self.fatal is not None:
            raise self.fatal


class _ClientWire:
    """One request's writes to its client, one at a time (the device loop's chunks and the heartbeat's keepalives).
    The first failed write is kept: a hang-up (BrokenPipeError) or a reader that stopped draining (TimeoutError);
    later writes raise it again at once instead of waiting on the socket."""

    def __init__(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        self.handler = handler
        self.lock = threading.Lock()
        self.streaming = False  # the SSE head went out: errors ride the stream, keepalives are due
        self.error: OSError | None = None

    def write(self, payload: bytes) -> None:
        with self.lock:
            if self.error is not None:
                raise self.error
            try:
                self.handler.wfile.write(payload)
                self.handler.wfile.flush()
            except OSError as error:
                self.error = error
                raise

    def event(self, document: Mapping[str, Any]) -> None:
        self.write(b"data: " + json.dumps(document).encode("utf-8") + b"\n\n")


class Qwen38ChatHandler(http.server.BaseHTTPRequestHandler):
    server_version = "qwen38-chat/2"
    protocol_version = "HTTP/1.0"
    server: Qwen38ChatHTTPServer

    def setup(self) -> None:
        # Every read and write on the connection is bounded (StreamRequestHandler applies ``timeout`` to the socket):
        # a reader that stopped draining raises TimeoutError, an OSError, and the request ends as "disconnected".
        self.timeout = self.server.socket_timeout_seconds
        self.error_body: dict[str, Any] | None = None  # the error document this connection answered, for the http event
        super().setup()

    def log_message(self, format: str, *args: Any) -> None:
        # The stdlib's request line and status; a refusal's body (its message, code and param) beside them, so the
        # log says WHY a 400 or 503 was answered, not only that it was (an operator reading a client's failure
        # otherwise has the status alone: the client keeps the body).
        fields = {} if self.error_body is None else {"error": self.error_body}
        _log("http", client=self.address_string(), line=format % args, **fields)

    def _peer_closed(self) -> bool:
        """The client hung up: its socket is readable with nothing left to read (FIN) or reset.  A client that sent
        its request and waits for the reply sends nothing more, so anything readable is the close."""

        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _send_json(self, status: int, document: Mapping[str, Any], **headers: str) -> None:
        body = json.dumps(document).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # The stdlib's own refusals (a method without a handler, a malformed request line) in the JSON error shape,
        # naming the methods served, instead of its HTML page.
        self.close_connection = True
        self._send_error_json(
            code,
            message or self.responses.get(code, ("", ""))[0],
            "invalid_request_error" if code < 500 or code == 501 else "server_error",  # 501: a method not served
            Allow=ALLOWED_METHODS,
        )

    def do_OPTIONS(self) -> None:
        # The methods served; no CORS headers (a browser page on another origin needs a deployment decision).
        self.send_response(204)
        self.send_header("Allow", ALLOWED_METHODS)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()  # the same status and headers; _send_json writes no body for HEAD

    def _send_error_json(
        self, status: int, message: str, kind: str, *, code: str | None = None, param: str | None = None, **headers: str
    ) -> None:
        self.error_body = {"message": message, "type": kind, "param": param, "code": code or kind}
        self._send_json(status, {"error": self.error_body}, **headers)

    def _model_document(self, session: Qwen38ChatSession) -> dict[str, Any]:
        """The model object of ``/v1/models`` (its one entry) and of ``/v1/models/<id>``: OpenAI clients read the
        context length from the object by id, so both routes serve the same document."""

        return {
            "id": MODEL_ID,
            "object": "model",
            "created": self.server.created,
            "owned_by": "tenstorrent",
            "context_length": session.context_limit,
            "max_model_len": session.context_limit,
            "limits": {
                **MAX_TOKENS_RULE,
                "thinking_token_caps": protocol.THINKING_TOKEN_CAPS,
                "answer_reserve_tokens": protocol.ANSWER_RESERVE_TOKENS,
            },
        }

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        session = self.server.session
        if path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [self._model_document(session)]})
        elif path.startswith("/v1/models/"):
            # The model by id (the id's slash literal or percent-encoded); another id is 404 in the OpenAI shape.
            wanted = unquote(path[len("/v1/models/") :])
            if wanted == MODEL_ID:
                self._send_json(200, self._model_document(session))
            else:
                self._send_error_json(
                    404,
                    f"no such model: {wanted!r} (this server serves {MODEL_ID})",
                    "invalid_request_error",
                    code="model_not_found",
                    param="model",
                )
        elif path == "/health":
            self._send_json(
                200,
                {
                    "status": "stopping" if self.server.stopping else "ready",
                    "model": MODEL_ID,
                    "busy": self.server.device_busy,
                    "queue_depth": self.server.queue_depth,
                    "lanes": None if self.server.lanes is None else self.server.lanes.status(),
                    "vision": None if self.server.vision is None else _vision_health(self.server.vision),
                    "current_request": self.server.current_request(),
                    "committed_tokens": len(session.committed),
                    "requests_served": session.requests_served,
                    "last_tokens_per_second": session.last_tokens_per_second,
                    "last_prefill_ms_per_token": self.server.last_prefill_ms_per_token,
                    "context_limit": session.context_limit,
                    "prefill_mode": session.prefill_mode,
                    "chunk_trace_available": session.chunk_trace_available,
                    "limits": {
                        "context_length": session.context_limit,
                        **MAX_TOKENS_RULE,
                        "max_tokens_bound": MAX_TOKENS_BOUND,
                        "stop_strings": protocol.MAX_STOP_STRINGS,
                        "queue_depth": self.server.queue_limit,
                        "request_deadline_seconds": self.server.request_deadline_seconds,
                        "socket_timeout_seconds": self.server.socket_timeout_seconds,
                        "stall_seconds": self.server.stall_seconds,
                        "request_bytes": MAX_REQUEST_BYTES,
                    },
                    "defaults": {
                        "enable_thinking": ENABLE_THINKING_DEFAULT,
                        "reasoning_effort": REASONING_EFFORT_DEFAULT,
                        "system_prompt": None,  # the client's messages render as sent; none is added
                        "thinking_token_caps": protocol.THINKING_TOKEN_CAPS,
                        "answer_reserve_tokens": protocol.ANSWER_RESERVE_TOKENS,
                    },
                    "supports": ["tools", "streaming", "reasoning_content", "stop", "thinking_budget", "ignore_eos"]
                    + (["sampling", "seed", "logprobs"] if session.sampling is not None else []),
                    "sampling": (
                        "greedy"
                        if session.sampling is None
                        else (
                            "candidate_row_device_sampler"
                            if getattr(session.sampling, "sampler", None) is not None
                            else "candidate_row_host_sampler"
                        )
                    ),
                    # logprobs are relative to the read candidates (above the vocabulary's by -log of the row's mass).
                    "logprobs_normalizer": None if session.sampling is None else "candidate_row",
                    "mtp": None if session.mtp is None else session.mtp.summary(),
                    # the chains captured at open by draft count (one entry without QWEN38_MTP_DRAFTS_PER_REQUEST)
                    "mtp_chains": {
                        str(drafts): chain_mtp.summary() for drafts, chain_mtp in sorted(session.mtp_chains.items())
                    },
                    "sampling_defaults": {
                        "thinking": sampling_step.parameters_as_dict(
                            sampling_step.Qwen38SamplingParameters.official_thinking(seed=0)
                        )
                        | {"seed": "random"},
                        "non_thinking": sampling_step.parameters_as_dict(
                            sampling_step.Qwen38SamplingParameters.official_non_thinking(seed=0)
                        )
                        | {"seed": "random"},
                        "top_k_limit": sampling_step.CANDIDATE_TOP_K_LIMIT,
                        "top_logprobs_limit": sampling_step.MAX_TOP_LOGPROBS,
                    },
                    "system_fingerprint": self.server.system_fingerprint,
                    "host_vmrss_kib": _vmrss_kib(),
                    "program_cache_entries": getattr(session.chain, "program_cache_entries", None),
                    "dram_after_captures": self.server.dram_after_captures,
                },
            )
        else:
            self._send_error_json(404, f"no such path: {path}", "not_found")

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path != "/v1/chat/completions":
            self._send_error_json(404, f"no such path: {path}", "not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:  # absent (a chunked body is not read), or not a number int accepts
            length = -1
        if not 0 <= length <= MAX_REQUEST_BYTES:
            self._send_error_json(
                400,
                f"Content-Length must be a number up to {MAX_REQUEST_BYTES}, got {self.headers.get('Content-Length')!r}",
                "invalid_request_error",
            )
            return
        session = self.server.session
        try:
            request = parse_chat_request(
                json.loads(self.rfile.read(int(length)).decode("utf-8")),
                sampling_available=session.sampling is not None,
                mtp_drafts_admitted=self.server.mtp_drafts_admitted,
            )
            # The reference render is the device prompt (usage.prompt_tokens is the client's own count); it validates
            # the whole request and resolves the budget (the remaining context when max_tokens is absent) before any
            # queueing.  The thinking budget is carved out of that.
            if request.get("images"):
                # An image request needs the resident tower (refused with the reason otherwise: text keeps serving)
                # and renders with the images' grids: one <|image_pad|> per merged token, the 3-axis rotary positions.
                refusal = self.server.vision_refusal(request["images"])
                if refusal is not None:
                    raise Qwen38ChatRequestRejected(refusal, param="messages", code="vision_unavailable")
                prompt_ids = protocol.render_prompt(
                    session.template.tokenizer,
                    request["raw_messages"],
                    request["tools"],
                    enable_thinking=request["enable_thinking"],
                    reasoning_effort=request["reasoning_effort"],
                    images=[],
                    image_grids=[image.grid for image in request["images"]],
                )
                request["vision_positions"] = mrope.mrope_positions(
                    prompt_ids, [image.grid for image in request["images"]]
                )
                # What the device section requires of an image prompt, checked here so a shape it cannot take is a
                # 400 and never a device-thread failure: the chunked prefill (image pads never take 1-row steps), a
                # prompt long enough for the chunk trace from position 0, a text tail after the last image.
                if session.prefill_mode != "chunked":
                    raise Qwen38ChatRequestRejected(
                        "image prompts need the chunked prefill mode (their pads never take 1-row steps)",
                        param="messages",
                        code="vision_unavailable",
                    )
                if len(prompt_ids) - 1 < CHUNK_PREFILL_MIN_ROWS:
                    raise Qwen38ChatRequestRejected("image prompt too short for the chunked prefill", param="messages")
                if not request["vision_positions"].tail_is_plain(len(prompt_ids)):
                    raise Qwen38ChatRequestRejected(
                        "an image ends within the last index block of the prompt: the prompt must end with text "
                        "after its last image (the chat template's assistant header does)",
                        param="messages",
                    )
            else:
                prompt_ids = session.render(
                    request["messages"],
                    tools=request["tools"],
                    enable_thinking=request["enable_thinking"],
                    reasoning_effort=request["reasoning_effort"],
                )
            max_tokens = session.require_budget(len(prompt_ids), request["max_tokens"])
        except (ValueError, UnicodeDecodeError) as error:
            code = getattr(error, "code", None)
            if code is None:
                code = "context_length_exceeded" if str(error).startswith("context_length_exceeded") else "bad_request"
            self._send_error_json(
                400, str(error), "invalid_request_error", code=code, param=getattr(error, "param", None)
            )
            return
        except Exception as error:  # noqa: BLE001  a validation defect: answered, not a dropped connection
            _log(
                "request_validation_failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
            )
            self._send_error_json(500, f"{type(error).__name__}: {error}", "server_error")
            return
        if request["ignored"]:
            _log("ignored_request_fields", fields=request["ignored"])
        request = {
            **request,
            "max_tokens_requested": request["max_tokens"],
            "max_tokens": max_tokens,
            "think_budget": (
                protocol.thinking_budget(request["reasoning_effort"], max_tokens, request["thinking_budget"])
                if request["enable_thinking"]
                else None
            ),
        }
        received_utc = utc_now()
        request_id = _completion_id()
        created = int(time.time())
        wire = _ClientWire(self)
        if self.server.lanes is not None:
            self._serve_lanes(session, request, prompt_ids, request_id, received_utc, created, wire)
            return
        try:
            ticket = self.server.enqueue()
        except Qwen38ServerBusy as error:
            self._send_error_json(503, str(error), "server_busy", Retry_After=str(RETRY_AFTER_SECONDS))
            return
        assembler = protocol.Qwen38ReplyAssembler(
            template_decoder(session.template),
            thinking_open=request["enable_thinking"],
            stop_strings=request["stop"],
            tools=request["tools"],
        )
        heartbeat_stop = threading.Event()
        heartbeats = [0]
        started = time.perf_counter()

        def chunk(choice: dict[str, Any], **extra: Any) -> dict[str, Any]:
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "system_fingerprint": self.server.system_fingerprint,
                "choices": [{"index": 0, **choice}],
                **extra,
            }

        # Every heartbeat_seconds: a progress log line, and on an open stream an SSE comment so a proxy or client
        # idle timeout does not cut a long queue wait or prefill (nothing else is written before the first token).
        # With --stall-seconds, a device turn whose last poll is older than that ends the server (fatal): a wedged
        # device call never returns to the poll, and a long prefill polls every few chunks.
        def heartbeat() -> None:
            while not heartbeat_stop.wait(self.server.heartbeat_seconds):
                heartbeats[0] += 1
                progress = self.server.current_request()
                silent = (
                    None if progress is None or progress["id"] != request_id else progress["seconds_since_progress"]
                )
                _log(
                    "heartbeat",
                    request_id=request_id,
                    elapsed_seconds=round(time.perf_counter() - started, 3),
                    committed_tokens=len(session.committed),
                    emitted_tokens=assembler.tokens,
                    reasoning_tokens=assembler.reasoning_tokens,
                    tool_calls=len(assembler.calls),
                    seconds_since_progress=silent,
                    host_vmrss_kib=_vmrss_kib(),
                )
                stall = self.server.stall_seconds
                if stall is not None and silent is not None and silent > stall and self.server.fatal is None:
                    self.server.fatal = Qwen38ChatChainError(
                        f"request {request_id} made no progress for {silent:.1f} s (--stall-seconds {stall})"
                    )
                    _log("stalled", request_id=request_id, seconds_since_progress=silent, stall_seconds=stall)
                if wire.streaming and wire.error is None:
                    try:
                        wire.write(b": keepalive\n\n")
                    except (OSError, ValueError):  # the wire remembers; should_stop sees the peer closed
                        pass

        try:
            if request["stream"]:
                # The stream opens before the queue wait: the client sees the head and the role chunk at once, then
                # keepalives while it waits and while the prompt prefills.
                try:
                    self._send_stream_head()
                    wire.streaming = True
                    wire.event(chunk({"delta": {"role": "assistant", "content": ""}, "finish_reason": None}))
                except OSError as error:
                    self.server.withdraw(ticket)
                    _log(
                        "client_disconnected",
                        request_id=request_id,
                        phase="queued",
                        error=f"{type(error).__name__}: {error}",
                    )
                    return
            threading.Thread(target=heartbeat, name=f"heartbeat-{request_id}", daemon=True).start()
            try:
                queue_wait = self.server.await_turn(ticket, self._peer_closed)
            except Qwen38ClientGone as error:
                _log("client_disconnected", request_id=request_id, phase="queued", error=str(error))
                return
            except Qwen38ServerBusy as error:  # the stop came while this request waited
                self._answer_error(wire, 503, str(error), "server_busy", Retry_After=str(RETRY_AFTER_SECONDS))
                return
            self._serve(
                session, request, prompt_ids, request_id, received_utc, wire, chunk, assembler, heartbeats, queue_wait
            )
        finally:
            heartbeat_stop.set()

    def _serve(
        self,
        session: Qwen38ChatSession,
        request: dict[str, Any],
        prompt_ids: list[int],
        request_id: str,
        received_utc: str,
        wire: _ClientWire,
        chunk: Callable[..., dict[str, Any]],
        assembler: protocol.Qwen38ReplyAssembler,
        heartbeats: list[int],
        queue_wait: float,
    ) -> None:
        """The device turn: the run over the reference render (the session continues its committed prefix where
        the render extends it and resets otherwise), the reply, the ledger.  The device is released before anything
        is written to the client."""

        started = time.perf_counter()
        deadline = self.server.request_deadline_seconds
        progress = {
            "id": request_id,
            "started_utc": utc_now(),
            "started": started,
            "last_progress": started,
            "polls": 0,
        }
        self.server.progress = progress

        # Polled between steps and at the prefill's event syncs (so each poll is a device step completed: the
        # progress record, /health current_request and the stall watchdog read it): the stop string, the stop
        # signal, the deadline, and the client's socket (a hang-up ends the request as "disconnected" whether or not
        # anything was being streamed to it).
        def should_stop() -> str | None:
            progress["last_progress"] = time.perf_counter()
            progress["polls"] += 1
            if assembler.stop_hit:
                return "stop"
            if self.server.stopping:
                return "shutdown"
            if deadline is not None and time.perf_counter() - started > deadline:
                return "deadline"
            if wire.error is not None or self._peer_closed():
                return "disconnected"
            return None

        sampling = (
            None
            if request["sampling"] is None
            else sampling_step.Qwen38SamplingRequest(
                request["sampling"], top_logprobs=request["top_logprobs"], logprobs=bool(request["logprobs"])
            )
        )
        extension_of = lambda completion: _extension(  # noqa: E731
            completion, assembler, queue_wait=queue_wait, think_budget=request["think_budget"], sampling=sampling
        )
        # One logprobs item per token the client sees (the sampled loop appends its sample before yielding).
        decode_one = lambda token_id: template_decoder(session.template)([token_id])  # noqa: E731
        logprobs_of = lambda token_id: sampling_step.logprobs_content_item(  # noqa: E731
            sampling.samples[-1], token_id, decode_one
        )
        logprob_items: list[dict[str, Any]] = []

        def on_token(token_id: int) -> None:
            if request["logprobs"]:
                logprob_items.append(logprobs_of(token_id))
            assembler.push(token_id)

        # Streaming: one chunk per assembled piece (reasoning_content, content, or a completed tool call), flushed
        # per token; with logprobs every token's item rides on its first chunk (an empty delta when the assembler
        # held the text back).  A write that fails (the client went away, or stopped reading: TimeoutError) raises
        # OSError out of on_token and the session ends the request as "disconnected".
        def on_token_streaming(token_id: int) -> None:
            deltas = assembler.push(token_id)
            if request["logprobs"]:
                deltas = deltas or [{}]
                wire.event(
                    chunk({"delta": deltas[0], "logprobs": {"content": [logprobs_of(token_id)]}, "finish_reason": None})
                )
                deltas = deltas[1:]
            for delta in deltas:
                wire.event(chunk({"delta": delta, "finish_reason": None}))

        failure: BaseException | None = None
        vision_record: dict[str, Any] | None = None
        try:
            vision_inputs_kw: dict[str, Any] = {}
            if request.get("images"):
                # The tower on this (device) thread: one forward per image in its row bucket, the feature rows for
                # the splice; then the prompt through the chain with its 3-axis positions.
                vision_prompt, vision_record = self.server.vision_prompt_for(
                    prompt_ids, request["vision_positions"], request["images"]
                )
                vision_inputs_kw = {"vision": vision_prompt}
            completion = session.complete(
                prompt_ids,
                request["max_tokens"],
                on_token=on_token_streaming if request["stream"] else on_token,
                stop_ids=() if request["ignore_eos"] else EOS_TOKEN_IDS,
                think_budget=request["think_budget"],
                should_stop=should_stop,
                prefill_mode=request["prefill_mode"],
                sampling=sampling,
                mtp_drafts=request["mtp_drafts"],
                **vision_inputs_kw,
            )
            final_deltas = assembler.finish()
        except Exception as error:  # noqa: BLE001  the device loop failed: report, then end the server
            failure = error
            _log(
                "request_failed",
                request_id=request_id,
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
            if session.poisoned:
                self.server.fatal = error
        finally:
            self.server.progress = None
            self.server.release_device()
        if failure is not None:
            self._answer_error(wire, 500, f"{type(failure).__name__}: {failure}", "server_error")
            return
        finish_reason = _finish_reason(completion.finish_reason, assembler)
        if completion.prefill_tokens:
            self.server.last_prefill_ms_per_token = round(
                1e3 * completion.prefill_seconds / completion.prefill_tokens, 3
            )
        extension = extension_of(completion)
        if vision_record is not None:  # an image request: the images, their tokens and the tower's time
            extension["vision"] = vision_record
        # The evidence (the ledger line, the log line) cannot cost the reply: a full disk or a lost evidence directory
        # loses the record, never the answer the device already produced.
        try:
            append_phase_record(
                self.server.ledger,
                {
                    "phase": "chat-request",
                    "request_id": request_id,
                    "received_utc": received_utc,
                    "stream": request["stream"],
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": len(completion.token_ids),
                    "max_tokens": request["max_tokens"],
                    "max_tokens_requested": request["max_tokens_requested"],
                    "finish_reason": finish_reason,
                    "text_characters": sum(len(piece) for piece in assembler.content),
                    "reasoning_characters": sum(len(piece) for piece in assembler.reasoning),
                    "enable_thinking": request["enable_thinking"],
                    "reasoning_effort": request["reasoning_effort"],
                    "think_budget": request["think_budget"],
                    "tools_offered": len(request["tools"]),
                    "ignore_eos": request["ignore_eos"],
                    "logprobs": request["logprobs"],
                    "deadline_seconds": deadline,
                    "heartbeats": heartbeats[0],
                    "host_vmrss_kib": _vmrss_kib(),
                    "program_cache_entries": getattr(session.chain, "program_cache_entries", None),
                    **extension,
                    "position_after": completion.position,
                },
            )
            _log(
                "request",
                request_id=request_id,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=len(completion.token_ids),
                finish_reason=finish_reason,
                **extension,
            )
        except OSError as error:
            try:
                _log("evidence_write_failed", request_id=request_id, error=f"{type(error).__name__}: {error}")
            except OSError:
                pass
        usage = _usage(completion.prompt_tokens, len(completion.token_ids), queue_wait)
        if completion.finish_reason == "disconnected":
            _log(
                "client_disconnected",
                request_id=request_id,
                phase="streaming" if request["stream"] else "generating",
                completion_tokens=len(completion.token_ids),
                error=None if wire.error is None else f"{type(wire.error).__name__}: {wire.error}",
            )
        else:
            try:
                if request["stream"]:
                    for delta in final_deltas:
                        wire.event(chunk({"delta": delta, "finish_reason": None}))
                    wire.event(chunk({"delta": {}, "finish_reason": finish_reason}, usage=usage, qwen38=extension))
                    wire.write(b"data: [DONE]\n\n")
                else:
                    choice: dict[str, Any] = {
                        "index": 0,
                        "message": assembler.message(),
                        "finish_reason": finish_reason,
                    }
                    if request["logprobs"]:
                        choice["logprobs"] = {"content": logprob_items}
                    self._send_json(
                        200,
                        {
                            "id": request_id,
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": MODEL_ID,
                            "system_fingerprint": self.server.system_fingerprint,
                            "choices": [choice],
                            "usage": usage,
                            "qwen38": extension,
                        },
                    )
            except OSError as error:
                _log(
                    "client_disconnected",
                    request_id=request_id,
                    phase="reply",
                    error=f"{type(error).__name__}: {error}",
                )

    def _serve_lanes(
        self,
        session: Qwen38ChatSession,
        request: dict[str, Any],
        prompt_ids: list[int],
        request_id: str,
        received_utc: str,
        created: int,
        wire: _ClientWire,
    ) -> None:
        """The request on the lanes server: a ticket in the scheduler's FIFO (503 with Retry-After beyond its limit),
        the stream head and keepalives while it waits and while its prompt prefills, then its tokens from the ticket's
        queue as the driver thread delivers them pass by pass; a stop string, the deadline or a hang-up end it at the
        next pass boundary (the ticket's cancel).  No device call on this thread."""

        scheduler = self.server.lanes
        ticket = Qwen38LaneTicket(
            request_id=request_id,
            prompt_ids=list(prompt_ids),
            max_tokens=request["max_tokens"],
            stop_ids=() if request["ignore_eos"] else tuple(EOS_TOKEN_IDS),
            think_budget=request["think_budget"],
            # An image prompt's decoded images and rotary positions: the driver thread runs the tower inside the
            # admission (the "tower" segment) and prefills with the feature rows; the lane takes the prompt's shift.
            images=list(request.get("images") or ()),
            vision_positions=request.get("vision_positions"),
        )
        try:
            scheduler.submit(ticket)
        except Qwen38LaneSchedulerBusy as error:
            self._send_error_json(503, str(error), "server_busy", Retry_After=str(RETRY_AFTER_SECONDS))
            return
        assembler = protocol.Qwen38ReplyAssembler(
            template_decoder(session.template),
            thinking_open=request["enable_thinking"],
            stop_strings=request["stop"],
            tools=request["tools"],
        )
        started = time.perf_counter()
        deadline = self.server.request_deadline_seconds
        heartbeat_stop = threading.Event()
        heartbeats = [0]

        def chunk(choice: dict[str, Any], **extra: Any) -> dict[str, Any]:
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "system_fingerprint": self.server.system_fingerprint,
                "choices": [{"index": 0, **choice}],
                **extra,
            }

        def heartbeat() -> None:
            while not heartbeat_stop.wait(self.server.heartbeat_seconds):
                heartbeats[0] += 1
                status = scheduler.status()
                silent = status["seconds_since_progress"] if (ticket.lane is not None or status["active"]) else None
                _log(
                    "heartbeat",
                    request_id=request_id,
                    elapsed_seconds=round(time.perf_counter() - started, 3),
                    lane=ticket.lane,
                    committed_tokens=ticket.committed,
                    emitted_tokens=assembler.tokens,
                    reasoning_tokens=assembler.reasoning_tokens,
                    tool_calls=len(assembler.calls),
                    seconds_since_progress=silent,
                    lanes_active=status["active"],
                    lanes_waiting=status["waiting"],
                    host_vmrss_kib=_vmrss_kib(),
                )
                stall = self.server.stall_seconds
                if stall is not None and silent is not None and silent > stall and self.server.fatal is None:
                    self.server.fatal = Qwen38ChatChainError(
                        f"the lanes made no progress for {silent:.1f} s with request {request_id} in them (--stall-seconds {stall})"
                    )
                    _log("stalled", request_id=request_id, seconds_since_progress=silent, stall_seconds=stall)
                if wire.streaming and wire.error is None:
                    try:
                        wire.write(b": keepalive\n\n")
                    except (OSError, ValueError):
                        pass

        def poll() -> None:
            # Between deliveries: the client's socket, the deadline, the stop signal (the driver ends every lane with
            # ``shutdown`` itself); a cancel reaches the driver at the next pass boundary.
            if ticket.cancelled is not None:
                return
            if wire.error is not None or self._peer_closed():
                ticket.cancel("disconnected")
            elif deadline is not None and time.perf_counter() - started > deadline:
                ticket.cancel("deadline")

        first_token_at: float | None = None
        last_token_at: float | None = None
        consumed = 0  # tokens the reply took (up to the one that completed a stop string)
        try:
            if request["stream"]:
                try:
                    self._send_stream_head()
                    wire.streaming = True
                    wire.event(chunk({"delta": {"role": "assistant", "content": ""}, "finish_reason": None}))
                except OSError as error:
                    if scheduler.withdraw(ticket):
                        _log(
                            "client_disconnected",
                            request_id=request_id,
                            phase="queued",
                            error=f"{type(error).__name__}: {error}",
                        )
                        return
                    ticket.cancel("disconnected")
            threading.Thread(target=heartbeat, name=f"heartbeat-{request_id}", daemon=True).start()
            for token_id, _finish in ticket.items(poll_seconds=QUEUE_POLL_SECONDS, on_idle=poll):
                last_token_at = time.perf_counter()
                if first_token_at is None:
                    first_token_at = last_token_at
                if ticket.cancelled is not None:
                    continue
                consumed += 1  # the EOS counts as the single stream counts it (a consumed token, no text)
                if token_id in ticket.stop_ids:
                    continue
                try:
                    if request["stream"]:
                        for delta in assembler.push(token_id):
                            wire.event(chunk({"delta": delta, "finish_reason": None}))
                    else:
                        assembler.push(token_id)
                except OSError:
                    ticket.cancel("disconnected")
                    continue
                if assembler.stop_hit:
                    ticket.cancel("stop")
                poll()
        finally:
            heartbeat_stop.set()
        finish = ticket.finish or "error"
        if finish == "error" and scheduler.fatal is not None:
            self.server.fatal = scheduler.fatal
        if finish == "error":
            self._answer_error(wire, 500, f"the lanes ended request {request_id} with an error", "server_error")
            return
        final_deltas = assembler.finish()
        finish_reason = _finish_reason(finish, assembler)
        decode_seconds = 0.0 if first_token_at is None or last_token_at is None else last_token_at - first_token_at
        tokens_per_second = (consumed - 1) / decode_seconds if consumed >= 2 and decode_seconds > 0 else None
        admitted = ticket.admitted_at if ticket.admitted_at is not None else started
        extension = {
            "finish": finish,
            "queue_wait_seconds": round(ticket.queue_wait, 4),
            "decode_loop": "greedy",
            "served_reasoning_tokens": 0,
            "sampling": None,
            "seed": None,
            "reasoning_tokens": assembler.reasoning_tokens,
            "thinking_forced": ticket.stream is not None and ticket.stream.forced_think_ends > 0,
            "stop_string_hit": assembler.stop_hit,
            "tool_calls": len(assembler.calls),
            "tool_parse_errors": assembler.parse_errors,
            "truncated_tool_call": assembler.truncated_tool_call,
            "prefix_reused": 0,
            "reset": True,
            # The snapshot vocabulary of the single stream's ledger: a lane admission is a fresh prefill of the prompt
            # from position 0 (the chunk driver, or the teacher-forced form of a short prompt) and leaves no snapshot.
            "snapshot_schedule": "chunked",
            "snapshot_captured": False,
            "prefill_tokens": len(prompt_ids),
            "prefill_seconds": round(ticket.admission_segments.get("prefill", 0.0), 4),
            "prefill_mode": ticket.prefill.get("mode"),
            "prefill_forced_tokens": ticket.prefill.get("forced_tokens"),
            "prefill_chunks": ticket.prefill.get("chunks"),
            "prefill_long_chunks": ticket.prefill.get("long_chunks"),
            "prefill_slabs": ticket.prefill.get("slabs"),
            "ttft_seconds": None if first_token_at is None else round(first_token_at - ticket.submitted, 4),
            "decode_seconds": round(decode_seconds, 4),
            "tokens_per_second": None if tokens_per_second is None else round(tokens_per_second, 3),
            "position": ticket.position,
            "mtp": {
                "k": scheduler.drafts,
                "passes": ticket.passes,
                "tokens_per_pass": None if not ticket.passes else round(ticket.committed / ticket.passes, 3),
            },
            **({} if ticket.vision is None else {"vision": ticket.vision}),
            "lanes": {
                "lane": ticket.lane,
                "admission_seconds": round(ticket.admission_seconds, 4),
                "admission_wall_seconds": round(ticket.admission_wall_seconds, 4),
                "admission": {name: round(value, 4) for name, value in ticket.admission_segments.items()},
                "interleaved_passes": ticket.interleaved_passes,
                "stalled_seconds": round(ticket.stalled_seconds, 4),
                "passes": ticket.passes,
                "committed_tokens": ticket.committed,
                "finish_detail": ticket.finish_detail,
                "forced_think_ends": 0 if ticket.stream is None else ticket.stream.forced_think_ends,
                "forced_think_end_inside_pass": 0 if ticket.stream is None else ticket.stream.forced_inside,
                "forced_think_end_on_last_token": 0 if ticket.stream is None else ticket.stream.forced_last,
                "admitted_after_seconds": round(admitted - ticket.submitted, 4),
            },
        }
        usage = _usage(len(prompt_ids), consumed, ticket.queue_wait)
        try:
            append_phase_record(
                self.server.ledger,
                {
                    "phase": "chat-request",
                    "request_id": request_id,
                    "received_utc": received_utc,
                    "stream": request["stream"],
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": consumed,
                    "max_tokens": request["max_tokens"],
                    "max_tokens_requested": request["max_tokens_requested"],
                    "finish_reason": finish_reason,
                    "text_characters": sum(len(piece) for piece in assembler.content),
                    "reasoning_characters": sum(len(piece) for piece in assembler.reasoning),
                    "enable_thinking": request["enable_thinking"],
                    "reasoning_effort": request["reasoning_effort"],
                    "think_budget": request["think_budget"],
                    "tools_offered": len(request["tools"]),
                    "ignore_eos": request["ignore_eos"],
                    "logprobs": False,
                    "deadline_seconds": deadline,
                    "heartbeats": heartbeats[0],
                    "host_vmrss_kib": _vmrss_kib(),
                    "program_cache_entries": getattr(session.chain, "program_cache_entries", None),
                    **extension,
                    "position_after": ticket.position,
                },
            )
            _log(
                "request",
                request_id=request_id,
                prompt_tokens=len(prompt_ids),
                completion_tokens=consumed,
                finish_reason=finish_reason,
                lane=ticket.lane,
                queue_wait_seconds=extension["queue_wait_seconds"],
                tokens_per_second=extension["tokens_per_second"],
                stalled_seconds=extension["lanes"]["stalled_seconds"],
            )
        except OSError as error:
            try:
                _log("evidence_write_failed", request_id=request_id, error=f"{type(error).__name__}: {error}")
            except OSError:
                pass
        if finish == "disconnected" or wire.error is not None:
            _log(
                "client_disconnected",
                request_id=request_id,
                phase="streaming" if request["stream"] else "generating",
                completion_tokens=consumed,
            )
            return
        try:
            if request["stream"]:
                for delta in final_deltas:
                    wire.event(chunk({"delta": delta, "finish_reason": None}))
                wire.event(chunk({"delta": {}, "finish_reason": finish_reason}, usage=usage, qwen38=extension))
                wire.write(b"data: [DONE]\n\n")
            else:
                self._send_json(
                    200,
                    {
                        "id": request_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "system_fingerprint": self.server.system_fingerprint,
                        "choices": [{"index": 0, "message": assembler.message(), "finish_reason": finish_reason}],
                        "usage": usage,
                        "qwen38": extension,
                    },
                )
        except OSError as error:
            _log("client_disconnected", request_id=request_id, phase="reply", error=f"{type(error).__name__}: {error}")

    def _answer_error(
        self, wire: _ClientWire, status: int, message: str, kind: str, *, code: str | None = None, **headers: str
    ) -> None:
        """An error after the request was admitted: HTTP ``status`` before the stream opened, an SSE error event
        (then ``[DONE]``) once it has."""

        try:
            if wire.streaming:
                wire.event({"error": {"message": message, "type": kind, "param": None, "code": code or kind}})
                wire.write(b"data: [DONE]\n\n")
            else:
                self._send_error_json(status, message, kind, code=code, **headers)
        except OSError:
            pass

    def _send_stream_head(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()


# -- acceptance replay ---------------------------------------------------------------------------


def load_acceptance_records(directory: Path) -> list[dict[str, Any]]:
    """The CPU study's ``prompt-*-greedy.json`` records, digest-checked against the directory's SHA256SUMS."""

    directory = directory.resolve(strict=True)
    sums = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        sums[name.strip()] = digest
    records = []
    for path in sorted(directory.glob("prompt-*-greedy.json")):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if sums.get(path.name) != actual:
            raise ValueError(f"{path.name}: sha256 {actual} vs SHA256SUMS {sums.get(path.name)}")
        document = json.loads(path.read_text(encoding="utf-8"))
        prompt_ids = document.get("prompt_token_ids")
        generated = document.get("generated_token_ids")
        if (
            document.get("mode") != "greedy"
            or not isinstance(document.get("prompt"), str)
            or not isinstance(prompt_ids, list)
            or not isinstance(generated, list)
            or not prompt_ids
            or not generated
            or any(type(value) is not int or not 0 <= value < VOCAB_SIZE for value in prompt_ids + generated)
            or document.get("prompt_length") != len(prompt_ids)
            or document.get("stop_reason") not in ("eos", "max_continuation")
        ):
            raise ValueError(f"{path.name} is not a greedy prompt record with vocabulary-ranged token lists")
        records.append(
            {
                "prompt": document["prompt"],
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated,
                "stop_reason": document["stop_reason"],
                "sha256": actual,
            }
        )
    if not records:
        raise ValueError(f"no prompt-*-greedy.json records in {directory}")
    # The gate prompt first: its 96/96 result is the go/no-go for serving.
    records.sort(key=lambda record: (record["prompt"] != ACCEPTANCE_GATE_PROMPT, record["prompt"]))
    return records


# A record of ACCEPTANCE_SLAB_RECORD_TOKENS or more prompt tokens exists to run a prefill slab (the twelve records under
# 2048 tokens never reach one); a server without --prefill-slab skips it, logged by name, so the chunked forms' start-up
# stays the twelve records' replay.
ACCEPTANCE_SLAB_RECORD_TOKENS = 2048


def slab_records_admitted(records: list[dict[str, Any]], *, slab_rows: int | None) -> list[dict[str, Any]]:
    """The records this server replays: every record with a prefill slab, else those under the slab record length."""

    admitted = []
    for record in records:
        prompt_tokens = len(record["prompt_token_ids"])
        if slab_rows is None and prompt_tokens >= ACCEPTANCE_SLAB_RECORD_TOKENS:
            _log(
                "acceptance_record_skipped",
                prompt=record["prompt"],
                prompt_tokens=prompt_tokens,
                reason="a slab record (2048 or more prompt tokens) on a server without --prefill-slab",
            )
            continue
        admitted.append(record)
    return admitted


def _divergence(actual: Sequence[int], expected: Sequence[int]) -> int | None:
    divergence = next((index for index, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    if divergence is None and len(actual) != len(expected):
        divergence = min(len(actual), len(expected))
    return divergence


def replay_acceptance(
    session: Qwen38ChatSession,
    records: list[dict[str, Any]],
    *,
    continuation: int = ACCEPTANCE_CONTINUATION,
    require_gate: bool,
    handoff_gate: bool = False,
    handoff_split: int = ACCEPTANCE_HANDOFF_SPLIT,
) -> dict[str, Any]:
    """Prefill each record's prompt ids, generate greedily, report the first index where the device differs.

    Stops are disabled so an early EOS record is compared over its whole CPU
    stream; the gate is the ``json`` record's full ``continuation`` match.
    ``handoff_gate`` (MTP chains) replays the gate record twice more, split at
    ``handoff_split`` tokens: the pass loop then the 1-row loop, and the 1-row
    loop then the pass loop, each half continuing the other's device state, so
    both mode switches must reproduce the CPU stream to count as exact.
    """

    results = []
    for record in records:
        expected = record["generated_token_ids"][:continuation]
        session.reset()
        completion = session.complete(record["prompt_token_ids"], len(expected), stop_ids=())
        actual = completion.token_ids
        divergence = _divergence(actual, expected)
        result = {
            "prompt": record["prompt"],
            "prompt_tokens": len(record["prompt_token_ids"]),
            "compared_tokens": len(expected),
            "divergence_index": divergence,
            "matched_tokens": len(expected) if divergence is None else divergence,
            "first_token_device": actual[0] if actual else None,
            "first_token_cpu": expected[0],
            "device_token_ids": actual,
            "cpu_token_ids": expected,
            "prefill_seconds": completion.prefill_seconds,
            "prefill_mode": completion.prefill_mode,
            "prefill_chunks": completion.prefill_chunks,
            "prefill_long_chunks": completion.prefill_long_chunks,
            "prefill_slabs": completion.prefill_slabs,
            "prefill_forced_tokens": completion.prefill_forced_tokens,
            "prefill_handoff_ms": completion.prefill_handoff_ms,
            "tokens_per_second": completion.tokens_per_second,
            "mtp": completion.mtp,
        }
        results.append(result)
        _log("acceptance_prompt", **{key: value for key, value in result.items() if not key.endswith("_ids")})
    gate = next((result for result in results if result["prompt"] == ACCEPTANCE_GATE_PROMPT), None)
    gate_pass = gate is not None and gate["divergence_index"] is None and gate["compared_tokens"] == continuation
    if require_gate and not gate_pass:
        raise Qwen38ChatChainError(
            f"acceptance gate: {ACCEPTANCE_GATE_PROMPT} matched "
            f"{None if gate is None else gate['matched_tokens']} of {continuation} CPU greedy tokens"
        )
    handoff = None
    if handoff_gate and gate is not None:
        record = next(record for record in records if record["prompt"] == ACCEPTANCE_GATE_PROMPT)
        expected = record["generated_token_ids"][:continuation]
        handoff = {"split": handoff_split, "orders": {}}
        for name, first_speculative in (("mtp_then_one_row", True), ("one_row_then_mtp", False)):
            session.reset()
            prompt = list(record["prompt_token_ids"])
            first = session.complete(prompt, handoff_split, stop_ids=(), speculative=first_speculative)
            # The second half extends the exact repeat: the first half's last token sits unconsumed in the row.
            second = session.complete(
                prompt + first.token_ids[:-1],
                len(expected) - handoff_split + 1,
                stop_ids=(),
                speculative=not first_speculative,
            )
            actual = first.token_ids[:-1] + second.token_ids
            handoff["orders"][name] = {
                "divergence_index": _divergence(actual, expected),
                "compared_tokens": len(expected),
                "second_half_prefix_reused": second.prefix_reused,
                "second_half_reset": second.reset,
                "first_half_mtp": first.mtp,
                "second_half_mtp": second.mtp,
                "device_token_ids": actual,
            }
            _log(
                "acceptance_handoff",
                order=name,
                **{k: v for k, v in handoff["orders"][name].items() if k != "device_token_ids"},
            )
        handoff["pass"] = all(order["divergence_index"] is None for order in handoff["orders"].values())
        if require_gate and not handoff["pass"]:
            raise Qwen38ChatChainError(
                f"acceptance hand-off gate: {ACCEPTANCE_GATE_PROMPT} split at {handoff_split} diverged: "
                f"{ {name: order['divergence_index'] for name, order in handoff['orders'].items()} }"
            )
    # The replays are not client requests: /health and result.json count clients from zero.
    session.requests_served = 0
    session.last_tokens_per_second = None
    return {
        "schema": "qwen38-chat-server-acceptance/v1",
        "continuation": continuation,
        "gate_prompt": ACCEPTANCE_GATE_PROMPT,
        "gate_pass": gate_pass,
        "prompts": results,
        "handoff": handoff,
    }


def record_agreement(
    session: Qwen38ChatSession,
    *,
    reference: Path,
    corpus_dir: Path,
    parts: Sequence[str],
    out_dir: Path,
    producer: dict[str, Any],
    full_logits_dir: Path | None = None,
) -> dict[str, Any]:
    """A1's device column: every corpus item teacher-forced through the session (a prompt followed by the HF
    reference's argmax chain), the resolved argmax and the candidate row (every shard's top-32) per scored position as
    agreement records (``agreement-records/<item>.json``), then the column (``agreement-device.json``) and its score against
    the HF reference (``agreement-score.json``), the files the corpus tool's ``device`` and ``score`` modes write.

    The chunked mode scores what its path produces rows for: a prompt item's continuation (the prompt goes through
    the chunk trace), a long item's windows, and for a text item without windows the 32 positions at its end.

    ``full_logits_dir`` also keeps every recorded position's full-vocabulary logits per item there, in the reference
    tool's ``--full-logits`` format (``<item>.pt``: ``positions`` and ``logits_fp16`` [n, V]): the LM head's bf16 row,
    gathered eagerly after each recorded TAIL; fp16 holds every bf16 logit exactly except values below 2^-17 in
    magnitude (rounded to 2^-24 steps; the record counts them).
    """

    manifest, items = reference_corpus.load_corpus(corpus_dir)
    _document, hf = reference_corpus.load_reference(reference, manifest=manifest)
    records_dir = out_dir / "agreement-records"
    records_dir.mkdir(exist_ok=True)
    if full_logits_dir is not None:
        import torch

        full_logits_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    recorded = []
    gather_seconds = 0.0
    for item in reference_corpus.select_items(items, parts, ()):
        if item.item_id not in hf:
            raise Qwen38ChatChainError(f"{item.item_id}: not in the HF reference {reference}")
        stream = list(item.token_ids)
        if item.continuation_tokens:
            teacher = dict(zip(hf[item.item_id].positions, hf[item.item_id].teacher_ids))
            stream += [teacher[p] for p in range(item.prompt_tokens - 1, item.positions)]
        scored = item.scored_positions()
        if session.prefill_mode == "chunked" and not item.windows:
            scored = (
                [p for p in scored if p >= item.prompt_tokens - 1]
                if item.continuation_tokens
                else scored[-reference_corpus.LONG_WINDOW :]
            )
        replay = session.teacher_force(
            stream[: item.positions], positions=scored, full_logits=full_logits_dir is not None
        )
        document = {
            "schema": reference_corpus.AGREEMENT_RECORDS_SCHEMA,
            "item_id": item.item_id,
            "part": item.part,
            "prefill_mode": replay.prefill_mode,
            "chunks": replay.chunks,
            "forced_tokens": replay.forced_tokens,
            "seconds": round(replay.seconds, 3),
            "positions": [
                {
                    "position": position,
                    "teacher_id": stream[position + 1],
                    "argmax": argmax,
                    "candidate_ids": row.ids.reshape(-1).tolist(),
                    "candidate_logits": row.values.reshape(-1).tolist(),
                }
                for position, (argmax, row) in sorted(replay.rows.items())
            ],
        }
        if full_logits_dir is not None:
            positions = sorted(replay.full_logits)
            rows = torch.stack([replay.full_logits[position] for position in positions])
            kept = rows.to(torch.float16)
            # fp16 holds every bf16 logit of magnitude 2^-17 and above exactly; below that it rounds to 2^-24 steps
            rounding = (kept.to(torch.float32) - rows).abs()
            torch.save(
                {"item_id": item.item_id, "positions": positions, "logits_fp16": kept, "device_dtype": "bfloat16"},
                full_logits_dir / f"{item.item_id}.pt",
            )
            document["full_logits_file"] = str(full_logits_dir / f"{item.item_id}.pt")
            document["full_logits_seconds"] = round(replay.full_logits_seconds, 3)
            document["fp16_rounded_values"] = int((rounding > 0).sum())
            document["fp16_max_abs_error"] = float(rounding.max())
            gather_seconds += replay.full_logits_seconds
        (records_dir / f"{item.item_id}.json").write_text(json.dumps(document) + "\n", encoding="utf-8")
        recorded.append(item.item_id)
        _log(
            "agreement_item",
            **{key: value for key, value in document.items() if key != "positions"},
            rows=len(document["positions"]),
        )
    session.reset()
    column = reference_corpus.device_reference_items(records_dir, items)
    reference_corpus.write_reference(
        out_dir / "agreement-device.json", manifest=manifest, producer=producer, items=column
    )
    score = reference_corpus.score_references(
        hf, {entry.item_id: entry for entry in column}, items, clear_margin=reference_corpus.CLEAR_MARGIN
    )
    score["column"] = producer
    (out_dir / "agreement-score.json").write_text(json.dumps(score, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "reference": str(reference),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "records": str(records_dir),
        "items": recorded,
        "positions": score["corpus"]["positions"],
        "prefill_mode": session.prefill_mode,
        "seconds": round(time.perf_counter() - started, 1),
        "full_logits": None if full_logits_dir is None else str(full_logits_dir),
        "full_logits_seconds": round(gather_seconds, 3),
        "corpus": score["corpus"],
        "parts": score["parts"],
    }
    _log("agreement_score", **{key: value for key, value in summary.items() if key != "items"})
    return summary


# -- the lanes' startup replay and driver --------------------------------------------------------


def replay_acceptance_lanes(
    lanes_session: Any,
    records: list[dict[str, Any]],
    *,
    lanes: int,
    drafts: int,
    single_stream: Mapping[str, Any],
    require_gate: bool,
    continuation: int = ACCEPTANCE_CONTINUATION,
    stall_budget_seconds: float | None = DEFAULT_STALL_BUDGET_SECONDS,
) -> dict[str, Any]:
    """The acceptance records through the lane scheduler (every record submitted at once, ``lanes`` admitted, the
    rest in the FIFO and admitted as lanes free, their admissions interleaved with the decoding lanes' passes under
    the served ``stall_budget_seconds``), each stream compared with the single-stream replay of the same process
    (``single_stream``: ``replay_acceptance``'s record on the same chain form) and with the CPU record.  The pin is
    exactness against the single stream on every record and the ``json`` gate; ``require_gate`` refuses a miss."""

    scheduler = Qwen38LaneScheduler(
        lanes=lanes, drafts=drafts, queue_limit=max(len(records), 1), stall_budget_seconds=stall_budget_seconds
    )
    tickets = []
    for record in records:
        expected = record["generated_token_ids"][:continuation]
        ticket = Qwen38LaneTicket(
            request_id=f"lanes-acceptance-{record['prompt']}",
            prompt_ids=list(record["prompt_token_ids"]),
            max_tokens=len(expected),
            stop_ids=(),
            think_budget=None,
        )
        tickets.append((record, expected, scheduler.submit(ticket)))
    scheduler.run(lanes_session, forever=False)
    reference = {row["prompt"]: row for row in single_stream["prompts"]}
    results = []
    for record, expected, ticket in tickets:
        if not ticket.done.is_set():
            raise Qwen38ChatChainError(f"lanes acceptance: ticket {ticket.request_id} did not finish")
        actual = _drain_ticket(ticket)
        single = reference.get(record["prompt"], {}).get("device_token_ids")
        row = {
            "prompt": record["prompt"],
            "prompt_tokens": len(record["prompt_token_ids"]),
            "compared_tokens": len(expected),
            "divergence_index": _divergence(actual, expected),
            "matched_tokens": len(expected) if _divergence(actual, expected) is None else _divergence(actual, expected),
            "device_token_ids": actual,
            "cpu_token_ids": expected,
            "single_stream_token_ids": single,
            "equals_single_stream": single is not None and actual == list(single),
            "single_stream_divergence_index": None if single is None else _divergence(actual, list(single)),
            "lane": ticket.lane,
            "finish": ticket.finish,
            "queue_wait_seconds": round(ticket.queue_wait, 4),
            "admission_seconds": round(ticket.admission_seconds, 4),
            "admission_wall_seconds": round(ticket.admission_wall_seconds, 4),
            "admission": {name: round(value, 4) for name, value in ticket.admission_segments.items()},
            "interleaved_passes": ticket.interleaved_passes,
            "stalled_seconds": round(ticket.stalled_seconds, 4),
            "passes": ticket.passes,
            "tokens_per_pass": None if not ticket.passes else round(ticket.committed / ticket.passes, 3),
        }
        results.append(row)
        _log("acceptance_lanes_prompt", **{key: value for key, value in row.items() if not key.endswith("_ids")})
    gate = next((row for row in results if row["prompt"] == ACCEPTANCE_GATE_PROMPT), None)
    gate_pass = gate is not None and gate["divergence_index"] is None and gate["compared_tokens"] == continuation
    equal = sum(1 for row in results if row["equals_single_stream"])
    equals_single_stream = equal == len(results)
    if require_gate and not (gate_pass and equals_single_stream):
        unequal = [row["prompt"] for row in results if not row["equals_single_stream"]]
        raise Qwen38ChatChainError(
            f"lanes acceptance gate: json {None if gate is None else gate['matched_tokens']} of {continuation} CPU greedy "
            f"tokens; {equal} of {len(results)} lane streams equal the single-stream replay (unequal: {unequal})"
        )
    return {
        "schema": "qwen38-chat-server-acceptance-lanes/v1",
        "continuation": continuation,
        "lanes": lanes,
        "drafts": drafts,
        "gate_prompt": ACCEPTANCE_GATE_PROMPT,
        "gate_pass": gate_pass,
        "equals_single_stream": equals_single_stream,
        "equal_prompts": equal,
        "compared_prompts": len(results),
        "passes": scheduler.passes,
        "admissions": scheduler.admissions,
        "interleaved_passes": scheduler.interleaved_passes,
        "stall_budget_seconds": scheduler.stall_budget_seconds,
        "pass_seconds_median": (
            None
            if not scheduler.pass_seconds
            else round(sorted(scheduler.pass_seconds)[len(scheduler.pass_seconds) // 2], 4)
        ),
        "prompts": results,
    }


def _drain_ticket(ticket: Qwen38LaneTicket) -> list[int]:
    """Every delivered id of a finished ticket."""

    tokens: list[int] = []
    while True:
        try:
            token, _finish = ticket.tokens.get_nowait()
        except queue.Empty:
            return tokens
        if token is None:
            return tokens
        tokens.append(token)


def _run_lanes_driver(scheduler: Qwen38LaneScheduler, lanes_session: Any, server: "Qwen38ChatHTTPServer") -> None:
    """The lane driver thread: the scheduler's loop over the device; a failure is the server's fatal error."""

    try:
        scheduler.run(lanes_session, forever=True)
    except BaseException as error:  # noqa: BLE001
        _log("lanes_driver_failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if server.fatal is None:
            server.fatal = error


# -- main --------------------------------------------------------------------------------------

# The launcher's ``common`` block: the model inputs and caches; the runtime identity's arguments are ``runtime_admission``'s.
PATH_ARGUMENTS = (
    "checkpoint",
    "component-cache-root",
    "routed-bf4-scratch-root",
    "model-io-cache-root",
    "phase-log",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PATH_ARGUMENTS:
        parser.add_argument(f"--{name}", type=Path, required=True)
    runtime_admission.add_arguments(parser)
    parser.add_argument(
        "--bf4-corpus",
        type=Path,
        default=None,
        help="a CPU-staged BF4 expert corpus root (tools/stage_full_bf4_cpu.py); without it the production cache "
        "under --routed-bf4-scratch-root is used and the missing layers are converted on the first start",
    )
    parser.add_argument("--bf4-corpus-verification", type=Path, default=None, help="the corpus's verification.json")
    parser.add_argument(
        "--bf4-producer-identity",
        type=Path,
        default=None,
        help="JSON with the corpus's expected source_head, runtime and checkpoint (pins what the record claims)",
    )
    parser.add_argument(
        "--bf4-stage-limit",
        type=int,
        default=None,
        help="convert at most N missing BF4 layers this run (a host that bounds a job's wall time; resumable)",
    )
    parser.add_argument(
        "--cpu-oracle", type=Path, default=None, help="the two-step CPU oracle (default: the shipped one)"
    )
    parser.add_argument(
        "--device-nodes",
        default=None,
        help="four KMD device nodes for a profile on other chips of a larger host, e.g. 4,5,6,7",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="open the mesh, convert the missing BF4 layers into the cache, close: no traces, no serving",
    )
    parser.add_argument("--evidence", type=Path, default=None, help="run directory (READY, STOPPED, ledgers)")
    parser.add_argument("--host", default="127.0.0.1", help="loopback unless the profile serves the LAN or --allow-lan")
    parser.add_argument(
        "--allow-lan", action="store_true", help="accept a non-loopback --host on a profile that does not serve the LAN"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--acceptance-prompts", type=Path, default=None, help="the CPU study's prompt-*-greedy.json")
    parser.add_argument("--require-json-96", action="store_true", help="refuse to serve unless json matches 96/96")
    parser.add_argument(
        "--agreement-reference",
        type=Path,
        default=None,
        help="the HF reference file of the corpus (tools/qwen38_reference_corpus.py hf): before serving, teacher-force "
        "the corpus through this chain and write the agreement records, the device column and its score against the "
        "reference into the evidence directory (needs --sampling: the candidate row)",
    )
    parser.add_argument(
        "--agreement-parts",
        nargs="*",
        choices=reference_corpus.PARTS,
        default=list(reference_corpus.PARTS),
        help="the corpus parts the agreement records cover (default: all)",
    )
    parser.add_argument(
        "--agreement-full-logits",
        type=Path,
        default=None,
        help="with --agreement-reference: also keep every recorded position's full-vocabulary logits per item in this "
        "directory (the corpus tool's --full-logits .pt format; one eager gather and readback per position)",
    )
    parser.add_argument(
        "--agreement-corpus",
        type=Path,
        default=reference_corpus.REFERENCE_DIR,
        help="the corpus directory (default: the committed Q38-REF-v1)",
    )
    parser.add_argument(
        "--prefill-mode",
        choices=PREFILL_MODES,
        default=DEFAULT_PREFILL_MODE,
        help="chunked: capture the chunk trace and prefill through it; teacher_forced: the decode traces only",
    )
    parser.add_argument(
        "--long-chunks",
        action="store_true",
        help="accepted for compatibility: the 128-row chunk trace and the 128-row chunks ahead of the 32-row ones are "
        "the served default of the chunked prefill (bitwise the 32-row chunks, docs/PREFILL.md); --prefill-mode "
        "teacher_forced has no chunks at all",
    )
    parser.add_argument(
        "--prefill-slab",
        type=int,
        default=None,
        metavar="ROWS",
        help="chunked prefill only: also capture a prefill slab of ROWS rows (a multiple of 128 in 256..4096; every "
        "dense linear as one matmul, the GDN state carried through the slab in one kernel call) and run slabs ahead "
        "of the 128-row chunks; implies --long-chunks; tolerance-class against the chunk bodies (docs/PREFILL.md)",
    )
    parser.add_argument(
        "--allocated-context",
        type=int,
        choices=RESIDENT_QSA_CACHE_CAPACITIES,
        default=RESIDENT_MAX_QSA_CACHE_CAPACITY,
        help="the resident build's allocated context; the KV caches, RoPE tables, the reserve and the context limit "
        "follow it (each non-default context builds its own component-cache identity; 262144 is single-user)",
    )
    parser.add_argument("--queue-limit", type=int, default=QUEUE_LIMIT, help="queued requests before 503")
    parser.add_argument(
        "--request-deadline-seconds", type=float, default=None, help="end a request with finish 'deadline' after this"
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=HEARTBEAT_SECONDS, help="progress log interval")
    parser.add_argument(
        "--socket-timeout-seconds",
        type=float,
        default=SOCKET_TIMEOUT_SECONDS,
        help="a client socket read or write blocked this long ends the request as disconnected",
    )
    parser.add_argument(
        "--stall-seconds",
        type=float,
        default=None,
        help="a device turn with no completed step for this long ends the server as fatal (exit 1) so a supervisor "
        "restarts it; every decode step, prefill event (16 forced tokens, 4 chunks) and admission from the queue "
        "restarts the clock, so a long prompt is never a stall; must exceed --socket-timeout-seconds; default: no "
        "watchdog (the launchers pass 300)",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=tuple(hardware_profiles.hardware_profile_table()),
        default=None,
        help="tt-quietbox | p150-line | tt-quietbox-2[-instance-1] (a private table adds development hosts)",
    )
    parser.add_argument("--validate-only", action="store_true", help="provenance and CPU preparation, no mesh")
    parser.add_argument(
        "--sampling",
        dest="sampling",
        action="store_true",
        default=False,
        help="capture TAIL with the candidate-row epilogue: sampled requests served (+0.3 ms per greedy token); a "
        "request without sampling fields stays the bitwise greedy loop; the launchers pass this",
    )
    parser.add_argument(
        "--no-sampling",
        dest="sampling",
        action="store_false",
        default=False,
        help="the argparse default, explicit: capture TAIL without the candidate row, greedy requests only (explicit "
        "sampling fields are refused with 400)",
    )
    parser.add_argument(
        "--device-sampler",
        dest="device_sampler",
        action="store_true",
        default=None,
        help="the default with --sampling: the on-device sampler after the candidate row (TAIL writes the sampled "
        "token; a sampled request the device policy admits runs the greedy loop plus one draw write per step)",
    )
    parser.add_argument(
        "--host-sampler",
        dest="device_sampler",
        action="store_false",
        default=None,
        help="with --sampling: every sampled request draws on the host over the read candidate row",
    )
    parser.add_argument(
        "--sampling-discriminator",
        action="store_true",
        help="after the acceptance replay run the sampling chain arms on the gate prompt, write "
        "sampling-discriminator.json and stop without serving",
    )
    parser.add_argument(
        "--mtp",
        type=int,
        choices=MTP_DRAFTS,
        default=None,
        help="draft K tokens per pass with the MTP layer (verify / draft / commit traces; greedy chunked-mode "
        "requests generate through the pass loop); default off",
    )
    parser.add_argument(
        "--mtp-gdn-anchor",
        choices=MTP_GDN_ANCHORS,
        default="off",
        help="MTP: the GDN state re-anchor (layer0 = layer 0's commits run the 1-row fp32 step recurrence)",
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=0,
        help="serve up to B concurrent greedy requests through the MTP lane chain (B in 2..8 with B x (k + 1) <= 32 "
        "rows, needs --mtp; greedy only: no --sampling, the second-queue early read and the per-request drafts "
        "field are refused); 0 (the default): the single-stream chain with its queue",
    )
    parser.add_argument(
        "--lanes-stall-budget",
        type=stall_budget_argument,
        default=DEFAULT_STALL_BUDGET_SECONDS,
        metavar="SECONDS",
        help="--lanes only: the admission work (seconds) the decoding lanes wait for before their pass runs between "
        "two segments of an admission (0: a pass at every segment; 'off': every admission runs whole and stalls the "
        f"lanes for its length); default {DEFAULT_STALL_BUDGET_SECONDS} (docs/SERVER.md)",
    )
    return parser


def stall_budget_argument(text: str) -> float | None:
    """``--lanes-stall-budget``: a non-negative number of seconds, or ``off`` / ``none`` for whole admissions."""

    if text.strip().lower() in ("off", "none"):
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"a number of seconds or 'off', got {text!r}") from error
    if value != value or value < 0:
        raise argparse.ArgumentTypeError(f"a non-negative number of seconds or 'off', got {text!r}")
    return value


def main() -> int:
    args = _parser().parse_args()
    if args.prefill_slab is not None and not is_slab_rows(args.prefill_slab):
        raise SystemExit(f"--prefill-slab takes a multiple of 128 in 256..4096, got {args.prefill_slab}")
    if args.prefill_slab is not None:
        # the slab MoE switches (QWEN38_MOE_SLAB_ONE_CALL / QWEN38_MOE_SLAB_RINGS) are admitted here, before any
        # device is opened: a refused ring count is a start-up error, not a poisoned model 79 s into the warm pass
        try:
            admit_slab_moe_switches()
        except ValueError as error:
            raise SystemExit(str(error)) from error
    try:
        hardware_profile = hardware_profiles.resolve_hardware_profile(args.hardware_profile)
    except hardware_profiles.HardwareProfileError as error:
        raise SystemExit(str(error)) from error
    if args.host not in ("127.0.0.1", "::1", "localhost") and not (args.allow_lan or hardware_profile.lan_serving):
        raise SystemExit(f"--host must be loopback on {hardware_profile.host} without --allow-lan, got {args.host!r}")
    if not 1024 <= args.port <= 65535:
        raise SystemExit(f"--port must be in [1024, 65535], got {args.port}")
    if args.evidence is None and not args.validate_only:
        raise SystemExit("--evidence is required to serve")
    if not 1 <= args.queue_limit <= 64 or args.heartbeat_seconds <= 0:
        raise SystemExit(
            f"--queue-limit must be in [1, 64] and --heartbeat-seconds positive, got {args.queue_limit} {args.heartbeat_seconds}"
        )
    if args.request_deadline_seconds is not None and args.request_deadline_seconds <= 0:
        raise SystemExit(f"--request-deadline-seconds must be positive, got {args.request_deadline_seconds}")
    if args.socket_timeout_seconds <= 0:
        raise SystemExit(f"--socket-timeout-seconds must be positive, got {args.socket_timeout_seconds}")
    if args.stall_seconds is not None and args.stall_seconds <= args.socket_timeout_seconds:
        # A client write blocked for the socket timeout is not a device step; the watchdog must outlast it.
        raise SystemExit(
            f"--stall-seconds must exceed --socket-timeout-seconds {args.socket_timeout_seconds}, got {args.stall_seconds}"
        )
    args.device_sampler = effective_device_sampler(args)
    if args.sampling_discriminator and (not args.sampling or args.acceptance_prompts is None):
        raise SystemExit("--sampling-discriminator needs --sampling and the acceptance prompt records")
    lanes_switch = lanes_switches(os.environ, args)  # the lanes' geometry and refusals
    if lanes_switch is None and args.lanes_stall_budget != DEFAULT_STALL_BUDGET_SECONDS:
        raise SystemExit("--lanes-stall-budget needs --lanes (the budget is the lane scheduler's)")
    # The 128-row chunks are the served default of the chunked prefill (bitwise the 32-row chunks; the 560-token chat
    # prompt prefills in 0.81 s against 1.19 s in 32-row chunks alone, docs/PREFILL.md): the flag is accepted, the
    # teacher-forced mode has no chunk trace to extend.
    args.long_chunks = args.prefill_mode == "chunked"
    if args.agreement_reference is not None and not args.sampling:
        raise SystemExit("--agreement-reference needs --sampling (the records read the candidate row)")
    if args.agreement_reference is not None and not args.agreement_reference.is_file():
        raise SystemExit(f"--agreement-reference {args.agreement_reference}: not a file")
    if args.agreement_full_logits is not None and args.agreement_reference is None:
        raise SystemExit("--agreement-full-logits needs --agreement-reference (the positions it keeps logits for)")
    if args.bf4_stage_limit is not None and args.bf4_stage_limit <= 0:
        raise SystemExit(f"--bf4-stage-limit must be positive, got {args.bf4_stage_limit}")
    if args.device_nodes is not None:
        try:
            hardware_profile = hardware_profile.with_device_nodes(
                tuple(int(node) for node in args.device_nodes.split(","))
            )
        except (ValueError, hardware_profiles.HardwareProfileError) as error:
            raise SystemExit(f"--device-nodes {args.device_nodes!r}: {error}") from error

    def marker(phase: str) -> None:
        append_marker(args.phase_log, phase)
        _log("phase", phase=phase)

    for name, expected in (
        ("TT_VISIBLE_DEVICES", hardware_profile.visible_devices),
        ("QWEN38_HARDWARE_MODE", "diagnostic_non_promoting"),
        ("TT_METAL_TRACE_ALLOC_TRACKING", "1"),
    ):
        if os.environ.get(name) != expected:
            raise SystemExit(f"{name} is {os.environ.get(name)!r}, expected {expected!r}")
    mtp_sampled = mtp_sampled_switch(os.environ, applicable=args.mtp is not None and bool(args.sampling))
    mtp_drafts_admitted = mtp_drafts_per_request_switch(os.environ, drafts=args.mtp)
    if mtp_sampled and (args.mtp is None or not args.sampling):
        raise SystemExit(
            f"{MTP_SAMPLED_VARIABLE}=1 needs --mtp and --sampling (the pass loop drafts for sampled requests)"
        )
    profiler = tuple(name for name in PROFILER_VARIABLES if os.environ.get(name) is not None)
    if profiler:
        raise SystemExit(f"profiler instrumentation is set: {profiler}")
    runtime = runtime_admission.admit_runtime(args)
    _log("runtime", **{key: value for key, value in runtime.items() if key != "bundle"})
    lock_proof = hardware_profiles.verify_inherited_locks(hardware_profile)
    # The route: pinned by the profile, or (the p150 line) derived from the cluster descriptor now and recorded.
    hardware_profile, route_derivation = resolve_route(hardware_profile)
    _log(
        "route",
        lane=hardware_profile.lane,
        route=list(hardware_profile.route),
        nodes=list(hardware_profile.route_nodes),
    )
    prepared, _oracle = runtime_admission.prepare_cpu(
        args, marker=marker, hardware_profile=hardware_profile, identity=runtime
    )
    template = Qwen38OfficialChatTemplate(prepared.checkpoint.root)
    records = load_acceptance_records(args.acceptance_prompts) if args.acceptance_prompts is not None else []
    records = slab_records_admitted(records, slab_rows=args.prefill_slab)
    resident_context = Qwen38ResidentContext(args.allocated_context)
    # The MTP admission decides on the live allocator once the resident weights are built (the chain's open,
    # Qwen38TracedChain.open); the 2026-09-04 table is the no-device fallback, logged here for the record, never a
    # refusal before the mesh opens.  The table row counts the verify forms the open will capture (one table, the
    # switch), so its record and the live decision's agree on them.
    mtp_device_accept = device_accept_switch()  # the acceptance module's switch; the extension builds from it at open
    if mtp_device_accept and not mtp_sampled:
        raise SystemExit(
            f"{DEVICE_ACCEPT_SWITCH}=1 needs the split verify ({MTP_SAMPLED_VARIABLE} on: --mtp and --sampling)"
        )
    # The second command queue's early rows read: on with --mtp unless QWEN38_MTP_PLE_EARLY=0; an explicit 1 needs --mtp.
    mtp_ple_early = ple_early_switch(os.environ, applicable=args.mtp is not None)
    if mtp_ple_early and args.mtp is None:
        raise SystemExit(f"{PLE_EARLY_VARIABLE}=1 needs --mtp (the early rows are the MTP pass loop's)")
    if lanes_switch is not None:
        mtp_ple_early = False  # the lanes run the one-queue form: the second queue's early read has no lanes form
    # The open captures these forms; its gate evaluates the same configuration.
    forms = mtp_verify_forms(mtp_sampled, mtp_device_accept)
    # QWEN38_MTP_MOE_ROWS (diagnostic, default unset): the verify MoE row count forced on the chain (5, 6 or 32);
    # it keys the admission's states term and reaches the chain open; needs --mtp.
    mtp_moe_rows = mtp_v2.moe_rows_override()
    if mtp_moe_rows is not None and args.mtp is None:
        raise SystemExit(f"{mtp_v2.MOE_ROWS_SWITCH} needs --mtp (the verify MoE form is the MTP chain's)")
    mtp_admission_table = (
        None
        if args.mtp is None
        else mtp_capacity_admission(
            resident_context.allocated_context,
            drafts=args.mtp,
            verify_forms=len(forms),
            long_chunks=bool(args.long_chunks) or args.prefill_slab is not None,
            moe_rows=mtp_moe_rows,
            gdn_rows_scan=fused_module.enabled(gdn_rows_scan_module.NAME),
            slab_rows=args.prefill_slab,
            # the tower's MODELED terms (the image path is the default path; the live decision is the warm hook's)
            vision_resident_bytes_per_bank=-(-vision_resident_layout(mesh_size=4)["device_total"] // 8),
            vision_peak_activation_bytes_per_bank=-(
                -VISION_ROW_BUCKETS[-1] * PEAK_ACTIVATION_BYTES_PER_ROW_PER_DIE // 8
            ),
        )
    )
    if mtp_admission_table is not None:
        mtp_admission_table["verify_forms_captured"] = list(forms)
        _log("mtp_admission_table_fallback", **mtp_admission_table)
    summary = {
        "mode": "chat_server_single_trace_chain" if lanes_switch is None else "chat_server_mtp_lanes",
        "model": MODEL_ID,
        "hardware_profile": hardware_profile.host,
        "hardware_partition": hardware_profile.partition,
        "device_nodes": list(hardware_profile.device_nodes),
        "route": list(hardware_profile.route),
        "route_nodes": list(hardware_profile.route_nodes),
        "route_derivation": route_derivation["route_derivation"],
        # a ring: the ring walk from the lowest chip next to the fabric's line order the mesh opens in (QuietBox: equal)
        "route_ring_walk": route_derivation.get("ring_walk_route"),
        "route_ring_walk_agrees": route_derivation.get("ring_walk_agrees"),
        "allocated_context": resident_context.allocated_context,
        "context_limit": resident_context.context_limit,
        "source": {
            "worktree": runtime["repo"],
            "head": runtime["head"],
            "tree": runtime["tree"],
            "clean": not runtime["dirty"],
        },
        "runtime": runtime,
        "prepared": prepared.summary(),
        "template_sha256": PINNED_TOKENIZER_ARTIFACTS["chat_template.jinja"],
        "acceptance_records": [record["prompt"] for record in records],
        "require_json_96": bool(args.require_json_96),
        "prefill_mode": args.prefill_mode,
        "host": args.host,
        "lan_serving": bool(args.allow_lan or hardware_profile.lan_serving),
        "port": args.port,
        "queue_limit": args.queue_limit,
        "request_deadline_seconds": args.request_deadline_seconds,
        "socket_timeout_seconds": args.socket_timeout_seconds,
        "stall_seconds": args.stall_seconds,
        "defaults": {"enable_thinking": ENABLE_THINKING_DEFAULT, "reasoning_effort": REASONING_EFFORT_DEFAULT},
        "sampling": (
            "candidate_row_device_sampler"
            if args.device_sampler
            else "candidate_row_host_sampler"
            if args.sampling
            else "greedy"
        ),
        "sampling_discriminator": bool(args.sampling_discriminator),
        "agreement": (
            None
            if args.agreement_reference is None
            else {
                "reference": str(args.agreement_reference),
                "corpus": str(args.agreement_corpus),
                "parts": list(args.agreement_parts),
                "full_logits": None if args.agreement_full_logits is None else str(args.agreement_full_logits),
            }
        ),
        "mtp": {
            "k": args.mtp,
            "anchor": args.mtp_gdn_anchor if args.mtp is not None else None,
            "admission": None,  # the chain's live admission once it opens (report["mtp"]["admission"])
            "admission_table_fallback": mtp_admission_table,
            "sampled": mtp_sampled,
            "device_accept": mtp_device_accept,
            "ple_early": mtp_ple_early,
            "moe_rows": mtp_moe_rows,  # the switch's forced verify MoE rows, None = moe_rows_for(k + 1)
            # QWEN38_MTP_DRAFTS_PER_REQUEST=1: the chains captured at open, the default first; a request picks one
            # with extra_body.mtp_drafts (the fingerprint and the acceptance baselines are the default chain's)
            "drafts_admitted": list(mtp_drafts_admitted),
        },
        # --lanes B: the lane chain's geometry, the fold's off switch and the admission (filled at the chain's warm hook)
        "lanes": (
            None
            if lanes_switch is None
            else {
                "lanes": lanes_switch["lanes"],
                "drafts": lanes_switch["drafts"],
                "rows": lanes_switch["rows"],
                "greedy_only": True,
                "admission": None,
            }
        ),
        # the vision tower's residency: decided and made in the chain's warm hook (ttnn/vision_residency), None until then
        "vision": None,
        "fused_kernels": sorted(fused.enabled_names()),
        # What a seed reproduces against: the source head and the runtime; with the pass loop drafting for sampled
        # requests the draw order is the pass's, so the switch and k are part of the identity.
        "system_fingerprint": f"{runtime['head'][:12]}-{runtime['extension_sha256'][:12]}"
        + (f"-mtp{args.mtp}-sampled" if mtp_sampled else "")
        + ("-device-accept" if mtp_sampled and device_accept_switch() else "")
        + ("" if lanes_switch is None else f"-lanes{lanes_switch['lanes']}"),
    }
    if args.validate_only:
        print(json.dumps({"status": "pass", "mesh_open_requested": False, **summary}, sort_keys=True))
        return 0

    evidence = args.evidence.resolve(strict=True)
    ready_marker, stopped_marker = evidence / "READY", evidence / "STOPPED"
    for path in (ready_marker, stopped_marker):
        if path.exists():
            raise SystemExit(f"{path} exists: this evidence directory was already used")
    report: dict[str, Any] = {
        "schema": "qwen38-chat-server-run/v1",
        "status": "fail",
        "start_utc": utc_now(),
        "pid": os.getpid(),
        "lock_proof": lock_proof,
        "mesh_closed": False,
        "fabric_disabled": False,
        "cleanup_errors": [],
        **summary,
    }
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    started_ns = time.perf_counter_ns()
    mesh = chain = server = None
    lanes_session = None
    scheduler = None
    if lanes_switch is not None:
        from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lanes_session import Qwen38LanesSession

        lanes_session = Qwen38LanesSession(lanes=lanes_switch["lanes"], drafts=lanes_switch["drafts"], marker=marker)
        scheduler = Qwen38LaneScheduler(
            lanes=lanes_switch["lanes"],
            drafts=lanes_switch["drafts"],
            queue_limit=args.queue_limit,
            stall_budget_seconds=args.lanes_stall_budget,
        )
    if not (args.prepare_only or args.sampling_discriminator):
        # The port is claimed before the minutes of mesh open, captures and replay: a taken port fails here.
        try:
            server = Qwen38ChatHTTPServer(
                (args.host, args.port),
                None,
                ledger=evidence / "requests.jsonl",
                queue_limit=args.queue_limit,
                request_deadline_seconds=args.request_deadline_seconds,
                heartbeat_seconds=args.heartbeat_seconds,
                socket_timeout_seconds=args.socket_timeout_seconds,
                stall_seconds=args.stall_seconds,
                system_fingerprint=summary["system_fingerprint"],
                mtp_drafts_admitted=mtp_drafts_admitted,
                listen=False,
                lanes=scheduler,
            )
        except OSError as error:
            raise SystemExit(f"--host {args.host} --port {args.port}: cannot bind: {error}") from error
    fabric_enabled = False
    uncertain = False
    cleanup_errors: list[str] = []
    try:
        mesh, report["topology"] = open_partition_b_mesh(
            marker, hardware_profile, command_queues=2 if mtp_ple_early else 1
        )
        fabric_enabled = True
        if args.prepare_only:
            # The caches only: the builder converts the missing BF4 layers (bounded by --bf4-stage-limit) and the
            # run stops; a later start finds them.  The component and model-I/O caches fill at the first target build.
            construction = construct_live_decode_diagnostic(
                prepared,
                mesh_device=mesh,
                collective_topology=ttnn.Topology.Linear,
                marker=marker,
                bf4_stage_limit=args.bf4_stage_limit,
                require_complete=False,
            )
            missing = missing_bf4_layers(construction.builder)
            report["prepare"] = {
                "bf4_cache_root": str(construction.production_cache.root),
                "bf4_admission": construction.bf4_admission,
                "staged_bf4_layers": [list(slot) for slot in construction.staged_bf4_layers],
                "missing_bf4_layers": [list(slot) for slot in missing],
            }
            _log(
                "prepared",
                staged=len(construction.staged_bf4_layers),
                missing=len(missing),
                admission=construction.bf4_admission,
            )
            ttnn.synchronize_device(mesh)
            report["status"] = "stopped"
            raise Qwen38ChatServerStop("prepare-only run complete")
        warm_hook = None
        if lanes_session is not None:

            def warm_hook(opened_chain) -> None:
                # The lanes' admission on the live allocator (after the resident build, the MTP states and the warm
                # pass; before any capture), then their allocation and compile in the chain's warm hook.  What the
                # chain still allocates after this reading (its traces) is reserved on the free side.
                live = hardware_profiles.symmetric_mesh_dram_memory(mesh, hardware_profile.route)
                reserved = RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND + int(
                    opened_chain.mtp.admission["mtp_growth_estimate_bytes_per_bank"]["traces"]
                )
                if bool(args.long_chunks) or args.prefill_slab is not None:
                    reserved += LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES
                admission = lanes_capacity_admission(
                    resident_context.allocated_context,
                    lanes=lanes_switch["lanes"],
                    drafts=lanes_switch["drafts"],
                    live=live,
                    reserved_bytes_per_bank=reserved,
                    gdn_rows_scan=fused_module.enabled(gdn_rows_scan_module.NAME),
                )
                lanes_session.admission = admission
                summary["lanes"]["admission"] = admission
                _log(
                    "lanes_admission",
                    **{key: value for key, value in admission.items() if key != "decided_by"},
                    **admission["decided_by"],
                )
                if not admission["fits"]:
                    raise Qwen38ChatChainError(
                        f"--lanes {lanes_switch['lanes']} refused at allocated context {resident_context.allocated_context}: "
                        f"{admission['decided_by']['shortfalls']} (required {admission['required_free_bytes_per_bank']} B per bank "
                        f"against {admission['free_bytes_per_bank']} free, contiguous {admission['largest_contiguous_bytes_free_per_bank']} "
                        f"against {admission['required_largest_contiguous_bytes_per_bank']})"
                    )
                lanes_session.prepare(opened_chain)

        vision_state: dict[str, Any] = {"residency": None}

        def vision_warm_hook(opened_chain) -> None:
            # The image path is the default path: the tower's residency is decided on the live allocator here (after
            # the resident build, the MTP states, the warm pass and the lanes' allocations; before any capture) and
            # made here (the weights resident, one forward per row bucket compiled).  A shortfall leaves the process
            # text-only with the reason on READY / health and on every image refusal.
            reserved = RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND
            if opened_chain.mtp is not None:
                reserved += int(opened_chain.mtp.admission["mtp_growth_estimate_bytes_per_bank"]["traces"])
            if bool(args.long_chunks) or args.prefill_slab is not None:
                reserved += LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES
            residency = opened_chain.construction.builder.enable_vision(
                lambda: hardware_profiles.symmetric_mesh_dram_memory(mesh, hardware_profile.route),
                reserved_bytes_per_bank=reserved,
            )
            residency.vision_warm_hook(opened_chain)
            vision_state["residency"] = residency
            summary["vision"] = residency.vision_summary()
            _log(
                "vision_residency",
                **{
                    key: value
                    for key, value in summary["vision"].items()
                    if key
                    not in ("prewarm", "prewarm_plan_modeled", "ladders", "dram_after_load", "dram_after_prewarm")
                },
                prewarm_seconds=[round(row["ready_seconds_measured"], 2) for row in summary["vision"]["prewarm"]],
            )

        warm_hook = compose_warm_hooks(warm_hook, vision_warm_hook)

        chain = construct_chain(
            prepared,
            mesh,
            marker=marker,
            chunked_prefill=args.prefill_mode == "chunked",
            sampling=bool(args.sampling),
            bf4_stage_limit=args.bf4_stage_limit,
            long_chunks=bool(args.long_chunks) or args.prefill_slab is not None,
            slab_rows=args.prefill_slab,
            mtp=args.mtp,
            mtp_gdn_anchor=args.mtp_gdn_anchor,
            device_sampler=bool(args.device_sampler),
            mtp_sampled=mtp_sampled,
            mtp_device_accept=mtp_device_accept,
            mtp_moe_rows=mtp_moe_rows,
            mtp_alternates=mtp_drafts_admitted[1:],
            mtp_ple_early=mtp_ple_early,
            warm_hook=warm_hook,
        )
        if chain.allocated_context != resident_context.allocated_context:
            raise Qwen38ChatChainError(
                f"chain allocated context {chain.allocated_context} vs requested {resident_context.allocated_context}"
            )
        session = Qwen38ChatSession(chain, template, prefill_mode=args.prefill_mode)
        if server is not None:
            server.vision = vision_state["residency"]
            if lanes_session is not None:
                # the driver thread runs the resident tower inside an image admission (the lanes session's "tower" segment)
                lanes_session.tower = server.vision_prompt_for
        if os.environ.get(DEVICE_ACCEPT_DUMP_VARIABLE):
            session.device_accept_dump = Path(os.environ[DEVICE_ACCEPT_DUMP_VARIABLE])
            session.device_accept_dump.mkdir(parents=True, exist_ok=True)
        if session.prefill_mode != args.prefill_mode:
            raise Qwen38ChatChainError(f"session prefill mode {session.prefill_mode} vs requested {args.prefill_mode}")
        if (session.sampling is not None) != bool(args.sampling):
            raise Qwen38ChatChainError(f"session sampling {session.sampling is not None} vs requested {args.sampling}")
        if (getattr(session.sampling, "sampler", None) is not None) != bool(args.device_sampler):
            raise Qwen38ChatChainError(f"session device sampler vs requested {args.device_sampler}")
        if (session.mtp is not None) != (args.mtp is not None):
            raise Qwen38ChatChainError(f"session mtp {session.mtp is not None} vs requested {args.mtp}")
        if session.mtp is not None and session.mtp.sampled != mtp_sampled:
            raise Qwen38ChatChainError(f"session mtp sampled {session.mtp.sampled} vs requested {mtp_sampled}")
        if session.mtp is not None and session.mtp.device_accept != mtp_device_accept:
            raise Qwen38ChatChainError(
                f"session mtp device_accept {session.mtp.device_accept} vs requested {mtp_device_accept}"
            )
        if chain.mtp is not None:
            report["mtp"]["admission"] = chain.mtp.admission
        if session.mtp is not None and (
            session.mtp.admission["verify_forms"] != mtp_admission_table["verify_forms"]
            or session.mtp.admission["verify_forms_captured"] != mtp_admission_table["verify_forms_captured"]
        ):
            raise Qwen38ChatChainError(
                f"the chain's live admission counts {session.mtp.admission['verify_forms']} verify form(s) "
                f"{session.mtp.admission['verify_forms_captured']}, the table fallback evaluated "
                f"{mtp_admission_table['verify_forms']} {mtp_admission_table['verify_forms_captured']}"
            )
        if session.context_limit != resident_context.context_limit:
            raise Qwen38ChatChainError(
                f"session context limit {session.context_limit} vs the build's {resident_context.context_limit}"
            )
        report["chain"] = {
            "allocated_context": chain.allocated_context,
            "context_limit": session.context_limit,
            "open_seconds": chain.open_seconds,
            "capture_ms": chain.capture_ms,
            "program_cache_entries": chain.program_cache_entries,
            "head_traces": len(chain.head_trace_ids),
            "tail_traces": len(chain.tail_trace_ids),
            "chunk_traces": len(
                [t for t in (chain.chunk_trace_id, chain.long_chunk_trace_id, chain.slab_trace_id) if t is not None]
            ),
            "prefill_slab_rows": args.prefill_slab,
            "prefill_long_chunks": chain.long_chunk_trace_id is not None,
            "chunk_capture_ms": chain.chunk_capture_ms,
            "long_chunk_capture_ms": chain.long_chunk_capture_ms,
            "prefill_mode": session.prefill_mode,
            "sampling": chain.sampling is not None,
            "dram_workers_per_bank": chain.construction.builder.decode_dram_workers_per_bank,
            "dram_workers_fallback": chain.construction.builder.decode_dram_workers_fallback,
            "dram_workers_placement": chain.construction.builder.decode_dram_workers_placement,
            "dense_weight_dtype": chain.construction.builder.dense_weight_plan.describe(),
            "mtp": (
                None
                if chain.mtp is None
                else {
                    "k": chain.mtp.drafts,
                    "anchor": chain.mtp.anchor,
                    "sampled": chain.mtp.sampled,
                    "device_accept": chain.mtp.device_accept,
                    "traces": len(chain.mtp.captured_trace_ids()),
                    "long_chunk_extension": chain.mtp.long_chunk_extension is not None,
                    "capture_ms": chain.mtp.capture_ms,
                    "trace_dram_bytes_per_bank": chain.mtp.trace_dram_bytes_per_bank,
                    "dram_bytes_per_bank": chain.mtp.dram_bytes_per_bank,
                    "admission": chain.mtp.admission,
                }
            ),
        }
        # The allocator after every capture (the traces bake their addresses in; nothing is allocated after this
        # point on the serving path): the build's per-device headroom, in the report, READY and /health.
        ttnn.synchronize_device(mesh)
        report["chain"]["dram_after_captures"] = hardware_profiles.symmetric_mesh_dram_memory(
            mesh, hardware_profile.route
        )
        _log("dram_after_captures", allocated_context=chain.allocated_context, **report["chain"]["dram_after_captures"])
        if records:
            marker("before-chat-acceptance-replay")
            report["acceptance"] = replay_acceptance(
                session, records, require_gate=args.require_json_96, handoff_gate=session.mtp is not None
            )
            (evidence / "acceptance.json").write_text(
                json.dumps(report["acceptance"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            marker("after-chat-acceptance-replay")
        if lanes_session is not None:
            # The three lane traces after the chain's, the tracker over every trace, then the HARD gate: the lanes'
            # measured growth (states at the hook, traces now) against the admission's estimate.
            lanes_session.capture(chain, session)
            growth = lanes_session.growth_bytes_per_bank()
            estimate = lanes_session.admission["required_free_bytes_per_bank"]
            _log("lanes_captured", **lanes_session.summary())
            if sum(growth.values()) > estimate:
                raise Qwen38ChatChainError(
                    f"lanes DRAM growth {sum(growth.values())} bytes per bank {growth} exceeds the admission's estimate "
                    f"{estimate} {lanes_session.admission['lanes_growth_estimate_bytes_per_bank']} for {lanes_switch['lanes']} lanes "
                    f"at k={lanes_switch['drafts']}, allocated context {chain.allocated_context}"
                )
            report["chain"]["lanes"] = lanes_session.summary()
            ttnn.synchronize_device(mesh)
            report["chain"]["dram_after_captures"] = hardware_profiles.symmetric_mesh_dram_memory(
                mesh, hardware_profile.route
            )
            if records:
                # The lanes' pin: the same records through the scheduler, B at a time, against the single-stream
                # replay just made in this process (the same chain form, the wrap): every stream must equal it.
                marker("before-lanes-acceptance-replay")
                report["acceptance_lanes"] = replay_acceptance_lanes(
                    lanes_session,
                    records,
                    stall_budget_seconds=args.lanes_stall_budget,
                    lanes=lanes_switch["lanes"],
                    drafts=lanes_switch["drafts"],
                    single_stream=report["acceptance"],
                    require_gate=args.require_json_96,
                )
                (evidence / "acceptance-lanes.json").write_text(
                    json.dumps(report["acceptance_lanes"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                marker("after-lanes-acceptance-replay")
        if args.agreement_reference is not None:
            marker("before-agreement-records")
            report["agreement"] = record_agreement(
                session,
                reference=args.agreement_reference,
                corpus_dir=args.agreement_corpus,
                parts=args.agreement_parts,
                out_dir=evidence,
                producer={
                    "kind": "device",
                    "label": f"served chain, {session.prefill_mode} prefill, {summary['system_fingerprint']}",
                    "prefill_mode": session.prefill_mode,
                    "sampling": summary["sampling"],
                    "mtp": args.mtp,
                    "allocated_context": chain.allocated_context,
                    "source_head": runtime["head"],
                    "runtime_sha256": runtime["extension_sha256"],
                    "normalisation": "log-softmax over the candidate row",
                },
                full_logits_dir=args.agreement_full_logits,
            )
            marker("after-agreement-records")
        if args.sampling_discriminator:
            # The chain arms (arm b is the replay above) on the creative record, whose distribution is flat enough
            # for a sampled stream to leave the greedy one (the json gate prompt is near-deterministic), then a
            # clean stop: no serving.
            marker("before-sampling-discriminator")
            record = next(
                (record for record in records if record["prompt"] == DISCRIMINATOR_PROMPT),
                next(record for record in records if record["prompt"] == ACCEPTANCE_GATE_PROMPT),
            )
            result = sampling_step.run_discriminator(session, record["prompt_token_ids"])
            result["prompt"] = record["prompt"]
            result["program_cache_delta"] = resident_decode.program_cache_count(mesh) - chain.program_cache_entries
            result["acceptance_gate_pass"] = report["acceptance"]["gate_pass"]
            result["pass"] = bool(
                result["pass"] and result["program_cache_delta"] == 0 and result["acceptance_gate_pass"]
            )
            report["sampling_discriminator"] = result
            (evidence / "sampling-discriminator.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            marker("after-sampling-discriminator")
            _log(
                "sampling_discriminator",
                **{key: value for key, value in result.items() if key not in ("rows", "greedy", "sampled")},
            )
            report["status"] = "stopped"
        else:
            server.session = session
            server.dram_after_captures = report["chain"]["dram_after_captures"]
            if lanes_session is not None:
                # The driver thread: the only device thread from here on (the handler threads submit and read tickets).
                server.lanes_thread = threading.Thread(
                    target=_run_lanes_driver, args=(scheduler, lanes_session, server), name="lane-driver", daemon=True
                )
                server.lanes_thread.start()
            server.server_activate()
            ready = {
                "pid": os.getpid(),
                "host": args.host,
                "port": args.port,
                "utc": utc_now(),
                "startup_seconds": (time.perf_counter_ns() - started_ns) / 1e9,
                "capture_ms": chain.capture_ms,
                "allocated_context": chain.allocated_context,
                "free_bytes_per_bank": report["chain"]["dram_after_captures"]["free_bytes_per_bank"],
                "acceptance_gate_pass": None if not records else report["acceptance"]["gate_pass"],
                "mtp": report["chain"]["mtp"],
                # the drafting chains a request may pick (QWEN38_MTP_DRAFTS_PER_REQUEST; one entry = the --mtp chain only)
                "mtp_drafts_admitted": list(mtp_drafts_admitted),
                "fused_kernels": sorted(fused.enabled_names()),
                "dram_workers_per_bank": report["chain"]["dram_workers_per_bank"],
                "moe_local_output": moe_local_output_enabled(),
                "moe_rows_form": moe_rows_form(),
                "dram_workers_fallback": report["chain"]["dram_workers_fallback"],
                "dram_workers_placement": report["chain"].get("dram_workers_placement"),
                "dense_weight_dtype": report["chain"]["dense_weight_dtype"],
                "route": list(hardware_profile.route),
                "route_derivation": route_derivation["route_derivation"],
                "mode": summary["mode"],
                "vision": summary["vision"],
                "lanes": (
                    None
                    if lanes_session is None
                    else {
                        **lanes_session.summary(),
                        "greedy_only": True,
                        "queue_limit": args.queue_limit,
                        "stall_budget_seconds": args.lanes_stall_budget,
                        "acceptance": (
                            None
                            if "acceptance_lanes" not in report
                            else {
                                key: report["acceptance_lanes"][key]
                                for key in ("gate_pass", "equals_single_stream", "equal_prompts", "compared_prompts")
                            }
                        ),
                    }
                ),
            }
            ready_marker.write_text(json.dumps(ready, sort_keys=True) + "\n", encoding="utf-8")
            marker("chat-server-ready")
            _log("ready", **ready, base_url=f"http://{args.host}:{args.port}/v1", model=MODEL_ID)
            try:
                server.serve_forever(poll_interval=0.5)
            except Qwen38ChatServerStop:
                # The drain: no new connections, queued requests refused, the request in flight ends at its next
                # step with finish "shutdown" and gets its reply; the chain is released only once the device is free
                # (a second signal or a drain past DRAIN_SECONDS leaves the boundary uncertain).
                uncertain = True
                report["status"] = "stopped_mid_request"
                _log("draining", busy=server.device_busy, queue_depth=server.queue_depth, seconds=DRAIN_SECONDS)
                if server.drain(DRAIN_SECONDS) and not session.poisoned:
                    uncertain = False
                    report["status"] = "stopped"
    except Qwen38ChatServerStop as stop:
        if report["status"] != "stopped":
            report["error"] = f"{type(stop).__name__}: {stop}"
            _log("failed", error=report["error"])
    except BaseException as error:
        uncertain = uncertain or (
            server is not None
            and (
                server.device_busy
                or (server.session is not None and server.session.poisoned)
                or (scheduler is not None and scheduler.fatal is not None)
            )
        )
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        _log("failed", error=report["error"])
    finally:
        if server is not None:
            server.server_close()
        if scheduler is not None and scheduler.running:
            scheduler.stop()
            if server is not None and server.lanes_thread is not None:
                server.lanes_thread.join(timeout=DRAIN_SECONDS)
                uncertain = uncertain or server.lanes_thread.is_alive()
        # The runner's cleanup: a failure mid-loop leaves queue and trace ownership
        # uncertain; then the mesh is closed without release work.
        if lanes_session is not None and not uncertain:
            try:
                marker("before-lanes-close")
                lanes_session.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"lanes_close:{type(error).__name__}:{error}")
        if chain is not None and not uncertain:
            try:
                marker("before-chat-chain-close")
                chain.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"chain_close:{type(error).__name__}:{error}")
        if mesh is not None:
            if not uncertain:
                try:
                    ttnn.synchronize_device(mesh)
                    mesh.disable_and_clear_program_cache()
                except BaseException as error:  # noqa: BLE001
                    cleanup_errors.append(f"program_cache_disable:{type(error).__name__}:{error}")
            try:
                marker("before-mesh-close")
                ttnn.close_mesh_device(mesh)
                report["mesh_closed"] = True
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"mesh_close:{type(error).__name__}:{error}")
        if fabric_enabled:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                report["fabric_disabled"] = True
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"fabric_disable:{type(error).__name__}:{error}")
        if cleanup_errors:
            report["status"] = "fail"
        report.update(
            end_utc=utc_now(),
            cleanup_errors=cleanup_errors,
            uncertain_boundary=uncertain,
            requests_served=(
                None
                if server is None or server.session is None
                else (server.session.requests_served if scheduler is None else scheduler.requests_done)
            ),
        )
        stopped_marker.write_text(
            json.dumps({"pid": os.getpid(), "utc": utc_now(), "status": report["status"]}) + "\n",
            encoding="utf-8",
        )
        write_result(evidence / "result.json", report)
        _log("stopped", status=report["status"], cleanup_errors=cleanup_errors)
    return 0 if report["status"] == "stopped" else 1


if __name__ == "__main__":
    sys.exit(main())
