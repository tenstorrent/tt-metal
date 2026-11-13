# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP server on the full ttnn ``DeepSeekV41Model``.

The V4.1 twin of ``models/experimental/deepseek_v4_flash/demo/server.py``. Chat, completions, sessions, tool
calling, thinking mode, the scheduler, the live console and traced prefill are that server's (read that
docstring), on the V4.1 decode model and :class:`DeepSeekV41PrefillModel` instead:

* the engine (:class:`V41ChatEngine`) builds :class:`DeepSeekV41Model` and its ``lm_head``; its layers are split
  evenly over the stages in contiguous runs (five per stage at the default eight), a placement the prefill model
  shares, as in ``tests/prefill/test_prefill_decode_demo.py``. The full mesh stays open;
  ``DEEPSEEK_V4_NUM_STAGES`` (default 8) is how many ``1 x 4`` submeshes are carved from it;
* the prefill (:class:`_V41Prefiller`) is :meth:`DeepSeekV41Model.build_prefill` (sharing the routed experts,
  embedding, ``lm_head``, sockets and the Engram host lookup) with no CSA position bias to commit. By default it
  is KV-only (``--prefill-kv-only``): layers 0-20 on stages 0-4, which write all of the shared compressed KV, and
  the last 128 prompt tokens are replayed through decode, which fills the sliding windows of every layer and
  gives the first token. ``--prefill-all-stages`` (on by default with a KV-only prefill; ``--no-prefill-all-stages``
  or ``DEEPSEEK_V41_PREFILL_ALL_STAGES=0`` turns it off) spreads layers 0-20 over every stage instead (three on each
  of stages 0-4, two on each of stages 5-7 at eight stages; three on each of the seven with ``--dspark``), loading a
  second copy of the routed experts of every layer it moves off its decode stage; to leave the DRAM for them,
  ``--num-users`` then defaults to 40.

``GET /models`` and ``GET /v1/models`` return DeepSeek's list-models record (:func:`model_record`).

A completion's ``usage`` is DeepSeek's (:meth:`_V41Handler._usage`): ``prompt_tokens`` split into
``prompt_cache_hit_tokens`` (already in the session's KV cache) and ``prompt_cache_miss_tokens`` (fed this turn),
``prompt_tokens_details.cached_tokens`` and ``completion_tokens_details.reasoning_tokens``. A stream carries it on
its last chunk (the one with ``finish_reason``), and with ``"stream_options": {"include_usage": true}`` every
earlier chunk has ``"usage": null``.

A turn stops, mid-prefill or mid-decode, when its client disconnects or on
``POST /v1/chat/completions/<id>/cancel`` (the ``id`` of the first streamed chunk, also the ``X-Request-Id`` header)
or ``POST /v1/sessions/<user>/cancel``, and every running turn stops on the live console's ``k`` key; see
"Stopping" in the V4 server's docstring.

What V4.1 imposes on the serving side:

* Users. ``--num-users`` sessions (40 with ``--dspark`` or ``--prefill-all-stages``, else 64) stay open at once, each with its own ring and window slot and its
  own rows of the compressed-KV page tables; their compressed KV shares pools sized for ``--total-context`` tokens (default
  ``num_users * 204800``). A request is admitted as soon as it has a slot: every pending prompt is prefilled (the traced
  prefill, then the tokens replayed through decode) before any of them generates, and a prompt that arrives while a reply
  is decoding waits only for the traces already in the dispatch queue (with ``--dspark``, for the verify round in
  progress). ``--max-running`` (8, one per stage) then bounds how many of those turns decode at
  once. While a prompt waits, its Engram rows are looked up
  on the host, so the prefill does not gather them. A step decodes one user, and switching users costs no device work: the step's packet
  names the user's slot. The scheduler interleaves the decoding users' steps through the stages, so up to one
  decoding user per stage (``DEEPSEEK_V4_NUM_STAGES``, default eight) keeps every stage busy; seating and posting,
  trace dispatch, the H2D packet writes and the D2H reads each run on their own thread (see :class:`V41ChatEngine`).
  A user decoding alone keeps ``--decode-ahead`` (32) steps posted, as the decode demo does; a reply that stops
  early feeds the steps still posted a rerun of its last one (:meth:`V41ChatEngine.repeat_traced`).
* Speculative decode (``--dspark``, on by default; ``--no-dspark`` or ``DEEPSEEK_V41_DSPARK=0`` turns it off). The DSpark drafter takes the
  mesh's last row, so decode runs on the first seven, and a reply decodes in verify rounds
  (:class:`_SpecScheduler`, :class:`~..tt.spec.DSparkSpeculator`): ~81 tok/s at three drafts against ~56 a step at a
  time, for a sampled (temperature 1) chat reply in ``tests/decode/test_dspark_spec.py``. Every session slot has its
  own rows of the drafter's rings, as of the decode rings. The drafts a round checks follow the load
  (``--dspark-drafts``, default 3 / 2 / 1 / 0 for 1 / 2 / 3 / 4+ seated decoding users): each user's draft takes the
  drafter's time whatever its length, so under load the replies decode a step at a time, sharing the pipeline as
  without ``--dspark``. Its rounds accept against the steps' logits on the host, and its
  drafter has no image rows, so ``--dspark`` turns off ``--vision`` and
  ``--device-sampling``; a request's ``temperature`` and ``top_p`` apply. The rounds run on the Rust host runtime
  as single steps do (its threads replay the stages' traces, write the packets and Engram rows and read the
  outputs), and the web monitor watches them.
* Shared prefixes. A prompt that continues no conversation starts from the longest prefix of it any other slot's
  cache holds, running or parked (a common system prompt, a forked conversation): the two slots share that prefix's
  compressed KV blocks, counted per slot so either can rewind or reset without the other losing them, and the
  prefill model rebuilds the new slot's sliding window from them by replaying the 128 tokens before the prefix's end
  (``DeepSeekV41Model.attach_session_prefix``, ``_Scheduler._share_prefix`` in the V4 server). The prefix is matched
  on the tokens the other slot was fed, in multiples of 128, up to the first image token, and is counted as
  ``prompt_cache_hit_tokens``.
* Slot reuse. A request that continues no slot's conversation, but whose tokens differ from a parked slot's cache
  only in the cache's last 512 (``_SLOT_REUSE_TOKENS``) -- a trace replay that sends its own recording of the last
  reply rather than the server's, an edited or regenerated last turn -- takes that slot instead of a new one: the
  cache is rewound in place to the prefill model's grid at or before the first differing token, the compressed KV
  before it kept and the sliding window rebuilt by a bounded replay, and the tokens past it are overwritten
  (:class:`_V41SlotPool`, :class:`_SlotReuse`).
* Context. ``--max-context`` is 1048576 tokens by default (prompt + reply), the model's full context. The pools
  hold ``--total-context`` tokens, by default 204800 per user (64 x 204800 is what fits in device DRAM), and are
  paged, so any one user can grow to the full context while the others leave blocks free; a user that needs a
  block when the pools are full is refused. Past
  ``candidate_topk_blocks * candidate_block_size`` (16384) compressed entries the select is two-level, in the
  traced prefill and in decode, so a prompt prefills up to ``--prefill-max-len`` (by default the whole context, at
  most 131072 tokens, see :class:`_V41Prefiller`) and decode replays the rest.
* Engram layers 1 and 14 read host n-gram tables (~101 GB each). Before the mesh is opened they are staged in
  tmpfs (``STAGE_DIR``, ``DEEPSEEK_V41_ENGRAM_STAGE_DIR``, default ``/dev/shm/deepseek_v41_engram``) and mapped
  from there, so one copy serves every run and process instead of each reading the checkpoint into RAM; the
  model then refuses to build only if even the staged copy does not fit. ``--no-engram-staging`` (or
  ``DEEPSEEK_V41_ENGRAM_STAGE=0``) keeps the old behaviour, and a startup with too little RAM names
  ``tools/stage_engram_tables.py``, which stages ahead of time (and ``--remove`` drops the copies).
* Images (``--vision``, on by default). A request's messages may carry up to 256 distinct images, as long as their rows
  fit ``--image-rows`` together (up to 1024 an image, fewer for a small one): an uploaded file (a
  ``file`` part with a ``file_id``) or inline base64 (a ``file`` part's ``file_data`` or an ``image_url``, holding a
  ``data:`` URL, or an Anthropic ``image`` with a base64 ``source``); a URL or path is refused.
  The vision tower runs on stage ``DEEPSEEK_V41_VISION_STAGE`` (5), with no step in flight, and writes each image's
  rows into ``--image-rows`` rows past the vocabulary of the embedding table, where its span's token ids point (see
  :class:`_ImageBook`); prefill and decode then read them like any token's, and the router picks the image tokens'
  experts with ``bias_vl``. Speculative decode does not take images. The whole image path is traced: the tower (one
  trace per ViT bucket) and the row write (one trace on stage 0) are captured at the end of startup, after every
  other allocation, and an image only replays them, its inputs streamed in and its rows read back over host sockets
  the model opens before its decode rings (:meth:`V41ChatEngine.capture_vision`, :class:`~..tt.vision.ImageSockets`).
* Files, as `DeepSeek's Files API <https://api-docs.deepseek.com/api/create-file>`_ (also under ``/v1``):
  ``POST /files`` (multipart ``file``, ``purpose=user_data``, optional ``expires_after[anchor]`` /
  ``expires_after[seconds]``), ``GET /files`` (``after``, ``limit``, ``order``), ``GET /files/<id>`` and
  ``DELETE /files/<id>``. An upload's image is encoded on the vision stage before the upload returns
  (:meth:`V41GenerationServer.upload_file`), so the chat request that references it only reads its rows; one
  whose rows were evicted meanwhile is written again at that request's admission. With ``--cache-dir`` both outlive
  a restart, beside the weight cache (``<cache-dir>/<model>/``): the uploads in ``files/``, and every image's
  vision rows -- which depend on the image alone, not on the prompt -- in ``vision_rows/<sha1>.pt``, so an image
  seen before (uploaded or inline) skips the vision tower. Without it, files live in host RAM only.

Run it (ttnn venv, from the repo root; 32-chip Galaxy for the prefill model)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    python models/experimental/deepseek_v41_flash/demo/server.py --host 0.0.0.0 --port 8000 --debug

Then talk to it as to the V4 server (``models/experimental/deepseek_v4_flash/demo/client.py``)::

    curl -s --no-buffer -X POST http://127.0.0.1:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'

    curl -s http://127.0.0.1:8000/models

    curl -s http://127.0.0.1:8000/files -F file=@cat.jpg -F purpose=user_data    # -> {"id": "file-api-...", ...}
    curl -s -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \\
        -d '{"messages": [{"role": "user", "content": [{"type": "file", "file_id": "file-api-..."},
                                                       {"type": "text", "text": "What is this?"}]}]}'

With ``--no-rust-host`` (or ``DEEPSEEK_V41_RUST_HOST=0``) the decode step's host work runs on Python threads (the
live console stays). By default it runs on the Rust host runtime (``tt/host_rs.py``), which the server builds at
startup (``host_rs/build.sh``, a no-op when up to date) and refuses to start without, and the live console gives way to a web monitor in a process of its own, so watching costs the server nothing but a raw
snapshot a second (``DEEPSEEK_V41_MONITOR_PERIOD``) and the counters the runtime keeps anyway (see
``demo/monitor.py``). The server starts it on ``--monitor-port`` (8090; 0 starts none) unless one of the user's is
already running, and leaves it running when it exits; open http://<host>:8090. By hand::

    python models/experimental/deepseek_v41_flash/demo/monitor.py --port 8090
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import email.policy
import hashlib
import itertools
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from email.parser import BytesHeaderParser
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

import torch
from loguru import logger

from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _PAGE_BLOCK_SIZE,
    _WEIGHT_DTYPE,
    _build_rope,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _TP_SIZE,
)
from models.experimental.deepseek_v41_flash import encoding_dsv41
from models.experimental.deepseek_v41_flash.demo import monitor
from models.experimental.deepseek_v41_flash.tt import host_rs
from models.experimental.deepseek_v41_flash.tt.config import (
    DEFAULT_MODEL_DIR,
    VISION_STAGE,
    dequantized,
    load_config,
    load_vision_config,
    pipeline_stage_count,
)
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    ENGRAM_RAM_MARGIN,
    STAGE_DIR,
    engram_ram_needed,
    host_mem_available,
    stage_engram_tables,
)
from models.experimental.deepseek_v41_flash.tt.lm_head import DeepSeekV41LMHead
from models.experimental.deepseek_v41_flash.tt.model import DeepSeekV41Model
from models.experimental.deepseek_v41_flash.tt.vision import (
    DELIMITER_ROWS,
    FIRST_IMAGE_ROW,
    IMAGE,
    DeepSeekV41Vision,
    EmbeddingRowWriter,
    image_processor,
    span_ids,
    write_embedding_rows,
)
from models.experimental.deepseek_v4_flash.demo.chat_cli import ChatEngine, ChatFormat, UserSession
from models.experimental.deepseek_v4_flash.demo.server import (
    DECODE_RATE_WINDOW,
    GenerationServer,
    RequestError,
    _content_text,
    _continues,
    _conversation_key,
    _Handler,
    _HostConversation,
    _Lease,
    _normalized_messages,
    _Prefiller,
    _PromptStopped,
    _Scheduler,
    _SlotPool,
    _text_content,
    _Turn,
    build_parser,
    check_args,
    serve,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

_NUM_USERS = 256  # default --num-users
# Default --num-users with --prefill-all-stages. Its expert copies take ~5.7 GB more on each chip of stages 2-4
# and ~3.8 GB on stages 1 and 5-7; stage 4 also holds the layer-20 pool, so 40 x 204800 tokens (~10.5 GB of
# its ~17 GB free) leave them room.
_NUM_USERS_ALL_STAGES = 40
# Default --num-users with --dspark: slots resident at once.
_NUM_USERS_DSPARK = 40
# Default --max-context (prompt + reply): what one user's page tables address, the model's full context.
_MAX_CONTEXT = 1048576
# Default --total-context is --num-users x this. The layer-20 pool costs ~1.25 KiB per token on each chip of
# stages 4-7, so 64 users x 1048576 tokens would not fit; 64 x 204800 take ~16 GB of the ~17 GB free. The pools
# are paged, so any one user can still grow to --max-context while the others leave the blocks free.
_KV_TOKENS_PER_USER = 204800
# Default and most --decode-ahead: the decode steps a lone decoding turn keeps posted, the window
# tests/decode/test_full_model_decode_demo.py keeps (its ``replay_ahead``).
_DECODE_AHEAD = 32
# --dspark: default --dspark-drafts, the drafts a verify round checks with 1, 2, 3, 4 or more seated decoding users
# (0: plain decode), and the decode stages, which leave the 8 x 4 mesh's last row to the drafter.
_DSPARK_DRAFTS = (3, 2, 1, 0)
_DSPARK_STAGES = 7
# The most --dspark-drafts takes: one user's rounds take ~1 ms more per draft up to 3, then stall to twice the round
# (test_dspark_spec.py's draft-count sweep: 81 tok/s at 3 drafts, 43 at 4, against 56 greedy).
_DSPARK_MAX_DRAFTS = 3
# Fewer seated users raise the drafts only once they have stayed fewer this long, so a reply ending just before the
# next one is seated does not switch speculation on and off.
_DSPARK_RAISE_SECONDS = 2.0
# A request that continues no slot's conversation reuses a parked slot whose cache it differs from only in the
# cache's last this-many tokens, overwriting them, instead of taking a new slot (_V41SlotPool, _SlotReuse).
_SLOT_REUSE_TOKENS = 512

# The checkpoint's own encoding (``encoding/encoding.py``): V4.1 renamed the DSML tool-call tags and takes no
# "medium" reasoning effort, which the V4 server accepts and V4 renders as "high".
V41_CHAT_FORMAT = ChatFormat(
    encoding_dsv41.render_message,
    encoding_dsv41.merge_tool_messages,
    encoding_dsv41.sort_tool_results_by_call_order,
    reasoning_efforts=("low", "medium", "high", "max"),
    calls_tag=encoding_dsv41.tool_calls_block_name,
    invoke_tag=encoding_dsv41.tool_call_tag_name,
    parameter_tag=encoding_dsv41.tool_parameter_tag_name,
    effort_aliases={"medium": "high"},
)

# -- images -------------------------------------------------------------------------------------- #
# An image part of a user message becomes a text block holding the checkpoint's placeholder, tagged with the image's
# sha1 (``"image"``), as ``encoding_dsv41.process_image_messages`` lays the message out. The tag rides on the block, so
# it survives ``merge_tool_messages`` and takes part in ``_conversation_key``: a continued conversation is recognised
# by its images too. Rendering tokenizes the placeholder to ``image_token_id``, which :meth:`V41GenerationServer.
# _render_ids` expands into the image's span of token ids past the vocabulary (see :mod:`..tt.vision`).
# Every cached ``all_reduce`` program keeps 4 global semaphores (16 B a bank each) in L1_SMALL, and the vision tower
# compiles one per ViT and aligner bucket on its stage (~1.2 KB), beside decode's and prefill's: 4096 B overflows.
# The prefill's semaphores also land in the holes the released prefetch pipes leave, so at 8192 their restore finds
# no contiguous 432 B. Every byte comes out of L1 too, where 12288 leaves the decode fused_experts CBs 256 B short.
_L1_SMALL_SIZE = 10240
_PREPARED_IMAGES = 256  # decoded images (their ViT patches, ~11 MB at most each) kept for admission
# Distinct images per request, across its whole history (a repeated image shares one span). Past the decoded images
# kept, a request would evict its own before its admission; its rows must also fit --image-rows (ImageBook.rows_of).
_MAX_IMAGES = _PREPARED_IMAGES
_IMAGE_PARTS = ("image", "image_url", "file")  # content part types that carry an image

# Files API (https://api-docs.deepseek.com/api/create-file): uploaded images, referenced by a chat ``file`` part's
# ``file_id``. Their bytes stay in host RAM for as long as the file exists (and on disk, see :class:`_FileStore`).
_MAX_FILE_BYTES = 64 << 20  # per upload, DeepSeek's limit
_MAX_STORED_FILE_BYTES = 4 << 30  # every live upload together
_FILE_LIFETIME = (3600, 2592000)  # bounds of expires_after[seconds]
_UPLOADS = itertools.count(1)  # names an upload in the live console until its bytes are in


def _spec_fields(spec) -> dict:
    """A turn's DSpark fields for the web monitor: its drafts per round and its reply's tokens per round so far
    while it speculates, none while it decodes a step at a time."""
    if spec is None:
        return {}
    rounds = len(spec.stats.accepted)
    return {"drafts": spec.drafts, "tokens_per_round": spec.stats.tokens / rounds if rounds else None}


def _host_cpus() -> list[int]:
    """``DEEPSEEK_V41_HOST_CPUS`` (e.g. ``"2-13"`` or ``"2,4,6"``): the CPUs the host runtime's threads are pinned
    to, in their creation order (output, packet, Engram layers, stages); unset, they float."""
    cpus = []
    for part in filter(None, os.environ.get("DEEPSEEK_V41_HOST_CPUS", "").split(",")):
        low, _, high = part.partition("-")
        cpus.extend(range(int(low), int(high or low) + 1))
    return cpus


def _image_bytes(part: dict, index: int, files: "_FileStore") -> bytes:
    """The bytes of one image part: an uploaded file (a ``file`` part's ``file_id``), or inline base64 -- a ``file``
    part's ``file_data`` or an OpenAI ``image_url``, holding a ``data:`` URL, or an Anthropic ``image`` with a base64
    ``source`` (or ``data``). A URL or a path is refused: fetching or opening it would let a request reach the
    network or the server's files."""
    kind = part.get("type")
    if kind == "image":
        source = part.get("source")
        payload = source.get("data") if isinstance(source, dict) else part.get("data")
        if not isinstance(payload, str):
            raise RequestError(400, f"messages[{index}]: an image part needs base64 data (a URL is not fetched)")
    else:
        if kind == "file":
            # DeepSeek's file part, flat or (as OpenAI's) under "file".
            fields = part["file"] if isinstance(part.get("file"), dict) else part
            file_id, url = fields.get("file_id"), fields.get("file_data")
            if (file_id is None) == (url is None):
                raise RequestError(400, f"messages[{index}]: a file part needs exactly one of file_id and file_data")
            if file_id is not None:
                file = files.get(file_id)
                if file is None:
                    raise RequestError(400, f"messages[{index}]: no file {file_id!r} (deleted or expired?)")
                return file.data
        else:
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
        if not isinstance(url, str) or not url.startswith("data:"):
            raise RequestError(
                400, f"messages[{index}]: an image must be an inline data: URL (base64) or an uploaded file_id"
            )
        header, _, payload = url.partition(",")
        if ";base64" not in header:
            raise RequestError(400, f"messages[{index}]: an image data: URL must be base64-encoded")
    try:
        return base64.b64decode(payload)
    except (binascii.Error, ValueError) as e:
        raise RequestError(400, f"messages[{index}]: malformed base64 image: {e}") from e


def _is_image(data: bytes) -> bool:
    """Whether ``data`` starts as a JPEG, PNG, GIF or WebP file does, the formats the Files API takes."""
    return (
        data.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a"))
        or data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
    )


def _multipart_fields(content_type: str, body: bytes) -> dict[str, tuple[str | None, bytes]]:
    """The fields of a ``multipart/form-data`` body, by name: ``(filename, value)``. Split on the boundary here, and
    only each part's headers handed to :mod:`email`, so a binary value comes back byte for byte."""
    parser = BytesHeaderParser(policy=email.policy.HTTP)
    header = parser.parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode("latin-1", "replace"))
    boundary = header.get_param("boundary")
    if header.get_content_type() != "multipart/form-data" or not isinstance(boundary, str) or not boundary:
        raise RequestError(400, "an upload must be multipart/form-data, with a boundary")
    fields = {}
    for section in (b"\r\n" + body).split(b"\r\n--" + boundary.encode("latin-1"))[1:]:
        if section.startswith(b"--"):
            return fields
        head, sep, value = section.partition(b"\r\n\r\n")
        if not sep:
            raise RequestError(400, "malformed multipart body: a part has no blank line after its headers")
        # The rest of the boundary line (transport padding) comes first.
        part = parser.parsebytes(head.partition(b"\r\n")[2] + b"\r\n\r\n")
        name = part.get_param("name", header="content-disposition")
        if isinstance(name, str):
            fields[name] = (part.get_filename(), value)
    raise RequestError(400, "malformed multipart body: no closing boundary")


def _files_path(path: str) -> str | None:
    """For a Files API path (``/files`` or ``/files/<id>``, also under ``/v1``) the file id, ``""`` for the
    collection; None for any other path."""
    for prefix in ("/files", "/v1/files"):
        if path == prefix:
            return ""
        if path.startswith(prefix + "/"):
            return unquote(path[len(prefix) + 1 :])
    return None


@dataclass
class _File:
    record: dict  # its file object, as the Files API returns it
    data: bytes


class _FileStore:
    """The Files API's uploads by id, in upload order (so by ``created_at``). An expired file is dropped the next
    time the store is read.

    With a ``directory`` every file is also kept there, ``<id>.bin`` (its bytes) and ``<id>.json`` (its file object,
    written last, so a file without one is a torn upload), and the store starts from what is there: uploads
    outlive a restart.
    """

    def __init__(self, directory: str | None = None):
        self.directory = directory
        self._files: dict[str, _File] = {}
        self._bytes = 0
        self._lock = threading.Lock()
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
            self._load()

    def usage(self) -> dict:
        """The files stored and their bytes, for the live console."""
        with self._lock:
            return {"count": len(self._files), "bytes": self._bytes}

    def _load(self) -> None:
        loaded = []
        for name in os.listdir(self.directory):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.directory, name)
            try:
                with open(path) as f:
                    record = json.load(f)
                with open(path[: -len(".json")] + ".bin", "rb") as f:
                    data = f.read()
            except (OSError, ValueError) as e:
                logger.warning(f"ignoring the uploaded file {path}: {e}")
                continue
            if not isinstance(record, dict) or record.get("bytes") != len(data) or f"{record.get('id')}.json" != name:
                logger.warning(f"ignoring the uploaded file {path}: its record does not match its bytes")
                continue
            loaded.append(_File(record, data))
        for file in sorted(loaded, key=lambda file: file.record["created_at"]):
            self._files[file.record["id"]] = file
            self._bytes += len(file.data)
        self._expire()
        logger.info(f"{len(self._files)} uploaded files ({self._bytes >> 20} MiB) in {self.directory}")

    def _unlink(self, file_id: str) -> None:
        if self.directory is None:
            return
        for suffix in (".json", ".bin"):  # the record first: a file without one is never loaded
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(self.directory, file_id + suffix))

    def _expire(self) -> None:
        now = time.time()
        for file_id in [i for i, f in self._files.items() if f.record.get("expires_at", now + 1) <= now]:
            self._bytes -= len(self._files.pop(file_id).data)
            self._unlink(file_id)

    def add(self, data: bytes, filename: str, lifetime: int | None) -> dict:
        created = int(time.time())
        record = {
            "id": "file-api-" + uuid.uuid4().hex,
            "object": "file",
            "bytes": len(data),
            "created_at": created,
            "filename": filename,
            "purpose": "user_data",
        }
        if lifetime is not None:
            record["expires_at"] = created + lifetime
        with self._lock:
            self._expire()
            if self._bytes + len(data) > _MAX_STORED_FILE_BYTES:
                raise RequestError(
                    400, f"the server holds {self._bytes >> 20} MiB of files already; delete some to upload more"
                )
            self._bytes += len(data)
        if self.directory is not None:
            path = os.path.join(self.directory, record["id"])
            try:
                with open(path + ".bin", "wb") as f:
                    f.write(data)
                with open(path + ".json.tmp", "w") as f:
                    json.dump(record, f)
                os.replace(path + ".json.tmp", path + ".json")
            except OSError as e:
                with self._lock:
                    self._bytes -= len(data)
                self._unlink(record["id"])
                raise RequestError(500, f"cannot save the file to {self.directory}: {e}", "server_error") from e
        with self._lock:
            self._files[record["id"]] = _File(record, data)
        return record

    def get(self, file_id: str) -> _File | None:
        with self._lock:
            self._expire()
            return self._files.get(file_id)

    def delete(self, file_id: str) -> bool:
        with self._lock:
            self._expire()
            file = self._files.pop(file_id, None)
            if file is not None:
                self._bytes -= len(file.data)
                self._unlink(file_id)
        return file is not None

    def page(self, after: str | None, limit: int, descending: bool) -> dict:
        """One page of `GET /files <https://api-docs.deepseek.com/api/list-files>`_."""
        with self._lock:
            self._expire()
            records = [file.record for file in self._files.values()]
        if descending:
            records.reverse()
        if after is not None:
            ids = [record["id"] for record in records]
            if after not in ids:
                raise RequestError(400, f"after: no file {after!r}")
            records = records[ids.index(after) + 1 :]
        data = records[:limit]
        return {
            "object": "list",
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": len(records) > limit,
        }


def _image_keys(messages) -> list[str]:
    """The sha1s of the images in ``messages``, in prompt order (the order their placeholders render in)."""
    return [
        block["image"]
        for message in messages
        for block in message.get("content_blocks") or ()
        if isinstance(block, dict) and "image" in block
    ]


@dataclass
class _Image:
    types: torch.Tensor  # image_token_types of its span
    prepared: tuple | None  # (patches, n_vit_h, n_vit_w) until encoded, unless rows are on disk
    slots: torch.Tensor | None = None  # its IMAGE rows' slots in the embedding table, once encoded
    used: float = 0.0
    rows: torch.Tensor | None = None  # its vision rows read from disk, until written into the table
    decode_seconds: float = 0.0  # host time to decode it (or read its rows from disk), for the live console


class _ImageStatus:
    """What the image path is doing, for the live console: each image in flight by a label (its key, or an upload's
    name before its bytes are in) with its stage -- ``uploading``, ``decoding``, ``reading cache``, ``queued``
    (waiting for the scheduler), ``vit``, ``embedding`` -- and the last few images done, with their timings. An entry
    lives only while its stage runs, so a request that fails part-way leaves nothing behind."""

    def __init__(self):
        self._live: dict[str, dict] = {}
        self._done: deque[dict] = deque(maxlen=3)
        self._lock = threading.Lock()

    def stage(self, label: str, stage: str, **info) -> None:
        with self._lock:
            entry = self._live.setdefault(label, {"label": label[:12]})
            entry.update(info, stage=stage, since=time.perf_counter())

    def clear(self, label: str) -> None:
        with self._lock:
            self._live.pop(label, None)

    def done(self, label: str, **info) -> None:
        with self._lock:
            self._live.pop(label, None)
            self._done.append({"label": label[:12], "at": time.time(), **info})

    def snapshot(self) -> dict:
        now = time.perf_counter()
        with self._lock:
            live = [{**e, "seconds": now - e["since"]} for e in self._live.values()]
            return {"live": live, "done": list(self._done)}


class _ImageBook:
    """Images by sha1 -> their rows in the model's extended embedding table.

    :meth:`add` (an HTTP thread) decodes an image into ViT patches; :meth:`span` (the scheduler, at admission or for
    an upload, with no step in flight) encodes it on the vision stage the first time, writes its rows into slots of the table and
    returns the span's token ids. A span's ids stay valid while its rows stay put, so an image keeps its slots while
    any conversation holds it -- a session's cached tokens, a host conversation, a queued turn -- and only images no
    conversation holds are evicted, least recently used first, to make room.

    The vision tower's rows depend on the image alone (its pixels, after the processor's resize) -- no prompt text
    reaches it -- so with a ``rows_dir`` each image's rows are kept there, ``<sha1>.pt``, beside the processor
    settings they were made with. An image found there, after an eviction or a restart, skips the decode and the
    tower and only has its rows written into the table.
    """

    def __init__(self, engine: "V41ChatEngine", snapshot_dir, rows_dir: str | None = None):
        self.engine = engine
        self.rows_dir = rows_dir
        if rows_dir is not None:
            os.makedirs(rows_dir, exist_ok=True)
        self.vocab_size = engine.config.vocab_size
        config = engine.vision.config
        self.processor = image_processor(snapshot_dir)
        self.processor_args = SimpleNamespace(
            vision_patch_size=config.patch_size,
            vision_downsample_ratio=config.downsample_ratio,
            vision_max_n_token=config.max_image_tokens,
            vision_min_pixels=config.min_pixels,
            vision_max_wh_ratio=config.max_wh_ratio,
        )
        self.free = deque(range(engine.model.image_rows - FIRST_IMAGE_ROW))
        self.total_rows = len(self.free)
        self.entries: OrderedDict[str, _Image] = OrderedDict()  # least recently used first
        self._lock = threading.Lock()
        self.status = _ImageStatus()

    def rows_of(self, keys) -> int:
        """The image rows images ``keys`` take in the table together (each held, as :meth:`add` left it)."""
        with self._lock:
            return sum(int((self.entries[key].types == IMAGE).sum()) for key in keys if key in self.entries)

    def stats(self) -> dict:
        """For the live console: the image rows, the images held (and how many have rows), what is in flight."""
        with self._lock:
            encoded = sum(entry.slots is not None for entry in self.entries.values())
            held, free = len(self.entries), len(self.free)
        return {
            "rows_free": free,
            "rows_total": self.total_rows,
            "held": held,
            "encoded": encoded,
            **self.status.snapshot(),
        }

    def add(self, data: bytes, where: str) -> str:
        """Decode an image (if not already held) and return its key. ``where`` names it in an error."""
        key = hashlib.sha1(data).hexdigest()
        with self._lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return key
        t0 = time.perf_counter()
        try:
            self.status.stage(key, "reading cache", bytes=len(data))
            saved = self._saved_rows(key)
            if saved is not None:
                entry = _Image(saved["types"], None, rows=saved["rows"])
            else:
                self.status.stage(key, "decoding")
                try:
                    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self.processor.load_image(
                        {"data": data}, self.processor_args
                    )
                except Exception as e:  # noqa: BLE001 - PIL raises a zoo of types for a bad image
                    raise RequestError(400, f"{where}: cannot read the image: {e}") from e
                entry = _Image(self.processor.image_token_types(n_llm_h, n_llm_w), (patches, n_vit_h, n_vit_w))
        finally:
            self.status.clear(key)
        entry.decode_seconds = time.perf_counter() - t0
        with self._lock:
            self.entries.setdefault(key, entry)
            self.entries.move_to_end(key)
            prepared = [k for k, e in self.entries.items() if e.slots is None]
            for stale in prepared[: max(0, len(prepared) - _PREPARED_IMAGES)]:
                del self.entries[stale]
        return key

    def span(self, key: str, pinned) -> list[int]:
        """The token ids of image ``key``'s span, its rows encoded and written first if they are not yet.
        ``pinned()`` returns the keys whose rows must stay put (consulted only to make room)."""
        with self._lock:
            entry = self.entries.get(key)
            if entry is None:
                raise RequestError(503, "an image of this request is no longer held; send the request again")
            self.entries.move_to_end(key)
            if entry.slots is not None:
                return span_ids(entry.types, entry.slots, self.vocab_size).tolist()
        n = int((entry.types == IMAGE).sum())
        if n > len(self.free):
            self._make_room(n, pinned() | {key})
        engine = self.engine
        if engine._replay_q:
            raise RuntimeError("encoding an image with decode steps in flight")
        if engine.row_writer is None:
            raise RuntimeError("the vision traces are not captured (V41ChatEngine.capture_vision)")
        try:
            with engine.model._cq_lock:
                t0 = time.perf_counter()
                if entry.rows is not None:
                    rows, source = entry.rows, "read from disk"
                else:
                    patches, n_vit_h, n_vit_w = entry.prepared
                    self.status.stage(key, "vit", rows=n, patches=f"{n_vit_h}x{n_vit_w}")
                    rows = engine.vision.encode(patches, n_vit_h, n_vit_w)
                    source = f"{n_vit_h}x{n_vit_w} patches, encoded"
                t1 = time.perf_counter()
                self.status.stage(key, "embedding", rows=n)
                with self._lock:
                    slots = torch.tensor([self.free.popleft() for _ in range(n)], dtype=torch.int64)
                try:
                    engine.row_writer.write(rows, self.vocab_size + FIRST_IMAGE_ROW + slots)
                except Exception:
                    with self._lock:
                        self.free.extend(slots.tolist())
                    raise
        finally:
            self.status.clear(key)
        write_seconds = time.perf_counter() - t1
        vit_seconds = None if entry.rows is not None else t1 - t0
        self.status.done(key, rows=n, decode=entry.decode_seconds, vit=vit_seconds, embedding=write_seconds)
        logger.info(
            f"image {key[:12]}: {n} rows, {source} in {t1 - t0:.2f}s, "
            f"written in {write_seconds:.2f}s ({len(self.free)} image rows free)"
        )
        if entry.rows is None:
            self._save_rows(key, entry.types, rows)
        with self._lock:
            entry.slots, entry.prepared, entry.rows = slots, None, None
        return span_ids(entry.types, slots, self.vocab_size).tolist()

    def _saved_rows(self, key: str) -> dict | None:
        """``{"types", "rows"}`` of image ``key`` from :attr:`rows_dir`, if saved there with these processor
        settings and of the span's shape."""
        if self.rows_dir is None:
            return None
        path = os.path.join(self.rows_dir, f"{key}.pt")
        try:
            saved = torch.load(path, weights_only=True)
        except FileNotFoundError:
            return None
        except Exception as e:  # noqa: BLE001 - a torn or foreign file is re-encoded, not fatal
            logger.warning(f"image {key[:12]}: ignoring unreadable {path}: {e}")
            return None
        if not isinstance(saved, dict) or saved.get("processor") != vars(self.processor_args):
            return None
        types, rows = saved.get("types"), saved.get("rows")
        hidden = self.engine.vision.config.text_hidden_size
        if not isinstance(rows, torch.Tensor) or tuple(rows.shape) != (int((types == IMAGE).sum()), hidden):
            logger.warning(f"image {key[:12]}: ignoring {path}, its rows do not fit its span")
            return None
        return saved

    def _save_rows(self, key: str, types: torch.Tensor, rows: torch.Tensor) -> None:
        if self.rows_dir is None:
            return
        path = os.path.join(self.rows_dir, f"{key}.pt")
        tmp = f"{path}.{os.getpid()}.tmp"  # another server may share the cache
        try:
            torch.save({"processor": vars(self.processor_args), "types": types, "rows": rows}, tmp)
            os.replace(tmp, path)
        except OSError as e:
            logger.warning(f"image {key[:12]}: cannot save its rows to {path}: {e}")

    def _make_room(self, n: int, pinned: set) -> None:
        """Free the rows of unpinned encoded images, least recently used first, until ``n`` are free."""
        with self._lock:
            for key in [k for k, e in self.entries.items() if e.slots is not None and k not in pinned]:
                if len(self.free) >= n:
                    break
                self.free.extend(self.entries.pop(key).slots.tolist())
            if len(self.free) < n:
                raise RequestError(
                    503,
                    f"the image rows are taken by the conversations in flight ({len(self.free)} free, {n} needed); "
                    "retry later or raise --image-rows",
                )


# Display name from https://api-docs.deepseek.com/api/list-models . The served ``id`` stays ``--model-id``
# (default ``deepseek-v4.1-flash``); the published catalog id is ``deepseek-flash``.
_MODEL_NAME = "DeepSeek-V4.1-Flash"
_MODEL_OWNER = "deepseek"


def model_record(api) -> dict:
    """One model object for `GET /models <https://api-docs.deepseek.com/api/list-models>`_.

    Limits are this process's, not the published 1,048,576 / 393,216 cloud numbers: ``context_window`` is
    the live prompt-plus-reply cap, and ``max_output_tokens`` is the smaller of that and ``--max-new-tokens``.
    ``input_modalities`` includes ``image`` unless the server runs ``--no-vision``.
    ``effort.supported_levels`` omits ``medium`` (accepted, then rendered as ``high``) and ``none`` (thinking off). ``system_prompt_update`` is
    ``in-history`` because a system message after the first turn is rendered; it is kept beside the leading
    one rather than replacing it. There is no Anthropic Messages route.
    """
    engine = api.engine
    aliases = engine.chat_format.effort_aliases
    levels = [level for level in engine.chat_format.reasoning_efforts if level not in aliases]
    effort = {"supported_levels": levels}
    if encoding_dsv41.DEFAULT_REASONING_EFFORT in levels:
        effort["default_level"] = encoding_dsv41.DEFAULT_REASONING_EFFORT
    return {
        "id": api.model_id,
        "object": "model",
        "created": api._created,
        "owned_by": _MODEL_OWNER,
        "name": _MODEL_NAME,
        "context_window": engine.max_seq,
        "max_output_tokens": min(engine.max_new_tokens, engine.max_seq),
        "input_modalities": ["text", "image"] if engine.images is not None else ["text"],
        "output_modalities": ["text"],
        "effort": effort,
        "api_capabilities": {"anthropic_messages": {"system_prompt_update": "in-history"}},
    }


class _V41Handler(_Handler):
    """:class:`_Handler` plus DeepSeek ``GET /models`` (``GET /v1/models`` returns the same body) and DeepSeek's
    ``usage`` (:meth:`_usage`), which a stream's last chunk carries too."""

    stream_usage = True

    def _usage(self, stats: dict) -> dict:
        """``usage`` as in `create-chat-completion <https://api-docs.deepseek.com/api/create-chat-completion>`_.

        A cache hit is a prompt token already in the slot's KV cache, i.e. the conversation a continuation (or a
        resume from an earlier turn) picks up from; the tokens the turn feeds, re-prefilled ones included, are
        misses. ``reasoning_tokens`` count the ``<think>`` block, its tags included.
        """
        hit, miss = stats["cached_tokens"], stats["prompt_tokens"]
        prompt = hit + miss
        return {
            "completion_tokens": stats["completion_tokens"],
            "prompt_tokens": prompt,
            "prompt_tokens_details": {"cached_tokens": hit},
            "prompt_cache_hit_tokens": hit,
            "prompt_cache_miss_tokens": miss,
            "total_tokens": prompt + stats["completion_tokens"],
            "completion_tokens_details": {"reasoning_tokens": stats["reasoning_tokens"]},
        }

    def do_GET(self) -> None:
        url = urlparse(self.path)
        file_id = _files_path(url.path)
        if url.path not in ("/models", "/v1/models") and file_id is None:
            super().do_GET()
            return
        try:
            if file_id is None:
                self._send_json({"object": "list", "data": [model_record(self.api)]})
            elif file_id:
                file = self.api.files.get(file_id)
                if file is None:
                    raise RequestError(404, f"no file {file_id!r}")
                self._send_json(file.record)
            else:
                query = parse_qs(url.query)
                limit = query.get("limit", ["1000"])[-1]
                if not limit.isdigit() or not 1 <= int(limit) <= 1000:
                    raise RequestError(400, "limit must be an integer in [1, 1000]")
                order = query.get("order", ["asc"])[-1]
                if order not in ("asc", "desc"):
                    raise RequestError(400, "order must be 'asc' or 'desc'")
                if query.get("purpose", ["user_data"])[-1] != "user_data":
                    raise RequestError(400, "purpose must be 'user_data'")
                self._send_json(self.api.files.page(query.get("after", [None])[-1], int(limit), order == "desc"))
        except Exception as e:
            self._send_error_json(e)

    def do_POST(self) -> None:
        if _files_path(urlparse(self.path).path) is None:
            super().do_POST()
            return
        try:
            length = self.headers.get("Content-Length")
            if not length or not length.isdigit():
                self.close_connection = True
                raise RequestError(411, "an upload needs a Content-Length")
            if int(length) > _MAX_FILE_BYTES + (1 << 20):  # the multipart framing on top of the file
                self.close_connection = True
                raise RequestError(413, f"a file may be at most {_MAX_FILE_BYTES >> 20} MiB")
            fields = _multipart_fields(self.headers.get("Content-Type", ""), self._read_upload(int(length)))
            self._send_json(self.api.upload_file(fields))
        except Exception as e:
            self._send_error_json(e)

    def _read_upload(self, length: int) -> bytes:
        """The request body, read in pieces so the live console can show the upload's progress."""
        images = self.api.engine.images
        if images is None:
            return self.rfile.read(length)
        label = f"upload {next(_UPLOADS)}"
        body = bytearray()
        try:
            while len(body) < length:
                images.status.stage(label, "uploading", received=len(body), bytes=length)
                piece = self.rfile.read(min(1 << 20, length - len(body)))
                if not piece:
                    raise RequestError(400, f"the upload ended after {len(body)} of {length} bytes")
                body += piece
        finally:
            images.status.clear(label)
        return bytes(body)

    def do_DELETE(self) -> None:
        file_id = _files_path(urlparse(self.path).path)
        if file_id is None:
            super().do_DELETE()
            return
        try:
            if not file_id:
                raise RequestError(400, "missing file id in path")
            if not self.api.files.delete(file_id):
                raise RequestError(404, f"no file {file_id!r}")
            self._send_json({"id": file_id, "object": "file", "deleted": True})
        except Exception as e:
            self._send_error_json(e)


def _shared_messages(keys: list[str], stored: list[dict]) -> int:
    """How many leading messages of a request, given by their :func:`_conversation_key`, equal ``stored``'s."""
    n = 0
    for key, old in zip(keys, stored):
        if key != _conversation_key(old):
            break
        n += 1
    return n


@dataclass
class _DeviceParkedConversation:
    """A conversation evicted from its slot/page-table while its KV cache blocks remain in device RAM."""

    owner: str | None
    messages: list[dict]
    tokens: list[int]
    marks: list[tuple]
    pending_id: str | None
    next_render: int
    thinking_mode: str | None
    reasoning_effort: str | None
    system_prompt: str | None
    snapshot: object  # DeviceParkedSnapshot
    used_at: float


class _V41SlotPool(_SlotPool):
    """:class:`_SlotPool` that, when no slot continues a request's conversation, hands it the parked slot whose cache
    the request shares the longest start with and differs from only in the cache's last :data:`_SLOT_REUSE_TOKENS`
    tokens, rather than an empty or the longest-idle slot. That is judged here from messages: the request's first
    differing message against the slot's turn marks and where its last reply started. Admission checks the tokens
    themselves and rewinds the cache in place (:class:`_SlotReuse`), or treats the slot as a new conversation's."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.near_misses: set[int] = set()  # slots handed out for an in-place rewind, until their turn's admission
        # slot -> (index of the assistant message its last turn stored, the position that reply started at)
        self.reply_at: dict[int, tuple[int, int]] = {}
        self.device_parked: list[_DeviceParkedConversation] = []

    def acquire(self, user_key: str, messages: list[dict]) -> _Lease:
        """Claim a slot for one turn, restoring from device-parked RAM or host RAM when matched."""
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
            if isinstance(restore, _DeviceParkedConversation):
                self.device_parked.remove(restore)
            elif restore is not None:
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
        return _Lease(slot, previous, restore)

    def _pick(self, messages: list[dict]):
        picked = super()._pick(messages)
        free = [i for i, busy in enumerate(self._busy) if not busy]
        # Prefer device-parked continuation over host RAM (or on tie with an empty/older parked slot).
        best_dev, best_dev_len = None, 0
        if free and self.device_parked:
            for conv in self.device_parked:
                if len(conv.messages) > best_dev_len and _continues(messages, conv.messages):
                    best_dev, best_dev_len = conv, len(conv.messages)
        if picked is not None:
            slot, restore = picked
            self.near_misses.discard(slot)
            if _continues(messages, self.engine.users[slot].messages):
                parked_len = len(self.engine.users[slot].messages)
                if best_dev is not None and best_dev_len > parked_len:
                    return slot, best_dev
                return picked
            if best_dev is not None:
                host_len = len(restore.messages) if restore is not None else 0
                if best_dev_len >= host_len:
                    empty = next((i for i in free if not self.engine.users[i].pos), None)
                    target = empty if empty is not None else slot
                    return target, best_dev
            if restore is not None:
                return picked
        elif free and best_dev is not None:
            empty = next((i for i in free if not self.engine.users[i].pos), None)
            if empty is not None or self.engine.tokens_left() > 0:
                target = empty if empty is not None else min(free, key=lambda i: self._used_at[i])
                return target, best_dev
        slot = self._near_miss(messages)
        if slot is not None:
            self.near_misses.add(slot)
            return slot, None
        if picked is not None:
            return picked
        # When all slots are full but the page table/pool still has space, assign the oldest parked slot
        # and keep the old user's KV cache resident in device RAM (device_parked).
        if free and self.engine.tokens_left() > 0:
            oldest = min(free, key=lambda i: self._used_at[i])
            return oldest, None
        return None

    def drop_host(self, user_key: str) -> int:
        dropped = super().drop_host(user_key)
        with self._cv:
            kept = []
            evicted = []
            for conv in self.device_parked:
                if conv.owner == user_key:
                    evicted.append(conv)
                else:
                    kept.append(conv)
            self.device_parked = kept
        for conv in evicted:
            self.engine.model.free_device_park(conv.snapshot)
        return dropped + len(evicted)

    def rows(self) -> list[dict]:
        rows = super().rows()
        with self._cv:
            parked = list(self.device_parked)
        for conv in sorted(parked, key=lambda c: c.used_at, reverse=True):
            rows.append(
                {
                    "id": conv.owner or "",
                    "index": None,
                    "tier": "parked",
                    "tokens": conv.snapshot.pos,
                    "max_context": self.engine.max_seq,
                    "messages": len(conv.messages),
                    "thinking": conv.thinking_mode == "thinking",
                    "busy": False,
                    "device_parked": True,
                }
            )
        return rows

    def _near_miss(self, messages: list[dict]) -> int | None:
        keys = [_conversation_key(m) for m in messages]
        best, best_start = None, 0
        for i, busy in enumerate(self._busy):
            user = self.engine.users[i]
            if busy or not user.pos or len(user.tokens) != user.pos:
                continue
            shared = _shared_messages(keys, user.messages)
            reply = self.reply_at.get(i)
            if reply is not None and reply[0] == shared == len(user.messages) - 1 and reply[1] <= user.pos:
                start = reply[1]  # only the last reply differs: the cache agrees up to where it started
            else:
                start = max((pos for n, pos, _ in user.marks if n <= shared), default=0)
            if start > best_start and user.pos - start <= _SLOT_REUSE_TOKENS:
                best, best_start = i, start
        return best


@dataclass
class _Rewind:
    """An in-place rewind of a parked slot's cache for the turn being admitted (:meth:`_SlotReuse._reuse_plan`)."""

    shared: int  # leading tokens the request and the cache agree on
    resume_at: int  # the cache is cut back to here: on the prefill model's grid, a window past its start
    ids: list[int]  # the request's tokens from resume_at on
    device_prefill: int  # ... how many of them the prefill model takes
    messages: int  # leading messages the request and the slot's conversation agree on


class _SlotReuse:
    """Mixin for a :class:`_Scheduler` served by :class:`_V41SlotPool`: a turn handed a slot whose cache it differs
    from only in the cache's last :data:`_SLOT_REUSE_TOKENS` tokens overwrites them in place. The cache is cut back
    (``truncate_session``) to the prefill model's grid at or before the first differing token, the sliding window
    rebuilt by replaying the ``replay_window`` cached tokens before the cut (SWA bounded replay), and the rest of the
    request prefilled from there, as :meth:`_Scheduler._resume` does from a turn mark. The slot's old conversation is
    not offloaded to host RAM: the request supersedes it. A slot whose tokens differ by more is admitted as the new
    conversation it is."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reuse: dict[int, _Rewind] = {}  # slot -> the rewind its turn being admitted takes

    def _admit(self, turn: _Turn) -> None:
        pool = self.server.pool
        with pool._cv:
            near_miss = turn.slot in pool.near_misses
            pool.near_misses.discard(turn.slot)
        if near_miss and not turn.cancelled.is_set() and turn.restore is None and turn.slot not in self._cut_short:
            plan = self._reuse_plan(self.engine.users[turn.slot], turn)
            if plan is not None:
                self._reuse[turn.slot] = plan
        try:
            super()._admit(turn)
        finally:
            self._reuse.pop(turn.slot, None)

    def _reuse_plan(self, user: UserSession, turn: _Turn) -> _Rewind | None:
        prefiller = self.server.prefiller
        if prefiller is None or not prefiller.replay_window or not prefiller.continue_alignment:
            return None
        messages = turn.body["messages"]
        if not user.pos or len(user.tokens) != user.pos or _continues(messages, user.messages):
            return None
        # The request rendered whole, as a new conversation in this slot would be.
        stored = (user.messages, user._next_render, user.pending_id, user.thinking_mode, user.reasoning_effort)
        try:
            user.messages, user._next_render, user.pending_id = list(messages), 0, None
            user.thinking_mode, user.reasoning_effort = self.server.thinking_mode, self.server.reasoning_effort
            self.server._apply_options(user, turn.body)
            ids = self.server._render_ids(user, include_assistant=True)
        except Exception as e:  # noqa: BLE001 - admission proper reports it to the client
            logger.debug(f"user {turn.user_key!r} (slot {turn.slot}): not rewinding in place: {type(e).__name__}: {e}")
            return None
        finally:
            user.messages, user._next_render, user.pending_id, user.thinking_mode, user.reasoning_effort = stored
        # Image rows are matched by their ids, which another image may take later: the shared start ends before one.
        vocab = self.engine.model.config.vocab_size
        shared = next(
            (i for i, (new, old) in enumerate(zip(ids, user.tokens)) if new != old or new >= vocab),
            min(len(ids), user.pos),
        )
        if user.pos - shared > _SLOT_REUSE_TOKENS:
            logger.debug(
                f"user {turn.user_key!r} (slot {turn.slot}): the request differs from the slot's last "
                f"{user.pos - shared} cached tokens, more than {_SLOT_REUSE_TOKENS}: a new conversation"
            )
            return None
        window, step = prefiller.replay_window, prefiller.continue_alignment
        start = (shared - window) // step * step  # the replay starts on the prefill's grid
        while start > 0:
            n = prefiller.aligned(len(ids) - start, start, bounded=True)
            if n:
                return _Rewind(
                    shared,
                    start + window,
                    ids[start + window :],
                    n - window,
                    _shared_messages([_conversation_key(m) for m in messages], user.messages),
                )
            start -= step
        return None

    def _offload(self, slot: int, owner: str | None) -> None:
        if slot in self._reuse:
            return
        pool, user = self.server.pool, self.engine.users[slot]
        if not user.pos or len(user.tokens) != user.pos or slot in self._cut_short:
            super()._offload(slot, owner)
            return
        try:
            snapshot = self.engine.model.evict_session_to_device_park(user.sid)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"user {owner!r} (slot {slot}): could not park on device, offloading to host: {e}")
            super()._offload(slot, owner)
            return
        with pool._cv:
            pool.device_parked.append(
                _DeviceParkedConversation(
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
                    used_at=pool._used_at[slot],
                )
            )
        logger.info(
            f"user {owner!r} (slot {slot}): {user.pos}-token conversation evicted from page table "
            f"(KV cache kept in device RAM; {len(pool.device_parked)} device-parked)"
        )

    def _restore(self, user: UserSession, turn: _Turn) -> bool:
        if isinstance(turn.restore, _DeviceParkedConversation):
            conv, turn.restore = turn.restore, None
            if self.engine.model._session_pos.get(user.sid):
                self.engine.model.reset_session(user.sid)
            self.engine.model.reload_session_from_device_park(user.sid, conv.snapshot)
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
                f"user {turn.user_key!r} (slot {turn.slot}): reloaded {user.pos}-token conversation from device RAM"
            )
            return True
        return super()._restore(user, turn)

    def _make_room(self, n_tokens: int, keep: int) -> None:
        """Evict the user with the oldest last request across device-parked conversations and parked slots
        to host RAM (freeing its device KV blocks) until ``n_tokens`` fit the page table/pool."""
        pool = self.server.pool
        while n_tokens > self.engine.tokens_left():
            with pool._cv:
                oldest_dev = min(pool.device_parked, key=lambda c: c.used_at, default=None)
                parked_slots = [
                    i
                    for i, u in enumerate(self.engine.users)
                    if not pool._busy[i] and i != keep and u.pos and not pool._closing
                ]
                oldest_slot = min(parked_slots, key=lambda i: pool._used_at[i], default=None)
                if oldest_dev is None and oldest_slot is None:
                    return
                evict_dev = oldest_dev is not None and (
                    oldest_slot is None or oldest_dev.used_at <= pool._used_at[oldest_slot]
                )
                if evict_dev:
                    pool.device_parked.remove(oldest_dev)
                else:
                    pool._busy[oldest_slot] = True
                    slot_owner = pool.owner[oldest_slot]
            if evict_dev:
                if pool.host_users:
                    try:
                        snapshot = self.engine.model.export_device_park(oldest_dev.snapshot)
                        pool.store(
                            _HostConversation(
                                owner=oldest_dev.owner,
                                messages=list(oldest_dev.messages),
                                tokens=list(oldest_dev.tokens),
                                marks=list(oldest_dev.marks),
                                pending_id=oldest_dev.pending_id,
                                next_render=oldest_dev.next_render,
                                thinking_mode=oldest_dev.thinking_mode,
                                reasoning_effort=oldest_dev.reasoning_effort,
                                system_prompt=oldest_dev.system_prompt,
                                snapshot=snapshot,
                                stored_at=oldest_dev.used_at,
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"user {oldest_dev.owner!r}: could not export device-parked to host: {e}")
                        self.engine.model.free_device_park(oldest_dev.snapshot)
                else:
                    self.engine.model.free_device_park(oldest_dev.snapshot)
                logger.debug(
                    f"freed device-parked user {oldest_dev.owner!r} for {n_tokens} tokens: {self._pool_summary()}"
                )
            else:
                try:
                    super()._offload(oldest_slot, slot_owner)
                    self.engine.users[oldest_slot].reset()
                    self._cut_short.discard(oldest_slot)
                finally:
                    pool.release(oldest_slot, vacated=True)
                logger.debug(
                    f"freed parked slot {oldest_slot} (user {slot_owner!r}) for {n_tokens} tokens: {self._pool_summary()}"
                )

    def _resume(self, user: UserSession, turn: _Turn) -> bool:
        plan = self._reuse.get(turn.slot)
        if plan is None:
            return super()._resume(user, turn)
        window = self.server.prefiller.replay_window
        logger.info(
            f"user {turn.user_key!r} (slot {turn.slot}): reusing the slot, the request differs from its last "
            f"{user.pos - plan.shared} of {user.pos} cached tokens: rewinding the cache to {plan.resume_at} in "
            f"place ({window} cached tokens replayed for the window, {plan.device_prefill} by the prefill model, "
            f"{len(plan.ids) - plan.device_prefill} through decode)"
        )
        turn.resumed_from = plan.resume_at  # a failure from here resets the session
        self.engine.model.truncate_session(user.sid, plan.resume_at)
        turn.replay_ids = user.tokens[plan.resume_at - window : plan.resume_at]
        del user.tokens[plan.resume_at :]
        user.pos = plan.resume_at
        user.pending_id = None
        user.marks = [m for m in user.marks if m[1] <= plan.resume_at and m[0] <= plan.messages]
        user.messages = list(turn.body["messages"])
        turn.ids = plan.ids
        turn.device_prefill = plan.device_prefill
        turn.prefill_from = 0
        return True

    def _finish(self, turn: _Turn) -> None:
        reply_at = turn.cached + len(turn.ids)
        super()._finish(turn)
        user = self.engine.users[turn.slot]
        if user.messages and user.messages[-1].get("role") == "assistant":
            self.server.pool.reply_at[turn.slot] = (len(user.messages) - 1, reply_at)


class _EngramPrefetch:
    """One prompt's Engram rows, gathered while the turn waits, against a copy of its session history.

    :meth:`consume` is what the traced prefill's chunk feed calls. A hit still hashes the chunk into the live
    session (so decode's history stays in step) and skips the table gather, which is the slow part of the lookup.
    """

    def __init__(self, lookup, slot: int):
        self.lookup = lookup
        self.slot = slot
        self._cv = threading.Condition()
        self._rows: dict[tuple, dict] = {}
        self._done = False
        self.error: Exception | None = None

    def consume(self, chunk_ids: torch.Tensor, start: int) -> dict | None:
        """This chunk's prefetched rows ``{layer: [1, t, kin]}``.

        Waits until the lookup has produced this chunk (or finished without it). ``None`` when it failed or
        this chunk was not part of the prompt: the caller looks the chunk up itself then."""
        ids = tuple(int(token) for token in chunk_ids.reshape(-1).tolist())
        key = (int(start), ids)
        with self._cv:
            while key not in self._rows and not self._done:
                self._cv.wait()
            if self.error is not None:
                return None
            found = self._rows.get(key)
        if found is None:
            return None
        # The rows were gathered against a copy. Extend the live session the way the lookup would have.
        self.lookup.hash_session(self.slot, chunk_ids.reshape(1, -1), int(start))
        return found

    def run(
        self,
        history: torch.Tensor,
        hashed_ids: list[int],
        hash_start: int,
        prefill_ids: list[int],
        prefill_start: int,
        chunk: int,
    ) -> None:
        """Gather ``hashed_ids`` at ``hash_start`` (lead tokens included, so their n-grams are in the history) and
        keep the slices the traced prefill will ask for, ``prefill_ids`` at ``prefill_start``."""
        t0 = time.perf_counter()
        try:
            ids_t = torch.tensor(hashed_ids, dtype=torch.long).view(1, -1).expand(history.shape[0], -1).contiguous()
            rows = self.lookup.lookup_history(history, ids_t, hash_start)
            skip = prefill_start - hash_start
            cached = {}
            for offset in range(0, len(prefill_ids), chunk):
                piece = tuple(prefill_ids[offset : offset + chunk])
                src = skip + offset
                cached[(prefill_start + offset, piece)] = {
                    layer: tensor[:, src : src + len(piece)].contiguous() for layer, tensor in rows.items()
                }
            with self._cv:
                self._rows = cached
                self._cv.notify_all()
            logger.info(
                f"engram prefetch (slot {self.slot}): {len(prefill_ids)} tokens in {time.perf_counter() - t0:.2f}s, "
                f"{len(cached)} chunk(s)"
            )
        except Exception as exc:  # the prefill looks the chunk up itself when this missed
            self.error = exc
            logger.warning(f"engram prefetch (slot {self.slot}) failed: {exc}")
        finally:
            with self._cv:
                self._done = True
                self._cv.notify_all()


class _PrefillFirst(_Scheduler):
    """Finish every pending prompt before decoding, and look Engram rows up while a prompt waits.

    A turn is admitted as soon as it holds a KV slot (:meth:`V41GenerationServer._await_seat`), not when a decode
    seat is free. While any turn is still prefilling, or one is waiting to be admitted, decode does not start
    another step: traces already posted are fed (they cannot be cancelled mid-dispatch) and then every prompt
    runs, the traced prefill one prompt at a time and the tokens replayed through decode pipelined. After that,
    the first ``--max-running`` turns in admission order decode together; the rest stay admitted, prompts already
    in the cache, until a seat frees.
    """

    def _admit(self, turn: _Turn) -> None:
        super()._admit(turn)
        if turn in self._active:
            self._prefetch_engram(turn)

    def _prefill_waiting(self) -> bool:
        """A prompt is still to be prefilled, or one has been submitted and not yet admitted.

        ``_new`` is appended under the scheduler lock and read here the way :meth:`_Scheduler._post_ahead` reads
        it: a request that arrived during this round."""
        if self._new:
            return True
        return any(turn.phase == _Turn.PREFILL and turn.error is None for turn in self._active)

    def _holds_decode_seat(self, turn: _Turn) -> bool:
        """Whether ``turn`` is one of the first ``--max-running`` decoding turns. A prompt does not take a seat."""
        if turn.phase != _Turn.DECODE:
            return True
        seats = self.server.pool.max_running
        held = 0
        for other in self._active:
            if other.phase != _Turn.DECODE or other.cancelled.is_set():
                continue
            if other is turn:
                return True
            held += 1
            if held >= seats:
                return False
        return True

    def _prefetch_engram(self, turn: _Turn) -> None:
        """Start the host Engram lookup for the tokens the traced prefill will see, on a copy of the session."""
        prefiller = self.server.prefiller
        lookup = getattr(self.engine.model, "_engram_lookup", None)
        if prefiller is None or lookup is None or not (turn.device_prefill or turn.replay_ids):
            return
        user = self.engine.users[turn.slot]
        slot = self.engine.model._slot_of.get(user.sid)
        if slot is None:
            return
        replay = list(turn.replay_ids)
        lead = list(turn.ids[: turn.prefill_from])
        body = list(turn.ids[turn.prefill_from : turn.prefill_from + turn.device_prefill])
        # Lead tokens are replayed through decode before the traced prefill, so they are not in the prefill's
        # chunks, but its n-grams read them. Hash them into the copy ahead of the prefill tokens. A window replay
        # and a lead together do not occur; the prefill looks those up itself.
        if replay and lead:
            return
        prefill_ids = replay + body
        if not prefill_ids:
            return
        prefill_start = user.pos + len(lead) - len(replay)
        if prefill_start < 0:
            return
        if lead:
            hashed_ids, hash_start = lead + body, user.pos
        else:
            hashed_ids, hash_start = prefill_ids, prefill_start
        try:
            history = lookup.export_history(slot, hash_start)
        except Exception as exc:  # the prefill looks the chunks up itself
            logger.warning(f"user {turn.user_key!r}: engram prefetch skipped: {exc}")
            return
        traced = getattr(prefiller.model, "_traced", None)
        chunk = (getattr(traced, "_chunk_size", None) if traced is not None else None) or prefiller.chunk_size
        job = _EngramPrefetch(lookup, slot)
        turn.engram_prefetch = job
        threading.Thread(
            target=job.run,
            args=(history, hashed_ids, hash_start, prefill_ids, prefill_start, chunk),
            name=f"engram-prefetch-{slot}",
            daemon=True,
        ).start()

    def _prefill_on_device(self, turn: _Turn, drain_seconds: float = 0.0, drained: int = 0) -> None:
        traced = getattr(getattr(self.server.prefiller, "model", None), "_traced", None)
        prefetch = getattr(turn, "engram_prefetch", None)
        if traced is not None:
            traced._engram_prefetch = prefetch
        try:
            super()._prefill_on_device(turn, drain_seconds, drained)
        finally:
            if traced is not None:
                traced._engram_prefetch = None
            turn.engram_prefetch = None

    def _drain_decode_queue(self) -> bool:
        """Feed the decode steps whose traces are already posted, and do not post any more.

        Returns whether the pipeline is still usable (``False`` once a readback has aborted every turn)."""
        if not self._posted and not self._inflight:
            return True
        logger.info(
            f"prefill waiting: finishing {len(self._posted)} posted decode step(s) "
            f"({len(self._inflight)} in flight) before prefilling"
        )
        return self._settle()

    def _prefill_only_round(self) -> None:
        """One round of prompt work only: traced prefills and prompt tokens replayed through decode."""
        self._stalled = False
        for turn in list(self._active):
            if turn not in self._active or turn.phase == _Turn.DECODE:
                continue
            if turn.error is not None:
                if not turn.pending:
                    self._fail(turn, turn.error)
                continue
            try:
                if (
                    (turn.device_prefill or turn.replay_ids)
                    and turn.fed == turn.prefill_from
                    and not turn.cancelled.is_set()
                ):
                    t_drain = time.perf_counter()
                    drained = len(self._inflight) + len(self.engine._replay_q)
                    if not self._settle():
                        return
                    self._prefill_on_device(turn, time.perf_counter() - t_drain, drained)
                # A prefill that consumed the whole prompt is now decoding; it waits until every other
                # prompt has finished too.
                steps = [] if turn.phase == _Turn.DECODE else self._plan(turn)
            except _PromptStopped:
                self._stop_prompt(turn)
                continue
            except Exception as exc:  # one bad turn must not stop the others
                if turn.pending:
                    turn.error = exc
                else:
                    self._fail(turn, exc)
                continue
            if steps:
                self._dispatch(turn, steps)
        if self._inflight:
            try:
                self._collect(block=True)
            except Exception as exc:
                self._abort_all(exc)

    def _round_turns(self) -> None:
        if self._prefill_waiting():
            if self._drain_decode_queue():
                self._prefill_only_round()
            return
        waiting = [
            turn
            for turn in self._active
            if turn.phase == _Turn.DECODE and not turn.cancelled.is_set() and not self._holds_decode_seat(turn)
        ]
        if not waiting:
            super()._round_turns()
            return
        # Keep turns whose prompt is in but which have no decode seat out of this round, so a posted step of a
        # seated turn is not stalled on them and lookahead follows the seated turns only.
        self._active = [turn for turn in self._active if turn not in waiting]
        try:
            super()._round_turns()
        finally:
            self._active.extend(turn for turn in waiting if turn not in self._active)


class _V41Scheduler(_SlotReuse, _PrefillFirst):
    """:class:`_PrefillFirst` with :class:`_SlotReuse`'s in-place rewinds."""


class _SpecScheduler(_SlotReuse, _PrefillFirst):
    """:class:`_PrefillFirst` (with :class:`_SlotReuse`'s in-place rewinds) that decodes each reply by DSpark speculative decode (``--dspark``).

    A turn's prompt goes in as without it (the traced prefill, then decode steps, each of which also writes its
    row into its session's rows of the drafter's rings). Its reply then runs on this thread in verify rounds
    (:class:`~..tt.spec.DSparkSpeculator`, at the request's ``temperature`` and ``top_p``), one round per scheduler
    round, and the session is handed back as plain decode would leave it.

    The drafts a round checks follow the load (:meth:`_pick_drafts`, ``--dspark-drafts``): every user's draft takes
    the drafter's time whatever its length, so the more users decode, the fewer drafts pay, and with none every
    reply decodes a step at a time as without ``--dspark``. A speculator whose length changes is finished and
    restarted at the new one (a fixup and one draft).
    With the host runtime and two or more seated replies ready, each scheduler round runs one round of every one
    of them as a batch (:meth:`_spec_batch`): every verify dispatched, then each collected in turn, so the
    pipeline is kept busy while the host accepts and the drafter drafts. Otherwise the seated replies take their
    rounds one after another. Either way each speculator keeps its session's drafts and rejected rows between
    rounds (:meth:`~..tt.spec.DSparkSpeculator.pause`; each session's fixup restores and snapshots its own ring
    rows). A prompt submitted during a reply waits for the round in progress, and that prompt (and any others
    waiting) is prefilled before decode resumes. The last tokens before the end of the context, where a round
    would run past it, are decoded a step at a time.

    At most one speculator is live, the one whose lone round ran last: its next round's rows may be in the
    pipeline. The others are paused. Another session's round pauses it, and anything else on the device (a plain
    step, a prefill, a chore: everything that follows :meth:`_settle`) waits for every speculator to finish first.

    Every verify round adds its tokens to :attr:`steps` and the rate window as it ends, so the status pane's tok/s
    and tok/s/user follow the replies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec_accounted = 0  # steps of this round the verify rounds have already put in the rate window
        self._specs: dict = {}  # turn -> its reply's DSparkSpeculator, kept between its rounds
        self._spec_live: _Turn | None = None
        self._batched: set = set()  # the turns whose verify round this scheduler round ran in a batch
        # Drafts per verify round this scheduler round (_pick_drafts; 0: plain decode), and since when fewer seated
        # users have asked for more.
        self._drafts = self.engine.dspark_drafts[0]
        self._raise_since: float | None = None

    def _round(self) -> None:
        self.rounds += 1
        before = self.steps
        self._spec_accounted = 0
        self._round_turns()
        self._account(self.steps - before - self._spec_accounted)
        self._log_pool_change()

    def _round_turns(self) -> None:
        # A prompt's decode steps follow no verify round.
        if self._specs and self._prefill_waiting() and not self._end_specs():
            return
        self._batched = set()
        self._pick_drafts()
        if self._drafts and not self._prefill_waiting() and self.engine.model.host_runtime_running:
            ready = [turn for turn in self._active if self._holds_decode_seat(turn) and self._spec_budget(turn) > 0]
            if len(ready) > 1:
                self._batched = set(ready)
                self._spec_batch(ready)
        super()._round_turns()

    def _settle(self) -> bool:
        if self._specs and not self._end_specs():
            return False
        return super()._settle()

    def _post_ahead(self) -> None:
        # Verify rounds post their own steps. Steps posted ahead for a plain turn beside them (one near the end of
        # its context) would have the next round settle them, which finishes every speculator.
        if self._specs:
            return
        super()._post_ahead()

    def _retire(self, turn: _Turn, outcome: str) -> None:
        self._specs.pop(turn, None)
        if self._spec_live is turn:
            self._spec_live = None
        super()._retire(turn, outcome)

    def _plan(self, turn: _Turn) -> list[tuple[int, int, bool]]:
        if turn.phase != _Turn.DECODE:
            return super()._plan(turn)
        engine = self.engine
        user = engine.users[turn.slot]
        # A posted step still has to be fed (_flush_ahead). Anything else waits while a prompt is pending, and
        # only a seated turn takes a verify round.
        if not self._flushing and (self._prefill_waiting() or not self._holds_decode_seat(turn)):
            return []
        if turn in self._batched:
            if turn in self._specs or turn not in self._active:
                return []
        else:
            budget = self._spec_budget(turn)
            if budget > 0:
                self._spec_round(turn, budget)
                if turn in self._specs or turn not in self._active:
                    return []
        if turn.pending:
            return super()._plan(turn)
        finishing = (
            turn.cancelled.is_set()
            or (turn.next_id == engine.eos_id and not turn.ignore_eos)
            or len(turn.generated) >= turn.max_tokens
            or user.pos >= engine.max_seq - 1
        )
        # A reply that ends (EOS, max_tokens, stopped) hands only its own session back; one stepped on near the end
        # of the context by plain decode needs every speculator finished first.
        if not self._end_specs([turn] if finishing else None):
            return []
        return super()._plan(turn)

    def _pick_drafts(self) -> None:
        """Set this scheduler round's drafts per verify round from ``--dspark-drafts`` by the seated decoding users:
        fewer as soon as more users decode, more only once fewer have for :data:`_DSPARK_RAISE_SECONDS` (at once
        with none decoding, when there is nothing to switch)."""
        schedule = self.engine.dspark_drafts
        users = sum(
            1
            for turn in self._active
            if turn.phase == _Turn.DECODE and not turn.cancelled.is_set() and self._holds_decode_seat(turn)
        )
        target = schedule[min(max(users, 1), len(schedule)) - 1]
        if target > self._drafts and users:
            now = time.monotonic()
            if self._raise_since is None:
                self._raise_since = now
            if now - self._raise_since < _DSPARK_RAISE_SECONDS:
                return
        self._raise_since = None
        if target != self._drafts:
            logger.info(
                f"DSpark: {users} seated decoding user(s): "
                + (f"{target} drafts a round" if target else "plain decode")
                + f" (was {self._drafts or 'plain decode'})"
            )
            self._drafts = target

    def _spec_budget(self, turn: _Turn) -> int:
        """The tokens a verify round of ``turn``'s reply may emit now; 0 when it takes none (no drafts under this
        load, a posted step to feed, a reply that ends, or too near the end of the context for a whole round and the
        next one, dispatched ahead)."""
        engine = self.engine
        if not self._drafts:
            return 0
        if turn.phase != _Turn.DECODE or turn.pending or turn.cancelled.is_set() or self._flushing:
            return 0
        if turn.next_id == engine.eos_id and not turn.ignore_eos:
            return 0
        room = engine.max_seq - engine.users[turn.slot].pos - 2 * (self._drafts + 1)
        return max(0, min(turn.max_tokens - len(turn.generated), room))

    def _speculator(self, turn: _Turn):
        """``turn``'s speculator at this round's drafts, started on its session (which must be resident) at its
        first round, or restarted when the drafts per round changed: a speculator's rounds keep one length."""
        from models.experimental.deepseek_v41_flash.tt.spec import DSparkSpeculator

        spec = self._specs.get(turn)
        if spec is not None and spec.drafts == self._drafts:
            return spec
        if spec is not None:
            del self._specs[turn]
            if self._spec_live is turn:
                self._spec_live = None
            spec.finish()
            logger.debug(f"user {turn.user_key!r}: DSpark {spec.stats.summary(spec.drafts)}")
        temperature = turn.body.get("temperature")
        if temperature is None:
            temperature = self.server.temperature
        spec = DSparkSpeculator(
            self.engine.model,
            self._drafts,
            temperature=float(temperature or 0.0),
            top_p=float(turn.body.get("top_p") or 1.0),
            seed=int(torch.randint(1, 2**31 - 1, ())),
        )
        spec.start(turn.next_id, self.engine.users[turn.slot].pos)
        self._specs[turn] = spec
        return spec

    def _spec_round(self, turn: _Turn, budget: int) -> None:
        """One verify round of ``turn``'s reply, at most ``budget`` tokens from ``turn.next_id`` at the session's
        position (see :meth:`_spec_tokens`). The turn's speculator starts with its first round and ends with its
        last; a round that dispatches the next one's rows ahead is one no other seated reply follows."""
        engine = self.engine
        user = engine.users[turn.slot]
        if (self._inflight or self._posted) and not self._settle():
            return
        # The rounds write their packets themselves, after every packet the engine's thread has queued.
        engine._packets.join()
        others = any(
            other is not turn
            and other.phase == _Turn.DECODE
            and not other.cancelled.is_set()
            and self._holds_decode_seat(other)
            for other in self._active
        )
        try:
            self._pause_live(turn)
            user.activate()
            spec = self._speculator(turn)
            self._spec_live = turn
            kept, done = spec.advance(budget, stop_ids=() if turn.ignore_eos else (engine.eos_id,), ahead=not others)
        except Exception as e:  # noqa: BLE001 - see _abort_all: a round's outputs and rows are in an unknown state
            self._abort_all(e)
            return
        self._spec_tokens(turn, spec, kept, done)

    def _spec_batch(self, turns: list[_Turn]) -> None:
        """One verify round of each of ``turns``' replies, their verifies dispatched back to back so each one's
        rows run through the pipeline while the host accepts the one before and the drafter drafts it.

        Rounds are collected in the order they were dispatched (the rows leave the pipeline in that order), and
        none dispatches the next round's rows ahead, so every speculator ends the batch paused. Needs the
        host runtime: on the Python path the packets of rounds queued behind unread outputs would wedge the
        packet writes (see :meth:`~..tt.model.DeepSeekV41Model.spec_verify_async`)."""
        engine = self.engine
        if (self._inflight or self._posted) and not self._settle():
            return
        engine._packets.join()
        budgets = {turn: self._spec_budget(turn) for turn in turns}
        try:
            self._pause_live()
            for turn in turns:
                engine.users[turn.slot].activate()
                self._speculator(turn)
            for turn in turns:
                engine.users[turn.slot].activate()
                self._specs[turn].dispatch()
            rounds = []
            for turn in turns:
                engine.users[turn.slot].activate()
                rounds.append(self._specs[turn].collect(budgets[turn], () if turn.ignore_eos else (engine.eos_id,)))
        except Exception as e:  # noqa: BLE001 - see _abort_all
            self._abort_all(e)
            return
        for turn, (kept, done) in zip(turns, rounds):
            if turn not in self._active:
                return
            self._spec_tokens(turn, self._specs[turn], kept, done)

    def _spec_tokens(self, turn: _Turn, spec, kept: list[int], done: bool) -> None:
        """Take a verify round's ``kept`` tokens into ``turn`` as single steps would: every token fed, the last one
        produced left in ``next_id``. Ends the turn's speculator once ``done``."""
        engine = self.engine
        user = engine.users[turn.slot]
        before = len(turn.generated)
        for token in kept:
            turn.generated.append(turn.next_id)
            turn.mark_token()
            user.tokens.append(turn.next_id)
            user.pos += 1
            turn.next_id = token
        if spec.position != user.pos:
            self._abort_all(RuntimeError(f"DSpark is at {spec.position}, the session at {user.pos}"))
            return
        turn.stream.push(turn.generated)
        # Each verify round rolls its tokens into the status pane's tok/s and tok/s/user as it ends.
        self.steps += len(kept)
        self._spec_accounted += len(kept)
        self._account(len(kept))
        if len(turn.generated) // 32 > before // 32:
            logger.debug(
                f"user {turn.user_key!r}: {len(turn.generated)}/{turn.max_tokens} tokens "
                f"at {turn.decode_rate:.2f} tok/s over the last {DECODE_RATE_WINDOW} "
                f"({turn.mean_decode_rate:.2f} tok/s for the reply so far), "
                f"cache at {user.pos}/{engine.max_seq}, DSpark {spec.stats.summary(spec.drafts)}"
            )
        if done:
            self._end_specs([turn])

    def _pause_live(self, turn: _Turn | None = None) -> None:
        """Pause the live speculator unless it is ``turn``'s, before another session's round or end."""
        live = self._spec_live
        if live is None or live is turn:
            return
        self._spec_live = None
        self.engine.users[live.slot].activate()
        self._specs[live].pause()

    def _end_specs(self, turns=None) -> bool:
        """Hand ``turns``' sessions (every speculator's when ``None``) back to plain decode
        (:meth:`~..tt.spec.DSparkSpeculator.finish`). Returns ``False`` once a failure has aborted every turn."""
        ending = list(self._specs) if turns is None else [turn for turn in turns if turn in self._specs]
        try:
            for turn in ending:
                spec = self._specs.pop(turn)
                if self._spec_live is turn:
                    self._spec_live = None
                self.engine.users[turn.slot].activate()
                spec.finish()
                logger.debug(f"user {turn.user_key!r}: DSpark {spec.stats.summary(spec.drafts)}")
        except Exception as e:  # noqa: BLE001 - see _abort_all
            self._abort_all(e)
            return False
        return True


class V41GenerationServer(GenerationServer):
    """:class:`GenerationServer` that takes images in user messages (see the images section above): uploaded files
    or inline base64 parts, at most :data:`_MAX_IMAGES` per request, each expanded into its span of image rows when
    rendered."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pool = self.pool
        self.pool = _V41SlotPool(self.engine, pool.host_users, pool.host_bytes, pool.max_running, pool.idle_timeout)
        scheduler = _SpecScheduler if self.engine.dspark else _V41Scheduler
        self.scheduler = scheduler(self, self.scheduler.prefill_chunk, self.scheduler.decode_ahead)
        # serve() builds the server after the warmup and the prefill snapshot, and before the scheduler starts: the
        # point past every other device allocation of startup.
        self.engine.capture_vision()
        self.monitor_feed = None
        self._monitor_sink = None
        if host_rs.enabled():
            # After every capture: from here the decode step's replays, packets, Engram rows and outputs run on the
            # Rust host runtime's threads, which keep their counters where the web monitor reads them.
            prefix = monitor.default_prefix()
            self.engine.model.start_host_runtime(_host_cpus(), counters_path=monitor.counters_path(prefix))
            self.monitor_feed = monitor.MonitorFeed(
                self.monitor_snapshot,
                monitor.snapshot_path(prefix),
                period=float(os.environ.get("DEEPSEEK_V41_MONITOR_PERIOD", "1.0")),
            )
        cache = self.engine.weight_cache
        self.files = _FileStore(os.path.join(cache.path, "files") if cache else None)

    def _await_seat(self, turn, lease, client_gone) -> bool:
        """Admit the turn now. ``--max-running`` limits decode, not prefill: the scheduler prefills every pending
        prompt before generating, and only then seats ``--max-running`` replies."""
        del turn, client_gone
        with self.pool._cv:
            if self.pool._closing:
                raise RequestError(503, "server is shutting down", "server_error")
            self.pool._running.add(lease.slot)
            self.pool._cv.notify_all()
        return True

    def start(self) -> None:
        super().start()
        if self.monitor_feed is not None:
            self._monitor_sink = logger.add(self.monitor_feed.log_sink, level="DEBUG", format="{message}")
            self.monitor_feed.start()
            logger.info(f"web monitor feed: {self.monitor_feed.path}")

    def stop(self) -> None:
        if self.monitor_feed is not None:
            self.monitor_feed.stop()
            logger.remove(self._monitor_sink)
        super().stop()

    def monitor_snapshot(self) -> dict:
        """The raw state :class:`~.monitor.MonitorFeed` publishes for the web monitor: counts, sizes and ages,
        read without locking the scheduler, as the live console reads them. Every rate is the monitor's to work out,
        in its own process."""
        scheduler, pool = self.scheduler, self.pool
        now = time.perf_counter()
        host_users, host_bytes = pool.host_usage()
        # Before the active turns: see _Scheduler._retire.
        ended = list(scheduler.ended_turns)[-monitor.FEED_ENDED_TURNS :]
        specs = dict(getattr(scheduler, "_specs", {}))  # turn -> its DSpark speculator, while it speculates
        snapshot = {
            "t": time.time(),
            "model_id": self.model_id,
            "started": self._created,
            "max_seq": self.engine.max_seq,
            "slots": len(self.engine.users),
            "slot_states": pool.counts(),
            "users": pool.rows(),
            "turns": [
                {
                    "serial": turn.serial,
                    "user": turn.user_key,
                    "slot": turn.slot,
                    "phase": turn.phase,
                    "prompt_tokens": len(turn.ids),
                    "cached_tokens": turn.cached,
                    "prefilled": turn.fed,
                    "generated": len(turn.generated),
                    "max_tokens": turn.max_tokens,
                    "prefill_s": turn.prefill_seconds,
                    "age_s": now - turn.t_submit,
                    "cancelled": turn.cancelled.is_set(),
                    **_spec_fields(specs.get(turn)),
                }
                for turn in list(scheduler._active)
            ],
            "inflight": len(scheduler._inflight),
            "rounds": scheduler.rounds,
            "steps": scheduler.steps,
            "ended_turns": ended,
            "pool": scheduler.pool_usage(),
            "tokens_left": self.engine.tokens_left(),
            "host": {
                "users": host_users,
                "bytes": host_bytes,
                "max_users": pool.host_users,
                "max_bytes": pool.host_bytes,
            },
            "broken": None if scheduler._broken is None else str(scheduler._broken),
        }
        if self.engine.images is not None:
            snapshot["images"] = {**self.engine.images.stats(), "files": self.files.usage()}
        if self.engine.dspark:
            snapshot["dspark"] = {
                "drafts": scheduler._drafts,
                "schedule": list(self.engine.dspark_drafts),
                "speculating": len(specs),
            }
        return snapshot

    def stats(self) -> dict:
        stats = super().stats()
        host = self.engine.model.host_runtime_stats()
        if host is not None:
            stats["host_runtime"] = host
        if self.engine.images is not None:
            stats["images"] = {**self.engine.images.stats(), "files": self.files.usage()}
        return stats

    def upload_file(self, fields: dict) -> dict:
        """`POST /files <https://api-docs.deepseek.com/api/create-file>`_: store the image, then decode it and encode
        its rows on the vision stage at once (a scheduler chore, between rounds), so a chat request that references
        the file later finds its span ready in the embedding table (and its rows are saved, see :class:`_ImageBook`).
        Not written now (no rows free), or evicted before it is referenced, it is written at that request's
        admission instead, as an inline image is."""
        images = self.engine.images
        if images is None:
            raise RequestError(400, "this server runs without vision, so it takes no image files")
        filename, data = fields.get("file", (None, None))
        if data is None:
            raise RequestError(400, "file is required")
        purpose = fields.get("purpose", (None, b""))[1]
        if purpose != b"user_data":
            raise RequestError(400, "purpose must be 'user_data'")
        anchor, seconds = fields.get("expires_after[anchor]"), fields.get("expires_after[seconds]")
        if (anchor is None) != (seconds is None):
            raise RequestError(400, "expires_after[anchor] and expires_after[seconds] go together")
        lifetime = None
        if anchor is not None:
            if anchor[1] != b"created_at":
                raise RequestError(400, "expires_after[anchor] must be 'created_at'")
            low, high = _FILE_LIFETIME
            if not seconds[1].isdigit() or not low <= int(seconds[1]) <= high:
                raise RequestError(400, f"expires_after[seconds] must be an integer in [{low}, {high}]")
            lifetime = int(seconds[1])
        if len(data) > _MAX_FILE_BYTES:
            raise RequestError(413, f"a file may be at most {_MAX_FILE_BYTES >> 20} MiB")
        if not _is_image(data):
            raise RequestError(400, "file must be a JPEG, PNG, GIF or WebP image")
        key = images.add(data, "file")
        record = self.files.add(data, filename or "file", lifetime)
        images.status.stage(key, "queued")
        try:
            self.scheduler.run_on_scheduler(lambda: images.span(key, self._pinned_images))
        except RequestError as e:
            logger.warning(f"{record['id']}: not encoded at upload ({e}); a request that references it encodes it")
        finally:
            images.status.clear(key)
        logger.info(f"{record['id']}: {filename!r}, {len(data)} bytes, image {key[:12]}")
        return record

    def normalized_messages(self, body: dict) -> list[dict]:
        messages = _normalized_messages(body.get("messages"), body.get("tools"), self.engine.chat_format, self._content)
        keys = set(_image_keys(messages))
        if len(keys) > _MAX_IMAGES:
            raise RequestError(
                400, f"a request may carry at most {_MAX_IMAGES} distinct images (this conversation has {len(keys)})"
            )
        if not keys:
            return messages
        rows = self.engine.images.rows_of(keys)
        if rows > self.engine.images.total_rows:
            raise RequestError(
                400,
                f"the {len(keys)} images of this conversation take {rows} image rows, more than the "
                f"{self.engine.images.total_rows} the server has (--image-rows); start a new conversation or drop some images",
            )
        return messages

    def _content(self, role: str, content, index: int) -> dict:
        """``{"content": text}``, or for a user message with image parts the checkpoint's block form: ``content_blocks``
        (text, and a placeholder block per image tagged with its key) and ``content``, their text joined."""
        placeholder = encoding_dsv41.IMAGE_PLACEHOLDER
        if not isinstance(content, list) or not any(
            isinstance(part, dict) and part.get("type") in _IMAGE_PARTS for part in content
        ):
            fields = _text_content(role, content, index)
            if placeholder in fields["content"]:
                raise RequestError(400, f"messages[{index}] contains {placeholder!r}; send images as image parts")
            return fields
        if self.engine.images is None:
            raise RequestError(400, f"messages[{index}] carries an image, but this server runs without vision")
        if role != "user":
            raise RequestError(400, f"messages[{index}] carries an image, which only a user message may")
        blocks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in _IMAGE_PARTS:
                key = self.engine.images.add(_image_bytes(part, index, self.files), f"messages[{index}]")
                blocks.append({"type": "text", "text": placeholder, "image": key})
                continue
            text = _content_text([part], index)
            if placeholder in text:
                raise RequestError(400, f"messages[{index}] contains {placeholder!r}; send images as image parts")
            blocks.append({"type": "text", "text": text})
        return {"content": "\n\n".join(block["text"] for block in blocks), "content_blocks": blocks}

    def _pinned_images(self) -> set:
        """The images some conversation holds: a session's, a host conversation's, or a running or queued turn's."""
        conversations = [user.messages for user in self.engine.users]
        with self.pool._cv:
            conversations += [conv.messages for conv in self.pool.host]
        with self._turns_lock:
            turns = list(self._turns.values())
        for turn in turns:
            conversations.append(turn.body["messages"])
            if turn.restore is not None:
                conversations.append(turn.restore.messages)
        return {key for messages in conversations for key in _image_keys(messages)}

    def _render_ids(self, user: UserSession, include_assistant: bool, closing: bool | None = None) -> list[int]:
        """:meth:`GenerationServer._render_ids`, each image placeholder replaced by its image's span. Encodes (on the
        device) an image seen for the first time: only call it at admission, with no step in flight."""
        first = user._next_render
        ids = super()._render_ids(user, include_assistant, closing)
        keys = _image_keys(m for m in user.messages[first:] if include_assistant or m["role"] != "assistant")
        token = self.engine.image_token_id
        at = [i for i, t in enumerate(ids) if t == token]
        if len(at) != len(keys):
            raise RequestError(400, f"the prompt renders {len(at)} image placeholders for {len(keys)} images")
        if not keys:
            return ids
        out, prev = [], 0
        for i, key in zip(at, keys):
            out += ids[prev:i]
            out += self.engine.images.span(key, self._pinned_images)
            prev = i + 1
        return out + ids[prev:]


class V41ChatEngine(ChatEngine):
    """:class:`ChatEngine` on :class:`DeepSeekV41Model`: the same users, decode and warmup over V4.1's model.

    Construction differs: the V4.1 config and ``lm_head``, the context capped by the compressed-KV limit, the EOS
    from the tokenizer (the V4.1 config has none), and the layers split evenly over the stages in contiguous runs,
    whatever the profile's pipeline group size, so the traced prefill can share them.

    And :meth:`write_traced` hands the packet to a thread of its own. A V4.1 packet write also looks up both Engram
    layers' host rows, which would otherwise hold the scheduler between seating one user's step and the next: with
    it, the scheduler only seats and posts, the model's replay thread issues the traces, this thread writes the H2D
    socket and the scheduler's output reader drains the D2H socket, each waiting on the others only for its data.

    Prompts are rendered and tool calls parsed in V4.1's chat format (:data:`V41_CHAT_FORMAT`).
    """

    chat_format = V41_CHAT_FORMAT
    # A turn that stops early reruns its newest step in the steps still posted (repeat_traced).
    max_decode_ahead = _DECODE_AHEAD
    # Every posted step keeps the session it was posted for, so decoding turns can each keep one posted.
    interleave_ahead = True

    def __init__(self, mesh_device, args, prefetcher: contextlib.ExitStack):
        from transformers import AutoTokenizer

        loader = DeepseekV4WeightLoader(args.model_dir)
        config = load_config(loader.snapshot_dir)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
        self.max_new_tokens = args.max_new_tokens
        self.eos_id = self.tokenizer.eos_token_id
        num_layers = min(args.num_layers or config.num_hidden_layers, config.num_hidden_layers)
        self.max_seq = _traced_max_seq(config, args.max_context, num_layers)
        self.rope = _build_rope(config, loader.snapshot_dir, self.max_seq)
        # Kept for the prefill model, built on the same weights.
        self.loader = loader
        model_name = os.path.basename(os.path.normpath(str(args.model_dir)))
        self.weight_cache = WeightCache(os.path.join(args.cache_dir, model_name)) if args.cache_dir else None

        system_config = load_system_config(
            profile=args.system_profile,
            path=args.system_config_file,
            variant=getattr(args, "system_variant", None),
            mesh_device=mesh_device,
        ).log()
        num_stages = getattr(args, "num_stages", None)
        if num_stages is None:
            num_stages = pipeline_stage_count()
        vision = getattr(args, "vision", False)
        if vision and VISION_STAGE >= num_stages:
            raise ValueError(f"the vision stage {VISION_STAGE} is past the {num_stages} pipeline stages")
        self.vision = self.images = self.row_writer = None
        self.image_token_id = load_vision_config(loader.snapshot_dir).image_token_id
        prefill_placement = None
        if getattr(args, "prefill_all_stages", False):
            kv_layers = max(i for i in config.kv_source_layer_ids if i < num_layers) + 1
            prefill_placement = plan_layer_placement(kv_layers, num_stages, 1)
            logger.info(f"KV-only prefill placement over {num_stages} stages: {prefill_placement}")
        self.model = DeepSeekV41Model(
            config,
            loader,
            mesh_device,
            cache=self.weight_cache,
            weight_dtype=_WEIGHT_DTYPE,
            max_layers=num_layers,
            system_config=system_config,
            tp_size=getattr(args, "tp_size", _TP_SIZE),
            num_stages=num_stages,
            submeshes=getattr(args, "submeshes", None),
            use_prefetcher=getattr(args, "use_prefetcher", None),
            layer_submesh_ids=plan_layer_placement(num_layers, num_stages, 1),
            image_rows=args.image_rows if vision else 0,
            vision_stage=VISION_STAGE if vision else None,
            prefill_layer_submesh_ids=prefill_placement,
        )
        self.lm_head = DeepSeekV41LMHead(
            dequantized(loader, "lm_head.weight", translate=True),
            self.model.last_device,
            self.weight_cache.file("lm_head") if self.weight_cache else None,
            dtype=_WEIGHT_DTYPE,
        )
        if vision:
            # Built and compiled before the KV pools, the prefill model and the decode traces take the DRAM and the
            # program cache; captured last (capture_vision), so at serving time an image only replays traces.
            t0 = time.perf_counter()
            self.vision = DeepSeekV41Vision(
                load_vision_config(loader.snapshot_dir),
                loader,
                self.model.submeshes[VISION_STAGE],
                cache=self.weight_cache,
            )
            self.vision.warmup()
            vocab = config.vocab_size
            write_embedding_rows(
                self.model.embed_tokens.embedding_weight,
                torch.stack([self.vision.delimiters[kind] for kind in DELIMITER_ROWS]),
                [vocab + row for row in DELIMITER_ROWS.values()],
                vocab,
            )
            self.images = _ImageBook(
                self,
                loader.snapshot_dir,
                os.path.join(self.weight_cache.path, "vision_rows") if self.weight_cache else None,
            )
            logger.info(
                f"vision tower on stage {VISION_STAGE}, built and compiled in {time.perf_counter() - t0:.1f}s; "
                f"{len(self.images.free)} image rows on stage 0"
            )
        logger.info(
            f"built DeepSeekV41Model with {self.model.num_layers}/{config.num_hidden_layers} layers, "
            f"TP{self.model.tp_size} over {self.model.pipeline_stages} pipeline stages, "
            f"context {self.max_seq} tokens"
        )
        # The prefetcher session spans the whole process, and the replay thread stops before it (LIFO); see
        # ChatEngine.__init__.
        prefetcher.enter_context(self.model.prefetcher_session())
        prefetcher.callback(self.model.shutdown)
        logger.info(f"tensor prefetcher: {'on' if self.model.use_prefetcher else 'off'}")

        self.model.prepare_static_decode(
            self.rope,
            self.max_seq,
            lm_head=self.lm_head,
            num_sessions=args.num_users,
            # max_seq may be rounded above --max-context; every session still has to fit the pools.
            total_tokens=max(args.total_context, self.max_seq),
            block_size=_PAGE_BLOCK_SIZE,
            device_sampling=bool(getattr(args, "device_sampling", False)),
        )
        if self.model.context_limit is not None and self.model.context_limit < self.max_seq:
            logger.warning(f"context capped at {self.model.context_limit} tokens by the compressed KV")
            self.max_seq = self.model.context_limit
        if self.model.device_sampler is not None:
            # ``--temperature`` is the one temperature every turn samples at (a request cannot set its
            # own), and ``--seed`` seeds the PRNGs, which still advance from step to step. This engine
            # does not run ``ChatEngine.__init__``, so the sampler is switched here.
            self.temperature = float(getattr(args, "temperature", None) or 0.0)
            self.seed = int(getattr(args, "seed", 0) or 0)
            self.model.device_sampler.set_sampling(self.temperature, self.seed)
            logger.info(
                f"device sampling: the output stage picks the token "
                f"(temperature {self.temperature}, seed {self.seed}); requests cannot set their own"
            )

        self.dspark = bool(getattr(args, "dspark", False))
        # Drafts per verify round with 1, 2, ... seated decoding users, the last entry for more (--dspark-drafts).
        self.dspark_drafts = tuple(getattr(args, "dspark_drafts", _DSPARK_DRAFTS))
        if self.dspark:
            # Before anything is compiled; the drafter's prefetcher rings are on its own row.
            self.model.prepare_spec_decode(self.rope["main"])
            if self.model.use_prefetcher:
                prefetcher.enter_context(tensor_prefetcher_session(self.model.dspark.device))
            logger.info(
                f"DSpark speculative decode: {', '.join(map(str, self.dspark_drafts))} drafts a round with "
                f"1, 2, ... seated users, the drafter on mesh row {self.model.num_submeshes}"
            )
        if self.model._eager_decode:
            # An eager step runs as its packet is written, at the position it was posted for.
            self.max_decode_ahead = 1
        self.users = [UserSession(self, i, args) for i in range(args.num_users)]
        self.active = 0
        self.users[0].activate()
        self._replay_q: deque[int] = deque()
        self._packets: queue.Queue = queue.Queue()
        # The server's output reader also writes decoding turns' posted steps (_OutputReader.arm, write_prefed).
        self.packet_lock = threading.Lock()
        self._packet_error: BaseException | None = None
        self._packet_writer = threading.Thread(target=self._write_packets, name="decode-packets", daemon=True)
        self._packet_writer.start()
        # Registered after the model's shutdown so it unwinds first: the thread's closure holds the model.
        prefetcher.callback(self._stop_packet_writer)

    def warmup(self) -> None:
        """:meth:`ChatEngine.warmup`; with DSpark, capture the speculative traces too (every stage's fixup and the
        drafter's, :meth:`DeepSeekV41Model.capture_spec_traces`) along with the decode variants."""
        if not self.dspark:
            super().warmup()
            return
        print("[warming up: compiling kernels and capturing decode and DSpark traces...]", flush=True)
        t0 = time.perf_counter()
        user = self.users[0]
        user.activate()
        self.model.capture_spec_traces(self.tokenizer.bos_token_id or 0, 0)  # rows discarded by the reset
        user.reset()
        print(f"[warmup done in {time.perf_counter() - t0:.1f}s]", flush=True)

    def capture_vision(self) -> None:
        """Capture the vision traces: one per ViT bucket on the vision stage, and the embedding-row write on stage 0.
        The last device allocations of startup, after the KV pools, the prefill and decode traces and the resident
        snapshot: a trace's buffers must not land under one allocated after it. An image then replays them only,
        so serving allocates nothing for it and compiles nothing (no CCL semaphores taken out of L1_SMALL), and its
        data moves over the model's image sockets only."""
        if self.vision is None:
            return
        t0 = time.perf_counter()
        with self.model._cq_lock:
            self.vision.capture(self.model.image_sockets)
            self.row_writer = EmbeddingRowWriter(
                self.model.embed_tokens.embedding_weight, self.config.vocab_size, self.model.image_sockets
            )
        logger.info(
            f"vision traces captured in {time.perf_counter() - t0:.1f}s: {len(self.vision.traces)} ViT buckets on "
            f"stage {VISION_STAGE}, the row write on stage 0"
        )

    def write_traced(self, token_id: int, pos: int) -> None:
        """Queue the packet for the oldest :meth:`post_traced` step on the packet thread, which writes the packets
        in queueing order, i.e. in the order the steps were posted. Raises once that thread has failed."""
        if self._packet_error is not None:
            raise RuntimeError("the decode packet writer failed") from self._packet_error
        queued = self._replay_q.popleft()
        if queued != int(pos):
            raise RuntimeError(f"traced packet pos {pos} does not match queued replay {queued}")
        self._put_packet(int(token_id), int(pos))

    def _put_packet(self, token_id: int, pos: int) -> None:
        if not self.model.host_runtime_running:
            self._packets.put((token_id, pos))
            return
        # The host runtime queues the write on its own packet thread, so no Python thread is needed.
        with self.packet_lock:
            self.model.write_step_packet(token_id, pos)

    def repeat_traced(self, token_id: int, pos: int) -> None:
        """Feed the oldest :meth:`post_traced` step the packet of ``pos``, the session's newest step, fed
        ``token_id``, instead of its own: a turn that stops early cannot take its posted steps, and fed at their
        own positions they would overwrite rows of the sliding window. The caller then ends the session at
        ``pos + 1`` (:meth:`DeepSeekV41Model.end_session_at`)."""
        if self._packet_error is not None:
            raise RuntimeError("the decode packet writer failed") from self._packet_error
        queued = self._replay_q.popleft()
        if queued <= int(pos):
            raise RuntimeError(f"rerunning pos {pos} in the step posted for pos {queued}, which is not after it")
        self._put_packet(int(token_id), int(pos))

    def write_prefed(self, token_id: int, pos: int) -> None:
        """The output reader's packet write (:class:`_OutputReader`): straight to the socket, as it is on the
        token path, once this thread has written every packet queued before it, so they land in posting order."""
        self._packets.join()
        if self._packet_error is not None:
            raise RuntimeError("the decode packet writer failed") from self._packet_error
        with self.packet_lock:
            self.model.write_step_packet(token_id, pos)

    def _write_packets(self) -> None:
        # Every item is marked done, written or not, so write_prefed never waits on a dead writer.
        while True:
            item = self._packets.get()
            if item is None:
                self._packets.task_done()
                return
            token_id, pos = item
            try:
                if self._packet_error is None:
                    with self.packet_lock:
                        self.model.write_step_packet(token_id, pos)
            except BaseException as e:  # noqa: BLE001 - surfaced by the next write_traced
                logger.exception(f"writing the decode packet for pos {pos} failed: {e}")
                self._packet_error = e
            finally:
                self._packets.task_done()

    def _stop_packet_writer(self) -> None:
        self._packets.put(None)
        self._packet_writer.join(timeout=30)


class _V41Prefiller(_Prefiller):
    """:class:`_Prefiller` with V4.1's prefill model, which has no CSA position bias for the commit.

    The prefill selects two-level, as decode does, and every chunk reads only the entries it can see (or its
    candidates' pool), so its device memory past the weights is the entry buffers ``--prefill-max-len`` sizes
    (the context when unset).

    With ``--prefill-kv-only`` (the default) the prefill model is layers 0-20 only (up to the last kv source, on
    stages 0-4, or on every stage with ``--prefill-all-stages``) with no head: they write all of the shared compressed KV. :meth:`aligned` leaves at least the last
    ``sliding_window`` (128) prompt tokens to decode, whose replay of them fills every layer's sliding window, the
    rings of layers 21-39 included, and gives the first token's logits.
    """

    def __init__(self, engine: V41ChatEngine, args, prefetcher: contextlib.ExitStack):
        self.kv_only = args.prefill_kv_only
        # The DSpark drafter attends the last sliding_window positions, and only decode steps write its rings.
        self.decode_window = self.kv_only or engine.dspark
        super().__init__(engine, args, prefetcher)

    def _build_model(self, engine: V41ChatEngine):
        return engine.model.build_prefill(
            engine.rope,
            lm_head=engine.lm_head,
            kv_only=self.kv_only,
            cache=engine.weight_cache,
            weight_dtype=_ATTENTION_WEIGHT_DTYPE,
            # Fresh-window traces, so a conversation edited past an earlier turn resumes from its cached
            # compressed KV (SWA bounded replay) instead of re-prefilling from 0.
            bounded_replay=True,
            progress=self._note,
        )

    def _bias_slots(self):
        return None

    @property
    def continue_alignment(self) -> int | None:
        return self.model.continue_alignment

    @property
    def replay_window(self) -> int | None:
        return self.model.config.sliding_window

    def aligned(self, n_tokens: int, start: int = 0, bounded: bool = False) -> int:
        """Also continues a session: from a multiple of :attr:`continue_alignment` the prefill model starts from
        the session's decode caches (see :meth:`DeepSeekV41Model.seed_prefill`), or, ``bounded``, from their
        compressed KV alone. KV-only or with DSpark, the last ``sliding_window`` of ``n_tokens`` are always left to
        decode."""
        if self.decode_window:
            n_tokens -= self.replay_window
            if n_tokens <= 0:
                return 0
        if not start:
            return super().aligned(n_tokens)
        return self.model.continuation_tokens(start, n_tokens, bounded_replay=bounded)

    def _prefill(self, user, ids, on_chunk, replayed: int = 0):
        if not user.pos:
            return super()._prefill(user, ids, on_chunk)
        model = self.engine.model
        if replayed:
            return self.model.prefill_traced(
                ids,
                on_chunk=on_chunk,
                start=user.pos - replayed,
                seed=lambda traced: model.seed_prefill(traced, user.sid, tails=False),
                bounded_replay=True,
            )
        return self.model.prefill_traced(
            ids, on_chunk=on_chunk, start=user.pos, seed=lambda traced: model.seed_prefill(traced, user.sid)
        )

    def _commit(self, user, states) -> int:
        if not user.pos:
            return super()._commit(user, states)
        return self.engine.model.commit_prefill_state(states, user.sid, start=user.pos)


def _stage_engram_tables(args) -> bool:
    """Stage the Engram tables in ``STAGE_DIR`` (tmpfs) before the mesh is opened, and check that what is left
    for the model to read or map fits in host RAM.

    The build reads the tables into RAM itself otherwise (~203 GB, once per process), and a copy made here holds
    the one copy every later run maps instead: startup then reads nothing, and two servers can serve from it at
    once (see :func:`~..tt.engram_lookup.stage_engram_tables`). Do it with the card still free, since the copy
    takes ~40 s from NVMe. ``--no-engram-staging`` (or ``DEEPSEEK_V41_ENGRAM_STAGE=0``) skips it, and then the
    model's own RAM check is the only gate.
    """
    config = load_config(args.model_dir)
    layers = min(args.num_layers or config.num_hidden_layers, config.num_hidden_layers)
    ids = [i for i in config.engram_layer_ids if i < layers]
    if not ids:
        return True
    loader = DeepseekV4WeightLoader(args.model_dir)
    if args.engram_staging:
        _, copied = stage_engram_tables(loader, ids)
        if copied:
            logger.info(f"Engram tables of layers {ids} staged in {STAGE_DIR}: {copied / 1e9:.0f} GB copied")
    else:
        logger.info("Engram staging off: each process reads the tables from the checkpoint")
    need = engram_ram_needed(loader, ids)
    available = host_mem_available()
    if available < ENGRAM_RAM_MARGIN * need:
        print(
            f"Engram tables of layers {ids} need {ENGRAM_RAM_MARGIN * need / 1e9:.0f} GB of free RAM to load, "
            f"and {available / 1e9:.0f} GB is available. Stage them once with\n"
            f"  python models/experimental/deepseek_v41_flash/tools/stage_engram_tables.py\n"
            f"to share one {STAGE_DIR} copy between runs instead of reading the checkpoint per process.",
            file=sys.stderr,
        )
        return False
    return True


def _launch_monitor(port: int) -> None:
    """Start ``demo/monitor.py`` on ``port``, on this user's feed, unless one of this user's monitors already runs
    or something else holds the port. In a session of its own, so it outlives the server (and a Ctrl-C of it),
    and its page and history carry over a restart; its output goes to ``<prefix>.monitor.log``."""
    script = Path(monitor.__file__).resolve()
    uid = os.getuid()
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            if pid_dir.stat().st_uid != uid:
                continue
            argv = (pid_dir / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(arg.endswith(b"demo/monitor.py") for arg in argv):
            logger.info(f"web monitor already running (pid {pid_dir.name})")
            return
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            logger.warning(f"not starting the web monitor: port {port} is taken (--monitor-port picks another)")
            return
    prefix = monitor.default_prefix()
    log_path = f"{prefix}.monitor.log"
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, str(script), "--port", str(port), "--prefix", prefix],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    logger.info(f"web monitor started (pid {proc.pid}) on http://{socket.gethostname()}:{port}, log {log_path}")


def parse_args(argv: list[str] | None = None):
    p, sys_cfg = build_parser(argv, description=__doc__)
    # The V4 parser's sizing defaults come from the machine profile, which is tuned for V4.
    p.set_defaults(
        model_dir=str(DEFAULT_MODEL_DIR),
        model_id=os.environ.get("DEEPSEEK_V4_MODEL_ID", "deepseek-v4.1-flash"),
        num_layers=None,
        num_users=None,  # _NUM_USERS, or _NUM_USERS_ALL_STAGES with --prefill-all-stages
        max_context=_MAX_CONTEXT,
        total_context=None,
        # None means "no reply cap": a request runs until EOS or the context is full.
        # ``--max-new-tokens N`` still caps a reply at N.
        max_new_tokens=None,
        # Requests that omit temperature sample at 1; a request's own temperature, or 0, still wins. With
        # --device-sampling this is the temperature the device sampler is left at, since a request cannot
        # set its own there.
        temperature=1.0,
        decode_ahead=_DECODE_AHEAD,
        device_sampling=os.environ.get("DEEPSEEK_V4_DEVICE_SAMPLING", "1") not in ("0", "", "false", "False"),
    )
    p.add_argument(
        "--prefill-kv-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="traced prefill of layers 0-20 only (up to the last kv source, no head), the last 128 prompt tokens "
        "replayed through decode to fill the other layers' sliding windows (default: on); "
        "--no-prefill-kv-only prefills all 40 layers",
    )
    p.add_argument(
        "--prefill-all-stages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="place the KV-only prefill's layers 0-20 evenly over every stage (3 on each of stages 0-4, 2 on each "
        "of stages 5-7 at the default eight) rather than on their decode stages (0-4), loading a copy of the routed "
        "experts of each layer moved off its decode stage; needs --prefill-kv-only, and --num-users then defaults "
        f"to {_NUM_USERS_ALL_STAGES} (instead of {_NUM_USERS}) to leave them the DRAM "
        "(default: on whenever the prefill is KV-only; DEEPSEEK_V41_PREFILL_ALL_STAGES=0 turns it off)",
    )
    p.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"accept images in user messages: the vision tower on stage {VISION_STAGE} "
        "(DEEPSEEK_V41_VISION_STAGE), its rows in an extended embedding table on stage 0 (default: on)",
    )
    p.add_argument(
        "--image-rows",
        type=int,
        default=40960,
        help="rows past the vocabulary in the embedding table for image spans, a multiple of 32 (10 KB each on "
        "every chip of stage 0: 400 MiB by default, beside the vocabulary's 1.2 GiB); an image of up to 1024 "
        "tokens takes as many (~600 for a typical screenshot), and every image of a request, and of any "
        "conversation held, keeps its rows at once (default: 40960)",
    )
    p.add_argument(
        "--engram-staging",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DEEPSEEK_V41_ENGRAM_STAGE", "1") not in ("0", "", "false", "False"),
        help="stage the Engram n-gram tables (~203 GB) in tmpfs before opening the mesh, so one copy is mapped "
        "by every run and process instead of each reading it into RAM (default: on; "
        "DEEPSEEK_V41_ENGRAM_STAGE=0 turns it off)",
    )
    p.add_argument(
        "--dspark",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DEEPSEEK_V41_DSPARK", "1") not in ("0", "", "false", "False"),
        help="decode replies by DSpark speculative decode, --dspark-drafts drafts a round, the drafter on the "
        f"mesh's last row (decode on the first {_DSPARK_STAGES}); --num-users slots (default {_NUM_USERS_DSPARK}), "
        "host sampling at the "
        "request's temperature / top_p, and no --vision or --device-sampling "
        "(default: on; --no-dspark or DEEPSEEK_V41_DSPARK=0 turns it off)",
    )
    p.add_argument(
        "--dspark-drafts",
        default=os.environ.get("DEEPSEEK_V41_DSPARK_DRAFTS", ",".join(map(str, _DSPARK_DRAFTS))),
        help="with --dspark, the drafts a verify round checks by the number of seated decoding users: a comma list "
        "for 1, 2, 3, ... users, its last entry for any more, each 0 (plain decode) to "
        f"{_DSPARK_MAX_DRAFTS}. Fewer drafts apply as soon as more users decode, more once fewer have for "
        f"{_DSPARK_RAISE_SECONDS:g} s; one number fixes it (default: DEEPSEEK_V41_DSPARK_DRAFTS, else "
        f"{','.join(map(str, _DSPARK_DRAFTS))})",
    )
    p.add_argument(
        "--rust-host",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DEEPSEEK_V41_RUST_HOST", "1") not in ("0", "", "false", "False"),
        help="run the decode step's host work on the Rust host runtime (tt/host_rs.py), built at startup by "
        "host_rs/build.sh; a failed build stops the server (default: on; --no-rust-host or "
        "DEEPSEEK_V41_RUST_HOST=0 serves on Python threads instead)",
    )
    p.add_argument(
        "--monitor-port",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V41_MONITOR_PORT", "8090")),
        help="port of the web monitor (demo/monitor.py) started beside the server on the Rust host runtime, unless "
        "one of this user's is already running; it outlives the server, so its page stays up across restarts "
        "(default: 8090, DEEPSEEK_V41_MONITOR_PORT; 0 starts none)",
    )
    for action in p._actions:
        if action.dest == "temperature":
            action.help = (
                "sampling temperature for requests that send none (default: 1); "
                "a request's own temperature overrides it, and 0 asks for greedy"
            )
        elif "--decode-ahead" in action.option_strings:
            action.help = (
                "decode steps a lone decoding turn keeps posted before their tokens are known, as "
                f"test_full_model_decode_demo.py keeps them (default and most: {_DECODE_AHEAD})"
            )
        elif action.dest == "max_running":
            action.default = None
            action.help = (
                "turns decoding at once, in admission order, once every pending prompt is in (default: "
                "DEEPSEEK_V4_MAX_RUNNING, else 8, one per pipeline stage; with --dspark, --dspark-drafts follows "
                "how many decode). Capped at --num-users"
            )
        elif action.dest == "l1_small_size":
            action.default = None
            action.help = (
                "bytes of L1_SMALL reserved for the CCL semaphores and the decode prefetcher's pipe config pages "
                f"when the prefill model is on (must not be 0; default: DEEPSEEK_V4_L1_SMALL_SIZE, else "
                f"{_L1_SMALL_SIZE})"
            )
    args = p.parse_args(argv)
    if args.max_running is None:
        env = os.environ.get("DEEPSEEK_V4_MAX_RUNNING")
        args.max_running = int(env) if env else 8
    if args.max_running < 1:
        p.error("--max-running must be at least 1")
    try:
        args.dspark_drafts = tuple(int(k) for k in str(args.dspark_drafts).split(",") if k.strip())
    except ValueError:
        p.error(f"--dspark-drafts takes comma-separated integers, got {args.dspark_drafts!r}")
    if not args.dspark_drafts or not all(0 <= k <= _DSPARK_MAX_DRAFTS for k in args.dspark_drafts):
        p.error(f"--dspark-drafts takes 0 to {_DSPARK_MAX_DRAFTS} drafts per entry, got {args.dspark_drafts}")
    if args.dspark:
        # Speculative decode accepts against the steps' logits on the host and has no image rows. Its turns take
        # verify rounds in turn, each session with its own rows of the drafter's rings.
        args.vision = False
        args.device_sampling = False
        # Each verify round's log line, read when tt/spec.py is imported; DEEPSEEK_V4_SPEC_VERBOSE=1 brings them back.
        os.environ.setdefault("DEEPSEEK_V4_SPEC_VERBOSE", "0")
    # host_rs.enabled(), which the model reads too, follows the flag.
    os.environ["DEEPSEEK_V41_RUST_HOST"] = "1" if args.rust_host else "0"
    if args.rust_host:
        args.tui = False  # the web monitor (demo/monitor.py) watches it, from a process of its own
    if args.vision and (args.image_rows % 32 or args.image_rows <= FIRST_IMAGE_ROW):
        p.error(f"--image-rows must be a positive multiple of 32, got {args.image_rows}")
    if args.l1_small_size is None:
        args.l1_small_size = int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", _L1_SMALL_SIZE))
    if args.max_new_tokens is None:
        args.max_new_tokens = args.max_context
    if args.prefill_all_stages is None:
        args.prefill_all_stages = (
            args.prefill
            and args.prefill_kv_only
            and os.environ.get("DEEPSEEK_V41_PREFILL_ALL_STAGES", "1") not in ("0", "", "false", "False")
        )
    if args.prefill_all_stages and not (args.prefill and args.prefill_kv_only):
        p.error("--prefill-all-stages places the KV-only prefill: it needs --prefill and --prefill-kv-only")
    if args.num_users is None:
        args.num_users = _NUM_USERS
    if args.total_context is None:
        kv_users = _NUM_USERS_ALL_STAGES if args.prefill_all_stages else (_NUM_USERS_DSPARK if args.dspark else 64)
        args.total_context = max(kv_users * _KV_TOKENS_PER_USER, args.max_context)
    return check_args(p, args, sys_cfg)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(args.model_dir).expanduser().exists():
        print(f"checkpoint not found: {args.model_dir}", file=sys.stderr)
        return 1
    if args.rust_host:
        host_rs.build()
        if args.monitor_port:
            _launch_monitor(args.monitor_port)
    if not _stage_engram_tables(args):
        return 1
    return serve(
        args,
        engine_cls=V41ChatEngine,
        prefiller_cls=_V41Prefiller,
        num_stages=min(pipeline_stage_count(), _DSPARK_STAGES) if args.dspark else pipeline_stage_count(),
        handler_cls=_V41Handler,
        server_cls=V41GenerationServer,
    )


if __name__ == "__main__":
    sys.exit(main())
