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
  gives the first token.

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

* Users. ``--num-users`` sessions (64 by default) stay open at once, each with its own ring and window slot and its
  own rows of the compressed-KV page tables; their compressed KV shares pools sized for ``--total-context`` tokens (default
  ``num_users * 204800``). At most ``--max-running`` (8, one per stage) generate at once, the rest parked
  or queued; see "Users & KV cache" in the V4 server's docstring. A step decodes one user, and switching users costs no device work: the step's packet
  names the user's slot. The scheduler interleaves the active users' steps through the stages, so up to one
  decoding user per stage (``DEEPSEEK_V4_NUM_STAGES``, default eight) keeps every stage busy; seating and posting,
  trace dispatch, the H2D packet writes and the D2H reads each run on their own thread (see :class:`V41ChatEngine`).
  Speculative decode is single-user.
* Context. ``--max-context`` is 1048576 tokens by default (prompt + reply), the model's full context. The pools
  hold ``--total-context`` tokens, by default 204800 per user (64 x 204800 is what fits in device DRAM), and are
  paged, so any one user can grow to the full context while the others leave blocks free; a user that needs a
  block when the pools are full is refused. Past
  ``candidate_topk_blocks * candidate_block_size`` (16384) compressed entries the select is two-level, in the
  traced prefill and in decode, so a prompt prefills up to ``--prefill-max-len`` (by default the whole context, at
  most 131072 tokens, see :class:`_V41Prefiller`) and decode replays the rest.
* Engram layers 1 and 14 read host n-gram tables (~101 GB each); the model refuses to build without that RAM free,
  unless ``tools/stage_engram_tables.py`` has already put them in RAM.
* Images (``--vision``, on by default). A request's messages may carry up to 64 distinct images, as long as their rows
  fit ``--image-rows`` together (up to 1024 an image, fewer for a small one): an uploaded file (a
  ``file`` part with a ``file_id``) or inline base64 (a ``file`` part's ``file_data`` or an ``image_url``, holding a
  ``data:`` URL, or an Anthropic ``image`` with a base64 ``source``); a URL or path is refused.
  The vision tower runs on stage ``DEEPSEEK_V41_VISION_STAGE`` (5), with no step in flight, and writes each image's
  rows into ``--image-rows`` rows past the vocabulary of the embedding table, where its span's token ids point (see
  :class:`_ImageBook`); prefill and decode then read them like any token's, and the router picks the image tokens'
  experts with ``bias_vl``. Speculative decode does not take images. The whole image path is traced: the tower (one
  trace per ViT bucket) and the row write (one trace on stage 0) are captured at the end of startup, after every
  other allocation, and an image only copies its inputs in and replays them (:meth:`V41ChatEngine.capture_vision`).
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
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from email.parser import BytesHeaderParser
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
from models.experimental.deepseek_v41_flash.tt.config import (
    DEFAULT_MODEL_DIR,
    VISION_STAGE,
    dequantized,
    load_config,
    load_vision_config,
    pipeline_stage_count,
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
    GenerationServer,
    RequestError,
    _content_text,
    _Handler,
    _normalized_messages,
    _Prefiller,
    _text_content,
    build_parser,
    check_args,
    serve,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_NUM_USERS = 64  # default --num-users
# Default --max-context (prompt + reply): what one user's page tables address, the model's full context.
_MAX_CONTEXT = 1048576
# Default --total-context is --num-users x this. The layer-20 pool costs ~1.25 KiB per token on each chip of
# stages 4-7, so 64 users x 1048576 tokens would not fit; 64 x 204800 take ~16 GB of the ~17 GB free. The pools
# are paged, so any one user can still grow to --max-context while the others leave the blocks free.
_KV_TOKENS_PER_USER = 204800

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
# The prefill's semaphores also land in the holes the released prefetch GCBs leave, so at 8192 their restore finds
# no contiguous 432 B. Every byte comes out of L1 too, where 12288 leaves the decode fused_experts CBs 256 B short.
_L1_SMALL_SIZE = 10240
_PREPARED_IMAGES = 64  # decoded images (their ViT patches, ~11 MB at most each) kept for admission
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


class V41GenerationServer(GenerationServer):
    """:class:`GenerationServer` that takes images in user messages (see the images section above): uploaded files
    or inline base64 parts, at most :data:`_MAX_IMAGES` per request, each expanded into its span of image rows when
    rendered."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # serve() builds the server after the warmup and the prefill snapshot, and before the scheduler starts: the
        # point past every other device allocation of startup.
        self.engine.capture_vision()
        cache = self.engine.weight_cache
        self.files = _FileStore(os.path.join(cache.path, "files") if cache else None)

    def stats(self) -> dict:
        stats = super().stats()
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
        )
        if self.model.context_limit is not None and self.model.context_limit < self.max_seq:
            logger.warning(f"context capped at {self.model.context_limit} tokens by the compressed KV")
            self.max_seq = self.model.context_limit

        self.users = [UserSession(self, i, args) for i in range(args.num_users)]
        self.active = 0
        self.users[0].activate()
        self._replay_q: deque[int] = deque()
        self._packets: queue.Queue = queue.Queue()
        # The server's output reader also writes a lone turn's posted step (_OutputReader.arm).
        self.packet_lock = threading.Lock()
        self._packet_error: BaseException | None = None
        self._packet_writer = threading.Thread(target=self._write_packets, name="decode-packets", daemon=True)
        self._packet_writer.start()
        # Registered after the model's shutdown so it unwinds first: the thread's closure holds the model.
        prefetcher.callback(self._stop_packet_writer)

    def capture_vision(self) -> None:
        """Capture the vision traces: one per ViT bucket on the vision stage, and the embedding-row write on stage 0.
        The last device allocations of startup, after the KV pools, the prefill and decode traces and the resident
        snapshot: a trace's buffers must not land under one allocated after it. An image then replays them only,
        so serving allocates nothing for it and compiles nothing (no CCL semaphores taken out of L1_SMALL)."""
        if self.vision is None:
            return
        t0 = time.perf_counter()
        with self.model._cq_lock:
            self.vision.capture()
            self.row_writer = EmbeddingRowWriter(self.model.embed_tokens.embedding_weight, self.config.vocab_size)
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
        self._packets.put((int(token_id), int(pos)))

    def _write_packets(self) -> None:
        for token_id, pos in iter(self._packets.get, None):
            try:
                with self.packet_lock:
                    self.model.write_step_packet(token_id, pos)
            except BaseException as e:  # noqa: BLE001 - surfaced by the next write_traced
                logger.exception(f"writing the decode packet for pos {pos} failed: {e}")
                self._packet_error = e
                return

    def _stop_packet_writer(self) -> None:
        self._packets.put(None)
        self._packet_writer.join(timeout=30)


class _V41Prefiller(_Prefiller):
    """:class:`_Prefiller` with V4.1's prefill model, which has no CSA position bias for the commit.

    The prefill selects two-level, as decode does, and every chunk reads only the entries it can see (or its
    candidates' pool), so its device memory past the weights is the entry buffers ``--prefill-max-len`` sizes
    (the context when unset).

    With ``--prefill-kv-only`` (the default) the prefill model is layers 0-20 only (up to the last kv source, on
    stages 0-4) with no head: they write all of the shared compressed KV. :meth:`aligned` leaves at least the last
    ``sliding_window`` (128) prompt tokens to decode, whose replay of them fills every layer's sliding window, the
    rings of layers 21-39 included, and gives the first token's logits.
    """

    def __init__(self, engine: V41ChatEngine, args, prefetcher: contextlib.ExitStack):
        self.kv_only = args.prefill_kv_only
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
        compressed KV alone. KV-only, the last ``sliding_window`` of ``n_tokens`` are always left to decode."""
        if self.kv_only:
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


def parse_args(argv: list[str] | None = None):
    p, sys_cfg = build_parser(argv, description=__doc__)
    # The V4 parser's sizing defaults come from the machine profile, which is tuned for V4.
    p.set_defaults(
        model_dir=str(DEFAULT_MODEL_DIR),
        model_id=os.environ.get("DEEPSEEK_V4_MODEL_ID", "deepseek-v4.1-flash"),
        num_layers=None,
        num_users=_NUM_USERS,
        max_context=_MAX_CONTEXT,
        total_context=None,
        # None means "no reply cap": a request runs until EOS or the context is full.
        # ``--max-new-tokens N`` still caps a reply at N.
        max_new_tokens=None,
        # Requests that omit temperature sample at 1. 0, or a request's own temperature, still wins.
        temperature=1.0,
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
    for action in p._actions:
        if action.dest == "temperature":
            action.help = (
                "sampling temperature for requests that send none (default: 1); "
                "a request's own temperature overrides it, and 0 asks for greedy"
            )
        elif action.dest == "l1_small_size":
            action.default = None
            action.help = (
                "bytes of L1_SMALL reserved for the CCL semaphores and the decode prefetcher's GCB config pages "
                f"when the prefill model is on (must not be 0; default: DEEPSEEK_V4_L1_SMALL_SIZE, else "
                f"{_L1_SMALL_SIZE})"
            )
    args = p.parse_args(argv)
    if args.vision and (args.image_rows % 32 or args.image_rows <= FIRST_IMAGE_ROW):
        p.error(f"--image-rows must be a positive multiple of 32, got {args.image_rows}")
    if args.l1_small_size is None:
        args.l1_small_size = int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", _L1_SMALL_SIZE))
    if args.max_new_tokens is None:
        args.max_new_tokens = args.max_context
    if args.total_context is None:
        args.total_context = max(args.num_users * _KV_TOKENS_PER_USER, args.max_context)
    return check_args(p, args, sys_cfg)


def main(argv: list[str] | None = None) -> int:
    return serve(
        parse_args(argv),
        engine_cls=V41ChatEngine,
        prefiller_cls=_V41Prefiller,
        num_stages=pipeline_stage_count(),
        handler_cls=_V41Handler,
        server_cls=V41GenerationServer,
    )


if __name__ == "__main__":
    sys.exit(main())
