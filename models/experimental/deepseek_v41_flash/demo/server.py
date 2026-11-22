# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP server on the full ttnn ``DeepSeekV41Model``.

The V4.1 twin of ``models/experimental/deepseek_v4_flash/demo/server.py``. Chat, completions, sessions, tool
calling, thinking mode, the scheduler, the live console and traced prefill are that server's (read that
docstring), on the V4.1 decode model and :class:`DeepSeekV41PrefillModel` instead:

* the engine (:class:`V41ChatEngine`) builds :class:`DeepSeekV41Model` and its ``lm_head``; its layers are split
  evenly over the stages in contiguous runs (five per stage at the default eight), a placement the prefill model
  shares, as in ``tests/pipeline_parallel_prefill/test_prefill_decode_demo.py``. The full mesh stays open;
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
* Persistent MoE (``--persistent-moe``, on by default; ``--no-persistent-moe`` or ``DEEPSEEK_V41_PERSISTENT_MOE=0``
  serves the stage pipeline described below instead). Every layer's MoE runs on the persistent TP12 ``fused_experts``
  server on rows 5-7 (``tt/decode/persistent_moe.py``), which ``--instances`` (default 5) independent model instances
  share: instance j is a whole ``1 x 4`` model on mesh row 4 - j -- every layer's attention, Engram, hyper-connections
  and norms, the ``lm_head`` and its DSpark drafter (``DeepSeekV41Model(drafter_on_stage=True)``) -- and the server's
  requester j, which it serves in whatever order the instances send. Each instance has its own weights, session,
  scheduler and Rust host runtime. What they share is the MoE and one Engram n-gram table mapping (~203 GB,
  :func:`~..tt.model.load_engram_tables`): each runtime keeps its own session histories over that mapping. The
  instances load in parallel, one thread per mesh row; trace capture stays one instance at a time, because a capture
  selects the shared server's request mode. A request goes to the instance holding its conversation, else to the
  idlest one (:class:`_Instances`). With ``--dspark`` each round of a reply is either a 4-row verify step (3 drafts + 1 checked in
  ONE traced step, ``DeepSeekV41Model.decode_verify_block``; the draft's FFNs on the server as 4-row requests too) or a
  one-row step: ``--dspark-drafts`` (entries 3 or 0) picks by how many users decode on all the instances, default
  :data:`_INSTANCE_DSPARK_DRAFTS` (4-row rounds at any load: with five users decoding at once, one per instance, GSM8K
  measured 127 tok/s/user, 635 in all, against 100 / 500 a step at a time). The server is launched once for 4-row
  blocks and runs one-row requests natively on the same launch, so a switch costs nothing. Up to ``--instances`` users
  decode in parallel, one session each. A conversation stays on the instance that holds it: a request for it that
  finds that session busy gets a 429 and is not started on another instance. A new conversation goes to the idlest
  instance, and to the next when that one turns it away with a 429 before replying; it takes a parked session only
  if it repeats that session's start (:class:`_V41SlotPool`) or once the session has been idle
  ``--slot-idle-timeout`` seconds. One instance alone measured ~216 tok/s/user on GSM8K and ~145 on chat text
  against ~149 a step at a time (``tests/decode/test_dspark_spec.py``). What that path takes: one session per
  instance (``--num-users 1``), greedy decoding with the sampler on device (``--temperature 0``; a request's
  ``temperature`` and ``top_p`` are ignored) and no ``--vision``. Prompts are prefilled by each instance's own traced
  prefill model (:class:`_V41Prefiller`, one user per instance on that instance's row; no pipeline over rows), its MoE on
  the persistent TP12 prefill MoE (``tt/prefill/persistent_moe.py``) launched with the server over the same blocks:
  the trace holds each layer's request half, the MoE mesh serves each replayed chunk. A prefill stops the shared
  decode server, so the other instances settle their posted steps and park meanwhile (:class:`_MoeGate`); the last
  ``sliding_window`` prompt tokens are replayed through decode (KV-only prefill, and the DSpark drafter's rings).
  ``--no-prefill`` replays whole prompts through decode instead. The 4-row verify attends the compressed groups densely, so it covers
  positions below ~508; past them the reply decodes a step at a time. ``--max-context`` defaults to
  :data:`_PERSISTENT_MAX_CONTEXT` here (one-level select; longer contexts are untested on this layout), and
  ``--total-context`` to :data:`_PERSISTENT_TOTAL_CONTEXT` (4M tokens per instance, shared by its running user and
  any number of conversations parked in its device RAM).
  ``--instances 1`` serves the single stage on row 4 with the drafter on row 3. The rest of this docstring describes
  the stage pipeline.
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
* Shared prefixes. A prompt that continues no conversation starts from the longest cached prefix of its tokens.
  With a traced prefill model that prefix comes from another slot's cache, running or parked (a common system
  prompt, a forked conversation): the two slots share that prefix's compressed KV blocks, counted per slot so
  either can rewind or reset without the other losing them, and the prefill model rebuilds the new slot's sliding
  window by replaying the 128 tokens before the prefix's end (``DeepSeekV41Model.attach_session_prefix``,
  ``_Scheduler._share_prefix`` in the V4 server). With prompts replayed through decode (persistent MoE,
  ``--no-prefill``), every instance shares one prefix table (:class:`~models.experimental.deepseek_v4_flash.demo.server._PrefixCache`).
  A hit loads a decode-ready snapshot of that prefix — rings, compressor windows, compressed KV and Engram
  history — with ``import_session``, and decode replays the tail. The table is keyed by the token prefix, so a
  hit is the same whichever user or slot produced it. The match stops at the first image token and leaves the
  last prompt token to decode. Either way the hit is ``prompt_cache_hit_tokens``.
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
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
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
from models.experimental.deepseek_v41_flash.tests.pipeline_parallel_prefill.test_full_model_prefill_demo import (
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
from models.experimental.deepseek_v41_flash.tt.decode import persistent_moe as pmoe
from models.experimental.deepseek_v41_flash.tt.model import DeepSeekV41Model, load_engram_tables
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import (
    PrefillClient,
    PrefillServerError,
    parse_prefill_address,
)
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
    _DEFAULT_USER,
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
    _PrefixCache,
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
from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import PagedCacheFull
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT as _PREFILL_ALIGNMENT
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tt_py_test_utils_common.tensor_prefetcher import tensor_prefetcher_session

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
# --persistent-moe: the rows of a DSpark verify step (3 drafts + 1), the L1_SMALL its CCLs need (as
# tests/decode/test_dspark_spec.py opens the mesh), and the default --max-context (below the two-level select's
# 16384 compressed entries; the persistent layout has not run longer contexts).
_PERSISTENT_VERIFY_ROWS = 4
_PERSISTENT_L1_SMALL_SIZE = 16384
_PERSISTENT_MAX_CONTEXT = 16384
# Default --total-context with --persistent-moe: each instance's compressed-KV pool, shared by its running user and
# every conversation parked in its device RAM (any number of them, until the pool is full). ~13.4 GB per chip of the
# instance's TP4 row (~3.2 KB per token: 4 compressed groups of bf16 KV and index keys).
_PERSISTENT_TOTAL_CONTEXT = 4 * 1024 * 1024
# Total slots per persistent-MoE instance: 1 active slot at a time plus up to 255 device-parked slots sharing the
# instance's 4M-token KV pool. Once all 256 slots or the KV pool are full, the oldest slot is evicted to host RAM.
_PERSISTENT_SLOTS = 256
# Default --num-users with --persistent-moe and no --dspark (one stage holds every user's pools).
_NUM_USERS_PERSISTENT = 8
# Default --instances with --persistent-moe: one whole TP4 model (and its drafter) on each of mesh rows 4, 3, 2, 1, 0,
# all sharing the TP12 MoE on rows 5-7 (pmoe.INSTANCE_ROWS).
_INSTANCES = 5
# Default --dspark-drafts with several instances, by the users decoding on all of them: 4-row verify rounds (3 drafts)
# at every load. tests/decode/test_instances_dspark.py (GSM8K, 300 tokens, one user per instance, Nov 17) measured
# DSpark against one-row steps at 1-5 users decoding at once: 184.0 / 176.0 / 171.9 / 142.6 / 127.0 tok/s/user against
# 125.1 / 121.1 / 118.8 / 109.8 / 100.0 (635 against 500 tok/s in all with five), 3.25 tokens a round throughout. The
# shared MoE saturates from four users (verify 13.7 -> 16.8 ms a round) but DSpark still wins there.
_INSTANCE_DSPARK_DRAFTS = (3,)
# Fewer seated users raise the drafts only once they have stayed fewer this long, so a reply ending just before the
# next one is seated does not switch speculation on and off.
_DSPARK_RAISE_SECONDS = 2.0
# A request that continues no slot's conversation reuses a parked slot whose cache it differs from only in the
# cache's last this-many tokens, overwriting them, instead of taking a new slot (_V41SlotPool, _SlotReuse).
_SLOT_REUSE_TOKENS = 512
# Free pool room kept before each scheduler round for every active turn, besides its unfed prompt (_keep_headroom):
# more than a round writes (a verify round, posted-ahead decode steps, 16 replay blocks) plus a capacity grow-ahead.
_ROUND_HEADROOM_TOKENS = 2048

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

    def __init__(self, *args, max_slots: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.near_misses: set[int] = set()  # slots handed out for an in-place rewind, until their turn's admission
        # slot -> (index of the assistant message its last turn stored, the position that reply started at)
        self.reply_at: dict[int, tuple[int, int]] = {}
        self.max_slots = max(len(self._busy), max_slots or len(self._busy))
        self.max_device_parked = self.max_slots - len(self._busy)
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
        # When all active slots are occupied by idle conversations, assign the oldest free active slot and park its
        # previous conversation in device RAM (evicting the oldest to host RAM if all slots or the KV pool are full).
        if free and (self.engine.tokens_left() > 0 or self.device_parked):
            oldest = min(free, key=lambda i: self._used_at[i])
            return oldest, None
        return None

    def counts(self) -> dict:
        """How many slots are ``running`` / ``queued`` / ``parked`` / ``empty``, plus ``max_running`` / ``max_slots``."""
        with self._cv:
            running, queued = len(self._running), len(self._queue)
            busy = sum(1 for b in self._busy if b)
            free = [i for i, b in enumerate(self._busy) if not b]
            parked_active = sum(1 for i in free if self.engine.users[i].pos)
            parked = parked_active + len(self.device_parked)
            empty = max(0, self.max_slots - busy - parked)
            return {
                "running": running,
                "queued": queued,
                "parked": parked,
                "empty": empty,
                "max_running": self.max_running,
                "max_slots": self.max_slots,
            }

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

    def _round(self) -> None:
        self._keep_headroom()
        super()._round()

    def _keep_headroom(self) -> None:
        """Evict parked conversations (device-parked ones and idle slots, least recently used first, to host RAM)
        until the pools hold what the active turns may write before the next round: each one's unfed prompt and
        :data:`_ROUND_HEADROOM_TOKENS`. Admission makes room for the prompt only; the reply, prompt tokens fed
        through decode and verify rounds allocate blocks as they go, and a verify round that finds none raises
        ``PagedCacheFull`` part-way, which the scheduler cannot recover from (:meth:`_abort_all`)."""
        if not self._active:
            return
        need = sum(turn.prompt_left + _ROUND_HEADROOM_TOKENS for turn in self._active)
        if need <= self.engine.tokens_left():
            return
        # Evicting reads the parked KV back off the device: no step may sit there waiting for its packet.
        if not self._settle():
            return
        self._make_room(need, keep=-1)

    def _room_mid_step(self, n_tokens: int, turn: _Turn) -> bool:
        """A step of ``turn`` found the pools full before anything of it was posted: evict the least recently used
        conversations to host RAM there and then, for its ``n_tokens`` and :data:`_ROUND_HEADROOM_TOKENS`, so the
        turn goes on. Only while every step on the device already has its packet (none posted ahead) and no verify
        round is live, as the eviction's reads queue behind them. True when that freed room."""
        if self._posted or self.engine._replay_q or getattr(self, "_spec_live", None) is not None:
            return False
        before = self.engine.tokens_left()
        logger.info(
            f"user {turn.user_key!r} (slot {turn.slot}): the KV pool is full mid-turn ({before} tokens left), "
            "evicting the least recently used conversations to host RAM"
        )
        self._make_room(n_tokens + _ROUND_HEADROOM_TOKENS, keep=turn.slot)
        return self.engine.tokens_left() > before

    def _post(self, turn: _Turn, positions, token_ids=None) -> None:
        positions = list(positions)
        try:
            super()._post(turn, positions, token_ids)
        except PagedCacheFull:
            # post_traced grows the session's blocks before it posts any trace.
            if not self._room_mid_step(len(positions), turn):
                raise
            super()._post(turn, positions, token_ids)

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

    def _evict_device_parked_to_host(self, oldest_dev: _DeviceParkedConversation, reason: str) -> None:
        """Evict one device-parked conversation to host RAM (or free its blocks if the host tier is disabled)."""
        pool = self.server.pool
        if pool.host_users:
            exported = False  # export_device_park frees the parked KV blocks itself
            try:
                snapshot = self.engine.model.export_device_park(oldest_dev.snapshot)
                exported = True
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
                if not exported:
                    self.engine.model.free_device_park(oldest_dev.snapshot)
        else:
            self.engine.model.free_device_park(oldest_dev.snapshot)
        logger.debug(f"evicted device-parked user {oldest_dev.owner!r} ({reason}): {self._pool_summary()}")

    def _offload(self, slot: int, owner: str | None) -> None:
        if slot in self._reuse:
            return
        pool, user = self.server.pool, self.engine.users[slot]
        if not user.pos or len(user.tokens) != user.pos or slot in self._cut_short or pool.max_device_parked <= 0:
            super()._offload(slot, owner)
            return
        try:
            snapshot = self.engine.model.evict_session_to_device_park(user.sid)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"user {owner!r} (slot {slot}): could not park on device, offloading to host: {e}")
            super()._offload(slot, owner)
            return
        evicted: list[_DeviceParkedConversation] = []
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
            while len(pool.device_parked) > pool.max_device_parked:
                oldest = min(pool.device_parked, key=lambda c: c.used_at)
                pool.device_parked.remove(oldest)
                evicted.append(oldest)
        for oldest in evicted:
            self._evict_device_parked_to_host(oldest, f"all {pool.max_slots} slots full")
        logger.info(
            f"user {owner!r} (slot {slot}): {user.pos}-token conversation evicted from page table "
            f"(KV cache kept in device RAM; {len(pool.device_parked)}/{pool.max_device_parked} device-parked)"
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
                self._evict_device_parked_to_host(oldest_dev, f"for {n_tokens} tokens")
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
        if getattr(self.server, "prefix_cache", None) is not None and not self._inflight and not self._posted:
            self._capture_prefix(user)


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


class _MoeGate:
    """Instances sharing one persistent MoE take it in turn for a traced prefill (``--instances`` > 1).

    A prefill stops the shared decode server (:meth:`~..tt.prefill.persistent_moe.PersistentPrefillMoE.window`), which
    sends a stop from every requester row, so no instance may have a decode step posted or in flight while it runs.
    :meth:`exclusive` (the prefilling instance's scheduler thread, its own steps already settled) posts a park chore
    to every other instance's scheduler: a chore runs on that scheduler's thread right after its ``_settle()``, so it
    parks with nothing in flight, busy or idle, and resumes once the prefill is done. An instance waiting for its own
    turn at the gate has settled too, so it counts as parked: two instances prefilling at once take turns.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._holder = None
        self._parked: set = set()
        self.schedulers: list = []

    def _park(self, scheduler) -> None:
        with self._cv:
            while self._holder is not None and self._holder is not scheduler:
                self._parked.add(scheduler)
                self._cv.notify_all()
                self._cv.wait()
            self._parked.discard(scheduler)

    @staticmethod
    def _post_chore(scheduler, fn) -> None:
        """``scheduler.run_on_scheduler(fn)`` without waiting for it."""
        with scheduler._wake:
            scheduler._chores.append((fn, threading.Event(), []))
            scheduler._wake.notify_all()

    @contextlib.contextmanager
    def exclusive(self, scheduler):
        others = [s for s in self.schedulers if s is not scheduler]
        if not others:
            yield
            return
        with self._cv:
            while self._holder is not None:
                self._parked.add(scheduler)
                self._cv.notify_all()
                self._cv.wait()
            self._parked.discard(scheduler)
            self._holder = scheduler
        try:
            t0 = time.perf_counter()
            for other in others:
                self._post_chore(other, lambda other=other: self._park(other))
            with self._cv:
                while not all(other in self._parked for other in others):
                    self._cv.wait()
            logger.debug(f"MoE gate: {len(others)} other instance(s) parked in {time.perf_counter() - t0:.3f}s")
            yield
        finally:
            with self._cv:
                self._holder = None
                self._cv.notify_all()


class _PrefillFirst(_Scheduler):
    """Finish every pending prompt before decoding, and look Engram rows up while a prompt waits.

    A turn is admitted as soon as it holds a KV slot (:meth:`V41GenerationServer._await_seat`), not when a decode
    seat is free. While any turn is still prefilling, or one is waiting to be admitted, decode does not start
    another step: traces already posted are fed (they cannot be cancelled mid-dispatch) and then every prompt
    runs, the traced prefill one prompt at a time and the tokens replayed through decode pipelined. After that,
    the first ``--max-running`` turns in admission order decode together; the rest stay admitted, prompts already
    in the cache, until a seat frees.

    With disaggregated prefill (:class:`_RemotePrefiller`) a prompt is sent to the prefill server at admission
    (:meth:`_send_ahead`), and until its KV is back the turn neither holds decode up nor waits on the scheduler
    thread: the other turns decode meanwhile. Only loading the KV and the tail replayed through decode settle them.
    """

    def _admit(self, turn: _Turn) -> None:
        super()._admit(turn)
        if turn in self._active:
            if self.server.prefiller is not None and getattr(self.server, "prefix_cache", None) is not None:
                user = self.engine.users[turn.slot]
                self._remember_prefix(user, turn)
                # For disaggregated prefill, if there is an uncached shared history prefix just below
                # device_prefill (leaving a short tail <= 1024 tokens), trim device_prefill to the aligned
                # boundary before it so the first request populates prefix_cache for subsequent requests.
                prefiller = self.server.prefiller
                if (
                    isinstance(prefiller, _RemotePrefiller)
                    and turn.device_prefill > 0
                    and not turn.prefill_from
                    and turn.prefix_cuts
                ):
                    full_len = user.pos + len(turn.ids)
                    history_cuts = [c for c in turn.prefix_cuts if user.pos < c < full_len - 1]
                    if history_cuts:
                        history = max(history_cuts)
                        if history <= user.pos + turn.device_prefill:
                            aligned_h = prefiller.aligned(history - user.pos, user.pos)
                            max_trim_tail = 2048 if getattr(self.engine.model, "verify_rows", 1) > 1 else 1024
                            if full_len - (user.pos + aligned_h) <= max_trim_tail:
                                turn.device_prefill = aligned_h
            self._send_ahead(turn)
            self._prefetch_engram(turn)

    def _send_ahead(self, turn: _Turn) -> None:
        """Send the prompt a remote prefill takes to the prefill server now, from a thread of its own
        (:meth:`_RemotePrefiller.start`); :meth:`_prefill_on_device` loads the reply once it is in."""
        prefiller = self.server.prefiller
        if (
            not isinstance(prefiller, _RemotePrefiller)
            or not turn.device_prefill
            or turn.prefill_from
            or turn.replay_ids
        ):
            return
        user = self.engine.users[turn.slot]
        if user.pos % prefiller.align or len(user.tokens) < user.pos:
            return

        # Not mark_prefill_chunk: it advances turn.fed, which must stay at prefill_from until the KV is loaded.
        def progress(done: int) -> None:
            if turn.cancelled.is_set():
                raise _PromptStopped()

        turn.remote_fetch = prefiller.start(user, turn.ids[: turn.device_prefill], progress, self._wake)

    @staticmethod
    def _fetching(turn: _Turn) -> bool:
        """``turn``'s prompt is at the prefill server and its KV not back yet."""
        job = getattr(turn, "remote_fetch", None)
        return job is not None and not job.done()

    def _retire(self, turn: _Turn, outcome: str) -> None:
        job = getattr(turn, "remote_fetch", None)
        if job is not None:
            turn.remote_fetch = None
            self.server.prefiller.forget(self.engine.users[turn.slot], job)
        super()._retire(turn, outcome)

    def _share_prefix(self, user: UserSession, turn: _Turn) -> bool:
        if super()._share_prefix(user, turn):
            return True
        prefiller = self.server.prefiller
        cache = getattr(self.server, "prefix_cache", None)
        if user.pos or prefiller is None or cache is None or not hasattr(self.engine.model, "import_session"):
            return False
        vocab = getattr(getattr(self.engine.model, "config", None), "vocab_size", 0) or 0
        hit = cache.lookup(turn.ids, vocab)
        if hit is None:
            return False
        pos, _ = hit
        tail = len(turn.ids) - pos
        n_whole = prefiller.aligned(len(turn.ids))
        # Prefer the unpolluted block-replay rate (excludes import_session and prefix-capture exports) when available.
        rate = getattr(self, "_block_replay_rate", None) or self.replay_rate
        whole_seconds = (
            prefiller.seconds(n_whole) + (len(turn.ids) - n_whole) / rate if n_whole else len(turn.ids) / rate
        )
        if tail / rate >= whole_seconds:
            return False
        return self._share_decoded_prefix(user, turn)

    def _prompt_done(self, turn: _Turn) -> None:
        super()._prompt_done(turn)
        if self.server.prefiller is not None and getattr(self.server, "prefix_cache", None) is not None:
            pos = self.engine.users[turn.slot].pos
            # _remember_prefix already captures end - 1 during the decode tail; only snapshot at pos if
            # the prefiller covered the entire prompt with no decode tail.
            if pos > 0 and turn.prefilled_on_device >= len(turn.ids):
                turn.capture_at = pos

    def _prefill_waiting(self) -> bool:
        """A prompt is still to be prefilled, or one has been submitted and not yet admitted. A prompt still at the
        prefill server does not count: decode goes on until its KV is back.

        ``_new`` is appended under the scheduler lock and read here the way :meth:`_Scheduler._post_ahead` reads
        it: a request that arrived during this round."""
        if self._new:
            return True
        return any(
            turn.phase == _Turn.PREFILL and turn.error is None and not self._fetching(turn) for turn in self._active
        )

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

    def _choose_feed(self, user: UserSession, turn: _Turn, continuing: bool) -> None:
        prefiller = self.server.prefiller
        if isinstance(prefiller, _RemotePrefiller) and prefiller.vocab and any(t >= prefiller.vocab for t in turn.ids):
            return
        super()._choose_feed(user, turn, continuing)

    def _prefetch_engram(self, turn: _Turn) -> None:
        """Start the host Engram lookup for the tokens the traced prefill will see, on a copy of the session."""
        prefiller = self.server.prefiller
        lookup = getattr(self.engine.model, "_engram_lookup", None)
        if (
            prefiller is None
            or getattr(prefiller, "model", None) is None
            or lookup is None
            or not (turn.device_prefill or turn.replay_ids)
        ):
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
        gate = getattr(self.server, "moe_gate", None)
        try:
            # Several instances on one persistent MoE: the others parked while this one's prefill holds the MoE.
            with gate.exclusive(self) if gate is not None else contextlib.nullcontext():
                super()._prefill_on_device(turn, drain_seconds, drained)
            if isinstance(self.server.prefiller, _RemotePrefiller):
                turn.prefill_cached = self.server.prefiller.last_cached
        finally:
            if traced is not None:
                traced._engram_prefetch = None
            turn.engram_prefetch = None
            turn.remote_fetch = None

    def _drain_decode_queue(self) -> bool:
        """Feed the decode steps whose traces are already posted, and do not post any more.

        Returns whether the pipeline is still usable (``False`` once a readback has aborted every turn)."""
        if not self._posted and not self._inflight:
            return True
        # Every scheduler round while a prompt replays through decode (each round's steps are in flight): debug only.
        logger.debug(
            f"prefill waiting: finishing {len(self._posted)} posted decode step(s) "
            f"({len(self._inflight)} in flight) before prefilling"
        )
        return self._settle()

    def _prefill_only_round(self) -> None:
        """One round of prompt work only: traced prefills and prompt tokens replayed through decode."""
        self._stalled = False
        for turn in list(self._active):
            if turn not in self._active or turn.phase == _Turn.DECODE or self._fetching(turn):
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
        fetching = [turn for turn in self._active if self._fetching(turn)]
        for turn in fetching:
            if turn.cancelled.is_set():
                turn.remote_fetch.abort()
        if fetching and len(fetching) == len(self._active) and not self._inflight and not self._posted:
            # Nothing to run until a prompt's KV is back (which notifies _wake), a request or chore comes in, or a
            # turn is stopped (polled: a cancel does not notify).
            with self._wake:
                if not (self._new or self._chores or self._stop or any(turn.remote_fetch.done() for turn in fetching)):
                    self._wake.wait(0.1)
            return
        if self._prefill_waiting():
            if self._drain_decode_queue():
                self._prefill_only_round()
            return
        waiting = [
            turn
            for turn in self._active
            if turn.phase == _Turn.DECODE and not turn.cancelled.is_set() and not self._holds_decode_seat(turn)
        ]
        held = waiting + fetching
        if not held:
            super()._round_turns()
            return
        # Keep turns whose prompt is in but which have no decode seat out of this round, so a posted step of a
        # seated turn is not stalled on them and lookahead follows the seated turns only; and turns whose prompt is
        # still at the prefill server, so the round does not wait for it.
        order = list(self._active)
        self._active = [turn for turn in self._active if turn not in held]
        try:
            super()._round_turns()
        finally:
            kept = self._active
            self._active = [turn for turn in order if turn in kept or turn in held]
            self._active.extend(turn for turn in kept if turn not in order)


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
        self._block_replay_rate: float | None = None
        if getattr(self.engine.model, "verify_rows", 1) > 1:
            # 4-row block replay runs ~4x faster than single-row decode replay; seed both rates so _share_prefix
            # uses true block-replay speed without being dragged down by import_session or prefix-capture exports.
            self.replay_rate = max(self.replay_rate, 220.0)
            self._block_replay_rate = 220.0

    def _round(self) -> None:
        self._keep_headroom()
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

    def _block_replay_prompt(self, turn: _Turn) -> bool:
        """Replay prompt tail tokens in ``verify_rows``-token blocks via ``decode_verify_block`` (~4x faster than
        single-row decode replay), stopping before any prefix cut, image token, or the final prompt token."""
        model = self.engine.model
        t = getattr(model, "verify_rows", 1)
        if (
            t <= 1
            or not getattr(model, "_block_verify", False)
            or turn.cancelled.is_set()
            or turn.device_prefill
            or turn.replay_ids
        ):
            return False
        user = self.engine.users[turn.slot]
        # Leave at least the last prompt token for single-row decode so sampling and _prompt_done run identically.
        left = turn.prompt_left - 1
        if left < t:
            return False
        n = self._prefix_batch(turn, user, left)
        vocab = getattr(getattr(model, "config", None), "vocab_size", 0) or 0
        if vocab > 0:
            for i in range(n):
                if int(turn.ids[turn.fed + i]) >= vocab:
                    n = i
                    break
        n_blocks = n // t
        if n_blocks <= 0:
            return False
        if not self._settle():
            return True
        # Blocks ending at least a tail before the next state that is read -- a prefix snapshot or the prompt's end --
        # run only layers 0..20: the full blocks of that tail rewrite the later layers' window rings and the drafter's.
        kv_end = -1
        if getattr(model, "kv_only_blocks", False):
            tail = max(model.sliding_window, model.dspark.ring_rows if model.dspark is not None else 0)
            reads = [user.pos + turn.prompt_left] + [cut for cut in turn.prefix_cuts if cut > user.pos]
            if turn.capture_at is not None and turn.capture_at > user.pos:
                reads.append(turn.capture_at)
            kv_end = min(reads) - tail
        user.activate()
        ran = 0
        t0 = time.perf_counter()
        try:
            for _ in range(min(n_blocks, 16)):
                if turn.cancelled.is_set():
                    break
                chunk = [int(x) for x in turn.ids[turn.fed : turn.fed + t]]
                kv_only = user.pos + t <= kv_end
                try:
                    model.decode_verify_block(chunk, user.pos, kv_only=kv_only)
                except PagedCacheFull:
                    # From the block's capacity check, before its trace replays; every earlier block was read back.
                    if not self._room_mid_step(t, turn):
                        raise
                    model.decode_verify_block(chunk, user.pos, kv_only=kv_only)
                user.tokens.extend(chunk)
                user.pos += t
                turn.fed += t
                ran += t
        except PagedCacheFull as e:
            # From the block's capacity check, before its trace replays: only this turn's prompt is cut short.
            self._cut_short.add(turn.slot)
            self._fail(turn, e)
        except Exception as e:  # noqa: BLE001 - see _abort_all
            self._abort_all(e)
            return True
        finally:
            if ran:
                elapsed = time.perf_counter() - t0
                if elapsed > 0 and getattr(self, "_block_replay_rate", None) is not None:
                    self._block_replay_rate = 0.7 * self._block_replay_rate + 0.3 * (ran / elapsed)
                if hasattr(model, "_spec_dirty"):
                    model._spec_dirty.discard(user.index)
                self.steps += ran
                self._spec_accounted += ran
                self._account(ran)
        return True

    def _plan(self, turn: _Turn) -> list[tuple[int, int, bool]]:
        if turn.phase != _Turn.DECODE:
            if turn.phase == _Turn.PREFILL:
                user = self.engine.users[turn.slot]
                if self._capture_due(turn, user):
                    return []
                if self._block_replay_prompt(turn):
                    return []
            return super()._plan(turn)
        engine = self.engine
        user = engine.users[turn.slot]
        if self._capture_due(turn, user):
            return []
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

    def decoding_users(self) -> int:
        """This scheduler's seated decoding users (read from other instances' threads too, unlocked, as the live
        console reads it)."""
        return sum(
            1
            for turn in list(self._active)
            if turn.phase == _Turn.DECODE and not turn.cancelled.is_set() and self._holds_decode_seat(turn)
        )

    def _pick_drafts(self) -> None:
        """Set this scheduler round's drafts per verify round from ``--dspark-drafts`` by the seated decoding users:
        fewer as soon as more users decode, more only once fewer have for :data:`_DSPARK_RAISE_SECONDS` (at once
        with none decoding, when there is nothing to switch). With several instances on one persistent MoE the
        count is every instance's (they share the MoE, which a 4-row round loads more than a one-row step), and an
        entry is 0 (one-row steps) or 3 (4-row verify rounds)."""
        schedule = self.engine.dspark_drafts
        mine = self.decoding_users()
        peers = [p.scheduler for p in getattr(self.server, "peers", ()) if p is not self.server]
        users = mine + sum(p.decoding_users() for p in peers if isinstance(p, _SpecScheduler))
        target = schedule[min(max(users, 1), len(schedule)) - 1]
        if target > self._drafts and mine:
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
        pos = engine.users[turn.slot].pos
        room = engine.max_seq - pos - 2 * (self._drafts + 1)
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
        if self.engine.block_verify:
            # The persistent stage verifies a round in one 4-row step and samples it greedily on device.
            spec = DSparkSpeculator(self.engine.model, self._drafts, parallel_verify=True, temperature=0.0)
        else:
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
        self._capture_due(turn, user)
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
        for turn in turns:
            self._capture_due(turn, engine.users[turn.slot])
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

    def __init__(self, *args, monitor_feed: bool = True, files: "_FileStore | None" = None, cpus=None, **kwargs):
        """``monitor_feed`` publishes this server's state for the web monitor (with several instances only the first
        publishes, and its snapshot covers every instance, see :meth:`monitor_snapshot`);
        ``files`` shares another server's upload store; ``cpus`` pins the host runtime's threads (default
        :func:`_host_cpus`)."""
        if kwargs.get("prefiller") is None and args and getattr(args[0], "prefill_url", None):
            engine = args[0]
            kwargs["prefiller"] = _RemotePrefiller(
                engine,
                engine.prefill_url,
                max_len=getattr(engine, "prefill_max_len", None),
                push=getattr(engine, "prefill_push", False),
                connect_wait=getattr(engine, "prefill_wait", 0.0),
            )
        if (
            kwargs.get("prefix_cache") is None
            and args
            and isinstance(kwargs.get("prefiller"), _RemotePrefiller)
            and hasattr(args[0].model, "import_session")
        ):
            host_bytes = kwargs.get("host_bytes", 0)
            kwargs["prefix_cache"] = _PrefixCache(host_bytes if host_bytes > 0 else 32 << 30)
        super().__init__(*args, **kwargs)
        # Every instance's server, this one included (set by _Instances): the DSpark schedule counts their users.
        self.peers = [self]
        pool = self.pool
        max_slots = _PERSISTENT_SLOTS if getattr(self.engine, "persistent_moe", False) else len(self.engine.users)
        self.pool = _V41SlotPool(
            self.engine,
            pool.host_users,
            pool.host_bytes,
            pool.max_running,
            pool.idle_timeout,
            max_slots=max_slots,
        )
        scheduler = _SpecScheduler if self.engine.dspark else _V41Scheduler
        self.scheduler = scheduler(self, self.scheduler.prefill_chunk, self.scheduler.decode_ahead)
        # serve() builds the server after the warmup and the prefill snapshot, and before the scheduler starts: the
        # point past every other device allocation of startup.
        self.engine.capture_vision()
        self.monitor_feed = None
        self._monitor_sink = None
        if host_rs.enabled():
            # After every capture: from here the decode step's replays, packets, Engram rows and outputs run on the
            # Rust host runtime's threads, which keep their counters where the web monitor reads them. Several
            # instances each write ``<prefix>.<j>.counters``; the monitor sums them. Only the first publishes the
            # JSON snapshot (monitor_feed), and that snapshot already covers every instance.
            prefix = monitor.default_prefix()
            n = getattr(self.engine, "instances", 1)
            slot = self.engine.instance if n > 1 else None
            self.engine.model.start_host_runtime(
                _host_cpus() if cpus is None else cpus,
                counters_path=monitor.counters_path(prefix, slot) if monitor_feed or n > 1 else None,
            )
            if monitor_feed:
                self.monitor_feed = monitor.MonitorFeed(
                    self.monitor_snapshot,
                    monitor.snapshot_path(prefix),
                    period=float(os.environ.get("DEEPSEEK_V41_MONITOR_PERIOD", "1.0")),
                )
        cache = self.engine.weight_cache
        self.files = files if files is not None else _FileStore(os.path.join(cache.path, "files") if cache else None)

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
        if self.prefiller is not None and hasattr(self.prefiller, "close"):
            self.prefiller.close()

    def monitor_snapshot(self) -> dict:
        """The raw state :class:`~.monitor.MonitorFeed` publishes for the web monitor: counts, sizes and ages,
        read without locking the scheduler, as the live console reads them. Every rate is the monitor's to work out,
        in its own process.

        With several instances (:class:`_Instances`) only the first publishes, and its snapshot covers them all:
        every instance's turns -- a prompt replayed through decode on any of them shows its ``prefilled`` progress --
        slots, ended turns and totals. The slots are numbered across the instances (instance j's slot i is
        ``i`` plus the slots of the instances before it), so the monitor's slot table tells them apart."""
        now = time.perf_counter()
        users, turns, ended = [], [], []
        totals = dict.fromkeys(("total_turns", "total_prefilled", "total_generated", "total_input", "total_cached"), 0)
        slot_states: dict = {}
        kv_pool: dict = {}
        host_users = host_bytes = max_host_users = max_host_bytes = 0
        inflight = rounds = steps = tokens_left = speculating = 0
        tokens_left_min = None
        broken = None
        offset = 0
        for j, server in enumerate(self.peers):
            scheduler, pool = server.scheduler, server.pool
            # Before the active turns: see _Scheduler._retire. The newest record carries this instance's totals.
            records = list(scheduler.ended_turns)
            if records:
                for key in totals:
                    totals[key] += records[-1].get(key, 0)
            ended += [
                {**record, "slot": None if record["slot"] is None else record["slot"] + offset, "instance": j}
                for record in records[-monitor.FEED_ENDED_TURNS :]
            ]
            specs = dict(getattr(scheduler, "_specs", {}))  # turn -> its DSpark speculator, while it speculates
            speculating += len(specs)
            turns += [
                {
                    "serial": turn.serial,
                    "user": turn.user_key,
                    "slot": None if turn.slot is None else turn.slot + offset,
                    "instance": j,
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
            ]
            users += [
                {**row, "index": None if row.get("index") is None else row["index"] + offset, "instance": j}
                for row in pool.rows()
            ]
            for key, value in pool.counts().items():
                slot_states[key] = slot_states.get(key, 0) + value
            for name, (used, total) in scheduler.pool_usage().items():
                prev_used, prev_total = kv_pool.get(name, (0, 0))
                kv_pool[name] = (prev_used + used, prev_total + total)
            n_users, n_bytes = pool.host_usage()
            host_users += n_users
            host_bytes += n_bytes
            max_host_users += pool.host_users
            max_host_bytes += pool.host_bytes
            inflight += len(scheduler._inflight)
            rounds += scheduler.rounds
            steps += scheduler.steps
            left = server.engine.tokens_left()
            tokens_left += left
            tokens_left_min = left if tokens_left_min is None else min(tokens_left_min, left)
            if broken is None and scheduler._broken is not None:
                broken = f"instance {j}: {scheduler._broken}" if len(self.peers) > 1 else str(scheduler._broken)
            offset += len(server.engine.users)
        ended.sort(key=lambda record: record["ended"])
        ended = ended[-monitor.FEED_ENDED_TURNS :]
        if ended:  # the monitor reads the running totals off the newest record: make them every instance's
            ended[-1] = {**ended[-1], **totals}
        snapshot = {
            "t": time.time(),
            "model_id": self.model_id,
            "started": self._created,
            "max_seq": self.engine.max_seq,
            "slots": offset,
            "instances": len(self.peers),
            "slot_states": slot_states,
            "users": users,
            "turns": turns,
            "inflight": inflight,
            "rounds": rounds,
            "steps": steps,
            "ended_turns": ended,
            "pool": kv_pool,
            "tokens_left": tokens_left,
            "tokens_left_min": 0 if tokens_left_min is None else tokens_left_min,
            "host": {
                "users": host_users,
                "bytes": host_bytes,
                "max_users": max_host_users,
                "max_bytes": max_host_bytes,
            },
            "broken": broken,
        }
        if self.engine.images is not None:
            snapshot["images"] = {**self.engine.images.stats(), "files": self.files.usage()}
        if self.engine.dspark:
            # Each scheduler picks drafts from the same load (every instance's decoding users) on its own round,
            # so they agree once a switch has landed. Until then the monitor shows the fewer.
            drafts = [getattr(server.scheduler, "_drafts", 0) for server in self.peers]
            snapshot["dspark"] = {
                "drafts": min(drafts) if drafts else 0,
                "schedule": list(self.engine.dspark_drafts),
                "speculating": speculating,
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


class _PoolsView:
    """The handler's view of every instance's slot pool at once (``/health``)."""

    def __init__(self, servers: list):
        self._servers = servers

    def counts(self) -> dict:
        total: dict = {}
        for server in self._servers:
            for key, value in server.pool.counts().items():
                total[key] = total.get(key, 0) + value
        return total

    def host_usage(self) -> tuple[int, int]:
        usages = [server.pool.host_usage() for server in self._servers]
        return sum(u[0] for u in usages), sum(u[1] for u in usages)

    def rows(self) -> list[dict]:
        return [{**row, "instance": j} for j, server in enumerate(self._servers) for row in server.pool.rows()]


class _TurnsView:
    """The handler's view of every instance's scheduler at once (``/health``'s ``active_turns``)."""

    def __init__(self, servers: list):
        self._servers = servers

    @property
    def active_turns(self) -> int:
        return sum(server.scheduler.active_turns for server in self._servers)


class _Instances:
    """Several :class:`V41GenerationServer` s -- one per model instance on the shared persistent MoE
    (:func:`_serve_instances`) -- behind the one HTTP server, as the handler's ``api``.

    A request goes to the instance holding its conversation (a slot ``user_key`` owns, or whose stored messages the
    request continues) and stays there: a 429 from that instance is the answer, not a reason to start the same
    conversation on another instance. A new conversation goes to the idlest instance (an empty slot, then a parked
    one, then the fewest running and queued turns), and an instance that turns it away with a 429 before replying
    passes it to the next. Each instance keeps its own sessions, scheduler and Rust host runtime; the DSpark
    schedule counts every instance's decoding users (``peers``)."""

    def __init__(self, servers: list):
        self.servers = list(servers)
        first = self.servers[0]
        for server in self.servers:
            server.peers = self.servers
        self.model_id = first.model_id
        self.max_body_bytes = first.max_body_bytes
        self._created = first._created
        self.files = first.files
        self.pool = _PoolsView(self.servers)
        self.engine = SimpleNamespace(
            users=[user for server in self.servers for user in server.engine.users],
            images=first.engine.images,
            max_seq=first.engine.max_seq,
            max_new_tokens=first.engine.max_new_tokens,
            chat_format=first.engine.chat_format,
        )
        self.scheduler = _TurnsView(self.servers)
        self._lock = threading.Lock()

    def __getattr__(self, name):
        # Request parsing, uploads and the like are the same on every instance: the first one's.
        return getattr(self.servers[0], name)

    def start(self) -> None:
        for server in self.servers:
            server.start()

    def stop(self) -> None:
        for server in self.servers:
            server.stop()

    def _order(self, user_key: str, messages: list[dict]) -> list:
        def rank(j: int):
            server = self.servers[j]
            pool = server.pool
            holds = self._holds(server, user_key, messages)
            parked_hit = any(
                conv.messages and _continues(messages, conv.messages) for conv in getattr(pool, "device_parked", ())
            )
            counts = pool.counts()
            busy = counts.get("running", 0) + counts.get("queued", 0)
            # "empty" counts the device-parked capacity too, so an instance holding idle conversations still looks
            # free: without the parked count, new conversations pile onto the lowest j and then contend for its
            # one active session.
            return (
                not holds,
                not parked_hit,
                busy > 0,
                counts.get("empty", 0) == 0,
                busy,
                counts.get("parked", 0),
                j,
            )

        with self._lock:
            return [self.servers[j] for j in sorted(range(len(self.servers)), key=rank)]

    def _holds(self, server, user_key: str, messages: list[dict]) -> bool:
        """Whether ``server`` already has ``user_key``'s conversation: a slot a named ``user_key`` owns, or stored
        messages this request continues. That instance is the only one that may serve the request."""
        pool = server.pool
        if user_key != _DEFAULT_USER and pool.slots_of(user_key):
            return True
        return any(user.messages and _continues(messages, user.messages) for user in server.engine.users)

    def generate(self, user_key: str, body: dict, on_chunk, request_id: str | None = None, client_gone=None) -> dict:
        messages = self.servers[0].normalized_messages(body)
        replied = [False]

        def relay(reasoning, content):
            replied[0] = True
            on_chunk(reasoning, content)

        order = self._order(user_key, messages)
        # A conversation stays on the instances that hold it (usually one). A new one tries the idlest, then
        # the next, when an instance turns it away with a 429 before replying.
        sticky = [server for server in order if self._holds(server, user_key, messages)]
        error = None
        for server in sticky or order:
            try:
                return server.generate(user_key, body, relay, request_id, client_gone)
            except RequestError as e:
                if e.status != 429 or replied[0]:
                    raise
                error = e
        raise error

    def stats(self) -> dict:
        stats = self.servers[0].stats()
        stats["instances"] = [server.stats() for server in self.servers]
        return stats

    def session_rows(self) -> list[dict]:
        return self.pool.rows()

    def reset_user(self, user_key: str) -> None:
        for server in self.servers:
            server.reset_user(user_key)

    def cancel(self, request_id: str) -> bool:
        return any([server.cancel(request_id) for server in self.servers])

    def cancel_user(self, user_key: str) -> int:
        return sum(server.cancel_user(user_key) for server in self.servers)

    def cancel_all(self) -> int:
        return sum(server.cancel_all() for server in self.servers)


class _LockedExitStack:
    """``ExitStack.enter_context`` / ``callback`` from the instance-load threads.

    Each instance registers its own callbacks in order, and the stack is one sequence, so an instance's later
    callback stays above its earlier ones whichever way the threads interleave. The lock only covers the push.
    """

    def __init__(self, stack: contextlib.ExitStack):
        self._stack = stack
        self._lock = threading.Lock()

    def enter_context(self, cm):
        with self._lock:
            return self._stack.enter_context(cm)

    def callback(self, callback, *args, **kwargs):
        with self._lock:
            return self._stack.callback(callback, *args, **kwargs)


@torch.no_grad()
def _serve_instances(args) -> int:
    """:func:`serve` for ``--instances`` > 1: open the 8x4 mesh, launch one persistent MoE for every instance
    (:func:`~..tt.decode.persistent_moe.launch_instances_server`), build, warm up and serve each instance -- a whole
    TP4 model and its DSpark drafter on mesh row 4 - j -- behind one HTTP server (:class:`_Instances`).

    The instances load together, one thread per mesh row (each row has its own allocator and command queues). They
    share one Engram table mapping. Trace capture stays one instance at a time: a capture selects the shared MoE's
    request mode.
    """
    from models.experimental.deepseek_v4_flash.demo import server as v4_server

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logger.remove()
    logger.add(sys.stderr, level="WARNING" if args.quiet else "DEBUG" if args.debug else "INFO")
    torch.manual_seed(0)
    v4_server._LOG_FULL_PROMPT[0] = not args.prompt_snippet
    n = args.instances
    logger.info(
        f"{n} instances on mesh rows {list(pmoe.INSTANCE_ROWS[:n])}, sharing the persistent MoE on rows 5-7: "
        f"{args.num_users} session(s) x {args.max_context} tokens each"
        + (f", DSpark drafts {','.join(map(str, args.dspark_drafts))} by users decoding" if args.dspark else "")
    )
    loader = DeepseekV4WeightLoader(args.model_dir)
    config = load_config(loader.snapshot_dir)
    num_layers = min(args.num_layers or config.num_hidden_layers, config.num_hidden_layers)
    model_name = os.path.basename(os.path.normpath(str(args.model_dir)))
    cache = WeightCache(os.path.join(args.cache_dir, model_name)) if args.cache_dir else None
    args.tp_size, args.num_stages = _TP_SIZE, 1
    with v4_server.open_mesh_device(
        args.trace_region_size,
        system_config=args.system_config,
        mesh_shape=pmoe.GALAXY_MESH,
        l1_small_size=args.l1_small_size,
    ) as mesh_device:
        with contextlib.ExitStack() as stack:
            # The server first, before any model allocates on the requester rows; stopped last (the stack unwinds
            # LIFO), after every instance's replay threads and prefetcher sessions.
            t0 = time.perf_counter()
            server = pmoe.launch_instances_server(
                mesh_device, config, loader, cache, num_layers, n, dspark=args.dspark, prefill=args.prefill
            )
            stack.callback(server.stop)
            logger.info(f"persistent MoE up in {time.perf_counter() - t0:.0f}s")
            engram_ids = [i for i in config.engram_layer_ids if i < num_layers]
            # One mapping for every instance. Each runtime builds its own lookup (its own session histories) over it.
            engram_tables = load_engram_tables(loader, engram_ids) if engram_ids else None
            locked = _LockedExitStack(stack)

            def load_instance(j: int):
                t0 = time.perf_counter()
                engine = V41ChatEngine(
                    mesh_device,
                    args,
                    locked,
                    moe_server=server,
                    instance=j,
                    engram_tables=engram_tables,
                )
                logger.info(
                    f"instance {j} (mesh row {pmoe.INSTANCE_ROWS[j]}) loaded in {time.perf_counter() - t0:.0f}s"
                )
                return engine

            logger.info(f"loading {n} instances in parallel")
            # transformers' lazy module is not thread-safe: resolve AutoTokenizer here, before the loaders race for it.
            from transformers import AutoTokenizer  # noqa: F401

            with ThreadPoolExecutor(max_workers=n, thread_name_prefix="instance-load") as pool:
                engines = list(pool.map(load_instance, range(n)))
            prefillers = []
            for j, engine in enumerate(engines):
                t0 = time.perf_counter()
                # Compiled and captured one instance at a time: a capture picks the server's request mode, which the
                # instances share (see DeepSeekV41Model._use_server_rows). Each instance's traced prefill is its own
                # (its row's stage, its requester of the prefill MoE): decode compile, then the prefill's build,
                # compile (the decode server stopped meanwhile) and capture, then the decode capture and the snapshot.
                if args.prefill_url:
                    prefiller = _RemotePrefiller(
                        engine,
                        args.prefill_url,
                        max_len=args.prefill_max_len,
                        push=args.prefill_push,
                        connect_wait=args.prefill_wait,
                    )
                    stack.callback(prefiller.close)
                else:
                    prefiller = _V41Prefiller(engine, args, locked) if args.prefill else None
                engine.warmup()
                if prefiller is not None:
                    prefiller.snapshot()
                prefillers.append(prefiller)
                logger.info(
                    f"instance {j} (mesh row {pmoe.INSTANCE_ROWS[j]}) captured in {time.perf_counter() - t0:.0f}s"
                )
            cpus = _host_cpus()
            per = len(cpus) // n if cpus else 0
            servers = []
            # One prefix table for every instance: a prompt replayed on any row can resume on any other.
            host_bytes = int(args.host_cache_gb * 2**30)
            prefix_cache = v4_server._PrefixCache(host_bytes if host_bytes > 0 else 32 << 30)
            for j, engine in enumerate(engines):
                servers.append(
                    V41GenerationServer(
                        engine,
                        args.model_id,
                        args.max_body_bytes,
                        args.prefill_chunk,
                        prefiller=prefillers[j],
                        decode_ahead=args.decode_ahead,
                        temperature=args.temperature,
                        host_users=args.host_users,
                        host_bytes=host_bytes,
                        max_running=args.max_running,
                        slot_idle_timeout=args.slot_idle_timeout,
                        think=args.think,
                        reasoning_effort=args.reasoning_effort,
                        prefix_cache=prefix_cache,
                        monitor_feed=j == 0,
                        files=servers[0].files if servers else None,
                        # DEEPSEEK_V41_HOST_CPUS split evenly over the instances' host runtimes.
                        cpus=cpus[j * per : (j + 1) * per] if per else [],
                    )
                )
            if args.prefill:
                gate = _MoeGate()
                gate.schedulers = [s.scheduler for s in servers]
                for s in servers:
                    s.moe_gate = gate
            api = _Instances(servers)
            sys.setswitchinterval(v4_server._GIL_SWITCH_SECONDS)
            api.start()
            httpd = v4_server._ModelHTTPServer((args.host, args.port), _V41Handler, api)
            logger.info(
                f"serving {args.model_id!r} on http://{args.host}:{args.port} ({n} instances x "
                f"{args.num_users} session(s), context {engines[0].max_seq}/user, "
                + (
                    f"disaggregated prefill via {args.prefill_url})"
                    if args.prefill_url
                    else (
                        "traced prefill on each instance's row, MoE on the shared prefill MoE)"
                        if args.prefill
                        else "prompts replayed through decode)"
                    )
                )
            )
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                logger.info("interrupted, shutting down")
            finally:
                httpd.server_close()
                api.stop()
    return 0


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

    def __init__(
        self,
        mesh_device,
        args,
        prefetcher: contextlib.ExitStack,
        moe_server=None,
        instance: int = 0,
        engram_tables=None,
    ):
        """``moe_server`` (a persistent MoE launched for several instances, :func:`_serve_instances`) makes this engine
        instance ``instance``: a whole model on that requester's mesh row, its DSpark drafter on the same row.

        ``engram_tables`` is the shared n-gram mapping (:func:`~..tt.model.load_engram_tables`). Several instances
        pass the same dict; each lookup keeps its own session histories. ``prefetcher`` may be a
        :class:`_LockedExitStack` when instances load on several threads.
        """
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
        # --persistent-moe: one stage, every MoE on the persistent server; with DSpark, 4-row verify steps.
        self.persistent_moe = bool(getattr(args, "persistent_moe", False)) or moe_server is not None
        self.dspark = bool(getattr(args, "dspark", False))
        self.prefill_url = getattr(args, "prefill_url", None)
        self.prefill_max_len = getattr(args, "prefill_max_len", None)
        self.prefill_push = bool(getattr(args, "prefill_push", False))
        self.prefill_wait = float(getattr(args, "prefill_wait", 0.0))
        self.instance = instance
        self.instances = 1 if moe_server is None else len(moe_server.request_meshes)
        persistent = {}
        if self.persistent_moe:
            persistent = dict(
                persistent_moe=True,
                verify_rows=_PERSISTENT_VERIFY_ROWS if self.dspark else 1,
                balance_dram_banks=os.environ.get("DEEPSEEK_V4_BALANCE_DRAM_BANKS", "1")
                not in ("0", "", "false", "False"),
            )
        if moe_server is not None:
            # One of several instances: requester `instance` of the shared server, on its row, drafter included.
            persistent.update(moe_server=moe_server, moe_requester=instance, drafter_on_stage=self.dspark)
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
            # The persistent layout places its own stage (row 4, or the shared server's requester row).
            submeshes=None if self.persistent_moe else getattr(args, "submeshes", None),
            use_prefetcher=getattr(args, "use_prefetcher", None),
            layer_submesh_ids=None if self.persistent_moe else plan_layer_placement(num_layers, num_stages, 1),
            image_rows=args.image_rows if vision else 0,
            vision_stage=VISION_STAGE if vision else None,
            prefill_layer_submesh_ids=prefill_placement,
            **persistent,
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
        if self.persistent_moe:
            row = pmoe.INSTANCE_ROWS[instance] if moe_server is not None else pmoe.REQUESTER_ROW
            logger.info(
                f"built DeepSeekV41Model{f' instance {instance}' if moe_server is not None else ''} with "
                f"{self.model.num_layers}/{config.num_hidden_layers} layers: attention on one TP{self.model.tp_size} "
                f"stage (mesh row {row}), every MoE on the persistent TP12 server (rows 5-7), verify rows "
                f"{self.model.verify_rows}, context {self.max_seq} tokens"
            )
        else:
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
            engram_tables=engram_tables,
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

        # Drafts per verify round with 1, 2, ... seated decoding users, the last entry for more (--dspark-drafts).
        self.dspark_drafts = tuple(getattr(args, "dspark_drafts", _DSPARK_DRAFTS))
        # The persistent stage's 4-row verify rounds (DSpark on --persistent-moe).
        self.block_verify = self.dspark and self.persistent_moe
        if self.dspark:
            # Before anything is compiled; the drafter's prefetcher rings are on its own row (none on the stage's).
            self.model.prepare_spec_decode(self.rope["main"])
            if self.model.use_prefetcher and not self.model.drafter_on_stage:
                prefetcher.enter_context(tensor_prefetcher_session(self.model.dspark.device))
            if self.block_verify:
                where = (
                    f"the drafter on the instance's own row {pmoe.INSTANCE_ROWS[instance]}"
                    if self.model.drafter_on_stage
                    else "the drafter on mesh row 3"
                )
                logger.info(
                    f"DSpark speculative decode: {self.model.verify_rows - 1} drafts + 1 verified in one "
                    f"{self.model.verify_rows}-row step, greedy, {where} (its MoE on the persistent server)"
                )
            else:
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


class _RemoteFetch:
    """One prompt sent to the prefill server from a thread of its own (:meth:`_RemotePrefiller.start`).

    The scheduler keeps decoding the other turns meanwhile; :meth:`_RemotePrefiller.run` takes :meth:`result` and
    only the KV load touches the device. ``wake`` (the scheduler's condition) is notified when the reply is in.
    """

    def __init__(
        self, prefiller: _RemotePrefiller, ids: list[int], have: int, on_progress, wake: threading.Condition | None
    ):
        self.ids = list(ids)  # the whole prompt from position 0
        self.have = have
        self.t0 = time.perf_counter()
        self.t_done = 0.0
        self.aborted = False
        self.client: PrefillClient | None = None  # the connection carrying the request, for abort()
        self._prefiller = prefiller
        self._on_progress = on_progress
        self._wake = wake
        self._states: dict | None = None
        self._error: BaseException | None = None
        self._done = threading.Event()
        threading.Thread(target=self._run, name="remote-prefill", daemon=True).start()

    def _run(self) -> None:
        try:
            self._states = self._prefiller._fetch(self.ids, self.have, self._on_progress, self)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the scheduler thread by result()
            self._error = exc
        finally:
            self.t_done = time.perf_counter()
            self._done.set()
            if self._wake is not None:
                with self._wake:
                    self._wake.notify_all()

    def done(self) -> bool:
        return self._done.is_set()

    def abort(self) -> None:
        """Stop waiting for the reply: the connection is shut down, so the thread ends and :meth:`result` raises
        :class:`_PromptStopped`."""
        self.aborted = True
        client = self.client
        if client is not None and not self.done():
            with contextlib.suppress(OSError):
                client.sock.shutdown(socket.SHUT_RDWR)

    def result(self) -> dict:
        self._done.wait()
        if self.aborted:
            raise _PromptStopped()
        if self._error is not None:
            raise self._error
        return self._states


class _RemotePrefiller:
    """Disaggregated prefill over TCP (:class:`~..tt.prefill.d_p_wire.PrefillClient`).

    Sends the aligned prefix of a prompt to a remote prefill server, receives the bf16 KV cache entries over TCP,
    and loads them into the decode device's KV cache via :meth:`DeepSeekV41Model.commit_host_prefill_state`.

    The server keeps recent prompts' state in prefix-cache slots (protocol version 2): a prompt that starts with one
    it kept (the next turn of a conversation, a retry) computes only the rest, and the reply's ``cached`` says how
    many tokens it skipped. What the reply carries is decided by ``have`` alone: the whole prompt from position 0 is
    always sent, with ``have`` the leading tokens whose entries this session already holds (``user.pos`` when the
    prefill continues the session, else 0), and the reply holds the compressed entries from ``have`` on. A session
    is continued from any position on the :attr:`align` grid (``continue_alignment``); a window replay from an
    earlier point (``bounded``) is not offered, since the server's hits are exact continuations only.

    A connection carries one request at a time, so concurrent prompts each take one from a pool of idle
    connections (a new one when none is idle). :meth:`start` sends a prompt from a thread of its own while the
    scheduler keeps decoding; :meth:`run` then only waits for what is left of it and loads the KV.
    """

    # Signals to _PrefillFirst._prefetch_engram that there is no local traced prefill model.
    model = None

    def __init__(
        self,
        engine: V41ChatEngine,
        url: str,
        max_len: int | None = None,
        timeout: float = 3600.0,
        push: bool = False,
        connect_wait: float = 0.0,
    ):
        """``push``: the server writes the KV cache into this server's device memory over TT-Fabric (protocol
        ``prefill_push``) instead of sending it; both servers must be ranks of one fabric job. ``connect_wait``:
        seconds to keep retrying a server that is not listening yet (one started with this one, still loading)."""
        self.engine = engine
        self.url = url
        self.timeout = timeout
        self.push = push
        self._lock = threading.Lock()
        self._idle: list[PrefillClient] = []
        self._closed = False
        self._fetches: dict[int, _RemoteFetch] = {}  # user index -> its prompt in flight (start)
        deadline = time.monotonic() + connect_wait
        while True:
            try:
                client = PrefillClient(self.url, timeout=self.timeout)
                break
            except ConnectionRefusedError as exc:
                if time.monotonic() >= deadline:
                    raise
                logger.info(f"prefill server {self.url} not listening yet ({exc}); retrying")
                time.sleep(10.0)
        self._idle.append(client)
        self.address = client.address
        info = client.info
        self.info = info

        decode_model = engine.model
        self.window = int(info["window"])
        if self.window != decode_model.sliding_window:
            self.close()
            raise ValueError(
                f"prefill server {self.address} has window {self.window}, decode model has {decode_model.sliding_window}"
            )
        server_kv_sources = {int(k): int(v) for k, v in info["kv_sources"].items()}
        for src in decode_model.config.kv_source_layer_ids:
            if src >= decode_model.num_layers:
                continue
            want_ratio = int(decode_model.config.compress_ratios[src])
            got_ratio = server_kv_sources.get(src)
            if got_ratio != want_ratio:
                self.close()
                raise ValueError(
                    f"prefill server {self.address} kv_source layer {src} ratio {got_ratio} != decode ratio {want_ratio}"
                )

        self.align = math.lcm(int(info["align"]), _PREFILL_ALIGNMENT, self.window)
        self.chunk_size = int(info.get("chunk") or self.align)
        self.vocab = int(info.get("vocab") or 0)
        server_max = int(info["max_seq"])
        cap = min(max_len or server_max, server_max, engine.max_seq - 1)
        self.max_len = cap // self.align * self.align
        if self.max_len < self.align:
            self.close()
            raise ValueError(
                f"disaggregated prefill needs a context of at least {self.align + 1} tokens "
                f"(server max_seq={server_max}, decode max_seq={engine.max_seq})"
            )
        announced_layers = {int(i) for i in info["layers"]}
        self.kv_only = announced_layers != set(range(decode_model.num_layers))
        # The rings a push writes: the server's layers (a layer it does not run gets zeros, as its host states do).
        self.ring_layers = sorted(li for li in announced_layers if li < decode_model.num_layers)
        # Leave sliding_window tokens for decode when the prefill server is KV-only or DSpark drafter rings need them.
        self.decode_window = self.kv_only or bool(getattr(engine, "dspark", False))
        self.fill_seconds = 0.5
        self.chunk_seconds = 0.15
        self.t_prefill_out = 0.0
        self.last_cached = 0  # of the last run's tokens, how many the server took from its prefix cache
        logger.info(
            f"disaggregated prefill connected to {self.address}: up to {self.max_len} tokens "
            f"(align {self.align}, chunk {self.chunk_size}, {len(announced_layers)}/{decode_model.num_layers} layers"
            f"{', kv-only' if self.kv_only else ''}, tail {self.window if self.decode_window else 1} via decode, "
            f"{info.get('cache_slots', 0)} prefix-cache slot(s) on the server"
            f"{', KV pushed over TT-Fabric' if self.push else ''})"
        )

    @property
    def replay_window(self) -> int:
        return self.window

    @property
    def continue_alignment(self) -> int:
        return self.align

    def snapshot(self) -> None:
        pass

    def _take_client(self) -> PrefillClient:
        with self._lock:
            if self._idle:
                return self._idle.pop()
        return PrefillClient(self.url, timeout=self.timeout)

    def _give_client(self, client: PrefillClient) -> None:
        with self._lock:
            if not self._closed:
                self._idle.append(client)
                return
        with contextlib.suppress(Exception):
            client.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
            fetches, self._fetches = list(self._fetches.values()), {}
        for job in fetches:
            job.abort()
        for client in idle:
            with contextlib.suppress(Exception):
                client.close()

    def _fetch(
        self,
        ids: list[int],
        have: int = 0,
        on_progress=None,
        job: _RemoteFetch | None = None,
        targets: dict | None = None,
    ) -> dict:
        """The KV entries of ``ids`` (the whole prompt) from ``have`` on. A server that refuses ``have`` is
        answered by asking again for every entry (``have`` 0, the reply's ``base`` says which it got); not with
        push ``targets``, which name the pages of the entries from ``have`` on."""
        try:
            return self._request(ids, have, on_progress, job, targets)
        except PrefillServerError as exc:
            if not have or "have" not in str(exc) or targets is not None:
                raise
            logger.error(
                f"prefill server {self.address} refused have={have} for a {len(ids)}-token prompt ({exc}): "
                f"a decode-side bug; asking for every entry instead"
            )
            return self._request(ids, 0, on_progress, job)

    def _request(
        self, ids: list[int], have: int, on_progress, job: _RemoteFetch | None, targets: dict | None = None
    ) -> dict:
        """One prefill request on one pooled connection (retried once on a new one if the connection drops; a
        push's writes are idempotent, so it may run twice). No device work: safe off the scheduler thread."""
        for attempt in range(2):
            client = None
            try:
                client = self._take_client()
                if job is not None:
                    job.client = client
                    if job.aborted:
                        raise _PromptStopped()
                if targets is None:
                    states = client.prefill(ids, on_progress=on_progress, have=have)
                else:
                    states = client.prefill_push(ids, targets, on_progress=on_progress, have=have)
            except PrefillServerError:
                if client is not None and client.sock.fileno() >= 0:
                    self._give_client(client)  # an error reply leaves the connection usable
                raise
            except (ConnectionError, OSError) as exc:
                if client is not None:
                    with contextlib.suppress(Exception):
                        client.close()
                if attempt == 0 and not (job is not None and job.aborted):
                    logger.warning(f"prefill server {self.address} connection lost ({exc}); reconnecting")
                    continue
                raise
            except BaseException:
                # Cancellation (_PromptStopped) or unexpected error mid-stream: discard unread response bytes.
                if client is not None:
                    with contextlib.suppress(Exception):
                        client.close()
                raise
            self._give_client(client)
            if states["seq_len"] != len(ids) or states["base"] != have:
                raise RuntimeError(
                    f"prefill server {self.address} answered a {len(ids)}-token prompt with have={have} with "
                    f"entries of [{states['base']}, {states['seq_len']})"
                )
            return states

    def _whole(self, user: UserSession, ids: list[int]) -> tuple[list[int], int]:
        """The whole prompt to send for ``ids``, ``user``'s tokens from its position on, and its ``have``."""
        if user.pos % self.align or len(user.tokens) < user.pos:
            raise ValueError(
                f"disaggregated prefill continues a session from a multiple of {self.align} it holds the tokens of "
                f"(user.pos={user.pos}, {len(user.tokens)} tokens)"
            )
        return list(user.tokens[: user.pos]) + list(ids), user.pos

    def start(self, user: UserSession, ids: list[int], on_progress=None, wake: threading.Condition | None = None):
        """Send ``ids`` (``user``'s prompt from its position on) now, from a thread of its own; :meth:`run` for the
        same user and ids loads the reply. Replaces (aborts) a prompt of ``user``'s still in flight. Nothing is sent
        ahead in push mode (None): the push's pages are reserved on the device, on the scheduler thread."""
        if self.push:
            return None
        whole, have = self._whole(user, ids)
        job = _RemoteFetch(self, whole, have, on_progress, wake)
        with self._lock:
            old, self._fetches[user.index] = self._fetches.get(user.index), job
        if old is not None:
            old.abort()
        return job

    def forget(self, user: UserSession, job: _RemoteFetch) -> None:
        """Abort ``job``, ``user``'s prompt sent by :meth:`start` (its turn ended before loading it)."""
        with self._lock:
            if self._fetches.get(user.index) is job:
                del self._fetches[user.index]
        job.abort()

    def _push(self, user: UserSession, whole: list[int], have: int, on_progress) -> dict:
        """Prefill ``whole`` on the server, which writes the entries from ``have`` on into ``user``'s pages over
        TT-Fabric; returns its reply. Until the server has replied its writes may land, so the session's pages stay
        reserved: a cancellation (``on_progress`` raising) is held until then, and an error reply rewinds the
        session to 0."""
        model = self.engine.model
        if not have:
            model.reset_session(user.sid)
            model.reset_static_caches()
        targets = model.remote_commit_targets(user.sid, len(whole), self.ring_layers, start=have)
        stopped: list[BaseException] = []

        def progress(end: int) -> None:
            if on_progress is not None and not stopped:
                try:
                    on_progress(end)
                except _PromptStopped as exc:
                    stopped.append(exc)

        try:
            reply = self._fetch(whole, have, progress, targets=targets)
        except BaseException:
            model.abort_remote_commit(user.sid)
            raise
        if stopped:
            model.abort_remote_commit(user.sid)
            raise stopped[0]
        return reply

    def aligned(self, n_tokens: int, start: int = 0, bounded: bool = False) -> int:
        if start % self.align or bounded:
            return 0
        tail = self.window if self.decode_window else 1
        n_tokens -= tail
        if n_tokens <= 0:
            return 0
        return max(0, min(n_tokens // self.align * self.align, self.max_len - start))

    def seconds(self, n_tokens: int) -> float:
        return self.fill_seconds + max(0, -(-n_tokens // self.chunk_size) - 1) * self.chunk_seconds

    def run(
        self,
        user: UserSession,
        ids: list[int],
        on_progress=None,
        drain_seconds: float = 0.0,
        drained: int = 0,
        replayed: int = 0,
    ) -> torch.Tensor | None:
        engine, model = self.engine, self.engine.model
        if engine._replay_q:
            raise RuntimeError("prefill with decode traces still posted")
        if replayed:
            raise ValueError(f"disaggregated prefill takes no window replay (replayed={replayed})")
        whole, have = self._whole(user, ids)
        t_seat = time.perf_counter()
        user.activate()
        seat_seconds = time.perf_counter() - t_seat
        n = len(ids)

        with self._lock:
            job = self._fetches.pop(user.index, None)
        if job is not None and (job.ids != whole or job.have != have):
            job.abort()
            job = None
        t_wait = time.perf_counter()
        if job is None:
            t0 = t_wait
            if self.push:
                try:
                    states = self._push(user, whole, have, on_progress)
                except Exception:
                    with contextlib.suppress(Exception):
                        user.reset()
                    raise
            else:
                states = self._fetch(whole, have, on_progress)
        else:
            t0 = job.t0
            states = job.result()
        self.t_prefill_out = time.perf_counter()
        # A prompt sent ahead may have been back a while before the scheduler came to load it.
        net_seconds = (job.t_done if job is not None else self.t_prefill_out) - t0
        # What of the round trip the scheduler sat out (the rest overlapped other turns' decode).
        wait_seconds = self.t_prefill_out - t_wait

        # The reply starts at base: have, or 0 when the server refused have and sent every entry.
        base = states["base"]
        t_commit = time.perf_counter()
        try:
            if self.push:
                committed = model.finish_remote_commit(user.sid, ids=whole)
            else:
                if not base:
                    model.reset_session(user.sid)
                    model.reset_static_caches()
                committed = model.commit_host_prefill_state(states, user.sid, ids=whole, start=base)
        except Exception:
            with contextlib.suppress(Exception):
                user.reset()
            raise
        user.pos = committed
        commit_seconds = time.perf_counter() - t_commit
        total_seconds = net_seconds + commit_seconds
        # Of this run's tokens (the session's from have on), those the server did not compute.
        self.last_cached = max(0, min(states["cached"], len(whole)) - have)

        computed = len(whole) - states["cached"]
        num_chunks = -(-computed // self.chunk_size)
        if num_chunks > 1 and total_seconds > 0:
            self.chunk_seconds = 0.5 * self.chunk_seconds + 0.5 * (total_seconds / num_chunks)
        if computed > 0:
            fill_seconds = max(total_seconds - max(num_chunks - 1, 0) * self.chunk_seconds, 0.0)
            self.fill_seconds = 0.5 * self.fill_seconds + 0.5 * fill_seconds
        hit = f"{states['cached']} from the server's prefix cache" if states["cached"] else "no prefix-cache hit"
        logger.info(
            f"remote prefill {self.address} (user {user.index}): {n} tokens from {have} "
            f"({hit}, entries from {base}) in {total_seconds:.2f}s "
            f"({n / total_seconds if total_seconds > 0 else 0.0:.1f} tok/s), server {states['seconds']:.2f}s, "
            f"queued {states['queued']:.2f}s, "
            f"transfer {states['bytes'] / 1e6:.1f} MB ({net_seconds:.2f}s RTT, "
            f"{'sent ahead, ' if job is not None else ''}{wait_seconds:.2f}s blocking the scheduler), "
            f"KV load {1e3 * commit_seconds:.1f} ms, "
            f"drain {1e3 * drain_seconds:.1f} ms ({drained} step(s)), seat {1e3 * seat_seconds:.1f} ms"
        )
        return None


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
        max_context=None,
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
        "--prefill-url",
        default=os.environ.get("DEEPSEEK_V41_PREFILL_URL"),
        help="TCP URL or host:port of a disaggregated prefill server (e.g. tcp://10.82.97.40:9041 or "
        "10.82.97.40:9041); sends prompts over TCP and loads the returned KV cache into the decode device "
        "instead of running a local prefill model (default: DEEPSEEK_V41_PREFILL_URL, else unset)",
    )
    p.add_argument(
        "--prefill-push",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DEEPSEEK_V41_PREFILL_PUSH") == "1",
        help="with --prefill-url: the prefill server writes the KV cache straight into this server's device memory "
        "over TT-Fabric (both servers ranks of one tt-run job, demo/run_d_p_servers.sh) instead of sending it over "
        "TCP (default: DEEPSEEK_V41_PREFILL_PUSH=1, else off)",
    )
    p.add_argument(
        "--prefill-wait",
        type=float,
        default=float(os.environ.get("DEEPSEEK_V41_PREFILL_WAIT", 0)),
        help="with --prefill-url: seconds to keep retrying a prefill server that is not listening yet, as when both "
        "servers start together (default: DEEPSEEK_V41_PREFILL_WAIT, else 0)",
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
        "--persistent-moe",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DEEPSEEK_V41_PERSISTENT_MOE", "1") not in ("0", "", "false", "False"),
        help="decode on TP4 stages above the persistent TP12 MoE server (rows 5-7), --instances of them, and, with "
        "--dspark, each reply in 4-row verify rounds (3 drafts + 1, one step) or one-row steps as --dspark-drafts "
        "picks by load; then one session per instance, greedy on-device sampling, prompts replayed through decode "
        f"and no --vision (default: on, --max-context {_PERSISTENT_MAX_CONTEXT}; --no-persistent-moe or "
        "DEEPSEEK_V41_PERSISTENT_MOE=0 serves the stage pipeline)",
    )
    p.add_argument(
        "--instances",
        type=int,
        default=int(os.environ.get("DEEPSEEK_V41_INSTANCES", _INSTANCES)),
        help="with --persistent-moe, independent model instances sharing the TP12 MoE (rows 5-7): instance j is a whole "
        "TP4 model on mesh row 4 - j (rows 4, 3, 2, 1, 0), its DSpark drafter on the same row, with --num-users "
        "sessions each (1 with --dspark); a request goes to the instance holding its conversation, else to the least "
        "busy one. 1 serves the single stage on row 4 with the drafter on row 3 (default: DEEPSEEK_V41_INSTANCES, else "
        f"{_INSTANCES})",
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
    if args.prefill_url:
        try:
            parse_prefill_address(args.prefill_url)
        except ValueError as exc:
            p.error(f"--prefill-url: {exc}")
        # Prefill runs on the remote prefill server over TCP, so do not build a local prefill model on this device.
        args.prefill = False
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
    if args.persistent_moe:
        # The traced prefill runs on each instance's own stage (one user per instance, no pipeline over rows), its MoE
        # on the persistent TP12 prefill MoE (--no-prefill replays prompts through decode instead); no image rows;
        # the output stage samples.
        args.prefill_all_stages = False
        args.vision = False
        if args.prefill:
            # Room above the prefetch rings for decode's host sockets, so releasing the rings frees one block of L1
            # for the prefill's circular buffers (without it the sockets land mid-L1, under the 4-row step buffers).
            os.environ.setdefault("DEEPSEEK_V4_GCB_HEADROOM", "16384")
        args.device_sampling = True
        if args.max_context is None:
            args.max_context = _MAX_CONTEXT if args.prefill_url else _PERSISTENT_MAX_CONTEXT
        if not 1 <= args.instances <= len(pmoe.INSTANCE_ROWS):
            p.error(f"--instances takes 1 to {len(pmoe.INSTANCE_ROWS)} (the mesh rows above the MoE)")
        if args.dspark:
            # The 4-row verify: one session per instance, greedy, and each round either a 4-row verify (3 drafts)
            # or a one-row step.
            if args.num_users not in (None, 1):
                logger.warning(
                    f"--persistent-moe --dspark serves one session per instance: --num-users 1, not {args.num_users}"
                )
            if args.temperature and args.temperature != p.get_default("temperature"):
                logger.warning(f"--persistent-moe --dspark decodes greedily: --temperature 0, not {args.temperature}")
            args.num_users = 1
            args.max_running = 1
            args.temperature = 0.0
            drafts = _PERSISTENT_VERIFY_ROWS - 1
            if "DEEPSEEK_V41_DSPARK_DRAFTS" not in os.environ and not any(
                a.startswith("--dspark-drafts") for a in (argv if argv is not None else sys.argv)
            ):
                args.dspark_drafts = _INSTANCE_DSPARK_DRAFTS if args.instances > 1 else (drafts,)
            if not all(k in (0, drafts) for k in args.dspark_drafts):
                p.error(
                    f"--persistent-moe --dspark verifies 4-row blocks: --dspark-drafts entries are 0 (one-row steps) "
                    f"or {drafts}, got {args.dspark_drafts}"
                )
        elif args.num_users is None:
            args.num_users = _NUM_USERS_PERSISTENT
    else:
        args.instances = 1
    if args.max_context is None:
        args.max_context = _MAX_CONTEXT
    # host_rs.enabled(), which the model reads too, follows the flag.
    os.environ["DEEPSEEK_V41_RUST_HOST"] = "1" if args.rust_host else "0"
    if args.rust_host:
        args.tui = False  # the web monitor (demo/monitor.py) watches it, from a process of its own
    if args.vision and (args.image_rows % 32 or args.image_rows <= FIRST_IMAGE_ROW):
        p.error(f"--image-rows must be a positive multiple of 32, got {args.image_rows}")
    if args.l1_small_size is None:
        args.l1_small_size = int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", _L1_SMALL_SIZE))
    if args.persistent_moe:
        args.l1_small_size = max(args.l1_small_size, _PERSISTENT_L1_SMALL_SIZE)
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
    if args.total_context is None and args.persistent_moe:
        # One stage holds every layer's pools, shared by the running user and the conversations parked on device.
        args.total_context = max(_PERSISTENT_TOTAL_CONTEXT, args.num_users * args.max_context)
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
    if args.persistent_moe and args.prefill and args.instances == 1:
        # The model launches its own server: the prefill MoE's routed op needs the L1 the resident router columns
        # would take (launch_instances_server does the same for several instances).
        pmoe.ROUTER_L1_BUDGET = 0
    if args.persistent_moe and args.instances > 1:
        return _serve_instances(args)
    return serve(
        args,
        engine_cls=V41ChatEngine,
        prefiller_cls=_V41Prefiller,
        num_stages=(
            1
            if args.persistent_moe
            else min(pipeline_stage_count(), _DSPARK_STAGES)
            if args.dspark
            else pipeline_stage_count()
        ),
        handler_cls=_V41Handler,
        server_cls=V41GenerationServer,
    )


if __name__ == "__main__":
    sys.exit(main())
