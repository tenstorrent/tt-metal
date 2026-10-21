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
  embedding, ``lm_head``, sockets and the Engram host lookup) with no CSA position bias to commit.

``GET /models`` and ``GET /v1/models`` return DeepSeek's list-models record (:func:`model_record`).

A turn stops, mid-prefill or mid-decode, when its client disconnects or on
``POST /v1/chat/completions/<id>/cancel`` (the ``id`` of the first streamed chunk, also the ``X-Request-Id`` header)
or ``POST /v1/sessions/<user>/cancel``, and every running turn stops on the live console's ``k`` key; see
"Stopping" in the V4 server's docstring.

What V4.1 imposes on the serving side:

* Users. ``--num-users`` sessions stay open at once, each with its own ring and window slot and its own rows of the
  compressed-KV page tables; their compressed KV shares pools sized for ``--total-context`` tokens (default
  ``num_users * max_context``). A step decodes one user, and switching users costs no device work: the step's packet
  names the user's slot. The scheduler interleaves the active users' steps through the stages, so up to one
  decoding user per stage (``DEEPSEEK_V4_NUM_STAGES``, default eight) keeps every stage busy; seating and posting,
  trace dispatch, the H2D packet writes and the D2H reads each run on their own thread (see :class:`V41ChatEngine`).
  Speculative decode is single-user.
* Context. ``--max-context`` is 1048576 tokens by default (prompt + reply). Past
  ``candidate_topk_blocks * candidate_block_size`` (16384) compressed entries the select is two-level, in the
  traced prefill and in decode, so a prompt prefills up to ``--prefill-max-len`` (by default the whole context, at
  most 131072 tokens, see :class:`_V41Prefiller`) and decode replays the rest.
* Engram layers 1 and 14 read host n-gram tables (~101 GB each); the model refuses to build without that RAM free.

Run it (ttnn venv, from the repo root; 32-chip Galaxy for the prefill model)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    python models/experimental/deepseek_v41_flash/demo/server.py --host 0.0.0.0 --port 8000 --num-users 8 --debug

Then talk to it as to the V4 server (``models/experimental/deepseek_v4_flash/demo/client.py``)::

    curl -s --no-buffer -X POST http://127.0.0.1:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'

    curl -s http://127.0.0.1:8000/models
"""

from __future__ import annotations

import contextlib
import os
import queue
import sys
import threading
from collections import deque
from urllib.parse import urlparse

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
    dequantized,
    load_config,
    pipeline_stage_count,
)
from models.experimental.deepseek_v41_flash.tt.model import DeepSeekV41Model
from models.experimental.deepseek_v4_flash.demo.chat_cli import ChatEngine, ChatFormat, UserSession
from models.experimental.deepseek_v4_flash.demo.server import _Handler, _Prefiller, build_parser, check_args, serve
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_MAX_CONTEXT = 1048576  # default --max-context (prompt + reply)

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

# Display name from https://api-docs.deepseek.com/api/list-models . The served ``id`` stays ``--model-id``
# (default ``deepseek-v4.1-flash``); the published catalog id is ``deepseek-flash``.
_MODEL_NAME = "DeepSeek-V4.1-Flash"
_MODEL_OWNER = "deepseek"


def model_record(api) -> dict:
    """One model object for `GET /models <https://api-docs.deepseek.com/api/list-models>`_.

    Limits are this process's, not the published 1,048,576 / 393,216 cloud numbers: ``context_window`` is
    the live prompt-plus-reply cap, and ``max_output_tokens`` is the smaller of that and ``--max-new-tokens``.
    ``input_modalities`` is text only (an image part is refused). ``effort.supported_levels`` omits ``medium``
    (accepted, then rendered as ``high``) and ``none`` (thinking off). ``system_prompt_update`` is
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
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "effort": effort,
        "api_capabilities": {"anthropic_messages": {"system_prompt_update": "in-history"}},
    }


class _V41Handler(_Handler):
    """:class:`_Handler` plus DeepSeek ``GET /models`` (``GET /v1/models`` returns the same body)."""

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/models", "/v1/models"):
            super().do_GET()
            return
        try:
            self._send_json({"object": "list", "data": [model_record(self.api)]})
        except Exception as e:
            self._send_error_json(e)


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
        )
        self.lm_head = Linear(
            dequantized(loader, "lm_head.weight", translate=True),
            self.model.last_device,
            self.weight_cache.file("lm_head") if self.weight_cache else None,
            dtype=_WEIGHT_DTYPE,
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
        self._packet_error: BaseException | None = None
        self._packet_writer = threading.Thread(target=self._write_packets, name="decode-packets", daemon=True)
        self._packet_writer.start()
        # Registered after the model's shutdown so it unwinds first: the thread's closure holds the model.
        prefetcher.callback(self._stop_packet_writer)

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
    """

    def _build_model(self, engine: V41ChatEngine):
        return engine.model.build_prefill(
            engine.rope,
            lm_head=engine.lm_head,
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
        compressed KV alone."""
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
        num_users=1,
        max_context=_MAX_CONTEXT,
        total_context=None,
        # None means "no reply cap": a request runs until EOS or the context is full.
        # ``--max-new-tokens N`` still caps a reply at N.
        max_new_tokens=None,
        # Requests that omit temperature sample at 1. 0, or a request's own temperature, still wins.
        temperature=1.0,
    )
    for action in p._actions:
        if action.dest == "temperature":
            action.help = (
                "sampling temperature for requests that send none (default: 1); "
                "a request's own temperature overrides it, and 0 asks for greedy"
            )
    args = p.parse_args(argv)
    if args.max_new_tokens is None:
        args.max_new_tokens = args.max_context
    return check_args(p, args, sys_cfg)


def main(argv: list[str] | None = None) -> int:
    return serve(
        parse_args(argv),
        engine_cls=V41ChatEngine,
        prefiller_cls=_V41Prefiller,
        num_stages=pipeline_stage_count(),
        handler_cls=_V41Handler,
    )


if __name__ == "__main__":
    sys.exit(main())
