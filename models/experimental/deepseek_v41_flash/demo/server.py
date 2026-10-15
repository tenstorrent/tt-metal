# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP server on the full ttnn ``DeepSeekV41Model``.

The V4.1 twin of ``models/experimental/deepseek_v4_flash/demo/server.py``, which it runs unchanged -- the same
endpoints, OpenAI request and response shapes, tool calling, thinking mode, scheduler, live console and traced
prefill (read that docstring) -- on the V4.1 decode model and :class:`DeepSeekV41PrefillModel` instead:

* the engine (:class:`V41ChatEngine`) builds :class:`DeepSeekV41Model` and its ``lm_head``; its layers are split
  evenly over the stages in contiguous runs (five per stage on eight stages), a placement the prefill model shares,
  as in ``tests/prefill/test_prefill_decode_demo.py``;
* the prefill (:class:`_V41Prefiller`) is :meth:`DeepSeekV41Model.build_prefill` (sharing the routed experts,
  embedding, ``lm_head``, sockets and the Engram host lookup) with no CSA position bias to commit.

What V4.1 imposes on the serving side:

* One user. V4.1 decode seats a single session (the select op decodes one user and there is no page-table swap), so
  ``--num-users`` is 1 and requests take turns on it; a request that does not continue the stored conversation
  re-prefills it.
* Context. ``--max-context`` is 262144 tokens by default (prompt + reply). Past
  ``candidate_topk_blocks * candidate_block_size`` (16384) compressed entries the select is two-level, in the
  traced prefill and in decode, so a prompt prefills up to ``--prefill-max-len`` (by default the whole context, at
  most 262144 tokens, see :class:`_V41Prefiller`) and decode replays the rest.
* Engram layers 1 and 14 read host n-gram tables (~101 GB each); the model refuses to build without that RAM free.

Run it (ttnn venv, from the repo root; 32-chip Galaxy for the prefill model)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    python models/experimental/deepseek_v41_flash/demo/server.py --host 0.0.0.0 --port 8000 --debug

Then talk to it as to the V4 server (``models/experimental/deepseek_v4_flash/demo/client.py``)::

    curl -s --no-buffer -X POST http://127.0.0.1:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections import deque

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
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, dequantized, load_config
from models.experimental.deepseek_v41_flash.tt.model import DeepSeekV41Model
from models.experimental.deepseek_v4_flash.demo.chat_cli import ChatEngine, UserSession
from models.experimental.deepseek_v4_flash.demo.server import _Prefiller, build_parser, check_args, serve
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_NUM_STAGES = 8  # one 1 x 4 TP4 stage per Galaxy row: V4.1's 40 layers, five on each
# Default --max-context, and --prefill-max-len when a larger context is asked for: the longest traced prefill whose
# last position tier compiled on every stage (a 524288-token tier ran out of DRAM).
_MAX_CONTEXT = 262144


class V41ChatEngine(ChatEngine):
    """:class:`ChatEngine` on :class:`DeepSeekV41Model`: the same users, decode and warmup over V4.1's model.

    Only construction differs: the V4.1 config and ``lm_head``, the context capped by the compressed-KV limit, the
    EOS from the tokenizer (the V4.1 config has none), and the layers split evenly over the stages in contiguous runs,
    whatever the profile's pipeline group size, so the traced prefill can share them.
    """

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
        num_stages = getattr(args, "num_stages", _NUM_STAGES)
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
            total_tokens=self.max_seq,
            block_size=_PAGE_BLOCK_SIZE,
        )
        if self.model.context_limit is not None and self.model.context_limit < self.max_seq:
            logger.warning(f"context capped at {self.model.context_limit} tokens by the compressed KV")
            self.max_seq = self.model.context_limit

        self.users = [UserSession(self, i, args) for i in range(args.num_users)]
        self.active = 0
        self.users[0].activate()
        self._replay_q: deque[int] = deque()


class _V41Prefiller(_Prefiller):
    """:class:`_Prefiller` with V4.1's prefill model, which has no CSA position bias for the commit.

    The prefill selects two-level past ``candidate_topk_blocks * candidate_block_size`` compressed entries, as
    decode does. Its attention stays dense, so every per-layer mask and score tensor of the last position tier
    spans the whole ``--prefill-max-len``; with decode's caches resident, a 512K-token tier runs out of DRAM.
    Unset, ``--prefill-max-len`` is therefore the context capped at ``_MAX_CONTEXT``, and decode replays the rest of
    a longer prompt; an explicit value is used as given.
    """

    def __init__(self, engine: V41ChatEngine, args, prefetcher: contextlib.ExitStack):
        if not args.prefill_max_len and engine.max_seq > _MAX_CONTEXT:
            logger.info(
                f"traced prefill sized for {_MAX_CONTEXT} tokens (dense attention past that does not fit beside "
                f"decode's {engine.max_seq}-token caches); decode replays the rest of a longer prompt"
            )
            args.prefill_max_len = _MAX_CONTEXT
        super().__init__(engine, args, prefetcher)

    def _build_model(self, engine: V41ChatEngine):
        return engine.model.build_prefill(
            engine.rope,
            lm_head=engine.lm_head,
            cache=engine.weight_cache,
            weight_dtype=_ATTENTION_WEIGHT_DTYPE,
            # Sliding layers attend without an additive mask (the compressed ones stay masked): the masked SDPA's
            # mask buffers do not fit beside the L1 decode keeps resident.
            maskless=True,
            progress=self._note,
        )

    def _bias_slots(self):
        return None


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
    )
    args = p.parse_args(argv)
    if args.num_users != 1:
        p.error("V4.1 decode seats one session: --num-users must be 1")
    if args.max_new_tokens is None:
        args.max_new_tokens = args.max_context
    return check_args(p, args, sys_cfg)


def main(argv: list[str] | None = None) -> int:
    return serve(parse_args(argv), engine_cls=V41ChatEngine, prefiller_cls=_V41Prefiller)


if __name__ == "__main__":
    sys.exit(main())
