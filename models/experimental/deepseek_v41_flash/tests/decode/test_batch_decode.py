# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""B=4 batch decode on one persistent-MoE instance (``DeepSeekV41Model(batch_block=True)``): four sessions, each at
its own position, decode one token each per :meth:`~DeepSeekV41Model.decode_batch_block` step, and every step's
picks must be what one-row decode (:meth:`decode_traced`) picks for that session alone.

The instance is the persistent layout's one TP4 stage (row 4) with every MoE on the persistent TP12 server below it;
the batch step is the DSpark 4-row block step with each row an independent session (its own ring and window slot,
page-table row and ``dense_kv`` rows), and its MoE requests are 4-row ones.

1. Every session decodes alone with one-row steps: its prompt, then ``DEEPSEEK_V4_MAX_NEW_TOKENS`` greedy tokens. Its
   picks at every position are the reference.
2. The sessions are rewound. Session ``i`` replays the first ``start_i`` tokens of that sequence with one-row steps
   (``start_i`` differs per session, so the rows sit at different positions), then the rest in batch steps, each
   session fed its reference tokens (teacher forcing). Sessions run out at different steps: the tail steps carry
   fewer than four rows, padded with the scratch session.
3. Every batch pick should equal the one-row pick at the same position, but for bf16 near-ties (the block step's
   (4, 32)-tiled ops, 4-row MoE requests and attention sum in another order than the one-row step's): every session's
   first must, and at least ``_MIN_MATCH`` of them.
4. If any differ, the sessions are rewound and the same batch steps run again eagerly
   (:meth:`~DeepSeekV41Model.debug_batch_logits`): at each differing position the batch step's logit of its own pick
   may exceed its logit of the one-row pick by at most ``_NEAR_TIE`` (``DEEPSEEK_V41_BATCH_NEAR_TIE``).

``DEEPSEEK_V41_BATCH_LONG=N`` prefixes session 0's prompt with ``N`` filler tokens, so the batch steps' furthest row
passes ``index_topk`` closed groups and the steps run the scored select variants for every row, short ones included
(past position 16384 the two-level select, past 31744 the candidate source's key split). ``DEEPSEEK_V41_BATCH_TAIL=N``
runs only each session's last ``N`` positions in batch steps (the rest one row a step), which keeps a long run short. It needs the TP12 expert cache (``tools/fill_expert_cache.py --all-layers --tp-size 12
--mesh-rows 3``)::

    DEEPSEEK_V4_MAX_NEW_TOKENS=32 pytest models/experimental/deepseek_v41_flash/tests/decode/test_batch_decode.py
    DEEPSEEK_V41_BATCH_LONG=2100 pytest models/experimental/deepseek_v41_flash/tests/decode/test_batch_decode.py
    DEEPSEEK_V41_BATCH_LONG=32000 DEEPSEEK_V41_BATCH_TAIL=400 pytest \
      models/experimental/deepseek_v41_flash/tests/decode/test_batch_decode.py
"""

from __future__ import annotations

import contextlib
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _L1_SMALL_SIZE,
    _PAGE_BLOCK_SIZE,
    _SEED,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _stage_engram_or_skip,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tests.decode.test_multi_session_decode import _TEXTS
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# DEEPSEEK_V41_BATCH_ROWS=2: B=2 batch steps (2-row blocks) instead of B=4.
_ROWS = int(os.environ.get("DEEPSEEK_V41_BATCH_ROWS", "4"))
assert _ROWS in (2, 4), f"DEEPSEEK_V41_BATCH_ROWS takes 2 or 4, got {_ROWS}"
_MAX_NEW_TOKENS = int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "32"))
_LONG = int(os.environ.get("DEEPSEEK_V41_BATCH_LONG", "0"))
_MIN_MATCH = float(os.environ.get("DEEPSEEK_V41_BATCH_MIN_MATCH", "0.9"))
_NEAR_TIE = float(os.environ.get("DEEPSEEK_V41_BATCH_NEAR_TIE", "1.5"))
# Batch rows for at most the last this many positions of each session (0: from about its middle on).
_TAIL = int(os.environ.get("DEEPSEEK_V41_BATCH_TAIL", "0"))
# The batch step's programs (the 4-row lm_head sampler's all_gather last) need L1_SMALL past the 10 KB of
# test_full_model_decode_demo, as DSpark's do (demo/server.py reserves 16 KB under --persistent-moe).
_BATCH_L1_SMALL_SIZE = max(_L1_SMALL_SIZE, 16384)  # as demo/server.py under --persistent-moe (B=2 needs past 12 KB)


def _pick(out: torch.Tensor) -> int:
    out = out.reshape(-1)
    return int(out[0]) if out.dtype == torch.int32 else int(out.argmax())


def _decode_alone(model, sid: int, ids: list[int], start: int = 0) -> list[int]:
    """One-row steps of ``ids[start:]`` at positions ``start ..`` on session ``sid``: each position's pick."""
    model.activate_session(sid)
    return [_pick(model.decode_traced(token, pos)) for pos, token in enumerate(ids) if pos >= start]


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            "l1_small_size": _BATCH_L1_SMALL_SIZE,
        }
    ],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["tp4_1stage_persistent_moe12"])
def test_batch_decode_matches_one_row(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    _stage_engram_or_skip(loader, config)
    prompts = [_tokenize_chat(tokenizer, text) for text in _TEXTS[:_ROWS]]
    if _LONG:
        filler = tokenizer.encode(" ".join(_TEXTS) + " ", add_special_tokens=False)
        prompts[0] = prompts[0][:1] + (filler * (_LONG // len(filler) + 1))[:_LONG] + prompts[0][1:]
    longest = max(len(p) for p in prompts) + _MAX_NEW_TOKENS
    max_seq = _traced_max_seq(config, longest + 1, num_layers)

    with contextlib.ExitStack() as stack:
        model, lm_head, _, _ = _construct_model(
            mesh_device,
            stack,
            loader=loader,
            config=config,
            num_stages=1,
            persistent_moe=True,
            verify_rows=_ROWS,
            batch_block=True,
        )
        rope = _build_rope(config, loader.snapshot_dir, max_seq)
        # One slot per user and the scratch slot that pads a short step.
        model.prepare_static_decode(
            rope,
            max_seq,
            lm_head=lm_head,
            num_sessions=_ROWS + 1,
            total_tokens=(_ROWS + 1) * max_seq,
            block_size=_PAGE_BLOCK_SIZE,
            device_sampling=True,
        )
        model.device_sampler.set_sampling(0.0, _SEED)
        model.prepare_batch_decode()
        sids = [model.open_session() for _ in range(_ROWS)]
        start = time.perf_counter()
        model.capture_batch_traces(prompts[0][0])
        logger.info(f"one-row and batch traces captured in {time.perf_counter() - start:.0f}s")
        for sid in sids:
            model.ensure_session_capacity(max_seq - 1, sid)

        # 1. Every session alone, one row a step: the reference picks at every position.
        fed, ref = [], []
        for i, sid in enumerate(sids):
            picks = _decode_alone(model, sid, prompts[i])
            tokens = list(prompts[i])
            for _ in range(_MAX_NEW_TOKENS):
                tokens.append(picks[-1])
                picks.append(_pick(model.decode_traced(picks[-1], len(tokens) - 1)))
            fed.append(tokens)
            ref.append(picks)
            logger.info(
                f"session {i}: {len(prompts[i])} prompt tokens, reply {tokenizer.decode(picks[-_MAX_NEW_TOKENS:])!r}"
            )

        # 2. Rewind; one-row steps up to a different position per session, then batch steps for the rest.
        starts = [max(1, len(tokens) // 2 - 7 * i) for i, tokens in enumerate(fed)]
        if _TAIL:
            starts = [max(start, len(tokens) - _TAIL) for start, tokens in zip(starts, fed)]
        for i, sid in enumerate(sids):
            model.reset_session(sid)
            model.ensure_session_capacity(max_seq - 1, sid)
            _decode_alone(model, sid, fed[i][: starts[i]])
        pos = list(starts)
        got = [[] for _ in sids]
        steps = 0
        t0 = time.perf_counter()
        while True:
            rows = [i for i in range(_ROWS) if pos[i] < len(fed[i])]
            if not rows:
                break
            picks = model.decode_batch_block(
                [fed[i][pos[i]] for i in rows], [sids[i] for i in rows], positions=[pos[i] for i in rows]
            )
            for i, pick in zip(rows, picks):
                got[i].append(pick)
                pos[i] += 1
            steps += 1
        elapsed = time.perf_counter() - t0
        tokens = sum(len(g) for g in got)
        logger.info(
            f"{steps} batch steps, {tokens} tokens in {elapsed:.2f}s: {tokens / elapsed:.1f} tok/s over the batch, "
            f"{steps / elapsed:.1f} steps/s"
        )

        # 3. Every pick as the one-row step's.
        matched = total = 0
        differ = []
        for i in range(_ROWS):
            want = ref[i][starts[i] :]
            assert len(got[i]) == len(want), (i, len(got[i]), len(want))
            diffs = [k for k, (a, b) in enumerate(zip(got[i], want)) if a != b]
            matched += len(want) - len(diffs)
            total += len(want)
            logger.info(
                f"session {i}: batch rows from position {starts[i]}, {len(want) - len(diffs)}/{len(want)} picks match"
                + (f"; first differences at positions {[starts[i] + k for k in diffs[:8]]}" if diffs else "")
            )
            differ += [(i, starts[i] + k) for k in diffs]
        logger.info(f"{matched}/{total} batch picks match the one-row ones")
        for i in range(_ROWS):
            want = ref[i][starts[i]]
            assert got[i][0] == want, f"session {i}: the first batch pick {got[i][0]} != the one-row pick {want}"
        assert matched >= _MIN_MATCH * total, f"only {matched}/{total} batch picks match the one-row ones"
        if not differ:
            return

        # 4. The same steps eagerly, with logits: a differing pick must be a near-tie of the batch step's.
        for i, sid in enumerate(sids):
            model.reset_session(sid)
            model.ensure_session_capacity(max_seq - 1, sid)
            _decode_alone(model, sid, fed[i][: starts[i]])
        pos = list(starts)
        gaps = {}
        same = 0
        while True:
            rows = [i for i in range(_ROWS) if pos[i] < len(fed[i])]
            if not rows:
                break
            logits = model.debug_batch_logits(
                [fed[i][pos[i]] for i in rows], [sids[i] for i in rows], positions=[pos[i] for i in rows]
            )
            for r, i in enumerate(rows):
                top = int(logits[r].argmax())
                same += top == got[i][pos[i] - starts[i]]
                if (i, pos[i]) in differ:
                    gaps[(i, pos[i])] = (float(logits[r, top] - logits[r, ref[i][pos[i]]]), top, ref[i][pos[i]])
                pos[i] += 1
        logger.info(f"eager batch steps: {same}/{total} argmax picks equal the traced batch steps'")
        for (i, p), (gap, top, want) in sorted(gaps.items()):
            logger.info(
                f"session {i} position {p}: batch pick {top} ({tokenizer.decode([top])!r}) leads the one-row pick "
                f"{want} ({tokenizer.decode([want])!r}) by {gap:.3f} logits"
            )
        worst = max(gap for gap, _, _ in gaps.values())
        assert worst <= _NEAR_TIE, f"a batch pick leads the one-row pick by {worst:.3f} logits (> {_NEAR_TIE})"
