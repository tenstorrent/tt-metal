# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decode demo on the full pipeline-parallel GLM-5.3-Flash (after deepseek_v4_flash's
``tests/decode/test_full_model_decode_demo.py``).

The whole model is built once on the Galaxy: eight 1 x 4 tensor-parallel stages, one per mesh row, the 45 layers
split contiguously across them (embedding on the first stage, head on the last). A chat-templated prompt is
prefilled into a ``Session`` -- in chunks through the prefill path by default, or, with ``GLM_PP_PREFILL=decode``,
by replaying the decode step once per prompt token as the DeepSeek demo does -- and generation then continues
greedily one token per step against the session's caches (KDA conv tail + recurrent state, MLA latent + DSA
indexer pools), with no recompute over the prior context.

DSA attends densely until a query sees more than ``index_topk`` tokens (position ``index_topk + index_kpool - 1``,
2051) and through the indexer's top-k pool selection from there on, so the sparse path is only exercised by runs
that reach that far (the default ``GLM_PP_MAX_NEW_TOKENS`` does).

Run it (ttnn venv)::

    GLM_PP_MAX_NEW_TOKENS=64 PYTHONPATH=$PWD TT_METAL_HOME=$PWD pytest -s \\
      models/experimental/glm53_flash/tests/pipeline_parallel_prefill/test_full_model_decode_demo.py

``GLM_PP_LAYERS=N`` caps the stack for bring-up (the output is then not meaningful text), ``GLM_PP_CACHE`` points at
the converted ttnn weight tiles, ``GLM_PP_PROMPT`` replaces the prompt and ``GLM_PP_CHUNK`` sets the prefill chunk.
Ctrl-C stops generation at the next step boundary and still reports what was generated.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.glm53_flash.tt.config import Glm5NextTextConfig
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.model import GLMPipeline
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.weights import ModelSource

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
from checkpoint import default_snapshot_dir  # noqa: E402

try:
    SNAPSHOT = default_snapshot_dir()
except FileNotFoundError:
    SNAPSHOT = None

pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason="GLM-5.3-Flash checkpoint not available")

CACHE = Path(os.environ.get("GLM_PP_CACHE", Path.home() / ".cache/ttnn/models/glm53_flash_pp"))
MAX_NEW_TOKENS = int(os.environ.get("GLM_PP_MAX_NEW_TOKENS", "2560"))
PREFILL_MODE = os.environ.get("GLM_PP_PREFILL", "prefill")
CHUNK = int(os.environ.get("GLM_PP_CHUNK", "2048"))
TP, N_STAGES = 4, 8
REPORT_EVERY = 64
_DEFAULT_TEXT = (
    "Tell me the name of the top 10 movies of all time. Also list out the top 10 worst movies of all time. "
    "Give me details of why you choose those movies. Try to make your response as humours as possible. "
    "For context, I am putting together a light-hearted movie night for my friends, who love classic dramas, "
    "science fiction, and over-the-top comedies alike. Keep every entry short, mention the director and the "
    "release year, and end with a one-line joke about each film. Please format the answer as two clearly "
    "numbered lists, one for the best and one for the worst."
)
TEXT = os.environ.get("GLM_PP_PROMPT", _DEFAULT_TEXT if MAX_NEW_TOKENS >= 10 else "I")


class _InterruptFlag:
    """Record SIGINT during the generation loop so a Ctrl-C ends it at a step boundary (with the session's caches
    consistent and the tokens so far reported) rather than unwinding from inside a device step."""

    def __init__(self):
        self.hit = False
        self._previous = None

    def __enter__(self):
        self._previous = signal.signal(signal.SIGINT, self._record)
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._previous)

    def _record(self, signum, frame):
        self.hit = True


def _assert_parallelism(model: GLMPipeline, n_layers: int) -> None:
    """The layout the demo pins: eight 1 x 4 TP stages covering every layer once, in order."""
    assert len(model.stages) == N_STAGES
    assert all(st.ctx.device.get_num_devices() == TP and st.ctx.tp == TP for st in model.stages)
    assert sum(model.placement, []) == list(range(n_layers))
    assert model.stages[0].first and model.stages[-1].last
    logger.info(f"parallelism: {N_STAGES} pipeline stages x TP{TP} ({N_STAGES * TP} chips), layers {model.placement}")


@pytest.mark.timeout(14400)  # a cold weight cache converts every expert to bfloat4_b; then thousands of steps
@torch.no_grad()
def test_full_model_decode_demo(stages):
    from models.experimental.glm53_flash.tt.tokenizer_chat import load_tokenizer

    assert PREFILL_MODE in ("prefill", "decode"), f"GLM_PP_PREFILL must be prefill or decode, got {PREFILL_MODE!r}"
    cfg = Glm5NextTextConfig.from_json(str(SNAPSHOT))
    n_layers = min(int(os.environ.get("GLM_PP_LAYERS", cfg.num_hidden_layers)), cfg.num_hidden_layers)
    tokenizer = load_tokenizer(SNAPSHOT)

    t0 = time.perf_counter()
    model = GLMPipeline(stages, cfg, ModelSource(SNAPSHOT), n_layers=n_layers, cache_dir=CACHE)
    logger.info(f"built {n_layers}/{cfg.num_hidden_layers} layers in {time.perf_counter() - t0:.0f}s")
    _assert_parallelism(model, n_layers)

    prompt_ids = tokenizer.encode_chat([{"role": "user", "content": TEXT}])
    real_len = len(prompt_ids)
    sparse_from = cfg.index_topk + cfg.index_kpool - 1  # first position whose DSA selection is sparse
    end_pos = real_len + MAX_NEW_TOKENS - 1
    if end_pos < sparse_from:
        logger.warning(f"this run ends at pos {end_pos} < {sparse_from}: the DSA indexer selection is never sparse")
    else:
        logger.info(f"DSA indexer: dense below pos {sparse_from}, sparse top-{cfg.index_topk} selection from there on")

    # --- prefill ------------------------------------------------------------------------------------------- #
    session = model.new_session()
    t0 = time.perf_counter()
    if PREFILL_MODE == "prefill":
        logits = model.prefill(prompt_ids, chunk=CHUNK, session=session)
    else:
        for tok in prompt_ids:
            logits = model.decode(tok, session)
    t_prefill = time.perf_counter() - t0
    assert session.pos == real_len
    next_id = int(logits.argmax())
    logger.info(
        f"prefill ({real_len} tokens, {PREFILL_MODE} path) in {t_prefill:.2f}s "
        f"({real_len / t_prefill:.1f} tok/s) -> token id {next_id} {tokenizer.decode([next_id])!r}"
    )

    # --- decode: feed the previous token at its position, read back the next-token logits ------------------ #
    generated = [next_id]
    window_tokens, window_time = 0, 0.0
    total_tokens, total_time = 0, 0.0
    with _InterruptFlag() as interrupt:
        while len(generated) < MAX_NEW_TOKENS:
            if interrupt.hit:
                logger.info("interrupted; stopping generation")
                break
            if tokenizer.is_stop_token(next_id):
                logger.info(f"hit stop token {next_id}; stopping")
                break
            pos = session.pos
            t0 = time.perf_counter()
            logits = model.decode(next_id, session)
            next_id = int(logits.argmax())
            dt = time.perf_counter() - t0
            generated.append(next_id)
            window_tokens, window_time = window_tokens + 1, window_time + dt
            total_tokens, total_time = total_tokens + 1, total_time + dt
            if window_tokens == REPORT_EVERY:
                logger.info(
                    f"decode throughput: {window_tokens / window_time:.2f} tok/s ({window_tokens} tokens in "
                    f"{window_time:.2f}s), DSA selection: {'sparse' if pos >= sparse_from else 'dense'} (pos {pos})"
                )
                window_tokens, window_time = 0, 0.0

    if total_tokens:
        logger.info(
            f"decode throughput (overall): {total_tokens / total_time:.2f} tok/s "
            f"({total_tokens} steps in {total_time:.2f}s, {1e3 * total_time / total_tokens:.0f} ms/token)"
        )
    assert generated, "no tokens were generated"
    assert session.pos == real_len + len(generated) - 1
    assert session.tokens == prompt_ids + generated[:-1]
    text = tokenizer.decode(generated)  # raises on an id outside the tokenizer vocabulary (an lm_head pad row)
    logger.info(f"PROMPT    : {tokenizer.decode(prompt_ids)!r}")
    logger.info(f"GENERATED : {text!r}  ({len(generated)} tokens, final pos {session.pos})")
    if n_layers == cfg.num_hidden_layers:
        reply = tokenizer.parse_assistant(generated)
        logger.info(f"reasoning: {len(reply.reasoning_content or '')} chars; content: {len(reply.content)} chars")
