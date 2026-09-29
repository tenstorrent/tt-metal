# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo: prefill a real prompt with the prefill model, then generate with the traced decode model.

The prompt is prefilled in 128-aligned chunks by :class:`DeepSeekV4PrefillModel`; its per-layer attention state
is then written into the decode model's own buffers (:func:`load_prefill_state_into_decode`: same ring / compressed
KV / CSA overlap window / paged HCA pool that decode fills itself, nothing in decode is changed), the prompt's
ragged tail (``len % 128`` tokens) is fed through ``decode_traced`` one token at a time, and generation continues
with ``decode_traced`` exactly like ``tests/decode/test_full_model_decode_demo.py``. The prompt and the generated
text are printed; there is no reference: the check is whether the answer is sensible for the prompt.

Chips (Galaxy 8 x 4, ``1 x 4`` TP4 stages, two of each): the decode model takes mesh rows 0-1 as it always does; the
prefill model takes rows ``DEEPSEEK_V4_E2E_PREFILL_ROW`` .. +1 (default 4-5), because both models keep the routed
experts resident and one chip cannot hold both. Row 2 is left alone too: the decode model parks its DSpark/MTP link
there unless ``DEEPSEEK_V4_DSPARK=0``, which this test always sets (nothing here reads the link, and an unread link is one
more thing a pipelined step can stall on). The prefill state crosses over through the host.

Limits: the CSA layers run dense in both models, and prefill does not fill the lightning indexer's key cache, so the
whole conversation (prompt + generated tokens) must stay below ``index_topk * 4 = 2048`` tokens.

Order of events (each step is logged as ``[n/8]``, with the heartbeat of the prefill demo):

1. tokenizer, prompt; 2. decode model built, static decode state prepared, session opened; 3. one throw-away decode
step, which captures the traces (its scratch writes to the caches are overwritten in step 6); 4. prefill model built
on other rows; 5. the 128-aligned part of the prompt prefilled; 6. the prefill state written into the decode buffers;
7. the ragged tail replayed through decode; 8. up to ``DEEPSEEK_V4_MAX_NEW_TOKENS`` tokens generated.

Run it (ttnn venv)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v4_flash/tests/prefill/test_prefill_decode_demo.py

Knobs (environment): ``DEEPSEEK_V4_E2E_PROMPT_LEN`` (1000 tokens; not a multiple of 128 on purpose, so the tail path
runs), ``DEEPSEEK_V4_MAX_NEW_TOKENS`` (128), ``DEEPSEEK_V4_E2E_CHUNK`` (1024 = prefill chunk),
``DEEPSEEK_V4_E2E_PREFILL_ROW`` (4), ``DEEPSEEK_V4_E2E_TEXT`` (a plain user message instead of the book prompt; ``decode_demo`` = the prompt of
``tests/decode/test_full_model_decode_demo.py``, exactly 128 tokens),
``DEEPSEEK_V4_DECODE_LAYERS`` (bring-up: first N layers in both models; the text is then gibberish, the flow is not),
``DEEPSEEK_V4_PREFILL_PROMPT`` (another prompt file), ``DEEPSEEK_V4_PREFILL_HEARTBEAT`` / ``_STALL_SECS``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.tests.decode.test_full_model_decode_demo import (
    _CACHE_DIR,
    _DEFAULT_MODEL_DIR,
    _DEFAULT_TEXT,
    _PAGE_BLOCK_SIZE,
    _assert_decode_parallelism,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _DEFAULT_PROMPT_FILE,
    _TP_SIZE,
    _Progress,
    _build_prompt_ids,
    _env_int,
)
from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import round_context
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT
from models.experimental.deepseek_v4_flash.tt.prefill.handoff import load_prefill_state_into_decode
from models.experimental.deepseek_v4_flash.tt.prefill.model import DeepSeekV4PrefillModel
from models.experimental.deepseek_v4_flash.tt.prefill.weights import checkpoint_expert_provider, checkpoint_weights
from models.experimental.deepseek_v4_flash.tt.system_config import set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_MAX_CONTEXT = 2048  # index_topk * CSA rate: the dense CSA / no-indexer-state limit of the hand-off


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "num_command_queues": 2}],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_prefill_decode_demo(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    progress = _Progress(
        interval=float(os.environ.get("DEEPSEEK_V4_PREFILL_HEARTBEAT", 30)),
        stall=float(os.environ.get("DEEPSEEK_V4_PREFILL_STALL_SECS", 600)),
    )
    progress.verbose = False
    os.environ["DEEPSEEK_V4_DSPARK"] = "0"  # MTP disabled for now (read by DeepSeekV4Model.__init__): no link on row 2
    # The decode model's prefetcher session spans everything (as in the decode demo).
    with progress, contextlib.ExitStack() as prefetcher:
        _run(mesh_device, progress, prefetcher, AutoTokenizer)


def _eos_ids(config) -> set[int]:
    eos = config.eos_token_id
    return {int(eos)} if isinstance(eos, int) else {int(e) for e in (eos or [])}


def _run(mesh_device, progress: _Progress, prefetcher: contextlib.ExitStack, AutoTokenizer) -> None:
    max_new = _env_int("DEEPSEEK_V4_MAX_NEW_TOKENS", 128)
    prompt_len = _env_int("DEEPSEEK_V4_E2E_PROMPT_LEN", 1000)
    chunk_size = _env_int("DEEPSEEK_V4_E2E_CHUNK", 1024)
    prefill_row = _env_int("DEEPSEEK_V4_E2E_PREFILL_ROW", 4)
    if chunk_size <= 0 or chunk_size % ALIGNMENT:
        raise ValueError(f"DEEPSEEK_V4_E2E_CHUNK={chunk_size} must be a positive multiple of {ALIGNMENT}")

    # --- prompt ----------------------------------------------------------------------------------- #
    progress.step("[1/8] tokenizer and prompt")
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else config.eos_token_id
    if isinstance(pad_id, list):
        pad_id = pad_id[0]
    if "DEEPSEEK_V4_E2E_TEXT" in os.environ:
        text = os.environ["DEEPSEEK_V4_E2E_TEXT"]
        if text == "decode_demo":  # the decode demo's own prompt (128 tokens chat-templated)
            text = _DEFAULT_TEXT
        prompt_ids = _tokenize_chat(tokenizer, text)
    else:
        prompt_file = Path(os.environ.get("DEEPSEEK_V4_PREFILL_PROMPT", _DEFAULT_PROMPT_FILE))
        prompt_ids, info = _build_prompt_ids(tokenizer, json.loads(prompt_file.read_text())[0], prompt_len)
        logger.info(f"prompt from {prompt_file.name}: {info}")
    real_len = len(prompt_ids)
    aligned = real_len // ALIGNMENT * ALIGNMENT
    tail = real_len - aligned
    if real_len + max_new >= _MAX_CONTEXT:
        raise ValueError(
            f"prompt ({real_len}) + max new tokens ({max_new}) must stay below {_MAX_CONTEXT}: the hand-off does not "
            "carry the CSA indexer's key cache (lower DEEPSEEK_V4_E2E_PROMPT_LEN / DEEPSEEK_V4_MAX_NEW_TOKENS)"
        )
    logger.info(
        f"prompt: {real_len} tokens = {aligned} prefilled ({math.ceil(aligned / chunk_size) if aligned else 0} "
        f"chunk(s) of up to {chunk_size}) + {tail} replayed through decode; up to {max_new} new tokens"
    )
    logger.info(f"prompt starts : {tokenizer.decode(prompt_ids[:24])!r}")
    logger.info(f"prompt ends   : {tokenizer.decode(prompt_ids[-80:])!r}")

    # --- the decode model (rows 0-1), static state, session ---------------------------------------- #
    progress.step("[2/8] decode model (mesh rows 0-1) - a warm weight cache just uploads")
    needed = real_len + max_new
    max_seq = round_context(_traced_max_seq(config, needed), set(config.compress_rates.values()), _PAGE_BLOCK_SIZE)
    rope = _build_rope(config, max_seq)
    t0 = time.perf_counter()
    decode, lm_head, loader, config = _construct_model(
        mesh_device, prefetcher, tp_size=_TP_SIZE, system_config=None, loader=loader, config=config
    )
    _assert_decode_parallelism(decode, _TP_SIZE)
    num_layers = decode.num_layers
    logger.info(f"decode model: {num_layers} layers built in {time.perf_counter() - t0:.1f}s")
    decode.prepare_static_decode(
        rope, max_seq, lm_head=lm_head, num_sessions=1, total_tokens=max_seq, block_size=_PAGE_BLOCK_SIZE
    )
    if decode.context_limit is not None and decode.context_limit < needed:
        raise ValueError(f"decode context capped at {decode.context_limit} < the {needed} tokens needed")
    sid = decode.open_session()
    decode.activate_session(sid)

    progress.step("[3/8] throw-away decode step: captures the traces (its cache writes are overwritten later)")
    t0 = time.perf_counter()
    decode.decode_traced(pad_id, 0)
    logger.info(f"trace capture + first step: {time.perf_counter() - t0:.1f}s")

    # --- prefill ---------------------------------------------------------------------------------- #
    next_id = None
    if aligned:
        progress.step(f"[4/8] prefill model (mesh rows {prefill_row}-{prefill_row + 1}) - {num_layers} layers")
        system_config = decode.system_config
        set_active_system_config(system_config)
        num_stages = _env_int("DEEPSEEK_V4_PREFILL_STAGES", 2)
        submeshes = [
            mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(prefill_row + i, 0))
            for i in range(num_stages)
        ]
        placement = plan_layer_placement(num_layers, num_stages, 1)
        layer_devices = [submeshes[k] for k in placement]
        cache = WeightCache(os.path.join(_CACHE_DIR, os.path.basename(_DEFAULT_MODEL_DIR))) if _CACHE_DIR else None
        t0 = time.perf_counter()
        prefill = DeepSeekV4PrefillModel(
            config,
            checkpoint_weights(loader, config, num_layers),
            layer_devices[0],
            rope,
            expert_provider=checkpoint_expert_provider(loader),
            num_layers=num_layers,
            cache=cache,
            weight_dtype=_ATTENTION_WEIGHT_DTYPE,
            expert_dtype=system_config.decode.ttnn_weight_dtype,
            tp_size=_TP_SIZE,
            layer_devices=layer_devices,
            dense_csa=True,
            progress=progress,
        )
        prefill.synchronize("uploads")
        logger.info(f"prefill model built in {time.perf_counter() - t0:.1f}s")

        progress.step(f"[5/8] prefill of {aligned} tokens")
        ids = torch.tensor(prompt_ids[:aligned], dtype=torch.long).unsqueeze(0)
        chunk_times: list[float] = []

        def on_chunk(index: int, start: int, end: int, seconds: float) -> None:
            chunk_times.append(seconds)
            logger.info(f"prefill chunk {index + 1} tokens [{start}, {end}): {seconds:.2f}s")

        logits, states = prefill.prefill(ids, chunk_size=chunk_size, on_chunk=on_chunk)
        prefill_seconds = sum(chunk_times)
        logger.info(
            f"prefill: {aligned} tokens in {prefill_seconds:.2f}s ({prefill_seconds / aligned * 1000:.2f} ms/token, "
            f"{aligned / prefill_seconds:.1f} tok/s; includes first-run compilation)"
        )
        row = prefill.to_host(logits, prefill.head_device).reshape(-1)
        assert torch.isfinite(row).all(), "non-finite prefill logits"
        top = row.topk(5)
        logger.info(
            f"prefill's own next token after position {aligned - 1} (top 5): "
            + ", ".join(f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices))
        )
        if tail == 0:
            next_id = int(row.argmax())

        # --- the hand-off ------------------------------------------------------------------------- #
        progress.step("[6/8] hand-off: prefill state -> decode buffers")
        t0 = time.perf_counter()
        handed = load_prefill_state_into_decode(
            decode, prefill, states, sid, progress=lambda m: progress(m, important=True)
        )
        assert handed == aligned
        logger.info(f"hand-off of {handed} tokens: {time.perf_counter() - t0:.2f}s")
    else:
        logger.warning("prompt is shorter than one 128-token block: nothing to prefill, decode replays it all")
        decode.reset_session(sid)  # the throw-away step's window state must not leak into the prompt
        decode.reset_static_caches()

    # --- the ragged tail (or the whole short prompt) through decode ------------------------------------ #
    progress.step(f"[7/8] replaying {tail if aligned else real_len} prompt token(s) through decode")
    t0 = time.perf_counter()
    for pos in range(aligned, real_len):
        logits = decode.decode_traced(prompt_ids[pos], pos).reshape(1, -1).float()
        next_id = int(logits[0].argmax())
    if real_len > aligned:
        logger.info(f"tail replay: {real_len - aligned} tokens in {time.perf_counter() - t0:.2f}s")
    assert next_id is not None
    logger.info(f"first generated token: {next_id} {tokenizer.decode([next_id])!r}")

    # --- generation ------------------------------------------------------------------------------- #
    progress.step(f"[8/8] decode: up to {max_new} tokens")
    eos = _eos_ids(config)
    generated = [next_id]
    started = time.perf_counter()
    step_times: list[float] = []
    for step in range(1, max_new):
        if generated[-1] in eos:
            logger.info("hit EOS; stopping")
            break
        t0 = time.perf_counter()
        logits = decode.decode_traced(generated[-1], real_len + step - 1).reshape(1, -1).float()
        generated.append(int(logits[0].argmax()))
        step_times.append(time.perf_counter() - t0)
        if step % 32 == 0:
            recent = step_times[-32:]
            logger.info(f"decode {step}/{max_new}: {len(recent) / sum(recent):.2f} tok/s (last {len(recent)} tokens)")
    if step_times:
        logger.info(
            f"decode: {len(generated)} tokens in {time.perf_counter() - started:.1f}s, "
            f"{len(step_times) / sum(step_times):.2f} tok/s steady (first step {step_times[0]:.2f}s)"
        )

    logger.info("PROMPT (last 600 chars):\n" + tokenizer.decode(prompt_ids)[-600:])
    logger.info(f"GENERATED ({len(generated)} tokens):\n{tokenizer.decode(generated)}")
    logger.info(f"pool usage: {decode.session_usage()}")
    progress.step("done")
    assert generated and all(0 <= t < config.vocab_size for t in generated)
