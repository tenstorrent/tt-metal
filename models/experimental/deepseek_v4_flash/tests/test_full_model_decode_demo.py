# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decode demo on the full ttnn ``DeepSeekV4Model``.

Builds the whole model once via :class:`DeepSeekV4Model`. There is no dedicated
prefill: a chat prompt is "prefilled" by replaying the decode step once per
prompt token (at ascending absolute positions), seeding every layer's sliding
K=V + compressor cache in place, then generation continues one token per step
(``S = 1``) against that cache. The RoPE tables are produced once for the
maximum length; each decode step slices the single position row(s) it needs.

All weights live on device in ``bfloat4_b``. The full 43-layer stack does not fit
a single Blackhole's 32 GB; cap it with ``DEEPSEEK_V4_DECODE_LAYERS=N`` and set
``DEEPSEEK_V4_CACHE_DIR`` to reuse the converted ttnn weight tiles across runs.

Run it (ttnn venv)::

    DEEPSEEK_V4_DECODE_LAYERS=4 DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    DEEPSEEK_V4_MAX_NEW_TOKENS=16 \\
    pytest -s models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.encoding_dsv4 import render_message
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4Model
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.weight_loader import (
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)

_DEFAULT_MODEL_DIR = "/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
_DEFAULT_TEXT = "Tell me the name of the top 10 songs of all time."
if int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "1024")) < 10:
    _DEFAULT_TEXT = "Tell"
_WEIGHT_DTYPE = ttnn.bfloat4_b
_CACHE_DIR = os.environ.get("DEEPSEEK_V4_CACHE_DIR", "../cache")


def _pad_to_tile(n: int) -> int:
    return ((n + 31) // 32) * 32


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(Path(_DEFAULT_MODEL_DIR))
    except FileNotFoundError:
        return False
    return True


def _w(loader: DeepseekV4WeightLoader, name: str):
    """Lazy (dequantized) fetch -> thunk (a populated tile cache skips the read)."""
    return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


def _build_rope(config, max_seq: int) -> dict:
    """YaRN RoPE tables (cos/sin halves) spanning ``max_seq`` for every layer family.

    ``win[cr]`` holds one windowed table per distinct compress-rate (CSA / HCA
    layers); decode slices the rows it needs from the max-length tables.
    """
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M

    dummy = torch.zeros(1, max_seq, 1, dtype=torch.float32)
    rotary = M.DeepseekV4RotaryEmbedding(config).to(torch.float32)

    def half(layer_type: str, position_ids: torch.Tensor):
        cos, sin = rotary(dummy, position_ids=position_ids, layer_type=layer_type)
        return cos[0].contiguous(), sin[0].contiguous()

    positions = torch.arange(max_seq).unsqueeze(0)
    rope = {
        "main": half("main", positions),
        "compress": half("compress", positions),
        "win": {},
    }
    for cr in sorted({int(v) for v in config.compress_rates.values()}):
        win_pos = (torch.arange(max_seq // cr) * cr).unsqueeze(0)
        rope["win"][cr] = half("compress", win_pos)
    return rope


def _build_and_prefill(mesh_device, text: str):
    """Build the full ttnn model, prepare the static traced-decode buffers, and
    prefill ``text`` one token at a time. Returns the populated state shared by
    the decode demo and the max-perf measurement tests."""
    from transformers import AutoTokenizer
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)

    max_new_tokens = int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "1024"))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else config.eos_token_id

    # Wrap the user input in the V4 chat template, tokenize, and build the RoPE
    # tables for the longest sequence we might decode (prompt + new tokens).
    # ``DEEPSEEK_V4_TRACED_DECODE``: replay one captured ttnn trace per submesh per
    # step (fixed-size in-place caches) instead of the host-bound eager decode.
    traced = os.environ.get("DEEPSEEK_V4_TRACED_DECODE", "1") not in ("0", "", "false", "False")

    prompt = render_message(0, [{"role": "user", "content": text}], "chat")
    prompt_ids: list[int] = list(tokenizer(prompt)["input_ids"])
    real_len = len(prompt_ids)
    max_seq = _pad_to_tile(real_len + max_new_tokens)
    if traced:
        # The fixed compressor buffers tile cleanly into windows only if the
        # capacity is a multiple of every compress-rate, so round the span up.
        crs = {int(v) for v in config.compress_rates.values()}
        step = math.lcm(32, *crs) if crs else 32
        max_seq = ((max_seq + step - 1) // step) * step
    rope = _build_rope(config, max_seq)

    max_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    top_cache = WeightCache(os.path.join(_CACHE_DIR, "full_decode", "ttnn")) if _CACHE_DIR else None

    # --- build the full model + lm_head once -------------------------------- #
    model = DeepSeekV4Model(
        config,
        loader,
        mesh_device,
        cache=top_cache,
        weight_dtype=_WEIGHT_DTYPE,
        max_layers=max_layers,
        use_submeshes=True,
    )
    lm_head = Linear(
        _w(loader, "lm_head.weight"),
        model.last_device,
        top_cache.file("lm_head") if top_cache else None,
        dtype=_WEIGHT_DTYPE,
    )
    logger.info(f"built DeepSeekV4Model with {model.num_layers}/{config.num_hidden_layers} layers")

    # --- prefill the prompt by replaying decode one token at a time --------- #
    # There is no dedicated prefill: each prompt token is fed at its absolute
    # position through the (eager or traced) decode path, filling the in-place
    # caches exactly as a full-sequence prefill would. The logits after the final
    # prompt token give the first generated token. The fixed-size traced caches
    # are allocated empty here (lm_head folded into the last submesh's trace).
    if traced:
        model.prepare_static_decode(rope, max_seq, lm_head=lm_head)
        logger.info("traced decode: prepared empty static buffers; trace captured on first prefill step")
    else:
        model.reset_caches()

    next_id = pad_id
    for pos in range(real_len):
        if traced:
            logits_tt = model.decode_traced(prompt_ids[pos], pos)  # [1, 1, vocab] (lm_head in-trace)
            logits = ttnn.to_torch(logits_tt).reshape(1, -1).float()
        else:
            hidden = model.decode(prompt_ids[pos], pos, rope)  # [1, 1, D]
            logits = ttnn.to_torch(lm_head(hidden)).reshape(1, -1).float()
        next_id = int(logits[0].argmax().item())
    logger.info(f"prefill ({real_len} tokens) -> token id {next_id} {tokenizer.decode([next_id])!r}")

    return {
        "model": model,
        "lm_head": lm_head,
        "tokenizer": tokenizer,
        "config": config,
        "rope": rope,
        "prompt_ids": prompt_ids,
        "real_len": real_len,
        "max_seq": max_seq,
        "max_new_tokens": max_new_tokens,
        "eos_id": config.eos_token_id,
        "next_id": next_id,
        "traced": traced,
    }


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)  # heavy: bf4 conversion of every expert + many decode steps
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "num_command_queues": 2})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("text", (_DEFAULT_TEXT,))
def test_full_model_decode_demo(mesh_device, reset_seeds, text: str) -> None:
    import time

    state = _build_and_prefill(mesh_device, text)
    model, lm_head, tokenizer = state["model"], state["lm_head"], state["tokenizer"]
    rope, prompt_ids, real_len = state["rope"], state["prompt_ids"], state["real_len"]
    max_seq, max_new_tokens, eos_id = state["max_seq"], state["max_new_tokens"], state["eos_id"]
    traced, next_id = state["traced"], state["next_id"]
    generated: list[int] = [next_id]

    # Each step feeds the previously generated token at its absolute position and
    # reads back the single-token logits (no recompute over the prior context).
    decode_tokens = 0
    decode_time = 0.0
    for step in range(1, max_new_tokens):
        if next_id == eos_id:
            logger.info("hit EOS; stopping")
            break
        pos = real_len + step - 1  # absolute position of the token being fed back
        if pos >= max_seq:  # ran past the precomputed RoPE span
            logger.warning(f"hit max RoPE length {max_seq}; stopping at {len(generated)} tokens")
            break
        t0 = time.perf_counter()
        if traced:
            logits_tt = model.decode_traced(next_id, pos)  # [1, 1, vocab] (lm_head in-trace)
            logits = ttnn.to_torch(logits_tt).reshape(1, -1).float()  # forces device sync
        else:
            hidden = model.decode(next_id, pos, rope)  # [1, 1, D]
            logits = ttnn.to_torch(lm_head(hidden)).reshape(1, -1).float()  # forces device sync
        next_id = int(logits[0].argmax().item())
        decode_time += time.perf_counter() - t0
        decode_tokens += 1
        generated.append(next_id)
        logger.info(f"step {step:3d} (pos {pos:4d}): token id {next_id} {tokenizer.decode([next_id])!r}")

        # Running decode throughput, reported every 10 generated tokens.
        if decode_tokens % 10 == 0:
            logger.info(
                f"decode throughput: {decode_tokens / decode_time:.2f} tok/s "
                f"({decode_tokens} tokens in {decode_time:.2f}s)"
            )
            decode_tokens = 0
            decode_time = 0.0

    if decode_tokens:
        logger.info(
            f"decode throughput (final): {decode_tokens / decode_time:.2f} tok/s "
            f"({decode_tokens} tokens in {decode_time:.2f}s)"
        )

    assert generated, "no tokens were generated"
    logger.info(f"PROMPT    : {tokenizer.decode(prompt_ids)!r}")
    logger.info(f"GENERATED : {tokenizer.decode(generated)!r}  ({len(generated)} tokens)")


@pytest.mark.skip(reason="perf benchmark; run explicitly with DEEPSEEK_V4_PERF_ITERS set")
@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)  # heavy: bf4 conversion of every expert + trace capture
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "num_command_queues": 2})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("text", (_DEFAULT_TEXT,))
def test_full_model_decode_max_perf(mesh_device, reset_seeds, text: str) -> None:
    """Measure the traced-decode throughput ceiling.

    Replays the captured trace back-to-back with a *fixed* input at a fixed
    position, never reading the output. This drops the autoregressive dependency
    (host argmax + device->host sync per step) so the device runs the trace
    uninterrupted. Skipped by default; enable by setting ``DEEPSEEK_V4_PERF_ITERS``.
    """
    import time

    state = _build_and_prefill(mesh_device, text)
    model, real_len, next_id, traced = state["model"], state["real_len"], state["next_id"], state["traced"]
    assert traced, "max-perf measurement requires the traced decode path"

    perf_iters = int(os.environ.get("DEEPSEEK_V4_PERF_ITERS", "100"))
    fixed_pos = real_len  # any valid in-range position; correctness is irrelevant here
    model._set_step_inputs(next_id, fixed_pos)  # write the (fixed) inputs once
    if not model._traced_captured:
        model._capture_traces()

    def _replay() -> None:
        for sm in model.submeshes_io:
            ttnn.execute_trace(sm["device"], sm["tid"], cq_id=0, blocking=False)

    def _sync() -> None:
        for sm in model.submeshes_io:
            ttnn.synchronize_device(sm["device"])

    _replay()  # warmup
    _sync()
    t0 = time.perf_counter()
    for _ in range(perf_iters):
        _replay()
    _sync()  # single host sync after all replays
    dt = time.perf_counter() - t0
    logger.info(f"MAX PERF: {perf_iters / dt:.2f} tok/s ({perf_iters} iters in {dt:.3f}s)")


@pytest.mark.skip("Test disabled by default")
@pytest.mark.timeout(14400)  # heavy: bf4 conversion of every expert + trace capture
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "num_command_queues": 2})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("text", (_DEFAULT_TEXT,))
def test_full_model_decode_on_device_sampling(mesh_device, reset_seeds, text: str) -> None:
    """Greedy (top-1) sampling done on device, fed back without going to host.

    After prefill, runs a burst of decode steps where each step argmaxes the logits
    on device and copies the sampled id straight back into the embedding input
    buffer (no per-step device->host round trip). All sampled token ids are read
    back to the host in a single transfer at the end.
    """
    import time

    n_iters = int(os.environ.get("DEEPSEEK_V4_SAMPLE_ITERS", "25"))

    state = _build_and_prefill(mesh_device, text)
    model, tokenizer = state["model"], state["tokenizer"]
    real_len, next_id, traced = state["real_len"], state["next_id"], state["traced"]
    assert traced, "on-device sampling requires the traced decode path"

    t0 = time.perf_counter()
    tokens = model.decode_sampled_burst(next_id, real_len, n_iters)  # single host transfer at the end
    dt = time.perf_counter() - t0

    assert len(tokens) == n_iters
    logger.info(f"on-device sampling: {n_iters} tokens in {dt:.3f}s ({n_iters / dt:.2f} tok/s)")
    logger.info(f"SAMPLED IDS : {tokens}")
    logger.info(f"SAMPLED TEXT: {tokenizer.decode(tokens)!r}")
