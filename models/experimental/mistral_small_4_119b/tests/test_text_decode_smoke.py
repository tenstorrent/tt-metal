# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill + decode smoke test for Mistral-Small-4 text model.

Runs a short prefill then steps through N decode iterations.
Checks: finite logits, correct shape at every step.

Run manually::

    export MISTRAL4_DECODE_SMOKE=1
    export MISTRAL4_DECODE_N_LAYERS=2        # optional; default 2
    export MISTRAL4_DECODE_PREFILL_LEN=4     # optional; default 4
    export MISTRAL4_DECODE_N_STEPS=5         # optional; default 5
    export MESH_DEVICE=P150x4                # optional
    pytest models/experimental/mistral_small_4_119b/tests/test_text_decode_smoke.py -v -s --timeout=0
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from tracy import signpost
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_VOCAB_SIZE,
    HF_MODEL_ID,
    TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")

_N_LAYERS = int(os.environ.get("MISTRAL4_DECODE_N_LAYERS", "2"))
_PREFILL_LEN = int(os.environ.get("MISTRAL4_DECODE_PREFILL_LEN", "4"))
_N_STEPS = int(os.environ.get("MISTRAL4_DECODE_N_STEPS", "5"))
_PROFILE_STEPS = max(1, int(os.environ.get("MISTRAL4_DECODE_PROFILE_STEPS", "1")))
_ON_DEVICE_ARGMAX = os.environ.get("MISTRAL4_DECODE_ON_DEVICE_ARGMAX", "0") == "1"
_MAX_SEQ_LEN = _PREFILL_LEN + _N_STEPS + 64  # small cache for smoke


def _state_dict_prefixes(n_layers: int) -> tuple:
    p = ["language_model.model.embed_tokens."]
    for i in range(n_layers):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    return tuple(p)


def _mesh_params():
    shape = mesh_device_request_param()
    base = {"trace_region_size": 30000000, "num_command_queues": 1}
    fabric = ttnn.FabricConfig.DISABLED if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D
    return [pytest.param(shape, {**base, "fabric_config": fabric}, id=f"mesh{shape[0]}x{shape[1]}")]


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_DECODE_SMOKE") != "1",
    reason="Set MISTRAL4_DECODE_SMOKE=1 to run the prefill+decode smoke test.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_text_decode_smoke(reset_seeds, mesh_device):
    """Prefill N tokens, greedily decode M steps, check shapes and finiteness."""
    from transformers import AutoConfig
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    text = cfg.text_config
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(text, attr):
            setattr(text, attr, "eager")

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_N_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    vocab = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY].shape[0]
    assert vocab == EXPECTED_VOCAB_SIZE

    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)

    # ── Build model ──────────────────────────────────────────────────────
    logger.info(f"Building TtMistral4TextModel ({_N_LAYERS} layers, max_seq={_MAX_SEQ_LEN})...")
    try:
        model = TtMistral4TextModel(
            mesh_device=mesh_device,
            state_dict=state_dict,
            text_config=text,
            num_decoder_layers=_N_LAYERS,
            max_seq_len=_MAX_SEQ_LEN,
        )
    except Exception as exc:
        pytest.fail(f"TtMistral4TextModel init failed: {exc}")

    # ── Prefill ──────────────────────────────────────────────────────────
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, _PREFILL_LEN), dtype=torch.long)

    # Cache RoPE for every position prefill + decode will touch (one HF call).
    total_positions = _PREFILL_LEN + _N_STEPS
    full_position_ids = torch.arange(total_positions, dtype=torch.long).unsqueeze(0)
    cos_full, sin_full = rotary(torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16), full_position_ids)
    model.cache_rope_tables(cos_full, sin_full)

    logger.info(f"Running prefill (seq_len={_PREFILL_LEN})...")
    prefill_logits = model.prefill(input_ids)

    assert prefill_logits.shape == (
        1,
        _PREFILL_LEN,
        vocab,
    ), f"Prefill logits shape {tuple(prefill_logits.shape)} != (1, {_PREFILL_LEN}, {vocab})"
    assert torch.isfinite(prefill_logits.to(torch.float32)).all(), "Non-finite values in prefill logits"
    logger.info(f"Prefill OK: shape={tuple(prefill_logits.shape)}")

    # ── Greedy decode loop ───────────────────────────────────────────────
    next_token = prefill_logits[0, -1, :].argmax().item()
    logger.info(f"First decode token (greedy from prefill): {next_token}")
    logger.info(f"On-device decode argmax mode: {_ON_DEVICE_ARGMAX}")

    # Step 0 is the JIT-compile pass (slow). Steps after the signpost are
    # the perf-measured iterations. Wall-clock is logged for each step so
    # we can see steady-state decode latency.
    decode_times = []
    profile_start_step = max(1, _N_STEPS - _PROFILE_STEPS)
    for step in range(_N_STEPS):
        if step == profile_start_step:
            ttnn.synchronize_device(mesh_device)
            signpost("Performance pass")
        current_pos = _PREFILL_LEN + step
        tok_tensor = torch.tensor([[next_token]], dtype=torch.long)

        t0 = time.perf_counter()
        if _ON_DEVICE_ARGMAX:
            next_token = model.decode_next_token(tok_tensor, current_pos)
            decode_logits = None
        else:
            decode_logits = model.decode(tok_tensor, current_pos)
        ttnn.synchronize_device(mesh_device)
        dt = time.perf_counter() - t0
        if step >= profile_start_step:
            decode_times.append(dt)

        if decode_logits is not None:
            assert decode_logits.shape == (
                1,
                1,
                vocab,
            ), f"Step {step}: decode logits shape {tuple(decode_logits.shape)} != (1, 1, {vocab})"
            assert torch.isfinite(
                decode_logits.to(torch.float32)
            ).all(), f"Step {step}: non-finite values in decode logits"
            next_token = decode_logits[0, 0, :].argmax().item()
        logger.info(f"Decode step {step} (pos={current_pos}): {dt*1e3:.2f} ms, next token={next_token}")

    if decode_times:
        ts = sorted(decode_times)
        med = ts[len(ts) // 2] * 1e3
        logger.info(
            f"Steady-state decode wall-clock: median={med:.2f} ms"
            f" (min={ts[0]*1e3:.2f}, max={ts[-1]*1e3:.2f}, n={len(decode_times)})"
        )
    logger.info(f"{_N_LAYERS}-layer prefill({_PREFILL_LEN})+decode({_N_STEPS}) smoke: PASSED")
