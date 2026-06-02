# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Language demo smoke test — mirrors langauge_demo.py's generate() flow.

Uses real tokenization, prefill_next_token (on-device argmax), and decode_next_token
exactly as the demo does.  Default is 1 layer so the Tracy profiler run is fast.

Run::

    export MISTRAL4_LANG_DEMO_SMOKE=1
    export MISTRAL4_LANG_DEMO_N_LAYERS=1        # optional; default 1
    export MISTRAL4_LANG_DEMO_PROMPT="The capital of France is"   # optional
    export MISTRAL4_LANG_DEMO_MAX_NEW_TOKENS=8  # optional; default 8
    export MISTRAL4_WEIGHT_CACHE_DIR=/tmp/mistral4_weights  # optional; cache quantized weights to skip re-quantization
    export MESH_DEVICE=P150x8
    pytest models/experimental/mistral_small_4_119b/tests/test_language_demo_smoke.py -v -s --timeout=0
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
    HF_MODEL_ID,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")

_N_LAYERS = int(os.environ.get("MISTRAL4_LANG_DEMO_N_LAYERS", "1"))
_PROMPT = os.environ.get("MISTRAL4_LANG_DEMO_PROMPT", "The capital of France is")
_MAX_NEW_TOKENS = int(os.environ.get("MISTRAL4_LANG_DEMO_MAX_NEW_TOKENS", "8"))


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


def _precompute_rope_table(rotary_cls, text_config, max_seq_len: int):
    rotary = rotary_cls(text_config).eval().to(torch.bfloat16)
    dummy = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    pos_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    return rotary(dummy, pos_ids)


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_LANG_DEMO_SMOKE") != "1",
    reason="Set MISTRAL4_LANG_DEMO_SMOKE=1 to run the language demo smoke test.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_language_demo_smoke(reset_seeds, mesh_device):
    """
    Mirror of langauge_demo.generate():
      - tokenize a real prompt
      - prefill_next_token (on-device argmax)
      - decode_next_token loop with Tracy signpost around steady-state steps
    """
    from transformers import AutoConfig, AutoTokenizer
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

    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception as exc:
        pytest.skip(f"Tokenizer load failed: {exc}")

    input_ids = tokenizer(_PROMPT, return_tensors="pt").input_ids  # [1, seq_len]
    seq_len = input_ids.shape[1]
    logger.info(f"Prompt: {_PROMPT!r}  →  {seq_len} tokens")

    total_positions = seq_len + _MAX_NEW_TOKENS
    max_seq_len = total_positions + 64

    logger.info(f"Building TtMistral4TextModel ({_N_LAYERS} layers, max_seq_len={max_seq_len})…")
    try:
        model = TtMistral4TextModel(
            mesh_device=mesh_device,
            state_dict=state_dict,
            text_config=text,
            num_decoder_layers=_N_LAYERS,
            max_seq_len=max_seq_len,
        )
    except Exception as exc:
        pytest.fail(f"TtMistral4TextModel init failed: {exc}")

    del state_dict

    cos_full, sin_full = _precompute_rope_table(Mistral4RotaryEmbedding, text, total_positions)
    model.cache_rope_tables(cos_full, sin_full)

    # ── Prefill ──────────────────────────────────────────────────────────────
    logger.info(f"Running prefill (seq_len={seq_len})…")
    t0 = time.perf_counter()
    next_id = model.prefill_next_token(input_ids)
    prefill_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Prefill done in {prefill_ms:.0f} ms, first token id={next_id}")

    assert isinstance(next_id, int), f"prefill_next_token should return int, got {type(next_id)}"

    print(f"\n{_PROMPT}", end="", flush=True)
    print(tokenizer.decode([next_id], skip_special_tokens=False), end="", flush=True)

    # ── Decode loop (matches langauge_demo.generate exactly) ─────────────────
    # Step 0 is the first decode (JIT warm-up). Tracy signpost fires before
    # the last _MAX_NEW_TOKENS-1 steps so the profiler captures steady-state.
    next_id_tensor = torch.tensor([[next_id]], dtype=torch.long)
    generated_ids = [next_id]
    decode_times = []
    profile_start_step = 1  # everything after the first step is profiled

    for step in range(1, _MAX_NEW_TOKENS):
        if step == profile_start_step:
            ttnn.synchronize_device(mesh_device)
            signpost("Performance pass")

        current_pos = seq_len + step - 1
        t_dec = time.perf_counter()
        tok_id = model.decode_next_token(next_id_tensor, current_pos)
        decode_times.append((time.perf_counter() - t_dec) * 1000)

        generated_ids.append(tok_id)
        print(tokenizer.decode([tok_id], skip_special_tokens=False), end="", flush=True)

        if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
            logger.info(f"EOS at step {step}")
            break

        next_id_tensor = torch.tensor([[tok_id]], dtype=torch.long)

    print()

    if decode_times:
        avg_ms = sum(decode_times) / len(decode_times)
        logger.info(
            f"Generated {len(generated_ids)} tokens | " f"decode avg {avg_ms:.0f} ms/tok ({1000/avg_ms:.1f} tok/s)"
        )

    logger.info(f"{_N_LAYERS}-layer language demo smoke: PASSED")
