# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill + decode smoke test for Mistral-Small-4 text model.

Runs prefill_next_token then N decode_next_token steps (on-device argmax).
Every token id is read back from device; no logits ever cross the PCIe bus.

Run manually::

    export MISTRAL4_DECODE_SMOKE=1
    export MISTRAL4_DECODE_N_LAYERS=2        # optional; default 2
    export MISTRAL4_DECODE_PREFILL_LEN=4     # optional; default 4
    export MISTRAL4_DECODE_N_STEPS=3         # optional; default 3
    export MESH_DEVICE=P150x8                # optional
    pytest models/experimental/mistral_small_4_119b/tests/test_text_decode_smoke.py -v -s --timeout=0

Device-perf profiling (per-op timing for one decode step)::

    TT_METAL_PROFILER_MID_RUN_DUMP=1 python -m tracy -r -m -p -v \
        --op-support-count 10000 pytest \
        models/experimental/mistral_small_4_119b/tests/test_text_decode_smoke.py -v -s --timeout=0

A tracy signpost ("Performance pass") is emitted right before the last decode
iteration; tracy's per-op CSV is filtered to ops after the last signpost, so
the report contains exactly one measured decode step.

The device profiler DRAM buffer is sized to hold a fixed number of programs;
the default (1000, see DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT in
tt_metal/impl/profiler/profiler_state_manager.cpp) is too small for a full
36-layer prefill + decode run and you get "Profiler DRAM buffers were full,
markers were dropped" followed by the post-run AssertionError
"Op N not present in cpp_device_perf_report.csv". Bump it with
--op-support-count 10000 (or higher if it still overflows). MID_RUN_DUMP=1
just makes tracy write the host-side CSV more often; it does NOT drain the
device buffer faster.
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
_N_STEPS = int(os.environ.get("MISTRAL4_DECODE_N_STEPS", "3"))
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
    base = {"num_command_queues": 1}
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
    """Prefill N tokens, greedily decode M steps on-device, check token ids are valid."""
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

    logger.info(f"Building TtMistral4TextModel ({_N_LAYERS} layers, max_seq={_MAX_SEQ_LEN})...")
    model = TtMistral4TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text,
        num_decoder_layers=_N_LAYERS,
        max_seq_len=_MAX_SEQ_LEN,
    )

    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, _PREFILL_LEN), dtype=torch.long)

    # Cache RoPE for every position prefill + decode will touch (one HF call).
    total_positions = _PREFILL_LEN + _N_STEPS
    full_position_ids = torch.arange(total_positions, dtype=torch.long).unsqueeze(0)
    cos_full, sin_full = rotary(torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16), full_position_ids)
    model.cache_rope_tables(cos_full, sin_full)

    logger.info(f"Running prefill (seq_len={_PREFILL_LEN})...")
    next_token = model.prefill_next_token(input_ids)
    assert (
        isinstance(next_token, int) and 0 <= next_token < vocab
    ), f"prefill_next_token returned invalid token id: {next_token}"
    logger.info(f"Prefill OK. First decode token (on-device argmax): {next_token}")

    decode_times = []
    for step in range(_N_STEPS):
        # Emit a tracy signpost right before the LAST decode step so the
        # per-op CSV is filtered to a single measured iteration. Earlier
        # steps act as warmup (step 0 is the JIT-compile pass).
        if step == _N_STEPS - 1:
            ttnn.synchronize_device(mesh_device)
            signpost("Performance pass")

        current_pos = _PREFILL_LEN + step
        tok_tensor = torch.tensor([[next_token]], dtype=torch.long)

        t0 = time.perf_counter()
        next_token = model.decode_next_token(tok_tensor, current_pos)
        ttnn.synchronize_device(mesh_device)
        dt = time.perf_counter() - t0
        decode_times.append(dt)

        assert (
            isinstance(next_token, int) and 0 <= next_token < vocab
        ), f"Step {step}: decode_next_token returned invalid token id: {next_token}"
        logger.info(f"Decode step {step} (pos={current_pos}): {dt*1e3:.2f} ms, next token={next_token}")

    # Step 0 is the JIT-compile pass, exclude from steady-state stats.
    steady = sorted(decode_times[1:]) if len(decode_times) > 1 else sorted(decode_times)
    med = steady[len(steady) // 2] * 1e3
    logger.info(
        f"Steady-state decode wall-clock: median={med:.2f} ms"
        f" (min={steady[0]*1e3:.2f}, max={steady[-1]*1e3:.2f}, n={len(steady)})"
    )
    logger.info(f"{_N_LAYERS}-layer prefill({_PREFILL_LEN})+decode({_N_STEPS}) smoke: PASSED")
