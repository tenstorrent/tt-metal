# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN-only smoke test for Mistral-Small-4 prefill — no HF reference, no PCC.

Loads weights, builds TtMistral4TextModel, runs ``prefill_next_token`` (the
canonical end-to-end inference path: prefill + last-token argmax fully on
device, returns one uint32 across PCIe). Validates the token id is a valid
vocab index. Use this to iterate on prefill performance and memory without
the ~40-minute CPU reference forward pass.

The inference path is strictly on device — no torch fallback, no logits ever
crossing the PCIe bus.

Run::

    export MISTRAL4_PREFILL_SMOKE=1
    export MISTRAL4_PREFILL_N_LAYERS=36
    export MESH_DEVICE=T3K
    pytest models/experimental/mistral_small_4_119b/tests/test_text_prefill_smoke.py -v -s --timeout=0

Device-perf profiling (per-op timing of one prefill step)::

    TT_METAL_PROFILER_MID_RUN_DUMP=1 python -m tracy -r -m -p -v \
        --op-support-count 10000 pytest \
        models/experimental/mistral_small_4_119b/tests/test_text_prefill_smoke.py -v -s --timeout=0

A tracy signpost ("Performance pass") is emitted between the compile pass and
the measured pass; tracy's per-op CSV is filtered to ops after the last
signpost so the report contains exactly one measured prefill.
"""

from __future__ import annotations

import os
import time

import psutil
import pytest
import torch
from loguru import logger


def _log_mem(tag: str) -> None:
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / 1e9
    sys = psutil.virtual_memory()
    logger.info(f"[MEM {tag}] RSS={rss_gb:.1f} GB  sys avail={sys.available / 1e9:.1f} GB")


import ttnn
from tracy import signpost
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_VOCAB_SIZE,
    HF_MODEL_ID,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")

_N_LAYERS = int(os.environ.get("MISTRAL4_PREFILL_N_LAYERS", "1"))


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
    os.environ.get("MISTRAL4_PREFILL_SMOKE") != "1",
    reason="Set MISTRAL4_PREFILL_SMOKE=1 to run.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_prefill_smoke(reset_seeds, mesh_device):
    """Build TtMistral4TextModel and run prefill_next_token end-to-end on device."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    text = cfg.text_config

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_N_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception as exc:
        pytest.skip(f"Tokenizer load failed: {exc}")

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Optional: pad/truncate to a fixed seq_len for perf profiling.
    # Set MISTRAL4_PREFILL_SEQ_LEN=128 to run prefill at a fixed length.
    target_seq_len = int(os.environ.get("MISTRAL4_PREFILL_SEQ_LEN", "0"))
    if target_seq_len > 0:
        cur = input_ids.shape[1]
        if cur < target_seq_len:
            # Repeat the last token to pad up. Padding token id doesn't matter
            # for perf profiling — we only check that we get back a valid id.
            pad_token = input_ids[:, -1:].repeat(1, target_seq_len - cur)
            input_ids = torch.cat([input_ids, pad_token], dim=1)
        else:
            input_ids = input_ids[:, :target_seq_len]

    seq_len = input_ids.shape[1]
    logger.info(f"Prompt: {prompt!r}  →  {seq_len} tokens")

    logger.info(f"Building TtMistral4TextModel ({_N_LAYERS} layers)...")
    _log_mem("before model construction")
    model = TtMistral4TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text,
        num_decoder_layers=_N_LAYERS,
        max_seq_len=seq_len + 64,
    )
    _log_mem("after model construction")

    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    hidden0 = torch.nn.functional.embedding(
        input_ids,
        state_dict["language_model.model.embed_tokens.weight"].to(torch.bfloat16),
    )
    cos_full, sin_full = rotary(hidden0, position_ids)
    model.cache_rope_tables(cos_full, sin_full)

    # Compile pass: JIT-compile programs. The next call (under the signpost) is
    # what tracy will profile.
    logger.info(f"Compile pass: prefill_next_token (seq_len={seq_len})...")
    _log_mem("before compile pass")
    t_compile = time.perf_counter()
    _ = model.prefill_next_token(input_ids)
    ttnn.synchronize_device(mesh_device)
    t_compile = time.perf_counter() - t_compile
    logger.info(f"Compile pass wall-clock: {t_compile*1e3:.2f} ms")
    _log_mem("after compile pass")

    # prefill_next_token consumes the cached RoPE table (full-range ttnn.slice
    # aliases the buffer, and the trailing ttnn.deallocate frees it). Re-cache
    # before the measured pass. Done before the signpost so the upload is not
    # in the report.
    model.cache_rope_tables(cos_full, sin_full)

    # ── Signposted measured pass — what shows up in the perf report ─────────
    # Single clean prefill_next_token call. Tracy filters per-op data to ops
    # AFTER the last signpost, so anything we run after this would pollute the
    # report.
    signpost("Performance pass")
    logger.info("Measured prefill pass...")
    t_measured = time.perf_counter()
    next_token = model.prefill_next_token(input_ids)
    ttnn.synchronize_device(mesh_device)
    t_measured = time.perf_counter() - t_measured
    logger.info(f"Measured prefill wall-clock: {t_measured*1e3:.2f} ms")
    _log_mem("after measured pass")

    assert (
        isinstance(next_token, int) and 0 <= next_token < EXPECTED_VOCAB_SIZE
    ), f"prefill_next_token returned invalid token id: {next_token}"
    logger.info(f"Top predicted token id (on-device argmax): {next_token}")
    logger.info(f"PASSED — model constructed and prefill completed for {_N_LAYERS} layers")
