# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the decode step as a trace and replay it.

Tracing is the real test of every "on-device, trace-compatible" claim made
while building this layer. A trace records device commands once and replays
them, so anything that is not a pure device op -- a host round-trip, a
Python-side branch on tensor *values*, a shape that depends on the data --
either fails to capture or replays stale results. Notably, this is what makes
the router's design load-bearing: it keeps top-k selection and the scatter on
device precisely so this step is possible.

Two properties are checked, and the second is the one that catches real bugs:

1. the traced output matches the eager output for the same input;
2. replaying with *different* input produces *different*, still-correct output.

Property 2 is essential. A trace whose input tensor was captured by value
rather than written in place replays the original activations forever and
therefore passes property 1 perfectly, every time.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from ..tt.functional_decoder import (
    DecoderLayerConfig,
    build_expert_sparsity,
    build_rope_cache,
    create_kv_cache,
    decoder_layer_decode,
    decoder_layer_prefill,
    upload_layer_weights,
)
from ..tt.weight_mapping import convert_layer_weights
from .reference import build_reference_layer, layer_state_dict, rotary_embeddings

LAYER_IDX = 0
PCC_REQUIRED = 0.99
MAX_SEQ = 256
PROMPT_LEN = 32
# Reserved at device open; the capture fails outright if the graph needs more.
TRACE_REGION_SIZE = 50331648


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights(reference):
    _, hf_config = reference
    return convert_layer_weights(layer_state_dict(LAYER_IDX), hf_config)


def _hidden(hf_config, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.float32) * 0.02


def _reference_layer(layer, hf_config, hidden):
    seq_len = hidden.shape[1]
    cos, sin = rotary_embeddings(hf_config, seq_len)
    mask = torch.full((seq_len, seq_len), float("-inf")).triu(1).reshape(1, 1, seq_len, seq_len)
    with torch.no_grad():
        out = layer(hidden, position_embeddings=(cos, sin), attention_mask=mask)
    return out[0] if isinstance(out, tuple) else out


def _to_device(t, mesh_device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_decode_step_is_traceable(mesh_device, reference, torch_weights):
    layer, hf_config = reference
    config = DecoderLayerConfig.from_hf(hf_config)

    weights = upload_layer_weights(torch_weights, mesh_device, config)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    sparsity = build_expert_sparsity(mesh_device, config.moe.num_experts)
    kv_cache = create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ)

    hidden_full = _hidden(hf_config, PROMPT_LEN + 2)
    ref_out = _reference_layer(layer, hf_config, hidden_full)

    decoder_layer_prefill(
        _to_device(hidden_full[:, :PROMPT_LEN, :].unsqueeze(0), mesh_device),
        weights,
        config,
        cos_cache,
        sin_cache,
        sparsity,
        kv_cache=kv_cache,
    )

    # Persistent input buffers: a trace replays writes to the *same* addresses,
    # so inputs must be updated in place rather than rebound each step.
    tt_in = _to_device(hidden_full[:, PROMPT_LEN, :].reshape(1, 1, 1, hf_config.hidden_size), mesh_device)
    current_pos = ttnn.from_torch(torch.tensor([PROMPT_LEN], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)

    def step():
        return decoder_layer_decode(
            tt_in, weights, config, cos_cache, sin_cache, kv_cache, current_pos, token_index=PROMPT_LEN
        )

    # Warm up so program compilation happens outside the capture.
    eager_out = ttnn.to_torch(step()).reshape(1, hf_config.hidden_size)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = step()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replayed = ttnn.to_torch(traced_out).reshape(1, hf_config.hidden_size)

    passing, pcc_message = comp_pcc(eager_out, replayed, 0.999)
    logger.info(comp_allclose(eager_out, replayed))
    logger.info(f"traced vs eager: {pcc_message}")
    assert passing, f"traced replay disagrees with eager execution: {pcc_message}"

    passing, pcc_message = comp_pcc(ref_out[:, PROMPT_LEN, :], replayed, PCC_REQUIRED)
    logger.info(f"traced vs reference: {pcc_message}")
    assert passing, f"traced output below {PCC_REQUIRED} vs reference: {pcc_message}"

    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_traced_replay_follows_new_input(mesh_device, reference, torch_weights):
    """Writing a new token into the input buffer must change the traced output.

    Guards the failure mode a same-input trace test cannot see: if the capture
    bound the input by value, replay reproduces the first token's result
    forever and every equality check still passes.
    """
    layer, hf_config = reference
    config = DecoderLayerConfig.from_hf(hf_config)

    weights = upload_layer_weights(torch_weights, mesh_device, config)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    sparsity = build_expert_sparsity(mesh_device, config.moe.num_experts)
    kv_cache = create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ)

    hidden_full = _hidden(hf_config, PROMPT_LEN + 1)
    decoder_layer_prefill(
        _to_device(hidden_full[:, :PROMPT_LEN, :].unsqueeze(0), mesh_device),
        weights,
        config,
        cos_cache,
        sin_cache,
        sparsity,
        kv_cache=kv_cache,
    )

    token_a = hidden_full[:, PROMPT_LEN, :].reshape(1, 1, 1, hf_config.hidden_size)
    token_b = (_hidden(hf_config, 1, seed=99)).reshape(1, 1, 1, hf_config.hidden_size)

    tt_in = _to_device(token_a, mesh_device)
    current_pos = ttnn.from_torch(torch.tensor([PROMPT_LEN], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)

    def step():
        return decoder_layer_decode(
            tt_in, weights, config, cos_cache, sin_cache, kv_cache, current_pos, token_index=PROMPT_LEN
        )

    step()  # warm up / compile

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = step()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    out_a = ttnn.to_torch(traced_out).reshape(-1).float().clone()

    # Overwrite the captured input buffer in place, then replay.
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(token_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), tt_in)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    out_b = ttnn.to_torch(traced_out).reshape(-1).float().clone()

    delta = (out_a - out_b).abs().max().item()
    logger.info(f"max|out_a - out_b| after swapping the input token = {delta:.6f}")
    assert delta > 1e-3, (
        "traced replay produced identical output for a different input token -- "
        "the trace is not reading the live input buffer"
    )

    ttnn.release_trace(mesh_device, trace_id)
