# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed functional-decoder profiler entry points.

Run through ``python -m tracy -r -p -v -m pytest ...``. Signposts isolate one
warmed prefill and five steady-state decode trace replays.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tracy import signpost

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _assert_pcc,
    _config,
    _page_table,
    _positions,
    _real_state,
    _reference_decode,
    _reference_decode_zero_prefix,
    _reference_prefill,
    _synthetic_state,
    _to_torch_decode,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder

RECORDED_ACTIVATIONS = Path(__file__).parents[1] / "doc/optimized_decoder/activations/layer0_inputs.safetensors"


def _decode_inputs(batch, config):
    if os.environ.get("PHI35_REAL_WEIGHTS") == "1":
        recorded = load_file(RECORDED_ACTIVATIONS)
        hidden = recorded["token_embeddings"][127 : 127 + batch].unsqueeze(1)
        print(f"ACTIVATION_SOURCE recorded_target path={RECORDED_ACTIVATIONS} batch={batch}")
        return hidden
    return torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(20 + batch)).to(
        torch.bfloat16
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_profile_warmed_prefill(mesh_device, batch):
    config = _config()
    decoder = FunctionalDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
    )
    hidden = torch.randn(batch, 128, config.hidden_size, generator=torch.Generator().manual_seed(11 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_prefill(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    decoder.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL")
    start = time.perf_counter()
    output = decoder.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    signpost("PERF_PREFILL_END")
    assert tuple(output.shape) == (1, batch, 128, config.hidden_size)
    print(f"PERF_RESULT mode=prefill batch={batch} sequence=128 warmed_ms={elapsed_ms:.6f}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_profile_traced_decode(mesh_device, batch):
    config = _config()
    state = _real_state() if os.environ.get("PHI35_REAL_WEIGHTS") == "1" else _synthetic_state(config)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
    )
    hidden = _decode_inputs(batch, config)
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    current_positions = _positions([127] * batch, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    recorded_past = None
    if os.environ.get("PHI35_REAL_WEIGHTS") == "1":
        tokens = load_file(RECORDED_ACTIVATIONS)["token_embeddings"]
        prefixes = torch.stack([tokens[user : user + 127] for user in range(batch)])
        _, recorded_past = _reference_prefill(config, state, prefixes)
        decoder.prefill_forward(
            _to_tt_prefill(prefixes, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
        )
        ttnn.synchronize_device(mesh_device)
        print(f"CACHE_SOURCE functional_prefill recorded_target_prefix prefix_length=127 batch={batch}")

    def decode():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            use_long_rope=False,
        )

    decode()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples = []
        signpost(f"PERF_DECODE_B{batch}")
        for _ in range(10):
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append(1000 * (time.perf_counter() - start))
        signpost(f"PERF_DECODE_B{batch}_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    reference = (
        _reference_decode(config, state, hidden, 127, recorded_past)
        if recorded_past is not None
        else _reference_decode_zero_prefix(config, state, hidden, [127] * batch, use_long=False)
    )
    _assert_pcc(f"functional-perf-decode-recorded-b{batch}", reference, _to_torch_decode(output))
    print(
        f"PERF_RESULT mode=decode batch={batch} context=128 trace_replays=10 "
        f"mean_ms={sum(samples) / len(samples):.6f} min_ms={min(samples):.6f}"
    )
