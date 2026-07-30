# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed functional-decoder profiler entry points.

Run through ``python -m tracy -r -p -v -m pytest ...``. Signposts isolate one
warmed prefill and five steady-state decode trace replays.
"""

from __future__ import annotations

import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _page_table,
    _positions,
    _synthetic_state,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_profile_warmed_prefill(mesh_device):
    config = _config()
    decoder = FunctionalDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=128,
    )
    hidden = torch.randn(1, 128, config.hidden_size, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
    tt_hidden = _to_tt_prefill(hidden, mesh_device)
    page_table = _page_table(1, 128, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    decoder.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL")
    start = time.perf_counter()
    output = decoder.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    signpost("PERF_PREFILL_END")
    assert tuple(output.shape) == (1, 1, 128, config.hidden_size)
    print(f"PERF_RESULT mode=prefill batch=1 sequence=128 warmed_ms={elapsed_ms:.6f}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_profile_traced_decode(mesh_device, batch):
    config = _config()
    decoder = FunctionalDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
    )
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(20 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    current_positions = _positions([0] * batch, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()

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
        for _ in range(5):
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append(1000 * (time.perf_counter() - start))
        signpost(f"PERF_DECODE_B{batch}_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert tuple(output.shape) == (1, 1, batch, config.hidden_size)
    print(
        f"PERF_RESULT mode=decode batch={batch} context=128 trace_replays=5 "
        f"mean_ms={sum(samples) / len(samples):.6f} min_ms={min(samples):.6f}"
    )
