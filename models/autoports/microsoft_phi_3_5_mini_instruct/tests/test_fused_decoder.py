# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import time

import pytest
import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _assert_pcc,
    _config,
    _page_table,
    _positions,
    _reference_decode_zero_prefix,
    _reference_prefill,
    _synthetic_state,
    _to_torch_decode,
    _to_torch_prefill,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.fused_decoder import FusedDecoder


def test_fused_runtime_dispatch_contract():
    source = inspect.getsource(FusedDecoder._mlp)
    assert "input_tensor_a_activations=[ttnn.UnaryOpType.SILU]" in source
    assert "ttnn.silu" not in source
    assert FusedDecoder._mlp is not FunctionalDecoder._mlp


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "mode,batch,seq_len",
    [
        ("prefill", 1, 31),
        ("prefill", 1, 33),
        ("prefill", 1, 65),
        ("decode", 1, 1),
        ("decode", 32, 1),
    ],
)
def test_fused_matches_functional_on_device(mesh_device, mode, batch, seq_len):
    config = _config()
    state = _synthetic_state(config)
    max_context = 96
    functional = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=max_context,
    )
    fused = FusedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=max_context,
    )
    page_table = _page_table(batch, max_context, mesh_device, permute=True)
    functional_cache = functional.create_paged_kv_cache()
    fused_cache = fused.create_paged_kv_cache()
    generator = torch.Generator().manual_seed(7100 + batch)
    if mode == "prefill":
        hidden = torch.randn(batch, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
        tt_hidden = _to_tt_prefill(hidden, mesh_device)
        functional_out = functional.prefill_forward(
            tt_hidden,
            key_cache=functional_cache[0],
            value_cache=functional_cache[1],
            page_table=page_table,
        )
        fused_out = fused.prefill_forward(
            tt_hidden,
            key_cache=fused_cache[0],
            value_cache=fused_cache[1],
            page_table=page_table,
        )
        _assert_pcc("fused-vs-functional-prefill", _to_torch_prefill(functional_out), _to_torch_prefill(fused_out))
        reference, _ = _reference_prefill(config, state, hidden)
        _assert_pcc(f"fused-prefill-reference-{seq_len}", reference, _to_torch_prefill(fused_out))
    else:
        hidden = torch.randn(batch, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
        tt_hidden = _to_tt_decode(hidden, mesh_device)
        positions = _positions(list(range(batch)), mesh_device)
        kwargs = dict(page_table=page_table, current_positions=positions, use_long_rope=False)
        functional_out = functional.decode_forward(
            tt_hidden,
            key_cache=functional_cache[0],
            value_cache=functional_cache[1],
            **kwargs,
        )
        fused_out = fused.decode_forward(
            tt_hidden,
            key_cache=fused_cache[0],
            value_cache=fused_cache[1],
            **kwargs,
        )
        _assert_pcc(
            f"fused-vs-functional-decode-b{batch}",
            _to_torch_decode(functional_out),
            _to_torch_decode(fused_out),
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_mlp_candidate_latency_probe(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    functional = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=64
    )
    fused = FusedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=64
    )
    hidden = _to_tt_prefill(
        torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(7111)).to(torch.bfloat16),
        mesh_device,
    )

    def measure(decoder):
        decoder._mlp(hidden)
        ttnn.synchronize_device(mesh_device)
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            decoder._mlp(hidden)
            ttnn.synchronize_device(mesh_device)
            samples.append(time.perf_counter() - start)
        return sum(samples) / len(samples)

    functional_seconds = measure(functional)
    fused_seconds = measure(fused)
    print(f"FUSION_AB functional_ms={functional_seconds * 1000:.6f} " f"fused_ms={fused_seconds * 1000:.6f}")
    # End-to-end traced decode is the performance selection gate.  This small
    # component probe is intentionally informational because sub-millisecond
    # host timing is noisy across device reopenings.


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_fused_trace_replay_is_deterministic(mesh_device, batch):
    config = _config()
    state = _synthetic_state(config)
    decoder = FusedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=64,
    )
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(7200 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    positions = [33] if batch == 1 else list(range(1, batch + 1))
    current_positions = _positions(positions, mesh_device)
    page_table = _page_table(batch, 64, mesh_device, permute=True)
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
    trace_output = decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    replayed = []
    try:
        for _ in range(5):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replayed.append(ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0]).clone())
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert all(torch.equal(replayed[0], item) for item in replayed[1:])
    reference = _reference_decode_zero_prefix(config, state, hidden, positions, use_long=False)
    _assert_pcc(f"fused-trace-reference-b{batch}", reference, replayed[0].squeeze(0).transpose(0, 1))
