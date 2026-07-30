# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness gates for the optimized Phi-3.5 decoder path."""

from __future__ import annotations

import inspect

import pytest
import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests import test_functional_decoder as functional_tests
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _assert_pcc,
    _config,
    _page_table,
    _positions,
    _real_state,
    _reference_decode,
    _reference_prefill,
    _to_torch_decode,
    _to_torch_prefill,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder


def test_optimized_class_owns_policy_and_mlp_path():
    assert OptimizedDecoder is not FunctionalDecoder
    assert OptimizedDecoder._mlp is not FunctionalDecoder._mlp
    assert OptimizedDecoder.prefill_forward is not FunctionalDecoder.prefill_forward
    assert OptimizedDecoder.decode_forward is not FunctionalDecoder.decode_forward
    assert OptimizedDecoder.forward is not FunctionalDecoder.forward
    runtime = (
        OptimizedDecoder._mlp,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder._decode_mlp,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder.forward,
    )
    for method in runtime:
        source = inspect.getsource(method)
        for forbidden in ("torch", "from_torch", "to_torch", ".cpu(", "FunctionalDecoder."):
            assert forbidden not in source, (method.__name__, forbidden)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kv_cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_real_weight_optimized_prefill_and_decode(mesh_device, kv_cache_dtype):
    config = _config()
    state = _real_state()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=64,
        kv_cache_dtype=kv_cache_dtype,
    )
    assert type(decoder) is OptimizedDecoder
    assert set(decoder.selected_decode_weight_dtypes.values()) == {functional_tests.ttnn.bfloat4_b}
    assert set(decoder.decode_fidelities.values()) == {functional_tests.ttnn.MathFidelity.LoFi}
    page_table = _page_table(1, 64, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    generator = torch.Generator().manual_seed(3500)
    prefill = (torch.randn(1, 33, config.hidden_size, generator=generator) * 0.2).to(torch.bfloat16)
    prefill_reference, past = _reference_prefill(config, state, prefill)
    prefill_actual = decoder.prefill_forward(
        _to_tt_prefill(prefill, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    dtype_name = str(kv_cache_dtype).split(".")[-1]
    _assert_pcc(f"optimized-real-prefill-33-cache-{dtype_name}", prefill_reference, _to_torch_prefill(prefill_actual))

    hidden = (torch.randn(1, 1, config.hidden_size, generator=generator) * 0.2).to(torch.bfloat16)
    decode_reference = _reference_decode(config, state, hidden, 33, past)
    decode_actual = decoder.decode_forward(
        _to_tt_decode(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions([33], mesh_device),
        use_long_rope=False,
    )
    _assert_pcc(f"optimized-real-decode-33-cache-{dtype_name}", decode_reference, _to_torch_decode(decode_actual))


def _run_functional_gate_with_optimized(monkeypatch, gate, *args):
    """Reuse the mature semantic gate while proving its constructor is optimized."""
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", OptimizedDecoder)
    return gate(*args)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [31, 33, 65])
def test_optimized_prefill_non_aligned_semantics(monkeypatch, mesh_device, seq_len):
    _run_functional_gate_with_optimized(
        monkeypatch, functional_tests.test_paged_prefill_synthetic_matches_reference, mesh_device, seq_len
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_prefill_batch2_cache_routing(monkeypatch, mesh_device):
    _run_functional_gate_with_optimized(
        monkeypatch, functional_tests.test_paged_prefill_batch2_cache_routing, mesh_device
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_decode_trace_replay_is_deterministic(monkeypatch, mesh_device, batch):
    _run_functional_gate_with_optimized(
        monkeypatch, functional_tests.test_decode_trace_replay_is_deterministic, mesh_device, batch
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_real_weight_decode_at_advertised_context(monkeypatch, mesh_device):
    _run_functional_gate_with_optimized(
        monkeypatch, functional_tests.test_real_weight_decode_at_advertised_context, mesh_device
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_long_rope_trace_matches_reference(monkeypatch, mesh_device):
    _run_functional_gate_with_optimized(
        monkeypatch, functional_tests.test_long_rope_decode_trace_matches_reference, mesh_device
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_trace_replay_stress(mesh_device, batch):
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        functional_tests._synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
    )
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(8000 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    positions = _positions([33] * batch, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()

    def decode():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=positions,
            use_long_rope=False,
        )

    decode()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        samples = []
        for index in range(50):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            if index in (0, 24, 49):
                samples.append(ttnn.to_torch(ttnn.get_device_tensors(output)[0]).clone())
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert all(torch.isfinite(sample).all() for sample in samples)
    assert torch.equal(samples[0], samples[1])
    assert torch.equal(samples[1], samples[2])
    print(f"STRESS_RESULT batch={batch} trace_replays=50 finite=true bitwise_stable=true")
