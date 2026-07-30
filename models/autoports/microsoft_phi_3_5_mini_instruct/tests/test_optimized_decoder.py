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
    _real_state,
    _reference_decode_zero_prefix,
    _reference_prefill,
    _synthetic_state,
    _to_torch_decode,
    _to_torch_prefill,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder


def test_optimized_runtime_dispatch_contract():
    source = inspect.getsource(OptimizedDecoder._mlp)
    assert "input_tensor_a_activations=[ttnn.UnaryOpType.SILU]" in source
    assert "ttnn.silu" not in source
    assert OptimizedDecoder._mlp is not FunctionalDecoder._mlp
    assert OptimizedDecoder.decode_forward is not FunctionalDecoder.decode_forward
    constructor = inspect.getsource(OptimizedDecoder.from_state_dict)
    assert '"dram_sharded"' in constructor
    assert '"bfp4"' in constructor


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
def test_optimized_matches_functional_on_device(mesh_device, mode, batch, seq_len):
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
    optimized = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=max_context,
    )
    page_table = _page_table(batch, max_context, mesh_device, permute=True)
    functional_cache = functional.create_paged_kv_cache()
    optimized_cache = optimized.create_paged_kv_cache()
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
        optimized_out = optimized.prefill_forward(
            tt_hidden,
            key_cache=optimized_cache[0],
            value_cache=optimized_cache[1],
            page_table=page_table,
        )
        _assert_pcc(
            "optimized-vs-functional-prefill", _to_torch_prefill(functional_out), _to_torch_prefill(optimized_out)
        )
        reference, _ = _reference_prefill(config, state, hidden)
        _assert_pcc(f"optimized-prefill-reference-{seq_len}", reference, _to_torch_prefill(optimized_out))
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
        optimized_out = optimized.decode_forward(
            tt_hidden,
            key_cache=optimized_cache[0],
            value_cache=optimized_cache[1],
            **kwargs,
        )
        _assert_pcc(
            f"optimized-vs-functional-decode-b{batch}",
            _to_torch_decode(functional_out),
            _to_torch_decode(optimized_out),
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_mlp_candidate_latency_probe(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    functional = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=64
    )
    optimized = OptimizedDecoder.from_state_dict(
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
    optimized_seconds = measure(optimized)
    print(f"FUSION_AB functional_ms={functional_seconds * 1000:.6f} " f"optimized_ms={optimized_seconds * 1000:.6f}")
    # End-to-end traced decode is the performance selection gate.  This small
    # component probe is intentionally informational because sub-millisecond
    # host timing is noisy across device reopenings.


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_trace_replay_is_deterministic(mesh_device, batch):
    config = _config()
    state = _synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
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
    _assert_pcc(f"optimized-trace-reference-b{batch}", reference, replayed[0].squeeze(0).transpose(0, 1))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_real_weight_decode_matches_functional(mesh_device):
    config = _config()
    state = _real_state()
    functional = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, batch=1, max_context=64
    )
    optimized = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, batch=1, max_context=64
    )
    hidden = torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(7301)).to(torch.bfloat16)
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    kwargs = {
        "page_table": _page_table(1, 64, mesh_device, permute=True),
        "current_positions": _positions([0], mesh_device),
        "use_long_rope": False,
    }
    functional_cache = functional.create_paged_kv_cache()
    optimized_cache = optimized.create_paged_kv_cache()
    functional_out = functional.decode_forward(
        tt_hidden,
        key_cache=functional_cache[0],
        value_cache=functional_cache[1],
        **kwargs,
    )
    optimized_out = optimized.decode_forward(
        tt_hidden,
        key_cache=optimized_cache[0],
        value_cache=optimized_cache[1],
        **kwargs,
    )
    _assert_pcc(
        "optimized-real-weight-decode",
        _to_torch_decode(functional_out),
        _to_torch_decode(optimized_out),
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_decode_consumes_non_aligned_paged_prefill(mesh_device, batch):
    config = _config()
    state = _synthetic_state(config)
    functional = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, batch=batch, max_context=64
    )
    optimized = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, batch=batch, max_context=64
    )
    page_table = _page_table(batch, 64, mesh_device, permute=True)
    functional_cache = functional.create_paged_kv_cache()
    optimized_cache = optimized.create_paged_kv_cache()
    generator = torch.Generator().manual_seed(7400 + batch)
    prompt = torch.randn(batch, 33, config.hidden_size, generator=generator).to(torch.bfloat16)
    functional.prefill_forward(
        _to_tt_prefill(prompt, mesh_device),
        key_cache=functional_cache[0],
        value_cache=functional_cache[1],
        page_table=page_table,
    )
    optimized.prefill_forward(
        _to_tt_prefill(prompt, mesh_device),
        key_cache=optimized_cache[0],
        value_cache=optimized_cache[1],
        page_table=page_table,
    )
    token = torch.randn(batch, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    kwargs = {
        "page_table": page_table,
        "current_positions": _positions([33] * batch, mesh_device),
        "use_long_rope": False,
    }
    functional_out = functional.decode_forward(
        _to_tt_decode(token, mesh_device),
        key_cache=functional_cache[0],
        value_cache=functional_cache[1],
        **kwargs,
    )
    optimized_out = optimized.decode_forward(
        _to_tt_decode(token, mesh_device),
        key_cache=optimized_cache[0],
        value_cache=optimized_cache[1],
        **kwargs,
    )
    _assert_pcc(
        f"optimized-prefill-decode-cache-b{batch}",
        _to_torch_decode(functional_out),
        _to_torch_decode(optimized_out),
    )
