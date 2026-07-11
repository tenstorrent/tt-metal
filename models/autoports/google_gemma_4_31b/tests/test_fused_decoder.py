# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import statistics
import time

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

import ttnn
from models.autoports.google_gemma_4_31b.tests import test_functional_decoder as functional_tests
from models.autoports.google_gemma_4_31b.tests.test_functional_decoder import (
    LAYER_KINDS,
    _assert_pcc,
    _hf_prefill,
    _layer_state,
    _local_state,
    _paged_state,
    _rope_host,
    _rope_device,
    _to_host,
    _tt_input,
    hf_config,
    mesh_device,
)
from models.autoports.google_gemma_4_31b.tt.fused_decoder import FusedDecoder, _FusedSharedMLP
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.shared_mlp import SharedMLP


def _long_gelu_program_config(*, per_core_m: int, out_block_h: int):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=7,
        out_block_h=out_block_h,
        out_block_w=7,
        per_core_M=per_core_m,
        per_core_N=7,
        fuse_batch=False,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
        mcast_in0=True,
        num_global_cb_receivers=0,
    )


@pytest.mark.skipif("GEMMA4_LONG_GELU" not in os.environ, reason="select one long GELU candidate")
def test_long_gelu_real_weight_gate_candidate(hf_config, mesh_device):
    """Focused B0/F4 admission, equivalence, topology, and warmed-latency gate."""
    candidate = os.environ["GEMMA4_LONG_GELU"]
    candidates = {
        "F4": (4096, 128, 4),
        "F2": (4096, 128, 2),
        "F1": (4096, 128, 1),
        "C2048": (2048, 64, 4),
        "C1024": (1024, 32, 4),
        "C128": (128, 4, 4),
    }
    if candidate not in candidates:
        pytest.fail(f"unsupported one-at-a-time long GELU candidate: {candidate}")
    m, per_core_m, out_block_h = candidates[candidate]

    layer_state = _local_state(_layer_state(0), 0)
    mlp_state = {key.removeprefix("mlp."): value for key, value in layer_state.items() if key.startswith("mlp.")}
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
    mlp = SharedMLP(
        mesh_device,
        Gemma4ModelArgs.from_hf_config(hf_config),
        mlp_state,
        mesh_config,
        dtype=ttnn.bfloat16,
    )
    torch.manual_seed(2026071102)
    hidden = _tt_input(torch.randn(1, m, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
    config = _long_gelu_program_config(per_core_m=per_core_m, out_block_h=out_block_h)

    def baseline():
        gate = ttnn.linear(hidden, mlp.gate_proj)
        return ttnn.gelu(gate, fast_and_approximate_mode=True)

    def fused():
        return ttnn.linear(hidden, mlp.gate_proj, program_config=config)

    baseline_output = baseline()
    ttnn.synchronize_device(mesh_device)
    fused_output = fused()
    ttnn.synchronize_device(mesh_device)
    pcc = _assert_pcc(_to_host(baseline_output), _to_host(fused_output), threshold=0.99)
    baseline_output.deallocate(True)
    fused_output.deallocate(True)

    timings = {"B0": [], candidate: []}
    for label, call in (("B0", baseline), (candidate, fused), (candidate, fused), ("B0", baseline)):
        for _ in range(3):
            start = time.perf_counter_ns()
            output = call()
            ttnn.synchronize_device(mesh_device)
            timings[label].append((time.perf_counter_ns() - start) / 1_000)
            output.deallocate(True)

    if "GEMMA4_LONG_GELU_PROFILE" in os.environ:
        from tracy import signpost

        signpost("LONG_GELU_B0")
        output = baseline()
        ttnn.synchronize_device(mesh_device)
        signpost("LONG_GELU_B0_END")
        output.deallocate(True)
        signpost(f"LONG_GELU_{candidate}")
        output = fused()
        ttnn.synchronize_device(mesh_device)
        signpost(f"LONG_GELU_{candidate}_END")
        output.deallocate(True)

    print(
        "LONG_GELU_GATE "
        f"candidate={candidate} shape=M{m}_K{hf_config.hidden_size}_N{hf_config.intermediate_size} "
        f"chunks_per_4096={4096 // m} "
        f"pcc={pcc} B0_median_us={statistics.median(timings['B0'])} "
        f"candidate_median_us={statistics.median(timings[candidate])} "
        f"B0_normalized_4096_us={statistics.median(timings['B0']) * (4096 // m)} "
        f"candidate_normalized_4096_us={statistics.median(timings[candidate]) * (4096 // m)} "
        f"B0_samples_us={timings['B0']} candidate_samples_us={timings[candidate]}"
    )


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", [32, 33, 128])
def test_fused_real_weight_paged_prefill_pcc(hf_config, mesh_device, layer_idx, layer_kind, seq_len):
    state = _layer_state(layer_idx)
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    decoder = FusedDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    assert type(decoder) is FusedDecoder
    assert type(decoder.layer.shared_mlp) is _FusedSharedMLP
    cache, page_table = _paged_state(hf_config, layer_idx, max(seq_len, 128), mesh_device)
    rope = _rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False)
    torch.manual_seed(20260711 + layer_idx + seq_len)
    hidden = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    reference = _hf_prefill(hf_layer, hf_config, layer_idx, hidden)
    actual = decoder.prefill_forward(
        _tt_input(hidden, mesh_device),
        rope_mats=rope,
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=seq_len,
    )
    _assert_pcc(reference, _to_host(actual))


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_fused_real_weight_paged_decode_trace_pcc(hf_config, mesh_device, layer_idx, layer_kind):
    prompt_len = 32
    state = _layer_state(layer_idx)
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    decoder = FusedDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    assert type(decoder) is FusedDecoder
    cache, page_table = _paged_state(hf_config, layer_idx, 128, mesh_device)
    prefill_rope = _rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False)
    decode_rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=True)
    torch.manual_seed(20260721 + layer_idx)
    prompt = torch.randn(1, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    token = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    decoder.prefill_forward(
        _tt_input(prompt, mesh_device),
        rope_mats=prefill_rope,
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=prompt_len,
    )
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = prompt_len
    pos_u = ttnn.from_torch(pos_u_host, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_i = ttnn.from_torch(
        torch.tensor([prompt_len], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_token = _tt_input(token, mesh_device)

    dynamic_cache = DynamicCache()
    positions = torch.arange(prompt_len).reshape(1, -1)
    causal = torch.triu(torch.full((1, 1, prompt_len, prompt_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        hf_layer(
            prompt,
            position_embeddings=_rope_host(hf_config, layer_idx, positions),
            attention_mask=causal,
            past_key_values=dynamic_cache,
        )
        reference = hf_layer(
            token,
            position_embeddings=_rope_host(hf_config, layer_idx, [prompt_len]),
            attention_mask=torch.zeros(1, 1, 1, prompt_len + 1),
            past_key_values=dynamic_cache,
        )

    def call():
        return decoder.decode_forward(
            tt_token,
            rope_mats=decode_rope,
            page_table=page_table,
            kv_cache=cache,
            current_position=pos_u,
            current_position_cache=pos_i,
        )

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay_one = _to_host(output).clone()
        _assert_pcc(reference, replay_one)
        for _ in range(7):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            replay = _to_host(output).clone()
            assert torch.equal(replay_one, replay)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif("GEMMA4_FUSED_PERF" not in os.environ, reason="set GEMMA4_FUSED_PERF to collect Tracy evidence")
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("perf_mode", ["prefill", "decode"])
def test_fused_warmed_performance(hf_config, mesh_device, layer_idx, layer_kind, perf_mode):
    from tracy import signpost

    state = _layer_state(layer_idx)
    decoder = FusedDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    cache, page_table = _paged_state(hf_config, layer_idx, 128, mesh_device)
    torch.manual_seed(20261200 + layer_idx)
    if perf_mode == "prefill":
        seq_len = 128
        hidden = _tt_input(torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
        rope = _rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False)

        def call():
            return decoder.prefill_forward(
                hidden, rope_mats=rope, page_table=page_table, kv_cache=cache, valid_seq_len=seq_len
            )

        warm = call()
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)
        signpost("FUSED_PERF_PREFILL")
        output = call()
        ttnn.synchronize_device(mesh_device)
        signpost("FUSED_PERF_PREFILL_END")
        output.deallocate(True)
        return

    prompt_len = 32
    prompt = _tt_input(torch.randn(1, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
    decoder.prefill_forward(
        prompt,
        rope_mats=_rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=prompt_len,
    )
    token = _tt_input(torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
    rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=True)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = prompt_len
    pos_u = ttnn.from_torch(pos_u_host, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_i = ttnn.from_torch(
        torch.tensor([prompt_len], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def call():
        return decoder.decode_forward(
            token,
            rope_mats=rope,
            page_table=page_table,
            kv_cache=cache,
            current_position=pos_u,
            current_position_cache=pos_i,
        )

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        signpost("FUSED_PERF_DECODE")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost("FUSED_PERF_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)


def test_fused_runtime_path_source_audit():
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            FusedDecoder._concatenate_heads,
            FusedDecoder._prefill_attention,
            FusedDecoder._fill_bounded_sliding_cache_exact,
            FusedDecoder._streaming_full_prefill_attention,
            FusedDecoder._decode_attention,
            FusedDecoder._forward_device,
            _FusedSharedMLP.__call__,
        )
    )
    for required in (
        "nlp_concat_heads",
        "UnaryOpType.GELU",
        "UnaryOpType.MUL_UNARY_SFPU",
    ):
        assert required in source
    for forbidden in ("torch", "ttnn.from_torch", "ttnn.to_torch", "super()._forward_device"):
        assert forbidden not in source


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("batch_size", [2, 32])
def test_fused_batched_nonaligned_paged_prefill(
    monkeypatch, hf_config, mesh_device, layer_idx, layer_kind, batch_size
):
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)
    functional_tests.test_batched_nonaligned_paged_prefill(
        hf_config, mesh_device, layer_idx, layer_kind, batch_size
    )


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_fused_batch_32_paged_decode_trace_pcc(monkeypatch, hf_config, mesh_device, layer_idx, layer_kind):
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)
    functional_tests.test_batch_32_paged_decode_pcc(hf_config, mesh_device, layer_idx, layer_kind)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("position_case", ["random", "window_wrap"])
def test_fused_changed_trace_buffers_random_and_boundaries(
    monkeypatch, hf_config, mesh_device, layer_idx, layer_kind, position_case
):
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)
    functional_tests.test_changed_trace_buffers_random_and_boundaries(
        hf_config, mesh_device, layer_idx, layer_kind, position_case
    )


@pytest.mark.parametrize("seq_len", [1025, 1057])
def test_fused_sliding_padding_cache_ownership_and_decode(monkeypatch, hf_config, mesh_device, seq_len):
    """Padded prefill rows must not overwrite live slots after window wrap."""
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)
    functional_tests.test_sliding_padding_cache_ownership_and_decode(hf_config, mesh_device, seq_len)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_fused_advertised_context_paged_decode(monkeypatch, hf_config, mesh_device, layer_idx, layer_kind):
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)
    functional_tests.test_advertised_context_paged_decode(hf_config, mesh_device, layer_idx, layer_kind)


@pytest.mark.skipif("GEMMA4_LONG_PREFILL" not in os.environ, reason="set GEMMA4_LONG_PREFILL for capacity probe")
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_fused_long_nonaligned_prefill_capacity(monkeypatch, hf_config, mesh_device, layer_idx, layer_kind):
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)
    functional_tests.test_long_nonaligned_prefill_capacity(hf_config, mesh_device, layer_idx, layer_kind)


@pytest.mark.skipif(
    "GEMMA4_LONG_DECODE" not in os.environ,
    reason="set GEMMA4_LONG_DECODE for genuine advertised-context decode",
)
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_fused_exact_context_distinct_traced_decode(
    monkeypatch, hf_config, mesh_device, layer_idx, layer_kind
):
    """Run the distinct-token advertised-context HF gate through the fused graph."""
    constructed = []
    original_from_state_dict = FusedDecoder.from_state_dict.__func__

    def checked_from_state_dict(cls, *args, **kwargs):
        decoder = original_from_state_dict(cls, *args, **kwargs)
        assert type(decoder) is FusedDecoder
        assert type(decoder.layer.shared_mlp) is _FusedSharedMLP
        constructed.append(decoder)
        return decoder

    monkeypatch.setattr(FusedDecoder, "from_state_dict", classmethod(checked_from_state_dict))
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)
    functional_tests.test_exact_context_distinct_traced_decode(
        hf_config, mesh_device, layer_idx, layer_kind
    )
    assert len(constructed) == 1
