# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import hashlib
import json
import math
import os
from pathlib import Path
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
    _paged_state as _functional_paged_state,
    _rope_device,
    _rope_host,
    _to_host,
    _tt_input,
    hf_config,
    mesh_device,
)
from models.autoports.google_gemma_4_31b.tt.optimized_decoder import (
    DEFAULT_OPTIMIZATION_POLICY,
    DecoderOptimizationPolicy,
    OptimizedDecoder,
    _OptimizedSharedMLP,
)
from models.autoports.google_gemma_4_31b.tt.fused_decoder import FusedDecoder


def _dtype(name: str):
    return {
        "bfp4": ttnn.bfloat4_b,
        "bfp8": ttnn.bfloat8_b,
        "bf16": ttnn.bfloat16,
    }[name.lower()]


def _fidelity(name: str):
    return {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }[name.lower()]


def _policy_from_env() -> DecoderOptimizationPolicy:
    base = DEFAULT_OPTIMIZATION_POLICY
    return DecoderOptimizationPolicy(
        name=os.getenv("GEMMA4_OPT_NAME", base.name),
        attention_weight_dtype=_dtype(os.getenv("GEMMA4_OPT_ATTN_DTYPE", "bfp8")),
        mlp_gate_up_weight_dtype=_dtype(os.getenv("GEMMA4_OPT_GATE_UP_DTYPE", "bfp4")),
        mlp_down_weight_dtype=_dtype(os.getenv("GEMMA4_OPT_DOWN_DTYPE", "bfp4")),
        attention_math_fidelity=_fidelity(os.getenv("GEMMA4_OPT_ATTN_FIDELITY", "lofi")),
        mlp_gate_up_math_fidelity=_fidelity(os.getenv("GEMMA4_OPT_GATE_UP_FIDELITY", "lofi")),
        mlp_down_math_fidelity=_fidelity(os.getenv("GEMMA4_OPT_DOWN_FIDELITY", "lofi")),
        decode_num_cores=int(os.getenv("GEMMA4_OPT_CORES", str(base.decode_num_cores))),
        qkv_in0_block_w=int(os.getenv("GEMMA4_OPT_QKV_BLOCK_W", str(base.qkv_in0_block_w))),
        o_proj_in0_block_w=int(os.getenv("GEMMA4_OPT_O_BLOCK_W", str(base.o_proj_in0_block_w))),
        gate_up_in0_block_w=int(os.getenv("GEMMA4_OPT_GATE_UP_BLOCK_W", str(base.gate_up_in0_block_w))),
        down_in0_block_w=int(os.getenv("GEMMA4_OPT_DOWN_BLOCK_W", str(base.down_in0_block_w))),
        kv_cache_dtype=_dtype(os.getenv("GEMMA4_OPT_KV_DTYPE", "bfp8")),
        attention_projection_topology=os.getenv(
            "GEMMA4_OPT_ATTN_TOPOLOGY", base.attention_projection_topology
        ),
        mlp_gate_up_topology=os.getenv("GEMMA4_OPT_MLP_TOPOLOGY", base.mlp_gate_up_topology),
        mlp_packed_output_dtype=_dtype(os.getenv("GEMMA4_OPT_MLP_OUTPUT_DTYPE", "bf16")),
        prefill_qkv_topology=os.getenv("GEMMA4_OPT_PREFILL_QKV", base.prefill_qkv_topology),
    )


def _source_hashes():
    paths = {
        "optimized_decoder": Path(inspect.getfile(OptimizedDecoder)),
        "optimized_tests": Path(__file__),
        "fused_decoder": Path(inspect.getfile(FusedDecoder)),
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _policy_record(policy):
    if policy is None:
        return None
    return {
        field: str(getattr(policy, field))
        for field in policy.__dataclass_fields__
    }


def _print_run_binding(*, implementation, policy):
    env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("GEMMA4_OPT_") or key in {"GEMMA4_LONG_PREFILL", "GEMMA4_LONG_DECODE"}
    }
    print(
        "RUN_BINDING "
        + json.dumps(
            {
                "implementation": implementation,
                "policy": _policy_record(policy),
                "source_hashes": _source_hashes(),
                "environment": env,
            },
            sort_keys=True,
        )
    )


def _paged_state(hf_config, layer_idx, max_context, mesh_device, *, permutation=True, batch_size=1):
    cache_dtype = _policy_from_env().kv_cache_dtype
    if cache_dtype == ttnn.bfloat16:
        return _functional_paged_state(
            hf_config,
            layer_idx,
            max_context,
            mesh_device,
            permutation=permutation,
            batch_size=batch_size,
        )
    args = functional_tests.Gemma4ModelArgs.from_hf_config(hf_config)
    cfg = functional_tests.Gemma4AttentionConfig(args, layer_idx)
    physical_context = cfg.sliding_window if cfg.is_sliding else max_context
    num_blocks = math.ceil(physical_context / functional_tests.BLOCK_SIZE)
    paged = functional_tests.PagedAttentionConfig(
        block_size=functional_tests.BLOCK_SIZE,
        max_num_blocks=num_blocks * batch_size,
    )
    cache = functional_tests.init_kv_cache(
        mesh_device,
        cfg,
        paged_attention_config=paged,
        cache_dtype=cache_dtype,
        max_num_blocks_override=num_blocks * batch_size,
    )
    rows = []
    for batch_idx in range(batch_size):
        block_ids = torch.arange(
            batch_idx * num_blocks,
            (batch_idx + 1) * num_blocks,
            dtype=torch.int32,
        )
        if permutation:
            block_ids = torch.roll(block_ids, shifts=1)
        rows.append(block_ids)
    page_table = ttnn.from_torch(
        torch.stack(rows),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    return cache, page_table


def _decoder(state, hf_config, layer_idx, mesh_device):
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        optimization_policy=_policy_from_env(),
    )
    assert type(decoder) is OptimizedDecoder
    assert type(decoder.layer.shared_mlp) is _OptimizedSharedMLP
    assert decoder.policy == _policy_from_env()
    _print_run_binding(implementation="optimized", policy=decoder.policy)
    return decoder


def _benchmark_decoder(state, hf_config, layer_idx, mesh_device):
    """Select only the benchmark baseline; correctness always uses OptimizedDecoder."""
    implementation = os.getenv("GEMMA4_OPT_BENCH_IMPLEMENTATION", "optimized")
    if implementation == "optimized":
        return implementation, _decoder(state, hf_config, layer_idx, mesh_device)
    if implementation == "fused":
        decoder = FusedDecoder.from_state_dict(
            state,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
        )
        assert type(decoder) is FusedDecoder
        _print_run_binding(implementation="fused", policy=None)
        return implementation, decoder
    raise ValueError(f"unknown GEMMA4_OPT_BENCH_IMPLEMENTATION={implementation!r}")


def _bind_optimized_inherited_helpers(monkeypatch):
    """Run inherited contract tests with both the optimized graph and BFP8 cache."""
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", OptimizedDecoder)
    monkeypatch.setattr(functional_tests, "_paged_state", _paged_state)
    _print_run_binding(implementation="optimized", policy=_policy_from_env())


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", [32, 33, 128])
def test_optimized_real_weight_paged_prefill_pcc(hf_config, mesh_device, layer_idx, layer_kind, seq_len):
    state = _layer_state(layer_idx)
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    decoder = _decoder(state, hf_config, layer_idx, mesh_device)
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
def test_optimized_real_weight_paged_decode_trace_pcc(hf_config, mesh_device, layer_idx, layer_kind):
    prompt_len = 32
    state = _layer_state(layer_idx)
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    decoder = _decoder(state, hf_config, layer_idx, mesh_device)
    cache, page_table = _paged_state(hf_config, layer_idx, 128, mesh_device)
    prompt_rope = _rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False)
    decode_rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=True)
    torch.manual_seed(20260721 + layer_idx)
    prompt = torch.randn(1, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    token = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    decoder.prefill_forward(
        _tt_input(prompt, mesh_device),
        rope_mats=prompt_rope,
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
            assert torch.equal(replay_one, _to_host(output))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif("GEMMA4_OPT_BENCH" not in os.environ, reason="set GEMMA4_OPT_BENCH for warmed timing")
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("perf_mode", ["prefill", "decode"])
def test_optimized_warmed_latency(hf_config, mesh_device, layer_idx, layer_kind, perf_mode):
    state = _layer_state(layer_idx)
    implementation, decoder = _benchmark_decoder(state, hf_config, layer_idx, mesh_device)
    cache_factory = _functional_paged_state if implementation == "fused" else _paged_state
    cache, page_table = cache_factory(hf_config, layer_idx, 128, mesh_device)
    torch.manual_seed(20261300 + layer_idx)
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
        samples = []
        for _ in range(8):
            start = time.perf_counter_ns()
            output = call()
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            output.deallocate(True)
        policy = decoder.policy.name if implementation == "optimized" else "fused_bf16"
        print(f"OPT_WARMED implementation={implementation} policy={policy} kind={layer_kind} mode=prefill median_ms={statistics.median(samples)} samples_ms={samples}")
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
    pos_i = ttnn.from_torch(torch.tensor([prompt_len], dtype=torch.int32), device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def call():
        return decoder.decode_forward(token, rope_mats=rope, page_table=page_table, kv_cache=cache, current_position=pos_u, current_position_cache=pos_i)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        policy = decoder.policy.name if implementation == "optimized" else "fused_bf16"
        print(f"OPT_WARMED implementation={implementation} policy={policy} kind={layer_kind} mode=decode median_ms={statistics.median(samples)} samples_ms={samples}")
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif("GEMMA4_OPT_PROFILE" not in os.environ, reason="set GEMMA4_OPT_PROFILE for Tracy evidence")
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("perf_mode", ["prefill", "decode"])
def test_optimized_profile(hf_config, mesh_device, layer_idx, layer_kind, perf_mode):
    from tracy import signpost

    state = _layer_state(layer_idx)
    decoder = _decoder(state, hf_config, layer_idx, mesh_device)
    cache, page_table = _paged_state(hf_config, layer_idx, 128, mesh_device)
    torch.manual_seed(20261400 + layer_idx)
    if perf_mode == "prefill":
        seq_len = 128
        hidden = _tt_input(torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
        rope = _rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False)

        def call():
            return decoder.prefill_forward(hidden, rope_mats=rope, page_table=page_table, kv_cache=cache, valid_seq_len=seq_len)

        warm = call()
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)
        signpost("OPT_PERF_PREFILL")
        output = call()
        ttnn.synchronize_device(mesh_device)
        signpost("OPT_PERF_PREFILL_END")
        output.deallocate(True)
        return

    prompt_len = 32
    prompt = _tt_input(torch.randn(1, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
    decoder.prefill_forward(prompt, rope_mats=_rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False), page_table=page_table, kv_cache=cache, valid_seq_len=prompt_len)
    token = _tt_input(torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
    rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=True)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = prompt_len
    pos_u = ttnn.from_torch(pos_u_host, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_i = ttnn.from_torch(torch.tensor([prompt_len], dtype=torch.int32), device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def call():
        return decoder.decode_forward(token, rope_mats=rope, page_table=page_table, kv_cache=cache, current_position=pos_u, current_position_cache=pos_i)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        signpost("OPT_PERF_DECODE")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost("OPT_PERF_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)


def test_optimized_runtime_path_source_audit():
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            OptimizedDecoder._decode_attention,
            OptimizedDecoder._decode_matmul_program_config,
            OptimizedDecoder._forward_device,
            _OptimizedSharedMLP.__call__,
        )
    )
    for required in ("MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig", "paged_scaled_dot_product_attention_decode"):
        assert required in source
    for forbidden in ("torch", "ttnn.from_torch", "ttnn.to_torch", "super()._forward_device"):
        assert forbidden not in source


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("batch_size", [2, 32])
def test_optimized_batched_nonaligned_paged_prefill(monkeypatch, hf_config, mesh_device, layer_idx, layer_kind, batch_size):
    _bind_optimized_inherited_helpers(monkeypatch)
    functional_tests.test_batched_nonaligned_paged_prefill(hf_config, mesh_device, layer_idx, layer_kind, batch_size)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_optimized_batch_32_paged_decode_trace_pcc(monkeypatch, hf_config, mesh_device, layer_idx, layer_kind):
    _bind_optimized_inherited_helpers(monkeypatch)
    functional_tests.test_batch_32_paged_decode_pcc(hf_config, mesh_device, layer_idx, layer_kind)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("position_case", ["random", "window_wrap"])
def test_optimized_changed_trace_buffers_random_and_boundaries(monkeypatch, hf_config, mesh_device, layer_idx, layer_kind, position_case):
    _bind_optimized_inherited_helpers(monkeypatch)
    functional_tests.test_changed_trace_buffers_random_and_boundaries(hf_config, mesh_device, layer_idx, layer_kind, position_case)


@pytest.mark.parametrize("seq_len", [1025, 1057])
def test_optimized_sliding_padding_cache_ownership_and_decode(monkeypatch, hf_config, mesh_device, seq_len):
    _bind_optimized_inherited_helpers(monkeypatch)
    functional_tests.test_sliding_padding_cache_ownership_and_decode(hf_config, mesh_device, seq_len)


@pytest.mark.skipif("GEMMA4_LONG_PREFILL" not in os.environ, reason="set GEMMA4_LONG_PREFILL for capacity probe")
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_optimized_long_nonaligned_prefill_capacity(monkeypatch, hf_config, mesh_device, layer_idx, layer_kind):
    _bind_optimized_inherited_helpers(monkeypatch)
    functional_tests.test_long_nonaligned_prefill_capacity(hf_config, mesh_device, layer_idx, layer_kind)


@pytest.mark.skipif(
    "GEMMA4_LONG_DECODE" not in os.environ,
    reason="set GEMMA4_LONG_DECODE for exact-context distinct-token oracle",
)
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_optimized_exact_context_distinct_traced_decode(
    monkeypatch, hf_config, mesh_device, layer_idx, layer_kind
):
    _bind_optimized_inherited_helpers(monkeypatch)
    functional_tests.test_exact_context_distinct_traced_decode(
        hf_config, mesh_device, layer_idx, layer_kind
    )
