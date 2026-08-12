# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import time

import pytest
import torch
import tracy

import ttnn
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tests.test_functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    LAYER_IDX,
    _assert_pcc,
    _config,
    _empty_caches,
    _hf_layer,
    _real_state,
    _reference_layer,
    _synthetic_state,
    _to_host,
    _tt_tensor,
)
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.functional_decoder import FunctionalDecoder
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.optimized_decoder import OptimizedDecoder

PERF_IMPL_ENV = "MISTRAL_SMALL_24B_PERF_IMPL"
PERF_ITERS_ENV = "MISTRAL_SMALL_24B_PERF_ITERS"
OPT_POLICY_ENV = "MISTRAL_SMALL_24B_OPT_POLICY"
KV_DTYPE_ENV = "MISTRAL_SMALL_24B_KV_DTYPE"
CONTEXT_PROBE_LEN_ENV = "MISTRAL_SMALL_24B_OPT_CONTEXT_PROBE_LEN"
CONTEXT_EXPECT_OOM_ENV = "MISTRAL_SMALL_24B_OPT_CONTEXT_EXPECT_OOM"


def _empty_optimized_caches(
    config, mesh_device, *, batch: int = EMITTED_BATCH, max_cache_len: int = EMITTED_CACHE_LENGTH
):
    dtype_name = os.environ.get(KV_DTYPE_ENV, "bfp8")
    dtypes = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
    if dtype_name not in dtypes:
        raise ValueError(f"Unknown {KV_DTYPE_ENV}={dtype_name!r}; expected one of {sorted(dtypes)}")
    if dtype_name == "bf16":
        return _empty_caches(config, mesh_device, batch=batch, max_cache_len=max_cache_len)
    shape = (batch, config.num_key_value_heads, max_cache_len, config.head_dim)
    key = _tt_tensor(torch.zeros(shape, dtype=torch.bfloat16), mesh_device, dtype=dtypes[dtype_name])
    value = _tt_tensor(torch.zeros(shape, dtype=torch.bfloat16), mesh_device, dtype=dtypes[dtype_name])
    return key, value


def _optimized_policy_kwargs():
    policy = os.environ.get(OPT_POLICY_ENV, "selected")
    packed_bf16 = {
        "mlp_weight_dtype": ttnn.bfloat16,
        "mlp_math_fidelity": ttnn.MathFidelity.HiFi2,
        "use_dram_sharded_mlp": False,
        "mlp_geometry": "80x64",
        "attention_weight_dtype": ttnn.bfloat16,
        "attention_math_fidelity": ttnn.MathFidelity.HiFi2,
        "use_dram_sharded_attention": False,
        "use_prefill_program_configs": False,
        "use_advisor_decode_layout": False,
    }
    policies = {
        "selected": {},
        "selected_attention_40x48_32x40": {"attention_geometry": "40x48_32x40"},
        "selected_attention_20x24_16x20": {"attention_geometry": "20x24_16x20"},
        "selected_attention_10x12_8x10": {"attention_geometry": "10x12_8x10"},
        "selected_mlp_10x32": {"mlp_geometry": "10x32"},
        "selected_mlp_10x64": {"mlp_geometry": "10x64"},
        "advisor_1d": {"use_advisor_1d_matmuls": True},
        "bf16_hifi2": packed_bf16,
        "bfp8_hifi2": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat8_b,
            "mlp_math_fidelity": ttnn.MathFidelity.HiFi2,
        },
        "bfp8_lofi": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat8_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
        },
        "bfp4_lofi": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
        },
        "bfp4_gate_up_bfp8_down": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "down_weight_dtype": ttnn.bfloat8_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
        },
        "dram_bfp4_lofi": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_mlp": True,
        },
        "dram_bfp4_lofi_40x32": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_mlp": True,
            "mlp_geometry": "40x32",
        },
        "dram_bfp4_lofi_20x16": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_mlp": True,
            "mlp_geometry": "20x16",
        },
        "dram_bfp4_hifi2": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "mlp_math_fidelity": ttnn.MathFidelity.HiFi2,
            "use_dram_sharded_mlp": True,
        },
        "dram_bfp8_lofi": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat8_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_mlp": True,
        },
        "dram_bfp8_hifi2": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat8_b,
            "mlp_math_fidelity": ttnn.MathFidelity.HiFi2,
            "use_dram_sharded_mlp": True,
        },
        "full_dram_bfp4_lofi": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_mlp": True,
            "mlp_geometry": "40x32",
            "attention_weight_dtype": ttnn.bfloat4_b,
            "attention_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_attention": True,
            "use_prefill_program_configs": True,
        },
        "full_dram_bfp4_lofi_advisor_layout": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_mlp": True,
            "mlp_geometry": "40x32",
            "attention_weight_dtype": ttnn.bfloat4_b,
            "attention_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_attention": True,
            "use_prefill_program_configs": True,
            "use_advisor_decode_layout": True,
        },
        "full_dram_bfp4_hifi2_advisor_layout": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "mlp_math_fidelity": ttnn.MathFidelity.HiFi2,
            "use_dram_sharded_mlp": True,
            "mlp_geometry": "40x32",
            "attention_weight_dtype": ttnn.bfloat4_b,
            "attention_math_fidelity": ttnn.MathFidelity.HiFi2,
            "use_dram_sharded_attention": True,
            "use_prefill_program_configs": True,
            "use_advisor_decode_layout": True,
        },
        "full_packed_bfp4_lofi": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat4_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
            "attention_weight_dtype": ttnn.bfloat4_b,
            "attention_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_attention": True,
            "use_prefill_program_configs": True,
        },
        "full_dram_bfp8_hifi2": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat8_b,
            "mlp_math_fidelity": ttnn.MathFidelity.HiFi2,
            "use_dram_sharded_mlp": True,
            "mlp_geometry": "80x64",
            "attention_weight_dtype": ttnn.bfloat8_b,
            "attention_math_fidelity": ttnn.MathFidelity.HiFi2,
            "use_dram_sharded_attention": True,
        },
        "full_dram_bfp8_lofi": {
            **packed_bf16,
            "mlp_weight_dtype": ttnn.bfloat8_b,
            "mlp_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_mlp": True,
            "mlp_geometry": "80x64",
            "attention_weight_dtype": ttnn.bfloat8_b,
            "attention_math_fidelity": ttnn.MathFidelity.LoFi,
            "use_dram_sharded_attention": True,
        },
    }
    if policy not in policies:
        raise ValueError(f"Unknown {OPT_POLICY_ENV}={policy!r}; expected one of {sorted(policies)}")
    return policy, policies[policy]


def test_optimized_runtime_is_owned_and_has_no_host_fallback():
    assert OptimizedDecoder is not FunctionalDecoder
    methods = (OptimizedDecoder._mlp_forward, OptimizedDecoder.prefill_forward, OptimizedDecoder.decode_forward)
    for method in methods:
        assert method.__module__.endswith("optimized_decoder")
        source = inspect.getsource(method)
        for token in ("torch", "from_torch", "to_torch"):
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"
    mlp_source = inspect.getsource(OptimizedDecoder._mlp_forward)
    assert "gate_up_weight" in mlp_source
    assert "input_tensor_a_activations" in mlp_source


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_graph_rewrite_prefill_decode_pcc(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
    )
    assert decoder.use_dram_sharded_attention
    assert decoder.use_dram_sharded_mlp
    assert decoder.use_advisor_decode_layout
    assert decoder.use_prefill_program_configs
    assert decoder.mlp_geometry == (40, 32, 40, 4, 16)
    assert decoder.attention_geometry == (10, 12, 8, 10, 16)
    assert decoder.use_advisor_1d_matmuls is False
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _empty_optimized_caches(config, mesh_device)

    generator = torch.Generator().manual_seed(2502)
    for seq_len in (1, 17, 33, EMITTED_PREFILL_SEQUENCE):
        prefill_hidden = torch.randn(
            (1, EMITTED_BATCH, seq_len, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        reference_prefill, reference_key, reference_value, reference_cache = _reference_layer(
            reference_layer, prefill_hidden, config
        )
        actual_prefill = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
        _assert_pcc(reference_prefill, _to_host(actual_prefill), 0.99, f"optimized prefill seq={seq_len}")
        _assert_pcc(
            reference_key,
            _to_host(key_cache)[:, :, :seq_len, :],
            0.99,
            f"optimized prefill seq={seq_len} key cache",
        )
        _assert_pcc(
            reference_value,
            _to_host(value_cache)[:, :, :seq_len, :],
            0.99,
            f"optimized prefill seq={seq_len} value cache",
        )

    decode_hidden = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference_decode, decode_key, decode_value, reference_cache = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual_decode = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device), key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE
    )
    _assert_pcc(reference_decode, _to_host(actual_decode), 0.99, "graph rewrite decode")
    _assert_pcc(
        decode_key,
        _to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        0.99,
        "graph rewrite decode key append",
    )
    _assert_pcc(
        decode_value,
        _to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        0.99,
        "graph rewrite decode value append",
    )

    for current_pos in range(EMITTED_PREFILL_SEQUENCE + 1, EMITTED_PREFILL_SEQUENCE + 5):
        decode_hidden = torch.randn(
            (1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16
        )
        reference_decode, _, _, reference_cache = _reference_layer(
            reference_layer,
            decode_hidden,
            config,
            start_pos=current_pos,
            cache=reference_cache,
        )
        actual_decode = decoder.decode_forward(
            _tt_tensor(decode_hidden, mesh_device), key_cache, value_cache, current_pos=current_pos
        )
        _assert_pcc(reference_decode, _to_host(actual_decode), 0.99, f"optimized repeated decode pos={current_pos}")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_optimized_policy_pcc(mesh_device):
    if OPT_POLICY_ENV not in os.environ:
        pytest.skip(f"Set {OPT_POLICY_ENV} to run an explicit real-weight precision policy")
    policy, policy_kwargs = _optimized_policy_kwargs()
    config = _config()
    state = _real_state()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        **policy_kwargs,
    )
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _empty_optimized_caches(config, mesh_device)
    generator = torch.Generator().manual_seed(2504)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference_prefill, _, _, reference_cache = _reference_layer(reference_layer, prefill_hidden, config)
    actual_prefill = decoder.prefill_forward(_tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache)
    _assert_pcc(reference_prefill, _to_host(actual_prefill), 0.99, f"{policy} real prefill")

    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference_decode, decode_key, decode_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual_decode = decoder.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    _assert_pcc(reference_decode, _to_host(actual_decode), 0.99, f"{policy} real decode")
    _assert_pcc(
        decode_key,
        _to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        0.99,
        f"{policy} real decode key append",
    )
    _assert_pcc(
        decode_value,
        _to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
        0.99,
        f"{policy} real decode value append",
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_optimized_batch32_context_capacity_probe(mesh_device):
    """Opt-in, one-length-per-process capacity probe for the selected path."""

    length_text = os.environ.get(CONTEXT_PROBE_LEN_ENV)
    if not length_text:
        pytest.skip(f"Set {CONTEXT_PROBE_LEN_ENV} to run an isolated optimized capacity measurement")
    seq_len = int(length_text)
    config = _config()
    state = _synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=seq_len,
    )
    key_cache, value_cache = _empty_optimized_caches(config, mesh_device, max_cache_len=seq_len)
    hidden = torch.zeros((1, EMITTED_BATCH, seq_len, config.hidden_size), dtype=torch.bfloat16)

    expect_oom = os.environ.get(CONTEXT_EXPECT_OOM_ENV) == "1"
    try:
        actual = decoder.prefill_forward(_tt_tensor(hidden, mesh_device), key_cache, value_cache)
    except RuntimeError as error:
        if not expect_oom:
            raise
        message = str(error)
        assert "Out of Memory" in message, message
        detail = next(line.strip() for line in message.splitlines() if "Out of Memory" in line)
        print(f"optimized capacity probe EXPECTED OOM: batch={EMITTED_BATCH}, seq_len={seq_len}: {detail}")
        return

    if expect_oom:
        pytest.fail(f"Expected a DRAM allocation failure at batch={EMITTED_BATCH}, seq_len={seq_len}")
    host = _to_host(actual)
    assert tuple(host.shape) == tuple(hidden.shape)
    print(f"optimized capacity probe PASS: batch={EMITTED_BATCH}, seq_len={seq_len}, output_shape={tuple(host.shape)}")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_warmed_prefill_and_traced_decode_perf(mesh_device):
    """Opt-in same-harness functional/optimized latency and trace probe."""

    implementation = os.environ.get(PERF_IMPL_ENV)
    if implementation not in {"functional", "optimized"}:
        pytest.skip(f"Set {PERF_IMPL_ENV}=functional|optimized to run the performance probe")
    decoder_cls = FunctionalDecoder if implementation == "functional" else OptimizedDecoder
    iterations = int(os.environ.get(PERF_ITERS_ENV, "20"))

    config = _config()
    state = _real_state()
    policy, policy_kwargs = _optimized_policy_kwargs() if implementation == "optimized" else ("functional", {})
    decoder = decoder_cls.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        **policy_kwargs,
    )
    key_cache, value_cache = _empty_optimized_caches(config, mesh_device)
    generator = torch.Generator().manual_seed(2503)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    tt_prefill = _tt_tensor(prefill_hidden, mesh_device)

    decoder.prefill_forward(tt_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    tracy.signpost("PERF_PREFILL")
    prefill_start = time.perf_counter()
    decoder.prefill_forward(tt_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
    tracy.signpost("PERF_PREFILL_END")

    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    tt_decode = _tt_tensor(decode_hidden, mesh_device)
    eager_output = decoder.decode_forward(tt_decode, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
    ttnn.synchronize_device(mesh_device)
    eager_host = _to_host(eager_output)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(tt_decode, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first_replay_host = _to_host(traced_output)
    first_replay_key = _to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :]
    first_replay_value = _to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :]
    _assert_pcc(eager_host, first_replay_host, 0.9999, f"{policy} eager versus traced decode")

    for _ in range(3):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    tracy.signpost("PERF_DECODE")
    decode_start = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0 / iterations
    tracy.signpost("PERF_DECODE_END")

    replayed_host = _to_host(traced_output)
    replayed_key = _to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :]
    replayed_value = _to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :]
    assert tuple(replayed_host.shape) == (1, EMITTED_BATCH, 1, config.hidden_size)
    assert torch.equal(first_replay_host, replayed_host), "traced decode output changed across identical replays"
    assert torch.equal(first_replay_key, replayed_key), "traced decode key-cache update is nondeterministic"
    assert torch.equal(first_replay_value, replayed_value), "traced decode value-cache update is nondeterministic"
    ttnn.release_trace(mesh_device, trace_id)
    print(
        f"PERF_RESULT impl={implementation} policy={policy} prefill_seq={EMITTED_PREFILL_SEQUENCE} "
        f"prefill_ms={prefill_ms:.6f} traced_decode_ms={decode_ms:.6f} iterations={iterations}"
    )
