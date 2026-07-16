# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.autoports.qwen_qwen3_32b.tests.test_functional_decoder import (
    REAL_WEIGHT_DIR_ENV,
    _assert_pcc,
    _config,
    _hf_layer,
    _real_state,
    _reference_layer,
    _synthetic_state,
    _to_host,
    _tt_tensor,
)
from models.autoports.qwen_qwen3_32b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_PREFILL_SEQUENCE,
    REPRESENTATIVE_LAYER,
    FunctionalDecoder,
)
from models.autoports.qwen_qwen3_32b.tt.optimized_decoder import OptimizedDecoder

RESULTS_DIR_ENV = "QWEN3_32B_RESULTS_DIR"
RESULT_NAME_ENV = "QWEN3_32B_RESULT_NAME"
ACTIVATION_PATH_ENV = "QWEN3_32B_ACTIVATION_PATH"
DEFAULT_ACTIVATION_PATH = (
    Path(__file__).resolve().parents[1] / "doc/optimized_decoder/activations/layer32_prompt_inputs.pt"
)
EXTENDED_ACTIVATION_PATH = (
    Path(__file__).resolve().parents[1] / "doc/optimized_decoder/activations/layer32_prompt_repeated35_inputs.pt"
)


def _source_identity() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[4]
    relative_paths = (
        "models/autoports/qwen_qwen3_32b/tt/optimized_decoder.py",
        "models/autoports/qwen_qwen3_32b/tests/test_optimized_decoder.py",
        "models/autoports/qwen_qwen3_32b/doc/context_contract.json",
    )
    return {
        relative_path: hashlib.sha256((repo_root / relative_path).read_bytes()).hexdigest()
        for relative_path in relative_paths
    }


def _write_result(name: str, payload: dict) -> None:
    directory = os.getenv(RESULTS_DIR_ENV)
    if not directory:
        return
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "source_sha256": _source_identity()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _measure_pcc(reference, actual, threshold: float, label: str) -> tuple[bool, float]:
    passed, pcc = comp_pcc(reference.float(), actual.float(), pcc=threshold)
    print(f"{label}: {pcc}")
    return bool(passed), float(pcc)


def _recorded_hidden(
    position_start: int, length: int, *, artifact_path: Path | None = None
) -> tuple[torch.Tensor, dict]:
    path = artifact_path or Path(os.getenv(ACTIVATION_PATH_ENV, DEFAULT_ACTIVATION_PATH))
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    sequence = artifact["hidden_states"]
    if position_start < 0 or position_start + length > sequence.shape[1]:
        raise ValueError(
            f"recorded activation range [{position_start}, {position_start + length}) exceeds {sequence.shape[1]}"
        )
    hidden = sequence[:, position_start : position_start + length, :]
    hidden = hidden.unsqueeze(1).expand(1, EMITTED_BATCH, length, hidden.shape[-1]).contiguous()
    metadata = {
        **artifact["metadata"],
        "artifact_path": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "batch_policy": "repeat one prompt-derived HF boundary activation across 32 decoder slots",
    }
    return hidden, metadata


def _empty_cache(config, mesh_device, *, dtype=ttnn.bfloat16, max_cache_len=128):
    shape = (EMITTED_BATCH, config.num_key_value_heads, max_cache_len, config.head_dim)
    return (
        ttnn.zeros(
            shape,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.zeros(
            shape,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def _run_prefill(model, mesh_device, hidden):
    if isinstance(model, OptimizedDecoder):
        key_cache, value_cache = model.allocate_kv_cache()
    else:
        config = _config()
        key_cache, value_cache = _empty_cache(config, mesh_device)
    output = model.prefill_forward(_tt_tensor(hidden, mesh_device), key_cache, value_cache)
    return _to_host(output), key_cache, value_cache


def _capture_and_time_decode(model, mesh_device, hidden, key_cache, value_cache, current_pos, iterations):
    tt_hidden = _tt_tensor(hidden, mesh_device)
    warm = model.decode_forward(tt_hidden, key_cache, value_cache, current_pos=current_pos)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(tt_hidden, key_cache, value_cache, current_pos=current_pos)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
        first = _to_host(trace_output)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        second = _to_host(trace_output)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return elapsed_ms, first, second


def _time_prefill(model, mesh_device, hidden, key_cache, value_cache, iterations=5):
    tt_hidden = _tt_tensor(hidden, mesh_device)
    for _ in range(2):
        output = model.prefill_forward(tt_hidden, key_cache, value_cache)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output, True)
    measurements = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = model.prefill_forward(tt_hidden, key_cache, value_cache)
        ttnn.synchronize_device(mesh_device)
        measurements.append((time.perf_counter() - start) * 1000.0)
        ttnn.deallocate(output, True)
    return sorted(measurements)[len(measurements) // 2]


def test_optimized_runtime_is_implementation_owned_and_host_free():
    assert OptimizedDecoder.prefill_forward is not FunctionalDecoder.prefill_forward
    assert OptimizedDecoder.decode_forward is not FunctionalDecoder.decode_forward
    for method in (
        OptimizedDecoder._prefill_mlp_chunk,
        OptimizedDecoder._prefill_linear,
        OptimizedDecoder._prefill_mlp,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder._decode_mlp,
        OptimizedDecoder.allocate_decode_position_buffers,
        OptimizedDecoder.prepare_decode_position_buffers,
        OptimizedDecoder.decode_forward,
    ):
        source = inspect.getsource(method)
        for token in ("torch", "from_torch", "to_torch", "super()"):
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_optimized_synthetic_non_aligned_prefill_decode_and_repeats(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    model = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        # A deliberately conservative optimized policy keeps broad seeded-random
        # stress as a diagnostic without vetoing a faster real-activation policy.
        precision_policy="mlp_bfp4_attention_lofi",
    )
    reference_layer = _hf_layer(state, config)
    generator = torch.Generator().manual_seed(3231)
    seq_len = 31
    hidden = torch.randn(
        (1, EMITTED_BATCH, seq_len, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference, reference_key, reference_value, reference_cache = _reference_layer(reference_layer, hidden, config)
    actual, key_cache, value_cache = _run_prefill(model, mesh_device, hidden)
    prefill_pcc = _assert_pcc(reference, actual, 0.99, "optimized synthetic non-aligned prefill output")
    key_pcc = _assert_pcc(
        reference_key,
        _to_host(key_cache)[:, :, :seq_len, :],
        0.99,
        "optimized synthetic key cache",
    )
    value_pcc = _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, :seq_len, :],
        0.99,
        "optimized synthetic value cache",
    )

    outputs = []
    decode_pccs = []
    for step in range(4):
        decode_hidden = torch.randn(
            (1, EMITTED_BATCH, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        expected, _, _, reference_cache = _reference_layer(
            reference_layer,
            decode_hidden,
            config,
            start_pos=seq_len + step,
            cache=reference_cache,
        )
        output = model.decode_forward(
            _tt_tensor(decode_hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=seq_len + step,
        )
        host = _to_host(output)
        outputs.append(host)
        decode_pccs.append(_assert_pcc(expected, host, 0.99, f"optimized synthetic decode step {step}"))
    assert all(torch.isfinite(output).all() for output in outputs)
    _write_result(
        "synthetic_conservative_stress.json",
        {
            "role": (
                "seeded-random diagnostic on a conservative optimized precision policy; "
                "real recorded activations select the final precision policy"
            ),
            "batch": EMITTED_BATCH,
            "sequence_length": seq_len,
            "prefill_pcc": prefill_pcc,
            "key_cache_pcc": key_pcc,
            "value_cache_pcc": value_pcc,
            "decode_positions": list(range(seq_len, seq_len + len(decode_pccs))),
            "decode_pccs": decode_pccs,
            "optimization_config": model.optimization_summary(),
        },
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_optimized_real_recorded_non_aligned_prefill_decode_and_repeats(mesh_device):
    config = _config()
    state = _real_state()
    model = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        precision_policy=os.getenv("QWEN3_32B_PRECISION_POLICY", "all_bfp4_lofi"),
        decode_down_target_cores=int(os.getenv("QWEN3_32B_DECODE_DOWN_TARGET_CORES", "32")),
    )
    reference_layer = _hf_layer(state, config)
    seq_len = 31
    hidden, activation_metadata = _recorded_hidden(0, seq_len, artifact_path=EXTENDED_ACTIVATION_PATH)
    reference, reference_key, reference_value, reference_cache = _reference_layer(reference_layer, hidden, config)
    actual, key_cache, value_cache = _run_prefill(model, mesh_device, hidden)
    prefill_pcc = _assert_pcc(reference, actual, 0.99, "optimized recorded non-aligned prefill output")
    key_pcc = _assert_pcc(
        reference_key,
        _to_host(key_cache)[:, :, :seq_len, :],
        0.99,
        "optimized recorded non-aligned key cache",
    )
    value_pcc = _assert_pcc(
        reference_value,
        _to_host(value_cache)[:, :, :seq_len, :],
        0.99,
        "optimized recorded non-aligned value cache",
    )

    decode_pccs = []
    for step in range(4):
        decode_hidden, _ = _recorded_hidden(seq_len + step, 1, artifact_path=EXTENDED_ACTIVATION_PATH)
        expected, _, _, reference_cache = _reference_layer(
            reference_layer,
            decode_hidden,
            config,
            start_pos=seq_len + step,
            cache=reference_cache,
        )
        output = model.decode_forward(
            _tt_tensor(decode_hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=seq_len + step,
        )
        decode_pccs.append(
            _assert_pcc(expected, _to_host(output), 0.99, f"optimized recorded non-aligned decode step {step}")
        )
    _write_result(
        "real_recorded_non_aligned_correctness.json",
        {
            "activation_kind": "prompt-derived HF layer-32 input with repeated token sequence",
            "activation_metadata": activation_metadata,
            "batch": EMITTED_BATCH,
            "sequence_length": seq_len,
            "prefill_pcc": prefill_pcc,
            "key_cache_pcc": key_pcc,
            "value_cache_pcc": value_pcc,
            "decode_positions": list(range(seq_len, seq_len + len(decode_pccs))),
            "decode_pccs": decode_pccs,
            "optimization_config": model.optimization_summary(),
        },
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_optimized_real_weight_prefill_decode(mesh_device):
    config = _config()
    state = _real_state()
    model = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        precision_policy=os.getenv("QWEN3_32B_PRECISION_POLICY", "all_bfp4_lofi"),
        decode_target_cores=int(os.getenv("QWEN3_32B_DECODE_TARGET_CORES", "40")),
        decode_down_target_cores=int(os.getenv("QWEN3_32B_DECODE_DOWN_TARGET_CORES", "32")),
        decode_in0_block_w_limit=(
            int(os.environ["QWEN3_32B_DECODE_IN0_BLOCK_W_LIMIT"])
            if "QWEN3_32B_DECODE_IN0_BLOCK_W_LIMIT" in os.environ
            else None
        ),
        decode_matmul_mode=os.getenv("QWEN3_32B_DECODE_MATMUL_MODE", "dram_sharded"),
        prefill_in0_block_w=int(os.getenv("QWEN3_32B_PREFILL_IN0_BLOCK_W", "10")),
        use_packed_mlp=os.getenv("QWEN3_32B_PACKED_MLP", "0") == "1",
        advisor_head_layouts=os.getenv("QWEN3_32B_ADVISOR_HEAD_LAYOUTS", "1") == "1",
    )
    reference_layer = _hf_layer(state, config)
    hidden, activation_metadata = _recorded_hidden(0, EMITTED_PREFILL_SEQUENCE)
    expected, expected_key, expected_value, reference_cache = _reference_layer(reference_layer, hidden, config)
    actual, key_cache, value_cache = _run_prefill(model, mesh_device, hidden)
    prefill_pcc = _assert_pcc(expected, actual, 0.99, "optimized real prefill output")
    key_pcc = _assert_pcc(
        expected_key,
        _to_host(key_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
        0.99,
        "optimized real prefill key cache",
    )
    value_pcc = _assert_pcc(
        expected_value,
        _to_host(value_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
        0.99,
        "optimized real prefill value cache",
    )
    decode_hidden, _ = _recorded_hidden(EMITTED_PREFILL_SEQUENCE, 1)
    expected, _, _, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    actual = model.decode_forward(
        _tt_tensor(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    decode_pcc = _assert_pcc(expected, _to_host(actual), 0.99, "optimized real decode output")
    payload = {
        "activation_kind": "prompt-derived HF layer-32 input",
        "activation_metadata": activation_metadata,
        "batch": EMITTED_BATCH,
        "sequence_length": EMITTED_PREFILL_SEQUENCE,
        "decode_position": EMITTED_PREFILL_SEQUENCE,
        "prefill_pcc": prefill_pcc,
        "key_cache_pcc": key_pcc,
        "value_cache_pcc": value_pcc,
        "decode_pcc": decode_pcc,
        "optimization_config": model.optimization_summary(),
    }
    if os.getenv("TT_METAL_WATCHER"):
        repo_root = Path(__file__).resolve().parents[4]
        watcher_path = repo_root / "generated/watcher/watcher.log"
        watcher_text = watcher_path.read_text(errors="replace")
        fault_patterns = ("error", "assert", "hang", "stuck", "timeout")
        matches = [pattern for pattern in fault_patterns if pattern in watcher_text.lower()]
        assert not matches, f"watcher fault signatures: {matches}"
        payload["watcher"] = {
            "enabled": os.environ["TT_METAL_WATCHER"],
            "log_path": str(watcher_path),
            "fault_patterns": list(fault_patterns),
            "matches": matches,
            "log_sha256": hashlib.sha256(watcher_text.encode()).hexdigest(),
        }
    _write_result("real_weight_correctness.json", payload)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.timeout(600)
def test_traced_decode_advances_position_and_cache(mesh_device):
    config = _config()
    state = _real_state()
    model = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        precision_policy=os.getenv("QWEN3_32B_PRECISION_POLICY", "all_bfp4_lofi"),
        decode_down_target_cores=int(os.getenv("QWEN3_32B_DECODE_DOWN_TARGET_CORES", "32")),
    )
    reference_layer = _hf_layer(state, config)
    hidden, activation_metadata = _recorded_hidden(0, EMITTED_PREFILL_SEQUENCE)
    _, _, _, reference_cache = _reference_layer(reference_layer, hidden, config)
    _, key_cache, value_cache = _run_prefill(model, mesh_device, hidden)

    first_decode, _ = _recorded_hidden(EMITTED_PREFILL_SEQUENCE, 1)
    stable_hidden = _tt_tensor(first_decode, mesh_device)
    position_buffers = model.allocate_decode_position_buffers(EMITTED_PREFILL_SEQUENCE)
    stable_addresses = [stable_hidden.buffer_address()] + [
        tensor.buffer_address() for tensor in position_buffers.tensors()
    ]

    warm_key, warm_value = model.allocate_kv_cache()
    warm = model.decode_forward(
        stable_hidden,
        warm_key,
        warm_value,
        current_pos=EMITTED_PREFILL_SEQUENCE,
        position_buffers=position_buffers,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    # Compile the preallocated slice/fill refresh before a trace exists, then
    # restore the capture position. No device allocation is allowed while the
    # live trace owns its address range.
    model.prepare_decode_position_buffers(position_buffers, EMITTED_PREFILL_SEQUENCE + 1)
    model.prepare_decode_position_buffers(position_buffers, EMITTED_PREFILL_SEQUENCE)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        stable_hidden,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
        position_buffers=position_buffers,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    decode_pccs = []
    key_pccs = []
    value_pccs = []
    trace_outputs = []
    try:
        for current_pos in range(EMITTED_PREFILL_SEQUENCE, EMITTED_PREFILL_SEQUENCE + 4):
            decode_hidden, _ = _recorded_hidden(current_pos, 1)
            host_hidden = ttnn.from_torch(decode_hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_hidden, stable_hidden)
            model.prepare_decode_position_buffers(position_buffers, current_pos)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            actual = _to_host(trace_output)
            expected, _, _, reference_cache = _reference_layer(
                reference_layer,
                decode_hidden,
                config,
                start_pos=current_pos,
                cache=reference_cache,
            )
            decode_pccs.append(_assert_pcc(expected, actual, 0.99, f"traced advancing decode {current_pos}"))
            expected_key = reference_cache.layers[REPRESENTATIVE_LAYER].keys
            expected_value = reference_cache.layers[REPRESENTATIVE_LAYER].values
            actual_key = _to_host(key_cache)[:, :, : current_pos + 1, :]
            actual_value = _to_host(value_cache)[:, :, : current_pos + 1, :]
            key_pccs.append(_assert_pcc(expected_key, actual_key, 0.99, f"traced key history {current_pos}"))
            value_pccs.append(_assert_pcc(expected_value, actual_value, 0.99, f"traced value history {current_pos}"))
            assert stable_addresses == [stable_hidden.buffer_address()] + [
                tensor.buffer_address() for tensor in position_buffers.tensors()
            ]
            trace_outputs.append(actual)
        assert any(
            not torch.equal(trace_outputs[0], later) for later in trace_outputs[1:]
        ), "trace output did not change as hidden state, position, and KV history advanced"
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    _write_result(
        "traced_advancing_decode.json",
        {
            "activation_kind": "prompt-derived HF layer-32 input",
            "activation_metadata": activation_metadata,
            "positions": list(range(EMITTED_PREFILL_SEQUENCE, EMITTED_PREFILL_SEQUENCE + 4)),
            "decode_pccs": decode_pccs,
            "key_history_pccs": key_pccs,
            "value_history_pccs": value_pccs,
            "stable_buffer_addresses": stable_addresses,
            "position_buffer_payload_bytes": 16_640,
            "outputs_changed_across_positions": True,
        },
    )


@pytest.mark.skipif(os.getenv("QWEN3_32B_RUN_PERF") != "1", reason="manual candidate performance gate")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.timeout(1800)
def test_warmed_prefill_and_traced_decode_candidates(mesh_device):
    config = _config()
    state = _real_state()
    reference_layer = _hf_layer(state, config)
    activation_kind = os.getenv("QWEN3_32B_PERF_ACTIVATION_KIND", "recorded")
    if activation_kind == "recorded":
        hidden, activation_metadata = _recorded_hidden(0, EMITTED_PREFILL_SEQUENCE)
        decode_hidden, _ = _recorded_hidden(EMITTED_PREFILL_SEQUENCE, 1)
        activation_label = "prompt-derived HF layer-32 input"
    elif activation_kind == "synthetic":
        generator = torch.Generator().manual_seed(3231)
        hidden = torch.randn(
            (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        decode_hidden = torch.randn(
            (1, EMITTED_BATCH, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
        )
        activation_metadata = {"generator": "torch.randn", "seed": 3231}
        activation_label = "seeded synthetic stress input"
    else:
        raise ValueError(f"unsupported QWEN3_32B_PERF_ACTIVATION_KIND={activation_kind!r}")
    expected_prefill, _, _, reference_cache = _reference_layer(reference_layer, hidden, config)
    expected_decode, _, _, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    constructors = {
        "functional_bf16": lambda: FunctionalDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
        ),
        "optimized_bfp8_hifi2_80c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="bfp8_hifi2",
            decode_target_cores=80,
            prefill_in0_block_w=4,
        ),
        "optimized_bfp8_lofi_80c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="bfp8_lofi",
            decode_target_cores=80,
            prefill_in0_block_w=4,
        ),
        "optimized_mlp_bfp4_lofi_40c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
        ),
        "optimized_mlp_bfp4_lofi_40c_block10": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            decode_in0_block_w_limit=10,
        ),
        "optimized_mlp_bfp4_lofi_40c_block5": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            decode_in0_block_w_limit=5,
        ),
        "optimized_mlp_bfp4_lofi_40c_block2": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            decode_in0_block_w_limit=2,
        ),
        "optimized_mlp_bfp4_hifi2_40c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_hifi2",
            decode_target_cores=40,
        ),
        "optimized_mlp_bfp4_lofi_kv_bf16_40c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi_kv_bf16",
            decode_target_cores=40,
        ),
        "optimized_mlp_bfp4_lofi_sdpa_8x4": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            decode_sdpa_grid=(8, 4),
        ),
        "optimized_mlp_bfp4_lofi_sdpa_exp": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            decode_sdpa_exp_approx=True,
        ),
        "optimized_mlp_bfp4_lofi_prefill_block2": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            prefill_in0_block_w=2,
        ),
        "optimized_mlp_bfp4_lofi_prefill_block8": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            prefill_in0_block_w=8,
        ),
        "optimized_mlp_bfp4_lofi_prefill_block10": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            prefill_in0_block_w=10,
        ),
        "optimized_mlp_bfp4_lofi_prefill_block16": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            prefill_in0_block_w=16,
        ),
        "optimized_advisor_exact": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_matmul_mode="shard_advisor",
        ),
        "optimized_advisor_matmuls_only": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_matmul_mode="shard_advisor",
            advisor_head_layouts=False,
        ),
        "optimized_mlp_bfp4_lofi_32c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=32,
        ),
        "optimized_mlp_bfp4_lofi_packed_40c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=40,
            use_packed_mlp=True,
            decode_in0_block_w_limit=2,
            prefill_in0_block_w=2,
        ),
        "optimized_mlp_bfp4_lofi_80c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=80,
        ),
        "optimized_mlp_bfp4_lofi_packed_80c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=80,
            use_packed_mlp=True,
            prefill_in0_block_w=2,
        ),
        "optimized_mlp_bfp4_lofi_20c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=20,
            decode_in0_block_w_limit=4,
        ),
        "optimized_mlp_bfp4_lofi_16c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_lofi",
            decode_target_cores=16,
            decode_in0_block_w_limit=5,
        ),
        "optimized_gate_up_bfp4_lofi_40c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_gate_up_bfp4_lofi",
            decode_target_cores=40,
        ),
        "optimized_mlp_bfp4_attention_lofi_40c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="mlp_bfp4_attention_lofi",
            decode_target_cores=40,
        ),
        "optimized_attention_bfp4_lofi_80c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="attention_bfp4_lofi",
            decode_target_cores=80,
            decode_down_target_cores=80,
            prefill_in0_block_w=4,
        ),
        "optimized_all_bfp4_lofi_40c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=40,
        ),
        "optimized_all_bfp4_lofi_gate16c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_mlp_target_cores=16,
            decode_down_target_cores=40,
        ),
        "optimized_all_bfp4_lofi_gate16c_block5": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_mlp_target_cores=16,
            decode_down_target_cores=40,
            decode_gate_in0_block_w_limit=5,
        ),
        "optimized_all_bfp4_lofi_gate20c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_mlp_target_cores=20,
            decode_down_target_cores=40,
        ),
        "optimized_all_bfp4_lofi_gate20c_block4": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_mlp_target_cores=20,
            decode_down_target_cores=40,
            decode_gate_in0_block_w_limit=4,
        ),
        "optimized_all_bfp4_lofi_gate32c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_mlp_target_cores=32,
            decode_down_target_cores=40,
        ),
        "optimized_all_bfp4_lofi_gate80c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_mlp_target_cores=80,
            decode_down_target_cores=40,
        ),
        "optimized_all_bfp4_lofi_down16c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=16,
        ),
        "optimized_all_bfp4_lofi_down16c_block5": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=16,
            decode_down_in0_block_w_limit=5,
        ),
        "optimized_all_bfp4_lofi_down16c_block10": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=16,
            decode_down_in0_block_w_limit=10,
        ),
        "optimized_all_bfp4_lofi_down16c_block25": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=16,
            decode_down_in0_block_w_limit=25,
        ),
        "optimized_all_bfp4_lofi_down20c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=20,
        ),
        "optimized_all_bfp4_lofi_down20c_block5": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=20,
            decode_down_in0_block_w_limit=5,
        ),
        "optimized_all_bfp4_lofi_down20c_block8": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=20,
            decode_down_in0_block_w_limit=8,
        ),
        "optimized_all_bfp4_lofi_down20c_block10": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=20,
            decode_down_in0_block_w_limit=10,
        ),
        "optimized_all_bfp4_lofi_down20c_block20": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=20,
            decode_down_in0_block_w_limit=20,
        ),
        "optimized_all_bfp4_lofi_down32c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=32,
        ),
        "optimized_all_bfp4_lofi_down80c": lambda: OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh_device,
            precision_policy="all_bfp4_lofi",
            decode_target_cores=40,
            decode_down_target_cores=80,
        ),
    }
    selected_names = os.getenv("QWEN3_32B_PERF_CANDIDATES", "").split(",")
    selected_names = [name for name in selected_names if name]
    if selected_names:
        constructors = {name: constructors[name] for name in selected_names}
    results = {}
    for name, constructor in constructors.items():
        model = constructor()
        try:
            actual_prefill, key_cache, value_cache = _run_prefill(model, mesh_device, hidden)
            prefill_passed, prefill_pcc = _measure_pcc(expected_prefill, actual_prefill, 0.99, f"{name} prefill")
            prefill_ms = _time_prefill(
                model,
                mesh_device,
                hidden,
                key_cache,
                value_cache,
                iterations=int(os.getenv("QWEN3_32B_PREFILL_PERF_ITERATIONS", "5")),
            )
            decode_ms, first, second = _capture_and_time_decode(
                model,
                mesh_device,
                decode_hidden,
                key_cache,
                value_cache,
                EMITTED_PREFILL_SEQUENCE,
                iterations=int(os.getenv("QWEN3_32B_DECODE_PERF_ITERATIONS", "50")),
            )
        except RuntimeError as error:
            if not os.getenv("QWEN3_32B_ALLOW_RUNTIME_FAILURES"):
                raise
            detail = next(
                (line.strip() for line in str(error).splitlines() if "beyond max L1 size" in line),
                str(error).splitlines()[0],
            )
            results[name] = {
                "activation_kind": activation_label,
                "activation_metadata": activation_metadata,
                "runtime_failed": True,
                "error": detail,
                "optimization_config": model.optimization_summary(),
            }
            print(f"PERF_RESULT {name}: runtime_failed={detail}")
            continue
        decode_passed, decode_pcc = _measure_pcc(expected_decode, first, 0.99, f"{name} traced decode")
        assert torch.equal(first, second), f"{name} traced decode is not deterministic"
        results[name] = {
            "activation_kind": activation_label,
            "activation_metadata": activation_metadata,
            "prefill_passed": prefill_passed,
            "prefill_pcc": prefill_pcc,
            "decode_passed": decode_passed,
            "decode_pcc": decode_pcc,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "optimization_config": (
                model.optimization_summary() if isinstance(model, OptimizedDecoder) else {"path": "functional_bf16"}
            ),
            "roofline": (
                model.roofline_summary(EMITTED_PREFILL_SEQUENCE) if isinstance(model, OptimizedDecoder) else None
            ),
        }
        print(f"PERF_RESULT {name}: prefill_ms={prefill_ms:.6f} traced_decode_ms={decode_ms:.6f}")
        if not os.getenv("QWEN3_32B_ALLOW_PCC_FAILURE"):
            assert prefill_passed, f"{name} prefill PCC {prefill_pcc} is below 0.99"
            assert decode_passed, f"{name} decode PCC {decode_pcc} is below 0.99"
    if "functional_bf16" in results and len(results) > 1:
        assert (
            min(value["decode_ms"] for name, value in results.items() if name != "functional_bf16")
            < results["functional_bf16"]["decode_ms"]
        )
    _write_result(
        os.getenv(RESULT_NAME_ENV, "initial_candidates.json"),
        {"batch": EMITTED_BATCH, "sequence_length": EMITTED_PREFILL_SEQUENCE, "results": results},
    )


@pytest.mark.skipif(os.getenv("QWEN3_32B_RUN_PROFILE") != "1", reason="manual Tracy profile")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.timeout(1800)
def test_profile_selected_decoder(mesh_device):
    from tracy import signpost

    config = _config()
    state = _real_state()
    model = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        precision_policy=os.getenv("QWEN3_32B_PRECISION_POLICY", "all_bfp4_lofi"),
        decode_target_cores=int(os.getenv("QWEN3_32B_DECODE_TARGET_CORES", "40")),
        decode_down_target_cores=int(os.getenv("QWEN3_32B_DECODE_DOWN_TARGET_CORES", "32")),
        decode_in0_block_w_limit=(
            int(os.environ["QWEN3_32B_DECODE_IN0_BLOCK_W_LIMIT"])
            if "QWEN3_32B_DECODE_IN0_BLOCK_W_LIMIT" in os.environ
            else None
        ),
        decode_matmul_mode=os.getenv("QWEN3_32B_DECODE_MATMUL_MODE", "dram_sharded"),
        prefill_in0_block_w=int(os.getenv("QWEN3_32B_PREFILL_IN0_BLOCK_W", "10")),
        use_packed_mlp=os.getenv("QWEN3_32B_PACKED_MLP", "0") == "1",
        advisor_head_layouts=os.getenv("QWEN3_32B_ADVISOR_HEAD_LAYOUTS", "1") == "1",
    )
    hidden, activation_metadata = _recorded_hidden(0, EMITTED_PREFILL_SEQUENCE)
    decode_hidden, _ = _recorded_hidden(EMITTED_PREFILL_SEQUENCE, 1)
    _, key_cache, value_cache = _run_prefill(model, mesh_device, hidden)
    tt_hidden = _tt_tensor(hidden, mesh_device)
    warm = model.prefill_forward(tt_hidden, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm)
    signpost(header="PERF_PREFILL")
    profile_prefill = model.prefill_forward(tt_hidden, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    signpost(header="PERF_PREFILL_END")

    tt_decode = _tt_tensor(decode_hidden, mesh_device)
    warm = model.decode_forward(
        tt_decode,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        tt_decode,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        signpost(header="PERF_DECODE")
        start = time.perf_counter()
        for _ in range(3):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        wall_ms_per_replay = (time.perf_counter() - start) * 1000.0 / 3
        signpost(header="PERF_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    for tensor in (profile_prefill, trace_output, tt_hidden, tt_decode, key_cache, value_cache):
        ttnn.deallocate(tensor)
    _write_result(
        os.getenv(RESULT_NAME_ENV, "profile_run.json"),
        {
            "batch": EMITTED_BATCH,
            "sequence_length": EMITTED_PREFILL_SEQUENCE,
            "decode_position": EMITTED_PREFILL_SEQUENCE,
            "trace_replays": 3,
            "profiled_wall_ms_per_replay": wall_ms_per_replay,
            "activation_kind": "prompt-derived HF layer-32 input",
            "activation_metadata": activation_metadata,
            "optimization_config": model.optimization_summary(),
            "roofline": model.roofline_summary(EMITTED_PREFILL_SEQUENCE),
        },
    )
