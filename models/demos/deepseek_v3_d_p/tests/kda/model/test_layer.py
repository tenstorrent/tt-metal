# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Blackhole PCC tests for the composed TTNN KDA layer."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    collect_mesh_accuracy_and_determinism_results,
    make_config,
    make_program_config,
    random_weights,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 256 * 1024 * 1024}),
]


def _forward(
    layer: ttKDA,
    hidden: torch.Tensor,
    state: KdaState,
) -> tuple[torch.Tensor, KdaState]:
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=layer.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, next_state = layer.forward(hidden_tt, state)
    return ttnn.to_torch(output), next_state


def test_composed_layer_pcc(device: ttnn.Device) -> None:
    config = make_config()
    weights = random_weights(config)
    sequence = 32
    hidden = torch.randn(
        1,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed(41 + sequence),
    ).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)

    layer = ttKDA(device, config, weights)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output, state = layer.forward(hidden_tt, layer.allocate_state())
        return output, state.recurrent, state.convolution

    (output_tt, recurrent_tt, convolution_tt), mismatch_markers = collect_mesh_accuracy_and_determinism_results(run)
    actual_output = ttnn.to_torch(output_tt)
    actual_recurrent = ttnn.to_torch(recurrent_tt)
    actual_convolution = ttnn.to_torch(convolution_tt)
    golden_convolution = torch.cat(
        (
            golden_state.q_convolution,
            golden_state.k_convolution,
            golden_state.v_convolution,
        ),
        dim=-1,
    ).to(torch.bfloat16)
    assert_accurate(golden_output, actual_output, name=f"T={sequence} output", pcc_threshold=0.999)
    assert_accurate(golden_state.recurrent, actual_recurrent, name=f"T={sequence} recurrent state", pcc_threshold=0.999)
    assert_accurate(golden_convolution, actual_convolution, name=f"T={sequence} convolution state", pcc_threshold=0.999)
    assert all(marker.item() == 0 for marker in mismatch_markers), "composed layer is not bit-identical across runs"


def test_offline_cache_and_cache_only_layer_pcc(device: ttnn.Device, tmp_path: Path, expect_error) -> None:
    config = make_config()
    state_dict = random_weights(config)
    hidden = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(151), dtype=torch.bfloat16)
    golden_output, _ = kda_forward_reference(hidden, state_dict, config)
    cache_prefix = "layer_0.kda"

    assert not KDAWeights.check_cache_complete(tmp_path, cache_prefix, config, device)
    with expect_error(FileNotFoundError, "incomplete KDA TTNN cache"):
        KDAWeights.from_cache(tmp_path, cache_prefix, config, device)

    KDAWeights.build_ttnn_cache(state_dict, tmp_path, cache_prefix, config, device)
    assert KDAWeights.check_cache_complete(tmp_path, cache_prefix, config, device)
    cached_weights = KDAWeights.from_cache(tmp_path, cache_prefix, config, device)
    cached_layer = ttKDA(device, config, weights=cached_weights)
    cached_output, _ = _forward(cached_layer, hidden, cached_layer.allocate_state())
    assert_accurate(golden_output, cached_output, name="loaded-cache output", pcc_threshold=0.999)

    cache_only_layer = ttKDA(device, config, None, weight_cache_path=tmp_path, layer_idx=0)
    cache_only_output, _ = _forward(cache_only_layer, hidden, cache_only_layer.allocate_state())
    assert_accurate(golden_output, cache_only_output, name="cache-only output", pcc_threshold=0.999)


def test_cache_only_load_rejects_corrupt_tensorbin(device: ttnn.Device, tmp_path: Path, expect_error) -> None:
    config = make_config()
    cache_prefix = "layer_0.kda"
    KDAWeights.build_ttnn_cache(random_weights(config), tmp_path, cache_prefix, config, device)
    next(tmp_path.glob("*.tensorbin")).write_bytes(b"corrupt")

    with expect_error(RuntimeError, "too small"):
        KDAWeights.from_cache(tmp_path, cache_prefix, config, device)


def test_program_config_controls_tp_topology(device: ttnn.Device) -> None:
    config = make_config()
    program_config = replace(make_program_config(), tp_ccl_topology=ttnn.Topology.Ring)
    layer = ttKDA(device, config, random_weights(config), program_config=program_config)
    assert layer.tp_ccl_topology == ttnn.Topology.Ring


def test_non_tile_aligned_sequence_is_rejected(device: ttnn.Device, expect_error) -> None:
    config = make_config()
    hidden = torch.randn(
        1,
        4,
        config.hidden_size,
        generator=torch.Generator().manual_seed(45),
    ).to(torch.bfloat16)
    layer = ttKDA(device, config, random_weights(config))
    with expect_error(ValueError, r"requires local T .* divisible by 32, got T=4"):
        _forward(layer, hidden, layer.allocate_state())


def test_grouped_scan_rejects_unequal_key_value_dims_at_forward_boundary(device: ttnn.Device, expect_error) -> None:
    config = replace(make_config(), head_v_dim=64)
    base_program_config = make_program_config()
    program_config = replace(
        base_program_config,
        recurrence=replace(
            base_program_config.recurrence,
            local_scan_strategy="grouped",
            summary_group_chunks=1,
        ),
    )
    layer = ttKDA(device, config, random_weights(config), program_config=program_config)
    hidden = torch.randn(1, 32, config.hidden_size, dtype=torch.bfloat16)

    with expect_error(ValueError, "grouped KDA currently requires K == V"):
        _forward(layer, hidden, layer.allocate_state())


def test_batch_greater_than_one_is_rejected_at_state_setup(device: ttnn.Device, expect_error) -> None:
    config = make_config()
    layer = ttKDA(device, config, random_weights(config))

    with expect_error(ValueError, "batch_size=1"):
        layer.allocate_state(batch_size=2)


def test_segmented_prefill_rebinds_cache_hit_runtime_inputs(device: ttnn.Device, isolated_program_cache: None) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden = torch.randn(
        1,
        64,
        config.hidden_size,
        generator=torch.Generator().manual_seed(73),
    ).to(torch.bfloat16)
    golden_first, golden_state = kda_forward_reference(hidden[:, :32], weights, config)
    golden_second, golden_state = kda_forward_reference(hidden[:, 32:], weights, config, golden_state)

    layer = ttKDA(device, config, weights)
    state = layer.allocate_state()
    actual_first, state = _forward(layer, hidden[:, :32], state)
    cache_entries_after_first = device.num_program_cache_entries()
    assert (
        cache_entries_after_first > 0
    ), "KDA forward populated no program cache entries; the cache-hit assertion below would be vacuous"
    actual_second, state = _forward(layer, hidden[:, 32:], state)
    cache_entries_after_second = device.num_program_cache_entries()
    assert cache_entries_after_second == cache_entries_after_first, (
        "second same-shape prefill segment must hit cached programs, entries grew "
        f"{cache_entries_after_first} -> {cache_entries_after_second}"
    )
    actual_recurrent = ttnn.to_torch(state.recurrent)
    actual_convolution = ttnn.to_torch(state.convolution)
    golden_convolution = torch.cat(
        (
            golden_state.q_convolution,
            golden_state.k_convolution,
            golden_state.v_convolution,
        ),
        dim=-1,
    ).to(torch.bfloat16)
    assert_accurate(golden_first, actual_first, name="first prefill segment output", pcc_threshold=0.999)
    assert_accurate(golden_second, actual_second, name="second prefill segment output", pcc_threshold=0.999)
    assert_accurate(golden_state.recurrent, actual_recurrent, name="cache recurrent state", pcc_threshold=0.999)
    assert_accurate(golden_convolution, actual_convolution, name="cache convolution state", pcc_threshold=0.999)


def test_explicit_fp32_state_is_replaced_without_mutating_input(device: ttnn.Device) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden = torch.randn(
        1,
        64,
        config.hidden_size,
        generator=torch.Generator().manual_seed(109),
    ).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)

    layer = ttKDA(
        device,
        config,
        weights,
        program_config=make_program_config(),
    )
    input_state = layer.allocate_state()
    external_recurrent = input_state.recurrent
    external_convolution = input_state.convolution
    recurrent_address = external_recurrent.buffer_address()
    convolution_address = external_convolution.buffer_address()
    first, next_state = _forward(layer, hidden[:, :32], input_state)
    second, next_state = _forward(layer, hidden[:, 32:], next_state)
    actual_output = torch.cat((first, second), dim=1)

    assert input_state.recurrent is external_recurrent
    assert input_state.convolution is external_convolution
    assert input_state.recurrent.buffer_address() == recurrent_address
    assert input_state.convolution.buffer_address() == convolution_address
    assert next_state.recurrent.dtype == ttnn.float32
    actual_recurrent = ttnn.to_torch(next_state.recurrent)
    actual_convolution = ttnn.to_torch(next_state.convolution)
    golden_convolution = torch.cat(
        (
            golden_state.q_convolution,
            golden_state.k_convolution,
            golden_state.v_convolution,
        ),
        dim=-1,
    ).to(torch.bfloat16)
    assert_accurate(golden_output, actual_output, name="external FP32 output", pcc_threshold=0.999)
    assert_accurate(
        golden_state.recurrent,
        actual_recurrent,
        name="external FP32 recurrent state",
        pcc_threshold=0.999,
    )
    assert_accurate(
        golden_convolution,
        actual_convolution,
        name="external BF16 convolution state",
        pcc_threshold=0.999,
    )


def test_composed_layer_immutable_state_trace_replay(device: ttnn.Device) -> None:
    config = make_config()
    hidden = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(1917)).to(torch.bfloat16)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    layer = ttKDA(device, config, random_weights(config))

    # Eager reference for the replay comparison. Reading the traced output between capture and
    # replay would read a buffer the capture never wrote: trace capture records dispatch commands
    # without executing them.
    with ttnn.manage_config("throw_exception_on_fallback", True):
        eager_output, _ = layer.forward(hidden_tt, layer.allocate_state())
    ttnn.synchronize_device(device)
    eager_result = ttnn.to_torch(eager_output)

    input_state = layer.allocate_state()
    input_state_before = (
        ttnn.to_torch(input_state.recurrent),
        ttnn.to_torch(input_state.convolution),
    )
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, next_state = layer.forward(hidden_tt, input_state)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    assert next_state.recurrent is not input_state.recurrent
    assert next_state.convolution is not input_state.convolution

    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    replayed_output = ttnn.to_torch(output)
    input_state_after = (
        ttnn.to_torch(input_state.recurrent),
        ttnn.to_torch(input_state.convolution),
    )
    ttnn.release_trace(device, trace_id)

    assert_bit_identical(eager_result, replayed_output, name="traced layer output")
    for name, expected, actual in zip(("recurrent", "convolution"), input_state_before, input_state_after):
        assert_bit_identical(expected, actual, name=f"traced input {name} state")
