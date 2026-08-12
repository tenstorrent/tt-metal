# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

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
    _hf_layer,
    _real_state,
    _reference_layer,
    _synthetic_state,
)
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.multichip_decoder import TP_DEGREE, MultichipDecoder
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.optimized_decoder import (
    OptimizedDecoder,
    _dram_matmul_program_config,
    _l1_width_sharded_memory_config,
)

BASELINE_ARTIFACT_ENV = "MISTRAL_SMALL_24B_MULTICHIP_BASELINE_ARTIFACT"
BASELINE_IMPL_ENV = "MISTRAL_SMALL_24B_MULTICHIP_BASELINE_IMPL"
BASELINE_REAL_ENV = "MISTRAL_SMALL_24B_MULTICHIP_BASELINE_REAL"
PERF_IMPL_ENV = "MISTRAL_SMALL_24B_MULTICHIP_PERF_IMPL"
PERF_ITERS_ENV = "MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS"
CAPACITY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_CAPACITY"
COLLECTIVE_CANDIDATE_ENV = "MISTRAL_SMALL_24B_MULTICHIP_COLLECTIVE_CANDIDATE"
ATTENTION_GEOMETRY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_ATTENTION_GEOMETRY"
MLP_GEOMETRY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_MLP_GEOMETRY"
GEOMETRY_ARTIFACT_ENV = "MISTRAL_SMALL_24B_MULTICHIP_GEOMETRY_ARTIFACT"
GEOMETRY_ARTIFACT_MODE_ENV = "MISTRAL_SMALL_24B_MULTICHIP_GEOMETRY_ARTIFACT_MODE"
DENSE_KERNEL_FAMILY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_DENSE_KERNEL_FAMILY"
MLP_PROJECTION_FAMILY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_MLP_PROJECTION_FAMILY"
COLLECTIVE_FAMILY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_COLLECTIVE_FAMILY"
COLLECTIVE_DTYPE_ENV = "MISTRAL_SMALL_24B_MULTICHIP_COLLECTIVE_DTYPE"
DECODE_ACTIVATION_DTYPE_ENV = "MISTRAL_SMALL_24B_MULTICHIP_DECODE_ACTIVATION_DTYPE"
ATTENTION_ACTIVATION_DTYPE_ENV = "MISTRAL_SMALL_24B_MULTICHIP_ATTENTION_ACTIVATION_DTYPE"
MLP_ACTIVATION_DTYPE_ENV = "MISTRAL_SMALL_24B_MULTICHIP_MLP_ACTIVATION_DTYPE"
PREFILL_ACTIVATION_FAMILY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_PREFILL_ACTIVATION_FAMILY"
PREFILL_COLLECTIVE_FAMILY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_PREFILL_COLLECTIVE_FAMILY"
PREFILL_COLLECTIVE_DTYPE_ENV = "MISTRAL_SMALL_24B_MULTICHIP_PREFILL_COLLECTIVE_DTYPE"
PROFILE_STACKED_PREFILL_ENV = "MISTRAL_SMALL_24B_MULTICHIP_PROFILE_STACKED_PREFILL"
TRACE_STACKED_PREFILL_ENV = "MISTRAL_SMALL_24B_MULTICHIP_TRACE_STACKED_PREFILL"
ATTENTION_WEIGHT_DTYPE_ENV = "MISTRAL_SMALL_24B_MULTICHIP_ATTENTION_WEIGHT_DTYPE"
MLP_WEIGHT_DTYPE_ENV = "MISTRAL_SMALL_24B_MULTICHIP_MLP_WEIGHT_DTYPE"
ATTENTION_FIDELITY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_ATTENTION_FIDELITY"
MLP_FIDELITY_ENV = "MISTRAL_SMALL_24B_MULTICHIP_MLP_FIDELITY"
ACTIVE_BATCH_ENV = "MISTRAL_SMALL_24B_MULTICHIP_ACTIVE_BATCH"

MESH_PARAMS = [(1, 4)]
DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 100000000}]
STACK_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200000000}]
RING_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200000000}]


def _mesh_input(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _weight_dtype_from_env(name: str):
    value = os.environ.get(name)
    if value is None:
        return None
    if value == "bfp4":
        return ttnn.bfloat4_b
    if value == "bfp8":
        return ttnn.bfloat8_b
    raise ValueError(f"{name} must be bfp4|bfp8, got {value!r}")


def _fidelity_from_env(name: str):
    value = os.environ.get(name)
    if value is None:
        return None
    if value == "lofi":
        return ttnn.MathFidelity.LoFi
    if value == "hifi2":
        return ttnn.MathFidelity.HiFi2
    raise ValueError(f"{name} must be lofi|hifi2, got {value!r}")


def _mesh_host_input(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        layout=layout,
        dtype=dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _single_input(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _multichip_caches(
    config,
    mesh_device,
    *,
    batch: int = EMITTED_BATCH,
    max_cache_len: int = EMITTED_CACHE_LENGTH,
    dtype=ttnn.bfloat8_b,
):
    shape = (batch, config.num_key_value_heads, max_cache_len, config.head_dim)
    host = torch.zeros(shape, dtype=torch.bfloat16)
    kwargs = {
        "device": mesh_device,
        "layout": ttnn.TILE_LAYOUT,
        "dtype": dtype,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "mesh_mapper": ttnn.ShardTensorToMesh(mesh_device, dim=1),
    }
    return ttnn.from_torch(host, **kwargs), ttnn.from_torch(host, **kwargs)


def _single_caches(config, mesh_device, *, batch: int = EMITTED_BATCH, dtype=ttnn.bfloat8_b):
    shape = (batch, config.num_key_value_heads, EMITTED_CACHE_LENGTH, config.head_dim)
    host = torch.zeros(shape, dtype=torch.bfloat16)
    return _single_input(host, mesh_device, dtype=dtype), _single_input(host, mesh_device, dtype=dtype)


def _geometry_from_env(name: str, expected: int):
    text = os.environ.get(name)
    if text is None:
        return None
    values = tuple(int(value) for value in text.split(","))
    if len(values) != expected:
        pytest.fail(f"{name} requires {expected} comma-separated integers, got {text!r}")
    return values


def _replicated_host(tensor) -> torch.Tensor:
    locals_ = ttnn.get_device_tensors(tensor)
    assert len(locals_) in (1, TP_DEGREE)
    first = ttnn.to_torch(locals_[0])
    if len(locals_) == TP_DEGREE:
        for rank, local in enumerate(locals_[1:], start=1):
            other = ttnn.to_torch(local)
            assert torch.equal(first, other), f"replicated tensor differs on rank {rank}"
    return first


def _cache_host(tensor, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))


def _assert_mesh_contract(decoder, output, key_cache, value_cache):
    assert decoder.tp_degree == 4
    assert decoder.tp_axis == 1
    assert decoder.num_heads == 8
    assert decoder.num_kv_heads == 2
    assert decoder.attention_width == 1024
    assert decoder.intermediate_size == 8192
    assert tuple(output.shape) == (1, EMITTED_BATCH, 1, decoder.hidden_size)
    assert len(ttnn.get_device_tensors(output)) == TP_DEGREE
    assert tuple(key_cache.shape) == (EMITTED_BATCH, 2, decoder.max_cache_len, decoder.head_dim)
    assert tuple(value_cache.shape) == tuple(key_cache.shape)


def test_multichip_runtime_is_owned_and_has_no_host_fallback():
    assert issubclass(MultichipDecoder, OptimizedDecoder)
    owned_methods = (
        MultichipDecoder.prefill_forward,
        MultichipDecoder.decode_forward,
        MultichipDecoder._mlp_forward,
    )
    for method in owned_methods:
        assert method.__module__.endswith("multichip_decoder")
        source = inspect.getsource(method)
        for token in ("torch", "from_torch", "to_torch"):
            assert token not in source, f"{method.__name__} contains runtime fallback token {token!r}"
    for method in (MultichipDecoder.prefill_forward, MultichipDecoder.decode_forward):
        source = inspect.getsource(method)
        assert "_all_reduce_hidden" in source
    collective_source = inspect.getsource(MultichipDecoder._all_reduce_hidden)
    assert "ttnn.all_reduce" in collective_source
    assert "ttnn.experimental.all_reduce_async" in collective_source
    defaults = inspect.signature(MultichipDecoder.from_state_dict).parameters
    assert defaults["collective_family"].default == "persistent"
    assert defaults["collective_dtype"].default == "bfp8"
    assert defaults["dense_kernel_family"].default == "dram_sharded"
    assert defaults["mlp_projection_family"].default == "separate"
    assert defaults["prefill_activation_family"].default == "dram"
    assert defaults["prefill_collective_family"].default == "default"
    assert defaults["prefill_collective_dtype"].default == "bf16"
    for method in (
        MultichipDecoder.prepare_decode_residual,
        MultichipDecoder.finish_decode_residual,
        MultichipDecoder.decode_forward_stacked,
        MultichipDecoder.prepare_prefill_residual,
        MultichipDecoder.finish_prefill_residual,
        MultichipDecoder.prefill_forward_stacked,
    ):
        assert method.__module__.endswith("multichip_decoder")
    assert "super().prefill_forward" not in inspect.getsource(MultichipDecoder.prefill_forward)
    assert "super().decode_forward" not in inspect.getsource(MultichipDecoder.decode_forward)


@pytest.mark.parametrize("mesh_shape", [(2, 2), (4, 1)])
def test_multichip_rejects_non_target_mesh_shape(mesh_shape, expect_error):
    class FakeFourDeviceMesh:
        shape = mesh_shape

        @staticmethod
        def get_num_devices():
            return TP_DEGREE

    with expect_error(ValueError, "requires the target logical 1x4 mesh"):
        MultichipDecoder.from_state_dict({}, hf_config=None, layer_idx=LAYER_IDX, mesh_device=FakeFourDeviceMesh())


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_multichip_synthetic_prefill_decode_pcc_layout_and_trace(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device)
    reference_layer = _hf_layer(state, config)
    key_cache, value_cache = _multichip_caches(config, mesh_device)
    generator = torch.Generator().manual_seed(2510)

    # Includes both tile-aligned and non-aligned public sequence lengths.
    reference_cache = None
    for seq_len in (17, EMITTED_PREFILL_SEQUENCE, 32):
        hidden = torch.randn((1, EMITTED_BATCH, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        reference, reference_key, reference_value, reference_cache = _reference_layer(reference_layer, hidden, config)
        actual = decoder.prefill_forward(_mesh_input(hidden, mesh_device), key_cache, value_cache)
        _assert_pcc(reference, _replicated_host(actual), 0.99, f"multichip prefill seq={seq_len}")
        _assert_pcc(
            reference_key,
            _cache_host(key_cache, mesh_device)[:, :, :seq_len, :],
            0.99,
            f"multichip prefill seq={seq_len} key cache",
        )
        _assert_pcc(
            reference_value,
            _cache_host(value_cache, mesh_device)[:, :, :seq_len, :],
            0.99,
            f"multichip prefill seq={seq_len} value cache",
        )

    current_pos = 32
    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference_decode, decode_key, decode_value, reference_cache = _reference_layer(
        reference_layer, decode_hidden, config, start_pos=current_pos, cache=reference_cache
    )
    tt_decode = _mesh_input(decode_hidden, mesh_device)
    actual_decode = decoder.decode_forward(tt_decode, key_cache, value_cache, current_pos=current_pos)
    _assert_mesh_contract(decoder, actual_decode, key_cache, value_cache)
    _assert_pcc(reference_decode, _replicated_host(actual_decode), 0.99, "multichip decode")
    _assert_pcc(
        decode_key,
        _cache_host(key_cache, mesh_device)[:, :, current_pos : current_pos + 1, :],
        0.99,
        "multichip decode key append",
    )
    _assert_pcc(
        decode_value,
        _cache_host(value_cache, mesh_device)[:, :, current_pos : current_pos + 1, :],
        0.99,
        "multichip decode value append",
    )

    # CCL, cache update, SDPA, and both TP matmul groups must all be trace-safe.
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(tt_decode, key_cache, value_cache, current_pos=current_pos)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = _replicated_host(traced_output)
    first_key = _cache_host(key_cache, mesh_device)[:, :, current_pos : current_pos + 1, :]
    first_value = _cache_host(value_cache, mesh_device)[:, :, current_pos : current_pos + 1, :]
    for _ in range(3):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(first, _replicated_host(traced_output))
    assert torch.equal(first_key, _cache_host(key_cache, mesh_device)[:, :, current_pos : current_pos + 1, :])
    assert torch.equal(first_value, _cache_host(value_cache, mesh_device)[:, :, current_pos : current_pos + 1, :])
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_multichip_paged_cache_page_table_and_positions(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device)
    reference_layer = _hf_layer(state, config)
    block_size = 32
    blocks_per_user = EMITTED_CACHE_LENGTH // block_size
    num_blocks = EMITTED_BATCH * blocks_per_user
    cache_shape = (num_blocks, config.num_key_value_heads, block_size, config.head_dim)
    host_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    cache_kwargs = {
        "device": mesh_device,
        "layout": ttnn.TILE_LAYOUT,
        "dtype": ttnn.bfloat8_b,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "mesh_mapper": ttnn.ShardTensorToMesh(mesh_device, dim=1),
    }
    key_cache = ttnn.from_torch(host_cache, **cache_kwargs)
    value_cache = ttnn.from_torch(host_cache, **cache_kwargs)
    contiguous_key_cache, contiguous_value_cache = _multichip_caches(config, mesh_device)

    # Reverse physical block ownership so the test cannot pass by treating the
    # logical position as a contiguous physical cache index.
    page_table_host = torch.arange(num_blocks - 1, -1, -1, dtype=torch.int32).reshape(EMITTED_BATCH, blocks_per_user)
    page_table = _mesh_input(page_table_host, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    generator = torch.Generator().manual_seed(2511)
    hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference, _, _, reference_cache = _reference_layer(reference_layer, hidden, config)
    contiguous_prefill = decoder.prefill_forward(
        _mesh_input(hidden, mesh_device), contiguous_key_cache, contiguous_value_cache
    )
    actual = decoder.prefill_forward(_mesh_input(hidden, mesh_device), key_cache, value_cache, page_table=page_table)
    _assert_pcc(reference, _replicated_host(actual), 0.99, "multichip paged prefill")
    _assert_pcc(
        _replicated_host(contiguous_prefill),
        _replicated_host(actual),
        0.9999,
        "matched paged/contiguous prefill",
    )

    def logical_paged_cache(cache):
        physical = _cache_host(cache, mesh_device)
        return torch.stack(
            [
                torch.cat([physical[page_table_host[user, block]] for block in range(blocks_per_user)], dim=1)
                for user in range(EMITTED_BATCH)
            ]
        )

    _assert_pcc(
        _cache_host(contiguous_key_cache, mesh_device),
        logical_paged_cache(key_cache),
        0.9999,
        "matched paged/contiguous prefill K cache",
    )
    _assert_pcc(
        _cache_host(contiguous_value_cache, mesh_device),
        logical_paged_cache(value_cache),
        0.9999,
        "matched paged/contiguous prefill V cache",
    )

    current_pos = EMITTED_PREFILL_SEQUENCE
    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference, decode_key, decode_value, _ = _reference_layer(
        reference_layer, decode_hidden, config, start_pos=current_pos, cache=reference_cache
    )
    contiguous_actual = decoder.decode_forward(
        _mesh_input(decode_hidden, mesh_device),
        contiguous_key_cache,
        contiguous_value_cache,
        current_pos=current_pos,
    )
    actual = decoder.decode_forward(
        _mesh_input(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=current_pos,
        page_table=page_table,
    )
    _assert_pcc(reference, _replicated_host(contiguous_actual), 0.99, "matched contiguous decode")
    _assert_pcc(reference, _replicated_host(actual), 0.99, "multichip paged decode")
    _assert_pcc(
        _replicated_host(contiguous_actual),
        _replicated_host(actual),
        0.9999,
        "matched paged/contiguous decode",
    )

    physical_key = _cache_host(key_cache, mesh_device)
    physical_value = _cache_host(value_cache, mesh_device)
    logical_block = current_pos // block_size
    block_offset = current_pos % block_size
    gathered_key = torch.stack(
        [physical_key[page_table_host[user, logical_block], :, block_offset, :] for user in range(EMITTED_BATCH)]
    ).unsqueeze(2)
    gathered_value = torch.stack(
        [physical_value[page_table_host[user, logical_block], :, block_offset, :] for user in range(EMITTED_BATCH)]
    ).unsqueeze(2)
    _assert_pcc(decode_key, gathered_key, 0.99, "paged physical key mapping")
    _assert_pcc(decode_value, gathered_value, 0.99, "paged physical value mapping")


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_multichip_mutable_nonuniform_positions_trace(mesh_device):
    """One persistent position tensor must drive RoPE, cache, and SDPA on replay."""

    config = _config()
    state = _synthetic_state(config)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device)
    reference_layer = _hf_layer(state, config)
    generator = torch.Generator().manual_seed(2515)
    prefill_len = 33
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, prefill_len, config.hidden_size), generator=generator, dtype=torch.bfloat16
    )
    decode_inputs = [
        torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    initial_positions = torch.tensor([32 if user % 2 == 0 else 33 for user in range(EMITTED_BATCH)], dtype=torch.int32)

    def run_trace(*, paged: bool):
        if paged:
            block_size = 32
            blocks_per_user = EMITTED_CACHE_LENGTH // block_size
            num_blocks = EMITTED_BATCH * blocks_per_user
            cache_shape = (num_blocks, config.num_key_value_heads, block_size, config.head_dim)
            host_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
            cache_kwargs = {
                "device": mesh_device,
                "layout": ttnn.TILE_LAYOUT,
                "dtype": ttnn.bfloat8_b,
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "mesh_mapper": ttnn.ShardTensorToMesh(mesh_device, dim=1),
            }
            key_cache = ttnn.from_torch(host_cache, **cache_kwargs)
            value_cache = ttnn.from_torch(host_cache, **cache_kwargs)
            page_table_host = torch.arange(num_blocks - 1, -1, -1, dtype=torch.int32).reshape(
                EMITTED_BATCH, blocks_per_user
            )
            page_table = _mesh_input(page_table_host, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
        else:
            block_size = None
            page_table_host = None
            page_table = None
            key_cache, value_cache = _multichip_caches(config, mesh_device)

        decoder.prefill_forward(
            _mesh_input(prefill_hidden, mesh_device),
            key_cache,
            value_cache,
            page_table=page_table,
        )

        # Even users have a 32-token prompt; odd users have a 33-token prompt.
        # The TT cache was filled to 33 for every user, but cur_pos masks the
        # even users at 32 and their decode write overwrites that physical slot.
        reference_caches = []
        for user, position in enumerate(initial_positions.tolist()):
            _, _, _, cache = _reference_layer(
                reference_layer,
                prefill_hidden[:, user : user + 1, :position, :],
                config,
            )
            reference_caches.append(cache)

        persistent_hidden = _mesh_input(decode_inputs[0], mesh_device)
        persistent_positions = _mesh_input(
            initial_positions, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        # Compile every decode program once and flush prefill/cache writes plus
        # host-to-device initialization before capture. TTNN deliberately rejects
        # program-load transfers or reads that overlap a trace.
        decoder.decode_forward(
            persistent_hidden,
            key_cache,
            value_cache,
            current_pos_tensor=persistent_positions,
            page_table=page_table,
        )
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = decoder.decode_forward(
            persistent_hidden,
            key_cache,
            value_cache,
            current_pos_tensor=persistent_positions,
            page_table=page_table,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        for step in range(3):
            positions = initial_positions + step
            if step:
                ttnn.copy_host_to_device_tensor(
                    _mesh_host_input(decode_inputs[step], mesh_device), persistent_hidden, cq_id=0
                )
                ttnn.copy_host_to_device_tensor(
                    _mesh_host_input(positions, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32),
                    persistent_positions,
                    cq_id=0,
                )
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)

            for rank, local in enumerate(ttnn.get_device_tensors(persistent_positions)):
                assert torch.equal(ttnn.to_torch(local), positions), f"position copy missed rank {rank}"

            reference_outputs = []
            reference_keys = []
            reference_values = []
            for user, position in enumerate(positions.tolist()):
                output, key, value, cache = _reference_layer(
                    reference_layer,
                    decode_inputs[step][:, user : user + 1, :, :],
                    config,
                    start_pos=position,
                    cache=reference_caches[user],
                )
                reference_caches[user] = cache
                reference_outputs.append(output)
                reference_keys.append(key)
                reference_values.append(value)
            reference_output = torch.cat(reference_outputs, dim=1)
            reference_key = torch.cat(reference_keys, dim=0)
            reference_value = torch.cat(reference_values, dim=0)
            _assert_pcc(
                reference_output,
                _replicated_host(traced_output),
                0.99,
                f"mutable trace {'paged' if paged else 'contiguous'} step={step}",
            )

            physical_key = _cache_host(key_cache, mesh_device)
            physical_value = _cache_host(value_cache, mesh_device)
            if paged:
                actual_key = torch.stack(
                    [
                        physical_key[page_table_host[user, position // block_size], :, position % block_size, :]
                        for user, position in enumerate(positions.tolist())
                    ]
                ).unsqueeze(2)
                actual_value = torch.stack(
                    [
                        physical_value[page_table_host[user, position // block_size], :, position % block_size, :]
                        for user, position in enumerate(positions.tolist())
                    ]
                ).unsqueeze(2)
            else:
                actual_key = torch.stack(
                    [physical_key[user, :, position, :] for user, position in enumerate(positions.tolist())]
                ).unsqueeze(2)
                actual_value = torch.stack(
                    [physical_value[user, :, position, :] for user, position in enumerate(positions.tolist())]
                ).unsqueeze(2)
            _assert_pcc(reference_key, actual_key, 0.99, f"mutable trace K step={step}")
            _assert_pcc(reference_value, actual_value, 0.99, f"mutable trace V step={step}")

        stable_output = _replicated_host(traced_output).clone()
        stable_key = _cache_host(key_cache, mesh_device).clone()
        stable_value = _cache_host(value_cache, mesh_device).clone()
        for _ in range(3):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        assert torch.equal(stable_output, _replicated_host(traced_output))
        assert torch.equal(stable_key, _cache_host(key_cache, mesh_device))
        assert torch.equal(stable_value, _cache_host(value_cache, mesh_device))
        ttnn.release_trace(mesh_device, trace_id)

    run_trace(paged=False)
    run_trace(paged=True)


def _artifact_mesh_param():
    return 1 if os.environ.get(BASELINE_IMPL_ENV) == "optimized" else (1, 4)


def _artifact_device_params():
    if os.environ.get(BASELINE_IMPL_ENV) == "multichip":
        return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 100000000}
    return {"trace_region_size": 100000000}


@pytest.mark.parametrize("device_params", [_artifact_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_artifact_mesh_param()], indirect=True)
def test_multichip_pcc_against_single_chip_optimized_baseline(mesh_device):
    artifact_text = os.environ.get(BASELINE_ARTIFACT_ENV)
    implementation = os.environ.get(BASELINE_IMPL_ENV)
    if not artifact_text or implementation not in {"optimized", "multichip"}:
        pytest.skip(f"Set {BASELINE_ARTIFACT_ENV} and {BASELINE_IMPL_ENV}=optimized|multichip")
    artifact_path = Path(artifact_text)
    config = _config()
    active_batch = int(os.environ.get(ACTIVE_BATCH_ENV, EMITTED_BATCH))
    assert 1 <= active_batch <= EMITTED_BATCH
    state = _real_state() if os.environ.get(BASELINE_REAL_ENV) == "1" else _synthetic_state(config)
    generator = torch.Generator().manual_seed(2512)
    prefill_hidden = torch.randn((1, active_batch, 17, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode_hidden = torch.randn((1, active_batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)

    if implementation == "optimized":
        # The inherited TP1 decoder exposes only its emitted batch-32 return
        # shape. For a batch-1 reference, place active users first and pad only
        # the independent inactive users; compare/snapshot active rows below.
        reference_batch = EMITTED_BATCH
        if active_batch < reference_batch:
            prefill_hidden = torch.cat(
                (
                    prefill_hidden,
                    torch.zeros((1, reference_batch - active_batch, 17, config.hidden_size), dtype=torch.bfloat16),
                ),
                dim=1,
            )
            decode_hidden = torch.cat(
                (
                    decode_hidden,
                    torch.zeros((1, reference_batch - active_batch, 1, config.hidden_size), dtype=torch.bfloat16),
                ),
                dim=1,
            )
        decoder = OptimizedDecoder.from_state_dict(
            state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, batch=reference_batch
        )
        key_cache, value_cache = _single_caches(config, mesh_device, batch=reference_batch)
        prefill = decoder.prefill_forward(_single_input(prefill_hidden, mesh_device), key_cache, value_cache)
        decode = decoder.decode_forward(
            _single_input(decode_hidden, mesh_device), key_cache, value_cache, current_pos=17
        )
        ttnn.synchronize_device(mesh_device)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prefill": ttnn.to_torch(prefill)[:, :active_batch],
                "decode": ttnn.to_torch(decode)[:, :active_batch],
                "key": ttnn.to_torch(key_cache)[:active_batch, :, :18, :],
                "value": ttnn.to_torch(value_cache)[:active_batch, :, :18, :],
            },
            artifact_path,
        )
        print(
            f"BASELINE_ARTIFACT wrote={artifact_path} active_batch={active_batch} " f"reference_batch={reference_batch}"
        )
        return

    assert artifact_path.is_file(), f"Run the optimized artifact command first: {artifact_path}"
    baseline = torch.load(artifact_path, weights_only=True)
    decoder = MultichipDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, batch=active_batch
    )
    key_cache, value_cache = _multichip_caches(config, mesh_device, batch=active_batch)
    prefill = decoder.prefill_forward(_mesh_input(prefill_hidden, mesh_device), key_cache, value_cache)
    decode = decoder.decode_forward(_mesh_input(decode_hidden, mesh_device), key_cache, value_cache, current_pos=17)
    _assert_pcc(baseline["prefill"], _replicated_host(prefill), 0.99, "optimized/multichip prefill")
    _assert_pcc(baseline["decode"], _replicated_host(decode), 0.99, "optimized/multichip decode")
    _assert_pcc(baseline["key"], _cache_host(key_cache, mesh_device)[:, :, :18, :], 0.99, "optimized/multichip K")
    _assert_pcc(baseline["value"], _cache_host(value_cache, mesh_device)[:, :, :18, :], 0.99, "optimized/multichip V")


def _perf_mesh_param():
    return 1 if os.environ.get(PERF_IMPL_ENV) == "optimized" else (1, 4)


def _perf_device_params():
    if os.environ.get(PERF_IMPL_ENV) == "multichip":
        return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 100000000}
    return {"trace_region_size": 100000000}


@pytest.mark.parametrize("device_params", [_perf_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_perf_mesh_param()], indirect=True)
def test_warmed_single_chip_and_multichip_perf(mesh_device):
    implementation = os.environ.get(PERF_IMPL_ENV)
    if implementation not in {"optimized", "multichip"}:
        pytest.skip(f"Set {PERF_IMPL_ENV}=optimized|multichip")
    iterations = int(os.environ.get(PERF_ITERS_ENV, "20"))
    config = _config()
    state = _real_state()
    decoder_cls = OptimizedDecoder if implementation == "optimized" else MultichipDecoder
    decoder_kwargs = {}
    if implementation == "multichip":
        attention_geometry = _geometry_from_env(ATTENTION_GEOMETRY_ENV, 6)
        mlp_geometry = _geometry_from_env(MLP_GEOMETRY_ENV, 5)
        decoder_kwargs["dense_kernel_family"] = os.environ.get(DENSE_KERNEL_FAMILY_ENV, "dram_sharded")
        decoder_kwargs["mlp_projection_family"] = os.environ.get(MLP_PROJECTION_FAMILY_ENV, "separate")
        decoder_kwargs["collective_family"] = os.environ.get(COLLECTIVE_FAMILY_ENV, "persistent")
        decoder_kwargs["collective_dtype"] = os.environ.get(COLLECTIVE_DTYPE_ENV, "bfp8")
        decoder_kwargs["decode_activation_dtype"] = os.environ.get(DECODE_ACTIVATION_DTYPE_ENV, "bf16")
        attention_activation_dtype = os.environ.get(ATTENTION_ACTIVATION_DTYPE_ENV)
        mlp_activation_dtype = os.environ.get(MLP_ACTIVATION_DTYPE_ENV)
        if attention_activation_dtype is not None:
            decoder_kwargs["attention_activation_dtype"] = attention_activation_dtype
        if mlp_activation_dtype is not None:
            decoder_kwargs["mlp_activation_dtype"] = mlp_activation_dtype
        decoder_kwargs["prefill_activation_family"] = os.environ.get(PREFILL_ACTIVATION_FAMILY_ENV, "dram")
        decoder_kwargs["prefill_collective_family"] = os.environ.get(PREFILL_COLLECTIVE_FAMILY_ENV, "default")
        decoder_kwargs["prefill_collective_dtype"] = os.environ.get(PREFILL_COLLECTIVE_DTYPE_ENV, "bf16")
        attention_weight_dtype = _weight_dtype_from_env(ATTENTION_WEIGHT_DTYPE_ENV)
        mlp_weight_dtype = _weight_dtype_from_env(MLP_WEIGHT_DTYPE_ENV)
        attention_fidelity = _fidelity_from_env(ATTENTION_FIDELITY_ENV)
        mlp_fidelity = _fidelity_from_env(MLP_FIDELITY_ENV)
        if attention_weight_dtype is not None:
            decoder_kwargs["attention_weight_dtype"] = attention_weight_dtype
        if mlp_weight_dtype is not None:
            decoder_kwargs["mlp_weight_dtype"] = mlp_weight_dtype
        if attention_fidelity is not None:
            decoder_kwargs["attention_math_fidelity"] = attention_fidelity
        if mlp_fidelity is not None:
            decoder_kwargs["mlp_math_fidelity"] = mlp_fidelity
        if attention_geometry is not None:
            decoder_kwargs["attention_geometry"] = attention_geometry
        if mlp_geometry is not None:
            decoder_kwargs["mlp_geometry"] = mlp_geometry
    decoder = decoder_cls.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, **decoder_kwargs
    )
    if implementation == "optimized":
        key_cache, value_cache = _single_caches(config, mesh_device)
        to_device = lambda tensor: _single_input(tensor, mesh_device)
        to_host = ttnn.to_torch
        cache_to_host = lambda tensor: ttnn.to_torch(tensor)
    else:
        key_cache, value_cache = _multichip_caches(config, mesh_device)
        to_device = lambda tensor: _mesh_input(tensor, mesh_device)
        to_host = _replicated_host
        cache_to_host = lambda tensor: _cache_host(tensor, mesh_device)

    generator = torch.Generator().manual_seed(2513)
    prefill_hidden = torch.randn(
        (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    tt_prefill = to_device(prefill_hidden)
    decoder.prefill_forward(tt_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    tracy.signpost("MULTICHIP_PREFILL")
    start = time.perf_counter()
    decoder.prefill_forward(tt_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.perf_counter() - start) * 1000
    tracy.signpost("MULTICHIP_PREFILL_END")

    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    tt_decode = to_device(decode_hidden)
    eager = decoder.decode_forward(tt_decode, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
    ttnn.synchronize_device(mesh_device)
    eager_host = to_host(eager)
    geometry_artifact = os.environ.get(GEOMETRY_ARTIFACT_ENV)
    geometry_artifact_mode = os.environ.get(GEOMETRY_ARTIFACT_MODE_ENV)
    if implementation == "multichip" and geometry_artifact:
        if geometry_artifact_mode == "write":
            torch.save(eager_host, geometry_artifact)
        elif geometry_artifact_mode == "compare":
            _assert_pcc(
                torch.load(geometry_artifact, weights_only=True),
                eager_host,
                0.9999,
                "geometry/default output",
            )
        else:
            pytest.fail(f"{GEOMETRY_ARTIFACT_MODE_ENV} must be write|compare when an artifact path is set")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = decoder.decode_forward(tt_decode, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(4):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    stable_output = to_host(traced).clone()
    stable_key = cache_to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE, :].clone()
    stable_value = cache_to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE, :].clone()
    _assert_pcc(eager_host, stable_output, 0.9999, f"{implementation} eager/traced")

    tracy.signpost("MULTICHIP_DECODE")
    start = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    decode_ms = (time.perf_counter() - start) * 1000 / iterations
    tracy.signpost("MULTICHIP_DECODE_END")
    assert torch.equal(stable_output, to_host(traced)), f"{implementation} trace output was not deterministic"
    assert torch.equal(
        stable_key, cache_to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE, :]
    ), f"{implementation} traced K-cache writes were not deterministic"
    assert torch.equal(
        stable_value, cache_to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE, :]
    ), f"{implementation} traced V-cache writes were not deterministic"
    ttnn.release_trace(mesh_device, trace_id)
    print(
        f"MULTICHIP_PERF_RESULT impl={implementation} prefill_seq={EMITTED_PREFILL_SEQUENCE} "
        f"prefill_ms={prefill_ms:.6f} traced_decode_ms={decode_ms:.6f} iterations={iterations}"
    )


@pytest.mark.parametrize("device_params", STACK_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_multichip_two_layer_stacked_decode_layout_perf(mesh_device):
    """Compare repeated public restores with a stack-compatible residual layout."""

    iterations = int(os.environ.get(PERF_ITERS_ENV, "50"))
    active_batch = int(os.environ.get(ACTIVE_BATCH_ENV, EMITTED_BATCH))
    assert 1 <= active_batch <= EMITTED_BATCH
    config = _config()
    state = _real_state()
    dense_kernel_family = os.environ.get(DENSE_KERNEL_FAMILY_ENV, "dram_sharded")
    mlp_projection_family = os.environ.get(MLP_PROJECTION_FAMILY_ENV, "separate")
    collective_family = os.environ.get(COLLECTIVE_FAMILY_ENV, "persistent")
    collective_dtype = os.environ.get(COLLECTIVE_DTYPE_ENV, "bfp8")
    decode_activation_dtype = os.environ.get(DECODE_ACTIVATION_DTYPE_ENV, "bf16")
    attention_activation_dtype = os.environ.get(ATTENTION_ACTIVATION_DTYPE_ENV)
    mlp_activation_dtype = os.environ.get(MLP_ACTIVATION_DTYPE_ENV)
    attention_geometry = _geometry_from_env(ATTENTION_GEOMETRY_ENV, 6)
    mlp_geometry = _geometry_from_env(MLP_GEOMETRY_ENV, 5)
    attention_weight_dtype = _weight_dtype_from_env(ATTENTION_WEIGHT_DTYPE_ENV)
    mlp_weight_dtype = _weight_dtype_from_env(MLP_WEIGHT_DTYPE_ENV)
    attention_fidelity = _fidelity_from_env(ATTENTION_FIDELITY_ENV)
    mlp_fidelity = _fidelity_from_env(MLP_FIDELITY_ENV)
    decoder_kwargs = {
        "dense_kernel_family": dense_kernel_family,
        "mlp_projection_family": mlp_projection_family,
        "collective_family": collective_family,
        "collective_dtype": collective_dtype,
        "decode_activation_dtype": decode_activation_dtype,
        "attention_activation_dtype": attention_activation_dtype,
        "mlp_activation_dtype": mlp_activation_dtype,
        "batch": active_batch,
    }
    for name, value in (
        ("attention_geometry", attention_geometry),
        ("mlp_geometry", mlp_geometry),
        ("attention_weight_dtype", attention_weight_dtype),
        ("mlp_weight_dtype", mlp_weight_dtype),
        ("attention_math_fidelity", attention_fidelity),
        ("mlp_math_fidelity", mlp_fidelity),
    ):
        if value is not None:
            decoder_kwargs[name] = value
    first = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        **decoder_kwargs,
    )
    second = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        shared_rope=first.shared_rope,
        shared_collective=first.shared_collective,
        **decoder_kwargs,
    )
    first_caches = _multichip_caches(config, mesh_device, batch=active_batch)
    second_caches = _multichip_caches(config, mesh_device, batch=active_batch)
    generator = torch.Generator().manual_seed(2514)
    hidden = torch.randn((1, active_batch, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    tt_hidden = _mesh_input(hidden, mesh_device)
    current_pos_tensor = _mesh_input(
        torch.full((active_batch,), EMITTED_PREFILL_SEQUENCE, dtype=torch.int32),
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )

    def public_pair():
        output = first.decode_forward(tt_hidden, *first_caches, current_pos_tensor=current_pos_tensor)
        return second.decode_forward(output, *second_caches, current_pos_tensor=current_pos_tensor)

    stacked_input = first.prepare_decode_residual(tt_hidden)

    def stacked_pair():
        output = first.decode_forward_stacked(stacked_input, *first_caches, current_pos_tensor=current_pos_tensor)
        return second.decode_forward_stacked(output, *second_caches, current_pos_tensor=current_pos_tensor)

    eager_public = public_pair()
    eager_stacked = stacked_pair()
    ttnn.synchronize_device(mesh_device)
    public_host = _replicated_host(eager_public)
    stacked_host = _replicated_host(eager_stacked).reshape(public_host.shape)
    _assert_pcc(public_host, stacked_host, 0.9999, "public/stacked two-layer output")
    geometry_artifact = os.environ.get(GEOMETRY_ARTIFACT_ENV)
    geometry_artifact_mode = os.environ.get(GEOMETRY_ARTIFACT_MODE_ENV)
    if geometry_artifact:
        if geometry_artifact_mode == "write":
            torch.save(stacked_host, geometry_artifact)
        elif geometry_artifact_mode == "compare":
            _assert_pcc(
                torch.load(geometry_artifact, weights_only=True),
                stacked_host,
                0.9999,
                "stack candidate/default output",
            )
        else:
            pytest.fail(f"{GEOMETRY_ARTIFACT_MODE_ENV} must be write|compare when an artifact path is set")

    def capture_and_time(run_pair):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = run_pair()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        stable = _replicated_host(traced_output).clone()
        start = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
        assert torch.equal(stable, _replicated_host(traced_output))
        ttnn.release_trace(mesh_device, trace_id)
        return elapsed_ms, stable

    tracy.signpost("MULTICHIP_PUBLIC_STACK_DECODE")
    public_ms, traced_public = capture_and_time(public_pair)
    tracy.signpost("MULTICHIP_PUBLIC_STACK_DECODE_END")
    tracy.signpost("MULTICHIP_INTERNAL_STACK_DECODE")
    stacked_ms, traced_stacked = capture_and_time(stacked_pair)
    tracy.signpost("MULTICHIP_INTERNAL_STACK_DECODE_END")
    _assert_pcc(traced_public, traced_stacked.reshape(traced_public.shape), 0.9999, "traced public/stacked")
    assert stacked_ms < public_ms
    print(
        "MULTICHIP_STACK_LAYOUT_RESULT "
        f"active_batch={active_batch} "
        f"dense_kernel_family={dense_kernel_family} "
        f"mlp_projection_family={mlp_projection_family} "
        f"collective_family={collective_family} "
        f"collective_dtype={collective_dtype} "
        f"attention_activation_dtype={attention_activation_dtype or decode_activation_dtype} "
        f"mlp_activation_dtype={mlp_activation_dtype or decode_activation_dtype} "
        f"attention_geometry={first.attention_geometry} mlp_geometry={first.mlp_geometry} "
        f"attention_weight_dtype={os.environ.get(ATTENTION_WEIGHT_DTYPE_ENV, 'bfp4')} "
        f"mlp_weight_dtype={os.environ.get(MLP_WEIGHT_DTYPE_ENV, 'bfp4')} "
        f"attention_math_fidelity={os.environ.get(ATTENTION_FIDELITY_ENV, 'lofi')} "
        f"mlp_math_fidelity={os.environ.get(MLP_FIDELITY_ENV, 'lofi')} "
        f"public_two_layer_ms={public_ms:.6f} internal_two_layer_ms={stacked_ms:.6f} "
        f"public_per_layer_ms={public_ms / 2:.6f} internal_per_layer_ms={stacked_ms / 2:.6f} "
        f"speedup={public_ms / stacked_ms:.6f} iterations={iterations}"
    )


@pytest.mark.parametrize("device_params", STACK_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_multichip_two_layer_stacked_prefill_layout_perf(mesh_device):
    """Compare repeated public prefill transposes with one stack-boundary conversion."""

    iterations = int(os.environ.get(PERF_ITERS_ENV, "10"))
    active_batch = int(os.environ.get(ACTIVE_BATCH_ENV, EMITTED_BATCH))
    assert 1 <= active_batch <= EMITTED_BATCH
    config = _config()
    state = _real_state()
    prefill_activation_family = os.environ.get(PREFILL_ACTIVATION_FAMILY_ENV, "dram")
    prefill_collective_family = os.environ.get(PREFILL_COLLECTIVE_FAMILY_ENV, "default")
    prefill_collective_dtype = os.environ.get(PREFILL_COLLECTIVE_DTYPE_ENV, "bf16")
    first = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=active_batch,
        prefill_activation_family=prefill_activation_family,
        prefill_collective_family=prefill_collective_family,
        prefill_collective_dtype=prefill_collective_dtype,
    )
    second = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=active_batch,
        shared_rope=first.shared_rope,
        shared_collective=first.shared_collective,
        shared_prefill_collective=first.shared_prefill_collective,
        prefill_activation_family=prefill_activation_family,
        prefill_collective_family=prefill_collective_family,
        prefill_collective_dtype=prefill_collective_dtype,
    )
    first_caches = _multichip_caches(config, mesh_device, batch=active_batch)
    second_caches = _multichip_caches(config, mesh_device, batch=active_batch)
    generator = torch.Generator().manual_seed(2515)
    hidden = torch.randn(
        (1, active_batch, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    tt_hidden = _mesh_input(hidden, mesh_device)

    def public_pair():
        output = first.prefill_forward(tt_hidden, *first_caches)
        return second.prefill_forward(output, *second_caches)

    stacked_input, logical_seq_len = first.prepare_prefill_residual(tt_hidden)

    def stacked_pair():
        output = first.prefill_forward_stacked(stacked_input, *first_caches, logical_seq_len=logical_seq_len)
        return second.prefill_forward_stacked(output, *second_caches, logical_seq_len=logical_seq_len)

    eager_public = public_pair()
    eager_stacked = stacked_pair()
    ttnn.synchronize_device(mesh_device)
    public_host = _replicated_host(eager_public)
    stacked_host = (
        _replicated_host(eager_stacked)
        .reshape(1, logical_seq_len, active_batch, config.hidden_size)
        .permute(0, 2, 1, 3)
    )
    _assert_pcc(public_host, stacked_host, 0.9999, "public/stacked two-layer prefill")
    geometry_artifact = os.environ.get(GEOMETRY_ARTIFACT_ENV)
    geometry_artifact_mode = os.environ.get(GEOMETRY_ARTIFACT_MODE_ENV)
    if geometry_artifact:
        if geometry_artifact_mode == "write":
            torch.save(stacked_host, geometry_artifact)
        elif geometry_artifact_mode == "compare":
            _assert_pcc(
                torch.load(geometry_artifact, weights_only=True),
                stacked_host,
                0.999,
                "prefill candidate/default output",
            )
        else:
            pytest.fail(f"{GEOMETRY_ARTIFACT_MODE_ENV} must be write|compare when an artifact path is set")

    def time_warmed(run_pair):
        run_pair()
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(iterations):
            run_pair()
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - start) * 1000 / iterations

    tracy.signpost("MULTICHIP_PUBLIC_STACK_PREFILL")
    public_ms = time_warmed(public_pair)
    tracy.signpost("MULTICHIP_PUBLIC_STACK_PREFILL_END")
    tracy.signpost("MULTICHIP_INTERNAL_STACK_PREFILL")
    stacked_ms = time_warmed(stacked_pair)
    tracy.signpost("MULTICHIP_INTERNAL_STACK_PREFILL_END")
    assert stacked_ms < public_ms
    print(
        "MULTICHIP_PREFILL_STACK_LAYOUT_RESULT "
        f"active_batch={active_batch} logical_seq_len={logical_seq_len} "
        f"prefill_activation_family={prefill_activation_family} "
        f"prefill_collective_family={prefill_collective_family} "
        f"prefill_collective_dtype={prefill_collective_dtype} public_two_layer_ms={public_ms:.6f} "
        f"internal_two_layer_ms={stacked_ms:.6f} public_per_layer_ms={public_ms / 2:.6f} "
        f"internal_per_layer_ms={stacked_ms / 2:.6f} speedup={public_ms / stacked_ms:.6f} "
        f"iterations={iterations}"
    )


@pytest.mark.parametrize("device_params", STACK_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_multichip_profile_internal_stacked_prefill(mesh_device):
    """Profile only the stack-preserving two-layer prefill region."""

    if os.environ.get(PROFILE_STACKED_PREFILL_ENV) != "1":
        pytest.skip(f"Set {PROFILE_STACKED_PREFILL_ENV}=1 to profile stacked prefill")
    iterations = int(os.environ.get(PERF_ITERS_ENV, "1"))
    active_batch = int(os.environ.get(ACTIVE_BATCH_ENV, EMITTED_BATCH))
    config = _config()
    decoder = MultichipDecoder.from_state_dict(
        _real_state(), hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, batch=active_batch
    )
    first_caches = _multichip_caches(config, mesh_device, batch=active_batch)
    second_caches = _multichip_caches(config, mesh_device, batch=active_batch)
    generator = torch.Generator().manual_seed(2519)
    hidden = torch.randn(
        (1, active_batch, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    stacked_input, logical_seq_len = decoder.prepare_prefill_residual(_mesh_input(hidden, mesh_device))

    def stacked_pair():
        output = decoder.prefill_forward_stacked(stacked_input, *first_caches, logical_seq_len=logical_seq_len)
        return decoder.prefill_forward_stacked(output, *second_caches, logical_seq_len=logical_seq_len)

    stacked_pair()
    ttnn.synchronize_device(mesh_device)
    tracy.signpost("MULTICHIP_INTERNAL_STACK_PREFILL")
    start = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = stacked_pair()
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
    tracy.signpost("MULTICHIP_INTERNAL_STACK_PREFILL_END")
    assert tuple(output.shape) == (1, 1, active_batch * logical_seq_len, config.hidden_size)
    print(
        "MULTICHIP_INTERNAL_STACK_PREFILL_PROFILE_RESULT "
        f"active_batch={active_batch} logical_seq_len={logical_seq_len} two_layer_ms={elapsed_ms:.6f} "
        f"per_layer_ms={elapsed_ms / 2:.6f} iterations={iterations}"
    )


@pytest.mark.parametrize("device_params", STACK_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_multichip_two_layer_stacked_prefill_trace_perf(mesh_device):
    """Measure the profiler's op-gap advice on the decoder-local prefill graph."""

    if os.environ.get(TRACE_STACKED_PREFILL_ENV) != "1":
        pytest.skip(f"Set {TRACE_STACKED_PREFILL_ENV}=1 to trace stacked prefill")
    iterations = int(os.environ.get(PERF_ITERS_ENV, "50"))
    active_batch = int(os.environ.get(ACTIVE_BATCH_ENV, "1"))
    assert 1 <= active_batch <= EMITTED_BATCH
    config = _config()
    state = _real_state()
    first = MultichipDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, batch=active_batch
    )
    second = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=active_batch,
        shared_rope=first.shared_rope,
        shared_collective=first.shared_collective,
        shared_prefill_collective=first.shared_prefill_collective,
    )
    first_caches = _multichip_caches(config, mesh_device, batch=active_batch)
    second_caches = _multichip_caches(config, mesh_device, batch=active_batch)
    generator = torch.Generator().manual_seed(2521)
    hidden = torch.randn(
        (1, active_batch, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    stacked_input, logical_seq_len = first.prepare_prefill_residual(_mesh_input(hidden, mesh_device))

    def stacked_pair():
        output = first.prefill_forward_stacked(stacked_input, *first_caches, logical_seq_len=logical_seq_len)
        return second.prefill_forward_stacked(output, *second_caches, logical_seq_len=logical_seq_len)

    eager_output = stacked_pair()
    ttnn.synchronize_device(mesh_device)
    eager_host = _replicated_host(eager_output).clone()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = stacked_pair()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        traced_host = _replicated_host(traced_output).clone()
        _assert_pcc(eager_host, traced_host, 0.9999, "eager/traced stacked prefill")
        tracy.signpost("MULTICHIP_INTERNAL_STACK_PREFILL_TRACE")
        start = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
        tracy.signpost("MULTICHIP_INTERNAL_STACK_PREFILL_TRACE_END")
        stable_host = _replicated_host(traced_output)
        assert torch.equal(traced_host, stable_host)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    print(
        "MULTICHIP_PREFILL_TRACE_RESULT "
        f"active_batch={active_batch} logical_seq_len={logical_seq_len} two_layer_ms={elapsed_ms:.6f} "
        f"per_layer_ms={elapsed_ms / 2:.6f} iterations={iterations} eager_traced_pcc=1.0"
    )


@pytest.mark.parametrize("device_params", STACK_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_multichip_full_context_paged_cache_capacity(mesh_device):
    if os.environ.get(CAPACITY_ENV) != "1":
        pytest.skip(f"Set {CAPACITY_ENV}=1 to run the full-stack 32K decode allocation envelope")
    config = _config()
    max_cache_len = int(config.max_position_embeddings)
    state = _synthetic_state(config)
    decoder = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=max_cache_len,
    )
    assert not decoder.prefill_weights_released
    shared_decoder = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=max_cache_len,
        shared_rope=decoder.shared_rope,
        shared_collective=decoder.shared_collective,
    )
    assert shared_decoder.rotary_cos is decoder.rotary_cos
    assert shared_decoder.decode_rotary_cos is decoder.decode_rotary_cos
    assert shared_decoder.collective_workspace is decoder.collective_workspace
    assert shared_decoder.collective_semaphore is decoder.collective_semaphore
    assert not shared_decoder.prefill_weights_released

    # Keep the complete 40-layer decode matrix/norm lifetime resident. Layer 0
    # is the real decoder above; the remaining 39 use the same physical local
    # shapes, dtypes, and DRAM shard specs without staging host contents. The
    # shared_decoder above is the second real layer, so 38 synthetic allocations
    # complete the 40-layer stack.
    decode_weight_shapes = (
        (tuple(decoder.qkv_weight.shape), decoder.qkv_weight.dtype, decoder.qkv_weight.memory_config()),
        (tuple(decoder.output_weight.shape), decoder.output_weight.dtype, decoder.output_weight.memory_config()),
        (tuple(decoder.gate_weight.shape), decoder.gate_weight.dtype, decoder.gate_weight.memory_config()),
        (tuple(decoder.up_weight.shape), decoder.up_weight.dtype, decoder.up_weight.memory_config()),
        (tuple(decoder.down_weight.shape), decoder.down_weight.dtype, decoder.down_weight.memory_config()),
        (tuple(decoder.input_norm.shape), decoder.input_norm.dtype, decoder.input_norm.memory_config()),
        (
            tuple(decoder.post_attention_norm.shape),
            decoder.post_attention_norm.dtype,
            decoder.post_attention_norm.memory_config(),
        ),
    )
    prefill_weight_shapes = tuple(
        (
            tuple(getattr(decoder, name).shape),
            getattr(decoder, name).dtype,
            getattr(decoder, name).memory_config(),
        )
        for name in (
            "prefill_qkv_weight",
            "prefill_output_weight",
            "prefill_gate_up_weight",
            "prefill_down_weight",
        )
    )
    stack_constants = []
    for _ in range(38):
        for shape, dtype, memory_config in decode_weight_shapes:
            stack_constants.append(
                ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mesh_device, memory_config)
            )
        for shape, dtype, memory_config in prefill_weight_shapes:
            stack_constants.append(
                ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mesh_device, memory_config)
            )

    # Simulate TP4 BF16 embedding + untied LM head local vocabulary shards and
    # the replicated final norm expected by the full-model handoff.
    local_vocab = config.vocab_size // TP_DEGREE
    stack_constants.extend(
        [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([local_vocab, config.hidden_size]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([config.hidden_size, local_vocab]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([config.hidden_size]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            ),
        ]
    )
    block_size = 32
    blocks_per_user = max_cache_len // block_size
    # allocate_tensor_on_device takes the local TP cache shape and allocates it
    # on every mesh rank without staging a multi-gigabyte host tensor.
    local_shape = ttnn.Shape(
        [EMITTED_BATCH * blocks_per_user, config.num_key_value_heads // TP_DEGREE, block_size, config.head_dim]
    )
    layer_caches = []
    for _ in range(40):
        layer_caches.append(
            (
                ttnn.allocate_tensor_on_device(
                    local_shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
                ),
                ttnn.allocate_tensor_on_device(
                    local_shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
                ),
            )
        )
    key_cache, value_cache = layer_caches[0]
    page_table_host = torch.arange(EMITTED_BATCH * blocks_per_user, dtype=torch.int32).reshape(
        EMITTED_BATCH, blocks_per_user
    )
    page_table = _mesh_input(page_table_host, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    current_positions = _mesh_input(
        torch.full((EMITTED_BATCH,), max_cache_len - 1, dtype=torch.int32),
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    rotary_positions = _mesh_input(
        torch.full((1, EMITTED_BATCH), max_cache_len - 1, dtype=torch.int32),
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    # Physically reserve 1.5 GiB/rank for runtime activations, programs, and
    # collective scratch after all steady-state constants, both optimized
    # decode and prefill matrix representations, full caches, and the fixture's
    # separate 200 MB-per-DRAM-bank (1.6 GB/device) trace region exist.
    runtime_reserve = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([8192, 32768]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(3)
    ]

    # Exercise the retained public prefill matrices while every full-context
    # cache and all 40 layers' physical weight representations are resident.
    prefill_hidden = torch.zeros((1, EMITTED_BATCH, block_size, config.hidden_size), dtype=torch.bfloat16)
    prefill_output = decoder.prefill_forward(
        _mesh_input(prefill_hidden, mesh_device),
        key_cache,
        value_cache,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    assert tuple(_replicated_host(prefill_output).shape) == tuple(prefill_hidden.shape)

    hidden = torch.zeros((1, EMITTED_BATCH, 1, config.hidden_size), dtype=torch.bfloat16)
    current_pos = max_cache_len - 1
    output = decoder.decode_forward(
        _mesh_input(hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos_tensor=current_positions,
        rotary_pos_tensor=rotary_positions,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    assert tuple(_replicated_host(output).shape) == tuple(hidden.shape)
    print(
        f"MULTICHIP_CAPACITY_PASS layers=40 batch={EMITTED_BATCH} max_cache_len={max_cache_len} "
        f"local_cache_shape={tuple(local_shape)} current_pos={current_pos} "
        "prefill_weights_resident=true prefill_tokens_per_user=32 "
        f"runtime_reserve_bytes={3 * 8192 * 32768 * 2}"
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_collective_chain_all_reduce_vs_hidden_sharded(mesh_device):
    """Measure collectives through distributed RMSNorm and the next QKV GEMM."""

    if os.environ.get(COLLECTIVE_CANDIDATE_ENV) != "1":
        pytest.skip(f"Set {COLLECTIVE_CANDIDATE_ENV}=1 to run the collective consumer-chain comparison")
    iterations = int(os.environ.get(PERF_ITERS_ENV, "50"))
    config = _config()
    state = _synthetic_state(config)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device)
    generator = torch.Generator().manual_seed(2514)
    partial_host = torch.randn((1, 1, EMITTED_BATCH, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    partial = _mesh_input(partial_host, mesh_device)
    norm_host = state[f"model.layers.{LAYER_IDX}.input_layernorm.weight"]
    local_norm = ttnn.from_torch(
        norm_host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    def replicated_chain():
        hidden = ttnn.all_reduce(
            partial,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = ttnn.to_memory_config(hidden, decoder.decode_norm_mem_config)
        hidden = ttnn.rms_norm(
            hidden,
            epsilon=decoder.rms_norm_eps,
            weight=decoder.input_norm,
            memory_config=decoder.decode_norm_mem_config,
            program_config=decoder.decode_norm_program_config,
        )
        hidden = ttnn.to_memory_config(hidden, decoder.decode_qkv_input_mem_config)
        return ttnn.matmul(
            hidden,
            decoder.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=decoder.decode_qkv_output_mem_config,
            program_config=decoder.decode_qkv_program_config,
            compute_kernel_config=decoder.attention_compute_kernel_config,
        )

    def hidden_sharded_chain():
        hidden = ttnn.reduce_scatter(
            partial,
            dim=3,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        stats = ttnn.rms_norm_pre_all_gather(
            hidden, compute_kernel_config=decoder.attention_compute_kernel_config, dtype=ttnn.bfloat16
        )
        stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = ttnn.rms_norm_post_all_gather(
            hidden,
            stats,
            epsilon=decoder.rms_norm_eps,
            weight=local_norm,
            compute_kernel_config=decoder.attention_compute_kernel_config,
        )
        hidden = ttnn.all_gather(
            hidden,
            dim=3,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = ttnn.to_memory_config(hidden, decoder.decode_qkv_input_mem_config)
        return ttnn.matmul(
            hidden,
            decoder.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=decoder.decode_qkv_output_mem_config,
            program_config=decoder.decode_qkv_program_config,
            compute_kernel_config=decoder.attention_compute_kernel_config,
        )

    def trace_and_time(chain):
        chain()
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = chain()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
        host = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(output)]
        ttnn.release_trace(mesh_device, trace_id)
        return elapsed_ms, host

    replicated_ms, replicated_host = trace_and_time(replicated_chain)
    sharded_ms, sharded_host = trace_and_time(hidden_sharded_chain)
    for rank, (reference, actual) in enumerate(zip(replicated_host, sharded_host, strict=True)):
        _assert_pcc(reference, actual, 0.999, f"collective chain rank={rank}")
    print(
        f"COLLECTIVE_CHAIN_RESULT replicated_ms={replicated_ms:.6f} hidden_sharded_ms={sharded_ms:.6f} "
        f"ratio={sharded_ms / replicated_ms:.6f} iterations={iterations}"
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_collective_chain_default_vs_explicit_persistent(mesh_device):
    """Compare the exact decoder consumer chain with the persistent-buffer overload."""

    if os.environ.get(COLLECTIVE_CANDIDATE_ENV) != "persistent":
        pytest.skip(f"Set {COLLECTIVE_CANDIDATE_ENV}=persistent to run the persistent all-reduce comparison")
    iterations = int(os.environ.get(PERF_ITERS_ENV, "50"))
    config = _config()
    state = _synthetic_state(config)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device)
    generator = torch.Generator().manual_seed(2516)
    partial_host = torch.randn((1, 1, EMITTED_BATCH, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    partial = ttnn.to_memory_config(_mesh_input(partial_host, mesh_device), decoder.decode_mlp_output_mem_config)

    intermediate_mem_config = _l1_width_sharded_memory_config(
        mesh_device, ttnn.TILE_SIZE, config.hidden_size * TP_DEGREE, decoder.mlp_geometry[2]
    )
    persistent_buffer = ttnn.from_torch(
        torch.zeros((1, 1, EMITTED_BATCH, config.hidden_size * TP_DEGREE), dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=intermediate_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    grid_size = mesh_device.compute_with_storage_grid_size()
    worker_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    global_semaphore = ttnn.create_global_semaphore(mesh_device, worker_grid, 0)

    def consume(hidden):
        hidden = ttnn.to_memory_config(hidden, decoder.decode_norm_mem_config)
        hidden = ttnn.rms_norm(
            hidden,
            epsilon=decoder.rms_norm_eps,
            weight=decoder.input_norm,
            memory_config=decoder.decode_norm_mem_config,
            program_config=decoder.decode_norm_program_config,
        )
        hidden = ttnn.to_memory_config(hidden, decoder.decode_qkv_input_mem_config)
        return ttnn.matmul(
            hidden,
            decoder.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=decoder.decode_qkv_output_mem_config,
            program_config=decoder.decode_qkv_program_config,
            compute_kernel_config=decoder.attention_compute_kernel_config,
        )

    def default_chain():
        hidden = ttnn.to_memory_config(partial, ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.all_reduce(
            hidden,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return consume(hidden)

    def persistent_chain():
        hidden = ttnn.experimental.all_reduce_async(
            partial,
            persistent_buffer,
            cluster_axis=1,
            mesh_device=mesh_device,
            multi_device_global_semaphore=global_semaphore,
            dtype=ttnn.bfloat16,
            memory_config=decoder.decode_mlp_output_mem_config,
            topology=ttnn.Topology.Linear,
            num_links=2,
        )
        return consume(hidden)

    def trace_and_time(chain):
        chain()
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = chain()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
        host = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(output)]
        ttnn.release_trace(mesh_device, trace_id)
        return elapsed_ms, host

    default_ms, default_host = trace_and_time(default_chain)
    persistent_ms, persistent_host = trace_and_time(persistent_chain)
    for rank, (reference, actual) in enumerate(zip(default_host, persistent_host, strict=True)):
        _assert_pcc(reference, actual, 0.9999, f"persistent all-reduce chain rank={rank}")
    print(
        f"PERSISTENT_ALL_REDUCE_RESULT default_ms={default_ms:.6f} persistent_ms={persistent_ms:.6f} "
        f"ratio={persistent_ms / default_ms:.6f} iterations={iterations}"
    )


@pytest.mark.parametrize("device_params", RING_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_fused_matmul_ccl_fractured_residual_chain(mesh_device):
    """Keep a hidden shard through distributed norm and fused next-layer QKV."""

    candidate = os.environ.get(COLLECTIVE_CANDIDATE_ENV)
    if candidate not in {"fused", "fused_interleaved"}:
        pytest.skip(f"Set {COLLECTIVE_CANDIDATE_ENV}=fused|fused_interleaved to run the fused CCL family")
    use_interleaved_family = candidate == "fused_interleaved"
    iterations = int(os.environ.get(PERF_ITERS_ENV, "50"))
    config = _config()
    state = _real_state()
    decoder = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Ring,
        num_links=1,
    )
    generator = torch.Generator().manual_seed(2517)
    down_input_host = torch.randn(
        (1, 1, EMITTED_BATCH, config.intermediate_size), generator=generator, dtype=torch.bfloat16
    )
    down_input = ttnn.from_torch(
        down_input_host,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    if not use_interleaved_family:
        down_input_mem_config = _l1_width_sharded_memory_config(
            mesh_device, ttnn.TILE_SIZE, config.intermediate_size // TP_DEGREE, decoder.mlp_geometry[1]
        )
        down_input = ttnn.to_memory_config(down_input, down_input_mem_config)
    local_norm = ttnn.from_torch(
        state[f"model.layers.{LAYER_IDX}.input_layernorm.weight"],
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    grid_size = mesh_device.compute_with_storage_grid_size()
    worker_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([worker_grid])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    mrs_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_grid, 0) for _ in range(3)]
    ag_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_grid, 0) for _ in range(2)]
    ag_barrier = ttnn.create_global_semaphore(mesh_device, worker_grid, 0)

    mrs_intermediate = _mesh_input(
        torch.zeros((1, 1, EMITTED_BATCH, config.hidden_size), dtype=torch.bfloat16), mesh_device
    )
    if not use_interleaved_family:
        mrs_intermediate = ttnn.to_memory_config(mrs_intermediate, decoder.decode_mlp_output_mem_config)
    mrs_output = _mesh_input(
        torch.zeros((1, 1, EMITTED_BATCH, config.hidden_size // TP_DEGREE), dtype=torch.bfloat16),
        mesh_device,
    )
    ag_output = _mesh_input(torch.zeros((1, 1, EMITTED_BATCH, config.hidden_size), dtype=torch.bfloat16), mesh_device)

    if use_interleaved_family:
        down_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 6),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=20,
            out_block_w=10,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 6),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=3,
            per_core_M=1,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        down_weight = decoder.prefill_down_weight
        qkv_weight = decoder.prefill_qkv_weight
        fused_mm_memory_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        down_program_config = _dram_matmul_program_config(
            ttnn.TILE_SIZE,
            config.intermediate_size // TP_DEGREE,
            config.hidden_size,
            decoder.mlp_geometry[1],
            decoder.mlp_geometry[2],
            max_in0_block_w=decoder.mlp_geometry[4],
        )
        qkv_program_config = decoder.decode_qkv_program_config
        down_weight = decoder.down_weight
        qkv_weight = decoder.qkv_weight
        fused_mm_memory_config = decoder.decode_mlp_output_mem_config
    qkv_weight_4d = ttnn.reshape(
        qkv_weight,
        [1, 1, config.hidden_size, int(qkv_weight.shape[-1])],
    )

    def baseline_chain():
        partial = ttnn.matmul(
            down_input,
            down_weight,
            dtype=ttnn.bfloat16,
            memory_config=fused_mm_memory_config,
            program_config=down_program_config,
            compute_kernel_config=decoder.mlp_compute_kernel_config,
        )
        hidden = ttnn.all_reduce(
            partial,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if use_interleaved_family:
            hidden = ttnn.rms_norm(
                hidden,
                epsilon=decoder.rms_norm_eps,
                weight=decoder.input_norm,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            hidden = ttnn.to_memory_config(hidden, decoder.decode_norm_mem_config)
            hidden = ttnn.rms_norm(
                hidden,
                epsilon=decoder.rms_norm_eps,
                weight=decoder.input_norm,
                memory_config=decoder.decode_norm_mem_config,
                program_config=decoder.decode_norm_program_config,
            )
            hidden = ttnn.to_memory_config(hidden, decoder.decode_qkv_input_mem_config)
        return ttnn.matmul(
            hidden,
            qkv_weight_4d,
            dtype=ttnn.bfloat16,
            memory_config=(ttnn.DRAM_MEMORY_CONFIG if use_interleaved_family else decoder.decode_qkv_output_mem_config),
            program_config=qkv_program_config,
            compute_kernel_config=decoder.attention_compute_kernel_config,
        )

    def fused_chain():
        _, hidden_shard = ttnn.experimental.matmul_reduce_scatter_async(
            down_input,
            down_weight,
            persistent_intermediate_buffer=mrs_intermediate,
            persistent_output_buffer=mrs_output,
            dim=3,
            multi_device_global_semaphore=mrs_semaphores,
            reduce_scatter_core_grid_offset=(0, 6),
            num_links=1,
            memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            memory_config_mm=fused_mm_memory_config,
            program_config=down_program_config,
            compute_kernel_config=decoder.mlp_compute_kernel_config,
        )
        stats = ttnn.rms_norm_pre_all_gather(
            hidden_shard, compute_kernel_config=decoder.attention_compute_kernel_config, dtype=ttnn.bfloat16
        )
        stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normed_shard = ttnn.rms_norm_post_all_gather(
            hidden_shard,
            stats,
            epsilon=decoder.rms_norm_eps,
            weight=local_norm,
            compute_kernel_config=decoder.attention_compute_kernel_config,
        )
        _, qkv = ttnn.experimental.all_gather_matmul_async(
            normed_shard,
            qkv_weight_4d,
            persistent_output_buffer=ag_output,
            dim=3,
            multi_device_global_semaphore=ag_semaphores,
            all_gather_core_grid_offset=(0, 6),
            barrier_semaphore=ag_barrier,
            num_links=1,
            memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
            memory_config_mm=(
                ttnn.DRAM_MEMORY_CONFIG if use_interleaved_family else decoder.decode_qkv_output_mem_config
            ),
            program_config=qkv_program_config,
            compute_kernel_config=decoder.attention_compute_kernel_config,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        return qkv

    def trace_and_time(chain):
        chain()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = chain()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
        start = time.perf_counter()
        for _ in range(iterations):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
        host = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(output)]
        ttnn.release_trace(mesh_device, trace_id)
        return elapsed_ms, host

    baseline_ms, baseline_host = trace_and_time(baseline_chain)
    fused_ms, fused_host = trace_and_time(fused_chain)
    for rank, (reference, actual) in enumerate(zip(baseline_host, fused_host, strict=True)):
        _assert_pcc(reference, actual, 0.999, f"fused fractured residual chain rank={rank}")
    print(
        f"FUSED_CCL_FAMILY_RESULT candidate={candidate} baseline_ms={baseline_ms:.6f} fused_ms={fused_ms:.6f} "
        f"ratio={fused_ms / baseline_ms:.6f} iterations={iterations}"
    )
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize("device_params", RING_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_fused_matmul_ccl_exact_final_boundaries(mesh_device):
    """Compare both fused fractured-residual boundaries with exact final controls."""

    if os.environ.get(COLLECTIVE_CANDIDATE_ENV) != "fused_exact_boundaries":
        pytest.skip(f"Set {COLLECTIVE_CANDIDATE_ENV}=fused_exact_boundaries to run the exact fused-boundary comparison")
    iterations = int(os.environ.get(PERF_ITERS_ENV, "50"))
    config = _config()
    state = _real_state()

    # Use real layer activations at every material boundary. The public layer
    # input is the attention residual; the hooks capture the actual WO input,
    # post-attention residual, and SwiGLU down input.
    reference_layer = _hf_layer(state, config)
    captures = {}
    hook_handles = []

    def capture_input(name):
        def hook(_module, args):
            captures[name] = args[0].detach().to(torch.bfloat16).clone()

        return hook

    hook_handles.extend(
        (
            reference_layer.self_attn.o_proj.register_forward_pre_hook(capture_input("wo_input")),
            reference_layer.post_attention_layernorm.register_forward_pre_hook(
                capture_input("post_attention_residual")
            ),
            reference_layer.mlp.down_proj.register_forward_pre_hook(capture_input("down_input")),
        )
    )
    generator = torch.Generator().manual_seed(2519)
    hidden_host = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    try:
        _reference_layer(reference_layer, hidden_host, config)
    finally:
        for handle in hook_handles:
            handle.remove()

    assert set(captures) == {"wo_input", "post_attention_residual", "down_input"}
    attention_residual_host = hidden_host.reshape(1, 1, EMITTED_BATCH, config.hidden_size)
    wo_input_host = captures["wo_input"].reshape(1, 1, EMITTED_BATCH, config.num_attention_heads * config.head_dim)
    post_attention_residual_host = captures["post_attention_residual"].reshape(1, 1, EMITTED_BATCH, config.hidden_size)
    down_input_host = captures["down_input"].reshape(1, 1, EMITTED_BATCH, config.intermediate_size)
    assert tuple(wo_input_host.shape) == (1, 1, EMITTED_BATCH, 4096)
    assert tuple(attention_residual_host.shape) == (1, 1, EMITTED_BATCH, 5120)
    assert tuple(post_attention_residual_host.shape) == (1, 1, EMITTED_BATCH, 5120)
    assert tuple(down_input_host.shape) == (1, 1, EMITTED_BATCH, 32768)

    # A Ring-capable fabric session is required by MRS/AGMM. The control
    # decoder deliberately retains the final Linear/two-link persistent policy.
    decoder = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
    )
    assert decoder.collective_family == "persistent"
    assert decoder.collective_dtype == ttnn.bfloat8_b
    assert decoder.collective_topology == ttnn.Topology.Linear
    assert decoder.num_links == 2
    assert decoder.collective_workspace is decoder.shared_collective[0]
    assert decoder.collective_semaphore is decoder.shared_collective[1]
    assert decoder.attention_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert decoder.mlp_compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi
    assert decoder.qkv_weight.dtype == ttnn.bfloat4_b
    assert decoder.output_weight.dtype == ttnn.bfloat4_b
    assert decoder.gate_weight.dtype == ttnn.bfloat4_b
    assert decoder.up_weight.dtype == ttnn.bfloat4_b
    assert decoder.down_weight.dtype == ttnn.bfloat4_b
    for program_config in (
        decoder.decode_qkv_program_config,
        decoder.decode_o_program_config,
    ):
        assert isinstance(program_config, ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig)

    def mesh_shard(host, *, dim, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            host,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )

    wo_input = mesh_shard(wo_input_host, dim=3)
    down_input = mesh_shard(down_input_host, dim=3)
    attention_residual = ttnn.to_memory_config(
        _mesh_input(attention_residual_host, mesh_device), decoder.decode_norm_mem_config
    )
    post_attention_residual = ttnn.to_memory_config(
        _mesh_input(post_attention_residual_host, mesh_device), decoder.decode_norm_mem_config
    )
    attention_residual_shard = mesh_shard(attention_residual_host, dim=3)
    post_attention_residual_shard = mesh_shard(post_attention_residual_host, dim=3)
    control_wo_input = ttnn.to_memory_config(wo_input, decoder.decode_o_input_mem_config)

    mlp_input_cores, mlp_intermediate_cores, mlp_output_cores, gate_max_block, down_max_block = decoder.mlp_geometry
    control_mlp_input_mem_config = _l1_width_sharded_memory_config(
        mesh_device, ttnn.TILE_SIZE, config.hidden_size, mlp_input_cores
    )
    control_intermediate_mem_config = _l1_width_sharded_memory_config(
        mesh_device, ttnn.TILE_SIZE, decoder.intermediate_size, mlp_intermediate_cores
    )
    control_gate_program_config = _dram_matmul_program_config(
        ttnn.TILE_SIZE,
        config.hidden_size,
        decoder.intermediate_size,
        mlp_input_cores,
        mlp_intermediate_cores,
        max_in0_block_w=gate_max_block,
    )
    control_down_program_config = _dram_matmul_program_config(
        ttnn.TILE_SIZE,
        decoder.intermediate_size,
        config.hidden_size,
        mlp_intermediate_cores,
        mlp_output_cores,
        max_in0_block_w=down_max_block,
    )
    assert isinstance(control_gate_program_config, ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig)
    assert isinstance(control_down_program_config, ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig)
    control_down_input = ttnn.to_memory_config(down_input, control_intermediate_mem_config)

    # Candidate weights contain the same real values and BFP4 policy as the
    # final path, but use API-compatible DRAM-interleaved storage.
    layer_prefix = f"model.layers.{LAYER_IDX}."
    candidate_gate_weight = mesh_shard(
        state[f"{layer_prefix}mlp.gate_proj.weight"].transpose(0, 1).contiguous(),
        dim=1,
        dtype=ttnn.bfloat4_b,
    )
    candidate_up_weight = mesh_shard(
        state[f"{layer_prefix}mlp.up_proj.weight"].transpose(0, 1).contiguous(),
        dim=1,
        dtype=ttnn.bfloat4_b,
    )
    candidate_gate_weight = ttnn.reshape(candidate_gate_weight, [1, 1, config.hidden_size, decoder.intermediate_size])
    candidate_up_weight = ttnn.reshape(candidate_up_weight, [1, 1, config.hidden_size, decoder.intermediate_size])
    candidate_qkv_weight = ttnn.reshape(
        decoder.prefill_qkv_weight,
        [1, 1, config.hidden_size, int(decoder.prefill_qkv_weight.shape[-1])],
    )
    candidate_wo_weight = decoder.prefill_output_weight
    candidate_down_weight = decoder.prefill_down_weight

    local_input_norm = mesh_shard(state[f"{layer_prefix}input_layernorm.weight"], dim=0, dtype=ttnn.bfloat16)
    local_post_attention_norm = mesh_shard(
        state[f"{layer_prefix}post_attention_layernorm.weight"], dim=0, dtype=ttnn.bfloat16
    )

    candidate_wo_down_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 6),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=1,
        per_core_N=20,
        out_block_w=10,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    candidate_gate_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 6),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=1,
        per_core_N=32,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    candidate_qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 6),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=1,
        per_core_N=6,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    grid_size = mesh_device.compute_with_storage_grid_size()
    worker_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([worker_grid])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    subdevice_loaded = False

    def allocate_bfp8(shape):
        return _mesh_input(torch.zeros(shape, dtype=torch.bfloat16), mesh_device, dtype=ttnn.bfloat8_b)

    # Each boundary owns distinct preallocated buffers and semaphores so its
    # independently captured trace has a stable persistent-buffer contract.
    a_mrs_intermediate = allocate_bfp8((1, 1, EMITTED_BATCH, config.hidden_size))
    a_mrs_output = allocate_bfp8((1, 1, EMITTED_BATCH, config.hidden_size // TP_DEGREE))
    a_ag_output = allocate_bfp8((1, 1, EMITTED_BATCH, config.hidden_size))
    b_mrs_intermediate = allocate_bfp8((1, 1, EMITTED_BATCH, config.hidden_size))
    b_mrs_output = allocate_bfp8((1, 1, EMITTED_BATCH, config.hidden_size // TP_DEGREE))
    b_ag_output = allocate_bfp8((1, 1, EMITTED_BATCH, config.hidden_size))

    def buffer_addresses(tensor):
        return tuple(int(local.buffer_address()) for local in ttnn.get_device_tensors(tensor))

    def print_tensor_policy(label, tensor):
        local_policies = []
        for rank, local in enumerate(ttnn.get_device_tensors(tensor)):
            tile_bytes = {
                ttnn.bfloat16: 2048,
                ttnn.bfloat8_b: 1088,
                ttnn.bfloat4_b: 576,
            }[local.dtype]
            padded_volume = 1
            for extent in local.padded_shape:
                padded_volume *= int(extent)
            assert padded_volume % (ttnn.TILE_SIZE * ttnn.TILE_SIZE) == 0
            payload_bytes = padded_volume // (ttnn.TILE_SIZE * ttnn.TILE_SIZE) * tile_bytes
            local_policies.append(
                f"rank={rank}:shape={tuple(local.shape)}:padded={tuple(local.padded_shape)}:"
                f"dtype={local.dtype}:layout={local.layout}:memory={local.memory_config()}:"
                f"address={int(local.buffer_address())}:payload_bytes={payload_bytes}"
            )
        print(f"FUSED_EXACT_TENSOR_POLICY label={label} " + " | ".join(local_policies))

    for label, tensor in (
        ("control_workspace", decoder.collective_workspace),
        ("control_wo_input", control_wo_input),
        ("control_wo_weight", decoder.output_weight),
        ("control_gate_weight", decoder.gate_weight),
        ("control_up_weight", decoder.up_weight),
        ("control_down_input", control_down_input),
        ("control_down_weight", decoder.down_weight),
        ("control_qkv_weight", decoder.qkv_weight),
        ("a_fractured_residual", attention_residual_shard),
        ("b_fractured_residual", post_attention_residual_shard),
        ("candidate_wo_weight", candidate_wo_weight),
        ("candidate_gate_weight", candidate_gate_weight),
        ("candidate_up_weight", candidate_up_weight),
        ("candidate_down_weight", candidate_down_weight),
        ("candidate_qkv_weight", candidate_qkv_weight),
        ("candidate_input_norm", local_input_norm),
        ("candidate_post_attention_norm", local_post_attention_norm),
        ("a_mrs_intermediate", a_mrs_intermediate),
        ("a_mrs_output", a_mrs_output),
        ("a_ag_output", a_ag_output),
        ("b_mrs_intermediate", b_mrs_intermediate),
        ("b_mrs_output", b_mrs_output),
        ("b_ag_output", b_ag_output),
    ):
        print_tensor_policy(label, tensor)
    print(
        "FUSED_EXACT_CONTROL_POLICY "
        f"topology={decoder.collective_topology} num_links={decoder.num_links} "
        f"collective_family={decoder.collective_family} collective_dtype={decoder.collective_dtype} "
        f"workspace_identity={decoder.collective_workspace is decoder.shared_collective[0]} "
        f"workspace_addresses={buffer_addresses(decoder.collective_workspace)} "
        f"o_program={decoder.decode_o_program_config} gate_program={control_gate_program_config} "
        f"down_program={control_down_program_config} qkv_program={decoder.decode_qkv_program_config} "
        f"attention_compute={decoder.attention_compute_kernel_config} mlp_compute={decoder.mlp_compute_kernel_config}"
    )
    print(
        "FUSED_EXACT_CANDIDATE_POLICY topology=Topology.Ring num_links=1 "
        "mrs_semaphore_count=3 ag_semaphore_count=2 barrier_count=1 "
        "rs_offset=(0,6) ag_offset=(0,6) chunks_per_sync=10 num_workers_per_link=2 "
        f"num_buffers_per_channel=2 wo_down_program={candidate_wo_down_program_config} "
        f"gate_program={candidate_gate_program_config} qkv_program={candidate_qkv_program_config}"
    )

    try:
        mesh_device.load_sub_device_manager(sub_device_manager)
        subdevice_loaded = True
        mesh_device.set_sub_device_stall_group([worker_sub_device_id])
        a_mrs_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_grid, 0) for _ in range(3)]
        a_ag_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_grid, 0) for _ in range(2)]
        a_ag_barrier = ttnn.create_global_semaphore(mesh_device, worker_grid, 0)
        b_mrs_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_grid, 0) for _ in range(3)]
        b_ag_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_grid, 0) for _ in range(2)]
        b_ag_barrier = ttnn.create_global_semaphore(mesh_device, worker_grid, 0)

        def distributed_norm(hidden_shard, local_weight, compute_kernel_config):
            stats = ttnn.rms_norm_pre_all_gather(
                hidden_shard,
                compute_kernel_config=compute_kernel_config,
                dtype=ttnn.bfloat16,
            )
            stats = ttnn.all_gather(
                stats,
                dim=3,
                cluster_axis=1,
                topology=ttnn.Topology.Ring,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.rms_norm_post_all_gather(
                hidden_shard,
                stats,
                epsilon=decoder.rms_norm_eps,
                weight=local_weight,
                compute_kernel_config=compute_kernel_config,
            )

        def a_control_chain():
            attention = ttnn.matmul(
                control_wo_input,
                decoder.output_weight,
                dtype=ttnn.bfloat16,
                memory_config=decoder.decode_o_output_mem_config,
                program_config=decoder.decode_o_program_config,
                compute_kernel_config=decoder.attention_compute_kernel_config,
            )
            attention = decoder._all_reduce_hidden(attention, mode="decode")
            attention = ttnn.to_memory_config(attention, decoder.decode_norm_mem_config)
            hidden = ttnn.add(
                attention_residual,
                attention,
                dtype=ttnn.bfloat16,
                memory_config=decoder.decode_norm_mem_config,
            )
            hidden = ttnn.rms_norm(
                hidden,
                epsilon=decoder.rms_norm_eps,
                weight=decoder.post_attention_norm,
                memory_config=decoder.decode_norm_mem_config,
                program_config=decoder.decode_norm_program_config,
            )
            hidden = ttnn.to_memory_config(hidden, control_mlp_input_mem_config)
            gate = ttnn.matmul(
                hidden,
                decoder.gate_weight,
                dtype=ttnn.bfloat16,
                memory_config=control_intermediate_mem_config,
                program_config=control_gate_program_config,
                compute_kernel_config=decoder.mlp_compute_kernel_config,
            )
            up = ttnn.matmul(
                hidden,
                decoder.up_weight,
                dtype=ttnn.bfloat16,
                memory_config=control_intermediate_mem_config,
                program_config=control_gate_program_config,
                compute_kernel_config=decoder.mlp_compute_kernel_config,
            )
            return gate, up

        def a_fused_chain():
            _, hidden_shard = ttnn.experimental.matmul_reduce_scatter_async(
                wo_input,
                candidate_wo_weight,
                persistent_intermediate_buffer=a_mrs_intermediate,
                persistent_output_buffer=a_mrs_output,
                dim=3,
                multi_device_global_semaphore=a_mrs_semaphores,
                reduce_scatter_core_grid_offset=(0, 6),
                num_links=1,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=worker_sub_device_id,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                program_config=candidate_wo_down_program_config,
                compute_kernel_config=decoder.attention_compute_kernel_config,
            )
            assert buffer_addresses(hidden_shard) == buffer_addresses(a_mrs_output)
            hidden_shard = ttnn.add(
                attention_residual_shard,
                hidden_shard,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            normed_shard = distributed_norm(hidden_shard, local_post_attention_norm, decoder.mlp_compute_kernel_config)
            normed_shard = ttnn.typecast(normed_shard, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gathered, gate = ttnn.experimental.all_gather_matmul_async(
                normed_shard,
                candidate_gate_weight,
                persistent_output_buffer=a_ag_output,
                dim=3,
                multi_device_global_semaphore=a_ag_semaphores,
                all_gather_core_grid_offset=(0, 6),
                barrier_semaphore=a_ag_barrier,
                num_links=1,
                memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=worker_sub_device_id,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                program_config=candidate_gate_program_config,
                compute_kernel_config=decoder.mlp_compute_kernel_config,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            assert buffer_addresses(gathered) == buffer_addresses(a_ag_output)
            up = ttnn.matmul(
                gathered,
                candidate_up_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=candidate_gate_program_config,
                compute_kernel_config=decoder.mlp_compute_kernel_config,
            )
            return gate, up

        def b_control_chain():
            hidden = ttnn.matmul(
                control_down_input,
                decoder.down_weight,
                dtype=ttnn.bfloat16,
                memory_config=decoder.decode_mlp_output_mem_config,
                program_config=control_down_program_config,
                compute_kernel_config=decoder.mlp_compute_kernel_config,
            )
            hidden = decoder._all_reduce_hidden(hidden, mode="decode")
            hidden = ttnn.to_memory_config(hidden, decoder.decode_norm_mem_config)
            hidden = ttnn.add(
                post_attention_residual,
                hidden,
                dtype=ttnn.bfloat16,
                memory_config=decoder.decode_norm_mem_config,
            )
            hidden = ttnn.rms_norm(
                hidden,
                epsilon=decoder.rms_norm_eps,
                weight=decoder.input_norm,
                memory_config=decoder.decode_norm_mem_config,
                program_config=decoder.decode_norm_program_config,
            )
            hidden = ttnn.to_memory_config(hidden, decoder.decode_qkv_input_mem_config)
            return ttnn.matmul(
                hidden,
                decoder.qkv_weight,
                dtype=ttnn.bfloat16,
                memory_config=decoder.decode_qkv_output_mem_config,
                program_config=decoder.decode_qkv_program_config,
                compute_kernel_config=decoder.attention_compute_kernel_config,
            )

        def b_fused_chain():
            _, hidden_shard = ttnn.experimental.matmul_reduce_scatter_async(
                down_input,
                candidate_down_weight,
                persistent_intermediate_buffer=b_mrs_intermediate,
                persistent_output_buffer=b_mrs_output,
                dim=3,
                multi_device_global_semaphore=b_mrs_semaphores,
                reduce_scatter_core_grid_offset=(0, 6),
                num_links=1,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=worker_sub_device_id,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                program_config=candidate_wo_down_program_config,
                compute_kernel_config=decoder.mlp_compute_kernel_config,
            )
            assert buffer_addresses(hidden_shard) == buffer_addresses(b_mrs_output)
            hidden_shard = ttnn.add(
                post_attention_residual_shard,
                hidden_shard,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            normed_shard = distributed_norm(hidden_shard, local_input_norm, decoder.attention_compute_kernel_config)
            normed_shard = ttnn.typecast(normed_shard, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gathered, qkv = ttnn.experimental.all_gather_matmul_async(
                normed_shard,
                candidate_qkv_weight,
                persistent_output_buffer=b_ag_output,
                dim=3,
                multi_device_global_semaphore=b_ag_semaphores,
                all_gather_core_grid_offset=(0, 6),
                barrier_semaphore=b_ag_barrier,
                num_links=1,
                memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=worker_sub_device_id,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                program_config=candidate_qkv_program_config,
                compute_kernel_config=decoder.attention_compute_kernel_config,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            assert buffer_addresses(gathered) == buffer_addresses(b_ag_output)
            return qkv

        def trace_and_time(label, chain):
            trace_id = None
            capture_ended = False
            try:
                chain()
                ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                outputs = chain()
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                capture_ended = True
                for _ in range(4):
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
                start = time.perf_counter()
                for _ in range(iterations):
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
                elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
                host_outputs = tuple(
                    [ttnn.to_torch(local) for local in ttnn.get_device_tensors(output)] for output in outputs
                )
                print(f"FUSED_EXACT_TRACE label={label} elapsed_ms={elapsed_ms:.6f} iterations={iterations}")
                return elapsed_ms, host_outputs
            finally:
                if trace_id is not None:
                    if not capture_ended:
                        try:
                            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                        except Exception:
                            pass
                    try:
                        ttnn.release_trace(mesh_device, trace_id)
                    except Exception:
                        pass

        def measure_once(repeat):
            a_control_ms, a_control_host = trace_and_time(f"a_control_repeat_{repeat}", a_control_chain)
            a_fused_ms, a_fused_host = trace_and_time(f"a_fused_repeat_{repeat}", a_fused_chain)
            b_control_ms, b_control_host = trace_and_time(f"b_control_repeat_{repeat}", b_control_chain)
            b_fused_ms, b_fused_host = trace_and_time(f"b_fused_repeat_{repeat}", b_fused_chain)
            for projection, references, actuals in (
                ("gate", a_control_host[0], a_fused_host[0]),
                ("up", a_control_host[1], a_fused_host[1]),
                ("qkv", b_control_host[0], b_fused_host[0]),
            ):
                for rank, (reference, actual) in enumerate(zip(references, actuals, strict=True)):
                    _assert_pcc(
                        reference,
                        actual,
                        0.999,
                        f"fused exact {projection} repeat={repeat} rank={rank}",
                    )
            control_sum_ms = a_control_ms + b_control_ms
            fused_sum_ms = a_fused_ms + b_fused_ms
            print(
                "FUSED_EXACT_BOUNDARY_RESULT "
                f"repeat={repeat} a_control_ms={a_control_ms:.6f} a_fused_ms={a_fused_ms:.6f} "
                f"a_ratio={a_fused_ms / a_control_ms:.6f} b_control_ms={b_control_ms:.6f} "
                f"b_fused_ms={b_fused_ms:.6f} b_ratio={b_fused_ms / b_control_ms:.6f} "
                f"control_sum_ms={control_sum_ms:.6f} fused_sum_ms={fused_sum_ms:.6f} "
                f"summed_ratio={fused_sum_ms / control_sum_ms:.6f}"
            )
            return control_sum_ms, fused_sum_ms

        control_sum_ms, fused_sum_ms = measure_once(0)
        combined_followup_required = fused_sum_ms <= control_sum_ms * 1.01
        if combined_followup_required:
            for repeat in range(1, 4):
                measure_once(repeat)
        print(
            "FUSED_EXACT_DECISION "
            f"control_sum_ms={control_sum_ms:.6f} fused_sum_ms={fused_sum_ms:.6f} "
            f"summed_ratio={fused_sum_ms / control_sum_ms:.6f} "
            f"combined_followup_required={combined_followup_required}"
        )
    finally:
        if subdevice_loaded:
            mesh_device.reset_sub_device_stall_group()
            mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize("device_params", RING_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_fused_wo_matmul_reduce_scatter_tuned_blocker(mesh_device):
    """Exercise fused WO+RS with the final DRAM-sharded decode program config."""

    if os.environ.get(COLLECTIVE_CANDIDATE_ENV) != "fused_wo":
        pytest.skip(f"Set {COLLECTIVE_CANDIDATE_ENV}=fused_wo to run the tuned WO boundary")
    config = _config()
    decoder = MultichipDecoder.from_state_dict(
        _real_state(),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Ring,
        num_links=1,
    )
    generator = torch.Generator().manual_seed(2518)
    attention = ttnn.from_torch(
        torch.randn(
            (1, 1, EMITTED_BATCH, decoder.global_attention_width),
            generator=generator,
            dtype=torch.bfloat16,
        ),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    attention = ttnn.to_memory_config(attention, decoder.decode_o_input_mem_config)

    grid_size = mesh_device.compute_with_storage_grid_size()
    worker_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([worker_grid])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    semaphores = [ttnn.create_global_semaphore(mesh_device, worker_grid, 0) for _ in range(3)]
    intermediate = ttnn.to_memory_config(
        _mesh_input(
            torch.zeros((1, 1, EMITTED_BATCH, config.hidden_size), dtype=torch.bfloat16),
            mesh_device,
        ),
        decoder.decode_o_output_mem_config,
    )
    output = _mesh_input(
        torch.zeros((1, 1, EMITTED_BATCH, config.hidden_size // TP_DEGREE), dtype=torch.bfloat16),
        mesh_device,
    )
    ttnn.experimental.matmul_reduce_scatter_async(
        attention,
        decoder.output_weight,
        persistent_intermediate_buffer=intermediate,
        persistent_output_buffer=output,
        dim=3,
        multi_device_global_semaphore=semaphores,
        reduce_scatter_core_grid_offset=(0, 6),
        num_links=1,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        subdevice_id=worker_sub_device_id,
        memory_config_mm=decoder.decode_o_output_mem_config,
        program_config=decoder.decode_o_program_config,
        compute_kernel_config=decoder.attention_compute_kernel_config,
    )
