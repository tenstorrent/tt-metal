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
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.optimized_decoder import OptimizedDecoder

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

MESH_PARAMS = [(1, 4)]
DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 100000000}]


def _mesh_input(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


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


def _single_caches(config, mesh_device, *, dtype=ttnn.bfloat8_b):
    shape = (EMITTED_BATCH, config.num_key_value_heads, EMITTED_CACHE_LENGTH, config.head_dim)
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
    assert "ttnn.all_reduce" in inspect.getsource(MultichipDecoder._all_reduce_hidden)
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
    state = _real_state() if os.environ.get(BASELINE_REAL_ENV) == "1" else _synthetic_state(config)
    generator = torch.Generator().manual_seed(2512)
    prefill_hidden = torch.randn((1, EMITTED_BATCH, 17, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    decode_hidden = torch.randn((1, EMITTED_BATCH, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)

    if implementation == "optimized":
        decoder = OptimizedDecoder.from_state_dict(
            state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device
        )
        key_cache, value_cache = _single_caches(config, mesh_device)
        prefill = decoder.prefill_forward(_single_input(prefill_hidden, mesh_device), key_cache, value_cache)
        decode = decoder.decode_forward(
            _single_input(decode_hidden, mesh_device), key_cache, value_cache, current_pos=17
        )
        ttnn.synchronize_device(mesh_device)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prefill": ttnn.to_torch(prefill),
                "decode": ttnn.to_torch(decode),
                "key": ttnn.to_torch(key_cache)[:, :, :18, :],
                "value": ttnn.to_torch(value_cache)[:, :, :18, :],
            },
            artifact_path,
        )
        print(f"BASELINE_ARTIFACT wrote={artifact_path}")
        return

    assert artifact_path.is_file(), f"Run the optimized artifact command first: {artifact_path}"
    baseline = torch.load(artifact_path, weights_only=True)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device)
    key_cache, value_cache = _multichip_caches(config, mesh_device)
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


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
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
    decoder.release_prefill_weights()
    assert decoder.prefill_weights_released
    assert all(
        getattr(decoder, name) is None
        for name in (
            "prefill_qkv_weight",
            "prefill_output_weight",
            "prefill_gate_up_weight",
            "prefill_down_weight",
        )
    )
    shared_decoder = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=max_cache_len,
        shared_rope=decoder.shared_rope,
    )
    assert shared_decoder.rotary_cos is decoder.rotary_cos
    assert shared_decoder.decode_rotary_cos is decoder.decode_rotary_cos
    shared_decoder.release_prefill_weights()

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
    stack_constants = []
    for _ in range(38):
        for shape, dtype, memory_config in decode_weight_shapes:
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

    # Physically reserve 4 GiB/rank for runtime activations, traces, programs,
    # and collective scratch after all steady-state constants and caches exist.
    runtime_reserve = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([8192, 32768]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(8)
    ]
    hidden = torch.zeros((1, EMITTED_BATCH, 1, config.hidden_size), dtype=torch.bfloat16)
    current_pos = max_cache_len - 1
    output = decoder.decode_forward(
        _mesh_input(hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos_tensor=current_positions,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    assert tuple(_replicated_host(output).shape) == tuple(hidden.shape)
    print(
        f"MULTICHIP_CAPACITY_PASS layers=40 batch={EMITTED_BATCH} max_cache_len={max_cache_len} "
        f"local_cache_shape={tuple(local_shape)} current_pos={current_pos} "
        f"runtime_reserve_bytes={8 * 8192 * 32768 * 2}"
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
