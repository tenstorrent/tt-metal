# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Explicitly-invoked optimization sweeps; not part of default pytest discovery."""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _page_table,
    _positions,
    _synthetic_state,
    _to_tt_decode,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder

GEOMETRIES = {
    "advisor": {},
    "qkv_wide_block": {"qkv": (12, 3)},
    "o_fewer_cores": {"o_proj": (12, 2)},
    "o_32_cores": {"o_proj": (12, 3)},
    "gate_wide_block": {"gate_up": (12, 5)},
    "gate_block_8": {"gate_up": (8, 5)},
    "gate_64_cores": {"gate_up": (6, 8)},
    "down_wide_block": {"down": (32, 1)},
    "down_48_cores": {"down": (16, 2)},
    "down_32_cores": {"down": (16, 3)},
}

ADAPTED_GATE_CANDIDATES = {
    "six_core_block_8": {"decode_config_overrides": {"gate_up": (8, 5)}},
    "six_core_block_16": {"decode_config_overrides": {"gate_up": (16, 5)}},
}

TOPOLOGY_CANDIDATES = {
    "bf16_cache_packed": {"kv_cache_dtype": ttnn.bfloat16},
    "bfp8_cache_packed": {"kv_cache_dtype": ttnn.bfloat8_b},
    "bfp8_cache_split": {"kv_cache_dtype": ttnn.bfloat8_b, "split_decode_projections": True},
}

POLICIES = {
    "all_bfp4_lofi": {},
    "qkv_bfp4_hifi2": {"fidelity_overrides": {"qkv": ttnn.MathFidelity.HiFi2}},
    "o_bfp4_hifi2": {"fidelity_overrides": {"o_proj": ttnn.MathFidelity.HiFi2}},
    "gate_bfp4_hifi2": {"fidelity_overrides": {"gate_up": ttnn.MathFidelity.HiFi2}},
    "down_bfp4_hifi2": {"fidelity_overrides": {"down": ttnn.MathFidelity.HiFi2}},
    "all_bfp4_hifi2": {
        "fidelity_overrides": {
            "qkv": ttnn.MathFidelity.HiFi2,
            "o_proj": ttnn.MathFidelity.HiFi2,
            "gate_up": ttnn.MathFidelity.HiFi2,
            "down": ttnn.MathFidelity.HiFi2,
        }
    },
}


def _measure_trace(decoder, hidden, page_table, positions, key_cache, value_cache):
    def decode():
        return decoder.decode_forward(
            hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=positions,
            use_long_rope=False,
        )

    decode()
    ttnn.synchronize_device(decoder.mesh_device)
    trace_id = ttnn.begin_trace_capture(decoder.mesh_device, cq_id=0)
    output = decode()
    ttnn.end_trace_capture(decoder.mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(decoder.mesh_device)
    try:
        ttnn.execute_trace(decoder.mesh_device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            ttnn.execute_trace(decoder.mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(decoder.mesh_device)
            samples.append(1000 * (time.perf_counter() - start))
    finally:
        ttnn.release_trace(decoder.mesh_device, trace_id)
    return output, samples


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("candidate", GEOMETRIES)
def test_geometry_sweep(mesh_device, batch, candidate):
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
        decode_config_overrides=GEOMETRIES[candidate],
    )
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(700 + batch)).to(
        torch.bfloat16
    )
    hidden = _to_tt_decode(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    positions = _positions([0] * batch, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()

    output, samples = _measure_trace(decoder, hidden, page_table, positions, key_cache, value_cache)
    assert tuple(output.shape) == (1, 1, batch, config.hidden_size)
    print(
        f"SWEEP_RESULT axis=geometry candidate={candidate} batch={batch} "
        f"mean_ms={sum(samples)/len(samples):.6f} min_ms={min(samples):.6f} "
        f"configs={decoder.decode_config_values}"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("candidate", ADAPTED_GATE_CANDIDATES)
def test_adapted_gate_shard_sweep(mesh_device, batch, candidate):
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
        decode_input_core_overrides={"gate_up": 6},
        **ADAPTED_GATE_CANDIDATES[candidate],
    )
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(1300 + batch)).to(
        torch.bfloat16
    )
    try:
        output, samples = _measure_trace(
            decoder,
            _to_tt_decode(hidden, mesh_device),
            _page_table(batch, 128, mesh_device, permute=True),
            _positions([0] * batch, mesh_device),
            *decoder.create_paged_kv_cache(),
        )
    except RuntimeError as error:
        if candidate != "six_core_block_16":
            raise
        assert "beyond max L1 size" in str(error)
        print(
            f"SWEEP_REJECT axis=adapted_gate candidate={candidate} batch={batch} "
            "reason=static_circular_buffers_2077440_exceed_L1_1572864"
        )
        return
    assert candidate != "six_core_block_16", "expected the block-16 L1 rejection"
    assert tuple(output.shape) == (1, 1, batch, config.hidden_size)
    print(
        f"SWEEP_RESULT axis=adapted_gate candidate={candidate} batch={batch} "
        f"mean_ms={sum(samples)/len(samples):.6f} min_ms={min(samples):.6f} "
        f"input_cores=6 configs={decoder.decode_config_values}"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("candidate", POLICIES)
def test_policy_sweep(mesh_device, batch, candidate):
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
        **POLICIES[candidate],
    )
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(900 + batch)).to(
        torch.bfloat16
    )
    hidden = _to_tt_decode(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    positions = _positions([0] * batch, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    output, samples = _measure_trace(decoder, hidden, page_table, positions, key_cache, value_cache)
    assert tuple(output.shape) == (1, 1, batch, config.hidden_size)
    print(
        f"SWEEP_RESULT axis=policy candidate={candidate} batch={batch} "
        f"mean_ms={sum(samples)/len(samples):.6f} min_ms={min(samples):.6f} "
        f"dtypes={decoder.selected_decode_weight_dtypes} fidelities={decoder.decode_fidelities}"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("candidate", TOPOLOGY_CANDIDATES)
def test_topology_sweep(mesh_device, batch, candidate):
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
        **TOPOLOGY_CANDIDATES[candidate],
    )
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(1200 + batch)).to(
        torch.bfloat16
    )
    output, samples = _measure_trace(
        decoder,
        _to_tt_decode(hidden, mesh_device),
        _page_table(batch, 128, mesh_device, permute=True),
        _positions([0] * batch, mesh_device),
        *decoder.create_paged_kv_cache(),
    )
    assert tuple(output.shape) == (1, 1, batch, config.hidden_size)
    print(
        f"SWEEP_RESULT axis=topology candidate={candidate} batch={batch} "
        f"mean_ms={sum(samples)/len(samples):.6f} min_ms={min(samples):.6f} "
        f"cache={decoder.kv_cache_dtype} split={decoder.split_decode_projections}"
    )
