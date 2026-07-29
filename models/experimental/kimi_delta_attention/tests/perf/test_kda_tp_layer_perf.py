# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler harness for target-shape TP8 and 2D SP KDA layouts."""

import os
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.experimental.kimi_delta_attention.checkpoint import load_kda_layer_state_dict
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.kimi_k3_config import (
    KimiK3Config,
    kimi_k3_kda_config,
    kimi_k3_program_config,
)
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.tt_transformers.tt.ccl import TT_CCL

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.parametrize(
        "device_params",
        [
            {
                "l1_small_size": 24576,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 256 * 1024 * 1024,
            }
        ],
        indirect=True,
    ),
]


def _profile_eager(
    mesh_device: ttnn.MeshDevice,
    layer: KimiDeltaAttention,
    hidden: ttnn.Tensor,
    repetitions: int,
) -> float:
    outputs: list[ttnn.Tensor] = []
    signpost(header="start")
    start = time.perf_counter()
    for _ in range(repetitions):
        outputs.append(layer.forward(hidden))
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    signpost(header="stop")
    for output in outputs:
        ttnn.deallocate(output)
    return elapsed / repetitions


def _profile_trace(
    mesh_device: ttnn.MeshDevice,
    layer: KimiDeltaAttention,
    hidden: ttnn.Tensor,
    repetitions: int,
) -> float:
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = layer.forward(hidden)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    signpost(header="start")
    start = time.perf_counter()
    for _ in range(repetitions):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    signpost(header="stop")

    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(output)
    return elapsed / repetitions


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["TP8", "SP2xTP4", "SP4xTP2"],
)
def test_kda_tp_layer_device_perf(mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int) -> None:
    """Profile ten warm target-shape trace replays for TP8 and both 2D SP layouts."""
    sequence = int(os.getenv("PERF_SEQ", "5120"))
    if sequence % 32:
        raise ValueError(f"PERF_SEQ must be divisible by 32, got {sequence}")
    config = KDAConfig(
        hidden_size=2304,
        num_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(1607)).to(
        torch.bfloat16
    )
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=tuple(1 if axis == 1 - tensor_parallel_axis else None for axis in range(2)),
            mesh_shape=tuple(mesh_device.shape),
        ),
    )
    layer = KimiDeltaAttention(
        mesh_device,
        config,
        random_weights(config),
        tt_ccl=TT_CCL(mesh_device),
        tensor_parallel_axis=tensor_parallel_axis,
    )
    layer.reset_state(batch_size=1)
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    layer.set_external_state(layer.recurrent_state, layer.convolution_state)

    warm_output = layer.forward(hidden_tt)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_output)

    repetitions = int(os.getenv("PERF_REPS", "10"))
    if os.getenv("PERF_TRACE", "0") == "1":
        wall_seconds = _profile_trace(mesh_device, layer, hidden_tt, repetitions)
    else:
        wall_seconds = _profile_eager(mesh_device, layer, hidden_tt, repetitions)
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    tp_size = tuple(mesh_device.shape)[tensor_parallel_axis]
    print(
        f"KDA SP{sp_size}xTP{tp_size} B=1 T={sequence}: "
        f"wall={wall_seconds * 1e3:.3f} ms/replay over {repetitions} replays"
    )


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0), ((1, 8), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2", "SP8xTP1"],
)
@pytest.mark.parametrize("weight_source", ["random", "real"])
def test_kimi_k3_layer_1_device_perf(
    mesh_device: ttnn.MeshDevice, tensor_parallel_axis: int, weight_source: str
) -> None:
    """Compare K3 device time across SP/TP layouts with random and layer-1 weights."""
    checkpoint_value = os.getenv("KIMI_K3_CKPT")
    if checkpoint_value is None:
        pytest.skip("set KIMI_K3_CKPT to the pinned Kimi-K3 checkpoint subset")
    checkpoint_dir = Path(checkpoint_value)
    config = kimi_k3_kda_config()
    if weight_source == "real":
        state_dict = load_kda_layer_state_dict(checkpoint_dir, KimiK3Config.FIRST_KDA_LAYER, config)
        tensor_cache_path = checkpoint_dir / "ttnn_cache" / "layer_1"
        tensor_cache_path.mkdir(parents=True, exist_ok=True)
    else:
        state_dict = random_weights(config)
        tensor_cache_path = None

    sequence = int(os.getenv("KIMI_K3_PERF_SEQ", "672"))
    if sequence % 32:
        raise ValueError(f"KIMI_K3_PERF_SEQ must be divisible by 32, got {sequence}")
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(1607)).to(
        torch.bfloat16
    )
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=tuple(1 if axis == 1 - tensor_parallel_axis else None for axis in range(2)),
            mesh_shape=tuple(mesh_device.shape),
        ),
    )
    layer = KimiDeltaAttention(
        mesh_device,
        config,
        state_dict,
        tensor_cache_path=tensor_cache_path,
        tt_ccl=TT_CCL(mesh_device),
        tensor_parallel_axis=tensor_parallel_axis,
        program_config=kimi_k3_program_config(),
    )
    layer.reset_state(batch_size=1)
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    layer.set_external_state(layer.recurrent_state, layer.convolution_state)

    warm_output = layer.forward(hidden_tt)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_output)

    repetitions = int(os.getenv("PERF_REPS", "10"))
    if os.getenv("PERF_TRACE", "0") == "1":
        wall_seconds = _profile_trace(mesh_device, layer, hidden_tt, repetitions)
    else:
        wall_seconds = _profile_eager(mesh_device, layer, hidden_tt, repetitions)
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    tp_size = tuple(mesh_device.shape)[tensor_parallel_axis]
    print(
        f"Kimi-K3 KDA layer 1 weights={weight_source} SP{sp_size}xTP{tp_size} B=1 T={sequence}: "
        f"wall={wall_seconds * 1e3:.3f} ms/replay over {repetitions} replays"
    )
