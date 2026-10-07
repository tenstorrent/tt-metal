# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prefill-shape probes for dit_fused_distributed_rmsnorm.

These tests are intentionally non-gating: they log correctness and latency for
agents/engineers optimizing the op, but they do not assert thresholds.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


_TP_AXIS = 1
_NUM_LINKS = 1
_TOPOLOGY = ttnn.Topology.Linear
_WARMUP_ITERS = 3
_MEASURE_ITERS = 10
_EPS = 1e-5


def _torch_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x.float(), normalized_shape=(x.shape[-1],), weight=weight.float(), eps=_EPS)


def _make_sharded_input(mesh_device: ttnn.MeshDevice, torch_input: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 3)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_sharded_weight(mesh_device: ttnn.MeshDevice, torch_weight: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch_weight.reshape(1, 1, 1, -1),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 3)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch(mesh_device: ttnn.MeshDevice, tt_output: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 3)),
    ).float()


def _run_once(
    mesh_device: ttnn.MeshDevice,
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    semaphores,
    stats_buffer: ttnn.Tensor,
) -> ttnn.Tensor:
    return ttnn.experimental.dit_fused_distributed_rmsnorm(
        x,
        _TP_AXIS,
        mesh_device,
        semaphores,
        topology=_TOPOLOGY,
        epsilon=_EPS,
        weight=weight,
        persistent_output_buffer=stats_buffer,
        num_preferred_links=_NUM_LINKS,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
@pytest.mark.parametrize(
    "case_name,seq_len,hidden_dim",
    [
        pytest.param("kimi_k3_latent_moe_h3584", 640, 3584, id="kimi-k3-latent-moe-h3584"),
        pytest.param("deepseek_v4_flash_h4096", 640, 4096, id="deepseek-v4-flash-h4096"),
        pytest.param("glm_5_3_h6144", 640, 6144, id="glm-5-3-h6144"),
        pytest.param("kimi_k2_7_h7168", 640, 7168, id="kimi-k2-7-h7168"),
    ],
)
@pytest.mark.parametrize("seed", [0], ids=["seed0"])
def test_fused_rms_norm_prefill(mesh_device, case_name, seq_len, hidden_dim, seed):
    """Run correctness + perf logging for prefill RMSNorm shapes on a TP=4 mesh.

    The tensor is sharded on hidden only. With a (1, 4) mesh the device-local
    shapes are therefore [1, 1, seq_len, hidden_dim / 4], matching the RMSNorm
    kernel shapes seen by TP=4 prefill runs after sequence sharding has already
    selected a local row count.
    """
    torch.manual_seed(seed)
    torch_input = torch.randn(1, 1, seq_len, hidden_dim, dtype=torch.bfloat16)
    torch_weight = (torch.randn(hidden_dim, dtype=torch.bfloat16) * 0.2 + 1.0).to(torch.bfloat16)
    torch_reference = _torch_rms_norm(torch_input, torch_weight)

    x = _make_sharded_input(mesh_device, torch_input)
    weight = _make_sharded_weight(mesh_device, torch_weight)

    ccl = CCLManager(mesh_device=mesh_device, num_links=_NUM_LINKS, topology=_TOPOLOGY)
    semaphores = [ccl.get_ag_ping_pong_semaphore(_TP_AXIS) for _ in range(2)]
    stats_buffers = [
        ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
            x,
            _TP_AXIS,
            mesh_device,
            num_links=_NUM_LINKS,
            weight=weight,
        )
        for _ in range(2)
    ]
    ttnn.synchronize_device(mesh_device)

    output = None
    for i in range(_WARMUP_ITERS):
        output = _run_once(mesh_device, x, weight, semaphores[i % 2], stats_buffers[i % 2])
    ttnn.synchronize_device(mesh_device)

    start = time.perf_counter()
    for i in range(_MEASURE_ITERS):
        output = _run_once(mesh_device, x, weight, semaphores[i % 2], stats_buffers[i % 2])
    ttnn.synchronize_device(mesh_device)
    elapsed_s = time.perf_counter() - start

    tt_output = _to_torch(mesh_device, output)
    _, pcc_msg = comp_pcc(torch_reference, tt_output)
    max_abs = torch.max(torch.abs(torch_reference.float() - tt_output)).item()
    mean_abs = torch.mean(torch.abs(torch_reference.float() - tt_output)).item()
    avg_ms = elapsed_s * 1000.0 / _MEASURE_ITERS

    logger.info(
        "dit_fused_distributed_rmsnorm prefill probe "
        f"case={case_name} global_shape={(1, 1, seq_len, hidden_dim)} "
        f"local_shape={(1, 1, seq_len, hidden_dim // 4)} tp=4 "
        f"avg_ms={avg_ms:.3f} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} {pcc_msg}"
    )
