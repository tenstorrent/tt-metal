# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration test for `models.common.modules.moe.tt_moe_decode.TTMoEDecode`.

Setup mirrors `test_optimized_moe_decode_block.py`: build torch weights / inputs,
push them through the TTMoEDecode module, and verify the final output against a
torch reference. Combine output verification is intentionally skipped — that
intermediate buffer is exercised by the optimized-block test directly.

Parametrized over every YAML model config in `models/common/modules/moe/configs/`.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.moe.tt_moe_decode import TTMoEDecode
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig
from models.common.utility_functions import comp_allclose, comp_pcc

# ---------------------------------------------------------------------------
# torch reference helpers (mirrors `test_optimized_moe_decode_block.py`)
# ---------------------------------------------------------------------------


def _matmul_golden(token: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """SiLU SwiGLU MoE expert reference (num_layers=1 throughout)."""
    silu = torch.nn.functional.silu(token @ w0)
    gate = token @ w1
    return (silu * gate) @ w2


def _create_per_expert_weights(num_layers: int, num_experts: int, h: int, n: int) -> torch.Tensor:
    """Returns a [num_layers, num_experts, *, *] tensor of expert weights."""
    return torch.cat(
        [torch.rand((num_layers, 1, h, n), dtype=torch.bfloat16) - 0.5 for _ in range(num_experts)],
        dim=1,
    )


def _create_dispatch_input(batch: int, hidden_size: int) -> torch.Tensor:
    """[batch, 1, 1, hidden_size] — seq=1 for decode."""
    return torch.rand((batch, 1, 1, hidden_size), dtype=torch.bfloat16) - 0.5


def _create_expert_indices(batch: int, num_experts: int, select_k: int) -> torch.Tensor:
    """[batch, 1, 1, select_k] — random unique experts per token."""
    out = torch.full((batch, 1, 1, select_k), -1, dtype=torch.int32)
    for b in range(batch):
        choices = random.sample(range(num_experts), select_k)
        for k, e in enumerate(choices):
            out[b, 0, 0, k] = e
    return out.to(torch.int32)


def _create_expert_scores(batch: int, select_k: int) -> torch.Tensor:
    """[batch, 1, 1, select_k] — softmax-normalized random scores."""
    s = torch.rand((batch, 1, 1, select_k), dtype=torch.bfloat16)
    return s / s.sum(dim=-1, keepdim=True)


def _gen_output_golden(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_scores: torch.Tensor,
    w0_per_expert: list[torch.Tensor],
    w1_per_expert: list[torch.Tensor],
    w2_per_expert: list[torch.Tensor],
    batch: int,
    hidden_size: int,
    select_k: int,
) -> torch.Tensor:
    """[batch, 1, 1, hidden_size] — sum_k(score_k * matmul(token, expert_k))."""
    out = torch.zeros((batch, 1, 1, hidden_size), dtype=torch.bfloat16)
    for t in range(batch):
        for k in range(select_k):
            e = expert_indices[t, 0, 0, k].item()
            contrib = _matmul_golden(tokens[t], w0_per_expert[e], w1_per_expert[e], w2_per_expert[e])
            out[t] = out[t] + expert_scores[t, 0, 0, k] * contrib
    return out


def _linearized_expert_to_device(
    expert_id: int,
    cluster_axis: int,
    num_replicated_devices: int,
    experts_per_cluster: int,
    experts_per_device: int,
) -> int:
    """Returns the linear device id (in mesh row-major order) hosting `expert_id`."""
    if cluster_axis == 0:
        cluster_id = expert_id // experts_per_cluster
        device_within_cluster = (expert_id % experts_per_cluster) // experts_per_device
        return device_within_cluster * num_replicated_devices + cluster_id
    return expert_id // experts_per_device


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def _verify_output(
    iteration: int,
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    tt_output: ttnn.Tensor,
    output_reference: torch.Tensor,
    pcc_threshold: float = 0.988,
    atol_threshold: float = 450.0,
) -> bool:
    tt_torch = ttnn.to_torch(
        tt_output,
        dtype=torch.bfloat16,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(-2, -1)),
    )
    # [1, 1, batch, hidden_size] -> [batch, 1, 1, hidden_size]
    tt_torch = tt_torch.reshape(tt_torch.shape[-2], 1, 1, tt_torch.shape[-1])

    pcc_passed, pcc_str = comp_pcc(tt_torch, output_reference, pcc=pcc_threshold)
    logger.info(f"Final Output - Iteration {iteration} - PCC: {pcc_str}")
    if not pcc_passed:
        logger.warning(f"FAILED PCC - iteration {iteration}: {pcc_str}")

    allclose_passed, allclose_str = comp_allclose(output_reference, tt_torch, atol=atol_threshold, rtol=0)
    logger.info(f"Final Output - Iteration {iteration} - AllClose: {allclose_str}")
    if not allclose_passed:
        logger.warning(f"FAILED AllClose - iteration {iteration}: {allclose_str}")

    return pcc_passed and allclose_passed


CONFIGS_DIR = Path(__file__).resolve().parents[3] / "modules" / "moe" / "configs"
CONFIG_PATHS = sorted(CONFIGS_DIR.glob("*.yaml"))


def _config_id(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD"])
@pytest.mark.skipif(os.getenv("USE_TORUS_MODE") is None, reason="Requires ring fabric")
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((1, 16), (1, 16), id="1x16"),
        pytest.param((1, 8), (1, 8), id="1x8"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500_000,
        }
    ],
    ids=["fabric_1D_ring"],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [3])
@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=_config_id)
@torch.no_grad()
def test_tt_moe_decode(
    mesh_shape: tuple[int, int],
    mesh_device: ttnn.MeshDevice,
    config_path: Path,
    num_iterations: int,
):
    torch.manual_seed(2005)
    random.seed(2005)

    config = TTMoEDecodeConfig.from_yaml(config_path.read_text())
    if config.mesh_shape != mesh_shape:
        pytest.skip(f"config mesh_shape {config.mesh_shape} != fixture mesh_shape {mesh_shape}")

    # --- derived sizes (all from config) ---
    cluster_axis = config.cluster_axis
    routed_experts = config.num_routed_experts
    hidden_size = config.hidden_size
    intermediate_size = config.compute.intermediate_size
    select_experts_k = config.select_experts_k
    batches_per_device = config.batch_per_device

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * num_dispatch_devices

    shard_dim = 0
    shard_dims = (shard_dim, None) if cluster_axis == 0 else (None, shard_dim)

    logger.info(
        f"Setup [{config_path.stem}]: mesh_shape={mesh_shape} cluster_axis={cluster_axis} "
        f"num_devices={num_devices} batch={batch} hidden={hidden_size} N={intermediate_size} "
        f"routed_experts={routed_experts} select_experts_k={select_experts_k} "
        f"has_bias={config.has_bias} activation={config.compute.activation_type.name}"
    )

    # --- weights: [num_layers=1, routed_experts, H/N, N/H] ---
    num_layers = 1
    torch_w0 = _create_per_expert_weights(num_layers, routed_experts, hidden_size, intermediate_size)
    torch_w1 = _create_per_expert_weights(num_layers, routed_experts, hidden_size, intermediate_size)
    torch_w2 = _create_per_expert_weights(num_layers, routed_experts, intermediate_size, hidden_size)
    w0_per_expert = [torch_w0[:, e : e + 1, ...] for e in range(routed_experts)]
    w1_per_expert = [torch_w1[:, e : e + 1, ...] for e in range(routed_experts)]
    w2_per_expert = [torch_w2[:, e : e + 1, ...] for e in range(routed_experts)]

    # --- build module ---
    decode = TTMoEDecode(
        mesh_device=mesh_device,
        config=config,
        torch_w0=torch_w0,
        torch_w1=torch_w1,
        torch_w2=torch_w2,
    )

    # --- per-iteration inputs + goldens ---
    tt_dispatch_inputs = []
    tt_dispatch_indices = []
    tt_dispatch_scores = []
    output_goldens = []
    for _ in range(num_iterations):
        tokens = _create_dispatch_input(batch, hidden_size)
        indices = _create_expert_indices(batch, routed_experts, select_experts_k)
        scores = _create_expert_scores(batch, select_experts_k)

        tt_dispatch_inputs.append(
            ttnn.from_torch(
                tokens,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
            )
        )
        tt_dispatch_indices.append(
            ttnn.from_torch(
                indices,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
            )
        )
        tt_dispatch_scores.append(
            ttnn.from_torch(
                scores,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
            )
        )

        output_goldens.append(
            _gen_output_golden(
                tokens,
                indices,
                scores,
                w0_per_expert,
                w1_per_expert,
                w2_per_expert,
                batch,
                hidden_size,
                select_experts_k,
            )
        )

    # --- run + collect outputs ---
    logger.info("Running forward iterations")
    tt_outputs = []
    for it in range(num_iterations):
        tt_outputs.append(
            decode.forward(
                tt_x=tt_dispatch_inputs[it],
                tt_scores=tt_dispatch_scores[it],
                tt_indices=tt_dispatch_indices[it],
                layer_id=0,
            )
        )

    ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])

    # --- verify ---
    logger.info("Verifying outputs")
    all_passed = True
    for it in range(num_iterations):
        if not _verify_output(it, mesh_device, mesh_shape, tt_outputs[it], output_goldens[it]):
            all_passed = False

    assert all_passed, f"TTMoEDecode output verification failed for {config_path.stem}"
