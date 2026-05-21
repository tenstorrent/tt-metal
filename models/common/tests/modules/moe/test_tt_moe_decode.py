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

import faulthandler
import random
import sys
import traceback
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.common.modules.moe.tt_moe_decode import TTMoEDecode
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig
from models.demos.deepseek_v3.tests.fused_op_unit_tests.moe.test_optimized_moe_decode_block import (
    create_torch_dispatch_input_expert_scores_tensor,
    create_torch_dispatch_input_tensor,
    verify_output,
)
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import _swiglu_reference

faulthandler.enable()


@pytest.fixture(autouse=True)
def _hang_watchdog():
    # If a test wedges (typically on device teardown after an async failure),
    # dump tracebacks and SIGABRT the process instead of hanging indefinitely.
    faulthandler.dump_traceback_later(300, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _print_exception_and_fail(reason: str) -> None:
    """Print the active exception to the original (uncaptured) stderr and pytest.fail.

    pytest's stderr capture buffers `logger.exception()` output and only flushes it
    after the test exits. When the watchdog `_exit()`s the process (e.g. because
    a ttnn tensor `__repr__` was waiting on a wedged device), that buffer is lost.
    Writing to `sys.__stderr__` and flushing bypasses capture so the trace survives.
    Then pytest.fail(pytrace=False) skips pytest's own saferepr-driven traceback
    rendering, which is itself prone to deadlocking on hung device tensors.
    """
    traceback.print_exc(file=sys.__stderr__)
    sys.__stderr__.flush()
    pytest.fail(reason, pytrace=False)


# ---------------------------------------------------------------------------
# torch reference helpers
#
# `_swiglu_reference`, `create_torch_dispatch_input_tensor`,
# `create_torch_dispatch_input_expert_scores_tensor`, and `verify_output` are
# imported above from the existing MoE tests — same logic, no need to duplicate.
# Helpers that diverge (per-expert weight/bias init, activation/bias-aware
# matmul, output-golden assembly) are defined locally.
# ---------------------------------------------------------------------------


def _matmul_golden(
    token: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation_type: MoEActivationFunction = MoEActivationFunction.SILU,
    b0: torch.Tensor | None = None,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> torch.Tensor:
    """MoE expert reference (num_layers=1 throughout).

    SILU:   `silu(x @ w0 + b0) * (x @ w1 + b1) @ w2 + b2`
    SWIGLU: `(up + 1) * gate * sigmoid(alpha * gate) @ w2 + b2` with clamping (GPT-OSS),
            where `gate = x @ w0 + b0` and `up = x @ w1 + b1`.

    Per-expert bias shapes: `b0`/`b1` are `[num_layers, 1, N]`, `b2` is
    `[num_layers, 1, hidden_size]`. `unsqueeze(-2)` broadcasts over the token dim.
    """
    gate = token @ w0
    if b0 is not None:
        gate = gate + b0.unsqueeze(-2)
    up = token @ w1
    if b1 is not None:
        up = up + b1.unsqueeze(-2)

    if activation_type == MoEActivationFunction.SILU:
        intermediate = torch.nn.functional.silu(gate) * up
    elif activation_type == MoEActivationFunction.SWIGLU:
        intermediate = _swiglu_reference(gate, up)
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")

    output = intermediate @ w2
    if b2 is not None:
        output = output + b2.unsqueeze(-2)
    return output


def _create_per_expert_weights(num_layers: int, num_experts: int, h: int, n: int) -> torch.Tensor:
    """Returns a [num_layers, num_experts, *, *] tensor of expert weights."""
    return torch.rand((num_layers, num_experts, h, n), dtype=torch.bfloat16) - 0.5


def _create_per_expert_biases(num_layers: int, num_experts: int, dim: int) -> torch.Tensor:
    """Returns a [num_layers, num_experts, dim] tensor of expert biases.

    Matches the bias init in `test_moe_compute_6U.py` (std=0.12 then cast to bfloat16).
    """
    _bias_std = 0.12
    return (torch.randn(num_layers, num_experts, dim, dtype=torch.float32) * _bias_std).to(torch.bfloat16)


def _create_expert_indices(batch: int, num_experts: int, select_k: int) -> torch.Tensor:
    """[batch, 1, 1, select_k] — random unique experts per token."""
    out = torch.full((batch, 1, 1, select_k), -1, dtype=torch.int32)
    for b in range(batch):
        for k, e in enumerate(random.sample(range(num_experts), select_k)):
            out[b, 0, 0, k] = e
    return out


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
    activation_type: MoEActivationFunction = MoEActivationFunction.SILU,
    b0_per_expert: list[torch.Tensor] | None = None,
    b1_per_expert: list[torch.Tensor] | None = None,
    b2_per_expert: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """[batch, 1, 1, hidden_size] — sum_k(score_k * matmul(token, expert_k))."""
    out = torch.zeros((batch, 1, 1, hidden_size), dtype=torch.bfloat16)
    for t in range(batch):
        for k in range(select_k):
            e = expert_indices[t, 0, 0, k].item()
            contrib = _matmul_golden(
                tokens[t],
                w0_per_expert[e],
                w1_per_expert[e],
                w2_per_expert[e],
                activation_type,
                b0=b0_per_expert[e] if b0_per_expert is not None else None,
                b1=b1_per_expert[e] if b1_per_expert is not None else None,
                b2=b2_per_expert[e] if b2_per_expert is not None else None,
            )
            out[t] = out[t] + expert_scores[t, 0, 0, k] * contrib
    return out


CONFIGS_DIR = Path(__file__).resolve().parents[3] / "modules" / "moe" / "configs"
CONFIG_PATHS = filter(lambda x: "gpt_oss" in str(x), sorted(CONFIGS_DIR.glob("*.yaml")))
assert CONFIG_PATHS, f"no YAML configs found in {CONFIGS_DIR}"


def _config_id(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device",
    [
        # pytest.param((16, 4), id="16x4"),
        pytest.param((8, 4), id="8x4"),
    ],
    indirect=True,
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
    mesh_device: ttnn.MeshDevice,
    config_path: Path,
    num_iterations: int,
):
    torch.manual_seed(2005)
    random.seed(2005)

    mesh_shape = tuple(mesh_device.shape)
    config = TTMoEDecodeConfig.from_yaml(config_path.read_text())
    if config.mesh_shape != mesh_shape:
        pytest.skip(f"config mesh_shape {config.mesh_shape} != device mesh_shape {mesh_shape}")

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

    # --- biases (optional): [num_layers=1, routed_experts, N or hidden_size] ---
    torch_b0 = torch_b1 = torch_b2 = None
    b0_per_expert = b1_per_expert = b2_per_expert = None
    if config.has_bias:
        torch_b0 = _create_per_expert_biases(num_layers, routed_experts, intermediate_size)
        torch_b1 = _create_per_expert_biases(num_layers, routed_experts, intermediate_size)
        torch_b2 = _create_per_expert_biases(num_layers, routed_experts, hidden_size)
        b0_per_expert = [torch_b0[:, e : e + 1, :] for e in range(routed_experts)]
        b1_per_expert = [torch_b1[:, e : e + 1, :] for e in range(routed_experts)]
        b2_per_expert = [torch_b2[:, e : e + 1, :] for e in range(routed_experts)]

    print("Created inputs")

    # --- build module ---
    # Wrap in try/except + pytest.fail(pytrace=False) — pytest's pretty-traceback
    # saferepr'ing the deeply nested config/mesh args takes long enough that the
    # 300s faulthandler watchdog _exit()s the process before any output is shown.
    try:
        decode = TTMoEDecode(
            mesh_device=mesh_device,
            config=config,
            torch_w0=torch_w0,
            torch_w1=torch_w1,
            torch_w2=torch_w2,
            torch_b0=torch_b0,
            torch_b1=torch_b1,
            torch_b2=torch_b2,
        )
    except Exception as e:
        _print_exception_and_fail(f"TTMoEDecode init failed: {type(e).__name__}")

    logger.info("Module Setup complete")

    # --- per-iteration inputs + goldens ---
    tt_dispatch_inputs = []
    tt_dispatch_indices = []
    tt_dispatch_scores = []
    output_goldens = []
    for _ in range(num_iterations):
        tokens = create_torch_dispatch_input_tensor(batch, 1, hidden_size, ttnn.bfloat16)
        indices = _create_expert_indices(batch, routed_experts, select_experts_k)
        scores = create_torch_dispatch_input_expert_scores_tensor(batch, 1, select_experts_k, ttnn.bfloat16)

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
                activation_type=config.compute.activation_type,
                b0_per_expert=b0_per_expert,
                b1_per_expert=b1_per_expert,
                b2_per_expert=b2_per_expert,
            )
        )

    logger.info("Goldens computed")

    # --- run + collect outputs ---
    logger.info("Running forward iterations")
    tt_outputs = []
    for it in range(num_iterations):
        try:
            tt_outputs.append(
                decode.forward(
                    tt_x=tt_dispatch_inputs[it],
                    tt_scores=tt_dispatch_scores[it],
                    tt_indices=tt_dispatch_indices[it],
                    layer_id=0,
                )
            )
            # Surface async device errors immediately rather than letting them
            # silently corrupt state and then deadlock at mesh_device teardown.
            ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        except Exception as e:
            _print_exception_and_fail(f"forward iteration {it} failed: {type(e).__name__}")
        logger.info(f"Op iteration {it} complete")

    # --- verify ---
    logger.info("Verifying outputs")
    all_passed = True
    for it in range(num_iterations):
        if not verify_output(it, mesh_device, mesh_shape, tt_outputs[it], output_goldens[it]):
            all_passed = False

    assert all_passed, f"TTMoEDecode output verification failed for {config_path.stem}"
