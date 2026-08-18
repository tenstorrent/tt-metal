# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn

from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

TILE = 32
COUNT = 64
GRID = ttnn.CoreCoord(11, 8)
GLOBAL_EXPERT_ID = 137
INPUT_SCALES = (1.0, 1.0e-1, 1.0e-2, 1.0e-3)
WEIGHT_SCALE = 2.0e-2


def _to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def _pcc(reference, actual):
    reference = reference.float().flatten()
    actual = actual.float().flatten()
    finite = torch.isfinite(reference) & torch.isfinite(actual)
    if finite.count_nonzero() < 2:
        return math.nan
    reference = reference[finite]
    actual = actual[finite]
    reference = reference - reference.mean()
    actual = actual - actual.mean()
    denom = torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(actual)
    return (torch.dot(reference, actual) / denom).item() if denom != 0 else math.nan


def _metrics(actual, reference):
    actual = actual.float()
    reference = reference.float()
    actual_finite = torch.isfinite(actual)
    joint_finite = actual_finite & torch.isfinite(reference)
    delta = actual[joint_finite] - reference[joint_finite]
    finite_columns = actual_finite.all(dim=0)
    finite_indices = finite_columns.nonzero().flatten().tolist()
    finite_column_runs = []
    for index in finite_indices:
        if not finite_column_runs or index != finite_column_runs[-1][1]:
            finite_column_runs.append([index, index + 1])
        else:
            finite_column_runs[-1][1] = index + 1
    return {
        "finite_fraction": actual_finite.float().mean().item(),
        "nan_fraction": torch.isnan(actual).float().mean().item(),
        "inf_fraction": torch.isinf(actual).float().mean().item(),
        "fully_finite_columns": finite_columns.count_nonzero().item(),
        "finite_column_runs": finite_column_runs,
        "total_columns": actual.shape[-1],
        "max_abs": actual[actual_finite].abs().max().item() if actual_finite.any() else math.inf,
        "mean_abs": actual[actual_finite].abs().mean().item() if actual_finite.any() else math.inf,
        "rmse": torch.sqrt(torch.mean(delta * delta)).item() if delta.numel() else math.inf,
        "pcc": _pcc(reference, actual),
    }


def _reference(x, weights):
    gate, up, down = (weight.float() for weight in weights)
    x = x.float()
    return (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down


@pytest.mark.parametrize("emb,hidden", [(6144, 2048), (7168, 2048)], ids=["glm", "kimi-ds"])
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_fused_placement_and_unified_scale_sweep(device, emb, hidden, expect_error):
    torch.manual_seed(20260818)
    base_x = torch.randn((1, 1, COUNT, emb), dtype=torch.bfloat16)
    host_weights = [
        torch.randn(shape, dtype=torch.bfloat16) * WEIGHT_SCALE
        for shape in ((emb, hidden), (emb, hidden), (hidden, emb))
    ]

    gate_nd, down_nd = weight_memory_configs(device, emb, hidden, core_grid=GRID)
    interleaved_weights = [_to_device(weight, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device) for weight in host_weights]
    nd_weights = [
        _to_device(weight, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device, memory_config)
        for weight, memory_config in zip(host_weights, (gate_nd, gate_nd, down_nd))
    ]

    counts_host = torch.zeros(256, dtype=torch.int32)
    counts_host[GLOBAL_EXPERT_ID] = COUNT
    counts = _to_device(counts_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    expert_ids = _to_device(
        torch.tensor([GLOBAL_EXPERT_ID], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )
    config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )

    # Verify that placement itself did not change the quantized tensor contents.
    for name, interleaved, nd in zip(("gate", "up", "down"), interleaved_weights, nd_weights):
        interleaved_host = ttnn.to_torch(interleaved).float()
        nd_host = ttnn.to_torch(nd).float()
        assert torch.equal(interleaved_host, nd_host), f"{name} differs after interleaved/ND upload"

    results = {}
    for input_scale in INPUT_SCALES:
        x_host = (base_x * input_scale).to(torch.bfloat16)
        reference = _reference(x_host[0, 0], host_weights)
        x = _to_device(x_host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)

        fused_outputs = {}
        for placement, weights in (("interleaved", interleaved_weights), ("nd", nd_weights)):
            output = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                x,
                *weights,
                counts,
                expert_ids,
                0,
                input_m_tiles=COUNT // TILE,
                compute_kernel_config=config,
                core_grid=GRID,
            )
            fused_outputs[placement] = ttnn.to_torch(output)[0, 0, :COUNT].float()

        unified_output_tensor = ttnn.empty(
            x.shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        unified_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
            x,
            *interleaved_weights,
            counts,
            expert_ids,
            0,
            compute_kernel_config=config,
            output=unified_output_tensor,
            input_m_tiles=COUNT // TILE,
            x_is_row_major=True,
        )
        unified_host = ttnn.to_torch(unified_output)[0, 0, :COUNT].float()

        scale_key = f"{input_scale:.0e}"
        placement_joint_finite = torch.isfinite(fused_outputs["interleaved"]) & torch.isfinite(fused_outputs["nd"])
        placement_delta = (
            fused_outputs["interleaved"][placement_joint_finite] - fused_outputs["nd"][placement_joint_finite]
        )
        results[scale_key] = {
            "fused_interleaved": _metrics(fused_outputs["interleaved"], reference),
            "fused_nd": _metrics(fused_outputs["nd"], reference),
            "unified_interleaved": _metrics(unified_host, reference),
            "fused_placement_pcc": _pcc(fused_outputs["interleaved"], fused_outputs["nd"]),
            "fused_placement_finite_mask_equal": torch.equal(
                torch.isfinite(fused_outputs["interleaved"]), torch.isfinite(fused_outputs["nd"])
            ),
            "fused_placement_max_finite_abs_diff": (
                placement_delta.abs().max().item() if placement_delta.numel() else math.nan
            ),
            "fused_vs_unified_pcc": _pcc(fused_outputs["interleaved"], unified_host),
        }

    # The unified operation currently documents and validates interleaved-only weights.
    with expect_error(RuntimeError, "DRAM.interleaved"):
        ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
            x,
            *nd_weights,
            counts,
            expert_ids,
            0,
            compute_kernel_config=config,
            output=unified_output_tensor,
            input_m_tiles=COUNT // TILE,
            x_is_row_major=True,
        )

    for result in results.values():
        assert result["fused_placement_pcc"] >= 0.999
        assert result["fused_placement_finite_mask_equal"]
        assert result["fused_placement_max_finite_abs_diff"] == 0.0
        assert result["fused_interleaved"]["finite_fraction"] == 1.0
        assert result["fused_nd"]["finite_fraction"] == 1.0
        assert result["unified_interleaved"]["finite_fraction"] == 1.0
        assert result["fused_interleaved"]["pcc"] >= 0.97
        assert result["fused_nd"]["pcc"] >= 0.97
        assert result["unified_interleaved"]["pcc"] >= 0.97
        assert result["fused_vs_unified_pcc"] >= 0.999
