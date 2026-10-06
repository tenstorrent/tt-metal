# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host views of ttGDN device results and the GDN accuracy gates (design gdn-on-kda §6.3)."""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.gdn import GDNConfig, GDNReferenceState
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    mla_row_permutation,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tt.kda.linear_attention import KdaState
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import accuracy_metrics

# Shared accuracy definition (assert_accurate / accuracy_metrics) with the §6.3 thresholds.
PCC_THRESHOLD = 0.9995
OUTPUT_REL_RMSE_THRESHOLD = 0.0316
OUTPUT_NORM_RATIO_TOLERANCE = 0.02
# D5 (user-approved): worst V-head final-state relative RMSE; aggregate PCC missed a head-localized K3 failure.
HEAD_STATE_REL_RMSE_THRESHOLD = 0.10
# Trace capture needs a trace region on every mesh, including 1x1 where the fixture sets none.
_TRACE_REGION_SIZE = 64 * 1024 * 1024


def fixture_mesh_shape(mesh_shape: tuple[int, int]) -> tuple[int, int]:
    """Mesh the device fixture opens for a layout's mesh.

    A 1x4 mesh is a submesh of the LoudBox 2x4: opening only those four chips with fabric times out in the router
    handshake on links to the unopened chips (observed in tt_metal_tracker-g1b.5.4.3), so the fixture opens the
    full 2x4 and ``layout_mesh`` carves the 1x4 out of it.
    """
    return (2, 4) if mesh_shape == (1, 4) else mesh_shape


def layout_mesh(mesh_device: ttnn.MeshDevice, mesh_shape: tuple[int, int]) -> ttnn.MeshDevice:
    """The layout's mesh within the fixture's mesh (see ``fixture_mesh_shape``)."""
    if tuple(mesh_device.shape) == tuple(mesh_shape):
        return mesh_device
    return mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))


def gdn_device_params(mesh_shape: tuple[int, int]) -> dict:
    """Device fixture parameters of a GDN layout: FABRIC_1D whenever a collective runs."""
    if mesh_shape == (1, 1):
        return {"trace_region_size": _TRACE_REGION_SIZE}
    return fabric_1d_device_params(trace_region_size=_TRACE_REGION_SIZE)


def gdn_local_widths(config: GDNConfig, tensor_parallel_size: int) -> tuple[int, int, int]:
    """TP-local q, k, v convolution channel widths."""
    return (
        config.q_dim // tensor_parallel_size,
        config.k_dim // tensor_parallel_size,
        config.v_dim // tensor_parallel_size,
    )


def reconstruct_convolution_at_sp_rank(
    tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int, sp_rank: int, config: GDNConfig
) -> torch.Tensor:
    """Full ``[3, conv_dim]`` convolution carry in HF ``[q | k | v]`` channel order from per-rank ``[q_r|k_r|v_r]``."""
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
    columns = tuple(mesh_device.shape)[1]
    tp_size = tuple(mesh_device.shape)[tp_axis]
    widths = gdn_local_widths(config, tp_size)
    per_rank = []
    for tp_rank in range(tp_size):
        row, column = (sp_rank, tp_rank) if sp_axis == 0 else (tp_rank, sp_rank)
        per_rank.append(shards[row * columns + column].reshape(-1, sum(widths)).split(widths, dim=-1))
    return torch.cat([torch.cat([rank[part] for rank in per_rank], dim=-1) for part in range(3)], dim=-1)


def snapshot(
    config: GDNConfig,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    output: ttnn.Tensor,
    state: KdaState,
    chunk_start: int,
    valid_tokens: int,
) -> dict[str, torch.Tensor]:
    """Host copy of one chunk's valid output (natural order) and the carries at every SP rank."""
    sp_size = tuple(mesh_device.shape)[sp_axis]
    local_rows = output.shape[1]
    permutation = mla_row_permutation(chunk_start, sp_size, local_rows)
    rotated = reconstruct_sp_tp_tensor(output, mesh_device, sp_axis, tp_axis, tp_dim=2, sp_dim=1)
    natural = torch.empty_like(rotated)
    natural[:, permutation, :] = rotated
    result = {"output": natural[0, :valid_tokens].clone()}
    for rank in range(sp_size):
        result[f"recurrent_sp{rank}"] = reconstruct_state_at_sp_rank(
            state.recurrent, mesh_device, sp_axis, tp_axis, rank
        )[0].clone()
        result[f"convolution_sp{rank}"] = reconstruct_convolution_at_sp_rank(
            state.convolution, mesh_device, sp_axis, tp_axis, rank, config
        ).clone()
    return result


def expected_tensor(output: torch.Tensor, state: GDNReferenceState, name: str) -> torch.Tensor:
    """Reference counterpart of a snapshot entry, in the device dtype (output and convolution carry are BF16)."""
    if name == "output":
        return output.bfloat16()
    if name.startswith("recurrent"):
        return state.recurrent
    return state.conv.bfloat16()


def per_head_relative_rmse(expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Relative RMSE of each V head of a ``[HV, K, V]`` state."""
    difference = (actual.float() - expected.float()).pow(2).mean((-1, -2)).sqrt()
    scale = expected.float().pow(2).mean((-1, -2)).sqrt()
    return difference / scale.clamp_min(torch.finfo(torch.float32).tiny)


def chunk_gate_rows(
    chunk: int, snapshot_tensors: dict[str, torch.Tensor], output: torch.Tensor, state: GDNReferenceState
) -> tuple[list[dict], list[str]]:
    """Metrics and §6.3 gate failures of one chunk: output, every rank's carries and the D5 per-head state gate."""
    rows, failures = [], []
    for kind in ("output", "recurrent", "convolution"):
        names = [name for name in snapshot_tensors if name.split("_sp")[0] == kind]
        per_rank = {
            name: accuracy_metrics(expected_tensor(output, state, name), snapshot_tensors[name]) for name in names
        }
        worst = min(per_rank, key=lambda name: per_rank[name]["pcc"])
        row = {"chunk": chunk, "tensor": kind, "worst": worst, **per_rank[worst]}
        if row["pcc"] < PCC_THRESHOLD:
            failures.append(f"chunk {chunk} {worst} PCC {row['pcc']:.6f} < {PCC_THRESHOLD}")
        if kind == "output":
            expected = expected_tensor(output, state, "output").float()
            ratio = float(snapshot_tensors["output"].float().norm() / expected.norm())
            row["norm_ratio"] = ratio
            if row["rel_rmse"] > OUTPUT_REL_RMSE_THRESHOLD:
                failures.append(f"chunk {chunk} output rel RMSE {row['rel_rmse']:.4f} > {OUTPUT_REL_RMSE_THRESHOLD}")
            if abs(ratio - 1) > OUTPUT_NORM_RATIO_TOLERANCE:
                failures.append(
                    f"chunk {chunk} output norm ratio {ratio:.4f} outside 1 +- {OUTPUT_NORM_RATIO_TOLERANCE}"
                )
        if kind == "recurrent":
            head_errors = torch.stack(
                [per_head_relative_rmse(state.recurrent, snapshot_tensors[name]) for name in names]
            )
            worst_head = int(head_errors.max(0).values.argmax())
            row["head_rel_rmse"] = [round(float(error), 6) for error in head_errors.max(0).values]
            row["worst_head"] = worst_head
            worst_rank = int(head_errors[:, worst_head].argmax())
            row["worst_head_rank"] = names[worst_rank]
            row["worst_head_rms"] = {
                "expected": float(state.recurrent[worst_head].float().pow(2).mean().sqrt()),
                "actual": float(snapshot_tensors[names[worst_rank]][worst_head].float().pow(2).mean().sqrt()),
            }
            row["worst_head_rel_rmse"] = float(head_errors.max())
            if row["worst_head_rel_rmse"] > HEAD_STATE_REL_RMSE_THRESHOLD:
                failures.append(
                    f"chunk {chunk} V head {worst_head} state rel RMSE {row['worst_head_rel_rmse']:.4f} > "
                    f"{HEAD_STATE_REL_RMSE_THRESHOLD} (D5)"
                )
        rows.append(row)
    return rows, failures
