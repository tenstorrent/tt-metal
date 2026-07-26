# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LoudBox correctness gate for the Galaxy-shaped SP=8 affine prefix."""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.tt.sp_affine_prefix import SP8AffinePrefixProbe

_SP_SIZE = 8
_HEADS_PER_TP4_RANK = 8
_DIM = 128

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
        indirect=True,
    ),
]


def _compose(
    right_a: torch.Tensor, right_b: torch.Tensor, left_a: torch.Tensor, left_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(right_a, right_b) o (left_a, left_b)`` in FP32."""
    return right_a @ left_a, right_a @ left_b + right_b


def _production_rank_transforms() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Create the exact 512 KiB A + 512 KiB B TP4-rank transport payload."""
    generator = torch.Generator().manual_seed(8441)
    eye = torch.eye(_DIM, dtype=torch.float32).expand(_HEADS_PER_TP4_RANK, -1, -1)
    transforms_a = [
        (0.88 + 0.01 * span) * eye + 0.002 * torch.randn(_HEADS_PER_TP4_RANK, _DIM, _DIM, generator=generator)
        for span in range(_SP_SIZE)
    ]
    transforms_b = [0.01 * torch.randn(_HEADS_PER_TP4_RANK, _DIM, _DIM, generator=generator) for _ in range(_SP_SIZE)]
    initial_state = 0.01 * torch.randn(_HEADS_PER_TP4_RANK, _DIM, _DIM, generator=generator)
    return transforms_a, transforms_b, initial_state


def test_sp8_affine_prefix_production_tp4_payload(mesh_device: ttnn.MeshDevice) -> None:
    """Three device stages derive all eight SP entry states with no host handoff.

    Every A and B tensor is [8, 128, 128] FP32: 512 KiB each, exactly the
    production TP4-rank payload.  The test checks the inclusive transform and
    its application to a nonzero initial recurrent state at every span.
    """
    host_a, host_b, initial_state = _production_rank_transforms()
    probe = SP8AffinePrefixProbe(mesh_device)
    device_a = tuple(
        ttnn.from_torch(
            tensor,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        for tensor, device in zip(host_a, probe.span_devices, strict=True)
    )
    device_b = tuple(
        ttnn.from_torch(
            tensor,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        for tensor, device in zip(host_b, probe.span_devices, strict=True)
    )

    with ttnn.manage_config("throw_exception_on_fallback", True):
        device_barrier = os.getenv("KDA_SP_DEVICE_BARRIER", "0") == "1"
        actual_a, actual_b = probe.run(
            device_a,
            device_b,
            synchronize_stages=not device_barrier,
            device_barrier=device_barrier,
        )
    probe._synchronize()

    expected_a: list[torch.Tensor] = []
    expected_b: list[torch.Tensor] = []
    accumulated_a, accumulated_b = host_a[0], host_b[0]
    expected_a.append(accumulated_a)
    expected_b.append(accumulated_b)
    for span in range(1, _SP_SIZE):
        accumulated_a, accumulated_b = _compose(host_a[span], host_b[span], accumulated_a, accumulated_b)
        expected_a.append(accumulated_a)
        expected_b.append(accumulated_b)

    for span, (expected_transform_a, expected_transform_b, device_transform_a, device_transform_b, device) in enumerate(
        zip(expected_a, expected_b, actual_a, actual_b, probe.span_devices, strict=True)
    ):
        actual_transform_a = ttnn.to_torch(device_transform_a, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        actual_transform_b = ttnn.to_torch(device_transform_b, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        expected_entry = expected_transform_a @ initial_state + expected_transform_b
        actual_entry = actual_transform_a @ initial_state + actual_transform_b
        for name, expected, actual in (
            ("A", expected_transform_a, actual_transform_a),
            ("B", expected_transform_b, actual_transform_b),
            ("entry state", expected_entry, actual_entry),
        ):
            assert torch.isfinite(actual).all(), f"SP=8 prefix span {span} {name} contains non-finite values"
            passed, pcc = comp_pcc(expected, actual, pcc=0.999)
            assert passed, f"SP=8 prefix span {span} {name} PCC {pcc:.6f} < 0.999"
