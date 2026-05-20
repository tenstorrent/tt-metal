# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
import ttnn


pytestmark = pytest.mark.use_module_device({"l1_small_size": 1 << 15})

_REPRO_DIR = Path(__file__).parent / "test_data" / "rms_norm_bad_user"
_PCC_BAD_USER_THRESHOLD = 0.94


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def test_rms_norm_reproduces_llama_bad_user_pattern(device):
    input_parts = sorted(_REPRO_DIR.glob("input_part_*.pth"))
    input_tensor = torch.cat([torch.load(part_path, map_location="cpu") for part_path in input_parts], dim=0)
    weight_tensor = torch.load(_REPRO_DIR / "weight.pth", map_location="cpu")

    assert tuple(input_tensor.shape) == (576, 2048)
    assert tuple(weight_tensor.shape) == (2048,)

    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 8:
        pytest.skip("Repro requires at least an 8x8 compute grid.")

    sharded_l1 = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
            [576, 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    interleaved_dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

    x = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=sharded_l1,
    )
    w = ttnn.from_torch(
        weight_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=interleaved_dram,
    )

    out = ttnn.rms_norm(
        x,
        epsilon=9.999999747378752e-06,
        weight=w,
        bias=None,
        residual_input_tensor=None,
        memory_config=sharded_l1,
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    out_torch = ttnn.to_torch(ttnn.from_device(out)).cpu()
    out_by_user = out_torch.reshape(32, 18, 2048)

    pcc_vs_user0 = [_pcc(out_by_user[user], out_by_user[0]) for user in range(32)]
    bad_users = tuple(user for user, score in enumerate(pcc_vs_user0) if score < _PCC_BAD_USER_THRESHOLD)

    # The exact failing lanes can vary by machine/run, but we still expect
    # lane divergence to be visible for this captured repro input.
    assert len(bad_users) > 0, (
        "Expected at least one divergent user lane, but none were found. " f"pcc_vs_user0={pcc_vs_user0}"
    )
