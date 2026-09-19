# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Bisect the sharded-conv3d corruption that blocks C-09 phase 2 (T-sharded transposed convs / conv_pre), on one chip:
the halo'd per-shard shape (2, 81, 2048) -> 1024, k7, no internal padding (T_out 75 = 9 full T blocks of 8 + a 3-row
tail) against torch in float64, then the variants that remove one suspect at a time: a T that fills its blocks (86 -> 80
rows), a 1-row tail (87), C_in_block 256 (8 reduction blocks instead of 16), T_out_block 25 (divides 75).
    pytest models/tt_dit/tests/models/minimax_h3/tools/conv_shard_shape_probe.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from .....layers.audio_ops import Conv1dViaConv3d
from .....models.audio_vae.minimax_h3.blockings_minimax_h3_audio import register_h3_audio_blockings

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
C_IN, C_OUT, K = 2048, 1024, 7


def _config_like(base, *, t_out=None, c_in=None, grid):
    return ttnn.Conv3dConfig(
        weights_dtype=base.weights_dtype,
        output_layout=base.output_layout,
        T_out_block=base.T_out_block if t_out is None else t_out,
        W_out_block=base.W_out_block,
        H_out_block=base.H_out_block,
        C_out_block=base.C_out_block,
        C_in_block=base.C_in_block if c_in is None else c_in,
        compute_with_storage_grid_size=grid,
        operand_split=base.operand_split,
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_conv_shard_shape(mesh_device):
    register_h3_audio_blockings()
    grid = mesh_device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    w = torch.randn(C_OUT, C_IN, K) * (1.0 / (C_IN * K) ** 0.5)
    b = torch.randn(C_OUT) * 0.1
    cases = [
        ("T81 (75 out: 9 blocks + 3-row tail), table blocking", 81, {}),
        ("T86 (80 out: 10 full blocks)", 86, {}),
        ("T87 (81 out: 1-row tail)", 87, {}),
        ("T81, C_in_block 256", 81, {"c_in": 256}),
        ("T81, T_out_block 25 (divides 75)", 81, {"t_out": 25}),
        ("T81, T_out_block 1", 81, {"t_out": 1}),
    ]
    for c_in_block in (None, 256):
        pass
    for name, t_in, over in cases:
        conv = Conv1dViaConv3d(C_IN, C_OUT, kernel_size=K, padding=0, mesh_device=mesh_device, dtype=ttnn.float32, split_mode="kernel")
        base = conv.conv_config
        if over:
            conv.conv_config = _config_like(base, grid=grid, **over)
        conv.load_torch_state_dict({"weight": w.clone(), "bias": b.clone()})  # prepared per C_in_block, so after the config
        x = torch.randn(2, t_in, C_IN)
        x_dev = ttnn.from_torch(x, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        try:
            out = ttnn.to_torch(conv(x_dev)).double()
        except Exception as exc:  # noqa: BLE001
            logger.info(f"CONVSHAPE {name}: FAILED {type(exc).__name__}: {str(exc)[:140]}")
            continue
        ref = torch.nn.functional.conv1d(x.double().transpose(1, 2), w.double(), b.double(), padding=0).transpose(1, 2)
        out = out[:, : ref.shape[1], : ref.shape[2]]
        err = (out - ref).abs()
        rel = float(err.norm() / ref.norm())
        per_row = err.amax(dim=(0, 2))
        worst = int(per_row.argmax())
        logger.info(
            f"CONVSHAPE {name}: cfg Cin{conv.conv_config.C_in_block} Cout{conv.conv_config.C_out_block} T{conv.conv_config.T_out_block}; "
            f"rel-RMSE {rel:.3e} max {float(err.max()):.3e} (worst row {worst} of {ref.shape[1]}; rows with err>1e-2: "
            f"{int((per_row > 1e-2).sum())}, first {int(torch.nonzero(per_row > 1e-2)[0]) if (per_row > 1e-2).any() else -1})"
        )
        ttnn.deallocate(x_dev)
