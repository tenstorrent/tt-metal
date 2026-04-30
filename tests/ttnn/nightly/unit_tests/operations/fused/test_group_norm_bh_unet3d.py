# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Regression test for the U-Net 3D GroupNorm PCC drop on Blackhole (issue #42991).

These shapes correspond to three GroupNorm call-sites observed in the U-Net 3D
model. Originally reproduced via tt_metal_repro/test_unet3d_group_norm_blackhole.py
which loaded fixture activations/weights from disk; here we substitute torch.randn
of the same shapes so the test is self-contained and can run as a nightly.

The interesting case is `dec1_upsample_cat_gn1_conv1_relu1_then_gn2`: when no
core_grid is supplied, ttnn.group_norm auto-selects an 11x10 grid on Blackhole,
which previously hit a stale-L1 bug in the multicast-sender reader kernel
(`reader_mcast_sender_unary_gn.cpp`) where partial-sum tiles in `cb_ex_external`
were not fully cleared between out-blocks, causing PCC to drop from ~0.999 to
~0.88. The fix tile-clears `cb_ex_external` whenever a new tile is started so
the `REDUCE_SCALAR` in the compute kernel does not pick up garbage.

We deliberately do not pass `core_grid=` so the auto-selection path
(GroupNormMcastProgramFactory) is exercised, which is the path that hit the bug.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole


# (case_id, N, C, D, H, W, num_groups, eps)
UNET3D_CASES = [
    ("enc1_pool_then_gn", 1, 32, 64, 120, 120, 8, 1e-5),
    ("dec1_upsample_cat_gn1_conv1_relu1_then_gn2", 1, 64, 64, 120, 120, 8, 1e-5),
    ("dec2_upsample_cat_gn1_conv1_relu1_then_gn2", 1, 32, 128, 240, 240, 8, 1e-5),
]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient on flattened fp64 tensors. Same
    convention as tt-xla's compute_pcc and tt-metal's assert_with_pcc."""
    a64 = a.detach().to(torch.float64).flatten()
    b64 = b.detach().to(torch.float64).flatten()
    if torch.allclose(a64, b64, rtol=1e-2, atol=1e-2):
        return 1.0
    va = a64 - a64.mean()
    vb = b64 - b64.mean()
    denom = float(va.norm() * vb.norm())
    if denom == 0.0:
        return float("nan")
    return float((va @ vb) / denom)


def _to_ttnn_input(x_ncdhw: torch.Tensor, device) -> ttnn.Tensor:
    """Convert (1, C, D, H, W) torch -> (1, 1, D*H*W, C) ttnn TILE in DRAM,
    mirroring the layout the model pipeline produces in the failing IR."""
    n, c, d, h, w = x_ncdhw.shape
    assert n == 1, "Only batch size 1 is exercised by this test."
    x_nhwc_flat = x_ncdhw.permute(0, 2, 3, 4, 1).contiguous().reshape(1, 1, d * h * w, c)
    return ttnn.from_torch(
        x_nhwc_flat.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_ttnn_weight_or_bias(w_flat: torch.Tensor, num_channels: int, device) -> ttnn.Tensor:
    """Use the canonical tt-metal helper so the layout matches the IR
    (``1 x 1 x ceil(C/32) x 32`` row-major bf16 in DRAM)."""
    assert w_flat.numel() == num_channels, f"Expected weight/bias of length {num_channels}, got {w_flat.numel()}"
    padded = ttnn.create_group_norm_weight_bias_rm(w_flat.to(torch.bfloat16), num_channels, num_cores_x=1)
    return ttnn.from_torch(
        padded,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _torch_reference(
    x_ncdhw: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int, eps: float
) -> torch.Tensor:
    """Reference ``group_norm`` computed in fp32 on the bf16-quantised input,
    then re-cast to bf16. This mirrors what the device sees while keeping fp32
    accumulation in the math, matching the wormhole result we compare against."""
    x_bf = x_ncdhw.to(torch.bfloat16).to(torch.float32)
    w_bf = weight.to(torch.bfloat16).to(torch.float32)
    b_bf = bias.to(torch.bfloat16).to(torch.float32)
    out = torch.nn.functional.group_norm(x_bf, num_groups, w_bf, b_bf, eps=eps)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "case_id, N, C, D, H, W, num_groups, eps",
    UNET3D_CASES,
    ids=[c[0] for c in UNET3D_CASES],
)
@run_for_blackhole(reason_str="auto-selected core_grid path differs on Blackhole; this is the BH regression case")
def test_group_norm_bh_unet3d(device, case_id, N, C, D, H, W, num_groups, eps):
    torch.manual_seed(0)

    x = torch.randn((N, C, D, H, W), dtype=torch.float32)
    weight = torch.randn((C,), dtype=torch.float32)
    bias = torch.randn((C,), dtype=torch.float32)

    golden = _torch_reference(x, weight, bias, num_groups, eps)

    x_tt = _to_ttnn_input(x, device)
    w_tt = _to_ttnn_weight_or_bias(weight, C, device)
    b_tt = _to_ttnn_weight_or_bias(bias, C, device)

    y_tt = ttnn.group_norm(
        x_tt,
        num_groups=num_groups,
        epsilon=eps,
        weight=w_tt,
        bias=b_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        inplace=False,
        num_out_blocks=-1,
    )
    ttnn.synchronize_device(device)

    y_torch_4d = ttnn.to_torch(y_tt)
    y_torch_5d = y_torch_4d.reshape(1, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    pcc = _pcc(golden, y_torch_5d)
    print(f"[{case_id}] PCC(ttnn vs torch.group_norm) = {pcc:.6f}")

    assert pcc >= 0.99, f"[{case_id}] PCC={pcc:.6f} below 0.99 threshold. This reproduces tt-metal#42991."
