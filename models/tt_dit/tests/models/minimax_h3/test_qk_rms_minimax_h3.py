# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""What the decoder's query/key RMS costs in precision, with and without an fp32 round-trip.

The reference computes this norm in fp32, and the port matched it by casting its bfloat16 input up
and the result back down. A Tracy capture of one decoder forward bills those casts at 144 ops and
14.9 ms of 172.8 -- 9 % of the decoder -- so whether they buy anything is worth a gate of its own:
the input is already bfloat16, so the upcast carries no information the tensor did not have, and
`fp32_dest_acc_en` accumulates in fp32 either way. This asserts that dropping them costs nothing
measurable against a float64 reference at the shape the decoder actually runs.

The VAE's other tests reach the reference implementation through `diffusers`
`autoencoder_kl_minimax_h3`, which not every environment carries; this one needs only ttnn.
"""

import pytest
import torch

import ttnn

from ....utils.check import assert_quality

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]

# `MiniMaxH3Attention3d`'s SDPA shape: [batch, heads, padded sequence, head_dim].
QK_SHAPE = (1, 32, 1824, 64)
EPS = 1e-6


def _reference_rms(x_bf16: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS over the last dim in float64, on exactly the values the device sees."""
    x = x_bf16.to(torch.float64)
    return (x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)).float()


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_qk_rms_needs_no_fp32_round_trip(mesh_device, reset_seeds):
    torch.manual_seed(0)
    x_bf16 = torch.randn(QK_SHAPE).bfloat16()
    expected = _reference_rms(x_bf16, EPS)

    config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    device_bf16 = ttnn.from_torch(x_bf16, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    # Today: cast up, normalize, cast back.
    up = ttnn.typecast(device_bf16, ttnn.float32)
    round_trip = ttnn.typecast(ttnn.rms_norm(up, epsilon=EPS, compute_kernel_config=config), ttnn.bfloat16)
    round_trip_host = ttnn.to_torch(round_trip).float()

    # Proposed: normalize the bfloat16 operands directly, accumulation still fp32.
    direct_host = ttnn.to_torch(ttnn.rms_norm(device_bf16, epsilon=EPS, compute_kernel_config=config)).float()

    for label, actual in (("fp32 round-trip", round_trip_host), ("bf16 direct", direct_host)):
        error = (actual - expected).abs().max().item()
        print(f"{label}: max |error| {error:.3e}, rmse {(actual - expected).pow(2).mean().sqrt().item():.3e}")
        assert_quality(expected, actual, pcc=0.999, relative_rmse=0.01)

    # The claim under test: dropping the casts is not the worse of the two.
    worse = (direct_host - expected).abs().max().item()
    better = (round_trip_host - expected).abs().max().item()
    assert worse <= better * 1.5 + 1e-3, (
        f"bf16-direct RMS is {worse / max(better, 1e-12):.2f}x the round-trip's max error, "
        "so the fp32 cast is doing something after all"
    )
