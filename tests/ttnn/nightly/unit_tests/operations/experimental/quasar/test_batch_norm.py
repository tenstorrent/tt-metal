# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 32, 64),  # 4 planes of 2 tiles: one round, batch 2
        (2, 3, 64, 128),  # 6 planes of 8 tiles: two rounds, duplicate planes
        (1, 2, 32, 32),  # 2 planes of 1 tile: batch 1
        (1, 1, 96, 96),  # 1 plane of 9 tiles: batch 3
    ],
)
@pytest.mark.parametrize("affine", ["none", "weight", "weight_bias"])
def test_quasar_batch_norm(device, shape, affine):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)")
    torch.manual_seed(0)
    eps = 1e-5
    N, C, H, W = shape
    x = torch.randn(shape, dtype=torch.bfloat16)
    mean = torch.randn((1, C, 1, 1), dtype=torch.bfloat16)
    var = torch.rand((1, C, 1, 1), dtype=torch.bfloat16) + 0.5
    weight = torch.randn((1, C, 1, 1), dtype=torch.bfloat16) if affine != "none" else None
    bias = torch.randn((1, C, 1, 1), dtype=torch.bfloat16) if affine == "weight_bias" else None

    ref = torch.nn.functional.batch_norm(
        x.float(),
        mean.float().reshape(C),
        var.float().reshape(C),
        weight=weight.float().reshape(C) if weight is not None else None,
        bias=bias.float().reshape(C) if bias is not None else None,
        training=False,
        eps=eps,
    )

    to_dev = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device) if t is not None else None
    out = ttnn.experimental.quasar.batch_norm(
        to_dev(x), running_mean=to_dev(mean), running_var=to_dev(var), eps=eps, weight=to_dev(weight), bias=to_dev(bias)
    )
    out = ttnn.to_torch(out).float()
    assert_with_pcc(ref, out, 0.999)
