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
        (1, 1, 32, 64),  # Wt=2 < window: no reload
        (1, 1, 32, 256),  # Wt=8: tail=4, reload 4
        (1, 1, 96, 1024),  # 3 rows: uneven across 4 NEOs
        (2, 3, 64, 2048),  # many rows, Wt=64
        (1, 1, 32, 100),  # ragged W
    ],
)
@pytest.mark.parametrize("affine", ["none", "gamma", "gamma_beta"])
def test_quasar_layer_norm(device, shape, affine):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)")
    torch.manual_seed(0)
    eps = 1e-5
    x = torch.randn(shape, dtype=torch.bfloat16)
    W = shape[-1]
    gamma = torch.randn((1, 1, 1, W), dtype=torch.bfloat16) if affine != "none" else None
    beta = torch.randn((1, 1, 1, W), dtype=torch.bfloat16) if affine == "gamma_beta" else None

    ref = torch.nn.functional.layer_norm(
        x.float(),
        (W,),
        weight=gamma.float().reshape(W) if gamma is not None else None,
        bias=beta.float().reshape(W) if beta is not None else None,
        eps=eps,
    )

    tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(gamma, layout=ttnn.TILE_LAYOUT, device=device) if gamma is not None else None
    tb = ttnn.from_torch(beta, layout=ttnn.TILE_LAYOUT, device=device) if beta is not None else None

    out = ttnn.experimental.quasar.layer_norm(tx, epsilon=eps, weight=tg, bias=tb)
    out = ttnn.to_torch(out).float()
    assert_with_pcc(ref, out, 0.999)
