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
        (1, 1, 32, 64),  # Wt=2: one block
        (1, 1, 32, 256),  # Wt=8: several blocks, partial sums
        (1, 1, 96, 1024),  # 3 rows: uneven across 4 NEOs
        (2, 3, 64, 2048),  # many rows, Wt=64
        (1, 1, 32, 96),  # Wt=3: batch 3, block 2 + 1
    ],
)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_quasar_softmax(device, shape, numeric_stable):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    if not numeric_stable:
        x = x.clamp(-4, 4)  # keep exp(x) in range without the max subtraction
    ref = torch.softmax(x.float(), dim=-1)

    tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.experimental.quasar.softmax(tx, dim=-1, numeric_stable=numeric_stable)
    out = ttnn.to_torch(out).float()
    assert_with_pcc(ref, out, 0.999)
