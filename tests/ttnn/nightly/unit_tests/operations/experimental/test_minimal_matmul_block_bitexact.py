# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def make_input(dist, m, k):
    x = torch.randn(1, 1, m, k)
    if dist == "outliers":
        x[torch.rand_like(x) < 0.01] *= 1e4
    return x


@pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole only")
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@pytest.mark.parametrize("packer_l1_acc", [True, False])
@pytest.mark.parametrize("w_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("dist", ["randn", "outliers"])
@pytest.mark.parametrize(
    "m, k, n",
    [
        (2048, 2048, 3072),
        (1024, 8192, 2048),
        (2048, 3072, 5120),
        (1024, 14336, 4096),
    ],
)
def test_minimal_matmul_block_bitexact(device, m, k, n, dist, w_dtype, packer_l1_acc, fp32_dest_acc_en):
    torch.manual_seed(0)
    torch_x = make_input(dist, m, k)
    torch_w = torch.randn(1, 1, k, n) * 0.02
    x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(torch_w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    def run(M_block_size, subblock_w):
        config = ttnn.MinimalMatmulConfig(
            M_block_size=M_block_size,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        )
        out = ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=compute_kernel_config, config=config)
        return ttnn.to_torch(out)

    reference = run(8, 1)
    assert_with_pcc(torch_x @ torch_w, reference, 0.999)
    for M_block_size, subblock_w in [(4, 4), (8, 4), (4, 1)]:
        assert torch.equal(
            run(M_block_size, subblock_w), reference
        ), f"M_block_size={M_block_size} subblock_w={subblock_w}"
