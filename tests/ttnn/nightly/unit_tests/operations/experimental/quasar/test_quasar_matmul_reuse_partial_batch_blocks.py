# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""#57954 in the Quasar copy of the Reuse factory: with per_core_M < Mt and several blocks per core, a core's
consecutive blocks must walk the M blocks of a batch before moving to the next batch. The output starts out NaN so
that any tile the kernels skip fails the comparison."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Every test in this module takes the function-scoped `device` fixture without
# parametrizing device_params, so the device is opened once per module rather than
# once per test.
pytestmark = pytest.mark.use_module_device

qsr = ttnn._ttnn.operations.experimental.quasar


@pytest.mark.parametrize(
    "batch_shape, m, k, n, per_core_M",
    [
        ((5,), 256, 256, 64, 1),  # 40 blocks on 16 cores: runs of 2-3 blocks cross batch boundaries
        ((64,), 256, 256, 64, 4),  # 128 blocks, 8 per core
        ((4, 12), 1024, 1024, 64, 16),  # 96 blocks, 6 per core
        ((48,), 256, 256, 64, 8),  # whole batch matrices (per_core_M == Mt)
    ],
)
@pytest.mark.parametrize("transpose_a", [False, True])
def test_quasar_matmul_reuse_config_partial_batch_blocks(device, batch_shape, m, k, n, per_core_M, transpose_a):
    torch.manual_seed(0)
    a_shape = (*batch_shape, k, m) if transpose_a else (*batch_shape, m, k)
    torch_a = torch.randn(a_shape, dtype=torch.bfloat16)
    torch_b = torch.randn((*batch_shape, k, n), dtype=torch.bfloat16)
    torch_output = (torch_a.transpose(-1, -2) if transpose_a else torch_a).float() @ torch_b.float()

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.from_torch(
        torch.full((*batch_shape, m, n), float("nan"), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    program_config = qsr.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(4, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=per_core_M,
        per_core_N=n // 32,
    )
    ttnn.experimental.quasar.matmul(
        a, b, transpose_a=transpose_a, program_config=program_config, optional_output_tensor=output
    )

    output = ttnn.to_torch(output).float()
    assert not torch.isnan(output).any(), "output tiles were not written"
    assert_with_pcc(torch_output, output, 0.999)
