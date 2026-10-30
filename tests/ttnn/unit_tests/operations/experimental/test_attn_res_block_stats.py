# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole

pytestmark = run_for_blackhole()

SHARD_WIDTH = 1792  # 7168 over a 4-way tensor-parallel axis


@pytest.mark.parametrize("planes", [[1], [6], [6, 1], [3, 1, 1, 2]], ids=lambda p: "+".join(map(str, p)))
@pytest.mark.parametrize("num_sites", [1, 24])
@pytest.mark.parametrize("num_tokens", [64, 640])
def test_attn_res_block_stats(device, planes, num_sites, num_tokens):
    torch.manual_seed(0)
    snapshots = [torch.randn(1, s, num_tokens, SHARD_WIDTH).bfloat16() for s in planes]
    queries = torch.zeros(1, 1, SHARD_WIDTH, 32)
    queries[..., 1 : 1 + num_sites] = torch.randn(SHARD_WIDTH, num_sites) * 0.02
    queries = queries.bfloat16()

    to_device = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.experimental.deepseek_prefill.attn_res_block_stats([to_device(s) for s in snapshots], to_device(queries))
    assert tuple(out.shape) == (1, sum(planes), num_tokens, 32)
    assert out.dtype == ttnn.float32
    out = ttnn.to_torch(out)

    v = torch.cat(snapshots, dim=1).float()
    sum_squares = (v * v).sum(dim=-1)
    dots = v @ queries.float()[0, 0, :, 1 : 1 + num_sites]

    # The squares are packed through bf16 before they are summed, as the RMSNorm statistics
    # kernel this replaces does; the dots accumulate in fp32 dest at HiFi4.
    torch.testing.assert_close(out[..., 0], sum_squares, rtol=5e-3, atol=0)
    passed, pcc = comp_pcc(dots, out[..., 1 : 1 + num_sites], 0.99999)
    assert passed, f"dots PCC {pcc}"
    torch.testing.assert_close(out[..., 1 : 1 + num_sites], dots, rtol=2e-2, atol=2e-2)
    assert out[..., 1 + num_sites :].abs().max() == 0, "columns past the last site must stay zero"
