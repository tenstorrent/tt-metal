# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for rms_norm: PCC, abs/rel-RMS error, ULP-ish, got/true ratio spread."""
import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

TORCH_DT = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
SHAPES = [(1, 1, 32, 64), (2, 1, 100, 47), (1, 1, 128, 4096), (1, 1, 32, 8192)]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", SHAPES)
def test_precision_baseline(device, shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.float32)
    xin = x.to(TORCH_DT[dtype]).float()
    ref = xin / torch.sqrt(xin.pow(2).mean(-1, keepdim=True) + 1e-6) * g.to(TORCH_DT[dtype]).float()
    tx = ttnn.from_torch(x.to(TORCH_DT[dtype]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(g.to(TORCH_DT[dtype]), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tx, gamma=tg, epsilon=1e-6)).float()
    ok, pcc = assert_with_pcc(ref, out, 0.99)
    err = (out - ref).abs()
    rms = ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    m = (ref.abs() > 1e-6) & torch.isfinite(out)
    r = (out[m] / ref[m])
    q = torch.quantile(r[:1_000_000], torch.tensor([0.05, 0.5, 0.95]))
    print(
        f"BASELINE shape={shape} dtype={dtype} pcc={pcc} max_abs={err.max():.5f} mean_abs={err.mean():.6f} "
        f"rel_rms={rms:.5f} ratio_p5/med/p95={q.tolist()}"
    )
    print(comp_allclose(ref, out))
    assert 0.9 < q[1] < 1.1
