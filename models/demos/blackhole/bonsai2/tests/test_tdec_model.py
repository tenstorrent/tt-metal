# The ternary decode linear as the model builds it (from_packed: inline group16 records, PAGE_SPLIT), for profiling.
# TDEC_DEFINES=a,b: extra defines for every kernel (e.g. FAKE_STREAM, SKIP_MM, SKIP_SFPU, TDEC_ZONES).
import os

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import (
    TernaryDecodeLinear3,
    pack_records_inline,
    reference,
)

SHAPES = [(17408, 5120), (6144, 5120), (5120, 16384)]


@pytest.mark.parametrize("K, N", SHAPES, ids=[f"m{k}x{n}" for k, n in SHAPES])
def test_tdec_model(device, K, N):
    ed = [(v, "1") for v in os.environ.get("TDEC_DEFINES", "").split(",") if v]
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    rec, ppr = pack_records_inline(t, s)
    lin = TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)
    x = torch.randn(K).bfloat16()
    act = ttnn.from_torch(
        x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    if not ed:
        y = ttnn.to_torch(lin(act)).float().reshape(-1, N)[0]
        ref = reference(x, t, s)
        rel = ((y - ref).norm() / ref.norm()).item()
        print(f"TDEC_MODEL K={K} N={N} rel={rel:.5f}")
        assert rel < 0.02, rel
    for _ in range(10):
        lin(act, extra_defines=ed, dm_defines=ed)
    ttnn.synchronize_device(device)
