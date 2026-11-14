# W_STAGED: a ternary decode linear reading weight pages that tdec3_stage.cpp copied into L1 beforehand.
import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3, pack_records_inline


@pytest.mark.parametrize("K, N", [(6144, 5120)])
def test_stage(device, K, N):
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    rec, ppr = pack_records_inline(t, s)
    lin = TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)
    x = torch.randn(K).bfloat16()
    act = ttnn.from_torch(
        x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ref = ttnn.to_torch(lin(act)).float().reshape(-1, N)[0]
    stage = lin.alloc_stage(device)
    _, kernels = lin.stage_kernels(device, stage)
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
    for _ in range(3):
        ttnn.generic_op([lin.w, stage], prog)
        y = ttnn.to_torch(lin(act, stage=stage)).float().reshape(-1, N)[0]
    ttnn.synchronize_device(device)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"STAGE rel vs DRAM-reading op {rel:.6f}")
    assert rel == 0.0, rel
