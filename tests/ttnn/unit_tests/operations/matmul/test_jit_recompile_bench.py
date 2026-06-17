# SPDX-License-Identifier: Apache-2.0
#
# Artificial matmul benchmark that FORCES many distinct kernel compiles (so the JIT path -- local vs
# remote server -- is exercised under real recompiles, not build-once dedup). Distinctness comes from
# the ComputeKernelConfig: math_fidelity / fp32_dest_acc_en / packer_l1_acc / math_approx_mode /
# dst_full_sync_en are all COMPILE-TIME, so each combination produces a distinct compute-kernel hash.
# 4 * 2 * 2 * 2 * 2 = 64 distinct compute kernels on the same fixed shape.
#
# Cold (cleared cache) => ~64 real compiles. Warm with the JIT server + Option B => 0 round-trips.
#   scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/matmul/test_jit_recompile_bench.py
import itertools

import pytest
import torch
import ttnn

FIDELITIES = [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4]
CONFIGS = list(itertools.product(FIDELITIES, [False, True], [False, True], [False, True], [False, True]))


def _id(c):
    fid, fp32, packer, approx, dst = c
    return f"{fid}-fp32_{int(fp32)}-pk_{int(packer)}-approx_{int(approx)}-dst_{int(dst)}"


@pytest.mark.parametrize("fid,fp32,packer,approx,dst", CONFIGS, ids=[_id(c) for c in CONFIGS])
def test_jit_recompile_bench(device, fid, fp32, packer, approx, dst):
    torch.manual_seed(0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=packer,
        dst_full_sync_en=dst,
    )
    a = ttnn.from_torch(
        torch.randn(512, 512, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = ttnn.from_torch(
        torch.randn(512, 512, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = ttnn.matmul(a, b, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    assert tuple(out.shape) == (512, 512)
