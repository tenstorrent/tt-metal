# Scratch, not for merge: the MoE DRAM expert matmul (matmul_expert_compressed_dram.hpp) at the fused MoE's gate/up and
# down configurations, 8 active experts, one device.
import hashlib
import os
import sys

import pytest
import ttnn

from models.demos.deepseek_v3_b1.tests.unit_tests.per_core_allocation.test_matmul_expert import (
    _run_hybrid_expert_multi_device as run,
)

MIX = {
    "p": (["bfp4", "bfp2", "bfp0"], {"bfp4": 55, "bfp2": 41, "bfp0": 4}),
    "b": (["bfp4", "bfp2", "bfp0"], {"bfp4": 74.4, "bfp2": 7.3, "bfp0": 18.3}),
}
# Every tensor read back to the host is hashed, so arms can be compared bit for bit
_to_torch = ttnn.to_torch
_seen = {"id": None, "n": 0}


def _hashed_to_torch(*a, **k):
    r = _to_torch(*a, **k)
    tid = os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]
    if tid != _seen["id"]:
        _seen["id"], _seen["n"] = tid, 0
    try:
        h = hashlib.md5(r.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:12]
    except Exception as ex:  # noqa: BLE001
        h = f"nohash:{type(ex).__name__}"
    sys.__stderr__.write(f"BITS {tid.split('[')[-1].rstrip(']')} {_seen['n']} {h}\n")
    _seen["n"] += 1
    return r


ttnn.to_torch = _hashed_to_torch

SPLIT = int(os.environ.get("MEXP_SPLIT", "1"))
if SPLIT > 1:
    import functools

    import models.demos.deepseek_v3_b1.tests.unit_tests.per_core_allocation.test_matmul_expert as tme

    tme.create_dram_expert_tensors_multi_device = functools.partial(
        tme.create_dram_expert_tensors_multi_device, num_in1_buffers=3 * SPLIT
    )
CASES = [c for c in os.environ.get("MEXP_CASES", "g_p_1,g_p_100,g_b_1,g_b_100,d_p_1,d_p_100,d_b_1,d_b_100").split(",") if c]


@pytest.mark.parametrize("case", CASES)
def test_mexp(device, case):
    shape, mix, loops = case.split("_")
    fmts, ratios = MIX[mix]
    common = dict(
        M=1,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[fmts],
        fmt_distribution="uniform",
        fmt_ratios=ratios,
        num_loop_iters=int(loops),
    )
    if shape == "g":
        run(device, K=7168, N=256, dram_fuse_silu=True, num_subblocks_k=4 * SPLIT, num_subblocks_n=1,
            n_parallel_per_bank=1, k_parallel_per_bank=2, **common)
    else:
        run(device, K=256, N=7168, accum_experts=True, num_subblocks_k=1 * SPLIT, num_subblocks_n=2,
            n_parallel_per_bank=2, k_parallel_per_bank=1, **common)
