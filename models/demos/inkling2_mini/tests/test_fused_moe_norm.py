# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""E10d measurement for M1: the MoE input norm (tt/moe.py MoE.norm: reshard + sharded rms_norm + to L1 bf16,
3 programs) vs FusedRowNorm with an L1 output (1 program). Not bit-equal (HiFi2 FPU vs fp32 SFPU), so this
records how many bf16 outputs differ, the PCC and both traced times; it asserts only PCC."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.norm import FusedRowNorm

C = 2304
RESULTS = os.environ.get("INKLING_E10_RESULTS")


def traced(device, fn, n=200):
    fn()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / n * 1e6
    r = ttnn.to_torch(out).float().reshape(-1)[:C]
    ttnn.release_trace(device, tid)
    return r, us


@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_moe_norm(device):
    g = torch.Generator().manual_seed(11)
    w = (1 + 0.1 * torch.randn(C, generator=g)).bfloat16()
    h = ttnn.from_torch(
        torch.randn(1, 1, 1, C, generator=g) * 3, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    # tt/moe.py MoE.__init__ / norm
    norm_w = ttnn.from_torch(w.reshape(1, 1, C // 32, 32), dtype=ttnn.bfloat16, device=device)
    shard = ttnn.create_sharded_memory_config(
        (32, C // 18), core_grid=ttnn.CoreGrid(y=2, x=9), strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
    )  # fmt: skip
    prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(9, 2), subblock_w=4, block_h=1, block_w=C // 18 // 32, inplace=False
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )  # fmt: skip

    def old():
        hs = ttnn.to_memory_config(h, shard)
        x = ttnn.rms_norm(
            hs, weight=norm_w, epsilon=1e-6, program_config=prog, compute_kernel_config=ckc, memory_config=shard
        )
        ttnn.deallocate(hs)
        out = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(x)
        return out

    fused = FusedRowNorm(device, [w.float()], out_l1=True)
    a, us_old = traced(device, old)
    b, us_new = traced(device, lambda: fused(h)[0])
    ref = h_t = ttnn.to_torch(h).float().reshape(-1)[:C]
    want = (h_t * torch.rsqrt(h_t.pow(2).mean() + 1e-6) * w.float()).bfloat16().float()
    pcc = lambda x, y: torch.corrcoef(torch.stack([x.double(), y.double()]))[0, 1].item()  # noqa: E731
    res = dict(
        name="moe_norm", us_3ops=us_old, us_fused=us_new, n_diff=int((a != b).sum()), max_diff=float((a - b).abs().max()),
        pcc_old_ref=pcc(a, want), pcc_fused_ref=pcc(b, want), n_diff_old_ref=int((a != want).sum()),
        n_diff_fused_ref=int((b != want).sum()),
    )  # fmt: skip
    del ref
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(res) + "\n")
    print(res)
    assert res["pcc_fused_ref"] > 0.9999, res
