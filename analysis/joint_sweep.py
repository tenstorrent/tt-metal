# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-only joint SDPA (SD3/Flux) sweep for perf-counter capture (TEN-4716 joint validation).

joint_scaled_dot_product_attention(q,k,v, joint_q,joint_k,joint_v) attends over the concatenated
[main + joint] sequence (non-causal), so compute == one non-causal SDPA over S_eff = seq + joint.
Mirrors analysis/sdpa_sweep.py. Env: JT_NH, JT_D, JT_SEQS (csv), JT_JOINT, JT_QC, JT_KC, JT_ITERS.
"""
from __future__ import annotations
import os
import time
import pytest


@pytest.mark.parametrize("seq_len", [int(x) for x in os.environ.get("JT_SEQS", "2048,4096,8192").split(",")])
def test_joint_sweep(device, seq_len):
    import torch
    import ttnn

    nh = int(os.environ.get("JT_NH", "16"))
    d = int(os.environ.get("JT_D", "128"))
    joint = int(os.environ.get("JT_JOINT", "512"))
    qc = int(os.environ.get("JT_QC", "128"))
    kc = int(os.environ.get("JT_KC", "128"))
    b = 1

    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=True,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    def mk(s):
        return ttnn.from_torch(
            torch.randn(b, nh, s, d),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    q, k, v = mk(seq_len), mk(seq_len), mk(seq_len)
    jq, jk, jv = mk(joint), mk(joint), mk(joint)
    scale = d**-0.5
    iters = int(os.environ.get("JT_ITERS", "40"))

    def run():
        o, jo = ttnn.transformer.joint_scaled_dot_product_attention(
            q, k, v, jq, jk, jv, joint_strategy="rear", program_config=pc, compute_kernel_config=ck
        )
        o.deallocate()
        jo.deallocate()

    for _ in range(3):
        run()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    ttnn.synchronize_device(device)
    lat_us = (time.perf_counter() - t0) / iters * 1e6
    print(
        f"[joint_sweep] S={seq_len} joint={joint} nh={nh} d={d} S_eff={seq_len+joint} lat_us={lat_us:.1f}", flush=True
    )
    for t in (q, k, v, jq, jk, jv):
        t.deallocate()
