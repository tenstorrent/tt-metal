# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch CI only (llk_analysis #58790): the production Blackhole SDPA calls whose streaming QK or softmax pack is 3 tiles
wide, at single-device equivalents of their per-device shapes, chunks and compute configs. Prints a sha256 of each
output for a bitwise compare between two CI builds."""

import hashlib
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

F = ttnn.MathFidelity
# id: (Q shape, K/V shape, q_chunk, k_chunk, causal, fidelity (None = op default), math_approx, packer_l1_acc, exp_approx)
CASES = {
    "minimax_h3_vae": ((1, 32, 1797, 64), (1, 32, 1797, 64), 192, 192, False, F.HiFi2, False, False, False),
    "sdxl_base_xattn_640": ((1, 10, 4096, 64), (1, 10, 77, 64), 128, 128, False, F.LoFi, False, True, False),
    "sdxl_base_xattn_1280": ((1, 20, 1024, 64), (1, 20, 77, 64), 128, 128, False, F.LoFi, False, True, False),
    "sdxl_ref_xattn_768": ((1, 12, 4096, 64), (1, 12, 77, 64), 128, 128, False, F.LoFi, False, True, False),
    "sdxl_ref_xattn_1536": ((1, 24, 1024, 64), (1, 24, 77, 64), 128, 128, False, F.LoFi, False, True, False),
    "sdxl_ref_xattn_mid": ((1, 24, 256, 64), (1, 24, 77, 64), 128, 128, False, F.LoFi, False, True, False),
    "wan720_glx_shard7": ((1, 10, 9472, 128), (1, 10, 9296, 128), 288, 512, False, F.HiFi2, False, False, False),
    "wan720_lb_shard3": ((1, 20, 18912, 128), (1, 20, 18864, 128), 256, 256, False, F.HiFi2, False, False, False),
    "ltx_s1_145f_glx": ((1, 8, 1216, 128), (1, 8, 1216, 128), 128, 512, False, F.HiFi2, False, False, False),
    "ltx_s1_145f_lb_shard3": ((1, 16, 2432, 128), (1, 16, 2394, 128), 256, 256, False, F.HiFi2, False, False, False),
    "ltx_i2v_smoke_s2_glx": ((1, 8, 192, 128), (1, 8, 192, 128), 128, 512, False, F.HiFi2, False, False, False),
    "mmh3_10s_glx_shard7": ((1, 14, 9600, 128), (1, 14, 6225, 128), 256, 512, False, F.HiFi2, False, False, False),
    "mmh3_15s_glx_shard7": ((1, 14, 13952, 128), (1, 14, 11437, 128), 256, 512, False, F.HiFi2, False, False, False),
    "mmh3_5s_quad_shard27": ((1, 14, 1376, 128), (1, 14, 597, 128), 128, 512, False, F.HiFi2, False, False, False),
    "mmh3_10s_quad_shard27": ((1, 14, 2688, 128), (1, 14, 849, 128), 256, 384, False, F.HiFi2, False, False, False),
    "mmh3_15s_quad_local": ((1, 14, 3776, 128), (1, 14, 3776, 128), 192, 512, False, F.HiFi2, False, False, False),
    "ideogram4_1024_lb_n80": ((1, 9, 1536, 256), (1, 9, 1104, 256), 128, 256, False, F.HiFi2, False, False, False),
    "mochi_glx_L80": ((1, 6, 5568, 128), (1, 6, 80, 128), 128, 512, False, F.HiFi2, False, False, False),
    "qwen38_t96_causal": ((1, 6, 96, 256), (1, 1, 96, 256), 32, 128, True, None, None, None, True),
}
# torch reference only where the score matrix is small; the gate is the bitwise compare against main
REF_LIMIT = 2**27


@pytest.mark.parametrize("case", list(CASES))
def test_w3_sdpa(device, case):
    qs, ks, qc, kc, causal, fid, approx, l1, expa = CASES[case]
    calls = int(os.environ.get("W3_CALLS", "1"))
    torch.manual_seed(0)
    q, k, v = torch.randn(qs, dtype=torch.bfloat16), torch.randn(ks, dtype=torch.bfloat16), torch.randn(ks, dtype=torch.bfloat16)
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=expa,
    )
    ckc = (
        None
        if fid is None
        else ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=False, packer_l1_acc=l1
        )
    )
    concat = case == "minimax_h3_vae"
    out = None
    for _ in range(calls):
        if out is not None:
            ttnn.deallocate(out)
        out = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=causal, program_config=pc, compute_kernel_config=ckc, output_concat_heads=concat
        )
    if os.environ.get("W3_LOG"):
        with open(os.environ["W3_LOG"], "a") as f:
            f.write(f"{case} {calls}\n")
    if os.environ.get("W3_PROF") and hasattr(ttnn, "ReadDeviceProfiler"):
        ttnn.ReadDeviceProfiler(device)
    res = ttnn.to_torch(out)
    print(f"W3HASH {case} {tuple(res.shape)} {hashlib.sha256(res.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()}")
    if os.environ.get("BITID_OUT"):
        os.makedirs(os.environ["BITID_OUT"], exist_ok=True)
        torch.save({"out": res}, os.path.join(os.environ["BITID_OUT"], f"{case}.pt"))
    if qs[1] * qs[2] * ks[2] <= REF_LIMIT:
        kr, vr = k.float(), v.float()
        if ks[1] != qs[1]:
            kr, vr = kr.repeat_interleave(qs[1] // ks[1], dim=1), vr.repeat_interleave(qs[1] // ks[1], dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), kr, vr, is_causal=causal)
        if concat:
            ref = ref.permute(0, 2, 1, 3).reshape(qs[0], 1, qs[2], qs[1] * qs[3])
        passing, msg = comp_pcc(ref, res.float(), 0.99)
        assert passing, f"{case}: {msg}"
