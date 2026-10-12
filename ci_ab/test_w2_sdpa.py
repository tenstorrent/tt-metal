# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Farm-only (llk_analysis #58790, threshold 2): production Blackhole SDPA calls whose streaming pack rows are 2 tiles
wide (a survey of main), at single-device equivalents of their per-device shapes, chunks and compute configs. W3_CALLS calls per case in
one process (the first compiles); BITID_OUT saves the last output; W3_LOG records the calls issued per case, in order."""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

F = ttnn.MathFidelity
# id: (Q shape, K/V shape, q_chunk, k_chunk, causal, fidelity (None = op default), math_approx, packer_l1_acc, exp_approx)
CASES = {
    # every P V row 2 tiles wide (V head dim 64), SALAD batches of 2 from a Q chunk's second K chunk
    "sdxl_base_self_db1": ((1, 10, 4096, 64), (1, 10, 4096, 64), 128, 512, False, F.HiFi2, False, True, False),
    "sdxl_base_self_ub0": ((1, 20, 1024, 64), (1, 20, 1024, 64), 128, 1024, False, F.HiFi2, False, True, False),
    "sdxl_ref_self_1024": ((1, 24, 1024, 64), (1, 24, 1024, 64), 128, 512, False, F.LoFi, False, True, False),
    "sdxl_ref_self_mid": ((1, 24, 256, 64), (1, 24, 256, 64), 128, 512, False, F.LoFi, False, True, False),
    "whisper_enc_self": ((2, 20, 1500, 64), (2, 20, 1500, 64), 256, 256, False, F.HiFi2, True, True, True),
    "whisper_cross_decode": ((2, 20, 1, 64), (2, 20, 1500, 64), 256, 256, False, F.HiFi2, True, True, True),
    "gpt_oss20b_p150_s128": ((1, 64, 128, 64), (1, 8, 128, 64), 32, 32, True, F.HiFi4, False, False, False),
    "gpt_oss_qb2_s2048": ((1, 16, 2048, 64), (1, 2, 2048, 64), 256, 256, True, F.HiFi4, False, False, False),
    "ltx_audio_text_xattn": ((1, 8, 32, 64), (1, 8, 1024, 64), 256, 256, False, F.HiFi2, False, False, False),
    "ltx_a2v_s1": ((1, 8, 1216, 64), (1, 8, 256, 64), 128, 128, False, F.HiFi2, False, False, False),
    "xtts_cond_enc": ((1, 16, 345, 64), (1, 16, 345, 64), 32, 32, False, None, None, None, True),
    "xtts_gpt_decode": ((1, 16, 1, 64), (1, 16, 704, 64), 32, 32, False, None, None, None, True),
    "sd35_lb_shard": ((1, 19, 2048, 64), (1, 19, 2048, 64), 256, 512, False, F.HiFi2, False, False, False),
    "motif_glx_shard": ((1, 8, 1280, 64), (1, 8, 1280, 64), 128, 256, False, F.HiFi2, False, False, True),
    # every QK and exp row 2 tiles wide
    "gemma3_4b_p150_s1024": ((1, 8, 1024, 256), (1, 4, 1024, 256), 64, 64, True, F.HiFi4, False, True, False),
    "gemma3_4b_p150_s128": ((1, 8, 128, 256), (1, 4, 128, 256), 64, 64, True, F.HiFi4, False, True, False),
    "ltx_i2v_smoke_s1_self": ((1, 8, 64, 128), (1, 8, 64, 128), 128, 512, False, F.HiFi2, False, False, False),
    # one K chunk narrowed to 2 tiles per Q chunk or shard
    "mochi_glx_local174_q512": ((1, 6, 512, 128), (1, 6, 5568, 128), 128, 512, False, F.HiFi2, False, False, False),
    "mmh3_glx_r66048": ((1, 14, 512, 128), (1, 14, 8256, 128), 256, 512, False, F.HiFi2, False, False, False),
    "qwen_image21_ntext40": ((1, 4, 4096, 128), (1, 4, 4136, 128), 512, 256, False, F.HiFi2, False, False, False),
}
# torch reference only where the score matrix is small; the gate is the bitwise compare against main
REF_LIMIT = 2**27


@pytest.mark.parametrize("case", list(CASES))
def test_w2_sdpa(device, case):
    qs, ks, qc, kc, causal, fid, approx, l1, expa = CASES[case]
    calls = int(os.environ.get("W3_CALLS", "1"))
    torch.manual_seed(int(os.environ.get("W3_SEED", "0")))
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
