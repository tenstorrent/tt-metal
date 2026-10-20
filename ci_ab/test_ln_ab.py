# Scratch, not for merge. Single-device layer_norm_post_all_gather (second half of distributed LayerNorm) at LLM shapes;
# each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

LF, H2, H4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
CASES = {  # name: (rows, W, fidelity, fp32_dest, gamma, beta)
    "dec_hf2": (32, 512, H2, False, True, True),
    "p_hf2": (2048, 512, H2, False, True, True),
    "p_lofi": (2048, 512, LF, False, True, True),
    "p_hf4": (2048, 512, H4, False, True, True),
    "p32_hf2": (2048, 1024, H2, False, True, True),
    "p_hf2_f32": (2048, 512, H2, True, True, True),
    "p_hf2_nobeta": (2048, 512, H2, False, True, False),
}


@pytest.mark.parametrize("case", list(CASES))
def test_ln_post(device, case, request):
    rows, W, fid, f32, gamma, beta = CASES[case]
    torch.manual_seed(0)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=f32, packer_l1_acc=False)
    x = torch.randn(1, 1, rows, W).bfloat16()
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mk = lambda: ttnn.from_torch(torch.rand(1, 1, W // 32, 32).bfloat16(), dtype=ttnn.bfloat16,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = mk() if gamma else None
    b = mk() if beta else None
    stats = ttnn.layer_norm_pre_all_gather(tt_x, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
    out = None
    for _ in range(2):
        out = ttnn.layer_norm_post_all_gather(tt_x, stats, epsilon=1e-5, weight=w, bias=b, compute_kernel_config=cfg,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")
