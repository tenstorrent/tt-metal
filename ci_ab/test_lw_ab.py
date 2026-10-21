# Scratch, not for merge. Interleaved Welford layer_norm (layernorm_welford.cpp); each output's sha1 goes to $EB_BITS_OUT.
import hashlib
import os

import pytest
import torch
import ttnn

LF, H2, H4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
CASES = {  # name: (rows, W, fidelity, fp32_dest, gamma, beta, residual, dtype)
    "l_hf4_f32": (2048, 1024, H4, True, True, True, False, "bf16"),
    "l_hf2": (2048, 1024, H2, False, True, True, False, "bf16"),
    "l_lofi": (2048, 1024, LF, False, True, True, False, "bf16"),
    "l_hf4_f32_w2k": (1024, 2048, H4, True, True, True, False, "bf16"),
    "l_hf2_nobeta": (2048, 1024, H2, False, True, False, False, "bf16"),
    "l_hf2_plain": (2048, 1024, H2, False, False, False, False, "bf16"),
    "l_hf2_res": (2048, 1024, H2, False, True, True, True, "bf16"),
    "l_lofi_w4k": (512, 4096, LF, False, True, True, False, "bf16"),
    "l_hf4_f32in": (1024, 1024, H4, True, True, True, False, "f32"),
}


def _bits(request, out):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


@pytest.mark.parametrize("case", list(CASES))
def test_ln_welford(device, case, request):
    rows, W, fid, f32, gamma, beta, res, dt = CASES[case]
    torch.manual_seed(0)
    tdt, ddt = (torch.bfloat16, ttnn.bfloat16) if dt == "bf16" else (torch.float32, ttnn.float32)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=f32, packer_l1_acc=False)
    mk = lambda shape: ttnn.from_torch(torch.randn(*shape).to(tdt), dtype=ddt, layout=ttnn.TILE_LAYOUT, device=device,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = mk((1, 1, rows, W))
    r = mk((1, 1, rows, W)) if res else None
    w = mk((1, 1, 1, W)) if gamma else None
    b = mk((1, 1, 1, W)) if beta else None
    grid = device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    recip = ttnn.create_layer_norm_reciprocals(device, crs, W)
    pc = ttnn.LayerNormDefaultProgramConfig(use_welford=True)
    out = None
    for _ in range(2):
        out = ttnn.layer_norm(x, epsilon=1e-5, weight=w, bias=b, residual_input_tensor=r, program_config=pc,
                              compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, recip_tensor=recip)
    ttnn.synchronize_device(device)
    _bits(request, out)
