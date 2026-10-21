# Scratch, not for merge. Single-device Welford post all-gather kernels: ttnn.layer_norm_post_all_gather(use_welford) and
# ttnn.experimental.dit_layernorm_post_allgather; each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

LF, H2, H4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
CASES = {  # name: (rows, W, fidelity, fp32_dest, gamma, beta)
    "p_hf4_f32": (2048, 1024, H4, True, True, True),
    "p_hf2": (2048, 1024, H2, False, True, True),
    "p_lofi": (2048, 1024, LF, False, True, True),
    "p_hf4_f32_w2k": (1024, 2048, H4, True, True, True),
    "p_hf2_nobeta": (2048, 1024, H2, False, True, False),
}


def _bits(request, out):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


def _setup(device, rows, W, fid, f32, gamma, beta, rm):
    torch.manual_seed(0)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=f32, packer_l1_acc=False)
    x = torch.randn(1, 1, rows, W).bfloat16()
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if rm:
        mk = lambda: ttnn.from_torch(torch.rand(1, 1, W // 32, 32).bfloat16(), dtype=ttnn.bfloat16,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        mk = lambda: ttnn.from_torch(torch.rand(1, 1, 1, W).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    grid = device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    recip = ttnn.create_layer_norm_reciprocals(device, crs, W)
    return cfg, tt_x, (mk() if gamma else None), (mk() if beta else None), recip


@pytest.mark.parametrize("case", list(CASES))
def test_ln_welford_post(device, case, request):
    rows, W, fid, f32, gamma, beta = CASES[case]
    cfg, tt_x, w, b, recip = _setup(device, rows, W, fid, f32, gamma, beta, rm=True)
    pc = ttnn.LayerNormDefaultProgramConfig(use_welford=True)
    stats = ttnn.layer_norm_pre_all_gather(tt_x, compute_kernel_config=cfg, dtype=ttnn.bfloat16, program_config=pc,
                                           recip_tensor=recip)
    out = None
    for _ in range(2):
        out = ttnn.layer_norm_post_all_gather(tt_x, stats, epsilon=1e-5, weight=w, bias=b, compute_kernel_config=cfg,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc)
    ttnn.synchronize_device(device)
    _bits(request, out)


@pytest.mark.parametrize("case", list(CASES))
def test_dit_post(device, case, request):
    rows, W, fid, f32, gamma, beta = CASES[case]
    cfg, tt_x, w, b, recip = _setup(device, rows, W, fid, f32, gamma, beta, rm=False)
    stats = ttnn.experimental.dit_layernorm_pre_allgather(tt_x, recip, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
    out = None
    for _ in range(2):
        out = ttnn.experimental.dit_layernorm_post_allgather(tt_x, stats, epsilon=1e-5, weight=w, bias=b,
                                                             compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    _bits(request, out)
