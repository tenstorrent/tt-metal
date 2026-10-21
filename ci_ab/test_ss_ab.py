# Scratch, not for merge. softmax_sharded.cpp (ttnn.scale_mask_softmax_in_place, height sharded); output sha1 to $EB_BITS_OUT.
import hashlib
import os

import pytest
import torch
import ttnn

LF, H2, H4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
CASES = {  # name: (batch, heads, h, w, fidelity, fp32_dest, mask, dtype)
    "ss_mask_hf4": (8, 4, 384, 384, H4, False, True, ttnn.bfloat16),
    "ss_mask_hf2": (8, 4, 384, 384, H2, False, True, ttnn.bfloat16),
    "ss_mask_lofi": (8, 4, 384, 384, LF, False, True, ttnn.bfloat16),
    "ss_mask_hf4_f32": (8, 4, 384, 384, H4, True, True, ttnn.bfloat16),
    "ss_nomask_hf4": (8, 4, 384, 384, H4, False, False, ttnn.bfloat16),
    "ss_nomask_lofi": (8, 4, 384, 384, LF, False, False, ttnn.bfloat16),
    "ss_mask_bfp8_hf2": (8, 4, 384, 384, H2, False, True, ttnn.bfloat8_b),
    "ss_mask_w1024_hf4": (8, 4, 128, 1024, H4, False, True, ttnn.bfloat16),
}


def _bits(request, out):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


@pytest.mark.parametrize("case", list(CASES))
def test_softmax_sharded(device, case, request):
    b, nh, h, w, fid, f32, mask, dt = CASES[case]
    torch.manual_seed(0)
    grid = (b, nh)
    am = torch.rand(b, 1, 1, w)
    am = (am > 0.5).float().masked_fill(am <= 0.5, -1e4).masked_fill(am > 0.5, 0)
    mt = ttnn.from_torch(am, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x = torch.randn(b, nh, h, w).bfloat16()
    mc = ttnn.create_sharded_memory_config(x.shape, core_grid=ttnn.CoreGrid(y=grid[1], x=grid[0]),
                                           strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)
    pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(compute_with_storage_grid_size=grid, subblock_w=6 if not f32 else 3,
                                                   block_h=h // 32, block_w=w // 32)
    cfg = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fid, math_approx_mode=False,
                                                 fp32_dest_acc_en=f32, packer_l1_acc=False)
    out = None
    for _ in range(3):
        xt = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        if mask:
            out = ttnn.scale_mask_softmax_in_place(xt, 1.0, mt, program_config=pc, compute_kernel_config=cfg,
                                                   numeric_stable=True)
        else:
            out = ttnn.scale_mask_softmax_in_place(xt, program_config=pc, compute_kernel_config=cfg, numeric_stable=True)
    ttnn.synchronize_device(device)
    _bits(request, out)
