# Scratch, not for merge. softmax_large_tensor.cpp (rows too wide for the small kernel); each output's sha1 goes to $EB_BITS_OUT.
import hashlib
import os

import pytest
import torch
import ttnn

LF, H2, H4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
CASES = {  # name: (rows, W, fidelity, fp32_dest, numeric_stable, mask)
    "sl_plain_hf4": (512, 32000, H4, False, False, False),
    "sl_ns_hf4": (512, 32000, H4, False, True, False),
    "sl_ns_hf2": (512, 32000, H2, False, True, False),
    "sl_ns_lofi": (512, 32000, LF, False, True, False),
    "sl_ns_hf4_f32": (512, 32000, H4, True, True, False),
    "sl_mask_ns_hf4": (512, 32000, H4, False, True, True),
    "sl_mask_hf2": (512, 32000, H2, False, False, True),
    "sl_mask_ns_lofi": (512, 32000, LF, False, True, True),
}


def _bits(request, out):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


@pytest.mark.parametrize("case", list(CASES))
def test_softmax_large(device, case, request):
    rows, W, fid, f32, ns, mask = CASES[case]
    torch.manual_seed(0)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=f32, packer_l1_acc=False)
    x = ttnn.from_torch(torch.randn(1, 1, rows, W).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device)
    out = None
    if mask:
        m = torch.rand(1, 1, 1, W)
        m = (m > 0.5).float().masked_fill(m <= 0.5, -1e4).masked_fill(m > 0.5, 0)
        mt = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for _ in range(2):
            out = ttnn.scale_mask_softmax(x, 0.5, mt, compute_kernel_config=cfg, numeric_stable=ns)
    else:
        for _ in range(2):
            out = ttnn.softmax(x, dim=-1, compute_kernel_config=cfg, numeric_stable=ns)
    ttnn.synchronize_device(device)
    _bits(request, out)
