import pytest
import torch
import torch.nn.functional as F

import ttnn


@pytest.mark.parametrize("approx", [False, True])
@pytest.mark.parametrize("scale", [1, 4])
def test_probe(device, approx, scale):
    torch.manual_seed(0)
    torch_input = torch.randn((1, 16, 26, 26), dtype=torch.bfloat16) * scale
    golden = F.softmax(torch_input.float(), dim=-1).to(torch.bfloat16)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=approx, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.softmax(tt_input, dim=-1, compute_kernel_config=cfg))
    out_def = ttnn.to_torch(ttnn.softmax(tt_input, dim=-1))
    r = (out == golden).float().mean().item()
    rd = (out_def == golden).float().mean().item()
    ulp = ((out.float() - golden.float()).abs() / golden.float().abs().clamp_min(1e-30)).max().item()
    print(f"PROBE approx={approx} scale={scale} explicit_correct={r:.4f} default_correct={rd:.4f} max_rel={ulp:.4g}")
