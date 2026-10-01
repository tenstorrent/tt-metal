# Single-chip A/B of the 145f conv-decoder blockings changed in 9a02249fdfb (old 4x8/8x4 vs 153f-sweep 16x2).
# Per-device input shapes of the 4x8 decode at 1080p/145f; one chip suffices because conv3d is device-local.
# Run: pytest tmp/t17/test_blk_ab.py -s --timeout=600
import pytest
import torch

import ttnn
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import (
    MATH_FIDELITY,
    TRACE_REGION_SIZE,
    _invoke,
    _trace_us,
)
from models.tt_dit.utils.conv3d import aligned_channels

# (name, C_in, C_out, T_in, H_in, W_in, old blocking, new blocking)
_SITES = [
    ("s1_res", 512, 512, 39, 19, 17, (64, 256, 1, 4, 8), (64, 256, 1, 16, 2)),
    ("s1_up", 512, 4096, 39, 19, 17, (128, 64, 5, 4, 8), (128, 64, 5, 16, 2)),
    ("s2_res", 512, 512, 75, 36, 32, (64, 256, 1, 8, 4), (64, 256, 1, 16, 2)),
    ("s3_res", 256, 256, 147, 36, 32, (64, 256, 1, 8, 4), (64, 256, 1, 16, 2)),
    ("s3_chg", 256, 512, 147, 36, 32, (64, 256, 1, 8, 4), (64, 256, 1, 16, 2)),
]


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["1x1"],
    indirect=True,
)
def test_blk_ab(mesh_device):
    device = mesh_device
    grid = device.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=MATH_FIDELITY, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    kernel = (3, 3, 3)
    rows = []
    for name, C_in, C_out, T, H, W, old, new in _SITES:
        torch.manual_seed(0)
        cin_p = aligned_channels(C_in)
        x = ttnn.from_torch(
            torch.randn(1, T, H, W, cin_p), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        w_host = torch.randn(C_out, cin_p, *kernel) * 0.02
        b = ttnn.from_torch(torch.randn(1, C_out), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        res = {}
        for tag, (cin, cout, tb, hb, wb) in (("old", old), ("new", new)):
            cfg = ttnn.Conv3dConfig(
                weights_dtype=ttnn.bfloat16,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                T_out_block=tb,
                W_out_block=wb,
                H_out_block=hb,
                C_out_block=cout,
                C_in_block=cin,
                compute_with_storage_grid_size=grid,
            )
            w = ttnn.from_torch(w_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            wp = ttnn.experimental.prepare_conv3d_weights(weight_tensor=w, C_in_block=cin, device=device)
            args = (device, x, wp, b, cfg, C_out, kernel, (1, 1, 1), (0, 0, 0), ckc)
            out = _invoke(args)
            res[tag] = (ttnn.to_torch(out).float(), _trace_us(args))
            ttnn.deallocate(out)
            ttnn.deallocate(wp)
        (o_old, us_old), (o_new, us_new) = res["old"], res["new"]
        rows.append(
            (name, us_old, us_new, torch.equal(o_old, o_new), _pcc(o_old, o_new), (o_old - o_new).abs().max().item())
        )
        print(
            f"AB {name}: old {us_old:.0f}us new {us_new:.0f}us identical={rows[-1][3]} pcc={rows[-1][4]:.6f}",
            flush=True,
        )
        for t in (x, b):
            ttnn.deallocate(t)
    print("AB_TABLE name old_us new_us identical pcc maxabs")
    for r in rows:
        print(f"AB_ROW {r[0]} {r[1]:.0f} {r[2]:.0f} {r[3]} {r[4]:.6f} {r[5]:.3g}")
    print(f"AB_TOTAL old {sum(r[1] for r in rows):.0f}us new {sum(r[2] for r in rows):.0f}us")
    assert all(r[4] >= 0.999 for r in rows)
