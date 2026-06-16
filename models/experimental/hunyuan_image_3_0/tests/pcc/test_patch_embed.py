# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — TTNN UNetDown (patch_embed) / UNetUp (final_layer) vs the PyTorch
# reference (ref/image_gen/patch_embed.py), with random weights at the released
# config shapes (patch_size=1, latent=32, hidden=1024, hsize=4096).
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_patch_embed.py -v
#   python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_patch_embed.py

import sys

import torch

ROOT = "/home/iguser/Christy/tt-metal"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp

PCC_THR = 0.99
PATCH, LATENT, HID, HSZ = 1, 32, 1024, 4096
B, GRID = 1, 8


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _prefixed(module, prefix):
    return {f"{prefix}.{k}": v for k, v in module.state_dict().items()}


def _run(device):
    torch.manual_seed(0)
    ref_down = RefDown(PATCH, LATENT, HSZ, HID, HSZ).eval()
    ref_up = RefUp(PATCH, HSZ, HSZ, HID, LATENT, out_norm=True).eval()

    x = torch.randn(B, LATENT, GRID, GRID)
    t = torch.randn(B, HSZ)

    with torch.no_grad():
        ref_seq, th, tw = ref_down(x, t)  # [B, H*W, HSZ]
        ref_lat = ref_up(ref_seq, t, th, tw)  # [B, LATENT, H, W]

    down_sd = _prefixed(ref_down, "patch_embed")
    up_sd = _prefixed(ref_up, "final_layer")

    tt_down = HunyuanTtUNetDown(device, down_sd, in_channels=LATENT, hidden_channels=HID, out_channels=HSZ)
    tt_up = HunyuanTtUNetUp(device, up_sd, in_channels=HSZ, hidden_channels=HID, out_channels=LATENT)

    t_tt = ttnn.from_torch(t.reshape(1, 1, B, HSZ), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    seq_tt, h2, w2 = tt_down(x, t_tt)
    lat_tt, h3, w3 = tt_up(seq_tt, t_tt, h2, w2, B=B)

    # down output: ttnn [1,1,B*H*W,HSZ] == ref [B, H*W, HSZ] (B==1)
    seq_t = ttnn.to_torch(seq_tt).reshape(B, th * tw, HSZ)
    pcc_down = _pcc(ref_seq, seq_t)

    # up output: ttnn NHWC flat [1,1,B*H*W,LATENT] -> [B,H,W,LATENT] -> NCHW
    lat_t = ttnn.to_torch(lat_tt).reshape(B, h3, w3, LATENT).permute(0, 3, 1, 2)
    pcc_up = _pcc(ref_lat, lat_t)

    return pcc_down, pcc_up, (th, tw)


def test_unetdown_pcc(device):
    pcc_down, _, _ = _run(device)
    print(f"UNetDown PCC={pcc_down:.6f}")
    assert pcc_down >= PCC_THR, f"UNetDown PCC {pcc_down:.6f} < {PCC_THR}"


def test_unetup_pcc(device):
    _, pcc_up, _ = _run(device)
    print(f"UNetUp PCC={pcc_up:.6f}")
    assert pcc_up >= PCC_THR, f"UNetUp PCC {pcc_up:.6f} < {PCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        pd, pu, (th, tw) = _run(dev)
    finally:
        ttnn.close_device(dev)
    okd, oku = pd >= PCC_THR, pu >= PCC_THR
    print("=" * 64)
    print(f"patch_embed / final_layer — latent {B}x{LATENT}x{GRID}x{GRID}, tokens {th}x{tw}")
    print(f"  [{'PASS' if okd else 'FAIL'}] UNetDown PCC={pd:.6f} (>= {PCC_THR})")
    print(f"  [{'PASS' if oku else 'FAIL'}] UNetUp   PCC={pu:.6f} (>= {PCC_THR})")
    print("=" * 64)
    sys.exit(0 if (okd and oku) else 1)
