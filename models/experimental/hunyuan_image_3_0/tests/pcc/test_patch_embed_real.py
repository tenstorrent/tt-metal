# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — TTNN UNetDown (patch_embed) / UNetUp (final_layer) vs the PyTorch
# reference, on the REAL HunyuanImage-3.0 checkpoint weights.
#
# Unlike test_patch_embed.py (random weights at config shapes), this loads the
# actual `patch_embed.*` / `final_layer.*` tensors from the sharded safetensors,
# runs the fp32 reference and the bf16 TT port on identical inputs, and gates on
# PCC. The timestep conditioning `t_emb` is a random [B, hidden] vector (i.e. a
# stand-in for time_embed(timesteps)); both paths receive the same one, so this
# isolates the patch-projection numerics.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_patch_embed_real.py -v
#   python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_patch_embed_real.py

import glob
import json
import sys

import torch
from safetensors import safe_open

ROOT = "/home/iguser/ign-tt/tt-metal"
HUNYUAN = "/home/iguser/ign-tt/hunyan_instruct"
WEIGHTS = "/home/iguser/ign-tt/base"
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp

PCC_THR = 0.99
PATCH, B, GRID = 1, 1, 16  # latent grid (e.g. 256px / 16 downsample)


# --- sharded weight loading (same approach as test_model.py) ----------------
_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _dims(down_sd):
    # Derive shapes from the weights (config.json omits patch_embed_hidden_dim):
    #   model.0.weight        : [HID, LATENT, 3, 3]
    #   model.1.in_layers.2.w : [HSZ, HID, 3, 3]
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def _run(device):
    down_sd = _load_prefix("patch_embed")  # keys: model.0.*, model.1.*
    up_sd = _load_prefix("final_layer")  # keys: model.0.*, model.1.*
    LATENT, HID, HSZ = _dims(down_sd)
    print(f"config: latent={LATENT} hidden={HID} hsize={HSZ}  grid={GRID}x{GRID}")

    # ---- reference (fp32) ----
    ref_down = RefDown(PATCH, LATENT, HSZ, HID, HSZ).eval()
    ref_up = RefUp(PATCH, HSZ, HSZ, HID, LATENT, out_norm=True).eval()
    ref_down.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ref_up.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)

    torch.manual_seed(0)
    x = torch.randn(B, LATENT, GRID, GRID)
    t = torch.randn(B, HSZ)

    with torch.no_grad():
        ref_seq, th, tw = ref_down(x, t)
        ref_lat = ref_up(ref_seq, t, th, tw)

    # ---- TT (bf16) on the same real weights ----
    tt_down = HunyuanTtUNetDown(
        device,
        {f"patch_embed.{k}": v for k, v in down_sd.items()},
        in_channels=LATENT,
        hidden_channels=HID,
        out_channels=HSZ,
    )
    tt_up = HunyuanTtUNetUp(
        device,
        {f"final_layer.{k}": v for k, v in up_sd.items()},
        in_channels=HSZ,
        hidden_channels=HID,
        out_channels=LATENT,
    )
    t_tt = ttnn.from_torch(t.reshape(1, 1, B, HSZ), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    seq_tt, h2, w2 = tt_down(x, t_tt)
    lat_tt, h3, w3 = tt_up(seq_tt, t_tt, h2, w2, B=B)

    seq_t = ttnn.to_torch(seq_tt).reshape(B, th * tw, HSZ)
    lat_t = ttnn.to_torch(lat_tt).reshape(B, h3, w3, LATENT).permute(0, 3, 1, 2)
    return _pcc(ref_seq, seq_t), _pcc(ref_lat, lat_t), (th, tw)


def test_unetdown_real_pcc(device):
    pcc_down, _, _ = _run(device)
    print(f"UNetDown (real weights) PCC={pcc_down:.6f}")
    assert pcc_down >= PCC_THR, f"UNetDown PCC {pcc_down:.6f} < {PCC_THR}"


def test_unetup_real_pcc(device):
    _, pcc_up, _ = _run(device)
    print(f"UNetUp (real weights) PCC={pcc_up:.6f}")
    assert pcc_up >= PCC_THR, f"UNetUp PCC {pcc_up:.6f} < {PCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        pd, pu, (th, tw) = _run(dev)
    finally:
        ttnn.close_device(dev)
    okd, oku = pd >= PCC_THR, pu >= PCC_THR
    print("=" * 64)
    print(f"patch_embed / final_layer — REAL weights, tokens {th}x{tw}")
    print(f"  [{'PASS' if okd else 'FAIL'}] UNetDown PCC={pd:.6f} (>= {PCC_THR})")
    print(f"  [{'PASS' if oku else 'FAIL'}] UNetUp   PCC={pu:.6f} (>= {PCC_THR})")
    print("=" * 64)
    sys.exit(0 if (okd and oku) else 1)
