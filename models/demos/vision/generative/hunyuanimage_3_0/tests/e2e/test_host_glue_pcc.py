# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 PCC gate for the host-glue TT port: `final_layer` (velocity head) on TT vs host.

HunyuanImage-3.0's hybrid gen_image path currently runs `final_layer` (a UNetUp:
ResBlock(4096->1024) + GroupNorm/SiLU/conv(1024->32)) on the HOST CPU each diffusion
step, and round-trips the [1,4116,4096] hidden device<->host. Porting it to TTNN
(host_glue_tt.FinalLayerTT) lets the transformer hidden stay ON-DEVICE and downloads
only the small velocity [1,32,64,64] — the #1 perf lever (~57% "other").

This gate feeds an identical random post-ln_f image hidden [1, H*W, 4096] + timestep
embedding into the host torch `final_layer` and the TTNN `FinalLayerTT`, and asserts:

    velocity PCC(host, TT) >= 0.99   (VAE-resnet reference bar; bf8_b conv weights + bf16 acts)

Only `final_layer` + `time_embed_2` weights matter here, so load at reduced depth.
Runs on a single chip by default (port is replicated / single-device-equivalent) for
fast iteration; set HUNYUAN_HG_MESH="4,8" to validate on the full mesh.

Run:  ./python_env/bin/python -m pytest \
        models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_host_glue_pcc.py -s
Env:  HUNYUAN_HG_MESH (default "1,1"), HUNYUAN_GENIMG_NUM_LAYERS (default 2),
      HUNYUAN_HG_TIMESTEP (default 500), HUNYUAN_HG_PCC (default 0.99)
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.vision.generative.hunyuanimage_3_0.tt import gen_image as gi
from models.demos.vision.generative.hunyuanimage_3_0.tt.host_glue_tt import _repl, build_final_layer, build_patch_embed
from models.demos.vision.generative.hunyuanimage_3_0.tt.pipeline import _mesh_to_torch

_MESH = tuple(int(x) for x in os.environ.get("HUNYUAN_HG_MESH", "1,1").split(","))
PCC_TARGET = float(os.environ.get("HUNYUAN_HG_PCC", "0.99"))
N_LAYERS = int(os.environ.get("HUNYUAN_GENIMG_NUM_LAYERS", "2"))
TIMESTEP = float(os.environ.get("HUNYUAN_HG_TIMESTEP", "500"))
TOKEN_H = TOKEN_W = 64  # 1024^2 latent -> 64x64 image tokens

_DEV = {"l1_small_size": 24576}
if _MESH[0] * _MESH[1] > 1:  # fabric is multi-chip only; single chip opens without it
    _DEV["fabric_config"] = ttnn.FabricConfig.FABRIC_1D


@pytest.mark.parametrize("device_params", [_DEV], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_final_layer_pcc(device_params, mesh_device):
    torch.manual_seed(0)
    model = gi.load_model(num_layers=N_LAYERS)
    fl = model.final_layer.float()
    C = int(fl.model[0].in_layers[2].weight.shape[1])  # transformer dim (4096)

    # identical inputs to host + TT
    x = torch.randn(1, TOKEN_H * TOKEN_W, C, dtype=torch.float32)  # post-ln_f image hidden tokens
    try:
        emb = model.time_embed_2(torch.tensor([TIMESTEP], dtype=torch.float32)).float()
        emb_src = "time_embed_2"
    except Exception as e:  # noqa: BLE001 - fall back to random emb (still same for host+TT)
        emb = torch.randn(1, C, dtype=torch.float32)
        emb_src = f"random(fallback: {type(e).__name__}: {e})"
    print(f"[hg] emb_src={emb_src} emb={tuple(emb.shape)} x={tuple(x.shape)} C={C}", flush=True)

    with torch.no_grad():
        vel_host = fl(x, emb, TOKEN_H, TOKEN_W)  # [1,32,64,64] NCHW
    print(f"[hg] host velocity={tuple(vel_host.shape)}", flush=True)

    fl_tt = build_final_layer(mesh_device, model, TOKEN_H, TOKEN_W)
    x_tt = _repl(mesh_device, x)  # [1,H*W,C] bf16 TILE (replicated)
    emb_tt = _repl(mesh_device, emb)  # [1,C]     bf16 TILE (replicated)
    out = fl_tt(x_tt, emb_tt)  # [1,1,H*W,32] NHWC-flat
    out_t = _mesh_to_torch(out, mesh_device).to(torch.float32)
    vel_tt = out_t.reshape(1, TOKEN_H, TOKEN_W, 32).permute(0, 3, 1, 2).contiguous()  # -> [1,32,64,64]
    print(f"[hg] tt velocity={tuple(vel_tt.shape)}", flush=True)

    ok, pcc = comp_pcc(vel_host, vel_tt, PCC_TARGET)
    print(f"HOST_GLUE_FINAL_LAYER_PCC={pcc} target={PCC_TARGET} ok={ok}", flush=True)
    assert ok, f"final_layer TT-vs-host velocity PCC {pcc} < {PCC_TARGET}"


@pytest.mark.parametrize("device_params", [_DEV], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_patch_embed_pcc(device_params, mesh_device):
    """Stage-2: patch_embed (UNetDown) TT vs host. VAE latent [1,32,64,64] + time_embed(t)
    -> image tokens [1, H*W, 4096]; PCC(host, TT) >= 0.99."""
    torch.manual_seed(0)
    model = gi.load_model(num_layers=N_LAYERS)
    pe = model.patch_embed.float()
    in_ch = int(pe.model[0].weight.shape[1])  # VAE latent channels (32)

    latent = torch.randn(1, in_ch, TOKEN_H, TOKEN_W, dtype=torch.float32)  # NCHW VAE latent
    try:
        emb = model.time_embed(torch.tensor([TIMESTEP], dtype=torch.float32)).float()
        emb_src = "time_embed"
    except Exception as e:  # noqa: BLE001
        emb = torch.randn(1, int(pe.model[1].in_layers[2].weight.shape[0]), dtype=torch.float32)
        emb_src = f"random(fallback: {type(e).__name__}: {e})"
    print(f"[hg] patch_embed emb_src={emb_src} emb={tuple(emb.shape)} latent={tuple(latent.shape)}", flush=True)

    with torch.no_grad():
        tok_host, th, tw = pe(latent, emb)  # [1, H*W, 4096]
    print(f"[hg] host tokens={tuple(tok_host.shape)} token_hw=({th},{tw})", flush=True)

    pe_tt = build_patch_embed(mesh_device, model, TOKEN_H, TOKEN_W)
    emb_tt = _repl(mesh_device, emb)
    out = pe_tt(latent, emb_tt)  # [1, H*W, 4096]
    tok_tt = _mesh_to_torch(out, mesh_device).to(torch.float32).reshape(tok_host.shape)
    print(f"[hg] tt tokens={tuple(tok_tt.shape)}", flush=True)

    ok, pcc = comp_pcc(tok_host, tok_tt, PCC_TARGET)
    print(f"HOST_GLUE_PATCH_EMBED_PCC={pcc} target={PCC_TARGET} ok={ok}", flush=True)
    assert ok, f"patch_embed TT-vs-host tokens PCC {pcc} < {PCC_TARGET}"
