# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for the ttnn HunyuanVideo15 VAE decoder port (tt/vae_decoder.py).

Validates individual blocks and the full decoder against the real diffusers
`AutoencoderKLHunyuanVideo15` weights on a single device.

    pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_vae_decoder.py -s
"""
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.hunyuanvideo_1_5.tt.vae_decoder import (
    AttnBlock,
    HunyuanVideo15Decoder,
    ResnetBlock,
    RMSNorm,
    Upsample,
)

_COMMUNITY = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"


def _load_decoder():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from diffusers import AutoencoderKLHunyuanVideo15
    from huggingface_hub import snapshot_download

    path = snapshot_download(_COMMUNITY)
    vae = AutoencoderKLHunyuanVideo15.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32).eval()
    return vae


def _to_bthwc(dev, x):
    return ttnn.from_torch(
        x.permute(0, 2, 3, 4, 1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )


def _from_bthwc(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().permute(0, 4, 1, 2, 3).contiguous()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_vae_decoder_blocks(mesh_device):
    """Per-block PCC: RMSNorm, ResnetBlock, AttnBlock, Upsample (spatial + temporal)."""
    torch.manual_seed(0)
    dec = _load_decoder().decoder
    dev = mesh_device

    def check(name, ref_mod, tt_mod, x, thr=0.99):
        with torch.no_grad():
            y_ref = ref_mod(x).float()
        y_tt = _from_bthwc(tt_mod(_to_bthwc(dev, x)))
        ok, pcc = comp_pcc(y_ref, y_tt, thr)
        print(f"[vae {name}] PCC={pcc} -> {'PASS' if ok else 'FAIL'}", flush=True)
        assert ok, f"{name} PCC {pcc} < {thr}"

    norm = dec.norm_out
    check("RMSNorm", norm, RMSNorm(norm.gamma.detach(), device=dev), torch.randn(1, norm.gamma.shape[0], 2, 8, 8))

    rb = dec.up_blocks[-1].resnets[-1]
    check("ResnetBlock", rb, ResnetBlock(rb, device=dev), torch.randn(1, rb.conv1.conv.in_channels, 2, 8, 8))

    attn = dec.mid_block.attentions[0]
    check("AttnBlock", attn, AttnBlock(attn, device=dev), torch.randn(1, attn.in_channels, 2, 4, 4))

    up = dec.up_blocks[3].upsamplers[0]
    check("Upsample-spatial", up, Upsample(up, device=dev), torch.randn(1, up.conv.conv.in_channels, 2, 6, 6))

    up = dec.up_blocks[0].upsamplers[0]
    check("Upsample-temporal", up, Upsample(up, device=dev), torch.randn(1, up.conv.conv.in_channels, 2, 6, 6))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_vae_decoder_full(mesh_device):
    """Full decoder PCC vs CPU on a tiny latent (2,8,8) -> (5,128,128)."""
    torch.manual_seed(0)
    vae = _load_decoder()
    dev = mesh_device

    z = torch.randn(1, vae.config.latent_channels, 2, 8, 8)
    with torch.no_grad():
        dec_ref = vae.decode(z, return_dict=False)[0].float()

    dec_tt = HunyuanVideo15Decoder(vae.decoder, device=dev)
    out = _from_bthwc(dec_tt(_to_bthwc(dev, z)))

    ok, pcc = comp_pcc(dec_ref, out, 0.95)
    print(f"[vae full decode] {tuple(out.shape)} PCC={pcc} -> {'PASS' if ok else 'FAIL'}", flush=True)
    assert ok, f"full decode PCC {pcc} < 0.95"
