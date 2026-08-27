# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-chip PCC ladder for the on-device HunyuanImage-3.0 VAE decode.

Validates each tt primitive (and the full tiny-latent decoder) against the REAL HF
``AutoencoderKLConv3D`` reference modules (random-init) at factor=1, fabric disabled.

Run (on box):
  ./python_env/bin/python -m pytest -o timeout=0 -s \
    models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_vae_decode_pcc.py \
    -k conv_in            # or resnet / attn / upsample_temporal / upsample_spatial / decoder_tiny
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0.tt.vae_decode import (
    AttnBlock3D,
    Conv3dSym,
    MeshCtx,
    ResnetBlock3D,
    UpsampleDCAE,
    VaeDecoder,
    gather_output_ncthw,
    reference_autoencoder_path,
    shard_input_nthwc,
    to_device_nthwc,
    to_host_ncthw,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def _load_hf_ref():
    path = reference_autoencoder_path()
    spec = importlib.util.spec_from_file_location("akl3d_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["akl3d_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


HF = _load_hf_ref()

DEV_PARAMS = [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.DISABLED}]


def _pcc(gt_ncthw, tt_nthwc, out_c, gate):
    y = to_host_ncthw(tt_nthwc, out_c)
    ok, msg = check_with_pcc(gt_ncthw, y, pcc=gate)
    return ok, msg


@pytest.mark.parametrize("device_params", DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_conv_in(device_params, mesh_device):
    """conv_in(32->1024) + z.repeat_interleave(32) residual."""
    dev = mesh_device
    torch.manual_seed(0)
    N, C, T, H, W = 1, 32, 1, 16, 16
    z = torch.randn(N, C, T, H, W)
    conv = HF.Conv3d(32, 1024, kernel_size=3, stride=1, padding=1)
    with torch.no_grad():
        gt = conv(z) + z.repeat_interleave(1024 // 32, dim=1)

    tt_conv = Conv3dSym(conv, dev, T, H, W)
    zt = to_device_nthwc(z, dev)
    h = tt_conv(zt)
    res = ttnn.repeat_interleave(zt, 1024 // 32, dim=4)
    h = ttnn.add(h, res)
    ok, msg = _pcc(gt, h, 1024, 0.99)
    print(f"\nPCC_CONV_IN {msg} -> {'OK' if ok else 'FAIL'}", flush=True)
    assert ok, msg


@pytest.mark.parametrize("device_params", DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("C,T,H,W", [(512, 1, 16, 16), (128, 2, 16, 16)])
def test_resnet(device_params, mesh_device, C, T, H, W):
    dev = mesh_device
    torch.manual_seed(0)
    x = torch.randn(1, C, T, H, W)
    ref = HF.ResnetBlock(C, C).eval()
    with torch.no_grad():
        gt = ref(x)
    tt = ResnetBlock3D(ref.state_dict(), "", C, T, H, W, dev)
    y = tt(to_device_nthwc(x, dev))
    ok, msg = _pcc(gt, y, C, 0.99)
    print(f"\nPCC_RESNET C={C} T={T} H={H} W={W} {msg} -> {'OK' if ok else 'FAIL'}", flush=True)
    assert ok, msg


@pytest.mark.parametrize("device_params", DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_attn(device_params, mesh_device):
    dev = mesh_device
    torch.manual_seed(0)
    C, T, H, W = 1024, 1, 16, 16
    x = torch.randn(1, C, T, H, W)
    ref = HF.AttnBlock(C).eval()
    with torch.no_grad():
        gt = ref(x)
    tt = AttnBlock3D(ref.state_dict(), "", C, T, H, W, dev)
    y = tt(to_device_nthwc(x, dev))
    ok, msg = _pcc(gt, y, C, 0.99)
    print(f"\nPCC_ATTN C={C} T={T} H={H} W={W} {msg} -> {'OK' if ok else 'FAIL'}", flush=True)
    assert ok, msg


@pytest.mark.parametrize("device_params", DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "in_c,out_c,add_temporal,tag",
    [(512, 256, False, "spatial"), (256, 128, True, "temporal")],  # temporal: exercise texp=2 path
)
def test_upsample(device_params, mesh_device, in_c, out_c, add_temporal, tag):
    dev = mesh_device
    torch.manual_seed(0)
    T, H, W = 2, 16, 16
    x = torch.randn(1, in_c, T, H, W)
    ref = HF.UpsampleDCAE(in_c, out_c, add_temporal).eval()
    with torch.no_grad():
        gt = ref(x)
    tt = UpsampleDCAE(ref.state_dict(), "", in_c, out_c, add_temporal, T, H, W, dev)
    y = tt(to_device_nthwc(x, dev))
    ok, msg = _pcc(gt, y, out_c, 0.99)
    print(
        f"\nPCC_UPSAMPLE_{tag.upper()} in={in_c} out={out_c} temporal={add_temporal} {msg} -> {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    assert ok, msg


MESH_DEV_PARAMS = [
    {"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 20000000}
]


def _num_links(mesh_device):
    return 2 if (mesh_device.shape[0] > 1 and mesh_device.shape[1] > 1) else 1


@pytest.mark.parametrize("device_params", MESH_DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("C,T,H,W", [(128, 2, 64, 64)])
def test_resnet_mesh(device_params, mesh_device, C, T, H, W):
    """ResnetBlock on the full (8,4) mesh, H/W-fractured (8x4): exercises the conv H/W
    halo (zeros-boundary) + gather-GroupNorm-partition end-to-end vs torch."""
    dev = mesh_device
    hf, wf = int(dev.shape[0]), int(dev.shape[1])  # H on axis0, W on axis1
    torch.manual_seed(0)
    ctx = MeshCtx(dev, h_factor=hf, h_axis=0, w_factor=wf, w_axis=1, num_links=_num_links(dev))
    x = torch.randn(1, C, T, H, W)
    ref = HF.ResnetBlock(C, C).eval()
    with torch.no_grad():
        gt = ref(x)
    tt = ResnetBlock3D(ref.state_dict(), "", C, T, H, W, dev, ctx=ctx)
    xt = shard_input_nthwc(x, ctx)
    yt = tt(xt)
    y = gather_output_ncthw(yt, ctx, C, H, W)
    ok, msg = check_with_pcc(gt, y, pcc=0.99)
    print(f"\nPCC_RESNET_MESH_{hf}x{wf} C={C} T={T} H={H} W={W} {msg} -> {'OK' if ok else 'FAIL'}", flush=True)
    assert ok, msg


@pytest.mark.parametrize("device_params", MESH_DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("in_c,out_c,add_temporal,tag", [(512, 256, False, "spatial"), (256, 128, True, "temporal")])
def test_upsample_mesh(device_params, mesh_device, in_c, out_c, add_temporal, tag):
    """UpsampleDCAE on (8,4) mesh: conv halo + LOCAL depth_to_spacetime (spatial doubling
    needs no CCL — the shard just grows locally)."""
    dev = mesh_device
    hf, wf = int(dev.shape[0]), int(dev.shape[1])
    torch.manual_seed(0)
    ctx = MeshCtx(dev, h_factor=hf, h_axis=0, w_factor=wf, w_axis=1, num_links=_num_links(dev))
    T, H, W = 2, 32, 32
    x = torch.randn(1, in_c, T, H, W)
    ref = HF.UpsampleDCAE(in_c, out_c, add_temporal).eval()
    with torch.no_grad():
        gt = ref(x)
    tt = UpsampleDCAE(ref.state_dict(), "", in_c, out_c, add_temporal, T, H, W, dev, ctx=ctx)
    xt = shard_input_nthwc(x, ctx)
    yt = tt(xt)
    y = gather_output_ncthw(yt, ctx, out_c, H * 2, W * 2)
    ok, msg = check_with_pcc(gt, y, pcc=0.99)
    print(
        f"\nPCC_UPSAMPLE_MESH_{tag.upper()} in={in_c} out={out_c} temporal={add_temporal} {msg} -> {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    assert ok, msg


@pytest.mark.parametrize("device_params", MESH_DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_decoder_mesh(device_params, mesh_device):
    """Full decoder on (8,4) mesh, latent [1,32,1,16,16] -> [1,3,4,256,256] vs HF decoder."""
    dev = mesh_device
    hf, wf = int(dev.shape[0]), int(dev.shape[1])
    torch.manual_seed(0)
    # num_links=1: the decoder starts at T=1 (latent), and neighbor_pad requires outer_dim(T) >= num_links.
    ctx = MeshCtx(dev, h_factor=hf, h_axis=0, w_factor=wf, w_axis=1, num_links=1)
    cfg = dict(
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        block_out_channels=[128, 256, 512, 1024, 1024],
        layers_per_block=2,
        ffactor_spatial=16,
        ffactor_temporal=4,
        sample_size=384,
        sample_tsize=96,
        scaling_factor=0.562679178327931,
    )
    vae = HF.AutoencoderKLConv3D(**cfg).eval()
    dec = vae.decoder
    z = torch.randn(1, 32, 1, 16, 16)  # 16 divisible by 8(H) and 4(W); all levels divide cleanly
    with torch.no_grad():
        gt = dec(z)  # [1,3,4,256,256]
    tt = VaeDecoder(dec, dev, latent_t=1, latent_h=16, latent_w=16, ctx=ctx)
    zt = shard_input_nthwc(z, ctx)
    yt = tt._forward(zt)
    y = gather_output_ncthw(yt, ctx, 3, 256, 256)
    ok, msg = check_with_pcc(gt, y, pcc=0.99)
    print(f"\nPCC_DECODER_MESH {msg} -> {'OK' if ok else 'FAIL'} gt_shape={tuple(gt.shape)}", flush=True)
    assert ok, msg


@pytest.mark.parametrize("device_params", MESH_DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_decoder_mesh_fullres(device_params, mesh_device):
    """Full decoder at REAL resolution: latent [1,32,1,64,64] -> [1,3,4,1024,1024] on (8,4).
    Stresses the norm gather at 1024^2 (~1GB) — the memory/perf reality check. Times it."""
    import time

    dev = mesh_device
    hf, wf = int(dev.shape[0]), int(dev.shape[1])
    torch.manual_seed(0)
    ctx = MeshCtx(dev, h_factor=hf, h_axis=0, w_factor=wf, w_axis=1, num_links=1)
    cfg = dict(
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        block_out_channels=[128, 256, 512, 1024, 1024],
        layers_per_block=2,
        ffactor_spatial=16,
        ffactor_temporal=4,
        sample_size=384,
        sample_tsize=96,
        scaling_factor=0.562679178327931,
    )
    vae = HF.AutoencoderKLConv3D(**cfg).eval()
    dec = vae.decoder
    z = torch.randn(1, 32, 1, 64, 64)
    with torch.no_grad():
        gt = dec(z)  # [1,3,4,1024,1024]
    tt = VaeDecoder(dec, dev, latent_t=1, latent_h=64, latent_w=64, ctx=ctx)
    zt = shard_input_nthwc(z, ctx)
    ttnn.synchronize_device(dev)
    t0 = time.time()
    yt = tt._forward(zt)
    ttnn.synchronize_device(dev)
    dt = time.time() - t0
    y = gather_output_ncthw(yt, ctx, 3, 1024, 1024)
    ok, msg = check_with_pcc(gt, y, pcc=0.99)
    print(
        f"\nPCC_DECODER_MESH_FULLRES {msg} -> {'OK' if ok else 'FAIL'} decode_forward_s={dt:.2f} gt_shape={tuple(gt.shape)}",
        flush=True,
    )
    assert ok, msg


def _load_real_vae():
    """Load ONLY the VAE (vae.* keys, shards 31-32/32) — no 80B transformer, no render."""
    from safetensors.torch import load_file

    snap = os.path.dirname(reference_autoencoder_path())
    sd = {}
    for shard in ("model-0031-of-0032.safetensors", "model-0032-of-0032.safetensors"):
        for k, v in load_file(os.path.join(snap, shard)).items():
            if k.startswith("vae."):
                sd[k[len("vae.") :]] = v
    cfg = dict(
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        block_out_channels=[128, 256, 512, 1024, 1024],
        layers_per_block=2,
        ffactor_spatial=16,
        ffactor_temporal=4,
        sample_size=384,
        sample_tsize=96,
        scaling_factor=0.562679178327931,
    )
    vae = HF.AutoencoderKLConv3D(**cfg).eval()
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    dec_missing = [k for k in missing if k.startswith("decoder.")]
    assert not dec_missing, f"decoder weights missing: {dec_missing[:5]}"
    return vae


@pytest.mark.parametrize("device_params", MESH_DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_decoder_mesh_realweights(device_params, mesh_device):
    """REAL VAE weights: on-device decode vs host model.vae.decode at 1024^2 (the wiring oracle).
    Catches any weight-load / decode()-pipeline bug before the expensive e2e render."""
    dev = mesh_device
    hf, wf = int(dev.shape[0]), int(dev.shape[1])
    torch.manual_seed(0)
    ctx = MeshCtx(dev, h_factor=hf, h_axis=0, w_factor=wf, w_axis=1, num_links=1)
    vae = _load_real_vae()
    z = torch.randn(1, 32, 1, 64, 64)  # already post-scaling latent (compare decode() directly)
    with torch.no_grad():
        gt = vae.decode(z.float(), return_dict=False)[0]  # host oracle [1,3,1,1024,1024]
    tt = VaeDecoder(vae.decoder, dev, latent_t=1, latent_h=64, latent_w=64, ctx=ctx)
    y = tt.decode(z)  # [1,3,1,1024,1024]
    ok, msg = check_with_pcc(gt, y, pcc=0.99)
    print(f"\nPCC_DECODER_MESH_REALWEIGHTS {msg} -> {'OK' if ok else 'FAIL'} gt_shape={tuple(gt.shape)}", flush=True)
    assert ok, msg


@pytest.mark.parametrize("device_params", DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_decoder_tiny(device_params, mesh_device):
    """Full decoder on a tiny latent [1,32,1,8,8] -> [1,3,4,128,128] vs HF decoder.forward."""
    dev = mesh_device
    torch.manual_seed(0)
    cfg = dict(
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        block_out_channels=[128, 256, 512, 1024, 1024],
        layers_per_block=2,
        ffactor_spatial=16,
        ffactor_temporal=4,
        sample_size=384,
        sample_tsize=96,
        scaling_factor=0.562679178327931,
    )
    vae = HF.AutoencoderKLConv3D(**cfg).eval()
    dec = vae.decoder
    z = torch.randn(1, 32, 1, 8, 8)
    with torch.no_grad():
        gt = dec(z)  # [1,3,4,128,128] (pre last-frame slice)
    tt = VaeDecoder(dec, dev, latent_t=1, latent_h=8, latent_w=8)
    y = tt._forward(to_device_nthwc(z, dev))
    ok, msg = _pcc(gt, y, 3, 0.99)
    print(f"\nPCC_DECODER_TINY {msg} -> {'OK' if ok else 'FAIL'} gt_shape={tuple(gt.shape)}", flush=True)
    assert ok, msg
