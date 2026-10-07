# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Production 4x8 checks for the re-picked LTX conv3d blockings (blx03, full mesh, 1080p/145f shards).

test_ups_ab: latent upsampler at the S1 latent (19x17x30), one arm per blocking in T119_UPS_ARMS
("Cin,Cout,T,H,W/..."), patched into the ups_initial key. Each arm is timed (whole forward_device and
initial_conv alone) and compared to the fp32 diffusers reference, overall and in bands at the chip seams.
test_vae: conv VAE decode at 1088x1920: a halo-off eager decode records/serves the reference, then the
default (halo-only) decoder is timed eager and traced and checked against it (VAE_REF gate).
"""

import gc
import math
import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import prefetch_shard_fits
from models.tt_dit.utils import conv3d

UPS_KEY = (4, 8, 128, 1024, (3, 3, 3), 21, 5, 4)
TRACE_REGION = int(os.environ.get("LTX_TRACE_REGION", "500000000"))
DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}
SEAM_BAND = 2  # latent px each side; a dropped halo corrupts the conv's 1-px border, so 2 covers it


def _pc():
    return VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=4, mesh_axis=0),
        width_parallel=ParallelFactor(factor=8, mesh_axis=1),
    )


def _sync_times(mesh_device, fn, n):
    times = []
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
    return times


def _fmt(ts):
    return f"min={min(ts) * 1e3:.3f}ms med={sorted(ts)[len(ts) // 2] * 1e3:.3f}ms n={len(ts)}"


def _seam_idx(size, chip, band=SEAM_BAND):
    idx = set()
    for b in range(chip, size, chip):
        idx.update(range(max(0, b - band), min(size, b + band)))
    return sorted(idx)


def _psnr(err, peak):
    mse = err.double().pow(2).mean().item()
    return 99.0 if mse == 0 else 10 * math.log10(peak**2 / mse)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_ups_ab(mesh_device, device_params):
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    from models.tt_dit.models.upsampler.latent_upsampler_ltx import LTXLatentUpsampler

    arms = [tuple(int(v) for v in a.split(",")) for a in os.environ["T119_UPS_ARMS"].split("/")]
    in_c, mid_c, T, H, W = 128, 1024, 19, 17, 30
    torch.manual_seed(0xC0FFEE)
    ref_model = LTX2LatentUpsamplerModel(
        in_channels=in_c,
        mid_channels=mid_c,
        num_blocks_per_stage=4,
        dims=3,
        spatial_upsample=True,
        temporal_upsample=False,
        rational_spatial_scale=2.0,
        use_rational_resampler=False,
    ).eval()
    latent = torch.randn(1, in_c, T, H, W)
    with torch.no_grad():
        ref = ref_model(latent)
    peak = ref.abs().max().item()
    # Output-latent chip seams: the 17x30 input pads to 20x32, so 5x4 per chip, doubled by the upsample.
    rows, cols = _seam_idx(2 * H, 2 * 5), _seam_idx(2 * W, 2 * 4)

    outs = {}
    for blk in arms:
        assert prefetch_shard_fits(*blk, UPS_KEY[4], UPS_KEY[2]), f"{blk} gets no L1 prefetch shard"
        conv3d._BLOCKINGS[UPS_KEY] = blk
        conv3d._BLOCKINGS_BY_SPATIAL = None
        tt = LTXLatentUpsampler(
            input_hw=(H, W),
            in_channels=in_c,
            mid_channels=mid_c,
            num_blocks_per_stage=4,
            spatial_upsample=True,
            temporal_upsample=False,
            spatial_scale=2.0,
            rational_resampler=False,
            mesh_device=mesh_device,
            num_frames=T,
            parallel_config=_pc(),
            ccl_manager=CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear),
        )
        tt.load_torch_state_dict(ref_model.state_dict())
        c = tt.initial_conv.conv_config
        used = (c.C_in_block, c.C_out_block, c.T_out_block, c.H_out_block, c.W_out_block)
        x, lh, lw = tt._encode_input(latent)
        tt.forward_device(x, lh, lw)  # compile + program cache

        def full():
            tt.forward_device(tt._encode_input(latent)[0], lh, lw)

        def conv_only():
            tt.initial_conv(x, causal=False, logical_h=lh, logical_w=lw)

        t_full = _sync_times(mesh_device, full, 10)
        t_conv = _sync_times(mesh_device, conv_only, 30)
        out = tt.forward(latent).float()
        err = out - ref
        pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
        tag = ",".join(map(str, blk))
        print(
            f"T119_UPS arm={tag} used={used} full {_fmt(t_full)} | initial_conv {_fmt(t_conv)} | "
            f"pcc={pcc:.6f} psnr={_psnr(err, peak):.2f}dB seam_rows={_psnr(err[..., rows, :], peak):.2f}dB "
            f"seam_cols={_psnr(err[..., cols], peak):.2f}dB max_abs={err.abs().max().item():.4f}",
            flush=True,
        )
        outs[tag] = out
        del tt, x
        gc.collect()
    base = next(iter(outs.values()))
    for tag, out in outs.items():
        print(f"T119_UPS_CMP arm={tag} identical_to_first={torch.equal(out, base)}", flush=True)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae(mesh_device, device_params):
    from models.tt_dit.models.vae import vae_ltx
    from models.tt_dit.tests.models.ltx.test_vae_ltx import (
        _LTX_PROD_DECODER_BLOCKS,
        _diffusers_decoder_state_to_tt,
        _require_diffusers_ltx_vae,
        _TorchLTXVideoDecoder,
    )
    from models.tt_dit.tests.models.ltx.test_vae_ltx_fold_time_pad_ab import NF, H, W, _latent
    from models.tt_dit.tests.models.ltx.tools.vae_ref_check import check_against_reference

    assert (H, W) == (1088, 1920), "set LTX_VAE_AB_HW=1088,1920"
    torch.manual_seed(42)
    tdec = _TorchLTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        spatial_padding_mode="zeros",
        vae_mods=_require_diffusers_ltx_vae(),
    ).eval()
    state = tdec.state_dict()
    state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    state["per_channel_statistics.std-of-means"] = torch.ones(128)
    tt_state = _diffusers_decoder_state_to_tt(state)
    lat = _latent()

    def build():
        dec = vae_ltx.LTXVideoDecoder(
            decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
            in_channels=128,
            out_channels=3,
            patch_size=4,
            base_channels=128,
            causal=False,
            num_frames=NF,
            height=H,
            width=W,
            mesh_device=mesh_device,
            parallel_config=_pc(),
            ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2),
        )
        dec.load_torch_state_dict(tt_state)
        return dec

    def check(out, arm, dec):
        check_against_reference(
            out, arm, latent=lat, num_frames=NF, height=H, width=W, mesh_shape=(4, 8), exact_shard=dec.exact_shard
        )

    # The halo path is chosen at conv construction, so the env switch must precede build().
    os.environ["LTX_VAE_HALO_ONLY"] = "0"
    dec = build()
    out = torch.as_tensor(dec(lat, output_type="yuv")).clone()
    check(out, "halo_off", dec)
    del dec, out
    gc.collect()

    os.environ["LTX_VAE_HALO_ONLY"] = "1"
    dec = build()
    dec(lat, output_type="yuv")  # compile + program cache
    t_eager = _sync_times(mesh_device, lambda: dec(lat, output_type="yuv"), 3)
    eager = torch.as_tensor(dec(lat, output_type="yuv")).clone()
    dec(lat, output_type="yuv", traced=True)  # capture
    t_traced = _sync_times(mesh_device, lambda: dec(lat, output_type="yuv", traced=True), 5)
    traced = torch.as_tensor(dec(lat, output_type="yuv", traced=True)).clone()
    dec.release_trace()
    print(f"T119_VAE eager {_fmt(t_eager)} | traced {_fmt(t_traced)}", flush=True)
    print(f"T119_VAE traced_vs_eager identical={torch.equal(traced, eager)}", flush=True)
    check(eager, "default_eager", dec)
    check(traced, "default_traced", dec)
