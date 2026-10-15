# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gates for the Wan2.2 TI2V-5B device image encoder (sprint 7, lever A10).

1. ``test_patchify_roundtrip`` (host): ``patchify_bcthw`` equals diffusers' ``patchify`` and inverts
   ``unpatchify_bcthw``.
2. ``test_avg_down3d_host_algorithm`` (host): the slice / add / permutation-matmul algorithm of
   ``WanAvgDown3D``, emulated in torch, equals the HF ``AvgDown3D`` for the three 5B instances.
3. ``test_avg_down3d_device`` (device, 1x1 mesh): ``WanAvgDown3D`` vs HF ``AvgDown3D`` on the 5B
   per-device shapes, T = 1 and T = 5.
4. ``test_residual_encoder_vs_torch`` (device, 4x8): the full residual ``WanEncoder`` with random
   weights at the 5B config vs ``AutoencoderKLWan.encode`` on one patchified 1280x704 frame.
5. ``test_pipeline_device_encode_matches_host`` (device, 4x8): the production gate: the pipeline's
   device encode of a real image vs its host fp32 encode (latent PCC, max |diff|).

    pytest models/tt_dit/tests/models/wan2_2/test_i2v_device_encoder_ti2v_5b.py -sv --timeout=0
"""

import os
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanAvgDown3D, patchify_bcthw, unpatchify_bcthw
from models.tt_dit.models.vae.vae_wan2_1_encoder_host import AvgDown3D as HostAvgDown3D
from models.tt_dit.utils.test import ring_params_req_exact_devices

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests dir is not a package

# (name, in_dim, out_dim, factor_t, factor_s, T, H, W) at per-device 5B production shapes (1280x704 -> patchified
# 640x352 -> per device 80x88 on the 4x8 mesh; stage 1 at 40x44, stage 2 at 20x22).
_CASES = [
    ("down0_160_160_ft1", 160, 160, 1, 2, 1, 88, 80),
    ("down1_160_320_ft2", 160, 320, 2, 2, 1, 44, 40),
    ("down2_320_640_ft2", 320, 640, 2, 2, 1, 22, 20),
    ("down1_t5", 160, 320, 2, 2, 5, 44, 40),
    ("down3_identity", 640, 640, 1, 1, 1, 11, 10),
]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a0, b0 = a - a.mean(), b - b.mean()
    return float((a0 @ b0) / (a0.norm() * b0.norm()))


def _torch_emulation(x_BTHWC: torch.Tensor, in_c, out_c, ft, fs) -> torch.Tensor:
    """What WanAvgDown3D.forward computes, in torch: pad, strided slices, adds, permutation matmul."""
    x = x_BTHWC.clone()
    B, T, H, W, C = x.shape
    pad_t = (ft - T % ft) % ft
    if pad_t:
        x = torch.cat([torch.zeros(B, pad_t, H, W, C, dtype=x.dtype), x], dim=1)
        T += pad_t
    if fs > 1:
        x = sum(x[:, :, dh::fs] for dh in range(fs))
        x = sum(x[:, :, :, dw::fs] for dw in range(fs)) / (fs * fs)
        H //= fs
        W //= fs
    if ft == 1:
        return x
    slices = [x[:, dt::ft] for dt in range(ft)]
    cat = torch.cat(slices, dim=-1)
    perm = torch.zeros(ft * C, ft * C, dtype=x.dtype)
    for dt in range(ft):
        for c in range(C):
            perm[dt * C + c, c * ft + dt] = 1.0
    return cat @ perm


def test_patchify_roundtrip():
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify as hf_patchify

    x = torch.randn(1, 3, 2, 32, 48)
    ours = patchify_bcthw(x, 2)
    theirs = hf_patchify(x, 2)
    assert ours.shape == theirs.shape == (1, 12, 2, 16, 24)
    assert torch.equal(ours, theirs), "patchify_bcthw differs from diffusers patchify"
    assert torch.equal(unpatchify_bcthw(ours, 2), x), "unpatchify(patchify(x)) != x"


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_avg_down3d_host_algorithm(case):
    name, in_c, out_c, ft, fs, T, H, W = case
    torch.manual_seed(0)
    x_BCTHW = torch.randn(1, in_c, T, H, W)
    ref = HostAvgDown3D(in_c, out_c, factor_t=ft, factor_s=fs)(x_BCTHW)  # (B, out, To, H/fs, W/fs)
    got = _torch_emulation(x_BCTHW.permute(0, 2, 3, 4, 1), in_c, out_c, ft, fs).permute(0, 4, 1, 2, 3)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert torch.allclose(got, ref, atol=1e-5), f"{name}: max abs diff {(got - ref).abs().max().item()}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["single"], indirect=True)
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_avg_down3d_device(mesh_device, case):
    name, in_c, out_c, ft, fs, T, H, W = case
    torch.manual_seed(0)
    x_BCTHW = torch.randn(1, in_c, T, H, W).to(torch.bfloat16).to(torch.float32)
    ref = HostAvgDown3D(in_c, out_c, factor_t=ft, factor_s=fs)(x_BCTHW)
    module = WanAvgDown3D(in_c, out_c, factor_t=ft, factor_s=fs, mesh_device=mesh_device)
    x_BTHWC = x_BCTHW.permute(0, 2, 3, 4, 1).contiguous()
    tt_x = ttnn.from_torch(x_BTHWC, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    tt_out = module(tt_x)
    out = ttnn.to_torch(ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)).to(torch.float32)
    out = out[..., : ref.shape[1]].permute(0, 4, 1, 2, 3)
    pcc = _pcc(out, ref)
    max_abs = (out - ref).abs().max().item()
    logger.info(f"{name}: shape {tuple(out.shape)} PCC {pcc:.7f} max_abs {max_abs:.4g}")
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert pcc > 0.9999 and max_abs < 2e-2, f"{name}: PCC {pcc}, max abs {max_abs}"


@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    [((4, 8), 0, 1, 2)],
    ids=["4x8h0w1nl2"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}],
    ids=["line"],
    indirect=True,
)
def test_residual_encoder_vs_torch(mesh_device, h_axis, w_axis, num_links):
    """Random-weight residual encoder at the 5B config on one patchified 1280x704 frame."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan as TorchAutoencoderKLWan

    from models.tt_dit.models.vae.vae_wan2_1 import WanEncoder
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.conv3d import aligned_channels, conv_pad_height, conv_pad_width
    from models.tt_dit.utils.tensor import bf16_tensor_2dshard, fast_device_to_host
    from models.tt_dit.utils.test import skip_if_unsupported_num_links

    skip_if_unsupported_num_links(mesh_device, num_links)
    torch.manual_seed(0)
    cfg = dict(
        base_dim=160,
        in_channels=12,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        out_channels=12,
        is_residual=True,
        decoder_base_dim=256,
        patch_size=2,
    )
    torch_model = TorchAutoencoderKLWan(**cfg).eval()
    height, width = 704 // 2, 1280 // 2  # patchified frame
    raw = torch.randn(1, 3, 1, 704, 1280)
    pixels = patchify_bcthw(raw, 2)  # (1, 12, 1, 352, 640); diffusers patchifies inside encode()
    with torch.no_grad():
        ref = torch_model.encode(raw).latent_dist.mode()  # (1, 48, 1, 44, 80), unnormalized

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_model = WanEncoder(
        base_dim=cfg["base_dim"],
        in_channels=cfg["in_channels"],
        z_dim=cfg["z_dim"],
        dim_mult=cfg["dim_mult"],
        num_res_blocks=cfg["num_res_blocks"],
        attn_scales=cfg["attn_scales"],
        temperal_downsample=cfg["temperal_downsample"],
        is_residual=True,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        dtype=ttnn.bfloat16,
        height=height,
        width=width,
        encoder_t_chunk_size=1,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    frame_BTHWC = pixels.permute(0, 2, 3, 4, 1).contiguous()
    frame_BTHWC, logical_h = conv_pad_height(frame_BTHWC, parallel_config.height_parallel.factor * 8)
    frame_BTHWC, logical_w = conv_pad_width(frame_BTHWC, parallel_config.width_parallel.factor * 8)
    tt_frame = bf16_tensor_2dshard(
        frame_BTHWC, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
    )
    pad_c = aligned_channels(12) - 12
    if pad_c:
        tt_frame = ttnn.pad(tt_frame, [(0, 0), (0, 0), (0, 0), (0, 0), (0, pad_c)], value=0.0)
    enc_BCTHW, lh, lw = tt_model(tt_frame, logical_h, logical_w=logical_w, encoder_t_chunk_size=None)
    concat_dims = [None, None]
    concat_dims[h_axis] = 3
    concat_dims[w_axis] = 4
    got = fast_device_to_host(enc_BCTHW, mesh_device, concat_dims, ccl_manager=ccl_manager)[:, :, :, :lh, :lw].float()
    pcc = _pcc(got, ref)
    rmse = (got - ref).pow(2).mean().sqrt().item()
    logger.info(
        f"residual encoder: out {tuple(got.shape)} ref {tuple(ref.shape)} PCC {pcc:.6f} RMSE {rmse:.4f} (ref std {ref.std():.4f})"
    )
    assert got.shape == ref.shape
    assert pcc > 0.99, f"PCC {pcc}"


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    [[(4, 8), (4, 8), {**ring_params_req_exact_devices, "trace_region_size": 150000000}, ttnn.Topology.Ring]],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_device_encode_matches_host(mesh_device, mesh_shape, topology):
    """Production gate: pipeline device encode vs host fp32 encode of a real image (latent PCC, max |diff|)."""
    from PIL import Image

    from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b_i2v import WanTI2V5BI2VPipeline
    from models.tt_dit.utils.test import skip_if_unsupported_num_links

    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)
    os.environ["WAN5B_I2V_DEVICE_ENCODER"] = "1"
    height, width = 704, 1280
    pipeline = WanTI2V5BI2VPipeline.create_pipeline(
        mesh_device=mesh_device, height=height, width=width, num_frames=81, run_warmup=False
    )
    assert pipeline.tt_vae_encoder is not None, "device encoder was not built"
    img_path = os.environ.get("WAN5B_I2V_IMAGE")
    if img_path:
        image = Image.open(img_path).convert("RGB")
    else:
        torch.manual_seed(0)
        image = Image.fromarray((torch.rand(height, width, 3) * 255).to(torch.uint8).numpy())
    import time

    t0 = time.perf_counter()
    device_latent = pipeline.encode_condition_image(image, height=height, width=width, device=None)
    t_dev = time.perf_counter() - t0
    pipeline.tt_vae_encoder = None  # host path
    os.environ["WAN5B_I2V_ENCODE_FP32"] = "1"
    t0 = time.perf_counter()
    host_latent = pipeline.encode_condition_image(image, height=height, width=width, device=None)
    t_host = time.perf_counter() - t0
    pcc = _pcc(device_latent, host_latent)
    max_abs = (device_latent - host_latent).abs().max().item()
    rmse = (device_latent - host_latent).pow(2).mean().sqrt().item()
    print(
        f"I2V_DEVICE_ENCODER device={t_dev:.3f}s host_fp32={t_host:.3f}s shape={tuple(device_latent.shape)} "
        f"PCC={pcc:.7f} max_abs={max_abs:.4f} RMSE={rmse:.4f} host_std={host_latent.std():.4f}"
    )
    assert device_latent.shape == host_latent.shape
    assert pcc > 0.999, f"latent PCC {pcc} below 0.999"
