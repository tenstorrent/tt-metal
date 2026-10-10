# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One arm of the conv VAE conv_out unpatchify fusion A/B on the 4x8 mesh at 1088x1920/145f.

The arm is set by LTX_VAE_FUSE_UNPATCH (0: reshape + permute conv_out's output to CHWT, then rgb_to_yuv;
1: rgb_to_yuv reads conv_out's patchified output directly), on the fused YUV path (LTX_FUSE_YUV_OUTPUT=1).
Weights: $VAE_CKPT (LTX conv VAE safetensors, decoder built from its header config) if set, else random
init. Latents: $AB_LATENT_DIR/seed<s>.pt (1080p latents, 128x19x34x60) if set, else seeded randn. Each run
prints one md5 per seed and saves the yuv outputs to $AB_OUT_DIR/yuv_r<0|1>_s<seed>.pt, then prints AB
timing lines for the first seed. Both arms' outputs must be bit-identical.
"""

import hashlib
import json
import os
import time

import pytest
import torch

import ttnn

NF, H, W = 145, 1088, 1920
TIMED_DECODES = 3


def _latent(seed):
    lt, lh, lw = (NF - 1) // 8 + 1, H // 32, W // 32
    latent_dir = os.environ.get("AB_LATENT_DIR")
    if latent_dir:
        lat = torch.load(os.path.join(latent_dir, f"seed{seed}.pt"), map_location="cpu")
        if isinstance(lat, dict):
            lat = lat["video"]
        lat = lat.float()
        if lat.shape[-1] == 128 and lat.ndim <= 3:
            return lat.reshape(1, lt, lh, lw, 128).permute(0, 4, 1, 2, 3)
        return lat.reshape(1, 128, lt, lh, lw)
    torch.manual_seed(seed)
    return torch.randn(1, 128, lt, lh, lw)


def _vae_header_config(path):
    with open(path, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    return json.loads(header.get("__metadata__", {}).get("config", "{}")).get("vae", {})


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_ltx_fuse_unpatch_ab(mesh_device, device_params):
    from safetensors.torch import load_file

    from models.tt_dit.models.vae.vae_ltx import LTXVideoDecoder, vae_key_map
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.ltx.test_vae_ltx import (
        _LTX_PROD_DECODER_BLOCKS,
        _diffusers_decoder_state_to_tt,
        _require_diffusers_ltx_vae,
        _TorchLTXVideoDecoder,
    )

    assert os.environ.get("LTX_FUSE_YUV_OUTPUT") == "1", "the unpatchify fusion only applies to the fused YUV path"
    fuse = os.environ.get("LTX_VAE_FUSE_UNPATCH", "0")
    arm = f"r{fuse}"
    ckpt = os.environ.get("VAE_CKPT")
    cfg = _vae_header_config(ckpt) if ckpt else {}
    blocks = cfg.get("decoder_blocks") or _LTX_PROD_DECODER_BLOCKS
    causal = cfg.get("causal_decoder", False)
    base = cfg.get("decoder_base_channels", 128)
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=4, mesh_axis=0),
        width_parallel=ParallelFactor(factor=8, mesh_axis=1),
    )
    dec = LTXVideoDecoder(
        decoder_blocks=blocks,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=base,
        causal=causal,
        num_frames=NF,
        height=H,
        width=W,
        mesh_device=mesh_device,
        parallel_config=pc,
        ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2),
    )
    assert dec.fuse_unpatch == (fuse == "1")
    if ckpt:
        raw = load_file(ckpt)
        dec.load_torch_state_dict({short: raw[k] for k, short in vae_key_map(raw, "decoder").items()})
    else:
        torch.manual_seed(42)
        tdec = _TorchLTXVideoDecoder(
            decoder_blocks=blocks,
            in_channels=128,
            out_channels=3,
            patch_size=4,
            base_channels=base,
            causal=False,
            spatial_padding_mode="zeros",
            vae_mods=_require_diffusers_ltx_vae(),
        ).eval()
        state = tdec.state_dict()
        state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
        state["per_channel_statistics.std-of-means"] = torch.ones(128)
        dec.load_torch_state_dict(_diffusers_decoder_state_to_tt(state))
    print(f"AB arm={arm} weights={ckpt or 'random'} latents={os.environ.get('AB_LATENT_DIR') or 'randn'}")

    out_dir = os.environ.get("AB_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in os.environ.get("AB_SEEDS", "0,1,2,3,4").split(",")]
    for seed in seeds:
        out = torch.as_tensor(dec(_latent(seed), output_type="yuv"))
        md5 = hashlib.md5(out.contiguous().numpy().tobytes()).hexdigest()
        print(f"AB arm={arm} seed={seed} shape={tuple(out.shape)} md5={md5}")
        if out_dir:
            torch.save(out, os.path.join(out_dir, f"yuv_{arm}_s{seed}.pt"))

    lat = _latent(seeds[0])
    times = []
    for _ in range(TIMED_DECODES):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        dec(lat, output_type="yuv")
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
    print(f"AB arm={arm} decode_s={' '.join(f'{t:.4f}' for t in times)} min={min(times):.4f}")
