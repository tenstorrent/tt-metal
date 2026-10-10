# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-profile one eager conv VAE decode on the full 4x8 mesh at 1088x1920/145f (t334).

Production layout: H on mesh axis 0 (4), W on axis 1 (8), Linear CCL with 2 links, the decoder's own 4x8
conv3d blockings, t48 defaults (HALO_ONLY, FOLD_TIME_PAD, EXACT_SHARD). Random-init decoder weights (per-op
timing does not depend on them). Latent: $AB_LATENT (saved 1080p 2.5 latent, 19x34x60) if present, else
seeded randn. Run under `python -m tracy -p -r`; the decode sits between "start"/"stop" signposts.
"""

import os

import pytest
import torch
from tracy import signpost

import ttnn

NF, H, W = 145, 1088, 1920


def _latent():
    lt, lh, lw = (NF - 1) // 8 + 1, H // 32, W // 32
    path = os.environ.get("AB_LATENT")
    if path and os.path.exists(path):
        print(f"PROF latent {path}")
        return torch.load(path, map_location="cpu")["video"].float().reshape(1, lt, lh, lw, 128).permute(0, 4, 1, 2, 3)
    print("PROF latent randn seed 0")
    torch.manual_seed(0)
    return torch.randn(1, 128, lt, lh, lw)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_ltx_prof_4x8(mesh_device, device_params):
    from models.tt_dit.models.vae.vae_ltx import LTXVideoDecoder
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.ltx.test_vae_ltx import (
        _LTX_PROD_DECODER_BLOCKS,
        _diffusers_decoder_state_to_tt,
        _require_diffusers_ltx_vae,
        _TorchLTXVideoDecoder,
    )

    assert tuple(mesh_device.shape) == (4, 8)
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
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=4, mesh_axis=0),
        width_parallel=ParallelFactor(factor=8, mesh_axis=1),
    )
    dec = LTXVideoDecoder(
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
        parallel_config=pc,
        ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2),
    )
    dec.load_torch_state_dict(_diffusers_decoder_state_to_tt(state))
    print(f"PROF exact_shard={dec.exact_shard} halo_only={os.environ.get('LTX_VAE_HALO_ONLY', 'default')}")
    lat = _latent()

    # One decode only: it issues thousands of ops, and a second would overflow the profiler buffer.
    signpost("start")
    out = dec(lat, output_type="yuv")
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    ttnn.ReadDeviceProfiler(mesh_device)
    print(f"PROF out {getattr(out, 'shape', None)}")
