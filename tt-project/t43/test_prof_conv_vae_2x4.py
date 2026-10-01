# Task #43: device-profile one LTX-2.5 conv VAE decode on a 2x4 mesh.
# 544x960/145f on 2x4 gives each chip the same shard (latent 9x8 per device, same pad masks) as
# 1088x1920/145f on the production 4x8, so per-op device times match production per-device work.
import os

import pytest
import torch
import ttnn
from tracy import signpost

from models.tt_dit.models.vae.vae_ltx import LTXVideoDecoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.ltx.test_vae_ltx import (
    _LTX_PROD_DECODER_BLOCKS,
    _TorchLTXVideoDecoder,
    _diffusers_decoder_state_to_tt,
    _require_diffusers_ltx_vae,
)

NF, H, W = 145, 544, 960
LAT = "/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt"


def _latent():
    lt, lh, lw = (NF - 1) // 8 + 1, H // 32, W // 32
    if os.path.exists(LAT):
        # Saved 2.5 stage-2 latent (1080p: T=19, H=34, W=60, packed tokens), center crop to 17x30.
        v = torch.load(LAT, map_location="cpu")["video"].float().reshape(1, 19, 34, 60, 128)
        h0, w0 = (34 - lh) // 2, (60 - lw) // 2
        return v[:, :lt, h0 : h0 + lh, w0 : w0 + lw].permute(0, 4, 1, 2, 3).contiguous()
    return torch.randn(1, 128, lt, lh, lw)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_prof_conv_vae_2x4(mesh_device, device_params):
    # Opening a bare 2x4 on the galaxy fails fabric router sync (job 989: links to unopened chips
    # never handshake), so open the system mesh and run only on a 2x4 submesh.
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    vae_mods = _require_diffusers_ltx_vae()
    torch.manual_seed(42)
    tdec = _TorchLTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        spatial_padding_mode="zeros",
        vae_mods=vae_mods,
    ).eval()
    state = tdec.state_dict()
    state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    state["per_channel_statistics.std-of-means"] = torch.ones(128)
    ccl = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2)
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=2, mesh_axis=0),
        width_parallel=ParallelFactor(factor=4, mesh_axis=1),
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
        ccl_manager=ccl,
    )
    dec.load_torch_state_dict(_diffusers_decoder_state_to_tt(state))
    lat = _latent()
    print(f"T43 latent {tuple(lat.shape)} from {'saved' if os.path.exists(LAT) else 'randn'}")

    # One forward only: the decoder issues thousands of ops; more would overflow the profiler buffer.
    signpost("start")
    out = dec(lat, output_type="yuv")
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    ttnn.ReadDeviceProfiler(mesh_device)
    print(f"T43 out {getattr(out, 'shape', None)}")
