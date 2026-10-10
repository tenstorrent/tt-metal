# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One arm of the conv VAE decode halo-only A/B on a 2x4 submesh at 544x960/145f.

On 2x4 this gives each chip the same shard as 1088x1920/145f on the production 4x8. One run per arm; the
arm is set by LTX_VAE_HALO_ONLY (0: neighbor_pad full-pad copy, 1: neighbor_pad_halo + conv3d halo_buffer).
Each run saves its yuv output to $AB_OUT_DIR/yuv_h<0|1>.pt and prints AB timing lines; both arms' outputs
must be identical. Each arm must also match the stored reference decode (tools/vae_ref_check.py).
"""

import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.tests.models.ltx.tools.vae_ref_check import check_against_reference

NF, H, W = 145, 544, 960
TIMED_DECODES = 3


def _latent():
    lt, lh, lw = (NF - 1) // 8 + 1, H // 32, W // 32
    path = os.environ.get("AB_LATENT")
    if path and os.path.exists(path):
        # Saved 2.5 stage-2 latent (1080p: T=19, H=34, W=60, packed tokens), center crop to 17x30.
        v = torch.load(path, map_location="cpu")["video"].float().reshape(1, 19, 34, 60, 128)
        h0, w0 = (34 - lh) // 2, (60 - lw) // 2
        return v[:, :lt, h0 : h0 + lh, w0 : w0 + lw].permute(0, 4, 1, 2, 3).contiguous()
    torch.manual_seed(0)
    return torch.randn(1, 128, lt, lh, lw)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_ltx_halo_only_ab(mesh_device, device_params):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    from models.tt_dit.models.vae import vae_ltx
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.ltx.test_vae_ltx import (
        _LTX_PROD_DECODER_BLOCKS,
        _diffusers_decoder_state_to_tt,
        _require_diffusers_ltx_vae,
        _TorchLTXVideoDecoder,
    )

    halo = os.environ.get("LTX_VAE_HALO_ONLY", "1")
    arm = f"h{halo}"
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
        height_parallel=ParallelFactor(factor=2, mesh_axis=0),
        width_parallel=ParallelFactor(factor=4, mesh_axis=1),
    )
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
        parallel_config=pc,
        ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2),
    )
    dec.load_torch_state_dict(_diffusers_decoder_state_to_tt(state))

    def walk(m):
        yield m
        for _, c in m.named_children():
            yield from walk(c)

    n_halo = sum(1 for m in walk(dec) if getattr(m, "halo_only", False))
    print(f"AB arm={arm} halo_only_convs={n_halo}")
    assert (n_halo > 0) == (halo == "1")

    lat = _latent()
    out = dec(lat, output_type="yuv")  # warmup: kernel compile + program cache
    # Only a conv that took the halo path builds a pad offset (the 2x4 latent has W and H pad to mask).
    n_offset = sum(1 for m in walk(dec) if getattr(m, "_pad_offset_cache", None))
    print(f"AB arm={arm} convs_with_pad_offset={n_offset}")
    assert (n_offset > 0) == (halo == "1")
    times = []
    for _ in range(TIMED_DECODES):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = dec(lat, output_type="yuv")
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
    print(f"AB arm={arm} decode_s={' '.join(f'{t:.4f}' for t in times)} min={min(times):.4f}")

    out_dir = os.environ.get("AB_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(torch.as_tensor(out), os.path.join(out_dir, f"yuv_{arm}.pt"))

    check_against_reference(
        out,
        arm,
        latent=lat,
        num_frames=NF,
        height=H,
        width=W,
        mesh_shape=(2, 4),
        exact_shard=getattr(dec, "exact_shard", False),
    )
