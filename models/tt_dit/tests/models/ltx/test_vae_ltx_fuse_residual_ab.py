# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One arm of the conv VAE decode resnet skip-add fusion A/B on the 4x8 mesh at 1088x1920/145f.

The arm is set by LTX_VAE_FUSE_RESIDUAL (0: conv2, then ttnn.add of the skip tensor; 1: conv2 adds the skip
tensor in its epilogue). Each run decodes AB_SEEDS random latents, prints one md5 per seed and saves
the yuv outputs to $AB_OUT_DIR/yuv_r<0|1>_s<seed>.pt, then prints AB timing lines for the first seed.
Both arms' outputs must be bit-identical. AB_MESH=2x4 runs on a 2x4 submesh at 544x960 instead.
"""

import hashlib
import os
import time

import pytest
import torch

import ttnn

NF = 145
TIMED_DECODES = 3


def _latent(seed, h, w):
    lt, lh, lw = (NF - 1) // 8 + 1, h // 32, w // 32
    torch.manual_seed(seed)
    return torch.randn(1, 128, lt, lh, lw)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_ltx_fuse_residual_ab(mesh_device, device_params):
    from models.tt_dit.models.vae import vae_ltx
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.ltx.test_vae_ltx import (
        _LTX_PROD_DECODER_BLOCKS,
        _diffusers_decoder_state_to_tt,
        _require_diffusers_ltx_vae,
        _TorchLTXVideoDecoder,
    )

    if os.environ.get("AB_MESH", "4x8") == "2x4":
        # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
        mesh_device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
        H, W = 544, 960
    else:
        H, W = 1088, 1920
    rows, cols = tuple(mesh_device.shape)

    fuse = os.environ.get("LTX_VAE_FUSE_RESIDUAL", "1")
    arm = f"r{fuse}"
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
        height_parallel=ParallelFactor(factor=rows, mesh_axis=0),
        width_parallel=ParallelFactor(factor=cols, mesh_axis=1),
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

    out_dir = os.environ.get("AB_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in os.environ.get("AB_SEEDS", "0,1,2,3,4").split(",")]
    for seed in seeds:
        out = torch.as_tensor(dec(_latent(seed, H, W), output_type="yuv"))
        md5 = hashlib.md5(out.contiguous().numpy().tobytes()).hexdigest()
        if seed == seeds[0]:
            n_fused = sum(1 for m in walk(dec) if getattr(m, "fused", False))
            print(f"AB arm={arm} mesh={rows}x{cols} fused_resnets={n_fused}")
            assert (n_fused > 0) == (fuse == "1")
        print(f"AB arm={arm} seed={seed} shape={tuple(out.shape)} md5={md5}")
        if out_dir:
            torch.save(out, os.path.join(out_dir, f"yuv_{arm}_s{seed}.pt"))

    lat = _latent(seeds[0], H, W)
    times = []
    for _ in range(TIMED_DECODES):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        dec(lat, output_type="yuv")
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
    print(f"AB arm={arm} decode_s={' '.join(f'{t:.4f}' for t in times)} min={min(times):.4f}")
