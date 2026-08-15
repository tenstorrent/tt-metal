# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture the DiffVAE's device half as a ttnn trace and replay it.

The decode is split so a trace can hold it: host does the ghost pad, the noise patchify and the
final gather/unpatchify, and ``decode_device`` is device-only in between. This checks the half
that matters — that the capture succeeds at all, that a replay reproduces the eager result, and
what it costs in trace region.

Two things make this less obvious than it looks. The halo exchange is a fabric CCL op inside the
captured region, which works only because it writes into the CCL manager's persistent ping-pong
buffers. And the region has to hold every op of every block, which is the risk this test exists
to measure rather than assume.
"""

import os
import time
from pathlib import Path

import pytest
import torch

import ttnn

from ....models.vae.diffvae_ltx import DiffVAEDecoder, MeshShardConfig, decoder_config
from ....models.vae.diffvae_ltx_stage5 import Grid
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tracing import Tracer

CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)

# 500 MB matches what the LTX distilled configs reserve for two DiT stage traces. Whether a
# 24-block DiffVAE decode fits in it is the open question.
_FABRIC_TRACE = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "require_exact_physical_num_devices": True,
    "trace_region_size": 500_000_000,
}


@pytest.mark.parametrize(
    "mesh_device, device_params, latent_frames, latent_hw, enter_stage",
    [
        # Crop scale: ops are small enough that dispatch is most of the eager time, so the
        # speedup here is an upper bound and does not carry to production.
        ((4, 8), _FABRIC_TRACE, 4, (10, 10), 3),
        # 145 frames of 1088x1920 — the only size whose number means anything.
        ((4, 8), _FABRIC_TRACE, 19, (34, 60), 1),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_decode_device_traces_and_replays(*, mesh_device, latent_frames, latent_hw, enter_stage):
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")

    config = decoder_config(CHECKPOINT)
    decoder = DiffVAEDecoder(config, mesh_device=mesh_device)
    decoder.load_checkpoint(CHECKPOINT)

    shard = MeshShardConfig(
        ccl=CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear),
        mesh=tuple(mesh_device.shape),
        enter_stage=enter_stage,
    )

    generator = torch.Generator().manual_seed(0)
    latent = torch.randn(1, config["in_channels"], latent_frames, *latent_hw, generator=generator)
    latent_tt, padded_dims = decoder.upload_latent(latent)
    grid = Grid(
        batch=1,
        t=decoder.context_frames(latent.shape[2]),
        h=padded_dims[1] * decoder.space_scale[0],
        w=padded_dims[2] * decoder.space_scale[1],
    )
    noise = torch.randn(
        1,
        config["out_channels"],
        grid.t,
        grid.h * decoder.patch_size,
        grid.w * decoder.patch_size,
        generator=generator,
    )
    noise_tt = decoder.stage5.upload_x_t(noise, shard=shard)
    timestep = ttnn.from_torch(
        torch.tensor([[[[1.0]]]]), device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )

    def device_half(latent_in, noise_in, timestep_in):
        return decoder.decode_device(
            latent_in,
            noise_in,
            timestep_in,
            grid,
            padded_dims=padded_dims,
            latent_frames=latent.shape[2],
            shard=shard,
        )

    start = time.perf_counter()
    eager = device_half(latent_tt, noise_tt, timestep)
    ttnn.synchronize_device(mesh_device)
    eager_s = time.perf_counter() - start
    expected = decoder.stage5.unpack_pixels(eager, grid, shard=shard)

    traced = Tracer(device_half, device=mesh_device)
    start = time.perf_counter()
    out = traced(latent_tt, noise_tt, timestep)
    ttnn.synchronize_device(mesh_device)
    capture_s = time.perf_counter() - start

    start = time.perf_counter()
    out = traced(latent_tt, noise_tt, timestep)
    ttnn.synchronize_device(mesh_device)
    replay_s = time.perf_counter() - start

    actual = decoder.stage5.unpack_pixels(out, grid, shard=shard)
    print(
        f"\nTRACE grid={grid} enter_stage={enter_stage}\n"
        f"  eager   {eager_s:7.2f}s\n  capture {capture_s:7.2f}s\n  replay  {replay_s:7.2f}s"
        f"   ({eager_s / replay_s:.2f}x vs eager)"
    )
    assert_quality(expected, actual, pcc=0.999)
