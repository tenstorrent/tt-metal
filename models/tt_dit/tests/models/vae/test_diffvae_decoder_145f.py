# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The DiffVAE decoder at the resolution it exists to reach: 145 frames of 1088x1920.

This is the claim the sharding work was built to test. Replicated, stage 5's context is 9.7 GB
per chip and its ``context_and_x`` buffer 19.4 GB, which is where a whole-volume decode dies;
split over a 4x8 the same tensors are 0.405 GB and 0.81 GB. Every other test here runs on
volumes small enough that both paths fit, so none of them can tell whether that arithmetic is
right.

There is no upstream capture at this size — the intermediates would be tens of GB — so what is
checked is that the decode completes and returns the pixel grid the frame arithmetic predicts.
Correctness against upstream is covered at crop scale by ``test_diffvae_decoder_sharded``.

``enter_stage`` is 1: the latent grid is 34 wide and a mesh of 4 does not divide it, but the
first upsample takes it to 68 and everything from there divides.
"""

import os
import time
from pathlib import Path

import pytest
import torch

import ttnn

from ....models.vae.diffvae_ltx import DiffVAEDecoder, MeshShardConfig, decoder_config
from ....parallel.manager import CCLManager
from ....utils.ltx import TEMPORAL_COMPRESSION

CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)

_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}


def _dram_in_use(mesh_device) -> str:
    """Per-bank DRAM after the decode, as a figure rather than a survival claim."""
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        used = view.total_bytes_allocated_per_bank * view.num_banks
        total = view.total_bytes_per_bank * view.num_banks
        return f"{used / 1e9:.2f} GB of {total / 1e9:.2f} GB per device across {view.num_banks} banks"
    except Exception as err:  # noqa: BLE001 - reporting only
        return f"unavailable ({type(err).__name__}: {err})"


@pytest.mark.parametrize(
    "mesh_device, device_params, num_frames, height, width",
    [((4, 8), _FABRIC, 145, 1088, 1920)],
    indirect=["mesh_device", "device_params"],
)
def test_sharded_decode_at_production_size(*, mesh_device, num_frames, height, width):
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")

    mesh = tuple(mesh_device.shape)
    config = decoder_config(CHECKPOINT)
    spatial = 32  # latent grid to pixels, composed over the four upsamples and the patch size
    latent = torch.randn(
        1,
        config["in_channels"],
        (num_frames - 1) // TEMPORAL_COMPRESSION + 1,
        height // spatial,
        width // spatial,
        generator=torch.Generator().manual_seed(0),
    )

    decoder = DiffVAEDecoder(config, mesh_device=mesh_device)
    decoder.load_checkpoint(CHECKPOINT)

    shard = MeshShardConfig(
        ccl=CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear),
        mesh=mesh,
        enter_stage=1,
    )

    from ....models.vae.diffvae_ltx_stage5 import Grid

    # Warm first: a cold call folds every program's compilation into whichever phase touches it.
    decoder.decode(latent, seed=0, shard=shard)

    start = time.perf_counter()
    pixels = decoder.decode(latent, seed=0, shard=shard)
    elapsed = time.perf_counter() - start

    # Same work again, split, so the host halves are attributable rather than inferred.
    t0 = time.perf_counter()
    latent_tt, padded_dims = decoder.upload_latent(latent)
    grid = Grid(
        batch=1,
        t=decoder.context_frames(latent.shape[2]),
        h=padded_dims[1] * decoder.space_scale[0],
        w=padded_dims[2] * decoder.space_scale[1],
    )
    noise = torch.randn(
        (1, config["out_channels"], grid.t, grid.h * decoder.patch_size, grid.w * decoder.patch_size),
        generator=torch.Generator().manual_seed(0),
    )
    t1 = time.perf_counter()
    noise_tt = decoder.stage5.upload_x_t(noise, shard=shard)
    t2 = time.perf_counter()
    timestep = ttnn.from_torch(
        torch.tensor([[[[1.0]]]]), device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    out = decoder.decode_device(
        latent_tt, noise_tt, timestep, grid, padded_dims=padded_dims, latent_frames=latent.shape[2], shard=shard
    )
    ttnn.synchronize_device(mesh_device)
    t3 = time.perf_counter()
    decoder.stage5.unpack_pixels(out, grid, shard=shard)
    t4 = time.perf_counter()
    print(
        f"\n  PHASES  latent+noise host {t1 - t0:6.2f}s | upload_x_t {t2 - t1:6.2f}s | "
        f"device {t3 - t2:6.2f}s | unpack {t4 - t3:6.2f}s"
    )

    expected = (1, config["out_channels"], num_frames, height, width)
    print(
        f"\n145f SPLIT DECODE: latent {tuple(latent.shape)} -> pixels {tuple(pixels.shape)} "
        f"in {elapsed:.1f}s on {mesh[0]}x{mesh[1]}\n  DRAM: {_dram_in_use(mesh_device)}"
    )
    assert tuple(pixels.shape) == expected, f"{tuple(pixels.shape)} != {expected}"
    assert torch.isfinite(pixels).all(), "decode produced non-finite pixels"
