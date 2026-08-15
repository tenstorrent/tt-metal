# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The whole DiffVAE decoder, split across the mesh, against upstream's own pixels.

``test_diffvae_decoder`` holds the replicated decoder to this same capture. This holds the
split one to it, so a passing run means the sharded path reproduces upstream end to end and not
merely itself: deterministic stages, the ghost pad and crop, the handoff into stage 5, the
noise upload, eight diffusion blocks and the unpatchify back to pixels.

``enter_stage`` is 3 rather than 1 because of this capture's geometry. Its stage-1 and stage-2
grids are 20 wide, which a mesh of 8 does not divide; by stage 3 the volume is 40 wide and it
does. The split is checked stage by stage, so a too-early entry fails at the split instead of
quietly rebalancing.
"""

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn

from ....models.vae.diffvae_ltx import DiffVAEDecoder, MeshShardConfig, decoder_config
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality

CAPTURE = Path(os.environ.get("DIFFVAE_CAPTURE", "/home/noblewoodall/ltx25_diffvae/stages/crop10.safetensors"))
CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)

_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}


def _captured(*names: str) -> tuple[torch.Tensor, ...]:
    with safe_open(str(CAPTURE), "pt") as handle:
        available = set(handle.keys())
        if missing := [name for name in names if name not in available]:
            pytest.skip(f"{CAPTURE.name} lacks {missing}; regenerate without --pixels-only")
        return tuple(handle.get_tensor(name).float() for name in names)


@pytest.mark.parametrize(
    "mesh_device, device_params, enter_stage",
    [((4, 8), _FABRIC, 3)],
    indirect=["mesh_device", "device_params"],
)
def test_sharded_decode_matches_upstream(*, mesh_device, enter_stage):
    if not os.access(CAPTURE, os.R_OK):
        pytest.skip(f"missing or unreadable {CAPTURE}; run capture_stages.py first")
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")

    latent, noise, expected = _captured("input.latent", "stage5.noise", "output.pixels")

    decoder = DiffVAEDecoder(decoder_config(CHECKPOINT), mesh_device=mesh_device)
    decoder.load_checkpoint(CHECKPOINT)

    shard = MeshShardConfig(
        ccl=CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear),
        mesh=tuple(mesh_device.shape),
        enter_stage=enter_stage,
    )
    pixels = decoder.decode(latent, noise=noise, shard=shard)

    assert tuple(pixels.shape) == tuple(expected.shape), f"{tuple(pixels.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, pixels, pcc=0.99)
