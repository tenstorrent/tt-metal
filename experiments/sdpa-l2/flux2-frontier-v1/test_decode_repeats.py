# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode saved repeatability latents with the same TT VAE configuration."""

import hashlib
import json
import os
from pathlib import Path

from PIL import Image
import pytest
import torch
import ttnn


@pytest.fixture
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 256 * 1024**2}


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
@pytest.mark.timeout(600)
def test_decode_repeats(mesh_device, monkeypatch):
    from diffusers import AutoencoderKLFlux2
    from models.tt_dit.models.vae.vae import VaeAttention
    from models.tt_dit.models.vae.vae_flux2 import Flux2VaeDecoder
    from models.tt_dit.parallel.config import Flux2VaeParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import from_torch, fast_device_to_host, float_to_uint8

    torch.set_num_threads(8)
    source = Path(os.environ["FLUX2_REPEAT_SOURCE"])
    output = source / "decoded"
    output.mkdir(exist_ok=False)
    report = json.loads((source / "report.json").read_text())
    assert report["status"] == "completed"
    checkpoint = os.environ["FLUX2_CHECKPOINT"]
    monkeypatch.setattr(VaeAttention, "sdpa_chunk_size_map", {(True, 2, 1, 4): (64, 64)})
    reference = AutoencoderKLFlux2.from_pretrained(checkpoint, subfolder="vae")
    manager = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear)
    parallel = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=1, h_axis=0, w_axis=None)
    decoder = Flux2VaeDecoder(
        out_channels=3,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        z_channels=32,
        device=mesh_device,
        parallel_config=parallel,
        ccl_manager=manager,
        use_conv3d=False,
    )
    decoder.load_torch_state_dict(reference.state_dict())
    rows = []
    for record in report["rows"]:
        if record["mode"] == "capture":
            continue
        name = f"{record['index']}-{record['mode']}"
        latent_path = source / f"{name}.pt"
        value = torch.load(latent_path, weights_only=True)
        assert bool(torch.isfinite(value).all())
        tensor = from_torch(value, device=mesh_device, mesh_axes=[None, 0, None])
        tensor = decoder.preprocess_and_unpatchify(tensor, height=128, width=128)
        decoded = decoder.forward(tensor, traced=False)
        first = ttnn.to_torch(ttnn.get_device_tensors(decoded)[0])
        assert bool(torch.isfinite(first).all()), "Non-finite VAE output before uint8 conversion"
        host = fast_device_to_host(
            decoded, mesh_device, [None, None], ccl_manager=manager, pre_transfer_fn=float_to_uint8
        )
        image_path = output / f"{name}.png"
        Image.fromarray(host[0].numpy()).save(image_path)
        rows.append(
            dict(
                index=record["index"],
                mode=record["mode"],
                image=image_path.name,
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                latents_sha256=hashlib.sha256(latent_path.read_bytes()).hexdigest(),
            )
        )
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                status="completed",
                source=str(source),
                decoder="TT VAE, untraced, 2x4 SP2/TP4, Q64/K64; decoder scheduling held fixed across repeats",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
