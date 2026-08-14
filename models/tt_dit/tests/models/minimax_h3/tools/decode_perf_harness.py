# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Whole-stage `MiniMaxH3Vae.decode` timing -- measurement driver, not a gate.

`test_performance_vae_minimax_h3.py` times one decoder wave; this times the stage the pipeline
actually bills to "VAE decode", including tiling, upload, readback, unpatchify and stitching, and
prints `_report_profile`'s breakdown. That breakdown is the point: a stage total alone cannot say
which of device / readback / stitch moved.

Built with a `CCLManager` because `_read_wave_units` picks its reader on `ccl_manager is None` --
without one a single-host run silently takes the `ConcatMeshToTensor` path and measures something
production never runs.

    export MINIMAX_H3_MODEL_PATH=/home/kevinmi/minimax-h3
    ./python_env/bin/python -m pytest models/tt_dit/tests/models/minimax_h3/tools/decode_perf_harness.py \\
      -k "4x8 and 5s" -s --timeout 1800
"""

import json
import os
import time

import pytest
import torch

import ttnn

from .....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig
from .....parallel.manager import CCLManager
from ..common import weights_subdir

# 1344x768 at 16x spatial compression. Latent frame counts follow the 17n -> 5n rule less
# `token_drop`, so 124 frames (5 s) is 37 and 362 (15 s) is 107 -- the two rows in `MiniMaxH3.md`.
LATENT_HEIGHT, LATENT_WIDTH = 48, 84
LATENT_FRAMES = {5: 37, 15: 107}
# 7 latent frames is one chunk, the cheapest input that still exercises a whole wave.
WARM_LATENT_FRAMES = 7

MESH_4X8 = [
    pytest.param(
        (4, 8),
        {
            # RING, not FABRIC_1D: `_decode_clip_device_stitched` hardcodes
            # `all_gather(topology=ttnn.Topology.Ring)`, so the `yuv420` mode cannot run on a linear
            # fabric at all. Ring is also the production tt_dit configuration, so measuring the other
            # two modes here keeps all three comparable on one fabric.
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
            # The traced mode captures one per-chunk graph (~1270 ops); 150 MB is what the denoise
            # loop uses on the quad. Harmless for the untraced modes.
            "trace_region_size": 150000000,
        },
        id="4x8",
    )
]


def _decoder_state(weights_dir: str) -> dict[str, torch.Tensor]:
    """Just the decoder-side tensors. The checkpoint is 10.4 GB and over half of it is the encoder."""
    from safetensors.torch import load_file

    index_path = os.path.join(weights_dir, "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return {
            k: v
            for k, v in load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")).items()
            if k.startswith(("decoder.", "post_quant_conv."))
        }

    weight_map = json.loads(open(index_path).read())["weight_map"]
    wanted = {k: f for k, f in weight_map.items() if k.startswith(("decoder.", "post_quant_conv."))}
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(wanted.values())):
        loaded = load_file(os.path.join(weights_dir, shard))
        state.update({k: loaded[k] for k in wanted if k in loaded})
    return state


# ImageNet constants the decoder's pixels are still normalized by. Duplicated from
# `pipelines/minimax_h3/conditioning.py` rather than imported, so this driver does not drag the whole
# pipeline module in for two tuples.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

# Each mode is one lever, so a row of this table is a measurement of that lever alone.
MODES = {
    # Today's readback: bf16 tiles in token space, blended and unpatchified on host.
    "float": {},
    # Same path, uint8 across PCIe. Halves the transfer; costs <=1 LSB at the seams.
    "uint8": {"pixel_denorm": (PIXEL_MEAN, PIXEL_STD), "readback_uint8": True},
    # Stitch, clamp and colour-convert on device; read one planar canvas at 1.5 bytes/pixel.
    "yuv420": {"pixel_denorm": (PIXEL_MEAN, PIXEL_STD), "device_stitch": True},
    # Same, with the per-chunk device graph captured once and replayed. Isolates host dispatch.
    "traced": {"pixel_denorm": (PIXEL_MEAN, PIXEL_STD), "device_stitch": True, "trace_device_stitch": True},
}


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("seconds", [5, 15], ids=["5s", "15s"])
@pytest.mark.parametrize("mode", sorted(MODES), ids=sorted(MODES))
def test_decode_stage(mesh_device, seconds, mode):
    from loguru import logger

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")

    config = MiniMaxH3VaeConfig.from_pretrained(weights_dir)
    # Ring to match the fabric above. Unused on a single host -- `fast_device_to_host` and
    # `fast_device_to_host_yuv` only consult it inside their `using_distributed_env()` branches --
    # but Linear here would be wrong the moment this runs on the quad.
    ccl_manager = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    vae = MiniMaxH3Vae(config, mesh_device=mesh_device, ccl_manager=ccl_manager, **MODES[mode])
    output_type = "yuv420" if mode in ("yuv420", "traced") else "float"

    t0 = time.time()
    vae.load_decoder_state(_decoder_state(weights_dir))
    logger.info(f"decoder state read in {time.time() - t0:.1f}s")

    # Build the per-shape decoder outside the timed region, as `_prepare_vae(decode_shape=...)` does:
    # its weight upload is seconds and would otherwise land inside the first `decode`.
    t0 = time.time()
    vae._decoder_for(config.tokens_chunk_size + config.token_overlap, 16, 16)
    logger.info(f"per-shape decoder built in {time.time() - t0:.1f}s")

    torch.manual_seed(0)
    warm = torch.randn(1, config.latent_channels, WARM_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    latents = torch.randn(1, config.latent_channels, LATENT_FRAMES[seconds], LATENT_HEIGHT, LATENT_WIDTH)

    logger.info("warming (program cache + allocator)")
    vae.decode(warm, output_type=output_type)

    logger.info(f"MEASURING mode={mode} {seconds}s: latents {tuple(latents.shape)}")
    started = time.time()
    video = vae.decode(latents, output_type=output_type)
    elapsed = time.time() - started

    profile = dict(vae.last_decode_profile)
    logger.info(f"DECODE_STAGE mode={mode} seconds={seconds} total={elapsed:.3f}s video={tuple(video.shape)}")
    logger.info(
        f"DECODE_SPLIT mode={mode} seconds={seconds} "
        + " ".join(
            f"{k}={profile.get(k, 0.0):.3f}"
            for k in (
                "device",
                "readback",
                "stitch",
                "unpatchify",
                "blend",
                "concat",
                "dispatch",
                "compute",
                "tiling",
                "upload",
                "host_prep",
                "residual",
            )
        )
        + f" waves={int(profile.get('waves', 0))} units={int(profile.get('units', 0))}"
        + f" readback_gb={profile.get('readback_mb', 0.0) / 1000:.2f}"
    )
