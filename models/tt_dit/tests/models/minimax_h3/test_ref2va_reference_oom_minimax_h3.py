# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Reproduce tt-inference-server#5044's ref2va reference-count OOMs at the conditioning-encode stage.

The measured matrix (quad BH galaxy, fresh worker per case, 16:9 / 5 s target):

    6 images                     pass, 475 s
    7 images                     OOM  2,868,903,936 B, 158 s
    9 images                     OOM  2,868,903,936 B, 111 s
    3 images + 3 videos          OOM  2,868,903,936 B, 250 s
    6 images + 3 videos + 3 aud  OOM    717,225,984 B, 325 s

The pass/fail boundary is monotone in conditioning-encoder load (presentation tokens / tower
patches) and NOT in the DiT packed length (3 videos pass at 150k padded rows while 7 images fail
at 95k), so this test drives exactly the suspect stage -- ``prepare_references`` +
``encode_prompt`` -- and nothing after it: no reference VAE encode, no transformer, no denoise.

Two deliberate differences from production, both of which give the stage MORE headroom here:

* Standalone, the DiT / VAE / warmup residue is not resident (the quad preset is coresident), so
  a case that OOMs in production may pass here. ``H3_OOM_BALLAST_GB`` allocates that many GiB of
  DRAM per device before the encode to stand in for the missing floor -- raise it until the
  passing control (6 images) still passes and the ticket's failing cases fail.
* On a 4x8 the tower runs sp8 (per-device patch shards 4x the quad's sp32); the text conditioner
  is TP-only on both meshes, so its per-device, presentation-length-scaled footprint -- where the
  byte counts point -- is identical to the quad's.

Run one case per process (an OOM poisons in-process state for later cases -- the ticket measured
exactly that), e.g.:

    MINIMAX_H3_MODEL_PATH=/data/cglagovich/MiniMax-H3-diffusers \
        pytest models/tt_dit/tests/models/minimax_h3/test_ref2va_reference_oom_minimax_h3.py \
        -k 7img -s
"""

import os

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.references import prepare_references
from .common import MESH_4X8_RING
from .common_av import weights_dir

PROMPT = "Family theme park not too far from the road, with a nice picnic area and families having a good time."
NUM_FRAMES = 124  # the ticket's 5 s target: 17n + 5 aligned, 5.167 s at 24 fps

# (images, videos, audios) per case, with the ticket's measured outcome in the id. The 6-image
# control brackets the boundary: if it OOMs here too, the ballast is set too high to be meaningful.
CASES = [
    pytest.param(6, 0, 0, id="6img_ticket_pass"),
    pytest.param(7, 0, 0, id="7img_ticket_oom_2.87GB"),
    pytest.param(9, 0, 0, id="9img_ticket_oom_2.87GB"),
    pytest.param(3, 3, 0, id="3img_3vid_ticket_oom_2.87GB"),
    pytest.param(6, 3, 3, id="6img_3vid_3aud_ticket_oom_0.72GB"),
]

MESHES = [
    pytest.param(shape, {**params, "l1_small_size": 16384}, id=param.id)
    for param in [MESH_4X8_RING]
    for shape, params in [param.values]
]


def _dram_line(mesh_device, label: str) -> None:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    gib = 1024**3
    allocated = view.total_bytes_allocated_per_bank * view.num_banks
    free = view.total_bytes_free_per_bank * view.num_banks
    largest = view.largest_contiguous_bytes_free_per_bank * view.num_banks
    logger.info(
        f"DRAM[{label}] allocated={allocated / gib:.2f} GiB free={free / gib:.2f} GiB "
        f"largest_contiguous={largest / gib:.2f} GiB (per device, {view.num_banks} banks)"
    )


def _references(num_images: int, num_videos: int, num_audios: int, sample_rate: int) -> list[MiniMaxH3Reference]:
    """Synthetic references matching the ticket's shapes.

    Images are 512x512 noise ON PURPOSE: the 512 -> 2048 short-edge upscale inside
    ``prepare_references`` is part of the mechanism under test, and the ticket measured 256/512/1024
    px images failing identically. Videos are generated at 24 fps directly on the 16:9 canvas
    (768x1344) so host prep is a no-op -- the ticket's vid_5s was 16:9 too, and the conditioner
    load depends on the canvas, not the source pixels.
    """
    references = []
    generator = torch.Generator().manual_seed(0)
    for _ in range(num_images):
        pixels = (torch.rand(512, 512, 3, generator=generator) * 255).to(torch.uint8).numpy()
        references.append(MiniMaxH3Reference(image=Image.fromarray(pixels)))
    for _ in range(num_videos):
        frames = (torch.rand(120, 768, 1344, 3, generator=generator) * 255).to(torch.uint8).numpy()
        references.append(MiniMaxH3Reference(video=frames, fps=24.0))
    for _ in range(num_audios):
        waveform = torch.randn(2, 5 * sample_rate, generator=generator) * 0.1
        references.append(MiniMaxH3Reference(audio=waveform, sample_rate=sample_rate))
    return references


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(("num_images", "num_videos", "num_audios"), CASES)
def test_ref2va_reference_encode(mesh_device, num_images, num_videos, num_audios, reset_seeds):
    # topology=Linear: this galaxy has no wrap-around cabling on mesh axis 0 (the fabric realizes
    # TORUS_Y, not TORUS_XY), so the preset's Ring topology dies in every axis-0 TP collective.
    # Linear routes all of them; see test_ref2va_sample_run.py.
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("text_encoder"),
        topology=ttnn.Topology.Linear,
    )

    references = _references(num_images, num_videos, num_audios, pipeline.audio_sampling_rate)
    prepared, num_frames = prepare_references(references, NUM_FRAMES, pipeline.audio_sampling_rate)
    assert num_frames == NUM_FRAMES

    _dram_line(mesh_device, "before encode")

    # Stand-in for the resident DiT + VAE + warmup residue the standalone stage doesn't carry.
    # One DRAM tensor per device, held alive across the encode.
    ballast = None
    ballast_gib = float(os.environ.get("H3_OOM_BALLAST_GB", "0") or 0)
    if ballast_gib > 0:
        rows = int(ballast_gib * 1024**3) // (1024 * 2)  # [rows, 1024] bf16
        ballast = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        _dram_line(mesh_device, f"after {ballast_gib:g} GiB ballast")

    # The suspect stage. On the ticket's failing cases this is where the OOM should surface;
    # the TT_FATAL traceback then names the allocating op, which is the point of this test.
    embeds, tags = pipeline.encode_prompt(PROMPT, references=prepared)

    _dram_line(mesh_device, "after encode")
    del ballast

    # A/B hook: dump the embeds so two parallel configs of the conditioner can be PCC-compared
    # (e.g. the TP-only decoder against the SP one) without holding both on device at once.
    dump = os.environ.get("H3_OOM_DUMP_EMBEDS")
    if dump:
        torch.save({"embeds": embeds, "tags": tags}, dump)
        logger.info(f"dumped embeds {tuple(embeds.shape)} to {dump}")

    expected_image_tokens = num_images * 4096  # every image is upscaled to 2048^2 -> 4096 merged tokens
    vision_tags = int((tags == 0).sum())
    logger.info(
        f"encode completed: L={embeds.shape[1]} tokens, {vision_tags} video-tagged, "
        f"references={num_images}img+{num_videos}vid+{num_audios}aud"
    )
    assert embeds.ndim == 3 and embeds.shape[-1] == 5120, f"unexpected embeds shape {tuple(embeds.shape)}"
    assert embeds.shape[1] == tags.shape[0]
    assert torch.isfinite(embeds).all(), "prompt embeds contain NaN or Inf"
    assert vision_tags >= expected_image_tokens, (
        f"presentation is missing image vision rows: {vision_tags} video-tagged tokens < "
        f"{expected_image_tokens} expected from {num_images} images alone"
    )
