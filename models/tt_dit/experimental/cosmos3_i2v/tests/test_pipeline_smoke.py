# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 MVP smoke tests for Cosmos3-Super-Image2Video.

Three tests in increasing scope:

  - `test_cosmos3_trunk_load_and_wrap`: 4-layer truncated trunk on 1 chip
    with stock (replicating) TTNNLinear. Validates the load + wrap +
    placement mechanic end-to-end. Currently passes.

  - `test_cosmos3_trunk_sharded_galaxy`: FULL 64-layer trunk on a 4x8 BH
    mesh with `TTNNLinearMeshShard`. Validates that the full 64B model
    fits when its Linear weights are sharded across the mesh. Does NOT
    run a forward — the sharded Linear is placement-only for now.

  - `test_cosmos3_i2v_smoke`: full pipeline. Skipped pending
    Qwen3VLVisionModel availability (needs transformers >= 4.57).

Run on a dev box with a TT device:

    pytest models/tt_dit/experimental/cosmos3_i2v/tests/test_pipeline_smoke.py::test_cosmos3_trunk_load_and_wrap -s
    pytest models/tt_dit/experimental/cosmos3_i2v/tests/test_pipeline_smoke.py::test_cosmos3_trunk_sharded_galaxy -s
"""

from __future__ import annotations

import pytest

from models.tt_dit.utils.test import line_params


@pytest.mark.timeout(7200)
def test_cosmos3_trunk_load_and_wrap(device):
    """Phase 1 mechanic test: load truncated 4-layer transformer, wrap, place on 1 chip.

    Full 64B doesn't fit on 1 chip; tt-symbiote stock Linear replicates rather than
    shards, so even on a 4x8 mesh each chip independently OOMs. We truncate to 4
    layers (~4B params, ~8GB FP16, fits in 32GB per BH chip with headroom) to validate
    the load + wrap + device-place mechanic end-to-end.
    """
    from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_i2v import build_cosmos3_i2v_transformer_only

    transformer, all_modules, param_count, num_layers = build_cosmos3_i2v_transformer_only(device, max_layers=4)

    print(f"Loaded transformer: {num_layers} layers, {param_count / 1e9:.2f}B params (pre-wrap count)")
    print(f"tt-symbiote wrapped modules: {len(all_modules)}")

    assert param_count > 1_000_000_000, f"expected >1B params after truncation, got {param_count / 1e9:.2f}B"
    assert len(all_modules) > 0, "no modules were wrapped by tt-symbiote"
    assert num_layers == 4, f"expected truncation to 4 layers, got {num_layers}"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(4, 8), line_params],  # full 32-chip BH Galaxy
    ],
    indirect=True,
    ids=["bh_4x8"],
)
@pytest.mark.timeout(7200)
def test_cosmos3_trunk_sharded_galaxy(mesh_device):
    """Phase 1 sharding test: full 64-layer trunk on 4x8 BH mesh with sharded Linear.

    Validates that `TTNNLinearMeshShard` places the full 64B-param trunk
    across the Galaxy by sharding each Linear's weight along its
    output-feature dim across mesh axis 1. Expected per-chip footprint:
    ~128GB / 8 = 16GB along axis 1, replicated 4x along axis 0.

    Does NOT run a forward (the sharded Linear's forward is placement-only);
    success means weights successfully placed across the mesh.
    """
    from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_i2v import build_cosmos3_i2v_transformer_only

    transformer, all_modules, param_count, num_layers = build_cosmos3_i2v_transformer_only(
        mesh_device, shard_linear=True
    )

    print(f"Loaded full transformer: {num_layers} layers, {param_count / 1e9:.2f}B params (pre-wrap count)")
    print(f"tt-symbiote wrapped modules: {len(all_modules)} (sharded across {tuple(mesh_device.shape)} mesh)")

    assert num_layers == 64, f"expected full 64 layers, got {num_layers}"
    assert param_count > 50_000_000_000, f"expected >50B params, got {param_count / 1e9:.2f}B"
    assert len(all_modules) > 0, "no modules were wrapped by tt-symbiote"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(4, 8), line_params],  # full 32-chip BH Galaxy for the sharded 64B model
    ],
    indirect=True,
    ids=["bh_4x8"],
)
@pytest.mark.timeout(7200)
def test_cosmos3_i2v_smoke(mesh_device):
    """Full pipeline end-to-end: image + prompt → 1 latent frame.

    Loads transformer (sharded across the 4x8 BH mesh), vision encoder,
    VAE, scheduler, and tokenizer; runs 1 inference step with
    output_type='latent' to skip VAE decode. Validates the entire wiring
    on Galaxy.
    """
    import torch
    from PIL import Image

    from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_i2v import build_cosmos3_i2v_pipeline

    pipe = build_cosmos3_i2v_pipeline(mesh_device, dtype=torch.bfloat16, shard_linear=True)

    # Minimum viable I2V: small resolution, few frames, single step.
    # num_frames must be > 1 (the docstring notes image is ignored when num_frames == 1)
    # and probably needs to be 4k+1 for the Wan2.2-TI2V VAE's 4x temporal compression.
    ref_image = Image.new("RGB", (256, 256), (128, 128, 128))
    prompt = "a cat walks across a sunlit kitchen floor"

    result = pipe(
        image=ref_image,
        prompt=prompt,
        num_frames=17,  # 4*4 + 1 = 17, matches TI2V VAE temporal stride
        height=256,
        width=256,
        num_inference_steps=1,
        output_type="latent",
    )

    # Cosmos3OmniPipelineOutput's main field is `video`, which with
    # output_type="latent" holds the raw latent tensor.
    latent = result.video
    print(f"Generated latent type: {type(latent).__name__}, shape: {tuple(latent.shape)}")
    assert latent.numel() > 0


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(1, 8), line_params],
        [(4, 8), line_params],
    ],
    indirect=True,
    ids=["wh_1x8", "bh_4x8"],
)
@pytest.mark.timeout(14400)
def test_cosmos3_i2v_generate_video(mesh_device):
    """Full I2V generate: image + prompt → VAE-decoded MP4 on disk.

    Same load path as the smoke tests but drops `output_type="latent"` so
    the pipeline runs the VAE decode and returns PIL frames; we then write
    those frames to an MP4 via `diffusers.utils.export_to_video`.

    On `(1, 8)` (WH LoudBox / T3K) this uses `weight_dtype=ttnn.bfloat8_b`
    to fit in 12 GB/chip; on `(4, 8)` (BH Galaxy) it stays on bfloat16.

    The VAE (`AutoencoderKLWan`, ~700M) is NOT wrapped by tt-symbiote, so
    decode happens as plain PyTorch on host. For a 17-frame 256×256 clip
    that's ~30–60 s on CPU; bigger sizes are slower in proportion. VAE
    tiling is enabled (set in `build_cosmos3_i2v_pipeline`) so memory is
    bounded.

    Knobs via env vars (so this test doubles as a sweep driver):
      COSMOS3_STEPS       UniPC denoise steps (default 10).
                          1 ≈ noise, 8–15 ≈ structured, 30–40 ≈ full quality.
      COSMOS3_FRAMES      Number of frames (default 17). Must be `4k+1` for
                          the TI2V VAE's 4× temporal compression.
      COSMOS3_HW          Square frame size (default 256).
      COSMOS3_IMAGE       Path to a reference image. Default is a solid gray —
                          model can't extrapolate much from a flat color, so
                          a real image is recommended for any meaningful view.
      COSMOS3_PROMPT      Text prompt (default "a cat walks across a sunlit
                          kitchen floor").
      COSMOS3_OUT         Output MP4 path (default
                          `/tmp/cosmos3_i2v_<mesh>.mp4`).
      COSMOS3_FPS         MP4 framerate (default 16, matches base_fps in config).
    """
    import os

    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    import ttnn
    from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_i2v import build_cosmos3_i2v_pipeline

    mesh_shape = tuple(mesh_device.shape)
    is_loudbox = mesh_shape == (1, 8)
    weight_dtype = ttnn.bfloat8_b if is_loudbox else ttnn.bfloat16

    steps = int(os.environ.get("COSMOS3_STEPS", "10"))
    frames = int(os.environ.get("COSMOS3_FRAMES", "17"))
    hw = int(os.environ.get("COSMOS3_HW", "256"))
    fps = int(os.environ.get("COSMOS3_FPS", "16"))
    prompt = os.environ.get("COSMOS3_PROMPT", "a cat walks across a sunlit kitchen floor")
    mesh_tag = f"{mesh_shape[0]}x{mesh_shape[1]}"
    out_path = os.environ.get("COSMOS3_OUT", f"/tmp/cosmos3_i2v_{mesh_tag}.mp4")

    image_path = os.environ.get("COSMOS3_IMAGE")
    if image_path:
        ref_image = Image.open(image_path).convert("RGB").resize((hw, hw))
        print(f"Reference image: {image_path} ({ref_image.size})")
    else:
        ref_image = Image.new("RGB", (hw, hw), (128, 128, 128))
        print(f"Reference image: solid-gray {hw}x{hw} (set COSMOS3_IMAGE for real visuals)")

    print(
        f"Generating: mesh={mesh_shape}, weight_dtype={weight_dtype}, "
        f"steps={steps}, frames={frames}, {hw}x{hw}, fps={fps}"
    )

    pipe = build_cosmos3_i2v_pipeline(
        mesh_device,
        dtype=torch.bfloat16,
        shard_linear=True,
        weight_dtype=weight_dtype,
    )

    result = pipe(
        image=ref_image,
        prompt=prompt,
        num_frames=frames,
        height=hw,
        width=hw,
        num_inference_steps=steps,
    )

    video_frames = result.video
    print(f"Decoded {len(video_frames)} frames; writing MP4 to {out_path}")
    export_to_video(video_frames, out_path, fps=fps)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"MP4 written: {out_path} ({size_kb:.1f} KB)")
    assert size_kb > 1, "MP4 file suspiciously tiny; VAE decode or muxing likely failed"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(1, 8), line_params],  # WH LoudBox / T3K — 8 chips, 12 GB DRAM each
    ],
    indirect=True,
    ids=["wh_1x8"],
)
@pytest.mark.timeout(7200)
def test_cosmos3_i2v_smoke_loudbox(mesh_device):
    """LoudBox variant: BFP8 weights on a 1x8 WH mesh.

    Same image+prompt → latent flow as `test_cosmos3_i2v_smoke`, but uses
    `weight_dtype=ttnn.bfloat8_b` so the 64B trunk fits on a box with only
    12 GB DRAM per chip. At BFP8 the per-chip footprint is ~8 GB sharded
    8-way along mesh axis 1, leaving ~4 GB of headroom for activations.
    bfloat16 weights would want ~16 GB/chip and OOM.
    """
    import torch
    from PIL import Image

    import ttnn
    from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_i2v import build_cosmos3_i2v_pipeline

    pipe = build_cosmos3_i2v_pipeline(
        mesh_device,
        dtype=torch.bfloat16,
        shard_linear=True,
        weight_dtype=ttnn.bfloat8_b,
    )

    ref_image = Image.new("RGB", (256, 256), (128, 128, 128))
    prompt = "a cat walks across a sunlit kitchen floor"

    result = pipe(
        image=ref_image,
        prompt=prompt,
        num_frames=17,
        height=256,
        width=256,
        num_inference_steps=1,
        output_type="latent",
    )

    latent = result.video
    print(f"Generated latent type: {type(latent).__name__}, shape: {tuple(latent.shape)}")
    assert latent.numel() > 0
