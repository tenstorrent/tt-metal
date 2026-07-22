# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from huggingface_hub import snapshot_download

import ttnn

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _fibo_local():
    try:
        return snapshot_download(FIBO_PATH, local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO not cached: {e}")


@pytest.mark.parametrize("mesh_device", [(2, 2), (4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=["device_params"],
)
def test_fibo_pipeline_smoke(*, mesh_device):
    """Full end-to-end FIBO text->image smoke on the mesh (2x2 sp=2/tp=2, or 4x8 Galaxy sp=4/tp=8).

    Encode (SmolLM3, replicated) -> 30-step CFG flow-match denoise (BriaFibo transformer) -> Wan 2.2
    residual VAE decode -> 1024x1024 image. No reference comparison at full steps (the per-step path
    is PCC-gated elsewhere); this only asserts the pipeline runs and produces a finite (1024, 1024, 3)
    image, and saves the PNG for visual inspection.
    """
    import numpy as np

    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig

    ckpt = _fibo_local()
    pipe = BriaFiboPipeline(
        device=mesh_device,
        config=BriaFiboPipelineConfig.default(
            mesh_shape=mesh_device.shape, checkpoint_name=ckpt, height=1024, width=1024
        ),
    )
    # force_device_decode=True: the Wan 2.2 residual VAE decode runs on-device (full 2x2 submesh,
    # hw-parallel) with NO host-torch fallback -- if the on-device path regresses, the test fails here.
    imgs = pipe("a luxury sports car", num_inference_steps=30, guidance_scale=5.0, seed=0, force_device_decode=True)

    arr = np.asarray(imgs[0])
    # (1024,1024,3) AND non-degenerate: `np.isfinite` on a uint8 array is vacuously true, so instead
    # assert the image has real variation (a black/uniform frame would fail both bounds below).
    assert arr.shape == (1024, 1024, 3), f"unexpected image shape {arr.shape}"
    assert arr.std() > 1.0, f"image looks degenerate (std={arr.std():.4f})"
    assert np.unique(arr).size > 16, f"image looks degenerate ({np.unique(arr).size} unique values)"
    imgs[0].save("fibo_smoke.png")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=["device_params"],
)
def test_fibo_pipeline_e2e_image_golden(*, mesh_device):
    """End-to-end WITH golden: the tt pipeline's decoded IMAGE matches the diffusers reference image.

    Companion to ``test_fibo_pipeline_smoke`` (end-to-end WITHOUT golden). Runs the full text->image
    path (encode -> CFG denoise -> on-device VAE decode) on BOTH the tt pipeline and the host diffusers
    reference, fed IDENTICAL injected noise, and compares the two decoded images via PCC.

    Reduced resolution (512x512 -> 32x32 = 1024 tokens, the smallest no-attention-padding size; steps=4)
    to keep the CPU reference tractable -- a native-res 30-step reference is many CPU-hours. The tt image
    is decoded ON-DEVICE (``force_device_decode=True``). Slow (loads + runs the fp32 reference incl. its
    VAE decode) -- run on-demand, not in fast CI.
    """
    import gc

    import torch

    from models.tt_dit.utils.check import assert_quality

    ckpt = _fibo_local()

    steps = 4
    guidance_scale = 5.0
    height = width = 512
    in_channels = 48
    latent_h = height // 16
    latent_w = width // 16
    prompt = (
        "A luxury sports car in vivid detail: sleek aerodynamic silver bodywork with sharp creases, "
        "large matte black alloy wheels, low ground clearance, glowing LED headlights, parked on a wet "
        "city street at night reflecting neon signs, cinematic photography, shallow depth of field, ultra "
        "realistic, 8k resolution, dramatic lighting, professional automotive advertisement style."
    )

    # Shared initial noise, packed to the (1, h*w, C) layout both pipelines' ``latents=`` expects.
    torch.manual_seed(0)
    init_bchw = torch.randn(1, in_channels, latent_h, latent_w, dtype=torch.float32)
    init_packed = init_bchw.permute(0, 2, 3, 1).reshape(1, latent_h * latent_w, in_channels).contiguous()

    # --- Reference (diffusers, host CPU, fp32) -> decoded image; then free ~44 GB before the tt build. ---
    from diffusers import BriaFiboPipeline as RefBriaFiboPipeline

    ref = RefBriaFiboPipeline.from_pretrained(ckpt, torch_dtype=torch.float32)
    ref_img = ref(
        prompt,
        negative_prompt="",
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        latents=init_packed.clone(),
        output_type="pt",
    ).images  # (1, 3, 512, 512) in [0, 1]
    ref_img = ref_img.detach().float().cpu()
    del ref
    gc.collect()

    # --- TT pipeline (2x2 Blackhole mesh), on-device VAE decode. ---
    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig

    pipe = BriaFiboPipeline(
        device=mesh_device,
        config=BriaFiboPipelineConfig.default(mesh_shape=mesh_device.shape, checkpoint_name=ckpt),
    )
    tt_img = (
        pipe(
            prompt,
            negative_prompt="",
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            latents=init_packed.clone(),
            output_type="pt",
            force_device_decode=True,
        )
        .float()
        .cpu()
    )

    assert tuple(tt_img.shape) == tuple(ref_img.shape), f"{tt_img.shape} != {ref_img.shape}"
    # Image PCC is lower than the pre-VAE latent PCC (the bf16 VAE decode adds drift on top of the
    # denoise drift). 0.95 is a conservative floor; tighten to the measured value once run on-device.
    assert_quality(ref_img, tt_img, pcc=0.95)


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=["device_params"],
)
def test_fibo_pipeline_vae_decode_on_device(*, mesh_device):
    """The pipeline's Wan 2.2 residual VAE decode runs ON-DEVICE and matches the host-torch decode.

    Exercises ``BriaFiboPipeline._decode_vae`` (the exact production decode path) at the production
    resolution (64x64 latent -> 1024x1024 image) on the 2x2 Blackhole mesh. The VAE decodes on the same
    2x2 submesh as the transformer (hw-parallel halo/CCL decode), so this also proves the residual
    decoder coexists in device memory with the resident transformer/encoder shards.

    Two assertions:
      1. ``force_device_decode=True`` must NOT raise -- i.e. the on-device weight-load + decode succeeds
         (before the ``decoder_base_dim`` fix in ``vae_wan2_1.py`` this raised ``LoadingError`` at
         ``decoder.conv_in.weight``: built (1728, 640) vs prepared (1728, 1024), on every mesh size).
      2. The on-device image matches the host-torch reference (``_host_decode_vae``, i.e. HF
         ``AutoencoderKLWan.decode``) at high PCC (~99.9%, matching the isolated residual decode).

    Feeds a synthetic normalized latent directly to ``_decode_vae`` (no denoise loop) for speed; the
    denoise->latent path is PCC-gated by ``test_fibo_pipeline_latent_pcc``.
    """
    import torch

    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig
    from models.tt_dit.utils.check import assert_quality

    ckpt = _fibo_local()
    pipe = BriaFiboPipeline(
        device=mesh_device,
        config=BriaFiboPipelineConfig.default(
            mesh_shape=mesh_device.shape, checkpoint_name=ckpt, height=1024, width=1024
        ),
    )

    # Synthetic normalized latent in the BCTHW form _decode_vae expects (adapter denormalizes internally).
    torch.manual_seed(0)
    latent_hw = 1024 // 16  # 64
    z = torch.randn(1, pipe._in_channels, 1, latent_hw, latent_hw, dtype=torch.float32)

    # 1. On-device decode -- force_device_decode=True re-raises instead of silently falling back to host.
    dev_img = pipe._decode_vae(z, force_device_decode=True)  # (1, 3, 1, 1024, 1024)
    assert tuple(dev_img.shape) == (1, 3, 1, 1024, 1024), f"unexpected decode shape {tuple(dev_img.shape)}"

    # 2. Host-torch reference decode of the SAME latent, compared at high PCC.
    host_img = pipe._host_decode_vae(z)
    assert_quality(host_img.float(), dev_img.float(), pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(2, 2), (4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=["device_params"],
)
@pytest.mark.parametrize("guidance_scale", [5.0, 1.0])
def test_fibo_pipeline_latent_pcc(*, mesh_device, guidance_scale):
    """Core end-to-end gate: the tt ``BriaFiboPipeline`` reproduces the diffusers reference
    ``BriaFiboPipeline``'s pre-VAE latent trajectory (PCC-gated) on the mesh (2x2 sp=2/tp=2, or
    4x8 Galaxy sp=4/tp=8).

    Both pipelines run the SAME prompt / negative_prompt / guidance_scale / steps / resolution and are
    fed IDENTICAL initial noise via latent INJECTION (a host-built tensor passed as ``latents=`` to
    both) -- this sidesteps the host/device RNG mismatch (``torch.manual_seed``+float32-randn->bf16 will
    NOT bit-match the reference's ``randn_tensor(generator, dtype=...)``). Both return the pre-VAE latent
    (``output_type="latent"``); the two ``(1, h*w, 48)`` latents are compared with ``assert_quality``.

    Reduced config (spec §5, §7 risk 2): height=width=512 -> 32x32 = 1024 spatial tokens. 1024 ==
    k_chunk_size(512) * sp_factor(2), so the spatial sequence needs NO attention padding (the pipeline
    has none) -- the smallest clean reduced resolution. The glue (noise->pack->rope->CFG->solver->latent)
    is resolution-independent and every component is already PCC-validated, so this faithfully exercises
    the wiring while keeping the CPU reference (8B transformer x2 CFG branches) tractable. steps=4
    (the whole test runs in ~3 min; 4 steps exercises 4 distinct sigma/timestep transitions).

    Threshold 0.99: a real wiring bug (wrong pack transpose, rope split, CFG sign, solver index,
    sigma/timestep mismatch) craters PCC; bf16-vs-fp32 drift over a few steps does not.
    """
    import gc

    import torch

    from models.tt_dit.utils.check import assert_quality

    ckpt = _fibo_local()

    # guidance_scale is parametrized: 5.0 exercises CFG (both branches); 1.0 exercises the no-CFG
    # gate (uncond branch skipped) -- both must reproduce the reference, which gates identically.
    steps = 4
    height = width = 512
    in_channels = 48
    latent_h = height // 16
    latent_w = width // 16
    # A realistic-length caption: short/empty prompts have a known ~1% position-0 causal-LM encoder PCC
    # gap (see test_wrapper_encode_matches_reference) that would confound the glue signal here.
    prompt = (
        "A luxury sports car in vivid detail: sleek aerodynamic silver bodywork with sharp creases, "
        "large matte black alloy wheels, low ground clearance, glowing LED headlights, parked on a wet "
        "city street at night reflecting neon signs, cinematic photography, shallow depth of field, ultra "
        "realistic, 8k resolution, dramatic lighting, professional automotive advertisement style."
    )

    # Shared initial noise: built on host in the reference's UNPACKED (1, C, h, w) form, then packed to
    # the (1, h*w, C) layout that BOTH pipelines' ``latents=`` expects (reference `_pack_latents_no_patch`;
    # the reference returns injected ``latents`` as-is, so it must already be packed).
    torch.manual_seed(0)
    init_bchw = torch.randn(1, in_channels, latent_h, latent_w, dtype=torch.float32)
    init_packed = init_bchw.permute(0, 2, 3, 1).reshape(1, latent_h * latent_w, in_channels).contiguous()

    # --- Reference (diffusers, host CPU, fp32) — run first, then free ~44 GB before the tt build. ---
    from diffusers import BriaFiboPipeline as RefBriaFiboPipeline

    ref = RefBriaFiboPipeline.from_pretrained(ckpt, torch_dtype=torch.float32)
    ref_latent = ref(
        prompt,
        negative_prompt="",
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        latents=init_packed.clone(),
        output_type="latent",
    ).images  # (1, h*w, 48) packed pre-VAE latent
    ref_latent = ref_latent.detach().float().cpu()
    del ref
    gc.collect()

    # --- TT pipeline (2x2 Blackhole mesh) ---
    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig

    pipe = BriaFiboPipeline(
        device=mesh_device,
        config=BriaFiboPipelineConfig.default(mesh_shape=mesh_device.shape, checkpoint_name=ckpt),
    )
    tt_latent = pipe(
        prompt,
        negative_prompt="",
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        latents=init_packed.clone(),
        output_type="latent",
    )

    assert tuple(tt_latent.shape) == tuple(ref_latent.shape), f"{tt_latent.shape} != {ref_latent.shape}"
    assert_quality(ref_latent, tt_latent.float(), pcc=0.99)


def test_build_text_encoder_layers_pads_37_to_46():
    from models.tt_dit.pipelines.bria_fibo.text_encoder import build_text_encoder_layers

    hs = [f"h{i}" for i in range(37)]  # stand-in objects
    out = build_text_encoder_layers(hs, 46)
    assert len(out) == 46
    assert out[:37] == hs
    assert out[37:] == [hs[-1]] * 9  # last state repeated 9x
    # right-trim when longer than num_blocks
    assert build_text_encoder_layers([f"h{i}" for i in range(50)], 46) == [f"h{i}" for i in range(4, 50)]
