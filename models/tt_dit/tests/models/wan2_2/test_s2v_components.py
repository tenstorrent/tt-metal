# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Component-level parity tests for S2V-specific modules.

Covers components that don't fall under the per-block PCC test
(``test_segmented_block_math_parity``) and aren't covered by the existing
audio / VAE / weight-load suites:

  * :class:`FramePackMotionerWan` — three Conv3d-style projections + rope
    precompute. Compare against the reference's ``FramePackMotioner``.
  * (TODO) Audio injector cross-attention with the block-diagonal mask vs
    the reference's per-frame ``rearrange`` injection.

Test bar is PCC ≥ 0.99 per ``feedback_wan_pcc_bar.md``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.motioner_wan import FramePackMotionerWan
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import local_device_to_torch
from ....utils.test import line_params

_REF_REPO = Path("/home/kevinmi/wan2_2_ref")
_S2V_ASSETS_REF = _REF_REPO / "examples"


def _install_wan_ref_stubs() -> None:
    """Stub the wan repo's CUDA/flash_attn/decord imports for CPU execution."""
    import importlib.machinery

    def _make_stub(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)  # type: ignore[attr-defined]
        return mod

    if "flash_attn" not in sys.modules:
        mod = _make_stub("flash_attn")
        mod.flash_attn_func = None  # type: ignore[attr-defined]
        sys.modules["flash_attn"] = mod
    if "decord" not in sys.modules:
        mod = _make_stub("decord")
        mod.VideoReader = None  # type: ignore[attr-defined]
        mod.cpu = lambda x=0: None  # type: ignore[attr-defined]
        sys.modules["decord"] = mod
    if not hasattr(torch.cuda, "_orig_current_device"):
        torch.cuda._orig_current_device = torch.cuda.current_device  # type: ignore[attr-defined]
        torch.cuda.current_device = lambda: 0  # type: ignore[assignment]


@pytest.mark.skipif(
    not (_REF_REPO / "wan" / "modules" / "s2v" / "motioner.py").exists(),
    reason="Wan-Video/Wan2.2 reference repo not at /home/kevinmi/wan2_2_ref",
)
@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8),
            (4, 8),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_4x8sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_frame_packer_parity(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Per-module parity for FramePackMotionerWan vs reference FramePackMotioner.

    Drives both on the same random motion-latent input. Both produce the same
    flat motion-token tensor ``[B, N_motion, inner_dim]``. Bar: PCC ≥ 0.99.
    """
    torch.manual_seed(0)
    if str(_REF_REPO) not in sys.path:
        sys.path.insert(0, str(_REF_REPO))
    _install_wan_ref_stubs()

    from wan.modules.s2v.motioner import FramePackMotioner

    # Small config — same shape contract as production but smaller dim.
    # Production: inner_dim=5120, num_heads=40, zip_frame_buckets=(1,2,16),
    # drop_mode="padd". We keep the same zip_frame_buckets so the motion-token
    # count matches production geometry; dim is smaller for fast test.
    INNER_DIM = 512
    NUM_HEADS = 4
    ZIP_FRAME_BUCKETS = (1, 2, 16)
    DROP_MODE = "padd"  # production default per the S2V config

    # Production motion latent shape (480p): [B=1, C=16, T_motion=5, lat_h=60, lat_w=104].
    # Keep T_motion=5 (production); shrink spatial to speed up the test.
    B, C, T_motion, lat_h, lat_w = 1, 16, 5, 8, 16

    # Reference module.
    ref = (
        FramePackMotioner(
            inner_dim=INNER_DIM,
            num_heads=NUM_HEADS,
            zip_frame_buckets=list(ZIP_FRAME_BUCKETS),
            drop_mode=DROP_MODE,
        )
        .eval()
        .to(torch.float32)
    )
    logger.info(
        f"Reference FramePackMotioner built: inner_dim={INNER_DIM}, "
        f"zip_buckets={ZIP_FRAME_BUCKETS}, drop_mode={DROP_MODE}"
    )

    # TT module (single TP shard for fast test; production uses tp=4).
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    tt = FramePackMotionerWan(
        in_channels=16,
        inner_dim=INNER_DIM,
        num_heads=NUM_HEADS,
        zip_frame_buckets=ZIP_FRAME_BUCKETS,
        drop_mode=DROP_MODE,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
    )

    # Transfer ref weights → tt. Both have ``proj``, ``proj_2x``, ``proj_4x``
    # whose ``weight`` shape is ``[inner_dim, 16, kT, kH, kW]`` (Conv3d format).
    # Our WanPatchEmbed accepts the same Conv3d-style state dict.
    ref_sd = ref.state_dict()
    # Filter to just the projection weights/biases (skip ``freqs`` which is a buffer).
    proj_keys = [k for k in ref_sd if k.startswith(("proj.", "proj_2x.", "proj_4x."))]
    block_sd = {k: ref_sd[k] for k in proj_keys}
    incompat = tt.load_torch_state_dict(block_sd, strict=False)
    logger.info(f"TT load: missing={len(incompat.missing_keys)} " f"unexpected={len(incompat.unexpected_keys)}")

    # Random motion-latents input. Same on both sides.
    motion = torch.randn(B, C, T_motion, lat_h, lat_w, dtype=torch.float32)

    # --- Reference forward ---
    with torch.no_grad():
        # Ref takes a list-per-batch and returns ``(mot_list, mot_rope_list)``;
        # each list element is the tokens / rope for that batch.
        ref_motion = [motion[0]]  # list of [C, T, H, W]
        ref_mot_list, _ref_rope_list = ref.forward(ref_motion, add_last_motion=2)
        ref_tokens = ref_mot_list[0]  # [1, N_motion, inner_dim]
    logger.info(f"Reference tokens: {tuple(ref_tokens.shape)}")

    # --- TT forward ---
    tt_tokens_dev, tt_rope = tt.forward(motion, add_last_motion=2)
    # tt_tokens_dev shape: [1, B, N_motion, inner_dim] TP-fractured on last dim.
    # Gather across TP to get full inner_dim.
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    if parallel_config.tensor_parallel.factor > 1:
        tt_tokens_gathered = ccl_manager.all_gather_persistent_buffer(tt_tokens_dev, dim=3, mesh_axis=tp_axis)
    else:
        tt_tokens_gathered = tt_tokens_dev
    tt_tokens_torch = local_device_to_torch(tt_tokens_gathered)
    # tt_tokens_torch shape: [1, B, N_motion, inner_dim]. Reshape to match ref [B, N, D].
    tt_tokens_torch = tt_tokens_torch.squeeze(0)  # [B, N, D]
    logger.info(f"TT tokens: {tuple(tt_tokens_torch.shape)}")

    assert_quality(tt_tokens_torch.float(), ref_tokens.float(), pcc=0.99)


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "h_axis", "w_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((4, 8), (4, 8), 0, 1, 2, line_params, ttnn.Topology.Linear, id="bh_4x8_h0w1"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_vae_encode_pipeline_vs_host(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    h_axis: int,
    w_axis: int,
    num_links: int,
    topology: ttnn.Topology,
) -> None:
    """Verify the **pipeline's** VAE-encode-then-normalize matches the host
    (Diffusers) reference for the canonical S2V input.

    The TT VAE encoder by itself is already covered by
    ``test_vae_wan2_1.py::test_wan_encoder`` (PCC ≥ 0.995 on raw mean). This
    test adds two pieces that test only validates indirectly:

      1. The S2V pipeline's ``_encode_normalized`` helper applies the
         production ``(mu - latents_mean) / latents_std`` normalization
         on top of the raw encoder output.
      2. The encode runs at the **exact** image size and chunk parameters
         the pipeline uses (``encoder_t_chunk_size=16``), so any blocking-
         lookup miss for the production H/W shows up here.

    Bar: PCC ≥ 0.99 vs ``diffusers.AutoencoderKLWan._encode``.
    """
    import PIL.Image
    import torch.nn.functional as F  # noqa: N812
    from diffusers.models import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    from ....models.vae.vae_wan2_1 import WanEncoder
    from ....parallel.config import VaeHWParallelConfig
    from ....utils.conv3d import conv_pad_height, conv_pad_in_channels
    from ....utils.tensor import bf16_tensor_2dshard, fast_device_to_host

    # Load canonical S2V input image — same as the pipeline test uses.
    img_path = _S2V_ASSETS_REF / "i2v_input.JPG"
    if not img_path.exists():
        pytest.skip(f"S2V reference image not at {img_path}")
    ref_pil = PIL.Image.open(img_path)

    HEIGHT, WIDTH = 480, 832
    Z_DIM = 16

    # Host VAE (Diffusers).
    host_vae = (
        AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae").eval().to(torch.float32)
    )
    latents_mean = torch.tensor(host_vae.config.latents_mean).view(1, Z_DIM, 1, 1, 1)
    latents_std = torch.tensor(host_vae.config.latents_std).view(1, Z_DIM, 1, 1, 1)

    # Pre-process to [B, C=3, T=1, H, W] in [-1, 1].
    vp = VideoProcessor(vae_scale_factor=host_vae.config.scale_factor_spatial)
    pixel = vp.preprocess(ref_pil, height=HEIGHT, width=WIDTH).to(torch.float32)
    pixel_BCTHW = pixel.unsqueeze(2)  # [1, 3, 1, H, W]

    # --- Host reference: raw encode then normalize. ---
    with torch.no_grad():
        raw = host_vae._encode(pixel_BCTHW)
        host_mean = raw[:, :Z_DIM]
        host_norm = (host_mean - latents_mean) / latents_std
    logger.info(
        f"Host VAE normalized latent: shape={tuple(host_norm.shape)} "
        f"mean={host_norm.float().mean():.3f} std={host_norm.float().std():.3f}"
    )

    # --- TT VAE encoder. ---
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_enc = WanEncoder(
        base_dim=host_vae.config.base_dim,
        in_channels=host_vae.config.in_channels,
        z_dim=Z_DIM,
        dim_mult=host_vae.config.dim_mult,
        num_res_blocks=host_vae.config.num_res_blocks,
        attn_scales=host_vae.config.attn_scales,
        temperal_downsample=host_vae.config.temperal_downsample,
        is_residual=host_vae.config.is_residual,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        dtype=ttnn.bfloat16,
    )
    tt_enc.load_torch_state_dict(host_vae.state_dict())

    # Upload pixel tensor as the pipeline does it.
    pix_BTHWC = pixel_BCTHW.permute(0, 2, 3, 4, 1)
    pix_BTHWC = conv_pad_in_channels(pix_BTHWC)
    pix_BTHWC, logical_h = conv_pad_height(
        pix_BTHWC, parallel_config.height_parallel.factor * host_vae.config.scale_factor_spatial
    )
    pix_dev = bf16_tensor_2dshard(
        pix_BTHWC,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={h_axis: 2, w_axis: 3},
    )
    raw_BCTHW, new_logical_h = tt_enc(pix_dev, logical_h, encoder_t_chunk_size=16)
    concat_dims = [None, None]
    concat_dims[h_axis] = 3
    concat_dims[w_axis] = 4
    raw_tt = fast_device_to_host(raw_BCTHW, mesh_device, concat_dims, ccl_manager=ccl_manager)[
        :, :, :, :new_logical_h, :
    ]
    tt_norm = (raw_tt.float() - latents_mean.float()) / latents_std.float()
    logger.info(
        f"TT VAE normalized latent: shape={tuple(tt_norm.shape)} "
        f"mean={tt_norm.float().mean():.3f} std={tt_norm.float().std():.3f}"
    )

    # Crop to common shape (TT may have extra padding rows).
    if tt_norm.shape != host_norm.shape:
        h_min = min(tt_norm.shape[3], host_norm.shape[3])
        w_min = min(tt_norm.shape[4], host_norm.shape[4])
        tt_norm = tt_norm[:, :, :, :h_min, :w_min]
        host_norm = host_norm[:, :, :, :h_min, :w_min]
        logger.info(f"Cropped both to {tuple(tt_norm.shape)}")

    assert_quality(tt_norm.float(), host_norm.float(), pcc=0.99)
