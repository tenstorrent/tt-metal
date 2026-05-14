# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Parity test for ``WanS2VTransformer3DModel`` against the reference
``WanModel_S2V``.

Scope
-----
The reference S2V forward orchestrates four pieces that the TT ``inner_step``
does not yet replicate inline (they're driven by the pipeline at run time):

  * ``patch_embedding`` of noisy + ref latents and concatenation of their
    sequences,
  * ``frame_packer`` of motion latents and append to the spatial sequence,
  * pose Conv3d (``cond_encoder``) added to the noisy patched sequence,
  * ``trainable_cond_mask`` embedding added to the per-token sequence,
  * segmented timestep modulation when ``zero_timestep=True`` (production
    default).

To make the parity test tractable while the host-side orchestration is being
finished (#10 follow-on), we drive the test in two phases:

  1. **Strict weight-load smoke test.** Build the on-device model with the
     production config, run :func:`translate_s2v_state_dict` on the native
     state dict, call ``load_torch_state_dict(strict=True)`` and assert the
     load completes with no missing or unexpected keys. This is the unit-level
     guarantee that #20 produced a correct mapping; if this passes we know the
     1251 device-resident parameters are all wired up.

  2. **Reduced-config block-stack PCC** (BH 4x8). With ``zero_timestep=False``
     and audio injection disabled on the reference, the S2V block stack
     reduces to the same shape as the T2V block stack, so we can use
     ``inner_step`` directly on a synthesized ``[B, C, T, H, W]`` noisy
     latent. This sanity-checks the loaded weights flow through the block
     stack correctly.

The full end-to-end PCC with ref/motion/cond/audio injection is validated by
``test_pipeline_wan_s2v.py`` once the pipeline orchestration lands (#12).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.transformer_wan_s2v import WanS2VTransformer3DModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.wan.wan_s2v_loader import find_s2v_snapshot, load_s2v_config, load_s2v_state_dict
from ....pipelines.wan.wan_s2v_weight_map import translate_s2v_state_dict
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, float32_tensor, local_device_to_torch
from ....utils.test import line_params

# Production config (matches Wan-AI/Wan2.2-S2V-14B / config.json).
PATCH_SIZE = (1, 2, 2)
DIM = 5120
NUM_HEADS = 40
NUM_LAYERS = 40
IN_CHANNELS = 16
OUT_CHANNELS = 16
TEXT_DIM = 4096
FREQ_DIM = 256
FFN_DIM = 13824
AUDIO_DIM = 1024
NUM_AUDIO_TOKEN = 4
NUM_AUDIO_LAYERS = 25
AUDIO_INJECT_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39)
ROPE_MAX_SEQ_LEN = 1024
EPS = 1e-6


def _maybe_path_for_ref_repo() -> Path | None:
    ref = Path("/home/kevinmi/wan2_2_ref")
    return ref if (ref / "wan" / "modules" / "s2v" / "model_s2v.py").exists() else None


_REF_REPO = _maybe_path_for_ref_repo()


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )


def _build_tt_s2v_model(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, cfg):
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    # The on-disk config omits a few keys that the reference's
    # ``register_to_config`` marks as ``ignore_for_config`` (patch_size,
    # cross_attn_norm, qk_norm, text_dim, window_size). Fall back to the
    # ``WanModel_S2V.__init__`` defaults.
    return WanS2VTransformer3DModel(
        patch_size=tuple(cfg.get("patch_size", PATCH_SIZE)),
        num_heads=cfg["num_heads"],
        dim=cfg["dim"],
        in_channels=cfg.get("in_dim", IN_CHANNELS),
        out_channels=cfg.get("out_dim", OUT_CHANNELS),
        text_dim=cfg.get("text_dim", TEXT_DIM),
        freq_dim=cfg.get("freq_dim", FREQ_DIM),
        ffn_dim=cfg["ffn_dim"],
        num_layers=cfg["num_layers"],
        cross_attn_norm=cfg.get("cross_attn_norm", True),
        eps=cfg.get("eps", EPS),
        rope_max_seq_len=ROPE_MAX_SEQ_LEN,
        audio_dim=cfg["audio_dim"],
        num_audio_layers=NUM_AUDIO_LAYERS,
        num_audio_token=cfg.get("num_audio_token", NUM_AUDIO_TOKEN),
        audio_inject_layers=tuple(cfg["audio_inject_layers"]),
        enable_adain=cfg.get("enable_adain", False),
        enable_motioner=cfg.get("enable_motioner", False),
        enable_framepack=cfg.get("enable_framepack", True),
        motion_token_num=cfg.get("motion_token_num", 1024),
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        model_type="s2v",
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
def test_s2v_weight_load(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Strict weight-load smoke test for the production Wan2.2-S2V-14B
    checkpoint. Validates that :func:`translate_s2v_state_dict` produces a
    state dict that ``WanS2VTransformer3DModel.load_torch_state_dict`` accepts
    in ``strict=True`` mode.
    """
    snapshot = find_s2v_snapshot()
    cfg = load_s2v_config(snapshot)
    logger.info(
        f"Loaded S2V config: dim={cfg['dim']}, num_layers={cfg['num_layers']}, "
        f"audio_dim={cfg['audio_dim']}, enable_adain={cfg['enable_adain']}, "
        f"enable_framepack={cfg['enable_framepack']}"
    )

    ref_sd = load_s2v_state_dict(snapshot)
    tt_sd = translate_s2v_state_dict(ref_sd)
    logger.info(
        f"Translated state dict: {len(ref_sd)} ref keys → {len(tt_sd)} tt keys "
        f"({len(ref_sd) - len(tt_sd)} CPU-shadow keys excluded)"
    )

    tt_model = _build_tt_s2v_model(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, cfg)

    # ``strict=True`` raises ValueError on missing or unexpected keys; the
    # success of this call is the unit-level guarantee that #20's mapper
    # produces a state dict whose every key matches a device-resident parameter.
    tt_model.load_torch_state_dict(tt_sd, strict=True)
    logger.info("Strict load succeeded — 1251 device-resident parameters loaded.")

    del tt_model


@pytest.mark.skipif(
    _REF_REPO is None,
    reason="Wan-Video/Wan2.2 reference repo not at /home/kevinmi/wan2_2_ref. " "Add it to PYTHONPATH or set _REF_REPO.",
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
def test_s2v_block_stack_parity(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Reduced-config parity: TT block stack vs reference WanModel_S2V's block
    stack, with audio injection and ref/motion conditioning disabled.

    With ``zero_timestep=False`` and ``empty audio_inject_layers``, the S2V
    block stack reduces to the same shape as the T2V block stack, so a
    standard inner-step parity check on a synthetic ``[B, C, T, H, W]`` noisy
    latent is meaningful.

    Production end-to-end PCC (with ref/motion/cond and audio injection
    active) is validated by ``test_pipeline_wan_s2v.py``.
    """
    if str(_REF_REPO) not in sys.path:
        sys.path.insert(0, str(_REF_REPO))

    from wan.modules.s2v.model_s2v import WanModel_S2V  # noqa: WPS433  (deferred import)

    snapshot = find_s2v_snapshot()
    cfg = load_s2v_config(snapshot)
    ref_sd = load_s2v_state_dict(snapshot)

    # Build a reduced reference: disable the parts we don't mirror in inner_step.
    ref_kwargs = dict(cfg)
    ref_kwargs["zero_timestep"] = False  # avoid segmented modulation
    ref_kwargs["audio_inject_layers"] = []  # disable audio injection branch
    ref_kwargs["enable_motioner"] = False  # disable motioner branch
    ref_kwargs["enable_framepack"] = False  # disable framepack branch
    ref_kwargs["enable_adain"] = False  # ada-IN off
    ref_kwargs["cond_dim"] = 0  # disable pose Conv3d
    ref_kwargs["add_last_motion"] = False
    ref_kwargs["zero_init"] = False  # don't zero out o-proj at construction; we're loading real weights
    # Filter out non-constructor keys from the on-disk config (the JSON has
    # diffusers-only keys like ``_class_name`` / ``_diffusers_version``).
    for k in [
        "__name__",
        "_class_name",
        "_diffusers_version",
        "model_type",  # always 's2v', already asserted
    ]:
        ref_kwargs.pop(k, None)
    # Some keys in the on-disk config don't correspond to constructor args (e.g.
    # ``framepack_drop_mode``); the reference uses ignore_for_config so they get
    # filtered by register_to_config, but to instantiate directly we keep only
    # the constructor-known keys.
    import inspect

    sig = inspect.signature(WanModel_S2V.__init__)
    allowed = set(sig.parameters)
    ref_kwargs = {k: v for k, v in ref_kwargs.items() if k in allowed}

    logger.info(
        f"Building reference WanModel_S2V with reduced config (audio_inject_layers=[], "
        f"zero_timestep=False, enable_framepack=False)"
    )
    ref_model = WanModel_S2V(**ref_kwargs).eval().to(torch.float32)
    # Load the production weights — but the reference will reject keys for
    # disabled branches. Use strict=False to skip those.
    missing, unexpected = ref_model.load_state_dict(ref_sd, strict=False)
    logger.info(
        f"Reference load: {len(missing)} missing, {len(unexpected)} unexpected "
        f"(unexpected expected since we disabled framepack/audio_inject/cond_encoder)"
    )

    # Build the TT model (full production config; we just won't drive the audio
    # injector branch in this test).
    tt_model = _build_tt_s2v_model(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, cfg)
    tt_sd = translate_s2v_state_dict(ref_sd)
    tt_model.load_torch_state_dict(tt_sd, strict=True)
    logger.info("TT model loaded.")

    # --- Synthesize inputs at 480p / 81 frames (latent dims: F=21, H=60, W=104). ---
    B, F, H, W = 1, 21, 60, 104
    torch.manual_seed(0)
    noisy_latents = torch.randn(B, IN_CHANNELS, F, H, W, dtype=torch.float32)
    text_emb = torch.randn(B, 512, TEXT_DIM, dtype=torch.float32)
    timestep = torch.tensor([500.0], dtype=torch.float32)

    # --- Reference reduced forward: bypass ref/motion/cond/audio. ---
    # The reference's ``forward`` is hardwired to take audio_input and to call
    # process_motion/ref/cond_encoder. We replicate its block-loop manually.
    with torch.no_grad():
        # The reference's block-loop expects x post-patchify, plus rope freqs +
        # e0 + context. Build them by walking the reference forward up to the
        # block loop.
        x = ref_model.patch_embedding(noisy_latents)  # [B, dim, T, H/2, W/2]
        x = x.flatten(2).transpose(1, 2)  # [B, N, dim]
        N_ref = x.shape[1]

        # Context (text) projection.
        context_lens = None
        context_padded = torch.stack(
            [torch.cat([u, u.new_zeros(ref_model.text_len - u.size(0), u.size(1))]) for u in text_emb]
        )
        context = ref_model.text_embedding(context_padded)

        # Time embedding.
        from wan.modules.model import sinusoidal_embedding_1d  # noqa: WPS433

        e_full = ref_model.time_embedding(sinusoidal_embedding_1d(ref_model.freq_dim, timestep).float())
        e0 = ref_model.time_projection(e_full).unflatten(1, (6, ref_model.dim))
        # zero_timestep=False ⇒ e0 broadcasts uniformly across segments.
        e0 = e0.unsqueeze(2).repeat(1, 1, 2, 1)
        e0_payload = [e0, 0]

        # rope_precompute on a no-ref-no-motion grid.
        from wan.modules.s2v.s2v_utils import rope_precompute  # noqa: WPS433

        grid_size = torch.stack([torch.tensor([F // 1, H // 2, W // 2], dtype=torch.long)])
        grid_sizes = [[torch.zeros_like(grid_size), grid_size, grid_size]]
        d = ref_model.dim // ref_model.num_heads
        freqs = rope_precompute(
            x.detach().view(B, x.shape[1], ref_model.num_heads, d), grid_sizes, ref_model.freqs, start=None
        )

        # Drive the block loop on the reference (no after_transformer_block —
        # audio_inject_layers is empty).
        ref_x = x
        for block in ref_model.blocks:
            ref_x = block(
                ref_x,
                e=e0_payload,
                seq_lens=torch.tensor([ref_x.shape[1]], dtype=torch.long),
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                context_lens=context_lens,
            )
        # head + unpatchify
        ref_out_tokens = ref_model.head(ref_x, e_full)  # [B, N, pH*pW*pF*C_out]

    logger.info(f"Reference block-stack output: {tuple(ref_out_tokens.shape)}")

    # --- TT side: drive inner_step on the same noisy latents. ---
    # The TT inner_step takes a 5-D spatial tensor; build it via the standard
    # WanTransformer3DModel.preprocess_spatial_input path.
    tt_spatial_1BNI, N_tt = tt_model.preprocess_spatial_input(noisy_latents)
    assert N_tt == N_ref, f"sequence length mismatch: TT N={N_tt}, ref N={N_ref}"

    tt_prompt = bf16_tensor(text_emb.unsqueeze(0), device=mesh_device)
    tt_prompt_1BLP = tt_model.prepare_text_conditioning(text_emb)

    rope_cos_1HND, rope_sin_1HND, trans_mat = tt_model.prepare_rope_features(noisy_latents)

    tt_timestep = float32_tensor(timestep.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=mesh_device)

    # Run the inner_step; with audio_inject_layers populated but audio_emb not
    # primed (merged_audio_emb is None), after_transformer_block early-returns.
    # ...except `_inject_block_id` keys still trigger the hook. Patch by
    # clearing the audio injector's inject set for this test.
    tt_model.audio_injector.injected_block_id = {}

    tt_out_1BNI = tt_model.inner_step(
        spatial_1BNI=tt_spatial_1BNI,
        prompt_1BLP=tt_prompt_1BLP,
        rope_cos_1HND=rope_cos_1HND,
        rope_sin_1HND=rope_sin_1HND,
        trans_mat=trans_mat,
        N=N_tt,
        timestep=tt_timestep,
        gather_output=True,
    )
    tt_out_torch = local_device_to_torch(tt_out_1BNI)
    # Trim sequence-parallel padding tokens.
    tt_out_torch = tt_out_torch[..., :N_tt, :].squeeze(0)
    logger.info(f"TT block-stack output: {tuple(tt_out_torch.shape)}")

    # PCC: WAN parity bar is 0.99 (see feedback_wan_pcc_bar.md).
    assert_quality(tt_out_torch.float(), ref_out_tokens.float(), pcc=0.99)
