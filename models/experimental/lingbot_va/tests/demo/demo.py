# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lingbot-VA demo: RobotWin-style observations and TTNN inference (VA_Server-compatible).

All PyTorch tensors use CPU; TT workloads run on Tenstorrent mesh via TTNN.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from copy import deepcopy
from pathlib import Path

# Before Hugging Face tokenizers import (avoids fork warning under multiprocessing).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

# Repo root: lingbot_va (parent of tests/demo)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# TT-Metal root so tt.utils can import models.tt_dit (for TT transformer)
_TT_METAL_ROOT = os.environ.get("TT_METAL_HOME") or str(_REPO_ROOT.parent.parent.parent)
if os.path.isdir(_TT_METAL_ROOT) and _TT_METAL_ROOT not in sys.path:
    sys.path.insert(0, _TT_METAL_ROOT)

import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from reference.utils import (
    VA_CONFIGS,
    FlowMatchScheduler,
    apply_robotwin_inference_overrides,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    load_tokenizer,
    load_vae,
    logger,
    WanVAEStreamingWrapper,
)

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager

from models.experimental.lingbot_va.tt.transformer_wan import NUM_HEADS as LINGBOT_NUM_HEADS
from tt.utils import (
    load_text_encoder as load_text_encoder_tt,
    load_transformer as load_transformer_tt,
    WanVAEStreamingWrapper as TTWanVAEStreamingWrapper,
    WanVAEDecoderWrapper as TTWanVAEDecoderWrapper,
)

OBS_CAM_HIGH = "observation.images.cam_high"
OBS_CAM_LEFT_WRIST = "observation.images.cam_left_wrist"
OBS_CAM_RIGHT_WRIST = "observation.images.cam_right_wrist"
OBS_STATE = "observation.state"

REPRODUCIBLE_SEED = 42


def _set_seed(seed: int = REPRODUCIBLE_SEED) -> None:
    """Set random seeds so that inference is reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _release_ttnn_runtime_configs(obj, _visited: set[int] | None = None) -> None:
    """Best-effort recursive teardown of TTNN runtime config objects."""
    if obj is None:
        return
    if _visited is None:
        _visited = set()
    oid = id(obj)
    if oid in _visited:
        return
    _visited.add(oid)

    if isinstance(obj, dict):
        for v in list(obj.values()):
            _release_ttnn_runtime_configs(v, _visited)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in list(obj):
            _release_ttnn_runtime_configs(v, _visited)
        return

    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict):
        return

    for _, v in list(d.items()):
        _release_ttnn_runtime_configs(v, _visited)

    hints = ("program_config", "memory_config", "compute_kernel_config", "kernel_config", "sharded_memory_config")
    for name in list(d.keys()):
        lname = name.lower()
        if any(h in lname for h in hints):
            try:
                setattr(obj, name, None)
            except Exception:
                # Best-effort teardown: setattr can fail on read-only or invalid descriptors.
                logger.debug(
                    "Could not clear TTNN runtime config attribute %r on %s",
                    name,
                    type(obj).__name__,
                    exc_info=True,
                )


class _TTTransformerAdapter:
    """Wraps TTNN WanTransformer3DModel to match the PyTorch transformer call interface used in _infer_impl."""

    def __init__(self, tt_model):
        self._tt_model = tt_model

    def clear_cache(self, cache_name):
        self._tt_model.clear_cache(cache_name)

    def cleanup_all(self):
        if hasattr(self._tt_model, "cleanup_all"):
            self._tt_model.cleanup_all()

    def deallocate_weights(self):
        if hasattr(self._tt_model, "deallocate_weights"):
            self._tt_model.deallocate_weights()

    def create_empty_cache(
        self,
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        dtype=None,
        device=None,
        batch_size=1,
    ):
        self._tt_model.create_empty_cache(
            cache_name,
            attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

    def clear_pred_cache(self, cache_name):
        self._tt_model.clear_pred_cache(cache_name)

    def __call__(self, input_dict, update_cache=0, cache_name="pos", action_mode=False, dump_iter=None):
        spatial = input_dict["noisy_latents"]
        prompt = input_dict["text_emb"]
        timesteps = input_dict["timesteps"]
        grid_id = input_dict["grid_id"]
        B = spatial.shape[0]
        device = spatial.device
        ts = timesteps.detach().to(device=device, dtype=torch.float32)
        if ts.dim() == 2:
            timestep_per_frame = ts.clone()
            timestep = timestep_per_frame[:, 0].contiguous()
        elif ts.dim() == 1:
            F_frames = ts.shape[0]
            timestep_per_frame = ts.unsqueeze(0).expand(B, F_frames).contiguous()
            timestep = timestep_per_frame[:, 0].contiguous()
        else:
            t0 = ts.reshape(-1)[0]
            F_spatial = spatial.shape[2]
            timestep_per_frame = t0.view(1, 1).expand(B, F_spatial).contiguous()
            timestep = t0.unsqueeze(0).expand(B).contiguous()

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        return self._tt_model(
            spatial=spatial,
            prompt=prompt,
            timestep=timestep,
            grid_id=grid_id,
            action_mode=action_mode,
            update_cache=update_cache,
            cache_name=cache_name,
            timestep_per_frame=timestep_per_frame,
            dump_iter=dump_iter,
        )


def build_infer_obs(
    cam_high: np.ndarray,
    cam_left_wrist: np.ndarray,
    cam_right_wrist: np.ndarray,
    prompt: str = "",
    state: np.ndarray | None = None,
):
    """Build the single observation dict that the server expects for an infer (get-action) call."""
    out = {
        OBS_CAM_HIGH: _as_rgb_uint8(cam_high),
        OBS_CAM_LEFT_WRIST: _as_rgb_uint8(cam_left_wrist),
        OBS_CAM_RIGHT_WRIST: _as_rgb_uint8(cam_right_wrist),
        "task": prompt,
    }
    if state is not None:
        out[OBS_STATE] = np.asarray(state, dtype=np.float64)
    return out


def build_infer_message(
    cam_high: np.ndarray,
    cam_left_wrist: np.ndarray,
    cam_right_wrist: np.ndarray,
    prompt: str,
    video_guidance_scale: float = 5.0,
    action_guidance_scale: float = 1.0,
    state: np.ndarray | None = None,
):
    """Build the full message dict for model.infer(...) to get one action chunk."""
    obs = build_infer_obs(cam_high, cam_left_wrist, cam_right_wrist, prompt=prompt, state=state)
    return {
        "obs": obs,
        "prompt": prompt,
        "video_guidance_scale": video_guidance_scale,
        "action_guidance_scale": action_guidance_scale,
    }


def build_reset_message(prompt: str):
    """Build the message for model.infer(...) to reset (start of episode)."""
    return {"reset": True, "prompt": prompt}


def _as_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure (H, W, 3) and uint8 for server (it will resize and normalize)."""
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {img.shape}")
    if img.dtype != np.uint8:
        if img.max() <= 1.0 + 1e-6:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def load_message_from_files(
    path_cam_high: str,
    path_cam_left_wrist: str,
    path_cam_right_wrist: str,
    prompt: str = "Lift the cup from the table",
) -> dict:
    """Load three images from paths and return the infer message dict."""
    cam_high = np.array(Image.open(path_cam_high).convert("RGB"))
    cam_left = np.array(Image.open(path_cam_left_wrist).convert("RGB"))
    cam_right = np.array(Image.open(path_cam_right_wrist).convert("RGB"))
    return build_infer_message(cam_high, cam_left, cam_right, prompt)


def _open_lingbot_mesh_device():
    """Open TTNN mesh for inference; shape from ``mesh_utils`` / ``MESH_DEVICE``.

    Env overrides (optional): ``LINGBOT_VA_NUM_COMMAND_QUEUES``, ``LINGBOT_VA_TRACE_REGION_SIZE``,
    ``LINGBOT_VA_L1_SMALL_SIZE``, ``LINGBOT_VA_WORKER_L1_SIZE``. Submesh: ``LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH``.
    """
    from models.experimental.lingbot_va.tests.mesh_utils import ttnn_mesh_shape_for_inference_demo

    kwargs: dict = {"mesh_shape": ttnn_mesh_shape_for_inference_demo()}
    n_cq = os.environ.get("LINGBOT_VA_NUM_COMMAND_QUEUES")
    if n_cq is not None and str(n_cq).strip() != "":
        kwargs["num_command_queues"] = int(n_cq)
    tr = os.environ.get("LINGBOT_VA_TRACE_REGION_SIZE")
    if tr is not None and str(tr).strip() != "":
        kwargs["trace_region_size"] = int(tr)
    l1 = os.environ.get("LINGBOT_VA_L1_SMALL_SIZE")
    if l1 is not None and str(l1).strip() != "":
        kwargs["l1_small_size"] = int(l1)
    wl1 = os.environ.get("LINGBOT_VA_WORKER_L1_SIZE")
    if wl1 is not None and str(wl1).strip() != "":
        kwargs["worker_l1_size"] = int(wl1)
    return ttnn.open_mesh_device(**kwargs)


def _lingbot_dit_parallel_config(mesh_device: ttnn.MeshDevice) -> DiTParallelConfig:
    """Align with PCC ``test_transformer_wan._make_parallel_config`` (sp_axis=0, tp_axis=1)."""
    rows, cols = tuple(mesh_device.shape)
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=1, factor=cols),
        sequence_parallel=ParallelFactor(mesh_axis=0, factor=rows),
        cfg_parallel=None,
    )


def _lingbot_vae_hw_parallel_config(mesh_device: ttnn.MeshDevice) -> VaeHWParallelConfig:
    """Shard VAE over mesh rows (H) and columns (W), same axes as tt_dit Wan VAE."""
    rows, cols = tuple(mesh_device.shape)
    return VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=rows, mesh_axis=0),
        width_parallel=ParallelFactor(factor=cols, mesh_axis=1),
    )


def _close_lingbot_mesh_stack(models: dict) -> None:
    """Close ``mesh_device`` and optional ``mesh_device_parent`` (multi-chip open + submesh)."""
    work = models.pop("mesh_device", None)
    parent = models.pop("mesh_device_parent", None)
    if work is None:
        return
    try:
        ttnn.synchronize_device(work)
    except Exception as e:
        logger.warning("synchronize_device before mesh close: %s", e)
    try:
        ttnn.close_mesh_device(work)
    except Exception as e:
        logger.warning("close_mesh_device(work): %s", e)
    if parent is not None:
        try:
            ttnn.close_mesh_device(parent)
        except Exception as e:
            logger.warning("close_mesh_device(parent): %s", e)


def _load_models_phase1(config, load_text_encoder=True):
    """Load tokenizer, VAE (CPU), mesh, optional TT text encoder. Transformer and TT VAE load in later phases."""
    init_logger()
    device = torch.device("cpu")
    dtype = config.param_dtype
    enable_offload = getattr(config, "enable_offload", True)
    ckpt = config.wan22_pretrained_model_name_or_path

    from models.experimental.lingbot_va.tests.mesh_utils import inference_work_mesh_from_opened

    opened_mesh = _open_lingbot_mesh_device()
    mesh_device, mesh_parent = inference_work_mesh_from_opened(opened_mesh)
    rows, cols = tuple(mesh_device.shape)
    if mesh_parent is not None:
        logger.info(
            "LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH: using (1,1) submesh inside %s-device open.",
            opened_mesh.get_num_devices(),
        )
    elif mesh_device.get_num_devices() > 1:
        logger.info(
            "Full mesh %sx%s: DiT tensor_parallel=%s (axis 1), sequence_parallel=%s (axis 0); "
            "VAE height_parallel=%s, width_parallel=%s. Text encoder TP follows mesh width (see load_text_encoder). "
            "Set LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH=1 to force one die only.",
            rows,
            cols,
            cols,
            rows,
            rows,
            cols,
        )
    tp_factor = cols
    if tp_factor > 1 and LINGBOT_NUM_HEADS % tp_factor != 0:
        raise RuntimeError(
            f"Lingbot WanTransformer NUM_HEADS={LINGBOT_NUM_HEADS} is not divisible by tensor_parallel "
            f"factor {tp_factor} (mesh columns). Use MESH_DEVICE=N150, unset MESH_DEVICE, or "
            "LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH=1."
        )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    dit_parallel_config = _lingbot_dit_parallel_config(mesh_device)

    text_encoder = None
    if load_text_encoder:
        text_encoder = load_text_encoder_tt(
            os.path.join(ckpt, "text_encoder"),
            mesh_device,
            ccl_manager=ccl_manager,
            torch_dtype=dtype,
            max_prompt_length=512,
        )

    vae = load_vae(
        os.path.join(ckpt, "vae"),
        torch_dtype=dtype,
        torch_device="cpu" if enable_offload else device,
    )
    streaming_vae = WanVAEStreamingWrapper(vae)
    vae_half = vae if config.env_type == "robotwin_tshape" else None
    streaming_vae_half = WanVAEStreamingWrapper(vae) if config.env_type == "robotwin_tshape" else None

    tokenizer = load_tokenizer(os.path.join(ckpt, "tokenizer"))

    scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
    action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    action_scheduler.set_timesteps(1000, training=True)

    return {
        "vae": vae,
        "vae_half": vae_half,
        "streaming_vae": streaming_vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "streaming_vae_half": streaming_vae_half,
        "scheduler": scheduler,
        "action_scheduler": action_scheduler,
        "device": device,
        "dtype": dtype,
        "cache_name": "pos",
        "config": config,
        "env_type": config.env_type,
        "transformer_is_tt": True,
        "mesh_device": mesh_device,
        "mesh_device_parent": mesh_parent,
        "parallel_config": dit_parallel_config,
        "ccl_manager": ccl_manager,
    }


def _free_tt_model(models: dict, key: str) -> None:
    """Remove a TT model from models with best-effort cleanup before deletion."""
    obj = models.pop(key, None)
    if obj is not None:
        _release_ttnn_runtime_configs(obj)
        try:
            if hasattr(obj, "cleanup_all"):
                obj.cleanup_all()
        except Exception as e:
            logger.warning("cleanup_all failed for %s: %s", key, e)
        try:
            if hasattr(obj, "deallocate_weights"):
                obj.deallocate_weights()
        except Exception as e:
            logger.warning("deallocate_weights failed for %s: %s", key, e)
        del obj
    gc.collect()


def _load_text_encoder_into_models(models: dict, config) -> None:
    """Load the TTNN UMT5 text encoder into ``models``."""
    ckpt = config.wan22_pretrained_model_name_or_path
    dtype = models["dtype"]
    logger.info(
        "Loading TT text encoder onto mesh %s (first graph build can take several minutes; this is not a hang).",
        models["mesh_device"].shape,
    )
    models["text_encoder"] = load_text_encoder_tt(
        os.path.join(ckpt, "text_encoder"),
        models["mesh_device"],
        ccl_manager=models["ccl_manager"],
        torch_dtype=dtype,
        max_prompt_length=512,
    )
    logger.info("Loaded TT text encoder (cache miss).")


def _prepare_state_for_vae_encode(state: dict, config) -> None:
    """Set state keys required by _encode_obs (height, width, latent_height, latent_width)."""
    state["height"] = config.height
    state["width"] = config.width
    if config.env_type == "robotwin_tshape":
        state["latent_height"] = ((config.height // 16) * 3) // 2
        state["latent_width"] = ((config.width // 16) * 3) // 2
    else:
        state["latent_height"] = config.height // 16
        state["latent_width"] = config.width // 16


def _load_tt_vae_into_models(models: dict, config) -> None:
    """Load TTNN VAE (streaming encoder + quant_conv) into models. Only one TT sub-network on device."""
    vae_parallel_config = _lingbot_vae_hw_parallel_config(models["mesh_device"])
    models["streaming_vae"] = TTWanVAEStreamingWrapper(
        models["vae"],
        models["mesh_device"],
        models["ccl_manager"],
        vae_parallel_config,
    )
    # One TT wrapper for high + wrist unless LINGBOT_VA_TT_USE_DUAL_ENCODER_WRAPPERS=1.
    use_dual_tt_wrappers = os.environ.get("LINGBOT_VA_TT_USE_DUAL_ENCODER_WRAPPERS", "0") == "1"
    if use_dual_tt_wrappers and config.env_type == "robotwin_tshape" and models.get("vae_half") is not None:
        models["streaming_vae_half"] = TTWanVAEStreamingWrapper(
            models["vae_half"],
            models["mesh_device"],
            models["ccl_manager"],
            vae_parallel_config,
        )
    else:
        models["streaming_vae_half"] = models["streaming_vae"]
        if config.env_type == "robotwin_tshape":
            logger.info(
                "Using single TT VAE wrapper for high + left_right "
                "(set LINGBOT_VA_TT_USE_DUAL_ENCODER_WRAPPERS=1 to force dual wrappers)."
            )
    logger.info("Loaded TT VAE encoder (streaming_vae) on device.")


def _free_tt_vae_from_models(models: dict, config) -> None:
    """Replace TT VAE in models with PyTorch wrappers and run gc to free device memory."""
    old_streaming_vae = models.get("streaming_vae")
    old_streaming_vae_half = models.get("streaming_vae_half")
    if old_streaming_vae is not None:
        _release_ttnn_runtime_configs(old_streaming_vae)
        try:
            if hasattr(old_streaming_vae, "cleanup_all"):
                old_streaming_vae.cleanup_all()
        except Exception as e:
            logger.warning("cleanup_all failed for streaming_vae: %s", e)
        try:
            if hasattr(old_streaming_vae, "deallocate_weights"):
                old_streaming_vae.deallocate_weights()
        except Exception as e:
            logger.warning("deallocate_weights failed for streaming_vae: %s", e)
    if old_streaming_vae_half is not None and old_streaming_vae_half is not old_streaming_vae:
        _release_ttnn_runtime_configs(old_streaming_vae_half)
        try:
            if hasattr(old_streaming_vae_half, "cleanup_all"):
                old_streaming_vae_half.cleanup_all()
        except Exception as e:
            logger.warning("cleanup_all failed for streaming_vae_half: %s", e)
        try:
            if hasattr(old_streaming_vae_half, "deallocate_weights"):
                old_streaming_vae_half.deallocate_weights()
        except Exception as e:
            logger.warning("deallocate_weights failed for streaming_vae_half: %s", e)

    models["streaming_vae"] = WanVAEStreamingWrapper(models["vae"])
    if config.env_type == "robotwin_tshape" and models.get("vae_half") is not None:
        models["streaming_vae_half"] = WanVAEStreamingWrapper(models["vae_half"])
    else:
        models["streaming_vae_half"] = None
    gc.collect()
    logger.info("Freed TT VAE from device; using PyTorch VAE wrapper for rest of run.")


def _load_transformer_into_models(models: dict, config) -> None:
    """Load TTNN WanTransformer3DModel and set models['transformer']. Call after freeing text_encoder."""
    ckpt = config.wan22_pretrained_model_name_or_path
    dtype = models["dtype"]
    transformer = load_transformer_tt(
        os.path.join(ckpt, "transformer"),
        models["mesh_device"],
        models["parallel_config"],
        ccl_manager=models["ccl_manager"],
        is_fsdp=False,
        torch_dtype=dtype,
    )
    models["transformer"] = _TTTransformerAdapter(transformer)


def _get_t5_prompt_embeds(models, prompt, num_videos_per_prompt=1, max_sequence_length=512):
    device = models["device"]
    dtype = models["dtype"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    # TTNN text encoder (always used; no PyTorch fallback).
    mesh_device = text_encoder.mesh_device
    tt_input = ttnn.from_torch(text_input_ids, dtype=ttnn.uint32, device=mesh_device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=mesh_device)
    tt_outputs = text_encoder(tt_input, attention_mask=tt_mask)
    last_hidden = tt_outputs[-1]
    prompt_embeds = ttnn.to_torch(last_hidden).float()
    try:
        ttnn.synchronize_device(mesh_device)
    except Exception as e:
        logger.warning("synchronize_device failed after text encoder pass: %s", e)
    try:
        ttnn.deallocate(tt_input)
    except Exception as e:
        logger.warning("Failed to deallocate tt_input: %s", e)
    try:
        ttnn.deallocate(tt_mask)
    except Exception as e:
        logger.warning("Failed to deallocate tt_mask: %s", e)
    try:
        ttnn.deallocate(last_hidden)
    except Exception as e:
        logger.warning("Failed to deallocate text encoder output: %s", e)
    while prompt_embeds.dim() > 3:
        prompt_embeds = prompt_embeds.squeeze(0)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
        dim=0,
    )
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    return prompt_embeds.to(device)


def _encode_prompt(models, _state, prompt, do_classifier_free_guidance=True, max_sequence_length=512):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if "text_encoder" not in models:
        raise RuntimeError("Text encoder was freed before prompt encoding.")
    prompt_embeds = _get_t5_prompt_embeds(
        models, prompt=prompt, num_videos_per_prompt=1, max_sequence_length=max_sequence_length
    )

    negative_prompt_embeds = None
    if do_classifier_free_guidance:
        negative_prompt = batch_size * [""]
        negative_prompt_embeds = _get_t5_prompt_embeds(
            models, prompt=negative_prompt, num_videos_per_prompt=1, max_sequence_length=max_sequence_length
        )

    return prompt_embeds, negative_prompt_embeds


def _normalize_latents(latents, latents_mean, latents_std):
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    return ((latents.float() - latents_mean) * latents_std).to(latents.dtype)


def _preprocess_action(models, state, action):
    config = models["config"]
    actions_q01 = state["actions_q01"]
    actions_q99 = state["actions_q99"]
    action_norm_method = state["action_norm_method"]

    action_model_input = torch.from_numpy(action)
    action_model_input = F.pad(action_model_input, [0, 0, 0, 0, 0, 1], mode="constant", value=0)
    action_model_input = action_model_input[config.inverse_used_action_channel_ids]
    if action_norm_method == "quantiles":
        action_model_input = (action_model_input - actions_q01) / (actions_q99 - actions_q01 + 1e-6) * 2.0 - 1.0
    else:
        raise NotImplementedError
    return action_model_input.unsqueeze(0).unsqueeze(-1)


def _postprocess_action(models, state, action):
    config = models["config"]
    actions_q01 = state["actions_q01"]
    actions_q99 = state["actions_q99"]
    action_norm_method = state["action_norm_method"]

    action = action.cpu()
    action = action[0, ..., 0]
    if action_norm_method == "quantiles":
        action = (action + 1) / 2 * (actions_q99 - actions_q01 + 1e-6) + actions_q01
    else:
        raise NotImplementedError
    action = action.squeeze(0).detach().cpu().numpy()
    return action[config.used_action_channel_ids]


def _repeat_input_for_cfg(models, state, input_dict):
    use_cfg = state["use_cfg"]
    prompt_embeds = state["prompt_embeds"]
    negative_prompt_embeds = state["negative_prompt_embeds"]
    dtype = models["dtype"]

    if use_cfg:
        input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1)
        input_dict["text_emb"] = torch.cat(
            [prompt_embeds.to(dtype).clone(), negative_prompt_embeds.to(dtype).clone()], dim=0
        )
        input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
        input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
    else:
        input_dict["grid_id"] = input_dict["grid_id"][None]
        input_dict["timesteps"] = input_dict["timesteps"][None]
    return input_dict


def _prepare_latent_input(
    models,
    state,
    latent_model_input,
    action_model_input,
    latent_t=0,
    action_t=0,
    latent_cond=None,
    action_cond=None,
    frame_st_id=0,
    patch_size=(1, 2, 2),
):
    device = models["device"]
    dtype = models["dtype"]
    prompt_embeds = state["prompt_embeds"]
    action_mask = state["action_mask"]

    input_dict = {}
    if latent_model_input is not None:
        input_dict["latent_res_lst"] = {
            "noisy_latents": latent_model_input,
            "timesteps": torch.ones([latent_model_input.shape[2]], dtype=torch.float32, device=device) * latent_t,
            "grid_id": get_mesh_id(
                latent_model_input.shape[-3] // patch_size[0],
                latent_model_input.shape[-2] // patch_size[1],
                latent_model_input.shape[-1] // patch_size[2],
                0,
                1,
                frame_st_id,
            ).to(device),
            "text_emb": prompt_embeds.to(dtype).clone(),
        }
        if latent_cond is not None:
            input_dict["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
            input_dict["latent_res_lst"]["timesteps"][0:1] *= 0

    if action_model_input is not None:
        input_dict["action_res_lst"] = {
            "noisy_latents": action_model_input,
            "timesteps": torch.ones([action_model_input.shape[2]], dtype=torch.float32, device=device) * action_t,
            "grid_id": get_mesh_id(
                action_model_input.shape[-3],
                action_model_input.shape[-2],
                action_model_input.shape[-1],
                1,
                1,
                frame_st_id,
                action=True,
            ).to(device),
            "text_emb": prompt_embeds.to(dtype).clone(),
        }
        if action_cond is not None:
            input_dict["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
            input_dict["action_res_lst"]["timesteps"][0:1] *= 0
        input_dict["action_res_lst"]["noisy_latents"][:, ~action_mask] *= 0
    return input_dict


def _encode_obs(models, state, obs):
    config = models["config"]
    device = models["device"]
    dtype = models["dtype"]

    env_type = models["env_type"]
    height = state["height"]
    width = state["width"]
    streaming_vae = models["streaming_vae"]
    streaming_vae_half = models["streaming_vae_half"]
    vae = models["vae"]

    images = obs["obs"]
    if not isinstance(images, list):
        images = [images]
    if len(images) < 1:
        return None

    videos = []
    for k_i, k in enumerate(config.obs_cam_keys):
        if env_type == "robotwin_tshape":
            height_i, width_i = (height, width) if k_i == 0 else (height // 2, width // 2)
        else:
            height_i, width_i = height, width
        history_video_k = torch.from_numpy(np.stack([each[k] for each in images])).float().permute(3, 0, 1, 2)
        history_video_k = F.interpolate(
            history_video_k, size=(height_i, width_i), mode="bilinear", align_corners=False
        ).unsqueeze(0)
        videos.append(history_video_k)

    if env_type == "robotwin_tshape":
        videos_high = videos[0] / 255.0 * 2.0 - 1.0
        videos_left_and_right = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
        vae_device = next(streaming_vae.vae.parameters()).device
        enc_out_high = streaming_vae.encode_chunk(videos_high.to(vae_device).to(dtype))
        encode_lr = streaming_vae_half if streaming_vae_half is not None else streaming_vae
        enc_out_left_and_right = encode_lr.encode_chunk(videos_left_and_right.to(vae_device).to(dtype))
        enc_out = torch.cat(
            [
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high,
            ],
            dim=-2,
        )
    else:
        videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
        vae_device = next(streaming_vae.vae.parameters()).device
        enc_out = streaming_vae.encode_chunk(videos.to(vae_device).to(dtype))

    mu, logvar = torch.chunk(enc_out, 2, dim=1)
    latents_mean = torch.tensor(vae.config.latents_mean).to(mu.device)
    latents_std = torch.tensor(vae.config.latents_std).to(mu.device)
    mu_norm = _normalize_latents(mu, latents_mean, 1.0 / latents_std)
    video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
    video_latent = video_latent.to(device)

    return video_latent


def _reset_state(models, state, prompt):
    config = models["config"]
    transformer = models["transformer"]
    streaming_vae = models["streaming_vae"]
    streaming_vae_half = models["streaming_vae_half"]
    device = models["device"]
    dtype = models["dtype"]
    cache_name = models["cache_name"]

    logger.info("Reset.")
    state["use_cfg"] = False
    state["frame_st_id"] = 0
    # Keep precomputed init latent when available (run_generate phase-2 preload).
    if "init_latent" not in state:
        state["init_latent"] = None

    transformer.clear_cache(cache_name)
    streaming_vae.clear_cache()
    if streaming_vae_half is not None:
        streaming_vae_half.clear_cache()

    state["action_per_frame"] = config.action_per_frame
    state["height"], state["width"] = config.height, config.width
    if config.env_type == "robotwin_tshape":
        state["latent_height"] = ((state["height"] // 16) * 3) // 2
        state["latent_width"] = state["width"] // 16
    else:
        state["latent_height"] = state["height"] // 16
        state["latent_width"] = state["width"] // 16 * len(config.obs_cam_keys)

    patch_size = config.patch_size
    latent_token_per_chunk = (config.frame_chunk_size * state["latent_height"] * state["latent_width"]) // (
        patch_size[0] * patch_size[1] * patch_size[2]
    )
    action_token_per_chunk = config.frame_chunk_size * state["action_per_frame"]
    transformer.create_empty_cache(
        cache_name,
        config.attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        dtype=dtype,
        device=device,
        batch_size=2 if state["use_cfg"] else 1,
    )

    state["action_mask"] = torch.zeros([config.action_dim], dtype=torch.bool, device=device)
    state["action_mask"][config.used_action_channel_ids] = True
    state["actions_q01"] = torch.tensor(config.norm_stat["q01"], dtype=torch.float32).reshape(-1, 1, 1)
    state["actions_q99"] = torch.tensor(config.norm_stat["q99"], dtype=torch.float32).reshape(-1, 1, 1)
    state["action_norm_method"] = config.action_norm_method

    if prompt is None:
        state["prompt_embeds"] = None
        state["negative_prompt_embeds"] = None
        state["_prompt_embeds_prompt"] = None
    else:
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        if state.get("_prompt_embeds_prompt") != prompt_list or "prompt_embeds" not in state:
            prompt_embeds, negative_prompt_embeds = _encode_prompt(
                models,
                state,
                prompt,
                do_classifier_free_guidance=(config.guidance_scale > 1),
                max_sequence_length=512,
            )
            state["prompt_embeds"] = prompt_embeds
            state["negative_prompt_embeds"] = negative_prompt_embeds
            state["_prompt_embeds_prompt"] = prompt_list


def _infer_impl(models, state, obs, frame_st_id=0):
    config = models["config"]
    device = models["device"]
    dtype = models["dtype"]
    cache_name = models["cache_name"]
    transformer = models["transformer"]
    scheduler = models["scheduler"]
    action_scheduler = models["action_scheduler"]

    frame_chunk_size = config.frame_chunk_size
    latent_height = state["latent_height"]
    latent_width = state["latent_width"]
    action_per_frame = state["action_per_frame"]
    action_mask = state["action_mask"]

    if frame_st_id == 0 and state.get("init_latent") is None:
        init_latent = _encode_obs(models, state, obs)
        state["init_latent"] = init_latent

    _set_seed()
    latents = torch.randn(1, 48, frame_chunk_size, latent_height, latent_width, device=device, dtype=dtype)
    actions = torch.randn(
        1,
        config.action_dim,
        frame_chunk_size,
        action_per_frame,
        1,
        device=device,
        dtype=dtype,
    )

    video_inference_step = config.num_inference_steps
    action_inference_step = config.action_num_inference_steps
    video_step = config.video_exec_step

    scheduler.set_timesteps(video_inference_step)
    action_scheduler.set_timesteps(action_inference_step)
    timesteps = scheduler.timesteps
    action_timesteps = action_scheduler.timesteps
    timesteps = F.pad(timesteps, (0, 1), mode="constant", value=0)
    if video_step != -1:
        timesteps = timesteps[:video_step]
    action_timesteps = F.pad(action_timesteps, (0, 1), mode="constant", value=0)

    use_cfg = state["use_cfg"]
    init_latent = state.get("init_latent")
    patch_size = config.patch_size

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            last_step = i == len(timesteps) - 1
            latent_cond = init_latent[:, :, 0:1].to(dtype) if frame_st_id == 0 else None
            input_dict = _prepare_latent_input(
                models,
                state,
                latents,
                None,
                latent_t=t,
                action_t=t,
                latent_cond=latent_cond,
                action_cond=None,
                frame_st_id=frame_st_id,
                patch_size=patch_size,
            )
            video_noise_pred = transformer(
                _repeat_input_for_cfg(models, state, input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=cache_name,
                action_mode=False,
                dump_iter=None,
            )
            if models.get("transformer_is_tt", False) and video_noise_pred.dtype != torch.float32:
                video_noise_pred = video_noise_pred.float()
            if not last_step or video_step != -1:
                if not models.get("transformer_is_tt", False):
                    video_noise_pred = data_seq_to_patch(
                        config.patch_size,
                        video_noise_pred,
                        frame_chunk_size,
                        latent_height,
                        latent_width,
                        batch_size=2 if use_cfg else 1,
                    )
                if config.guidance_scale > 1:
                    video_noise_pred = video_noise_pred[1:] + config.guidance_scale * (
                        video_noise_pred[:1] - video_noise_pred[1:]
                    )
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = scheduler.step(video_noise_pred, t, latents, return_dict=False)
            latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

        for i, t in enumerate(tqdm(action_timesteps)):
            last_step = i == len(action_timesteps) - 1
            action_cond = (
                torch.zeros(
                    [1, config.action_dim, 1, action_per_frame, 1],
                    device=device,
                    dtype=dtype,
                )
                if frame_st_id == 0
                else None
            )
            input_dict = _prepare_latent_input(
                models,
                state,
                None,
                actions,
                latent_t=t,
                action_t=t,
                latent_cond=None,
                action_cond=action_cond,
                frame_st_id=frame_st_id,
                patch_size=patch_size,
            )
            action_noise_pred = transformer(
                _repeat_input_for_cfg(models, state, input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=cache_name,
                action_mode=True,
                dump_iter=None,
            )
            if models.get("transformer_is_tt", False):
                if action_noise_pred.dtype != torch.float32:
                    action_noise_pred = action_noise_pred.float()
                action_noise_pred = action_noise_pred.contiguous()
            if not last_step:
                action_noise_pred = rearrange(action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
                if config.action_guidance_scale > 1:
                    action_noise_pred = action_noise_pred[1:] + config.action_guidance_scale * (
                        action_noise_pred[:1] - action_noise_pred[1:]
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
            actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

    actions[:, ~action_mask] *= 0
    actions_out = _postprocess_action(models, state, actions)
    return actions_out, latents


def _compute_kv_cache(models, state, obs):
    transformer = models["transformer"]
    cache_name = models["cache_name"]

    transformer.clear_pred_cache(cache_name)
    latent_model_input = _encode_obs(models, state, obs)
    if state["frame_st_id"] == 0:
        latent_model_input = (
            torch.cat([state["init_latent"], latent_model_input], dim=2)
            if latent_model_input is not None
            else state["init_latent"]
        )
    action_model_input = _preprocess_action(models, state, obs["state"])
    action_model_input = action_model_input.to(latent_model_input)
    logger.info(f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}")
    input_dict = _prepare_latent_input(
        models,
        state,
        latent_model_input,
        action_model_input,
        frame_st_id=state["frame_st_id"],
    )
    with torch.no_grad():
        transformer(
            _repeat_input_for_cfg(models, state, input_dict["latent_res_lst"]),
            update_cache=2,
            cache_name=cache_name,
            action_mode=False,
        )
        transformer(
            _repeat_input_for_cfg(models, state, input_dict["action_res_lst"]),
            update_cache=2,
            cache_name=cache_name,
            action_mode=True,
        )
    state["frame_st_id"] += latent_model_input.shape[2]


def _infer_entry(models, state, obs):
    """Dispatch: reset, compute_kv_cache, or infer one chunk. Returns result dict."""
    reset = obs.get("reset", False)
    prompt = obs.get("prompt", None)
    compute_kv_cache = obs.get("compute_kv_cache", False)

    if reset:
        logger.info("Reset server")
        _reset_state(models, state, prompt=prompt)
        return {}
    if compute_kv_cache:
        logger.info("Compute KV cache")
        _compute_kv_cache(models, state, obs)
        return {}
    logger.info("Infer one chunk")
    action, _ = _infer_impl(models, state, obs, frame_st_id=state["frame_st_id"])
    return {"action": action}


def _load_tt_vae_decoder_into_models(models: dict, config) -> None:
    """Load TTNN VAE decoder into models. Use before decode when running generate with TT path."""
    vae_parallel_config = _lingbot_vae_hw_parallel_config(models["mesh_device"])
    models["vae_decoder_tt"] = TTWanVAEDecoderWrapper(
        models["vae"],
        models["mesh_device"],
        ccl_manager=models["ccl_manager"],
        parallel_config=vae_parallel_config,
    )
    logger.info("Loaded TT VAE decoder on device.")


def _free_tt_vae_decoder_from_models(models: dict) -> None:
    """Remove TT VAE decoder from models and run gc to free device memory."""
    vae_decoder_tt = models.pop("vae_decoder_tt", None)
    if vae_decoder_tt is not None:
        _release_ttnn_runtime_configs(vae_decoder_tt)
        try:
            if hasattr(vae_decoder_tt, "cleanup_all"):
                vae_decoder_tt.cleanup_all()
        except Exception as e:
            logger.warning("cleanup_all failed for vae_decoder_tt: %s", e)
        try:
            if hasattr(vae_decoder_tt, "deallocate_weights"):
                vae_decoder_tt.deallocate_weights()
        except Exception as e:
            logger.warning("deallocate_weights failed for vae_decoder_tt: %s", e)
        del vae_decoder_tt
    gc.collect()
    logger.info("Freed TT VAE decoder from device.")


def _decode_one_video(models, latents, output_type="np"):
    vae = models["vae"]

    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean

    if models.get("vae_decoder_tt") is not None:
        video = models["vae_decoder_tt"].decode(latents)
    else:
        video = vae.decode(latents, return_dict=False)[0]

    video_processor = VideoProcessor(vae_scale_factor=1)
    video = video_processor.postprocess_video(video, output_type=output_type)
    return video


def _load_init_obs(config, input_img_path):
    obs_cam_keys = config.obs_cam_keys
    imf_dict = {v: np.array(Image.open(os.path.join(input_img_path, f"{v}.png")).convert("RGB")) for v in obs_cam_keys}
    return {"obs": [imf_dict]}


def run_inference(
    message: dict,
    checkpoint_path: str | Path,
    save_dir: str | Path | None = None,
    *,
    num_inference_steps: int | None = None,
    action_num_inference_steps: int | None = None,
    frame_chunk_size: int | None = None,
) -> dict:
    """
    Run Lingbot-VA inference on the input dict (same behavior as VA_Server.infer).

    Uses config and model loading from wan_va; no VA_Server class. Resets with
    message['prompt'], then runs infer one chunk and returns {'action': ...}.

    Optional keyword overrides set ``config`` fields; when omitted, env
    ``LINGBOT_VA_*`` inference overrides may apply (see ``apply_robotwin_inference_overrides``).
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_path}")

    _set_seed()
    os.chdir(_REPO_ROOT)
    config = deepcopy(VA_CONFIGS["robotwin"])
    config.wan22_pretrained_model_name_or_path = str(checkpoint_path)
    config.local_rank = 0
    config.rank = 0
    config.world_size = 1
    apply_robotwin_inference_overrides(
        config,
        num_inference_steps=num_inference_steps,
        action_num_inference_steps=action_num_inference_steps,
        frame_chunk_size=frame_chunk_size,
    )
    if save_dir is None:
        save_dir = _SCRIPT_DIR
    config.save_root = str(save_dir)

    # Phase 1: load shared assets (VAE once, tokenizer, mesh); load TT text encoder.
    models = _load_models_phase1(config, load_text_encoder=False)
    state = {}
    prompt = message.get("prompt", "")
    prompt_list = [prompt] if isinstance(prompt, str) else prompt
    _load_text_encoder_into_models(models, config)
    prompt_embeds, neg_embeds = _encode_prompt(
        models,
        state,
        prompt,
        do_classifier_free_guidance=(config.guidance_scale > 1),
        max_sequence_length=512,
    )
    state["prompt_embeds"] = prompt_embeds
    state["negative_prompt_embeds"] = neg_embeds
    state["_prompt_embeds_prompt"] = prompt_list
    _free_tt_model(models, "text_encoder")

    # Phase 2: load only TT VAE encoder; run _encode_obs, then free.
    _prepare_state_for_vae_encode(state, config)
    _load_tt_vae_into_models(models, config)
    state["init_latent"] = _encode_obs(models, state, message)
    _free_tt_vae_from_models(models, config)

    # Phase 3: load TT transformer and run inference.
    _load_transformer_into_models(models, config)

    reset_message = build_reset_message(prompt)
    _infer_entry(models, state, reset_message)
    result = _infer_entry(models, state, message)
    return result


def run_generate(
    checkpoint_path: str | Path,
    images_dir: str | Path,
    prompt: str,
    save_dir: str | Path,
    num_chunks: int = 10,
) -> str:
    """
    Run multi-chunk video generation (same behavior as VA_Server.generate).
    Loads init obs from images_dir, runs num_chunks of inference, decodes to video, saves demo.mp4.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    images_dir = Path(images_dir).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_path}")
    for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        if not (images_dir / f"{key}.png").exists():
            raise FileNotFoundError(f"Missing {images_dir / f'{key}.png'} for generate()")

    _set_seed()
    os.chdir(_REPO_ROOT)
    config = deepcopy(VA_CONFIGS["robotwin"])
    config.wan22_pretrained_model_name_or_path = str(checkpoint_path)
    config.local_rank = 0
    config.rank = 0
    config.world_size = 1
    config.save_root = str(save_dir)
    config.input_img_path = str(images_dir)
    config.prompt = prompt
    config.num_chunks_to_infer = num_chunks

    # Phase 1: load shared assets (VAE once, tokenizer, mesh); load TT text encoder.
    models = _load_models_phase1(config, load_text_encoder=False)
    state = {}
    prompt_list = [prompt] if isinstance(prompt, str) else prompt
    _load_text_encoder_into_models(models, config)
    prompt_embeds, neg_embeds = _encode_prompt(
        models,
        state,
        prompt,
        do_classifier_free_guidance=(config.guidance_scale > 1),
        max_sequence_length=512,
    )
    state["prompt_embeds"] = prompt_embeds
    state["negative_prompt_embeds"] = neg_embeds
    state["_prompt_embeds_prompt"] = prompt_list
    _free_tt_model(models, "text_encoder")

    # Phase 2: load only TT VAE encoder; run _encode_obs, then free.
    _prepare_state_for_vae_encode(state, config)
    init_obs = _load_init_obs(config, config.input_img_path)
    _load_tt_vae_into_models(models, config)
    state["init_latent"] = _encode_obs(models, state, init_obs)
    _free_tt_vae_from_models(models, config)

    # Phase 3: load TT transformer and run generation.
    _load_transformer_into_models(models, config)

    _reset_state(models, state, prompt)

    pred_latent_lst = []
    pred_action_lst = []
    logger.info("Generating %s chunks", config.num_chunks_to_infer)
    for chunk_id in range(config.num_chunks_to_infer):
        actions, latents = _infer_impl(models, state, init_obs, frame_st_id=(chunk_id * config.frame_chunk_size))
        actions = torch.from_numpy(actions)
        pred_latent_lst.append(latents)
        pred_action_lst.append(actions)

    pred_latent = torch.cat(pred_latent_lst, dim=2)

    # Teardown transformer + TT VAE encoders, then reopen mesh so the decoder has a clean device.
    transformer = models.get("transformer")
    if transformer is not None:
        _release_ttnn_runtime_configs(transformer)
        try:
            transformer.clear_cache(models["cache_name"])
        except Exception as e:
            logger.warning("transformer.clear_cache failed: %s", e)
        try:
            if hasattr(transformer, "cleanup_all"):
                transformer.cleanup_all()
        except Exception as e:
            logger.warning("transformer.cleanup_all failed: %s", e)
        try:
            if hasattr(transformer, "deallocate_weights"):
                transformer.deallocate_weights()
        except Exception as e:
            logger.warning("transformer.deallocate_weights failed: %s", e)
        models.pop("transformer", None)
        del transformer
    if models.get("streaming_vae_half"):
        _release_ttnn_runtime_configs(models["streaming_vae_half"])
        try:
            if hasattr(models["streaming_vae_half"], "cleanup_all"):
                models["streaming_vae_half"].cleanup_all()
            if hasattr(models["streaming_vae_half"], "deallocate_weights"):
                models["streaming_vae_half"].deallocate_weights()
        except Exception as e:
            logger.warning("streaming_vae_half cleanup failed: %s", e)
        del models["streaming_vae_half"]
    if models.get("streaming_vae"):
        _release_ttnn_runtime_configs(models["streaming_vae"])
        try:
            if hasattr(models["streaming_vae"], "cleanup_all"):
                models["streaming_vae"].cleanup_all()
            if hasattr(models["streaming_vae"], "deallocate_weights"):
                models["streaming_vae"].deallocate_weights()
        except Exception as e:
            logger.warning("streaming_vae cleanup failed: %s", e)
        del models["streaming_vae"]
    models.pop("text_encoder", None)
    _release_ttnn_runtime_configs(models)
    gc.collect()
    gc.collect()
    _close_lingbot_mesh_stack(models)
    from models.experimental.lingbot_va.tests.mesh_utils import inference_work_mesh_from_opened

    opened_mesh = _open_lingbot_mesh_device()
    mesh_device, mesh_parent = inference_work_mesh_from_opened(opened_mesh)
    models["mesh_device"] = mesh_device
    models["mesh_device_parent"] = mesh_parent
    models["ccl_manager"] = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    # TT decoder on fresh mesh; PyTorch fallback on OOM. Disable: LINGBOT_VA_USE_TT_DECODER=0.
    use_tt_decoder = os.environ.get("LINGBOT_VA_USE_TT_DECODER", "1").strip().lower() in ("1", "true", "yes")
    decoded_video = None
    if use_tt_decoder:
        try:
            _load_tt_vae_decoder_into_models(models, config)
            if getattr(config, "enable_offload", True):
                models["vae"] = models["vae"].to(models["device"]).to(models["dtype"])
            decoded_video = _decode_one_video(models, pred_latent, "np")[0]
        except RuntimeError as e:
            if "Out of Memory" in str(e) or "OOM" in str(e).upper():
                logger.warning("TT VAE decoder OOM, falling back to PyTorch decode: %s", e)
                _free_tt_vae_decoder_from_models(models)
                decoded_video = None
            else:
                _free_tt_vae_decoder_from_models(models)
                raise
        finally:
            if models.get("vae_decoder_tt") is not None:
                _free_tt_vae_decoder_from_models(models)
    if decoded_video is None:
        if getattr(config, "enable_offload", True):
            models["vae"] = models["vae"].to(models["device"]).to(models["dtype"])
        decoded_video = _decode_one_video(models, pred_latent, "np")[0]
    export_to_video(decoded_video, os.path.join(config.save_root, "demo.mp4"), fps=10)
    _close_lingbot_mesh_stack(models)
    return str(Path(save_dir) / "demo.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build input dict and run Lingbot-VA inference.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("LINGBOT_VA_CHECKPOINT", ""),
        help="Path to checkpoint dir (vae, tokenizer, text_encoder, transformer). Default: env LINGBOT_VA_CHECKPOINT.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="",
        help="Dir with observation.images.cam_high.png, cam_left_wrist.png, cam_right_wrist.png. Default: example/robotwin/.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Lift the cup from the table",
        help="Task instruction string.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save the action array (e.g. action.npy).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Output directory for generated demo.mp4. Default: tests/demo/.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run multi-chunk video generation instead of infer(): decode to RGB, save demo.mp4. Do not run infer().",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=2,
        help="Number of chunks for generate() (only used with --generate). Default: 2.",
    )
    args = parser.parse_args()
    init_logger()

    save_dir = args.save_dir or str(_SCRIPT_DIR)
    images_dir = Path(args.images_dir) if args.images_dir else _REPO_ROOT / "example" / "robotwin"

    if args.generate:
        if not args.checkpoint:
            logger.error("--generate requires --checkpoint (or LINGBOT_VA_CHECKPOINT).")
            sys.exit(1)
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            if not (images_dir / f"{key}.png").exists():
                logger.error("Missing %s for generate(). Use --images-dir.", images_dir / f"{key}.png")
                sys.exit(1)
        logger.info("=" * 60)
        logger.info("Running generate() (no infer): multi-chunk → decode → demo.mp4")
        logger.info("=" * 60)
        logger.info("Checkpoint: %s", args.checkpoint)
        logger.info("Images dir: %s", images_dir)
        logger.info("Prompt: %r", args.prompt)
        logger.info("Num chunks: %s", args.num_chunks)
        logger.info("Save dir: %s", save_dir)
        logger.info("=" * 60)
        out_path = run_generate(
            args.checkpoint,
            images_dir,
            args.prompt,
            save_dir,
            num_chunks=args.num_chunks,
        )
        logger.info("Generated video saved to: %s", out_path)
        return

    # Infer mode: build message, run reset + infer one chunk
    cam_high_path = images_dir / "observation.images.cam_high.png"
    cam_left_path = images_dir / "observation.images.cam_left_wrist.png"
    cam_right_path = images_dir / "observation.images.cam_right_wrist.png"
    for p in (cam_high_path, cam_left_path, cam_right_path):
        if not p.exists():
            logger.error("Missing image: %s", p)
            logger.error("  Use --images-dir to specify another dir.")
            sys.exit(1)

    message = load_message_from_files(
        str(cam_high_path),
        str(cam_left_path),
        str(cam_right_path),
        prompt=args.prompt,
    )

    logger.info("=" * 60)
    logger.info("Input dict (message for model.infer)")
    logger.info("=" * 60)
    logger.info("Top-level keys: %s", list(message.keys()))
    logger.info("Observation keys (message['obs']): %s", list(message["obs"].keys()))
    logger.info("Observation array shapes:")
    for k in (OBS_CAM_HIGH, OBS_CAM_LEFT_WRIST, OBS_CAM_RIGHT_WRIST):
        arr = message["obs"][k]
        logger.info("  %s: %s %s", k, arr.shape, arr.dtype)
    logger.info("Prompt: %r", message["prompt"])
    logger.info("=" * 60)

    if not args.checkpoint:
        logger.warning("No --checkpoint (or LINGBOT_VA_CHECKPOINT) set. Skipping inference.")
        logger.warning("Set checkpoint path to run inference on the above dict.")
        return

    logger.info("Running inference (reset + infer one chunk)...")
    logger.info("Output directory: %s", save_dir)
    result = run_inference(message, args.checkpoint, save_dir=save_dir)
    logger.info("=" * 60)
    logger.info("Inference result")
    logger.info("=" * 60)
    if "action" in result:
        action = result["action"]
        logger.info("action shape: %s dtype: %s", action.shape, action.dtype)
        if args.output:
            np.save(args.output, action)
            logger.info("action saved to: %s", args.output)
    else:
        logger.info("Keys: %s", list(result.keys()))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
