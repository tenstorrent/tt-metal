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
import time
from copy import deepcopy
from pathlib import Path

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
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from PIL import Image
from tqdm import tqdm

from reference.utils import (
    VA_CONFIGS,
    apply_robotwin_inference_overrides,
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
from models.experimental.lingbot_va.tt.utils import FlowMatchSchedulerTtnn, get_mesh_id_ttnn, data_seq_to_patch_ttnn
from tt.utils import (
    _safe_deallocate_tensor,
    load_text_encoder as load_text_encoder_tt,
    load_transformer as load_transformer_tt,
    WanVAEStreamingWrapper as TTWanVAEStreamingWrapper,
    WanVAEDecoderWrapper as TTWanVAEDecoderWrapper,
)
from models.experimental.lingbot_va.tests.mesh_utils import (
    inference_work_mesh_from_opened,
    ttnn_mesh_shape_for_inference_demo,
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
                # Best-effort teardown; some descriptors reject clearing.
                logger.debug("Could not clear %s on %s", name, type(obj).__name__, exc_info=True)


class _TTTransformerAdapter:
    """Wraps TTNN ``WanTransformer3DModel``: ``input_dict`` fields are ``ttnn.Tensor`` only; returns ``ttnn.Tensor``."""

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
        for name, t in (
            ("noisy_latents", spatial),
            ("text_emb", prompt),
            ("timesteps", timesteps),
            ("grid_id", grid_id),
        ):
            if not isinstance(t, ttnn.Tensor):
                raise TypeError(f"_TTTransformerAdapter expects {name} to be ttnn.Tensor, got {type(t).__name__}")

        ts = timesteps
        B = int(spatial.shape[0])
        nd = len(ts.shape)
        if nd == 2:
            timestep_per_frame = ts
            timestep = ttnn.squeeze(ttnn.slice(ts, [0, 0], [B, 1]), dim=1)
        elif nd == 1:
            ts_1f = ttnn.unsqueeze(ts, 0)
            timestep_per_frame = ttnn.repeat(ts_1f, (B, 1))
            timestep = ttnn.squeeze(ttnn.slice(timestep_per_frame, [0, 0], [B, 1]), dim=1)
        else:
            raise ValueError(
                f"timesteps must be 1D [F] or 2D [B, F] on mesh, got shape {tuple(int(x) for x in ts.shape)}"
            )
        if len(timestep.shape) == 0:
            timestep = ttnn.unsqueeze(timestep, 0)

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
    kwargs: dict = {"mesh_shape": ttnn_mesh_shape_for_inference_demo()}
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


def _load_models_phase1(config, load_text_encoder=True, mesh_device=None):
    """Load tokenizer, VAE (CPU), mesh, optional TT text encoder. Transformer and TT VAE load in later phases.

    Args:
        mesh_device: Optional pre-opened mesh (e.g. pytest ``mesh_device``). When ``None``, opens a mesh
            via ``_open_lingbot_mesh_device``. In both cases ``inference_work_mesh_from_opened`` is applied
            so ``LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH=1`` yields a ``(1,1)`` submesh when the open mesh
            has more than one device.
    """
    init_logger()
    device = torch.device("cpu")
    dtype = config.param_dtype
    enable_offload = getattr(config, "enable_offload", True)
    ckpt = config.wan22_pretrained_model_name_or_path

    if mesh_device is None:
        opened_mesh = _open_lingbot_mesh_device()
        mesh_device, mesh_parent = inference_work_mesh_from_opened(opened_mesh)
    else:
        mesh_device, mesh_parent = inference_work_mesh_from_opened(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    if mesh_parent is not None:
        logger.info(
            "LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH: using (1,1) submesh inside %s-device open.",
            mesh_parent.get_num_devices(),
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

    tokenizer = load_tokenizer(os.path.join(ckpt, "tokenizer"))

    scheduler = FlowMatchSchedulerTtnn(mesh_device, shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
    action_scheduler = FlowMatchSchedulerTtnn(
        mesh_device, shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True
    )
    # scheduler.set_timesteps(1000, training=True)
    # action_scheduler.set_timesteps(1000, training=True)

    return {
        "vae": vae,
        # "vae_half": vae_half,
        "streaming_vae": streaming_vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        # "streaming_vae_half": streaming_vae_half,
        "scheduler": scheduler,
        "action_scheduler": action_scheduler,
        "device": device,
        "dtype": dtype,
        "cache_name": "pos",
        "config": config,
        "env_type": config.env_type,
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


def _load_tt_vae_into_models(models: dict) -> None:
    """Load TTNN VAE (streaming encoder + quant_conv) into models. Only one TT sub-network on device."""
    vae_parallel_config = _lingbot_vae_hw_parallel_config(models["mesh_device"])
    models["streaming_vae"] = TTWanVAEStreamingWrapper(
        models["vae"],
        models["mesh_device"],
        models["ccl_manager"],
        vae_parallel_config,
    )
    logger.info("Loaded TT VAE encoder (streaming_vae) on device.")


def _free_tt_vae_from_models(models: dict) -> None:
    """Replace TT VAE in models with PyTorch wrappers and run gc to free device memory."""
    streaming_vae = models.get("streaming_vae")
    # old_streaming_vae_half = models.get("streaming_vae_half")
    if streaming_vae is not None:
        _release_ttnn_runtime_configs(streaming_vae)
        try:
            if hasattr(streaming_vae, "cleanup_all"):
                streaming_vae.cleanup_all()
        except Exception as e:
            logger.warning("cleanup_all failed for streaming_vae: %s", e)
        try:
            if hasattr(streaming_vae, "deallocate_weights"):
                streaming_vae.deallocate_weights()
        except Exception as e:
            logger.warning("deallocate_weights failed for streaming_vae: %s", e)
    gc.collect()
    logger.info("Freed TT VAE from device")


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


def _get_t5_prompt_embeds_ttnn(models, tt_input, tt_mask, num_videos_per_prompt=1, max_sequence_length=512):
    """TTNN text-encoder pass using TTNN pre-tokenized inputs, returns TTNN tensor."""
    _ = max_sequence_length
    text_encoder = models["text_encoder"]

    mesh_device = text_encoder.mesh_device
    tt_outputs = text_encoder(tt_input, attention_mask=tt_mask)
    prompt_embeds = tt_outputs[-1]

    # Keep this helper TTNN-only: normalize to [B, L, D], zero-out padded tokens with attention mask,
    # and apply num_videos_per_prompt expansion in TTNN.
    while len(prompt_embeds.shape) > 3:
        prompt_embeds = ttnn.squeeze(prompt_embeds, 0)
    if tt_mask.dtype != prompt_embeds.dtype:
        tt_mask = ttnn.typecast(tt_mask, prompt_embeds.dtype)
    tt_mask_3d = ttnn.unsqueeze(tt_mask, -1)
    prompt_embeds = ttnn.multiply(prompt_embeds, tt_mask_3d)
    if num_videos_per_prompt > 1:
        batch_size, seq_len, hidden_dim = prompt_embeds.shape
        prompt_embeds = ttnn.unsqueeze(prompt_embeds, 1)
        prompt_embeds = ttnn.repeat(prompt_embeds, (1, num_videos_per_prompt, 1, 1))
        prompt_embeds = ttnn.reshape(prompt_embeds, (batch_size * num_videos_per_prompt, seq_len, hidden_dim))
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
    return prompt_embeds


def _encode_prompt_ttnn(
    models,
    tt_input,
    tt_mask,
    negative_prompt_tokenized_inputs=None,
    do_classifier_free_guidance=True,
    max_sequence_length=512,
):
    """Encode prompt embeddings via TTNN text encoder using pre-tokenized inputs."""
    prompt_embeds = _get_t5_prompt_embeds_ttnn(
        models,
        tt_input,
        tt_mask,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )

    negative_prompt_embeds = None
    if do_classifier_free_guidance and negative_prompt_tokenized_inputs is not None:
        negative_prompt_embeds = _get_t5_prompt_embeds_ttnn(
            models,
            tt_input,
            tt_mask,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
    return prompt_embeds, negative_prompt_embeds


def _postprocess_action(models, state, action):
    config = models["config"]
    actions_q01 = state["actions_q01"]
    actions_q99 = state["actions_q99"]
    if isinstance(actions_q01, ttnn.Tensor):
        actions_q01 = ttnn.to_torch(actions_q01).float()
    if isinstance(actions_q99, ttnn.Tensor):
        actions_q99 = ttnn.to_torch(actions_q99).float()
    action_norm_method = state["action_norm_method"]

    action = action.cpu()
    action = action[0, ..., 0]
    if action_norm_method == "quantiles":
        action = (action + 1) / 2 * (actions_q99 - actions_q01 + 1e-6) + actions_q01
    else:
        raise NotImplementedError
    action = action.squeeze(0).detach().cpu().numpy()
    return action[config.used_action_channel_ids]


def _repeat_input_for_cfg_ttnn(models, state, input_dict):
    """Classifier-free guidance repeat for TTNN tensors (matches :func:`_repeat_input_for_cfg`)."""
    use_cfg = state["use_cfg"]
    prompt_embeds = state["prompt_embeds"]
    negative_prompt_embeds = state["negative_prompt_embeds"]
    if use_cfg:
        if negative_prompt_embeds is None:
            raise ValueError("use_cfg requires state['negative_prompt_embeds']")
        if not isinstance(prompt_embeds, ttnn.Tensor) or not isinstance(negative_prompt_embeds, ttnn.Tensor):
            raise TypeError("_repeat_input_for_cfg_ttnn expects prompt and negative prompt as ttnn.Tensor when use_cfg")
        input_dict["noisy_latents"] = ttnn.repeat(input_dict["noisy_latents"], (2, 1, 1, 1, 1))
        input_dict["text_emb"] = ttnn.concat([prompt_embeds, negative_prompt_embeds], dim=0)
        g = ttnn.unsqueeze(input_dict["grid_id"], 0)
        input_dict["grid_id"] = ttnn.repeat(g, (2, 1, 1))
        ts = ttnn.unsqueeze(input_dict["timesteps"], 0)
        input_dict["timesteps"] = ttnn.repeat(ts, (2, 1))
    else:
        input_dict["grid_id"] = ttnn.unsqueeze(input_dict["grid_id"], 0)
        input_dict["timesteps"] = ttnn.unsqueeze(input_dict["timesteps"], 0)
    return input_dict


def _timesteps_1d_ttnn(mesh_device, num_frames: int, t_val: float, *, zero_first: bool = False) -> ttnn.Tensor:
    """1-D timestep vector of length ``num_frames`` (matches torch ``ones([F]) * t``); optionally zero first slot."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if zero_first:
        if num_frames == 1:
            return ttnn.full(
                (1,),
                0.0,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            )
        first = ttnn.full(
            (1,),
            0.0,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        )
        rest = ttnn.full(
            (num_frames - 1,),
            float(t_val),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        )
        return ttnn.concat([first, rest], dim=0)
    return ttnn.full(
        (num_frames,),
        float(t_val),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )


def _action_channel_mask_ttnn(mesh_device, num_channels: int, action_mask: torch.Tensor | ttnn.Tensor) -> ttnn.Tensor:
    """Broadcast mask ``[C]`` (torch bool or TTNN 0/1) to ``[1, C, 1, 1, 1]`` float TTNN (1 = keep, 0 = zero)."""
    if isinstance(action_mask, ttnn.Tensor):
        am = ttnn.to_torch(action_mask).detach().cpu().numpy().squeeze() > 0.5
    else:
        am = action_mask.detach().cpu().numpy().astype(bool)
    m = np.ones((1, num_channels, 1, 1, 1), dtype=np.float32)
    m[0, ~am, 0, 0, 0] = 0.0
    return ttnn.from_torch(
        torch.from_numpy(m),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.float32,
        device=mesh_device,
    )


def _prepare_latent_input_ttnn(
    models,
    state,
    latent_model_input: ttnn.Tensor | None,
    action_model_input: ttnn.Tensor | None,
    latent_t: float = 0.0,
    action_t: float = 0.0,
    latent_cond: ttnn.Tensor | None = None,
    action_cond: ttnn.Tensor | None = None,
    frame_st_id: int = 0,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> dict:
    """
    TTNN-native analogue of :func:`_prepare_latent_input`: all activations are ``ttnn.Tensor`` on ``mesh_device``.

    Expects ``state['prompt_embeds']`` to be a ``ttnn.Tensor``. Does not call :func:`torch.ones` / ``.to`` on
    activations; timestep lines use :func:`ttnn.full` / :func:`ttnn.concat`. Grid ids use
    :func:`models.experimental.lingbot_va.tt.utils.get_mesh_id_ttnn`.

    Callers keep tensors on-device for the TTNN scheduler path.
    """
    mesh_device = models["mesh_device"]
    action_mask = state["action_mask"]
    prompt_embeds = state["prompt_embeds"]
    if not isinstance(prompt_embeds, ttnn.Tensor):
        raise TypeError("_prepare_latent_input_ttnn expects state['prompt_embeds'] to be a ttnn.Tensor")

    input_dict: dict = {}
    if latent_model_input is not None:
        noisy = latent_model_input
        f = int(noisy.shape[-3]) // patch_size[0]
        h = int(noisy.shape[-2]) // patch_size[1]
        w = int(noisy.shape[-1]) // patch_size[2]
        num_frames = int(noisy.shape[2])
        ts = _timesteps_1d_ttnn(mesh_device, num_frames, latent_t, zero_first=(latent_cond is not None))
        grid_id = get_mesh_id_ttnn(mesh_device, f, h, w, 0, 1, frame_st_id, action=False)
        if latent_cond is not None:
            if num_frames > 1:
                tail = noisy[:, :, 1:, :, :]
                noisy = ttnn.concat([latent_cond, tail], dim=2)
            else:
                noisy = latent_cond
        input_dict["latent_res_lst"] = {
            "noisy_latents": noisy,
            "timesteps": ts,
            "grid_id": grid_id,
            "text_emb": prompt_embeds,
        }

    if action_model_input is not None:
        noisy_a = action_model_input
        num_frames_a = int(noisy_a.shape[2])
        ts_a = _timesteps_1d_ttnn(mesh_device, num_frames_a, action_t, zero_first=(action_cond is not None))
        grid_id_a = get_mesh_id_ttnn(
            mesh_device,
            int(noisy_a.shape[-3]),
            int(noisy_a.shape[-2]),
            int(noisy_a.shape[-1]),
            0,
            1,
            frame_st_id,
            action=True,
        )
        if action_cond is not None:
            if num_frames_a > 1:
                tail_a = noisy_a[:, :, 1:, :, :]
                noisy_a = ttnn.concat([action_cond, tail_a], dim=2)
            else:
                noisy_a = action_cond
        ch_mask = _action_channel_mask_ttnn(mesh_device, int(noisy_a.shape[1]), action_mask)
        noisy_a = ttnn.multiply(noisy_a, ch_mask)
        input_dict["action_res_lst"] = {
            "noisy_latents": noisy_a,
            "timesteps": ts_a,
            "grid_id": grid_id_a,
            "text_emb": prompt_embeds,
        }

    return input_dict


def _randn_ttnn(mesh_device, shape: tuple[int, ...], *, torch_dtype: torch.dtype) -> ttnn.Tensor:
    """Gaussian samples on mesh via host ``torch.randn`` and :func:`ttnn.from_torch`."""
    host = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    tt_dtype = ttnn.bfloat16 if torch_dtype == torch.bfloat16 else ttnn.float32
    return ttnn.from_torch(
        host.contiguous(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=tt_dtype,
    )


def _ensure_prompt_embeds_ttnn(models, state) -> None:
    """Ensure ``state['prompt_embeds']`` (and negative when CFG) are TTNN for :func:`_prepare_latent_input_ttnn`."""
    mesh_device = models["mesh_device"]
    dtype = models["dtype"]
    tt_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    pe = state.get("prompt_embeds")
    if pe is None:
        raise RuntimeError("TTNN latent prepare requires encoded prompt embeddings; set prompt before infer.")
    if isinstance(pe, ttnn.Tensor):
        if state.get("use_cfg") and state.get("negative_prompt_embeds") is not None:
            neg = state["negative_prompt_embeds"]
            if not isinstance(neg, ttnn.Tensor):
                state["negative_prompt_embeds"] = ttnn.from_torch(
                    neg.to(dtype).contiguous(),
                    device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=tt_dtype,
                )
        return
    state["prompt_embeds"] = ttnn.from_torch(
        pe.to(dtype).contiguous(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=tt_dtype,
    )
    neg = state.get("negative_prompt_embeds")
    if state.get("use_cfg") and neg is not None and not isinstance(neg, ttnn.Tensor):
        state["negative_prompt_embeds"] = ttnn.from_torch(
            neg.to(dtype).contiguous(),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=tt_dtype,
        )


def _encode_obs_ttnn(models, state, obs):
    """Encode observations to TTNN latents.

    * ``obs['obs_ttnn']``: list of per-frame dicts mapping ``config.obs_cam_keys`` to **ttnn** tensors
      (E2E / device path).
    * ``obs['obs']``: same structure as :func:`_encode_obs` — list of dicts with numpy RGB (H, W, 3) per
      camera key (e.g. ``run_generate`` / ``_load_init_obs``, or ``message['obs']`` from
      :func:`build_infer_message`).
    """
    config = models["config"]
    mesh_device = models["mesh_device"]

    env_type = models["env_type"]
    streaming_vae = models["streaming_vae"]
    vae = models["vae"]

    obs_ttnn = obs.get("obs_ttnn")
    if obs_ttnn is not None:
        if not isinstance(obs_ttnn, list) or len(obs_ttnn) < 1:
            return None
        videos = []
        for k in config.obs_cam_keys:
            if k not in obs_ttnn[0]:
                raise KeyError(f"Missing camera key '{k}' in obs['obs_ttnn']")
            videos.append(obs_ttnn[0][k])

        if env_type == "robotwin_tshape":
            videos_high = ttnn.add(ttnn.multiply(videos[0], 2.0 / 255.0), -1.0)
            videos_left_and_right = ttnn.add(ttnn.multiply(ttnn.concat(videos[1:], dim=0), 2.0 / 255.0), -1.0)
            enc_out_high = streaming_vae.encode_chunk_ttnn(videos_high)
            enc_out_left_and_right = streaming_vae.encode_chunk_ttnn(videos_left_and_right)
            lr_parts = ttnn.chunk(enc_out_left_and_right, enc_out_left_and_right.shape[0], dim=0)
            enc_out = ttnn.concat([ttnn.concat(list(lr_parts), dim=-1), enc_out_high], dim=-2)
        else:
            videos_merged = ttnn.add(ttnn.multiply(ttnn.concat(videos, dim=0), 2.0 / 255.0), -1.0)
            enc_out = streaming_vae.encode_chunk_ttnn(videos_merged)
    else:
        images = obs.get("obs")
        if images is None:
            raise ValueError(
                "_encode_obs_ttnn requires obs['obs_ttnn'] (TTNN) or obs['obs'] (numpy/RGB), same as _encode_obs"
            )
        # Match _encode_obs preprocessing, then upload BCTHW to mesh for encode_chunk_ttnn.
        height = state["height"]
        width = state["width"]
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None

        videos_torch = []
        for k_i, k in enumerate(config.obs_cam_keys):
            if env_type == "robotwin_tshape":
                height_i, width_i = (height, width) if k_i == 0 else (height // 2, width // 2)
            else:
                height_i, width_i = height, width
            history_video_k = torch.from_numpy(np.stack([each[k] for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(
                history_video_k, size=(height_i, width_i), mode="bilinear", align_corners=False
            ).unsqueeze(0)
            videos_torch.append(history_video_k)

        def _encode_chunk_ttnn_from_torch(bcthw: torch.Tensor) -> ttnn.Tensor:
            t = bcthw.detach().to(torch.bfloat16).contiguous()
            if t.device.type != "cpu":
                t = t.cpu()
            v_tt = ttnn.from_torch(
                t,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                device=mesh_device,
            )
            try:
                return streaming_vae.encode_chunk_ttnn(v_tt)
            finally:
                _safe_deallocate_tensor(v_tt, "_encode_obs_ttnn from_torch upload")

        if env_type == "robotwin_tshape":
            videos_high = videos_torch[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos_torch[1:], dim=0) / 255.0 * 2.0 - 1.0
            enc_out_high = _encode_chunk_ttnn_from_torch(videos_high)
            enc_out_left_and_right = _encode_chunk_ttnn_from_torch(videos_left_and_right)
            lr_parts = ttnn.chunk(enc_out_left_and_right, enc_out_left_and_right.shape[0], dim=0)
            enc_out = ttnn.concat([ttnn.concat(list(lr_parts), dim=-1), enc_out_high], dim=-2)
        else:
            videos_merged = torch.cat(videos_torch, dim=0) / 255.0 * 2.0 - 1.0
            enc_out = _encode_chunk_ttnn_from_torch(videos_merged)

    mu, _ = ttnn.chunk(enc_out, 2, dim=1)
    latents_mean = ttnn.from_torch(
        torch.tensor(vae.config.latents_mean, dtype=torch.bfloat16).view(1, -1, 1, 1, 1),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
    )
    latents_std_inv = ttnn.from_torch(
        torch.tensor(1.0 / np.array(vae.config.latents_std), dtype=torch.bfloat16).view(1, -1, 1, 1, 1),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
    )
    mu_norm = ttnn.multiply(ttnn.subtract(mu, latents_mean), latents_std_inv)
    parts = ttnn.chunk(mu_norm, mu_norm.shape[0], dim=0)
    return ttnn.concat(list(parts), dim=-1)


def _ttnn_action_channel_mask_vector(mesh_device, action_dim: int, used_channel_ids) -> ttnn.Tensor:
    """1D float mask on mesh: ``1.0`` at ``used_channel_ids``, else ``0.0`` (ttnn only).

    Row-major ``typecast`` requires the last dimension to be a multiple of 32; we build on a
    padded length then ``slice`` back to ``action_dim``.
    """
    pad_to = ((action_dim + 31) // 32) * 32
    kw = dict(device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    ar = ttnn.arange(start=0, end=pad_to, step=1, **kw)
    acc = ttnn.zeros((pad_to,), **kw)
    for idx in sorted(set(int(i) for i in used_channel_ids)):
        if idx < 0 or idx >= action_dim:
            continue
        c = ttnn.full((pad_to,), float(idx), **kw)
        hit = ttnn.eq(ar, c)
        acc = ttnn.add(acc, ttnn.typecast(hit, ttnn.float32))
    return ttnn.slice(acc, [0], [action_dim])


def _ttnn_quantile_table_c11(mesh_device, values) -> ttnn.Tensor:
    """``(len(values), 1, 1)`` float32 on mesh from a Python iterable (ttnn ``full`` + ``concat``)."""
    fv = [float(x) for x in values]
    if not fv:
        raise ValueError("_ttnn_quantile_table_c11: values must be non-empty")
    kw = dict(device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    out = ttnn.full((1, 1, 1), fv[0], **kw)
    for v in fv[1:]:
        out = ttnn.concat([out, ttnn.full((1, 1, 1), v, **kw)], dim=0)
    return out


def _reset_state(models, state, prompt):
    config = models["config"]
    transformer = models["transformer"]
    streaming_vae = models["streaming_vae"]
    device = models["device"]
    mesh_device = models["mesh_device"]
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

    state["action_mask"] = _ttnn_action_channel_mask_vector(
        mesh_device, config.action_dim, config.used_action_channel_ids
    )
    state["actions_q01"] = _ttnn_quantile_table_c11(mesh_device, config.norm_stat["q01"])
    state["actions_q99"] = _ttnn_quantile_table_c11(mesh_device, config.norm_stat["q99"])
    state["action_norm_method"] = config.action_norm_method

    if prompt is None:
        state["prompt_embeds"] = None
        state["negative_prompt_embeds"] = None
        state["_prompt_embeds_prompt"] = None
    else:
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        if state.get("_prompt_embeds_prompt") != prompt_list or "prompt_embeds" not in state:
            if models.get("text_encoder") is not None:
                _free_tt_model(models, "text_encoder")
            _load_text_encoder_into_models(models, config)
            tokenizer = models["tokenizer"]
            text_encoder = models["text_encoder"]
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
            enc_mesh = text_encoder.mesh_device
            tt_input = ttnn.from_torch(text_input_ids, dtype=ttnn.uint32, device=enc_mesh)
            tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=enc_mesh)
            prompt_embeds, negative_prompt_embeds = _encode_prompt_ttnn(
                models,
                tt_input,
                tt_mask,
                do_classifier_free_guidance=(config.guidance_scale > 1),
                max_sequence_length=512,
            )
            state["prompt_embeds"] = prompt_embeds
            state["negative_prompt_embeds"] = negative_prompt_embeds
            state["_prompt_embeds_prompt"] = prompt_list
            _free_tt_model(models, "text_encoder")


def _infer_impl(models, state, obs, frame_st_id=0):
    config = models["config"]
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
        init_latent = _encode_obs_ttnn(models, state, obs)
        state["init_latent"] = init_latent

    mesh_device = models["mesh_device"]
    tt_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    _ensure_prompt_embeds_ttnn(models, state)

    _set_seed()
    latents_tt = _randn_ttnn(
        mesh_device,
        (1, 48, frame_chunk_size, latent_height, latent_width),
        torch_dtype=dtype,
    )
    actions_tt = _randn_ttnn(
        mesh_device,
        (1, config.action_dim, frame_chunk_size, action_per_frame, 1),
        torch_dtype=dtype,
    )

    video_inference_step = config.num_inference_steps
    action_inference_step = config.action_num_inference_steps
    video_step = config.video_exec_step

    scheduler.set_timesteps(video_inference_step)
    action_scheduler.set_timesteps(action_inference_step)
    video_sched_len = int(scheduler.timesteps.shape[0])
    action_sched_len = int(action_scheduler.timesteps.shape[0])
    video_total_steps = video_step if video_step != -1 else (video_sched_len + 1)
    action_total_steps = action_sched_len + 1
    if video_total_steps < 0:
        raise ValueError(f"video_exec_step must be >= 0 or -1, got {video_total_steps}")

    init_latent = state.get("init_latent")
    patch_size = config.patch_size

    def _schedule_timestep_at(schedule_tt: ttnn.Tensor, idx: int, length: int) -> ttnn.Tensor:
        if idx < length:
            return ttnn.slice(schedule_tt, [idx], [idx + 1])
        return ttnn.full((1,), 0.0, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)

    def _select_first_batch(x: ttnn.Tensor) -> ttnn.Tensor:
        shape = [int(s) for s in x.shape]
        starts = [0] * len(shape)
        ends = shape.copy()
        ends[0] = 1
        return ttnn.slice(x, starts, ends)

    def _apply_cfg(x: ttnn.Tensor, scale: float) -> ttnn.Tensor:
        shape = [int(s) for s in x.shape]
        if shape[0] < 2:
            raise RuntimeError("CFG requested but transformer output batch < 2")
        starts0 = [0] * len(shape)
        ends0 = shape.copy()
        ends0[0] = 1
        starts1 = [0] * len(shape)
        starts1[0] = 1
        ends1 = shape.copy()
        ends1[0] = 2
        cond = ttnn.slice(x, starts0, ends0)
        uncond = ttnn.slice(x, starts1, ends1)
        out = ttnn.add(uncond, ttnn.multiply(ttnn.subtract(cond, uncond), float(scale)))
        ttnn.deallocate(cond)
        ttnn.deallocate(uncond)
        return out

    for i in tqdm(range(video_total_steps)):
        last_step = i == video_total_steps - 1
        t_tt = _schedule_timestep_at(scheduler.timesteps, i, video_sched_len)
        t_scalar = float(t_tt.item())
        latent_cond_tt = init_latent[:, :, 0:1, :, :] if (frame_st_id == 0 and init_latent is not None) else None
        input_dict = _prepare_latent_input_ttnn(
            models,
            state,
            latents_tt,
            None,
            latent_t=t_scalar,
            action_t=t_scalar,
            latent_cond=latent_cond_tt,
            action_cond=None,
            frame_st_id=frame_st_id,
            patch_size=patch_size,
        )
        video_noise_pred = transformer(
            _repeat_input_for_cfg_ttnn(models, state, input_dict["latent_res_lst"]),
            update_cache=1 if last_step else 0,
            cache_name=cache_name,
            action_mode=False,
            dump_iter=None,
        )
        # Older paths may return sequence [B, N, C]; convert to [B, C, T, H, W] on device.
        if len(video_noise_pred.shape) == 3:
            video_noise_pred = data_seq_to_patch_ttnn(
                config.patch_size,
                video_noise_pred,
                frame_chunk_size,
                latent_height,
                latent_width,
                batch_size=int(video_noise_pred.shape[0]),
            )
        if len(video_noise_pred.shape) != 5:
            raise RuntimeError(f"Unexpected video_noise_pred shape: {tuple(int(x) for x in video_noise_pred.shape)}")

        if not last_step or video_step != -1:
            if config.guidance_scale > 1:
                video_noise_pred = _apply_cfg(video_noise_pred, config.guidance_scale)
            else:
                video_noise_pred = _select_first_batch(video_noise_pred)
            if video_noise_pred.dtype != latents_tt.dtype:
                video_noise_pred = ttnn.typecast(video_noise_pred, latents_tt.dtype)
            latents_tt = scheduler.step(video_noise_pred, t_tt, latents_tt, return_dict=False)

        if frame_st_id == 0 and init_latent is not None:
            b, c, f, h, w = (int(x) for x in latents_tt.shape)
            if f > 1:
                latent_tail = ttnn.slice(latents_tt, [0, 0, 1, 0, 0], [b, c, f, h, w])
                latents_tt = ttnn.concat([init_latent[:, :, 0:1, :, :], latent_tail], dim=2)
                ttnn.deallocate(latent_tail)
            else:
                latents_tt = init_latent[:, :, 0:1, :, :]

    for i in tqdm(range(action_total_steps)):
        last_step = i == action_total_steps - 1
        t_tt = _schedule_timestep_at(action_scheduler.timesteps, i, action_sched_len)
        t_scalar = float(t_tt.item())
        action_cond_tt = None
        if frame_st_id == 0:
            action_cond_tt = ttnn.full(
                (1, config.action_dim, 1, action_per_frame, 1),
                0.0,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=tt_dtype,
            )
        input_dict = _prepare_latent_input_ttnn(
            models,
            state,
            None,
            actions_tt,
            latent_t=t_scalar,
            action_t=t_scalar,
            latent_cond=None,
            action_cond=action_cond_tt,
            frame_st_id=frame_st_id,
            patch_size=patch_size,
        )
        action_noise_pred = transformer(
            _repeat_input_for_cfg_ttnn(models, state, input_dict["action_res_lst"]),
            update_cache=1 if last_step else 0,
            cache_name=cache_name,
            action_mode=True,
            dump_iter=None,
        )
        if not last_step:
            if len(action_noise_pred.shape) == 3:
                b_a = int(action_noise_pred.shape[0])
                n_a = int(action_noise_pred.shape[1])
                c_a = int(action_noise_pred.shape[2])
                expected_n = frame_chunk_size * action_per_frame
                if n_a != expected_n:
                    raise RuntimeError(f"Unexpected action sequence length: got {n_a}, expected {expected_n}")
                action_noise_pred = ttnn.reshape(action_noise_pred, (b_a, frame_chunk_size, action_per_frame, c_a))
                action_noise_pred = ttnn.permute(action_noise_pred, (0, 3, 1, 2))
                action_noise_pred = ttnn.unsqueeze(action_noise_pred, dim=4)
            if len(action_noise_pred.shape) != 5:
                raise RuntimeError(
                    f"Unexpected action_noise_pred shape: {tuple(int(x) for x in action_noise_pred.shape)}"
                )

            if config.action_guidance_scale > 1:
                action_noise_pred = _apply_cfg(action_noise_pred, config.action_guidance_scale)
            else:
                action_noise_pred = _select_first_batch(action_noise_pred)
            if action_noise_pred.dtype != actions_tt.dtype:
                action_noise_pred = ttnn.typecast(action_noise_pred, actions_tt.dtype)
            actions_tt = action_scheduler.step(action_noise_pred, t_tt, actions_tt, return_dict=False)

        if frame_st_id == 0:
            b, c, f, n, one = (int(x) for x in actions_tt.shape)
            zero_first = ttnn.full(
                (b, c, 1, n, one),
                0.0,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=tt_dtype,
            )
            if f > 1:
                action_tail = ttnn.slice(actions_tt, [0, 0, 1, 0, 0], [b, c, f, n, one])
                actions_tt = ttnn.concat([zero_first, action_tail], dim=2)
                ttnn.deallocate(action_tail)
            else:
                actions_tt = zero_first

    ch_mask = _action_channel_mask_ttnn(mesh_device, int(actions_tt.shape[1]), action_mask)
    actions_tt = ttnn.multiply(actions_tt, ch_mask)
    actions = ttnn.to_torch(actions_tt).to(dtype)
    latents = ttnn.to_torch(latents_tt).to(dtype)
    actions_out = _postprocess_action(models, state, actions)
    return actions_out, latents


def _infer_entry(models, state, obs):
    """Dispatch: reset, compute_kv_cache, or infer one chunk. Returns result dict."""
    reset = obs.get("reset", False)
    prompt = obs.get("prompt", None)

    if reset:
        logger.info("Reset server")
        _reset_state(models, state, prompt=prompt)
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


def _normalize_infer_obs_for_encode(message_obs: dict) -> dict:
    """Adapt ``message['obs']`` for :func:`_encode_obs_ttnn`.

    * ``build_infer_message`` / server style: a **single** frame dict (camera keys, ``task``, …) with no
      top-level ``obs`` / ``obs_ttnn`` — wrap as ``{"obs": [frame_dict]}``.
    * ``_load_init_obs`` style: already ``{"obs": [ ... ]}`` — returned unchanged.
    * Device path: ``{"obs_ttnn": ...}`` — returned unchanged.
    """
    if not isinstance(message_obs, dict):
        raise TypeError(f"message['obs'] must be a dict, got {type(message_obs)}")
    if message_obs.get("obs_ttnn") is not None:
        return message_obs
    if "obs" in message_obs:
        return message_obs
    return {"obs": [message_obs]}


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
    t0 = time.perf_counter()
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info("run_inference: start %s", start_ts)

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
    config.save_root = str(save_dir)
    config.num_chunks_to_infer = 1
    # Do not assign None over VA_CONFIGS defaults; ``apply_robotwin_inference_overrides`` applies
    # non-None kwargs and optional LINGBOT_VA_* env vars only.
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
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

    mesh_device = text_encoder.mesh_device
    tt_input = ttnn.from_torch(text_input_ids, dtype=ttnn.uint32, device=mesh_device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=mesh_device)
    prompt_embeds, neg_embeds = _encode_prompt_ttnn(
        models,
        tt_input,
        tt_mask,
        do_classifier_free_guidance=(config.guidance_scale > 1),
        max_sequence_length=512,
    )
    state["prompt_embeds"] = prompt_embeds
    state["negative_prompt_embeds"] = neg_embeds
    state["_prompt_embeds_prompt"] = prompt_list
    _free_tt_model(models, "text_encoder")

    # Phase 2: load only TT VAE encoder; run _encode_obs, then free.
    _prepare_state_for_vae_encode(state, config)
    init_obs = _normalize_infer_obs_for_encode(message["obs"])
    _load_tt_vae_into_models(models)
    state["init_latent"] = _encode_obs_ttnn(models, state, init_obs)
    _free_tt_vae_from_models(models)

    # Phase 3: load TT transformer and run generation.
    _load_transformer_into_models(models, config)

    _reset_state(models, state, prompt)
    action, _ = _infer_impl(models, state, init_obs, frame_st_id=state["frame_st_id"])

    elapsed_s = time.perf_counter() - t0
    end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info("run_inference: end %s (elapsed %.3f s)", end_ts, elapsed_s)

    return {"action": action}


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
    t0 = time.perf_counter()
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info("run_generate: start %s", start_ts)

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
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

    mesh_device = text_encoder.mesh_device
    tt_input = ttnn.from_torch(text_input_ids, dtype=ttnn.uint32, device=mesh_device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=mesh_device)
    prompt_embeds, neg_embeds = _encode_prompt_ttnn(
        models,
        tt_input,
        tt_mask,
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
    _load_tt_vae_into_models(models)
    state["init_latent"] = _encode_obs_ttnn(models, state, init_obs)
    _free_tt_vae_from_models(models)

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
    _close_lingbot_mesh_stack(models)
    opened_mesh = _open_lingbot_mesh_device()
    mesh_device, mesh_parent = inference_work_mesh_from_opened(opened_mesh)
    models["mesh_device"] = mesh_device
    models["mesh_device_parent"] = mesh_parent
    models["ccl_manager"] = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    _load_tt_vae_decoder_into_models(models, config)
    decoded_video = None
    if getattr(config, "enable_offload", True):
        models["vae"] = models["vae"].to(models["device"]).to(models["dtype"])
        decoded_video = _decode_one_video(models, pred_latent, "np")[0]
        _free_tt_vae_decoder_from_models(models)
        if models.get("vae_decoder_tt") is not None:
            _free_tt_vae_decoder_from_models(models)
    if decoded_video is not None:
        export_to_video(decoded_video, os.path.join(config.save_root, "demo.mp4"), fps=10)
    _close_lingbot_mesh_stack(models)

    elapsed_s = time.perf_counter() - t0
    end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info("run_generate: end %s (elapsed %.3f s)", end_ts, elapsed_s)

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
