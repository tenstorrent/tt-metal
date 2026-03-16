# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Build the observation dict and run Lingbot-VA inference locally (no server).

1. Build the input dict from three camera images (same format as client→server).
2. Run inference using the same logic as VA_Server.infer() from wan_va_server.py:
   reset with prompt, then infer one chunk on the observation dict.
   All logic is inlined here; no dependency on wan_va_server.py or VA_Server class.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

# Repo root (parent of evaluation/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    WanVAEStreamingWrapper,
    _configure_model,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
    logger,
    save_async,
    shard_model,
)

# Keys the server uses (va_robotwin_cfg.obs_cam_keys)
OBS_CAM_HIGH = "observation.images.cam_high"
OBS_CAM_LEFT_WRIST = "observation.images.cam_left_wrist"
OBS_CAM_RIGHT_WRIST = "observation.images.cam_right_wrist"
OBS_STATE = "observation.state"

# Cache file for VAE encode output; if present, skip running the VAE encoder (saves a lot of time).
# Use a distinct name from inference_ttnn so the two scripts never share the same VAE cache.
VAE_ENC_CACHE_FILENAME = "vae_encoded_obs_torch.pt"
# Cache file for text (prompt) embeddings; if present and prompt matches, skip running the text encoder.
# Use a distinct name from inference_ttnn so this script never loads TTNN embeddings (and vice versa).
TEXT_EMB_CACHE_FILENAME = "text_emb_cache_torch.pt"

# Seed for reproducible inference (latents/actions init, etc.).
REPRODUCIBLE_SEED = 42


def _set_seed(seed: int = REPRODUCIBLE_SEED) -> None:
    """Set random seeds so that inference is reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_kv_cache_message(key_frame_obs_list: list[dict], state: np.ndarray):
    """Build the message for model.infer(...) to update KV cache after executing a chunk."""
    return {
        "obs": key_frame_obs_list,
        "compute_kv_cache": True,
        "imagine": False,
        "state": np.asarray(state, dtype=np.float64),
    }


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


# -----------------------------------------------------------------------------
# Inference implementation (logic copied from wan_va_server.VA_Server, no class)
# -----------------------------------------------------------------------------


def _load_models(config):
    """Load VAE, tokenizer, text_encoder, transformer, schedulers. Returns models dict."""
    init_logger()
    device = torch.device("cpu")
    dtype = config.param_dtype
    enable_offload = getattr(config, "enable_offload", True)
    ckpt = config.wan22_pretrained_model_name_or_path

    vae = load_vae(
        os.path.join(ckpt, "vae"),
        torch_dtype=dtype,
        torch_device="cpu" if enable_offload else device,
    )
    streaming_vae = WanVAEStreamingWrapper(vae)

    tokenizer = load_tokenizer(os.path.join(ckpt, "tokenizer"))

    text_encoder = load_text_encoder(
        os.path.join(ckpt, "text_encoder"),
        torch_dtype=dtype,
        torch_device="cpu" if enable_offload else device,
    )

    transformer = load_transformer(
        os.path.join(ckpt, "transformer"),
        torch_dtype=dtype,
        torch_device=device,
    )
    transformer = _configure_model(
        model=transformer,
        shard_fn=shard_model,
        param_dtype=dtype,
        device=device,
        eval_mode=True,
    )

    scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
    action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    action_scheduler.set_timesteps(1000, training=True)

    streaming_vae_half = None
    if config.env_type == "robotwin_tshape":
        vae_half = load_vae(
            os.path.join(ckpt, "vae"),
            torch_dtype=dtype,
            torch_device="cpu" if enable_offload else device,
        )
        streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    return {
        "vae": vae,
        "streaming_vae": streaming_vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "transformer": transformer,
        "streaming_vae_half": streaming_vae_half,
        "scheduler": scheduler,
        "action_scheduler": action_scheduler,
        "device": device,
        "dtype": dtype,
        "cache_name": "pos",
        "config": config,
        "env_type": config.env_type,
    }


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

    text_encoder_device = next(text_encoder.parameters()).device
    prompt_embeds = text_encoder(text_input_ids.to(text_encoder_device), mask.to(text_encoder_device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
        dim=0,
    )
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    return prompt_embeds.to(device)


def _encode_prompt(models, state, prompt, do_classifier_free_guidance=True, max_sequence_length=512):
    device = models["device"]
    dtype = models["dtype"]
    config = models["config"]

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    cache_path = getattr(config, "text_emb_cache_path", None)
    if cache_path and os.path.isfile(cache_path):
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("prompt") == prompt and "prompt_embeds" in cached:
                logger.info("Loading text embeddings from cache: %s", cache_path)
                prompt_embeds = cached["prompt_embeds"].to(device=device, dtype=dtype)
                neg = cached.get("negative_prompt_embeds")
                negative_prompt_embeds = neg.to(device=device, dtype=dtype) if neg is not None else None
                return prompt_embeds, negative_prompt_embeds
        except Exception as e:
            logger.warning("Failed to load text emb cache: %s", e)

    prompt_embeds = _get_t5_prompt_embeds(
        models, prompt=prompt, num_videos_per_prompt=1, max_sequence_length=max_sequence_length
    )

    negative_prompt_embeds = None
    if do_classifier_free_guidance:
        negative_prompt = batch_size * [""]
        negative_prompt_embeds = _get_t5_prompt_embeds(
            models, prompt=negative_prompt, num_videos_per_prompt=1, max_sequence_length=max_sequence_length
        )

    if cache_path:
        try:
            torch.save(
                {
                    "prompt": prompt,
                    "prompt_embeds": prompt_embeds.cpu(),
                    "negative_prompt_embeds": negative_prompt_embeds.cpu()
                    if negative_prompt_embeds is not None
                    else None,
                },
                cache_path,
            )
            logger.info("Saved text embeddings to cache: %s", cache_path)
        except Exception as e:
            logger.warning("Failed to save text emb cache: %s", e)
    return prompt_embeds, negative_prompt_embeds


def _normalize_latents(latents, latents_mean, latents_std):
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    return ((latents.float() - latents_mean) * latents_std).to(latents.dtype)


def _preprocess_action(models, state, action):
    config = models["config"]
    device = models["device"]
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
    config = models["config"]
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

    cache_path = getattr(config, "vae_enc_cache_path", None)
    if cache_path and os.path.isfile(cache_path):
        logger.info("Loading VAE encode output from cache: %s", cache_path)
        video_latent = torch.load(cache_path, map_location=device, weights_only=True)
        return video_latent.to(dtype)

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
        enc_out_left_and_right = streaming_vae_half.encode_chunk(videos_left_and_right.to(vae_device).to(dtype))
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

    if cache_path:
        try:
            torch.save(video_latent.cpu(), cache_path)
            logger.info("Saved VAE encode output to cache: %s", cache_path)
        except Exception as e:
            logger.warning("Failed to save VAE encode cache: %s", e)
    return video_latent


def _reset_state(models, state, prompt):
    config = models["config"]
    transformer = models["transformer"]
    streaming_vae = models["streaming_vae"]
    streaming_vae_half = models["streaming_vae_half"]
    device = models["device"]
    dtype = models["dtype"]
    cache_name = models["cache_name"]
    save_root = config.save_root

    logger.info("Reset.")
    state["use_cfg"] = False  # Fixed to batch_size=1 for this demo
    state["frame_st_id"] = 0
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
    else:
        prompt_embeds, negative_prompt_embeds = _encode_prompt(
            models,
            state,
            prompt,
            do_classifier_free_guidance=(config.guidance_scale > 1),
            max_sequence_length=512,
        )
        state["prompt_embeds"] = prompt_embeds
        state["negative_prompt_embeds"] = negative_prompt_embeds

    state["exp_name"] = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
    state["exp_save_root"] = os.path.join(save_root, "real", state["exp_name"])
    os.makedirs(state["exp_save_root"], exist_ok=True)


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

    if frame_st_id == 0:
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
    init_latent = state["init_latent"]
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
            )
            if not last_step or video_step != -1:
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
            )
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
    save_async(latents, os.path.join(state["exp_save_root"], f"latents_{frame_st_id}.pt"))
    save_async(actions, os.path.join(state["exp_save_root"], f"actions_{frame_st_id}.pt"))
    actions_out = _postprocess_action(models, state, actions)
    return actions_out, latents


def _compute_kv_cache(models, state, obs):
    config = models["config"]
    transformer = models["transformer"]
    cache_name = models["cache_name"]

    transformer.clear_pred_cache(cache_name)
    save_async(
        obs["obs"],
        os.path.join(state["exp_save_root"], f"obs_data_{state['frame_st_id']}.pt"),
    )
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
        logger.info("******************* Reset server ******************")
        _reset_state(models, state, prompt=prompt)
        return {}
    if compute_kv_cache:
        logger.info("################# Compute KV Cache #################")
        _compute_kv_cache(models, state, obs)
        return {}
    logger.info("################# Infer One Chunk #################")
    action, _ = _infer_impl(models, state, obs, frame_st_id=state["frame_st_id"])
    return {"action": action}


def _decode_one_video(models, latents, output_type="np"):
    vae = models["vae"]
    device = models["device"]
    dtype = models["dtype"]

    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
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
) -> dict:
    """
    Run Lingbot-VA inference on the input dict (same behavior as VA_Server.infer).

    Uses config and model loading from wan_va; no VA_Server class. Resets with
    message['prompt'], then runs infer one chunk and returns {'action': ...}.
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
    if save_dir is None:
        save_dir = _SCRIPT_DIR / "out_inference"
    config.save_root = str(save_dir)
    config.vae_enc_cache_path = os.path.join(config.save_root, VAE_ENC_CACHE_FILENAME)
    config.text_emb_cache_path = os.path.join(config.save_root, TEXT_EMB_CACHE_FILENAME)

    models = _load_models(config)
    state = {}
    reset_message = build_reset_message(message.get("prompt", ""))
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
    config.vae_enc_cache_path = os.path.join(config.save_root, VAE_ENC_CACHE_FILENAME)
    config.text_emb_cache_path = os.path.join(config.save_root, TEXT_EMB_CACHE_FILENAME)
    config.input_img_path = str(images_dir)
    config.prompt = prompt
    config.num_chunks_to_infer = num_chunks

    models = _load_models(config)
    state = {}
    _reset_state(models, state, prompt)
    init_obs = _load_init_obs(config, config.input_img_path)

    pred_latent_lst = []
    pred_action_lst = []
    print(f"Generating {config.num_chunks_to_infer} chunks")
    for chunk_id in range(config.num_chunks_to_infer):
        actions, latents = _infer_impl(models, state, init_obs, frame_st_id=(chunk_id * config.frame_chunk_size))
        actions = torch.from_numpy(actions)
        pred_latent_lst.append(latents)
        pred_action_lst.append(actions)

    pred_latent = torch.cat(pred_latent_lst, dim=2)
    transformer = models["transformer"]
    streaming_vae = models["streaming_vae"]
    streaming_vae_half = models["streaming_vae_half"]
    transformer.clear_cache(models["cache_name"])
    streaming_vae.clear_cache()
    if streaming_vae_half:
        streaming_vae_half.clear_cache()
    del models["transformer"]
    if models.get("streaming_vae_half"):
        del models["streaming_vae_half"]
    del models["text_encoder"]

    if getattr(config, "enable_offload", True):
        models["vae"] = models["vae"].to(models["device"]).to(models["dtype"])
    decoded_video = _decode_one_video(models, pred_latent, "np")[0]
    export_to_video(decoded_video, os.path.join(config.save_root, "demo.mp4"), fps=10)
    return str(Path(save_dir) / "demo.mp4")


def _print_dict_shapes(d: dict, prefix: str = "") -> None:
    """Print dict keys and shapes of numpy arrays (for debugging)."""
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"  {prefix}{k}: shape {v.shape}, dtype {v.dtype}")
        elif isinstance(v, dict):
            print(f"  {prefix}{k}: (dict)")
            _print_dict_shapes(v, prefix=prefix + "    ")
        elif isinstance(v, (int, float, str, bool)) or v is None:
            print(f"  {prefix}{k}: {v!r}")
        else:
            print(f"  {prefix}{k}: type {type(v).__name__}")


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
        help="Dir for internal saves (latents_*.pt, actions_*.pt) or demo.mp4 when --generate. Default: evaluation/robotwin/out_inference.",
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
        help="Number of chunks for generate() (only used with --generate). Default: 10.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or str(_SCRIPT_DIR / "out_inference")
    images_dir = Path(args.images_dir) if args.images_dir else _REPO_ROOT / "example" / "robotwin"

    if args.generate:
        if not args.checkpoint:
            print("--generate requires --checkpoint (or LINGBOT_VA_CHECKPOINT).")
            sys.exit(1)
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            if not (images_dir / f"{key}.png").exists():
                print(f"Missing {images_dir / f'{key}.png'} for generate(). Use --images-dir.")
                sys.exit(1)
        print("=" * 60)
        print("Running generate() (no infer): multi-chunk → decode → demo.mp4")
        print("=" * 60)
        print("Checkpoint:", args.checkpoint)
        print("Images dir:", images_dir)
        print("Prompt:", repr(args.prompt))
        print("Num chunks:", args.num_chunks)
        print("Save dir:", save_dir)
        print("=" * 60)
        out_path = run_generate(
            args.checkpoint,
            images_dir,
            args.prompt,
            save_dir,
            num_chunks=args.num_chunks,
        )
        print("Generated video saved to:", out_path)
        return

    # Infer mode: build message, run reset + infer one chunk
    cam_high_path = images_dir / "observation.images.cam_high.png"
    cam_left_path = images_dir / "observation.images.cam_left_wrist.png"
    cam_right_path = images_dir / "observation.images.cam_right_wrist.png"
    for p in (cam_high_path, cam_left_path, cam_right_path):
        if not p.exists():
            print(f"Missing image: {p}")
            print(f"  Use --images-dir to specify another dir.")
            sys.exit(1)

    message = load_message_from_files(
        str(cam_high_path),
        str(cam_left_path),
        str(cam_right_path),
        prompt=args.prompt,
    )

    print("=" * 60)
    print("Input dict (message for model.infer)")
    print("=" * 60)
    print("Top-level keys:", list(message.keys()))
    print("Observation keys (message['obs']):", list(message["obs"].keys()))
    print("Observation array shapes:")
    for k in (OBS_CAM_HIGH, OBS_CAM_LEFT_WRIST, OBS_CAM_RIGHT_WRIST):
        arr = message["obs"][k]
        print(f"  {k}: {arr.shape} {arr.dtype}")
    print("Prompt:", repr(message["prompt"]))
    print("=" * 60)

    if not args.checkpoint:
        print("No --checkpoint (or LINGBOT_VA_CHECKPOINT) set. Skipping inference.")
        print("Set checkpoint path to run inference on the above dict.")
        return

    print("\nRunning inference (reset + infer one chunk)...")
    print("Internal saves (latents_*.pt, actions_*.pt):", save_dir)
    result = run_inference(message, args.checkpoint, save_dir=save_dir)
    print("=" * 60)
    print("Inference result")
    print("=" * 60)
    if "action" in result:
        action = result["action"]
        print("action shape:", action.shape, "dtype:", action.dtype)
        if args.output:
            np.save(args.output, action)
            print("action saved to:", args.output)
    else:
        print("Keys:", list(result.keys()))
        _print_dict_shapes(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
