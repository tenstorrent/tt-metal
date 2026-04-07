# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lingbot-VA demo: RobotWin-style observations and TTNN inference (VA_Server-compatible).

All PyTorch tensors use CPU; TT workloads run on Tenstorrent mesh via TTNN.

Imports: path bootstrap (``sys.path``) runs first using only ``os``, ``sys``, and ``pathlib``;
all other imports follow in one block (stdlib, then third-party, then repo packages).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Path bootstrap (must run before any ``models.*``, ``reference.*``, or ``tt.*`` imports).
# -----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TT_METAL_ROOT = os.environ.get("TT_METAL_HOME") or str(_REPO_ROOT.parent.parent.parent)
if os.path.isdir(_TT_METAL_ROOT) and _TT_METAL_ROOT not in sys.path:
    sys.path.insert(0, _TT_METAL_ROOT)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -----------------------------------------------------------------------------
# Imports (stdlib → third-party → first-party). Keep grouped; do not insert code above.
# -----------------------------------------------------------------------------
import argparse
import gc
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import ttnn
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from PIL import Image
from tqdm import tqdm

from models.experimental.lingbot_va.tests.download_pretrained_weights import (
    ensure_checkpoint_path_for_run,
    resolve_demo_checkpoint_arg,
)
from models.experimental.lingbot_va.tests.mesh_utils import (
    inference_work_mesh_from_opened,
    ttnn_mesh_shape_for_inference_demo,
)
from models.experimental.lingbot_va.tt.transformer_wan import NUM_HEADS as LINGBOT_NUM_HEADS
from models.experimental.lingbot_va.tt.utils import (
    FlowMatchSchedulerTtnn,
    data_seq_to_patch_ttnn,
    get_mesh_id_ttnn,
)
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from reference.utils import (
    VA_CONFIGS,
    init_logger,
    load_tokenizer,
    load_vae,
    logger,
)
from tt.utils import (
    WanVAEDecoderWrapper as TTWanVAEDecoderWrapper,
    WanVAEStreamingWrapper as TTWanVAEStreamingWrapper,
    _safe_deallocate_tensor,
    load_text_encoder as load_text_encoder_tt,
    load_transformer as load_transformer_tt,
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


def _log_phase_timings_table(title: str, rows: list[tuple[str, float | None]]) -> None:
    """Log ``rows`` as ``(phase_name, seconds | None)``; ``None`` prints as N/A."""
    if not rows:
        return
    label_w = max(len("Phase"), max(len(r[0]) for r in rows))
    lines = [
        title,
        f"{'Phase':<{label_w}}  {'Time (s)':>12}",
        f"{'-' * label_w}  {'-' * 12}",
    ]
    for label, sec in rows:
        if sec is None:
            lines.append(f"{label:<{label_w}}  {'N/A':>12}")
        else:
            lines.append(f"{label:<{label_w}}  {sec:12.4f}")
    logger.info("\n".join(lines))


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
                pass


class _TTTransformerAdapter:
    """Wraps TTNN ``WanTransformer3DModel``: ``input_dict`` fields are ``ttnn.Tensor`` only; returns ``ttnn.Tensor``."""

    def __init__(self, tt_model):
        self._tt_model = tt_model

    def clear_cache(self, cache_name):
        self._tt_model.clear_cache(cache_name)

    def cleanup_all(self):
        self._tt_model.cleanup_all()

    def deallocate_weights(self):
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
    except Exception:
        pass
    try:
        ttnn.close_mesh_device(work)
    except Exception:
        pass
    if parent is not None:
        try:
            ttnn.close_mesh_device(parent)
        except Exception:
            pass


def _load_models_phase1(
    config,
    mesh_device=None,
    timings_out: dict[str, float] | None = None,
):
    """Load tokenizer, VAE weights (CPU reference for config/weights), mesh, and schedulers.

    ``streaming_vae`` is ``None`` until :func:`_load_tt_vae_into_models` installs the TT VAE encoder.

    Args:
        mesh_device: Optional pre-opened mesh (e.g. pytest ``mesh_device``). When ``None``, opens a mesh
            via ``_open_lingbot_mesh_device``. In both cases ``inference_work_mesh_from_opened`` is applied
            so ``LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH=1`` yields a ``(1,1)`` submesh when the open mesh
            has more than one device.
        timings_out: If provided, stores ``load_tokenizer`` wall time (seconds) for demo phase tables.
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
    _, cols = tuple(mesh_device.shape)
    tp_factor = cols
    if tp_factor > 1 and LINGBOT_NUM_HEADS % tp_factor != 0:
        raise RuntimeError(
            f"Lingbot WanTransformer NUM_HEADS={LINGBOT_NUM_HEADS} is not divisible by tensor_parallel "
            f"factor {tp_factor} (mesh columns). Use MESH_DEVICE=N150, unset MESH_DEVICE, or "
            "LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH=1."
        )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    dit_parallel_config = _lingbot_dit_parallel_config(mesh_device)

    vae = load_vae(
        os.path.join(ckpt, "vae"),
        torch_dtype=dtype,
        torch_device="cpu" if enable_offload else device,
    )

    t_load_tok = time.perf_counter()
    tokenizer = load_tokenizer(os.path.join(ckpt, "tokenizer"))
    if timings_out is not None:
        timings_out["load_tokenizer"] = time.perf_counter() - t_load_tok

    scheduler = FlowMatchSchedulerTtnn(mesh_device, shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
    action_scheduler = FlowMatchSchedulerTtnn(
        mesh_device, shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True
    )

    return {
        "vae": vae,
        "streaming_vae": None,
        "tokenizer": tokenizer,
        "text_encoder": None,
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
            obj.cleanup_all()
        except Exception:
            pass
        try:
            obj.deallocate_weights()
        except Exception:
            pass
        del obj
    gc.collect()


def _load_text_encoder_into_models(models: dict, config) -> None:
    """Load the TTNN UMT5 text encoder into ``models``."""
    ckpt = config.wan22_pretrained_model_name_or_path
    dtype = models["dtype"]
    models["text_encoder"] = load_text_encoder_tt(
        os.path.join(ckpt, "text_encoder"),
        models["mesh_device"],
        ccl_manager=models["ccl_manager"],
        torch_dtype=dtype,
        max_prompt_length=512,
    )


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


def _free_tt_vae_from_models(models: dict) -> None:
    """Tear down TTNN VAE encoder on device and run gc. Expects :func:`_load_tt_vae_into_models` to have run."""
    streaming_vae = models.pop("streaming_vae")
    _release_ttnn_runtime_configs(streaming_vae)
    try:
        streaming_vae.cleanup_all()
    except Exception:
        pass
    try:
        streaming_vae.deallocate_weights()
    except Exception:
        pass
    del streaming_vae
    gc.collect()


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
    except Exception:
        pass
    try:
        ttnn.deallocate(tt_input)
    except Exception:
        pass
    try:
        ttnn.deallocate(tt_mask)
    except Exception:
        pass
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


def _repeat_input_for_cfg_ttnn(models, state, input_dict):
    """Classifier-free guidance repeat for TTNN tensors (matches :func:`_repeat_input_for_cfg`)."""
    use_cfg = state["use_cfg"]
    prompt_embeds = state["prompt_embeds"]
    negative_prompt_embeds = state["negative_prompt_embeds"]
    if use_cfg:
        if negative_prompt_embeds is None:
            raise ValueError("use_cfg requires state['negative_prompt_embeds']")
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


def _action_channel_mask_ttnn(mesh_device, num_channels: int, action_mask: ttnn.Tensor) -> ttnn.Tensor:
    """Broadcast mask ``[C]`` (TTNN 0/1) to ``[1, C, 1, 1, 1]`` float (1 = keep, 0 = zero)."""
    am = ttnn.to_torch(action_mask).detach().cpu().numpy().squeeze() > 0.5
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
    TTNN-only: activations are ``ttnn.Tensor`` on ``mesh_device``; ``state['prompt_embeds']`` must be TTNN.
    Timestep lines use :func:`ttnn.full` / :func:`ttnn.concat`; grid ids use
    :func:`models.experimental.lingbot_va.tt.utils.get_mesh_id_ttnn`.
    """
    mesh_device = models["mesh_device"]
    action_mask = state["action_mask"]
    prompt_embeds = state["prompt_embeds"]

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


def _ensure_prompt_embeds_ttnn(state) -> None:
    """Require ``state['prompt_embeds']`` from the TT text encoder path."""
    if state.get("prompt_embeds") is None:
        raise RuntimeError("TTNN latent prepare requires encoded prompt embeddings; set prompt before infer.")


def _encode_obs_ttnn(models, state, obs):
    """Encode observations to TTNN latents via ``encode_chunk_ttnn``.

    * ``obs['obs_ttnn']``: list of per-frame dicts mapping ``config.obs_cam_keys`` to **ttnn** tensors.
    * ``obs['obs']``: list of dicts with numpy RGB (H, W, 3) per camera key (uploaded to mesh before encode).
    """
    config = models["config"]
    mesh_device = models["mesh_device"]

    env_type = models["env_type"]
    streaming_vae = models["streaming_vae"]
    vae = models["vae"]

    obs_ttnn = obs.get("obs_ttnn")
    if obs_ttnn is not None:
        if not isinstance(obs_ttnn, list) or len(obs_ttnn) < 1:
            raise ValueError("obs['obs_ttnn'] must be a non-empty list of per-frame dicts")
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
            raise ValueError("_encode_obs_ttnn requires obs['obs_ttnn'] or obs['obs'] (numpy RGB per camera key)")
        # Match _encode_obs preprocessing, then upload BCTHW to mesh for encode_chunk_ttnn.
        height = state["height"]
        width = state["width"]
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            raise ValueError("obs['obs'] must contain at least one frame dict")

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


def _ttnn_latents_mean_std_bcthw(mesh_device, channel_values, z_dim: int) -> ttnn.Tensor:
    """``(1, z_dim, 1, 1, 1)`` bfloat16 on mesh from per-channel floats; ``ttnn.full`` + ``concat`` only."""
    if z_dim < 1:
        raise ValueError("_ttnn_latents_mean_std_bcthw: z_dim must be >= 1")
    kw = dict(device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    out = ttnn.full((1, 1, 1, 1, 1), float(channel_values[0]), **kw)
    for i in range(1, z_dim):
        out = ttnn.concat(
            [out, ttnn.full((1, 1, 1, 1, 1), float(channel_values[i]), **kw)],
            dim=1,
        )
    return out


def _reset_state(models, state, prompt):
    config = models["config"]
    transformer = models["transformer"]
    # TT VAE encoder may already be freed after ``_encode_obs_ttnn`` (full demo pipeline); optional for prepared runs.
    streaming_vae = models.get("streaming_vae")
    device = models["device"]
    mesh_device = models["mesh_device"]
    dtype = models["dtype"]
    cache_name = models["cache_name"]

    state["use_cfg"] = False
    state["frame_st_id"] = 0
    # Keep precomputed init latent when available (run_generate phase-2 preload).
    if "init_latent" not in state:
        state["init_latent"] = None

    transformer.clear_cache(cache_name)
    if streaming_vae is not None:
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
    """Video + action denoise loops.

    Returns ``(actions_tt, latents_tt)``: both are ``ttnn.Tensor`` on ``models['mesh_device']``.
    ``latents_tt`` is normalized BCTHW. ``actions_tt`` has the action-channel mask applied (still in
    model sample space). For host numpy in simulation scale, use :func:`ttnn.to_torch` then
    :func:`_postprocess_action`.
    """
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
    _ensure_prompt_embeds_ttnn(state)

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
        # Sequence head [B, N, C] -> [B, C, T, H, W] on device.
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

    # Masked on-device action samples; :func:`run_inference` returns this tensor as ``action`` unchanged.
    return actions_tt, latents_tt


def _load_tt_vae_decoder_into_models(models: dict, config) -> None:
    """Load TTNN VAE decoder into models. Use before decode when running generate with TT path."""
    vae_parallel_config = _lingbot_vae_hw_parallel_config(models["mesh_device"])
    models["vae_decoder_tt"] = TTWanVAEDecoderWrapper(
        models["vae"],
        models["mesh_device"],
        ccl_manager=models["ccl_manager"],
        parallel_config=vae_parallel_config,
    )


def _free_tt_vae_decoder_from_models(models: dict) -> None:
    """Remove TT VAE decoder from models and run gc to free device memory."""
    vae_decoder_tt = models.pop("vae_decoder_tt")
    _release_ttnn_runtime_configs(vae_decoder_tt)
    try:
        vae_decoder_tt.cleanup_all()
    except Exception:
        pass
    try:
        vae_decoder_tt.deallocate_weights()
    except Exception:
        pass
    del vae_decoder_tt
    gc.collect()


def _decode_one_video(models, latents: ttnn.Tensor, output_type="np"):
    """Decode latents to RGB video.

    ``latents`` must be a **normalized** ``ttnn.Tensor`` ``(B, C, T, H, W)`` on ``models['mesh_device']``,
    same scale as :func:`_infer_impl` / encoder output. Denormalizes on device with ``ttnn`` (same math as
    ``latents / (1/latents_std) + latents_mean``), then :class:`~models.experimental.lingbot_va.tt.utils.WanVAEDecoderWrapper.decode`.

    Mean/std broadcast tensors are built with ``ttnn.full`` / ``ttnn.concat`` only (no ``numpy`` or ``torch`` here).
    Diffusers :class:`~diffusers.video_processor.VideoProcessor` still consumes the decoder's ``torch.Tensor`` output.
    """
    vae = models["vae"]
    vae_decoder_tt = models["vae_decoder_tt"]
    mesh_device = models["mesh_device"]
    z_dim = int(vae.config.z_dim)
    # Encoder uses (x - mean) * (1/std); decoder inverse is x * std + mean with config latents_std per channel.
    mean_tt = _ttnn_latents_mean_std_bcthw(mesh_device, vae.config.latents_mean, z_dim)
    std_tt = _ttnn_latents_mean_std_bcthw(mesh_device, vae.config.latents_std, z_dim)

    x = latents
    if latents.dtype != ttnn.bfloat16:
        x = ttnn.typecast(latents, ttnn.bfloat16)

    denorm_tt = ttnn.add(ttnn.multiply(x, std_tt), mean_tt)
    _safe_deallocate_tensor(mean_tt, "_decode_one_video mean_tt")
    _safe_deallocate_tensor(std_tt, "_decode_one_video std_tt")
    if x is not latents:
        _safe_deallocate_tensor(x, "_decode_one_video latents typecast")

    try:
        video = vae_decoder_tt.decode(denorm_tt)
    finally:
        _safe_deallocate_tensor(denorm_tt, "_decode_one_video denorm_tt")

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
    prepared: tuple[dict, dict, dict] | None = None,
    return_latents: bool = False,
    log_time: bool = False,
) -> dict:
    """
    Run Lingbot-VA inference on the input dict (same behavior as VA_Server.infer).

    Uses config and model loading from wan_va; no VA_Server class. Resets with
    message['prompt'], then runs infer one chunk and returns {'action': ...}.

    Optional keyword overrides set ``config.num_inference_steps`` and
    ``config.action_num_inference_steps`` when not ``None``; otherwise ``VA_CONFIGS`` defaults apply.

    If ``prepared`` is ``(models, state, init_obs)`` from a prior load (e.g. perf ``TtLingbotVA.prepare``),
    skips Phases 1–3 and only runs ``_reset_state`` + ``_infer_impl``. ``checkpoint_path`` must match
    ``models['config'].wan22_pretrained_model_name_or_path``. When ``return_latents`` is True, the return
    dict also includes ``latents`` (``ttnn.Tensor`` BCTHW, normalized).

    When ``log_time`` is True, records per-phase wall times (``time.perf_counter``), logs start/end
    timestamps with total elapsed, and prints the phase timings table. Default is False (no timing overhead).
    """
    if log_time:
        t0 = time.perf_counter()
        start_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info("run_inference: start %s", start_ts)

    checkpoint_path = Path(checkpoint_path).resolve()

    if prepared is not None:
        models, state, init_obs = prepared
        cfg_ckpt = Path(models["config"].wan22_pretrained_model_name_or_path).resolve()
        if checkpoint_path != cfg_ckpt:
            raise ValueError(
                f"run_inference(prepared=...): checkpoint_path {checkpoint_path} != config checkpoint {cfg_ckpt}"
            )
        _set_seed()
        os.chdir(_REPO_ROOT)
        config = models["config"]
        if num_inference_steps is not None:
            config.num_inference_steps = num_inference_steps
        if action_num_inference_steps is not None:
            config.action_num_inference_steps = action_num_inference_steps
        prompt = message.get("prompt", "")
        _reset_state(models, state, prompt)
        if log_time:
            t_inf0 = time.perf_counter()
        action, latents = _infer_impl(models, state, init_obs, frame_st_id=state["frame_st_id"])
        if log_time:
            _log_phase_timings_table(
                "run_inference (prepared) phase timings",
                [
                    ("infer_impl", time.perf_counter() - t_inf0),
                ],
            )
            elapsed_s = time.perf_counter() - t0
            end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.info("run_inference: end %s (elapsed %.3f s, prepared path)", end_ts, elapsed_s)
        out: dict = {"action": action}
        if return_latents:
            out["latents"] = latents
        return out

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
    if num_inference_steps is not None:
        config.num_inference_steps = num_inference_steps
    if action_num_inference_steps is not None:
        config.action_num_inference_steps = action_num_inference_steps
    if save_dir is None:
        save_dir = _SCRIPT_DIR
    config.save_root = str(save_dir)

    # Phase 1: load shared assets (VAE once, tokenizer, mesh); load TT text encoder.
    if log_time:
        _phase1_timings: dict[str, float] = {}
        models = _load_models_phase1(config, timings_out=_phase1_timings)
    else:
        models = _load_models_phase1(config)
    state = {}
    prompt = message.get("prompt", "")
    prompt_list = [prompt] if isinstance(prompt, str) else prompt
    phase_timings: list[tuple[str, float | None]] = []
    if log_time:
        phase_timings.append(("load_tokenizer", _phase1_timings.get("load_tokenizer")))

    if log_time:
        t0_te = time.perf_counter()
    _load_text_encoder_into_models(models, config)
    if log_time:
        phase_timings.append(("load_text_encoder_tt", time.perf_counter() - t0_te))

    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    if log_time:
        t0_tok_call = time.perf_counter()
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    if log_time:
        phase_timings.append(("tokenizer_call", time.perf_counter() - t0_tok_call))
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

    mesh_device = text_encoder.mesh_device
    tt_input = ttnn.from_torch(text_input_ids, dtype=ttnn.uint32, device=mesh_device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=mesh_device)
    if log_time:
        t0_enc_prompt = time.perf_counter()
    prompt_embeds, neg_embeds = _encode_prompt_ttnn(
        models,
        tt_input,
        tt_mask,
        do_classifier_free_guidance=(config.guidance_scale > 1),
        max_sequence_length=512,
    )
    if log_time:
        phase_timings.append(("encode_prompt_ttnn", time.perf_counter() - t0_enc_prompt))
    state["prompt_embeds"] = prompt_embeds
    state["negative_prompt_embeds"] = neg_embeds
    state["_prompt_embeds_prompt"] = prompt_list
    _free_tt_model(models, "text_encoder")

    # Phase 2: load only TT VAE encoder; run _encode_obs, then free.
    _prepare_state_for_vae_encode(state, config)
    init_obs = _normalize_infer_obs_for_encode(message["obs"])
    if log_time:
        t0_vae_enc = time.perf_counter()
    _load_tt_vae_into_models(models)
    if log_time:
        phase_timings.append(("load_tt_vae_encoder", time.perf_counter() - t0_vae_enc))

    if log_time:
        t0_enc_obs = time.perf_counter()
    state["init_latent"] = _encode_obs_ttnn(models, state, init_obs)
    if log_time:
        phase_timings.append(("encode_obs_ttnn", time.perf_counter() - t0_enc_obs))
    _free_tt_vae_from_models(models)

    # Phase 3: load TT transformer and run generation.
    if log_time:
        t0_tr = time.perf_counter()
    _load_transformer_into_models(models, config)
    if log_time:
        phase_timings.append(("load_transformer", time.perf_counter() - t0_tr))

    _reset_state(models, state, prompt)
    if log_time:
        t0_infer = time.perf_counter()
    action, latents = _infer_impl(models, state, init_obs, frame_st_id=state["frame_st_id"])
    if log_time:
        phase_timings.append(("infer_impl", time.perf_counter() - t0_infer))
        phase_timings.append(("load_vae_decoder", None))
        phase_timings.append(("decode_one_video", None))

        _log_phase_timings_table("run_inference phase timings", phase_timings)

        elapsed_s = time.perf_counter() - t0
        end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info("run_inference: end %s (elapsed %.3f s)", end_ts, elapsed_s)

    out = {"action": action}
    if return_latents:
        out["latents"] = latents
    return out


def run_generate(
    checkpoint_path: str | Path,
    images_dir: str | Path,
    prompt: str,
    save_dir: str | Path,
    num_chunks: int = 10,
    *,
    log_time: bool = False,
) -> str:
    """
    Run multi-chunk video generation (same behavior as VA_Server.generate).
    Loads init obs from images_dir, runs num_chunks of inference, decodes to video, saves demo.mp4.

    When ``log_time`` is True, records per-phase wall times and logs start/end with total elapsed.
    Default is False (no timing overhead).
    """
    if log_time:
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
    if log_time:
        _phase1_timings: dict[str, float] = {}
        models = _load_models_phase1(config, timings_out=_phase1_timings)
    else:
        models = _load_models_phase1(config)
    state = {}
    prompt_list = [prompt] if isinstance(prompt, str) else prompt
    gen_timings: list[tuple[str, float | None]] = []
    if log_time:
        gen_timings.append(("load_tokenizer", _phase1_timings.get("load_tokenizer")))

    if log_time:
        t0_te = time.perf_counter()
    _load_text_encoder_into_models(models, config)
    if log_time:
        gen_timings.append(("load_text_encoder_tt", time.perf_counter() - t0_te))

    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    if log_time:
        t0_tok_call = time.perf_counter()
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    if log_time:
        gen_timings.append(("tokenizer_call", time.perf_counter() - t0_tok_call))
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

    mesh_device = text_encoder.mesh_device
    tt_input = ttnn.from_torch(text_input_ids, dtype=ttnn.uint32, device=mesh_device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=mesh_device)
    if log_time:
        t0_ep = time.perf_counter()
    prompt_embeds, neg_embeds = _encode_prompt_ttnn(
        models,
        tt_input,
        tt_mask,
        do_classifier_free_guidance=(config.guidance_scale > 1),
        max_sequence_length=512,
    )
    if log_time:
        gen_timings.append(("encode_prompt_ttnn", time.perf_counter() - t0_ep))
    state["prompt_embeds"] = prompt_embeds
    state["negative_prompt_embeds"] = neg_embeds
    state["_prompt_embeds_prompt"] = prompt_list
    _free_tt_model(models, "text_encoder")

    # Phase 2: load only TT VAE encoder; run _encode_obs, then free.
    _prepare_state_for_vae_encode(state, config)
    init_obs = _load_init_obs(config, config.input_img_path)
    if log_time:
        t0_vae = time.perf_counter()
    _load_tt_vae_into_models(models)
    if log_time:
        gen_timings.append(("load_tt_vae_encoder", time.perf_counter() - t0_vae))

    if log_time:
        t0_eo = time.perf_counter()
    state["init_latent"] = _encode_obs_ttnn(models, state, init_obs)
    if log_time:
        gen_timings.append(("encode_obs_ttnn", time.perf_counter() - t0_eo))
    _free_tt_vae_from_models(models)

    # Phase 3: load TT transformer and run generation.
    if log_time:
        t0_tr = time.perf_counter()
    _load_transformer_into_models(models, config)
    if log_time:
        gen_timings.append(("load_transformer", time.perf_counter() - t0_tr))

    _reset_state(models, state, prompt)

    pred_latent_lst = []
    if log_time:
        t0_chunks = time.perf_counter()
    for chunk_id in range(config.num_chunks_to_infer):
        actions_tt, latents = _infer_impl(models, state, init_obs, frame_st_id=(chunk_id * config.frame_chunk_size))
        pred_latent_lst.append(latents)
        _safe_deallocate_tensor(actions_tt, f"run_generate actions_tt chunk {chunk_id}")
    if log_time:
        gen_timings.append(("infer_chunks_total", time.perf_counter() - t0_chunks))

    pred_latent = pred_latent_lst[0] if len(pred_latent_lst) == 1 else ttnn.concat(pred_latent_lst, dim=2)
    if len(pred_latent_lst) > 1:
        for i, pl in enumerate(pred_latent_lst):
            _safe_deallocate_tensor(pl, f"run_generate pred_latent chunk {i} (post-concat)")
    pred_latent_lst.clear()

    # Teardown TT transformer on the **same** mesh (TT VAE encoder was already popped in ``_free_tt_vae_from_models``).
    # ``pred_latent`` stays valid on device; TT VAE decoder loads next on this mesh for ``_decode_one_video``.
    transformer = models.pop("transformer")
    _release_ttnn_runtime_configs(transformer)
    try:
        transformer.clear_cache(models["cache_name"])
    except Exception:
        pass
    try:
        transformer.cleanup_all()
    except Exception:
        pass
    try:
        transformer.deallocate_weights()
    except Exception:
        pass
    del transformer

    models.pop("text_encoder", None)
    _release_ttnn_runtime_configs(models)
    gc.collect()

    if log_time:
        t0_dec = time.perf_counter()
    _load_tt_vae_decoder_into_models(models, config)
    if log_time:
        gen_timings.append(("load_vae_decoder", time.perf_counter() - t0_dec))

    models["vae"] = models["vae"].to(models["device"]).to(models["dtype"])
    if log_time:
        t0_dv = time.perf_counter()
    decoded_video = None
    try:
        decoded_video = _decode_one_video(models, pred_latent, "np")[0]
    finally:
        _safe_deallocate_tensor(pred_latent, "run_generate pred_latent after decode")
    if log_time:
        gen_timings.append(("decode_one_video", time.perf_counter() - t0_dv))
    _free_tt_vae_decoder_from_models(models)

    if log_time:
        _log_phase_timings_table("run_generate phase timings", gen_timings)

    if decoded_video is not None:
        video_path = str(Path(config.save_root).resolve() / "demo.mp4")
        export_to_video(decoded_video, video_path, fps=10)
        logger.info("Generated video: %s", video_path)
    _close_lingbot_mesh_stack(models)

    if log_time:
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
        help=(
            "Path to checkpoint dir (vae, tokenizer, text_encoder, transformer). "
            "Default: env LINGBOT_VA_CHECKPOINT, else TT_METAL_HOME/"
            "models/experimental/lingbot_va/reference/checkpoints. Missing weights may be downloaded "
            "unless LINGBOT_VA_SKIP_CHECKPOINT_DOWNLOAD=1."
        ),
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
    parser.add_argument(
        "--log-time",
        action="store_true",
        help="Log phase timings and start/end wall times (enables perf_counter per phase).",
    )
    args = parser.parse_args()
    init_logger()

    save_dir = args.save_dir or str(_SCRIPT_DIR)
    images_dir = Path(args.images_dir) if args.images_dir else _REPO_ROOT / "example" / "robotwin"

    try:
        checkpoint_path = ensure_checkpoint_path_for_run(resolve_demo_checkpoint_arg(args.checkpoint))
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    if args.generate:
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            if not (images_dir / f"{key}.png").exists():
                logger.error("Missing %s for generate(). Use --images-dir.", images_dir / f"{key}.png")
                sys.exit(1)
        run_generate(
            str(checkpoint_path),
            images_dir,
            args.prompt,
            save_dir,
            num_chunks=args.num_chunks,
            log_time=args.log_time,
        )
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

    out = run_inference(message, str(checkpoint_path), save_dir=save_dir, log_time=args.log_time)


if __name__ == "__main__":
    main()
