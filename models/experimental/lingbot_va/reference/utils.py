# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference-side helpers used by Lingbot-VA demo scripts."""

import concurrent.futures
import logging
import math
import os

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from easydict import EasyDict
from transformers import T5TokenizerFast, UMT5EncoderModel

from .transformer_wan import WanTransformer3DModel

logger = logging.getLogger(__name__)


def init_logger():
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    os.environ.setdefault("KINETO_LOG_LEVEL", "5")


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        shift=None,
        dynamic_shift_len=None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.training = bool(training)

    def step(self, model_output, timestep, sample, to_final=False, return_dict=True, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b


def get_mesh_id(f, h, w, t, f_w=1, f_shift=0, action=False):
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1
    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


def data_seq_to_patch(
    patch_size,
    data_seq,
    latent_num_frames,
    latent_height,
    latent_width,
    batch_size=1,
):
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w
    data_patch = data_seq.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch


_save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def save_async(obj, file_path):
    # Save work runs in a single background thread to avoid blocking inference loops.
    if torch.is_tensor(obj) or (isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values())):
        if torch.is_tensor(obj):
            obj = obj.cpu()
        elif isinstance(obj, dict):
            obj = {k: v.cpu() if torch.is_tensor(v) else v for k, v in obj.items()}
        _save_executor.submit(torch.save, obj, file_path)
    elif isinstance(obj, np.ndarray):
        obj_copy = obj.copy()
        _save_executor.submit(np.save, file_path, obj_copy)
    else:
        _save_executor.submit(torch.save, obj, file_path)


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    if eval_mode:
        model.eval().requires_grad_(False)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)
    return model


def shard_model(model, param_dtype=torch.bfloat16, reduce_dtype=torch.float32):
    """No-op when not distributed; avoids dependency on torch.distributed.fsdp."""
    _ = (param_dtype, reduce_dtype)
    return model


va_shared_cfg = EasyDict()
va_shared_cfg.host = "0.0.0.0"
va_shared_cfg.port = 29536
va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = "./train_out"
va_shared_cfg.patch_size = (1, 2, 2)
va_shared_cfg.enable_offload = True

va_robotwin_cfg = EasyDict(__name__="Config: VA robotwin")
va_robotwin_cfg.update(va_shared_cfg)
va_robotwin_cfg.wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"
va_robotwin_cfg.attn_window = 72
va_robotwin_cfg.frame_chunk_size = 6
va_robotwin_cfg.env_type = "robotwin_tshape"
va_robotwin_cfg.height = 256
va_robotwin_cfg.width = 320
va_robotwin_cfg.action_dim = 30
va_robotwin_cfg.action_per_frame = 16
va_robotwin_cfg.obs_cam_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
va_robotwin_cfg.guidance_scale = 1
va_robotwin_cfg.action_guidance_scale = 1
va_robotwin_cfg.num_inference_steps = 25
va_robotwin_cfg.video_exec_step = -1
va_robotwin_cfg.action_num_inference_steps = 50
va_robotwin_cfg.snr_shift = 5.0
va_robotwin_cfg.action_snr_shift = 1.0
va_robotwin_cfg.used_action_channel_ids = (
    list(range(0, 7)) + list(range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
)
_inverse = [len(va_robotwin_cfg.used_action_channel_ids)] * va_robotwin_cfg.action_dim
for i, j in enumerate(va_robotwin_cfg.used_action_channel_ids):
    _inverse[j] = i
va_robotwin_cfg.inverse_used_action_channel_ids = _inverse
va_robotwin_cfg.action_norm_method = "quantiles"
va_robotwin_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055,
        -3.6716461181640625e-05,
        -0.08783501386642456,
        -1,
        -1,
        -1,
        -1,
        -0.3547105032205582,
        -1.3113021850585938e-06,
        -0.11975435614585876,
        -1,
        -1,
        -1,
        -1,
    ]
    + [0.0] * 16,
    "q99": [
        0.3462600058317184,
        0.39966784834861746,
        0.14745532035827624,
        1,
        1,
        1,
        1,
        0.034201726913452024,
        0.39142737388610793,
        0.1792279863357542,
        1,
        1,
        1,
        1,
    ]
    + [0.0] * 14
    + [1.0, 1.0],
}

VA_CONFIGS = {"robotwin": va_robotwin_cfg}


def _local_path(p):
    """Resolve to absolute path so from_pretrained treats it as local, not a HF repo id."""
    return str(os.path.abspath(os.path.expanduser(p)))


def load_vae(vae_path, torch_dtype, torch_device):
    vae = AutoencoderKLWan.from_pretrained(
        _local_path(vae_path),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    return vae.to(torch_device)


def load_text_encoder(text_encoder_path, torch_dtype, torch_device):
    text_encoder = UMT5EncoderModel.from_pretrained(
        _local_path(text_encoder_path),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path):
    return T5TokenizerFast.from_pretrained(
        _local_path(tokenizer_path),
        local_files_only=True,
    )


def load_transformer(transformer_path, torch_dtype, torch_device):
    # Override attn_mode: checkpoints may use "flex" which is not implemented here; use "torch".
    model = WanTransformer3DModel.from_pretrained(
        _local_path(transformer_path),
        torch_dtype=torch_dtype,
        local_files_only=True,
        attn_mode="torch",
    )
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(
        batch_size,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )
    return x


class WanVAEStreamingWrapper:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
