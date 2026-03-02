# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import concurrent.futures

import numpy as np
import torch

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

__all__ = ["get_mesh_id", "save_async", "data_seq_to_patch"]


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
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch


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

    grid_id = torch.cat(
        [
            ff.unsqueeze(0),
            hh.unsqueeze(0),
            ww.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


def save_async(obj, file_path):
    """
    todo
    """
    if torch.is_tensor(obj) or (isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values())):
        if torch.is_tensor(obj):
            if obj.is_cuda:
                obj = obj.cpu()
        elif isinstance(obj, dict):
            obj = {k: v.cpu() if torch.is_tensor(v) else v for k, v in obj.items()}
        executor.submit(torch.save, obj, file_path)
    elif isinstance(obj, np.ndarray):
        obj_copy = obj.copy()
        executor.submit(np.save, file_path, obj_copy)
    else:
        executor.submit(torch.save, obj, file_path)


def sample_timestep_id(
    batch_size: int = 1,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
):
    u = torch.rand(size=[batch_size])
    u = u * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    timestep_id = (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)
    return timestep_id


def warmup_constant_lambda(current_step, warmup_steps=1000):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0
