# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def position_ids_in_meshgrid_tt(tt_patch_embeds_list, max_width, device):
    position_ids_tt = []
    for tt_patch in tt_patch_embeds_list:
        shape = tt_patch.shape
        height, width = shape[-2], shape[-1]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid

        tt_ids = ttnn.from_torch(
            ids,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        position_ids_tt.append(tt_ids[:, 0])
    return ttnn.concat(position_ids_tt, dim=0)


def apply_scaling_vision(freqs: torch.Tensor, scale_factor: float, orig_context_len: int):
    return freqs / scale_factor


def precompute_mistral_vision_freqs(
    dim: int, max_patches_per_side: int, theta: float, scale_factor=None, orig_context_len=None
):
    base_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    if scale_factor is not None:
        base_freqs = apply_scaling_vision(base_freqs, scale_factor, orig_context_len)

    h_idx = torch.arange(max_patches_per_side)
    w_idx = torch.arange(max_patches_per_side)

    freqs_h = torch.outer(h_idx, base_freqs[::2])
    freqs_w = torch.outer(w_idx, base_freqs[1::2])

    inv_freq = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
            freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
        ],
        dim=-1,
    ).reshape(-1, dim // 2)

    full_freqs = torch.cat([inv_freq, inv_freq], dim=-1)
    return full_freqs.cos(), full_freqs.sin()
