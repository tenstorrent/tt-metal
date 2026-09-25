# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of `Qwen2_5_VisionTransformerPretrainedModel` (`model.visual`).

Composed of the graduated sub-module ports:

    x = vision_patch_embed(pixel_patches)             Conv3d(stride == kernel) == Linear(1176 -> 1280)
    x = x[window_index]                                reorder tokens window-major (groups of 4)
    x = v_l_vision_block_i(x, full | window mask)      32 blocks (heads + MLP split over TP, fp32 residual)
    merged = v_l_patch_merger(x)[argsort(window_index)]

Batched over images: N images of the same grid run as a leading axis [N, 1, s_pad, C]. Attention
never crosses images (HF's cu_seqlens blocks are exactly per-image), so the per-image window/full
masks, the rotary tables and the token permutations are shared by every image and broadcast over N.
Token order permutations and masks depend only on grid_thw (integer metadata): they are built once
per grid on host (numpy) and applied on device as one-hot matmuls / additive masks.

TP scheme: see encoder_stack.py (heads + MLP column/row-parallel, all_reduce on the TP axis; any DP
axis replicates).
"""

from __future__ import annotations

import numpy as np

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import (
    block_mask,
    hifi4_config,
    pad_rows,
    pad_to_tile,
    split_matmul,
    upload,
)
from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import _bf16, _fp32, _one_hot, _vision_metadata
from models.demos.qwen_image_edit_text_encoder._stubs.v_l_patch_merger import TtVLPatchMerger
from models.demos.qwen_image_edit_text_encoder._stubs.v_l_vision_block import TtVLVisionBlock
from models.demos.qwen_image_edit_text_encoder._stubs.vision_patch_embed import TtVisionPatchEmbed


class VisionConsts:
    """Per-grid device constants shared by every image of that grid."""

    def __init__(self, cos, sin, mask_full, mask_win, perm, unperm, s, s_pad, m, m_pad):
        self.cos, self.sin = cos, sin
        self.mask_full, self.mask_win = mask_full, mask_win
        self.perm, self.unperm = perm, unperm  # [N, 1, s_pad, s_pad] / [N, 1, m_pad, m_pad] one-hot
        self.s, self.s_pad, self.m, self.m_pad = s, s_pad, m, m_pad


class TtVisionTransformer:
    def __init__(self, device, torch_module):
        self.device = device
        cfg = torch_module.config
        self.merge = int(torch_module.spatial_merge_size)
        self.unit = self.merge * self.merge
        self.window_size = int(torch_module.window_size)
        self.patch_size = int(torch_module.patch_size)
        self.fullatt = set(int(i) for i in torch_module.fullatt_block_indexes)
        self.hidden = int(cfg.hidden_size)
        self.head_dim = self.hidden // int(cfg.num_heads)
        self.inv_freq = torch_module.rotary_pos_emb.inv_freq.detach().float().numpy()

        self.patch_embed = TtVisionPatchEmbed(device, torch_module.patch_embed)
        self.blocks = [TtVLVisionBlock(device, blk) for blk in torch_module.blocks]
        self.merger = TtVLPatchMerger(device, torch_module.merger)
        self.compute_cfg = hifi4_config()
        # precise: float32 rotary tables, exact-product patch embed / token permutations, float32 merger
        # input (the last blocks carry |x| ~ 2.6e4 activations that bf16 rounds by ~100)
        self.precise = False

    # ---- per-grid constants (integer metadata -> device tables) ---------------------------------
    def consts(self, grid, n_images):
        """grid: (t, h, w) shared by all n_images. Built once, outside the forward."""
        grid = tuple(int(v) for v in grid)
        pos, window_index, cu_window, cu_full = _vision_metadata([grid], self.merge, self.window_size, self.patch_size)
        s = grid[0] * grid[1] * grid[2]
        s_pad = pad_to_tile(s)
        rows = (window_index[:, None] * self.unit + np.arange(self.unit)[None, :]).reshape(-1)
        freqs = (pos[:, :, None].astype(np.float32) * self.inv_freq[None, None, :]).reshape(s, -1)[rows]
        emb = np.concatenate([freqs, freqs], axis=-1)
        D = emb.shape[-1]
        if self.precise:  # float32 tables (the bf16 tables round every angle's cos/sin to 8 bits)
            cos = upload(self.device, pad_rows(np.cos(emb), s_pad).reshape(1, 1, s_pad, D), dtype=ttnn.float32)
            sin = upload(self.device, pad_rows(np.sin(emb), s_pad).reshape(1, 1, s_pad, D), dtype=ttnn.float32)
        else:
            cos, sin = self.blocks[0].block.attn.rope_tables(np.cos(emb), np.sin(emb), s_pad)
        mask_full = upload(self.device, block_mask(cu_full, s, s_pad))
        mask_win = upload(self.device, block_mask(cu_window, s, s_pad))
        m = s // self.unit
        m_pad = pad_to_tile(s_pad // self.unit)
        perm = np.repeat(_one_hot(rows, s_pad), n_images, axis=0)
        unperm = np.repeat(_one_hot(np.argsort(window_index), m_pad), n_images, axis=0)
        return VisionConsts(
            cos,
            sin,
            mask_full,
            mask_win,
            upload(self.device, perm, dtype=ttnn.float32),
            upload(self.device, unperm),
            s,
            s_pad,
            m,
            m_pad,
        )

    # ---- device forward ---------------------------------------------------------------------------
    def forward_batched(self, pixels, c: VisionConsts):
        """pixels: [N, 1, s_pad, 1176] fp32 replicated (rows >= s are zero).
        Returns (last_hidden_state [N, 1, s_pad, C] fp32, merged image embeddings [N, m, out])."""
        n = pixels.shape[0]
        cfg = self.compute_cfg
        if self.precise:
            x = self.patch_embed(pixels, dtype=ttnn.float32, precise=True)
            x = split_matmul(c.perm, x, compute_kernel_config=cfg, limbs=getattr(self, "limbs", 2))
        else:
            x = self.patch_embed(pixels, dtype=ttnn.float32)
            x = ttnn.matmul(c.perm, x, compute_kernel_config=cfg, dtype=ttnn.float32)
        for i, blk in enumerate(self.blocks):
            x = blk.forward_padded(x, c.cos, c.sin, c.mask_full if i in self.fullatt else c.mask_win)

        merged = self.merger.forward_padded(_fp32(x) if self.precise else _bf16(x))  # [N, 1, s_pad // unit, out]
        m_rows = merged.shape[-2]
        if c.m_pad != m_rows:
            merged = ttnn.pad(merged, [(0, 0), (0, 0), (0, c.m_pad - m_rows), (0, 0)], 0.0)
        if self.precise:
            merged = split_matmul(c.unperm, merged, compute_kernel_config=cfg, limbs=getattr(self, "limbs", 2))
        else:
            merged = ttnn.matmul(c.unperm, merged, compute_kernel_config=cfg)
        out = merged.shape[-1]
        merged = ttnn.reshape(ttnn.slice(merged, [0, 0, 0, 0], [n, 1, c.m, out]), (n, c.m, out))
        return x, merged

    def __call__(self, hidden_states, grid_thw=None, **kwargs):
        """HF signature: hidden_states [sum_i t*h*w, 1176] (images concatenated), grid_thw [N, 3]."""
        grids = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        assert len(set(grids)) == 1, "batched vision port needs every image on the same grid"
        n = len(grids)
        c = self.consts(grids[0], n)
        s, s_pad = c.s, c.s_pad
        x = ttnn.reshape(hidden_states, (n, 1, s, hidden_states.shape[-1]))
        if s_pad != s:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, s_pad - s), (0, 0)], 0.0)
        x, merged = self.forward_batched(_fp32(x), c)
        if s_pad != s:
            x = ttnn.slice(x, [0, 0, 0, 0], [n, 1, s, self.hidden])
        return ttnn.reshape(x, (n * s, self.hidden)), ttnn.reshape(merged, (n * c.m, merged.shape[-1]))


def build(device, torch_module=None):
    return TtVisionTransformer(device, torch_module)


def vision_transformer_pretrained_model(device, torch_module=None):
    return build(device, torch_module)
