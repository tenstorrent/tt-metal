# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of the Qwen2.5-VL vision encoder stack
(`Qwen2_5_VisionTransformerPretrainedModel`, `model.visual`).

    x = patch_embed(pixel_patches)                     Conv3d(stride == kernel) == Linear(1176 -> 1280)
    x = x[window_index]                                reorder tokens window-major (groups of 4)
    for blk in blocks:                                 32 x (RMSNorm -> attn -> +, RMSNorm -> MLP -> +)
        x = blk(x, cu_seqlens = full if blk in fullatt_block_indexes else windows)
    merged = merger(x)[argsort(window_index)]          RMSNorm, 4-token merge, Linear-GELU-Linear
    returns (last_hidden_state, merged)

TP scheme on the TP (column) axis of the mesh, DP axis replicated:
    * attention: heads split (column-parallel qkv, row-parallel proj + all_reduce) -- see attention.py
    * MLP: gate/up column-parallel, down row-parallel + all_reduce. intermediate 3420 is zero-padded to
      a multiple of TP*32; the padded gate/up columns are 0 -> silu(0)*0 = 0 and meet zero down rows.
    * merger: fc1 column-parallel, GELU, fc2 row-parallel + all_reduce.
    * patch_embed, norms, biases added after reductions, rotary tables: replicated.
Token order permutations and masks depend only on grid_thw (integer metadata): they are built on
host in numpy and applied on device as one-hot matmuls / additive masks.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import (
    TtVisionAttention,
    block_mask,
    exact_all_reduce,
    hifi4_config,
    mesh_shape,
    pad_to_tile,
    replicate,
    shard_mapper,
    split_linear,
    upload,
    upload_rows,
)


def _pad_cols(w, n):
    return torch.cat([w, torch.zeros(w.shape[0], n - w.shape[1], dtype=w.dtype)], dim=1) if n > w.shape[1] else w


def _pad_rows(w, n):
    return torch.cat([w, torch.zeros(n - w.shape[0], w.shape[1], dtype=w.dtype)], dim=0) if n > w.shape[0] else w


def _bf16(t):
    return t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)


def _fp32(t):
    return t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32)


class TtRMSNorm:
    def __init__(self, device, torch_module, pair=None):
        """pair: a second norm module whose weight goes on mesh row 1 (row-staged layers)."""
        w = torch_module.weight.detach().float()
        self.eps = float(torch_module.variance_epsilon)
        if pair is not None:
            w2 = pair.weight.detach().float()
            self.weight = upload_rows(
                device, [w.reshape(1, 1, -1, 32), w2.reshape(1, 1, -1, 32)], layout=ttnn.ROW_MAJOR_LAYOUT
            )
        else:
            self.weight = ttnn.from_torch(
                w.reshape(1, 1, -1, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=replicate(device),
            )
        self.compute_cfg = hifi4_config()
        # precise: mean(x^2) / rsqrt / scale spelled out in float32 (the fused rms_norm carries ~2x the
        # error of this on the Qwen2.5-VL residual stream; measured per-block branch error 1.7e-3 -> 7.7e-4)
        w32 = [w.reshape(1, -1)] + ([pair.weight.detach().float().reshape(1, -1)] if pair is not None else [])
        self._w32_host = w32
        self._device = device
        self.weight32 = None
        self.precise = False

    def _ensure_w32(self):
        if self.weight32 is None:
            if len(self._w32_host) == 2:
                self.weight32 = upload_rows(self._device, self._w32_host, dtype=ttnn.float32)
            else:
                self.weight32 = upload(self._device, self._w32_host[0], dtype=ttnn.float32)
        return self.weight32

    def __call__(self, x):
        if self.precise:
            x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
            ms = ttnn.mean(ttnn.multiply(x32, x32), dim=-1, keepdim=True, compute_kernel_config=self.compute_cfg)
            return ttnn.multiply(ttnn.multiply(x32, ttnn.rsqrt(ttnn.add(ms, self.eps))), self._ensure_w32())
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight, compute_kernel_config=self.compute_cfg)


class TtVisionMLP:
    """down(act(gate(x)) * up(x)) -- gate/up column-parallel, down row-parallel + all_reduce."""

    def __init__(self, device, torch_module):
        _, self.tp = mesh_shape(device)
        inter = torch_module.gate_proj.out_features
        inter_pad = ((inter + self.tp * 32 - 1) // (self.tp * 32)) * (self.tp * 32)

        def _t(lin):
            return lin.weight.detach().float().t().contiguous()

        def _b(lin, n):
            b = lin.bias.detach().float().reshape(1, -1) if lin.bias is not None else torch.zeros(1, n)
            return _pad_cols(b, n).reshape(1, 1, 1, -1)

        col = shard_mapper(device, -1)
        self.w_gate = upload(device, _pad_cols(_t(torch_module.gate_proj), inter_pad), mapper=col)
        self.w_up = upload(device, _pad_cols(_t(torch_module.up_proj), inter_pad), mapper=col)
        self.b_gate = upload(device, _b(torch_module.gate_proj, inter_pad), mapper=col)
        self.b_up = upload(device, _b(torch_module.up_proj, inter_pad), mapper=col)
        self.w_down = upload(device, _pad_rows(_t(torch_module.down_proj), inter_pad), mapper=shard_mapper(device, 0))
        down_b = torch_module.down_proj.bias
        self.b_down = upload(device, down_b.detach().float().reshape(1, 1, 1, -1)) if down_b is not None else None
        self.compute_cfg = hifi4_config()
        self.device = device
        # precise: gate/up/product in float32, down projection out in float32 with an exact TP reduce.
        self.precise = False

    def __call__(self, x):
        if self.precise:
            cfg = self.compute_cfg
            L = getattr(self, "limbs", 2)
            gate = split_linear(x, self.w_gate, bias=self.b_gate, compute_kernel_config=cfg, limbs=L)
            up = split_linear(x, self.w_up, bias=self.b_up, compute_kernel_config=cfg, limbs=L)
            h = ttnn.multiply(ttnn.silu(gate), up)
            out = split_linear(h, self.w_down, compute_kernel_config=cfg, limbs=L)
            if self.tp > 1:
                out = exact_all_reduce(out, self.device)
            # float32 bias (a bf16 operand rounds the float32 sum to bf16)
            return ttnn.add(out, ttnn.typecast(self.b_down, ttnn.float32)) if self.b_down is not None else out
        gate = ttnn.linear(x, self.w_gate, bias=self.b_gate, compute_kernel_config=self.compute_cfg)
        up = ttnn.linear(x, self.w_up, bias=self.b_up, compute_kernel_config=self.compute_cfg)
        h = ttnn.multiply(ttnn.silu(gate), up)
        out = ttnn.linear(h, self.w_down, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=1, topology=ttnn.Topology.Linear)
        if self.b_down is not None:
            out = ttnn.add(out, self.b_down)
        return out


class TtVisionBlock:
    def __init__(self, device, torch_module):
        self.norm1 = TtRMSNorm(device, torch_module.norm1)
        self.norm2 = TtRMSNorm(device, torch_module.norm2)
        self.attn = TtVisionAttention(device, torch_module.attn)
        self.mlp = TtVisionMLP(device, torch_module.mlp)

    def forward_padded(self, x, tt_cos, tt_sin, tt_mask):
        """x: fp32 residual stream; the branches are accumulated back in fp32. With precise_inputs the
        normalised activations enter the branch matmuls in fp32 instead of being rounded to bf16 first."""
        cast = _fp32 if getattr(self, "precise_inputs", False) else _bf16
        a = self.attn.forward_padded(cast(self.norm1(x)), tt_cos, tt_sin, tt_mask)
        x = ttnn.add(x, _fp32(a))
        return ttnn.add(x, _fp32(self.mlp(cast(self.norm2(x)))))


class TtPatchMerger:
    def __init__(self, device, torch_module):
        _, self.tp = mesh_shape(device)
        self.hidden = int(torch_module.hidden_size)
        self.ln_q = TtRMSNorm(device, torch_module.ln_q)
        fc1, fc2 = torch_module.mlp[0], torch_module.mlp[2]
        col = shard_mapper(device, -1)
        self.w1 = upload(device, fc1.weight.detach().float().t().contiguous(), mapper=col)
        self.b1 = upload(device, fc1.bias.detach().float().reshape(1, 1, 1, -1), mapper=col)
        self.w2 = upload(device, fc2.weight.detach().float().t().contiguous(), mapper=shard_mapper(device, 0))
        self.b2 = upload(device, fc2.bias.detach().float().reshape(1, 1, 1, -1))
        self._device = device
        self.compute_cfg = hifi4_config()

    def __call__(self, x):
        """x: [N, 1, s_pad, C] -> [N, 1, s_pad // unit, out_dim] (row i merges tokens 4i..4i+3)."""
        x = self.ln_q(x)
        x = ttnn.reshape(x, (x.shape[0], 1, x.shape[-2] * x.shape[-1] // self.hidden, self.hidden))
        if getattr(self, "precise", False):
            cfg = self.compute_cfg
            L = getattr(self, "limbs", 2)
            h = ttnn.gelu(
                split_linear(x, self.w1, bias=self.b1, compute_kernel_config=cfg, limbs=L),
                fast_and_approximate_mode=False,
            )
            out = split_linear(h, self.w2, compute_kernel_config=cfg, limbs=L)
            if self.tp > 1:
                out = exact_all_reduce(out, self._device)
            return ttnn.add(out, ttnn.typecast(self.b2, ttnn.float32))
        h = ttnn.linear(x, self.w1, bias=self.b1, compute_kernel_config=self.compute_cfg)
        h = ttnn.gelu(h, fast_and_approximate_mode=False)
        out = ttnn.linear(h, self.w2, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=1, topology=ttnn.Topology.Linear)
        return ttnn.add(out, self.b2)


def _vision_metadata(grid_thw, merge, window_size, patch_size):
    """numpy port of HF get_vision_position_ids / get_vision_cu_seqlens / get_vision_window_index."""
    pos, window_index, cu_window, cu_full = [], [], [0], [0]
    win_id = 0
    vit_win = window_size // merge // patch_size
    unit = merge * merge
    for t, h, w in grid_thw:
        hp = np.broadcast_to(np.arange(h)[:, None], (h, w))
        wp = np.broadcast_to(np.arange(w)[None, :], (h, w))
        hp = hp.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
        wp = wp.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
        pos.append(np.tile(np.stack([hp, wp], axis=-1), (t, 1)))
        for _ in range(t):
            cu_full.append(cu_full[-1] + h * w)

        lh, lw = h // merge, w // merge
        index = np.arange(t * lh * lw).reshape(t, lh, lw)
        pad_h = vit_win - lh % vit_win
        pad_w = vit_win - lw % vit_win
        nwh, nww = (lh + pad_h) // vit_win, (lw + pad_w) // vit_win
        padded = np.full((t, lh + pad_h, lw + pad_w), -100, dtype=np.int64)
        padded[:, :lh, :lw] = index
        padded = padded.reshape(t, nwh, vit_win, nww, vit_win).transpose(0, 1, 3, 2, 4)
        padded = padded.reshape(t, nwh * nww, vit_win, vit_win)
        seqlens = (padded != -100).sum(axis=(2, 3)).reshape(-1)
        flat = padded.reshape(-1)
        window_index.append(flat[flat != -100] + win_id)
        cu_window.extend((np.cumsum(seqlens) * unit + cu_window[-1]).tolist())
        win_id += t * lh * lw
    cu_window = np.array(cu_window)
    keep = np.concatenate([[True], cu_window[1:] != cu_window[:-1]])  # unique_consecutive
    return np.concatenate(pos, 0), np.concatenate(window_index, 0), cu_window[keep].tolist(), cu_full


def _one_hot(rows, n_pad):
    """P with P[i, rows[i]] = 1, so (P @ x)[i] = x[rows[i]]; padded rows stay zero."""
    p = np.zeros((n_pad, n_pad), dtype=np.float32)
    p[np.arange(len(rows)), rows] = 1.0
    return p.reshape(1, 1, n_pad, n_pad)


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

        pe_w = torch_module.patch_embed.proj.weight.detach().float()  # [C, 3, t, p, p]
        self.w_patch = upload(device, pe_w.reshape(pe_w.shape[0], -1).t().contiguous())
        self.blocks = [TtVisionBlock(device, blk) for blk in torch_module.blocks]
        self.merger = TtPatchMerger(device, torch_module.merger)
        self.compute_cfg = hifi4_config()

    def __call__(self, hidden_states, grid_thw=None, **kwargs):
        s = hidden_states.shape[0]
        s_pad = pad_to_tile(s)
        grid = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        pos, window_index, cu_window, cu_full = _vision_metadata(grid, self.merge, self.window_size, self.patch_size)
        rows = (window_index[:, None] * self.unit + np.arange(self.unit)[None, :]).reshape(-1)

        # rotary tables in window order (HF: rotary_pos_emb(position_ids)[window_index], cat twice)
        freqs = (pos[:, :, None].astype(np.float32) * self.inv_freq[None, None, :]).reshape(s, -1)[rows]
        emb = np.concatenate([freqs, freqs], axis=-1)
        tt_cos, tt_sin = self.blocks[0].attn.rope_tables(np.cos(emb), np.sin(emb), s_pad)
        mask_full = upload(self.device, block_mask(cu_full, s, s_pad))
        mask_win = upload(self.device, block_mask(cu_window, s, s_pad))

        x = ttnn.reshape(hidden_states, (1, 1, s, hidden_states.shape[-1]))
        if s_pad != s:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, s_pad - s), (0, 0)], 0.0)
        x = ttnn.linear(x, self.w_patch, compute_kernel_config=self.compute_cfg, dtype=ttnn.float32)
        x = ttnn.matmul(
            upload(self.device, _one_hot(rows, s_pad), dtype=ttnn.float32),
            x,
            compute_kernel_config=self.compute_cfg,
            dtype=ttnn.float32,
        )

        for i, blk in enumerate(self.blocks):
            x = blk.forward_padded(x, tt_cos, tt_sin, mask_full if i in self.fullatt else mask_win)

        m = s // self.unit
        merged = self.merger(_bf16(x))  # [1, 1, s_pad // unit, out]
        m_rows = merged.shape[-2]
        m_pad = pad_to_tile(m_rows)
        if m_pad != m_rows:
            merged = ttnn.pad(merged, [(0, 0), (0, 0), (0, m_pad - m_rows), (0, 0)], 0.0)
        merged = ttnn.matmul(
            upload(self.device, _one_hot(np.argsort(window_index), m_pad)),
            merged,
            compute_kernel_config=self.compute_cfg,
        )
        merged = ttnn.reshape(ttnn.slice(merged, [0, 0, 0, 0], [1, 1, m, merged.shape[-1]]), (m, merged.shape[-1]))

        if s_pad != s:
            x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, s, self.hidden])
        return ttnn.reshape(x, (s, self.hidden)), merged


def build(device, torch_module=None):
    return TtVisionTransformer(device, torch_module)


def encoder_stack(device, torch_module=None):
    return TtVisionTransformer(device, torch_module)
