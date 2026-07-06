# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_vision_transformer_pretrained_model`
(transformers ``Qwen2_5_VisionTransformerPretrainedModel``) — the LongCat-Image
text encoder's Qwen2.5-VL vision tower, submodule ``text_encoder.model.visual``.

forward(hidden_states[num_patches, C*T*P*P], grid_thw) -> last_hidden_state::

    position_ids   = get_vision_position_ids(grid_thw, merge)      # host index math
    cu_seqlens     = get_vision_cu_seqlens(grid_thw)               # full-attn boundaries
    window_index,
    cu_window      = get_vision_window_index(grid_thw, ...)        # windowed boundaries
    h   = patch_embed(hidden_states)                              # Conv3d == linear
    h   = h[window_index]  (grouped by spatial_merge_unit)        # window reorder
    cos,sin = rotary(position_ids)[window_index] |> duplicate     # 2-D M-RoPE, reordered
    for i, blk in enumerate(blocks):
        cu = cu_seqlens if i in fullatt_block_indexes else cu_window
        h = blk(h, cu, (cos,sin))                                 # windowed / full attn
    return last_hidden_state = h                                  # (window order)

The harness compares ``last_hidden_state`` (``_normalize_out`` -> ``.last_hidden_state``),
which is exactly ``patch_embed -> window-reorder -> 32 blocks`` — the merger only feeds
``pooler_output`` (not compared here; it graduates separately as
`qwen2_v_l_patch_merger`), so it is omitted.

Key equivalences that keep this native + exact:
  * ``patch_embed`` is a Conv3d whose kernel == stride == the full patch extent, so it
    is a per-patch linear: flatten the conv weight to [embed, C*T*P*P] and matmul the
    already-flattened patch pixels (no bias).
  * The reference reorders patches AFTER patch_embed; patch_embed is row-wise, so
    reorder-then-embed == embed-then-reorder. We reorder the INPUT rows on host (pure
    index bookkeeping, like the rope tables / masks) — NO activation round-trips.
  * Windowed attention (contiguous windows in reordered order) is realised with a
    block-diagonal additive mask built from ``cu_window`` — numerically identical to the
    reference's per-chunk split. Full-attention blocks use an all-zero mask.

The 32 blocks reuse the graduated `qwen2_v_l_vision_block` native port verbatim
(``_VisionBlock._apply``): fp32 weights + fp32 activations, HiFi4 with fp32_dest_acc_en
+ packer_l1_acc, manual fp32 softmax.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.vision.generative.longcat_image._stubs.qwen2_v_l_vision_block import F32, _VisionBlock

DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = ttnn.TILE_LAYOUT


class _VisionModel:
    def __init__(self, device, visual):
        self.device = device
        v = visual.eval() if hasattr(visual, "eval") else visual
        cfg = getattr(v, "config", None)

        def _get(name, default=None):
            val = getattr(v, name, None)
            if val is None and cfg is not None:
                val = getattr(cfg, name, None)
            return default if val is None else val

        self.spatial_merge_size = int(_get("spatial_merge_size", 2))
        self.patch_size = int(_get("patch_size", 14))
        self.window_size = int(_get("window_size", 112))
        self.merge_unit = self.spatial_merge_size**2
        self.fullatt = set(int(i) for i in (_get("fullatt_block_indexes", []) or []))

        # rotary: parameter-free 2-D vision M-RoPE table (fixed inv_freq buffer).
        self.inv_freq = v.rotary_pos_emb.inv_freq.detach().float().reshape(-1)  # [head_dim/4]

        # patch_embed Conv3d (kernel==stride==full patch) -> per-patch linear weight.
        pw = v.patch_embed.proj.weight.detach()  # [embed, C, T, P, P]
        self.embed_dim = int(pw.shape[0])
        self._patch_w = ttnn.from_torch(
            pw.reshape(self.embed_dim, -1).to(torch.float32), dtype=F32, layout=TILE, device=device, memory_config=DRAM
        )

        # 32 native vision blocks (reuse the graduated per-block port).
        self.blocks = [_VisionBlock(device, blk) for blk in v.blocks]
        b0 = self.blocks[0]
        self.num_heads = b0.num_heads
        self.head_dim = b0.head_dim
        self._compute = None

    # ── helpers ──────────────────────────────────────────────────────────
    def _ck(self):
        if self._compute is None:
            self._compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        return self._compute

    def _to_ttnn(self, t):
        return ttnn.from_torch(t.to(torch.float32), dtype=F32, layout=TILE, device=self.device, memory_config=DRAM)

    def _reorder_groups(self, t2d, window_index):
        # t2d [seq, feat]; reorder groups of `merge_unit` consecutive rows by
        # window_index (a permutation of the seq//merge_unit groups). Host-side
        # index bookkeeping — identical to the reference's [window_index] gather.
        seq, feat = int(t2d.shape[0]), int(t2d.shape[1])
        g = t2d.reshape(seq // self.merge_unit, self.merge_unit, feat)
        g = g[window_index]
        return g.reshape(seq, feat)

    def _chunk_mask(self, cu, S):
        # Additive 0/-1e9 block-diagonal mask [1,1,S,S] from cumulative boundaries.
        add = torch.full((S, S), -1e9, dtype=torch.float32)
        cu = [int(x) for x in torch.as_tensor(cu).reshape(-1).tolist()]
        for i in range(len(cu) - 1):
            a, b = cu[i], cu[i + 1]
            add[a:b, a:b] = 0.0
        return self._to_ttnn(add.reshape(1, 1, S, S))

    # ── forward ──────────────────────────────────────────────────────────
    def __call__(self, _primary=None, hidden_states=None, grid_thw=None, **_ignored):
        if hidden_states is None or grid_thw is None:
            raise ValueError(
                "qwen2_vision_transformer_pretrained_model stub requires " "`hidden_states` and `grid_thw`"
            )
        from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids, get_vision_window_index

        gt = grid_thw if isinstance(grid_thw, torch.Tensor) else torch.as_tensor(grid_thw)
        gt = gt.reshape(-1, 3).to(torch.long)
        S = int((gt[:, 0] * gt[:, 1] * gt[:, 2]).sum().item())

        # ── host index bookkeeping (parameter-free) ──
        position_ids = get_vision_position_ids(gt, self.spatial_merge_size)  # [S,2]
        cu_seqlens = get_vision_cu_seqlens(gt)  # full boundaries
        window_index, cu_window = get_vision_window_index(
            gt, spatial_merge_size=self.spatial_merge_size, window_size=self.window_size, patch_size=self.patch_size
        )

        # 2-D vision rotary: rpe = (pos[...,None] * inv_freq).flatten -> reorder -> dup.
        rpe = (position_ids.unsqueeze(-1).float() * self.inv_freq).flatten(1)  # [S, head_dim/2]
        rpe = self._reorder_groups(rpe, window_index)
        emb = torch.cat([rpe, rpe], dim=-1)  # [S, head_dim]
        cos = self._to_ttnn(emb.cos().reshape(1, 1, S, self.head_dim))
        sin = self._to_ttnn(emb.sin().reshape(1, 1, S, self.head_dim))

        full_mask = self._chunk_mask(cu_seqlens, S)
        win_mask = self._chunk_mask(cu_window, S)

        # ── patch embed (reorder input rows first; patch_embed is row-wise) ──
        h = self._reorder_groups(hidden_states.reshape(S, -1), window_index)  # [S, C*T*P*P]
        x = self._to_ttnn(h)
        x = ttnn.linear(
            x, self._patch_w, transpose_b=True, memory_config=DRAM, dtype=F32, compute_kernel_config=self._ck()
        )  # [S, embed]
        x = ttnn.reshape(x, [1, S, self.embed_dim])

        # ── 32 blocks (windowed / full attention) ──
        for i, blk in enumerate(self.blocks):
            mask = full_mask if i in self.fullatt else win_mask
            x = blk._apply(x, cos, sin, mask, S)

        return ttnn.reshape(x, [S, self.embed_dim])  # last_hidden_state (window order)


def build(device, torch_module):
    """PCC-harness entry point: native TTNN Qwen2.5-VL vision tower (last_hidden_state)."""
    return _VisionModel(device, torch_module)
