# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading and preprocessing for Swin-L backbone.
Loads from mmdet checkpoint and converts to TTNN parameter dict.

Works with any mmdet checkpoint that contains a Swin-L backbone
(e.g., DINO-5scale, ATSS-DyHead, etc.).

mmdet checkpoint key structure (backbone only):
  backbone.patch_embed.projection.{weight,bias}
  backbone.patch_embed.norm.{weight,bias}
  backbone.stages.{s}.blocks.{b}.norm1.{weight,bias}
  backbone.stages.{s}.blocks.{b}.norm2.{weight,bias}
  backbone.stages.{s}.blocks.{b}.attn.w_msa.qkv.{weight,bias}
  backbone.stages.{s}.blocks.{b}.attn.w_msa.proj.{weight,bias}
  backbone.stages.{s}.blocks.{b}.attn.w_msa.relative_position_bias_table  [529, heads]
  backbone.stages.{s}.blocks.{b}.attn.w_msa.relative_position_index       [144, 144]
  backbone.stages.{s}.blocks.{b}.ffn.layers.0.0.{weight,bias}   (MLP fc1)
  backbone.stages.{s}.blocks.{b}.ffn.layers.1.{weight,bias}     (MLP fc2)
  backbone.stages.{s}.downsample.norm.{weight,bias}
  backbone.stages.{s}.downsample.reduction.weight
  backbone.norm{s}.{weight,bias}                                  (per-stage output norms)
"""

from typing import Dict, List, Optional, Tuple

import torch
import ttnn


def _get(sd: dict, key: str) -> torch.Tensor:
    """Get a key from state_dict, raising a clear error if missing."""
    if key not in sd:
        raise KeyError(f"Missing key in checkpoint: {key}")
    return sd[key]


def _to_ttnn_linear_weight(weight: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Transpose [out, in] -> [in, out] and convert to TTNN tile layout."""
    return ttnn.from_torch(weight.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _to_ttnn_linear_bias(bias: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Reshape [out] -> [1, out] and convert to TTNN tile layout."""
    return ttnn.from_torch(bias.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _to_ttnn_norm_param(param: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Reshape [C] -> [1, C] and convert to TTNN tile layout."""
    return ttnn.from_torch(param.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _compute_relative_position_bias(
    bias_table: torch.Tensor,
    bias_index: torch.Tensor,
    window_size: int,
) -> ttnn.Tensor:
    """
    Precompute relative position bias from table + index.
    Returns: ttnn tensor [1, num_heads, win*win, win*win] in TILE layout.
    """
    N = window_size * window_size
    rpb = bias_table[bias_index.view(-1).long()]  # [N*N, heads]
    rpb = rpb.view(N, N, -1).permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, heads, N, N]
    return ttnn.from_torch(rpb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _unfold_to_concat_perm(C: int) -> torch.Tensor:
    """
    Build permutation to reorder PatchMerging weights from mmdet's nn.Unfold
    channel ordering to our stride-slice concat ordering.

    mmdet Unfold order (per spatial pos): [ch0_tl, ch0_tr, ch0_bl, ch0_br, ch1_tl, ...]
    Our concat order: [all_tl (0..C-1), all_bl (C..2C-1), all_tr (2C..3C-1), all_br (3C..4C-1)]

    Returns a 4C index tensor such that: weight_reordered[:, j] = weight[:, perm[j]]
    """
    perm = torch.zeros(4 * C, dtype=torch.long)
    for j in range(C):
        perm[j] = j * 4 + 0  # tl -> unfold index c*4+0
        perm[C + j] = j * 4 + 2  # bl -> unfold index c*4+2
        perm[2 * C + j] = j * 4 + 1  # tr -> unfold index c*4+1
        perm[3 * C + j] = j * 4 + 3  # br -> unfold index c*4+3
    return perm


def load_backbone_weights(
    checkpoint_path: str,
    device: ttnn.Device,
    embed_dim: int = 192,
    depths: Tuple[int, ...] = (2, 2, 18, 2),
    num_heads: Tuple[int, ...] = (6, 12, 24, 48),
    window_size: int = 12,
    out_indices: Tuple[int, ...] = (0, 1, 2, 3),
) -> dict:
    """
    Load backbone weights from mmdet checkpoint and return TTNN parameter dict.

    Works with any mmdet checkpoint containing a Swin-L backbone.
    Only loads per-stage output norms for stages in `out_indices`
    (e.g., ATSS checkpoint won't have norm0 when out_indices=(1,2,3)).

    Returns nested dict matching the structure expected by TtSwinLBackbone:
      {
        "patch_embed": {"projection": {"weight", "bias"}, "norm": {"weight", "bias"}},
        "stages": {s: {"blocks": {b: {...}}, "downsample": {...}}},
        "norm{s}": {"weight", "bias"} for s in out_indices,
      }
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    params: Dict = {}

    # -- Patch embedding --
    params["patch_embed"] = {
        "projection": {
            "weight": ttnn.from_torch(_get(sd, "backbone.patch_embed.projection.weight"), dtype=ttnn.bfloat16),
            "bias": ttnn.from_torch(
                _get(sd, "backbone.patch_embed.projection.bias").reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
            ),
        },
        "norm": {
            "weight": _to_ttnn_norm_param(_get(sd, "backbone.patch_embed.norm.weight")),
            "bias": _to_ttnn_norm_param(_get(sd, "backbone.patch_embed.norm.bias")),
        },
    }

    # -- Stages --
    params["stages"] = {}
    for s in range(4):
        dim = embed_dim * (2**s)
        heads = num_heads[s]
        params["stages"][s] = {"blocks": {}}

        for b in range(depths[s]):
            prefix = f"backbone.stages.{s}.blocks.{b}"

            # Relative position bias (precomputed on CPU)
            rpb_table = _get(sd, f"{prefix}.attn.w_msa.relative_position_bias_table")
            rpb_index = _get(sd, f"{prefix}.attn.w_msa.relative_position_index")
            rpb = _compute_relative_position_bias(rpb_table, rpb_index, window_size)

            block_params = {
                "norm1": {
                    "weight": _to_ttnn_norm_param(_get(sd, f"{prefix}.norm1.weight")),
                    "bias": _to_ttnn_norm_param(_get(sd, f"{prefix}.norm1.bias")),
                },
                "norm2": {
                    "weight": _to_ttnn_norm_param(_get(sd, f"{prefix}.norm2.weight")),
                    "bias": _to_ttnn_norm_param(_get(sd, f"{prefix}.norm2.bias")),
                },
                "attn": {
                    "qkv": {
                        "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.attn.w_msa.qkv.weight")),
                        "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.attn.w_msa.qkv.bias")),
                    },
                    "proj": {
                        "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.attn.w_msa.proj.weight")),
                        "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.attn.w_msa.proj.bias")),
                    },
                    "relative_position_bias": rpb,
                },
                "mlp": {
                    "fc1": {
                        "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.ffn.layers.0.0.weight")),
                        "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.ffn.layers.0.0.bias")),
                    },
                    "fc2": {
                        "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.ffn.layers.1.weight")),
                        "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.ffn.layers.1.bias")),
                    },
                },
            }
            params["stages"][s]["blocks"][b] = block_params

        # Downsample (stages 0, 1, 2 only)
        # mmdet uses nn.Unfold which interleaves channels; our TTNN uses stride-slice concat.
        # Reorder norm and reduction weights from unfold order -> concat order.
        if s < 3:
            ds_prefix = f"backbone.stages.{s}.downsample"
            perm = _unfold_to_concat_perm(dim)

            # Reorder norm: norm was over 4C unfold-ordered features
            norm_w = _get(sd, f"{ds_prefix}.norm.weight")  # [4C]
            norm_b = _get(sd, f"{ds_prefix}.norm.bias")  # [4C]
            norm_w = norm_w[perm]
            norm_b = norm_b[perm]

            # Reorder reduction weight columns: W[out, in] -> W[:, perm]
            red_w = _get(sd, f"{ds_prefix}.reduction.weight")  # [2C, 4C]
            red_w = red_w[:, perm]

            params["stages"][s]["downsample"] = {
                "norm": {
                    "weight": _to_ttnn_norm_param(norm_w),
                    "bias": _to_ttnn_norm_param(norm_b),
                },
                "reduction": {
                    "weight": _to_ttnn_linear_weight(red_w),
                },
            }

    # -- Per-stage output norms (only for stages in out_indices) --
    for s in out_indices:
        params[f"norm{s}"] = {
            "weight": _to_ttnn_norm_param(_get(sd, f"backbone.norm{s}.weight")),
            "bias": _to_ttnn_norm_param(_get(sd, f"backbone.norm{s}.bias")),
        }

    # Move all parameters to device
    params = _to_device_recursive(params, device)
    return params


def _to_device_recursive(d, device):
    """Recursively move all ttnn.Tensor values in a nested dict to device."""
    if isinstance(d, ttnn.Tensor):
        return ttnn.to_device(d, device)
    if isinstance(d, dict):
        return {k: _to_device_recursive(v, device) for k, v in d.items()}
    return d


def compute_attn_masks(
    input_h: int,
    input_w: int,
    patch_size: int,
    window_size: int,
    device: ttnn.Device,
) -> List[Optional[ttnn.Tensor]]:
    """
    Precompute attention masks for shifted window attention at each of the 4 stages.
    Returns list of 4 ttnn tensors (or None if window >= feature size).
    """
    feat_h = input_h // patch_size
    feat_w = input_w // patch_size
    ws = window_size
    shift = ws // 2
    masks = []

    for s in range(4):
        pad_b = (ws - feat_h % ws) % ws
        pad_r = (ws - feat_w % ws) % ws
        pad_H = feat_h + pad_b
        pad_W = feat_w + pad_r

        if ws >= pad_H or ws >= pad_W:
            masks.append(None)
        else:
            num_windows = (pad_H // ws) * (pad_W // ws)
            attn_mask = torch.zeros((pad_H, pad_W))
            h_slices = ((0, -ws), (-ws, -shift), (-shift, None))
            w_slices = ((0, -ws), (-ws, -shift), (-shift, None))
            count = 0
            for hs in h_slices:
                for ws_ in w_slices:
                    attn_mask[hs[0] : hs[1], ws_[0] : ws_[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // ws, ws, pad_W // ws, ws)
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, ws * ws)
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(0)  # [1, 1, nW, ws*ws, ws*ws]
            masks.append(ttnn.from_torch(attn_mask, device=device, layout=ttnn.TILE_LAYOUT))

        # Next stage: spatial dims halve
        feat_h = feat_h // 2
        feat_w = feat_w // 2

    return masks
