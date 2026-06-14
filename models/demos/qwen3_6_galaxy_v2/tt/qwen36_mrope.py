# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-MROPE-1: M-RoPE 3D cos/sin tables for qwen3.6 text decoder.

qwen3.6's `text_config.rope_parameters`:
  mrope_interleaved: True
  mrope_section:     [11, 11, 10]    (sums to 32 = partial_rotary_dim / 2)
  partial_rotary_factor: 0.25
  rope_theta:        10_000_000
  rope_type:         'default'

head_dim = 256, partial_rotary_dim = head_dim * partial_rotary_factor = 64.

The M-RoPE math (mirrors HF transformers:
`models/qwen3_vl/modeling_qwen3_vl.py:Qwen3VLTextRotaryEmbedding.forward`).
HF's `compute_default_rope_parameters` actually uses
`dim = head_dim * partial_rotary_factor = partial_rotary_dim` (qwen3.6: 64)
when partial_rotary_factor < 1.0 — NOT the full head_dim:

  inv_freq = rope_theta^(-arange(0, partial_rotary_dim, 2) / partial_rotary_dim)
             shape [partial_rotary_dim // 2 = 32 for qwen3.6]
  Given position_ids [3, B, S] (T/H/W axes):
    freqs = inv_freq @ position_ids  → [3, B, S, partial_rotary_dim // 2]
    freqs_interleaved = apply_interleaved_mrope(freqs, mrope_section)
                      → [B, S, partial_rotary_dim // 2]
  Where `apply_interleaved_mrope` starts from axis-T's freqs and OVERWRITES
  specific indices with axis-H's freqs (offset=1, stride=3, length=mrope_section[1]*3=33)
  and axis-W's freqs (offset=2, stride=3, length=mrope_section[2]*3=30). For
  qwen3.6 with mrope_section sum=32=partial_rotary_dim/2, the interleaving
  touches all 32 freq indices.

  emb = cat(freqs_interleaved, freqs_interleaved, dim=-1) → [B, S, partial_rotary_dim]
  cos = emb.cos()
  sin = emb.sin()

For TEXT-ONLY input, all 3 axes of position_ids are identical (a 1D ramp
broadcast to 3 dims). The interleaved freqs then equal the 1D freqs, so the
output is mathematically identical to standard 1D RoPE. This is the
backward-compatibility guarantee that lets us drop M-RoPE into the existing
qwen3.6 v2 text decoder without breaking text-only tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def get_rope_index(
    input_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    image_token_id: int = 248056,
    video_token_id: int = 248057,
    vision_start_token_id: int = 248053,
    spatial_merge_size: int = 2,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 3D position_ids `[3, B, S]` for a multimodal input sequence.

    Faithful port of HF's `Qwen3VLModel.get_rope_index` covering the
    text + image + video case. For each batch:
      - Text tokens get (t, h, w) = (k, k, k) where k advances monotonically.
      - Each image/video placeholder range gets replaced by a 3D grid of patch
        coordinates (t, h, w) offset by the running position (`st_idx`).
      - After each vision segment, the running position advances to
        `max(position_id) + 1`.

    qwen3.6 vision_config: spatial_merge_size=2. Each vision frame inserts
    `llm_grid_t * llm_grid_h * llm_grid_w = t * (h/2) * (w/2)` placeholder tokens.

    Video handling mirrors HF: qwen3.6 uses timestamps (not absolute temporal
    position ids), so `video_grid_thw` is `repeat_interleave`'d by its temporal
    dim and the temporal dim is then set to 1 — i.e. each video frame becomes an
    independent `[1, h, w]` vision segment delimited in the token stream by its
    own `<|vision_start|><|video_pad|>...<|vision_end|>` block (the HF processor
    inserts these per temporal group).

    The image-only path is byte-identical to HF (and to the previous
    image-only implementation, verified via `torch.equal`).

    Returns:
      position_ids: `[3, B, S]` torch.long.
      mrope_position_deltas: `[B, 1]` torch.long — for incremental decode,
        the offset between the running token count and the max(position_id+1).
    """
    assert input_ids.ndim == 2, f"input_ids must be [B, S], got {input_ids.shape}"

    # HF splits each video's temporal frames into independent [1, h, w] grids
    # (timestamps encode time, so llm_grid_t is always 1 per segment).
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw = video_grid_thw.clone()
        video_grid_thw[:, 0] = 1

    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)
    attention_mask = attention_mask.to(total_input_ids.device)

    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=torch.long,
        device=input_ids.device,
    )
    mrope_deltas: list[int] = []
    image_index, video_index = 0, 0

    for i in range(total_input_ids.shape[0]):
        ids_row = total_input_ids[i][attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(ids_row == vision_start_token_id).squeeze(1)
        vision_tokens = ids_row[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum())
        video_nums = int((vision_tokens == video_token_id).sum())
        input_tokens = ids_row.tolist()

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                if image_grid_thw is None:
                    raise ValueError(f"image_token_id={image_token_id} found in input but no image_grid_thw provided")
                t, h, w = image_grid_thw[image_index].tolist()
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                if video_grid_thw is None:
                    raise ValueError(f"video_token_id={video_token_id} found in input but no video_grid_thw provided")
                t, h, w = video_grid_thw[video_index].tolist()
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            text_len = ed - st

            st_idx = int(llm_pos_ids_list[-1].max().item()) + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)

            t_idx = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_idx = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_idx = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_idx, h_idx, w_idx]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = int(llm_pos_ids_list[-1].max().item()) + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        mrope_deltas.append(int(llm_positions.max().item()) + 1 - int(total_input_ids[i].shape[0]))

    return position_ids, torch.tensor(mrope_deltas, dtype=torch.long).unsqueeze(1)


def build_mrope_inv_freq(rope_theta: float, partial_rotary_dim: int) -> torch.Tensor:
    """Compute inverse-frequency tensor for M-RoPE.

    HF's `compute_default_rope_parameters` uses `dim = head_dim * partial_rotary_factor`
    (i.e. partial_rotary_dim) when partial_rotary_factor < 1.0. For qwen3.6 with
    head_dim=256 and partial_rotary_factor=0.25, that's dim=64.

    Returns: `[partial_rotary_dim // 2]` float32.
    """
    return 1.0 / (
        rope_theta
        ** (torch.arange(0, partial_rotary_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / partial_rotary_dim)
    )


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: Sequence[int]) -> torch.Tensor:
    """Replicates HF's `Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope`.

    Args:
        freqs: `[3, B, S, partial_rotary_dim // 2]` — per-axis (T, H, W) freqs.
        mrope_section: 3-tuple; the head_dim-half / 3 cyclic split of axes.

    Returns:
        `[B, S, partial_rotary_dim // 2]` with the interleaved per-token freqs.
    """
    freqs_t = freqs[0].clone()  # start from axis T (covers all positions)
    for dim, offset in enumerate((1, 2), start=1):  # H @ offset 1, W @ offset 2
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def build_mrope_tt_tensors(
    position_ids_3d: torch.Tensor,
    *,
    rope_theta: float,
    partial_rotary_dim: int,
    mrope_section: Sequence[int],
    mesh_device,  # ttnn.MeshDevice (avoid circular import)
    attention_scaling: float = 1.0,
):
    """Build M-RoPE cos/sin on CPU and upload to mesh as ttnn tensors.

    Returns:
      (cos_tt, sin_tt) each ttnn tensor of shape `[1, 1, S, partial_rotary_dim]`,
      replicated across the mesh, bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG.

    This is the format the existing qwen3.6 attention `rot_mats` argument
    expects — a drop-in replacement for the 1D-gather-based path when
    running with multimodal 3D position_ids.
    """
    import ttnn  # local import to keep this file pure-torch otherwise

    cos, sin = build_mrope_cos_sin(
        position_ids_3d,
        rope_theta=rope_theta,
        partial_rotary_dim=partial_rotary_dim,
        mrope_section=mrope_section,
        attention_scaling=attention_scaling,
        dtype=torch.float32,
    )
    # cos, sin: [B, S, partial_rotary_dim] → [1, 1, S, partial_rotary_dim]
    assert cos.shape[0] == 1, f"M-RoPE TT upload assumes B=1 for now; got B={cos.shape[0]}"
    cos_4d = cos.unsqueeze(1)
    sin_4d = sin.unsqueeze(1)

    def _upload(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return _upload(cos_4d), _upload(sin_4d)


def build_mrope_cos_sin(
    position_ids: torch.Tensor,
    *,
    rope_theta: float,
    partial_rotary_dim: int,
    mrope_section: Sequence[int],
    attention_scaling: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build M-RoPE cos/sin tables for 3D position_ids.

    HF computes cos/sin over PARTIAL head_dim (`head_dim * partial_rotary_factor`)
    when partial_rotary_factor < 1.0. For qwen3.6 that's 64 dims. The mrope_section
    interleaving touches the first `sum(mrope_section) = partial_rotary_dim // 2`
    indices.

    Args:
        position_ids: `[3, B, S]` (T, H, W axes). For text-only, all 3 axes
            should be identical to recover standard 1D RoPE.
        rope_theta: base period (qwen3.6: 10_000_000).
        partial_rotary_dim: rotary dim for partial RoPE (qwen3.6: 64).
        mrope_section: 3-tuple summing to `partial_rotary_dim // 2`
            (qwen3.6: [11, 11, 10] summing to 32).
        attention_scaling: per HF (default 1.0 for 'default' rope_type).
        dtype: output dtype.

    Returns:
        `(cos, sin)` each `[B, S, partial_rotary_dim]`.
    """
    assert (
        position_ids.ndim == 3 and position_ids.shape[0] == 3
    ), f"position_ids must be [3, B, S]; got {position_ids.shape}"
    assert (
        sum(mrope_section) == partial_rotary_dim // 2
    ), f"sum(mrope_section)={sum(mrope_section)} must equal partial_rotary_dim/2={partial_rotary_dim // 2}"

    inv_freq = build_mrope_inv_freq(rope_theta, partial_rotary_dim)  # [partial_rotary_dim // 2]

    # Per-axis freqs: [3, B, S, head_dim // 2]
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()  # [3, B, 1, S]
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)  # [3, B, S, head_dim/2]

    # Interleave T/H/W into a single per-token freq tensor (first sum(mrope_section) indices only).
    freqs_interleaved = apply_interleaved_mrope(freqs, mrope_section)  # [B, S, head_dim/2]

    # Double and convert to cos/sin
    emb = torch.cat((freqs_interleaved, freqs_interleaved), dim=-1)  # [B, S, head_dim]
    cos = (emb.cos() * attention_scaling).to(dtype=dtype)
    sin = (emb.sin() * attention_scaling).to(dtype=dtype)
    return cos, sin
