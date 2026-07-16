# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Partial-RoPE helpers for the tensor-parallel attention path.

Ported from models/demos/qwen35_27b/tt/rope.py. Only the rotary portion
(rope_dim, e.g. 64 of 256) is rotated; the rest passes through. cos/sin are in
HuggingFace split-halves format. These operate on per-device head shards, so
they are unchanged by TP (each device rotates its local heads).
"""
import itertools

import torch

import ttnn


def build_rope_tables(device, rope_dim, max_seq_len, theta):
    """Precompute replicated cos/sin tables [1, max_seq_len, rope_dim] (HF split-halves)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, rope_dim]
    cos = ttnn.from_torch(
        emb.cos().unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    sin = ttnn.from_torch(
        emb.sin().unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    return cos, sin


def get_vision_position_ids(
    start_position: int,
    grid_thw: list[int, int, int] | torch.Tensor,
    temp_merge_size: int = 1,
    spatial_merge_size: int = 1,
    time_interval: int = 1,
    device: str | torch.device | None = None,
):
    """
    Compute 3D positional indices for vision tokens derived from a single image or video input.

    The positions are generated from the input grid defined by temporal (T), height (H), and
    width (W) dimensions. Temporal and spatial dimensions can be downscaled according to the
    merge sizes used in the vision backbone. The resulting positions are offset by `start_position`.

    Args:
        start_position (`int`):
            Offset added to all computed positional indices.
        grid_thw (`Sequence[int]` or `torch.Tensor` of shape `(3,)`):
            The (T, H, W) grid representing the feature layout of the current image or video after patch embedding.
        temp_merge_size (`int`, *optional*):
            Factor by which the temporal dimension is reduced in the backbone. The temporal grid size is divided
            by this value. Defaults to 1.
        spatial_merge_size (`int`, *optional*):
            Factor by which the spatial dimensions (H and W) are reduced in the backbone. Both H and W are divided
            by this value. Defaults to 1.
        time_interval (`int`, *optional*):
            Spacing factor applied between consecutive temporal position indices.Defaults to 1.
        device (`str` or `torch.device`, *optional*):
            Device on which the resulting tensor is allocated. If `None`, uses the current default device.

    Returns:
        torch.LongTensor of shape (3, sequence_length):
            Positional indices for temporal, height, and width dimensions,
            flattened into sequence form and offset by `start_position`.
    """
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0].item() // temp_merge_size,
        grid_thw[1].item() // spatial_merge_size,
        grid_thw[2].item() // spatial_merge_size,
    )

    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
        llm_grid_w * llm_grid_t
    )
    position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
    position_temporal = position_temporal * time_interval
    vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)

    return vision_position_ids


def get_rope_index(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    spatial_merge_size: int = 2,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Difference from Qwen2VL/Qwen2.5VL's get_rope_index:
    - Since Qwen3.5 use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split too.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
            Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """

    # Separate video grid thw into multiple grids because timestamps are used to seperate videos.
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    spatial_merge_size = spatial_merge_size

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            # text == 0
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            # image == 1, video == 2
            else:
                grid_thw = next(grid_iters[modality_type])
                vision_position_ids = get_vision_position_ids(
                    current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def compute_3d_position_ids(
    input_ids: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
) -> torch.Tensor | None:
    has_multimodal = image_grid_thw is not None or video_grid_thw is not None
    if has_multimodal and mm_token_type_ids is None and input_ids is not None:
        raise ValueError(
            "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
            "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
            "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
        )
    can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

    if can_compute_mrope:
        position_ids, rope_deltas = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
    return position_ids, rope_deltas


def get_rot_mats(inv_freq, position_ids, mrope_section, attention_scaling):
    # In contrast to other models, Qwen3_5 has different position ids for the grids
    # So we expand the inv_freq to shape (3, ...)
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    freqs = apply_interleaved_mrope(freqs, mrope_section)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling

    return cos, sin


def apply_interleaved_mrope(freqs, mrope_section):
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT], preserving frequency continuity.
    args:
        x: (3, bs, seq_len, head_dim // 2)
        mrope_section: (3,)
    returns:
        x_t: (bs, seq_len, head_dim // 2)
    """
    freqs_t = freqs[0]  # just overwrite the first dimension T
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def rot_mats_decode(device, rope_dim, max_seq_len, theta, positions):
    """Return [cos, sin] each [1, B, 1, rope_dim] for the given per-user positions.

    positions: torch.Tensor [B] of int positions. Built on host (small) then
    replicated to the mesh — matches apply_partial_rope_decode's expected layout.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    pos = positions.float()
    freqs = torch.outer(pos, inv_freq)  # [B, rope_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, rope_dim]
    B = positions.shape[0]
    cos = emb.cos().reshape(1, B, 1, rope_dim).to(torch.bfloat16)
    sin = emb.sin().reshape(1, B, 1, rope_dim).to(torch.bfloat16)
    cos_tt = ttnn.from_torch(
        cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    sin_tt = ttnn.from_torch(
        sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    return cos_tt, sin_tt


def rot_mats_prefill(device, rope_dim, seq_len, theta, position_ids=None, mrope_section=None, attention_scaling=1.0):
    """Return [cos, sin] each [1, 1, seq_len, rope_dim].

    position_ids: 3D M-RoPE indices [3, bs, seq_len] (or 2D [bs, seq_len], expanded inside
    get_rot_mats). When None, defaults to text positions arange(seq_len) — the (t==h==w) case
    where interleaved-mrope collapses to ordinary 1D RoPE, so the result is independent of
    mrope_section and identical to the pre-M-RoPE behaviour.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    if position_ids is None:
        position_ids = torch.arange(seq_len).view(1, -1)
    if mrope_section is None:
        # Any split works for text (t==h==w); use an even-ish T/H/W partition of rope_dim//2.
        half = rope_dim // 2
        base = half // 3
        mrope_section = [base, base, half - 2 * base]
    cos, sin = get_rot_mats(inv_freq, position_ids, mrope_section, attention_scaling)
    cos = ttnn.from_torch(
        cos.reshape(1, 1, seq_len, rope_dim).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    sin = ttnn.from_torch(
        sin.reshape(1, 1, seq_len, rope_dim).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    return cos, sin


def apply_partial_rope_decode(x, cos_tt, sin_tt, n_heads, batch_size, rope_dim):
    """x: [1, B, n_heads, HD]; cos/sin: [1, B, 1, rope_dim]; rotates first rope_dim dims."""
    hd = x.shape[-1]
    B = batch_size
    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, B, n_heads, rope_dim))
    x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, B, n_heads, hd))
    r1 = ttnn.slice(x_rope, (0, 0, 0, 0), (1, B, n_heads, rope_dim // 2))
    r2 = ttnn.slice(x_rope, (0, 0, 0, rope_dim // 2), (1, B, n_heads, rope_dim))
    x_rot = ttnn.concat([ttnn.neg(r2), r1], dim=-1)
    ttnn.deallocate(r1)
    ttnn.deallocate(r2)
    roped = ttnn.add(ttnn.multiply(x_rope, cos_tt), ttnn.multiply(x_rot, sin_tt))
    ttnn.deallocate(x_rope)
    ttnn.deallocate(x_rot)
    roped = ttnn.to_memory_config(roped, ttnn.DRAM_MEMORY_CONFIG)
    x_pass = ttnn.to_memory_config(x_pass, ttnn.DRAM_MEMORY_CONFIG)
    result = ttnn.concat([roped, x_pass], dim=-1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return result


def apply_partial_rope_prefill(x, cos_tt, sin_tt, n_heads, rope_dim):
    """x: [1, n_heads, seq_len, HD]; cos/sin: [1, 1, seq_len, rope_dim]."""
    hd = x.shape[-1]
    seq_len = x.shape[-2]
    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, n_heads, seq_len, rope_dim))
    x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, n_heads, seq_len, hd))
    r1 = ttnn.slice(x_rope, (0, 0, 0, 0), (1, n_heads, seq_len, rope_dim // 2))
    r2 = ttnn.slice(x_rope, (0, 0, 0, rope_dim // 2), (1, n_heads, seq_len, rope_dim))
    x_rot = ttnn.concat([ttnn.neg(r2), r1], dim=-1)
    ttnn.deallocate(r1)
    ttnn.deallocate(r2)
    roped = ttnn.add(ttnn.multiply(x_rope, cos_tt), ttnn.multiply(x_rot, sin_tt))
    ttnn.deallocate(x_rope)
    ttnn.deallocate(x_rot)
    roped = ttnn.to_memory_config(roped, ttnn.DRAM_MEMORY_CONFIG)
    x_pass = ttnn.to_memory_config(x_pass, ttnn.DRAM_MEMORY_CONFIG)
    result = ttnn.concat([roped, x_pass], dim=-1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return result
