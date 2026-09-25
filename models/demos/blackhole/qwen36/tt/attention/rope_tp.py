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
from models.common.utility_functions import is_blackhole

ROPE_PERM_VERSION = "v1"  # bump when the channel permutation changes; it is part of the weight-cache name


def rope_full_head_dim(args):
    """full_head_dim so cos/sin width matches TPAttention; None keeps HF-width partial rope."""
    return args.head_dim if getattr(args, "rope_permuted_enabled", False) else None


def rope_channel_perm(head_dim, rope_dim):
    """Index list P: permuted channel i holds HF channel P[i], so full-width rotate-half matches partial rope."""
    assert head_dim % 2 == 0 and rope_dim % 2 == 0, (head_dim, rope_dim)
    assert 0 < rope_dim <= head_dim, (head_dim, rope_dim)
    half, rh = head_dim // 2, rope_dim // 2
    perm = [None] * head_dim
    for j in range(rh):
        perm[j] = j
        perm[half + j] = rh + j
    # Pass-through slots land in matched (p, p+half) pairs (cos=1/sin=0).
    free = [i for i in range(rh, half)] + [i for i in range(half + rh, head_dim)]
    for slot, src in zip(free, range(rope_dim, head_dim)):
        perm[slot] = src
    assert sorted(perm) == list(range(head_dim)), "rope_channel_perm is not a permutation"
    return perm


def _rope_perm_row_index(device, out_rows, head_dim, rope_dim, stride):
    """Row-index tensor for the gather; identity segment when stride > head_dim (q_proj gate half)."""
    half, rh = head_dim // 2, rope_dim // 2

    def seg(a, b):
        return ttnn.arange(a, b, 1, device=device, dtype=ttnn.uint32)

    segments = []
    for base in range(0, out_rows, stride):
        segments += [
            seg(base, base + rh),
            seg(base + rope_dim, base + rope_dim + (half - rh)),
            seg(base + rh, base + rope_dim),
            seg(base + rope_dim + (half - rh), base + head_dim),
        ]
        if stride > head_dim:
            segments.append(seg(base + head_dim, base + stride))
    idx = segments[0] if len(segments) == 1 else ttnn.concat(segments, dim=0)
    return ttnn.reshape(idx, (1, out_rows))


def permute_rope_channels(w, head_dim, rope_dim, device, stride=None):
    """Permute head_dim output channels via ttnn.embedding. Use host reshape for 1D: ttnn.reshape corrupts 1D<->2D."""
    is_1d = w.dim() == 1
    if is_1d:
        assert w.shape[0] == head_dim, w.shape
        out_rows, stride = head_dim, head_dim
        w = w.reshape(head_dim, 1)
    else:
        stride = stride or head_dim
        out_rows = w.shape[0]
        assert out_rows % stride == 0, (w.shape, stride)

    idx = _rope_perm_row_index(device, out_rows, head_dim, rope_dim, stride)
    # Replicate the gather across the mesh, then read back one shard.
    table = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    gathered = ttnn.embedding(idx, table)
    shards = ttnn.get_device_tensors(gathered)
    out = ttnn.to_torch(ttnn.from_device(shards[0] if shards else gathered)).reshape(w.shape)
    ttnn.deallocate(table)
    ttnn.deallocate(gathered)
    return out.reshape(head_dim) if is_1d else out


def to_full_width_rot_mats(cos_r, sin_r, head_dim, rope_dim, device):
    """Widen HF cos/sin to head_dim in permuted order; pass-through slots are cos=1/sin=0."""
    if head_dim == rope_dim:
        return cos_r, sin_r
    half, rh = head_dim // 2, rope_dim // 2
    lead = list(cos_r.shape[:-1])
    start = [0] * (len(lead) + 1)

    def widen(src, fill_value):
        t = ttnn.from_torch(
            src,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        # Permuted layout: rope halves at 0 and head_dim/2, tail fills the rest.
        pieces = [
            ttnn.slice(t, start, lead + [rh]),
            ttnn.full(lead + [half - rh], fill_value, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.slice(t, start[:-1] + [rh], lead + [rope_dim]),
            ttnn.full(
                lead + [head_dim - half - rh],
                fill_value,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            ),
        ]
        out = ttnn.concat(pieces, dim=-1)
        shards = ttnn.get_device_tensors(out)
        res = ttnn.to_torch(ttnn.from_device(shards[0] if shards else out))
        ttnn.deallocate(t)
        ttnn.deallocate(out)
        return res

    return widen(cos_r, 1.0), widen(sin_r, 0.0)


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


_ROPE_DEV_TABLES = {}
# Minimum row count the device RoPE table is built at; unrelated to args.dim.
_ROPE_TABLE_MIN_ROWS = 4096


def _rope_dev_tables(device, rope_dim, n_rows, theta, full_head_dim=None):
    """ROW_MAJOR device cos/sin, grown on demand. full_head_dim widens via to_full_width_rot_mats."""
    key = (id(device), int(rope_dim), float(theta), int(full_head_dim or 0))
    ent = _ROPE_DEV_TABLES.get(key)
    if ent is not None and ent["rows"] >= n_rows:
        return ent["cos"], ent["sin"]
    rows = max(int(n_rows), 2 * ent["rows"] if ent else 0, _ROPE_TABLE_MIN_ROWS)
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    freqs = torch.outer(torch.arange(rows).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_t, sin_t = emb.cos(), emb.sin()
    if full_head_dim is not None and full_head_dim != rope_dim:
        cos_t, sin_t = to_full_width_rot_mats(cos_t, sin_t, full_head_dim, rope_dim, device)

    def _mk(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    ent = {"rows": rows, "cos": _mk(cos_t), "sin": _mk(sin_t)}
    _ROPE_DEV_TABLES[key] = ent
    return ent["cos"], ent["sin"]


def rot_mats_decode(device, rope_dim, max_seq_len, theta, positions, full_head_dim=None):
    """[cos, sin] each [1, B, 1, W] for per-user positions; a sequence of ints stays torch-free."""
    W = full_head_dim or rope_dim
    pos_i = [int(p) for p in (positions.reshape(-1).tolist() if isinstance(positions, torch.Tensor) else positions)]
    if is_blackhole():
        # Positions are ints; rebuilding the float row from the int list is bit-identical.
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        pos = torch.tensor(pos_i, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        B = len(pos_i)
        cos, sin = emb.cos(), emb.sin()
        if W != rope_dim:
            cos, sin = to_full_width_rot_mats(cos, sin, W, rope_dim, device)
        cos = cos.reshape(1, B, 1, W).to(torch.bfloat16)
        sin = sin.reshape(1, B, 1, W).to(torch.bfloat16)
        cos_tt = ttnn.from_torch(
            cos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        sin_tt = ttnn.from_torch(
            sin,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        return cos_tt, sin_tt
    assert min(pos_i) >= 0, f"negative rope position {min(pos_i)}"
    tbl_cos, tbl_sin = _rope_dev_tables(device, rope_dim, max(pos_i) + 1, theta, full_head_dim=full_head_dim)
    B = len(pos_i)
    idx = ttnn.Tensor(pos_i, [1, B], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    def _gather(tbl):
        r = ttnn.embedding(idx, tbl)
        r = ttnn.reshape(r, (1, B, 1, W))  # metadata-only while ROW_MAJOR
        return ttnn.to_layout(r, ttnn.TILE_LAYOUT)

    cos_tt, sin_tt = _gather(tbl_cos), _gather(tbl_sin)
    ttnn.deallocate(idx)
    return cos_tt, sin_tt


def rot_mats_prefill(
    device,
    rope_dim,
    seq_len,
    theta,
    position_ids=None,
    mrope_section=None,
    attention_scaling=1.0,
    full_head_dim=None,
):
    """Return [cos, sin] each [1, 1, seq_len, W].

    position_ids: 3D M-RoPE indices [3, bs, seq_len] (or 2D [bs, seq_len], expanded inside
    get_rot_mats). When None, defaults to text positions arange(seq_len) — the (t==h==w) case
    where interleaved-mrope collapses to ordinary 1D RoPE, so the result is independent of
    mrope_section and identical to the pre-M-RoPE behaviour.
    """
    W = full_head_dim or rope_dim
    if position_ids is None and not is_blackhole():
        # Text positions are arange(seq_len), a contiguous table prefix; slice on device.
        tbl_cos, tbl_sin = _rope_dev_tables(device, rope_dim, seq_len, theta, full_head_dim=full_head_dim)

        def _slice(tbl):
            r = ttnn.slice(tbl, [0, 0], [seq_len, W])  # ROW_MAJOR: no tile alignment
            r = ttnn.reshape(r, (1, 1, seq_len, W))  # metadata-only while ROW_MAJOR
            return ttnn.to_layout(r, ttnn.TILE_LAYOUT)

        return _slice(tbl_cos), _slice(tbl_sin)

    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    if position_ids is None:
        # Blackhole text-only: the original host path expects explicit positions.
        position_ids = torch.arange(seq_len).view(1, -1)
    if mrope_section is None:
        # Any split works for text (t==h==w); use an even-ish T/H/W partition of rope_dim//2.
        half = rope_dim // 2
        base = half // 3
        mrope_section = [base, base, half - 2 * base]
    cos, sin = get_rot_mats(inv_freq, position_ids, mrope_section, attention_scaling)
    if W != rope_dim:
        cos, sin = to_full_width_rot_mats(cos.reshape(-1, rope_dim), sin.reshape(-1, rope_dim), W, rope_dim, device)
    cos = ttnn.from_torch(
        cos.reshape(1, 1, seq_len, W).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    sin = ttnn.from_torch(
        sin.reshape(1, 1, seq_len, W).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    return cos, sin


def shard_rot_mats_decode(cos_tt, sin_tt, shard_cfg):
    """Shard decode cos/sin onto the same grid as the tensor being rotated."""
    cos_tt = ttnn.unsqueeze_to_4D(cos_tt)
    sin_tt = ttnn.unsqueeze_to_4D(sin_tt)
    return (
        ttnn.interleaved_to_sharded(cos_tt, shard_cfg),
        ttnn.interleaved_to_sharded(sin_tt, shard_cfg),
    )


def apply_rope_full_decode(x_sh, cos_sh, sin_sh, memory_config=None):
    """Permuted decode RoPE is one op; the output shard spec is copied from the input."""
    return ttnn.experimental.rotary_embedding_hf(
        x_sh, cos_sh, sin_sh, is_decode_mode=True, memory_config=memory_config or x_sh.memory_config()
    )


def apply_rope_full_prefill(x, cos_tt, sin_tt, memory_config=None):
    """Permuted prefill RoPE is one op."""
    return ttnn.experimental.rotary_embedding_hf(
        x, cos_tt, sin_tt, is_decode_mode=False, memory_config=memory_config or ttnn.L1_MEMORY_CONFIG
    )


def apply_partial_rope_decode(x, cos_tt, sin_tt, n_heads, batch_size, rope_dim):
    """x: [1, B, n_heads, HD]; cos/sin: [1, B, 1, rope_dim]; rotates first rope_dim dims.

    Fused HF-convention rotate-half via ttnn.experimental.rotary_embedding_hf. The op's native
    decode mode (is_decode_mode=True) hard-requires HEIGHT_SHARDED input + cos/sin, but qwen36's
    decode attention runs interleaved (q/k are sharded_to_interleaved right after head-split). To
    avoid the reshards that sharding would add, transpose the interleaved tensor to a prefill-shaped
    [1, n_heads, B, rope_dim] (batch plays the seq role) and use the interleaved-friendly prefill
    mode (is_decode_mode=False), then transpose back. Partial: only the first rope_dim is rotated;
    the tail passes through.
    """
    hd = x.shape[-1]
    B = batch_size
    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, B, n_heads, rope_dim))
    x_rope_t = ttnn.transpose(x_rope, 1, 2)  # [1, n_heads, B, rope_dim]
    ttnn.deallocate(x_rope)
    # decode cos/sin [1, B, 1, rope_dim] -> prefill [1, 1, B, rope_dim] (broadcast over heads)
    cos_p = ttnn.reshape(cos_tt, (1, 1, B, rope_dim))
    sin_p = ttnn.reshape(sin_tt, (1, 1, B, rope_dim))
    roped_t = ttnn.experimental.rotary_embedding_hf(
        x_rope_t, cos_p, sin_p, is_decode_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(x_rope_t)
    roped = ttnn.to_memory_config(ttnn.transpose(roped_t, 1, 2), ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(roped_t)
    if rope_dim == hd:
        return roped
    x_pass = ttnn.to_memory_config(ttnn.slice(x, (0, 0, 0, rope_dim), (1, B, n_heads, hd)), ttnn.DRAM_MEMORY_CONFIG)
    result = ttnn.concat([roped, x_pass], dim=-1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return result


def apply_partial_rope_prefill(x, cos_tt, sin_tt, n_heads, rope_dim):
    """x: [1, n_heads, seq_len, HD]; cos/sin: [1, 1, seq_len, rope_dim].

    Fused HF-convention rotate-half via ttnn.experimental.rotary_embedding_hf (replaces manual
    slice/neg/concat/mul/add). Partial: only the first rope_dim is rotated; tail passes through.
    """
    # Prefill-only: roped q/k feed SDPA directly; L1 is safe at S=2048 (SDPA CBs fit; verified).
    # chunked SDPA in forward_prefill_paged still clashes with this L1 output.
    _L1 = ttnn.L1_MEMORY_CONFIG
    hd = x.shape[-1]
    seq_len = x.shape[-2]
    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, n_heads, seq_len, rope_dim), memory_config=_L1)
    roped = ttnn.experimental.rotary_embedding_hf(x_rope, cos_tt, sin_tt, is_decode_mode=False, memory_config=_L1)
    ttnn.deallocate(x_rope)
    if rope_dim == hd:
        return roped
    x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, n_heads, seq_len, hd), memory_config=_L1)
    result = ttnn.concat([roped, x_pass], dim=-1, memory_config=_L1)
    ttnn.deallocate(roped)
    ttnn.deallocate(x_pass)
    return result
