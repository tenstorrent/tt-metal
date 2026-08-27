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

ROPE_PERM_VERSION = "v1"
"""Cache-busting tag for permuted-RoPE weight construction (rope_channel_perm /
permute_rope_channels). attention/tp.py::load_attention_weights_tp folds this into every ".rp"
cache file name alongside a content hash of the pre-permutation source weight, so a cached
tensorbin is used ONLY if both the source weights AND this version are unchanged from the run
that produced it. Bump this string (any change is enough, e.g. "v2") whenever the permutation's
semantics change -- rope_channel_perm's index derivation, permute_rope_channels' gather, the
stride convention, etc. -- so existing on-disk caches are automatically orphaned and rebuilt
instead of silently being served as if still correct. See README-N300-9B.md Known limitations."""


def rope_full_head_dim(args):
    """The ``full_head_dim`` every rot_mats_* producer must pass so its cos/sin match TPAttention.

    head_dim under permuted full-width RoPE, else None (HF-width cos/sin, partial-rope chain).
    Producers that go through Qwen36RoPESetup get this applied for them; direct rot_mats_decode /
    rot_mats_prefill callers need it explicitly, or the rotary rejects the width mismatch.
    """
    return args.head_dim if getattr(args, "rope_permuted_enabled", False) else None


def rope_channel_perm(head_dim, rope_dim):
    """head_dim-long index list P: permuted channel ``i`` holds HF channel ``P[i]``.

    Chosen so that full-width rotate-half (pairing i with i+head_dim/2) reproduces HF partial
    rope (pairing j with j+rope_dim/2 for j < rope_dim/2, tail untouched). Pure index bookkeeping
    (a plain Python list, no tensor framework) -- ``permute_rope_channels`` is what turns this into
    a device gather.
    """
    assert head_dim % 2 == 0 and rope_dim % 2 == 0, (head_dim, rope_dim)
    assert 0 < rope_dim <= head_dim, (head_dim, rope_dim)
    half, rh = head_dim // 2, rope_dim // 2
    perm = [None] * head_dim
    for j in range(rh):
        perm[j] = j  # rope first half stays put
        perm[half + j] = rh + j  # rope second half moves to the far half
    # Pass-through channels fill the remaining slots in order. The leftover slots are
    # [rh, half) and [half+rh, head_dim) -- equal counts, so they also land in
    # matched (p, p+half) pairs and get cos=1/sin=0 together.
    free = [i for i in range(rh, half)] + [i for i in range(half + rh, head_dim)]
    for slot, src in zip(free, range(rope_dim, head_dim)):
        perm[slot] = src
    assert sorted(perm) == list(range(head_dim)), "rope_channel_perm is not a permutation"
    return perm


def _rope_perm_row_index(device, out_rows, head_dim, rope_dim, stride):
    """The full [1, out_rows] row-index tensor ``permute_rope_channels`` gathers with, built from
    ``ttnn.arange`` + ``ttnn.concat`` only (no torch) -- one 4-segment block per ``rope_channel_perm``
    (see there for why these four ranges, in this order, are the permutation), plus a 5th identity
    segment per block when ``stride > head_dim`` (q_proj's gate half, left untouched). Verified
    against ``ttnn.embedding``'s row-index convention on real hardware: a plain arange+concat index
    tensor reproduces exact torch fancy-indexing (``w[perm]``) bit-for-bit.
    """
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
    """Apply rope_channel_perm to a weight's head_dim output channels, via a ttnn.embedding gather
    (a device row-gather op) rather than torch fancy-indexing.

    Applying it to q_proj and k_proj is safe because attention scores are q.k, a dot product over
    head_dim: permuting both operands the same way leaves the score unchanged. It means the KV
    cache stores K in this PERMUTED order, not HF's -- a test that compares raw cache bytes against
    an HF reference must permute the reference the same way first.

    ``w``: [out, in] projection weight (rows are output channels) or a 1-D [head_dim] norm weight.
    ``stride``: rows per head block, when a head's head_dim channels are not the whole block --
    q_proj is [q_head_0 | gate_head_0 | q_head_1 | ...] so it needs stride=2*head_dim to permute
    only the q half of each head. Defaults to head_dim (k_proj, and per-head norm weights).

    ``ttnn.embedding`` requires BFLOAT16 weights, so ``w`` is cast to bf16 before the gather
    regardless of its input dtype -- done inside ``ttnn.from_torch`` itself (its ``dtype=`` performs
    the identical cast a separate ``.to(torch.bfloat16)`` would, verified bit-for-bit on hardware),
    so no torch cast call is needed. This costs nothing accuracy-wise either way: a gather only
    reorders values, it never touches them, so bf16-then-permute and permute-then-bf16 round every
    value identically -- bit-identical to the previous torch-indexing implementation once the
    caller's own later cast to its target dtype (bfloat16 for norms, bfloat8_b for projections) is
    applied, exactly as before.

    The only torch touching this function is ``w`` itself and the returned tensor -- both are the
    ttnn/torch interop boundary (``ttnn.from_torch``/``ttnn.to_torch``), not computation: ``w``
    arrives as a torch.Tensor because that is what a checkpoint's ``state_dict`` is, and the caller
    (``load_attention_weights_tp``) needs a torch.Tensor back to hand to the rest of the (unrelated,
    pre-existing) weight-loading pipeline. The actual PERMUTE is ``ttnn.embedding``, a device op; the
    two shape adjustments the 1-D case needs (folding it to a [head_dim, 1] "table" for the gather,
    then back) are done as torch ``.reshape()`` -- pure host-side VIEW metadata, zero torch compute
    -- specifically because ``ttnn.reshape`` was found to SILENTLY CORRUPT DATA for a 1D<->2D rank
    change on this build (reproduced standalone: reshaping a device tensor [8] -> [8,1] returned
    [1.9766, 3.2969, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] against a correct [0.30, 1.30, 2.30, ...] --
    caught by this function's own bit-exactness test against the prior torch-indexing
    implementation, not by inspection). Given that, host-side reshape is the CORRECT choice here,
    not merely the convenient one -- the 2D case (q_proj/k_proj) needs no reshape at all and stays
    fully ttnn end to end.
    """
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
    # device may be a multi-device mesh (e.g. N300's 2 devices): replicate the identical gather onto
    # every device, then read back ONE shard: a replicated tensor is identical on
    # each device (same pattern as the other mesh-aware table builders here).
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
    """[..., rope_dim] HF cos/sin -> [..., head_dim] cos/sin in permuted channel order.

    Pass-through slots get cos=1/sin=0 so the full-width rotary is the identity there.

    This is a SCATTER (place the rope values at their permuted slots, fill the rest with the
    identity), which is exactly slice + fill + concat -- so it runs as ttnn device ops
    (``ttnn.slice`` / ``ttnn.full`` / ``ttnn.concat``), verified bit-exact against the equivalent
    torch scatter on hardware. It deliberately does NOT need an outer product: the position x
    inv_freq product that produces ``cos_r``/``sin_r`` happens in the caller and is untouched here
    (which matters, because ``ttnn.outer`` and the [N,1]*[1,M] broadcast-multiply workaround were
    both found to return silent garbage on this build).

    Everything is done at bfloat16. That is not a precision loss: a scatter only places values, it
    never computes on them, and every caller casts to bfloat16 anyway before the tensor reaches a
    device -- so bf16-then-scatter and scatter-then-bf16 yield identical bits, and the two fill
    values (1.0 and 0.0) are exactly representable. ``cos_r``/``sin_r`` arrive as torch tensors and
    a torch tensor is returned because these are host-side table builders (``_rope_dev_tables``'
    one-time build/grow, rope.py's M-RoPE machinery) whose callers index and concatenate the result
    on host; that boundary is ``ttnn.from_torch``/``ttnn.to_torch``, not computation.
    """
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
        # [:rh] | identity pad | [rh:rope_dim] | identity pad  -- the permuted layout, see
        # rope_channel_perm: the two rope halves land at 0 and head_dim/2, the tail fills the rest.
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


def _rope_dev_tables(device, rope_dim, n_rows, theta, full_head_dim=None):
    """ROW_MAJOR [rows, W] cos/sin tables resident ON DEVICE, built once and grown on demand.

    One table serves both decode (per-user row gather via ttnn.embedding) and prefill (contiguous
    ttnn.slice), so neither path needs host trig. ROW_MAJOR because ttnn.embedding wants a
    ROW_MAJOR weight and because slicing it carries no tile-alignment constraint.

    ``full_head_dim``: emit W=head_dim tables in PERMUTED channel order instead of W=rope_dim
    HF-order ones, by widening the finished cos/sin through ``to_full_width_rot_mats`` (a ttnn
    scatter) rather than scattering the raw angles -- equivalent, since an angle-0 pass-through slot
    gives cos=1/sin=0 and that is exactly the identity fill the widen writes. Cached separately (the
    width is part of the key).
    """
    key = (id(device), int(rope_dim), float(theta), int(full_head_dim or 0))
    ent = _ROPE_DEV_TABLES.get(key)
    if ent is not None and ent["rows"] >= n_rows:
        return ent["cos"], ent["sin"]
    rows = max(int(n_rows), 2 * ent["rows"] if ent else 0, 4096)
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    freqs = torch.outer(torch.arange(rows).float(), inv_freq)  # [rows, rope_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [rows, rope_dim]
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
    """Return [cos, sin] each [1, B, 1, W] for the given per-user positions.

    W is rope_dim (HF channel order) by default, or ``full_head_dim`` in permuted channel order
    when that is given (see ``rope_channel_perm``).

    Fully on device: the cos/sin tables are resident (built once, grown on demand) and the
    per-user rows are fetched with ttnn.embedding, so the only host->device traffic per step is
    the [B] index vector. A position past the current table end (M-RoPE rope_delta can push
    rope_pos beyond max_seq_len) grows the table rather than dropping to host trig.
    """
    W = full_head_dim or rope_dim
    if is_blackhole():
        # Blackhole executes the pre-migration statements verbatim (see e83017ce0ec).
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        pos = positions.float()
        freqs = torch.outer(pos, inv_freq)  # [B, rope_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, rope_dim]
        B = positions.shape[0]
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
    pos_i = positions.to(torch.int64).reshape(-1)
    assert int(pos_i.min()) >= 0, f"negative rope position {int(pos_i.min())}"
    tbl_cos, tbl_sin = _rope_dev_tables(device, rope_dim, int(pos_i.max()) + 1, theta, full_head_dim=full_head_dim)
    B = int(positions.shape[0])
    idx = ttnn.from_torch(
        pos_i.to(torch.int32).reshape(1, B),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    def _gather(tbl):
        r = ttnn.embedding(idx, tbl)  # ROW_MAJOR [1, B, W]
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

    W is rope_dim (HF channel order) by default, or ``full_head_dim`` in permuted channel order
    when that is given (see ``rope_channel_perm``).

    position_ids: 3D M-RoPE indices [3, bs, seq_len] (or 2D [bs, seq_len], expanded inside
    get_rot_mats). When None, defaults to text positions arange(seq_len) — the (t==h==w) case
    where interleaved-mrope collapses to ordinary 1D RoPE, so the result is independent of
    mrope_section and identical to the pre-M-RoPE behaviour.
    """
    W = full_head_dim or rope_dim
    if position_ids is None and not is_blackhole():
        # Text-only on Wormhole: positions are exactly arange(seq_len), i.e. a contiguous prefix of
        # the RoPE table, so slice on device instead of recomputing trig on host
        # a [1,1,seq_len,rope_dim] cos+sin pair across. (t==h==w here, so interleaved-mrope
        # collapses to ordinary 1D RoPE and mrope_section is irrelevant -- same values.)
        # Blackhole falls through to the original host path below: unchanged flow.
        tbl_cos, tbl_sin = _rope_dev_tables(device, rope_dim, seq_len, theta, full_head_dim=full_head_dim)

        def _slice(tbl):
            r = ttnn.slice(tbl, [0, 0], [seq_len, W])  # ROW_MAJOR: no tile alignment needed
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
    """cos/sin [1, B, 1, W] interleaved -> HEIGHT_SHARDED on ``shard_cfg`` for the decode-mode kernel.

    ``rotary_embedding_hf(is_decode_mode=True)`` requires sharded cos/sin, and its program factory
    lays every kernel (cos/sin CBs included) on the INPUT's shard grid -- so cos/sin must live on the
    same cores as the tensor being rotated. Q and K therefore need their own copies whenever they sit
    on different grids (K targets the KV-write's shifted half; see model_config's
    kv_cache_write_k_shard_cfg). Cheap enough to be worth it: this is per DECODE STEP, amortised over
    every full-attention layer, against ops removed per layer.
    """
    # Promote to 4D first: the decode-mode kernel matches cos/sin against the input on
    # padded_shape()[1] (the batch axis), and producers differ on rank -- rot_mats_decode and
    # Model._rope_decode_gather return [1,B,1,W] while Qwen36RoPESetup.get_rot_mats' B==T==1 fast
    # path returns [1,1,W], whose axis 1 would read as the (padded) head axis instead. Metadata-only.
    cos_tt = ttnn.unsqueeze_to_4D(cos_tt)
    sin_tt = ttnn.unsqueeze_to_4D(sin_tt)
    return (
        ttnn.interleaved_to_sharded(cos_tt, shard_cfg),
        ttnn.interleaved_to_sharded(sin_tt, shard_cfg),
    )


def apply_rope_full_decode(x_sh, cos_sh, sin_sh, memory_config=None):
    """Permuted-layout decode RoPE: ONE op. x_sh [1,B,n_heads,HD] HEIGHT_SHARDED, cos/sin sharded.

    Replaces apply_partial_rope_decode's 7-op slice/transpose/rotate/transpose/slice/copy/concat
    chain: once the head_dim channels are permuted at weight-load time (``rope_channel_perm``), a
    full-width rotary is exactly HF's partial rope, so this is the whole thing in one op.

    NOTE the output shard spec is COPIED FROM THE INPUT (rotary_embedding_hf_device_operation.cpp's
    compute_output_specs takes only layout+buffer_type from the requested memory_config when the
    input is sharded). So the caller places the result by choosing the INPUT's grid, which is how K
    lands directly in kv_cache_write_k_shard_cfg with no InterleavedToSharded of its own.
    """
    return ttnn.experimental.rotary_embedding_hf(
        x_sh, cos_sh, sin_sh, is_decode_mode=True, memory_config=memory_config or x_sh.memory_config()
    )


def apply_rope_full_prefill(x, cos_tt, sin_tt, memory_config=None):
    """Permuted-layout prefill RoPE: ONE op. x [1, n_heads, seq_len, HD], cos/sin [1,1,seq_len,HD].

    Replaces apply_partial_rope_prefill's slice/rotate/slice/concat with the same one-op trick
    apply_rope_full_decode uses (see ``rope_channel_perm``).
    """
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
    #
    # forward_prefill_paged's chunked_scaled_dot_product_attention still clashes with this at S=2048
    # even with q_chunk_size capped to 64 (see `cap` in attention/tp.py) -- MEASURED via
    # ttnn.dump_device_memory_state right before the SDPA call: the ONLY persistent >100KB L1
    # allocation there is this function's roped Q (196608 B at 85696), and
    # SDPA's static CBs start at L1 address 0 whatever the chunk size, so ANY
    # persistent buffer at a low address clashes regardless of its size or the
    # CB region's. TRIED moving this to DRAM at S>=2048: does NOT help even
    # combined with the `cap`=64 fix (MEASURED: CB region stays exactly 1393856 whether q_chunk_size
    # is 64 or 128 once the source is DRAM vs 454080 when L1: DRAM-source SDPA
    # pays a large fixed CB cost, so no chunk size makes DRAM win here).
    # This is a genuine conflict between the L1 allocator's first-come placement of Q and where
    # SDPA's CBs are laid out, not a simple memory-config choice -- needs a program-factory-level fix
    # (don't assume L1 address 0 for CBs, or force Q high) to fully
    # resolve at S=2048 for the paged/chunked SDPA path specifically. Left at unconditional L1 (only
    # lever that helped was `cap`, which shrank but kept the S=2048 overflow).
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
