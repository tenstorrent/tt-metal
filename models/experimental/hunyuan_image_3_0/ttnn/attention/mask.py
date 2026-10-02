# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN custom attention mask for HunyuanImage-3.0 (text=causal, image=bidirectional).
#
# A causal lower-triangular keep-mask, OR'd with a bidirectional block for each
# image span, then converted to an additive mask (0 = attend, large-negative =
# masked). Pattern matches the verified reference
# HunyuanImage3ForCausalMM._prepare_attention_mask_for_generation.
#
# The mask is built on host (torch) and uploaded once. At long I2I sequence
# lengths the dense ``[bsz,1,S,S]`` mask is multiple GiB (e.g. ~2.4 GiB bf16 at
# S≈34.7k); an on-device build via ``ttnn.ones``/``tril``/``matmul`` would hold
# several such tensors live at once and OOM DRAM next to a resident backbone.
# Building host-side and doing a single ``from_torch`` upload keeps the device
# peak to exactly one mask.

import torch
import ttnn

# Large negative additive value for masked positions (representable in bf16).
_NEG = -1.0e30


def _as_start_stop(span):
    if isinstance(span, slice):
        return int(span.start), int(span.stop)
    start, stop = span
    return int(start), int(stop)


def build_attention_mask_tt(
    device,
    seq_len,
    image_slices=None,
    bsz=1,
    dtype=ttnn.bfloat16,
):
    """Additive attention mask on device, shape [bsz, 1, S, S].

    0.0 where a query may attend to a key, `_NEG` where masked. Causal base with
    every image span made bidirectional. Built on host (torch) and uploaded once
    (replicated across the mesh) to keep the device peak to a single mask.

    Args:
        device:       TTNN device.
        seq_len:      sequence length S.
        image_slices: flat list of spans (python slice or (start, stop) tuple)
                      applied to all batch items, or a per-batch list of such
                      lists. None / empty -> pure causal mask.
        bsz:          batch size.
        dtype:        TTNN dtype for the returned mask (default bfloat16).
                      Prefer bf16: SP ``ttnn.pad`` rejects packed dtypes.

    Note:
        Per-batch-distinct spans are supported by passing a list-of-lists; each
        batch row is built on host and concatenated before upload.
    """
    S = int(seq_len)

    # Normalise to per-batch span lists.
    if image_slices and isinstance(image_slices[0], list):
        per_batch = image_slices
    else:
        per_batch = [image_slices or []] * bsz

    # Build the whole additive mask on host, then a single device upload. See the
    # module header: an on-device ttnn.ones/tril/matmul build holds several
    # ``[bsz,1,S,S]`` tensors live at once and OOMs at long I2I sequence lengths.
    host_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    rows = [_build_one_host(S, spans, host_dtype) for spans in per_batch]  # each [1,1,S,S]
    host = rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)

    multi = hasattr(device, "get_num_devices") and device.get_num_devices() > 1
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if multi else None,
    )


def build_attention_mask_tt_sp_sharded(
    device,
    seq_len,
    image_slices=None,
    *,
    bsz=1,
    sp_factor=2,
    tile=32,
    dtype=ttnn.bfloat16,
):
    """Build an additive mask directly in SP query-sharded form.

    Returns a TTNN tensor with global shape ``[bsz, 1, S_pad, S_pad]`` but uploaded
    with a mesh mapper that shards dim-2 (query rows) across the SP mesh axis and
    replicates over the TP axis. This avoids materializing a replicated full mask on
    each device and removes the extra ``ttnn.pad`` allocation peak in ``tt/model.py``.
    """
    if sp_factor <= 1:
        return build_attention_mask_tt(device, seq_len, image_slices=image_slices, bsz=bsz, dtype=dtype)

    S = int(seq_len)
    S_pad = ((S + sp_factor * tile - 1) // (sp_factor * tile)) * (sp_factor * tile)
    shard_q = S_pad // sp_factor

    if image_slices and isinstance(image_slices[0], list):
        per_batch = image_slices
    else:
        per_batch = [image_slices or []] * bsz

    host_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    host = torch.empty((bsz, 1, S_pad, S_pad), dtype=host_dtype)
    keys = torch.arange(S_pad).unsqueeze(0)  # [1,S_pad]

    for bi, spans in enumerate(per_batch):
        for si in range(sp_factor):
            q0, q1 = si * shard_q, (si + 1) * shard_q
            q = torch.arange(q0, q1).unsqueeze(1)  # [shard_q,1] absolute query positions
            keep = keys <= q  # causal base on absolute q positions

            for span in spans:
                s, e = _as_start_stop(span)
                q_in = (q >= s) & (q < e)  # [shard_q,1]
                if not bool(q_in.any()):
                    continue
                k_in = (keys >= s) & (keys < e)  # [1,S_pad]
                keep |= q_in & k_in  # bidirectional only within this image span

            add = torch.where(
                keep,
                torch.zeros((), dtype=torch.float32),
                torch.full((), _NEG, dtype=torch.float32),
            )

            # For padded query rows (q >= S), value is ignored (rows are sliced off later).
            # Set to 0 to avoid wasting host work.
            if q1 > S:
                add[max(0, S - q0) :, :] = 0.0
            # Real queries must not attend padded key columns.
            if S_pad > S and q0 < S:
                add[: max(0, min(q1, S) - q0), S:] = _NEG

            host[bi, 0, q0:q1, :] = add.to(host_dtype)

    mapper = ttnn.create_mesh_mapper(
        device,
        ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()]),
    )
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _build_one_host(S, spans, host_dtype):
    """Additive mask for a single batch item -> torch ``[1, 1, S, S]`` on host.

    ``keep[i,j] = 1`` where query ``i`` may attend key ``j``: causal lower-triangle
    (``j <= i``) OR'd with a bidirectional block for each image span. Each span is
    made bidirectional WITHIN itself only — distinct spans must not become mutually
    visible, so membership is AND'd per span (``q_in & k_in``) rather than combining
    all spans into one indicator (which would wrongly link a token in span A to one
    in span B). The result is the additive mask ``{0 where keep, _NEG where masked}``.

    Index arithmetic uses int64 ``arange`` so token positions (tens of thousands) are
    exact; only the final ``{0, _NEG}`` values are cast to ``host_dtype``. ``_NEG`` is
    chosen to be representable in bf16.
    """
    q = torch.arange(S).unsqueeze(1)  # [S,1] query positions
    keys = torch.arange(S).unsqueeze(0)  # [1,S] key positions
    keep = keys <= q  # causal base [S,S]

    for span in spans or []:
        s, e = _as_start_stop(span)
        q_in = (q >= s) & (q < e)  # [S,1]
        k_in = (keys >= s) & (keys < e)  # [1,S]
        keep |= q_in & k_in  # bidirectional only within this image span

    add = torch.where(
        keep,
        torch.zeros((), dtype=torch.float32),
        torch.full((), _NEG, dtype=torch.float32),
    )
    return add.to(host_dtype).reshape(1, 1, S, S)
