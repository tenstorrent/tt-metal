# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN custom attention mask for HunyuanImage-3.0 (text=causal, image=bidirectional).
#
# Built entirely with TTNN ops (no torch): a causal lower-triangular keep-mask,
# OR'd with a bidirectional block for each image span, then converted to an
# additive mask (0 = attend, large-negative = masked). Pattern matches the
# verified reference HunyuanImage3ForCausalMM._prepare_attention_mask_for_generation.

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
    every image span made bidirectional. Built with TTNN ops only.

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
        Per-batch-distinct spans are supported by passing a list-of-lists; the
        device path builds each batch row and stacks them.
    """
    S = seq_len

    # Normalise to per-batch span lists.
    if image_slices and isinstance(image_slices[0], list):
        per_batch = image_slices
    else:
        per_batch = [image_slices or []] * bsz

    rows = [_build_one(device, S, spans, dtype) for spans in per_batch]  # each [1,1,S,S]
    mask = rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=0)
    return mask


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


def _build_one(device, S, spans, dtype):
    """Additive mask for a single batch item -> [1, 1, S, S].

    The dense ``[1,1,S,S]`` working tensors (``ones``/``keep``/``block``/``logical_or``
    output) are built directly in ``dtype`` — bf16 by default. Since the mask holds only
    the values ``{0, 1}`` (exact in bf16) and finally ``{0, _NEG}`` (``_NEG`` is chosen to
    be representable in bf16), this is numerically identical to an fp32 build but halves
    peak DRAM, which at long S is what OOMs. Index arithmetic stays fp32 because bf16
    represents integers exactly only up to 256, and token positions here reach tens of
    thousands — a bf16 ``arange`` would alias positions and corrupt the span membership.
    """
    # Big tensors are floating point; if a non-float dtype is requested, build in bf16
    # and let the final typecast convert.
    build_dtype = dtype if dtype in (ttnn.bfloat16, ttnn.float32) else ttnn.bfloat16

    # Causal keep (1.0 lower-tri incl. diagonal, 0.0 above), built in build_dtype.
    ones = ttnn.ones([1, 1, S, S], dtype=build_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    keep = ttnn.tril(ones, diagonal=0)
    ttnn.deallocate(ones)

    if spans:
        # Each image span is made bidirectional WITHIN itself only — distinct
        # spans must not become mutually visible. So OR a separate within-span
        # block per span: block_s[i,j] = v_s[i] * v_s[j], where v_s indicates
        # membership in span s. (A single combined indicator would wrongly link
        # token i in span A to token j in span B.)
        # Index math in fp32 so positions (up to tens of thousands) are exact.
        idx = ttnn.arange(0, S, 1, dtype=ttnn.float32, device=device)  # [S]
        idx = ttnn.to_layout(ttnn.reshape(idx, [1, 1, 1, S]), ttnn.TILE_LAYOUT)
        for span in spans:
            s, e = _as_start_stop(span)
            ge = ttnn.ge(idx, float(s))
            lt = ttnn.lt(idx, float(e))
            v_s = ttnn.logical_and(ge, lt)  # [1,1,1,S] membership in this span
            ttnn.deallocate(ge)
            ttnn.deallocate(lt)
            # Cast the tiny membership vector to build_dtype so the S×S block (and its
            # matmul) stay in the low-memory dtype.
            if v_s.get_dtype() != build_dtype:
                v_cast = ttnn.typecast(v_s, build_dtype)
                ttnn.deallocate(v_s)
                v_s = v_cast

            v_col = ttnn.reshape(v_s, [1, 1, S, 1])
            # Outer product [S,1]@[1,S] -> [S,S]. Use DRAM: the result is up to
            # seq_len^2 and must not steal L1 from resident model / emb buffers
            # (L1-sharded matmul CBs clash once the backbone is loaded).
            block = ttnn.matmul(v_col, v_s, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(v_col)
            ttnn.deallocate(v_s)

            new_keep = ttnn.logical_or(keep, block)
            ttnn.deallocate(keep)
            ttnn.deallocate(block)
            keep = new_keep
        ttnn.deallocate(idx)

    # additive = (1 - keep) * _NEG  ->  0 where keep, _NEG where masked.
    inv = ttnn.rsub(keep, 1.0)
    ttnn.deallocate(keep)
    add = ttnn.multiply(inv, _NEG)
    ttnn.deallocate(inv)

    if add.get_dtype() != dtype:
        out = ttnn.typecast(add, dtype)
        ttnn.deallocate(add)
        add = out
    return add
