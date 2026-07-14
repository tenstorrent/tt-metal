# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN custom attention mask for HunyuanImage-3.0 (text=causal, image=bidirectional).
#
# Built entirely with TTNN ops (no torch): a causal lower-triangular keep-mask,
# OR'd with a bidirectional block for each image span, then converted to an
# additive mask (0 = attend, large-negative = masked). Pattern matches the
# verified reference HunyuanImage3ForCausalMM._prepare_attention_mask_for_generation.

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


def _build_one(device, S, spans, dtype):
    """Additive mask for a single batch item -> [1, 1, S, S]."""
    # Causal keep (1.0 lower-tri incl. diagonal, 0.0 above).
    ones = ttnn.ones([1, 1, S, S], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    keep = ttnn.tril(ones, diagonal=0)
    ttnn.deallocate(ones)

    if spans:
        # Each image span is made bidirectional WITHIN itself only — distinct
        # spans must not become mutually visible. So OR a separate within-span
        # block per span: block_s[i,j] = v_s[i] * v_s[j], where v_s indicates
        # membership in span s. (A single combined indicator would wrongly link
        # token i in span A to token j in span B.)
        idx = ttnn.arange(0, S, 1, dtype=ttnn.float32, device=device)  # [S]
        idx = ttnn.to_layout(ttnn.reshape(idx, [1, 1, 1, S]), ttnn.TILE_LAYOUT)
        for span in spans:
            s, e = _as_start_stop(span)
            ge = ttnn.ge(idx, float(s))
            lt = ttnn.lt(idx, float(e))
            v_s = ttnn.logical_and(ge, lt)  # [1,1,1,S] membership in this span
            ttnn.deallocate(ge)
            ttnn.deallocate(lt)

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
