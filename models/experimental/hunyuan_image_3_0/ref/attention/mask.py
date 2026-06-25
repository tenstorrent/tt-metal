# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 custom attention mask.
# Extracted from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     HunyuanImage3ForCausalMM._prepare_attention_mask_for_generation
#     (lines 2859-2882)
#
# The model mixes text and image tokens in one sequence:
#   * text tokens  -> causal attention (lower-triangular)
#   * image tokens -> full (bidirectional) attention WITHIN each image span
#
# Upstream algorithm (verbatim):
#     attention_mask = torch.ones(S, S, bool).tril(0).repeat(bsz, 1, 1)
#     for each image_slice:  attention_mask[i, image_slice, image_slice] = True
#     attention_mask = attention_mask.unsqueeze(1)           # [bsz, 1, S, S]
#
# The image spans come from image_processor.prepare_full_attn_slices(...), which
# returns a list of python `slice` objects (one per image region per batch item).
# Here we accept those slices (or (start, stop) tuples) directly so the mask math
# is decoupled from the tokenizer/image-processor plumbing.

import torch


def _as_slice(s):
    if isinstance(s, slice):
        return s
    start, stop = s
    return slice(int(start), int(stop))


def build_attention_mask(seq_len, image_slices=None, bsz=1, device=None):
    """Boolean attention mask, True = attend (keep), shape [bsz, 1, S, S].

    Matches HunyuanImage3ForCausalMM._prepare_attention_mask_for_generation:
    causal base, with each image span made fully bidirectional.

    Args:
        seq_len:      sequence length S.
        image_slices: list (per batch item) of lists of image spans, OR a single
                      flat list of spans applied to every batch item. Each span
                      is a python `slice` or an (start, stop) tuple. None / empty
                      -> pure causal mask.
        bsz:          batch size.
        device:       torch device.

    Returns:
        torch.BoolTensor [bsz, 1, seq_len, seq_len]  (True = attend).
    """
    attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(diagonal=0).repeat(bsz, 1, 1)

    if image_slices:
        # A flat list of spans is applied to every batch item; a list whose first
        # element is itself a list is treated as per-batch span lists.
        per_batch = image_slices if isinstance(image_slices[0], list) else [image_slices] * bsz

        for i in range(bsz):
            for span in per_batch[i]:
                sl = _as_slice(span)
                attention_mask[i, sl, sl] = True

    return attention_mask.unsqueeze(1)  # [bsz, 1, S, S]


def build_attention_mask_query_row(seq_len, query_pos, image_slices=None, bsz=1, device=None):
    """Additive-mask row for a single decode query position: [bsz, 1, 1, seq_len]."""
    full = build_attention_mask(seq_len, image_slices, bsz=bsz, device=device)
    return full[:, :, query_pos : query_pos + 1, :seq_len]


def to_additive(bool_mask, dtype=torch.float32):
    """Convert a boolean keep-mask to an additive mask: 0 where True (attend),
    a large negative value where False (masked). Useful for TT/SDPA paths that
    add the mask to the attention logits rather than consuming a bool.
    """
    neg = torch.finfo(dtype).min
    return torch.where(bool_mask, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    S = 16
    spans = [slice(4, 9)]  # one image region tokens [4,9)
    m = build_attention_mask(S, spans, bsz=1)
    print(f"mask shape: {tuple(m.shape)}  dtype={m.dtype}")
    grid = m[0, 0].int()
    print("causal lower-tri with bidirectional block over tokens 4..8:")
    for r in range(S):
        print(" ".join(str(int(v)) for v in grid[r].tolist()))
