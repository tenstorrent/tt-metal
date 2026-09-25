# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device top-k for the speculative sampler: ttnn.topk reduces [T, vocab] to [T, k].
Opt-in via QWEN36_SPEC_DEVICE_TOPK=1; the per-iteration pad allocates while traces are live.
"""

import math

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model_config import SPEC_TOPK_ALIGN, SPEC_TOPK_CHUNK

# Pad fill. Below any real logit, so padding never enters a chunk's top-k.
_NEG = -1e30


def topk_support(logits_dev, top_k, rows):
    """Bit-exact top-k values; tied ids may differ from torch.topk. Chunking is exact if each chunk returns >= top_k."""
    assert top_k > 0, "device support needs top_k > 0"
    vocab = int(logits_dev.shape[-1])
    top_k = min(top_k, vocab)  # dist() clamps the same way, so the two supports agree
    n = min(-(-top_k // SPEC_TOPK_ALIGN) * SPEC_TOPK_ALIGN, vocab)
    chunks = -(-vocab // SPEC_TOPK_CHUNK)
    assert SPEC_TOPK_CHUNK >= n, f"chunk {SPEC_TOPK_CHUNK} must hold {n} entries for the merge to be exact"
    # Logical shape, not volume(): that returns the tile-padded row count and oversizes the reshape.
    r_in = math.prod(int(d) for d in logits_dev.shape) // vocab

    padded = ttnn.pad(logits_dev, [(0, 0), (0, 0), (0, 0), (0, chunks * SPEC_TOPK_CHUNK - vocab)], value=_NEG)
    split = ttnn.reshape(padded, (1, 1, r_in * chunks, SPEC_TOPK_CHUNK))
    vals, idx = ttnn.topk(split, n, dim=-1)
    # One replica is the whole answer: the trace replicates the logits across the mesh.
    c_idx = ttnn.to_torch(ttnn.get_device_tensors(idx)[0]).reshape(-1, chunks, n)[:rows].long()
    c_vals = ttnn.to_torch(ttnn.get_device_tensors(vals)[0]).reshape(-1, chunks, n)[:rows].float()
    for t in (vals, idx, split, padded):
        ttnn.deallocate(t)

    # Chunk-local ids -> vocabulary ids, then one host merge.
    c_idx = c_idx + torch.arange(chunks).view(1, chunks, 1) * SPEC_TOPK_CHUNK
    flat_v, flat_i = c_vals.reshape(rows, -1), c_idx.reshape(rows, -1)
    h_vals, order = torch.topk(flat_v, top_k, dim=-1)
    h_idx = flat_i.gather(-1, order)
    assert bool((h_idx < vocab).all()), "pad column selected; _NEG is not below the real logits"
    return h_idx, h_vals
