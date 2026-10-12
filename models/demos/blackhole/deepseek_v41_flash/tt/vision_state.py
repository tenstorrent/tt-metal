# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host state of the images of one request (DeepSeek-V4.1-Flash image input): which prompt positions are inside an image span and the embedding rows of those
positions (the vision encoder's output, ``tt/vision.py``). The prefill takes windows of it (``window``); it is kept for the lifetime of the request because a
re-prefill (a live user is prefilled again next to a new request) rebuilds the prompt from the token ids, which carry only the image placeholder id."""

import torch

TEXT = -1


class VisionSeq:
    def __init__(self, types, spans):
        """``types`` [L] int (TEXT = -1 outside image spans, >= 0 inside: ``image_processor`` token types); ``spans`` [(start position, rows [n, dim] bf16)] (the
        rows of the whole span, delimiters included)."""
        self.types = torch.as_tensor(types, dtype=torch.int64)
        self.spans = [(int(s), r) for s, r in spans]
        for s, r in self.spans:
            assert bool((self.types[s : s + r.shape[0]] >= 0).all()), f"span at {s} is not inside image tokens"
        self.dim = self.spans[0][1].shape[1] if self.spans else 0

    def window(self, s0, n):
        """positions [s0, s0 + n) -> (image mask [n] bool, embedding rows [n, dim] bf16 (zero outside the image positions))."""
        L = self.types.numel()
        m = torch.zeros(n, dtype=torch.bool)
        k = max(0, min(n, L - s0))
        if k:
            m[:k] = self.types[s0 : s0 + k] >= 0
        emb = torch.zeros(n, self.dim, dtype=torch.bfloat16)
        for s, r in self.spans:
            lo, hi = max(s, s0), min(s + r.shape[0], s0 + n)
            if lo < hi:
                emb[lo - s0 : hi - s0] = r[lo - s : hi - s]
        return m, emb
