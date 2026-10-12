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


def locate_spans(tokens, span_lens, image_token_id, first=0):
    """Positions of the image spans in the prompt ``tokens`` (1-D): span j starts at the first ``image_token_id`` at or after the end of span j - 1 and has ``span_lens[j]`` tokens.
    -> list of starts for the spans fully inside ``tokens`` (a span that starts inside but does not fit raises: a scheduler chunk must not split an image).
    """
    t = torch.as_tensor(tokens).reshape(-1)
    starts, pos = [], first
    for n in span_lens:
        hit = (t[pos:] == image_token_id).nonzero()
        if hit.numel() == 0:
            break
        s = pos + int(hit[0])
        if s + n > t.numel() or not bool((t[s : s + n] == image_token_id).all()):
            raise ValueError(
                f"image span of {n} tokens at {s} is not contained in the {t.numel()} prompt tokens given (an image must not be split across prefill chunks)"
            )
        starts.append(s)
        pos = s + n
    return starts


class VisionBook:
    """Images of the live requests, per model user: the encoder rows computed so far and the spans located so far. ``update_many`` is called at every prefill call (the first chunk and every
    continuation chunk of each request in it); ``pixel_values`` / ``grids`` hold ALL images of the request (vLLM's multimodal kwargs), the spans of the images whose tokens are already in
    the prompt are located, and each image is encoded once. The images that still need the tower (of all the requests of the call) are encoded TOGETHER: ``DSV41Vision.encode_many`` puts
    up to 4 of them on the 4 mesh rows at once."""

    def __init__(self, vision, image_token_id, cache_images=None):
        self.vision, self.image_token_id = vision, image_token_id
        self.seqs, self._rows = {}, {}
        # encoder rows of recent images by content (an image that comes again, e.g. a shared system image or a retry, skips the tower): LRU of DSV41_VISION_CACHE entries (~10 MB each), 0 = off
        import collections
        import os

        self.cache_max = int(os.environ.get("DSV41_VISION_CACHE", "32")) if cache_images is None else cache_images
        self._cache = collections.OrderedDict()
        self.hits = 0

    def drop(self, p):
        self.seqs.pop(p, None)
        self._rows.pop(p, None)

    def get(self, p):
        return self.seqs.get(p)

    @staticmethod
    def _key(patches, h, w):
        import hashlib

        return hashlib.sha1(patches.contiguous().view(torch.int16).numpy().tobytes() + f"{h}x{w}".encode()).hexdigest()

    def update(self, p, tokens, end, pixel_values, grids, start):
        """-> the user's ``VisionSeq`` for positions < end (None: no image in the prompt so far)."""
        return self.update_many([(p, tokens, end, pixel_values, grids, start)])[p]

    def update_many(self, requests):
        """requests [(user, prompt tokens, end, pixel_values, grids, start)] -> {user: ``VisionSeq`` for positions < end or None}."""
        from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP

        a = self.vision.a
        plans, jobs = {}, {}  # jobs: key -> (patches, h, w, types) of the images that need the tower
        for p, tokens, end, pixel_values, grids, start in requests:
            if start == 0:
                self.drop(p)
            if pixel_values is None or grids is None or len(grids) == 0:
                continue
            grids = torch.as_tensor(grids).reshape(-1, 3)
            pv = torch.as_tensor(pixel_values)
            vit = [(int(g[1]), int(g[2])) for g in grids]
            llm = [(-(-h // a.ratio), -(-w // a.ratio)) for h, w in vit]
            span_lens = [h * (w + 1) + 2 for h, w in llm]
            starts = locate_spans(torch.as_tensor(tokens).reshape(-1)[:end], span_lens, self.image_token_id)
            offs = [0]
            for h, w in vit:
                offs.append(offs[-1] + h * w)
            rows = self._rows.setdefault(p, {})
            todo = {}
            for j in range(len(starts)):
                if j in rows:
                    continue
                h, w = vit[j]
                patches = pv[offs[j] : offs[j + 1]].reshape(h * w, 3, a.patch, a.patch)
                key = self._key(patches, h, w)
                if key in self._cache:
                    self._cache.move_to_end(key)
                    self.hits += 1
                    rows[j] = self._cache[key]
                else:
                    todo[j] = key
                    jobs.setdefault(key, (patches, h, w, VP._ip.image_token_types(*llm[j])))
            plans[p] = (end, starts, llm, todo)
        if jobs:
            keys = list(jobs)
            spans = self.vision.image_rows_many([jobs[k] for k in keys])
            done = dict(zip(keys, spans))
            if self.cache_max > 0:
                for k, rows_ in done.items():
                    self._cache[k] = rows_
                while len(self._cache) > self.cache_max:
                    self._cache.popitem(last=False)
            for p, (end, starts, llm, todo) in plans.items():
                for j, key in todo.items():
                    self._rows[p][j] = done[key]
        out = {}
        for p, tokens, end, pixel_values, grids, start in requests:
            if p not in plans or not plans[p][1]:
                out[p] = self.seqs.get(p)
                continue
            end, starts, llm, _ = plans[p]
            types = torch.full((end,), TEXT, dtype=torch.int64)
            spans = []
            for j, s in enumerate(starts):
                t = VP._ip.image_token_types(*llm[j])
                types[s : s + t.numel()] = t
                spans.append((s, self._rows[p][j]))
            self.seqs[p] = VisionSeq(types, spans)
            out[p] = self.seqs[p]
        return out
