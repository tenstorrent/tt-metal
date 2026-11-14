# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4PoolPrefill: model-level DIRECT-TO-POOL chunked prefill of the 78-layer chain (task pf-integrate).

Same build contract as ``Hy4PoolDecoder`` (INTERFACE.md of integ): the chain must be built with HY4_POOL=1 HY4_POOL_PF=1 HY4_KV_FMT=scaled_fp8 (every layer's attention is ``Hy4AttentionPF``), and the pool
(``Hy4PoolDecoder``: kv pools + key pools + PoolCtx) must exist BEFORE the first prefill. ``prefill(prompts)`` writes every prompt row into the sequence-sharded pool (no replicated cache, no hand-off)
and returns the first generated token per user; decode then continues with the unchanged ``Hy4PoolDecoder`` (``prepare_decode`` + capture / run / read_argmax).

One user per mesh row (4 on the 4x8 mesh; 1 on a 1x8 slot for wiring tests). Chunks of ``pf_rows`` (512 or 1024) rows per user advance in lock step (SPMD): a user whose prompt is shorter than the longest
one computes pad tokens (id PAD) in the remaining chunks; those pool rows lie at positions >= its length, are masked by the decode (ids < pos + 1) and overwritten as it generates.
MTP prompt KV (layer 78) is NOT written (see INTERFACE.md of this task): a spec round at the first step reads an empty MTP prompt cache (lower acceptance only; the target verification stays exact).
"""
import time
from dataclasses import dataclass, field

import torch

import ttnn

PAD = 120002  # same pad id as the needle runners


@dataclass
class PrefillResult:
    first: list  # first generated token per user (greedy argmax of the last prompt token)
    lens: list
    chunk_sec: list = field(default_factory=list)
    total_sec: float = 0.0
    hidden: list = (
        None  # optional: final normed hidden of the last prompt token per user ([6144] fp32), keep_hidden=True
    )
    logits: list = None  # optional: logits row of the last prompt token per user (fp32 [V]), keep_logits=True


def decode_bucket(max_len, pages):
    """bucket_tokens for ``dec.ctx.set_bucket(bucket, "topk")`` after a prefill of prompts of at most max_len tokens (as in the needle runners)."""
    return min(pages * 512 // 256 * 256, (-(-max_len // 4096)) * 4096)


class Hy4PoolPrefill:
    def __init__(self, dec, pf_rows=512, log=None):
        assert pf_rows in (512, 1024)
        self.dec, self.ch, self.md = dec, dec.ch, dec.ch.md
        self.rows = int(self.md.shape[0])
        self.cols = int(self.md.shape[1])
        self.pf_rows = pf_rows
        self.log = log or (lambda **kw: None)
        for ly in self.ch.layers:
            assert hasattr(ly.attn, "_pf_chunk512"), "chain not built with HY4_POOL=1 HY4_POOL_PF=1"

    def pages_needed(self, max_len):
        return (-(-max_len // self.pf_rows)) * self.pf_rows // 512

    def _host_rows(self, t, per_row_cols):
        """mesh tensor with one distinct value per mesh row (replicated over columns) -> torch [rows, ...] from device (r, 0)."""
        devs = ttnn.get_device_tensors(t)
        return torch.stack([ttnn.to_torch(devs[r * self.cols]).float() for r in range(self.rows)])

    def prefill(self, prompts, keep_hidden=False, keep_logits=False, sync_every_chunk=True):
        """prompts: list (<= rows) of token-id lists (a missing / empty user is a pad prompt). Returns PrefillResult."""
        ch, md, PF = self.ch, self.md, self.pf_rows
        prompts = [list(p) for p in prompts] + [[PAD]] * (self.rows - len(prompts))
        assert len(prompts) == self.rows
        lens = [max(len(p), 1) for p in prompts]
        L = max(lens)
        nch = -(-L // PF)
        assert self.dec.pages >= self.pages_needed(L), (self.dec.pages, self.pages_needed(L))
        res = PrefillResult(first=[None] * self.rows, lens=lens)
        if keep_hidden:
            res.hidden = [None] * self.rows
        if keep_logits:
            res.logits = [None] * self.rows
        ids_by_layer = {}
        t0 = time.time()
        for c in range(nch):
            ids = torch.full((self.rows, PF), PAD, dtype=torch.long)
            for r in range(self.rows):
                seg = prompts[r][c * PF : (c + 1) * PF]
                if seg:
                    ids[r, : len(seg)] = torch.tensor(seg)
            tc = time.time()
            x4, ids_by_layer = ch.prefill(ids, start=c * PF, ids_by_layer=ids_by_layer)
            enders = [r for r in range(self.rows) if c * PF <= lens[r] - 1 < (c + 1) * PF]
            if enders:
                hn, logits = ch.head(x4)
                lg = ttnn.to_torch(
                    logits, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))
                ).reshape(self.rows, PF, -1)
                hh = self._host_rows(hn, True).reshape(self.rows, PF, -1) if keep_hidden else None
                for r in enders:
                    last = lens[r] - 1 - c * PF
                    row = lg[r, last].float()
                    res.first[r] = int(row.argmax())
                    if keep_logits:
                        res.logits[r] = row
                    if keep_hidden:
                        res.hidden[r] = hh[r, last]
                ttnn.deallocate(hn)
                ttnn.deallocate(logits)
            if sync_every_chunk:
                ttnn.synchronize_device(md)
            ttnn.deallocate(x4)
            dt = time.time() - tc
            res.chunk_sec.append(dt)
            self.log(event="prefill_chunk", c=c, of=nch, chunk_sec=round(dt, 3), sec=round(time.time() - t0, 1))
        for t in ids_by_layer.values():
            ttnn.deallocate(t)
        res.total_sec = time.time() - t0
        return res

    def prepare_decode(self, res, set_bucket=True):
        """after prefill: bucket + first decode inputs (tokens = res.first, pos = prompt lengths). The caller then calls dec.capture() once and dec.set_inputs / dec.run / dec.read_argmax per step."""
        dec = self.dec
        if set_bucket:
            dec.ctx.set_bucket(decode_bucket(max(res.lens), dec.pages), "topk")
        dec.set_inputs(res.first, res.lens)
