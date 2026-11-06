# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Final head of GLM-5.3-Flash on one chip: stream collapse, final RMSNorm, LM head, greedy ids.

HF (tail of ``Glm5NextTextModel.forward`` + ``Glm5NextForConditionalGeneration.forward``)::

    h = self.hc_head(streams)            # Glm5NextTextHyperHead: streams.mean(dim=2), UNWEIGHTED (bf16 in, bf16 out)
    h = self.norm(h)                     # Glm5NextTextRMSNorm(4096, eps=1e-5): fp32 x*rsqrt(mean(x^2)+eps) -> bf16, * weight (bf16)
    logits = self.lm_head(h[:, -k:, :])  # nn.Linear(4096, 154880, bias=False), bf16 x bf16 -> bf16 logits; no softcap
                                         # lm_head.weight is its own checkpoint tensor (tie_word_embeddings = false)

Device, from the packed residual ``[1, 1, T, 4*4096]`` fp32 (T = tokens, or B user slots in decode):

    collapse  4 tile-aligned column slices summed in fp32, x 1/4                      -> [1, 1, T, 4096] fp32
    norm      ttnn.rms_norm fp32 (HiFi4, fp32 acc) with the bf16 checkpoint weight,
              cast to bf16 (the LM head input dtype, as in HF)                       -> [1, 1, T, 4096] bf16
    logits    bf16 x bfp8_b weight (HiFi2, fp32 acc). The vocab is split into
              ``vocab_chunks`` column blocks (154880 = 5 x 30976) run as separate
              matmuls and concatenated: 47 ms vs 88 ms for one 154880-wide matmul
              at T=4128, equal at T=32                                               -> [1, 1, T, 154880] bf16
    greedy    multicore untilize + ttnn.argmax(dim=-1, keepdim=True)                  -> [1, 1, T, 1] uint32 ids
              (argmax's output dtype; the bits equal int32 for any vocab id)

Everything is a device op on weights uploaded once, so a decode-shaped call is trace-capturable.
``last_token_logits`` slices the last valid row of the streams BEFORE the head (prefill only
needs that token's logits; HF ``logits_to_keep=1``).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.reference.checkpoint import (
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    default_snapshot_dir,
    load_tensors,
    shard_headers,
)
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.matmul_configs import DecodeLinear
from models.demos.glm53_flash.tt.mesh import GLMMesh, from_torch_host_layout, single_chip

TILE = 32


def _compute(device, fidelity, packer_l1_acc):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def _upload(device, t: torch.Tensor, dtype):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def vocab_chunk_bounds(vocab: int, chunks: int) -> list[tuple[int, int]]:
    """Tile-aligned ``[start, end)`` column blocks covering ``[0, vocab)`` as evenly as possible."""
    assert vocab % TILE == 0, vocab
    tiles = vocab // TILE
    base, rem = divmod(tiles, chunks)
    widths = [(base + (1 if i < rem else 0)) * TILE for i in range(chunks)]
    bounds, start = [], 0
    for w in widths:
        if w:
            bounds.append((start, start + w))
            start += w
    assert start == vocab
    return bounds


class TtModelHead(LightweightModule):
    """``hc_head`` (mean of the streams) -> final ``norm`` -> ``lm_head`` -> greedy ids."""

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        norm_weight: torch.Tensor,  # [hidden] bf16  (model.language_model.norm.weight)
        lm_head_weight: torch.Tensor,  # [vocab, hidden] bf16  (lm_head.weight, nn.Linear layout)
        *,
        lm_head_dtype=ttnn.bfloat8_b,
        vocab_chunks: int = 5,
        mesh: GLMMesh | None = None,
    ):
        """On a mesh the vocab is split over the TP columns (column ``c`` keeps the contiguous block
        ``[c*V/cols, (c+1)*V/cols)``, chunked inside the block); ``logits`` all-gathers the column
        blocks back into the natural vocab order so ``greedy`` / callers see ``[.., V]`` on every chip.
        Each chip then streams ``V/cols`` head columns per step instead of the whole 0.63 GB head."""
        H, V = cfg.hidden_size, cfg.vocab_size
        assert tuple(norm_weight.shape) == (H,), norm_weight.shape
        assert tuple(lm_head_weight.shape) == (V, H), lm_head_weight.shape
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        self.hidden, self.vocab, self.n = H, V, cfg.hc_mult
        self.eps = float(cfg.rms_norm_eps)
        self.norm_w = _upload(device, norm_weight.float().reshape(1, 1, 1, H), ttnn.bfloat16)
        wt = lm_head_weight.t()  # [H, V]
        cols = self.mesh.cols
        assert V % (cols * TILE) == 0, (V, cols)
        V_local = V // cols
        if cols > 1:
            vocab_chunks = 1  # the 4-way split already bounds the matmul width (38720 for GLM-5.3-Flash)
        self.bounds = vocab_chunk_bounds(V_local, vocab_chunks)  # chunk bounds inside one column block
        if cols == 1:
            self.lm_w = [_upload(device, wt[:, a:b], lm_head_dtype) for a, b in self.bounds]
        else:
            # chunk i host tensor = [block(c=0, i) | block(c=1, i) | ...]; ``upload(cols=-1)`` hands column c its block
            self.lm_w = [
                self.mesh.upload(
                    torch.cat([wt[:, c * V_local + a : c * V_local + b] for c in range(cols)], dim=-1),
                    lm_head_dtype,
                    cols=-1,
                )
                for a, b in self.bounds
            ]
        self.lm_lin = [DecodeLinear(device, w) for w in self.lm_w]  # 1D mcast configs for decode-shaped rows
        self.lm_head_dtype = lm_head_dtype
        # HiFi2 is the project policy for bfp8 matmuls; a bf16 head keeps the full mantissas.
        fid = ttnn.MathFidelity.HiFi2 if lm_head_dtype == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi4
        self.lm_compute = _compute(device, fid, packer_l1_acc=True)
        self.hifi4 = _compute(device, ttnn.MathFidelity.HiFi4, packer_l1_acc=False)

    @classmethod
    def from_checkpoint(cls, device, cfg: GLM53FlashConfig, snapshot: Path | None = None, headers=None, **kw):
        snapshot = snapshot or default_snapshot_dir()
        headers = headers or shard_headers(snapshot)
        w = load_tensors(snapshot, [FINAL_NORM_KEY, LM_HEAD_KEY], headers)
        return cls(device, cfg, w[FINAL_NORM_KEY], w[LM_HEAD_KEY], **kw)

    # ---- pieces -------------------------------------------------------------------------------
    @staticmethod
    def collapse_streams(streams: ttnn.Tensor, n: int) -> ttnn.Tensor:
        """Packed ``[1, 1, T, n*H]`` fp32 -> unweighted mean over the n streams ``[1, 1, T, H]`` fp32
        (4 slices + 3 adds + 1 scale; the last decoder layer's ``collapse=True`` emits this directly)."""
        T, H = streams.shape[-2], streams.shape[-1] // n
        acc = ttnn.slice(streams, [0, 0, 0, 0], [1, 1, T, H])
        for i in range(1, n):
            part = ttnn.slice(streams, [0, 0, 0, i * H], [1, 1, T, (i + 1) * H])
            acc = ttnn.add(acc, part, output_tensor=acc)
            ttnn.deallocate(part)
        return ttnn.multiply(acc, 1.0 / n, output_tensor=acc)

    def collapse(self, streams: ttnn.Tensor) -> ttnn.Tensor:
        """Packed ``[1, 1, T, n*H]`` fp32 -> unweighted mean over the n streams ``[1, 1, T, H]`` fp32;
        an already collapsed ``[1, 1, T, H]`` (``decode_forward(collapse=True)``) passes through."""
        if streams.shape[-1] == self.hidden:
            return streams
        return self.collapse_streams(streams, self.n)

    def norm(self, h: ttnn.Tensor) -> ttnn.Tensor:
        """Final RMSNorm in fp32, result in bf16 (the LM head input dtype)."""
        y = ttnn.rms_norm(h, epsilon=self.eps, weight=self.norm_w, compute_kernel_config=self.hifi4)
        yb = ttnn.typecast(y, ttnn.bfloat16)
        ttnn.deallocate(y)
        return yb

    def logits(self, h: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, T, H]`` bf16 -> ``[1, 1, T, vocab]`` bf16 (fp32 accumulation); on a mesh the column
        blocks are all-gathered over the TP columns into the natural vocab order."""
        outs = [lin(h, dtype=ttnn.bfloat16, compute_kernel_config=self.lm_compute) for lin in self.lm_lin]
        if len(outs) == 1:
            out = outs[0]
        else:
            out = ttnn.concat(outs, dim=-1)
            for t in outs:
                ttnn.deallocate(t)
        if self.mesh.cols > 1:
            full = self.mesh.all_gather_cols(out, dim=-1)
            ttnn.deallocate(out)
            out = full
        return out

    def greedy(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, T, vocab]`` bf16 TILE -> ``[1, 1, T, 1]`` uint32 ROW_MAJOR token ids (argmax over the vocab).

        ``ttnn.argmax`` on a TILE input would untilize single-core (hopeless at this width), so the
        logits are untilized multicore first."""
        u = ttnn.untilize(logits, use_multicore=True)
        ids = ttnn.argmax(u, dim=-1, keepdim=True)
        ttnn.deallocate(u)
        return ids

    # ---- entry points -------------------------------------------------------------------------
    def forward(self, streams: ttnn.Tensor) -> ttnn.Tensor:
        """Packed streams ``[1, 1, T, n*H]`` fp32 -> logits ``[1, 1, T, vocab]`` bf16 for every row."""
        h = self.collapse(streams)
        hn = self.norm(h)
        if h is not streams:  # the pass-through of an already collapsed hidden is the caller's tensor
            ttnn.deallocate(h)
        out = self.logits(hn)
        ttnn.deallocate(hn)
        return out

    def last_token_logits(self, streams: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """Prefill: logits ``[1, 1, 1, vocab]`` of the last valid token (row ``seq_len - 1``) only. Shape-static
        like ``tile_logits``: the collapse and the final norm run on every row, the tensor-argument ``ttnn.slice``
        cuts the 32-row tile holding the row (the program keys on ``T``, the offset is a runtime tensor), an
        exact one-hot matmul moves the row to the tile's row 0, the decode-shaped LM head runs on the tile and
        its row 0 is returned. No program depends on ``seq_len``: the former host-int slice of row ``seq_len - 1``
        was one program per prompt length, built at request time under live traces (the 2026-10-28 served
        seal hit)."""
        T = int(streams.shape[-2])
        assert 1 <= seq_len <= T, (seq_len, streams.shape)
        t0, r = ((seq_len - 1) // TILE) * TILE, (seq_len - 1) % TILE
        h = self.collapse(streams)
        hn = self.norm(h)
        if h is not streams:
            ttnn.deallocate(h)
        rm = lambda v: ttnn.from_torch(
            torch.tensor(v, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        start, end = rm([0, 0, t0, 0]), rm([1, 1, t0 + TILE, int(hn.shape[-1])])
        tile = ttnn.slice(input_tensor=hn, starts=start, ends=end, slice_dim=2, num_devices=T // TILE)
        ttnn.deallocate(hn)
        onehot = torch.zeros(1, 1, TILE, TILE)
        onehot[0, 0, 0, r] = 1.0
        sel = from_torch_host_layout(onehot, tile.dtype, ttnn.TILE_LAYOUT, self.device)  # no Tilize program
        picked = ttnn.matmul(sel, tile, dtype=tile.dtype, compute_kernel_config=self.hifi4)  # row 0 = row r
        logits = self.logits(picked)  # [1, 1, 32, vocab]
        out = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, 1, int(logits.shape[-1])])
        for t in (start, end, tile, sel, picked, logits):
            ttnn.deallocate(t)
        return out

    def last_token_logits_rows(self, streams: ttnn.Tensor, seq_lens: tuple[int, ...]) -> ttnn.Tensor:
        """``last_token_logits`` with a last token PER MESH ROW (a wave: every row holds its own chunk of ``T``
        rows, ``seq_lens[r]`` is row r's valid length; 0 on an idle row picks its row 0): the tile bounds
        (``[4]`` int32 per chip) and the one-hot (``[1, 1, 32, 32]``) are row-sharded uploads, the programs the
        single-prompt form's. Returns ``[1, 1, 1, vocab]`` per chip: row r's logits on row r's chips."""
        T = int(streams.shape[-2])
        rows = self.mesh.rows
        if len(seq_lens) != rows:
            raise ValueError(f"{len(seq_lens)} row lengths for a {rows}-row mesh")
        last = [max(1, min(int(n), T)) - 1 for n in seq_lens]
        h = self.collapse(streams)
        hn = self.norm(h)
        if h is not streams:
            ttnn.deallocate(h)
        W = int(hn.shape[-1])
        bounds = lambda t: self.mesh.upload(t.reshape(-1), ttnn.int32, rows=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        start = bounds(torch.tensor([[0, 0, (r // TILE) * TILE, 0] for r in last], dtype=torch.int32))
        end = bounds(torch.tensor([[1, 1, (r // TILE) * TILE + TILE, W] for r in last], dtype=torch.int32))
        onehot = torch.zeros(rows, 1, TILE, TILE)
        for r, n in enumerate(last):
            onehot[r, 0, 0, n % TILE] = 1.0
        sel = self.mesh.upload(onehot, hn.dtype, rows=0)  # [1, 1, 32, 32] per chip: row 0 picks its row's token
        out = self._last_token_logits_normed(hn, start, end, sel)
        for t in (start, end, sel):
            ttnn.deallocate(t)
        return out

    def last_token_logits_rows_from(
        self, streams: ttnn.Tensor, tile_start: ttnn.Tensor, tile_end: ttnn.Tensor, onehot: ttnn.Tensor
    ) -> ttnn.Tensor:
        """``last_token_logits_rows`` on persistent per-row device bounds / one-hot (the traced wave: no host value
        of the wave inside the graph). Returns ``[1, 1, 1, vocab]`` per chip."""
        h = self.collapse(streams)
        hn = self.norm(h)
        if h is not streams:
            ttnn.deallocate(h)
        return self._last_token_logits_normed(hn, tile_start, tile_end, onehot)

    def _last_token_logits_normed(self, hn, start, end, sel) -> ttnn.Tensor:
        T = int(hn.shape[-2])
        tile = ttnn.slice(input_tensor=hn, starts=start, ends=end, slice_dim=2, num_devices=T // TILE)
        ttnn.deallocate(hn)
        picked = ttnn.matmul(sel, tile, dtype=tile.dtype, compute_kernel_config=self.hifi4)
        logits = self.logits(picked)  # [1, 1, 32, vocab]
        out = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, 1, int(logits.shape[-1])])
        for t in (tile, picked, logits):
            ttnn.deallocate(t)
        return out

    def tile_logits(self, streams: ttnn.Tensor, tile_start: ttnn.Tensor, tile_end: ttnn.Tensor) -> ttnn.Tensor:
        """Traced prefill: logits ``[1, 1, 32, vocab]`` of one 32-row tile of the ``T`` packed rows, the tile
        chosen by the persistent device bounds ``tile_start`` / ``tile_end`` (``[4]`` int32:
        ``[0, 0, tile0, 0]`` / ``[1, 1, tile0 + 32, hidden]``): the collapse and the final norm run on every
        row (per-row ops), the tensor-argument ``ttnn.slice`` cuts the tile (the program keys on ``T``, the
        offset is a runtime tensor: the node ``tt_transformers`` uses after its bucketed prefill traces),
        then the same decode-shaped LM-head matmul ``last_token_logits`` runs on the tile."""
        h = self.collapse(streams)
        hn = self.norm(h)
        if h is not streams:
            ttnn.deallocate(h)
        T = int(hn.shape[-2])
        tile = ttnn.slice(input_tensor=hn, starts=tile_start, ends=tile_end, slice_dim=2, num_devices=T // TILE)
        ttnn.deallocate(hn)
        out = self.logits(tile)
        ttnn.deallocate(tile)
        return out

    def forward_greedy(self, streams: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Decode step: ``(logits [1, 1, B, vocab] bf16, ids [1, 1, B, 1] uint32)``; trace-capturable."""
        logits = self.forward(streams)
        return logits, self.greedy(logits)
