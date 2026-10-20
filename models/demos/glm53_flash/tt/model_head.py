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
from models.demos.glm53_flash.tt.mesh import GLMMesh, single_chip

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
        ttnn.deallocate(h)
        out = self.logits(hn)
        ttnn.deallocate(hn)
        return out

    def last_token_logits(self, streams: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """Prefill: logits ``[1, 1, 1, vocab]`` of the last valid token (row ``seq_len - 1``) only."""
        assert 1 <= seq_len <= streams.shape[-2], (seq_len, streams.shape)
        row = ttnn.slice(streams, [0, 0, seq_len - 1, 0], [1, 1, seq_len, self.n * self.hidden])
        out = self.forward(row)
        ttnn.deallocate(row)
        return out

    def forward_greedy(self, streams: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Decode step: ``(logits [1, 1, B, vocab] bf16, ids [1, 1, B, 1] uint32)``; trace-capturable."""
        logits = self.forward(streams)
        return logits, self.greedy(logits)
