# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Logits-level accuracy of the GLM-5.3-Flash chain vs HF (CPU helper, no ttnn).

The raw mHC residual PCC after 45 layers is depth-chaotic for this model (doc/mesh/README.md, chain note) and is
not a correctness metric. This module measures what a user sees: the logits. Given the last layer's streams
``[S, 4, 4096]`` (the device's, pulled to host as ``tests/test_mesh_model.py::_ttft`` does, or HF's ``x_out``) it
runs the host head exactly as ``_ttft`` does (stream mean -> final RMSNorm with the checkpoint's norm weight,
fp32 -> ``lm_head`` fp32) and compares with the HF logits of the dump (``logits.pt`` written by
``reference/dump_natural_prompt.py``; recomputed from the dump's ``layer{L}_prefill.pt`` ``x_out`` through the
same head when the built chain stops before the last layer or the dump carries no logits file):

* per-position logits PCC (Pearson over the vocab),
* top-1 agreement rate, top-5 overlap rate (``|top5_ref & top5_dev| / 5``, averaged over positions),
* mean KL(HF || device) in nats (full-vocab softmax, temperature 1).

``host_head(..., round_hidden=...)`` rounds the stream mean or the normalized hidden to bf16 before the next
stage, ``round_weight="bfp8"`` also rounds ``lm_head`` the way the device stores it (``tt/model_head.py``: ``W^T``
``[H, V]`` in bfloat8_b, 16-value shared-exponent blocks along V, bf16 hidden, fp32 accumulation): the CPU floor of
these numbers (what the device head cannot beat), measured by ``tests/test_logits_check.py``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from models.demos.glm53_flash.reference.checkpoint import (
        FINAL_NORM_KEY,
        LM_HEAD_KEY,
        default_snapshot_dir,
        load_tensors,
        shard_headers,
    )
except ImportError:  # plain torch venv: run from this directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from checkpoint import FINAL_NORM_KEY, LM_HEAD_KEY, default_snapshot_dir, load_tensors, shard_headers  # noqa: E402

LOGITS_FILE = "logits.pt"  # [S, vocab] fp32: the host head on the dump's last-layer x_out
TOPK_FILE = "top5.pt"  # {"top5_ids": [S, 5], "top5_logits": [S, 5], "top1_ids": [S], ...}
META_FILE = "prompt_meta.json"  # prompt text / ids / which layer the logits belong to
RMS_NORM_EPS = 1e-5  # GLM53FlashConfig.rms_norm_eps (config.json); passed explicitly by callers that have the config


def load_head_weights(snapshot: Path | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(norm_weight [H], lm_head [V, H])`` in their checkpoint dtype (bf16)."""
    snapshot = snapshot or default_snapshot_dir()
    w = load_tensors(snapshot, [FINAL_NORM_KEY, LM_HEAD_KEY], shard_headers(snapshot))
    return w[FINAL_NORM_KEY], w[LM_HEAD_KEY]


def collapse_streams(streams: torch.Tensor) -> torch.Tensor:
    """``[S, 4, H]`` -> ``[S, H]`` fp32 stream mean (the HF ``hc_head``); ``[S, H]`` passes through."""
    s = streams.detach().float()
    if s.ndim == 3:
        return s.mean(1)
    if s.ndim == 2:
        return s
    raise ValueError(f"streams must be [S, 4, H] or [S, H], got {tuple(s.shape)}")


def host_head(
    streams: torch.Tensor,
    norm_w: torch.Tensor,
    lm_w: torch.Tensor,
    eps: float = RMS_NORM_EPS,
    *,
    round_hidden: str | None = None,
    round_weight: str | None = None,
    chunk: int = 32,
) -> torch.Tensor:
    """``streams [S, 4, H]`` (or a collapsed ``[S, H]``) -> logits ``[S, V]`` fp32: the ``_ttft`` host head
    (``h = mean(streams); hn = h * rsqrt(mean(h^2) + eps) * norm_w; logits = hn @ lm_head.T``, all fp32).

    ``round_hidden``: ``None`` (fp32 throughout), ``"h"`` (the stream mean rounded to bf16 before the norm) or
    ``"hn"`` (the normalized hidden rounded to bf16 before ``lm_head``: a bf16-activation head's input).
    ``round_weight``: ``None`` (the checkpoint's bf16 ``lm_head`` as fp32) or ``"bfp8"`` (``bfp8_round`` of ``W^T``
    ``[H, V]`` along V, the device head's storage; the matmul fidelity is not emulated)."""
    if round_hidden not in (None, "h", "hn"):
        raise ValueError(f"round_hidden must be None, 'h' or 'hn', got {round_hidden!r}")
    if round_weight not in (None, "bfp8"):
        raise ValueError(f"round_weight must be None or 'bfp8', got {round_weight!r}")
    h = collapse_streams(streams)
    if round_hidden == "h":
        h = h.to(torch.bfloat16).float()
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w.float()
    if round_hidden == "hn":
        hn = hn.to(torch.bfloat16).float()
    w = lm_w.float().t().contiguous()  # [H, V] (2.5 GB for 154880 x 4096: shared by the chunks below)
    if round_weight == "bfp8":
        w = bfp8_round(w)
    out = torch.empty(hn.shape[0], w.shape[1], dtype=torch.float32)
    for s0 in range(0, hn.shape[0], chunk):  # bounded temporaries for long prompts
        out[s0 : s0 + chunk] = hn[s0 : s0 + chunk] @ w
    return out


def bfp8_round(w: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Host emulation of ttnn's ``bfloat8_b`` along the last dim: ``block`` consecutive values share the exponent of
    their largest magnitude and keep a sign + 7-bit explicit-leading-one mantissa, so every value of the block is
    rounded (to nearest) to a multiple of ``2 ** (floor(log2(max |x|)) - 6)``. A floor estimate: the device's
    conversion may differ in the last bit (rounding mode), and the matmul fidelity (HiFi2) is not modelled."""
    x = w.detach().float()
    K = x.shape[-1]
    if K % block:
        raise ValueError(f"last dim {K} is not a multiple of the bfp8 block {block}")
    xb = x.reshape(*x.shape[:-1], K // block, block)
    amax = xb.abs().amax(-1, keepdim=True)
    step = torch.exp2(torch.floor(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny))) - 6)
    q = torch.round(xb / step).clamp_(-127, 127) * step
    q = torch.where(amax > 0, q, torch.zeros_like(q))
    return q.reshape(x.shape)


def row_pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of every row of ``a`` with the same row of ``b`` (``[S, V]`` -> ``[S]``)."""
    a = a.detach().float()
    b = b.detach().float()
    if a.shape != b.shape:
        raise ValueError(f"row_pcc shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    ac = a - a.mean(-1, keepdim=True)
    bc = b - b.mean(-1, keepdim=True)
    return ((ac * bc).sum(-1) / (ac.norm(dim=-1) * bc.norm(dim=-1) + 1e-30)).clamp_(-1.0, 1.0)


@dataclass
class LogitsReport:
    """Per-position comparison of device logits with the HF reference logits (both ``[S, V]``)."""

    pcc: torch.Tensor  # [S] logits PCC over the vocab
    top1_match: torch.Tensor  # [S] bool: argmax equal
    topk_overlap: torch.Tensor  # [S] |top-k(ref) & top-k(dev)| / k
    top1_in_ref_topk: torch.Tensor  # [S] bool: the device argmax is one of HF's top-k
    kl: torch.Tensor  # [S] KL(HF || device) in nats
    ref_top1: torch.Tensor  # [S] HF argmax ids
    dev_top1: torch.Tensor  # [S] device argmax ids
    ref_topk: torch.Tensor  # [S, k] HF top-k ids (descending)
    dev_topk: torch.Tensor  # [S, k] device top-k ids (descending)
    k: int = 5

    @property
    def n(self) -> int:
        return int(self.pcc.numel())

    @property
    def mean_pcc(self) -> float:
        return float(self.pcc.mean())

    @property
    def min_pcc(self) -> float:
        return float(self.pcc.min())

    @property
    def top1_rate(self) -> float:
        return float(self.top1_match.float().mean())

    @property
    def topk_rate(self) -> float:
        return float(self.topk_overlap.mean())

    @property
    def top1_in_ref_topk_rate(self) -> float:
        return float(self.top1_in_ref_topk.float().mean())

    @property
    def mean_kl(self) -> float:
        return float(self.kl.mean())

    @property
    def max_kl(self) -> float:
        return float(self.kl.max())

    def summary(self, label: str = "logits") -> str:
        return (
            f"{label}: S={self.n} logits pcc mean {self.mean_pcc:.6f} min {self.min_pcc:.6f} (position {int(self.pcc.argmin())}); "
            f"top-1 agreement {self.top1_rate * 100:.1f} % ({int(self.top1_match.sum())}/{self.n}); "
            f"top-{self.k} overlap {self.topk_rate * 100:.1f} %; device top-1 in HF top-{self.k} {self.top1_in_ref_topk_rate * 100:.1f} %; "
            f"KL(HF||dev) mean {self.mean_kl:.5f} max {self.max_kl:.5f} nats (position {int(self.kl.argmax())}); "
            f"last position: pcc {float(self.pcc[-1]):.6f} HF top-1 {int(self.ref_top1[-1])} device top-1 {int(self.dev_top1[-1])}"
        )

    def rows(self, every: int = 1) -> list[str]:
        """One line per position (every ``every``-th, plus every top-1 mismatch) for the test log."""
        out = []
        for t in range(self.n):
            miss = not bool(self.top1_match[t])
            if t % every and not miss and t != self.n - 1:
                continue
            out.append(
                f"pos {t:4d}: pcc {float(self.pcc[t]):.6f} top-{self.k} overlap {float(self.topk_overlap[t]):.2f} "
                f"kl {float(self.kl[t]):.5f} HF top-{self.k} {self.ref_topk[t].tolist()} device top-{self.k} "
                f"{self.dev_topk[t].tolist()}{'  <-- top-1 differs' if miss else ''}"
            )
        return out

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "k": self.k,
            "mean_pcc": self.mean_pcc,
            "min_pcc": self.min_pcc,
            "top1_rate": self.top1_rate,
            "topk_rate": self.topk_rate,
            "top1_in_ref_topk_rate": self.top1_in_ref_topk_rate,
            "mean_kl": self.mean_kl,
            "max_kl": self.max_kl,
            "pcc": [round(float(v), 6) for v in self.pcc],
            "top1_match": [bool(v) for v in self.top1_match],
            "topk_overlap": [round(float(v), 3) for v in self.topk_overlap],
            "kl": [round(float(v), 6) for v in self.kl],
            "ref_top1": [int(v) for v in self.ref_top1],
            "dev_top1": [int(v) for v in self.dev_top1],
        }


def compare_logits(dev: torch.Tensor, ref: torch.Tensor, k: int = 5) -> LogitsReport:
    """Device logits ``[S, V]`` vs HF reference logits ``[S, V]`` (any float dtype; compared in fp32)."""
    dev = dev.detach().float()
    ref = ref.detach().float()
    if dev.shape != ref.shape:
        raise ValueError(f"logits shape mismatch: device {tuple(dev.shape)} vs reference {tuple(ref.shape)}")
    if torch.isnan(dev).any() or torch.isinf(dev).any():
        raise ValueError("device logits contain nan/inf")
    pcc = row_pcc(ref, dev)
    ref_topk = ref.topk(k, dim=-1).indices
    dev_topk = dev.topk(k, dim=-1).indices
    ref_top1, dev_top1 = ref_topk[:, 0], dev_topk[:, 0]
    top1_match = ref_top1 == dev_top1
    overlap = (ref_topk[:, :, None] == dev_topk[:, None, :]).any(-1).float().sum(-1) / k
    top1_in = (ref_topk == dev_top1[:, None]).any(-1)
    log_p = torch.log_softmax(ref, dim=-1)
    log_q = torch.log_softmax(dev, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return LogitsReport(pcc, top1_match, overlap, top1_in, kl, ref_top1, dev_top1, ref_topk, dev_topk, k)


def dump_last_layer(dump_dir: Path) -> int | None:
    """The layer whose ``x_out`` the dump's ``logits.pt`` came from (``prompt_meta.json``), else ``None``."""
    meta = Path(dump_dir) / META_FILE
    if not meta.exists():
        return None
    v = json.loads(meta.read_text()).get("logits_layer")
    return None if v is None else int(v)


def reference_logits(
    dump_dir: Path,
    last_layer: int,
    norm_w: torch.Tensor,
    lm_w: torch.Tensor,
    eps: float = RMS_NORM_EPS,
    S: int | None = None,
) -> tuple[torch.Tensor, str]:
    """HF reference logits ``[S, V]`` for the chain ending at ``last_layer``: the dump's ``logits.pt`` when it
    belongs to that layer, else the host head on the dump's ``layer{last_layer}_prefill.pt`` ``x_out`` (a partial
    chain, or a dump without logits). Returns ``(logits, source)``."""
    dump_dir = Path(dump_dir)
    f = dump_dir / LOGITS_FILE
    if f.exists() and dump_last_layer(dump_dir) == last_layer:
        logits = torch.load(f)
        return (logits if S is None else logits[:S]).float(), str(f)
    x_out = torch.load(dump_dir / f"layer{last_layer}_prefill.pt")["x_out"][0]  # [S, 4, H] bf16
    x_out = x_out if S is None else x_out[:S]
    return host_head(x_out, norm_w, lm_w, eps), f"host head on {dump_dir / f'layer{last_layer}_prefill.pt'} x_out"


def logits_vs_dump(
    streams: torch.Tensor,
    dump_dir: Path,
    last_layer: int,
    *,
    eps: float = RMS_NORM_EPS,
    snapshot: Path | None = None,
    k: int = 5,
    head_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[LogitsReport, torch.Tensor, torch.Tensor, str]:
    """The device test's one call: last-layer streams ``[S, 4, H]`` -> ``(report, device logits, HF logits,
    reference source)``, the HF side from the dump for the chain ending at ``last_layer``."""
    norm_w, lm_w = head_weights or load_head_weights(snapshot)
    S = int(streams.shape[0])
    ref, source = reference_logits(dump_dir, last_layer, norm_w, lm_w, eps, S=S)
    dev = host_head(streams, norm_w, lm_w, eps)
    return compare_logits(dev, ref, k=k), dev, ref, source
