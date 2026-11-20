# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Determinism probe (diagnostic, opt-in): which layer and which op first changed between two runs of one chunk.

The K3 determinism test compares the whole KV cache after each iteration, which names a 4-layer MLA window at best.
With a probe installed (`install(...)`), every K3 layer taps its input, its attention output and its FFN/MoE output:
each tap reduces the tensor to fp32 per-token sums (over the hidden dim) and copies them into a persistent slot of
its own. The ops are ordinary device ops, so they are captured into the trace like the rest of the forward and
rerun on every replay. The probe is armed only after one full forward (every MoE block has its persistent L1 arena by then, so the probe's
ops are built around it); the slots are allocated by the next eager forward, before any capture.

After each chunk (the test synchronizes there anyway) `snapshot()` concatenates the slots and reads them back once
(~0.2 MB per chip). The first armed iteration's snapshots are the reference; `compare(it, chunk)` reports the taps that
differ, in execution order. The first differing tap of the first differing chunk is the op that changed: earlier
chunks were identical, so nothing upstream (KV, KDA state) can have carried the change in.

A per-token fp32 sum of ~2K bf16 values moves with any 1-ulp change of an element above ~1e-3 of the row's sum; a
change that small in a tiny element can be absorbed, so "no tap differs" is strong but not absolute evidence.
"""

from __future__ import annotations

from typing import Optional

import torch
from loguru import logger

import ttnn

_PROBE: Optional["DetProbe"] = None


def active() -> Optional["DetProbe"]:
    return _PROBE


def install(mesh_device) -> "DetProbe":
    global _PROBE
    _PROBE = DetProbe(mesh_device)
    return _PROBE


def uninstall():
    global _PROBE
    if _PROBE is not None:
        _PROBE.release()
    _PROBE = None


def tap(layer_idx: int, stage: str, t):
    """Called by the model; a no-op unless a probe is installed."""
    if _PROBE is not None and t is not None:
        _PROBE.tap(layer_idx, stage, t)


class DetProbe:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.slots = {}  # (layer, stage) -> persistent [1, 1, 1, S] fp32 row-major (S local tokens)
        self.order = []  # tap keys in execution order
        self.frozen = False  # set once the slots must not move (before trace capture)
        self.armed = False  # off for the first forward: every MoE block's L1 arena exists before a probe op is built
        self.ref_it = None
        self.ref = {}  # chunk -> [chips, taps, S] reference (first iteration)
        self.first_bad = None  # (it, chunk, report lines)
        self.bad = {}  # it -> number of chunks that differed

    def arm(self):
        self.armed = True

    def freeze(self):
        self.frozen = True

    def tap(self, layer_idx, stage, t):
        if not self.armed:
            return
        key = (layer_idx, stage)
        x = ttnn.typecast(
            t, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # (always a DRAM copy: t may be L1-sharded)
        s = ttnn.sum(x, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [.., S, 1]
        ttnn.deallocate(x)
        st = ttnn.transpose(
            s, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [.., 1, S]: row-major without a 1-element row
        ttnn.deallocate(s)
        flat = ttnn.to_layout(st, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if flat is not st:
            ttnn.deallocate(st)
        slot = self.slots.get(key)
        if slot is None:
            assert not self.frozen, f"det probe: tap {key} first seen after the slots were frozen (under capture?)"
            slot = ttnn.allocate_tensor_on_device(flat.spec, self.mesh_device)
            self.slots[key] = slot
            self.order.append(key)
        ttnn.copy(flat, slot)
        ttnn.deallocate(flat)

    def snapshot(self) -> torch.Tensor:
        """[chips, taps, S] fp32 of the last chunk's taps (eager, between replays)."""
        cat = ttnn.concat(
            [self.slots[k] for k in self.order], dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, 1, taps, S]
        out = torch.stack([ttnn.to_torch(t).reshape(len(self.order), -1) for t in ttnn.get_device_tensors(cat.cpu())])
        ttnn.deallocate(cat)
        return out

    def compare(self, it: int, chunk: int) -> bool:
        """Record (it 0) or compare (later) the chunk's taps. True if any differ."""
        snap = self.snapshot()
        if chunk not in self.ref:  # the first armed iteration is the reference
            self.ref[chunk] = snap
            self.ref_it = it
            return False
        ref = self.ref[chunk]
        diff = snap != ref  # bitwise: NaN != NaN only if a NaN appeared, which is itself a change
        if not bool(diff.any()):
            return False
        self.bad[it] = self.bad.get(it, 0) + 1
        if self.first_bad is None:
            lines = []
            per_tap = diff.any(dim=2)  # [chips, taps]
            taps = [i for i in range(len(self.order)) if bool(per_tap[:, i].any())]
            for i in taps[:12]:
                chips = [c for c in range(diff.shape[0]) if bool(per_tap[c, i])]
                tok = diff[:, i, :].any(dim=0).nonzero().flatten()
                d = (snap[:, i, :] - ref[:, i, :]).abs()
                rel = float(d.max() / ref[:, i, :].abs().max().clamp_min(1e-30))
                L, st = self.order[i]
                lines.append(
                    f"  L{L} {st}: chips {chips}, {int(diff[:, i, :].sum())} token sums differ "
                    f"(local rows {int(tok[0])}..{int(tok[-1])}), max |d| {float(d.max()):.3e} (rel {rel:.2e})"
                )
            if len(taps) > 12:
                lines.append(f"  ... {len(taps) - 12} more taps differ")
            self.first_bad = (it, chunk, lines)
        logger.warning(
            f"[det probe] iteration {it} chunk {chunk}: {int(diff.any(dim=2).any(dim=0).sum())} of {len(self.order)} "
            f"taps differ, first L{self.order[int(diff.any(dim=2).any(dim=0).nonzero()[0])][0]} "
            f"{self.order[int(diff.any(dim=2).any(dim=0).nonzero()[0])][1]}"
        )
        return True

    def report(self) -> str:
        if self.first_bad is None:
            return f"[det probe] all {len(self.order)} taps identical to iteration {self.ref_it} in every chunk of every later iteration"
        it, chunk, lines = self.first_bad
        return (
            f"[det probe] first divergence from iteration {self.ref_it}: iteration {it}, chunk {chunk} (every earlier chunk "
            f"identical); taps in "
            f"execution order:\n"
            + "\n".join(lines)
            + f"\n  iterations with differing chunks: {dict(sorted(self.bad.items()))}"
        )

    def release(self):
        for t in self.slots.values():
            ttnn.deallocate(t)
        self.slots.clear()
