# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""DBCache for the FLUX 3 Klein DiT (the ``cache-dit`` algorithm, https://github.com/vipshop/cache-dit, re-expressed
for the Klein single stage; off unless a request asks for it).

Per step the first ``Fn_compute_blocks`` single blocks always run. Their residual ``F = h_Fn - h_in`` is compared
with the previous computed step's ``F`` by the cache-dit rule ``mean|F - F_prev| / mean|F_prev| < residual_diff_threshold``
(sums over every device's shard, so it is the global mean). The rule sees the CONTENT rows only (image and image_cond):
the diffusers transformer that cache-dit hooks keeps the text tokens in a separate ``encoder_hidden_states`` stream, so
its ``hidden_states`` residual never contains them, while the Klein single stage here is one joint sequence
``[image | image_cond | text | pad]``. The text rows' residual is far larger per row than the image rows' and nearly
constant across steps, so including it halves the ratio and roughly doubles the cached steps (CPU-reference probe
``bin/probes/probe_cache_diffs_cpu.py``, 2026-10-23). ``content_rows_only=False`` (spec token ``all``) keeps the
whole sequence for comparison. When the rule holds and the step is past the warmup and
inside the cached-step caps, the middle blocks are SKIPPED: ``h = h_Fn + R`` where ``R`` is the middle blocks'
residual ``h_Mn - h_Fn`` cached on the last computed step (or TaylorSeer's extrapolation of it), then the last
``Bn_compute_blocks`` blocks run. Otherwise every block runs and ``F`` and ``R`` are refreshed. The cache-dit
FLUX.3 adapter caches exactly this stack (the per-mode early stacks are recomputed every step), with
``forward_pattern`` Pattern_1 and no encoder stream (the Klein single stage is one joint sequence).

The TaylorSeer calibrator (``taylorseer=True``, ``taylorseer_order`` n) keeps a ladder of finite-difference
derivatives of ``R`` over the computed steps and evaluates ``R(t) = sum_i d^iR/dt^i * elapsed^i / i!`` on a cached
step, as cache-dit's ``TaylorSeerCalibrator``.

The whole thing is EAGER only (a skipped step is a different program sequence; the denoiser refuses a traced run
with a cache attached) and it is an approximation by construction: it is gated by the PSNR tests, not bit equality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import ttnn

from ...utils import tensor


@dataclass(frozen=True)
class Flux3KleinCacheConfig:
    """cache-dit's ``DBCacheConfig`` names and defaults (``F8B0``, 8 warmup steps, unlimited cached steps)."""

    Fn_compute_blocks: int = 8
    Bn_compute_blocks: int = 0
    residual_diff_threshold: float = 0.08
    max_warmup_steps: int = 8
    max_cached_steps: int = -1
    max_continuous_cached_steps: int = -1
    taylorseer: bool = False
    taylorseer_order: int = 1
    content_rows_only: bool = True  # the rule over image + image_cond rows (cache-dit's stream), not text / pad

    @classmethod
    def from_spec(cls, spec: str | dict | None) -> "Flux3KleinCacheConfig | None":
        """``None`` / ``""`` -> no cache. A dict of field names, or a string ``"F8B0:0.08:w8"`` (``F<n>B<m>``, the
        threshold, ``w<warmup>``, ``c<max_cached>``, ``k<max_continuous>``, ``ts`` / ``ts2`` for TaylorSeer order 1/2,
        ``all`` for the rule over every row including text)."""
        if spec is None or spec == "" or spec is False:
            return None
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, dict):
            return cls(**spec)
        kw: dict = {}
        for part in str(spec).split(":"):
            p = part.strip()
            if not p:
                continue
            if p[0] == "F" and "B" in p:
                f, b = p[1:].split("B")
                kw["Fn_compute_blocks"], kw["Bn_compute_blocks"] = int(f), int(b)
            elif p == "all":
                kw["content_rows_only"] = False
            elif p.startswith("ts"):
                kw["taylorseer"] = True
                kw["taylorseer_order"] = int(p[2:] or 1)
            elif p[0] == "w":
                kw["max_warmup_steps"] = int(p[1:])
            elif p[0] == "c":
                kw["max_cached_steps"] = int(p[1:])
            elif p[0] == "k":
                kw["max_continuous_cached_steps"] = int(p[1:])
            else:
                kw["residual_diff_threshold"] = float(p)
        return cls(**kw)

    def tag(self) -> str:
        return (
            f"F{self.Fn_compute_blocks}B{self.Bn_compute_blocks}_{self.residual_diff_threshold:g}_w{self.max_warmup_steps}"
            + (f"_ts{self.taylorseer_order}" if self.taylorseer else "")
            + ("" if self.content_rows_only else "_all")
        )


@dataclass
class Flux3KleinCacheStats:
    steps: int = 0
    cached_steps: list = field(default_factory=list)
    diffs: list = field(default_factory=list)  # (step, diff) for every step with a previous F


def _dealloc(t) -> None:
    if t is not None:
        ttnn.deallocate(t)


class Flux3KleinCache:
    """One request's DBCache state over a DiT's single-block stack (one instance per CFG branch)."""

    def __init__(self, config: Flux3KleinCacheConfig, mesh_device: ttnn.MeshDevice):
        self.config = config
        self.mesh_device = mesh_device
        self.stats = Flux3KleinCacheStats()
        self.step = -1
        self._F_prev = None  # previous computed step's Fn residual
        self._R = None  # middle-block residual of the last computed step
        self._ladder: list = []  # TaylorSeer: [R, dR/dt, d2R/dt2, ...] of the last computed step (level 0 is self._R)
        self._last_computed_step = -1
        self._continuous = 0
        self._masks: dict = {}  # (total_padded, content_rows, sp_axis) -> [1, total_padded, 1] bf16 column, 1 on content rows

    # --- per request / step ---------------------------------------------------------------------------------------
    def begin_request(self) -> None:
        self.release()
        self.stats = Flux3KleinCacheStats()
        self.step = -1
        self._last_computed_step = -1
        self._continuous = 0

    def begin_step(self, index: int) -> None:
        self.step = int(index)
        self.stats.steps = max(self.stats.steps, self.step + 1)

    def release(self) -> None:
        _dealloc(self._F_prev)
        _dealloc(self._R)
        for t in self._ladder[1:]:
            _dealloc(t)
        self._F_prev = self._R = None
        self._ladder = []

    def close(self) -> None:
        self.release()
        for m in self._masks.values():
            ttnn.deallocate(m)
        self._masks = {}

    def _content_mask(self, total_padded: int, content_rows: int, sp_axis: int) -> ttnn.Tensor:
        """``[1, total_padded, 1]`` bf16 (exact 0 / 1), sharded over the sequence like ``h``, 1 on the content rows."""
        key = (int(total_padded), int(content_rows), int(sp_axis))
        if key not in self._masks:
            mask = torch.zeros(1, total_padded, 1, dtype=torch.float32)
            mask[:, :content_rows] = 1.0
            self._masks[key] = tensor.from_torch(
                mask, device=self.mesh_device, dtype=ttnn.bfloat16, mesh_axes=[None, sp_axis, None]
            )
        return self._masks[key]

    # --- the rule -------------------------------------------------------------------------------------------------
    def _global_abs_sum(self, t: ttnn.Tensor) -> float:
        a = ttnn.abs(t)
        a32 = ttnn.typecast(a, ttnn.float32)
        ttnn.deallocate(a)
        s = ttnn.sum(a32)  # a scalar per device (each device holds its own shard)
        ttnn.deallocate(a32)
        total = 0.0
        for dev in ttnn.get_device_tensors(s):
            total += float(ttnn.to_torch(dev).double().sum().item())
        ttnn.deallocate(s)
        return total

    def _can_cache(self, F: ttnn.Tensor) -> bool:
        cfg = self.config
        if self.step < cfg.max_warmup_steps or self._F_prev is None or self._R is None:
            return False
        if cfg.max_cached_steps >= 0 and len(self.stats.cached_steps) >= cfg.max_cached_steps:
            return False
        if cfg.max_continuous_cached_steps >= 0 and self._continuous >= cfg.max_continuous_cached_steps:
            return False
        if cfg.residual_diff_threshold <= 0:
            return False
        d = ttnn.subtract(F, self._F_prev)
        num = self._global_abs_sum(d)
        ttnn.deallocate(d)
        den = self._global_abs_sum(self._F_prev)
        diff = num / den if den > 0 else float("inf")
        self.stats.diffs.append((self.step, diff))
        return diff < cfg.residual_diff_threshold

    # --- the stack --------------------------------------------------------------------------------------------------
    def run(
        self, blocks, h: ttnn.Tensor, call, *, content_rows: int | None = None, total_padded: int | None = None, sp_axis: int = 0
    ) -> ttnn.Tensor:
        """``call(block, h) -> h`` over ``blocks`` with the DBCache skip of the middle blocks. ``h`` is consumed.
        ``content_rows`` / ``total_padded`` (the sequence layout) restrict the rule to the content rows when
        ``config.content_rows_only``."""
        cfg = self.config
        n = len(blocks)
        fn = max(0, min(cfg.Fn_compute_blocks, n))
        bn = max(0, min(cfg.Bn_compute_blocks, n - fn))
        # A block's output is the CCL manager's reduce-scatter PING-PONG buffer (linear.py forward_fused_addcmul):
        # two persistent buffers alternate, so an output is overwritten two blocks later and must never be
        # deallocated here. Anything this cache keeps across blocks is a clone or a fresh elementwise result.
        h_in = h
        for block in blocks[:fn]:
            h = call(block, h)
        F = ttnn.subtract(h, h_in)  # fresh; h_in (the assembled sequence) is the caller's
        if cfg.content_rows_only and content_rows is not None and total_padded is not None and content_rows < total_padded:
            masked = ttnn.multiply(F, self._content_mask(total_padded, content_rows, sp_axis))
            ttnn.deallocate(F)
            F = masked
        if self._can_cache(F):
            ttnn.deallocate(F)
            self.stats.cached_steps.append(self.step)
            self._continuous += 1
            R = self._approximate() if cfg.taylorseer else self._R
            h = ttnn.add(h, R)  # fresh: the Bn blocks / the final layer read it
            if R is not self._R:
                ttnn.deallocate(R)
        else:
            _dealloc(self._F_prev)
            self._F_prev = F
            self._continuous = 0
            h_fn = ttnn.clone(h)  # the ping-pong buffer is reused inside the middle loop
            for block in blocks[fn : n - bn]:
                h = call(block, h)
            R = ttnn.subtract(h, h_fn)
            ttnn.deallocate(h_fn)
            if cfg.taylorseer:
                self._update_ladder(R)  # differences against the old ladder, before the old R goes
            _dealloc(self._R)
            self._R = R
            self._last_computed_step = self.step
        for block in blocks[n - bn :]:
            h = call(block, h)
        return h

    # --- TaylorSeer (finite-difference ladder over computed steps, cache-dit's TaylorSeerCalibrator) -------------
    def _update_ladder(self, R: ttnn.Tensor) -> None:
        """Called on a computed step with the new middle residual ``R`` BEFORE the old ``self._R`` is released:
        level 0 of the ladder is the last computed ``R`` itself, level i the i-th finite difference over the
        computed steps (cache-dit divides by the step window)."""
        window = self.step - self._last_computed_step if self._last_computed_step >= 0 else 1
        cur: list = [R]
        for i in range(self.config.taylorseer_order):
            if i < len(self._ladder) and self._ladder[i] is not None and window > 0:
                d = ttnn.subtract(cur[i], self._ladder[i])
                di = ttnn.multiply(d, 1.0 / window)
                ttnn.deallocate(d)
                cur.append(di)
            else:
                break
        for t in self._ladder[1:]:  # level 0 is the old self._R, released by the caller
            _dealloc(t)
        self._ladder = cur

    def _approximate(self) -> ttnn.Tensor:
        """``R(elapsed) = sum_i ladder[i] * elapsed^i / i!``; returns ``self._R`` itself when no derivative exists
        yet, otherwise a fresh tensor."""
        elapsed = self.step - self._last_computed_step
        if len(self._ladder) < 2:
            return self._R
        out = None
        for i, d in enumerate(self._ladder):
            term = ttnn.multiply(d, float((elapsed**i) / math.factorial(i))) if i > 0 else d
            if out is None:
                out = ttnn.add(term, 0.0)  # a fresh copy of level 0
            else:
                nxt = ttnn.add(out, term)
                ttnn.deallocate(out)
                ttnn.deallocate(term)
                out = nxt
        return out
