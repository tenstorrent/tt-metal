# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DBCache (cache-dit style) step skipping for the Wan 2.2 denoising loop.

The transformer forward is split into ``cache_head`` / ``cache_body`` / ``cache_tail`` (see
``WanTransformer3DModel``). Between head and body the host decides whether the middle blocks can
be skipped by re-applying the residual they produced on the last computed step. Decision logic
lives in :mod:`models.tt_dit.utils.dbcache`; this module owns the device buffers, the (optional)
tracers for the three pieces, and the CFG combination.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.dbcache import DBCacheConfig, DBCacheContext
from models.tt_dit.utils.tensor import from_torch
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from models.tt_dit.models.transformers.wan2_2.transformer_wan import WanTransformer3DModel


@dataclass(frozen=True)
class WanDBCacheConfig:
    """Per-expert DBCache configuration for Wan 2.2's two-stage MoE denoising.

    ``high_noise`` applies to ``transformer`` (timesteps above the boundary), ``low_noise`` to
    ``transformer_2``. Each expert keeps its own cache context, mirroring cache-dit's dual
    ``BlockAdapter`` setup for Wan 2.2.
    """

    high_noise: DBCacheConfig
    low_noise: DBCacheConfig

    @classmethod
    def default(cls, **shared: object) -> WanDBCacheConfig:
        """Default preset: cache-dit's Wan 2.2 example settings (F1B0, threshold 0.08, at most 2
        consecutive cached steps; the high-noise expert warms up for 4 steps and caches at most 8, the
        low-noise expert warms up for 2 and caches at most 20), plus ``cfg_diff_compute_separate=False``
        (the unconditional branch's diff tracks the conditional one to 3 decimals, so it reuses that
        decision and skips one host readback per step).

        Measured with the pipeline's default ``flow_shift=5`` schedule (BH 4x8, 81 frames, 40 steps):
        480p 1.60x denoising speedup at PCC 0.91 against the uncached video, 720p 1.52x at PCC 0.89.
        With ``flow_shift=12`` (the official Wan 2.2 A14B T2V schedule) 0.08 caches the fast-changing
        early high-noise steps and drifts to a different trajectory (PCC ~0.85); use
        ``default(residual_diff_threshold=0.05)`` there for PCC > 0.95 at ~1.2x.

        ``shared`` overrides the common fields.
        """
        base = DBCacheConfig(
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            residual_diff_threshold=0.08,
            max_continuous_cached_steps=2,
            cfg_diff_compute_separate=False,
        ).replace(**shared)
        return cls(
            high_noise=base.replace(max_warmup_steps=4, max_cached_steps=8),
            low_noise=base.replace(max_warmup_steps=2, max_cached_steps=20),
        )

    @classmethod
    def coerce(cls, config: WanDBCacheConfig | DBCacheConfig) -> WanDBCacheConfig:
        if isinstance(config, WanDBCacheConfig):
            return config
        if isinstance(config, DBCacheConfig):
            return cls(high_noise=config, low_noise=config)
        msg = f"expected WanDBCacheConfig or DBCacheConfig, got {type(config)}"
        raise TypeError(msg)

    def for_expert(self, idx: int) -> DBCacheConfig:
        return (self.high_noise, self.low_noise)[idx]


class ResidualForecaster:
    """Stores the Bn residual of computed steps and predicts it for cached steps.

    ``order == 0`` reuses the last computed residual unchanged (plain DBCache). ``order == k``
    is TaylorSeer with ``k`` finite-difference derivatives, mirroring cache-dit's
    ``TaylorSeerState``::

        update(Y, t):   dY[0] = Y;  dY[i+1] = (dY[i] - dY_prev[i]) / (t - t_prev)
        predict(t):     sum_i dY[i] * (t - t_last)^i / i!

    All buffers are allocated up front (``alloc``) so they never overlap trace scratch memory.
    """

    def __init__(self, order: int, alloc: Callable[[], ttnn.Tensor]) -> None:
        self.order = order
        self.cur = [alloc() for _ in range(order + 1)]
        self.prev = [alloc() for _ in range(order + 1)]
        self.reset()

    def reset(self) -> None:
        self.valid_cur = 0  # number of valid entries in `cur` (0 = nothing cached yet)
        self.valid_prev = 0
        self.last_step: int | None = None

    @property
    def has_data(self) -> bool:
        return self.valid_cur > 0

    def update(self, residual: ttnn.Tensor, step: int) -> None:
        """Record the residual of a computed step and refresh the derivative ladder."""
        self.cur, self.prev = self.prev, self.cur
        self.valid_prev = self.valid_cur
        ttnn.copy(residual, self.cur[0])
        valid = 1
        window = step - self.last_step if self.last_step is not None else 0
        for i in range(self.order):
            # cache-dit only forms the i-th derivative once the previous ladder has it.
            if self.valid_prev <= i or window <= 0 or step <= 1:
                break
            d = ttnn.multiply(ttnn.subtract(self.cur[i], self.prev[i]), 1.0 / window)
            ttnn.copy(d, self.cur[i + 1])
            ttnn.deallocate(d)
            valid = i + 2
        self.valid_cur = valid
        self.last_step = step

    def apply(self, spatial: ttnn.Tensor, step: int) -> ttnn.Tensor:
        """Return ``spatial + predicted_residual(step)``."""
        assert self.has_data and self.last_step is not None
        elapsed = step - self.last_step
        out = ttnn.add(spatial, self.cur[0])
        for i in range(1, self.valid_cur):
            term = ttnn.multiply(self.cur[i], float(elapsed**i) / math.factorial(i))
            out = ttnn.add(out, term)
            ttnn.deallocate(term)
        return out


class ExpertCacheRunner:
    """Buffers, tracers and decision context for one Wan expert (transformer)."""

    NUM_BRANCHES = 2  # 0 = conditional, 1 = unconditional

    def __init__(
        self,
        model: WanTransformer3DModel,
        *,
        config: DBCacheConfig,
        name: str,
    ) -> None:
        self.model = model
        self.mesh_device = model.mesh_device
        self.config = config
        self.context = DBCacheContext(config, name=name, num_branches=self.NUM_BRANCHES)

        if config.Fn_compute_blocks + config.Bn_compute_blocks >= len(model.blocks):
            msg = (
                f"Fn_compute_blocks + Bn_compute_blocks ({config.Fn_compute_blocks} + {config.Bn_compute_blocks}) "
                f"must be < num blocks ({len(model.blocks)})"
            )
            raise ValueError(msg)

        # prep_run: the pipeline's own warmup only exercises the un-split forward, so the first
        # traced call of each piece must run once untraced to compile its programs before capture
        # ("Cannot load new binaries during trace capture"). None of the pieces mutate their inputs.
        tracer_kwargs = {"device": self.mesh_device, "prep_run": True, "clone_prep_inputs": False}
        self.head = Tracer(model.cache_head, **tracer_kwargs)
        self.body = Tracer(model.cache_body, **tracer_kwargs)
        self.tail = Tracer(model.cache_tail, **tracer_kwargs)

        # Persistent device buffers, allocated once (before any trace capture) in `ensure_buffers`.
        self.prev_fn_residual: list[ttnn.Tensor | None] = [None] * self.NUM_BRANCHES
        self.forecasters: list[ResidualForecaster | None] = [None] * self.NUM_BRANCHES
        self.velocity: list[ttnn.Tensor | None] = [None] * self.NUM_BRANCHES
        # Private input slots for the shared (cond/uncond) tracers. A Tracer copies each call's
        # inputs into the tensors it saw at capture time, so passing pipeline-owned tensors (e.g. the
        # conditional prompt buffer) directly would let the unconditional branch overwrite them.
        self.spatial_in: ttnn.Tensor | None = None
        self.prompt_in: ttnn.Tensor | None = None
        self.prev_fn_in: ttnn.Tensor | None = None
        # sum(|prev_fn_residual|) per branch, kept on the host (read back when the step computed).
        self.prev_fn_norm: list[float | None] = [None] * self.NUM_BRANCHES
        self._buffer_key: tuple | None = None

        # Optional per-piece timing (WAN_DBCACHE_PROFILE=1): synchronizes after every piece, so it
        # perturbs the total slightly; use it to attribute time, not to benchmark.
        self.profile_enabled = os.environ.get("WAN_DBCACHE_PROFILE", "0") == "1"
        self.profile: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ buffers
    def ensure_buffers(self, spatial_1BNI: ttnn.Tensor, prompt_1BLP: ttnn.Tensor) -> None:
        """Allocate the input slots and per-branch residual / velocity buffers for this geometry.

        Must run before the first traced call so the buffers do not land in memory a captured trace
        treats as scratch (see ``Tracer`` caveats).
        """
        model = self.model
        _, B, n_local, _ = spatial_1BNI.shape
        d_local = model.dim // model.parallel_config.tensor_parallel.factor
        out_dim = model.proj_out.out_features
        order = self.config.taylorseer_order
        key = (
            tuple(spatial_1BNI.shape),
            spatial_1BNI.dtype,
            tuple(prompt_1BLP.shape),
            prompt_1BLP.dtype,
            d_local,
            out_dim,
            model.output_dtype,
            order,
        )
        if key == self._buffer_key:
            return
        if self._buffer_key is not None:
            msg = f"DBCache buffers were allocated for {self._buffer_key}, cannot re-allocate for {key}"
            raise RuntimeError(msg)

        num_residual_buffers = self.NUM_BRANCHES * (1 + 2 * (order + 1))
        logger.info(
            f"[{self.context.name}] allocating DBCache buffers: "
            f"residual [1,{B},{n_local},{d_local}] bf16 x{num_residual_buffers} (taylorseer_order={order})"
        )
        self.spatial_in = self._zeros(tuple(spatial_1BNI.shape), spatial_1BNI.dtype)
        self.prompt_in = self._zeros(tuple(prompt_1BLP.shape), prompt_1BLP.dtype)
        self.prev_fn_in = self._zeros((1, B, n_local, d_local), ttnn.bfloat16)
        for branch in range(self.NUM_BRANCHES):
            self.prev_fn_residual[branch] = self._zeros((1, B, n_local, d_local), ttnn.bfloat16)
            self.forecasters[branch] = ResidualForecaster(
                order, lambda: self._zeros((1, B, n_local, d_local), ttnn.bfloat16)
            )
            self.velocity[branch] = self._zeros((1, B, n_local, out_dim), model.output_dtype)
        self._buffer_key = key

    def _zeros(self, shape: tuple[int, ...], dtype: ttnn.DataType) -> ttnn.Tensor:
        # Replicated across the mesh: every device gets its own zero-initialised shard-sized buffer.
        return from_torch(torch.zeros(shape, dtype=torch.float32), device=self.mesh_device, dtype=dtype)

    # ------------------------------------------------------------------ diff readback
    def read_sums(self, sums_1B12: ttnn.Tensor) -> tuple[float, float]:
        """One mesh-wide readback of the packed head sums: (sum|fn - prev|, sum|fn|) over all shards."""
        if ttnn.using_distributed_env():
            msg = "DBCache residual-diff readback is not implemented for multi-host meshes"
            raise NotImplementedError(msg)
        composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        totals = ttnn.to_torch(sums_1B12, mesh_composer=composer).double().sum(dim=(0, 1, 2))
        return float(totals[0]), float(totals[1])

    def _prev_norm(self, branch: int) -> float:
        """sum(|prev_fn_residual|) for ``branch``; computed on device once if it was never read back."""
        norm = self.prev_fn_norm[branch]
        if norm is None:
            prev = self.prev_fn_residual[branch]
            assert prev is not None
            abs_prev = ttnn.abs(prev)
            local = ttnn.sum(abs_prev, dim=[2, 3], keepdim=True)
            ttnn.deallocate(abs_prev)
            composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            norm = float(ttnn.to_torch(local, mesh_composer=composer).double().sum())
            ttnn.deallocate(local)
            self.prev_fn_norm[branch] = norm
        return norm

    def _tick(self, key: str, t0: float) -> float:
        """Profiling helper: synchronize, record elapsed since ``t0`` under ``key``, return now."""
        if not self.profile_enabled:
            return t0
        ttnn.synchronize_device(self.mesh_device)
        now = time.perf_counter()
        self.profile.setdefault(key, []).append(now - t0)
        return now

    def profile_summary(self) -> dict[str, float]:
        """Mean milliseconds per recorded piece (empty unless WAN_DBCACHE_PROFILE=1)."""
        return {k: 1e3 * sum(v) / len(v) for k, v in self.profile.items() if v}

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Start a new denoising run for this expert."""
        self.context.reset()
        self.prev_fn_norm = [None] * self.NUM_BRANCHES
        self.profile = {}
        for forecaster in self.forecasters:
            if forecaster is not None:
                forecaster.reset()

    def release_traces(self) -> None:
        for tracer in (self.head, self.body, self.tail):
            tracer.release_trace()

    # ------------------------------------------------------------------ step
    def run_step(
        self,
        *,
        spatial_1BNI: ttnn.Tensor,
        prompts_1BLP: list[ttnn.Tensor],
        rope_cos_1HND: ttnn.Tensor,
        rope_sin_1HND: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        N: int,
        timestep: ttnn.Tensor,
        traced: bool,
        gather_output: bool = False,
    ) -> list[ttnn.Tensor]:
        """Run one denoising step for each CFG branch, returning one velocity tensor per branch.

        ``prompts_1BLP[0]`` is the conditional prompt, ``prompts_1BLP[1]`` (if present) the
        unconditional one. Returned tensors are this runner's persistent buffers when more than one
        branch runs (the tracers reuse their output tensors between calls).
        """
        cfg = self.config
        ctx = self.context
        self.ensure_buffers(spatial_1BNI, prompts_1BLP[0])
        ctx.mark_step_begin()

        rope = {"rope_cos_1HND": rope_cos_1HND, "rope_sin_1HND": rope_sin_1HND, "trans_mat": trans_mat}
        multi_branch = len(prompts_1BLP) > 1
        outputs: list[ttnn.Tensor] = []

        assert self.spatial_in is not None and self.prompt_in is not None and self.prev_fn_in is not None
        ttnn.copy(spatial_1BNI, self.spatial_in)
        spatial_1BNI = self.spatial_in

        for branch, branch_prompt_1BLP in enumerate(prompts_1BLP):
            t0 = time.perf_counter() if self.profile_enabled else 0.0
            # Stage this branch's inputs in the shared slots (see `spatial_in` comment).
            ttnn.copy(branch_prompt_1BLP, self.prompt_in)
            prompt_1BLP = self.prompt_in
            ttnn.copy(self.prev_fn_residual[branch], self.prev_fn_in)

            spatial_1BND, fn_residual, sums_1B12, temb_11BD, timestep_proj_1BTD = self.head(
                spatial_1BNI,
                prompt_1BLP,
                rope_cos_1HND,
                rope_sin_1HND,
                trans_mat,
                N,
                timestep,
                self.prev_fn_in,
                num_fn_blocks=cfg.Fn_compute_blocks,
                traced=traced,
            )

            t0 = self._tick("head", t0)

            decision = ctx.gate(branch)
            diff = None
            cur_norm: float | None = None
            if decision == "dynamic":
                diff_sum, cur_norm = self.read_sums(sums_1B12)
                prev_norm = self._prev_norm(branch)
                diff = diff_sum / prev_norm if prev_norm > 0.0 else float("inf")
                use_cache = ctx.decide(branch, diff)
            else:
                use_cache = decision == "cache"
            t0 = self._tick("decide", t0)

            forecaster = self.forecasters[branch]
            assert forecaster is not None
            if use_cache:
                ctx.add_cached_step(branch)
                spatial_1BND = forecaster.apply(spatial_1BND, ctx.current_step)
                t0 = self._tick("apply_cache", t0)
            else:
                spatial_1BND, bn_residual = self.body(
                    spatial_1BND,
                    prompt_1BLP,
                    timestep_proj_1BTD,
                    N,
                    **rope,
                    num_fn_blocks=cfg.Fn_compute_blocks,
                    num_bn_blocks=cfg.Bn_compute_blocks,
                    traced=traced,
                )
                ttnn.copy(fn_residual, self.prev_fn_residual[branch])
                self.prev_fn_norm[branch] = cur_norm  # None if not read this step -> computed lazily
                forecaster.update(bn_residual, ctx.current_step)
                ctx.add_computed_step(branch)
                t0 = self._tick("body", t0)

            logger.debug(
                f"[{ctx.name}] step {ctx.current_step} branch {branch}: "
                f"{'CACHE' if use_cache else 'compute'} (gate={decision}, diff={diff})"
            )

            velocity = self.tail(
                spatial_1BND,
                prompt_1BLP,
                temb_11BD,
                timestep_proj_1BTD,
                N,
                **rope,
                num_bn_blocks=cfg.Bn_compute_blocks,
                gather_output=gather_output,
                traced=traced,
            )
            if multi_branch:
                # The tail tracer hands back the same output tensor every call; park this branch's
                # result in a private buffer before the next branch overwrites it.
                ttnn.copy(velocity, self.velocity[branch])
                velocity = self.velocity[branch]
            outputs.append(velocity)
            self._tick("tail", t0)

        return outputs
