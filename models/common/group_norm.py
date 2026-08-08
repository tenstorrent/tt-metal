# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reusable GroupNorm module (proposal for PR #45292 — built on `main`).

main's `ttnn.group_norm` is already a stateless op: it CONSUMES the masks you pass
(it does not own/cache them). This module is the "user-owned state" half the reviewers
asked for: it creates + holds the masks and passes them to the stateless op, mirroring
what SDXL VAE / unet_3d / oft already do by hand, promoted into one reusable place built
on `LightweightModule`.

negative_mask modes
-------------------
  True  : always use the negative-mask path (create both masks, pass both).
  False : never use it (create only the input mask).
  "auto": create both; at forward() a PYTHON CB-size estimate picks the path.

The "auto" estimate (see `_needs_negative_mask`) replicates main's factory CB formulas
to predict whether the cheaper no-negative-mask path fits in L1. It is only a HINT:
forward() wraps the chosen call in try/except, so if the estimate is wrong the framework's
own L1 validation is the final authority — a wrong guess costs at most one failed
program-build (on a cache miss), never a crash. This keeps the decision visible in Python
and pure-Python (no C++), while staying correct regardless of estimate drift.
"""

import os

import ttnn
from models.common.lightweightmodule import LightweightModule

_TILE_HW = 32
# Tiled (32x32) byte sizes per dtype.
_TILE_BYTES = {
    ttnn.DataType.BFLOAT16: 2048,
    ttnn.DataType.BFLOAT8_B: 1088,
    ttnn.DataType.FLOAT32: 4096,
}
# Approximate usable compute-L1 per core (wormhole), used only as the auto-hint budget.
# The try/except safety net is the real authority, so this need not be exact.
# Overridable via env (handy for tuning the flip point, or for testing the safety net).
_L1_BUDGET_BYTES = int(os.environ.get("GN_L1_BUDGET", "1499136"))


def _tile_bytes(dt):
    return _TILE_BYTES.get(dt, 2048)


def _num_cores_across_channel(core_grid, shard_orientation):
    return core_grid.x if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR else core_grid.y


def _find_max_tile_span(width, group_size, tile_width=_TILE_HW):
    """Port of the factory's find_max_tile_span: worst-case tiles a group spans."""
    pos, span = 0, 0
    while pos < width:
        end = pos + group_size
        span = max(span, (end - 1) // tile_width - pos // tile_width + 1)
        pos = end
    return span


def _is_l1_overflow(err):
    """Did this failure come from circular buffers not fitting in L1?"""
    msg = str(err).lower()
    return ("circular buffer" in msg and "l1" in msg) or "exceeds l1" in msg or "out of memory" in msg


class GroupNorm(LightweightModule):
    """User-owned GroupNorm wrapper: owns the mask(s); passes them to the stateless op."""

    def __init__(
        self,
        num_channels,
        num_groups,
        core_grid,
        device,
        *,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        weight=None,  # torch gamma (optional)
        bias=None,  # torch beta  (optional)
        mask_dtype=ttnn.DataType.BFLOAT8_B,
        negative_mask="auto",  # True | False | "auto"
        num_cores_across_channel=None,  # override (e.g. 1 for HEIGHT_SHARDED, num_virtual_cols for DRAM); else derived from grid
        use_input_mask=True,  # set False to run the no-input-mask path
        use_welford=False,  # default Welford mode (overridable per forward())
        epsilon=1e-5,
    ):
        if negative_mask not in (True, False, "auto"):
            raise ValueError(f"negative_mask must be True, False or 'auto', got {negative_mask!r}")
        if negative_mask in (True, "auto") and not use_input_mask:
            # The negative-mask path overlaps the input-mask CBs; without an input mask the op
            # has nothing to overlap and crashes on device. Fail clearly at construction instead.
            raise ValueError("negative_mask requires use_input_mask=True (the negative-mask path needs an input mask)")

        self.num_channels = num_channels
        self.num_groups = num_groups
        self.core_grid = core_grid
        self.device = device
        self.shard_orientation = shard_orientation
        self.mask_dtype = mask_dtype
        self.epsilon = epsilon
        self._neg_mode = negative_mask
        self._use_welford = use_welford
        self._ncac = (
            num_cores_across_channel
            if num_cores_across_channel is not None
            else _num_cores_across_channel(core_grid, shard_orientation)
        )
        self.last_decision = None  # ("calc"|"fallback"|"forced", used_negative_mask) — for tests

        # mask SIZE is pure config arithmetic (no L1 / device query)
        self.input_mask = (
            ttnn.to_device(
                ttnn.create_group_norm_input_mask(num_channels, num_groups, self._ncac, mask_dtype),
                device,
            )
            if use_input_mask
            else None
        )
        # create the negative mask for True and "auto"; skip for False
        self.negative_mask = self._build_negative_mask() if negative_mask in (True, "auto") else None

        self.gamma = self._prep_weight_bias(weight)
        self.beta = self._prep_weight_bias(bias)

    def _prep_weight_bias(self, w):
        if w is None:
            return None
        return ttnn.from_torch(
            ttnn.create_group_norm_weight_bias_rm(w, self.num_channels, self._ncac),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _build_negative_mask(self):
        return ttnn.to_device(
            ttnn.create_group_norm_input_negative_mask(self.num_channels, self.num_groups, self._ncac, self.mask_dtype),
            self.device,
        )

    def _needs_negative_mask(self, x):
        """Estimate the no-negative-mask CB usage; return True if it likely won't fit L1.
        A hint only — the try/except in forward() is the real safety net.
        Covers the default (non-welford) path. Welford uses different CB sizes — will be added
        later if this proposal is accepted."""
        ss = x.memory_config().shard_spec
        per_core_M, per_core_N = ss.shape[0], ss.shape[1]
        per_core_Mt = per_core_M // _TILE_HW
        per_core_Nt = (per_core_N + _TILE_HW - 1) // _TILE_HW

        # The op runs on a reshaped (N, 1, W*H, C) tensor, so (matching the factory's own naming)
        # "W" here is the CHANNEL dim (shape[3]=C) and "H" is the spatial*batch extent.
        shape = x.padded_shape
        num_batches, W = shape[0], shape[3]
        H = shape[2] * num_batches
        group_size = W // self.num_groups
        num_shards_r = H // per_core_M
        num_batches_per_core = num_batches // num_shards_r if num_batches > num_shards_r else 1
        num_shards_c = W // per_core_N
        num_groups_per_core = self.num_groups // num_shards_c if self.num_groups > num_shards_c else 1

        block_wt = _find_max_tile_span(per_core_N, group_size)
        block_ht = per_core_Mt // num_batches_per_core
        interm_block_tiles = block_ht * block_wt
        in0_block_tiles = per_core_Nt * per_core_Mt

        ts = 2048  # bf16 compute/intermediate format (op-fixed)
        in_ts = _tile_bytes(x.dtype)
        gb_ts = 2048  # gamma/beta bf16
        mask_ts = _tile_bytes(self.mask_dtype)  # dtype-aware (matches main's factory)
        untilize_out = x.layout == ttnn.ROW_MAJOR_LAYOUT
        has_gamma, has_beta = self.gamma is not None, self.beta is not None
        reader_repack = (per_core_N % _TILE_HW) != 0

        in_cb = in0_block_tiles * in_ts
        cb = in_cb  # c_1 tilized input
        cb += in_cb if untilize_out else 0  # c_30 untilize-out copy (no-negative-mask path)
        cb += ts  # c_2 scaler
        cb += ts  # c_3 eps
        cb += ts  # c_4 scaler-c (non-welford)
        if has_gamma:
            cb += per_core_Nt * gb_ts  # c_5
        if has_beta:
            cb += per_core_Nt * gb_ts  # c_6
        cb += block_wt * mask_ts * 2  # c_7 input mask (non-welford double-buffer)
        if reader_repack:
            cb += per_core_Nt * in_ts * 2  # c_11/c_12 repack
        cb += ts * interm_block_tiles  # c_13 x
        cb += ts  # c_8 ex_partial
        cb += ts  # c_10 ex_external (non-welford)
        cb += ts  # c_9/c_15 ex_global (non-welford)
        cb += ts  # c_17 ex2pe (non-welford)
        cb += ts  # c_26 ones

        # input (already L1) + output buffer are the other L1 the op needs alongside the CBs.
        out_bytes = in0_block_tiles * 2048  # bf16 output
        in_bytes = in0_block_tiles * in_ts
        return (cb + out_bytes + in_bytes) > _L1_BUDGET_BYTES

    def _run(
        self,
        x,
        memory_config,
        negative_mask,
        *,
        inplace=None,
        use_welford=False,
        compute_kernel_config=None,
        num_out_blocks=None,
        output_layout=None,
        reciprocals=None,
    ):
        kwargs = dict(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            weight=self.gamma,
            bias=self.beta,
            input_mask=self.input_mask,
            negative_mask=negative_mask,
            memory_config=memory_config,
            core_grid=self.core_grid,
            use_welford=use_welford,
        )
        # Only forward optional args when set, so the op keeps its own defaults otherwise.
        if inplace is not None:
            kwargs["inplace"] = inplace
        if compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        if num_out_blocks is not None:
            kwargs["num_out_blocks"] = num_out_blocks
        if output_layout is not None:
            kwargs["output_layout"] = output_layout
        if reciprocals is not None:
            kwargs["reciprocals"] = reciprocals
        return ttnn.group_norm(x, **kwargs)

    def _oom_message(self, x):
        return (
            f"GroupNorm does not fit in L1 even with the negative-mask path "
            f"(input shape {tuple(x.padded_shape)}, grid {self.core_grid}, "
            f"channels {self.num_channels}, groups {self.num_groups}). "
            f"Reduce L1 pressure: use a larger core grid, a smaller per-core shard, "
            f"or a smaller mask dtype."
        )

    def forward(
        self,
        x,
        memory_config=None,
        *,
        inplace=None,
        use_welford=None,
        compute_kernel_config=None,
        num_out_blocks=None,
        output_layout=None,
        reciprocals=None,
    ):
        mc = memory_config if memory_config is not None else x.memory_config()
        run_kwargs = dict(
            inplace=inplace,
            use_welford=self._use_welford if use_welford is None else use_welford,
            compute_kernel_config=compute_kernel_config,
            num_out_blocks=num_out_blocks,
            output_layout=output_layout,
            reciprocals=reciprocals,
        )

        sharded = x.memory_config().is_sharded()
        if self._neg_mode is True:
            if not sharded:
                raise RuntimeError(
                    "negative_mask=True requires a sharded (block-sharded ROW_MAJOR) input, "
                    "but this input is not sharded."
                )
            self.last_decision = ("forced", True)
            return self._run(x, mc, self.negative_mask, **run_kwargs)
        # No negative mask available (False mode, or DRAM/interleaved where it's unsupported).
        if self._neg_mode is False or self.negative_mask is None or not sharded:
            self.last_decision = ("forced", False)
            return self._run(x, mc, None, **run_kwargs)

        # "auto": the Python CB estimate decides; try/except is the safety net.
        use_neg = self._needs_negative_mask(x)
        self.last_decision = ("calc", use_neg)
        try:
            return self._run(x, mc, self.negative_mask if use_neg else None, **run_kwargs)
        except RuntimeError as e:
            if not _is_l1_overflow(e):
                raise
            # Estimate was wrong (said it fit without the negative mask, but it didn't):
            # fall back to the negative mask. This is the safety net.
            if not use_neg and self.negative_mask is not None:
                self.last_decision = ("fallback", True)
                try:
                    return self._run(x, mc, self.negative_mask, **run_kwargs)
                except RuntimeError as e2:
                    if _is_l1_overflow(e2):
                        raise RuntimeError(self._oom_message(x)) from e2
                    raise
            raise RuntimeError(self._oom_message(x)) from e
