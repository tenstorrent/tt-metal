# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper around ``ttnn.conv1d`` / ``ttnn.conv_transpose2d`` for the VAE.

All convolutions in the Oobleck decoder share the same input layout convention:

    ``[batch, time, channels]`` (row-major)

Conv1d weights are stored in PyTorch's ``[out, in, k]`` layout; ConvTranspose1d
weights are stored in PyTorch's ``[in, out, k]`` layout. Both get fed through
the standard TTNN ``prepare_conv_*`` helpers which expect rank-4 layouts.

For conv_transpose, we use ``ttnn.conv_transpose2d`` with ``H=1`` and treat the
time axis as ``W``. PyTorch's ``ConvTranspose1d`` and TTNN's ``conv_transpose2d``
implement the same formula, so the time-axis output length matches.
"""

from __future__ import annotations

import os

import numpy as np

from .._ttnn import get_ttnn
from ..math_perf_env import (
    ace_step_concat_kwargs,
    ace_step_dense_linear_program_config,
    ace_step_ensure_dram_activation,
    ace_step_init_vae_conv_compute_kernel_config,
    ace_step_reshape_kwargs,
    ace_step_safe_deallocate,
    ace_step_vae_activation_compute_dtype,
    ace_step_vae_activation_storage_dtype,
    ace_step_vae_conv1d_im2col_matmul_enabled,
    ace_step_vae_conv1d_im2col_matmul_program_config,
    ace_step_vae_conv1d_memory_config,
    ace_step_vae_conv_weight_dtype,
    ace_step_vae_host_weight_staging_dtype,
    ace_step_vae_k1_prefer_conv1d_l1,
    ace_step_vae_k7_sharded_output_config,
    ace_step_vae_normalize_activation_output,
    ace_step_vae_synchronize,
)


def _vae_act_block_h_override(*, in_channels: int) -> int:
    """``act_block_h_override`` in ntiles = value / 32; must divide per-core output height in tiles.

    Use 32 (1 ntile) so short conv-transpose windows (1 ntile/core) do not log conv2d_utils warnings.
    """
    _ = in_channels
    return 32


# ``ttnn.conv1d`` im2col probes 640 cores when ``M`` has 640 TILE rows (``T=20480`` at ``B=1``).
_K1_AVOID_CONV1D_M_DIM = 20480
# Chunk long k=1 convs so each ``ttnn.conv1d`` sees ``M <= 7680`` (240 M-tiles, Tracy-safe).
_K1_CONV1D_CHUNK_T_DEFAULT = 7680


def _k1_conv1d_chunk_t() -> int:
    try:
        return max(32, int(os.environ.get("ACE_STEP_VAE_K1_CONV1D_CHUNK_T", str(_K1_CONV1D_CHUNK_T_DEFAULT))))
    except ValueError:
        return _K1_CONV1D_CHUNK_T_DEFAULT


def _conv1_wants_tile_output(*, return_tile: bool, return_sharded: bool, use_sharded: bool) -> bool:
    """True when k>7 ``_run_conv1d`` should skip Untilize and return TILE activations.

    ``return_sharded`` without a built sharded config (``use_sharded=False``) falls back to the
    same TILE output contract as ``return_tile=True`` (conv1→snake2 boundary).
    """
    if use_sharded:
        return False
    return return_tile or return_sharded


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl.vae")
    return ttnn


def _to_float32_numpy(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32, copy=False)
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.detach().to(dtype=torch.float32, device="cpu").numpy()
    except Exception:
        pass
    return np.asarray(arr, dtype=np.float32)


class TtConv1d:
    """Owns a ``ttnn.conv1d`` op + its prepared weights/bias for one Conv1d layer."""

    def __init__(
        self,
        *,
        weight_host,
        bias_host,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        device,
        activation_dtype=None,
        weights_dtype=None,
        math_fidelity=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

        self._storage_dtype = ace_step_vae_activation_storage_dtype(ttnn)
        self._compute_dtype = ace_step_vae_activation_compute_dtype(ttnn)
        if activation_dtype is not None:
            self._storage_dtype = activation_dtype
            self._compute_dtype = activation_dtype
        self.weights_dtype = weights_dtype or ace_step_vae_host_weight_staging_dtype(ttnn)
        self._vae_conv_perf = True
        # L1 only for 1×1 projections; k>1 uses DRAM (L1 output OOM + static-CB clash on Blackhole).
        self._l1_mem = ace_step_vae_conv1d_memory_config(ttnn, kernel_size=self.kernel_size)
        self._conv_weight_dtype = ace_step_vae_conv_weight_dtype(ttnn, self.weights_dtype, kernel_size=self.kernel_size)

        w = _to_float32_numpy(weight_host)
        if w.ndim != 3 or w.shape[0] != self.out_channels or w.shape[1] != self.in_channels:
            raise ValueError(
                f"Unexpected Conv1d weight shape {w.shape}; "
                f"expected ({self.out_channels}, {self.in_channels}, {self.kernel_size})"
            )
        if int(w.shape[2]) != self.kernel_size:
            raise ValueError(f"Conv1d kernel mismatch: got {w.shape[2]}, expected {self.kernel_size}")

        # Host staging: BF16 ROW_MAJOR (``prepare_conv_weights``); ``conv_config.weights_dtype`` packs BFP8.
        self._weight_host_tt = ttnn.as_tensor(w, dtype=self.weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._bias_np = None
        if bias_host is None:
            self._bias_host_tt = None
            self._has_bias = False
        else:
            b = _to_float32_numpy(bias_host).reshape(1, 1, 1, -1)
            self._bias_np = b
            self._bias_host_tt = ttnn.as_tensor(b, dtype=self.weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            self._has_bias = True

        # 32 = 1 tile height; valid for both long audio and 1-ntile/core conv-transpose tiles.
        _act_block_h = _vae_act_block_h_override(in_channels=self.in_channels)
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=self._conv_weight_dtype,
            shard_layout=None,
            deallocate_activation=bool(self._vae_conv_perf),
            act_block_h_override=_act_block_h,
            config_tensors_in_dram=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        self.compute_config = ace_step_init_vae_conv_compute_kernel_config(device)
        self._packed_for: tuple[int, int] | None = None
        self._weight_dev = None
        self._bias_dev = None
        self._linear_weight_tt = None
        self._linear_bias_tt = None
        # Per-shape cache for k=7 HEIGHT_SHARDED output: True = worked, False = unsupported.
        self._sharded_output_cache: dict = {}
        if self.kernel_size == 1:
            self._init_k1_linear_weights(w, self._bias_np)

    def _init_k1_linear_weights(self, weight_np: np.ndarray, bias_np: np.ndarray | None) -> None:
        """Device TILE weights for ``ttnn.linear`` (no ``prepare_conv_weights`` / im2col probe)."""
        ttnn = self.ttnn
        w2 = np.ascontiguousarray(weight_np.reshape(self.out_channels, self.in_channels).astype(np.float32))
        w_mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        self._linear_weight_tt = ttnn.as_tensor(
            w2,
            device=self.device,
            dtype=self._conv_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w_mem,
        )
        self._linear_bias_tt = None
        if bias_np is not None:
            b = np.ascontiguousarray(bias_np.reshape(1, 1, 1, -1).astype(np.float32))
            self._linear_bias_tt = ttnn.as_tensor(
                b,
                device=self.device,
                dtype=self._conv_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=w_mem,
            )

    def _k1_m_dim(self, batch_size: int, input_length: int) -> int:
        return int(batch_size) * int(input_length)

    def _k1_must_avoid_conv1d(self, m_dim: int) -> bool:
        return self.kernel_size == 1 and int(m_dim) >= _K1_AVOID_CONV1D_M_DIM

    def _k1_activation_memory_config(self, *, force_dram: bool = False):
        ttnn = self.ttnn
        if force_dram or max(self.in_channels, self.out_channels) >= 512:
            return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        return self._input_memory_config()

    def _input_memory_config(self):
        ttnn = self.ttnn
        if self._l1_mem is not None:
            return self._l1_mem
        return ttnn.DRAM_MEMORY_CONFIG

    def _output_memory_config(self):
        ttnn = self.ttnn
        if self._l1_mem is not None:
            return self._l1_mem
        return ttnn.DRAM_MEMORY_CONFIG

    def _maybe_l1(self, x):
        # Place activations in the conv program's expected buffer (L1 for 1×1, DRAM for k>1).
        target = self._l1_mem if self._l1_mem is not None else self.ttnn.DRAM_MEMORY_CONFIG
        if x.memory_config() == target:
            return x
        return self.ttnn.to_memory_config(x, target)

    def _k1_row_major_for_conv1d(self, x):
        """``ttnn.conv1d`` expects ROW_MAJOR activations; keep L1 when untilizing TILE from snake2."""
        ttnn = self.ttnn
        if x.layout == ttnn.ROW_MAJOR_LAYOUT:
            return x
        act_mc = self._input_memory_config()
        kw = {"memory_config": act_mc} if act_mc is not None else {}
        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, **kw)

    def _output_length(self, input_length: int) -> int:
        num = int(input_length) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        return num // self.stride + 1

    def _k1_im2col_matmul_program_config(self, *, batch_size: int, input_length: int):
        if self.kernel_size != 1:
            return None
        m_dim = int(batch_size) * int(input_length)
        return ace_step_vae_conv1d_im2col_matmul_program_config(
            self.device,
            m_dim=m_dim,
            k_dim=self.in_channels,
            n_dim=self.out_channels,
        )

    def _forward_k1_linear(
        self,
        x,
        *,
        batch_size: int,
        input_length: int,
        program_config,
        force_dram: bool = False,
    ):
        """``1×1`` conv as ``ttnn.linear`` — avoids ``prepare_conv_weights`` / conv1d im2col probes."""
        if self._linear_weight_tt is None:
            return None
        ttnn = self.ttnn
        _sr = ace_step_reshape_kwargs(ttnn)
        act_mc = self._k1_activation_memory_config(force_dram=force_dram)
        _til_kw = {"memory_config": act_mc} if act_mc is not None else {}
        if x.layout != ttnn.TILE_LAYOUT:
            x_tile = ttnn.to_layout(x, ttnn.TILE_LAYOUT, **_til_kw)
        elif act_mc is not None and x.memory_config() != act_mc:
            x_tile = ttnn.to_memory_config(x, act_mc)
        else:
            x_tile = x

        linear_kw: dict = {
            "compute_kernel_config": self.compute_config,
            "transpose_b": True,
        }
        if program_config is not None:
            linear_kw["program_config"] = program_config
        if act_mc is not None:
            linear_kw["memory_config"] = act_mc

        try:
            out = ttnn.linear(
                x_tile,
                self._linear_weight_tt,
                bias=self._linear_bias_tt,
                dtype=self._compute_dtype,
                **linear_kw,
            )
        except RuntimeError:
            return None
        finally:
            if self._vae_conv_perf and act_mc is not None and act_mc == self._l1_mem and x_tile is not x:
                try:
                    ttnn.deallocate(x_tile)
                except Exception:
                    pass

        if len(out.shape) == 4:
            out = ttnn.squeeze(out, 1)
        if len(out.shape) == 3 and int(out.shape[1]) != int(input_length):
            out = ttnn.reshape(out, (batch_size, self._output_length(input_length), self.out_channels), **_sr)
        return ace_step_vae_normalize_activation_output(
            ttnn, out, storage_dtype=self._storage_dtype, compute_dtype=self._compute_dtype
        )

    def _forward_k1_tuned_matmul(
        self,
        x,
        *,
        batch_size: int,
        input_length: int,
        force_dram: bool = False,
    ):
        """``1×1`` conv via tuned tall-M ``ttnn.linear`` (Tracy 61440/30720/7680 buckets)."""
        program_config = self._k1_im2col_matmul_program_config(
            batch_size=batch_size,
            input_length=input_length,
        )
        if program_config is None:
            return None
        return self._forward_k1_linear(
            x,
            batch_size=batch_size,
            input_length=input_length,
            program_config=program_config,
            force_dram=force_dram,
        )

    def _forward_k1_dense_matmul(
        self,
        x,
        *,
        batch_size: int,
        input_length: int,
        force_dram: bool = False,
    ):
        program_config = ace_step_dense_linear_program_config(
            self.device,
            seq_len=int(input_length),
            in_dim=self.in_channels,
            out_dim=self.out_channels,
            batch_size=int(batch_size),
        )
        if program_config is None:
            return None
        return self._forward_k1_linear(
            x,
            batch_size=batch_size,
            input_length=input_length,
            program_config=program_config,
            force_dram=force_dram,
        )

    def _forward_k1_via_linear(self, x, *, batch_size: int, input_length: int) -> object | None:
        """Try ``ttnn.linear`` for k=1 (never ``prepare_conv_weights`` on the hot path)."""
        if self.kernel_size != 1 or self._linear_weight_tt is None:
            return None
        m_dim = self._k1_m_dim(batch_size, input_length)
        must_avoid = self._k1_must_avoid_conv1d(m_dim)
        if not ace_step_vae_conv1d_im2col_matmul_enabled() and not must_avoid:
            return None
        for force_dram in (False, True):
            for forward_fn in (self._forward_k1_tuned_matmul, self._forward_k1_dense_matmul):
                out = forward_fn(
                    x,
                    batch_size=batch_size,
                    input_length=input_length,
                    force_dram=force_dram,
                )
                if out is not None:
                    return out
        return None

    def _ensure_packed(self, batch_size: int, input_length: int) -> None:
        ttnn = self.ttnn
        if self._packed_for == (batch_size, input_length) and self._weight_dev is not None:
            return
        input_mem = self._input_memory_config()
        self._weight_dev = ttnn.prepare_conv_weights(
            weight_tensor=self._weight_host_tt,
            input_memory_config=input_mem,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=input_length,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, self.dilation),
            has_bias=self._has_bias,
            groups=1,
            device=self.device,
            input_dtype=self._compute_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
        )
        if self._has_bias:
            self._bias_dev = ttnn.prepare_conv_bias(
                bias_tensor=self._bias_host_tt,
                input_memory_config=input_mem,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                input_height=1,
                input_width=input_length,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
                padding=(0, self.padding),
                dilation=(1, self.dilation),
                device=self.device,
                input_dtype=self._compute_dtype,
                groups=1,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
            )
        self._packed_for = (batch_size, input_length)

    def _run_conv1d(
        self,
        x,
        *,
        batch_size: int,
        input_length: int,
        return_tile: bool = False,
        return_sharded: bool = False,
        _cb_retry: bool = False,
    ):
        """Eager ``ttnn.conv1d`` on ``[B,T,C]`` row-major (prepares weights per shape).

        Args:
            return_tile: Skip the Untilize; return a DRAM ``TILE`` tensor in ``storage_dtype``.
                Used at the k=7 conv1→snake2 boundary to replace the
                ``Untilize(10 µs) + Tilize(12.7 µs)`` round-trip with a single DRAM→L1 DMA
                copy inside snake2's ``to_layout(TILE, L1)`` call.
            return_sharded: Attempt HEIGHT_SHARDED L1 output from ``ttnn.conv1d`` so the
                conv's internal S2I (DRAM write) is replaced by an on-chip L1 scatter.
                Requires ``ACE_STEP_VAE_K7_SHARDED_OUTPUT=1``; falls back to ``return_tile``
                behaviour per shape if the sharded config cannot be built.  Snake2 detects
                the L1 TILE and skips its expensive ``to_layout`` Tilize.
        """
        ttnn = self.ttnn
        _sr = ace_step_reshape_kwargs(ttnn)
        b = int(batch_size)
        t = int(input_length)
        self._ensure_packed(b, t)

        if self.kernel_size > 1:
            dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            ace_step_vae_synchronize(ttnn, self.device)
            x_dram = x
            x = ace_step_ensure_dram_activation(ttnn, x, dram_mc)
            if x is not x_dram:
                ace_step_safe_deallocate(ttnn, x_dram)
            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                kw = {"memory_config": dram_mc} if dram_mc is not None else {}
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, **kw)

        # Determine output memory config.
        out_mc = self._output_memory_config()  # DRAM for k>1 by default
        use_sharded = False
        if return_sharded and not return_tile and self._sharded_output_cache.get((b, t)) is not False:
            out_T = self._output_length(t)
            sharded_mc = ace_step_vae_k7_sharded_output_config(ttnn, self.device, out_T, self.out_channels)
            if sharded_mc is not None:
                out_mc = sharded_mc
                use_sharded = True

        conv_kw: dict = {}
        # k>1 conv2d always emits DRAM interleaved; passing memory_config only warns and can
        # confuse buffer planning on Blackhole (static CB vs activation L1 clash).
        if self.kernel_size == 1 or use_sharded:
            conv_kw["memory_config"] = out_mc

        try:
            ret = ttnn.conv1d(
                input_tensor=x,
                weight_tensor=self._weight_dev,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                device=self.device,
                bias_tensor=self._bias_dev,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                batch_size=b,
                input_length=t,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=1,
                return_output_dim=True,
                return_weights_and_bias=False,
                dtype=self._compute_dtype,
                **conv_kw,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if self.kernel_size > 1 and not _cb_retry and ("circular buffer" in msg or "statically allocated" in msg):
                if use_sharded:
                    self._sharded_output_cache[(b, t)] = False
                ace_step_vae_synchronize(ttnn, self.device)
                return self._run_conv1d(
                    x,
                    batch_size=b,
                    input_length=t,
                    return_tile=True,
                    return_sharded=False,
                    _cb_retry=True,
                )
            raise
        out, out_length = ret
        out = ttnn.squeeze(out, 0)
        out = ttnn.reshape(out, (b, out_length, self.out_channels), **_sr)

        if use_sharded:
            self._sharded_output_cache[(b, t)] = True
            # Cast compute dtype → storage dtype while keeping whatever memory config
            # squeeze/reshape left us in (HEIGHT_SHARDED L1 ideally, or DRAM if TTNN
            # de-sharded during reshape — snake handles both).
            if self._compute_dtype != self._storage_dtype and getattr(out, "dtype", None) != self._storage_dtype:
                try:
                    mc_kw = {"memory_config": out.memory_config()}
                except Exception:
                    mc_kw = {}
                out = ttnn.typecast(out, self._storage_dtype, **mc_kw)
            return out  # HEIGHT_SHARDED L1 TILE (or DRAM TILE if squeeze/reshape de-sharded)

        if _conv1_wants_tile_output(return_tile=return_tile, return_sharded=return_sharded, use_sharded=False):
            # ``return_sharded`` without a sharded config (or ``ACE_STEP_VAE_K7_SHARDED_OUTPUT`` off)
            # falls back here — keep TILE so snake2 avoids Tilize on ROW_MAJOR DRAM.
            # Skip Untilize — keep TILE so snake2's ``to_memory_config(TILE, L1)`` is a DMA copy, not a
            # layout conversion.  Cast compute dtype (BF8) → storage dtype (BF16) in DRAM if
            # needed; this preserves TILE layout and avoids a separate Untilize+Tilize round-trip.
            if self._compute_dtype != self._storage_dtype and getattr(out, "dtype", None) != self._storage_dtype:
                dram = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
                kw = {"memory_config": dram} if dram is not None else {}
                out = ttnn.typecast(out, self._storage_dtype, **kw)
            return out  # DRAM TILE in storage_dtype (or L1 TILE when sharded path de-sharded)

        return ace_step_vae_normalize_activation_output(
            ttnn, out, storage_dtype=self._storage_dtype, compute_dtype=self._compute_dtype
        )

    def _forward_k1_chunked_conv1d(self, x, *, batch_size: int, input_length: int):
        """Slice long k=1 conv along time — each slab stays below 640 M-tile im2col probes."""
        if self.kernel_size != 1:
            return None
        chunk_t = _k1_conv1d_chunk_t()
        total_t = int(input_length)
        if total_t <= chunk_t:
            return None
        ttnn = self.ttnn
        b = int(batch_size)
        parts = []
        start = 0
        while start < total_t:
            end = min(start + chunk_t, total_t)
            slab = ttnn.slice(x, (0, start, 0), (b, end, self.in_channels))
            parts.append(self._run_conv1d(slab, batch_size=b, input_length=end - start))
            start = end
        if len(parts) == 1:
            return parts[0]
        return (
            ttnn.concat(parts, dim=1, **ace_step_concat_kwargs(ttnn))
            if hasattr(ttnn, "concat")
            else ttnn.concatenate(parts, dim=1, **ace_step_concat_kwargs(ttnn))
        )

    def __call__(self, x, *, return_tile: bool = False, return_sharded: bool = False):
        """Run conv1d on a ``[B, T, C]`` activation tensor.

        Args:
            return_tile: Passed through to :meth:`_run_conv1d`.  Only meaningful for
                ``kernel_size > 1`` (k=7 dilated conv in residual units); k=1 paths
                return ROW_MAJOR regardless and ignore this flag.
            return_sharded: Attempt HEIGHT_SHARDED L1 TILE output.  Requires
                ``ACE_STEP_VAE_K7_SHARDED_OUTPUT=1``; silently degrades to
                ``return_tile`` if the sharded config cannot be built for this shape.
                Only meaningful for k>1; ignored on the k=1 fast paths.

        ``kernel_size == 1`` accepts **L1 TILE** rank-3 input (e.g. from ``TtSnake1d`` with
        ``return_tile=True``): ``ttnn.linear`` uses TILE in0 directly; ``ttnn.conv1d`` untilizes
        to ROW_MAJOR L1 once at the API boundary.

        Returns a ``[B, T_out, out_channels]`` row-major tensor, or a TILE tensor when
        ``return_tile=True`` / ``return_sharded=True`` and the k>1 path is taken.
        """
        ttnn = self.ttnn
        if len(x.shape) != 3:
            raise ValueError(f"TtConv1d expects rank-3 [B,T,C], got {x.shape}")
        b = int(x.shape[0])
        t = int(x.shape[1])
        c = int(x.shape[-1])
        if c != self.in_channels:
            raise ValueError(f"Conv1d input channels mismatch: got {c}, expected {self.in_channels}")

        x = self._maybe_l1(x)
        if x.dtype != self._storage_dtype:
            try:
                mc_kw = {"memory_config": x.memory_config()}
            except Exception:
                mc_kw = {"memory_config": self._input_memory_config()}
            x = ttnn.typecast(x, self._storage_dtype, **mc_kw)
        m_dim = self._k1_m_dim(b, t)
        linear_out = None
        # Wide midsize k=1: skip linear (DRAM mcast / L1 OOM on BH) → conv1d L1 (validated E2E).
        prefer_conv1d_l1 = ace_step_vae_k1_prefer_conv1d_l1(
            m_dim=m_dim, k_dim=self.in_channels, n_dim=self.out_channels
        )
        if self.kernel_size == 1 and m_dim < 7680 and not prefer_conv1d_l1:
            linear_out = self._forward_k1_via_linear(x, batch_size=b, input_length=t)
        if linear_out is not None:
            return linear_out
        if self._k1_must_avoid_conv1d(m_dim):
            chunked = self._forward_k1_chunked_conv1d(x, batch_size=b, input_length=t)
            if chunked is not None:
                return chunked
        if self.kernel_size == 1:
            x = self._k1_row_major_for_conv1d(x)
        if self.kernel_size > 1:
            ace_step_vae_synchronize(ttnn, self.device)
        return self._run_conv1d(x, batch_size=b, input_length=t, return_tile=return_tile, return_sharded=return_sharded)


class TtConvTranspose1d:
    """Owns a ``ttnn.conv_transpose2d`` op for one ConvTranspose1d layer.

    PyTorch ``ConvTranspose1d`` weight layout is ``[in, out, k]``; TTNN's
    ``conv_transpose2d`` expects ``[in, out, kH, kW]`` (matching PyTorch's
    ``ConvTranspose2d``), so we add an ``H=1`` axis on host.
    """

    def __init__(
        self,
        *,
        weight_host,
        bias_host,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        device,
        activation_dtype=None,
        weights_dtype=None,
        math_fidelity=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self._storage_dtype = ace_step_vae_activation_storage_dtype(ttnn)
        self._compute_dtype = ace_step_vae_activation_compute_dtype(ttnn)
        if activation_dtype is not None:
            self._storage_dtype = activation_dtype
            self._compute_dtype = activation_dtype
        self.weights_dtype = weights_dtype or ace_step_vae_host_weight_staging_dtype(ttnn)
        self._vae_conv_perf = True
        # k>1 conv-transpose stays DRAM I/O (static CB clash on Blackhole if activations live in L1).
        self._l1_mem = ace_step_vae_conv1d_memory_config(ttnn, kernel_size=self.kernel_size)
        # Conv-transpose kernels are always k>1 in Oobleck (``2 * stride``); use BFP8 weights for BW.
        self._conv_weight_dtype = ace_step_vae_conv_weight_dtype(ttnn, self.weights_dtype, kernel_size=self.kernel_size)

        w = _to_float32_numpy(weight_host)
        if w.ndim != 3 or w.shape[0] != self.in_channels or w.shape[1] != self.out_channels:
            raise ValueError(
                f"Unexpected ConvTranspose1d weight shape {w.shape}; "
                f"expected ({self.in_channels}, {self.out_channels}, {self.kernel_size})"
            )
        if int(w.shape[2]) != self.kernel_size:
            raise ValueError(f"ConvTranspose1d kernel mismatch: got {w.shape[2]}, expected {self.kernel_size}")

        w4 = w.reshape(self.in_channels, self.out_channels, 1, self.kernel_size)
        self._weight_host_tt = ttnn.as_tensor(w4, dtype=self.weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        if bias_host is None:
            self._bias_host_tt = None
            self._has_bias = False
        else:
            b = _to_float32_numpy(bias_host).reshape(1, 1, 1, -1)
            self._bias_host_tt = ttnn.as_tensor(b, dtype=self.weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            self._has_bias = True

        _act_block_h = _vae_act_block_h_override(in_channels=self.in_channels)
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=self._conv_weight_dtype,
            shard_layout=None,
            deallocate_activation=bool(self._vae_conv_perf),
            act_block_h_override=_act_block_h,
            config_tensors_in_dram=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        self.compute_config = ace_step_init_vae_conv_compute_kernel_config(device)
        self._weight_dev = self._weight_host_tt
        self._bias_dev = self._bias_host_tt
        self._uploaded = False

    def _output_memory_config(self):
        ttnn = self.ttnn
        if self._l1_mem is not None:
            return self._l1_mem
        return ttnn.DRAM_MEMORY_CONFIG

    def _maybe_l1(self, x):
        target = self._l1_mem if self._l1_mem is not None else self.ttnn.DRAM_MEMORY_CONFIG
        if x.memory_config() == target:
            return x
        return self.ttnn.to_memory_config(x, target)

    def __call__(self, x):
        """Run conv_transpose2d on a ``[B, T, C]`` row-major tensor.

        Returns ``[B, T_out, out_channels]`` row-major with
        ``T_out = (T-1)*stride - 2*padding + kernel`` (PyTorch formula).
        """
        ttnn = self.ttnn
        if len(x.shape) != 3:
            raise ValueError(f"TtConvTranspose1d expects rank-3 [B,T,C], got {x.shape}")
        b = int(x.shape[0])
        t = int(x.shape[1])
        c = int(x.shape[-1])
        if c != self.in_channels:
            raise ValueError(f"ConvT1d input channels mismatch: got {c}, expected {self.in_channels}")

        x = self._maybe_l1(x)
        if x.dtype != self._storage_dtype:
            x = ttnn.typecast(x, self._storage_dtype, memory_config=self._output_memory_config())
        # TTNN conv_transpose2d expects [B, H, W, C] NHWC; map T -> W with H=1.
        x4 = ttnn.unsqueeze(x, 1)  # [B, 1, T, C]

        out, [self._weight_dev, self._bias_dev] = ttnn.conv_transpose2d(
            input_tensor=x4,
            weight_tensor=self._weight_dev,
            bias_tensor=self._bias_dev,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=b,
            input_height=1,
            input_width=t,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=self.device,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_weights_and_bias=True,
            mirror_kernel=True,
            dtype=self._compute_dtype,
        )
        # conv_transpose2d returns NHWC [B, 1, T_out, out_channels] (rank-4)
        out = ttnn.squeeze(out, 1)
        return ace_step_vae_normalize_activation_output(
            ttnn, out, storage_dtype=self._storage_dtype, compute_dtype=self._compute_dtype
        )
