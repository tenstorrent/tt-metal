# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2SpeechEncoder`] — inference, pure device ops in ``forward``."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ttnn
import torch

from models.common.utility_functions import nearest_32
from models.experimental.seamless_m4t_v2_large.tt.common import (
    matmul_program_config,
    pick_largest_height_shard_nhw_cores,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import from_torch_bfloat16_tile


# Match ``torch.finfo(torch.bfloat16).min`` used by HF attention masking.
_BF16_ATTN_MASK_MIN = float(torch.finfo(torch.bfloat16).min)

# Drain on-device profiler markers every N conformer layers when device profiling is on.
_PROFILER_LAYER_DRAIN_INTERVAL = 8
# Short-seq linears use ``MatmulMultiCoreReuseMultiCast1DProgramConfig`` (see ``common.matmul_program_config``).


def _drain_device_profiler(device: ttnn.Device, *, trace_no_profiler: bool) -> None:
    """Flush on-device profiler markers when profiling is enabled."""
    if trace_no_profiler:
        return
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        ttnn.ReadDeviceProfiler(device)


def _conv1d_output_length(in_len: int, *, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> int:
    """Output sequence length for 1-D convolution (matches ``ttnn.conv1d`` / PyTorch)."""
    return (in_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@dataclass
class SpeechEncoderTraceMasks:
    """Prebuilt masks and depthwise left pads for trace-safe ``forward`` (built outside ``begin_trace_capture``)."""

    encoder_additive_4d: Optional[ttnn.Tensor]
    adapter_self_attn_4d: List[Optional[ttnn.Tensor]]
    conv_dw_left_pad: List[Optional[ttnn.Tensor]]


class TTSeamlessM4Tv2SpeechEncoder:
    """
    Device port of Hugging Face ``SeamlessM4Tv2SpeechEncoder`` (Conformer stack + adapter).

    Use ``create_speech_encoder_parameters`` for weights. ``forward`` uses only ``ttnn`` ops;
    host ``numpy`` is used only to build relative-position index tables (no PyTorch ops in
    ``forward``).
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters: Any,
        *,
        hidden_size: int,
        feature_projection_input_dim: int,
        speech_encoder_attention_heads: int,
        speech_encoder_intermediate_size: int,
        speech_encoder_layers: int,
        layer_norm_eps: float,
        speech_encoder_chunk_size: Optional[int],
        speech_encoder_left_chunk_num: int,
    ):
        self.device = device
        self.parameters = parameters
        self.hidden_size = hidden_size
        self.feature_projection_input_dim = feature_projection_input_dim
        self.speech_encoder_attention_heads = speech_encoder_attention_heads
        self.speech_encoder_intermediate_size = speech_encoder_intermediate_size
        self.speech_encoder_layers = speech_encoder_layers
        self.layer_norm_eps = layer_norm_eps
        self.speech_encoder_chunk_size = speech_encoder_chunk_size
        self.speech_encoder_left_chunk_num = speech_encoder_left_chunk_num
        self.has_adapter = parameters.adapter is not None
        # Relative-position index tensors keyed by ``(seq_len, left_max, right_max)``.
        self._rel_pos_idx_cache: dict[Tuple[int, int, int], ttnn.Tensor] = {}
        # Flat uint32 index on CPU for host-side ``torch.embedding`` (avoids device Embeddings JIT).
        self._rel_pos_idx_torch_cache: dict[Tuple[int, int, int], torch.Tensor] = {}
        # ``(batch, seq_len)`` already passed through ``pre_warm`` for this forward lifetime.
        self._runtime_warmed: set[Tuple[int, int]] = set()
        # Stage 4: cache the full embedded position table ``[S, S, head_dim]`` per
        # ``(seq_len, weight_id)`` — eliminates 24× ``ttnn.embedding`` + reshape per forward pass.
        # Stage 7: scale is pre-folded into the cached table and the tensor is stored in L1
        # when it fits (≤ 1 MB), eliminating 24× rel_logits scale-multiply + 24× DRAM reshape.
        self._rel_pos_tab_cache: dict[Tuple[int, int, float], ttnn.Tensor] = {}
        # Stage 4: cache depthwise conv left-pad zero tensors per ``(batch, lp, hidden_size)``
        # — eliminates 24–25 ``ttnn.zeros`` (FillPad) calls per forward pass (~1.1 ms).
        self._dw_left_pad_cache: dict[Tuple[int, int, int], ttnn.Tensor] = {}
        # Stage 5: cache the chunk-attention mask — it depends only on ``(batch, seq_len, dtype)``
        # since ``speech_encoder_chunk_size`` is fixed at construction time.
        self._chunk_attn_mask_cache: dict[Tuple[int, int, int], Optional[ttnn.Tensor]] = {}
        # Stage 7: cache the full encoder additive mask ``[B, 1, S, S]`` keyed by
        # ``(batch, seq_len, conv_mask_buffer_id)`` — same mask reused on every call with a
        # fixed-length input (common in streaming/chunked inference).
        self._encoder_additive_mask_cache: dict[Tuple[int, int, int], Optional[ttnn.Tensor]] = {}
        # ``ttnn.conv1d`` may re-upload preprocessed weights via host writes when raw parameters are
        # passed every call. Cache prepared (device) weights per conv geometry so trace capture
        # (which forbids ``write_shard_to_device``) reuses them — same pattern as SDXL / vadv2.
        self._conv1d_prepared_cache: Dict[Tuple[Any, ...], Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]] = {}
        self._conv_config_cache: Dict[Tuple[Any, ...], ttnn.Conv1dConfig] = {}
        # Stage 1 (TTNN model bringup): LoFi linears, HiFi2 attention matmuls, sharded LN cache.
        self._linear_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._attn_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._layernorm_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._conv_compute_cfg = self._linear_compute_cfg
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._ln_sharded_cache: dict = {}
        self._matmul_pc_cache: dict = {}
        self._sdpa_pc_cache: dict = {}

    @staticmethod
    def _activation_tile_counts(batch: int, seq_len: int, hidden_size: int) -> Tuple[int, int]:
        m_tiles = (batch * seq_len + 31) // 32
        n_tiles = hidden_size // 32
        return m_tiles, n_tiles

    def _build_ln_sharded_config(self, m_tiles: int, n_tiles: int):
        key = (m_tiles, n_tiles)
        cached = self._ln_sharded_cache.get(key)
        if cached is not None:
            return cached

        device_grid = self.device.compute_with_storage_grid_size()
        grid_x = device_grid.x
        while grid_x > 1 and n_tiles % grid_x != 0:
            grid_x -= 1
        block_w = n_tiles // grid_x
        grid_y = min(device_grid.y, m_tiles)
        while grid_y > 1 and m_tiles % grid_y != 0:
            grid_y -= 1
        block_h = m_tiles // grid_y
        subblock_w = min(block_w, 4)
        while block_w % subblock_w != 0:
            subblock_w -= 1

        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            subblock_w=subblock_w,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
            [block_h * 32, block_w * 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED if grid_y == 1 else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
        cached = (memory_config, program_config)
        self._ln_sharded_cache[key] = cached
        return cached

    def _layer_norm_sharded(
        self,
        x: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        eps: float,
        batch: int,
        seq_len: int,
        channel_size: int,
        input_sharded: bool = False,
        output_sharded: bool = False,
    ) -> ttnn.Tensor:
        m_tiles, n_tiles = self._activation_tile_counts(batch, seq_len, channel_size)
        sharded_mem_config, sharded_pc = self._build_ln_sharded_config(m_tiles, n_tiles)
        if input_sharded and ttnn.is_sharded(x):
            x_sharded = x
        else:
            x_sharded = ttnn.to_memory_config(x, sharded_mem_config)
        normed_sharded = ttnn.layer_norm(
            x_sharded,
            weight=weight,
            bias=bias,
            epsilon=eps,
            memory_config=sharded_mem_config,
            program_config=sharded_pc,
            compute_kernel_config=self._layernorm_compute_cfg,
        )
        ttnn.deallocate(x_sharded)
        if output_sharded:
            return normed_sharded
        normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        ttnn.deallocate(normed_sharded)
        return normed

    @staticmethod
    def _linear_token_rows(x: ttnn.Tensor) -> int:
        if len(x.shape) == 4:
            # ``nlp_concat_heads`` output ``[B, 1, S, H]`` — matmul M = B * S.
            return int(x.shape[0]) * int(x.shape[2])
        if len(x.shape) == 3:
            return int(x.shape[0]) * int(x.shape[1])
        if len(x.shape) == 2:
            return int(x.shape[0])
        return int(x.shape[-2])

    def _matmul_program_config(self, token_rows: int, in_dim: int, out_dim: int):
        key = (token_rows, in_dim, out_dim)
        cached = self._matmul_pc_cache.get(key)
        if cached is not None:
            return cached
        cached = matmul_program_config(
            self.device,
            token_rows=token_rows,
            in_dim=in_dim,
            out_dim=out_dim,
        )
        self._matmul_pc_cache[key] = cached
        return cached

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        key = (seq_q, seq_k)
        cached = self._sdpa_pc_cache.get(key)
        if cached is not None:
            return cached

        q_chunk = max(64, min(256, nearest_32(seq_q)))
        k_chunk = max(64, min(256, nearest_32(seq_k)))
        out = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        self._sdpa_pc_cache[key] = out
        return out

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        program_config=None,
        activation: Optional[str] = None,
    ) -> ttnn.Tensor:
        if program_config is None:
            program_config = self._matmul_program_config(
                self._linear_token_rows(x),
                int(weight.shape[-2]),
                int(weight.shape[-1]),
            )
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            activation=activation,
            program_config=program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._linear_compute_cfg,
        )

    def _layer_norm(
        self,
        x: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        eps: float,
        batch: int,
        seq_len: int,
        channel_size: Optional[int] = None,
        use_sharded: bool = True,
        input_sharded: bool = False,
        output_sharded: bool = False,
    ) -> ttnn.Tensor:
        ch = self.hidden_size if channel_size is None else channel_size
        n_tiles = ch // 32
        # Sharded LN needs enough N-tiles to spread across the grid (see text encoder).
        if use_sharded and n_tiles >= 8:
            return self._layer_norm_sharded(
                x,
                weight=weight,
                bias=bias,
                eps=eps,
                batch=batch,
                seq_len=seq_len,
                channel_size=ch,
                input_sharded=input_sharded,
                output_sharded=output_sharded,
            )
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._layernorm_compute_cfg,
        )

    @staticmethod
    def _tensor_stable_id(t: ttnn.Tensor) -> int:
        # Host tensors may be ``is_allocated()`` but still reject ``buffer_address()``.
        if t.is_allocated() and t.storage_type() == ttnn.StorageType.DEVICE:
            return int(t.buffer_address())
        return id(t)

    @staticmethod
    def _enumish_cache_key(x: Any) -> Any:
        """Nanobind enums (e.g. ``BufferType``) are not always ``int()``-castable; prefer ``.value``."""
        if x is None:
            return -1
        v = getattr(x, "value", None)
        if isinstance(v, int):
            return v
        if isinstance(x, int):
            return x
        return str(x)

    @staticmethod
    def _is_depthwise_conv(*, in_channels: int, out_channels: int, groups: int) -> bool:
        return groups > 1 and groups == in_channels and groups == out_channels

    def _conv1d_config(
        self,
        *,
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        use_dw_zero_pad: bool = False,
    ) -> ttnn.Conv1dConfig:
        """Per-shape ``Conv1dConfig`` for conformer depthwise and pointwise conv1d.

        Depthwise (k=31) uses height-sharded layout. Do not set ``act_block_h_override`` on
        depthwise paths. When ``nhw_tiles`` divides the compute grid, NHW sharding may use more cores.
        """
        key = (batch, input_length, in_channels, out_channels, kernel_size, groups, use_dw_zero_pad)
        cached = self._conv_config_cache.get(key)
        if cached is not None:
            return cached

        is_depthwise = self._is_depthwise_conv(in_channels=in_channels, out_channels=out_channels, groups=groups)
        conv_kwargs: Dict[str, Any] = dict(
            weights_dtype=ttnn.bfloat8_b,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        if is_depthwise:
            conv_kwargs["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            conv_kwargs["enable_weights_double_buffer"] = False
            conv_kwargs["enable_act_double_buffer"] = False
            if use_dw_zero_pad:
                conv_kwargs["padding_mode"] = ttnn.PaddingMode.Zeros
            nhw_tiles = (batch * input_length + 31) // 32
            nhw_cores = pick_largest_height_shard_nhw_cores(nhw_tiles, self.device)
            if nhw_cores > 2:
                grid = self.device.compute_with_storage_grid_size()
                conv_kwargs["override_sharding_config"] = True
                conv_kwargs["core_grid"] = ttnn.num_cores_to_corerangeset(nhw_cores, grid, row_wise=True)
        else:
            conv_kwargs["shard_layout"] = None
            conv_kwargs["enable_weights_double_buffer"] = True
            conv_kwargs["enable_act_double_buffer"] = True
            if kernel_size > 1 and (input_length > 64 or in_channels >= 512):
                conv_kwargs["act_block_h_override"] = 32

        conv_config = ttnn.Conv1dConfig(**conv_kwargs)
        self._conv_config_cache[key] = conv_config
        return conv_config

    @staticmethod
    def _conv1d_padding_key(padding: int | Tuple[int, int]) -> Tuple[int, ...]:
        if isinstance(padding, int):
            return (padding,)
        return (int(padding[0]), int(padding[1]))

    def _conv1d_prepared_cache_key(
        self,
        *,
        weight: ttnn.Tensor,
        bias: Optional[ttnn.Tensor],
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | Tuple[int, int],
        groups: int,
        dilation: int,
        x_nlc: Optional[ttnn.Tensor],
    ) -> Tuple[Any, ...]:
        is_depthwise = self._is_depthwise_conv(in_channels=in_channels, out_channels=out_channels, groups=groups)
        if is_depthwise:
            mem_key: Tuple[Any, ...] = ()
        elif x_nlc is None:
            mem_key = (-1, -1)
        else:
            mc = x_nlc.memory_config()
            if mc is None:
                mem_key = (-1, -1)
            else:
                mem_key = (
                    self._enumish_cache_key(mc.buffer_type),
                    self._enumish_cache_key(mc.memory_layout),
                )
        layout_key = -1 if x_nlc is None else self._enumish_cache_key(x_nlc.layout)
        return (
            self._tensor_stable_id(weight),
            self._tensor_stable_id(bias) if bias is not None else 0,
            batch,
            input_length,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            self._conv1d_padding_key(padding),
            groups,
            dilation,
            mem_key,
            layout_key,
        )

    def _prepare_conv1d_weights_on_device(
        self,
        *,
        weight: ttnn.Tensor,
        bias: Optional[ttnn.Tensor],
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | Tuple[int, int],
        groups: int,
        dilation: int = 1,
        use_dw_zero_pad: bool = False,
    ) -> None:
        """Upload depthwise (or other) conv weights once via ``prepare_conv_*`` — no forward conv."""
        cache_key = self._conv1d_prepared_cache_key(
            weight=weight,
            bias=bias,
            batch=batch,
            input_length=input_length,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            x_nlc=None,
        )
        if cache_key in self._conv1d_prepared_cache:
            return

        conv_config = self._conv1d_config(
            batch=batch,
            input_length=input_length,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            use_dw_zero_pad=use_dw_zero_pad,
        )
        if isinstance(padding, int):
            pad_arg: int | Tuple[int, int] = (0, padding)
        else:
            pad_arg = padding

        prep_w = ttnn.prepare_conv_weights(
            weight_tensor=weight,
            input_memory_config=ttnn.L1_MEMORY_CONFIG,
            input_layout=ttnn.TILE_LAYOUT,
            weights_format="OIHW",
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch,
            input_height=1,
            input_width=input_length,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=pad_arg,
            dilation=(1, dilation),
            has_bias=bias is not None,
            groups=groups,
            device=self.device,
            input_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
            conv_config=conv_config,
            compute_config=self._conv_compute_cfg,
        )
        prep_b = None
        if bias is not None:
            prep_b = ttnn.prepare_conv_bias(
                bias_tensor=bias,
                input_memory_config=ttnn.L1_MEMORY_CONFIG,
                input_layout=ttnn.TILE_LAYOUT,
                in_channels=in_channels,
                out_channels=out_channels,
                batch_size=batch,
                input_height=1,
                input_width=input_length,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=pad_arg,
                dilation=(1, dilation),
                groups=groups,
                device=self.device,
                input_dtype=ttnn.bfloat16,
                output_dtype=ttnn.bfloat16,
                conv_config=conv_config,
                compute_config=self._conv_compute_cfg,
            )
        self._conv1d_prepared_cache[cache_key] = (
            ttnn.clone(prep_w, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.clone(prep_b, memory_config=ttnn.DRAM_MEMORY_CONFIG) if prep_b is not None else None,
        )

    def _prewarm_depthwise_conv_weights(self, batch: int, seq_len: int) -> None:
        """Prepare all encoder depthwise weights for this shape (host upload only, no Conv2d forward)."""
        enc = self.parameters.encoder
        for i in range(self.speech_encoder_layers):
            dw = enc.layers[i].conv_module.depthwise_conv
            lp = int(dw.left_pad)
            padding: int | Tuple[int, int] = (lp, 0) if lp > 0 else int(dw.padding)
            self._prepare_conv1d_weights_on_device(
                weight=dw.weight,
                bias=dw.bias,
                batch=batch,
                input_length=seq_len,
                in_channels=int(dw.in_channels),
                out_channels=int(dw.out_channels),
                kernel_size=int(dw.kernel_size),
                stride=int(dw.stride),
                padding=padding,
                groups=int(dw.groups),
                use_dw_zero_pad=lp > 0,
            )

    def _conv1d(
        self,
        x_nlc: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: Optional[ttnn.Tensor],
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | Tuple[int, int],
        groups: int,
        dilation: int = 1,
        use_dw_zero_pad: bool = False,
    ) -> Tuple[ttnn.Tensor, int]:
        conv_config = self._conv1d_config(
            batch=batch,
            input_length=input_length,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            use_dw_zero_pad=use_dw_zero_pad,
        )
        cache_key = self._conv1d_prepared_cache_key(
            weight=weight,
            bias=bias,
            batch=batch,
            input_length=input_length,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            x_nlc=x_nlc,
        )
        cached = self._conv1d_prepared_cache.get(cache_key)
        if cached is not None:
            prep_w, prep_b = cached
            out, out_len = ttnn.conv1d(
                input_tensor=x_nlc,
                weight_tensor=prep_w,
                in_channels=in_channels,
                out_channels=out_channels,
                device=self.device,
                bias_tensor=prep_b,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batch_size=batch,
                input_length=input_length,
                conv_config=conv_config,
                compute_config=self._conv_compute_cfg,
                groups=groups,
                dilation=dilation,
                dtype=ttnn.bfloat16,
                return_output_dim=True,
                return_weights_and_bias=False,
            )
        else:
            packed = ttnn.conv1d(
                input_tensor=x_nlc,
                weight_tensor=weight,
                in_channels=in_channels,
                out_channels=out_channels,
                device=self.device,
                bias_tensor=bias,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batch_size=batch,
                input_length=input_length,
                conv_config=conv_config,
                compute_config=self._conv_compute_cfg,
                groups=groups,
                dilation=dilation,
                dtype=ttnn.bfloat16,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            out, out_len, wb = packed
            if isinstance(wb, (list, tuple)) and len(wb) >= 1:
                prep_w = wb[0]
                prep_b = wb[1] if len(wb) > 1 else None
            else:
                prep_w, prep_b = wb, None
            self._conv1d_prepared_cache[cache_key] = (
                ttnn.clone(prep_w, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                ttnn.clone(prep_b, memory_config=ttnn.DRAM_MEMORY_CONFIG) if prep_b is not None else None,
            )
        out_len = int(out_len)
        if ttnn.is_sharded(out):
            out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        out = ttnn.reshape(out, (batch, out_len, out_channels))
        return out, out_len

    @staticmethod
    def _heads(x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int) -> ttnn.Tensor:
        x = ttnn.reshape(x, (batch, seq, num_heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

    @staticmethod
    def _k_heads_transposed(x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int) -> ttnn.Tensor:
        """Stage 17: extract K directly as ``[B, H, D, S]`` in one permute.

        Standard ``_heads`` produces ``[B, H, S, D]``; the relative-attention path then
        applies a second permute ``(0,1,3,2)`` to get ``[B, H, D, S]`` for ``Q @ K^T``.
        By using permute ``(0, 2, 3, 1)`` here we go ``[B, S, H, D] → [B, H, D, S]``
        in a single kernel launch, eliminating 24 ``kh_t`` permutes per forward pass.
        """
        x = ttnn.reshape(x, (batch, seq, num_heads, head_dim))
        return ttnn.permute(x, (0, 2, 3, 1), memory_config=ttnn.L1_MEMORY_CONFIG)

    def _qkv_heads_slice(
        self,
        hidden_states: ttnn.Tensor,
        attn_module: Any,
        *,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        hsz: int,
        k_transposed: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, None]:
        """Slice Q|K|V then reshape/permute (adapter SDPA and fallback)."""
        pc_qkv = self._matmul_program_config(batch * seq_len, hsz, 3 * hsz)
        qkv = self._linear(hidden_states, attn_module.qkv.weight, attn_module.qkv.bias, program_config=pc_qkv)
        q = ttnn.slice(qkv, [0, 0, 0], [batch, seq_len, hsz], [1, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.slice(qkv, [0, 0, hsz], [batch, seq_len, 2 * hsz], [1, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.slice(qkv, [0, 0, 2 * hsz], [batch, seq_len, 3 * hsz], [1, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        qh = self._heads(q, batch, seq_len, num_heads, head_dim)
        kh = (self._k_heads_transposed if k_transposed else self._heads)(k, batch, seq_len, num_heads, head_dim)
        vh = self._heads(v, batch, seq_len, num_heads, head_dim)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return qh, kh, vh, None

    def _qkv_heads_fused(
        self,
        hidden_states: ttnn.Tensor,
        attn_module: Any,
        *,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        hsz: int,
        k_transposed: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """``split_query_key_value_and_split_heads`` on fused ``[B, S, 3*H]`` QKV matmul output."""
        _ = head_dim
        pc_qkv = self._matmul_program_config(batch * seq_len, hsz, 3 * hsz)
        qkv = self._linear(hidden_states, attn_module.qkv.weight, attn_module.qkv.bias, program_config=pc_qkv)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            transpose_key=k_transposed,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return q, k, v, qkv

    def _qkv_heads(
        self,
        hidden_states: ttnn.Tensor,
        attn_module: Any,
        *,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        hsz: int,
        k_transposed: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
        """QKV linear + head layout.

        Conformer relative attention (``k_transposed=True``) uses the fused TTNN op
        ``split_query_key_value_and_split_heads`` (kernel over ``nlp_create_qkv_heads``).
        Adapter SDPA (``k_transposed=False``) keeps slice+permute for PCC parity.

        When the fused path is used, returns ``qkv_src``; ``_mh_attention`` frees it
        after Q/K/V are consumed.
        """
        if not k_transposed:
            return self._qkv_heads_slice(
                hidden_states,
                attn_module,
                batch=batch,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                hsz=hsz,
                k_transposed=k_transposed,
            )
        return self._qkv_heads_fused(
            hidden_states,
            attn_module,
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            hsz=hsz,
            k_transposed=k_transposed,
        )

    def _relative_position_index_flat_torch(self, seq_len: int, *, left_max: int, right_max: int) -> torch.Tensor:
        """CPU flat uint32 distance indices ``[S*S]`` for host ``torch.embedding``."""
        idx_key = (seq_len, left_max, right_max)
        cached = self._rel_pos_idx_torch_cache.get(idx_key)
        if cached is not None:
            return cached
        r = np.arange(seq_len, dtype=np.int64)
        l = np.arange(seq_len, dtype=np.int64)
        dist = r[np.newaxis, :] - l[:, np.newaxis]
        dist = np.clip(dist, -left_max, right_max) + left_max
        cached = torch.from_numpy(dist.astype(np.int64).reshape(-1))
        self._rel_pos_idx_torch_cache[idx_key] = cached
        return cached

    def _relative_embedding_table(
        self,
        seq_len: int,
        *,
        distance_weight: ttnn.Tensor,
        left_max: int,
        right_max: int,
        scale: float,
    ) -> ttnn.Tensor:
        """Return pre-scaled ``[S, head_dim, S]`` relative position table for batched matmul.

        Cached per ``(seq_len, weight_id, scale)``. Layout is ``pos[s_q, d, s_k]`` so
        ``ttnn.bmm`` over query index ``s_q`` computes relative logits without a 5-D
        ``reshape + multiply + sum`` on activations.
        """
        weight_id = self._tensor_stable_id(distance_weight)
        tab_key = (seq_len, weight_id, scale)
        cached_tab = self._rel_pos_tab_cache.get(tab_key)
        if cached_tab is not None:
            return cached_tab

        head_dim = self.hidden_size // self.speech_encoder_attention_heads
        idx_flat = self._relative_position_index_flat_torch(seq_len, left_max=left_max, right_max=right_max)
        w_cpu = to_torch_replicated_first_shard(distance_weight).to(torch.bfloat16).contiguous()
        emb_cpu = (
            torch.nn.functional.embedding(idx_flat, w_cpu)
            .reshape(seq_len, seq_len, head_dim)
            .permute(0, 2, 1)
            .contiguous()
        )
        emb = from_torch_bfloat16_tile(self.device, emb_cpu, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Fold scale into the table when non-trivial (stage 7 / stage 8 compatibility).
        if scale != 1.0:
            emb_scaled = ttnn.multiply(emb, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(emb)
            emb = emb_scaled

        # Move to L1 when the tensor fits (≤ 1 MB): [S, D, S] × 2 bytes.
        _L1_POS_TAB_LIMIT = 1 * 1024 * 1024
        if seq_len * seq_len * head_dim * 2 <= _L1_POS_TAB_LIMIT:
            emb_l1 = ttnn.to_memory_config(emb, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(emb)
            emb = emb_l1

        self._rel_pos_tab_cache[tab_key] = emb
        return emb

    @staticmethod
    def _relative_logits_bmm(
        q: ttnn.Tensor,
        pos_bmm: ttnn.Tensor,
        *,
        batch: int,
        num_heads: int,
        seq_len: int,
        memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        """``einsum('bhld,lrd->bhlr', q, pos)`` via ``bmm`` over query positions (no 5-D broadcast)."""
        head_dim = int(q.shape[-1])
        q_bh = ttnn.reshape(q, (batch * num_heads, seq_len, head_dim))
        q_sid = ttnn.permute(q_bh, (1, 0, 2), memory_config=memory_config)
        ttnn.deallocate(q_bh)
        rel_sid = ttnn.operations.moreh.bmm(q_sid, pos_bmm, memory_config=memory_config)
        ttnn.deallocate(q_sid)
        rel_bh = ttnn.permute(rel_sid, (1, 0, 2), memory_config=memory_config)
        ttnn.deallocate(rel_sid)
        return ttnn.reshape(rel_bh, (batch, num_heads, seq_len, seq_len))

    def _mh_attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_module: Any,
        attention_mask_4d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
        use_relative: bool,
    ) -> ttnn.Tensor:
        num_heads = self.speech_encoder_attention_heads
        head_dim = self.hidden_size // num_heads
        hsz = self.hidden_size
        scale = 1.0 / math.sqrt(head_dim)

        token_m = batch * seq_len
        pc_out = self._matmul_program_config(token_m, hsz, hsz)
        # Stage 17: for relative attention, K is extracted directly as [B, H, D, S]
        # (k_transposed=True), eliminating the downstream kh_t = permute(k,(0,1,3,2)).
        q, k, v, qkv_src = self._qkv_heads(
            hidden_states,
            attn_module,
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            hsz=hsz,
            k_transposed=use_relative,
        )

        if not use_relative:
            # Stage 3c: adapter self-attn has no relative positions — use fused SDPA.
            # K is in [B, H, S, D] form (k_transposed=False) as SDPA requires.
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask_4d,
                is_causal=False,
                scale=scale,
                program_config=self._sdpa_program_config(seq_len, seq_len),
                compute_kernel_config=self._sdpa_compute_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        else:
            # Stage 4: scores [B, H, S, S] — L1 fits for seq<=256 (1×16×256×256×2B = 2MB).
            # Larger sequences fall back to DRAM to avoid L1 spills.
            scores_mc = ttnn.L1_MEMORY_CONFIG if seq_len <= 256 else ttnn.DRAM_MEMORY_CONFIG

            # Stage 17: k is already [B, H, D, S] — no permute needed.
            # Stage 8: Q weights are pre-scaled by 1/√head_dim during preprocessing so
            # scores = pre_scaled_q @ k^T is already the correct scaled dot product.
            scores = ttnn.matmul(
                q,
                k,
                memory_config=scores_mc,
                compute_kernel_config=self._attn_compute_cfg,
            )
            ttnn.deallocate(k)

            # Stage 8: Q is pre-scaled by 1/√head_dim, so pos_tab uses scale=1.0 here —
            # the einsum over the pre-scaled Q already provides the single correct factor.
            pos_bmm = self._relative_embedding_table(
                seq_len,
                distance_weight=attn_module.distance_embedding.weight,
                left_max=int(attn_module.left_max_position_embeddings),
                right_max=int(attn_module.right_max_position_embeddings),
                scale=1.0,
            )
            rel_logits = self._relative_logits_bmm(
                q,
                pos_bmm,
                batch=batch,
                num_heads=num_heads,
                seq_len=seq_len,
                memory_config=scores_mc,
            )
            ttnn.deallocate(q)
            scores = ttnn.add(scores, rel_logits, memory_config=scores_mc)
            ttnn.deallocate(rel_logits)

            if attention_mask_4d is not None:
                scores = ttnn.add(scores, attention_mask_4d, memory_config=scores_mc)

            probs = ttnn.softmax(scores, dim=-1, numeric_stable=True, memory_config=scores_mc)
            ttnn.deallocate(scores)

            attn_out = ttnn.matmul(
                probs,
                v,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self._attn_compute_cfg,
            )
            ttnn.deallocate(probs)
            ttnn.deallocate(v)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        out = self._linear(
            merged_4d,
            attn_module.linear_out.weight,
            attn_module.linear_out.bias,
            program_config=pc_out,
        )
        ttnn.deallocate(merged_4d)
        if len(out.shape) == 4:
            out_3d = ttnn.reshape(out, (batch, seq_len, hsz))
            ttnn.deallocate(out)
            out = out_3d
        if qkv_src is not None:
            ttnn.deallocate(qkv_src)
        return out

    def _glu_last_dim(self, x: ttnn.Tensor, *, batch: int, seq_len: int, width: int) -> ttnn.Tensor:
        half = width // 2
        # Stage 9: explicit L1 destination — conv1d output is DRAM; pulling slices into
        # L1 means sigmoid and mul below operate on L1 tensors (faster than DRAM reads).
        a = ttnn.slice(x, [0, 0, 0], [batch, seq_len, half], [1, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        b = ttnn.slice(x, [0, 0, half], [batch, seq_len, width], [1, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        sig = ttnn.sigmoid(b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(b)
        out = ttnn.mul(a, sig, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(sig)
        ttnn.deallocate(a)
        return out

    def _conformer_ffn(self, x: ttnn.Tensor, ffn: Any, *, token_rows: int) -> ttnn.Tensor:
        hdim = self.hidden_size
        ff_dim = self.speech_encoder_intermediate_size
        pc1 = self._matmul_program_config(token_rows, hdim, ff_dim)
        pc2 = self._matmul_program_config(token_rows, ff_dim, hdim)
        h = self._linear(
            x,
            ffn.intermediate_dense.weight,
            ffn.intermediate_dense.bias,
            program_config=pc1,
            activation="silu",
        )
        out = self._linear(h, ffn.output_dense.weight, ffn.output_dense.bias, program_config=pc2)
        ttnn.deallocate(h)
        return out

    def _relu_ffn(self, x: ttnn.Tensor, ffn: Any, *, token_rows: int) -> ttnn.Tensor:
        hdim = self.hidden_size
        ff_dim = self.speech_encoder_intermediate_size
        pc1 = self._matmul_program_config(token_rows, hdim, ff_dim)
        pc2 = self._matmul_program_config(token_rows, ff_dim, hdim)
        h = self._linear(
            x,
            ffn.intermediate_dense.weight,
            ffn.intermediate_dense.bias,
            program_config=pc1,
            activation="relu",
        )
        out = self._linear(h, ffn.output_dense.weight, ffn.output_dense.bias, program_config=pc2)
        ttnn.deallocate(h)
        return out

    def _conv_module(
        self,
        hidden: ttnn.Tensor,
        cm: Any,
        conv_mask_1d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
        hidden_size: int,
        prebuilt_dw_left_pad: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        h = self._layer_norm(
            hidden,
            weight=cm.layer_norm.weight,
            bias=cm.layer_norm.bias,
            eps=float(cm.layer_norm.eps),
            batch=batch,
            seq_len=seq_len,
        )
        if conv_mask_1d is not None:
            m = ttnn.reshape(conv_mask_1d, (batch, seq_len, 1))
            h = ttnn.mul(h, m, memory_config=ttnn.L1_MEMORY_CONFIG)

        pc1 = cm.pointwise_conv1
        h, t1 = self._conv1d(
            h,
            weight=pc1.weight,
            bias=pc1.bias,
            batch=batch,
            input_length=seq_len,
            in_channels=pc1.in_channels,
            out_channels=pc1.out_channels,
            kernel_size=pc1.kernel_size,
            stride=pc1.stride,
            padding=pc1.padding,
            groups=pc1.groups,
        )
        h = self._glu_last_dim(h, batch=batch, seq_len=t1, width=pc1.out_channels)

        _ = prebuilt_dw_left_pad  # trace API: pads unused when halo applies zero left pad
        _ = hidden_size
        dw = cm.depthwise_conv
        lp = int(dw.left_pad)
        dw_padding: int | Tuple[int, int] = (lp, 0) if lp > 0 else int(dw.padding)
        h, t2 = self._conv1d(
            h,
            weight=dw.weight,
            bias=dw.bias,
            batch=batch,
            input_length=t1,
            in_channels=dw.in_channels,
            out_channels=dw.out_channels,
            kernel_size=dw.kernel_size,
            stride=dw.stride,
            padding=dw_padding,
            groups=dw.groups,
            dilation=1,
            use_dw_zero_pad=lp > 0,
        )
        dln = cm.depthwise_layer_norm
        h = self._layer_norm(h, weight=dln.weight, bias=dln.bias, eps=float(dln.eps), batch=batch, seq_len=t2)
        h = ttnn.silu(h, memory_config=ttnn.L1_MEMORY_CONFIG)

        pc2 = cm.pointwise_conv2
        h, _ = self._conv1d(
            h,
            weight=pc2.weight,
            bias=pc2.bias,
            batch=batch,
            input_length=t2,
            in_channels=pc2.in_channels,
            out_channels=pc2.out_channels,
            kernel_size=pc2.kernel_size,
            stride=pc2.stride,
            padding=pc2.padding,
            groups=pc2.groups,
        )
        return h

    def _encoder_additive_mask(
        self,
        conv_mask_1d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
        dtype: ttnn.DataType,
    ) -> tuple[Optional[ttnn.Tensor], bool]:
        """Return ``(mask, owned)`` where ``owned`` is always ``False`` — the cache owns
        the tensor for the model lifetime and callers must NOT deallocate it.
        """
        # Stage 7: cache by (batch, seq_len, mask_buffer_id) for fixed-length inference.
        mask_id = self._tensor_stable_id(conv_mask_1d) if conv_mask_1d is not None else -1
        cache_key = (batch, seq_len, mask_id)
        if cache_key in self._encoder_additive_mask_cache:
            return self._encoder_additive_mask_cache[cache_key], False

        # Each entry is ``(tensor, owned)`` — ``chunk_bad`` is borrowed from
        # ``_chunk_attn_mask_cache`` and must NOT be deallocated by the merge loop,
        # otherwise the next call with a different ``conv_mask_1d`` hits a freed tensor.
        bad_parts: list[tuple[ttnn.Tensor, bool]] = []
        if conv_mask_1d is not None:
            m = ttnn.reshape(conv_mask_1d, (batch, 1, 1, seq_len))
            one = ttnn.ones(m.shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            inv = ttnn.subtract(one, m, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(one)
            # Stage 7: replace S-copy concat with a single ttnn.repeat — avoids allocating
            # S individual row tensors and the O(S²) intermediate concat output separately.
            pad_bad = ttnn.repeat(inv, [1, 1, seq_len, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(inv)
            bad_parts.append((pad_bad, True))

        chunk_bad = self._chunk_attention_mask_float01(batch, seq_len, dtype)
        if chunk_bad is not None:
            bad_parts.append((chunk_bad, False))

        if not bad_parts:
            self._encoder_additive_mask_cache[cache_key] = None
            return None, False

        bad, bad_owned = bad_parts[0]
        for extra, extra_owned in bad_parts[1:]:
            s = ttnn.add(bad, extra, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if bad_owned:
                ttnn.deallocate(bad)
            if extra_owned:
                ttnn.deallocate(extra)
            cap = ttnn.ones(s.shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            bad = ttnn.minimum(s, cap, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(s)
            ttnn.deallocate(cap)
            bad_owned = True

        out = ttnn.multiply(bad, _BF16_ATTN_MASK_MIN, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if bad_owned:
            ttnn.deallocate(bad)
        self._encoder_additive_mask_cache[cache_key] = out
        # Cache owns the tensor; return owned=False so callers never deallocate.
        return out, False

    def _chunk_attention_mask_float01(self, batch: int, seq_len: int, dtype: ttnn.DataType) -> Optional[ttnn.Tensor]:
        """Return chunk-causal float01 mask ``[B, 1, S, S]``, cached per ``(batch, seq_len, dtype)``."""
        cs = self.speech_encoder_chunk_size
        if cs is None:
            return None
        # Inference seq (≪ chunk_size) lies in a single chunk → mask is all-zero; skip build/merge.
        if seq_len <= cs:
            return None
        cache_key = (batch, seq_len, self._enumish_cache_key(dtype))
        cached = self._chunk_attn_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        lc = self.speech_encoder_left_chunk_num
        chunk_np = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
        chunk_indices = np.arange(seq_len, dtype=np.int64) // cs
        start_indices = np.zeros(seq_len, dtype=np.int64)
        if lc >= 0:
            start_indices = np.clip(chunk_indices - lc, 0, None) * cs
        end_indices = np.minimum((chunk_indices + 1) * cs, seq_len)
        idx_cols = np.arange(seq_len, dtype=np.int64)
        for qi in range(seq_len):
            bad = (idx_cols < start_indices[qi]) | (idx_cols >= end_indices[qi])
            chunk_np[0, 0, qi, bad] = 1.0
        chunk_host = torch.from_numpy(chunk_np).to(torch.bfloat16)
        chunk_tt = from_torch_bfloat16_tile(self.device, chunk_host, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if batch == 1:
            self._chunk_attn_mask_cache[cache_key] = chunk_tt
            return chunk_tt
        out = ttnn.repeat(chunk_tt, [batch, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(chunk_tt)
        self._chunk_attn_mask_cache[cache_key] = out
        return out

    def _expand_attention_mask_2d_to_4d(self, mask_2d: ttnn.Tensor, *, batch: int, s: int) -> ttnn.Tensor:
        """HF ``AttentionMaskConverter._expand_mask`` — ``mask`` is 1 keep, 0 pad."""
        m = ttnn.reshape(mask_2d, (batch, 1, 1, s))
        one = ttnn.ones(m.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        inv = ttnn.subtract(one, m, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(one)
        inv_pos = ttnn.gt(inv, 0.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        neg = ttnn.full(
            inv.shape, _BF16_ATTN_MASK_MIN, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        zero = ttnn.zeros(inv.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        add = ttnn.where(inv_pos, neg, zero, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(inv_pos)
        ttnn.deallocate(neg)
        ttnn.deallocate(zero)
        ttnn.deallocate(inv)
        # Stage 13b: replace concat([add]*s, dim=2) with repeat — avoids S intermediate
        # tensor references and the S-input concat dispatch.
        # SDPA requires the attn_mask to live in DRAM (kernel assertion).
        out = ttnn.repeat(add, [1, 1, s, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(add)
        return out

    def _adapter_subsample_lengths(
        self, conv_mask_1d: ttnn.Tensor, *, kernel: int, stride: int, pad: int
    ) -> ttnn.Tensor:
        """Per-batch subsampled lengths after strided conv (HF adapter)."""
        s = ttnn.sum(conv_mask_1d, dim=1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        two_pad = float(2 * pad)
        padded = ttnn.add(s, two_pad, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(s)
        num = ttnn.subtract(padded, float(kernel), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(padded)
        scaled = ttnn.divide(num, float(stride), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(num)
        one = ttnn.ones(scaled.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        out = ttnn.add(scaled, one, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(scaled)
        ttnn.deallocate(one)
        return ttnn.floor(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _adapter_new_attention_mask(self, seq_len_out: int, seq_lens: ttnn.Tensor, *, batch: int) -> ttnn.Tensor:
        """``[B, seq_len_out]`` with 1 for valid positions (approximate floor parity via BF16)."""
        idx = ttnn.arange(0, seq_len_out, 1, device=self.device, dtype=ttnn.bfloat16)
        idx = ttnn.reshape(idx, (1, seq_len_out))
        lens = ttnn.reshape(seq_lens, (batch, 1))
        ok = ttnn.lt(idx, lens, memory_config=ttnn.L1_MEMORY_CONFIG)
        one = ttnn.ones(ok.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        zero = ttnn.zeros(ok.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        mask = ttnn.where(ok, one, zero, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ok)
        ttnn.deallocate(one)
        ttnn.deallocate(zero)
        ttnn.deallocate(idx)
        return mask

    def _conformer_encoder_layer(
        self,
        hidden: ttnn.Tensor,
        layer: Any,
        attention_mask_4d: Optional[ttnn.Tensor],
        conv_mask_1d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
        prebuilt_dw_left_pad: Optional[ttnn.Tensor] = None,
        input_sharded: bool = False,
    ) -> Tuple[ttnn.Tensor, bool]:
        hsz = self.hidden_size
        token_m = batch * seq_len

        if input_sharded:
            res = ttnn.sharded_to_interleaved(hidden, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        else:
            res = hidden
        h = self._layer_norm(
            hidden,
            weight=layer.ffn1_layer_norm.weight,
            bias=layer.ffn1_layer_norm.bias,
            eps=self.layer_norm_eps,
            batch=batch,
            seq_len=seq_len,
            input_sharded=input_sharded,
        )
        ff = self._conformer_ffn(h, layer.ffn1, token_rows=token_m)
        ttnn.deallocate(h)
        # Stage 13a: 0.5 scale folded into output_dense weights at preprocessing time.
        hidden = ttnn.add(res, ff, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ff)
        ttnn.deallocate(res)

        res = hidden
        h = self._layer_norm(
            hidden,
            weight=layer.self_attn_layer_norm.weight,
            bias=layer.self_attn_layer_norm.bias,
            eps=self.layer_norm_eps,
            batch=batch,
            seq_len=seq_len,
        )
        attn = self._mh_attention(
            h,
            layer.self_attn,
            attention_mask_4d,
            batch=batch,
            seq_len=seq_len,
            use_relative=True,
        )
        ttnn.deallocate(h)
        hidden = ttnn.add(res, attn, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        ttnn.deallocate(res)

        res = hidden
        conv = self._conv_module(
            hidden,
            layer.conv_module,
            conv_mask_1d,
            batch=batch,
            seq_len=seq_len,
            hidden_size=hsz,
            prebuilt_dw_left_pad=prebuilt_dw_left_pad,
        )
        hidden = ttnn.add(res, conv, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(conv)
        ttnn.deallocate(res)

        res = hidden
        h = self._layer_norm(
            hidden,
            weight=layer.ffn2_layer_norm.weight,
            bias=layer.ffn2_layer_norm.bias,
            eps=self.layer_norm_eps,
            batch=batch,
            seq_len=seq_len,
        )
        ff2 = self._conformer_ffn(h, layer.ffn2, token_rows=token_m)
        ttnn.deallocate(h)
        # Stage 13a: 0.5 scale folded into output_dense weights at preprocessing time.
        hidden = ttnn.add(res, ff2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ff2)
        ttnn.deallocate(res)

        return (
            self._layer_norm(
                hidden,
                weight=layer.final_layer_norm.weight,
                bias=layer.final_layer_norm.bias,
                eps=self.layer_norm_eps,
                batch=batch,
                seq_len=seq_len,
                output_sharded=True,
            ),
            True,
        )

    def _adapter_layer(
        self,
        hidden: ttnn.Tensor,
        layer: Any,
        conv_mask_1d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
        prebuilt_self_attn_4d: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        use_prebuilt_attn = prebuilt_self_attn_4d is not None

        res_branch = self._layer_norm(
            hidden,
            weight=layer.residual_layer_norm.weight,
            bias=layer.residual_layer_norm.bias,
            eps=self.layer_norm_eps,
            batch=batch,
            seq_len=seq_len,
        )
        rc = layer.residual_conv
        res_out, lens_r = self._conv1d(
            res_branch,
            weight=rc.weight,
            bias=rc.bias,
            batch=batch,
            input_length=seq_len,
            in_channels=rc.in_channels,
            out_channels=rc.out_channels,
            kernel_size=rc.kernel_size,
            stride=rc.stride,
            padding=rc.padding,
            groups=rc.groups,
        )
        ttnn.deallocate(res_branch)
        res_out = self._glu_last_dim(res_out, batch=batch, seq_len=lens_r, width=rc.out_channels)

        h = self._layer_norm(
            hidden,
            weight=layer.self_attn_layer_norm.weight,
            bias=layer.self_attn_layer_norm.bias,
            eps=self.layer_norm_eps,
            batch=batch,
            seq_len=seq_len,
        )
        sc = layer.self_attn_conv
        attn_in, lens_a = self._conv1d(
            h,
            weight=sc.weight,
            bias=sc.bias,
            batch=batch,
            input_length=seq_len,
            in_channels=sc.in_channels,
            out_channels=sc.out_channels,
            kernel_size=sc.kernel_size,
            stride=sc.stride,
            padding=sc.padding,
            groups=sc.groups,
        )
        ttnn.deallocate(h)
        attn_in = self._glu_last_dim(attn_in, batch=batch, seq_len=lens_a, width=sc.out_channels)
        assert lens_a == lens_r

        if use_prebuilt_attn:
            attn_4d = prebuilt_self_attn_4d
        elif conv_mask_1d is not None:
            sub_lens = self._adapter_subsample_lengths(
                conv_mask_1d, kernel=layer.kernel_size, stride=layer.stride, pad=layer.kernel_size // 2
            )
            mask_2d = self._adapter_new_attention_mask(lens_a, sub_lens, batch=batch)
            ttnn.deallocate(sub_lens)
            attn_4d = self._expand_attention_mask_2d_to_4d(mask_2d, batch=batch, s=lens_a)
            ttnn.deallocate(mask_2d)
        else:
            attn_4d = None

        attn = self._mh_attention(
            attn_in,
            layer.self_attn,
            attn_4d,
            batch=batch,
            seq_len=lens_a,
            use_relative=False,
        )
        ttnn.deallocate(attn_in)
        if attn_4d is not None and not use_prebuilt_attn:
            ttnn.deallocate(attn_4d)

        out = ttnn.add(attn, res_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        ttnn.deallocate(res_out)

        res2 = out
        h2 = self._layer_norm(
            out,
            weight=layer.ffn_layer_norm.weight,
            bias=layer.ffn_layer_norm.bias,
            eps=self.layer_norm_eps,
            batch=batch,
            seq_len=lens_a,
        )
        ff = self._relu_ffn(h2, layer.ffn, token_rows=batch * lens_a)
        ttnn.deallocate(h2)
        return ttnn.add(res2, ff, memory_config=ttnn.L1_MEMORY_CONFIG)

    def warm_relative_position_caches_for_seq_lens(self, seq_lens: list[int]) -> None:
        """Populate ``_rel_pos_tab_cache`` and ``_rel_pos_idx_cache`` for each seq len.

        Conformer self-attention always uses ``scale=1.0`` (Stage 8: Q weights are
        pre-scaled during preprocessing). Cached tables are ``[S, D, S]`` for ``bmm``.
        """
        enc = self.parameters.encoder
        for slen in seq_lens:
            for i in range(self.speech_encoder_layers):
                sa = enc.layers[i].self_attn
                self._relative_embedding_table(
                    slen,
                    distance_weight=sa.distance_embedding.weight,
                    left_max=int(sa.left_max_position_embeddings),
                    right_max=int(sa.right_max_position_embeddings),
                    scale=1.0,
                )

    def _ensure_runtime_caches(self, batch: int, seq_len: int) -> None:
        """One-time per ``(batch, seq_len)`` warmup before the conformer stack runs."""
        key = (batch, seq_len)
        if key in self._runtime_warmed:
            return
        self.pre_warm(batch, seq_len)
        self._runtime_warmed.add(key)

    def pre_warm(self, batch: int, seq_len: int) -> None:
        """Pre-populate shape-dependent caches for ``(batch, seq_len)`` before the first forward.

        Caches populated:
        * ``_rel_pos_tab_cache``   — ``[S, D, S]`` tables (host embedding + TILE upload)
        * ``_chunk_attn_mask_cache`` — chunk mask (skipped when ``S <= chunk_size``)
        * ``_encoder_additive_mask_cache`` — no-padding-mask entry (``mask_id=-1``)
        * depthwise conv weight prep — ``prepare_conv_weights`` only (no Conv2d forward)
        """
        self._prewarm_depthwise_conv_weights(batch, seq_len)
        self.warm_relative_position_caches_for_seq_lens([seq_len])
        self._chunk_attention_mask_float01(batch, seq_len, ttnn.bfloat16)
        self._encoder_additive_mask(None, batch=batch, seq_len=seq_len, dtype=ttnn.bfloat16)

    def materialize_trace_attention_masks(
        self, conv_mask_1d_bf16: ttnn.Tensor, *, batch: int, seq: int
    ) -> SpeechEncoderTraceMasks:
        """Build conformer + adapter additive masks and depthwise left pads **outside** trace capture."""
        dtype = ttnn.bfloat16
        enc_4d, _enc_4d_owned = self._encoder_additive_mask(conv_mask_1d_bf16, batch=batch, seq_len=seq, dtype=dtype)

        # Depthwise causal left pad is applied inside Conv2d (``PaddingMode.Zeros``); no concat pads.
        pads: List[Optional[ttnn.Tensor]] = [None] * self.speech_encoder_layers

        adapter_masks: List[Optional[ttnn.Tensor]] = []
        cur = seq
        seq_lens_to_warm = {seq}

        if self.has_adapter:
            for layer in self.parameters.adapter.layers:
                rc = layer.residual_conv
                lens_r = _conv1d_output_length(
                    cur,
                    kernel_size=int(rc.kernel_size),
                    stride=int(rc.stride),
                    padding=int(rc.padding),
                )
                if conv_mask_1d_bf16 is not None:
                    sub_lens = self._adapter_subsample_lengths(
                        conv_mask_1d_bf16,
                        kernel=layer.kernel_size,
                        stride=layer.stride,
                        pad=layer.kernel_size // 2,
                    )
                    mask_2d = self._adapter_new_attention_mask(lens_r, sub_lens, batch=batch)
                    ttnn.deallocate(sub_lens)
                    a4 = self._expand_attention_mask_2d_to_4d(mask_2d, batch=batch, s=lens_r)
                    ttnn.deallocate(mask_2d)
                    adapter_masks.append(a4)
                else:
                    adapter_masks.append(None)
                cur = lens_r
                seq_lens_to_warm.add(cur)

        self.warm_relative_position_caches_for_seq_lens(sorted(seq_lens_to_warm))
        return SpeechEncoderTraceMasks(enc_4d, adapter_masks, pads)

    def forward(
        self,
        input_features: ttnn.Tensor,
        *,
        conv_attention_mask_1d: Optional[ttnn.Tensor] = None,
        trace_masks: Optional[SpeechEncoderTraceMasks] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_features: ``[batch, seq, feature_projection_input_dim]`` bfloat16 on device.
            conv_attention_mask_1d: optional ``[batch, seq]`` bfloat16 ``1``/``0`` mask (``1`` = real frame),
                same convention as HF ``attention_mask``.
            trace_masks: when set, use prebuilt conformer/adapter masks and depthwise pads (for trace capture);
                must be produced by ``materialize_trace_attention_masks`` outside ``begin_trace_capture``.

        Returns:
            Last hidden state ``[batch, seq_out, hidden_size]`` (``seq_out`` may differ if adapter subsamples).
        """
        p = self.parameters
        batch = int(input_features.shape[0])
        seq = int(input_features.shape[1])
        token_m = batch * seq

        dtype = ttnn.bfloat16
        if trace_masks is None:
            self._ensure_runtime_caches(batch, seq)
            if conv_attention_mask_1d is not None:
                self._encoder_additive_mask(conv_attention_mask_1d, batch=batch, seq_len=seq, dtype=dtype)

        fp = p.feature_projection
        h = self._layer_norm(
            input_features,
            weight=fp.layer_norm.weight,
            bias=fp.layer_norm.bias,
            eps=float(fp.layer_norm.eps),
            batch=batch,
            seq_len=seq,
            channel_size=self.feature_projection_input_dim,
        )
        pc_feat = self._matmul_program_config(token_m, self.feature_projection_input_dim, self.hidden_size)
        h = self._linear(h, fp.projection.weight, fp.projection.bias, program_config=pc_feat)

        if conv_attention_mask_1d is not None:
            m1 = ttnn.reshape(conv_attention_mask_1d, (batch, seq, 1))
            h = ttnn.mul(h, m1, memory_config=ttnn.L1_MEMORY_CONFIG)

        enc = p.encoder
        if trace_masks is not None:
            attn_4d = trace_masks.encoder_additive_4d
            own_encoder_attn_4d = False
        else:
            attn_4d, own_encoder_attn_4d = self._encoder_additive_mask(
                conv_attention_mask_1d, batch=batch, seq_len=seq, dtype=dtype
            )

        trace_no_profiler = trace_masks is not None
        h_sharded = False
        for i in range(self.speech_encoder_layers):
            layer = enc.layers[i]
            pad_i = trace_masks.conv_dw_left_pad[i] if trace_masks is not None else None
            h, h_sharded = self._conformer_encoder_layer(
                h,
                layer,
                attn_4d,
                conv_attention_mask_1d,
                batch=batch,
                seq_len=seq,
                prebuilt_dw_left_pad=pad_i,
                input_sharded=h_sharded,
            )
            if (i + 1) % _PROFILER_LAYER_DRAIN_INTERVAL == 0:
                _drain_device_profiler(self.device, trace_no_profiler=trace_no_profiler)

        if h_sharded:
            h = ttnn.sharded_to_interleaved(h, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)

        h = self._layer_norm(
            h,
            weight=enc.layer_norm.weight,
            bias=enc.layer_norm.bias,
            eps=self.layer_norm_eps,
            batch=batch,
            seq_len=seq,
        )

        if own_encoder_attn_4d and attn_4d is not None:
            ttnn.deallocate(attn_4d)

        im = p.intermediate_ffn
        # Stage 13a: 0.5 scale folded into intermediate_ffn output_dense weights.
        exp = self._relu_ffn(h, im, token_rows=token_m)
        h = ttnn.add(h, exp, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(exp)

        if self.has_adapter:
            for ai, ad_layer in enumerate(p.adapter.layers):
                pre_a = trace_masks.adapter_self_attn_4d[ai] if trace_masks is not None else None
                h = self._adapter_layer(
                    h,
                    ad_layer,
                    conv_attention_mask_1d,
                    batch=batch,
                    seq_len=int(h.shape[1]),
                    prebuilt_self_attn_4d=pre_a,
                )

        _drain_device_profiler(self.device, trace_no_profiler=trace_no_profiler)

        return self._layer_norm(
            h,
            weight=p.inner_layer_norm.weight,
            bias=p.inner_layer_norm.bias,
            eps=self.layer_norm_eps,
            batch=batch,
            seq_len=int(h.shape[1]),
        )

    def __call__(
        self,
        input_features: ttnn.Tensor,
        *,
        conv_attention_mask_1d: Optional[ttnn.Tensor] = None,
        trace_masks: Optional[SpeechEncoderTraceMasks] = None,
    ) -> ttnn.Tensor:
        return self.forward(
            input_features,
            conv_attention_mask_1d=conv_attention_mask_1d,
            trace_masks=trace_masks,
        )
