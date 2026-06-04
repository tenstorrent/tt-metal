# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN ``SeamlessM4Tv2CodeHifiGan`` — unit embeddings, duration predictor, and HiFi-GAN vocoder.

Matches Hugging Face ``modeling_seamless_m4t_v2`` layout. Conv1d uses ``ttnn.conv1d``; ConvTranspose1d
uses ``ttnn.conv_transpose2d`` with ``H=1``. Temporal lengths follow HF conv formulas; ``t_audio``
(one host scalar from duration cumsum) sets the repeat-interleave and HiFi-GAN ``input_length``.
Output waveform lengths are computed on host from ``t_audio`` and uploaded as int32.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.seamless_m4t_v2_large.tt.common import (
    TILE,
    core_grid,
    matmul_program_config,
    MATMUL_1D_SEQ_THRESHOLD,
    to_torch_replicated_first_shard,
)

# Both matmul dims must stay small on BH; ``t_audio`` alone can be thousands of frames.
_VOCODER_EXPAND_MATMUL_CHUNK = TILE
# Max upsampled time for a single ``ttnn.conv1d`` on BH (above this, use fixed-window chunks).
_HIFIGAN_MAX_CONV1D_TLEN = 4096
# Conv1d chunk interior (halo + interior must fit ``_HIFIGAN_MAX_CONV1D_TLEN``).
_VOCODER_CONV1D_INTERIOR = 3968
# Channel-aware chunking. HiFi-GAN halves channels at each upsample, so the late stages (16-64 ch)
# fit far more time rows in the same L1 as the widest conv — yet a fixed-row ``interior`` chunks them
# just as finely (the late, low-channel, long-timeline stages dominate the chunk count and thus the
# ~37k vocoder ops + the O(n²) per-chunk timeline slicing). Size the chunk interior to a constant
# *element* budget (``interior * in_channels``) so low-channel stages chunk much wider. Budget =
# baseline interior × widest HiFi-GAN channel (proven safe at 3968). The result is clamped to
# ``[_VOCODER_CONV1D_INTERIOR, _VOCODER_CONV1D_MAX_INTERIOR]`` (the floor keeps the widest convs
# unchanged; the cap is the conv1d single-shot row ceiling) and tile-aligned. See ``_vocoder_conv1d_chunk_interior``.
_VOCODER_CONV1D_ELEM_BUDGET = 3968 * 512
# 49152 fits conv1d L1_SMALL on BH with ``config_tensors_in_dram`` on chunked timelines (65536 OOMs).
# 32768 was the prior ceiling; raising ~1.5× cuts late (16–32 ch) chunk counts when the element
# budget allows interiors above 32768. Override via ``SEAMLESS_VOCODER_CONV1D_MAX_INTERIOR``.
_VOCODER_CONV1D_MAX_INTERIOR = 49152

# Round ``t_audio`` up to stabilize shape-specialized vocoder programs; padded tail is masked/cropped.
_VOCODER_TAUDIO_BUCKET = 256


def _vocoder_conv1d_max_interior() -> int:
    raw = os.environ.get("SEAMLESS_VOCODER_CONV1D_MAX_INTERIOR")
    if raw is not None:
        return max(_VOCODER_CONV1D_INTERIOR, int(raw))
    return _VOCODER_CONV1D_MAX_INTERIOR


def _vocoder_conv1d_elem_budget() -> int:
    raw = os.environ.get("SEAMLESS_VOCODER_CONV1D_ELEM_BUDGET")
    if raw is not None:
        return max(_VOCODER_CONV1D_ELEM_BUDGET, int(raw))
    return _VOCODER_CONV1D_ELEM_BUDGET


def _vocoder_conv1d_chunk_interior(in_channels: int) -> int:
    """Tile-aligned interior rows for one HiFi-GAN conv1d chunk (channel-aware element budget)."""
    c = max(1, int(in_channels))
    interior = (_vocoder_conv1d_elem_budget() // c // 32) * 32
    return max(_VOCODER_CONV1D_INTERIOR, min(_vocoder_conv1d_max_interior(), interior))


def _vocoder_conv1d_prep_length(timeline_length: int, *, in_channels: int, padding: int) -> Tuple[int, bool]:
    """Return ``(conv input_width, timeline_chunked)`` for prepare/forward on a long mel timeline."""
    seq = int(timeline_length)
    if seq <= _HIFIGAN_MAX_CONV1D_TLEN:
        return seq, False
    fixed_in = _vocoder_conv1d_chunk_interior(in_channels) + 2 * int(padding)
    if seq <= fixed_in:
        return seq, True
    return fixed_in, True


def _vocoder_dram_slice_count(input_length: int) -> int:
    """DRAM height slices for long vocoder upsample timelines on Blackhole."""
    il = int(input_length)
    if il <= MATMUL_1D_SEQ_THRESHOLD:
        return 16
    target_rows = 48
    return min(128, max(16, (il + target_rows - 1) // target_rows))


def _as_batch_time_2d(x: ttnn.Tensor, *, batch: int, seq: int) -> ttnn.Tensor:
    """Normalize ``[B, T]`` or mesh-broadcast ``[B, 1, T]`` to logical ``[B, T]``."""
    rank = len(x.shape)
    if rank == 2:
        return x
    if rank == 3:
        if int(x.shape[1]) == 1:
            return ttnn.reshape(x, (batch, int(x.shape[2])))
        return ttnn.reshape(x, (batch, seq))
    raise RuntimeError(f"expected batch-time rank 2 or 3, got shape {tuple(x.shape)}")


def _slice_batch_time(x: ttnn.Tensor, *, batch: int, seq: int) -> ttnn.Tensor:
    """Slice the logical ``[batch, seq]`` prefix (rank-aware for replicated TP meshes)."""
    rank = len(x.shape)
    if rank == 2:
        return ttnn.slice(x, [0, 0], [batch, seq], (1, 1))
    if rank == 3:
        mid = int(x.shape[1])
        if mid == 1:
            return ttnn.slice(x, [0, 0, 0], [batch, 1, seq], (1, 1, 1))
        return ttnn.slice(x, [0, 0, 0], [batch, seq, int(x.shape[2])], (1, 1, 1))
    raise RuntimeError(f"cannot slice batch-time tensor with shape {tuple(x.shape)}")


def _vocoder_hf_gather_index(input_ids: ttnn.Tensor, *, batch: int, seq: int, pad_id: int) -> int:
    """HF ``_get_dur_output_lengths`` index: ``clamp((input_ids != pad).sum(-1), 0, seq - 1)`` for row 0.

    ``from_device`` may yield a 1-D tensor (length ``batch * seq``); ``ids_t[0]`` would then be a
    single id and the count would be wrong (``t_audio`` stuck near 1 and ~320 output samples).
    """
    ids_t = to_torch_replicated_first_shard(input_ids).to(torch.long).reshape(batch, -1)
    if ids_t.shape[1] > seq:
        ids_t = ids_t[:, :seq]
    count = int((ids_t[0] != pad_id).sum().item())
    return max(0, min(count, seq - 1))


def _fused_relu() -> ttnn.UnaryWithParam:
    """Post-conv fused ReLU (same recipe as T2U ``Conv1dConfig.activation``)."""
    return ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)


def _fused_leaky_relu(negative_slope: float) -> ttnn.UnaryWithParam:
    """Post-conv fused LeakyReLU with HF ``negative_slope`` (block-float param on device)."""
    return ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, float(negative_slope))


def _conv1d_prep_tensor_id(t: ttnn.Tensor) -> int:
    if t.is_allocated() and t.storage_type() == ttnn.StorageType.DEVICE:
        return int(t.buffer_address())
    return id(t)


def _fused_activation_token(activation: Optional[ttnn.UnaryWithParam]) -> str:
    if activation is None:
        return ""
    op = activation.op_type
    if op == ttnn.UnaryOpType.RELU:
        return "relu"
    if op == ttnn.UnaryOpType.LEAKY_RELU:
        raw = getattr(activation, "params", None)
        if raw is None:
            raw = getattr(activation, "param", 0.0)
        slope = float(raw[0] if hasattr(raw, "__getitem__") else raw)
        return f"leaky:{slope}"
    return str(op)


# HEIGHT sharding when K fits; None lets the device auto-pick.
_HEIGHT_SHARD_K_MAX = 4096
_CONV_PRE_SHARD = None
_RESBLOCK_SHARD = None
# Upsample transpose: HEIGHT per DRAM slice when slice count is small enough.
_UPSAMPLE_SHARD = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_UPSAMPLE_HEIGHT_MAX_SLICES = 48


def _resolve_conv_shard_layout(
    prefer: Optional[ttnn.TensorMemoryLayout], *, in_channels: int, kernel_size: int
) -> Optional[ttnn.TensorMemoryLayout]:
    if prefer is None:
        return None
    if prefer == ttnn.TensorMemoryLayout.HEIGHT_SHARDED and int(in_channels) * int(kernel_size) > _HEIGHT_SHARD_K_MAX:
        return None
    return prefer


def _vocoder_conv1d_config(
    fused_post_activation: Optional[ttnn.UnaryWithParam],
    *,
    input_length: int,
    in_channels: int,
    shard_layout: Optional[ttnn.TensorMemoryLayout] = None,
    timeline_chunked: bool = False,
) -> ttnn.Conv1dConfig:
    # Match T2U conv1d L1 recipe: deallocate activations and cap act block height on long/wide ops.
    conv_kwargs: dict = dict(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=shard_layout,
        deallocate_activation=True,
        enable_weights_double_buffer=False,
        enable_act_double_buffer=False,
    )
    if fused_post_activation is not None:
        conv_kwargs["activation"] = fused_post_activation
    if int(input_length) > 64 or int(in_channels) >= 512:
        conv_kwargs["act_block_h_override"] = 32
    # Chunked HiFi-GAN timelines: spill conv indices to DRAM (frees L1_SMALL for wider interiors)
    # and emit ROW_MAJOR activations so chunk stitch avoids per-chunk untilize/tilize (vocoder5 TM).
    if timeline_chunked:
        conv_kwargs["config_tensors_in_dram"] = True
        conv_kwargs["output_layout"] = ttnn.ROW_MAJOR_LAYOUT
    return ttnn.Conv1dConfig(**conv_kwargs)


def _vocoder_conv2d_config(
    *, input_length: int, in_channels: int, shard_layout: Optional[ttnn.TensorMemoryLayout] = None
) -> ttnn.Conv2dConfig:
    conv_kwargs: dict = dict(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=shard_layout,
        deallocate_activation=True,
        output_layout=ttnn.TILE_LAYOUT,
        enable_weights_double_buffer=True,
        enable_act_double_buffer=True,
    )
    # Forced shard layouts may need config tensors in DRAM when L1_SMALL is tight.
    if shard_layout is not None:
        conv_kwargs["config_tensors_in_dram"] = True
    il = int(input_length)
    if il > 256:
        conv_kwargs["enable_weights_double_buffer"] = False
        conv_kwargs["enable_act_double_buffer"] = False
    # ``act_block_h_override`` must be a multiple of TILE (32); 8 → 0 ntiles and SIGFPE in conv2d.
    if il > 64 or int(in_channels) >= 512:
        conv_kwargs["act_block_h_override"] = 32
    return ttnn.Conv2dConfig(**conv_kwargs)


def _host_conv_out_length(n: int, kernel_size: int, stride: int, pad: int, dilation: int = 1) -> int:
    return (n + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1


def _host_transpose_conv_out_length(n: int, kernel_size: int, stride: int, pad: int, dilation: int = 1) -> int:
    return (n - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1


def _host_hifigan_output_length(cfg: Any, unit_length: int) -> int:
    """Port of HF ``_get_output_hifigan_lengths`` (``reference/torch_code_hifigan.py``)."""
    x = int(unit_length)
    x = _host_conv_out_length(x, 7, 1, 3)
    for upsample_rate, kernel_size in zip(cfg.upsample_rates, cfg.upsample_kernel_sizes):
        x = _host_transpose_conv_out_length(x, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2)
    for _ in range(len(cfg.upsample_rates)):
        for kernel_size, dilation in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
            for dil in dilation:
                x = _host_conv_out_length(x, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil)
            for _ in dilation:
                x = _host_conv_out_length(x, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)
    return _host_conv_out_length(x, 7, 1, 3)


@dataclass
class VocoderForwardTraceRuntime:
    """Captured Metal trace for one vocoder ``forward`` (fixed inputs)."""

    trace_id: int
    waveform_tt: ttnn.Tensor
    lengths_tt: ttnn.Tensor


class TTSeamlessM4Tv2CodeHifiGan:
    """Inference forward for HF ``SeamlessM4Tv2CodeHifiGan``."""

    def __init__(self, device: ttnn.Device, parameters: Any, config: Any):
        self.device = device
        self.p = parameters
        self.cfg = config
        self.leaky_slope = float(config.leaky_relu_slope)
        self.num_kernels = len(config.resblock_kernel_sizes)

        # HiFi4 math fidelity for waveform PCC.
        self._compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._frame_idx_cache: dict[int, ttnn.Tensor] = {}
        self._matmul_pc_cache: dict = {}
        # Populated only by ``prewarm_conv1d_weights`` (trace/E2E). Forward without prewarm uses raw weights (PCC path).
        self._conv1d_prepared_cache: Dict[Tuple[Any, ...], Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]] = {}
        self._forward_trace_rt: Optional[VocoderForwardTraceRuntime] = None
        self._last_t_audio: Optional[int] = None

    def _expand_unit_embeddings_matmul(
        self,
        use_BEC: ttnn.Tensor,
        H: ttnn.Tensor,
        *,
        batch: int,
        e_unit: int,
        seq: int,
        t_audio: int,
    ) -> ttnn.Tensor:
        """``use_BEC @ H`` with 2D tiles when ``seq`` or ``t_audio`` exceeds the 1D L1 budget."""
        mm_kwargs: dict = dict(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute,
        )
        if seq <= MATMUL_1D_SEQ_THRESHOLD and t_audio <= MATMUL_1D_SEQ_THRESHOLD:
            pc_expand = self._matmul_pc(batch * e_unit, seq, t_audio)
            if pc_expand is not None:
                mm_kwargs["program_config"] = pc_expand
            return ttnn.matmul(use_BEC, H, **mm_kwargs)

        chunk = _VOCODER_EXPAND_MATMUL_CHUNK
        t_cols: list[ttnn.Tensor] = []
        for t0 in range(0, t_audio, chunk):
            t1 = min(t0 + chunk, t_audio)
            t_span = t1 - t0
            acc: Optional[ttnn.Tensor] = None
            for s0 in range(0, seq, chunk):
                s1 = min(s0 + chunk, seq)
                s_span = s1 - s0
                use_sl = ttnn.slice(
                    use_BEC,
                    [0, 0, s0],
                    [batch, e_unit, s1],
                    (1, 1, 1),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                h_sl = ttnn.slice(
                    H,
                    [0, s0, t0],
                    [batch, s1, t1],
                    (1, 1, 1),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                pc = matmul_program_config(
                    self.device,
                    token_rows=batch * e_unit,
                    in_dim=s_span,
                    out_dim=t_span,
                )
                part = ttnn.matmul(use_sl, h_sl, program_config=pc, **mm_kwargs)
                ttnn.deallocate(use_sl)
                ttnn.deallocate(h_sl)
                if acc is None:
                    acc = part
                else:
                    acc_next = ttnn.add(acc, part, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(acc)
                    ttnn.deallocate(part)
                    acc = acc_next
            assert acc is not None
            t_cols.append(acc)

        if len(t_cols) == 1:
            return t_cols[0]
        return ttnn.concat(t_cols, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _matmul_pc(self, token_rows: int, in_dim: int, out_dim: int) -> Optional[ttnn.ProgramConfig]:
        if token_rows > MATMUL_1D_SEQ_THRESHOLD:
            return None
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

    def _cached_frame_idx_f32(self, t_audio: int) -> ttnn.Tensor:
        key = int(t_audio)
        cached = self._frame_idx_cache.get(key)
        if cached is not None:
            return cached
        frame_idx = ttnn.arange(
            start=0,
            end=key,
            step=1,
            dtype=ttnn.float32,
            device=self.device,
        )
        self._frame_idx_cache[key] = frame_idx
        return frame_idx

    def _output_lengths_dev(self, t_audio: int, *, batch: int) -> ttnn.Tensor:
        """Valid waveform prefix length per row; ``t_audio`` is HF unit length (same as forward)."""
        wav_len = _host_hifigan_output_length(self.cfg, t_audio)
        return ttnn.full(
            (batch,),
            wav_len,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _linear(self, x: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            core_grid=core_grid(self.device),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute,
        )

    def _layer_norm(self, x: ttnn.Tensor, *, weight: ttnn.Tensor, bias: ttnn.Tensor, eps: float) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _conv1d_prep_cache_key(
        self,
        *,
        weight: ttnn.Tensor,
        bias: Optional[ttnn.Tensor],
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        groups: int,
        dilation: int,
        fused_post_activation: Optional[ttnn.UnaryWithParam],
        timeline_chunked: bool = False,
    ) -> Tuple[Any, ...]:
        return (
            _conv1d_prep_tensor_id(weight),
            _conv1d_prep_tensor_id(bias) if bias is not None else 0,
            batch,
            input_length,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            groups,
            dilation,
            _fused_activation_token(fused_post_activation),
            bool(timeline_chunked),
        )

    def _prepare_conv1d_weights_for_prewarm(
        self,
        *,
        weight: ttnn.Tensor,
        bias: Optional[ttnn.Tensor],
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        groups: int = 1,
        dilation: int = 1,
        fused_post_activation: Optional[ttnn.UnaryWithParam] = None,
        shard_layout: Optional[ttnn.TensorMemoryLayout] = None,
        timeline_chunked: bool = False,
    ) -> None:
        """``prepare_conv_*`` + DRAM ``clone`` only — no Conv2d forward (T2U/speech-encoder recipe)."""
        prep_len, timeline_chunked = _vocoder_conv1d_prep_length(input_length, in_channels=in_channels, padding=padding)
        cache_key = self._conv1d_prep_cache_key(
            weight=weight,
            bias=bias,
            batch=batch,
            input_length=prep_len,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            dilation=dilation,
            fused_post_activation=fused_post_activation,
            timeline_chunked=timeline_chunked,
        )
        if cache_key in self._conv1d_prepared_cache:
            return

        conv_config = _vocoder_conv1d_config(
            fused_post_activation,
            input_length=prep_len,
            in_channels=in_channels,
            shard_layout=shard_layout,
            timeline_chunked=timeline_chunked,
        )
        prep_w = ttnn.prepare_conv_weights(
            weight_tensor=weight,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch,
            input_height=1,
            input_width=prep_len,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, padding),
            dilation=(1, dilation),
            has_bias=bias is not None,
            groups=groups,
            device=self.device,
            input_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
            conv_config=conv_config,
            compute_config=self._compute,
        )
        prep_b = None
        if bias is not None:
            prep_b = ttnn.prepare_conv_bias(
                bias_tensor=bias,
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                in_channels=in_channels,
                out_channels=out_channels,
                batch_size=batch,
                input_height=1,
                input_width=prep_len,
                kernel_size=(1, kernel_size),
                stride=(1, 1),
                padding=(0, padding),
                dilation=(1, dilation),
                groups=groups,
                device=self.device,
                input_dtype=ttnn.bfloat16,
                output_dtype=ttnn.bfloat16,
                conv_config=conv_config,
                compute_config=self._compute,
            )
        self._conv1d_prepared_cache[cache_key] = (
            ttnn.clone(prep_w, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.clone(prep_b, memory_config=ttnn.DRAM_MEMORY_CONFIG) if prep_b is not None else None,
        )

    def prewarm_conv1d_weights(self, *, batch: int = 1, seq: int, t_audio: int) -> None:
        """Prepare all vocoder ``conv1d`` weights for ``(seq, t_audio)`` (host upload only, no forward).

        Call before trace capture so replay avoids first-hit weight prep inside ``ttnn.conv1d``. Does not
        change the default PCC path when this is not called.
        """
        dp = self.p.dur_predictor
        c1, c2 = dp.conv1, dp.conv2
        self._prepare_conv1d_weights_for_prewarm(
            weight=c1.weight,
            bias=c1.bias,
            batch=batch,
            input_length=seq,
            in_channels=c1.in_channels,
            out_channels=c1.out_channels,
            kernel_size=c1.kernel_size,
            padding=c1.padding,
            fused_post_activation=_fused_relu(),
        )
        self._prepare_conv1d_weights_for_prewarm(
            weight=c2.weight,
            bias=c2.bias,
            batch=batch,
            input_length=seq,
            in_channels=c2.in_channels,
            out_channels=c2.out_channels,
            kernel_size=c2.kernel_size,
            padding=c2.padding,
            fused_post_activation=_fused_relu(),
        )

        hg = self.p.hifi_gan
        cp = hg.conv_pre
        tlen = int(t_audio)
        self._prepare_conv1d_weights_for_prewarm(
            weight=cp.weight,
            bias=cp.bias,
            batch=batch,
            input_length=tlen,
            in_channels=int(cp.in_channels),
            out_channels=int(cp.out_channels),
            kernel_size=int(cp.kernel_size),
            padding=int(cp.padding),
            shard_layout=_resolve_conv_shard_layout(
                _CONV_PRE_SHARD, in_channels=int(cp.in_channels), kernel_size=int(cp.kernel_size)
            ),
        )
        tlen = _host_conv_out_length(tlen, int(cp.kernel_size), 1, int(cp.padding))

        for i, up_layer in enumerate(hg.upsampler):
            k = int(up_layer["kernel_size"])
            s = int(up_layer["stride"])
            p = int(up_layer["padding"])
            tlen = _host_transpose_conv_out_length(tlen, k, s, p)
            channels = self.cfg.upsample_initial_channel // (2 ** (i + 1))
            for j in range(self.num_kernels):
                rb = hg.resblocks[i * self.num_kernels + j]
                t_rb = tlen
                for c1p, c2p in zip(rb.convs1, rb.convs2):
                    k1 = int(c1p["kernel_size"])
                    p1 = int(c1p["padding"])
                    d1 = int(c1p["dilation"])
                    self._prepare_conv1d_weights_for_prewarm(
                        weight=c1p["weight"],
                        bias=c1p["bias"],
                        batch=batch,
                        input_length=t_rb,
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=k1,
                        padding=p1,
                        dilation=d1,
                        fused_post_activation=_fused_leaky_relu(self.leaky_slope),
                        shard_layout=_resolve_conv_shard_layout(_RESBLOCK_SHARD, in_channels=channels, kernel_size=k1),
                    )
                    t_rb = _host_conv_out_length(t_rb, k1, 1, p1, dilation=d1)
                    k2 = int(c2p["kernel_size"])
                    p2 = int(c2p["padding"])
                    d2 = int(c2p["dilation"])
                    self._prepare_conv1d_weights_for_prewarm(
                        weight=c2p["weight"],
                        bias=c2p["bias"],
                        batch=batch,
                        input_length=t_rb,
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=k2,
                        padding=p2,
                        dilation=d2,
                        shard_layout=_resolve_conv_shard_layout(_RESBLOCK_SHARD, in_channels=channels, kernel_size=k2),
                    )
                    t_rb = _host_conv_out_length(t_rb, k2, 1, p2, dilation=d2)

        cpost = hg.conv_post
        self._prepare_conv1d_weights_for_prewarm(
            weight=cpost.weight,
            bias=cpost.bias,
            batch=batch,
            input_length=tlen,
            in_channels=int(cpost.in_channels),
            out_channels=int(cpost.out_channels),
            kernel_size=int(cpost.kernel_size),
            padding=int(cpost.padding),
        )

    def _conv1d_run(
        self,
        x_rm: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: Optional[ttnn.Tensor],
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
        dilation: int,
        fused_post_activation: Optional[ttnn.UnaryWithParam],
        deallocate_input: bool = False,
        use_prepared_weights: bool = True,
        shard_layout: Optional[ttnn.TensorMemoryLayout] = None,
        timeline_chunked: bool = False,
    ) -> Tuple[ttnn.Tensor, int]:
        """Single-shot ``ttnn.conv1d`` on row-major NLC activations."""
        conv_config = _vocoder_conv1d_config(
            fused_post_activation,
            input_length=input_length,
            in_channels=in_channels,
            shard_layout=shard_layout,
            timeline_chunked=timeline_chunked,
        )
        cache_key = self._conv1d_prep_cache_key(
            weight=weight,
            bias=bias,
            batch=batch,
            input_length=input_length,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            dilation=dilation,
            fused_post_activation=fused_post_activation,
            timeline_chunked=timeline_chunked,
        )
        cached = self._conv1d_prepared_cache.get(cache_key) if use_prepared_weights else None
        weight_tensor = cached[0] if cached is not None else weight
        bias_tensor = cached[1] if cached is not None else bias
        out, out_len = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=weight_tensor,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            bias_tensor=bias_tensor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            batch_size=batch,
            input_length=input_length,
            conv_config=conv_config,
            compute_config=self._compute,
            groups=groups,
            dilation=dilation,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
        )
        if deallocate_input:
            ttnn.deallocate(x_rm)
        out_len = int(out_len)
        if ttnn.is_sharded(out):
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (batch, out_len, out_channels))
        return out, out_len

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
        padding: int,
        groups: int,
        dilation: int = 1,
        fused_post_activation: Optional[ttnn.UnaryWithParam] = None,
        shard_layout: Optional[ttnn.TensorMemoryLayout] = None,
    ) -> Tuple[ttnn.Tensor, int]:
        # ``ttnn.conv1d`` reshapes activations for the conv2d path; TILE NLC (e.g. from ``embedding``)
        # can hit "reshape between two shapes with different volumes". Host ROW_MAJOR weights are fine.
        seq = int(input_length)
        x_in = x_nlc
        rm_buf: Optional[ttnn.Tensor] = None
        if x_nlc.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            rm_buf = ttnn.to_layout(x_nlc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x_in = rm_buf

        timeline_chunked = seq > _HIFIGAN_MAX_CONV1D_TLEN

        # Below this length one conv1d fits BH l1_small (double-buffer off). Only chunk above it.
        if seq <= _HIFIGAN_MAX_CONV1D_TLEN:
            return self._conv1d_run(
                x_in,
                weight=weight,
                bias=bias,
                batch=batch,
                input_length=seq,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                fused_post_activation=fused_post_activation,
                deallocate_input=rm_buf is not None,
                shard_layout=shard_layout,
                timeline_chunked=False,
            )

        # HF stores symmetric same-padding per layer; that is the overlap needed between chunks.
        halo = int(padding)
        interior = _vocoder_conv1d_chunk_interior(in_channels)
        fixed_in = interior + 2 * halo
        if seq <= fixed_in:
            return self._conv1d_run(
                x_in,
                weight=weight,
                bias=bias,
                batch=batch,
                input_length=seq,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                fused_post_activation=fused_post_activation,
                deallocate_input=rm_buf is not None,
                shard_layout=shard_layout,
                timeline_chunked=True,
            )

        chunks: list[ttnn.Tensor] = []
        for start in range(0, seq, interior):
            end = min(start + interior, seq)
            chunk_rows = end - start
            in_start = max(0, start - halo)
            in_end = min(seq, end + halo)
            x_win = ttnn.slice(
                x_in,
                [0, in_start, 0],
                [batch, in_end, in_channels],
                (1, 1, 1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            win_len = in_end - in_start
            if win_len < fixed_in:
                pad_rows = fixed_in - win_len
                pad = ttnn.zeros(
                    (batch, pad_rows, in_channels),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                x_win = ttnn.concat([x_win, pad], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(pad)
            out_win, _ = self._conv1d_run(
                x_win,
                weight=weight,
                bias=bias,
                batch=batch,
                input_length=fixed_in,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                fused_post_activation=fused_post_activation,
                deallocate_input=True,
                shard_layout=shard_layout,
                timeline_chunked=True,
            )
            out_start = start - in_start
            out_chunk = ttnn.slice(
                out_win,
                [0, out_start, 0],
                [batch, out_start + chunk_rows, out_channels],
                (1, 1, 1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(out_win)
            chunks.append(out_chunk)

        if rm_buf is not None:
            ttnn.deallocate(rm_buf)
        elif x_in is not x_nlc:
            ttnn.deallocate(x_in)

        if len(chunks) == 1:
            return chunks[0], seq
        out = ttnn.concat(chunks, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for c in chunks:
            ttnn.deallocate(c)
        return out, seq

    def _pad_nlc_time(
        self,
        x_nlc: ttnn.Tensor,
        *,
        batch: int,
        tlen: int,
        channels: int,
        pad_to: int,
    ) -> Tuple[ttnn.Tensor, bool]:
        """Right-pad ``[B, tlen, C]`` to ``pad_to`` rows; returns (tensor, created_new)."""
        if tlen >= pad_to:
            return x_nlc, False
        pad_rows = pad_to - tlen
        pad = ttnn.zeros(
            (batch, pad_rows, channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.concat([x_nlc, pad], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(pad)
        return out, True

    def _conv_transpose1d_nlc_dram_sliced(
        self,
        x_nlc: ttnn.Tensor,
        *,
        layer: Any,
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        conv_config: ttnn.Conv2dConfig,
    ) -> Tuple[ttnn.Tensor, int]:
        """DRAM height slices for long timelines (BH vocoder upsample L1 budget)."""
        k = int(layer["kernel_size"])
        s = int(layer["stride"])
        p = int(layer["padding"])
        weight = layer["weight"]
        bias = layer["bias"]
        num_slices = _vocoder_dram_slice_count(input_length)
        pad_h = ((int(input_length) + num_slices - 1) // num_slices) * num_slices
        x_work, padded = self._pad_nlc_time(
            x_nlc,
            batch=batch,
            tlen=input_length,
            channels=in_channels,
            pad_to=pad_h,
        )
        x_nhwc = ttnn.reshape(x_work, (batch, pad_h, 1, in_channels))
        if x_nhwc.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x_nhwc = ttnn.to_memory_config(x_nhwc, ttnn.DRAM_MEMORY_CONFIG)
        dram_slice = ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceHeight,
            num_slices=num_slices,
        )
        out_4d, out_hw = ttnn.conv_transpose2d(
            input_tensor=x_nhwc,
            weight_tensor=weight,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            bias_tensor=bias,
            kernel_size=(k, 1),
            stride=(s, 1),
            padding=(p, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            batch_size=batch,
            input_height=pad_h,
            input_width=1,
            conv_config=conv_config,
            compute_config=self._compute,
            dram_slice_config=dram_slice,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=ttnn.bfloat16,
        )
        if padded:
            ttnn.deallocate(x_work)
        out_h_padded = int(out_hw[0])
        out_nlc = ttnn.reshape(out_4d, (batch, out_h_padded, out_channels))
        if ttnn.is_sharded(out_nlc):
            out_nlc = ttnn.sharded_to_interleaved(out_nlc, ttnn.DRAM_MEMORY_CONFIG)
        out_h = _host_transpose_conv_out_length(input_length, k, s, p)
        if out_h < out_h_padded:
            out_nlc = ttnn.slice(
                out_nlc,
                [0, 0, 0],
                [batch, out_h, out_channels],
                (1, 1, 1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return out_nlc, out_h

    def _conv_transpose1d_nlc(
        self,
        x_nlc: ttnn.Tensor,
        *,
        layer: Any,
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
    ) -> Tuple[ttnn.Tensor, int]:
        """Maps HF ``ConvTranspose1d`` to ``ttnn.conv_transpose2d`` with singleton width."""
        k = int(layer["kernel_size"])
        s = int(layer["stride"])
        p = int(layer["padding"])
        weight = layer["weight"]
        bias = layer["bias"]

        # HEIGHT per DRAM slice when slice count and K are within caps; else auto layout.
        sliced = int(input_length) > 64
        num_slices = _vocoder_dram_slice_count(input_length) if sliced else 0
        prefer = _UPSAMPLE_SHARD if (not sliced or num_slices <= _UPSAMPLE_HEIGHT_MAX_SLICES) else None
        conv_config = _vocoder_conv2d_config(
            input_length=input_length,
            in_channels=in_channels,
            shard_layout=_resolve_conv_shard_layout(prefer, in_channels=in_channels, kernel_size=k),
        )
        if sliced:
            return self._conv_transpose1d_nlc_dram_sliced(
                x_nlc,
                layer=layer,
                batch=batch,
                input_length=input_length,
                in_channels=in_channels,
                out_channels=out_channels,
                conv_config=conv_config,
            )

        x_nhwc = ttnn.reshape(x_nlc, (batch, input_length, 1, in_channels))
        out_4d, out_hw = ttnn.conv_transpose2d(
            input_tensor=x_nhwc,
            weight_tensor=weight,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            bias_tensor=bias,
            kernel_size=(k, 1),
            stride=(s, 1),
            padding=(p, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            batch_size=batch,
            input_height=input_length,
            input_width=1,
            conv_config=conv_config,
            compute_config=self._compute,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=ttnn.bfloat16,
        )
        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        assert out_w == 1
        out_nlc = ttnn.reshape(out_4d, (batch, out_h, out_channels))
        if ttnn.is_sharded(out_nlc):
            out_nlc = ttnn.sharded_to_interleaved(out_nlc, ttnn.DRAM_MEMORY_CONFIG)
        return out_nlc, out_h

    def _dur_predictor_dev(self, x_nlc: ttnn.Tensor, *, batch: int, seq: int, dp: Any) -> ttnn.Tensor:
        """Returns log-duration prediction as a device tensor of shape ``[B, T_units]`` (bf16)."""
        c1, c2 = dp.conv1, dp.conv2
        h, tlen = self._conv1d(
            x_nlc,
            weight=c1.weight,
            bias=c1.bias,
            batch=batch,
            input_length=seq,
            in_channels=c1.in_channels,
            out_channels=c1.out_channels,
            kernel_size=c1.kernel_size,
            stride=1,
            padding=c1.padding,
            groups=1,
            fused_post_activation=_fused_relu(),
        )
        h = self._layer_norm(h, weight=dp.ln1.weight, bias=dp.ln1.bias, eps=dp.ln1.eps)
        h, tlen = self._conv1d(
            h,
            weight=c2.weight,
            bias=c2.bias,
            batch=batch,
            input_length=tlen,
            in_channels=c2.in_channels,
            out_channels=c2.out_channels,
            kernel_size=c2.kernel_size,
            stride=1,
            padding=c2.padding,
            groups=1,
            fused_post_activation=_fused_relu(),
        )
        h = self._layer_norm(h, weight=dp.ln2.weight, bias=dp.ln2.bias, eps=dp.ln2.eps)
        log_dur = self._linear(h, dp.proj.weight, dp.proj.bias)  # [B, T_units, 1]
        ttnn.deallocate(h)
        log_dur_2d = ttnn.squeeze(log_dur, -1)  # [B, T_units] (may be [B, 1, T] on mesh)
        # Caller frees after ``expm1`` (``squeeze`` may alias ``log_dur``).
        return _as_batch_time_2d(log_dur_2d, batch=batch, seq=seq)

    def _resblock(self, x_nlc: ttnn.Tensor, rb: Any, *, batch: int, tlen: int, channels: int) -> ttnn.Tensor:
        """One HF ``HifiGanResidualBlock``; ``x`` is ``[B,T,C]``."""
        for c1p, c2p in zip(rb.convs1, rb.convs2):
            residual = x_nlc
            x_nlc = ttnn.leaky_relu(x_nlc, negative_slope=self.leaky_slope, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x_nlc, tlen = self._conv1d(
                x_nlc,
                weight=c1p["weight"],
                bias=c1p["bias"],
                batch=batch,
                input_length=tlen,
                in_channels=channels,
                out_channels=channels,
                kernel_size=int(c1p["kernel_size"]),
                stride=1,
                padding=int(c1p["padding"]),
                groups=1,
                dilation=int(c1p["dilation"]),
                fused_post_activation=_fused_leaky_relu(self.leaky_slope),
                shard_layout=_resolve_conv_shard_layout(
                    _RESBLOCK_SHARD, in_channels=channels, kernel_size=int(c1p["kernel_size"])
                ),
            )
            x_nlc, tlen = self._conv1d(
                x_nlc,
                weight=c2p["weight"],
                bias=c2p["bias"],
                batch=batch,
                input_length=tlen,
                in_channels=channels,
                out_channels=channels,
                kernel_size=int(c2p["kernel_size"]),
                stride=1,
                padding=int(c2p["padding"]),
                groups=1,
                dilation=int(c2p["dilation"]),
                shard_layout=_resolve_conv_shard_layout(
                    _RESBLOCK_SHARD, in_channels=channels, kernel_size=int(c2p["kernel_size"])
                ),
            )
            x_nlc = ttnn.add(x_nlc, residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x_nlc

    def _hifi_gan(self, x_nlc: ttnn.Tensor, hg: Any, *, batch: int, tlen: int) -> ttnn.Tensor:
        """HiFi-GAN stack; long ``conv1d`` timelines are chunked inside ``_conv1d`` (not mel splits)."""
        return self._hifi_gan_once(x_nlc, hg, batch=batch, tlen=tlen)

    def _hifi_gan_once(self, x_nlc: ttnn.Tensor, hg: Any, *, batch: int, tlen: int) -> ttnn.Tensor:
        import os as _os_vt, time as _t_vt

        _vt_on = bool(_os_vt.environ.get("VOC_TIMING"))

        def _vt(name: str, t0: float) -> float:
            if _vt_on:
                ttnn.synchronize_device(self.device)
                now = _t_vt.time()
                print(f"[VOC-TIMING] {name} (tlen={tlen}): {now - t0:.1f}s", flush=True)
                return now
            return t0

        _vt0 = _t_vt.time()
        cp = hg.conv_pre
        h, tlen = self._conv1d(
            x_nlc,
            weight=cp.weight,
            bias=cp.bias,
            batch=batch,
            input_length=tlen,
            in_channels=int(cp.in_channels),
            out_channels=int(cp.out_channels),
            kernel_size=int(cp.kernel_size),
            stride=1,
            padding=int(cp.padding),
            groups=1,
            shard_layout=_resolve_conv_shard_layout(
                _CONV_PRE_SHARD, in_channels=int(cp.in_channels), kernel_size=int(cp.kernel_size)
            ),
        )
        ttnn.deallocate(x_nlc)
        _vt0 = _vt("conv_pre", _vt0)

        for i, up_layer in enumerate(hg.upsampler):
            h = ttnn.leaky_relu(h, negative_slope=self.leaky_slope, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            h, tlen = self._conv_transpose1d_nlc(
                h,
                layer=up_layer,
                batch=batch,
                input_length=tlen,
                in_channels=int(up_layer["in_channels"]),
                out_channels=int(up_layer["out_channels"]),
            )
            _vt0 = _vt(f"stage{i} conv_transpose -> tlen={tlen}", _vt0)
            channels = self.cfg.upsample_initial_channel // (2 ** (i + 1))
            acc = None
            for j in range(self.num_kernels):
                rb = hg.resblocks[i * self.num_kernels + j]
                br = self._resblock(h, rb, batch=batch, tlen=tlen, channels=channels)
                if acc is None:
                    acc = br
                else:
                    acc = ttnn.add(acc, br, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(br)
            scale = 1.0 / float(self.num_kernels)
            acc_scaled = ttnn.multiply(acc, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(acc)
            ttnn.deallocate(h)
            h = acc_scaled
            _vt0 = _vt(f"stage{i} resblocks", _vt0)

        # Match HF: line 2489 of ``modeling_seamless_m4t_v2.py`` calls
        # ``nn.functional.leaky_relu(hidden_states)`` with the default 0.01 slope, NOT
        # ``self.leaky_relu_slope``. The other leaky_relus in the upsample loop and inside the
        # residual blocks do use ``cfg.leaky_relu_slope``.
        h = ttnn.leaky_relu(h, negative_slope=0.01, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cpost = hg.conv_post
        h, tlen = self._conv1d(
            h,
            weight=cpost.weight,
            bias=cpost.bias,
            batch=batch,
            input_length=tlen,
            in_channels=int(cpost.in_channels),
            out_channels=int(cpost.out_channels),
            kernel_size=int(cpost.kernel_size),
            stride=1,
            padding=int(cpost.padding),
            groups=1,
        )
        h = ttnn.tanh(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return h

    def forward(
        self,
        input_ids: ttnn.Tensor,
        speaker_id: ttnn.Tensor,
        lang_id: ttnn.Tensor,
        *,
        input_ids_torch: Optional[torch.Tensor] = None,
        trace_no_profiler: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            input_ids: ``uint32`` ``[B, T]`` on device (unit tokens).
            speaker_id: ``uint32`` ``[B, 1]`` on device.
            lang_id: ``uint32`` ``[B, 1]`` on device.
            input_ids_torch: kept for API compatibility; unused.

        Returns:
            ``(waveform, lengths)`` — both device tensors. ``waveform`` is bfloat16
            ``[B, T_wav_max, 1]``; ``lengths`` is int32 ``[B]`` and gives the valid
            audio sample count per row. For ``B > 1`` rows are right-padded with zeros to
            ``T_wav_max = max_b T_wav_b``; consumers must crop using ``lengths``.

        Notes:
            * ``B == 1`` runs a single fused on-device program (the fast path).
            * ``B > 1`` mirrors the Hugging Face implementation, which also processes rows
              sequentially (see ``modeling_seamless_m4t_v2.py`` lines 2611-2623). Each row
              goes through exactly the same ``_forward_one`` device program, so per-row PCC
              is identical to the ``B == 1`` baseline. Padding and final concat happen on
              device — no PyTorch compute is introduced.
        """
        del input_ids_torch  # API-compat stub.
        batch = int(input_ids.shape[0])
        if batch == 1:
            return self._forward_one(input_ids, speaker_id, lang_id, trace_no_profiler=trace_no_profiler)
        return self._forward_batched(
            input_ids,
            speaker_id,
            lang_id,
            batch=batch,
            trace_no_profiler=trace_no_profiler,
        )

    # ------------------------------------------------------------------------------- B == 1

    def _forward_one(
        self,
        input_ids: ttnn.Tensor,
        speaker_id: ttnn.Tensor,
        lang_id: ttnn.Tensor,
        *,
        trace_no_profiler: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Single-sample forward (``B == 1``); fully on device modulo one shape int."""
        batch = int(input_ids.shape[0])
        seq = int(input_ids.shape[1])
        assert batch == 1, "_forward_one expects B == 1; use forward() for B > 1."

        # ---------- embeddings (all on device) ----------
        # Tables uploaded ROW_MAJOR (``_vocoder_embedding_weight_row_major``); ``layout`` here is the *output* layout.
        ue = self.p.unit_embedding.weight
        use = ttnn.embedding(input_ids, weight=ue, layout=ttnn.TILE_LAYOUT)  # [B, T_units, E_unit] (TILE-padded)

        sp = self.p.speaker_embedding.weight
        la = self.p.language_embedding.weight
        sp_e = ttnn.embedding(ttnn.squeeze(speaker_id, 1), weight=sp, layout=ttnn.TILE_LAYOUT)
        lang_e = ttnn.embedding(ttnn.squeeze(lang_id, 1), weight=la, layout=ttnn.TILE_LAYOUT)

        # ---------- duration prediction (device) ----------
        dp = self.p.dur_predictor
        e_unit = int(dp.conv1.in_channels)
        # Drop TILE padding on the time / channel axes so ``seq`` matches ``cumsum_*`` and
        # ``use_BEC @ H`` is ``[B,E,T] @ [B,T,t_audio]`` (otherwise last dim can be e.g. 106 vs T=7).
        use_trim = ttnn.slice(use, [0, 0, 0], [batch, seq, e_unit], (1, 1, 1))
        use = use_trim
        log_dur = self._dur_predictor_dev(use, batch=batch, seq=seq, dp=dp)  # [B, T_units] bf16

        # HF: dur_out = clamp(round(expm1(log_dur)), min=1).long()
        e = ttnn.expm1(log_dur, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(log_dur)
        r = ttnn.round(e, decimals=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(e)
        dur_bf = ttnn.maximum(r, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, T_units] bf16
        ttnn.deallocate(r)

        # ---------- exclusive cumulative duration (device) ----------
        # Use float32 internally so integer comparisons are exact for any plausible t_audio.
        dur_f32 = ttnn.typecast(dur_bf, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cumsum_inc = ttnn.cumsum(dur_f32, dim=-1, dtype=ttnn.float32)  # [B, T_units] inclusive
        cumsum_prev = ttnn.subtract(cumsum_inc, dur_f32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(dur_f32)
        ttnn.deallocate(dur_bf)

        cumsum_inc_bt = _as_batch_time_2d(cumsum_inc, batch=batch, seq=seq)
        cumsum_prev_bt = _as_batch_time_2d(cumsum_prev, batch=batch, seq=seq)

        # ``t_audio`` from HF gather index (not last TILE column, which may be padding).
        # Metal trace capture forbids host readbacks — reuse values cached on the compile forward.
        if trace_no_profiler and self._last_t_audio is not None:
            t_audio = int(self._last_t_audio)
        else:
            cumsum_pre_seq = _slice_batch_time(cumsum_inc_bt, batch=batch, seq=seq)
            cs_t = to_torch_replicated_first_shard(cumsum_pre_seq).float().reshape(batch, -1)
            pad_id = int(self.cfg.t2u_pad_token_id)
            idx = _vocoder_hf_gather_index(input_ids, batch=batch, seq=seq, pad_id=pad_id)
            t_audio = int(cs_t[0, idx].item())
            if t_audio < 1:
                raise RuntimeError(f"Computed t_audio={t_audio}; expected positive duration sum.")
            self._last_t_audio = t_audio

        # Bucket the upsampled length so run-to-run jitter reuses the shape-specialized vocoder
        # programs instead of recompiling. ``t_audio_real`` drives the (cropped) output length;
        # the bucketed ``t_audio`` drives every downstream shape. Frames in
        # ``[t_audio_real, t_audio)`` get an all-zero expansion (no unit's [cumsum_prev, cumsum)
        # interval covers them), so the valid waveform is unchanged (see _VOCODER_TAUDIO_BUCKET).
        t_audio_real = t_audio
        t_audio = ((t_audio + _VOCODER_TAUDIO_BUCKET - 1) // _VOCODER_TAUDIO_BUCKET) * _VOCODER_TAUDIO_BUCKET

        if not trace_no_profiler:
            ttnn.ReadDeviceProfiler(self.device)

        # ---------- expansion via embeddings @ H (device) ----------
        # Defer dealloc of views (slice/permute/reshape) until after concat — aliases are common.
        frame_idx = self._cached_frame_idx_f32(t_audio)
        # Reshape for broadcasting: [B, T_units, 1] vs [1, 1, t_audio].
        c_b = ttnn.reshape(cumsum_inc_bt, (batch, seq, 1))
        cp_b = ttnn.reshape(cumsum_prev_bt, (batch, seq, 1))
        f_b = ttnn.reshape(frame_idx, (1, 1, t_audio))

        lower = ttnn.ge(f_b, cp_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        upper = ttnn.lt(f_b, c_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        H_mask = ttnn.logical_and(lower, upper, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(lower)
        ttnn.deallocate(upper)
        H = ttnn.typecast(H_mask, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(H_mask)

        # use: [B, T_units, E_unit] -> [B, E_unit, T_units]; expand to [B, E_unit, t_audio].
        use_BEC = ttnn.permute(use, (0, 2, 1))
        expanded_BCT = self._expand_unit_embeddings_matmul(
            use_BEC,
            H,
            batch=batch,
            e_unit=e_unit,
            seq=seq,
            t_audio=t_audio,
        )

        # ---------- broadcast lang/spk to ``t_audio`` (device) ----------
        lang_dim = int(lang_e.shape[-1])
        spk_dim = int(sp_e.shape[-1])
        lang_BC1 = ttnn.reshape(lang_e, (batch, lang_dim, 1))
        spk_BC1 = ttnn.reshape(sp_e, (batch, spk_dim, 1))
        if t_audio == 1:
            lang_BCT = lang_BC1
            spk_BCT = spk_BC1
        else:
            lang_BCT = ttnn.repeat(lang_BC1, [1, 1, t_audio])
            spk_BCT = ttnn.repeat(spk_BC1, [1, 1, t_audio])
            ttnn.deallocate(lang_BC1)
            ttnn.deallocate(spk_BC1)

        merged_BCT = ttnn.concat([lang_BCT, expanded_BCT, spk_BCT], dim=1)
        ttnn.deallocate(lang_BCT)
        ttnn.deallocate(expanded_BCT)
        ttnn.deallocate(spk_BCT)
        ttnn.deallocate(use_BEC)
        ttnn.deallocate(use)
        ttnn.deallocate(H)
        merged_NLC = ttnn.permute(merged_BCT, (0, 2, 1))
        ttnn.deallocate(merged_BCT)

        # ---------- HiFi-GAN (device) ----------
        wav = self._hifi_gan(merged_NLC, self.p.hifi_gan, batch=batch, tlen=t_audio)

        # ---------- length (host formula, single int32 upload per row) ----------
        # Real (un-bucketed) length so the bucket-padded waveform tail is cropped out by consumers.
        lengths = self._output_lengths_dev(t_audio_real, batch=batch)
        ttnn.deallocate(cumsum_inc)
        ttnn.deallocate(cumsum_prev)
        return wav, lengths

    # ------------------------------------------------------------------------------- B > 1

    def _forward_batched(
        self,
        input_ids: ttnn.Tensor,
        speaker_id: ttnn.Tensor,
        lang_id: ttnn.Tensor,
        *,
        batch: int,
        trace_no_profiler: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Multi-sample forward for any ``B > 1``. Mirrors HF's per-sample loop in
        ``modeling_seamless_m4t_v2.py`` (lines 2611-2623). Each row runs through the
        identical ``_forward_one`` device program, so per-row PCC is unchanged. Padding
        each row's waveform up to the batch maximum and the final concat are done on
        device via ``ttnn.pad`` and ``ttnn.concat``; no PyTorch compute is introduced.
        """
        seq = int(input_ids.shape[1])

        wavs: list[ttnn.Tensor] = []  # one [1, T_wav_b, 1] device tensor per row
        wav_lens: list[int] = []  # T_wav_b (Python ints from tensor metadata)
        valid_lens: list[ttnn.Tensor] = []  # one [1] int32 device tensor per row

        for b in range(batch):
            ids_b = ttnn.slice(input_ids, [b, 0], [b + 1, seq])
            spk_b = ttnn.slice(speaker_id, [b, 0], [b + 1, 1])
            lang_b = ttnn.slice(lang_id, [b, 0], [b + 1, 1])

            wav_b, len_b = self._forward_one(
                ids_b,
                spk_b,
                lang_b,
                trace_no_profiler=trace_no_profiler,
            )
            ttnn.deallocate(ids_b)
            ttnn.deallocate(spk_b)
            ttnn.deallocate(lang_b)

            wavs.append(wav_b)
            wav_lens.append(int(wav_b.shape[1]))
            valid_lens.append(len_b)

        t_wav_max = max(wav_lens)

        # Pad each row to ``[1, t_wav_max, 1]`` on device. ``ttnn.pad`` requires ROW_MAJOR
        # for arbitrary (non-tile-aligned) trailing-dim sizes, so we briefly drop layout.
        padded: list[ttnn.Tensor] = []
        for wav_b, t_b in zip(wavs, wav_lens):
            if wav_b.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                wav_rm = ttnn.to_layout(wav_b, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(wav_b)
            else:
                wav_rm = wav_b
            if t_b == t_wav_max:
                padded.append(wav_rm)
            else:
                wav_pad = ttnn.pad(
                    wav_rm,
                    padding=[(0, 0), (0, t_wav_max - t_b), (0, 0)],
                    value=0.0,
                )
                ttnn.deallocate(wav_rm)
                padded.append(wav_pad)

        waveform = ttnn.concat(padded, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for w in padded:
            ttnn.deallocate(w)

        lengths = ttnn.concat(valid_lens, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for l in valid_lens:
            ttnn.deallocate(l)

        return waveform, lengths

    def release_forward_trace(self) -> None:
        """Release a captured vocoder forward Metal trace (safe if none is active)."""
        rt = self._forward_trace_rt
        if rt is None:
            return
        try:
            ttnn.release_trace(self.device, rt.trace_id)
        except Exception:
            pass
        self._forward_trace_rt = None

    def capture_forward_trace(
        self,
        input_ids: ttnn.Tensor,
        speaker_id: ttnn.Tensor,
        lang_id: ttnn.Tensor,
        *,
        after_compile: bool = False,
    ) -> None:
        """Capture Metal trace for one vocoder forward (compile outside trace, then capture on CQ0).

        When ``after_compile=True``, skip the compile forward and use ``_last_t_audio`` left by a
        prior ``forward()`` on the same inputs (trace PCC tests call ``forward`` once, then capture).
        Otherwise this method runs one compile ``forward`` to discover ``t_audio``, then prewarms.
        """
        if self._forward_trace_rt is not None:
            return

        batch = int(input_ids.shape[0])
        seq = int(input_ids.shape[1])
        if batch != 1:
            raise ValueError("capture_forward_trace supports batch == 1 only.")

        if not after_compile:
            compile_wav, compile_len = self.forward(
                input_ids,
                speaker_id,
                lang_id,
                trace_no_profiler=True,
            )
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(compile_wav)
            ttnn.deallocate(compile_len)
        elif self._last_t_audio is None:
            raise RuntimeError("after_compile=True requires a prior forward() that set _last_t_audio.")

        t_audio = self._last_t_audio
        if t_audio is None or t_audio < 1:
            raise RuntimeError("compile forward did not set _last_t_audio.")
        self.prewarm_conv1d_weights(batch=batch, seq=seq, t_audio=int(t_audio))
        ttnn.synchronize_device(self.device)

        capture_wav: Optional[ttnn.Tensor] = None
        capture_len: Optional[ttnn.Tensor] = None

        def traced_step() -> None:
            nonlocal capture_wav, capture_len
            capture_wav, capture_len = self.forward(
                input_ids,
                speaker_id,
                lang_id,
                trace_no_profiler=True,
            )

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        traced_step()
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        if capture_wav is None or capture_len is None:
            raise RuntimeError("vocoder forward trace capture produced no outputs.")
        ttnn.synchronize_device(self.device)
        self._forward_trace_rt = VocoderForwardTraceRuntime(
            trace_id=trace_id,
            waveform_tt=capture_wav,
            lengths_tt=capture_len,
        )

    def execute_forward_trace(self) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Replay a captured vocoder forward trace on CQ0 (inputs must match capture)."""
        rt = self._forward_trace_rt
        if rt is None:
            raise RuntimeError("vocoder forward trace not captured; call capture_forward_trace first.")
        ttnn.execute_trace(self.device, rt.trace_id, cq_id=0, blocking=True)
        return rt.waveform_tt, rt.lengths_tt
