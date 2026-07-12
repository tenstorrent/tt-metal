# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN ``SeamlessM4Tv2CodeHifiGan`` — unit embeddings, duration predictor, and HiFi-GAN vocoder.

Matches Hugging Face ``modeling_seamless_m4t_v2`` layout. Conv1d uses ``ttnn.conv1d``; ConvTranspose1d
uses ``ttnn.conv_transpose2d`` with ``H=1``. Temporal lengths follow HF conv formulas; ``t_audio``
(one host scalar from duration cumsum) sets the repeat-interleave and HiFi-GAN ``input_length``.
Output waveform lengths are computed on host from ``t_audio`` and uploaded as int32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.seamless_m4t_v2_large.tt.common import (
    TILE,
    core_grid,
    matmul_program_config,
    MATMUL_1D_SEQ_THRESHOLD,
    pick_largest_height_shard_nhw_cores,
    to_torch_replicated_first_shard,
)

# Both matmul dims must stay small on BH; ``t_audio`` alone can be thousands of frames.
_VOCODER_EXPAND_MATMUL_CHUNK = TILE
# Timelines up to this fit a single unsliced ``ttnn.conv1d`` on BH; longer ones are width-sliced in DRAM.
_HIFIGAN_MAX_CONV1D_TLEN = 4096
# Round ``t_audio`` up to stabilize shape-specialized vocoder programs; padded tail is masked/cropped.
_VOCODER_TAUDIO_BUCKET = 256

# Estimated per-core L1 ceiling (bytes) for a single-shot HEIGHT-sharded conv1d; above it the timeline is
# width-sliced. BH L1 is ~1.5 MB/core; keep headroom for conv CBs / halo / intermediates.
_VOCODER_L1_SINGLESHOT_BYTES = 1_200_000
# BH worker-grid core count used to estimate the per-core activation shard (conservative).
_VOCODER_L1_SINGLESHOT_CORES = 110


def _vocoder_fits_l1_singleshot(timeline: int, in_channels: int, kernel_size: int = 11, batch: int = 1) -> bool:
    """Whether one HEIGHT-sharded conv1d fits BH L1 single-shot (no slicing).

    NOT an element-count test: HEIGHT sharding splits only the timeline across cores, but the weight
    tensor (``in_ch*out_ch*k``, ``out==in`` for resblocks) is replicated on every core and dominates L1
    at high channel counts — 256ch/k11 (~720 KB bf8, ~1.44 MB double-buffered) overflows even at a short
    7680-row timeline, while 16ch/k11 fits at 491520 rows. A plain ``batch*timeline*in_channels`` budget
    mis-ranks these (256ch/7680 = 1.97M "fits" but crashes; 16ch/491520 = 7.86M is fine), so estimate the
    per-core footprint (replicated weights + timeline-sharded activation) instead."""
    ch = int(in_channels)
    k = int(kernel_size)
    cores = _VOCODER_L1_SINGLESHOT_CORES
    weight_bytes = ch * ch * k  # bfloat8_b weights (out_ch == in_ch for resblock convs), 1 B/elem
    rows_per_core = max(1, (int(batch) * int(timeline) + cores - 1) // cores)
    act_bytes = rows_per_core * ch * 2 * 3  # bf16 activation, ~3 live buffers (in / out / halo)
    est = weight_bytes * 2 + act_bytes  # weights counted twice for prep + runtime headroom
    return est <= _VOCODER_L1_SINGLESHOT_BYTES


def _resolve_resblock_shard_layout(
    *, in_channels: int, kernel_size: int, input_length: int, batch: int = 1
) -> Optional[ttnn.TensorMemoryLayout]:
    """HEIGHT sharding for a resblock conv, kept even on long timelines while K fits and the conv fits L1
    single-shot (no auto-slice); otherwise defer to the generic resolver."""
    if int(in_channels) * int(kernel_size) <= _HEIGHT_SHARD_K_MAX and _vocoder_fits_l1_singleshot(
        input_length, in_channels, kernel_size, batch
    ):
        return ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    return _resolve_conv_shard_layout(
        _RESBLOCK_SHARD, in_channels=in_channels, kernel_size=kernel_size, input_length=input_length
    )


def _vocoder_conv1d_prep_length(timeline_length: int, *, in_channels: int, padding: int) -> Tuple[int, bool]:
    """Return ``(conv input_width, timeline_chunked)``; timelines beyond the single-shot cap are DRAM width-sliced."""
    seq = int(timeline_length)
    return seq, seq > _HIFIGAN_MAX_CONV1D_TLEN


# Round conv1d timelines up to this step so nondeterministic decode/T2U output lengths map to a few stable,
# reused conv shapes instead of a cold compile per distinct length. Tile-aligned.
_VOCODER_CONV1D_BUCKET_STEP = 256


def _vocoder_timeline_bucket(length: int) -> int:
    """Tile-aligned bucket for any vocoder conv1d timeline."""
    L = int(length)
    if L <= 0:
        return L
    step = _VOCODER_CONV1D_BUCKET_STEP
    return ((L + step - 1) // step) * step


def _vocoder_conv1d_bucket(length: int) -> int:
    """Tile-aligned bucket capped at ``_HIFIGAN_MAX_CONV1D_TLEN`` (single-shot path in ``_conv1d_run``)."""
    return min(_vocoder_timeline_bucket(length), _HIFIGAN_MAX_CONV1D_TLEN)


def _slice_nlc_time(
    x: ttnn.Tensor,
    *,
    batch: int,
    start: int,
    end: int,
    channels: int,
) -> ttnn.Tensor:
    """Slice ``[B, start:end, C]`` (or mesh ``[B, 1, S, C]``); block-sharded inputs are interleaved first."""
    if ttnn.is_sharded(x):
        x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    rank = len(x.shape)
    if rank == 4 and int(x.shape[1]) == 1:
        return ttnn.slice(
            x,
            [0, 0, int(start), 0],
            [batch, 1, int(end), int(channels)],
            (1, 1, 1, 1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if rank == 3:
        return ttnn.slice(
            x,
            [0, int(start), 0],
            [batch, int(end), int(channels)],
            (1, 1, 1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    raise RuntimeError(f"cannot slice NLC tensor with shape {tuple(x.shape)}")


# Target elements (rows*in_channels) per conv_transpose DRAM height slice: few large slices (each pays its
# own PaddedSlice/SliceWrite) rather than many tiny ones. 512K keeps ~1.6x margin under the BH L1_SMALL floor.
_VOCODER_TRANSPOSE_SLICE_ELEMS = 512 * 1024
_VOCODER_TRANSPOSE_MIN_SLICES = 8


def _vocoder_dram_slice_count(input_length: int, in_channels: int = 0) -> int:
    """DRAM height slices for a long vocoder upsample timeline: size each to a fixed element budget."""
    il = int(input_length)
    if int(in_channels) > 0:
        slices = (il * int(in_channels) + _VOCODER_TRANSPOSE_SLICE_ELEMS - 1) // _VOCODER_TRANSPOSE_SLICE_ELEMS
        return min(128, max(_VOCODER_TRANSPOSE_MIN_SLICES, slices))
    if il <= MATMUL_1D_SEQ_THRESHOLD:
        return 16
    return min(128, max(16, (il + 127) // 128))


def _vocoder_dram_slice_pad_h(input_length: int, *, in_channels: int = 0) -> int:
    """Tile-aligned padded height for DRAM height-sliced transpose conv."""
    num_slices = _vocoder_dram_slice_count(input_length, in_channels)
    pad_h = ((int(input_length) + num_slices - 1) // num_slices) * num_slices
    if int(in_channels) >= 512:
        align = _VOCODER_ACT_BLOCK_H_ALIGN
        pad_h = ((pad_h + align - 1) // align) * align
    return pad_h


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


# HEIGHT sharding when K fits and timeline is short enough; None lets the device auto-pick.
_HEIGHT_SHARD_K_MAX = 4096
# Long conv1d timelines + forced HEIGHT shard → DRAM auto-slice failure (BH S2ST @ 4096).
_HEIGHT_SHARD_MAX_TLEN = 2048
_CONV_PRE_SHARD = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_RESBLOCK_SHARD = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
# Upsample transpose: HEIGHT per DRAM slice when slice count is small enough.
_UPSAMPLE_SHARD = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_UPSAMPLE_HEIGHT_MAX_SLICES = 128
# Wide conv act_block_h=64 needs padded_output_height_ntiles_per_core % 2 == 0; pad timelines to this.
_VOCODER_ACT_BLOCK_H_ALIGN = 128


def _resolve_conv_shard_layout(
    prefer: Optional[ttnn.TensorMemoryLayout],
    *,
    in_channels: int,
    kernel_size: int,
    input_length: int = 0,
) -> Optional[ttnn.TensorMemoryLayout]:
    if prefer is None:
        return None
    if prefer == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        if int(in_channels) * int(kernel_size) > _HEIGHT_SHARD_K_MAX:
            return None
        if int(input_length) > _HEIGHT_SHARD_MAX_TLEN:
            return None
    return prefer


def _vocoder_act_block_h_override(input_length: int, in_channels: int) -> Optional[int]:
    """Pick act_block_h_override only when TTNN divisibility is likely to accept it."""
    il = int(input_length)
    if int(in_channels) >= 512:
        if il > 64 and il % _VOCODER_ACT_BLOCK_H_ALIGN == 0:
            return 64
        if il > 64:
            return 32
        return None
    if il > 64:
        return 32
    return None


def _vocoder_conv_nhw_tiles(timeline_length: int, *, batch: int = 1) -> int:
    """Output NHW tile count for conv1d mapped to conv2d (``H=timeline``, ``W=1``)."""
    return (int(batch) * int(timeline_length) + TILE - 1) // TILE


def _vocoder_tiles_per_core(timeline_length: int, nhw_cores: int, *, batch: int = 1) -> int:
    """Per-core output height in tiles after TTNN height padding (``create_sharded_memory_config`` recipe)."""
    nhw = int(batch) * int(timeline_length)
    cores = max(1, int(nhw_cores))
    nhw_padded = ((nhw + cores * TILE - 1) // (cores * TILE)) * (cores * TILE)
    return nhw_padded // (cores * TILE)


def _pick_vocoder_height_shard_nhw_cores(
    timeline_length: int,
    device: ttnn.Device,
    *,
    batch: int = 1,
    act_block_h: Optional[int] = None,
) -> int:
    """Pick NHW core count for HEIGHT-sharded vocoder conv1d (speech-encoder recipe + even tiles for act_block_h=64)."""
    grid = device.compute_with_storage_grid_size()
    max_cores = max(1, int(grid.x) * int(grid.y))
    nhw_tiles = _vocoder_conv_nhw_tiles(timeline_length, batch=batch)
    cap = min(max_cores, max(1, nhw_tiles))
    prefer_even_tp = act_block_h is not None and int(act_block_h) >= 64

    def acceptable(cores: int) -> bool:
        if cores < 1 or cores > cap:
            return False
        tp = _vocoder_tiles_per_core(timeline_length, cores, batch=batch)
        if tp <= 0:
            return False
        return not (prefer_even_tp and tp % 2 != 0)

    for cores in range(cap, 2, -1):
        if acceptable(cores):
            return cores
    if acceptable(2):
        return 2
    return pick_largest_height_shard_nhw_cores(nhw_tiles, device)


def _vocoder_apply_height_shard_core_override(
    conv_kwargs: dict,
    device: ttnn.Device,
    *,
    input_length: int,
    batch: int,
    shard_layout: Optional[ttnn.TensorMemoryLayout],
    act_block_h: Optional[int],
) -> None:
    """Set ``override_sharding_config`` + ``core_grid`` for conv_pre / resblock HEIGHT sharding."""
    if shard_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return
    nhw_cores = _pick_vocoder_height_shard_nhw_cores(input_length, device, batch=batch, act_block_h=act_block_h)
    if nhw_cores <= 2:
        return
    grid = device.compute_with_storage_grid_size()
    conv_kwargs["override_sharding_config"] = True
    conv_kwargs["core_grid"] = ttnn.num_cores_to_corerangeset(nhw_cores, grid, row_wise=True)


def _vocoder_conv1d_config(
    fused_post_activation: Optional[ttnn.UnaryWithParam],
    *,
    input_length: int,
    in_channels: int,
    shard_layout: Optional[ttnn.TensorMemoryLayout] = None,
    timeline_chunked: bool = False,
    row_major_output: bool = False,
    device: Optional[ttnn.Device] = None,
    batch: int = 1,
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
    act_block_h = _vocoder_act_block_h_override(input_length, in_channels)
    if act_block_h is not None:
        conv_kwargs["act_block_h_override"] = act_block_h
    # Forced shard layouts may need config tensors in DRAM when L1_SMALL is tight.
    if shard_layout is not None:
        conv_kwargs["config_tensors_in_dram"] = True
    if device is not None:
        _vocoder_apply_height_shard_core_override(
            conv_kwargs,
            device,
            input_length=input_length,
            batch=batch,
            shard_layout=shard_layout,
            act_block_h=act_block_h,
        )
    # DRAM width-sliced timelines: spill conv indices to DRAM (frees L1_SMALL) and emit ROW_MAJOR output.
    if timeline_chunked:
        conv_kwargs["config_tensors_in_dram"] = True
        conv_kwargs["output_layout"] = ttnn.ROW_MAJOR_LAYOUT
    elif row_major_output:
        # Single-shot tail (e.g. conv_post): keep ROW_MAJOR through the tanh / waveform path.
        conv_kwargs["output_layout"] = ttnn.ROW_MAJOR_LAYOUT
    return ttnn.Conv1dConfig(**conv_kwargs)


def _vocoder_conv2d_config(
    *, input_length: int, in_channels: int, shard_layout: Optional[ttnn.TensorMemoryLayout] = None
) -> ttnn.Conv2dConfig:
    conv_kwargs: dict = dict(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=shard_layout,
        deallocate_activation=True,
        # Emit RM so upsample → resblock conv1d stays RM (skips TILE→RM untilize at each stage).
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
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
    act_block_h = _vocoder_act_block_h_override(il, in_channels)
    if act_block_h is not None:
        conv_kwargs["act_block_h_override"] = act_block_h
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

        # HiFi4 math fidelity for waveform PCC (HiFi2 costs PCC margin for little gain; LoFi fails outright).
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
        self._last_unit_seq: Optional[int] = None

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

    def _expand_unit_embeddings_gather(
        self,
        use: ttnn.Tensor,
        *,
        batch: int,
        e_unit: int,
        seq: int,
        t_audio: int,
        cumsum_inc_bt: ttnn.Tensor,
        frame_idx: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Frame-expand unit embeddings via a per-frame gather (replaces the one-hot ``use @ H`` matmul).

        Frame ``t`` belongs to unit ``u(t) = #{s : cumsum_inc[s] <= t}`` (searchsorted-right): the
        unique unit whose half-open ``[cumsum_prev, cumsum_inc)`` interval contains ``t``. Bucket-pad
        frames (``t >= t_audio_real``, where every ``cumsum_inc[s] <= t``) get ``u(t) == seq`` and
        gather an appended all-zero row — matching the old matmul's all-zero column exactly.

        Returns ``[B, t_audio, E_unit]`` (NLC) to slot into the NLC lang/spk concat (channel dim).
        Replaces ``O(t_audio/32 * seq/32)`` tiny matmuls + adds + slices with one reduce + one gather.
        """
        assert batch == 1, "_expand_unit_embeddings_gather expects B == 1 (forward loops rows)."
        # unit index per frame = count of inclusive-cumulative durations <= t. Accumulate in fp32:
        # bf16 rounds integers above 256, which would corrupt indices for seq up to 1024.
        c_b = ttnn.reshape(cumsum_inc_bt, (batch, seq, 1))  # [B, T_units, 1]
        f_b = ttnn.reshape(frame_idx, (1, 1, t_audio))  # [1, 1, t_audio]
        le_mask = ttnn.le(c_b, f_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, T_units, t_audio]
        le_f32 = ttnn.typecast(le_mask, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(le_mask)
        unit_idx_f = ttnn.sum(le_f32, dim=1)  # [B, t_audio] (values in [0, seq])
        ttnn.deallocate(le_f32)
        unit_idx = ttnn.typecast(unit_idx_f, ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(unit_idx_f)
        unit_idx = ttnn.reshape(unit_idx, (batch, t_audio))
        unit_idx = ttnn.to_layout(unit_idx, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Gather table: use [B, T_units, E_unit] -> row-major [T_units + 1, E_unit] with a zero tail
        # row so out-of-range (padding) indices gather zeros. ``use`` itself is owned by the caller,
        # so never deallocate it here — only intermediates this method creates.
        if use.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            use_rm = ttnn.to_layout(use, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            use_rm = use
        use_tbl = ttnn.reshape(use_rm, (seq, e_unit))
        table = ttnn.pad(use_tbl, [(0, 1), (0, 0)], value=0.0)  # [T_units + 1, E_unit]
        seen: set[int] = set()
        for t in (use_tbl, use_rm):
            if t is use or t is table or id(t) in seen:
                continue
            seen.add(id(t))
            ttnn.deallocate(t)

        # RM embedding output is already NLC ``[B, t_audio, E_unit]`` — return it as-is so the front-door
        # concat stays NLC (no BCT permute here, no NLC permute in the caller) and feeds ``_conv1d`` directly.
        gathered = ttnn.embedding(unit_idx, weight=table, layout=ttnn.ROW_MAJOR_LAYOUT)  # [B, t_audio, E_unit]
        ttnn.deallocate(unit_idx)
        ttnn.deallocate(table)
        return gathered

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
        row_major_output: bool = False,
        shard_layout: Optional[ttnn.TensorMemoryLayout] = None,
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
            bool(row_major_output),
            shard_layout,
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
        row_major_output: bool = False,
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
            row_major_output=row_major_output,
            shard_layout=shard_layout,
        )
        if cache_key in self._conv1d_prepared_cache:
            return

        conv_config = _vocoder_conv1d_config(
            fused_post_activation,
            input_length=prep_len,
            in_channels=in_channels,
            shard_layout=shard_layout,
            timeline_chunked=timeline_chunked,
            row_major_output=row_major_output,
            device=self.device,
            batch=batch,
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
                _CONV_PRE_SHARD,
                in_channels=int(cp.in_channels),
                kernel_size=int(cp.kernel_size),
                input_length=tlen,
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
                        shard_layout=_resolve_conv_shard_layout(
                            _RESBLOCK_SHARD, in_channels=channels, kernel_size=k1, input_length=t_rb
                        ),
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
                        shard_layout=_resolve_conv_shard_layout(
                            _RESBLOCK_SHARD, in_channels=channels, kernel_size=k2, input_length=t_rb
                        ),
                    )
                    t_rb = _host_conv_out_length(t_rb, k2, 1, p2, dilation=d2)

        cpost = hg.conv_post
        _, cpost_chunked = _vocoder_conv1d_prep_length(
            tlen, in_channels=int(cpost.in_channels), padding=int(cpost.padding)
        )
        self._prepare_conv1d_weights_for_prewarm(
            weight=cpost.weight,
            bias=cpost.bias,
            batch=batch,
            input_length=tlen,
            in_channels=int(cpost.in_channels),
            out_channels=int(cpost.out_channels),
            kernel_size=int(cpost.kernel_size),
            padding=int(cpost.padding),
            row_major_output=not cpost_chunked,
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
        row_major_output: bool = False,
        keep_sharded_output: bool = False,
        slice_config: Optional[Any] = None,
        l1_full: bool = False,
    ) -> Tuple[ttnn.Tensor, int]:
        """Single-shot ``ttnn.conv1d``; input is row-major NLC or an already-sharded activation.

        When ``slice_config`` is given, ``ttnn.conv1d`` slices the timeline (width) in DRAM itself —
        used for long HiFi-GAN timelines in place of the manual Python chunk loop."""
        # Short single-shot length-bucketing (``<= _HIFIGAN_MAX_CONV1D_TLEN``): pad timeline to a fixed
        # bucket so few stable shapes compile once. Long/chunked timelines are bucketed in ``_conv1d``
        # (with sharded-aware trim via ``_slice_nlc_time``) before the chunk loop.
        bucket_pad = 0
        if (
            not timeline_chunked
            and int(stride) == 1
            and 0 < int(input_length) <= _HIFIGAN_MAX_CONV1D_TLEN
            and len(x_rm.shape) == 3
            and not ttnn.is_sharded(x_rm)
        ):
            bucket = _vocoder_conv1d_bucket(int(input_length))
            if bucket > int(input_length):
                x_pad = ttnn.pad(x_rm, [(0, 0), (0, bucket - int(input_length)), (0, 0)], value=0.0)
                if deallocate_input:
                    ttnn.deallocate(x_rm)
                x_rm = x_pad
                deallocate_input = True  # we now own the padded copy
                bucket_pad = bucket - int(input_length)
                input_length = bucket
        conv_config = _vocoder_conv1d_config(
            fused_post_activation,
            input_length=input_length,
            in_channels=in_channels,
            shard_layout=shard_layout,
            timeline_chunked=timeline_chunked,
            row_major_output=row_major_output,
            device=self.device,
            batch=batch,
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
            row_major_output=row_major_output,
            shard_layout=shard_layout,
        )
        cached = self._conv1d_prepared_cache.get(cache_key) if use_prepared_weights else None
        weight_tensor = cached[0] if cached is not None else weight
        bias_tensor = cached[1] if cached is not None else bias
        conv1d_kwargs: dict = dict(
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
        if slice_config is None and l1_full:
            # Keep the whole conv resident in L1 (input->compute->output), so a sharded output can hand
            # straight to the next conv with no DRAM reshard. Needs a small act_block_h (set above) to fit L1.
            slice_config = ttnn.Conv2dL1FullSliceConfig
        if slice_config is not None:
            conv1d_kwargs["slice_config"] = slice_config
        out, out_len = ttnn.conv1d(**conv1d_kwargs)
        if deallocate_input:
            ttnn.deallocate(x_rm)
        out_len = int(out_len)
        if not keep_sharded_output and ttnn.is_sharded(out):
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (batch, out_len, out_channels))
        if bucket_pad:
            keep = out_len - bucket_pad  # stride-1 conv: out_len tracks input_length 1:1
            out = _slice_nlc_time(out, batch=batch, start=0, end=keep, channels=out_channels)
            out_len = keep
        return out, out_len

    def _conv1d_apply_timeline_bucket(
        self,
        x_in: ttnn.Tensor,
        *,
        x_nlc: ttnn.Tensor,
        rm_buf: Optional[ttnn.Tensor],
        batch: int,
        seq: int,
        in_channels: int,
        accept_sharded_input: bool,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], int, int]:
        """Right-pad a stride-1 timeline to a stable bucket; returns (x_in, rm_buf, seq, real_seq)."""
        real_seq = int(seq)
        if _VOCODER_CONV1D_BUCKET_STEP <= 1:
            return x_in, rm_buf, real_seq, real_seq
        bucket = _vocoder_timeline_bucket(real_seq)
        if bucket <= real_seq:
            return x_in, rm_buf, real_seq, real_seq
        if accept_sharded_input and ttnn.is_sharded(x_in):
            x_in = ttnn.sharded_to_interleaved(x_in, ttnn.DRAM_MEMORY_CONFIG)
        elif x_in.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            if rm_buf is not None:
                ttnn.deallocate(rm_buf)
            rm_buf = ttnn.to_layout(x_in, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x_in = rm_buf
        x_padded, created = self._pad_nlc_time(
            x_in,
            batch=batch,
            tlen=real_seq,
            channels=in_channels,
            pad_to=bucket,
        )
        if created:
            if rm_buf is x_in:
                ttnn.deallocate(rm_buf)
                rm_buf = None
            elif x_in is not x_nlc:
                ttnn.deallocate(x_in)
            rm_buf = x_padded
            x_in = rm_buf
        return x_in, rm_buf, bucket, real_seq

    def _conv1d_trim_timeline(
        self,
        out: ttnn.Tensor,
        out_len: int,
        *,
        real_seq: int,
        batch: int,
        out_channels: int,
    ) -> Tuple[ttnn.Tensor, int]:
        """Drop bucket-padded tail rows; stride-1 conv keeps ``out_len == input_length``."""
        if int(out_len) <= int(real_seq):
            return out, int(out_len)
        out = _slice_nlc_time(out, batch=batch, start=0, end=int(real_seq), channels=out_channels)
        return out, int(real_seq)

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
        row_major_output: bool = False,
        keep_sharded_output: bool = False,
        accept_sharded_input: bool = False,
        accept_tile_input: bool = False,
        l1_full: bool = False,
    ) -> Tuple[ttnn.Tensor, int]:
        # ``ttnn.conv1d`` reshapes activations for the conv2d path; TILE NLC (e.g. from ``embedding``) can
        # hit "reshape between two shapes with different volumes" (host ROW_MAJOR weights are fine).
        # ``accept_tile_input`` skips the defensive TILE->RM untilize for conv-derived TILE input, which ``ttnn.conv1d`` consumes interleaved directly.
        seq = int(input_length)
        x_in = x_nlc
        rm_buf: Optional[ttnn.Tensor] = None
        if accept_sharded_input and ttnn.is_sharded(x_nlc):
            pass
        elif (
            accept_tile_input
            and int(stride) == 1
            and not ttnn.is_sharded(x_nlc)
            and x_nlc.get_layout() == ttnn.TILE_LAYOUT
            and _vocoder_timeline_bucket(seq) <= seq  # no right-pad bucket (else needs RM concat)
        ):
            pass  # feed TILE interleaved straight to ttnn.conv1d (no untilize)
        elif x_nlc.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            rm_buf = ttnn.to_layout(x_nlc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x_in = rm_buf

        real_seq = seq
        if int(stride) == 1:
            x_in, rm_buf, seq, real_seq = self._conv1d_apply_timeline_bucket(
                x_in,
                x_nlc=x_nlc,
                rm_buf=rm_buf,
                batch=batch,
                seq=seq,
                in_channels=in_channels,
                accept_sharded_input=accept_sharded_input,
            )

        # One conv1d fits BH L1 single-shot either when the timeline is short, or (L1-resident mode) when
        # it is HEIGHT-sharded and the activation fits the L1 element budget — no DRAM slicing, and the
        # sharded output can be handed straight to the next conv (``keep_sharded_output``).
        run_single_shot_l1 = (
            shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            and int(stride) == 1
            and _vocoder_fits_l1_singleshot(seq, in_channels, kernel_size, batch)
        )
        # The single-shot HEIGHT-sharded L1 fit is an estimate; on a marginal overflow (op-slicing
        # "found_valid_config" or a CB/L1 clash) for the long-timeline path, keep ``x_in`` alive and fall
        # back to the DRAM width-slice below, which handles any timeline (S2ST @ 2048/4096: some resblock stages sit just past the true L1 limit).
        single_can_fallback = run_single_shot_l1 and seq > _HIFIGAN_MAX_CONV1D_TLEN
        if seq <= _HIFIGAN_MAX_CONV1D_TLEN or run_single_shot_l1:
            try:
                out, out_len = self._conv1d_run(
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
                    deallocate_input=(rm_buf is not None) and not single_can_fallback,
                    shard_layout=shard_layout,
                    timeline_chunked=False,
                    row_major_output=row_major_output,
                    keep_sharded_output=keep_sharded_output,
                    l1_full=l1_full and run_single_shot_l1,
                )
            except RuntimeError:
                if not single_can_fallback:
                    raise
                # x_in preserved (dealloc deferred); drop through to the DRAM width-slice below.
            else:
                if single_can_fallback and rm_buf is not None:
                    ttnn.deallocate(x_in)
                return self._conv1d_trim_timeline(
                    out, out_len, real_seq=real_seq, batch=batch, out_channels=out_channels
                )

        # Long timeline / single-shot fallback: HEIGHT sharding is only valid for the single-shot L1 conv
        # above; leaving it set here propagates a too-large shard spec into the DRAM-slice conv and makes
        # it fail too, so drop back to auto sharding.
        shard_layout = None
        # ``ttnn.conv1d`` width-slices the timeline in DRAM itself (one op, device-managed halo).
        # DRAM width-slicing needs the activation in DRAM; interleave a sharded input first.
        if ttnn.is_sharded(x_in):
            x_int = ttnn.sharded_to_interleaved(x_in, ttnn.DRAM_MEMORY_CONFIG)
            if rm_buf is not None:
                ttnn.deallocate(rm_buf)
            elif x_in is not x_nlc:
                ttnn.deallocate(x_in)
            x_in = x_int
            rm_buf = x_int
        out, out_len = self._conv1d_run(
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
            keep_sharded_output=False,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
        )
        return self._conv1d_trim_timeline(out, out_len, real_seq=real_seq, batch=batch, out_channels=out_channels)

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
            layout=ttnn.ROW_MAJOR_LAYOUT,
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
        num_slices = _vocoder_dram_slice_count(input_length, in_channels)
        pad_h = _vocoder_dram_slice_pad_h(input_length, in_channels=in_channels)
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

        # ``conv_transpose2d`` needs NHWC ([B,T,1,C]); the [B,T,C]->[B,T,1,C] reshape is a free view on
        # ROW_MAJOR but a full relayout on TILE. Untilize once here so the reshape is free and the conv
        # gets its natural ROW_MAJOR input.
        if x_nlc.get_layout() != ttnn.ROW_MAJOR_LAYOUT and not ttnn.is_sharded(x_nlc):
            x_rm = ttnn.to_layout(x_nlc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x_nlc)
            x_nlc = x_rm

        real_len = int(input_length)
        work_len = _vocoder_timeline_bucket(real_len)
        x_work = x_nlc
        padded_in = False
        if work_len > real_len:
            x_work, padded_in = self._pad_nlc_time(
                x_nlc,
                batch=batch,
                tlen=real_len,
                channels=in_channels,
                pad_to=work_len,
            )
            input_length = work_len
        real_out_h = _host_transpose_conv_out_length(real_len, k, s, p)

        # HEIGHT per DRAM slice when slice count and K are within caps; else auto layout.
        sliced = int(input_length) > 64
        num_slices = _vocoder_dram_slice_count(input_length, in_channels) if sliced else 0
        prefer = _UPSAMPLE_SHARD if (not sliced or num_slices <= _UPSAMPLE_HEIGHT_MAX_SLICES) else None
        config_len = (
            _vocoder_dram_slice_pad_h(input_length, in_channels=in_channels)
            if sliced and int(in_channels) >= 512
            else input_length
        )
        conv_config = _vocoder_conv2d_config(
            input_length=config_len,
            in_channels=in_channels,
            shard_layout=_resolve_conv_shard_layout(
                prefer, in_channels=in_channels, kernel_size=k, input_length=input_length
            ),
        )
        if sliced:
            out_nlc, out_h = self._conv_transpose1d_nlc_dram_sliced(
                x_work,
                layer=layer,
                batch=batch,
                input_length=input_length,
                in_channels=in_channels,
                out_channels=out_channels,
                conv_config=conv_config,
            )
            if padded_in and x_work is not x_nlc:
                ttnn.deallocate(x_work)
            if out_h > real_out_h:
                out_nlc = _slice_nlc_time(out_nlc, batch=batch, start=0, end=real_out_h, channels=out_channels)
            return out_nlc, real_out_h

        x_nhwc = ttnn.reshape(x_work, (batch, input_length, 1, in_channels))
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
        if padded_in and x_work is not x_nlc:
            ttnn.deallocate(x_work)
        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        assert out_w == 1
        out_nlc = ttnn.reshape(out_4d, (batch, out_h, out_channels))
        if ttnn.is_sharded(out_nlc):
            out_nlc = ttnn.sharded_to_interleaved(out_nlc, ttnn.DRAM_MEMORY_CONFIG)
        if out_h > real_out_h:
            out_nlc = _slice_nlc_time(out_nlc, batch=batch, start=0, end=real_out_h, channels=out_channels)
        return out_nlc, real_out_h

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

    def _resblock(
        self,
        x_nlc: ttnn.Tensor,
        rb: Any,
        *,
        batch: int,
        tlen: int,
        channels: int,
        first_leaky: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """One HF ``HifiGanResidualBlock``; ``x`` is ``[B,T,C]``.

        ``first_leaky`` is a precomputed ``leaky_relu(x_nlc)`` for the first conv pair — the caller shares
        it across the stage's ``num_kernels`` resblocks (all fed the same stage input), so ``leaky(h)`` runs
        once instead of once per resblock. The residual still uses the raw (non-activated) ``x_nlc``."""
        for idx, (c1p, c2p) in enumerate(zip(rb.convs1, rb.convs2)):
            residual = x_nlc
            if idx == 0 and first_leaky is not None:
                x_nlc = first_leaky
            else:
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
                shard_layout=_resolve_resblock_shard_layout(
                    in_channels=channels,
                    kernel_size=int(c1p["kernel_size"]),
                    input_length=tlen,
                    batch=batch,
                ),
                keep_sharded_output=True,
                # Conv-derived TILE input: ttnn.conv1d consumes it directly, skipping the TILE->RM untilize.
                accept_tile_input=True,
                # L1-full: keep conv1's output sharded in L1 so conv2 consumes it there (no DRAM reshard).
                l1_full=True,
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
                shard_layout=_resolve_resblock_shard_layout(
                    in_channels=channels,
                    kernel_size=int(c2p["kernel_size"]),
                    input_length=tlen,
                    batch=batch,
                ),
                accept_sharded_input=True,
                accept_tile_input=True,
                l1_full=True,
            )
            x_nlc = ttnn.add(x_nlc, residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x_nlc

    def _hifi_gan(self, x_nlc: ttnn.Tensor, hg: Any, *, batch: int, tlen: int) -> ttnn.Tensor:
        """HiFi-GAN stack; long ``conv1d`` timelines are chunked inside ``_conv1d`` (not mel splits)."""
        return self._hifi_gan_once(x_nlc, hg, batch=batch, tlen=tlen)

    def _hifi_gan_once(self, x_nlc: ttnn.Tensor, hg: Any, *, batch: int, tlen: int) -> ttnn.Tensor:
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
                _CONV_PRE_SHARD,
                in_channels=int(cp.in_channels),
                kernel_size=int(cp.kernel_size),
                input_length=tlen,
            ),
        )
        ttnn.deallocate(x_nlc)

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
            channels = self.cfg.upsample_initial_channel // (2 ** (i + 1))
            # The transpose emits ROW_MAJOR ``h``, but the resblock convs want TILE: a RM->sharded conv
            # I2S tilizes on the fly (~10x costlier than TILE->sharded) and the residual add re-tilizes RM.
            # Tilize ``h`` once per stage (shared as residual + first-leaky input across all resblocks).
            if h.get_layout() != ttnn.TILE_LAYOUT:
                h_t = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(h)
                h = h_t
            # All ``num_kernels`` resblocks consume the same stage input ``h`` and each starts with
            # ``leaky_relu(h)``; compute it once and share (residual still uses the raw ``h``).
            h_leaky = ttnn.leaky_relu(h, negative_slope=self.leaky_slope, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            acc = None
            for j in range(self.num_kernels):
                rb = hg.resblocks[i * self.num_kernels + j]
                br = self._resblock(h, rb, batch=batch, tlen=tlen, channels=channels, first_leaky=h_leaky)
                if acc is None:
                    acc = br
                else:
                    acc = ttnn.add(acc, br, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(br)
            ttnn.deallocate(h_leaky)
            scale = 1.0 / float(self.num_kernels)
            acc_scaled = ttnn.multiply(acc, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(acc)
            ttnn.deallocate(h)
            h = acc_scaled

        # HF applies leaky_relu with the default 0.01 slope here (not ``cfg.leaky_relu_slope``, which the
        # upsample-loop and residual-block leaky_relus use).
        h = ttnn.leaky_relu(h, negative_slope=0.01, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cpost = hg.conv_post
        _, cpost_chunked = _vocoder_conv1d_prep_length(
            tlen, in_channels=int(cpost.in_channels), padding=int(cpost.padding)
        )
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
            row_major_output=not cpost_chunked,
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
        self._last_unit_seq = seq
        assert batch == 1, "_forward_one expects B == 1; use forward() for B > 1."

        # Tables uploaded ROW_MAJOR (``_vocoder_embedding_weight_row_major``); ``layout`` here is the *output* layout.
        # Unit emb stays TILE for duration LN/linear; expand gather + HiFi front-door use RM (below).
        ue = self.p.unit_embedding.weight
        use = ttnn.embedding(input_ids, weight=ue, layout=ttnn.TILE_LAYOUT)  # [B, T_units, E_unit]

        # Lang/spk feed only the NLC front-door concat below, so emit them ROW_MAJOR to skip a later
        # TILE→RM untilize (they broadcast + concat directly with the RM gather output).
        sp = self.p.speaker_embedding.weight
        la = self.p.language_embedding.weight
        sp_e = ttnn.embedding(ttnn.squeeze(speaker_id, 1), weight=sp, layout=ttnn.ROW_MAJOR_LAYOUT)
        lang_e = ttnn.embedding(ttnn.squeeze(lang_id, 1), weight=la, layout=ttnn.ROW_MAJOR_LAYOUT)

        dp = self.p.dur_predictor
        e_unit = int(dp.conv1.in_channels)
        # Drop TILE padding on the time / channel axes so ``seq`` matches ``cumsum_*`` and
        # expand gather sees logical ``[B, T, E]``.
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

        # Use float32 internally so integer comparisons are exact for any plausible t_audio.
        dur_f32 = ttnn.typecast(dur_bf, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cumsum_inc = ttnn.cumsum(dur_f32, dim=-1, dtype=ttnn.float32)  # [B, T_units] inclusive
        ttnn.deallocate(dur_f32)
        ttnn.deallocate(dur_bf)

        cumsum_inc_bt = _as_batch_time_2d(cumsum_inc, batch=batch, seq=seq)

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

        # ``use @ H`` (H a one-hot mask) is exactly a gather: frame ``t`` copies the unit whose
        # ``[cumsum_prev, cumsum_inc)`` interval contains it; padded frames (``t >= t_audio_real``) gather
        # an all-zero row. Replaces the ``O(t_audio/32 * seq/32)`` tiny-matmul + slice + concat with one reduce + gather.
        frame_idx = self._cached_frame_idx_f32(t_audio)
        expanded_NLC = self._expand_unit_embeddings_gather(
            use,
            batch=batch,
            e_unit=e_unit,
            seq=seq,
            t_audio=t_audio,
            cumsum_inc_bt=cumsum_inc_bt,
            frame_idx=frame_idx,
        )

        # Build ``merged_NLC = [lang | unit | spk]`` directly in NLC (concat on channel dim): lang/spk go
        # RM ``[B, C]`` -> ``[B, 1, C]`` (free view) -> repeat over time, and the gather output is already
        # RM NLC. This avoids the old channel-major path's two permutes (gather->BCT, merged->NLC) and three TILE->RM untilizes.
        lang_dim = int(lang_e.shape[-1])
        spk_dim = int(sp_e.shape[-1])
        lang_1LC = ttnn.reshape(lang_e, (batch, 1, lang_dim))
        spk_1LC = ttnn.reshape(sp_e, (batch, 1, spk_dim))
        if t_audio == 1:
            lang_NLC = lang_1LC
            spk_NLC = spk_1LC
        else:
            lang_NLC = ttnn.repeat(lang_1LC, [1, t_audio, 1])
            spk_NLC = ttnn.repeat(spk_1LC, [1, t_audio, 1])
            ttnn.deallocate(lang_1LC)
            ttnn.deallocate(spk_1LC)

        merged_NLC = ttnn.concat([lang_NLC, expanded_NLC, spk_NLC], dim=2)
        ttnn.deallocate(lang_NLC)
        ttnn.deallocate(expanded_NLC)
        ttnn.deallocate(spk_NLC)
        ttnn.deallocate(use)

        wav = self._hifi_gan(merged_NLC, self.p.hifi_gan, batch=batch, tlen=t_audio)

        # Real (un-bucketed) length so the bucket-padded waveform tail is cropped out by consumers.
        lengths = self._output_lengths_dev(t_audio_real, batch=batch)
        ttnn.deallocate(cumsum_inc)
        return wav, lengths

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
            # Best-effort trace release during teardown; local runtime state is cleared below.
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
