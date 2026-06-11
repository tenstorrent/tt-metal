# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audio tokenizer convs via ``ttnn.conv1d``. Forward takes ``[B, 1, T, C_in]`` tile."""

from __future__ import annotations

import math
import os

from loguru import logger
import ttnn


# Chunk long conv outputs to avoid P150 L1 pressure.
MAX_CONV_TRANSPOSE_OUTPUT_CHUNK = 1024


def _conv_debug_enabled() -> bool:
    return os.environ.get("VOXTRAL_TTS_CONV_DEBUG", "").lower() in ("1", "true", "yes", "on")


def _tensor_debug(tensor: ttnn.Tensor) -> str:
    try:
        mem = tensor.memory_config()
    except Exception as exc:  # pragma: no cover - debug-only path
        mem = f"<memory_config unavailable: {exc}>"
    try:
        layout = tensor.layout
    except Exception as exc:  # pragma: no cover - debug-only path
        layout = f"<layout unavailable: {exc}>"
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} layout={layout} memory_config={mem}"


def _sharded_to_interleaved_dram(tensor: ttnn.Tensor) -> ttnn.Tensor:
    if tensor.memory_config().is_sharded():
        out = ttnn.sharded_to_interleaved(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tensor)
        return out
    return tensor


def _deshard_split_result(tensor: ttnn.Tensor, *, keep_sharded: bool) -> ttnn.Tensor:
    """Leave height-sharded L1 activations intact until an interleaved op is required."""
    if keep_sharded and tensor.memory_config().is_sharded():
        return tensor
    if tensor.memory_config().is_sharded():
        return ttnn.sharded_to_interleaved(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return tensor


def _causal_conv1d_pad_amounts(seq_len: int, kernel_size: int, stride: int, dilation: int = 1) -> tuple[int, int]:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    padding_total = effective_kernel_size - stride
    n_frames = (seq_len - effective_kernel_size + padding_total) / stride + 1
    target_length = (math.ceil(n_frames) - 1) * stride + (effective_kernel_size - padding_total)
    return int(padding_total), int(target_length - seq_len)


def _fuse_weight_norm_conv1d(g, v):
    norm = v.reshape(v.shape[0], -1).norm(dim=1).reshape(-1, 1, 1)
    return (g * (v / norm)).contiguous()


def resolve_input_proj_conv_weight(state_dict: dict):
    """Return ``[out_ch, in_ch, k]`` conv weight for ``input_proj`` (checkpoint keys)."""
    plain = "input_proj.conv.weight"
    if plain in state_dict:
        w = state_dict[plain]
        if w.dim() != 3:
            raise ValueError(f"Expected 3D conv weight for {plain}, got shape {tuple(w.shape)}")
        return w
    gk = "input_proj.conv.parametrizations.weight.original0"
    vk = "input_proj.conv.parametrizations.weight.original1"
    if gk in state_dict and vk in state_dict:
        return _fuse_weight_norm_conv1d(state_dict[gk], state_dict[vk])
    raise KeyError(
        "No input_proj conv weight found in state_dict (tried " + ", ".join(f"'{c}'" for c in (plain, gk, vk)) + ")."
    )


def get_audio_tokenizer_conv_configs(device):
    conv1d_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        config_tensors_in_dram=True,
        act_block_h_override=32,
    )
    conv1d_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    return conv1d_config, conv1d_compute_config


def get_audio_tokenizer_conv_transpose_configs(device):
    """Conv transpose config: ``packer_l1_acc=False`` avoids P150 static-CB clashes on long seqs."""
    conv1d_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        config_tensors_in_dram=True,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        act_block_h_override=32,
    )
    conv1d_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    return conv1d_config, conv1d_compute_config


class VoxtralTTAudioTokenizerInputProj:
    def __init__(
        self,
        device,
        *,
        state_dict: dict,
        in_channels: int = 240,
        out_channels: int = 1024,
        kernel_size: int = 7,
        stride: int = 1,
        causal: bool = True,
        pad_mode: str = "reflect",
        weight_dtype=ttnn.bfloat16,
        activations_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
    ) -> None:
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.left_pad = (kernel_size - 1) if causal else 0
        # Upstream input_proj is a CausalConv1d with the default pad_mode="reflect".
        self.pad_mode = pad_mode
        self.activations_dtype = activations_dtype
        self.output_dtype = output_dtype
        self.debug_name = "input_proj.conv"

        w_host = resolve_input_proj_conv_weight(state_dict).contiguous()
        w_dtype = ttnn.float32 if weight_dtype == ttnn.bfloat8_b else weight_dtype
        self.weight_tensor = ttnn.from_torch(
            w_host,
            dtype=w_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
        )

        self.conv_config, self.compute_config = get_audio_tokenizer_conv_configs(device)

    def __call__(self, mel_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """``mel_b1tc``: ``[B, 1, T, C_in]`` tile BF16 → ``[B, 1, T, C_out]`` tile."""
        if len(mel_b1tc.shape) != 4 or mel_b1tc.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T, C], got shape {tuple(mel_b1tc.shape)}")
        b, _, t, c = (int(mel_b1tc.shape[i]) for i in range(4))
        if c != self.in_channels:
            raise ValueError(f"Expected C_in={self.in_channels}, got {c}")

        x_rm = ttnn.to_layout(mel_b1tc, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, (b, t, c))

        if self.left_pad:
            if self.pad_mode == "reflect":
                x_rm = _reflect_pad_time_rm(
                    x_rm, pad_left=self.left_pad, pad_right=0, dtype=self.activations_dtype, device=self.device
                )
            elif self.pad_mode == "constant":
                pad = ttnn.zeros(
                    (b, self.left_pad, c),
                    dtype=self.activations_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                x_rm = ttnn.concat([pad, x_rm], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                raise ValueError(f"Unsupported pad_mode={self.pad_mode!r}")

        padded_len = t + self.left_pad

        if _conv_debug_enabled():
            logger.info(
                "[voxtraltts][conv1d] {} pre-conv: B={} T={} C_in={} C_out={} kernel={} stride={} padded_len={} "
                "input_rm={} weight_host_shape={}",
                self.debug_name,
                b,
                t,
                c,
                self.out_channels,
                self.kernel_size,
                self.stride,
                padded_len,
                _tensor_debug(x_rm),
                tuple(self.weight_tensor.shape),
            )
        conv_out, [self.weight_tensor, _] = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=self.weight_tensor,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=None,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            batch_size=b,
            input_length=padded_len,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            dtype=self.output_dtype,
            return_weights_and_bias=True,
        )

        conv_out = ttnn.sharded_to_interleaved(conv_out)

        out_len = int(conv_out.shape[1])
        if out_len > t:
            conv_out = ttnn.slice(conv_out, [0, 0, 0], [b, t, self.out_channels])
        elif out_len < t:
            raise RuntimeError(f"conv1d output length {out_len} < input time {t}")

        out4 = ttnn.reshape(conv_out, (b, 1, t, self.out_channels))
        return ttnn.to_layout(out4, ttnn.TILE_LAYOUT)


def resolve_encoder_block_strided_conv_weight(state_dict: dict, block_index: int):
    """Conv weight for ``encoder_blocks.{i}`` strided ``CausalConv1d`` (``[out, in, k]``)."""
    base = f"encoder_blocks.{block_index}.conv"
    plain = f"{base}.weight"
    if plain in state_dict:
        w = state_dict[plain]
        if w.dim() != 3:
            raise ValueError(f"Expected 3D conv weight for {plain}, got shape {tuple(w.shape)}")
        return w
    gk = f"{base}.parametrizations.weight.original0"
    vk = f"{base}.parametrizations.weight.original1"
    if gk in state_dict and vk in state_dict:
        return _fuse_weight_norm_conv1d(state_dict[gk], state_dict[vk])
    raise KeyError(
        f"No conv weight for encoder_blocks.{block_index} (tried {plain} and weight-norm parametrizations original0/1)."
    )


class VoxtralTTAudioTokenizerEncoderDownsampleConv:
    """Strided causal ``conv1d`` (e.g. ``encoder_blocks.1``: 1024→1024, ``k=4``, ``s=2``)."""

    def __init__(
        self,
        device,
        *,
        state_dict: dict,
        block_index: int = 1,
        in_channels: int = 1024,
        out_channels: int = 1024,
        kernel_size: int = 4,
        stride: int = 2,
        causal: bool = True,
        weight_dtype=ttnn.bfloat16,
        activations_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
    ) -> None:
        self.device = device
        self.block_index = block_index
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.left_pad = (kernel_size - 1) if causal else 0
        self.activations_dtype = activations_dtype
        self.output_dtype = output_dtype

        w_host = resolve_encoder_block_strided_conv_weight(state_dict, block_index).contiguous()
        w_dtype = ttnn.float32 if weight_dtype == ttnn.bfloat8_b else weight_dtype
        self.weight_tensor = ttnn.from_torch(
            w_host,
            dtype=w_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
        )
        self.conv_config, self.compute_config = get_audio_tokenizer_conv_configs(device)

    def expected_output_length(self, input_time: int) -> int:
        """Logical output time after left causal pad (matches ``F.conv1d`` ``padding=0``)."""
        padded = input_time + self.left_pad
        return (padded - self.kernel_size) // self.stride + 1

    def __call__(self, x_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, 1, T, C_in]`` tile → ``[B, 1, T', C_out]`` tile with ``T' = expected_output_length(T)``."""
        if len(x_b1tc.shape) != 4 or x_b1tc.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T, C], got shape {tuple(x_b1tc.shape)}")
        b, _, t, c = (int(x_b1tc.shape[i]) for i in range(4))
        if c != self.in_channels:
            raise ValueError(f"Expected C_in={self.in_channels}, got {c}")

        x_rm = ttnn.to_layout(x_b1tc, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, (b, t, c))

        if self.left_pad:
            pad = ttnn.zeros(
                (b, self.left_pad, c),
                dtype=self.activations_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x_rm = ttnn.concat([pad, x_rm], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        padded_len = t + self.left_pad
        t_out_expect = self.expected_output_length(t)

        conv_out, [self.weight_tensor, _] = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=self.weight_tensor,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=None,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            batch_size=b,
            input_length=padded_len,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            dtype=self.output_dtype,
            return_weights_and_bias=True,
        )

        conv_out = ttnn.sharded_to_interleaved(conv_out)
        out_len = int(conv_out.shape[1])
        if out_len > t_out_expect:
            conv_out = ttnn.slice(conv_out, [0, 0, 0], [b, t_out_expect, self.out_channels])
        elif out_len < t_out_expect:
            raise RuntimeError(f"conv1d length {out_len} < expected {t_out_expect}")

        out4 = ttnn.reshape(conv_out, (b, 1, t_out_expect, self.out_channels))
        return ttnn.to_layout(out4, ttnn.TILE_LAYOUT)


def resolve_causal_conv1d_fused_weight(state_dict: dict, base: str):
    """Fused ``[out, in, k]`` conv weight for ``{base}`` (plain or weight-norm parametrization)."""
    if f"{base}.weight" in state_dict:
        w = state_dict[f"{base}.weight"]
        if w.dim() != 3:
            raise ValueError(f"Expected 3D conv weight for {base}.weight, got {tuple(w.shape)}")
        return w
    gk = f"{base}.parametrizations.weight.original0"
    vk = f"{base}.parametrizations.weight.original1"
    if gk in state_dict and vk in state_dict:
        return _fuse_weight_norm_conv1d(state_dict[gk], state_dict[vk])
    raise KeyError(f"No conv weight for {base} (tried {base}.weight and weight-norm parametrizations original0/1).")


def resolve_decoder_block_causal_conv_fused_weight(state_dict: dict, block_index: int):
    """Fused ``[out, in, k]`` conv weight for ``decoder_blocks.{i}.conv`` (plain or weight-norm)."""
    return resolve_causal_conv1d_fused_weight(state_dict, f"decoder_blocks.{block_index}.conv")


def resolve_output_proj_causal_conv_fused_weight(state_dict: dict):
    """Fused weight for ``output_proj.conv``."""
    return resolve_causal_conv1d_fused_weight(state_dict, "output_proj.conv")


def resolve_decoder_block_conv_transpose_fused_weight(state_dict: dict, block_index: int):
    """Fused ``[in, out, k]`` weight for ``decoder_blocks.{i}.conv`` transpose conv."""
    return resolve_decoder_block_causal_conv_fused_weight(state_dict, block_index)


def _replicate_pad_time_rm(
    x_btc: ttnn.Tensor,
    *,
    pad_left: int,
    pad_right: int,
) -> ttnn.Tensor:
    """Time-axis replicate pad on ``[B,T,C]`` RM (matches ``F.pad(..., mode='replicate')`` on the time slice)."""
    b, t, c = (int(x_btc.shape[i]) for i in range(3))
    parts: list[ttnn.Tensor] = []
    if pad_left:
        left = ttnn.slice(x_btc, [0, 0, 0], [b, 1, c])
        for _ in range(pad_left):
            parts.append(left)
    parts.append(x_btc)
    if pad_right:
        right = ttnn.slice(x_btc, [0, t - 1, 0], [b, t, c])
        for _ in range(pad_right):
            parts.append(right)
    return ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if len(parts) > 1 else x_btc


def _reflect_pad_time_rm(
    x_btc: ttnn.Tensor,
    *,
    pad_left: int,
    pad_right: int,
    dtype,
    device,
) -> ttnn.Tensor:
    """Time-axis reflect pad on ``[B,T,C]`` RM (matches upstream ``pad1d(..., mode='reflect')``).

    Reflection excludes the edge sample (torch semantics). The short-sequence case (``T <= max_pad``)
    is handled exactly like upstream: right-extend with zeros, reflect, then trim the extension.
    """
    if pad_left == 0 and pad_right == 0:
        return x_btc
    b, t, c = (int(x_btc.shape[i]) for i in range(3))
    max_pad = max(pad_left, pad_right)
    extra = 0
    work = x_btc
    if t <= max_pad:
        extra = max_pad - t + 1
        z = ttnn.zeros(
            (b, extra, c),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        work = ttnn.concat([x_btc, z], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(z)
    wt = int(work.shape[1])
    parts: list[ttnn.Tensor] = []
    for i in range(pad_left, 0, -1):  # indices pad_left .. 1 (reflect around frame 0)
        parts.append(ttnn.slice(work, [0, i, 0], [b, i + 1, c]))
    parts.append(work)
    for i in range(wt - 2, wt - 2 - pad_right, -1):  # indices wt-2 .. wt-1-pad_right (reflect around last frame)
        parts.append(ttnn.slice(work, [0, i, 0], [b, i + 1, c]))
    padded = ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if len(parts) > 1 else work
    if extra:
        end = int(padded.shape[1]) - extra
        padded = ttnn.slice(padded, [0, 0, 0], [b, end, c])
    return padded


def _zero_insert_time_rm(x_btc: ttnn.Tensor, *, stride: int, dtype, device) -> ttnn.Tensor:
    """Insert ``stride - 1`` all-zero frames after each input frame on ``[B,T,C]`` RM.

    Equivalent to the per-frame loop (``f0, 0, …, f1, 0, …, f_{T-1}``) but uses one
    interleaved reshape+concat instead of ``T`` ``ttnn.slice`` ops per upsample stage.
    """
    if stride == 1:
        return x_btc
    b, t, c = (int(x_btc.shape[i]) for i in range(3))
    mem = ttnn.DRAM_MEMORY_CONFIG
    # [B,T,C] → [B,T,1,C]; concat zeros on the small axis → [B,T,stride,C] → [B,T*stride,C].
    x4 = ttnn.reshape(x_btc, (b, t, 1, c))
    z = ttnn.zeros(
        (b, t, stride - 1, c),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem,
    )
    interleaved = ttnn.concat([x4, z], dim=2, memory_config=mem)
    ttnn.deallocate(x4)
    ttnn.deallocate(z)
    flat = ttnn.reshape(interleaved, (b, t * stride, c))
    ttnn.deallocate(interleaved)
    out_len = t * stride - (stride - 1)
    out = ttnn.slice(flat, [0, 0, 0], [b, out_len, c], memory_config=mem)
    ttnn.deallocate(flat)
    return out


class VoxtralTTAudioTokenizerDecoderCausalConv1d:
    """``CausalConv1d``-compatible path (e.g. ``decoder_blocks.0``: latent → ``dim``)."""

    def __init__(
        self,
        device,
        *,
        state_dict: dict,
        block_index: int | None = None,
        conv_weight_base: str | None = None,
        kernel_size: int,
        stride: int,
        pad_mode: str = "replicate",
        dilation: int = 1,
        in_channels: int | None = None,
        out_channels: int | None = None,
        output_channel_splits: int = 4,
        input_channel_splits: int = 1,
        keep_sharded_splits: bool = False,
        weight_dtype=ttnn.bfloat16,
        activations_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
    ) -> None:
        self.device = device
        self.block_index = block_index if block_index is not None else -1
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad_mode = pad_mode
        self.output_channel_splits = output_channel_splits
        self.input_channel_splits = input_channel_splits
        self.keep_sharded_splits = keep_sharded_splits
        self.activations_dtype = activations_dtype
        self.output_dtype = output_dtype
        self.debug_name = (
            conv_weight_base if conv_weight_base is not None else f"decoder_blocks.{self.block_index}.conv"
        )
        self.max_conv_output_chunk = MAX_CONV_TRANSPOSE_OUTPUT_CHUNK

        if conv_weight_base is not None:
            w_host = resolve_causal_conv1d_fused_weight(state_dict, conv_weight_base).contiguous()
        elif block_index is not None:
            w_host = resolve_decoder_block_causal_conv_fused_weight(state_dict, block_index).contiguous()
        else:
            raise ValueError("Provide block_index or conv_weight_base for conv weights.")
        oc, ic, k = (int(w_host.shape[0]), int(w_host.shape[1]), int(w_host.shape[2]))
        if in_channels is not None and ic != in_channels:
            raise ValueError(f"Checkpoint expects in_channels={ic}, got {in_channels}")
        if out_channels is not None and oc != out_channels:
            raise ValueError(f"Checkpoint expects out_channels={oc}, got {out_channels}")
        if kernel_size != k:
            raise ValueError(f"Checkpoint kernel k={k}, expected kernel_size={kernel_size}")
        self.in_channels = ic
        self.out_channels = oc
        self.kernel_size = k
        if self.out_channels % self.output_channel_splits != 0:
            raise ValueError(
                f"out_channels={self.out_channels} must divide evenly by output_channel_splits={self.output_channel_splits}"
            )
        self.out_channels_per_split = self.out_channels // self.output_channel_splits
        if self.in_channels % self.input_channel_splits != 0:
            raise ValueError(
                f"in_channels={self.in_channels} must divide evenly by input_channel_splits={self.input_channel_splits}"
            )
        self.in_channels_per_split = self.in_channels // self.input_channel_splits

        w_dtype = ttnn.float32 if weight_dtype == ttnn.bfloat8_b else weight_dtype
        self.weight_tensors = [
            [
                ttnn.from_torch(
                    w_host[
                        o * self.out_channels_per_split : (o + 1) * self.out_channels_per_split,
                        i * self.in_channels_per_split : (i + 1) * self.in_channels_per_split,
                        :,
                    ].contiguous(),
                    dtype=w_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=None,
                )
                for i in range(self.input_channel_splits)
            ]
            for o in range(self.output_channel_splits)
        ]
        self.conv_config, self.compute_config = get_audio_tokenizer_conv_configs(device)

    def _conv_summed_over_inputs(self, inp: ttnn.Tensor, input_length: int, b: int, weight_group: list) -> ttnn.Tensor:
        """One output-channel split: conv each input-channel split and sum partials."""
        keep_sharded = self.keep_sharded_splits
        accum = None
        for in_idx, weight_tensor in enumerate(weight_group):
            if self.input_channel_splits == 1:
                x_in = inp
            else:
                ic0 = in_idx * self.in_channels_per_split
                slice_mem = None if keep_sharded else ttnn.DRAM_MEMORY_CONFIG
                x_in = ttnn.slice(
                    inp,
                    [0, 0, ic0],
                    [b, input_length, ic0 + self.in_channels_per_split],
                    memory_config=slice_mem,
                )
            conv_kwargs = dict(
                input_tensor=x_in,
                weight_tensor=weight_tensor,
                in_channels=self.in_channels_per_split,
                out_channels=self.out_channels_per_split,
                device=self.device,
                bias_tensor=None,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0,
                batch_size=b,
                input_length=input_length,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=1,
                dtype=self.output_dtype,
                return_weights_and_bias=False,
            )
            if keep_sharded:
                out_i = ttnn.conv1d(**conv_kwargs)
            else:
                out_i = ttnn.conv1d(**conv_kwargs, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if x_in is not inp and x_in.is_allocated():
                ttnn.deallocate(x_in)
            if not keep_sharded and out_i.memory_config().is_sharded():
                out_i = ttnn.sharded_to_interleaved(out_i, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if accum is None:
                accum = out_i
            else:
                add_mem = accum.memory_config() if keep_sharded else ttnn.DRAM_MEMORY_CONFIG
                summed = ttnn.add(accum, out_i, memory_config=add_mem)
                ttnn.deallocate(accum)
                ttnn.deallocate(out_i)
                accum = summed
        return _deshard_split_result(accum, keep_sharded=keep_sharded)

    def _concat_output_channel_splits(self, out_splits: list[ttnn.Tensor]) -> ttnn.Tensor:
        channel_dim = len(out_splits[0].shape) - 1
        if self.keep_sharded_splits:
            ready = [_sharded_to_interleaved_dram(split) for split in out_splits]
            return ttnn.concat(ready, dim=channel_dim)
        return ttnn.concat(out_splits, dim=channel_dim)

    def _concat_time_chunks(self, split_chunks: list[ttnn.Tensor]) -> ttnn.Tensor:
        if len(split_chunks) == 1:
            return split_chunks[0]
        time_dim = 2 if len(split_chunks[0].shape) == 4 else 1
        if self.keep_sharded_splits:
            split_chunks = [_sharded_to_interleaved_dram(chunk) for chunk in split_chunks]
        return ttnn.concat(split_chunks, dim=time_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def padded_sequence_length(self, time_len: int) -> tuple[int, int, int]:
        """``(pad_left, pad_right, padded_len)`` for this conv's causal padding."""
        pl, pr = _causal_conv1d_pad_amounts(time_len, self.kernel_size, self.stride, self.dilation)
        return pl, pr, int(time_len + pl + pr)

    def expected_output_length(self, time_len: int) -> int:
        pl, pr, l_in = self.padded_sequence_length(time_len)
        _ = pl, pr
        return (l_in - self.kernel_size) // self.stride + 1

    def __call__(self, x_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, 1, T, C_in]`` tile → ``[B, 1, T_out, C_out]`` tile."""
        if len(x_b1tc.shape) != 4 or x_b1tc.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T, C], got shape {tuple(x_b1tc.shape)}")
        b, _, t, c = (int(x_b1tc.shape[i]) for i in range(4))
        if c != self.in_channels:
            raise ValueError(f"Expected C_in={self.in_channels}, got {c}")

        x_rm = ttnn.to_layout(x_b1tc, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, (b, t, c))

        pl, pr, padded_len = self.padded_sequence_length(t)
        if self.pad_mode == "replicate":
            x_rm = _replicate_pad_time_rm(x_rm, pad_left=pl, pad_right=pr)
        elif self.pad_mode == "reflect":
            x_rm = _reflect_pad_time_rm(
                x_rm, pad_left=pl, pad_right=pr, dtype=self.activations_dtype, device=self.device
            )
        elif self.pad_mode == "constant":
            if pr != 0:
                raise NotImplementedError("constant pad with right pad is not implemented")
            if pl:
                z = ttnn.zeros(
                    (b, pl, c),
                    dtype=self.activations_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                x_rm = ttnn.concat([z, x_rm], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            raise ValueError(f"Unsupported pad_mode={self.pad_mode!r}")

        t_out = self.expected_output_length(t)

        if _conv_debug_enabled():
            logger.info(
                "[voxtraltts][conv1d] {} prepared: B={} T={} C_in={} C_out={} kernel={} stride={} dilation={} "
                "pad_mode={} padded_len={} expected_t_out={} output_splits={} input_rm={}",
                self.debug_name,
                b,
                t,
                c,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.dilation,
                self.pad_mode,
                padded_len,
                t_out,
                self.output_channel_splits,
                _tensor_debug(x_rm),
            )
        out_splits = []
        effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        use_chunking = t_out > self.max_conv_output_chunk
        for split_idx, weight_group in enumerate(self.weight_tensors):
            if not use_chunking:
                out_splits.append(self._conv_summed_over_inputs(x_rm, padded_len, b, weight_group))
            else:
                split_chunks = []
                for out_start in range(0, t_out, self.max_conv_output_chunk):
                    out_end = min(out_start + self.max_conv_output_chunk, t_out)
                    in_start = out_start * self.stride
                    in_end = (out_end - 1) * self.stride + effective_kernel
                    chunk_slice_mem = None if self.keep_sharded_splits else ttnn.DRAM_MEMORY_CONFIG
                    conv_chunk = ttnn.slice(
                        x_rm,
                        [0, in_start, 0],
                        [b, in_end, c],
                        memory_config=chunk_slice_mem,
                    )
                    if _conv_debug_enabled():
                        logger.info(
                            "[voxtraltts][conv1d] {} split {}/{} chunk out=[{}:{}) in=[{}:{}) input_length={} "
                            "in_channels={} in_channel_splits={} out_channels_per_split={} conv_chunk={}",
                            self.debug_name,
                            split_idx + 1,
                            self.output_channel_splits,
                            out_start,
                            out_end,
                            in_start,
                            in_end,
                            in_end - in_start,
                            self.in_channels,
                            self.input_channel_splits,
                            self.out_channels_per_split,
                            _tensor_debug(conv_chunk),
                        )
                    out_chunk = self._conv_summed_over_inputs(conv_chunk, in_end - in_start, b, weight_group)
                    ttnn.deallocate(conv_chunk)
                    split_chunks.append(out_chunk)

                out_splits.append(self._concat_time_chunks(split_chunks))
                for split_chunk in split_chunks:
                    if split_chunk.is_allocated():
                        ttnn.deallocate(split_chunk)

        conv_out = self._concat_output_channel_splits(out_splits)
        if not self.keep_sharded_splits:
            for out_split in out_splits:
                ttnn.deallocate(out_split)
        if self.keep_sharded_splits and conv_out.memory_config().is_sharded():
            conv_out = _sharded_to_interleaved_dram(conv_out)
        trim_mem = None if self.keep_sharded_splits else ttnn.DRAM_MEMORY_CONFIG
        if len(conv_out.shape) == 4:
            out_len = int(conv_out.shape[2])
            if out_len > t_out:
                conv_out = ttnn.slice(conv_out, [0, 0, 0, 0], [b, 1, t_out, self.out_channels], memory_config=trim_mem)
            elif out_len < t_out:
                raise RuntimeError(f"conv1d length {out_len} < expected {t_out}")
            out4 = conv_out
        else:
            out_len = int(conv_out.shape[1])
            if out_len > t_out:
                conv_out = ttnn.slice(conv_out, [0, 0, 0], [b, t_out, self.out_channels], memory_config=trim_mem)
            elif out_len < t_out:
                raise RuntimeError(f"conv1d length {out_len} < expected {t_out}")
            out4 = ttnn.reshape(conv_out, (b, 1, t_out, self.out_channels))
        return ttnn.to_layout(out4, ttnn.TILE_LAYOUT)


class VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d:
    """``CausalConvTranspose1d`` via zero-insert + ``ttnn.conv1d`` + trim. Uses 16 output-channel splits to stay within L1."""

    def __init__(
        self,
        device,
        *,
        state_dict: dict,
        block_index: int,
        kernel_size: int,
        stride: int,
        in_channels: int | None = None,
        out_channels: int | None = None,
        trim_ratio: float = 1.0,
        output_channel_splits: int = 16,
        weight_dtype=ttnn.bfloat16,
        activations_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
    ) -> None:
        self.device = device
        self.block_index = block_index
        self.kernel_size = kernel_size
        self.stride = stride
        self.trim_ratio = trim_ratio
        self.output_channel_splits = output_channel_splits
        self.activations_dtype = activations_dtype
        self.output_dtype = output_dtype
        self.max_conv_output_chunk = MAX_CONV_TRANSPOSE_OUTPUT_CHUNK
        self.debug_name = f"decoder_blocks.{self.block_index}.conv_transpose"

        w_t = resolve_decoder_block_conv_transpose_fused_weight(state_dict, block_index).contiguous()
        ic, oc, k = (int(w_t.shape[0]), int(w_t.shape[1]), int(w_t.shape[2]))
        if in_channels is not None and ic != in_channels:
            raise ValueError(f"Checkpoint expects in_channels={ic}, got {in_channels}")
        if out_channels is not None and oc != out_channels:
            raise ValueError(f"Checkpoint expects out_channels={oc}, got {out_channels}")
        if kernel_size != k:
            raise ValueError(f"Checkpoint kernel k={k}, expected kernel_size={kernel_size}")
        self.in_channels = ic
        self.out_channels = oc
        self.kernel_size = k
        if self.out_channels % self.output_channel_splits != 0:
            raise ValueError(
                f"out_channels={self.out_channels} must divide evenly by output_channel_splits={self.output_channel_splits}"
            )
        self.out_channels_per_split = self.out_channels // self.output_channel_splits

        w_conv = w_t.permute(1, 0, 2).flip(-1).contiguous()
        w_dtype = ttnn.float32 if weight_dtype == ttnn.bfloat8_b else weight_dtype
        self.weight_tensors = [
            ttnn.from_torch(
                w_conv[i * self.out_channels_per_split : (i + 1) * self.out_channels_per_split],
                dtype=w_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
            )
            for i in range(self.output_channel_splits)
        ]
        self.conv_config, self.compute_config = get_audio_tokenizer_conv_transpose_configs(device)

    def expected_output_length(self, time_len: int) -> int:
        raw = (time_len - 1) * self.stride + self.kernel_size
        total_padding = self.kernel_size - self.stride
        right_padding = math.ceil(total_padding * self.trim_ratio)
        left_padding = total_padding - right_padding
        return raw - left_padding - right_padding

    def __call__(self, x_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, 1, T, C_in]`` tile → ``[B, 1, T_out, C_out]`` tile."""
        if len(x_b1tc.shape) != 4 or x_b1tc.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T, C], got shape {tuple(x_b1tc.shape)}")
        b, _, t, c = (int(x_b1tc.shape[i]) for i in range(4))
        if c != self.in_channels:
            raise ValueError(f"Expected C_in={self.in_channels}, got {c}")

        # Move L1 activations to DRAM before conv compile (avoid CB/L1 clash on long seqs).
        if x_b1tc.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x_dram = ttnn.to_memory_config(x_b1tc, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x_b1tc)
            _free_x_dram = True
        else:
            x_dram = x_b1tc
            _free_x_dram = False

        x_rm = ttnn.to_layout(x_dram, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if _free_x_dram:
            ttnn.deallocate(x_dram)
        x_rm = ttnn.reshape(x_rm, (b, t, c))
        z_rm = _zero_insert_time_rm(x_rm, stride=self.stride, dtype=self.activations_dtype, device=self.device)
        ttnn.deallocate(x_rm)
        z_len = int(z_rm.shape[1])

        pad = self.kernel_size - 1
        zero_pad = ttnn.zeros(
            (b, pad, c),
            dtype=self.activations_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        conv_in = ttnn.concat([zero_pad, z_rm, zero_pad], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(z_rm)
        ttnn.deallocate(zero_pad)

        padded_len = z_len + 2 * pad
        raw_len = (t - 1) * self.stride + self.kernel_size
        total_padding = self.kernel_size - self.stride
        right_padding = math.ceil(total_padding * self.trim_ratio)
        left_padding = total_padding - right_padding
        t_out = self.expected_output_length(t)

        if _conv_debug_enabled():
            logger.info(
                "[voxtraltts][conv_transpose1d] {} prepared: B={} T={} C_in={} C_out={} kernel={} stride={} "
                "padded_len={} raw_len={} trimmed_t_out={} left_pad={} right_pad={} output_splits={} max_chunk={} conv_in={}",
                self.debug_name,
                b,
                t,
                c,
                self.out_channels,
                self.kernel_size,
                self.stride,
                padded_len,
                raw_len,
                t_out,
                left_padding,
                right_padding,
                self.output_channel_splits,
                self.max_conv_output_chunk,
                _tensor_debug(conv_in),
            )
        out_splits = []
        for split_idx, weight_tensor in enumerate(self.weight_tensors):
            if raw_len <= self.max_conv_output_chunk:
                out_chunk = ttnn.conv1d(
                    input_tensor=conv_in,
                    weight_tensor=weight_tensor,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels_per_split,
                    device=self.device,
                    bias_tensor=None,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                    batch_size=b,
                    input_length=padded_len,
                    conv_config=self.conv_config,
                    compute_config=self.compute_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    groups=1,
                    dtype=self.output_dtype,
                    return_weights_and_bias=False,
                )
                if out_chunk.memory_config().is_sharded():
                    out_splits.append(ttnn.sharded_to_interleaved(out_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                    ttnn.deallocate(out_chunk)
                else:
                    out_splits.append(out_chunk)
            else:
                split_chunks = []
                for out_start in range(0, raw_len, self.max_conv_output_chunk):
                    out_end = min(out_start + self.max_conv_output_chunk, raw_len)
                    in_start = out_start
                    in_end = out_end + self.kernel_size - 1
                    conv_chunk = ttnn.slice(
                        conv_in,
                        [0, in_start, 0],
                        [b, in_end, c],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    if _conv_debug_enabled():
                        logger.info(
                            "[voxtraltts][conv_transpose1d] {} split {}/{} chunk out=[{}:{}) in=[{}:{}) "
                            "input_length={} in_channels={} out_channels_per_split={} conv_chunk={} weight_shape={}",
                            self.debug_name,
                            split_idx + 1,
                            self.output_channel_splits,
                            out_start,
                            out_end,
                            in_start,
                            in_end,
                            in_end - in_start,
                            self.in_channels,
                            self.out_channels_per_split,
                            _tensor_debug(conv_chunk),
                            tuple(weight_tensor.shape),
                        )
                    out_chunk = ttnn.conv1d(
                        input_tensor=conv_chunk,
                        weight_tensor=weight_tensor,
                        in_channels=self.in_channels,
                        out_channels=self.out_channels_per_split,
                        device=self.device,
                        bias_tensor=None,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=0,
                        batch_size=b,
                        input_length=in_end - in_start,
                        conv_config=self.conv_config,
                        compute_config=self.compute_config,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        groups=1,
                        dtype=self.output_dtype,
                        return_weights_and_bias=False,
                    )
                    ttnn.deallocate(conv_chunk)
                    if out_chunk.memory_config().is_sharded():
                        split_chunks.append(
                            ttnn.sharded_to_interleaved(out_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        )
                        ttnn.deallocate(out_chunk)
                    else:
                        split_chunks.append(out_chunk)

                if len(split_chunks) == 1:
                    out_splits.append(split_chunks[0])
                else:
                    time_dim = 2 if len(split_chunks[0].shape) == 4 else 1
                    out_splits.append(ttnn.concat(split_chunks, dim=time_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                    for split_chunk in split_chunks:
                        ttnn.deallocate(split_chunk)
        ttnn.deallocate(conv_in)

        conv_out = ttnn.concat(out_splits, dim=len(out_splits[0].shape) - 1)
        for out_split in out_splits:
            ttnn.deallocate(out_split)
        if len(conv_out.shape) == 4:
            out_len = int(conv_out.shape[2])
            if out_len != raw_len:
                raise RuntimeError(f"raw conv transpose length {out_len} != expected {raw_len}")
            begin = [0, 0, left_padding, 0]
            end = [b, 1, left_padding + t_out, self.out_channels]
            out4 = ttnn.slice(conv_out, begin, end)
            ttnn.deallocate(conv_out)
        else:
            out_len = int(conv_out.shape[1])
            if out_len != raw_len:
                raise RuntimeError(f"raw conv transpose length {out_len} != expected {raw_len}")
            out3 = ttnn.slice(conv_out, [0, left_padding, 0], [b, left_padding + t_out, self.out_channels])
            ttnn.deallocate(conv_out)
            out4 = ttnn.reshape(out3, (b, 1, t_out, self.out_channels))
            ttnn.deallocate(out3)
        return ttnn.to_layout(out4, ttnn.TILE_LAYOUT)
