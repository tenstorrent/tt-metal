# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audio tokenizer convs via ``ttnn.conv1d``. Forward takes ``[B, 1, T, C_in]`` tile."""

from __future__ import annotations

import math

import ttnn


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
    """Same pattern as Whisper ``get_conv_configs`` (``ttnn_optimized_functional_whisper``)."""
    conv1d_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        config_tensors_in_dram=True,
    )
    conv1d_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
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
        self.activations_dtype = activations_dtype
        self.output_dtype = output_dtype

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
            pad_shape = (b, self.left_pad, c)
            pad = ttnn.zeros(
                pad_shape,
                dtype=self.activations_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x_rm = ttnn.concat([pad, x_rm], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        padded_len = t + self.left_pad

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


def _zero_insert_time_rm(x_btc: ttnn.Tensor, *, stride: int, dtype, device) -> ttnn.Tensor:
    """Insert ``stride - 1`` all-zero frames after each input frame on ``[B,T,C]``."""
    if stride == 1:
        return x_btc
    b, t, c = (int(x_btc.shape[i]) for i in range(3))
    zero = ttnn.zeros(
        (b, 1, c),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    parts: list[ttnn.Tensor] = []
    for idx in range(t):
        frame = ttnn.slice(x_btc, [0, idx, 0], [b, idx + 1, c])
        parts.append(frame)
        if idx != t - 1:
            for _ in range(stride - 1):
                parts.append(zero)
    out = ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(zero)
    for p in parts:
        if p is not zero:
            ttnn.deallocate(p)
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
        self.activations_dtype = activations_dtype
        self.output_dtype = output_dtype

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

        w_dtype = ttnn.float32 if weight_dtype == ttnn.bfloat8_b else weight_dtype
        self.weight_tensors = [
            ttnn.from_torch(
                w_host[i * self.out_channels_per_split : (i + 1) * self.out_channels_per_split],
                dtype=w_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
            )
            for i in range(self.output_channel_splits)
        ]
        self.conv_config, self.compute_config = get_audio_tokenizer_conv_configs(device)

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

        out_splits = []
        for i, weight_tensor in enumerate(self.weight_tensors):
            out_split, [weight_device, _] = ttnn.conv1d(
                input_tensor=x_rm,
                weight_tensor=weight_tensor,
                in_channels=self.in_channels,
                out_channels=self.out_channels_per_split,
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
            self.weight_tensors[i] = weight_device
            out_splits.append(ttnn.sharded_to_interleaved(out_split))

        conv_out = ttnn.concat(out_splits, dim=len(out_splits[0].shape) - 1)
        if len(conv_out.shape) == 4:
            out_len = int(conv_out.shape[2])
            if out_len > t_out:
                conv_out = ttnn.slice(conv_out, [0, 0, 0, 0], [b, 1, t_out, self.out_channels])
            elif out_len < t_out:
                raise RuntimeError(f"conv1d length {out_len} < expected {t_out}")
            out4 = conv_out
        else:
            out_len = int(conv_out.shape[1])
            if out_len > t_out:
                conv_out = ttnn.slice(conv_out, [0, 0, 0], [b, t_out, self.out_channels])
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

        # ConvTranspose1d weight is [in, out, k]. Equivalent conv1d uses [out, in, flipped_k].
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
        self.conv_config, self.compute_config = get_audio_tokenizer_conv_configs(device)

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

        x_rm = ttnn.to_layout(x_b1tc, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, (b, t, c))
        z_rm = _zero_insert_time_rm(x_rm, stride=self.stride, dtype=self.activations_dtype, device=self.device)
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

        out_splits = []
        for i, weight_tensor in enumerate(self.weight_tensors):
            out_split, [weight_device, _] = ttnn.conv1d(
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
                groups=1,
                dtype=self.output_dtype,
                return_weights_and_bias=True,
            )
            self.weight_tensors[i] = weight_device
            out_splits.append(ttnn.sharded_to_interleaved(out_split))
        ttnn.deallocate(conv_in)

        conv_out = ttnn.concat(out_splits, dim=len(out_splits[0].shape) - 1)
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
        _ = right_padding
        return ttnn.to_layout(out4, ttnn.TILE_LAYOUT)
