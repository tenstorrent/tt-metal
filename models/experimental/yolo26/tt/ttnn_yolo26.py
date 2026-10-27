# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN YOLO26 Object Detection Model.

Implements YOLO26 architecture optimized for Tenstorrent hardware using TTNN.

Architecture:
- Backbone: CSP-style with C2f blocks
- Neck: PAN (Path Aggregation Network) for multi-scale feature fusion
- Head: End-to-end detection head (NMS-free)

Key optimizations:
- BatchNorm folded into Conv weights
- SiLU activation (native TTNN support)
- Optimal sharding strategies per layer
- Tensor deallocation for memory efficiency
"""

import torch
import ttnn
from typing import Tuple, List, Optional, Dict
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    HeightShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    WidthShardedStrategyConfiguration,
)


# Attention matmuls at full precision (N <= 400 tokens, so the cost is small)
ATTN_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# Softmax over bf16 scores: fp32 dest accumulation only slows its exp (PCC unchanged, ~10 us per PSA block faster)
SOFTMAX_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

# uint8 input pixels carry RGB + 5 zero channels: a 16-byte row-major stick is the minimum L1-aligned size, so the
# image can be sharded by rows and viewed as one stick per pixel without any data movement (see TtYOLO26.preprocess)
UINT8_INPUT_CHANNELS = 8

# Largest batch that runs the whole network in L1. Larger batches run the 640..40 resolution stages in groups of this
# many images and the 20x20 stages in groups of up to LOW_RES_BATCH (see TtYOLO26._forward_grouped).
MAX_L1_BATCH = 4
# Largest batch of the 20x20 stages that fits L1 (at 16, SPPF's first conv no longer fits next to its input)
LOW_RES_BATCH = 8


def uint8_input_from_torch(images: torch.Tensor) -> torch.Tensor:
    """uint8 images [B, H, W, 3] -> host layout of the uint8 model input, [B, 1, H, W * UINT8_INPUT_CHANNELS]."""
    b, h, w, c = images.shape
    images = torch.nn.functional.pad(images.to(torch.uint8), (0, UINT8_INPUT_CHANNELS - c))
    return images.reshape(b, 1, h, w * UINT8_INPUT_CHANNELS)


# Channel multiplier relative to the yolo26n layout used throughout this file.
# n and s share depth (every C3k2 has n=1) and differ only in width (0.25 vs 0.5).
YOLO26_WIDTH_MULTIPLIER = {"yolo26n": 1, "yolo26s": 2}


def get_width_multiplier(variant: str) -> int:
    if variant not in YOLO26_WIDTH_MULTIPLIER:
        raise ValueError(f"Unsupported YOLO26 variant '{variant}', supported: {list(YOLO26_WIDTH_MULTIPLIER)}")
    return YOLO26_WIDTH_MULTIPLIER[variant]


def pad_to_tile(channels: int) -> int:
    """Channel count rounded up to a tile width, so channel concat/split never cuts through a tile."""
    return -(-channels // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def scatter_segments(t: torch.Tensor, segments: List[Tuple[int, int]], dim: int) -> torch.Tensor:
    """Place consecutive chunks of t along dim into zero-padded segments, segments = [(real, padded), ...]."""
    parts, start = [], 0
    for real, padded in segments:
        part = t.narrow(dim, start, real)
        start += real
        if padded > real:
            shape = list(part.shape)
            shape[dim] = padded - real
            part = torch.cat([part, part.new_zeros(shape)], dim)
        parts.append(part)
    assert start == t.shape[dim], f"segments cover {start} of {t.shape[dim]} channels"
    return torch.cat(parts, dim)


def slice_out_channels(weight: ttnn.Tensor, bias: ttnn.Tensor, start: int, end: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Output channels [start, end) of a host conv weight/bias pair (the conv that computes one chunk of another)."""
    w, b = ttnn.to_torch(weight), ttnn.to_torch(bias)
    return ttnn.from_torch(w[start:end].contiguous()), ttnn.from_torch(b[..., start:end].contiguous())


def _with_memory_config_of(x: ttnn.Tensor, ref: ttnn.Tensor) -> ttnn.Tensor:
    return x if x.memory_config() == ref.memory_config() else ttnn.to_memory_config(x, ref.memory_config())


def _height_sharded_like(ref: ttnn.Tensor, width: int) -> ttnn.MemoryConfig:
    """ref's height sharding (grid, rows per core) for a tensor with `width` channels."""
    spec = ref.memory_config().shard_spec
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(spec.grid, [spec.shape[0], width], spec.orientation),
    )


def _in_dram(x: ttnn.Tensor) -> bool:
    return x.memory_config().buffer_type == ttnn.BufferType.DRAM


def _is_height_sharded(x: ttnn.Tensor) -> bool:
    return x.is_sharded() and x.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _as_flat_tile(x: ttnn.Tensor) -> ttnn.Tensor:
    """x as a flattened [1, 1, NHW, C] tile tensor (block inputs may arrive e.g. as row-major NHWC)."""
    flat = [1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]]
    if x.layout == ttnn.TILE_LAYOUT and list(x.shape) == flat:
        return x
    if x.is_sharded():
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, flat)
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)


def _free(*tensors: ttnn.Tensor):
    """Release block-internal tensors as soon as they are consumed (keeps L1 free for larger batches)."""
    for t in tensors:
        ttnn.deallocate(t)


def add(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    """Elementwise add of two flattened [1, 1, NHW, C] tile tensors, staying in L1 (b follows a's memory config).

    b may also be a block input in another form (e.g. row-major NHWC when a block is driven on its own).
    """
    if not a.is_sharded() and a.memory_config().buffer_type != ttnn.BufferType.L1:
        a = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG)
    b_in = b
    b = _with_memory_config_of(_as_flat_tile(b), a)
    out = ttnn.add(a, b, memory_config=a.memory_config())
    if b is not b_in:
        _free(b)
    return out


def concat(tensors: List[ttnn.Tensor]) -> ttnn.Tensor:
    """Channel concat of flattened [1, 1, NHW, C] tile tensors with tile-aligned C, staying in L1.

    Height-sharded inputs are aligned to the first tensor's sharding and concatenated shard by shard; anything
    else is concatenated L1 interleaved.
    """
    inputs = tensors
    tensors = [_as_flat_tile(t) for t in tensors]
    ref = tensors[0]
    if _is_height_sharded(ref):
        aligned = []
        for t in tensors:
            mc = _height_sharded_like(ref, t.padded_shape[-1])
            aligned.append(t if t.memory_config() == mc else ttnn.to_memory_config(t, mc))
        out = ttnn.concat(
            aligned, dim=3, memory_config=_height_sharded_like(ref, sum(t.padded_shape[-1] for t in aligned))
        )
    else:
        aligned = [
            t
            if (not t.is_sharded() and t.memory_config().buffer_type == ttnn.BufferType.L1)
            else ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
            for t in tensors
        ]
        out = ttnn.concat(aligned, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    # release the converted copies (the inputs themselves belong to the caller)
    temps = {id(t): t for t in tensors + aligned if not any(t is i for i in inputs)}
    _free(*temps.values())
    return out


class TtConvBNSiLU:
    """
    YOLO26 Conv + BatchNorm + SiLU block for TTNN.

    BatchNorm is folded into Conv weights at initialization.
    SiLU activation is applied after convolution.
    """

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        name: str = "",
        activation: bool = True,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.name = name
        self.activation = activation
        self.weight = None
        self.bias = None
        self._conv_cache = {}

    def load_weights(
        self,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        in_segments: Optional[List[Tuple[int, int]]] = None,
        out_segments: Optional[List[Tuple[int, int]]] = None,
    ):
        """Load pre-folded Conv+BN weights.

        in_segments / out_segments ([(real, padded), ...]) zero-pad input / output channel groups, for convs whose
        input is a concat of tile-padded tensors or whose output must be tile-padded. Padded outputs are exactly 0
        (zero weights and bias, SiLU(0) = 0), so they add and concat like real channels.
        """
        needs_padding = any(r != p for r, p in (in_segments or []) + (out_segments or []))
        if needs_padding:
            w, b = ttnn.to_torch(weight).float(), ttnn.to_torch(bias).float().reshape(1, 1, 1, -1)
            if out_segments:
                w, b = scatter_segments(w, out_segments, 0), scatter_segments(b, out_segments, 3)
            if in_segments:
                w = scatter_segments(w, in_segments, 1)
            weight, bias = ttnn.from_torch(w), ttnn.from_torch(b)
        self.weight = weight
        self.bias = bias

    def load_weights_no_bn(self, weight_loader, prefix: str):
        """Load plain Conv2d weights (no BatchNorm)."""
        import torch

        state_dict = weight_loader.state_dict

        # Get raw conv weight and bias
        conv_weight = state_dict[f"{prefix}.weight"]
        conv_bias = state_dict.get(f"{prefix}.bias", None)

        # TTNN Conv2d expects OIHW format (same as PyTorch)
        # Keep original: [out_ch, in_ch, kH, kW]
        self.weight = ttnn.from_torch(conv_weight.to(torch.bfloat16), dtype=ttnn.bfloat16)

        # Bias needs shape (1, 1, 1, out_channels) for TTNN conv
        if conv_bias is not None:
            bias_reshaped = conv_bias.reshape(1, 1, 1, -1).to(torch.bfloat16)
            self.bias = ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16)
        else:
            self.bias = ttnn.from_torch(
                torch.zeros(1, 1, 1, self.out_channels, dtype=torch.bfloat16), dtype=ttnn.bfloat16
            )

    def _large_batch_settings(self, batch_size: int) -> Tuple[int, bool]:
        """(height-sharded act_block_h, weights double buffer) for this batch size.

        Up to batch 2 the conv defaults are fastest. From batch 3 on they no longer fit L1 (circular buffers clash
        with the activations): batch 3 needs single-buffered weights (e.g. 3x3x256x256 on every core of model.5)
        and smaller stem blocks, batch 4 also smaller blocks on every height-sharded conv. Measured on yolo26s;
        these settings would cost 10-15 % at batch 1-2. Batch 5 still fails (model.2.cv2, 1x1 conv at 160x160).
        """
        if batch_size <= 2:
            return 0, True
        if self.name == "model.0":
            return (256 if batch_size == 3 else 128), False
        return (0 if batch_size == 3 else 64), False

    def _get_sharding_strategy(self, batch_size: int, height: int, width: int):
        """Determine optimal sharding strategy based on tensor dimensions."""
        # Output rows: a stride-2 conv's work is a quarter of its input rows. Above MAX_L1_BATCH the layout is chosen
        # as at MAX_L1_BATCH: more rows would flip e.g. model.9.cv2 to height sharding, which puts all its weights
        # (1x1, 1024 -> 512) on every core
        nhw = min(batch_size, MAX_L1_BATCH) * height * width // (self.stride * self.stride)
        ratio = nhw / self.out_channels if self.out_channels > 0 else float("inf")

        # Height sharding for tall outputs, block sharding otherwise (output-NHW threshold swept over 2..16 on
        # yolo26n/s, 4 is best; input-NHW thresholds 4..64 were all slower)
        if ratio > 4:
            return HeightShardedStrategyConfiguration(act_block_h_override=self._large_batch_settings(batch_size)[0])
        elif ratio < 0.25:
            return WidthShardedStrategyConfiguration()
        else:
            return BlockShardedStrategyConfiguration(reshard_if_not_optimal=True)

    def _get_conv(self, batch_size: int, input_height: int, input_width: int):
        """Get or create cached conv layer for given input dimensions."""
        key = (batch_size, input_height, input_width)
        if key not in self._conv_cache:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=(self.stride, self.stride),
                padding=(self.padding, self.padding),
                groups=self.groups,
                dilation=(1, 1),
                weight=self.weight,
                bias=self.bias,
                sharding_strategy=self._get_sharding_strategy(batch_size, input_height, input_width),
                # bfp8 weights drop N5 cls PCC to ~0.8; bf16 + HiFi4 + fp32 accumulation gives cls ~0.999
                weights_dtype=ttnn.bfloat16,
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=self._large_batch_settings(batch_size)[1],
                # SiLU runs in the conv's packer instead of as a separate op on the conv output
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if self.activation else None,
            )
            self._conv_cache[key] = TtConv2d(config, self.device)
        return self._conv_cache[key]

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """
        Forward pass.

        Args:
            x: Input tensor
            batch_size: Batch size
            height: Input height
            width: Input width

        Returns:
            Tuple of (output_tensor, output_height, output_width)
        """
        conv = self._get_conv(batch_size, height, width)
        x, h_w = conv(x, return_output_dim=True)
        return x, h_w[0], h_w[1]


class TtBottleneck:
    """
    YOLO26 Bottleneck block.

    YOLO26 uses: Conv3x3 -> Conv3x3 pattern (not 1x1->3x3 like ResNet)
    Structure: Conv3x3 (reduce) -> Conv3x3 (expand) -> Add (if shortcut)

    padded_channels: input/output carry this many channels (real ones first, the rest zero), so the block can sit
    between tile-padded tensors; the hidden tensor is internal and stays unpadded.
    """

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        name: str = "",
        padded_channels: Optional[int] = None,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut = shortcut and in_channels == out_channels
        self.hidden_channels = int(out_channels * expansion)
        self.padded_in = padded_channels or in_channels
        self.padded_out = padded_channels or out_channels

        # cv1: 3x3 conv (in_ch -> hidden_ch)
        self.conv1 = TtConvBNSiLU(
            device, self.padded_in, self.hidden_channels, kernel_size=3, stride=1, padding=1, name=f"{name}.cv1"
        )
        # cv2: 3x3 conv (hidden_ch -> out_ch)
        self.conv2 = TtConvBNSiLU(
            device,
            self.hidden_channels,
            self.padded_out,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            name=f"{name}.cv2",
        )

    def load_weights(self, cv1_weight, cv1_bias, cv2_weight, cv2_bias):
        """Load weights for both conv layers."""
        self.conv1.load_weights(cv1_weight, cv1_bias, in_segments=[(self.in_channels, self.padded_in)])
        self.conv2.load_weights(cv2_weight, cv2_bias, out_segments=[(self.out_channels, self.padded_out)])

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward pass with optional residual connection."""
        hidden, h, w = self.conv1(x, batch_size, height, width)
        out, h, w = self.conv2(hidden, batch_size, h, w)
        _free(hidden)
        if self.shortcut:
            residual = add(out, x)
            _free(out)
            out = residual
        return out, h, w


class TtC2f:
    """
    YOLO26 C2f (CSP Bottleneck with 2 convolutions) module.

    This is the core building block of YOLO26/v8/v11 backbone.
    Structure:
    - cv1: 1x1 conv (in → 2*hidden_channels)
    - Split into x1, x2 (each hidden_channels)
    - n bottleneck blocks processing x2
    - Concat x1 + x2 + bottleneck outputs
    - cv2: 1x1 conv to fuse features

    On device the split is not materialized: cv1 produces [x1 | x2] for the concat and cv1_b (the second half of
    cv1's weights) produces x2 for the bottlenecks. Each hidden_channels group is tile-padded, so the concat is
    tile-aligned and stays sharded in L1.

    Note: hidden_channels must be provided explicitly as it varies
    based on width_multiple (0.25 for early stages, 0.5 for later stages in YOLO26n).
    """

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,  # Must be explicitly provided!
        n: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        name: str = "",
    ):
        self.device = device
        self.n = n
        self.hidden_channels = hidden_channels
        self.hidden_padded = pad_to_tile(hidden_channels)
        hp = self.hidden_padded

        # Initial conv: in → [x1 | x2], and x2 alone
        self.cv1 = TtConvBNSiLU(device, in_channels, 2 * hp, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1")
        self.cv1_b = TtConvBNSiLU(device, in_channels, hp, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1_b")

        # Bottleneck blocks - use expansion=0.5 to match PyTorch
        # PyTorch: hidden → hidden/2 → hidden
        self.bottlenecks = []
        for i in range(n):
            self.bottlenecks.append(
                TtBottleneck(
                    device, hidden_channels, hidden_channels, shortcut, groups, 0.5, f"{name}.m.{i}", padded_channels=hp
                )
            )

        # Final fusion conv
        # Input channels = hidden_channels * (2 + n) after concatenation
        self.cv2 = TtConvBNSiLU(
            device, hp * (2 + n), out_channels, kernel_size=1, stride=1, padding=0, name=f"{name}.cv2"
        )

    def load_weights(self, weight_loader, prefix: str):
        """Load weights from weight loader."""
        # YOLO26 naming: model.2.cv1.conv.weight, model.2.cv1.bn.weight
        h, hp = self.hidden_channels, self.hidden_padded
        cv1_w, cv1_b = weight_loader.get_conv_bn(f"{prefix}.cv1")
        self.cv1.load_weights(cv1_w, cv1_b, out_segments=[(h, hp)] * 2)
        self.cv1_b.load_weights(*slice_out_channels(cv1_w, cv1_b, h, 2 * h), out_segments=[(h, hp)])

        for i, bottleneck in enumerate(self.bottlenecks):
            cv1_w, cv1_b = weight_loader.get_conv_bn(f"{prefix}.m.{i}.cv1")
            cv2_w, cv2_b = weight_loader.get_conv_bn(f"{prefix}.m.{i}.cv2")
            bottleneck.load_weights(cv1_w, cv1_b, cv2_w, cv2_b)

        cv2_w, cv2_b = weight_loader.get_conv_bn(f"{prefix}.cv2")
        self.cv2.load_weights(cv2_w, cv2_b, in_segments=[(h, hp)] * (2 + self.n))

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """
        Forward pass.

        Splits input, processes through bottlenecks, concatenates all features.
        """
        x12, h, w = self.cv1(x, batch_size, height, width)
        y, _, _ = self.cv1_b(x, batch_size, height, width)

        features = [x12]
        for i, bottleneck in enumerate(self.bottlenecks):
            y_next, _, _ = bottleneck(y, batch_size, h, w)
            if i == 0:
                _free(y)  # cv1_b's copy of x2 (x12 carries x2 for the concat)
            y = y_next
            features.append(y)

        cat = concat(features)
        _free(*features)
        out, h, w = self.cv2(cat, batch_size, h, w)
        _free(cat)
        return out, h, w


class TtC3k:
    """
    YOLO26 C3k block - used in C3k2 for model.6 and model.8.

    Structure:
    - cv1: in_ch → hidden/2 (1x1 conv)
    - cv2: in_ch → hidden/2 (1x1 conv)
    - cv3: in_ch → in_ch (1x1 conv for output)
    - m: Sequential of 2 bottlenecks (hidden/2 → hidden/2)

    Flow:
    1. x1 = cv1(x)  # Split path 1
    2. x2 = cv2(x)  # Split path 2
    3. y = bottleneck_chain(x1)  # Process x1 through bottlenecks
    4. out = cv3(concat(y, x2))  # Fuse and project

    Input and output are tile-padded to pad_to_tile(in_channels) channels, as are the two inner paths.
    """

    def __init__(self, device, in_channels: int, name: str = ""):
        self.device = device
        self.in_channels = in_channels
        self.padded = pad_to_tile(in_channels)
        self.hidden = in_channels // 2  # 64 → 32
        self.hidden_padded = pad_to_tile(self.hidden)
        cp, hp = self.padded, self.hidden_padded

        # cv1: in → hidden/2
        self.cv1 = TtConvBNSiLU(device, cp, hp, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1")
        # cv2: in → hidden/2
        self.cv2 = TtConvBNSiLU(device, cp, hp, kernel_size=1, stride=1, padding=0, name=f"{name}.cv2")
        # cv3: in → in (after concat of 2 hidden/2 = in channels)
        self.cv3 = TtConvBNSiLU(device, 2 * hp, cp, kernel_size=1, stride=1, padding=0, name=f"{name}.cv3")

        # 2 bottlenecks - note: these use expansion=1.0 (32→32→32)
        self.bottlenecks = [
            TtBottleneck(device, self.hidden, self.hidden, True, 1, 1.0, name=f"{name}.m.{i}", padded_channels=hp)
            for i in range(2)
        ]

    def load_weights(self, weight_loader, prefix: str):
        """Load weights for C3k block."""
        c, cp, h, hp = self.in_channels, self.padded, self.hidden, self.hidden_padded
        w, b = weight_loader.get_conv_bn(f"{prefix}.cv1")
        self.cv1.load_weights(w, b, in_segments=[(c, cp)], out_segments=[(h, hp)])
        w, b = weight_loader.get_conv_bn(f"{prefix}.cv2")
        self.cv2.load_weights(w, b, in_segments=[(c, cp)], out_segments=[(h, hp)])
        w, b = weight_loader.get_conv_bn(f"{prefix}.cv3")
        self.cv3.load_weights(w, b, in_segments=[(h, hp)] * 2, out_segments=[(c, cp)])

        for i, bottleneck in enumerate(self.bottlenecks):
            cv1_w, cv1_b = weight_loader.get_conv_bn(f"{prefix}.m.{i}.cv1")
            cv2_w, cv2_b = weight_loader.get_conv_bn(f"{prefix}.m.{i}.cv2")
            bottleneck.load_weights(cv1_w, cv1_b, cv2_w, cv2_b)

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward pass through C3k."""
        y, h, w = self.cv1(x, batch_size, height, width)
        x2, _, _ = self.cv2(x, batch_size, height, width)
        for bottleneck in self.bottlenecks:
            y_next, _, _ = bottleneck(y, batch_size, h, w)
            _free(y)
            y = y_next
        cat = concat([y, x2])
        _free(y, x2)
        out, h, w = self.cv3(cat, batch_size, h, w)
        _free(cat)
        return out, h, w


class TtC3k2:
    """
    YOLO26 C3k2 block with C3k inner modules - for model.6 and model.8.

    Similar to TtC2f but uses TtC3k instead of TtBottleneck.
    """

    def __init__(self, device, in_channels: int, out_channels: int, hidden_channels: int, n: int = 1, name: str = ""):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.hidden_padded = pad_to_tile(hidden_channels)
        self.n = n
        hp = self.hidden_padded

        # cv1: in → [x1 | x2], cv1_b: in → x2
        self.cv1 = TtConvBNSiLU(device, in_channels, 2 * hp, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1")
        self.cv1_b = TtConvBNSiLU(device, in_channels, hp, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1_b")

        # C3k blocks instead of bottlenecks
        self.c3k_blocks = [TtC3k(device, hidden_channels, name=f"{name}.m.{i}") for i in range(n)]

        # cv2: (n+2)*hidden → out
        self.cv2 = TtConvBNSiLU(
            device, (n + 2) * hp, out_channels, kernel_size=1, stride=1, padding=0, name=f"{name}.cv2"
        )

    def load_weights(self, weight_loader, prefix: str):
        """Load weights."""
        h, hp = self.hidden_channels, self.hidden_padded
        w, b = weight_loader.get_conv_bn(f"{prefix}.cv1")
        self.cv1.load_weights(w, b, out_segments=[(h, hp)] * 2)
        self.cv1_b.load_weights(*slice_out_channels(w, b, h, 2 * h), out_segments=[(h, hp)])

        for i, c3k in enumerate(self.c3k_blocks):
            c3k.load_weights(weight_loader, f"{prefix}.m.{i}")

        w, b = weight_loader.get_conv_bn(f"{prefix}.cv2")
        self.cv2.load_weights(w, b, in_segments=[(h, hp)] * (2 + self.n))

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward pass."""
        x12, h, w = self.cv1(x, batch_size, height, width)
        y, _, _ = self.cv1_b(x, batch_size, height, width)

        features = [x12]
        for i, c3k in enumerate(self.c3k_blocks):
            y_next, _, _ = c3k(y, batch_size, h, w)
            if i == 0:
                _free(y)  # cv1_b's copy of x2 (x12 carries x2 for the concat)
            y = y_next
            features.append(y)

        cat = concat(features)
        _free(*features)
        out, h, w = self.cv2(cat, batch_size, h, w)
        _free(cat)
        return out, h, w


class TtAttentionBlock:
    """
    YOLO26 Attention block used in C2f+Attn modules.

    Structure (from weights):
    - qkv: Conv 1x1 (in_ch → 2*in_ch) for query, key, value
    - pe: Depthwise Conv 3x3 for positional encoding
    - proj: Conv 1x1 (in_ch → in_ch) output projection
    - ffn: Feedforward network (in_ch → 2*in_ch → in_ch)
    """

    def __init__(self, device, channels: int, name: str = ""):
        self.device = device
        self.channels = channels

        # QKV projection: in_ch → 2*in_ch (for k, v; q comes from identity)
        self.qkv = TtConvBNSiLU(
            device,
            channels,
            channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=False,
            name=f"{name}.attn.qkv",
        )

        # Positional encoding: depthwise conv
        self.pe = TtConvBNSiLU(
            device,
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            activation=False,
            name=f"{name}.attn.pe",
        )

        # Output projection
        self.proj = TtConvBNSiLU(
            device, channels, channels, kernel_size=1, stride=1, padding=0, activation=False, name=f"{name}.attn.proj"
        )

        # FFN layers
        self.ffn_0 = TtConvBNSiLU(
            device, channels, channels * 2, kernel_size=1, stride=1, padding=0, activation=True, name=f"{name}.ffn.0"
        )
        self.ffn_1 = TtConvBNSiLU(
            device, channels * 2, channels, kernel_size=1, stride=1, padding=0, activation=False, name=f"{name}.ffn.1"
        )

    def load_weights(self, weight_loader, prefix: str):
        """Load attention block weights."""
        qkv_w, qkv_b = weight_loader.get_conv_bn(f"{prefix}.attn.qkv")
        self.qkv.load_weights(qkv_w, qkv_b)

        pe_w, pe_b = weight_loader.get_conv_bn(f"{prefix}.attn.pe")
        self.pe.load_weights(pe_w, pe_b)

        proj_w, proj_b = weight_loader.get_conv_bn(f"{prefix}.attn.proj")
        self.proj.load_weights(proj_w, proj_b)

        ffn0_w, ffn0_b = weight_loader.get_conv_bn(f"{prefix}.ffn.0")
        self.ffn_0.load_weights(ffn0_w, ffn0_b)

        ffn1_w, ffn1_b = weight_loader.get_conv_bn(f"{prefix}.ffn.1")
        self.ffn_1.load_weights(ffn1_w, ffn1_b)

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """
        Forward pass with attention.

        Simplified attention: q=x, kv from qkv projection
        """
        # Ensure proper layout
        if x.memory_config().is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Identity for residual
        identity = x

        # QKV projection
        qkv_out, h, w = self.qkv(x, batch_size, height, width)

        # Split qkv into k and v (q is identity x)
        if qkv_out.memory_config().is_sharded():
            qkv_out = ttnn.sharded_to_interleaved(qkv_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        qkv_out = ttnn.reshape(qkv_out, [batch_size, h, w, self.channels * 2])
        qkv_out = ttnn.to_layout(qkv_out, ttnn.ROW_MAJOR_LAYOUT)

        k = ttnn.slice(qkv_out, (0, 0, 0, 0), (batch_size, h, w, self.channels))
        v = ttnn.slice(qkv_out, (0, 0, 0, self.channels), (batch_size, h, w, self.channels * 2))

        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)

        # Positional encoding on value
        v_pe, _, _ = self.pe(v, batch_size, h, w)
        if v_pe.memory_config().is_sharded():
            v_pe = ttnn.sharded_to_interleaved(v_pe, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.add(v, v_pe)

        # Simplified attention: use v directly (actual attention would do softmax(q@k.T)@v)
        # For inference with pre-trained weights, this approximation often works
        attn_out = v

        # Output projection
        attn_out, h, w = self.proj(attn_out, batch_size, h, w)

        # Residual connection
        if attn_out.memory_config().is_sharded():
            attn_out = ttnn.sharded_to_interleaved(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        if identity.memory_config().is_sharded():
            identity = ttnn.sharded_to_interleaved(identity, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.add(attn_out, identity)

        # FFN with residual
        identity = x
        x, h, w = self.ffn_0(x, batch_size, h, w)
        x, h, w = self.ffn_1(x, batch_size, h, w)

        if x.memory_config().is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.add(x, identity)

        return x, h, w


class TtC2fAttn:
    """
    YOLO26 C2f with Attention module.

    Used in model.10 and model.22 of YOLO26 neck.
    Similar to C2f but bottleneck is replaced with attention block.
    """

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,  # Must be explicitly provided!
        n: int = 1,
        name: str = "",
    ):
        self.device = device
        self.n = n

        # Initial conv to split
        self.cv1 = TtConvBNSiLU(
            device, in_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1"
        )

        # Attention blocks instead of bottlenecks
        self.attn_blocks = []
        for i in range(n):
            self.attn_blocks.append(TtAttentionBlock(device, hidden_channels, name=f"{name}.m.{i}"))

        # Final fusion conv
        self.cv2 = TtConvBNSiLU(
            device, hidden_channels * (2 + n), out_channels, kernel_size=1, stride=1, padding=0, name=f"{name}.cv2"
        )

        self.hidden_channels = hidden_channels

    def load_weights(self, weight_loader, prefix: str):
        """Load weights from weight loader."""
        cv1_w, cv1_b = weight_loader.get_conv_bn(f"{prefix}.cv1")
        self.cv1.load_weights(cv1_w, cv1_b)

        for i, attn in enumerate(self.attn_blocks):
            attn.load_weights(weight_loader, f"{prefix}.m.{i}")

        cv2_w, cv2_b = weight_loader.get_conv_bn(f"{prefix}.cv2")
        self.cv2.load_weights(cv2_w, cv2_b)

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward pass with attention."""
        # Initial conv
        x, h, w = self.cv1(x, batch_size, height, width)

        # Prepare for slicing
        if x.memory_config().is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.reshape(x, [batch_size, h, w, 2 * self.hidden_channels])
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Split
        x1 = ttnn.slice(x, (0, 0, 0, 0), (batch_size, h, w, self.hidden_channels))
        x2 = ttnn.slice(x, (0, 0, 0, self.hidden_channels), (batch_size, h, w, 2 * self.hidden_channels))

        x1 = ttnn.to_layout(x1, ttnn.TILE_LAYOUT)
        x2 = ttnn.to_layout(x2, ttnn.TILE_LAYOUT)

        # Collect features
        features = [x1, x2]

        # Process through attention blocks
        y = x2
        for attn in self.attn_blocks:
            y, _, _ = attn(y, batch_size, h, w)
            features.append(y)

        # Concatenate
        concat_features = []
        for f in features:
            if f.memory_config().is_sharded():
                f = ttnn.sharded_to_interleaved(f, memory_config=ttnn.L1_MEMORY_CONFIG)
            f = ttnn.to_layout(f, ttnn.TILE_LAYOUT)
            concat_features.append(f)

        out = ttnn.concat(concat_features, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Final fusion
        out, h, w = self.cv2(out, batch_size, h, w)

        return out, h, w


class TtSPPF:
    """
    Spatial Pyramid Pooling - Fast (SPPF).

    Efficient implementation using sequential max pooling. Narrow inputs pool row-major and height sharded (a few
    image rows per core, ~11 us per pool at 20x20x128 vs ~28 us in tile layout on 28 cores); their outputs share
    one sharding, so the concat is shard-local. For a single wide input (yolo26s, 256 channels) the untilize/tilize
    around the row-major pools costs more than it saves, so it pools in tile layout; from batch 2 on row-major wins.
    """

    def __init__(self, device, in_channels: int, out_channels: int, kernel_size: int = 5, name: str = ""):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        hidden_channels = in_channels // 2

        # IMPORTANT: cv1 has NO activation (Identity in PyTorch), cv2 has SiLU
        self.cv1 = TtConvBNSiLU(
            device,
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=False,
            name=f"{name}.cv1",
        )
        self.cv2 = TtConvBNSiLU(
            device,
            hidden_channels * 4,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=True,
            name=f"{name}.cv2",
        )

        self.hidden_channels = hidden_channels

    def load_weights(self, weight_loader, prefix: str):
        """Load weights from weight loader."""
        # YOLO26 naming: model.9.cv1.conv.weight, model.9.cv1.bn.weight
        cv1_w, cv1_b = weight_loader.get_conv_bn(f"{prefix}.cv1")
        self.cv1.load_weights(cv1_w, cv1_b)
        cv2_w, cv2_b = weight_loader.get_conv_bn(f"{prefix}.cv2")
        self.cv2.load_weights(cv2_w, cv2_b)

    def _row_major_pools(self, batch_size: int) -> bool:
        return self.hidden_channels <= 128 or batch_size > 1

    def _pool(self, x: ttnn.Tensor, batch_size: int, h: int, w: int) -> ttnn.Tensor:
        return ttnn.max_pool2d(
            input_tensor=x,
            batch_size=batch_size,
            input_h=h,
            input_w=w,
            channels=self.hidden_channels,
            kernel_size=[self.kernel_size, self.kernel_size],
            stride=[1, 1],
            padding=[self.padding, self.padding],
            dilation=[1, 1],
            # the first row-major pool shards its interleaved input; the later ones inherit that sharding
            applied_shard_scheme=None if x.is_sharded() else ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            output_layout=ttnn.ROW_MAJOR_LAYOUT if self._row_major_pools(batch_size) else ttnn.TILE_LAYOUT,
        )

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward with spatial pyramid pooling."""
        identity = x
        y, h, w = self.cv1(x, batch_size, height, width)
        temps = [y]

        if self._row_major_pools(batch_size):
            if y.is_sharded():
                y = ttnn.sharded_to_interleaved(y, memory_config=ttnn.L1_MEMORY_CONFIG)
                temps.append(y)
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
            temps.append(y)

        # Sequential max pooling
        pools = [y]
        for _ in range(3):
            pools.append(self._pool(pools[-1], batch_size, h, w))
        temps += pools[1:]

        if self._row_major_pools(batch_size):
            pools[0] = ttnn.to_memory_config(y, _height_sharded_like(pools[1], self.hidden_channels))
            cat = ttnn.concat(pools, dim=3, memory_config=_height_sharded_like(pools[1], 4 * self.hidden_channels))
            temps += [pools[0], cat]
            # a few image rows per core is not tile-aligned: hand cv2 an interleaved tensor to shard its own way
            cat = ttnn.sharded_to_interleaved(cat, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            cat = concat(pools)
        temps.append(cat)

        out, h, w = self.cv2(cat, batch_size, h, w)
        _free(*{id(t): t for t in temps}.values())

        # IMPORTANT: SPPF has residual connection (add=True in PyTorch)
        residual = add(out, identity)
        _free(out)
        return residual, h, w


class TtUpsample:
    """Nearest 2x upsample of a flattened [1, 1, N*H*W, C] tile tensor (or NHWC), returning row-major NHWC
    [N, 2H, 2W, C] in L1 (concat flattens and tilizes it)."""

    def __init__(self, scale_factor: int = 2):
        self.scale_factor = scale_factor

    def __call__(
        self, x: ttnn.Tensor, batch_size: int, height: int, width: int, channels: int
    ) -> Tuple[ttnn.Tensor, int, int]:
        """Upsample using nearest neighbor interpolation."""
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG) if x.is_sharded() else x
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [batch_size, height, width, channels])
        x = ttnn.upsample(x, scale_factor=self.scale_factor, mode="nearest", memory_config=ttnn.L1_MEMORY_CONFIG)
        return x, height * self.scale_factor, width * self.scale_factor


class TtAttention:
    """
    Multi-head self-attention module for YOLO26 PSA blocks.

    Uses QKV attention with positional encoding.
    """

    def __init__(self, device, dim: int, num_heads: int = 2, name: str = ""):
        self.device = device
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = self.head_dim // 2
        self.scale = self.key_dim**-0.5

        # QKV conv: [all q | all k | all v], q and k padded to head_dim per head (see load_weights)
        qkv_dim = 3 * num_heads * self.head_dim
        self.qkv = TtConvBNSiLU(
            device, dim, qkv_dim, kernel_size=1, stride=1, padding=0, activation=False, name=f"{name}.qkv"
        )
        self.proj = TtConvBNSiLU(
            device, dim, dim, kernel_size=1, stride=1, padding=0, activation=False, name=f"{name}.proj"
        )
        self.pe = TtConvBNSiLU(
            device, dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, activation=False, name=f"{name}.pe"
        )

    def load_weights(self, weight_loader, prefix: str):
        """Load attention weights.

        qkv output channels are reordered from per-head [q, k, v] groups to [all q | all k | all v], with q and k
        zero-padded from key_dim to head_dim channels per head (q.k is unchanged), which is the fused layout
        nlp_create_qkv_heads splits into per-head tensors. The attention scale is folded into the q weights.
        """
        nh, kd, hd = self.num_heads, self.key_dim, self.head_dim
        per_head = 2 * kd + hd
        qkv_w, qkv_b = weight_loader.get_conv_bn(f"{prefix}.qkv")
        w, b = ttnn.to_torch(qkv_w).float(), ttnn.to_torch(qkv_b).float().reshape(-1)
        q_rows = [i * per_head + j for i in range(nh) for j in range(kd)]
        k_rows = [i * per_head + kd + j for i in range(nh) for j in range(kd)]
        v_rows = [i * per_head + 2 * kd + j for i in range(nh) for j in range(hd)]
        segments = [(kd, hd)] * nh
        q_w = scatter_segments(w[q_rows] * self.scale, segments, 0)
        q_b = scatter_segments(b[q_rows] * self.scale, segments, 0)
        k_w, k_b = scatter_segments(w[k_rows], segments, 0), scatter_segments(b[k_rows], segments, 0)
        w = torch.cat([q_w, k_w, w[v_rows]], 0)
        b = torch.cat([q_b, k_b, b[v_rows]], 0)
        self.qkv.load_weights(ttnn.from_torch(w), ttnn.from_torch(b.reshape(1, 1, 1, -1)))
        proj_w, proj_b = weight_loader.get_conv_bn(f"{prefix}.proj")
        self.proj.load_weights(proj_w, proj_b)
        pe_w, pe_b = weight_loader.get_conv_bn(f"{prefix}.pe")
        self.pe.load_weights(pe_w, pe_b)

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward pass, token-major: x is [1, 1, B * N, C] with N = H * W tokens per image.

        PyTorch YOLO26 Attention flow:
        1. qkv = self.qkv(x)  # Conv projection
        2. qkv.view(B, num_heads, key_dim*2+head_dim, N)
        3. split into Q[B,heads,key_dim,N], K[B,heads,key_dim,N], V[B,heads,head_dim,N]
        4. attn = (Q^T @ K) * scale
        5. attn = softmax(attn)
        6. out = V @ attn^T
        7. out = out.view(B,C,H,W) + pe(v.view(B,C,H,W))
        8. out = proj(out)

        With tokens as rows this is softmax(Q K^T) V batched over heads: nlp_create_qkv_heads splits the fused
        qkv into [1, heads, N, head_dim] (K transposed), and the heads concatenate back to PyTorch's channel
        order (head-major). Softmax runs on the tile-padded [N, N] scores but only over the N logical columns.
        """
        nh, hd = self.num_heads, self.head_dim

        qkv, h, w = self.qkv(x, batch_size, height, width)
        if qkv.is_sharded():
            qkv = ttnn.sharded_to_interleaved(qkv, memory_config=ttnn.L1_MEMORY_CONFIG)
        rows = qkv.shape[-2]
        tokens = rows // batch_size

        # Each image attends over its own tokens: with several images the flattened rows become a batch dimension
        # (H * W = 400 is not tile-aligned, so through row-major)
        qkv_heads = qkv
        if batch_size > 1:
            qkv_heads = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)
            qkv_heads = ttnn.reshape(qkv_heads, [batch_size, 1, tokens, qkv.shape[-1]])
            qkv_heads = ttnn.to_layout(qkv_heads, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        q, k_t, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_heads, num_heads=nh, transpose_k_heads=True, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        # One 32-token block of one head per core: the default config for these batched matmuls uses one core per
        # head for q.k^T
        token_tiles = -(-tokens // ttnn.TILE_SIZE)
        head_tiles = hd // ttnn.TILE_SIZE
        grid = self.device.compute_with_storage_grid_size()
        scores_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=head_tiles,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=token_tiles,
        )
        values_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=token_tiles,
            out_subblock_h=1,
            out_subblock_w=head_tiles,
            per_core_M=1,
            per_core_N=head_tiles,
        )
        attn = ttnn.matmul(
            q,
            k_t,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=ATTN_COMPUTE_KERNEL_CONFIG,
            program_config=scores_config,
        )
        _free(q, k_t)
        scores = attn
        attn = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=SOFTMAX_COMPUTE_KERNEL_CONFIG)
        _free(scores)
        heads = ttnn.matmul(
            attn,
            v,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=ATTN_COMPUTE_KERNEL_CONFIG,
            program_config=values_config,
        )
        _free(attn, v)
        out = ttnn.experimental.nlp_concat_heads(heads, memory_config=ttnn.L1_MEMORY_CONFIG)
        _free(heads)
        if batch_size > 1:
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
            out = ttnn.reshape(out, [1, 1, rows, out.shape[-1]])
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Positional encoding: depthwise conv over v of all heads (the last section of the fused qkv)
        v_all = ttnn.slice(qkv, [0, 0, 0, 2 * nh * hd], [1, 1, rows, 3 * nh * hd], memory_config=ttnn.L1_MEMORY_CONFIG)
        _free(qkv)
        pe, _, _ = self.pe(v_all, batch_size, h, w)
        _free(v_all)
        mixed = add(pe, out)
        _free(pe, out)

        # Project back (TTNN conv)
        out, h, w = self.proj(mixed, batch_size, h, w)
        _free(mixed)
        return out, h, w


class TtPSABlock:
    """
    PSA (Positional Self-Attention) block for YOLO26.

    Combines attention with FFN.
    """

    def __init__(self, device, dim: int, num_heads: int = 2, name: str = ""):
        self.device = device
        self.dim = dim

        self.attn = TtAttention(device, dim, num_heads, name=f"{name}.attn")

        # FFN: Conv(SiLU)-Conv (indices 0 and 1, not 0 and 2!)
        self.ffn_cv1 = TtConvBNSiLU(device, dim, dim * 2, kernel_size=1, stride=1, padding=0, name=f"{name}.ffn.0")
        self.ffn_cv2 = TtConvBNSiLU(
            device, dim * 2, dim, kernel_size=1, stride=1, padding=0, activation=False, name=f"{name}.ffn.1"
        )

    def load_weights(self, weight_loader, prefix: str):
        """Load PSA block weights."""
        self.attn.load_weights(weight_loader, f"{prefix}.attn")
        ffn1_w, ffn1_b = weight_loader.get_conv_bn(f"{prefix}.ffn.0")
        self.ffn_cv1.load_weights(ffn1_w, ffn1_b)
        ffn2_w, ffn2_b = weight_loader.get_conv_bn(f"{prefix}.ffn.1")
        self.ffn_cv2.load_weights(ffn2_w, ffn2_b)

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward pass."""
        # Attention with residual
        attn_out, h, w = self.attn(x, batch_size, height, width)
        x1 = add(attn_out, x)
        _free(attn_out)

        # FFN with residual
        hidden, h, w = self.ffn_cv1(x1, batch_size, h, w)
        ffn_out, h, w = self.ffn_cv2(hidden, batch_size, h, w)
        out = add(ffn_out, x1)
        _free(hidden, ffn_out, x1)
        return out, h, w


class TtC3k2PSA:
    """
    C3k2 with PSA attention for model.22.

    Structure: cv1 -> split -> [a, b] -> b goes through Bottleneck+PSABlock -> concat -> cv2

    Different from regular C3k2:
    - m.0 is Sequential(Bottleneck, PSABlock) instead of C3k
    """

    def __init__(self, device, in_channels: int, out_channels: int, hidden_channels: int, n: int = 1, name: str = ""):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n = n

        # cv1: in_channels -> 2 * hidden_channels ([y0 | y1] for the concat), cv1_b: y1 alone
        self.cv1 = TtConvBNSiLU(
            device, in_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1"
        )
        self.cv1_b = TtConvBNSiLU(
            device, in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1_b"
        )

        # cv2: (2 + n) * hidden_channels -> out_channels
        self.cv2 = TtConvBNSiLU(
            device, (2 + n) * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, name=f"{name}.cv2"
        )

        # m.0 = Sequential(Bottleneck, PSABlock)
        # m.0.0 = Bottleneck (hidden_channels -> hidden_channels)
        self.bottleneck = TtBottleneck(
            device, hidden_channels, hidden_channels, shortcut=True, expansion=0.5, name=f"{name}.m.0.0"
        )

        # m.0.1 = PSABlock (hidden_channels)
        self.psa = TtPSABlock(device, hidden_channels, num_heads=max(hidden_channels // 64, 1), name=f"{name}.m.0.1")

    def load_weights(self, weight_loader, prefix: str):
        """Load weights."""
        cv1_w, cv1_b = weight_loader.get_conv_bn(f"{prefix}.cv1")
        self.cv1.load_weights(cv1_w, cv1_b)
        h = self.hidden_channels
        self.cv1_b.load_weights(*slice_out_channels(cv1_w, cv1_b, h, 2 * h))
        cv2_w, cv2_b = weight_loader.get_conv_bn(f"{prefix}.cv2")
        self.cv2.load_weights(cv2_w, cv2_b)

        # Load bottleneck weights (m.0.0)
        bn_cv1_w, bn_cv1_b = weight_loader.get_conv_bn(f"{prefix}.m.0.0.cv1")
        bn_cv2_w, bn_cv2_b = weight_loader.get_conv_bn(f"{prefix}.m.0.0.cv2")
        self.bottleneck.load_weights(bn_cv1_w, bn_cv1_b, bn_cv2_w, bn_cv2_b)

        # Load PSA weights (m.0.1)
        self.psa.load_weights(weight_loader, f"{prefix}.m.0.1")

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward pass.

        C3k2 flow:
        1. cv1(x) -> chunk into 2: [y0, y1]
        2. for m in self.m: y.extend(m(y[-1]))
        3. cv2(concat(y))

        So output of concat is [y0, y1, m(y1)] = 3 * hidden_channels = 384
        """
        x01, h, w = self.cv1(x, batch_size, height, width)
        y, _, _ = self.cv1_b(x, batch_size, height, width)

        # Process y1 through Sequential(Bottleneck, PSABlock)
        y1, h, w = self.bottleneck(y, batch_size, h, w)
        y2, h, w = self.psa(y1, batch_size, h, w)
        _free(y, y1)

        cat = concat([x01, y2])
        _free(x01, y2)
        out, h, w = self.cv2(cat, batch_size, h, w)
        _free(cat)
        return out, h, w


class TtC2PSA:
    """
    C2PSA module for YOLO26 - Cross Stage Partial with PSA attention.

    Structure:
    - cv1: Conv that doubles channels
    - split into two branches (a, b)
    - b goes through PSA blocks
    - concat(a, b) then cv2
    """

    def __init__(self, device, in_channels: int, out_channels: int, n: int = 1, name: str = ""):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c = in_channels // 2  # Hidden channels per branch

        # cv1 output is split into a and b: computed as two convs on the two halves of its weights
        self.cv1_a = TtConvBNSiLU(device, in_channels, self.c, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1_a")
        self.cv1_b = TtConvBNSiLU(device, in_channels, self.c, kernel_size=1, stride=1, padding=0, name=f"{name}.cv1_b")
        # cv2 takes concat of 2 branches
        self.cv2 = TtConvBNSiLU(
            device, in_channels, out_channels, kernel_size=1, stride=1, padding=0, name=f"{name}.cv2"
        )

        # PSA blocks process the second branch
        self.psa_blocks = [
            TtPSABlock(device, self.c, num_heads=max(self.c // 64, 1), name=f"{name}.m.{i}") for i in range(n)
        ]

    def load_weights(self, weight_loader, prefix: str):
        """Load C2PSA weights."""
        cv1_w, cv1_b = weight_loader.get_conv_bn(f"{prefix}.cv1")
        self.cv1_a.load_weights(*slice_out_channels(cv1_w, cv1_b, 0, self.c))
        self.cv1_b.load_weights(*slice_out_channels(cv1_w, cv1_b, self.c, 2 * self.c))
        cv2_w, cv2_b = weight_loader.get_conv_bn(f"{prefix}.cv2")
        self.cv2.load_weights(cv2_w, cv2_b)

        for i, block in enumerate(self.psa_blocks):
            block.load_weights(weight_loader, f"{prefix}.m.{i}")

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward pass."""
        a, h, w = self.cv1_a(x, batch_size, height, width)
        b, _, _ = self.cv1_b(x, batch_size, height, width)

        # Process b through PSA blocks
        for block in self.psa_blocks:
            b_next, h, w = block(b, batch_size, h, w)
            _free(b)
            b = b_next

        cat = concat([a, b])
        _free(a, b)
        out, h, w = self.cv2(cat, batch_size, h, w)
        _free(cat)
        return out, h, w


class TtYOLO26Backbone:
    """
    YOLO26 Backbone network.

    CSP-style backbone with C2f blocks.
    Outputs multi-scale features: P3, P4, P5

    YOLO26n actual structure (from weights analysis):
    - model.0: Conv 3→16, stride=2 (stem)
    - model.1: Conv 16→32, stride=2
    - model.2: C2f 32→64
    - model.3: Conv 64→64, stride=2
    - model.4: C2f 64→128 → P3
    - model.5: Conv 128→128, stride=2
    - model.6: C2f 128→128 → P4 (note: stays at 128, not 256!)
    - model.7: Conv 128→256, stride=2
    - model.8: C2f 256→256
    - model.9: SPPF 256→256 → P5
    """

    def __init__(self, device, variant: str = "yolo26n", input_channels: int = 3, input_scale: float = 1.0):
        self.device = device
        self.input_channels = input_channels
        self.input_scale = input_scale
        k = get_width_multiplier(variant)

        # YOLO26n channels from weights analysis, scaled by k for wider variants
        # All C3k2 layers have n=1 inner block (verified from weights)
        # hidden_channels varies: 0.25 expansion for early, 0.5 for later stages

        # Stem - model.0: 3→16
        self.stem = TtConvBNSiLU(device, input_channels, 16 * k, kernel_size=3, stride=2, padding=1, name="model.0")

        # Stage 1 - model.1: 16→32, model.2: C2f 32→64 (hidden=16, expansion=0.25)
        self.conv1 = TtConvBNSiLU(device, 16 * k, 32 * k, kernel_size=3, stride=2, padding=1, name="model.1")
        self.c2f_1 = TtC2f(device, 32 * k, 64 * k, hidden_channels=16 * k, n=1, name="model.2")

        # Stage 2 - model.3: 64→64, model.4: C2f 64→128 → P3 (hidden=32, expansion=0.25)
        self.conv2 = TtConvBNSiLU(device, 64 * k, 64 * k, kernel_size=3, stride=2, padding=1, name="model.3")
        self.c2f_2 = TtC2f(device, 64 * k, 128 * k, hidden_channels=32 * k, n=1, name="model.4")

        # Stage 3 - model.5: 128→128, model.6: C3k2 (C3k inner) 128→128 → P4 (hidden=64, expansion=0.5)
        self.conv3 = TtConvBNSiLU(device, 128 * k, 128 * k, kernel_size=3, stride=2, padding=1, name="model.5")
        self.c3k2_3 = TtC3k2(device, 128 * k, 128 * k, hidden_channels=64 * k, n=1, name="model.6")

        # Stage 4 - model.7: 128→256, model.8: C3k2 (C3k inner) 256→256 (hidden=128, expansion=0.5), model.9: SPPF → P5
        self.conv4 = TtConvBNSiLU(device, 128 * k, 256 * k, kernel_size=3, stride=2, padding=1, name="model.7")
        self.c3k2_4 = TtC3k2(device, 256 * k, 256 * k, hidden_channels=128 * k, n=1, name="model.8")
        self.sppf = TtSPPF(device, 256 * k, 256 * k, kernel_size=5, name="model.9")

        # Store channel info for neck (actual values from weights)
        self.channels = {
            "p3": 128 * k,  # From model.4 output
            "p4": 128 * k,  # From model.6 output (NOT 256!)
            "p5": 256 * k,  # From model.9 output
        }

    def load_weights(self, weight_loader):
        """Load backbone weights."""
        # Stem - model.0. Input normalization is folded in (conv(x * s, W) == conv(x, W * s)), and the input
        # channels are zero-padded when the input is padded (e.g. 3 -> 16)
        w, b = weight_loader.get_conv_bn("model.0")
        if self.input_channels != w.shape[1] or self.input_scale != 1.0:
            w_torch = ttnn.to_torch(w) * self.input_scale
            w_torch = torch.nn.functional.pad(w_torch, (0, 0, 0, 0, 0, self.input_channels - w_torch.shape[1]))
            w = ttnn.from_torch(w_torch)
        self.stem.load_weights(w, b)

        # Stage 1 - model.1, model.2
        w, b = weight_loader.get_conv_bn("model.1")
        self.conv1.load_weights(w, b)
        self.c2f_1.load_weights(weight_loader, "model.2")

        # Stage 2 - model.3, model.4
        w, b = weight_loader.get_conv_bn("model.3")
        self.conv2.load_weights(w, b)
        self.c2f_2.load_weights(weight_loader, "model.4")

        # Stage 3 - model.5, model.6
        w, b = weight_loader.get_conv_bn("model.5")
        self.conv3.load_weights(w, b)
        self.c3k2_3.load_weights(weight_loader, "model.6")

        # Stage 4 - model.7, model.8, model.9
        w, b = weight_loader.get_conv_bn("model.7")
        self.conv4.load_weights(w, b)
        self.c3k2_4.load_weights(weight_loader, "model.8")
        self.sppf.load_weights(weight_loader, "model.9")

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        deallocate_input: bool = False,
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Forward pass.

        Args:
            x: NHWC input, or flattened [1, 1, N*H*W, C] (e.g. sharded) with batch_size/height/width given
            deallocate_input: free x after the stem conv (needed for trace capture: the input buffer must be
                released inside the trace so the re-allocated input lands at the same address)

        Returns:
            Tuple of ((p3, h, w), (p4, h, w), (p5, h, w)) feature maps
        """
        batch_size = x.shape[0] if batch_size is None else batch_size
        h = x.shape[1] if height is None else height
        w = x.shape[2] if width is None else width

        # Stem
        x_in = x
        x, h, w = self.stem(x, batch_size, h, w)
        if deallocate_input:
            ttnn.deallocate(x_in)

        # Stage 1
        x1, h, w = self.conv1(x, batch_size, h, w)
        _free(x)
        x2, h, w = self.c2f_1(x1, batch_size, h, w)
        _free(x1)

        # Stage 2 → P3 output (128 channels)
        x3, h, w = self.conv2(x2, batch_size, h, w)
        _free(x2)
        p3, p3_h, p3_w = self.c2f_2(x3, batch_size, h, w)
        _free(x3)

        # Stage 3 → P4 output (128 channels)
        x4, h, w = self.conv3(p3, batch_size, p3_h, p3_w)
        p4, p4_h, p4_w = self.c3k2_3(x4, batch_size, h, w)
        _free(x4)

        # Stage 4 → P5 output (256 channels)
        x5, h, w = self.conv4(p4, batch_size, p4_h, p4_w)
        x6, h, w = self.c3k2_4(x5, batch_size, h, w)
        _free(x5)
        p5, p5_h, p5_w = self.sppf(x6, batch_size, h, w)
        _free(x6)

        return (p3, p3_h, p3_w), (p4, p4_h, p4_w), (p5, p5_h, p5_w)


class TtYOLO26Neck:
    """
    YOLO26 Neck (PAN - Path Aggregation Network).

    Fuses multi-scale features from backbone.

    YOLO26n Neck structure:
    - model.10: C2PSA 256→256 (with attention)
    - model.11: Upsample x2
    - model.12: Concat from [-1, 6] = 256+128=384
    - model.13: C3k2 384→128
    - model.14: Upsample x2
    - model.15: Concat from [-1, 4] = 128+128=256
    - model.16: C3k2 256→64 → N3 output
    - model.17: Conv 64→64 (downsample)
    - model.18: Concat from [-1, 13] = 64+128=192
    - model.19: C3k2 192→128 → N4 output
    - model.20: Conv 128→128 (downsample)
    - model.21: Concat from [-1, 10] = 128+256=384
    - model.22: C3k2 384→256 → N5 output
    """

    def __init__(self, device, variant: str = "yolo26n"):
        self.device = device
        self.upsample = TtUpsample(scale_factor=2)
        k = get_width_multiplier(variant)
        self.k = k

        # model.10: C2PSA 256→256
        self.c2psa_10 = TtC2PSA(device, 256 * k, 256 * k, n=1, name="model.10")

        # model.13: C3k2 384→128
        self.c3k2_13 = TtC3k2(device, 384 * k, 128 * k, hidden_channels=64 * k, n=1, name="model.13")

        # model.16: C3k2 256→64 → N3 output
        self.c3k2_16 = TtC3k2(device, 256 * k, 64 * k, hidden_channels=32 * k, n=1, name="model.16")

        # model.17: Conv 64→64 (downsample)
        self.conv_17 = TtConvBNSiLU(device, 64 * k, 64 * k, kernel_size=3, stride=2, padding=1, name="model.17")

        # model.19: C3k2 192→128 → N4 output
        self.c3k2_19 = TtC3k2(device, 192 * k, 128 * k, hidden_channels=64 * k, n=1, name="model.19")

        # model.20: Conv 128→128 (downsample)
        self.conv_20 = TtConvBNSiLU(device, 128 * k, 128 * k, kernel_size=3, stride=2, padding=1, name="model.20")

        # model.22: C3k2 (Bottleneck + PSABlock inner) 384→256 → N5 output
        self.c3k2_22 = TtC3k2PSA(device, 384 * k, 256 * k, hidden_channels=128 * k, n=1, name="model.22")

        # Output channels
        self.channels = {
            "n3": 64 * k,  # From model.16
            "n4": 128 * k,  # From model.19
            "n5": 256 * k,  # From model.22
        }

    def load_weights(self, weight_loader):
        """Load neck weights."""
        # model.10: C2PSA with attention
        self.c2psa_10.load_weights(weight_loader, "model.10")

        # model.13: C3k2
        self.c3k2_13.load_weights(weight_loader, "model.13")

        # model.16: C3k2 → N3
        self.c3k2_16.load_weights(weight_loader, "model.16")

        # model.17: Downsample conv
        w, b = weight_loader.get_conv_bn("model.17")
        self.conv_17.load_weights(w, b)

        # model.19: C3k2 → N4
        self.c3k2_19.load_weights(weight_loader, "model.19")

        # model.20: Downsample conv
        w, b = weight_loader.get_conv_bn("model.20")
        self.conv_20.load_weights(w, b)

        # model.22: C3k2 → N5
        self.c3k2_22.load_weights(weight_loader, "model.22")

    def __call__(
        self,
        p3_data: Tuple[ttnn.Tensor, int, int],
        p4_data: Tuple[ttnn.Tensor, int, int],
        p5_data: Tuple[ttnn.Tensor, int, int],
        batch_size: int,
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Forward pass through PAN neck.

        Data flow:
        - P5 (256ch) → C2PSA_10 → upsample → concat with P4 (128ch) = 384ch
        - → C3k2_13 (128ch) → upsample → concat with P3 (128ch) = 256ch
        - → C3k2_16 → N3 (64ch)
        - N3 → downsample → concat with C3k2_13 output (128ch) = 192ch
        - → C3k2_19 → N4 (128ch)
        - N4 → downsample → concat with C2PSA_10 output (256ch) = 384ch
        - → C3k2_22 → N5 (256ch)

        Returns:
            Tuple of (n3, n4, n5) neck output features
        """
        p3, p3_h, p3_w = p3_data  # 128 channels
        p4, p4_h, p4_w = p4_data  # 128 channels
        p5, p5_h, p5_w = p5_data  # 256 channels

        # ===== Top-down path =====
        # model.10: Process P5 with C2PSA (256→256)
        # Each feature map is released right after its last use (the backbone outputs are consumed here)
        p5_out, p5_out_h, p5_out_w = self.c2psa_10(p5, batch_size, p5_h, p5_w)
        _free(p5)

        # Upsample P5 output and concat with P4: 256 + 128 = 384 channels
        p5_up, _, _ = self.upsample(p5_out, batch_size, p5_out_h, p5_out_w, 256 * self.k)
        cat = concat([p5_up, p4])
        _free(p5_up, p4)

        # model.13: C3k2 (384→128)
        f_p4, f_p4_h, f_p4_w = self.c3k2_13(cat, batch_size, p4_h, p4_w)
        _free(cat)

        # Upsample and concat with P3: 128 + 128 = 256 channels
        f_p4_up, _, _ = self.upsample(f_p4, batch_size, f_p4_h, f_p4_w, 128 * self.k)
        cat = concat([f_p4_up, p3])
        _free(f_p4_up, p3)

        # model.16: C3k2 (256→64) → N3 output
        n3, n3_h, n3_w = self.c3k2_16(cat, batch_size, p3_h, p3_w)
        _free(cat)

        # ===== Bottom-up path =====
        # model.17: Downsample N3 (64→64), concat with f_p4: 64 + 128 = 192 channels
        n3_down, n3_down_h, n3_down_w = self.conv_17(n3, batch_size, n3_h, n3_w)
        cat = concat([n3_down, f_p4])
        _free(n3_down, f_p4)

        # model.19: C3k2 (192→128) → N4 output
        n4, n4_h, n4_w = self.c3k2_19(cat, batch_size, f_p4_h, f_p4_w)
        _free(cat)

        # model.20: Downsample N4 (128→128), concat with p5_out: 128 + 256 = 384 channels
        n4_down, n4_down_h, n4_down_w = self.conv_20(n4, batch_size, n4_h, n4_w)
        cat = concat([n4_down, p5_out])
        _free(n4_down, p5_out)

        # model.22: C3k2 (384→256) → N5 output
        n5, n5_h, n5_w = self.c3k2_22(cat, batch_size, p5_out_h, p5_out_w)
        _free(cat)

        return (n3, n3_h, n3_w), (n4, n4_h, n4_w), (n5, n5_h, n5_w)


class TtDetectHead:
    """
    YOLO26 Detection Head - one scale.

    Structure for each scale:
    - cv2: Conv -> Conv -> Conv2d (bbox output, 4 channels)
    - cv3: Sequential(DWConv, Conv) -> Sequential(DWConv, Conv) -> Conv2d (cls output, nc channels)
    """

    def __init__(
        self,
        device,
        in_channels: int,
        scale_idx: int,
        num_classes: int = 80,
        box_hidden: int = 16,
        cls_hidden: int = 80,
        name: str = "",
    ):
        self.device = device
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.scale_idx = scale_idx

        # cv2 branch: bbox (output 4 channels)
        # Conv 3x3 (in_ch -> c2), Conv 3x3 (c2 -> c2), Conv2d 1x1 (c2 -> 4)
        self.cv2_0 = TtConvBNSiLU(
            device,
            in_channels,
            box_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            name=f"{name}.one2one_cv2.{scale_idx}.0",
        )
        self.cv2_1 = TtConvBNSiLU(
            device, box_hidden, box_hidden, kernel_size=3, stride=1, padding=1, name=f"{name}.one2one_cv2.{scale_idx}.1"
        )
        self.cv2_2 = TtConvBNSiLU(
            device,
            box_hidden,
            4,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=False,
            name=f"{name}.one2one_cv2.{scale_idx}.2",
        )

        # cv3 branch: cls (output nc channels)
        # More complex - has depthwise separable convs
        # Simplified: use regular convs
        self.cv3_0_0 = TtConvBNSiLU(
            device,
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            name=f"{name}.one2one_cv3.{scale_idx}.0.0",
        )
        self.cv3_0_1 = TtConvBNSiLU(
            device,
            in_channels,
            cls_hidden,
            kernel_size=1,
            stride=1,
            padding=0,
            name=f"{name}.one2one_cv3.{scale_idx}.0.1",
        )
        self.cv3_1_0 = TtConvBNSiLU(
            device,
            cls_hidden,
            cls_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=cls_hidden,
            name=f"{name}.one2one_cv3.{scale_idx}.1.0",
        )
        self.cv3_1_1 = TtConvBNSiLU(
            device,
            cls_hidden,
            cls_hidden,
            kernel_size=1,
            stride=1,
            padding=0,
            name=f"{name}.one2one_cv3.{scale_idx}.1.1",
        )
        self.cv3_2 = TtConvBNSiLU(
            device,
            cls_hidden,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=False,
            name=f"{name}.one2one_cv3.{scale_idx}.2",
        )

    def load_weights(self, weight_loader, prefix: str):
        """Load detection head weights."""
        # cv2 weights
        w, b = weight_loader.get_conv_bn(f"{prefix}.one2one_cv2.{self.scale_idx}.0")
        self.cv2_0.load_weights(w, b)
        w, b = weight_loader.get_conv_bn(f"{prefix}.one2one_cv2.{self.scale_idx}.1")
        self.cv2_1.load_weights(w, b)
        # cv2.2 is plain Conv2d without BN
        self.cv2_2.load_weights_no_bn(weight_loader, f"{prefix}.one2one_cv2.{self.scale_idx}.2")

        # cv3 weights
        w, b = weight_loader.get_conv_bn(f"{prefix}.one2one_cv3.{self.scale_idx}.0.0")
        self.cv3_0_0.load_weights(w, b)
        w, b = weight_loader.get_conv_bn(f"{prefix}.one2one_cv3.{self.scale_idx}.0.1")
        self.cv3_0_1.load_weights(w, b)
        w, b = weight_loader.get_conv_bn(f"{prefix}.one2one_cv3.{self.scale_idx}.1.0")
        self.cv3_1_0.load_weights(w, b)
        w, b = weight_loader.get_conv_bn(f"{prefix}.one2one_cv3.{self.scale_idx}.1.1")
        self.cv3_1_1.load_weights(w, b)
        # cv3.2 is plain Conv2d without BN
        self.cv3_2.load_weights_no_bn(weight_loader, f"{prefix}.one2one_cv3.{self.scale_idx}.2")

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward pass. Returns (bbox_output, cls_output)."""
        # cv2 branch: bbox
        b0, h, w = self.cv2_0(x, batch_size, height, width)
        b1, h, w = self.cv2_1(b0, batch_size, h, w)
        bbox, h, w = self.cv2_2(b1, batch_size, h, w)
        _free(b0, b1)

        # cv3 branch: cls
        cls = x
        for i, conv in enumerate((self.cv3_0_0, self.cv3_0_1, self.cv3_1_0, self.cv3_1_1, self.cv3_2)):
            out, h, w = conv(cls, batch_size, h if i else height, w if i else width)
            if i:
                _free(cls)
            cls = out

        return bbox, cls, h, w


class TtYOLO26Head:
    """
    YOLO26 Detection Head.

    End-to-end detection head outputting class scores and bounding boxes.

    YOLO26 Head structure (model.23):
    - one2one_cv2: bbox predictions (4 channels per scale)
    - one2one_cv3: class predictions (80 channels per scale)
    - Inputs from neck: N3 (64ch), N4 (128ch), N5 (256ch)
    """

    def __init__(self, device, variant: str = "yolo26n", num_classes: int = 80):
        self.device = device
        self.num_classes = num_classes
        self.reg_max = 1  # YOLO26 uses reg_max=1
        k = get_width_multiplier(variant)
        ch = (64 * k, 128 * k, 256 * k)

        # Hidden widths as in Ultralytics Detect: c2 = max(16, ch0 // 4, 4 * reg_max), c3 = max(ch0, min(nc, 100))
        box_hidden = max(16, ch[0] // 4, 4 * self.reg_max)
        cls_hidden = max(ch[0], min(num_classes, 100))

        # Create detection heads for each scale
        self.head_n3, self.head_n4, self.head_n5 = [
            TtDetectHead(
                device,
                ch[i],
                scale_idx=i,
                num_classes=num_classes,
                box_hidden=box_hidden,
                cls_hidden=cls_hidden,
                name="model.23",
            )
            for i in range(3)
        ]

    def load_weights(self, weight_loader):
        """Load head weights."""
        self.head_n3.load_weights(weight_loader, "model.23")
        self.head_n4.load_weights(weight_loader, "model.23")
        self.head_n5.load_weights(weight_loader, "model.23")

    def __call__(
        self,
        n3_data: Tuple[ttnn.Tensor, int, int],
        n4_data: Tuple[ttnn.Tensor, int, int],
        n5_data: Tuple[ttnn.Tensor, int, int],
        batch_size: int,
    ) -> Dict[str, List[ttnn.Tensor]]:
        """
        Forward pass.

        Returns dict with 'boxes' and 'scores' lists (one per scale).
        """
        n3, n3_h, n3_w = n3_data
        n4, n4_h, n4_w = n4_data
        n5, n5_h, n5_w = n5_data

        # N3 detection (stride 8, 80x80)
        bbox_n3, cls_n3, h3, w3 = self.head_n3(n3, batch_size, n3_h, n3_w)
        _free(n3)

        # N4 detection (stride 16, 40x40)
        bbox_n4, cls_n4, h4, w4 = self.head_n4(n4, batch_size, n4_h, n4_w)
        _free(n4)

        # N5 detection (stride 32, 20x20)
        bbox_n5, cls_n5, h5, w5 = self.head_n5(n5, batch_size, n5_h, n5_w)
        _free(n5)

        return {
            "boxes": [(bbox_n3, h3, w3), (bbox_n4, h4, w4), (bbox_n5, h5, w5)],
            "scores": [(cls_n3, h3, w3), (cls_n4, h4, w4), (cls_n5, h5, w5)],
        }


class TtYOLO26:
    """
    Complete YOLO26 model for Tenstorrent hardware.

    Combines backbone, neck, and head for end-to-end object detection.
    """

    def __init__(
        self,
        device,
        variant: str = "yolo26n",
        num_classes: int = 80,
        input_channels: int = 3,
        uint8_input: bool = False,
    ):
        """
        Initialize YOLO26 model.

        Args:
            device: TTNN device
            variant: Model variant ('yolo26n', 'yolo26s')
            num_classes: Number of detection classes (default: 80 for COCO)
            input_channels: Input channels; > 3 means the input is zero-padded (stem weights are padded to match)
            uint8_input: Input is the raw uint8 RGB image as [B, 1, H, W * UINT8_INPUT_CHANNELS] (HWC rows, 0..255,
                channels zero-padded; see uint8_input_from_torch). It is sharded by rows, converted and viewed as one
                stick per pixel on device; the 1/255 normalization is folded into the stem weights. Wide rows keep
                both the host -> device upload and the sharding copy to H pages instead of H * W.
        """
        self.device = device
        self.variant = variant
        self.num_classes = num_classes
        if uint8_input:
            input_channels = UINT8_INPUT_CHANNELS
        self.input_channels = input_channels
        self.uint8_input = uint8_input

        self.backbone = TtYOLO26Backbone(
            device, variant, input_channels, input_scale=1.0 / 255.0 if uint8_input else 1.0
        )
        self.neck = TtYOLO26Neck(device, variant)
        self.head = TtYOLO26Head(device, variant, num_classes)

    def load_weights_from_state_dict(self, state_dict: dict):
        """
        Load weights from a state dictionary.

        Args:
            state_dict: PyTorch state dict from Ultralytics model
        """
        from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader

        weight_loader = YOLO26WeightLoader(state_dict)
        weight_loader.print_structure()

        self.backbone.load_weights(weight_loader)
        self.neck.load_weights(weight_loader)
        self.head.load_weights(weight_loader)

    def load_weights_from_ultralytics(self, variant: Optional[str] = None):
        """
        Load weights directly from Ultralytics.

        Args:
            variant: Model variant (defaults to self.variant)
        """
        from models.experimental.yolo26.tt.model_preprocessing import load_yolo26_from_ultralytics

        variant = variant or self.variant
        _, state_dict = load_yolo26_from_ultralytics(variant)
        self.load_weights_from_state_dict(state_dict)

    def input_memory_config(self, batch_size: int, height: int, width: int) -> ttnn.MemoryConfig:
        """L1 height sharding of the uint8 input [B, 1, H, W * C] by image rows (B * H wide pages over the grid).

        A traced runner can keep its persistent input in this layout and copy each new image straight into it,
        so the trace starts from L1 (see preprocess). Above MAX_L1_BATCH the input stays in DRAM and each group of
        images is sharded on its way in (see _forward_grouped).
        """
        if batch_size > MAX_L1_BATCH:
            return ttnn.DRAM_MEMORY_CONFIG
        rows = batch_size * height
        grid = self.device.compute_with_storage_grid_size()
        rows_per_core = -(-rows // (grid.x * grid.y))
        num_cores = -(-rows // rows_per_core)
        shard_spec = ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(num_cores, grid, True),
            [rows_per_core, width * self.input_channels],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    def preprocess(self, x: ttnn.Tensor) -> Tuple[ttnn.Tensor, int, int, int]:
        """uint8 [B, 1, H, W * C] -> bf16 [1, 1, B * H * W, C] height sharded in L1 (values 0..255).

        Image rows are sharded first (B * H wide pages), so the copy out of DRAM is a few large transfers; the
        row-major sharded tensor is then re-viewed as one C-channel stick per pixel, which is free because the
        shard bytes are identical. The stem conv consumes this sharding directly. An input that already has this
        sharding (see input_memory_config) skips the copy.
        """
        batch_size, height, width = x.shape[0], x.shape[2], x.shape[3] // self.input_channels
        rows = batch_size * height
        x = ttnn.reshape(x, [1, 1, rows, width * self.input_channels])
        sharded_copy = None
        if not x.is_sharded():
            x = sharded_copy = ttnn.to_memory_config(x, self.input_memory_config(batch_size, height, width))
        x = ttnn.typecast(x, ttnn.bfloat16)
        if sharded_copy is not None:
            ttnn.deallocate(sharded_copy)
        x = ttnn.experimental.view(x, [1, 1, rows * width, self.input_channels])
        return x, batch_size, height, width

    def __call__(
        self,
        x: ttnn.Tensor,
        batch_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        deallocate_input: bool = False,
    ) -> Dict[str, List[Tuple[ttnn.Tensor, int, int]]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, height, width, channels] in NHWC format, or flattened [1, 1, N*H*W, C]
               (e.g. L1 sharded) with batch_size/height/width given, or with uint8_input the raw image
               [B, 1, H, W * UINT8_INPUT_CHANNELS]
            deallocate_input: free x after the stem conv (used by the trace runner)

        Returns:
            Dict with 'boxes' and 'scores': one (tensor, h, w) per scale (P3, P4, P5)
        """
        if self.uint8_input and x.shape[0] > MAX_L1_BATCH:
            return self._forward_grouped(x)
        if self.uint8_input:
            x, batch_size, height, width = self.preprocess(x)
            deallocate_input = True  # x is now the model's own preprocessed copy
        batch_size = x.shape[0] if batch_size is None else batch_size

        # Backbone: extract multi-scale features
        p3_data, p4_data, p5_data = self.backbone(x, batch_size, height, width, deallocate_input)

        # Neck: fuse features
        n3_data, n4_data, n5_data = self.neck(p3_data, p4_data, p5_data, batch_size)

        # Head: generate detections
        outputs = self.head(n3_data, n4_data, n5_data, batch_size)

        return outputs

    def _forward_grouped(self, x: ttnn.Tensor) -> Dict[str, List[Tuple[ttnn.Tensor, int, int]]]:
        """
        Batch above MAX_L1_BATCH (uint8 input [B, 1, H, W * C] in DRAM). The network is the same per image, so the
        stages above 20x20 run in groups of MAX_L1_BATCH images with the batch-4 programs (all in L1), and the 20x20
        stages (model.8-10, model.22, the 20x20 head), where a larger batch keeps the cores busier, in groups of
        LOW_RES_BATCH. Stage outputs that cross between the two (P3, P4, model.7 / model.20 outputs, the C2PSA
        output, the head outputs) wait in DRAM, so a group's L1 holds no more than at batch 4 / LOW_RES_BATCH.
        """
        bb, nk, hd = self.backbone, self.neck, self.head
        batch_size = x.shape[0]
        dram = ttnn.DRAM_MEMORY_CONFIG
        groups = [(first, min(MAX_L1_BATCH, batch_size - first)) for first in range(0, batch_size, MAX_L1_BATCH)]
        low_groups = [(first, min(LOW_RES_BATCH, batch_size - first)) for first in range(0, batch_size, LOW_RES_BATCH)]

        def park(t):
            out = ttnn.to_memory_config(t, dram) if (t.is_sharded() or not _in_dram(t)) else t
            if out is not t:
                _free(t)
            return out

        def rows_of(t, first, n, h, w):
            """Images first .. first + n of a flattened [1, 1, B * h * w, C] tensor, L1 interleaved."""
            return ttnn.slice(
                t,
                [0, 0, first * h * w, 0],
                [1, 1, (first + n) * h * w, t.shape[3]],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        def join(parts):
            out = ttnn.concat(parts, dim=2, memory_config=dram) if len(parts) > 1 else parts[0]
            if len(parts) > 1:
                _free(*parts)
            return out

        # Backbone above 20x20, per group: P3 (80x80), P4 (40x40) and model.7's 20x20 output are kept
        p3s, p4s, x5s = [], [], []
        for first, n in groups:
            xg = ttnn.slice(x, [first, 0, 0, 0], [first + n, x.shape[1], x.shape[2], x.shape[3]], memory_config=dram)
            x0, n, h, w = self.preprocess(xg)
            _free(xg)
            t, h, w = bb.stem(x0, n, h, w)
            _free(x0)
            t1, h, w = bb.conv1(t, n, h, w)
            _free(t)
            t2, h, w = bb.c2f_1(t1, n, h, w)
            _free(t1)
            t3, h, w = bb.conv2(t2, n, h, w)
            _free(t2)
            p3, p3_h, p3_w = bb.c2f_2(t3, n, h, w)
            _free(t3)
            t4, h, w = bb.conv3(p3, n, p3_h, p3_w)
            p4, p4_h, p4_w = bb.c3k2_3(t4, n, h, w)
            _free(t4)
            x5, p5_h, p5_w = bb.conv4(p4, n, p4_h, p4_w)
            p3s.append(park(p3))
            p4s.append(park(p4))
            x5s.append(park(x5))

        # 20x20 in groups of LOW_RES_BATCH: model.8, SPPF, C2PSA
        x5 = join(x5s)
        p5_outs = []
        for first, n in low_groups:
            xl = rows_of(x5, first, n, p5_h, p5_w) if len(low_groups) > 1 else x5
            x6, h, w = bb.c3k2_4(xl, n, p5_h, p5_w)
            if xl is not x5:
                _free(xl)
            p5, h, w = bb.sppf(x6, n, h, w)
            _free(x6)
            t, h, w = nk.c2psa_10(p5, n, h, w)
            _free(p5)
            p5_outs.append(park(t))
        _free(x5)
        p5_out = join(p5_outs)

        # Neck and heads above 20x20, per group; model.20's 20x20 output is kept
        n4_downs, box3, cls3, box4, cls4 = [], [], [], [], []
        for (first, n), p3, p4 in zip(groups, p3s, p4s):
            p5g = rows_of(p5_out, first, n, p5_h, p5_w)
            up, _, _ = nk.upsample(p5g, n, p5_h, p5_w, p5g.shape[3])
            _free(p5g)
            cat = concat([up, p4])
            _free(up, p4)
            f_p4, f_h, f_w = nk.c3k2_13(cat, n, p4_h, p4_w)
            _free(cat)
            up, _, _ = nk.upsample(f_p4, n, f_h, f_w, f_p4.shape[3])
            cat = concat([up, p3])
            _free(up, p3)
            n3, n3_h, n3_w = nk.c3k2_16(cat, n, p3_h, p3_w)
            _free(cat)
            t, h, w = nk.conv_17(n3, n, n3_h, n3_w)
            cat = concat([t, f_p4])
            _free(t, f_p4)
            n4, n4_h, n4_w = nk.c3k2_19(cat, n, f_h, f_w)
            _free(cat)
            t, h, w = nk.conv_20(n4, n, n4_h, n4_w)
            n4_downs.append(park(t))
            b, c, h3, w3 = hd.head_n3(n3, n, n3_h, n3_w)
            _free(n3)
            box3.append(park(b))
            cls3.append(park(c))
            b, c, h4, w4 = hd.head_n4(n4, n, n4_h, n4_w)
            _free(n4)
            box4.append(park(b))
            cls4.append(park(c))

        # 20x20 in groups of LOW_RES_BATCH: model.22 and its head
        n4_down = join(n4_downs)
        box5, cls5 = [], []
        for first, n in low_groups:
            parts = [rows_of(t, first, n, p5_h, p5_w) for t in (n4_down, p5_out)]
            cat = concat(parts)
            _free(*parts)
            n5, h, w = nk.c3k2_22(cat, n, p5_h, p5_w)
            _free(cat)
            b, c, h5, w5 = hd.head_n5(n5, n, h, w)
            _free(n5)
            box5.append(park(b))
            cls5.append(park(c))
        _free(n4_down, p5_out)

        return {
            "boxes": [(join(box3), h3, w3), (join(box4), h4, w4), (join(box5), h5, w5)],
            "scores": [(join(cls3), h3, w3), (join(cls4), h4, w4), (join(cls5), h5, w5)],
        }


def create_yolo26_model(
    device, variant: str = "yolo26n", num_classes: int = 80, state_dict: Optional[dict] = None
) -> TtYOLO26:
    """
    Factory function to create YOLO26 model.

    Args:
        device: TTNN device
        variant: Model variant
        num_classes: Number of classes
        state_dict: Optional pre-loaded state dictionary

    Returns:
        Initialized TtYOLO26 model
    """
    model = TtYOLO26(device, variant, num_classes)
    if state_dict is not None:
        model.load_weights_from_state_dict(state_dict)
    return model
