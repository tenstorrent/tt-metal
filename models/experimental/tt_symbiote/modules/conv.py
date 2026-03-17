# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Linear layer implementations for TTNN."""

import torch
from torch import nn
import math

import ttnn
from typing import Optional

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.activation import TTNNReLU, TTNNGelu
from models.experimental.tt_symbiote.modules.attention import TTNNSAMAttention
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.modules.tensor import TTNNPermute, TTNNReshape
from models.tt_cnn.tt.builder import Conv2dConfiguration, MaxPool2dConfiguration, TtConv2d, TtMaxPool2d
from models.experimental.tt_symbiote.modules.transformer import TTNNNoTPTransformer


class TTNNSAMLayerNorm(TTNNLayerNorm):
    """SAM-only LayerNorm: passes epsilon from torch layer (default 1e-5) for PCC match without changing shared TTNNLayerNorm."""

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        eps = getattr(self.torch_layer, "eps", 1e-5)
        tt_output = ttnn.layer_norm(
            input_tensor,
            weight=self.tt_weight,
            bias=self.tt_bias,
            epsilon=eps,
        )
        return tt_output


class NHWCLayerNorm2dWrapper(nn.Module):
    """Wraps LayerNorm2d so that when called with NHWC (from TTNN path in DPL), it permutes to NCHW and back."""

    def __init__(self, layer_norm_2d: nn.Module):
        super().__init__()
        self.layer_norm_2d = layer_norm_2d

    @property
    def weight(self) -> torch.Tensor:
        return self.layer_norm_2d.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.layer_norm_2d.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) from TTNN neck path
        x = x.permute(0, 3, 1, 2)  # -> (B, C, H, W)
        x = self.layer_norm_2d(x)
        return x.permute(0, 2, 3, 1)  # -> (B, H, W, C)


def fold_batch_norm2d_into_conv2d(weight, bias, scale, shift, running_mean, running_var, eps):
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    return weight, bias


def get_shape_from_module_name(module_name, model_config):
    """Get input shape from model config based on module name."""
    if model_config is None or not isinstance(model_config, dict) or module_name not in model_config:
        return None
    config = model_config[module_name]
    if config.get("reshape_output", False):
        return None
    return config.get("input_shapes", None)


class NHWCConvPytorch(nn.Module):
    """A wrapper around nn.Conv2d to handle NHWC input/output."""

    def __init__(
        self,
        conv: nn.Conv2d,
    ) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv2d with NHWC input/output."""
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW to NHWC
        return x


class NHWCMaxpoolPytorch(nn.Module):
    """A wrapper around nn.MaxPool2d to handle NHWC input/output."""

    def __init__(
        self,
        maxpool: nn.MaxPool2d,
    ) -> None:
        super().__init__()
        self.maxpool = maxpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MaxPool2d with NHWC input/output."""
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.maxpool(x)
        x = x.permute(0, 2, 3, 1)  # NCHW to NHWC
        return x


class NHWCUpsamplePytorch(nn.Module):
    """A wrapper around nn.Upsample to handle NHWC input/output."""

    def __init__(
        self,
        upsample: nn.Upsample,
    ) -> None:
        super().__init__()
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Upsample with NHWC input/output."""
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)  # NCHW to NHWC
        return x


class NHWCConvBNPytorch(nn.Module):
    """A wrapper around nn.Conv2d to handle NHWC input/output."""

    def __init__(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv2d with NHWC input/output."""
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.conv(x)
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)  # NCHW to NHWC
        return x


class NHWCConvBNActivationPytorch(nn.Module):
    """A wrapper around nn.Conv2d to handle NHWC input/output."""

    def __init__(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv2d with NHWC input/output."""
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1)  # NCHW to NHWC
        return x


class TTNNConv2dNHWC(TTNNModule):
    """TTNN-accelerated Conv layer."""

    CACHED_TTCNN = {}

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups: int = 1,
        slice_config=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.slice_config = slice_config
        self.reshape = TTNNReshape()

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, slice_config=None) -> "TTNNConv2d":
        """Create TTNNConv2d from PyTorch Conv2d layer."""
        new_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            slice_config=slice_config,
        )
        new_conv._fallback_torch_layer = NHWCConvPytorch(conv)
        return new_conv

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = NHWCConvPytorch(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
        self.tt_weight, self.tt_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            self.torch_layer.conv.weight, self.torch_layer.conv.bias
        )
        super().preprocess_weights_impl()

    def deallocate_weights_impl(self):
        """Deallocate weights from device."""
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        input_shape = get_shape_from_module_name(self.module_name, self.model_config)
        if input_shape is None:
            batch_size, input_height, input_width, _ = input_tensor.shape
            if isinstance(self.model_config, dict):
                self.model_config[self.module_name] = {
                    "input_shapes": [list(input_tensor.shape)],
                    "reshape_output": reshape_output,
                }
        else:
            assert len(input_shape) == 1, f"Only single input shape is supported. Got {input_shape}."
            batch_size, input_height, input_width, _ = input_shape[0]
            reshape_output = self.model_config[self.module_name].get("reshape_output", reshape_output)

        hash = (
            input_height,
            input_width,
            self.in_channels,
            self.out_channels,
            batch_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.groups,
            self.dilation,
            self.tt_weight,
            self.tt_bias,
            self.slice_config,
        )
        if hash in TTNNConv2dNHWC.CACHED_TTCNN:
            layer = TTNNConv2dNHWC.CACHED_TTCNN[hash]
        else:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
                dilation=self.dilation,
                weight=self.tt_weight,
                bias=self.tt_bias,
                slice_strategy=self.slice_config,
            )
            layer = TtConv2d(config, input_tensor.device())
            TTNNConv2dNHWC.CACHED_TTCNN[hash] = layer
        if reshape_output:
            out, h_w = layer(input_tensor, return_output_dim=reshape_output)
            out = self.reshape(out, [batch_size, h_w[0], h_w[1], -1])
            return out
        if input_tensor.memory_config().is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor)
        return layer(input_tensor)


class TTNNConv2dBNNHWC(TTNNConv2dNHWC):
    """TTNN-accelerated Fused Conv BN layer."""

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, bn: nn.BatchNorm2d, slice_config=None) -> "TTNNConv2d":
        """Create TTNNConv2d from PyTorch Conv2d layer."""
        new_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            slice_config=slice_config,
        )
        new_conv._fallback_torch_layer = NHWCConvBNPytorch(conv, bn)
        return new_conv

    def _preprocess_weights_local(self):
        torch_weight, torch_bias = fold_batch_norm2d_into_conv2d(
            self.torch_layer.conv.weight,
            self.torch_layer.conv.bias,
            self.torch_layer.bn.weight,
            self.torch_layer.bn.bias,
            self.torch_layer.bn.running_mean,
            self.torch_layer.bn.running_var,
            self.torch_layer.bn.eps,
        )
        self.tt_weight, self.tt_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            torch_weight, torch_bias
        )

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = NHWCConvBNPytorch(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                ),
                nn.BatchNorm2d(self.out_channels),
            )
        self._preprocess_weights_local()
        # call method from TTNNConv2dNHWC's grandparent TTNNModule
        TTNNModule.preprocess_weights_impl(self)


class TTNNConv2dBNActivationNHWC(TTNNConv2dBNNHWC):
    """TTNN-accelerated Fused Conv BN layer."""

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, bn: nn.BatchNorm2d, activation, slice_config=None) -> "TTNNConv2d":
        """Create TTNNConv2d from PyTorch Conv2d layer."""
        new_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            slice_config=slice_config,
        )
        new_conv._fallback_torch_layer = NHWCConvBNActivationPytorch(conv, bn, nn.ReLU())
        assert isinstance(activation, nn.ReLU), "Only ReLU activation is supported in TTNNConv2dBNActivationNHWC."
        return new_conv

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = NHWCConvBNActivationPytorch(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(),
            )
        self._preprocess_weights_local()
        # call method from TTNNConv2dNHWC's grandparent TTNNModule
        TTNNModule.preprocess_weights_impl(self)

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        batch_size, input_height, input_width, _ = input_tensor.shape
        config = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
            weight=self.tt_weight,
            bias=self.tt_bias,
            slice_strategy=self.slice_config,
            activation=ttnn.UnaryOpType.RELU,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
        )
        layer = TtConv2d(config, input_tensor.device())
        if reshape_output:
            out, h_w = layer(input_tensor, return_output_dim=reshape_output)
            out = self.reshape(out, [batch_size, h_w[0], h_w[1], -1])
            return out
        return layer(input_tensor)


class TTNNBottleneck(TTNNModule):
    """TTNN-accelerated ResNet Bottleneck block."""

    def __init__(
        self,
        downsample=None,
    ) -> None:
        super().__init__()
        self.downsample = downsample

    def initilize_submodules(self):
        assert (
            self._fallback_torch_layer is not None
        ), "Fallback torch layer must be set before initializing submodules."
        assert isinstance(self.torch_layer.bn1, nn.BatchNorm2d), "Only BatchNorm2d is supported in Bottleneck blocks."
        self.conv1 = TTNNConv2dBNActivationNHWC.from_torch(
            self.torch_layer.conv1, self.torch_layer.bn1, self.torch_layer.relu
        )
        self.conv2 = TTNNConv2dBNActivationNHWC.from_torch(
            self.torch_layer.conv2, self.torch_layer.bn2, self.torch_layer.relu
        )
        self.conv3 = TTNNConv2dBNNHWC.from_torch(self.torch_layer.conv3, self.torch_layer.bn3)
        self.relu = TTNNReLU()
        self.permute = TTNNPermute()

    @classmethod
    def from_torch(cls, bottleneck: "torchvision.models.resnet.Bottleneck") -> "TTNNBottleneck":
        """Create TTNNBottleneck from PyTorch Bottleneck layer."""
        new_bottleneck = cls(
            downsample=bottleneck.downsample,
        )
        new_bottleneck._fallback_torch_layer = bottleneck
        new_bottleneck.initilize_submodules()
        return new_bottleneck

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through Bottleneck block."""
        if self.downsample is not None:
            identity = x
            x = self.permute(x, perm=[0, 2, 3, 1])
        else:
            x = self.permute(x, perm=[0, 2, 3, 1])
            identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            identity = self.downsample(TorchTTNNTensor(identity, dtype=torch.bfloat16))
            if identity.to_ttnn.device() != out.to_ttnn.device():
                identity = ttnn.to_device(identity.to_ttnn, out.to_ttnn.device())

            identity = self.permute(identity, perm=[0, 2, 3, 1])
        out = out + identity
        out = self.relu(out)
        out = self.permute(out, perm=[0, 3, 1, 2])
        return out


class TTNNSAMMLPBlock(TTNNModule):
    """TTNN version of SAM MLPBlock (lin1 -> act -> lin2). Input/output shape (B, H, W, C)."""

    def __init__(self):
        super().__init__()

    def initialize_submodules(self):
        assert (
            self._fallback_torch_layer is not None
        ), "Fallback torch layer must be set before initializing submodules."
        self.lin1 = TTNNLinear.from_torch(self.torch_layer.lin1)
        self.lin2 = TTNNLinear.from_torch(self.torch_layer.lin2)
        self.act = TTNNGelu()

    @classmethod
    def from_torch(cls, mlp_block: "nn.Module") -> "TTNNSAMMLPBlock":
        """Create from SAM Block.mlp (MLPBlock)."""
        new_mlp = cls()
        new_mlp._fallback_torch_layer = mlp_block
        new_mlp.initialize_submodules()
        return new_mlp

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward: lin2(act(lin1(x)))."""
        out = self.lin1(x)
        out = out.ttnn_tensor if hasattr(out, "ttnn_tensor") else out
        out = self.act(out)
        out = self.lin2(out)
        return out.ttnn_tensor if hasattr(out, "ttnn_tensor") else out


class TTNNSAMBlock(TTNNModule):
    """TTNN version of SAM Block (deepencoder.py).
    Residual: norm1 -> attn -> +shortcut; norm2 -> mlp -> +shortcut.
    attn is TTNNSAMAttention (single class; window_size on attn controls global vs windowed).
    """

    def __init__(self, window_size: int = 0):
        super().__init__()
        self.window_size = window_size

    def initialize_submodules(self):
        assert (
            self._fallback_torch_layer is not None
        ), "Fallback torch layer must be set before initializing submodules."
        blk = self.torch_layer
        self.norm1 = TTNNSAMLayerNorm.from_torch(blk.norm1)
        self.attn = TTNNSAMAttention.from_torch(blk.attn, window_size=self.window_size)
        self.norm2 = TTNNSAMLayerNorm.from_torch(blk.norm2)
        self.mlp = TTNNSAMMLPBlock.from_torch(blk.mlp)

    @classmethod
    def from_torch(cls, block: "nn.Module", window_size: int = 0) -> "TTNNSAMBlock":
        """Create TTNNSAMBlock from SAM Block (blocks[i])."""
        new_block = cls(window_size=window_size)
        new_block._fallback_torch_layer = block
        new_block.initialize_submodules()
        return new_block

    def forward(self, x):
        """x: (B, H, W, C). Output: (B, H, W, C)."""
        if hasattr(x, "to_ttnn"):
            x = x.to_ttnn
        elif isinstance(x, torch.Tensor):
            x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        shortcut = x
        x = self.norm1(x)
        x = x.ttnn_tensor if hasattr(x, "ttnn_tensor") else x
        attn_out = self.attn(x)
        attn_out = attn_out.ttnn_tensor if hasattr(attn_out, "ttnn_tensor") else attn_out
        ttnn.deallocate(x)
        x = ttnn.add(
            shortcut.to_ttnn if hasattr(shortcut, "to_ttnn") else shortcut,
            attn_out,
        )
        ttnn.deallocate(attn_out)
        shortcut = x
        x = self.norm2(x)
        x = x.ttnn_tensor if hasattr(x, "ttnn_tensor") else x
        x = self.mlp(x)
        x = x.ttnn_tensor if hasattr(x, "ttnn_tensor") else x
        x = ttnn.add(
            shortcut.to_ttnn if hasattr(shortcut, "to_ttnn") else shortcut,
            x,
        )
        ttnn.deallocate(shortcut)
        return x


class TTNNImageEncoderViT(TTNNModule):
    """TTNN SAM ImageEncoderViT: patch_embed, pos_embed, blocks, neck, net_2, net_3.
    Uses TTNNConv2dNHWC for patch_embed (SAM PatchEmbed is just .proj Conv2d), TTNNSAMBlock, TTNN Conv/LayerNorm for neck.
    Input NCHW, output BCHW.
    """

    def __init__(self, depth: int, window_size: int = 0):
        super().__init__()
        self.depth = depth
        self.window_size = window_size

    def initialize_submodules(self):
        assert (
            self._fallback_torch_layer is not None
        ), "Fallback torch layer must be set before initializing submodules."
        enc = self.torch_layer
        # SAM patch_embed is PatchEmbed(proj=Conv2d); reuse existing TTNNConv2dNHWC
        self.patch_embed = TTNNConv2dNHWC.from_torch(enc.patch_embed.proj)
        # Use list (not nn.ModuleList): TTNNSAMBlock is TTNNModule, not nn.Module.
        # Per-block window_size to match ref (e.g. SAM: 14 for most blocks, 0 for global-attn blocks 2,5,8,11).
        self.blocks = []
        for i in range(self.depth):
            win_sz = getattr(enc.blocks[i], "window_size", 0)
            self.blocks.append(TTNNSAMBlock.from_torch(enc.blocks[i], window_size=win_sz))
        # neck: Sequential(Conv2d, LayerNorm2d, Conv2d, LayerNorm2d)
        neck_list = list(enc.neck.children())
        self.neck_conv1 = TTNNConv2dNHWC.from_torch(neck_list[0])
        self.neck_ln1 = TTNNSAMLayerNorm.from_torch(neck_list[1])
        self.neck_ln1._fallback_torch_layer = NHWCLayerNorm2dWrapper(neck_list[1])  # DPL: neck receives NHWC
        self.neck_conv2 = TTNNConv2dNHWC.from_torch(neck_list[2])
        self.neck_ln2 = TTNNSAMLayerNorm.from_torch(neck_list[3])
        self.neck_ln2._fallback_torch_layer = NHWCLayerNorm2dWrapper(neck_list[3])  # DPL: neck receives NHWC
        self.net_2 = TTNNConv2dNHWC.from_torch(enc.net_2)
        self.net_3 = TTNNConv2dNHWC.from_torch(enc.net_3)

    @classmethod
    def from_torch(cls, encoder: "nn.Module", window_size: int = 0) -> "TTNNImageEncoderViT":
        """Create from SAM ImageEncoderViT (encoder); uses full depth."""
        depth = len(encoder.blocks)
        new_enc = cls(depth=depth, window_size=window_size)
        new_enc._fallback_torch_layer = encoder
        new_enc.initialize_submodules()
        return new_enc

    def preprocess_weights_impl(self):
        """Recurse into blocks (ModuleList) and other children."""
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.preprocess_weights()
        for blk in self.blocks:
            if isinstance(blk, TTNNModule):
                blk.preprocess_weights()
        return self

    def move_weights_to_device_impl(self):
        """Recurse into blocks (ModuleList) and other children."""
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.move_weights_to_device()
        for blk in self.blocks:
            if isinstance(blk, TTNNModule):
                blk.move_weights_to_device()
        return self

    def forward(self, x):
        """x: NCHW. Output: BCHW (same as ref)."""
        if hasattr(x, "to_ttnn"):
            x = x.to_ttnn
        elif isinstance(x, torch.Tensor):
            x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Patch embed: NCHW -> NHWC then TTNNConv2dNHWC (reuse existing conv)
        x = ttnn.permute(x, (0, 2, 3, 1))
        x = self.patch_embed(x)
        x = x.to_ttnn if hasattr(x, "to_ttnn") else x

        # Pos embed: add on host if needed (get_abs_pos_sam) then add on device
        if self.torch_layer.pos_embed is not None:
            from torch.nn import functional as F

            B, H, W, C = x.shape
            pos = self.torch_layer.pos_embed
            src_size = pos.shape[1]
            if src_size != H:
                pos_nchw = pos.permute(0, 3, 1, 2).float()
                pos_resized = F.interpolate(
                    pos_nchw, size=(H, W), mode="bicubic", antialias=True, align_corners=False
                ).to(pos.dtype)
                pos = pos_resized.permute(0, 2, 3, 1)
            else:
                pos = pos
            pos_tt = ttnn.from_torch(
                pos.expand(B, -1, -1, -1), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            x = ttnn.add(x, pos_tt)

        for blk in self.blocks:
            x = blk(x)
            x = x.to_ttnn if hasattr(x, "to_ttnn") else x

        # Neck: BHWC -> conv1 -> ln1 -> conv2 -> ln2
        x = self.neck_conv1(x)
        x = x.to_ttnn if hasattr(x, "to_ttnn") else x
        x = self.neck_ln1(x)
        x = x.to_ttnn if hasattr(x, "to_ttnn") else x
        x = self.neck_conv2(x)
        x = x.to_ttnn if hasattr(x, "to_ttnn") else x
        x = self.neck_ln2(x)
        x = x.to_ttnn if hasattr(x, "to_ttnn") else x

        x = self.net_2(x)
        x = x.to_ttnn if hasattr(x, "to_ttnn") else x
        x = self.net_3(x)
        x = x.to_ttnn if hasattr(x, "to_ttnn") else x

        # B H W C -> B C H W
        x = ttnn.permute(x, (0, 3, 1, 2))
        return x


class TorchPatchEmbeddings(nn.Module):
    """A wrapper around nn.Conv2d to handle ViT Patch Embeddings."""

    def __init__(
        self,
        patch_embeddings,
    ) -> None:
        super().__init__()
        self.patch_embeddings = patch_embeddings

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through Conv2d with NHWC input/output."""
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        x = self.patch_embeddings(x[:, :3, :, :], **kwargs)  # Use only first 3 channels
        return x


class TTNNPatchEmbedding(TTNNModule):
    """TTNN-accelerated Patch Embedding layer for ViT."""

    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.permute = TTNNPermute()

    @classmethod
    def from_torch(cls, patch_embedding: "ViTPatchEmbeddings") -> "TTNNPatchEmbedding":
        """Create TTNNPatchEmbedding from PyTorch Conv2d layer."""
        new_patch_embedding = cls(
            img_size=patch_embedding.projection.kernel_size[0] * patch_embedding.projection.stride[0],
            patch_size=patch_embedding.projection.kernel_size[0],
            in_channels=patch_embedding.projection.in_channels,
            embed_dim=patch_embedding.projection.out_channels,
        )
        new_patch_embedding.projection = patch_embedding.projection
        new_patch_embedding._fallback_torch_layer = TorchPatchEmbeddings(patch_embedding)
        return new_patch_embedding

    def preprocess_weights_impl(self):
        weight = self.projection.weight
        bias = self.projection.bias

        three_times_hidden_size, c, _, _ = weight.shape
        pad_value = 4 - c
        preprocessed_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
        preprocessed_weight = torch.reshape(
            preprocessed_weight, (int(three_times_hidden_size * (4 / c)), three_times_hidden_size)
        )

        self.ttnn_weight = ttnn.from_torch(preprocessed_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        self.ttnn_bias = ttnn.from_torch(bias.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.ttnn_weight = ttnn.to_device(self.ttnn_weight, self.device)
        self.ttnn_bias = ttnn.to_device(self.ttnn_bias, self.device)
        super().move_weights_to_device_impl()

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
        patch_size = self.patch_size
        patch_count = img_h // patch_size  # 14
        patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
        patch_count_all = int(patch_count * patch_count)  # 196
        stride_h = patch_size
        stride_w = 1
        pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
        folded_pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)  # 1568, 1024
        ttnn.deallocate(pixel_values)
        folded_pixel_values = ttnn.to_memory_config(folded_pixel_values, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Convert back to interleaved or otherwise to_layout will fail
        folded_pixel_values = ttnn.to_layout(folded_pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        patch_embedding_output = ttnn.linear(
            folded_pixel_values,
            self.ttnn_weight,
            bias=self.ttnn_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
        patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, patch_size_sq_trpl))

        return patch_embedding_output


class TorchVitEmbeddings(nn.Module):
    """A wrapper around nn.Conv2d to handle ViT Patch Embeddings."""

    def __init__(
        self,
        patch_embeddings,
        cls_token,
        position_embeddings,
    ) -> None:
        super().__init__()
        self.patch_embeddings = TorchPatchEmbeddings(patch_embeddings)
        self.cls_token = cls_token
        self.position_embeddings = position_embeddings

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through Conv2d with NHWC input/output."""
        batch_size, height, width, _ = x.shape
        embeddings = self.patch_embeddings(x, **kwargs)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings
        return embeddings


class TTNNViTEmbeddings(TTNNModule):
    """TTNN-accelerated ViT Embeddings layer."""

    @classmethod
    def from_torch(cls, patch_embeddings: "ViTPatchEmbeddings", cls_token, position_embeddings) -> "TTNNViTEmbeddings":
        """Create TTNNViTEmbeddings from PyTorch ViTEmbeddings layer."""
        new_embeddings = cls()
        new_embeddings.patch_embeddings = TTNNPatchEmbedding.from_torch(patch_embeddings)
        new_embeddings.cls_token = ttnn.from_torch(cls_token, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        new_embeddings.position_embeddings = ttnn.from_torch(
            position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        new_embeddings._fallback_torch_layer = TorchVitEmbeddings(patch_embeddings, cls_token, position_embeddings)
        return new_embeddings

    def preprocess_weights_impl(self):
        """Preprocess weights for TTNN."""
        self.patch_embeddings.preprocess_weights()
        self.cls_token = ttnn.to_device(self.cls_token, self.device)
        self.position_embeddings = ttnn.to_device(self.position_embeddings, self.device)
        super().preprocess_weights_impl()

    def forward(self, pixel_values, **kwargs):
        patch_embedding_output = self.patch_embeddings(pixel_values, **kwargs)
        batch = pixel_values.shape[0]
        # expand the cls token to the batch size
        patch_embedding_output = patch_embedding_output.to_ttnn
        if patch_embedding_output.layout != ttnn.TILE_LAYOUT:
            patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.TILE_LAYOUT)
        # add the [CLS] token to the embedded patch tokens
        cls_token = ttnn.reshape(self.cls_token, [1, 1, patch_embedding_output.shape[-1]])
        if batch > 1:
            cls_token = ttnn.repeat(cls_token, [batch, 1, 1])
        embedding_output = ttnn.concat([cls_token, patch_embedding_output], 1, memory_config=ttnn.L1_MEMORY_CONFIG)
        embedding_output = ttnn.to_layout(embedding_output, layout=ttnn.TILE_LAYOUT)
        embedding_output = ttnn.add(
            embedding_output, self.position_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        return embedding_output


class TTNNMaxPool2dNHWC(TTNNModule):
    """TTNN-accelerated MaxPool2d layer."""

    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        dilation,
        slice_config=None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.slice_config = slice_config
        self.reshape = TTNNReshape()

    @classmethod
    def from_torch(cls, maxpool: nn.MaxPool2d, slice_config=None) -> "TTNNMaxPool2dNHWC":
        """Create TTNNMaxPool2dNHWC from PyTorch MaxPool2d layer."""
        new_maxpool = cls(
            kernel_size=maxpool.kernel_size,
            stride=maxpool.stride,
            padding=maxpool.padding,
            dilation=maxpool.dilation,
            slice_config=slice_config,
        )
        assert isinstance(new_maxpool.kernel_size, int), "Only integer kernel_size is supported in TTNNMaxPool2dNHWC."
        assert isinstance(new_maxpool.stride, int), "Only integer stride is supported in TTNNMaxPool2dNHWC."
        assert isinstance(new_maxpool.padding, int), "Only integer padding is supported in TTNNMaxPool2dNHWC."
        assert isinstance(new_maxpool.dilation, int), "Only integer dilation is supported in TTNNMaxPool2dNHWC."
        new_maxpool._fallback_torch_layer = NHWCMaxpoolPytorch(maxpool)
        return new_maxpool

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        input_shape = get_shape_from_module_name(self.module_name, self.model_config)
        if input_shape is None:
            batch_size, input_height, input_width, channels = input_tensor.shape
            if isinstance(self.model_config, dict):
                self.model_config[self.module_name] = {
                    "input_shapes": [list(input_tensor.shape)],
                    "reshape_output": reshape_output,
                }
        else:
            assert len(input_shape) == 1, f"Only single input shape is supported. Got {input_shape}."
            batch_size, input_height, input_width, channels = input_shape[0]
            reshape_output = self.model_config[self.module_name].get("reshape_output", reshape_output)
        config = MaxPool2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            channels=channels,
            batch_size=batch_size,
            kernel_size=[self.kernel_size, self.kernel_size],
            stride=[self.stride, self.stride],
            padding=[self.padding, self.padding],
            dilation=[self.dilation, self.dilation],
            slice_strategy=self.slice_config,
        )
        output_h = (input_height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output_w = (input_width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        h_w = (output_h, output_w)
        layer = TtMaxPool2d(config, input_tensor.device())
        if reshape_output:
            out = layer(input_tensor)
            out = self.reshape(out, [batch_size, h_w[0], h_w[1], -1])
            return out
        return layer(input_tensor)


class TTNNUpsampleNHWC(TTNNModule):
    """TTNN-accelerated Upsample layer."""

    def __init__(
        self,
        scale_factor,
        mode="nearest",
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    @classmethod
    def from_torch(cls, upsample: nn.Upsample) -> "TTNNUpsampleNHWC":
        """Create TTNNUpsampleNHWC from PyTorch Upsample layer."""
        new_upsample = cls(
            scale_factor=upsample.scale_factor,
            mode=upsample.mode,
        )
        assert upsample.mode in [
            "nearest",
            "bilinear",
        ], "Only 'nearest' and 'bilinear' modes are supported in TTNNUpsampleNHWC."
        new_upsample._fallback_torch_layer = NHWCUpsamplePytorch(upsample)
        return new_upsample

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through Upsample layer."""
        batch_size, input_height, input_width, channels = input_tensor.shape
        output_height = input_height * self.scale_factor
        output_width = input_width * self.scale_factor
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.upsample(
            input_tensor,
            scale_factor=int(self.scale_factor),
            mode=self.mode,
        )
        return input_tensor


class TTNNConv2dNHWCInputMultipleOf16(TTNNConv2dNHWC):
    """TTNN-accelerated Conv InputMultipleOf16 layer."""

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, slice_config=None) -> "TTNNConv2dNHWCInputMultipleOf16":
        """Create TTNNConv2dNHWCInputMultipleOf16 from PyTorch Conv2d layer."""
        if conv.in_channels > 16 or conv.in_channels % 16 == 0:
            return TTNNConv2dNHWC.from_torch(conv, slice_config)
        new_conv = cls(
            in_channels=16,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            slice_config=slice_config,
        )
        conv.weight = nn.Parameter(
            torch.nn.functional.pad(conv.weight, (0, 0, 0, 0, 0, (16 - conv.in_channels % 16) % 16))
        )
        new_conv._fallback_torch_layer = NHWCConvPytorch(conv)
        return new_conv


class TTNNClipVisionEmbeddings(TTNNModule):
    """
    CLIP Vision Embeddings using TTNN operations.

    Converts image patches to embeddings with class token and positional embeddings.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
    ):
        """
        Initialize CLIP vision embeddings.

        Args:
            hidden_size: Embedding dimension
            image_size: Input image size
            patch_size: Patch size
            num_channels: Number of input channels
            weights: PyTorch weights dict (optional, for loading pretrained)
            device: TTNN device
        """

        super().__init__()

        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.torch_layer_cp = None

    @classmethod
    def from_torch(cls, visionEmbedding):
        """Create TTNN module from PyTorch equivalent."""
        new_clip = cls(
            hidden_size=visionEmbedding.embed_dim,
            image_size=visionEmbedding.image_size,
            patch_size=visionEmbedding.patch_size,
            num_channels=3,
        )

        new_clip.torch_layer_cp = visionEmbedding
        new_clip._fallback_torch_layer = visionEmbedding
        return new_clip

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        # Load from pretrained weights
        self.class_embedding = ttnn.from_torch(
            self.torch_layer_cp.class_embedding.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Patch embedding: Conv2d weight (out_channels, in_channels, kernel_h, kernel_w)
        conv_weight = self.torch_layer_cp.patch_embedding.weight.data  # (hidden_size, 3, patch_size, patch_size)

        conv_bias = None
        if self.torch_layer_cp.patch_embedding.bias is not None:
            conv_bias = self.torch_layer_cp.patch_embedding.bias.data

        # Convert Conv2d to linear format for TTNN
        # Flatten kernel: (hidden_size, 3, patch_size, patch_size) -> (hidden_size, 3*patch_size*patch_size)
        linear_weight = conv_weight.view(self.embed_dim, -1)  # (hidden_size, 3*patch_size*patch_size)
        linear_weight = linear_weight.T  # (3*patch_size*patch_size, hidden_size) for TTNN linear

        self.patch_embedding_weight = ttnn.from_torch(
            linear_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if conv_bias is not None:
            self.patch_embedding_bias = self.tensor_1d_to_2d_ttnn(conv_bias)
        else:
            self.patch_embedding_bias = None

        # Position embedding - shape (num_positions, embed_dim)
        position_embedding_weight = self.torch_layer_cp.position_embedding.weight.data
        # Reshape to (1, num_positions, embed_dim) for get_abs_pos_ttnn
        position_embedding_reshaped = position_embedding_weight.unsqueeze(0)
        self.position_embedding = ttnn.from_torch(
            position_embedding_reshaped,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.class_embedding = ttnn.to_device(self.class_embedding, self.device)
        self.patch_embedding_weight = ttnn.to_device(self.patch_embedding_weight, self.device)
        self.position_embedding = ttnn.to_device(self.position_embedding, self.device)
        if self.patch_embedding_bias is not None:
            self.patch_embedding_bias = ttnn.to_device(self.patch_embedding_bias, self.device)

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        ttnn.deallocate(self.class_embedding)
        ttnn.deallocate(self.patch_embedding_weight)
        ttnn.deallocate(self.position_embedding)
        if self.patch_embedding_bias is not None:
            ttnn.deallocate(self.patch_embedding_bias)

    def tensor_1d_to_2d_ttnn(self, tensor_1d: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
        """
        Convert 1D PyTorch tensor to 2D TTNN tensor (1, N) for bias operations.

        Args:
            tensor_1d: 1D PyTorch tensor
            device: TTNN device
            dtype: TTNN data type

        Returns:
            2D TTNN tensor of shape (1, N)
        """
        tensor_2d = tensor_1d.unsqueeze(0)
        return ttnn.from_torch(
            tensor_2d,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _unfold_patches(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        """
        Extract patches from image using TTNN operations.

        Args:
            pixel_values: TTNN tensor (batch_size, channels, height, width)

        Returns:
            TTNN tensor (batch_size, num_patches, patch_size * patch_size * channels)
        """
        batch_size = pixel_values.shape[0]
        img_h = pixel_values.shape[2]
        img_w = pixel_values.shape[3]

        patches_h = img_h // self.patch_size
        patches_w = img_w // self.patch_size

        # Reshape to extract patches: (B, C, H, W) -> (B, C, patches_h, patch_size, patches_w, patch_size)
        pixel_values = ttnn.reshape(
            pixel_values, (batch_size, self.num_channels, patches_h, self.patch_size, patches_w, self.patch_size)
        )

        # Permute to group patches: (B, patches_h, patches_w, patch_size, patch_size, C)
        pixel_values = ttnn.permute(pixel_values, (0, 2, 4, 1, 3, 5))

        # Flatten patches: (B, patches_h, patches_w, patch_size * patch_size * C)
        pixel_values = ttnn.reshape(
            pixel_values, (batch_size, patches_h * patches_w, self.patch_size * self.patch_size * self.num_channels)
        )

        return pixel_values

    def get_abs_pos_ttnn(
        self,
        abs_pos: ttnn.Tensor,
        tgt_size: int,
        device: ttnn.Device,
    ) -> ttnn.Tensor:
        """
        Get absolute positional embeddings, interpolating if needed.

        Args:
            abs_pos: TTNN tensor of shape (1, L, C) with positional embeddings
            tgt_size: Target sequence size (excluding CLS token)
            device: TTNN device

        Returns:
            TTNN tensor of shape (1, tgt_size + 1, C) with interpolated positional embeddings
        """
        # Convert to torch for interpolation (TTNN doesn't have bicubic interpolation)
        abs_pos_torch = ttnn.to_torch(abs_pos)

        # Extract CLS token and position embeddings
        cls_token = abs_pos_torch[:, :1, :]  # (1, 1, C)
        old_pos_embed = abs_pos_torch[:, 1:, :]  # (1, L-1, C)

        src_size = int(math.sqrt(old_pos_embed.shape[1]))
        tgt_size_sqrt = int(math.sqrt(tgt_size))

        if src_size != tgt_size_sqrt:
            # Reshape for interpolation: (1, L-1, C) -> (1, C, src_size, src_size)
            old_pos_embed_2d = old_pos_embed.view(1, src_size, src_size, -1).permute(0, 3, 1, 2).contiguous()
            old_pos_embed_2d = old_pos_embed_2d.to(torch.float32)

            # Interpolate using PyTorch
            new_pos_embed_2d = torch.nn.functional.interpolate(
                old_pos_embed_2d,
                size=(tgt_size_sqrt, tgt_size_sqrt),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            ).to(old_pos_embed.dtype)

            # Reshape back: (1, C, tgt_size, tgt_size) -> (1, tgt_size, C)
            new_pos_embed = new_pos_embed_2d.permute(0, 2, 3, 1).contiguous()
            new_pos_embed = new_pos_embed.view(1, tgt_size, -1)

            # Concatenate CLS token
            vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=1)  # (1, tgt_size + 1, C)
        else:
            vision_pos_embed = abs_pos_torch

        # Convert back to TTNN
        return ttnn.from_torch(
            vision_pos_embed,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, pixel_values: ttnn.Tensor, patch_embeds: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """
        Forward pass of CLIP vision embeddings.

        Args:
            pixel_values: TTNN tensor (batch_size, channels, height, width)
            patch_embeds: Optional pre-computed patch embeddings (batch_size, num_patches, embed_dim)

        Returns:
            TTNN tensor (batch_size, num_patches + 1, embed_dim)
        """

        if pixel_values.layout != ttnn.TILE_LAYOUT:
            pixel_values = ttnn.to_layout(pixel_values, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if patch_embeds is not None and patch_embeds.layout != ttnn.TILE_LAYOUT:
            patch_embeds = ttnn.to_layout(patch_embeds, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        batch_size = pixel_values.shape[0]

        # Get patch embeddings
        if patch_embeds is not None:
            patch_embeds = patch_embeds
        else:
            # Extract patches
            patches = self._unfold_patches(pixel_values)

            # Apply linear projection
            patch_embeds = ttnn.linear(
                patches,
                self.patch_embedding_weight,
                bias=self.patch_embedding_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(patches)

        # Expand class embedding: (embed_dim) -> (batch_size, 1, embed_dim)
        class_embeds = ttnn.reshape(self.class_embedding, (1, 1, self.embed_dim))
        class_embeds = ttnn.repeat(class_embeds, (batch_size, 1, 1))

        # Concatenate class token and patch embeddings
        embeddings = ttnn.concat([class_embeds, patch_embeds], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(class_embeds)
        # Note: Don't deallocate patch_embeds here - it's either passed in (user's responsibility)
        # or was just created and will be used in the concat, so it's part of embeddings now

        # Get position embeddings (with interpolation if needed)
        # Position embedding is already in shape (1, num_positions, embed_dim)
        # We need to interpolate if sequence length doesn't match
        # Note: embeddings.size(1) is the actual sequence length (num_patches + 1)
        # but get_abs_pos_ttnn expects the number of patches (excluding CLS token)
        actual_seq_len = embeddings.shape[1]  # This is num_patches + 1
        num_patches_actual = actual_seq_len - 1  # Exclude CLS token
        pos_embeds = self.get_abs_pos_ttnn(
            self.position_embedding,
            num_patches_actual,
            self.device,
        )

        # Add position embeddings
        embeddings = ttnn.add(embeddings, pos_embeds, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pos_embeds)

        return embeddings


class TTNNVitModel(TTNNModule):
    """
    Vision Transformer Model using TTNN operations.

    Complete ViT model with embeddings, pre-norm, and transformer encoder.
    """

    def __init__(
        self,
    ):
        """
        Initialize ViT model.

        Args:
            cfg: Configuration dict
            weights: PyTorch weights dict (optional)
            device: TTNN device
            freeze_embed: Whether to freeze embedding weights
            freeze_pre_norm: Whether to freeze pre-norm weights
        """
        super().__init__()
        self.torch_layer_cp = None
        self.embeddings = None
        self.transformer = None
        self.pre_layernorm_epsilon = 1e-5

    @classmethod
    def from_torch(cls, VitModel):
        """Create TTNN module from PyTorch equivalent."""
        new_VitModel = cls()

        new_VitModel.embeddings = TTNNClipVisionEmbeddings.from_torch(VitModel.embeddings)
        new_VitModel.transformer = TTNNNoTPTransformer.from_torch(VitModel.transformer)

        new_VitModel.torch_layer_cp = VitModel
        new_VitModel._fallback_torch_layer = VitModel
        return new_VitModel

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        pre_norm_weight = self.torch_layer_cp.pre_layrnorm.weight.data
        pre_norm_bias = self.torch_layer_cp.pre_layrnorm.bias.data

        self.pre_layrnorm_weight = ttnn.from_torch(
            pre_norm_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.pre_layrnorm_bias = ttnn.from_torch(
            pre_norm_bias,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.embeddings.preprocess_weights_impl()
        self.transformer.preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.pre_layrnorm_weight = ttnn.to_device(self.pre_layrnorm_weight, self.device)
        self.pre_layrnorm_bias = ttnn.to_device(self.pre_layrnorm_bias, self.device)
        self.embeddings.move_weights_to_device_impl()
        self.transformer.move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        ttnn.deallocate(self.pre_layrnorm_weight)
        ttnn.deallocate(self.pre_layrnorm_bias)
        self.embeddings.deallocate_weights_impl()
        self.transformer.deallocate_weights_impl()

    def forward(
        self,
        x: ttnn.Tensor,
        patch_embeds: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass of ViT model.

        Args:
            x: TTNN tensor (batch_size, channels, height, width) - input image
            patch_embeds: Optional pre-computed patch embeddings

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        if isinstance(x, torch.Tensor):
            x = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        if isinstance(patch_embeds, torch.Tensor):
            patch_embeds = ttnn.from_torch(
                patch_embeds,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if patch_embeds is not None and patch_embeds.layout != ttnn.TILE_LAYOUT:
            patch_embeds = ttnn.to_layout(patch_embeds, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if patch_embeds is not None and len(patch_embeds.shape) == 4:
            patch_embeds = ttnn.reshape(patch_embeds, shape=[patch_embeds.shape[0], patch_embeds.shape[1], -1])
            patch_embeds = ttnn.transpose(patch_embeds, 1, 2)

        # Embeddings
        x = self.embeddings.forward(x, patch_embeds)

        # Pre-layer norm
        hidden_states = ttnn.layer_norm(
            x,
            weight=self.pre_layrnorm_weight,
            bias=self.pre_layrnorm_bias,
            epsilon=self.pre_layernorm_epsilon,
        )
        ttnn.deallocate(x)

        # Transformer
        output = self.transformer.forward(hidden_states)
        ttnn.deallocate(hidden_states)

        return output
