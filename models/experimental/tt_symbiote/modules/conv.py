# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Linear layer implementations for TTNN."""

import torch
from torch import nn

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, MaxPool2dConfiguration, TtConv2d, TtMaxPool2d
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.activation import TTNNReLU
from models.experimental.tt_symbiote.modules.tensor import TTNNPermute, TTNNReshape


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
        new_conv = TTNNConv2dNHWC(
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
        new_conv = TTNNConv2dBNNHWC(
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
        new_conv = TTNNConv2dBNActivationNHWC(
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
        new_bottleneck = TTNNBottleneck(
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
        new_patch_embedding = TTNNPatchEmbedding(
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
        new_embeddings = TTNNViTEmbeddings()
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
        new_maxpool = TTNNMaxPool2dNHWC(
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
        new_upsample = TTNNUpsampleNHWC(
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
