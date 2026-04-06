# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Linear layer implementations for TTNN."""

import torch
from torch import nn

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, MaxPool2dConfiguration, TtConv2d, TtMaxPool2d
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig, trace_enabled
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import torch_dtype_to_ttnn_dtype, tree_map
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


class NHWCConvTransposePytorch(nn.Module):
    """A wrapper around nn.ConvTranspose2d to handle NHWC input/output."""

    def __init__(self, conv: nn.ConvTranspose2d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class NCDHWConv3dPytorch(nn.Module):
    """PyTorch fallback for :class:`TTNNConv3d` (NCDHW in/out)."""

    def __init__(self, conv: nn.Conv3d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


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


@trace_enabled
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


def _conv1d_to_height1_conv2d(conv1d: nn.Conv1d) -> nn.Conv2d:
    """Map ``nn.Conv1d`` to ``nn.Conv2d`` with ``kernel_size=(1, k)`` for TTNN 2D conv."""

    def _one(x):
        if isinstance(x, int):
            return x
        return int(x[0])

    k = _one(conv1d.kernel_size)
    s = _one(conv1d.stride)
    p = _one(conv1d.padding)
    d = _one(conv1d.dilation)

    conv2d = nn.Conv2d(
        in_channels=conv1d.in_channels,
        out_channels=conv1d.out_channels,
        kernel_size=(1, k),
        stride=(1, s),
        padding=(0, p),
        dilation=(1, d),
        groups=conv1d.groups,
        bias=conv1d.bias is not None,
        device=conv1d.weight.device,
        dtype=conv1d.weight.dtype,
    )
    with torch.no_grad():
        conv2d.weight.copy_(conv1d.weight.unsqueeze(2))
        if conv1d.bias is not None:
            conv2d.bias.copy_(conv1d.bias)
    return conv2d


class TTNNConv1d(TTNNModule):
    """1D convolution via TTNN ``conv2d`` with height 1 (activations ``[B, C, T]`` → NHWC ``[B, 1, T, C]``).

    Pairs with :class:`TTNNQwen3OmniMoeCausalConvNet` in ``activation`` for HF ``Qwen3OmniMoeCausalConvNet``.
    """

    def __init__(self):
        super().__init__()
        self.conv2d = None

    @classmethod
    def from_torch(cls, conv1d: nn.Conv1d, slice_config=None):
        new = cls()
        new._fallback_torch_layer = conv1d
        equiv = _conv1d_to_height1_conv2d(conv1d)
        new.conv2d = TTNNConv2dNHWC.from_torch(equiv, slice_config=slice_config)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, input_tensor, reshape_output=True):
        """``input_tensor``: ``ttnn.Tensor`` or ``torch.Tensor`` ``[B, C, T]``. Returns ``[B, C_out, T_out]``.

        TTNN children of TTNN parents use ``_bypass_tensor_wrapping`` (see ``device_management``), so
        :class:`TTNNQwen3OmniMoeCausalConvNet` passes host torch after ``F.pad``; convert here.
        """
        if isinstance(input_tensor, TorchTTNNTensor):
            input_tensor = input_tensor.ttnn_tensor if input_tensor.ttnn_tensor is not None else input_tensor.elem
        if isinstance(input_tensor, torch.Tensor):
            mesh_mapper = None
            dev = self.device
            if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
                mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
            input_tensor = ttnn.from_torch(
                input_tensor,
                dtype=torch_dtype_to_ttnn_dtype(input_tensor.dtype),
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=mesh_mapper,
            )
        x = ttnn.permute(input_tensor, (0, 2, 1))
        x = ttnn.unsqueeze(x, 1)
        out = self.conv2d(x, reshape_output=reshape_output)
        out = ttnn.squeeze(out, 1)
        return ttnn.permute(out, (0, 2, 1))


def _int_pair_2d(x) -> tuple:
    if isinstance(x, int):
        return (x, x)
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return (t[0], t[0])
    return (t[0], t[1])


def _transpose_conv_output_w(
    input_width: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    out_pad_w: int,
) -> int:
    """Output width for conv transpose (symmetric padding), matches PyTorch / TTNN sliding window."""
    return (input_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + out_pad_w + 1


def _conv_transpose2d_dram_width_slice_config(w_out: int):
    """Explicit DRAM width slicing for large transpose convs.

    Without this, TTNN auto-slice can pick a marginal slice count (e.g. 12 for ~5.4k-wide
    output) that still exhausts L1 vs circular buffers on Wormhole (TT_THROW CB clash).
    """
    max_slices = 172
    if w_out <= 512:
        return None
    # Smaller slices than auto (~output/450) to keep per-slice L1 well under free DRAM/L1 budget.
    num_slices = max(48, min(max_slices, (w_out + 79) // 80))
    return ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)


def _conv_transpose1d_to_height1_conv_transpose2d(c1: nn.ConvTranspose1d) -> nn.ConvTranspose2d:
    """Map ``nn.ConvTranspose1d`` to ``nn.ConvTranspose2d`` with ``kernel_size=(1, k)`` for TTNN 2D transpose conv."""

    k = _int_pair_2d(c1.kernel_size)[0]
    s = _int_pair_2d(c1.stride)[0]
    p = _int_pair_2d(c1.padding)[0]
    op = _int_pair_2d(c1.output_padding)[0]
    d = _int_pair_2d(c1.dilation)[0]

    c2 = nn.ConvTranspose2d(
        in_channels=c1.in_channels,
        out_channels=c1.out_channels,
        kernel_size=(1, k),
        stride=(1, s),
        padding=(0, p),
        output_padding=(0, op),
        dilation=(1, d),
        groups=c1.groups,
        bias=c1.bias is not None,
        device=c1.weight.device,
        dtype=c1.weight.dtype,
    )
    with torch.no_grad():
        c2.weight.copy_(c1.weight.unsqueeze(2))
        if c1.bias is not None:
            c2.bias.copy_(c1.bias)
    return c2


@trace_enabled
class TTNNConvTranspose2dNHWC(TTNNModule):
    """TTNN ``conv_transpose2d`` on NHWC activations (``[B, H, W, C]``)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
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
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.slice_config = slice_config
        self.reshape = TTNNReshape()

    @classmethod
    def from_torch(cls, conv: nn.ConvTranspose2d, slice_config=None) -> "TTNNConvTranspose2dNHWC":
        new_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=_int_pair_2d(conv.kernel_size),
            stride=_int_pair_2d(conv.stride),
            padding=_int_pair_2d(conv.padding),
            output_padding=_int_pair_2d(conv.output_padding),
            dilation=_int_pair_2d(conv.dilation),
            groups=conv.groups,
            slice_config=slice_config,
        )
        new_conv._fallback_torch_layer = NHWCConvTransposePytorch(conv)
        return new_conv

    def preprocess_weights_impl(self):
        if self.torch_layer is None:
            self._fallback_torch_layer = NHWCConvTransposePytorch(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            )
        inner = self.torch_layer.conv
        self.tt_weight, self.tt_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
            inner.weight, inner.bias
        )
        super().preprocess_weights_impl()

    def deallocate_weights_impl(self):
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        batch_size, input_height, input_width, _ = input_tensor.shape
        batch_size, input_height, input_width = int(batch_size), int(input_height), int(input_width)

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=True,
            config_tensors_in_dram=True,
        )
        if self.slice_config is not None:
            conv_config.shard_layout = self.slice_config

        dev = input_tensor.device()
        w_out_est = _transpose_conv_output_w(
            input_width,
            int(self.kernel_size[1]),
            int(self.stride[1]),
            int(self.padding[1]),
            int(self.dilation[1]),
            int(self.output_padding[1]),
        )
        dram_slice_config = _conv_transpose2d_dram_width_slice_config(w_out_est)
        # LoFi reduces circular-buffer pressure for very wide transpose paths (still DRAM-sliced).
        math_fidelity = ttnn.MathFidelity.LoFi if w_out_est > 4096 else ttnn.MathFidelity.HiFi4
        compute_config = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        if input_tensor.memory_config().is_sharded():
            input_tensor = ttnn.sharded_to_interleaved(input_tensor)

        kwargs = dict(
            input_tensor=input_tensor,
            weight_tensor=self.tt_weight,
            device=dev,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            bias_tensor=self.tt_bias,
            conv_config=conv_config,
            compute_config=compute_config,
            mirror_kernel=True,
            dtype=ttnn.bfloat16,
        )
        if dram_slice_config is not None:
            kwargs["dram_slice_config"] = dram_slice_config
        if reshape_output:
            out, (oh, ow) = ttnn.conv_transpose2d(**kwargs, return_output_dim=True)
            return self.reshape(out, [batch_size, oh, ow, -1])
        return ttnn.conv_transpose2d(**kwargs, return_output_dim=False)


class TTNNConvTranspose1d(TTNNModule):
    """1D transposed convolution via TTNN ``conv_transpose2d`` with height 1 (``[B,C,T]`` ↔ NHWC ``[B,1,T,C]``)."""

    def __init__(self):
        super().__init__()
        self.conv2d_t = None

    @classmethod
    def from_torch(cls, conv1d: nn.ConvTranspose1d, slice_config=None):
        new = cls()
        new._fallback_torch_layer = conv1d
        equiv = _conv_transpose1d_to_height1_conv_transpose2d(conv1d)
        new.conv2d_t = TTNNConvTranspose2dNHWC.from_torch(equiv, slice_config=slice_config)
        return new

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, input_tensor, reshape_output=True):
        if isinstance(input_tensor, TorchTTNNTensor):
            input_tensor = input_tensor.ttnn_tensor if input_tensor.ttnn_tensor is not None else input_tensor.elem
        if isinstance(input_tensor, torch.Tensor):
            mesh_mapper = None
            dev = self.device
            if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
                mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
            input_tensor = ttnn.from_torch(
                input_tensor,
                dtype=torch_dtype_to_ttnn_dtype(input_tensor.dtype),
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=mesh_mapper,
            )
        x = ttnn.permute(input_tensor, (0, 2, 1))
        x = ttnn.unsqueeze(x, 1)
        out = self.conv2d_t(x, reshape_output=reshape_output)
        out = ttnn.squeeze(out, 1)
        return ttnn.permute(out, (0, 2, 1))


# TTNN ``experimental.conv3d`` layout (see ``tests/ttnn/unit_tests/operations/conv/test_conv3d.py``).
_CONV3D_CHANNEL_ALIGNMENT = 32


def _conv3d_mesh_max_n_per_chunk(dev) -> int:
    """Upper bound on leading batch N for one conv3d upload on a mesh (patch embed flattens many patches).

    Without chunking, ``from_torch`` + ``experimental.conv3d`` + output readback can request 100+ GiB DRAM
    for large ``N``.  Set ``TT_SYMBIOTE_CONV3D_CHUNK_N`` to tune (default 1024).
    """
    if dev is None or not hasattr(dev, "get_num_devices") or dev.get_num_devices() <= 1:
        return 0
    import os

    return max(1, int(os.environ.get("TT_SYMBIOTE_CONV3D_CHUNK_N", "1024")))


def _int_triplet_3d(x) -> tuple:
    if isinstance(x, int):
        return (x, x, x)
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return (t[0], t[0], t[0])
    if len(t) == 3:
        return (t[0], t[1], t[2])
    raise ValueError(f"Expected int or length-1/3 tuple for 3D conv hyperparameter, got {x!r}")


def _conv3d_out_dim(size_in: int, pad: int, stride: int, kernel: int, dilation: int = 1) -> int:
    eff_k = dilation * (kernel - 1) + 1
    return (size_in + 2 * pad - eff_k) // stride + 1


def _conv3d_padding_triple_for_ttnn(padding) -> tuple:
    """Map ``nn.Conv3d.padding`` to a ``(pd, ph, pw)`` triple for TTNN (symmetric triple only)."""
    if isinstance(padding, int):
        return (padding, padding, padding)
    p = tuple(int(x) for x in padding)
    if len(p) == 3:
        return (p[0], p[1], p[2])
    if len(p) == 6:
        raise ValueError("TTNNConv3d does not support 6-tuple asymmetric padding; use PyTorch fallback.")
    raise ValueError(f"Unsupported Conv3d padding: {padding!r}")


def _conv3d_torch_weights_to_ttnn_layout(
    conv: nn.Conv3d,
    *,
    alignment: int = _CONV3D_CHANNEL_ALIGNMENT,
    c_in_block: int = 0,
) -> tuple:
    """Layout ``nn.Conv3d`` weights/bias like ``prepare_weights`` in ``test_conv3d`` (host torch)."""
    w = conv.weight.data  # [out, in, kD, kH, kW]
    c_in = int(conv.in_channels)
    out_ch = int(conv.out_channels)
    w = w.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out
    align_pad = (alignment - c_in % alignment) % alignment
    if align_pad:
        w = torch.nn.functional.pad(w, (0, 0, 0, align_pad))
    kD, kH, kW, c_pad, _ = w.shape
    c_in_block_eff = c_pad if c_in_block == 0 else int(c_in_block)
    num_c_in_blocks = c_pad // c_in_block_eff
    assert num_c_in_blocks * c_in_block_eff == c_pad
    w = w.reshape(kD, kH, kW, num_c_in_blocks, c_in_block_eff, out_ch)
    w = w.permute(3, 0, 1, 2, 4, 5)
    w = w.reshape(-1, out_ch).contiguous()
    b = conv.bias.data.reshape(1, -1).contiguous() if conv.bias is not None else None
    return w, b


def _conv3d_torch_input_to_ndhwc_row_major(
    x: torch.Tensor, *, alignment: int = _CONV3D_CHANNEL_ALIGNMENT
) -> torch.Tensor:
    """NCDHW → NDHWC and pad channel to ``alignment`` (TTNN conv3d input)."""
    if x.dim() != 5:
        raise RuntimeError(f"TTNNConv3d expects 5D NCDHW input, got shape {tuple(x.shape)}")
    c_in = int(x.shape[1])
    tt_input = x.permute(0, 2, 3, 4, 1).contiguous()
    align_pad = (alignment - c_in % alignment) % alignment
    if align_pad:
        tt_input = torch.nn.functional.pad(tt_input, (0, align_pad))
    return tt_input


def _ttnn_conv3d_output_to_torch_ncdhw(
    tt_out: ttnn.Tensor,
    *,
    n: int,
    d_out: int,
    h_out: int,
    w_out: int,
    out_channels: int,
    mesh_device,
) -> torch.Tensor:
    """Match ``reshape_output`` in ``test_conv3d``; host ``[N,C,D,H,W]``."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        t = ttnn.to_torch(tt_out).float()
    else:
        shards = ttnn.get_device_tensors(tt_out)
        if shards:
            t = ttnn.to_torch(shards[0]).float()
        else:
            composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
            t = ttnn.to_torch(tt_out, mesh_composer=composer).float()
            lead = int(tt_out.shape[0])
            if t.dim() >= 1 and int(t.shape[0]) > lead:
                t = t[:lead].contiguous()
    t = t.reshape(n, d_out, h_out, w_out, out_channels)
    return t.permute(0, 4, 1, 2, 3).contiguous()


def _ttnn_tensor_to_torch_batch_leading(tt: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Device/mesh tensor → host torch; for multi-device concat on dim 0, slice to logical batch."""
    n0 = int(tt.shape[0])
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices") or mesh_device.get_num_devices() <= 1:
        return ttnn.to_torch(tt)
    shards = ttnn.get_device_tensors(tt)
    if shards:
        return ttnn.to_torch(shards[0])
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    t = ttnn.to_torch(tt, mesh_composer=composer)
    if t.dim() >= 1 and int(t.shape[0]) > n0:
        t = t[:n0].contiguous()
    return t


@trace_enabled
class TTNNConv3d(TTNNModule):
    """3D convolution via ``ttnn.experimental.conv3d`` (NDHWC activations, Qwen3-Omni vision patch embed).

    Matches the weight/input layout used in ``tests/ttnn/unit_tests/operations/conv/test_conv3d.py``.
    For large patch kernels (e.g. ``kernel_size == stride`` with spatial product ≥ 196) and
    ``out_channels % 32 == 0``, uses Qwen-VL-style L1 blocking (``C_in_block=16``, ``C_out_block=32``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        groups: int,
        padding_mode: str,
        conv3d_blocking=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.conv3d_blocking = conv3d_blocking
        self.tt_weight = None
        self.tt_bias = None

    @property
    def weight(self):
        """HF ``Qwen3OmniMoeVisionPatchEmbed`` reads ``proj.weight.dtype``; mirror ``nn.Conv3d``."""
        if self.torch_layer is None:
            raise AttributeError("weight")
        return self.torch_layer.conv.weight

    @property
    def bias(self):
        if self.torch_layer is None:
            raise AttributeError("bias")
        return self.torch_layer.conv.bias

    @classmethod
    def from_torch(cls, conv: nn.Conv3d, conv3d_blocking=None) -> "TTNNConv3d":
        ks = _int_triplet_3d(conv.kernel_size)
        st = _int_triplet_3d(conv.stride)
        try:
            pad_triple = _conv3d_padding_triple_for_ttnn(conv.padding)
        except ValueError:
            pad_triple = (0, 0, 0)
        blocking = conv3d_blocking
        if blocking is None:
            if ks == st and ks[0] * ks[1] * ks[2] >= 196 and conv.out_channels % 32 == 0 and conv.groups == 1:
                blocking = (16, 32, 1, 1, 1)
        new_mod = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=ks,
            stride=st,
            padding=pad_triple,
            dilation=_int_triplet_3d(conv.dilation),
            groups=conv.groups,
            padding_mode=conv.padding_mode,
            conv3d_blocking=blocking,
        )
        new_mod._fallback_torch_layer = NCDHWConv3dPytorch(conv)
        return new_mod

    def preprocess_weights_impl(self):
        if self.torch_layer is None:
            self._fallback_torch_layer = NCDHWConv3dPytorch(
                nn.Conv3d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    bias=True,
                )
            )
        conv = self.torch_layer.conv
        c_in_block = 0
        if self.conv3d_blocking is not None:
            c_in_block = int(self.conv3d_blocking[0])
        w_torch, b_torch = _conv3d_torch_weights_to_ttnn_layout(conv, c_in_block=c_in_block)
        self.tt_weight = ttnn.from_torch(w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
        if b_torch is not None:
            self.tt_bias = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
        else:
            self.tt_bias = None
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        if self.tt_weight is not None:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        if self.tt_bias is not None:
            self.tt_bias = ttnn.to_device(self.tt_bias, self.device)
        super().move_weights_to_device_impl()

    def deallocate_weights_impl(self):
        if self.tt_weight is not None:
            ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def __call__(self, *args, **kwds):
        """Same as ``NormalRun.module_run`` but skip ``to_ttnn_wrap`` on inputs.

        Default mesh ``to_ttnn`` for 5D NCDHW activations can size buffers incorrectly; we keep
        ``TorchTTNNTensor.elem`` as torch until :meth:`forward` uploads NDHWC via ``from_torch``.
        """
        bypass = getattr(self, "_bypass_tensor_wrapping", False)
        if bypass:
            from models.experimental.tt_symbiote.core.run_config import NormalRun

            return NormalRun.module_run(self, *args, **kwds)

        import time

        from tracy import signpost

        from models.experimental.tt_symbiote.core.run_config import (
            DispatchManager,
            NormalRun,
            compose_transforms,
            post_process_ttnn_module_output,
            set_device_wrap,
            wrap_to_torch_ttnn_tensor,
        )

        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        assert self.device is not None, "Device must be set for TTNN module execution."
        begin_full = time.time()
        transform = compose_transforms(wrap_to_torch_ttnn_tensor, set_device_wrap(self.device))
        func_args = tree_map(transform, args)
        other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
        func_kwargs = tree_map(transform, other_kwargs)
        func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})
        begin = time.time()
        self.preprocess_weights()
        end = time.time()
        DispatchManager.set_current_module_name(self.module_name)
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_preprocess_weights", {}, end - begin
        )
        begin = time.time()
        self.move_weights_to_device()
        end = time.time()
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_move_weights_to_device", {}, end - begin
        )
        if NormalRun.signpost_mode is not None:
            signpost(f"{self.module_name}", f"{self.__class__.__name__}")
        begin = time.time()
        result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
        end = time.time()
        DispatchManager.record_timing("TTNN", self.module_name, self.__class__.__name__ + "_forward", {}, end - begin)
        DispatchManager.set_current_module_name(None)
        end_full = time.time()
        DispatchManager.record_timing(
            "TorchModules", self.module_name, self.__class__.__name__, {}, end_full - begin_full
        )
        return result

    def _experimental_conv3d_ncdhw_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``ttnn.experimental.conv3d`` for one CPU ``[N,C,D,H,W]`` batch; return host ``[N,C_out,D_out,H_out,W_out]``."""
        n, _, d_in, h_in, w_in = x.shape
        pd, ph, pw = self.padding
        sd, sh, sw = self.stride
        dd, dh, dw = self.dilation
        kd, kh, kw = self.kernel_size
        d_out = _conv3d_out_dim(d_in, pd, sd, kd, dd)
        h_out = _conv3d_out_dim(h_in, ph, sh, kh, dh)
        w_out = _conv3d_out_dim(w_in, pw, sw, kw, dw)

        dev = self.device
        mesh_mapper = None
        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(dev)
        ndhwc = _conv3d_torch_input_to_ndhwc_row_major(x)
        tt_in = ttnn.from_torch(
            ndhwc,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mesh_mapper,
        )
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        grid = dev.compute_with_storage_grid_size()
        grid_xy = (int(grid.x), int(grid.y))

        kwargs = dict(
            input_tensor=tt_in,
            weight_tensor=self.tt_weight,
            bias_tensor=self.tt_bias,
            dtype=ttnn.bfloat16,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            padding_mode=self.padding_mode,
            compute_kernel_config=compute_kernel_config,
        )
        if self.conv3d_blocking is not None:
            cin_b, cout_b, tb, hb, wb = self.conv3d_blocking
            kwargs["config"] = ttnn.Conv3dConfig(
                weights_dtype=ttnn.bfloat16,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                T_out_block=int(tb),
                W_out_block=int(wb),
                H_out_block=int(hb),
                C_out_block=int(cout_b),
                C_in_block=int(cin_b),
                dilation=self.dilation,
                compute_with_storage_grid_size=grid_xy,
            )
        tt_out = ttnn.experimental.conv3d(**kwargs)
        out_torch = _ttnn_conv3d_output_to_torch_ncdhw(
            tt_out,
            n=n,
            d_out=d_out,
            h_out=h_out,
            w_out=w_out,
            out_channels=self.out_channels,
            mesh_device=dev,
        )
        ttnn.deallocate(tt_in)
        ttnn.deallocate(tt_out)
        return out_torch

    def forward(self, x):
        if isinstance(x, TorchTTNNTensor):
            x = x.ttnn_tensor if x.ttnn_tensor is not None else x.elem
        dev = self.device
        conv_pad = self.torch_layer.conv.padding
        if isinstance(conv_pad, tuple) and len(conv_pad) == 6:
            if isinstance(x, ttnn.Tensor):
                x = _ttnn_tensor_to_torch_batch_leading(x, dev)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"TTNNConv3d expected torch or ttnn tensor, got {type(x)}")
            return self.torch_layer.conv(x)

        if self.groups != 1 or self.padding_mode != "zeros":
            if isinstance(x, ttnn.Tensor):
                x = _ttnn_tensor_to_torch_batch_leading(x, dev)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"TTNNConv3d expected torch or ttnn tensor, got {type(x)}")
            x = x.to(dtype=self.torch_layer.conv.weight.dtype)
            return self.torch_layer.conv(x)

        if isinstance(x, ttnn.Tensor):
            x = _ttnn_tensor_to_torch_batch_leading(x, dev)
            x = x.to(dtype=torch.bfloat16).contiguous()

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"TTNNConv3d expected torch or ttnn tensor, got {type(x)}")
        x = x.to(dtype=torch.bfloat16).contiguous()
        n = int(x.shape[0])
        max_chunk = _conv3d_mesh_max_n_per_chunk(dev)
        if max_chunk > 0 and n > max_chunk:
            parts = []
            for s in range(0, n, max_chunk):
                e = min(s + max_chunk, n)
                parts.append(self._experimental_conv3d_ncdhw_torch(x[s:e].contiguous()))
            out_torch = torch.cat(parts, dim=0)
        else:
            out_torch = self._experimental_conv3d_ncdhw_torch(x)

        mesh_mapper_out = None
        if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
            mesh_mapper_out = ttnn.ReplicateTensorToMesh(dev)
        return ttnn.from_torch(
            out_torch,
            dtype=torch_dtype_to_ttnn_dtype(out_torch.dtype),
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=mesh_mapper_out,
        )


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
                activation=ttnn.UnaryOpType.RELU,
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,
            )
            layer = TtConv2d(config, input_tensor.device())
            TTNNConv2dNHWC.CACHED_TTCNN[hash] = layer
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
                identity = ttnn.to_device(identity.to_ttnn, out.device())

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


def _qwen_omni_conv2d_mesh_output_config(mesh_device):
    """Mesh readback for Qwen Omni audio/vision conv outputs (replicated activations per device).

    Default ``DistributedConfig`` uses ``ConcatMesh2dToTensor`` + ``logical_shape_for_batch_channel_sharding``,
    which does **not** match full spatial NHWC conv outputs and can inflate a dimension (e.g. time/freq
    13 → 104 on an 8-device mesh), breaking ``padded_embed[mask]`` in the HF audio encoder.

    Use the same pattern as :func:`models.experimental.tt_symbiote.modules.qwen_omni_lm_head._lm_head_logits_dtensor_config`:
    replicate mapper, compose on batch dim, then slice to one replica. (TTNN ``conv2d`` selects cores from
    ``device.compute_with_storage_grid_size()`` in the runtime; this path only fixes host logical shape.)
    """
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return None
    return DistributedTensorConfig(
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        replicate_compose_slice_dim0_to_leading=True,
    )


@trace_enabled
class TTNNQwenOmniConv2dNHWC(TTNNConv2dNHWC):
    """Qwen3-Omni ``nn.Conv2d`` for thinker vision (NHWC) and audio tower (NCHW).

    Audio encoder feeds **NCHW** ``[B, C, F, T]`` into ``conv2d*``; base :class:`TTNNConv2dNHWC`
    assumes **NHWC** ``[B, H, W, C]``, which mis-sized the conv and caused reshape errors in
    ``ttnn.conv2d``. This class permutes using the **logical** ``conv.in_channels`` (before any
    width pad), pads activations when :class:`TTNNConv2dNHWCInputMultipleOf16` widened channels,
    then restores NCHW for the next HF layer when the input was NCHW.

    ``from_torch`` always returns this type (never a bare :class:`TTNNConv2dNHWC`) so this
    ``forward`` always runs.
    """

    def _logical_in_channels(self) -> int:
        tl = self.torch_layer
        if tl is None:
            return int(self.in_channels)
        inner = getattr(tl, "conv", None)
        if inner is not None:
            return int(inner.in_channels)
        return int(self.in_channels)

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, slice_config=None) -> "TTNNQwenOmniConv2dNHWC":
        # Mirror :class:`TTNNConv2dNHWCInputMultipleOf16` but always construct ``cls`` so
        # :meth:`forward` below is used (``InputMultipleOf16.from_torch`` returns ``TTNNConv2dNHWC``).
        if conv.in_channels > 16 or conv.in_channels % 16 == 0:
            new_mod = cls(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                slice_config=slice_config,
            )
            new_mod._fallback_torch_layer = NHWCConvPytorch(conv)
            return new_mod
        new_mod = cls(
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
        new_mod._fallback_torch_layer = NHWCConvPytorch(conv)
        return new_mod

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _qwen_omni_conv2d_mesh_output_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        logical_in = self._logical_in_channels()
        shape = input_tensor.shape
        nchw_in = False
        if len(shape) != 4:
            raise RuntimeError(f"TTNNQwenOmniConv2dNHWC: expected 4D input, got shape={list(shape)}")
        if int(shape[1]) == logical_in and int(shape[-1]) != logical_in:
            nchw_in = True
            input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
        elif int(shape[-1]) != logical_in and int(shape[1]) != logical_in:
            raise RuntimeError(
                f"TTNNQwenOmniConv2dNHWC: cannot match in_channels={logical_in} to layout " f"shape={list(shape)}"
            )

        if int(self.in_channels) > logical_in and int(input_tensor.shape[-1]) == logical_in:
            pad_c = int(self.in_channels) - logical_in
            layout_in = input_tensor.layout
            if layout_in == ttnn.TILE_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            input_tensor = ttnn.pad(
                input_tensor,
                ((0, 0), (0, 0), (0, 0), (0, pad_c)),
                value=0.0,
            )
            if layout_in == ttnn.TILE_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)

        out = TTNNConv2dNHWC.forward(self, input_tensor, reshape_output=reshape_output)
        if nchw_in:
            out = ttnn.permute(out, (0, 3, 1, 2))
        return out
