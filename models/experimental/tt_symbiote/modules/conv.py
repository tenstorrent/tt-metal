# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Linear layer implementations for TTNN."""

import os
import torch
from torch import nn

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, MaxPool2dConfiguration, TtConv2d, TtMaxPool2d
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.utils import tree_map
from models.experimental.tt_symbiote.modules.activation import TTNNReLU
from models.experimental.tt_symbiote.modules.tensor import TTNNPermute, TTNNReshape
from models.experimental.tt_symbiote.core.run_config import (
    DistributedTensorConfig,
    replicated_tensor_mesh_composer,
    trace_enabled,
)


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
        # Always use the live NHWC input shape. Qwen3-Omni (and similar) run the same conv with
        # varying batch sizes (e.g. ``split(conv_chunksize, dim=0)`` on the audio tower); reusing
        # a cached ``model_config`` shape from the first chunk corrupts the conv hash and reshape.
        batch_size, input_height, input_width, _ = input_tensor.shape
        if isinstance(self.model_config, dict):
            prev = self.model_config.get(self.module_name)
            if prev is not None:
                reshape_output = prev.get("reshape_output", reshape_output)
            self.model_config[self.module_name] = {
                "input_shapes": [list(input_tensor.shape)],
                "reshape_output": reshape_output,
            }

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

    def set_output_tensors_config_impl(self, output_tensors):
        """Do not apply default mesh ``logical_shape_fn`` to conv activations.

        NHWC conv outputs are replicated full tensors; the default
        ``logical_shape_for_batch_channel_sharding`` multiplies the **last** dim by
        mesh width (e.g. 8). For Qwen audio that last dim can be **time** (13) after
        permute-to-NCHW, producing bogus logical shapes like 13×8=104 and breaking
        ``padded_embed[padded_mask_after_cnn]``.
        """
        if (
            self.device_state is None
            or self.device_state.mesh_device is None
            or self.device_state.mesh_device.get_num_devices() <= 1
        ):
            return super().set_output_tensors_config_impl(output_tensors)

        mesh_device = self.device_state.mesh_device

        def _replicated_config(e):
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                e.set_distributed_tensor_config(
                    DistributedTensorConfig(
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                        mesh_composer=replicated_tensor_mesh_composer(mesh_device),
                    )
                )
            return e

        return tree_map(_replicated_config, output_tensors)


@trace_enabled
class TTNNConv2dNCHW(TTNNConv2dNHWC):
    """Conv2d when activations are PyTorch-style ``NCHW`` (e.g. Qwen3-Omni audio stem).

    Permutes to ``NHWC`` for :class:`TTNNConv2dNHWC` / ``ttnn`` conv2d, then permutes outputs back.
    """

    @classmethod
    def from_torch(cls, conv: nn.Conv2d, slice_config=None):
        return super().from_torch(conv, slice_config=slice_config)

    def forward(self, input_tensor: ttnn.Tensor, reshape_output=True) -> ttnn.Tensor:
        input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
        out = TTNNConv2dNHWC.forward(self, input_tensor, reshape_output=reshape_output)
        return ttnn.permute(out, (0, 3, 1, 2))


@trace_enabled
class TTNNConv1d(TTNNModule):
    """1D convolution using ``ttnn.conv1d``; activations follow PyTorch ``Conv1d`` ``[B, C, L]``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.tt_weight = None
        self.tt_bias = None
        self._conv1d_conv_config = None
        self._conv1d_compute_config = None

    @staticmethod
    def _scalar_1d(x):
        if isinstance(x, int):
            return x
        if isinstance(x, (tuple, list)):
            return int(x[0])
        return int(x)

    @classmethod
    def from_torch(cls, conv: nn.Conv1d):
        new = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=cls._scalar_1d(conv.kernel_size),
            stride=cls._scalar_1d(conv.stride),
            padding=cls._scalar_1d(conv.padding),
            dilation=cls._scalar_1d(conv.dilation),
            groups=conv.groups,
        )
        new._fallback_torch_layer = conv
        return new

    def preprocess_weights_impl(self):
        conv = self.torch_layer
        assert isinstance(conv, nn.Conv1d), type(conv)
        # ``ttnn.conv1d`` lowers to ``conv2d``; preparation / DRAM slicing indexes ``weight[3]``.
        # PyTorch Conv1d weights are ``[out, in/groups, k]`` — add a height dim ``[out, in/groups, 1, k]``.
        w = conv.weight.data
        if w.dim() == 3:
            w = w.unsqueeze(2).contiguous()
        self.tt_weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        if conv.bias is not None:
            # Host conv bias must be 4D ``[1, 1, 1, out_channels]`` (see ``validate_host_conv_bias`` /
            # ``prepare_conv_bias_internal`` in ``prepare_conv2d_weights.cpp``); 1D bias hits ``logical_shape()[3]``.
            b = conv.bias.data.reshape(1, 1, 1, -1).contiguous()
            self.tt_bias = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            self.tt_bias = None
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
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
        """Avoid default mesh ``logical_shape_fn`` on ``[B, C, L]`` conv outputs.

        Default sharding heuristics multiply the **last** dim by mesh width. For NCL activations
        that last dim is **sequence length** (e.g. code2wav upsample), producing ``L×8`` bogus
        logical shapes and breaking residual ``input + hidden_states``.
        """
        if (
            self.device_state is None
            or self.device_state.mesh_device is None
            or self.device_state.mesh_device.get_num_devices() <= 1
        ):
            return super().set_output_tensors_config_impl(output_tensors)

        mesh_device = self.device_state.mesh_device

        def _replicated_config(e):
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                e.set_distributed_tensor_config(
                    DistributedTensorConfig(
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                        mesh_composer=replicated_tensor_mesh_composer(mesh_device),
                    )
                )
            return e

        return tree_map(_replicated_config, output_tensors)

    @staticmethod
    def _conv1d_chunk_output_steps() -> int:
        """Outputs per ``ttnn.conv1d`` chunk (smaller → lower L1 CB pressure).

        Env ``TT_SYMBIOTE_TTNN_CONV1D_CHUNK_OUT`` (default ``128``).
        """
        return max(1, int(os.environ.get("TT_SYMBIOTE_TTNN_CONV1D_CHUNK_OUT", "128")))

    @staticmethod
    def _conv1d_chunk_if_input_len() -> int:
        """Use chunked device conv when ``L`` exceeds this and ``padding == 0``.

        Env ``TT_SYMBIOTE_TTNN_CONV1D_CHUNK_IF_LEN`` (default ``384``). ``0`` = always chunk when
        ``padding == 0``.
        """
        raw = os.environ.get("TT_SYMBIOTE_TTNN_CONV1D_CHUNK_IF_LEN", "384").strip().lower()
        if raw in ("0", "off", "false", "none", ""):
            return 0
        return max(0, int(raw))

    def _conv1d_nlc_to_ncl(self, tt_out: ttnn.Tensor, batch_size: int, out_len: int) -> ttnn.Tensor:
        tt_out = ttnn.reshape(tt_out, [batch_size, out_len, self.out_channels])
        return ttnn.permute(tt_out, (0, 2, 1))

    def _conv1d_once_nlc(self, x_nlc: ttnn.Tensor, batch_size: int, seg_len: int) -> ttnn.Tensor:
        """Single ``ttnn.conv1d`` on ``x_nlc`` ``[B,L,C]``; returns ``[B,C,L_out]``."""
        ret = ttnn.conv1d(
            input_tensor=x_nlc,
            weight_tensor=self.tt_weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.tt_bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=batch_size,
            input_length=seg_len,
            conv_config=self._conv1d_conv_config,
            compute_config=self._conv1d_compute_config,
            groups=self.groups,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_out, out_len, (self.tt_weight, self.tt_bias) = ret
        self.tt_weight = self._ensure_conv1d_weight_rank4(self.tt_weight)
        self.tt_bias = self._ensure_conv1d_bias_rank4(self.tt_bias)
        return self._conv1d_nlc_to_ncl(tt_out, batch_size, out_len)

    def _forward_conv1d_chunked_nlc(self, x_nlc: ttnn.Tensor, batch_size: int, input_length: int) -> ttnn.Tensor:
        """Sequence-chunked conv (``padding == 0`` only); matches one full conv."""
        K, S, D = self.kernel_size, self.stride, self.dilation
        P = int(self.padding)
        if P != 0:
            raise RuntimeError("TTNNConv1d: internal chunked forward requires padding == 0")
        L_out_total = (input_length + 2 * P - D * (K - 1) - 1) // S + 1
        if L_out_total <= 0:
            raise RuntimeError(f"TTNNConv1d: invalid conv geometry L_in={input_length} K={K} S={S} D={D} P={P}")
        chunk_out = self._conv1d_chunk_output_steps()
        pieces: list[ttnn.Tensor] = []
        o_start = 0
        while o_start < L_out_total:
            n_target = min(chunk_out, L_out_total - o_start)
            last_o = o_start + n_target - 1
            in_lo = o_start * S
            in_hi_excl = min(input_length, last_o * S + (K - 1) * D + 1)
            l_sub = in_hi_excl - in_lo
            min_l = (K - 1) * D + 1
            if l_sub < min_l:
                raise RuntimeError(
                    f"TTNNConv1d: chunk too short l_sub={l_sub} need >= {min_l} "
                    f"(o_start={o_start} L={input_length})"
                )
            x_sub = ttnn.slice(
                x_nlc,
                [0, in_lo, 0],
                [batch_size, in_hi_excl, self.in_channels],
                [1, 1, 1],
            )
            piece = self._conv1d_once_nlc(x_sub, batch_size, l_sub)
            pieces.append(piece)
            n_out = int(piece.shape[2])
            if n_out <= 0:
                raise RuntimeError("TTNNConv1d: empty chunk output")
            o_start += n_out

        if len(pieces) == 1:
            return pieces[0]
        return ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _ensure_conv_configs(self):
        if self._conv1d_conv_config is None:
            # Prefer width sharding along sequence (conv1d→conv2d ``W``) to cap per-core L1 CBs;
            # move conv config tables to DRAM; split reader reduces reader CB pressure.
            self._conv1d_conv_config = ttnn.Conv1dConfig(
                weights_dtype=ttnn.bfloat16,
                shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                deallocate_activation=True,
                reallocate_halo_output=True,
                config_tensors_in_dram=True,
                force_split_reader=True,
            )
            self._conv1d_compute_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
            )

    def _ensure_conv1d_weight_rank4(self, w: ttnn.Tensor) -> ttnn.Tensor:
        """``conv1d`` → ``conv2d`` expects 4D weights ``[O, I/groups, 1, K]``.

        Prepared weights from ``return_weights_and_bias=True`` may come back rank-1 (flattened)
        or rank-3; DRAM slicing uses ``weight.logical_shape()[3]``, which aborts otherwise.
        """
        if w is None:
            return w
        rank = int(w.shape.rank)
        oc = self.out_channels
        icg = self.in_channels // self.groups
        ks = self.kernel_size
        if rank == 4:
            return w
        if rank == 3:
            return ttnn.reshape(w, [int(w.shape[0]), int(w.shape[1]), 1, int(w.shape[2])])
        if rank == 1:
            return ttnn.reshape(w, [oc, icg, 1, ks])
        raise RuntimeError(f"TTNNConv1d: unsupported weight rank {rank}, shape={list(w.shape)}")

    def _ensure_conv1d_bias_rank4(self, b: ttnn.Tensor | None) -> ttnn.Tensor | None:
        """Conv2d bias prep expects rank-4 ``[1, 1, 1, out_channels]`` on host (and ``[3]`` for channel count)."""
        if b is None:
            return None
        rank = int(b.shape.rank)
        oc = self.out_channels
        if rank == 4:
            return b
        if rank == 1:
            return ttnn.reshape(b, [1, 1, 1, oc])
        raise RuntimeError(f"TTNNConv1d: unsupported bias rank {rank}, shape={list(b.shape)}")

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        self._ensure_conv_configs()
        batch_size = int(input_tensor.shape[0])
        input_length = int(input_tensor.shape[2])
        self.tt_weight = self._ensure_conv1d_weight_rank4(self.tt_weight)
        self.tt_bias = self._ensure_conv1d_bias_rank4(self.tt_bias)
        x = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.permute(x, (0, 2, 1))
        P = int(self.padding)
        thr = self._conv1d_chunk_if_input_len()
        use_chunks = P == 0 and (thr == 0 or input_length > thr)
        if use_chunks:
            return self._forward_conv1d_chunked_nlc(x, batch_size, input_length)
        return self._conv1d_once_nlc(x, batch_size, input_length)


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
