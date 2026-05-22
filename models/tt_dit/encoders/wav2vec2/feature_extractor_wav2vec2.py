# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ...layers.module import Module, ModuleList, Parameter
from ...utils.conv3d import aligned_channels, get_conv3d_config
from ...utils.tensor import local_device_to_torch
from .config_wav2vec2 import Wav2Vec2Config


def _conv1d_output_len(in_len: int, kernel: int, stride: int, padding: int = 0) -> int:
    return (in_len + 2 * padding - kernel) // stride + 1


class Wav2Vec2ConvLayer(Module):
    """One Conv1d layer of the wav2vec2 feature extractor, expressed as a
    `ttnn.experimental.conv3d` with `kernel_size=(k, 1, 1)`.

    Applies HF ``nn.LayerNorm(C, elementwise_affine=True)`` over the channel
    axis (the ``feat_extract_norm="layer"`` variant used by
    wav2vec2-large-xlsr-53), then GeLU.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        conv_bias: bool,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.unpadded_in_channels = in_channels
        self.in_channels = aligned_channels(in_channels)
        self.out_channels = out_channels
        self.kernel_size_t = kernel_size
        self.stride_t = stride
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            (kernel_size, 1, 1),
            dtype,
            grid_size=mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
        )

        # Matmul-style compute config (conv3d is matmul-based under the hood).
        # `packer_l1_acc=True` preserves L1 accumulation precision the same
        # way the WAN T2V transformer's Linear ops do; combined with HiFi4
        # this cuts the per-layer cumulative bf16 drift roughly in half.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.layer_norm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # conv3d weight layout after `prepare_conv3d_weights`: [d, C_out] where
        # d = kT * kH * kW * C_in. Replicated across the mesh — feature
        # extraction is tiny vs. the transformer trunk and shape parallelism
        # for H=W=1 isn't meaningful here.
        d = kernel_size * 1 * 1 * self.in_channels
        self.weight = Parameter(
            total_shape=[d, out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )
        self.bias = (
            Parameter(
                total_shape=[1, out_channels],
                device=mesh_device,
                pad_value=0,
                dtype=dtype,
            )
            if conv_bias
            else None
        )

        # LayerNorm affine params (`[1, out_channels]`). Stored as fp32 — these
        # tensors are tiny (~2 KB each) but their bf16 error compounds badly
        # across the 7 conv stages. With fp32 affine params + bf16 activations,
        # the LayerNorm kernel still uses fp32_dest_acc so the per-stage
        # precision loss is bounded.
        self.ln_weight = Parameter(total_shape=[1, out_channels], device=mesh_device, dtype=ttnn.float32)
        self.ln_bias = Parameter(total_shape=[1, out_channels], device=mesh_device, dtype=ttnn.float32)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # HF Conv1d weight shape: [out, in, k]. Reshape to [out, in, k, 1, 1]
        # (3D-conv weight) then prepare for the C_in_block layout.
        if "conv.weight" in state:
            w = state.pop("conv.weight")  # [out, in, k]
            assert w.shape == (self.out_channels, self.unpadded_in_channels, self.kernel_size_t), w.shape

            # Pad in_channels up to ALIGNMENT (=32) if needed; the conv3d kernel
            # always sees in_channels-aligned weights.
            if self.in_channels != self.unpadded_in_channels:
                pad = self.in_channels - self.unpadded_in_channels
                w = torch.nn.functional.pad(w, (0, 0, 0, pad))  # pad in-dim

            w_5d = w.unsqueeze(-1).unsqueeze(-1)  # [out, in, k, 1, 1]
            weight_tt = ttnn.from_torch(w_5d, dtype=self.dtype, pad_value=0)
            prepared = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=weight_tt,
                C_in_block=self.conv_config.C_in_block,
                device=self.mesh_device,
            )
            state["weight"] = local_device_to_torch(prepared)

        if "conv.bias" in state:
            state["bias"] = state.pop("conv.bias").reshape(1, -1)

        if "layer_norm.weight" in state:
            state["ln_weight"] = state.pop("layer_norm.weight").reshape(1, -1)
        if "layer_norm.bias" in state:
            state["ln_bias"] = state.pop("layer_norm.bias").reshape(1, -1)

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        """Apply the conv → (LayerNorm) → GeLU.

        Args:
            x_BTHWC: `[B, T_in, 1, 1, C_in]` ROW_MAJOR.

        Returns:
            `[B, T_out, 1, 1, C_out]` TILE_LAYOUT after GeLU.
        """
        x_BTHWC = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data if self.bias is not None else None,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=(self.kernel_size_t, 1, 1),
            stride=(self.stride_t, 1, 1),
            padding=(0, 0, 0),
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)

        # nn.LayerNorm(C, elementwise_affine=True) over the channel axis on
        # every conv layer. With 7 stages of conv + LayerNorm in bf16, the
        # default LayerNorm fidelity (HiFi2) loses ~6 % PCC end-to-end; HiFi4 +
        # fp32_dest_acc_en holds cumulative error well under 1 % per stage.
        x_BTHWC = ttnn.layer_norm(
            x_BTHWC,
            weight=self.ln_weight.data,
            bias=self.ln_bias.data,
            epsilon=1e-5,
            compute_kernel_config=self.layer_norm_compute_kernel_config,
        )

        return ttnn.gelu(x_BTHWC)


class Wav2Vec2FeatureExtractor(Module):
    """TTNN port of HuggingFace Wav2Vec2's `feature_extractor`, built on
    `ttnn.experimental.conv3d`. Maps 1D temporal convs to conv3d with
    `kernel_size=(k, 1, 1)` and `H=W=1`.

    Runs the ``feat_extract_norm="layer"`` variant (wav2vec2-large-xlsr-53):
    ``nn.LayerNorm`` over the channel axis on every layer.

    Expects input `[B, T_raw, 1, 1, 1]` ROW_MAJOR; returns the final-layer
    feature tensor in TILE_LAYOUT.
    """

    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        assert config.feat_extract_norm == "layer", "only feat_extract_norm='layer' (large-xlsr-53) is supported"
        self.config = config
        self.mesh_device = mesh_device

        # 7 stages of conv + LayerNorm compound bf16 error to ~7 % PCC loss, so
        # default to fp32. The audio encoder runs once per audio clip outside
        # the denoise loop, so the perf cost is amortized. Boundary cast back
        # to bf16 happens in `Wav2Vec2Encoder.forward` before the bf16
        # feature_projection.
        self.dtype = dtype

        in_channels_list = (1,) + tuple(config.conv_dim[:-1])
        layers = [
            Wav2Vec2ConvLayer(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=k,
                stride=s,
                conv_bias=config.conv_bias,
                mesh_device=mesh_device,
                dtype=dtype,
            )
            for in_c, out_c, k, s in zip(in_channels_list, config.conv_dim, config.conv_kernel, config.conv_stride)
        ]
        self.conv_layers = ModuleList(layers)

    def output_length(self, input_length: int) -> int:
        length = input_length
        for k, s in zip(self.config.conv_kernel, self.config.conv_stride):
            length = _conv1d_output_len(length, kernel=k, stride=s, padding=0)
        return length

    def forward(self, audio_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        """Run the 7-conv front-end.

        Args:
            audio_BTHWC: `[B, T_raw, 1, 1, 1]` ROW_MAJOR.

        Returns:
            `[B, T_out, 1, 1, 512]` TILE_LAYOUT.
        """
        x = audio_BTHWC
        for layer in self.conv_layers:
            x = layer(x)
            # Conv3d expects ROW_MAJOR input; convert back between layers.
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
