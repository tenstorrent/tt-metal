# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from ...layers.module import Module, ModuleList, Parameter
from ...utils.conv3d import aligned_channels, get_conv3d_config, register_conv3d_configs
from ...utils.tensor import local_device_to_torch
from .config_wav2vec2 import Wav2Vec2Config


def _conv1d_output_len(in_len: int, kernel: int, stride: int, padding: int = 0) -> int:
    return (in_len + 2 * padding - kernel) // stride + 1


# Register blocking configs for the wav2vec2 feature-extractor shapes so
# `get_conv3d_config` doesn't fall through to the conservative `(in, 32, 1, 1, 1)`
# default. These shapes have H=W=1 and a very long T (raw audio), the opposite
# of the VAE's blockings (which assume large H/W). The C_out_block is sized to
# divide cleanly into 512 (the feature-extractor output width); T_out_block is
# capped at 32 to keep the program circular buffers within L1.
register_conv3d_configs(
    {
        (1, 512, (10, 1, 1)): (32, 128, 32, 1, 1),  # layer 0: raw audio → 512
        (512, 512, (3, 1, 1)): (128, 128, 32, 1, 1),  # layers 1-4: kernel 3
        (512, 512, (2, 1, 1)): (128, 128, 32, 1, 1),  # layers 5-6: kernel 2
    }
)


class Wav2Vec2ConvLayer(Module):
    """One Conv1d layer of the wav2vec2 feature extractor, expressed as a
    `ttnn.experimental.conv3d` with `kernel_size=(k, 1, 1)`.

    Two normalization modes (mutually exclusive):

      * ``use_group_norm=True``: HF ``nn.GroupNorm(num_groups=C, num_channels=C, affine=True)``,
        which is equivalent to ``InstanceNorm1d``. Used only on layer 0 of the
        ``feat_extract_norm="group"`` variant (wav2vec2-base).
      * ``use_layer_norm=True``: HF ``nn.LayerNorm(C, elementwise_affine=True)``
        over the channel axis. Used on every layer of the
        ``feat_extract_norm="layer"`` variant (wav2vec2-large-xlsr-53).

    Both modes apply GeLU after the (optional) normalization.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_group_norm: bool = False,
        use_layer_norm: bool = False,
        conv_bias: bool,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        assert not (use_group_norm and use_layer_norm), "group_norm and layer_norm are mutually exclusive"
        self.unpadded_in_channels = in_channels
        self.in_channels = aligned_channels(in_channels)
        self.out_channels = out_channels
        self.kernel_size_t = kernel_size
        self.stride_t = stride
        self.use_group_norm = use_group_norm
        self.use_layer_norm = use_layer_norm
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

        # Affine params for the optional normalization. Both group-norm (used
        # only on layer 0 of the "group" variant) and per-channel layer-norm
        # use the same `[1, out_channels]` affine shape.
        #
        # Stored as fp32 — these tensors are tiny (~2 KB each) but their bf16
        # error compounds badly across the 7 conv stages in the "layer" variant
        # (wav2vec2-large-xlsr). With fp32 affine params + bf16 activations,
        # the LayerNorm kernel still uses fp32_dest_acc so the per-stage
        # precision loss is bounded.
        if use_group_norm or use_layer_norm:
            self.ln_weight = Parameter(total_shape=[1, out_channels], device=mesh_device, dtype=ttnn.float32)
            self.ln_bias = Parameter(total_shape=[1, out_channels], device=mesh_device, dtype=ttnn.float32)
        else:
            self.ln_weight = None
            self.ln_bias = None

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

        if self.use_group_norm or self.use_layer_norm:
            if "layer_norm.weight" in state:
                state["ln_weight"] = state.pop("layer_norm.weight").reshape(1, -1)
            if "layer_norm.bias" in state:
                state["ln_bias"] = state.pop("layer_norm.bias").reshape(1, -1)
        else:
            # Drop any layer_norm keys for layers without normalization.
            for k in list(state):
                if k.startswith("layer_norm."):
                    state.pop(k)

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

        if self.use_group_norm:
            # HF wav2vec2 uses GroupNorm(num_groups=C, num_channels=C, affine=True)
            # which is equivalent to InstanceNorm1d: for each (batch, channel)
            # pair, compute mean/var over the T dim, then apply per-channel
            # affine. NOT a LayerNorm over channels — different statistics.
            mean = ttnn.mean(x_BTHWC, dim=1, keepdim=True)
            sq = ttnn.multiply(x_BTHWC, x_BTHWC)
            mean_sq = ttnn.mean(sq, dim=1, keepdim=True)
            var = ttnn.subtract(mean_sq, ttnn.multiply(mean, mean))
            inv_std = ttnn.rsqrt(ttnn.add(var, 1e-5))
            x_norm = ttnn.multiply(ttnn.subtract(x_BTHWC, mean), inv_std)
            x_BTHWC = ttnn.add(ttnn.multiply(x_norm, self.ln_weight.data), self.ln_bias.data)
        elif self.use_layer_norm:
            # HF wav2vec2-large-xlsr uses nn.LayerNorm(C, elementwise_affine=True)
            # over the channel axis for every conv layer. With 7 stages of conv
            # + LayerNorm in bf16, the default LayerNorm fidelity (HiFi2) loses
            # ~6 % PCC end-to-end; bumping to HiFi4 + fp32_dest_acc_en holds
            # the cumulative error well under 1 % per stage.
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

    Two HF norm modes are supported:

      * ``feat_extract_norm="group"`` (wav2vec2-base): GroupNorm-as-InstanceNorm
        on layer 0 only; remaining layers are conv + GeLU.
      * ``feat_extract_norm="layer"`` (wav2vec2-large-xlsr-53): plain
        ``nn.LayerNorm`` over the channel axis on every layer.

    Expects input `[B, T_raw, 1, 1, 1]` ROW_MAJOR; returns the final-layer
    feature tensor in TILE_LAYOUT.
    """

    def __init__(
        self,
        config: Wav2Vec2Config,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device

        # The "group" variant (wav2vec2-base) has a single GroupNorm on layer 0;
        # bf16 holds <0.1 % PCC end-to-end. The "layer" variant (large-xlsr) has
        # a LayerNorm on every one of 7 conv layers — bf16 activations through
        # those compound to ~7 % PCC loss. Default to fp32 in "layer" mode; the
        # audio encoder runs once per audio clip outside the denoise loop, so
        # the perf cost is amortized. Boundary cast back to bf16 happens in
        # `Wav2Vec2Encoder.forward` before the bf16 feature_projection.
        if dtype is None:
            dtype = ttnn.float32 if config.feat_extract_norm == "layer" else ttnn.bfloat16
        self.dtype = dtype

        in_channels_list = (1,) + tuple(config.conv_dim[:-1])
        layers = []
        for i, (in_c, out_c, k, s) in enumerate(
            zip(in_channels_list, config.conv_dim, config.conv_kernel, config.conv_stride)
        ):
            # "group" mode: GroupNorm-as-InstanceNorm only on layer 0.
            # "layer" mode: LayerNorm over channels on every layer.
            use_group_norm = config.feat_extract_norm == "group" and i == 0
            use_layer_norm = config.feat_extract_norm == "layer"
            layers.append(
                Wav2Vec2ConvLayer(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    use_group_norm=use_group_norm,
                    use_layer_norm=use_layer_norm,
                    conv_bias=config.conv_bias,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
            )
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
