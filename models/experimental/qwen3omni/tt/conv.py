# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

from models.experimental.tt_symbiote.core.module import TTNNModule


def _ttnn_to_torch_mesh_safe(tensor: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Host read for tensors on a mesh (e.g. F.pad in Python). Avoids to_torch without mesh_composer."""
    n = getattr(mesh_device, "get_num_devices", lambda: 1)()
    if n <= 1:
        return ttnn.to_torch(tensor)
    shards = ttnn.get_device_tensors(tensor)
    return ttnn.to_torch(shards[0])


# Height-sharded conv2d uses conv_act_c_blocks == 1, so activation reads coalesce
# weight_size_w * (C * elem_size) bytes. Wormhole NOC burst cap is 8192 B.
_NOC_MAX_BURST_BYTES = 8192


def _depthwise_chunk_channels(kernel_size: int, in_channels: int, bytes_per_elem: int = 2) -> int:
    """Largest multiple-of-32 channel count per depthwise conv such that K*C*elem <= NOC burst limit."""
    if in_channels <= 0:
        return in_channels
    max_c = _NOC_MAX_BURST_BYTES // (kernel_size * bytes_per_elem)
    aligned = max(32, (max_c // 32) * 32)
    return min(aligned, in_channels)


# NOC-sized chunks (~576 ch @ K=7) still blow L1 CB budgets (~1.5 MiB/core on WH); cap channels per split.
_DEPTHWISE_L1_SAFE_CHUNK_CHANNELS = 128


def _depthwise_effective_chunk_channels(kernel_size: int, in_channels: int) -> int:
    """Chunk size honoring both NOC burst limit and a conservative L1 circular-buffer cap."""
    noc_ok = _depthwise_chunk_channels(kernel_size, in_channels)
    capped = min(noc_ok, _DEPTHWISE_L1_SAFE_CHUNK_CHANNELS)
    aligned = max(32, (capped // 32) * 32)
    return min(aligned, in_channels)


def _postprocess_conv1d_output(output: ttnn.Tensor) -> ttnn.Tensor:
    """Normalize conv1d output to (B, C, L_out) like PyTorch depthwise."""
    if len(output.shape) == 4:
        n, h, w, c = output.shape
        # Conv2d output can be height-sharded with 16-wide shards (not tile 32-aligned).
        # Move to interleaved DRAM before permute so layout is valid for later ops.
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        # (N, H, W, C) -> (N, C, W, H); then squeeze H to get (N, C, W).
        out = ttnn.permute(output, (0, 3, 2, 1))  # (N, C, W, H)
        if h != 1:
            raise RuntimeError(f"conv1d 4D output expected H=1 (conv2d height), got shape={tuple(output.shape)}")
        return ttnn.squeeze(out, 3)
    if len(output.shape) == 3:
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.permute(output, (0, 2, 1))
    return output


class TtQwen3OmniMoeCausalConvNet:
    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        # Calculate effective kernel size and padding (same as PyTorch)
        self.effective_kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.effective_kernel_size - self.stride

        # Wide depthwise (e.g. C=1024, K=7) coalesces 7*C*2 B reads; must stay <= NOC_MAX_BURST_SIZE (8192).
        # act_block_h_override keeps activation CB height small vs default "full output height" (L1 OOM otherwise).
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            force_split_reader=True,
            act_block_h_override=32,
            config_tensors_in_dram=True,
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _get_extra_padding_for_conv1d(self, hidden_state: ttnn.Tensor) -> int:
        """Calculate extra padding needed for causal convolution"""
        length = hidden_state.shape[-1]
        n_frames = (length - self.effective_kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.effective_kernel_size - self.padding)
        return ideal_length - length

    def forward(
        self, hidden_state: ttnn.Tensor, weight_tensor: ttnn.Tensor, bias_tensor: Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        # Apply custom causal padding on host (mesh-safe read; plain to_torch fails on multi-device mesh)
        hidden_state_torch = _ttnn_to_torch_mesh_safe(hidden_state, self.device)
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        padded_torch = torch.nn.functional.pad(
            hidden_state_torch,
            (self.padding, extra_padding),
            mode="constant",
            value=0,
        )
        # PyTorch-style padded activations (B, C, L) -> (B, L, C) NLC, same as tests/ttnn/unit_tests/operations/conv/test_conv1d.py
        b, _c, seq_len = padded_torch.shape
        padded_nlc = padded_torch.permute(0, 2, 1).contiguous()
        if getattr(self.device, "get_num_devices", lambda: 1)() > 1:
            with ttnn.distribute(ttnn.ReplicateTensorToMesh(self.device)):
                padded_tt = ttnn.from_torch(
                    padded_nlc,
                    dtype=ttnn.bfloat16,
                    device=self.device,
                )
        else:
            padded_tt = ttnn.from_torch(
                padded_nlc,
                dtype=ttnn.bfloat16,
                device=self.device,
            )

        # conv2d path expects bias as (1, 1, 1, out_channels); checkpoint bias is often (C,) or (1, C)
        bias_for_conv = bias_tensor
        if bias_tensor is not None:
            bd = tuple(bias_tensor.shape)
            if len(bd) == 1:
                bias_for_conv = ttnn.reshape(bias_tensor, (1, 1, 1, bd[0]))
            elif len(bd) == 2 and bd[0] == 1:
                bias_for_conv = ttnn.reshape(bias_tensor, (1, 1, 1, bd[1]))

        # Wide depthwise on HEIGHT_SHARDED still uses conv_act_c_blocks==1, so coalesced reads can exceed
        # NOC_MAX_BURST_SIZE; split along channels. Use _depthwise_effective_chunk_channels (not the NOC-only
        # helper): ~576-wide chunks still exceed WH L1 circular-buffer limits (~1.5 MiB/core).
        chunk_c = _depthwise_effective_chunk_channels(self.kernel_size, self.in_channels)
        if self.groups == self.in_channels and chunk_c < self.in_channels:
            outs = []
            for c0 in range(0, self.in_channels, chunk_c):
                c1 = min(c0 + chunk_c, self.in_channels)
                g = c1 - c0
                padded_chunk = ttnn.slice(padded_tt, (0, 0, c0), (b, seq_len, c1))
                w_chunk = ttnn.slice(weight_tensor, (c0, 0, 0), (c1, 1, self.kernel_size))
                b_chunk = None
                if bias_for_conv is not None:
                    b_chunk = ttnn.slice(bias_for_conv, (0, 0, 0, c0), (1, 1, 1, c1))
                out_c, _, _ = ttnn.conv1d(
                    input_tensor=padded_chunk,
                    weight_tensor=w_chunk,
                    device=self.device,
                    in_channels=g,
                    out_channels=g,
                    batch_size=b,
                    input_length=seq_len,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=0,
                    dilation=self.dilation,
                    groups=g,
                    bias_tensor=b_chunk,
                    conv_config=self.conv_config,
                    compute_config=self.compute_config,
                    return_output_dim=True,
                    return_weights_and_bias=True,
                )
                out_c = _postprocess_conv1d_output(out_c)
                # Move to interleaved DRAM so concat output has tile-aligned shards (avoids
                # "Physical shard shape (..., 16) must be tile {32, 32} sized" when concat preserves input sharding).
                out_c = ttnn.to_memory_config(out_c, ttnn.DRAM_MEMORY_CONFIG)
                outs.append(out_c)
            return ttnn.concat(outs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Use TTNN conv1d (wrapper around conv2d)
        output, _, _ = ttnn.conv1d(
            input_tensor=padded_tt,
            weight_tensor=weight_tensor,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=b,
            input_length=seq_len,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,  # Padding already applied
            dilation=self.dilation,
            groups=self.groups,
            bias_tensor=bias_for_conv,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return _postprocess_conv1d_output(output)


class TtQwen3OmniMoeCausalTransConvNet:
    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Calculate padding for causal transposed conv (match PyTorch: right_pad = left_pad)
        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = self.left_pad

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            force_split_reader=True,
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(
        self, hidden_state: ttnn.Tensor, weight_tensor: ttnn.Tensor, bias_tensor: Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        # Reshape for conv_transpose2d (add height=1 dimension)
        batch_size, seq_len, channels = hidden_state.shape
        hidden_state_4d = ttnn.reshape(hidden_state, (batch_size, 1, seq_len, channels))

        # Use TTNN conv_transpose2d with height=1
        output, _, _ = ttnn.conv_transpose2d(
            input_tensor=hidden_state_4d,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, 0),
            batch_size=batch_size,
            input_height=1,
            input_width=seq_len,
            device=self.device,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=False,
            return_weights_and_bias=True,
        )

        # Remove causal padding from output (layout N,H,W,C; slice width dim 2)
        output_sliced = ttnn.slice(
            output,
            starts=[0, 0, self.left_pad, 0],
            ends=[batch_size, 1, output.shape[2] - self.right_pad, output.shape[3]],
            steps=[1, 1, 1, 1],
        )

        # Reshape back to 3D (B, L_out, C)
        return ttnn.reshape(output_sliced, (batch_size, output_sliced.shape[2], self.out_channels))


class TtQwen3OmniMoeConvNeXtBlock(TTNNModule):
    def __init__(self, device, dim: int):
        super().__init__()
        if device is not None:
            self.to_device(device)
        self.dim = dim

        # Initialize depthwise conv (causal)
        self.dwconv = TtQwen3OmniMoeCausalConvNet(
            device=device,
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        )

        # Layer norm config
        self.norm_config = {
            "epsilon": 1e-6,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        }

        self.compute_config = (
            ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            if device is not None
            else None
        )

        # Gamma parameter for scaling (stored as torch; moved to device in forward or preprocess)
        self.gamma = torch.nn.Parameter(1e-6 * torch.ones(dim))

        # Weights set by load_state_dict; preprocessed in preprocess_weights_impl
        self.dwconv_weight = None
        self.dwconv_bias = None
        self.norm_weight = None
        self.norm_bias = None
        self.pwconv1_weight = None
        self.pwconv1_bias = None
        self.pwconv2_weight = None
        self.pwconv2_bias = None
        self._tt_gamma = None

    def load_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True, prefix: str = ""):
        """Load from PyTorch ConvNeXtBlock state (e.g. dwconv.conv.weight, norm.weight, pwconv1.weight, gamma)."""
        state = dict(state)
        self._prepare_torch_state(state, prefix)
        for key in [
            "dwconv_weight",
            "dwconv_bias",
            "norm_weight",
            "norm_bias",
            "pwconv1_weight",
            "pwconv1_bias",
            "pwconv2_weight",
            "pwconv2_bias",
            "gamma",
        ]:
            if key in state:
                setattr(self, key, state[key])
        if "gamma" in state:
            self.gamma = torch.nn.Parameter(state["gamma"])

    def _prepare_torch_state(self, state: dict[str, torch.Tensor], prefix: str = "") -> None:
        """Map PyTorch state keys to internal names."""
        mapping = {
            f"{prefix}dwconv.conv.weight": "dwconv_weight",
            f"{prefix}dwconv.conv.bias": "dwconv_bias",
            f"{prefix}norm.weight": "norm_weight",
            f"{prefix}norm.bias": "norm_bias",
            f"{prefix}pwconv1.weight": "pwconv1_weight",
            f"{prefix}pwconv1.bias": "pwconv1_bias",
            f"{prefix}pwconv2.weight": "pwconv2_weight",
            f"{prefix}pwconv2.bias": "pwconv2_bias",
            f"{prefix}gamma": "gamma",
        }
        for old_key, new_key in mapping.items():
            if old_key in state:
                state[new_key] = state.pop(old_key)

    def preprocess_weights_impl(self):
        if self.dwconv_weight is None:
            return
        # Host-side TT tensors (layout/dtype); move_weights_to_device_impl moves to device
        # ttnn.conv1d requires conv weights on host in ROW_MAJOR (see prepare_conv2d_weights)
        self._tt_dwconv_weight = ttnn.from_torch(
            self.dwconv_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._tt_dwconv_bias = (
            ttnn.from_torch(self.dwconv_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            if self.dwconv_bias is not None
            else None
        )
        self._tt_norm_weight = ttnn.from_torch(
            self.norm_weight.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self._tt_norm_bias = ttnn.from_torch(
            self.norm_bias.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        # preprocess_linear_weight applies .T internally (PyTorch nn.Linear is (out, in)).
        self._tt_pwconv1_weight = preprocess_linear_weight(
            self.pwconv1_weight.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self._tt_pwconv1_bias = (
            preprocess_linear_bias(self.pwconv1_bias.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if self.pwconv1_bias is not None
            else None
        )
        self._tt_pwconv2_weight = preprocess_linear_weight(
            self.pwconv2_weight.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self._tt_pwconv2_bias = (
            preprocess_linear_bias(self.pwconv2_bias.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if self.pwconv2_bias is not None
            else None
        )
        self._tt_gamma_host = ttnn.from_torch(
            self.gamma.unsqueeze(0).unsqueeze(-1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def move_weights_to_device_impl(self):
        if getattr(self, "_tt_dwconv_weight", None) is None:
            return
        self.tt_dwconv_weight = ttnn.to_device(self._tt_dwconv_weight, self.device)
        self.tt_dwconv_bias = (
            ttnn.to_device(self._tt_dwconv_bias, self.device) if self._tt_dwconv_bias is not None else None
        )
        self.tt_norm_weight = ttnn.to_device(self._tt_norm_weight, self.device)
        self.tt_norm_bias = ttnn.to_device(self._tt_norm_bias, self.device)
        self.tt_pwconv1_weight = ttnn.to_device(self._tt_pwconv1_weight, self.device)
        self.tt_pwconv1_bias = (
            ttnn.to_device(self._tt_pwconv1_bias, self.device) if self._tt_pwconv1_bias is not None else None
        )
        self.tt_pwconv2_weight = ttnn.to_device(self._tt_pwconv2_weight, self.device)
        self.tt_pwconv2_bias = (
            ttnn.to_device(self._tt_pwconv2_bias, self.device) if self._tt_pwconv2_bias is not None else None
        )
        self._tt_gamma = ttnn.to_device(self._tt_gamma_host, self.device)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        dwconv_weight: Optional[ttnn.Tensor] = None,
        dwconv_bias: Optional[ttnn.Tensor] = None,
        pwconv1_weight: Optional[ttnn.Tensor] = None,
        pwconv1_bias: Optional[ttnn.Tensor] = None,
        pwconv2_weight: Optional[ttnn.Tensor] = None,
        pwconv2_bias: Optional[ttnn.Tensor] = None,
        norm_weight: Optional[ttnn.Tensor] = None,
        norm_bias: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        # Use internal weights if loaded (TTNNModule pattern), else use passed-in tensors
        if getattr(self, "tt_dwconv_weight", None) is not None:
            dwconv_weight = self.tt_dwconv_weight
            dwconv_bias = self.tt_dwconv_bias
            norm_weight = self.tt_norm_weight
            norm_bias = self.tt_norm_bias
            pwconv1_weight = self.tt_pwconv1_weight
            pwconv1_bias = self.tt_pwconv1_bias
            pwconv2_weight = self.tt_pwconv2_weight
            pwconv2_bias = self.tt_pwconv2_bias
            gamma_tensor = self._tt_gamma
        else:
            gamma_tensor = ttnn.from_torch(
                self.gamma.unsqueeze(0).unsqueeze(0),  # (1, 1, C) for broadcast over (B, L, C)
                device=self.device,
            )
        if dwconv_weight is None or norm_weight is None or pwconv1_weight is None or pwconv2_weight is None:
            raise RuntimeError(
                "TtQwen3OmniMoeConvNeXtBlock: either load_state_dict + preprocess_weights + move_weights_to_device, "
                "or pass all weight tensors to forward()."
            )

        input_ = hidden_states

        # dwconv / ttnn.conv1d expect (B, C, L); block I/O is (B, L, C) like Code2Wav tensor layout
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
        hidden_states = self.dwconv.forward(hidden_states, dwconv_weight, dwconv_bias)
        hidden_states = ttnn.permute(hidden_states, (0, 2, 1))  # (B, C, L) -> (B, L, C)

        # Layer norm normalizes over last dim; keep (B, L, C) so last dim is C=1024
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=norm_weight,
            bias=norm_bias,
            **self.norm_config,
        )

        # Pointwise conv 1: 4x expansion
        hidden_states = ttnn.linear(
            hidden_states,
            pwconv1_weight,
            bias=pwconv1_bias,
            compute_kernel_config=self.compute_config,
        )

        # GELU activation
        hidden_states = ttnn.gelu(hidden_states)

        # Pointwise conv 2: projection back
        hidden_states = ttnn.linear(
            hidden_states,
            pwconv2_weight,
            bias=pwconv2_bias,
            compute_kernel_config=self.compute_config,
        )

        # Apply gamma scaling: ensure (1, 1, C) for broadcast over (B, L, C)
        if tuple(gamma_tensor.shape) != (1, 1, self.dim):
            gamma_tensor = ttnn.reshape(gamma_tensor, (1, 1, self.dim))
        hidden_states = ttnn.multiply(hidden_states, gamma_tensor)

        # Residual connection (hidden_states is already (B, L, C), same as input_)
        hidden_states = ttnn.add(input_, hidden_states)

        return hidden_states
