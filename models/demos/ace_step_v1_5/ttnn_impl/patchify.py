from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch.nn as nn

import ttnn


@dataclass(frozen=True)
class PatchifyMetadata:
    """Metadata needed to undo patchify padding/packing."""

    original_seq_len: int
    pad_length: int
    patch_size: int


def _pad_seq_len_to_patch_size(hidden_states: ttnn.Tensor, patch_size: int, *, value: float = 0.0) -> PatchifyMetadata:
    """
    Pads `hidden_states` along the sequence dimension so its length is divisible by `patch_size`.

    Expected shape: [B, T, C] (row-major).
    """
    if len(hidden_states.shape) != 3:
        raise ValueError(f"Expected hidden_states rank-3 [B, T, C], got shape={hidden_states.shape}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    original_seq_len = int(hidden_states.shape[1])
    remainder = original_seq_len % patch_size
    pad_length = 0 if remainder == 0 else (patch_size - remainder)

    if pad_length == 0:
        return PatchifyMetadata(original_seq_len=original_seq_len, pad_length=0, patch_size=patch_size)

    # ttnn.pad operates on rank-4 tensors with explicit (N, C, H, W) padding tuples.
    # Treat sequence length as H and channels as W: [B, 1, T, C]
    hs4 = ttnn.unsqueeze(hidden_states, 1)
    hs4 = ttnn.pad(hs4, padding=((0, 0), (0, 0), (0, pad_length), (0, 0)), value=value)
    hidden_states = ttnn.squeeze(hs4, 1)

    return PatchifyMetadata(original_seq_len=original_seq_len, pad_length=pad_length, patch_size=patch_size)


def _maybe_get_state_dict_key(state_dict: dict, candidates: Tuple[str, ...]) -> str:
    for k in candidates:
        if k in state_dict:
            return k
    raise KeyError(f"None of the candidate keys were found in state_dict: {candidates}")


class TtAceStepPatchEmbed1D(nn.Module):
    """
    TTNN equivalent of HF's `proj_in` patch embedding in `AceStepDiTModel`.

    HF reference (conceptually):
      x: [B, T, C]
      if T % patch_size != 0: pad T
      x = x.transpose(1,2)              # [B, C, T]
      x = Conv1d(C -> inner_dim, k=p, s=p)(x)  # [B, inner_dim, T/p]
      x = x.transpose(1,2)              # [B, T/p, inner_dim]

    TTNN conv1d expects input shaped [B, T, C], so we can call it directly without transposes.
    """

    def __init__(
        self,
        *,
        config,
        state_dict: dict,
        base_address: str,
        device: ttnn.Device,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2,
        math_approx_mode: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.patch_size = int(getattr(config, "patch_size"))
        self.in_channels = int(getattr(config, "in_channels"))
        self.out_channels = int(getattr(config, "hidden_size"))

        weight_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.weight",  # nn.Sequential(Lambda, Conv1d, Lambda)
                f"{base_address}.weight",
            ),
        )
        bias_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.bias",
                f"{base_address}.bias",
            ),
        )

        torch_weight = state_dict[weight_key]  # [out_channels, in_channels, patch_size]
        torch_bias = state_dict[bias_key]  # [out_channels]

        if torch_weight.shape[0] != self.out_channels:
            raise ValueError(
                f"Unexpected proj_in out_channels: got {torch_weight.shape[0]}, expected {self.out_channels}"
            )
        if torch_weight.shape[1] != self.in_channels:
            raise ValueError(
                f"Unexpected proj_in in_channels: got {torch_weight.shape[1]}, expected {self.in_channels}"
            )
        if int(torch_weight.shape[2]) != self.patch_size:
            raise ValueError(f"Unexpected proj_in kernel_size: got {torch_weight.shape[2]}, expected {self.patch_size}")

        # Keep weights/bias on host in TTNN tensor form; conv1d will materialize device copies as needed.
        self.weight = ttnn.from_torch(torch_weight, dtype=weights_dtype)
        self.bias = ttnn.from_torch(torch_bias.reshape(1, 1, 1, -1), dtype=weights_dtype)

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype, shard_layout=None, deallocate_activation=False
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.activation_dtype = activation_dtype

    def forward(self, hidden_states: ttnn.Tensor) -> Tuple[ttnn.Tensor, PatchifyMetadata]:
        if len(hidden_states.shape) != 3:
            raise ValueError(f"Expected hidden_states shape [B, T, C], got {hidden_states.shape}")
        if int(hidden_states.shape[-1]) != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got C={hidden_states.shape[-1]}")

        meta = _pad_seq_len_to_patch_size(hidden_states, self.patch_size, value=0.0)
        if meta.pad_length:
            # `_pad_seq_len_to_patch_size` returns meta; actual tensor was padded in-place via TTNN ops,
            # but we need to re-fetch the padded tensor. Recompute with padding and return it.
            hs4 = ttnn.unsqueeze(hidden_states, 1)
            hs4 = ttnn.pad(hs4, padding=((0, 0), (0, 0), (0, meta.pad_length), (0, 0)), value=0.0)
            hidden_states = ttnn.squeeze(hs4, 1)

        batch_size = int(hidden_states.shape[0])
        input_length = int(hidden_states.shape[1])

        [out, out_length, _] = ttnn.conv1d(
            input_tensor=hidden_states,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            batch_size=batch_size,
            input_length=input_length,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # conv1d returns rank-4 [1, B, out_length, out_channels]
        out = ttnn.squeeze(out, 0)
        out = ttnn.reshape(out, (batch_size, out_length, out.shape[-1]))
        return out, meta


class TtAceStepDePatchify1D(nn.Module):
    """
    TTNN equivalent of HF's `proj_out` de-patchify in `AceStepDiTModel`.

    HF reference (conceptually):
      x: [B, T_p, inner_dim]
      x = x.transpose(1,2)                       # [B, inner_dim, T_p]
      x = ConvTranspose1d(inner_dim -> out_ch, k=p, s=p)(x)  # [B, out_ch, T_p*p]
      x = x.transpose(1,2)                       # [B, T, out_ch]

    Because kernel_size == stride == patch_size and padding == 0, this ConvTranspose1d is non-overlapping and can be
    implemented as a per-token linear projection that expands each token into `patch_size` frames.
    """

    def __init__(
        self,
        *,
        config,
        state_dict: dict,
        base_address: str,
        device: ttnn.Device,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.patch_size = int(getattr(config, "patch_size"))
        self.in_channels = int(getattr(config, "hidden_size"))
        self.out_channels = int(getattr(config, "audio_acoustic_hidden_dim"))

        weight_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.weight",  # nn.Sequential(Lambda, ConvTranspose1d, Lambda)
                f"{base_address}.weight",
            ),
        )
        bias_key = _maybe_get_state_dict_key(
            state_dict,
            (
                f"{base_address}.1.bias",
                f"{base_address}.bias",
            ),
        )

        torch_weight = state_dict[weight_key]  # ConvTranspose1d weight: [in_channels, out_channels, patch_size]
        torch_bias = state_dict[bias_key]  # [out_channels]

        if int(torch_weight.shape[0]) != self.in_channels:
            raise ValueError(
                f"Unexpected proj_out in_channels: got {torch_weight.shape[0]}, expected {self.in_channels}"
            )
        if int(torch_weight.shape[1]) != self.out_channels:
            raise ValueError(
                f"Unexpected proj_out out_channels: got {torch_weight.shape[1]}, expected {self.out_channels}"
            )
        if int(torch_weight.shape[2]) != self.patch_size:
            raise ValueError(
                f"Unexpected proj_out kernel_size: got {torch_weight.shape[2]}, expected {self.patch_size}"
            )

        # Build a linear weight matrix W2d of shape [out_features, in_features] where out_features = out_channels*patch.
        # Ordering: for each out_channel, contiguous patch positions.
        # W2d[(out_ch * patch_size + i), :] = torch_weight[:, out_ch, i]
        w = torch_weight.permute(1, 2, 0).contiguous()  # [out_channels, patch_size, in_channels]
        w2d = w.view(self.out_channels * self.patch_size, self.in_channels)  # [out_features, in_features]

        self.weight = ttnn.from_torch(w2d, dtype=weights_dtype)
        self.bias = ttnn.from_torch(torch_bias.reshape(1, 1, 1, -1), dtype=activation_dtype)
        self.activation_dtype = activation_dtype

    def forward(self, hidden_states: ttnn.Tensor, meta: PatchifyMetadata) -> ttnn.Tensor:
        if meta.patch_size != self.patch_size:
            raise ValueError(
                f"Patch size mismatch: meta.patch_size={meta.patch_size} vs module.patch_size={self.patch_size}"
            )
        if len(hidden_states.shape) != 3:
            raise ValueError(f"Expected hidden_states shape [B, T_p, inner_dim], got {hidden_states.shape}")
        if int(hidden_states.shape[-1]) != self.in_channels:
            raise ValueError(f"Expected inner_dim={self.in_channels}, got {hidden_states.shape[-1]}")

        b = int(hidden_states.shape[0])
        t_p = int(hidden_states.shape[1])

        x2d = ttnn.reshape(hidden_states, (b * t_p, self.in_channels))
        y2d = ttnn.linear(
            x2d,
            self.weight,
            bias=None,
            transpose_b=True,  # y = x @ W^T
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        y = ttnn.reshape(y2d, (b, t_p, self.patch_size, self.out_channels))
        y = ttnn.reshape(y, (b, t_p * self.patch_size, self.out_channels))

        # Add bias per output channel (broadcast across batch and time)
        y4 = ttnn.unsqueeze(y, 1)  # [B, 1, T, C]
        y4 = y4 + self.bias
        y = ttnn.squeeze(y4, 1)

        # Trim any pad added during patchify
        if meta.pad_length:
            y = y[:, : meta.original_seq_len, :]
        return y
