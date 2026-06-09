# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 audio mel-VAE decoder (Stage A).

Conv2dViaConv3d operates on ``(B, H, W, C)`` (mel_bins is W); the torch reference
is BCHW, so inputs convert at the device boundary and back on exit. All convs are
causal on the height (time) axis. Single-chip.
"""

from __future__ import annotations

import einops
import torch

import ttnn

from ...layers.audio_ops import Conv2dViaConv3d
from ...layers.module import Module, ModuleList
from ...utils.conv3d import conv_pad_in_channels

LATENT_DOWNSAMPLE_FACTOR = 4


class AudioPatchifier:
    """Host-side patchifier with ``patch_size=1``: flatten over (channels, mel_bins).

    Per-channel un-normalize stats are applied to the patchified tensor so they
    broadcast across T and F, not just channels.
    """

    def __init__(
        self,
        *,
        patch_size: int = 1,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = LATENT_DOWNSAMPLE_FACTOR,
        is_causal: bool = True,
        shift: int = 0,
    ) -> None:
        self.patch_size = patch_size
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift

    def patchify(self, audio_latents: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(audio_latents, "b c t f -> b t (c f)")

    def unpatchify(self, audio_latents: torch.Tensor, channels: int, mel_bins: int) -> torch.Tensor:
        return einops.rearrange(audio_latents, "b t (c f) -> b c t f", c=channels, f=mel_bins)


class PixelNorm(Module):
    """Per-pixel RMS normalization over the channel (last) axis. No learned params."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        x_sq = ttnn.multiply(x_BHWC, x_BHWC)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_sq, self.eps))
        return ttnn.multiply(x_BHWC, ttnn.reciprocal(rms))


class ResnetBlock(Module):
    """LTX-2 audio mel-VAE Resnet block. No temb branch, no dropout."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_shortcut = in_channels != out_channels

        self.norm1 = PixelNorm()
        self.conv1 = Conv2dViaConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding_mode="causal_height",
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.norm2 = PixelNorm()
        self.conv2 = Conv2dViaConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding_mode="causal_height",
            mesh_device=mesh_device,
            dtype=dtype,
        )
        if self.has_shortcut:
            self.nin_shortcut = Conv2dViaConv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding_mode="causal_height",
                mesh_device=mesh_device,
                dtype=dtype,
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        keys_to_remove = [k for k in state if k.startswith("non_linearity") or k.startswith("dropout")]
        for k in keys_to_remove:
            del state[k]

        if not self.has_shortcut:
            keys_to_remove = [k for k in state if k.startswith("nin_shortcut") or k.startswith("conv_shortcut")]
            for k in keys_to_remove:
                del state[k]

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        residual = x_BHWC

        h = self.norm1(x_BHWC)
        h = ttnn.silu(h)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT) if h.layout != ttnn.ROW_MAJOR_LAYOUT else h
        h = self.conv1(h)

        h = self.norm2(h)
        h = ttnn.silu(h)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT) if h.layout != ttnn.ROW_MAJOR_LAYOUT else h
        h = self.conv2(h)

        if self.has_shortcut:
            residual = (
                ttnn.to_layout(residual, ttnn.ROW_MAJOR_LAYOUT)
                if residual.layout != ttnn.ROW_MAJOR_LAYOUT
                else residual
            )
            residual = self.nin_shortcut(residual)

        return ttnn.add(residual, h)


class AudioUpsample(Module):
    """Nearest-neighbour 2x upsample + causal conv + drop leading H row."""

    def __init__(
        self,
        *,
        in_channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv = Conv2dViaConv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding_mode="causal_height",
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        # ttnn.upsample requires ROW_MAJOR input and produces ROW_MAJOR output.
        x_BHWC = ttnn.to_layout(x_BHWC, ttnn.ROW_MAJOR_LAYOUT) if x_BHWC.layout != ttnn.ROW_MAJOR_LAYOUT else x_BHWC
        x_BHWC = ttnn.upsample(x_BHWC, scale_factor=2)
        x_BHWC = self.conv(x_BHWC)

        # Drop the leading H row to undo the causal pad: output length is 1 + 2*n_in.
        x_BHWC = x_BHWC[:, 1:, :, :]
        return x_BHWC


class _MidBlock(Module):
    """Mid block: two Resnet blocks, no attention."""

    def __init__(
        self,
        *,
        channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.block_1 = ResnetBlock(
            in_channels=channels,
            out_channels=channels,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.block_2 = ResnetBlock(
            in_channels=channels,
            out_channels=channels,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        keys_to_remove = [k for k in state if k.startswith("attn_1")]
        for k in keys_to_remove:
            del state[k]

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        x_BHWC = self.block_1(x_BHWC)
        x_BHWC = self.block_2(x_BHWC)
        return x_BHWC


class _UpStage(Module):
    """One upsampling level: ``block`` (ModuleList) + optional ``upsample``."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_res_blocks_in_stage: int,
        has_upsample: bool,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        # First block does the channel transition; the rest are out → out.
        self.block = ModuleList()
        cur_in = in_channels
        for _ in range(num_res_blocks_in_stage):
            self.block.append(
                ResnetBlock(
                    in_channels=cur_in,
                    out_channels=out_channels,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
            )
            cur_in = out_channels

        self.has_upsample = has_upsample
        if has_upsample:
            self.upsample = AudioUpsample(
                in_channels=out_channels,
                mesh_device=mesh_device,
                dtype=dtype,
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        keys_to_remove = [k for k in state if k.startswith("attn.")]
        for k in keys_to_remove:
            del state[k]
        if not self.has_upsample:
            keys_to_remove = [k for k in state if k.startswith("upsample.")]
            for k in keys_to_remove:
                del state[k]

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        for block in self.block:
            x_BHWC = block(x_BHWC)
        if self.has_upsample:
            x_BHWC = self.upsample(x_BHWC)
        return x_BHWC


class AudioDecoder(Module):
    """LTX-2 audio mel-VAE decoder.

    ``forward`` accepts a torch tensor ``(B, z_channels, frames, mel_bins)`` and
    returns ``(B, out_ch, T_target, mel_bins)``.
    """

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int,
        attn_resolutions: set[int] | tuple[int, ...] = (),
        resolution: int,
        z_channels: int,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = None,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        if attn_resolutions:
            raise NotImplementedError(
                "AudioDecoder Stage A does not support attention blocks; " f"got attn_resolutions={attn_resolutions}"
            )
        if mid_block_add_attention:
            raise NotImplementedError("AudioDecoder Stage A does not support mid_block_add_attention=True")

        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(self.ch_mult)
        self.resolution = resolution
        self.z_channels = z_channels
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.mesh_device = mesh_device
        self.dtype = dtype

        # Per-channel denormalize stats: non-Parameter host tensors consumed in
        # _denormalize_latents. Default to identity (std=1, mean=0) so an unset
        # path is a harmless passthrough rather than uninitialized garbage.
        self._stats_std = torch.ones(ch)
        self._stats_mean = torch.zeros(ch)

        self.patchifier = AudioPatchifier(
            patch_size=1,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            is_causal=is_causal,
        )

        base_block_channels = ch * self.ch_mult[-1]

        self.conv_in = Conv2dViaConv3d(
            z_channels,
            base_block_channels,
            kernel_size=3,
            stride=1,
            padding_mode="causal_height",
            mesh_device=mesh_device,
            dtype=dtype,
        )

        self.mid = _MidBlock(
            channels=base_block_channels,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        # Build high→low resolution so channel counts propagate, storing by level;
        # then append in level order so state-dict keys ``up.<level>.*`` match
        # (forward iterates these in reverse).
        stages: list[_UpStage] = [None] * self.num_resolutions
        block_in = base_block_channels
        for level in reversed(range(self.num_resolutions)):
            block_out = ch * self.ch_mult[level]
            stages[level] = _UpStage(
                in_channels=block_in,
                out_channels=block_out,
                num_res_blocks_in_stage=self.num_res_blocks + 1,
                has_upsample=level != 0,
                mesh_device=mesh_device,
                dtype=dtype,
            )
            block_in = block_out
        self.up = ModuleList()
        for stage in stages:
            self.up.append(stage)

        final_block_channels = block_in

        self.norm_out = PixelNorm()
        self.conv_out = Conv2dViaConv3d(
            final_block_channels,
            out_ch,
            kernel_size=3,
            stride=1,
            padding_mode="causal_height",
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # per_channel_statistics buffers carry hyphens (not valid Python idents),
        # so consume them on host rather than registering them as children.
        std_key = "per_channel_statistics.std-of-means"
        mean_key = "per_channel_statistics.mean-of-means"
        if std_key in state:
            self._stats_std = state.pop(std_key).detach().clone()
        if mean_key in state:
            self._stats_mean = state.pop(mean_key).detach().clone()

        keys_to_remove = [k for k in state if k.startswith("non_linearity")]
        for k in keys_to_remove:
            del state[k]

    def set_per_channel_stats(self, std: torch.Tensor, mean: torch.Tensor) -> None:
        """Re-inject the per-channel denormalize stats after a binary cache load.

        These are non-Parameter host buffers; the weight cache carries no
        non-Parameter state, so without re-injection the denormalize multiplies
        by uninitialised memory.
        """
        self._stats_std = std.detach().clone().float()
        self._stats_mean = mean.detach().clone().float()

    def _denormalize_latents(self, sample: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Apply ``un_normalize`` on the patchified latents, then unpatchify.

        Returns the denormalized BCTF tensor *and* the target output shape
        ``(B, out_ch, target_frames, target_mel_bins)``.
        """
        B, C, T, F = sample.shape

        patched = self.patchifier.patchify(sample)

        # Stats length depends on the checkpoint: either C*F (direct broadcast)
        # or C (broadcast over mel_bins via repeat_interleave).
        std = self._stats_std.to(patched.dtype).to(patched.device)
        mean = self._stats_mean.to(patched.dtype).to(patched.device)
        last_dim = patched.shape[-1]
        if std.shape[0] == last_dim:
            pass
        elif std.shape[0] == C and last_dim == C * F:
            std = std.repeat_interleave(F)
            mean = mean.repeat_interleave(F)
        else:
            raise ValueError(
                f"per-channel stats shape {tuple(std.shape)} does not broadcast against "
                f"patched last dim {last_dim} (C={C}, F={F})"
            )
        sample_denormalized = patched * std + mean

        sample = self.patchifier.unpatchify(sample_denormalized, channels=C, mel_bins=F)

        target_frames = T * LATENT_DOWNSAMPLE_FACTOR
        if self.is_causal:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)
        target_mel_bins = self.mel_bins if self.mel_bins is not None else F
        target_shape = (B, self.out_ch, target_frames, target_mel_bins)

        return sample, target_shape

    @staticmethod
    def _adjust_output_shape(decoded_output: torch.Tensor, target_shape: tuple[int, int, int, int]) -> torch.Tensor:
        """Crop / pad the decoded output to the exact target shape."""
        _, _, current_time, current_freq = decoded_output.shape
        _, target_channels, target_time, target_freq = target_shape

        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]
        if time_padding_needed > 0 or freq_padding_needed > 0:
            padding = (
                0,
                max(freq_padding_needed, 0),
                0,
                max(time_padding_needed, 0),
            )
            decoded_output = torch.nn.functional.pad(decoded_output, padding)

        return decoded_output[:, :target_channels, :target_time, :target_freq]

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """``latent``: ``(B, z_channels, frames, mel_bins)`` →
        ``(B, out_ch, target_frames, target_mel_bins)``.
        """
        sample_bcfk, target_shape = self._denormalize_latents(latent)

        # To BHWC for the device (mel_bins is W).
        sample_bhwc = sample_bcfk.permute(0, 2, 3, 1).contiguous().to(torch.float32)

        # The input tensor must carry the same padded channel count as the
        # Conv2dViaConv3d weight.
        sample_bhwc_padded = conv_pad_in_channels(sample_bhwc)

        x = ttnn.from_torch(
            sample_bhwc_padded, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )

        x = self.conv_in(x)
        x = self.mid(x)

        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            x = stage(x)

        x = self.norm_out(x)
        x = ttnn.silu(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT) if x.layout != ttnn.ROW_MAJOR_LAYOUT else x
        x = self.conv_out(x)

        out_bhwc = ttnn.to_torch(ttnn.get_device_tensors(x)[0])

        # Strip out_channels padding from Conv2dViaConv3d, then BHWC → BCHW.
        out_bhwc = out_bhwc[..., : self.out_ch]
        out_bchw = out_bhwc.permute(0, 3, 1, 2).contiguous()

        return self._adjust_output_shape(out_bchw, target_shape)
