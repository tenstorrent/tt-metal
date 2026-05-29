# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 audio mel-VAE decoder (Stage A).

Mirror of the torch reference ``AudioDecoder`` from
``LTX-2/packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py:276``.

The production LTX-2.3 22B distilled config consumes 8-channel latents
``(B, z=8, frames, mel_bins=64)`` and produces 2-channel log-mel spectrograms
``(B, out_ch=2, time, 64)``. The architecture:

- ``conv_in``: CausalConv2d k=3 (z → ch * ch_mult[-1])
- mid block: 2 ResnetBlocks (no attention; ``mid_block_add_attention=false``)
- upsampling path: ``num_resolutions`` levels reversed
  - each: ``num_res_blocks + 1`` ResnetBlocks (+ optional attention — never in
    production), followed by an Upsample on every level except the bottom
- ``norm_out`` (PixelNorm) → SiLU → ``conv_out`` (CausalConv2d k=3) → no tanh

All convs are CausalConv2d with ``causality_axis=HEIGHT`` (= front-pad on the
time axis). The pixel-norm path makes this a single-chip module — there is no
mesh-time work here, only a small ROW_MAJOR conv chain.

Layout: ``Conv2dViaConv3d`` operates on ``(B, H, W, C)``. The reference torch
code uses BCHW. We convert at the device boundary and convert back on exit.
"""

from __future__ import annotations

import einops
import torch

import ttnn

from ...layers.audio_ops import Conv2dViaConv3d
from ...layers.module import Module, ModuleList
from ...utils.conv3d import conv_pad_in_channels

# Mirrors LATENT_DOWNSAMPLE_FACTOR in audio_vae.py.
LATENT_DOWNSAMPLE_FACTOR = 4


# ---------------------------------------------------------------------------
# Host-side helpers (patchifier, per-channel statistics).
# ---------------------------------------------------------------------------


class LTXAudioPatchifier:
    """Host-side patchifier mirroring ``AudioPatchifier`` with ``patch_size=1``.

    With ``patch_size=1`` and ``audio_latent_downsample_factor=4`` the patchify
    is a simple flatten over (channels, mel_bins):

    - ``patchify``:   ``(B, C, T, F) → (B, T, C*F)``
    - ``unpatchify``: ``(B, T, C*F) → (B, C, T, F)`` given the original shape

    The per-channel statistics in ``PerChannelStatistics.un_normalize`` are
    applied to the patchified tensor (so they broadcast across ``T`` and ``F``
    rather than just channels) before unpatchifying back. We mirror this exactly
    on the host so the latent denormalization path matches the reference bit-
    exact.
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
        """``(B, C, T, F) → (B, T, C*F)``."""
        return einops.rearrange(audio_latents, "b c t f -> b t (c f)")

    def unpatchify(self, audio_latents: torch.Tensor, channels: int, mel_bins: int) -> torch.Tensor:
        """``(B, T, C*F) → (B, C, T, F)`` given the target channel/mel counts."""
        return einops.rearrange(audio_latents, "b t (c f) -> b c t f", c=channels, f=mel_bins)


# ---------------------------------------------------------------------------
# On-device modules.
# ---------------------------------------------------------------------------


class LTXAudioPixelNorm(Module):
    """Per-pixel RMS normalization for BHWC tensors.

    ``y = x / sqrt(mean(x², dim=-1, keepdim=True) + eps)`` — same maths as the
    reference ``PixelNorm`` (which uses ``dim=1`` in BCHW); in BHWC the channel
    dim is the last axis. No learned parameters. Defaults match the reference
    ``build_normalization_layer`` (eps=1e-6).
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        x_sq = ttnn.multiply(x_BHWC, x_BHWC)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_sq, self.eps))
        return ttnn.multiply(x_BHWC, ttnn.reciprocal(rms))


class LTXAudioResnetBlock(Module):
    """LTX-2 audio mel-VAE Resnet block.

    Forward:
        h = norm1(x); h = silu(h); h = conv1(h)
        h = norm2(h); h = silu(h); h = conv2(h)
        if in != out: x = nin_shortcut(x)  # CausalConv2d k=1
        return x + h

    No temb branch (production has ``temb_channels=0``). No dropout.
    """

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

        self.norm1 = LTXAudioPixelNorm()
        self.conv1 = Conv2dViaConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding_mode="causal_height",
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.norm2 = LTXAudioPixelNorm()
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
        # Strip non-parametric submodules (non_linearity, dropout) and the
        # absent shortcut keys when channels match.
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

        # add tolerates layout mismatch; pull both to TILE for the add since
        # conv output is ROW_MAJOR and ttnn.add infers layout from the inputs.
        return ttnn.add(residual, h)


class LTXAudioUpsample(Module):
    """Nearest-neighbour 2× upsample + causal conv + drop first row.

    Mirrors ``Upsample`` in upsample.py with ``causality_axis=HEIGHT``:

      x = nearest(x, 2)          # (B, H, W, C) → (B, 2H, 2W, C)
      x = causal_conv2d_k3(x)    # H-causal pad on the front
      x = x[:, 1:, :, :]         # drop the leading H row

    Always uses ``with_conv=True`` (production setting).
    """

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

        # Drop the first H row to undo the causal padding (see Upsample docstring
        # in the reference: this keeps output length 1 + 2 * n_in).
        # ttnn slicing on the row-major dim is supported.
        x_BHWC = x_BHWC[:, 1:, :, :]
        return x_BHWC


class _MidBlock(Module):
    """Mid block: two Resnet blocks, no attention.

    The torch reference structures this as ``mid.block_1`` / ``mid.attn_1`` /
    ``mid.block_2`` where ``attn_1 = nn.Identity()`` when
    ``mid_block_add_attention=false``. We mirror the same attribute names so
    state-dict keys match.
    """

    def __init__(
        self,
        *,
        channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.block_1 = LTXAudioResnetBlock(
            in_channels=channels,
            out_channels=channels,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        # NB: attn_1 is nn.Identity() in production — no params, no submodule
        # registered here. _prepare_torch_state strips any "attn_1.*" keys
        # (there should be none for production weights).
        self.block_2 = LTXAudioResnetBlock(
            in_channels=channels,
            out_channels=channels,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # attn_1 is nn.Identity() — should have no params, but defensively
        # strip the prefix.
        keys_to_remove = [k for k in state if k.startswith("attn_1")]
        for k in keys_to_remove:
            del state[k]

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        x_BHWC = self.block_1(x_BHWC)
        x_BHWC = self.block_2(x_BHWC)
        return x_BHWC


class _UpStage(Module):
    """One level of the upsampling path: ``block`` (ModuleList) + optional ``upsample``.

    Mirrors the per-level ``stage`` object created by ``build_upsampling_path``.
    The torch reference uses ``stage.block`` / ``stage.attn`` (always empty in
    production) / ``stage.upsample``. We omit ``stage.attn`` entirely (no params
    in production).
    """

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

        # ``num_res_blocks_in_stage`` is ``num_res_blocks + 1`` per the
        # reference's loop range. First block does the channel transition;
        # the rest are out_channels → out_channels.
        self.block = ModuleList()
        cur_in = in_channels
        for _ in range(num_res_blocks_in_stage):
            self.block.append(
                LTXAudioResnetBlock(
                    in_channels=cur_in,
                    out_channels=out_channels,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
            )
            cur_in = out_channels

        self.has_upsample = has_upsample
        if has_upsample:
            self.upsample = LTXAudioUpsample(
                in_channels=out_channels,
                mesh_device=mesh_device,
                dtype=dtype,
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # No attn in production — strip prophylactically.
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


# ---------------------------------------------------------------------------
# Top-level decoder.
# ---------------------------------------------------------------------------


class LTXAudioDecoder(Module):
    """LTX-2 audio mel-VAE decoder.

    Constructor mirrors the reference ``AudioDecoder.__init__`` keyword
    arguments. The state-dict layout matches the torch reference so
    ``load_torch_state_dict(torch_decoder.state_dict())`` works without
    remapping.

    ``forward`` accepts a torch tensor ``(B, z_channels, frames, mel_bins)``
    and returns a torch tensor ``(B, out_ch, T_target, mel_bins)`` per the
    reference's ``_denormalize_latents`` / ``_adjust_output_shape``.
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
                "LTXAudioDecoder Stage A does not support attention blocks; " f"got attn_resolutions={attn_resolutions}"
            )
        if mid_block_add_attention:
            raise NotImplementedError("LTXAudioDecoder Stage A does not support mid_block_add_attention=True")

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

        # Per-channel statistics: stored on host. The reference registers buffers
        # under the *exact* names ``std-of-means`` / ``mean-of-means`` (note the
        # hyphens — these are not valid Python identifiers). We hold them as
        # plain torch tensors and consume them in _denormalize_latents.
        self._stats_std = torch.empty(ch)
        self._stats_mean = torch.empty(ch)

        # Host-side patchifier (identical to reference AudioPatchifier).
        self.patchifier = LTXAudioPatchifier(
            patch_size=1,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            is_causal=is_causal,
        )

        base_block_channels = ch * self.ch_mult[-1]

        # conv_in: z → base_block_channels, CausalConv2d k=3
        self.conv_in = Conv2dViaConv3d(
            z_channels,
            base_block_channels,
            kernel_size=3,
            stride=1,
            padding_mode="causal_height",
            mesh_device=mesh_device,
            dtype=dtype,
        )

        # mid block (registered as `mid` to match the torch reference)
        self.mid = _MidBlock(
            channels=base_block_channels,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        # Upsampling path. The reference builds `up` as a ModuleList of length
        # ``num_resolutions`` indexed by the *original* (un-reversed) level
        # number, then iterates in reverse. We mirror that exactly so the
        # state-dict path ``up.<level>.block.<idx>.*`` matches.
        self.up = ModuleList()
        block_in = base_block_channels
        for level in range(self.num_resolutions):
            self.up.append(
                _UpStage(
                    in_channels=0,  # placeholder, will be overwritten below
                    out_channels=0,
                    num_res_blocks_in_stage=0,
                    has_upsample=False,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
            )
        # Now rebuild in reverse order to track block_in through the levels.
        for level in reversed(range(self.num_resolutions)):
            block_out = ch * self.ch_mult[level]
            has_upsample = level != 0
            stage = _UpStage(
                in_channels=block_in,
                out_channels=block_out,
                num_res_blocks_in_stage=self.num_res_blocks + 1,
                has_upsample=has_upsample,
                mesh_device=mesh_device,
                dtype=dtype,
            )
            # Replace the placeholder at index `level`.
            self.up._children[str(level)] = stage
            block_in = block_out

        final_block_channels = block_in

        # norm_out and conv_out
        self.norm_out = LTXAudioPixelNorm()
        self.conv_out = Conv2dViaConv3d(
            final_block_channels,
            out_ch,
            kernel_size=3,
            stride=1,
            padding_mode="causal_height",
            mesh_device=mesh_device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # State-dict prep: strip / remap the torch reference's non-Module keys.
    # ------------------------------------------------------------------

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # ``per_channel_statistics`` is registered as buffers on a separate
        # nn.Module in the reference. We consume the two buffers on host.
        # Keys: ``per_channel_statistics.std-of-means``, ``...mean-of-means``.
        std_key = "per_channel_statistics.std-of-means"
        mean_key = "per_channel_statistics.mean-of-means"
        if std_key in state:
            self._stats_std = state.pop(std_key).detach().clone()
        if mean_key in state:
            self._stats_mean = state.pop(mean_key).detach().clone()

        # non_linearity is nn.SiLU() — no params.
        keys_to_remove = [k for k in state if k.startswith("non_linearity")]
        for k in keys_to_remove:
            del state[k]

    # ------------------------------------------------------------------
    # Host-side latent denormalize / output shape adjustment.
    # ------------------------------------------------------------------

    def _denormalize_latents(self, sample: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Apply ``un_normalize`` on the patchified latents, then unpatchify.

        Returns the denormalized BCTF tensor *and* the target output shape
        ``(B, out_ch, target_frames, target_mel_bins)``.
        """
        B, C, T, F = sample.shape

        # Patchify: (B, C, T, F) → (B, T, C*F)
        patched = self.patchifier.patchify(sample)

        # Per-channel un-normalize: ``x * std + mean``. The reference stores
        # the stats as 1D tensors that must broadcast against the last dim of
        # the patched tensor (``B, T, C*F``). The exact length depends on the
        # checkpoint:
        #   - production checkpoints store stats of length C (and rely on
        #     ``C*F == C`` — i.e. mel_bins == 1 or patch_size == mel_bins)
        #   - the unit test overrides them at length C*F so the math actually
        #     works regardless of mel_bins
        # We accept both: if stats length matches the patched last dim use
        # them directly; otherwise broadcast over mel_bins via
        # ``repeat_interleave``.
        std = self._stats_std.to(patched.dtype).to(patched.device)
        mean = self._stats_mean.to(patched.dtype).to(patched.device)
        last_dim = patched.shape[-1]
        if std.shape[0] == last_dim:
            pass  # direct broadcast
        elif std.shape[0] == C and last_dim == C * F:
            std = std.repeat_interleave(F)
            mean = mean.repeat_interleave(F)
        else:
            raise ValueError(
                f"per-channel stats shape {tuple(std.shape)} does not broadcast against "
                f"patched last dim {last_dim} (C={C}, F={F})"
            )
        sample_denormalized = patched * std + mean

        # Unpatchify back to (B, C, T, F)
        sample = self.patchifier.unpatchify(sample_denormalized, channels=C, mel_bins=F)

        # Target shape per reference:
        target_frames = T * LATENT_DOWNSAMPLE_FACTOR
        if self.is_causal:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)
        target_mel_bins = self.mel_bins if self.mel_bins is not None else F
        target_shape = (B, self.out_ch, target_frames, target_mel_bins)

        return sample, target_shape

    @staticmethod
    def _adjust_output_shape(decoded_output: torch.Tensor, target_shape: tuple[int, int, int, int]) -> torch.Tensor:
        """Crop / pad the decoded output to the exact target shape (mirrors ref)."""
        _, _, current_time, current_freq = decoded_output.shape
        _, target_channels, target_time, target_freq = target_shape

        # Step 1: crop
        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        # Step 2: pad if needed
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

        # Step 3: final safety crop
        return decoded_output[:, :target_channels, :target_time, :target_freq]

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """``latent``: torch tensor ``(B, z_channels, frames, mel_bins)``.

        Returns torch tensor ``(B, out_ch, target_frames, target_mel_bins)``.
        """
        # Host: denormalize.
        sample_bcfk, target_shape = self._denormalize_latents(latent)

        # Convert to BHWC for the device. ``F`` (mel_bins) is the W axis.
        # (B, C=z, T=H, F=W) → (B, H, W, C)
        sample_bhwc = sample_bcfk.permute(0, 2, 3, 1).contiguous().to(torch.float32)

        # Pad C up to the conv3d alignment if needed (Conv2dViaConv3d pads
        # in_channels internally for its weight but the *input tensor* must
        # carry the same padded channel count).
        sample_bhwc_padded = conv_pad_in_channels(sample_bhwc)

        x = ttnn.from_torch(
            sample_bhwc_padded, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )

        # conv_in
        x = self.conv_in(x)

        # mid block
        x = self.mid(x)

        # upsampling path (reversed iteration)
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            x = stage(x)

        # norm_out → SiLU → conv_out
        x = self.norm_out(x)
        x = ttnn.silu(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT) if x.layout != ttnn.ROW_MAJOR_LAYOUT else x
        x = self.conv_out(x)
        # ``tanh_out=False`` in production — no tanh.

        # Pull off device.
        out_bhwc = ttnn.to_torch(ttnn.get_device_tensors(x)[0])

        # Strip channel padding introduced by Conv2dViaConv3d.out_channels
        # rounding (out_ch=2 gets padded up to 32). Then BHWC → BCHW.
        out_bhwc = out_bhwc[..., : self.out_ch]
        out_bchw = out_bhwc.permute(0, 3, 1, 2).contiguous()

        # Host: adjust output shape.
        return self._adjust_output_shape(out_bchw, target_shape)
