# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 audio VAE decode: a 1x1 latent projection into a BigVGAN vocoder.

Almost all of this already exists. ``audio_vae/vocoder_ltx.py``'s ``Vocoder`` is a
BigVGAN-v2 AMP1 stack whose ``forward`` is *identical* to H3's reference
``MiniMaxH3AudioBigVGANDecoder.forward`` -- ``conv_pre``, then per stage
``ups[i]`` followed by the mean of three parallel AMP blocks, then ``act_post``,
``conv_post`` and ``clamp(-1, 1)`` -- including the row-major
``resblocks[i * num_kernels + j]`` ordering. Every loop in it is generic over
``len(upsample_rates)``, so H3's **seven** stages need only configuration, not code.

The channel schedule falls out of the same ``upsample_initial_channel // 2**i`` rule:
1024 -> 512, 256, 128, 64, 32, 16, **8**, and ``conv_post`` takes that 8 to 1 mono
channel. Stereo is carried as batch 2, so nothing here is stereo-aware.

What this module adds is only:

* ``dec_in_proj``, the 1x1 conv from 32 latent channels to the vocoder's 2048;
* the wiring that keeps its output on device instead of round-tripping to host between
  the projection and the vocoder.

Naming is chosen so the converted checkpoint loads with no fixups: the vocoder is held as
a child called ``decoder``, matching ``decoder.conv_pre`` / ``decoder.ups.N`` /
``decoder.resblocks.N`` / ``decoder.act_post`` / ``decoder.conv_post``, and the projection
is ``dec_in_proj``. The ``ups.{i}.0 -> ups.{i}`` and ``activations -> acts1/acts2`` remaps,
and the weight-norm fusion, all happen in ``convert_minimax_h3_audio.py``.

fp32 throughout: ``vocoder_ltx``'s docstring records that bf16 accumulation measurably
degrades spectral metrics through its 108-conv chain, and H3's is longer still.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.audio_ops import DEFAULT_MAX_C_IN_BLOCK, _AlignedOutConv1d
from ....layers.module import Module
from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.tensor import local_device_to_torch
from ..vocoder_ltx import Vocoder
from .blockings_minimax_h3_audio import register_h3_audio_blockings

TILE_HEIGHT = 32


class MiniMaxH3AudioDecoder(Module):
    """``(B, 32, T)`` latents to ``(B, 1, T * 800)`` mono waveform at 32 kHz."""

    def __init__(
        self,
        *,
        latent_channels: int = 32,
        latent_dim: int = 2048,
        decoder_dim: int = 1024,
        decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2),
        decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
        split_mode: str = "full",
        tap_matmul: bool = True,
        prefer_mac: bool = True,
        max_c_in_block: int = DEFAULT_MAX_C_IN_BLOCK,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim
        self.hop_length = 1
        for rate in decoder_rates:
            self.hop_length *= rate

        # The precision levers default to accurate: all three on measures 0.0045 rel RMSE /
        # 99.9990 % PCC / 67.53 dB at 5 s stereo against 0.1046 / 99.5451 % / 40.29 dB all-fast --
        # 23x less error for ~3x on the stage. H3-only: LTX constructs the same conv classes with
        # its own fast defaults. Kept as attributes so the pipeline's device-weight cache key
        # (`weights_variant`) reads the exact values this module was built with.
        self.split_mode = split_mode
        self.tap_matmul = tap_matmul
        self.prefer_mac = prefer_mac
        self.max_c_in_block = max_c_in_block

        # H3's audio channel schedule differs from LTX's at both ends, so every conv misses
        # _FP32_BLOCKINGS. Seed stubs before any conv is built; see that module for why stubs.
        register_h3_audio_blockings(max_c_in_block=max_c_in_block)

        # k1 conv, so no padding mode to get wrong. _AlignedOutConv1d rather than the base
        # class per its own docstring: a non-32-multiple out count reaches conv3d and
        # produces a buffer whose page size does not divide its length.
        self.dec_in_proj = _AlignedOutConv1d(
            latent_channels,
            latent_dim,
            kernel_size=1,
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            split_mode=split_mode,
            tap_matmul=tap_matmul,
        )
        self.decoder = Vocoder(
            resblock_kernel_sizes=list(resblock_kernel_sizes),
            upsample_rates=list(decoder_rates),
            upsample_kernel_sizes=list(decoder_kernel_sizes),
            resblock_dilation_sizes=[list(d) for d in resblock_dilation_sizes],
            upsample_initial_channel=decoder_dim,
            in_channels=latent_dim,
            out_channels=1,
            use_tanh_at_final=False,
            apply_final_activation=True,
            use_bias_at_final=False,  # H3's conv_post has no bias
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            # H3-only opt-in: LTX's vocoder keeps the default fast conv1d filters and
            # single-conv weights.
            prefer_mac=prefer_mac,
            split_mode=split_mode,
            tap_matmul=tap_matmul,
        )

    def _project_latents_device(self, latents_BCT: torch.Tensor) -> torch.Tensor:
        """``(B, 32, T)`` -> ``(B, 2048, T)`` through ``dec_in_proj`` on device.

        Torch in, torch out, matching the stage convention ``bwe_ltx.VocoderWithBWE``
        uses (``_compute_mel_device``, ``_resample_device``): each stage owns its own
        upload and readback, which keeps the stages independently testable.
        """
        x = latents_BCT.transpose(1, 2).float().contiguous()  # (B, T, C)
        t_pad = self._t_padding(x.shape[1])
        if t_pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, t_pad))

        x_device = ttnn.from_torch(x, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.dtype)
        projected_device = self.dec_in_proj(x_device)
        projected = local_device_to_torch(projected_device).float()
        if t_pad:
            # Crop the alignment padding back off before handing T to the vocoder, which
            # applies its own padding for its own sharding.
            projected = projected[:, : x.shape[1] - t_pad]
        return projected.transpose(1, 2).contiguous()  # (B, 2048, T)

    def forward(self, latents_BCT: torch.Tensor, *, traced: bool = False) -> torch.Tensor:
        """``(B, latent_channels, T)`` torch in, ``(B, 1, T * hop_length)`` torch out.

        ``traced`` replays a captured device graph for the vocoder instead of dispatching it
        op by op. The vocoder is ~70 % host-bound, so this is its dominant lever -- unlike the
        *visual* halves, which measure 1.00x traced because they are device-bound. Needs a
        ``trace_region_size`` on the mesh device; the first call at a shape captures, later
        calls replay.
        """
        _, channels, _ = latents_BCT.shape
        assert channels == self.latent_channels, f"expected {self.latent_channels} latent channels, got {channels}"
        projected = self._project_latents_device(latents_BCT)
        return self.decoder.forward_BCT_traced(projected) if traced else self.decoder.forward_BCT(projected)

    def _t_padding(self, num_frames: int) -> int:
        """T padding needed for tile-aligned per-chip shards; zero when unsharded."""
        if self.parallel_config is None or self.parallel_config.factor <= 1:
            return 0
        align = TILE_HEIGHT * self.parallel_config.factor
        return (-num_frames) % align

    def release_trace(self) -> None:
        """Free the vocoder's captured traces; safe when none is active."""
        self.decoder.release_trace()
