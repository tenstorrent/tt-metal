# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 ``HifiDecoder``: on-device latent upsampling + HiFi-GAN generator.

The reference does two ``F.interpolate(mode="linear")`` steps on the GPT latents.
ttnn has no linear-1D interpolate, so both steps are folded into one fixed
``[L_out, T]`` resample matrix (``M2 @ M1`` — linear ops compose exactly) and
applied as a single on-device matmul along time. The matrix depends only on the
sequence length (not on the activations), so it is a per-length constant, not a
torch op in the data path.

Speaker embedding ``g`` is supplied (the speaker encoder is a later phase).
"""

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_hifi_decoder import (
    LATENT_SCALE,
    SR_SCALE,
    build_linear_interp_matrix,
)
from models.experimental.xtts.tt.xtts_hifigan import TtHifiganGenerator


class TtLatentUpsampler(LightweightModule):
    """Linear time-upsample of channels-last latents ``[1, T, C]`` -> ``[1, L_out, C]``
    via a composed resample matmul, matching the reference's two F.interpolate steps."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self._matrix_cache = {}  # T -> device resample matrix [L_out, T]

    def _resample_matrix(self, length_in: int) -> ttnn.Tensor:
        if length_in not in self._matrix_cache:
            m1 = build_linear_interp_matrix(length_in, LATENT_SCALE)  # [4T, T]
            m2 = build_linear_interp_matrix(m1.shape[0], SR_SCALE)  # [L_out, 4T]
            matrix = m2 @ m1  # [L_out, T]
            self._matrix_cache[length_in] = ttnn.from_torch(
                matrix, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32
            )
        return self._matrix_cache[length_in]

    def forward(self, x_blc: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, length_in, channels = x_blc.shape
        assert batch_size == 1, "latent upsampler assumes batch size 1"
        matrix = self._resample_matrix(length_in)  # [L_out, T]

        x = ttnn.to_layout(x_blc, ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, [length_in, channels])  # drop batch for the 2D matmul
        y = ttnn.matmul(matrix, x)  # [L_out, C]
        length_out = matrix.shape[0]
        y = ttnn.reshape(y, [1, length_out, channels])
        return ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)


class TtHifiDecoder(LightweightModule):
    """XTTS-v2 HifiDecoder: latent upsample -> HiFi-GAN generator.

    ``latents`` is channels-last ``[1, T, 1024]``, ``g`` is ``[1, 1, 512]``.
    Output is ``[1, L_out*256, 1]`` (the waveform).
    """

    def __init__(self, device, state_dict):
        super().__init__()
        self.upsampler = TtLatentUpsampler(device)
        self.generator = TtHifiganGenerator(device, state_dict)

    def forward(self, latents, g):
        return self.generator(self.upsampler(latents), g)
