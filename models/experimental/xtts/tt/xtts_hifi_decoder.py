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

import math

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_hifi_decoder import (
    LATENT_SCALE,
    SR_SCALE,
    build_linear_interp_matrix,
)
from models.experimental.xtts.tt.xtts_hifigan import TtHifiganGenerator

TILE = 32
# Tuned resample-matmul config (test_hifi_upsampler_matmul_sweep.py, Blackhole):
# a 2D-multicast config with per_core_N=3 (N=1024 -> gx=11) + M spread over rows, HiFi2,
# fp32_dest_acc off, and an L1 output — 4.80us -> 1.46us on the profiled L=32 shape. The
# grid is derived per shape so other latent lengths stay legal (falls back to auto if not).
_MATMUL_PER_CORE_N = 3


class TtLatentUpsampler(LightweightModule):
    """Linear time-upsample of channels-last latents ``[1, T, C]`` -> ``[1, L_out, C]``
    via a composed resample matmul, matching the reference's two F.interpolate steps."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self._matrix_cache = {}  # T -> device resample matrix [L_out, T]
        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _resample_matrix(self, length_in: int) -> ttnn.Tensor:
        if length_in not in self._matrix_cache:
            m1 = build_linear_interp_matrix(length_in, LATENT_SCALE)  # [4T, T]
            m2 = build_linear_interp_matrix(m1.shape[0], SR_SCALE)  # [L_out, 4T]
            matrix = m2 @ m1  # [L_out, T]
            self._matrix_cache[length_in] = ttnn.from_torch(
                matrix, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32
            )
        return self._matrix_cache[length_in]

    def _matmul_program_config(self, length_out: int, channels: int):
        """Tuned 2D-multicast config derived for this shape, or None if it doesn't fit
        the device grid (caller then falls back to the auto config)."""
        grid = self.device.compute_with_storage_grid_size()
        max_x, max_y = int(grid.x), int(grid.y)
        Mt = math.ceil(length_out / TILE)
        Nt = math.ceil(channels / TILE)
        per_core_N = _MATMUL_PER_CORE_N
        gx = math.ceil(Nt / per_core_N)
        per_core_M = math.ceil(Mt / max_y)  # spread M over rows, capping the grid
        gy = math.ceil(Mt / per_core_M)
        if gx > max_x or gy > max_y or per_core_N > Nt:
            return None
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=1,  # Kt = 1
            out_subblock_h=1,
            out_subblock_w=per_core_N,
            out_block_h=per_core_M,
            out_block_w=per_core_N,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    def forward(self, x_blc: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, length_in, channels = x_blc.shape
        assert batch_size == 1, "latent upsampler assumes batch size 1"
        matrix = self._resample_matrix(length_in)  # [L_out, T]
        length_out = matrix.shape[0]

        x = ttnn.to_layout(x_blc, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_blc)  # caller's latents temp, not reused after
        x = ttnn.reshape(x, [length_in, channels])  # drop batch for the 2D matmul
        program_config = self._matmul_program_config(length_out, channels)
        y = ttnn.matmul(  # [L_out, C]
            matrix,
            x,
            program_config=program_config,
            compute_kernel_config=self._compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        y2 = ttnn.reshape(y, [1, length_out, channels])
        out = ttnn.to_layout(y2, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(y2)
        return out


class TtHifiDecoder(LightweightModule):
    """XTTS-v2 HifiDecoder: latent upsample -> HiFi-GAN generator.

    ``latents`` is channels-last ``[1, T, 1024]``, ``g`` is ``[1, 1, 512]``.
    Output is ``[1, L_out*256, 1]`` (the waveform).
    """

    def __init__(self, device, state_dict, bf16_stages=None):
        super().__init__()
        self.upsampler = TtLatentUpsampler(device)
        self.generator = TtHifiganGenerator(device, state_dict, bf16_stages=bf16_stages)

    def forward(self, latents, g):
        return self.generator(self.upsampler(latents), g)
