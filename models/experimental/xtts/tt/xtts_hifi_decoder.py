# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_hifi_decoder import (
    LATENT_SCALE,
    SR_SCALE,
    build_linear_interp_matrix,
)
from models.experimental.xtts.tt.xtts_hifigan import TtHifiganGenerator

from models.experimental.xtts.config import TILE  # noqa: F401 — re-exported for callers

_MATMUL_PER_CORE_N = 3


class TtLatentUpsampler(LightweightModule):
    def __init__(self, device):
        """Initialize latent upsampler compute config and matrix cache."""
        super().__init__()
        self.device = device
        self._matrix_cache = {}
        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _resample_matrix(self, length_in: int) -> ttnn.Tensor:
        """Build or fetch the two-stage linear interpolation matrix."""
        if length_in not in self._matrix_cache:
            m1 = build_linear_interp_matrix(length_in, LATENT_SCALE)
            m2 = build_linear_interp_matrix(m1.shape[0], SR_SCALE)
            matrix = m2 @ m1
            # Keep per-length constant in L1 (matmul in0).
            self._matrix_cache[length_in] = ttnn.from_torch(
                matrix,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                dtype=ttnn.float32,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        return self._matrix_cache[length_in]

    def release_cache(self):
        """Drop cached interp matrices (L1). Safe after vocoder trace release."""
        for t in self._matrix_cache.values():
            if t.is_allocated():
                ttnn.deallocate(t)
        self._matrix_cache.clear()

    def _matmul_program_config(self, length_out: int, channels: int):
        """Build a block-sharded matmul program config for upsample."""
        grid = self.device.compute_with_storage_grid_size()
        max_x, max_y = int(grid.x), int(grid.y)
        Mt = math.ceil(length_out / TILE)
        Nt = math.ceil(channels / TILE)
        per_core_N = _MATMUL_PER_CORE_N
        gx = math.ceil(Nt / per_core_N)
        per_core_M = math.ceil(Mt / max_y)
        gy = math.ceil(Mt / per_core_M)
        if gx > max_x or gy > max_y or per_core_N > Nt:
            return None
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=1,
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

    @staticmethod
    def _out_memory_config(program_config):
        """Pick L1 interleaved or block-sharded output memory config."""
        if program_config is None:
            return ttnn.L1_MEMORY_CONFIG
        return ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1)

    def forward(self, x_blc: ttnn.Tensor) -> ttnn.Tensor:
        """Upsample latent frames via cached interpolation matmul."""
        batch_size, length_in, channels = x_blc.shape
        assert batch_size == 1, "latent upsampler assumes batch size 1"
        matrix = self._resample_matrix(length_in)
        length_out = matrix.shape[0]

        # to_layout on already-TILE shares the buffer; only deallocate after a real conversion.
        tiled_in = x_blc.layout == ttnn.TILE_LAYOUT
        x = x_blc if tiled_in else ttnn.to_layout(x_blc, ttnn.TILE_LAYOUT)
        if not tiled_in:
            ttnn.deallocate(x_blc)
        x = ttnn.reshape(x, [length_in, channels])
        program_config = self._matmul_program_config(length_out, channels)
        y = ttnn.matmul(
            matrix,
            x,
            program_config=program_config,
            compute_kernel_config=self._compute_kernel_config,
            memory_config=self._out_memory_config(program_config),
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(x)
        return ttnn.reshape(y, [1, length_out, channels])


class TtHifiDecoder(LightweightModule):
    def __init__(self, device, state_dict):
        """Compose latent upsampler with HiFi-GAN generator."""
        super().__init__()
        self.upsampler = TtLatentUpsampler(device)
        self.generator = TtHifiganGenerator(device, state_dict)

    def forward(self, latents, g):
        """Upsample latents and generate waveform with speaker emb."""
        return self.generator(self.upsampler(latents), g)
