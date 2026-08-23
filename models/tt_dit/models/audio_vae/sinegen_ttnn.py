# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F

import ttnn

from ...layers.linear import Linear
from ...layers.module import Module


class SineGen2(Module):
    """TTNN implementation of the non-causal CosyVoice2 HiFT SineGen2 path.

    Input/output layout: BTC.

    Phase accumulation intentionally uses float32.  The HiFT upsample scale is
    large (256 for CosyVoice2), so BF16 destroys fractional phase information
    before the final sine operation.
    """

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        *,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        causal: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        if causal:
            raise NotImplementedError(
                "CosyVoice2 bring-up currently implements the production " "non-causal SineGen2 path only"
            )

        self.sampling_rate = float(sampling_rate)
        self.upsample_scale = int(upsample_scale)
        self.harmonic_num = int(harmonic_num)
        self.dim = self.harmonic_num + 1

        self.sine_amp = float(sine_amp)
        self.noise_std = float(noise_std)
        self.voiced_threshold = float(voiced_threshold)

        self.mesh_device = mesh_device
        self.dtype = dtype
        self.phase_dtype = ttnn.float32

        self.phase_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Length-dependent interpolation/cumsum matrices.
        self._phase_matrix_cache: dict[int, tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]] = {}

        # Small constants are cached lazily on device.
        self._harmonic_scale = None
        self._initial_phase_mask = None

    def _constant(
        self,
        value: torch.Tensor,
        *,
        dtype: ttnn.DataType,
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            value,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
        )

    def _get_harmonic_scale(self) -> ttnn.Tensor:
        if self._harmonic_scale is None:
            scale = (torch.arange(1, self.dim + 1, dtype=torch.float32) / self.sampling_rate).reshape(1, 1, self.dim)

            self._harmonic_scale = self._constant(
                scale,
                dtype=self.phase_dtype,
            )

        return self._harmonic_scale

    def _get_initial_phase_mask(self) -> ttnn.Tensor:
        if self._initial_phase_mask is None:
            mask = torch.ones(1, 1, self.dim, dtype=torch.float32)
            mask[..., 0] = 0.0

            self._initial_phase_mask = self._constant(
                mask,
                dtype=self.phase_dtype,
            )

        return self._initial_phase_mask

    def _get_phase_matrices(
        self,
        length: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        cached = self._phase_matrix_cache.get(length)
        if cached is not None:
            return cached

        if length % self.upsample_scale != 0:
            raise ValueError(
                f"SineGen2 input length {length} must be divisible by " f"upsample_scale={self.upsample_scale}"
            )

        reduced_length = length // self.upsample_scale

        # These matrices reproduce PyTorch F.interpolate(..., mode="linear",
        # align_corners=False) exactly as linear transformations.
        down_matrix = F.interpolate(
            torch.eye(length, dtype=torch.float32).unsqueeze(1),
            size=reduced_length,
            mode="linear",
        ).squeeze(1)

        cumsum_matrix = torch.triu(
            torch.ones(
                reduced_length,
                reduced_length,
                dtype=torch.float32,
            )
        )

        up_matrix = F.interpolate(
            torch.eye(reduced_length, dtype=torch.float32).unsqueeze(1),
            size=length,
            mode="linear",
        ).squeeze(1)

        # Original:
        # phase = cumsum(rad) * 2*pi
        # phase = interpolate(phase * upsample_scale, ...)
        #
        # Fold constants into the matrix so the phase stays FP32 without an
        # additional scalar SFPU multiply.
        up_matrix *= self.upsample_scale * (2.0 * math.pi)

        down_tt = self._constant(down_matrix, dtype=self.phase_dtype)
        cumsum_tt = self._constant(cumsum_matrix, dtype=self.phase_dtype)
        up_tt = self._constant(up_matrix, dtype=self.phase_dtype)

        cached = (down_tt, cumsum_tt, up_tt)
        self._phase_matrix_cache[length] = cached
        return cached

    def _randn(
        self,
        shape: tuple[int, ...],
        *,
        dtype: ttnn.DataType,
    ) -> ttnn.Tensor:
        """Generate Gaussian noise on device with Box-Muller."""

        u1 = ttnn.rand(
            shape,
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            low=1.0e-6,
            high=1.0,
        )
        u2 = ttnn.rand(
            shape,
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            low=0.0,
            high=1.0,
        )

        log_u1 = ttnn.log(u1)
        radius = ttnn.sqrt(ttnn.multiply(log_u1, -2.0))
        angle = ttnn.multiply(u2, 2.0 * math.pi)
        cosine = ttnn.cos(angle)
        noise = ttnn.multiply(radius, cosine)

        ttnn.deallocate(u1)
        ttnn.deallocate(u2)
        ttnn.deallocate(log_u1)
        ttnn.deallocate(radius)
        ttnn.deallocate(angle)
        ttnn.deallocate(cosine)

        return noise

    def _f02uv(self, f0: ttnn.Tensor) -> ttnn.Tensor:
        uv = ttnn.gt(f0, self.voiced_threshold)

        if uv.dtype != self.dtype:
            uv = ttnn.typecast(uv, dtype=self.dtype)

        return uv

    def _f02sine(self, f0_values: ttnn.Tensor) -> ttnn.Tensor:
        batch, length, channels = f0_values.shape

        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} harmonic channels, got {channels}")

        if f0_values.dtype != self.phase_dtype:
            f0_values = ttnn.typecast(
                f0_values,
                dtype=self.phase_dtype,
            )

        # rad_values = (f0_values / sampling_rate) % 1
        rad_values = ttnn.multiply(
            f0_values,
            1.0 / self.sampling_rate,
        )
        rad_values = ttnn.remainder(rad_values, 1.0)

        # Original non-causal SineGen2 adds a random initial phase to all
        # harmonics except the fundamental.
        rand_ini = ttnn.rand(
            (batch, 1, self.dim),
            device=self.mesh_device,
            dtype=self.phase_dtype,
            layout=ttnn.TILE_LAYOUT,
            low=0.0,
            high=1.0,
        )
        rand_ini = ttnn.multiply(
            rand_ini,
            self._get_initial_phase_mask(),
        )

        first = ttnn.slice(
            rad_values,
            [0, 0, 0],
            [batch, 1, self.dim],
        )
        first = ttnn.add(first, rand_ini)

        if length > 1:
            rest = ttnn.slice(
                rad_values,
                [0, 1, 0],
                [batch, length, self.dim],
            )
            rad_values_with_phase = ttnn.concat(
                [first, rest],
                dim=1,
            )
            ttnn.deallocate(rest)
        else:
            rad_values_with_phase = first

        ttnn.deallocate(rand_ini)
        ttnn.deallocate(rad_values)

        down_matrix, cumsum_matrix, up_matrix = self._get_phase_matrices(length)

        # BTC -> BCT
        phase = ttnn.permute(
            rad_values_with_phase,
            (0, 2, 1),
        )
        ttnn.deallocate(rad_values_with_phase)

        # Linear downsample by 1 / upsample_scale.
        phase_next = ttnn.matmul(
            phase,
            down_matrix,
            compute_kernel_config=self.phase_compute_config,
        )
        ttnn.deallocate(phase)
        phase = phase_next

        # cumsum over time via upper-triangular matmul.
        phase_next = ttnn.matmul(
            phase,
            cumsum_matrix,
            compute_kernel_config=self.phase_compute_config,
        )
        ttnn.deallocate(phase)
        phase = phase_next

        # Linear upsample and fold (* upsample_scale * 2*pi) into matrix.
        phase_next = ttnn.matmul(
            phase,
            up_matrix,
            compute_kernel_config=self.phase_compute_config,
        )
        ttnn.deallocate(phase)
        phase = phase_next

        # BCT -> BTC
        phase = ttnn.permute(
            phase,
            (0, 2, 1),
        )

        sines = ttnn.sin(phase)
        ttnn.deallocate(phase)

        return sines

    def forward(
        self,
        f0: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        # Fundamental + harmonic overtones.
        f0_phase = f0 if f0.dtype == self.phase_dtype else ttnn.typecast(f0, dtype=self.phase_dtype)

        # fn = f0 * [1, 2, ..., harmonic_num + 1]
        harmonic_numbers = torch.arange(
            1,
            self.dim + 1,
            dtype=torch.float32,
        ).reshape(1, 1, self.dim)

        harmonic_numbers_tt = self._constant(
            harmonic_numbers,
            dtype=self.phase_dtype,
        )

        fn = ttnn.multiply(f0_phase, harmonic_numbers_tt)
        sines = self._f02sine(fn)

        ttnn.deallocate(harmonic_numbers_tt)
        ttnn.deallocate(fn)

        if f0_phase is not f0:
            ttnn.deallocate(f0_phase)

        if sines.dtype != self.dtype:
            sines = ttnn.typecast(
                sines,
                dtype=self.dtype,
            )

        sine_waves = ttnn.multiply(
            sines,
            self.sine_amp,
        )
        ttnn.deallocate(sines)

        uv = self._f02uv(f0)

        voiced_noise_amp = ttnn.multiply(
            uv,
            self.noise_std,
        )

        one_minus_uv = ttnn.multiply(uv, -1.0)
        one_minus_uv = ttnn.add(one_minus_uv, 1.0)

        unvoiced_noise_amp = ttnn.multiply(
            one_minus_uv,
            self.sine_amp / 3.0,
        )

        noise_amp = ttnn.add(
            voiced_noise_amp,
            unvoiced_noise_amp,
        )

        gaussian = self._randn(
            tuple(sine_waves.shape),
            dtype=self.dtype,
        )
        noise = ttnn.multiply(
            noise_amp,
            gaussian,
        )

        voiced_sines = ttnn.multiply(
            sine_waves,
            uv,
        )
        output = ttnn.add(
            voiced_sines,
            noise,
        )

        ttnn.deallocate(sine_waves)
        ttnn.deallocate(voiced_noise_amp)
        ttnn.deallocate(one_minus_uv)
        ttnn.deallocate(unvoiced_noise_amp)
        ttnn.deallocate(noise_amp)
        ttnn.deallocate(gaussian)
        ttnn.deallocate(voiced_sines)

        return output, uv, noise


class SourceModuleHnNSF(Module):
    """TTNN HiFT harmonic-plus-noise source module."""

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        *,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        causal: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.sine_amp = float(sine_amp)
        self.dtype = dtype
        self.mesh_device = mesh_device

        self.l_sin_gen = SineGen2(
            sampling_rate,
            upsample_scale,
            harmonic_num=harmonic_num,
            sine_amp=sine_amp,
            noise_std=add_noise_std,
            voiced_threshold=voiced_threshold,
            causal=causal,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        self.l_linear = Linear(
            harmonic_num + 1,
            1,
            bias=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )

    def forward(
        self,
        f0: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        sine_wavs, uv, sine_noise = self.l_sin_gen(f0)

        sine_merge = self.l_linear(sine_wavs)
        sine_merge = ttnn.tanh(sine_merge)

        noise = self.l_sin_gen._randn(
            tuple(uv.shape),
            dtype=self.dtype,
        )
        noise = ttnn.multiply(
            noise,
            self.sine_amp / 3.0,
        )

        ttnn.deallocate(sine_wavs)
        ttnn.deallocate(sine_noise)

        return sine_merge, noise, uv
