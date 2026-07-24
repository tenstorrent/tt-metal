# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Aligned ``SourceModuleHnNSF`` / SineGen RNG for Kokoro PCC and deterministic inference.

Reference uses ``torch.rand`` / ``torch.randn_like`` inside ``SineGen`` and
``SourceModuleHnNSF``. TT uses optional uploaded tensors plus init-time
``_noise_raw`` / ``_out_noise_raw``. This module patches the reference and
supplies explicit TT tensors so both sides see the same noise (typically zeros).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import ttnn


@dataclass(frozen=True)
class MSourceRngTensors:
    """CPU tensors shared by patched reference and explicit TT uploads."""

    rand_ini: torch.Tensor  # [B, dim]
    sinegen_noise: torch.Tensor  # [B, T_har, dim]
    source_noise: torch.Tensor  # [B, T_har, 1]


@dataclass
class MSourceRngTT:
    """Device tensors for :meth:`TTSourceModuleHnNSF.forward`."""

    rand_ini: ttnn.Tensor  # [B, 1, dim]
    sinegen_noise: ttnn.Tensor  # [B, T_har, dim]
    source_noise: ttnn.Tensor  # [B, T_har, 1]


def make_zero_m_source_rng(B: int, T_har: int, dim: int) -> MSourceRngTensors:
    """Deterministic all-zero noise (matches ``_zero_noise()`` reference intent)."""
    rand_ini = torch.zeros(B, dim)
    rand_ini[:, 0] = 0.0
    return MSourceRngTensors(
        rand_ini=rand_ini,
        sinegen_noise=torch.zeros(B, T_har, dim),
        source_noise=torch.zeros(B, T_har, 1),
    )


def m_source_rng_shapes_from_f0(
    F0_curve: torch.Tensor,
    *,
    upsample_scale_full: int,
    dim: int,
) -> tuple[int, int, int]:
    """Return ``(B, T_har, dim)`` for building :class:`MSourceRngTensors`."""
    f0 = F0_curve.detach().float().cpu()
    if f0.dim() == 1:
        f0 = f0.unsqueeze(0)
    B = int(f0.shape[0])
    T_f0 = int(f0.shape[-1])
    T_har = T_f0 * int(upsample_scale_full)
    return B, T_har, dim


@contextmanager
def patched_m_source_torch_rng(bundle: MSourceRngTensors):
    """Patch global ``torch.rand`` / ``torch.randn_like`` for ``m_source`` reference."""
    real_rand = torch.rand
    real_randn_like = torch.randn_like

    def _fake_rand(*size, **kwargs):
        out = bundle.rand_ini.to(kwargs.get("device", bundle.rand_ini.device))
        return out.to(kwargs.get("dtype", out.dtype))

    def _fake_randn_like(t: torch.Tensor, **kwargs):
        if int(t.shape[-1]) == 1:
            out = bundle.source_noise
        else:
            out = bundle.sinegen_noise
        return out.to(device=t.device, dtype=t.dtype).reshape(t.shape)

    torch.rand = _fake_rand
    torch.randn_like = _fake_randn_like
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def upload_m_source_rng(
    bundle: MSourceRngTensors,
    device: ttnn.Device,
    *,
    dtype=ttnn.float32,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> MSourceRngTT:
    """Upload CPU bundle for :class:`TTSourceModuleHnNSF` / :class:`TTSineGen`."""
    return MSourceRngTT(
        rand_ini=ttnn.from_torch(
            bundle.rand_ini.unsqueeze(1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        ),
        sinegen_noise=ttnn.from_torch(
            bundle.sinegen_noise.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        ),
        source_noise=ttnn.from_torch(
            bundle.source_noise.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        ),
    )


def deallocate_m_source_rng_tt(rng: Optional[MSourceRngTT]) -> None:
    if rng is None:
        return
    ttnn.deallocate(rng.rand_ini)
    ttnn.deallocate(rng.sinegen_noise)
    ttnn.deallocate(rng.source_noise)
