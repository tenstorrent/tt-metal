# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC hyper-connections for GLM-5.3-Flash, on top of the DeepSeek-V4 implementation.

GLM-5.3-Flash's ``Glm5NextTextHyperConnection`` is the DeepSeek-V4 parametrization exactly:
unweighted RMSNorm over the 4 packed streams, one fused projection ``fn [24, 4*4096]``, then
``pre = sigmoid(a_pre*raw + b) + eps``, ``post = 2*sigmoid(a_post*raw + b)``,
``comb = sinkhorn(softmax(a_res*raw + b) + eps)`` with one column normalisation followed by
``iters-1`` (row, column) pairs. ``TtMHCWrap`` runs that in fp32 with the fused
``mhc_split_sinkhorn`` kernel, so this module only adapts the configuration and the tensor
convention.

Tensor convention (from ``tt_mhc``): residual streams are packed on the last dim,
``[1, 1, T, 4*4096]`` fp32 with stream ``i`` at columns ``[i*4096, (i+1)*4096)`` and the
``T = B*S`` tokens down the rows. HF keeps ``[B, S, 4, 4096]``; the conversion is a reshape.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap, mhc_expand  # noqa: F401
from models.demos.glm53_flash.tt.config import GLM53FlashConfig


@dataclass(frozen=True)
class MHCKernelConfig:
    """The fields ``TtMHCWrap`` reads (``MHCConfig`` in the DeepSeek-V4 reference)."""

    dim: int
    n: int
    sinkhorn_iters: int
    eps: float
    norm_eps: float

    @property
    def mix_hc(self) -> int:
        return (2 + self.n) * self.n


def mhc_kernel_config(cfg: GLM53FlashConfig) -> MHCKernelConfig:
    # HF: input_norm = UnweightedRMSNorm(eps=rms_norm_eps); hc_eps is the stochasticity eps.
    return MHCKernelConfig(
        dim=cfg.hidden_size,
        n=cfg.hc_mult,
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        eps=cfg.hc_eps,
        norm_eps=cfg.rms_norm_eps,
    )


def build_mhc(device, cfg: GLM53FlashConfig, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor) -> TtMHCWrap:
    """One hyper-connection (attention or FFN) from the raw checkpoint tensors
    ``hc_*_fn [24, 16384]`` (bf16), ``hc_*_base [24]`` and ``hc_*_scale [3]`` (fp32)."""
    kcfg = mhc_kernel_config(cfg)
    assert tuple(fn.shape) == (kcfg.mix_hc, cfg.hc_mult * cfg.hidden_size), fn.shape
    assert base.numel() == kcfg.mix_hc and scale.numel() == 3
    return TtMHCWrap(device, kcfg, fn.float(), base.float(), scale.float(), dtype=ttnn.float32)


def streams_hf_to_packed(x: torch.Tensor) -> torch.Tensor:
    """HF ``[B, S, n, C]`` -> packed ``[1, 1, B*S, n*C]`` (host side, tests only)."""
    B, S, n, C = x.shape
    return x.reshape(1, 1, B * S, n * C)


def streams_packed_to_hf(x: torch.Tensor, batch: int, n: int) -> torch.Tensor:
    """Packed ``[1, 1, B*S, n*C]`` -> HF ``[B, S, n, C]`` (host side, tests only)."""
    T, nC = x.shape[-2], x.shape[-1]
    return x.reshape(batch, T // batch, n, nC // n)
