# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decision head on device: final norm -> last real token -> readout -> option mask -> / T -> softmax.

Reproduces ``DecisionModel.forward`` + ``predict`` (snapshot ``source/src/autojev/model.py:202-224``)::

    hidden = backbone(...).last_hidden_state[:, -1]          # after the final RMSNorm
    logits = readout(hidden).float()                          # Linear 5120 -> 255, no bias
    logits = logits.masked_fill(arange(255) >= count, -1e9)
    probs = (logits / temperature).softmax(-1)[:count]

The final norm is row-wise, so the last real token is sliced first and only one row is normed.
The readout weight is zero-padded from 255 to 256 output columns so the softmax runs over whole
tiles; column 255 is always masked (``count <= 255``), so it carries no probability.
The readout matmul output is BF16 (as the app's bf16 ``readout``); mask, scale and softmax run
in fp32. Only the [1, 1, 256] logits / probabilities leave the device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.demos.pplx_decider_v1_27b.tt.common import prefill_linear, resolve
from models.demos.pplx_decider_v1_27b.tt.optimizations import LinearOptimizations, NormOptimizations

NUM_OPTIONS = 255
PADDED_OPTIONS = 256
MASK_VALUE = -1e9  # the app's finite mask value


@dataclass
class DecisionHeadConfig:
    norm_weight: LazyWeight  # (1 + w) [1, 1, 5120]
    readout_weight: LazyWeight  # [5120, 256], column 255 zero
    eps: float
    temperature: float
    linear: LinearOptimizations
    norm: NormOptimizations
    mesh_device: object | None = None


class PplxDecisionHead(LightweightModule):
    def __init__(self, config: DecisionHeadConfig):
        super().__init__()
        self._build(config)

    @classmethod
    def from_config(cls, config: DecisionHeadConfig) -> "PplxDecisionHead":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance._build(config)
        return instance

    def _build(self, config: DecisionHeadConfig) -> None:
        import torch

        if not (config.temperature > 0 and config.temperature != float("inf")):
            raise ValueError(f"Temperature must be positive and finite, got {config.temperature}")
        self.config = _resolve(config)
        # Option index per column, compared against the option count on device.
        self.option_index = ttnn.from_torch(
            torch.arange(PADDED_OPTIONS, dtype=torch.float32).reshape(1, 1, PADDED_OPTIONS),
            device=self.config.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.softmax_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.norm_weight = None
        self.readout_weight = None

    def load_device_weights(self) -> None:
        if self.norm_weight is None:
            self.norm_weight = self.config.norm_weight.get_device_weight()
            self.readout_weight = self.config.readout_weight.get_device_weight()

    def forward(self, hidden: ttnn.Tensor, last_index: int, count: int, *, return_hidden: bool = False):
        """hidden [1, S, 5120] (output of the last layer) -> (probs, logits, normed_or_None).

        probs and logits are [1, 1, 256] fp32. ``last_index`` is the position of the last REAL
        token (right padding follows it); ``count`` the number of options (1..255). probs[count:]
        are 0, logits are unmasked. ``return_hidden`` also keeps the final-normed last-token
        hidden [1, 1, 5120] (for tests); otherwise it is freed and None is returned.
        """
        self.load_device_weights()
        if not 1 <= count <= NUM_OPTIONS:
            raise ValueError(f"Option count {count} outside 1..{NUM_OPTIONS}")
        if not 0 <= last_index < hidden.shape[1]:
            raise ValueError(f"last_index {last_index} outside the {hidden.shape[1]}-token input")
        c = self.config
        last = ttnn.slice(hidden, [0, last_index, 0], [1, last_index + 1, hidden.shape[-1]])
        normed = ttnn.rms_norm(
            last,
            weight=self.norm_weight,
            epsilon=c.eps,
            compute_kernel_config=c.norm.compute_kernel_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(last)
        logits = ttnn.typecast(prefill_linear(normed, self.readout_weight, "readout", c.linear), ttnn.float32)
        if not return_hidden:
            ttnn.deallocate(normed)
            normed = None
        masked = ttnn.where(ttnn.ge(self.option_index, float(count)), MASK_VALUE, logits)
        scaled = ttnn.multiply(masked, 1.0 / c.temperature)
        ttnn.deallocate(masked)
        # Softmax composed from fp32 eltwise ops: ttnn.softmax on this fp32 [1, 1, 256] row measured
        # 2e-3 relative error (probs summed to 0.9989); max/exp/sum/divide measured 4e-7.
        shifted = ttnn.subtract(scaled, ttnn.max(scaled, dim=-1, keepdim=True))
        ttnn.deallocate(scaled)
        exp = ttnn.exp(shifted)
        ttnn.deallocate(shifted)
        total = ttnn.sum(exp, dim=-1, keepdim=True, compute_kernel_config=self.softmax_compute_cfg)
        probs = ttnn.divide(exp, total)
        ttnn.deallocate(exp)
        ttnn.deallocate(total)
        return probs, logits, normed


def _resolve(config: DecisionHeadConfig) -> DecisionHeadConfig:
    device = config.mesh_device or config.readout_weight.device or ttnn.GetDefaultDevice()
    return replace(
        config,
        mesh_device=device,
        norm_weight=resolve(config.norm_weight, device),
        readout_weight=resolve(config.readout_weight, device),
    )
