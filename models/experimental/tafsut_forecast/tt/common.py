# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Precision policy (skill §2, CONTRACT §1) and compute configs.

Accuracy (the gated target): bf16 weights and activations; HiFi2 matmuls with fp32 accumulation, except QKV at
HiFi4; HiFi4 for the norms, RoPE and the patch embedding; HiFi4 with an fp32 output for the head. SDPA: HiFi4, fp32
accumulation,
``exp_approx_mode=False`` (round 0 is fold-free, D0 Q5).
Performance (Chronos-2's preset as the start): bfp8 QKV / FFN-up weights, LoFi encoder matmuls; ``wo`` / ``w_down``
stay bf16 (their residual add can fold into the matmul epilogue).
"""

from __future__ import annotations

from dataclasses import dataclass

import os

import ttnn

MASK_FILL = -1e9  # finite: an all-masked k-chunk must not compute (-inf) - (-inf) (CONTRACT §7 item 1)


@dataclass(frozen=True)
class Precision:
    name: str = "accuracy"
    bf8_weights: bool = False  # wqkv and wi as bfloat8_b
    lofi: bool = False  # encoder linears at LoFi
    qkv_fidelity: str = "HiFi4"  # QKV: HiFi4 (single_long head PCC 0.9930 -> 0.9988 with SDPA HiFi4, job 923)
    sdpa_fidelity: str = "HiFi4"
    exp_approx: bool = False

    @classmethod
    def accuracy(cls):
        return cls()

    @classmethod
    def performance(cls):
        return cls(name="performance", bf8_weights=True, lofi=True, qkv_fidelity="LoFi", sdpa_fidelity="HiFi2")

    @classmethod
    def by_name(cls, name: str):
        return {"accuracy": cls.accuracy, "performance": cls.performance}[name]()

    @property
    def in_weight_dtype(self):
        return ttnn.bfloat8_b if self.bf8_weights else ttnn.bfloat16


# screening knob (D7 (b)), default off: packer L1 accumulation in every compute config
PACKER_L1 = os.environ.get("TAFSUT_PACKER_L1", "0") == "1"


def ckc(fidelity="HiFi2", fp32_acc=True, approx=False):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=PACKER_L1,
    )
