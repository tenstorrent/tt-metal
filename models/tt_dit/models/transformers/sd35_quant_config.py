# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quantization / compute-precision policy for the SD3.5 DiT linears.

Mirrors the LTX quant profile (see ``ltx/quant_config.py``) but tailored to the SD3.5 joint-attention
block: the quantized projections are the fused ``to_qkv`` / ``add_qkv_proj`` and the two FFN linears
(``ff1`` / ``ff2``); the attention output projections (``to_out`` / ``to_add_out``) carve out to bf16.

The transformer modules stay precision-agnostic: they spread ``profile.qkv_linear_kwargs()`` /
``profile.out_linear_kwargs()`` / ``profile.ffn_kwargs()`` onto each linear and read
``profile.mm_compute_config`` for the matmul fidelity, with no dtype / fidelity literals of their own.

Weight quantization happens at ``Parameter`` construction (``ttnn.from_torch(dtype=...)`` is the only
quantizer; there is no torch bfloat8), so the profile sets each linear's ``dtype`` at build time.

Selection is env-driven so a run can A/B without touching the pipeline config:

  SD35_QUANT unset / "off" / "0"  -> None (bf16 everywhere, unchanged model)
  SD35_QUANT = "bf8w"             -> bf8 weights + LoFi matmul, activations stay bf16 (Increment 1)
  SD35_QUANT = "bf8"              -> bf8 weights + LoFi matmul + bf8 linear/SDPA activations (Increment 2)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class SD35QuantProfile:
    weight_dtype: ttnn.DataType  # to_qkv / add_qkv_proj, ff1 / ff2
    out_weight_dtype: ttnn.DataType  # to_out / to_add_out carve-out
    activation_dtype: ttnn.DataType | None  # linear activation cast (None => keep bf16 activations)
    mm_fidelity: ttnn.MathFidelity
    mm_fp32_dest_acc: bool
    sdpa_input_dtype: ttnn.DataType | None  # ring-SDPA q/k/v input cast (None => bf16)

    # ---- construction-kwarg helpers -------------------------------------------------

    def qkv_linear_kwargs(self) -> dict:
        """Kwargs for a quantized ColParallelLinear (to_qkv / add_qkv_proj)."""
        return {
            "dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "pin_output_bf16": self.activation_dtype is not None,
        }

    def out_linear_kwargs(self) -> dict:
        """Kwargs for the carved-out attention output projections (to_out / to_add_out).

        Weight stays bf16; still consumes the shared bf8 activation when activations are quantized, so
        it is pinned back to bf16 like the projections.
        """
        return {
            "dtype": self.out_weight_dtype,
            "activation_dtype": self.activation_dtype,
            "pin_output_bf16": self.activation_dtype is not None,
        }

    def ffn_kwargs(self) -> dict:
        """Kwargs for ParallelFeedForward (ff1 + ff2 both quantized)."""
        return {
            "ff1_dtype": self.weight_dtype,
            "ff2_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "pin_output_bf16": self.activation_dtype is not None,
        }

    def mm_compute_config(self, arch):
        """Compute-kernel config for the DiT-linear matmuls.

        ``packer_l1_acc`` stays on (near-free fp32 L1 partial accumulation); ``fp32_dest_acc_en`` is the
        pricier fp32 destination accumulation that the LoFi/bf8 tier drops.
        """
        return ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=self.mm_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=self.mm_fp32_dest_acc,
            packer_l1_acc=True,
        )

    # ---- presets --------------------------------------------------------------------

    @staticmethod
    def bf8_weights_lofi() -> SD35QuantProfile:
        """bf8 weights, LoFi matmul, bf16 activations. Lowest-risk quant tier (weights quantize well)."""
        return SD35QuantProfile(
            weight_dtype=ttnn.bfloat8_b,
            out_weight_dtype=ttnn.bfloat16,
            activation_dtype=None,
            mm_fidelity=ttnn.MathFidelity.LoFi,
            mm_fp32_dest_acc=False,
            sdpa_input_dtype=None,
        )

    @staticmethod
    def all_bf8_lofi() -> SD35QuantProfile:
        """bf8 weights + bf8 linear/SDPA activations, LoFi matmul. Attacks CCL payloads too."""
        return SD35QuantProfile(
            weight_dtype=ttnn.bfloat8_b,
            out_weight_dtype=ttnn.bfloat16,
            activation_dtype=ttnn.bfloat8_b,
            mm_fidelity=ttnn.MathFidelity.LoFi,
            mm_fp32_dest_acc=False,
            sdpa_input_dtype=ttnn.bfloat8_b,
        )

    @staticmethod
    def from_env() -> SD35QuantProfile | None:
        val = os.environ.get("SD35_QUANT", "off").strip().lower()
        if val in ("off", "0", "", "none", "bf16"):
            return None
        if val == "bf8w":
            return SD35QuantProfile.bf8_weights_lofi()
        if val == "bf8":
            return SD35QuantProfile.all_bf8_lofi()
        raise ValueError(f"Unknown SD35_QUANT={val!r}; expected one of off|bf8w|bf8")
