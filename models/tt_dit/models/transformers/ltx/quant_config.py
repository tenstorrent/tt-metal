# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quantization and compute-precision policy for the LTX DiT linears.

A single :class:`LtxQuantProfile` holds the whole policy — weight dtypes, the ``to_out`` carve-out,
activation/SDPA casts, and matmul/SDPA fidelities — and exposes it as construction kwargs. The
transformer modules stay precision-agnostic: they spread ``profile.linear_kwargs(role)`` onto each
linear and read ``profile.mm_compute_config`` / ``profile.sdpa_self_config`` for the op configs, with
no dtype or fidelity literals of their own.

Weight quantization happens at Parameter construction, not as a state_dict pass: there is no Torch
bfloat8, so ``ttnn.from_torch(dtype=...)`` is the only quantizer and the cache enforces the declared
``Parameter.dtype``. The profile therefore sets each linear's ``dtype`` at build time, so a cache miss
loads weights direct-to-quant and the cache write holds the quantized tensorbins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ttnn

# When set, activations are narrowed to bf8 wherever they cross the fabric — both the linear activation
# casts and the ring-SDPA Q/K/V inputs — so this is the one knob that changes what the collectives move
# (the linear payloads and the SP-gathered K/V; SDPA itself stays HiFi2, only its inputs narrow). More
# aggressive than the weight quant it rides on: weights are a fixed, well-conditioned distribution,
# activations are not. On by default because the shipped 1080p tier is measured and VBench-gated with it
# on; set to 0 to A/B against the weight-only preset. Keeping the SDPA input cast on this same knob means
# a run can't narrow the linear activations while silently leaving SDPA at bf16 (or vice versa).
LTX_QUANT_ACTIVATIONS = os.environ.get("LTX_QUANT_ACTIVATIONS", "1") in ("1", "true", "True")


@dataclass(frozen=True)
class LtxQuantProfile:
    """One quant profile for every DiT linear, with the ``to_out`` carve-out.

    Every quantized projection (qkv/q/kv, ff1/ff2) shares one weight dtype and one compute profile;
    only ``to_out`` carves out to ``out_weight_dtype`` (bf16), because it feeds the fused-addcmul
    epilogue whose bf16 ternary inputs must match the weight tile format. The activation and SDPA-input
    casts (and the output pins that go with them) ride ``LTX_QUANT_ACTIVATIONS`` together, so a
    weight-only run keeps activations bf16.
    """

    weight_dtype: ttnn.DataType  # qkv/q/kv, ff1/ff2, and any other quantized projection
    out_weight_dtype: ttnn.DataType  # to_out carve-out
    activation_dtype: ttnn.DataType  # linear activation cast (gated by LTX_QUANT_ACTIVATIONS)
    mm_fidelity: ttnn.MathFidelity
    mm_fp32_dest_acc: bool
    sdpa_input_dtype: ttnn.DataType  # self-attn ring-SDPA input cast (gated by LTX_QUANT_ACTIVATIONS)
    sdpa_fidelity: ttnn.MathFidelity
    sdpa_fp32_dest_acc: bool

    def linear_kwargs(self, role: str) -> dict:
        """Construction kwargs for a DiT linear of ``role`` ("out" carves out to bf16, else quantized).

        A ``gate`` keeps its bf16 weight but still consumes the shared bf8 activation, so it is pinned
        like the projections but gets no weight-dtype or activation-cast change.
        """
        pin = LTX_QUANT_ACTIVATIONS
        if role == "gate":
            return {"pin_output_bf16": pin}
        return {
            "dtype": self.out_weight_dtype if role == "out" else self.weight_dtype,
            "activation_dtype": self.activation_dtype if LTX_QUANT_ACTIVATIONS else None,
            "pin_output_bf16": pin,
        }

    def ffn_kwargs(self) -> dict:
        """Construction kwargs for the FFN (ParallelFeedForward), where ff1 and ff2 are both quantized."""
        lk = self.linear_kwargs("ff")
        return {
            "ff1_dtype": lk["dtype"],
            "ff2_dtype": lk["dtype"],
            "activation_dtype": lk["activation_dtype"],
            "pin_output_bf16": lk["pin_output_bf16"],
        }

    def mm_compute_config(self, arch):
        """Compute-kernel config for the DiT-linear matmuls (attention QKV/out and the FFN).

        ``packer_l1_acc`` stays on regardless of tier: it accumulates matmul partials in L1 at fp32,
        which is near-free and strictly helps. It is orthogonal to ``fp32_dest_acc_en`` — the pricier
        fp32 *destination* accumulation that the LoFi/bf8 tier deliberately drops (``mm_fp32_dest_acc``).
        """
        return ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=self.mm_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=self.mm_fp32_dest_acc,
            packer_l1_acc=True,
        )

    def sdpa_self_config(self, arch) -> tuple:
        """``(compute_config, input_dtype)`` for the self-attn ring SDPA; input cast rides the gate.

        Cross-attention keeps its default bf16/HiFi2 SDPA, so callers apply this to self-attn only.
        """
        compute_config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=self.sdpa_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=self.sdpa_fp32_dest_acc,
        )
        return compute_config, (self.sdpa_input_dtype if LTX_QUANT_ACTIVATIONS else None)

    @staticmethod
    def all_bf8_lofi() -> LtxQuantProfile:
        """Weights bf8, LoFi compute, bf8 activations. SDPA math stays HiFi2 with bf8 inputs.

        SDPA stays fully unquantized in math (FastVideo kept attention higher precision); only its
        inputs narrow to bf8 under ``LTX_QUANT_ACTIVATIONS``. ``ff2`` uses the RowParallel RS-fused
        addcmul, which Wan runs at bf8 with no issue, so it is quantized; only ``to_out`` carves out.
        """
        return LtxQuantProfile(
            weight_dtype=ttnn.bfloat8_b,
            out_weight_dtype=ttnn.bfloat16,
            activation_dtype=ttnn.bfloat8_b,
            mm_fidelity=ttnn.MathFidelity.LoFi,
            mm_fp32_dest_acc=False,
            sdpa_input_dtype=ttnn.bfloat8_b,
            sdpa_fidelity=ttnn.MathFidelity.HiFi2,
            sdpa_fp32_dest_acc=False,
        )

    @staticmethod
    def all_bf4_lofi() -> LtxQuantProfile:
        """Like ``all_bf8_lofi`` but 4-bit weights and 4-bit linear activations. Opt-in probe tier, not
        shipped: bf4 *activations* visibly destroy quality (bf4 weights alone hold up), so this is for
        A/B measurement only. Carve-outs and the ring-SDPA bf8 input match all_bf8_lofi.
        """
        return LtxQuantProfile(
            weight_dtype=ttnn.bfloat4_b,
            out_weight_dtype=ttnn.bfloat16,
            activation_dtype=ttnn.bfloat4_b,
            mm_fidelity=ttnn.MathFidelity.LoFi,
            mm_fp32_dest_acc=False,
            sdpa_input_dtype=ttnn.bfloat8_b,
            sdpa_fidelity=ttnn.MathFidelity.HiFi2,
            sdpa_fp32_dest_acc=False,
        )
