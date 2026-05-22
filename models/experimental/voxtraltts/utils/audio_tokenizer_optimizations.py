# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audio tokenizer decode perf presets (HiFi2/BFP8 weights, BF16 activations)."""

from __future__ import annotations

from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class AudioTokenizerOptimizations:
    """Runtime dtype and compute-kernel settings for ``VoxtralTTAudioTokenizer``."""

    weight_dtype: ttnn.DataType
    activation_dtype: ttnn.DataType
    matmul_compute_kernel_config: ttnn.WormholeComputeKernelConfig
    sdpa_compute_kernel_config: ttnn.WormholeComputeKernelConfig


def voxtral_audio_tokenizer_default_optimizations() -> AudioTokenizerOptimizations:
    """Production decode: BFP8 weights, HiFi2 matmuls/SDPA, BF16 activations."""
    return AudioTokenizerOptimizations(
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        matmul_compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
        sdpa_compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
    )


def voxtral_audio_tokenizer_high_accuracy_optimizations() -> AudioTokenizerOptimizations:
    """BF16 weights + HiFi4 for module PCC / golden comparisons."""
    return AudioTokenizerOptimizations(
        weight_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
        matmul_compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        sdpa_compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
