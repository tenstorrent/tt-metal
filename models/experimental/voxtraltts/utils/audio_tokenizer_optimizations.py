# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audio tokenizer decode perf presets (HiFi2/BFP8 weights, BF16 activations)."""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

from models.experimental.voxtraltts.utils.config_helpers import COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER


@dataclass(frozen=True)
class AudioTokenizerOptimizations:
    """Runtime dtype and compute-kernel settings for ``VoxtralTTAudioTokenizer``."""

    weight_dtype: ttnn.DataType
    activation_dtype: ttnn.DataType
    matmul_compute_kernel_config: ttnn.WormholeComputeKernelConfig
    sdpa_compute_kernel_config: ttnn.WormholeComputeKernelConfig
    # Use L1 for matmul in0 when T <= this (DRAM for longer decode stacks to avoid L1 overflow).
    matmul_l1_max_seq_len: int = 128


def voxtral_audio_tokenizer_default_optimizations() -> AudioTokenizerOptimizations:
    """Production decode: BFP8 weights, HiFi2 matmuls/SDPA, BF16 activations."""
    return AudioTokenizerOptimizations(
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        matmul_compute_kernel_config=COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER,
        sdpa_compute_kernel_config=COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER,
    )


def voxtral_audio_tokenizer_high_accuracy_optimizations() -> AudioTokenizerOptimizations:
    """BF16 weights + HiFi2 for module PCC / golden comparisons."""
    return AudioTokenizerOptimizations(
        weight_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
        matmul_compute_kernel_config=COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER,
        sdpa_compute_kernel_config=COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER,
    )
