# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audio tokenizer decode perf presets (HiFi2/BFP8 weights, BF16 activations)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

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
    # Explicit 2D multicast matmul program configs per forward (Tier 1 decode tuning).
    matmul_program_config: bool = True
    # Native TTNN causal + sliding_window SDPA (no dense [1,H,T,T] mask; ALiBi omitted).
    sdpa_native_sliding_window: bool = False


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def voxtral_audio_tokenizer_default_optimizations() -> AudioTokenizerOptimizations:
    """Production decode: dense ALiBi SDPA + BFP8 weights, HiFi2 matmuls, BF16 activations."""
    return AudioTokenizerOptimizations(
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        matmul_compute_kernel_config=COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER,
        sdpa_compute_kernel_config=COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER,
        matmul_program_config=not _env_flag("VOXTRAL_AUDIO_TOKENIZER_MATMUL_PROGCFG_OFF"),
        sdpa_native_sliding_window=_env_flag("VOXTRAL_AUDIO_TOKENIZER_SDPA_NATIVE_WINDOW"),
    )


def voxtral_audio_tokenizer_dense_mask_sdpa_optimizations() -> AudioTokenizerOptimizations:
    """Dense ALiBi + sliding-window ``attn_mask`` SDPA (same as production default)."""
    return replace(
        voxtral_audio_tokenizer_default_optimizations(),
        sdpa_native_sliding_window=False,
    )


def voxtral_audio_tokenizer_native_sdpa_optimizations() -> AudioTokenizerOptimizations:
    """Native sliding-window SDPA (fast decode; omits ALiBi — use for perf tests only)."""
    return replace(
        voxtral_audio_tokenizer_default_optimizations(),
        sdpa_native_sliding_window=True,
    )


def voxtral_audio_tokenizer_high_accuracy_optimizations() -> AudioTokenizerOptimizations:
    """BF16 weights + HiFi2 for module PCC / golden comparisons."""
    return AudioTokenizerOptimizations(
        weight_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
        matmul_compute_kernel_config=COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER,
        sdpa_compute_kernel_config=COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER,
        sdpa_native_sliding_window=False,
    )
