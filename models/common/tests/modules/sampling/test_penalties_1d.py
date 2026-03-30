# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Penalties1D module."""

import pytest
import torch

import ttnn
from models.common.modules.sampling.penalties_1d import (
    Penalties1D,
    Penalties1DConfig,
    PenaltyAccumulator,
    PenaltyParams,
    _materialize,
    _resolve_penalties1d_config,
)
from models.common.utility_functions import comp_pcc

# ---------------------------------------------------------------------------
# Model name constants (match test_mlp_1d.py naming convention)
# ---------------------------------------------------------------------------
LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_11B = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-v0.1"
QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
QWEN3_32B = "Qwen/Qwen3-32B"

_slow = pytest.mark.slow


def _list_collected_penalty_cases() -> list[pytest.param]:
    """
    Collected from TTTv1 demo runs (Phase B of test_case_collection.md).

    Each entry is:
        (mesh_shape, vocab_size, batch_size, seq_len,
         presence, frequency, repetition, pcc, hf_model_name)

    Source CSVs: sampling_generator_config_collected.csv,
                 sampling_generator_params_collected.csv,
                 penalties_prompt_tokens_collected.csv
    Deduplicated by (topology, vocab, batch, seq_len, penalties_active).
    """
    # fmt: off
    return [
        # --- (1,1) Mistral7B v32768 ---
        pytest.param((1, 1), 32768, 32, 128, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x1-Mistral7B-v32768-b32-s128-no-pen"),
        pytest.param((1, 1), 32768, 32, 128, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x1-Mistral7B-v32768-b32-s128-pen", marks=_slow),
        pytest.param((1, 1), 32768, 32, 1024, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x1-Mistral7B-v32768-b32-s1024-no-pen", marks=_slow),
        pytest.param((1, 1), 32768, 32, 1024, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x1-Mistral7B-v32768-b32-s1024-pen", marks=_slow),
        pytest.param((1, 1), 32768, 32, 2048, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x1-Mistral7B-v32768-b32-s2048-no-pen", marks=_slow),
        pytest.param((1, 1), 32768, 32, 2048, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x1-Mistral7B-v32768-b32-s2048-pen", marks=_slow),
        pytest.param((1, 1), 32768, 32, 4096, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x1-Mistral7B-v32768-b32-s4096-no-pen", marks=_slow),
        pytest.param((1, 1), 32768, 32, 4096, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x1-Mistral7B-v32768-b32-s4096-pen", marks=_slow),
        # --- (1,1) Llama8B v128256 ---
        pytest.param((1, 1), 128256, 1, 125, 0.0, 0.0, 1.0, 0.999, LLAMA_8B, id="1x1-Llama8B-v128256-b1-s125-no-pen"),
        # --- (1,1) Llama1B v128256 ---
        pytest.param((1, 1), 128256, 1, 71, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b1-s71-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 1, 80, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b1-s80-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 1, 115, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b1-s115-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 1, 337, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b1-s337-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 1, 512, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b1-s512-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 1, 785, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b1-s785-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 1, 16229, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b1-s16229-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 52, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s52-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 56, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s56-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 57, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s57-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 59, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s59-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 60, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s60-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 61, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s61-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 63, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s63-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 65, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s65-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 66, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s66-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 67, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s67-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 70, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s70-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 71, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s71-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 72, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s72-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 75, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s75-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 77, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s77-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 80, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s80-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 81, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s81-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 84, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s84-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 86, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s86-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 87, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s87-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 91, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s91-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 93, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s93-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 94, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s94-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 96, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s96-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 98, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s98-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 99, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s99-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 101, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s101-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 102, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s102-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 103, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s103-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 104, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s104-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 105, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s105-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 106, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s106-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 109, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s109-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 113, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s113-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 114, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s114-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 115, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s115-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 116, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s116-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 117, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s117-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 119, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s119-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 124, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s124-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 125, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s125-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 128, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s128-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 128, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s128-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 337, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s337-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 512, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s512-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 712, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s712-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 785, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s785-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 1024, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s1024-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 1024, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s1024-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 2048, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s2048-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 2048, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s2048-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 4096, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s4096-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 4096, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s4096-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 8192, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s8192-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 8192, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s8192-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 16229, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s16229-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 16384, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s16384-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 16384, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s16384-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 32768, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s32768-no-pen", marks=_slow),
        pytest.param((1, 1), 128256, 32, 32768, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x1-Llama1B-v128256-b32-s32768-pen", marks=_slow),
        # --- (1,2) Mistral7B v32768 ---
        pytest.param((1, 2), 32768, 32, 128, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x2-Mistral7B-v32768-b32-s128-no-pen"),
        pytest.param((1, 2), 32768, 32, 128, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x2-Mistral7B-v32768-b32-s128-pen", marks=_slow),
        pytest.param((1, 2), 32768, 32, 1024, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x2-Mistral7B-v32768-b32-s1024-no-pen", marks=_slow),
        pytest.param((1, 2), 32768, 32, 1024, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x2-Mistral7B-v32768-b32-s1024-pen", marks=_slow),
        pytest.param((1, 2), 32768, 32, 2048, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x2-Mistral7B-v32768-b32-s2048-no-pen", marks=_slow),
        pytest.param((1, 2), 32768, 32, 2048, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x2-Mistral7B-v32768-b32-s2048-pen", marks=_slow),
        pytest.param((1, 2), 32768, 32, 4096, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x2-Mistral7B-v32768-b32-s4096-no-pen", marks=_slow),
        pytest.param((1, 2), 32768, 32, 4096, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x2-Mistral7B-v32768-b32-s4096-pen", marks=_slow),
        # --- (1,2) Llama1B v128256 ---
        pytest.param((1, 2), 128256, 1, 71, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b1-s71-no-pen"),
        pytest.param((1, 2), 128256, 1, 80, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b1-s80-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 1, 115, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b1-s115-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 1, 337, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b1-s337-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 1, 512, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b1-s512-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 1, 785, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b1-s785-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 1, 16229, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b1-s16229-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 52, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s52-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 56, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s56-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 57, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s57-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 59, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s59-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 60, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s60-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 61, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s61-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 63, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s63-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 65, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s65-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 66, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s66-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 67, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s67-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 70, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s70-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 71, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s71-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 72, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s72-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 75, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s75-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 77, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s77-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 80, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s80-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 81, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s81-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 84, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s84-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 86, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s86-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 87, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s87-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 91, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s91-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 93, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s93-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 94, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s94-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 96, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s96-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 98, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s98-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 99, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s99-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 101, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s101-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 102, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s102-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 103, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s103-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 104, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s104-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 105, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s105-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 106, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s106-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 109, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s109-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 113, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s113-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 114, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s114-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 115, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s115-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 116, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s116-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 117, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s117-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 119, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s119-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 124, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s124-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 125, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s125-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 128, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s128-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 128, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s128-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 337, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s337-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 512, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s512-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 712, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s712-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 785, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s785-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 1024, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s1024-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 1024, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s1024-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 2048, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s2048-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 2048, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s2048-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 4096, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s4096-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 4096, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s4096-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 8192, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s8192-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 8192, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s8192-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 16229, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s16229-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 16384, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s16384-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 16384, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s16384-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 32768, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s32768-no-pen", marks=_slow),
        pytest.param((1, 2), 128256, 32, 32768, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x2-Llama1B-v128256-b32-s32768-pen", marks=_slow),
        # --- (1,8) Mixtral8x7B v32000 ---
        pytest.param((1, 8), 32000, 1, 87, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b1-s87-no-pen"),
        pytest.param((1, 8), 32000, 1, 393, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b1-s393-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 57, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s57-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 61, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s61-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 64, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s64-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 66, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s66-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 67, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s67-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 69, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s69-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 70, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s70-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 72, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s72-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 73, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s73-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 74, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s74-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 76, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s76-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 79, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s79-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 82, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s82-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 83, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s83-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 87, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s87-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 89, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s89-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 101, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s101-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 128, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s128-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 128, 1.2, 1.2, 1.5, 0.95, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s128-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 393, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s393-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 820, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s820-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 1024, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s1024-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 1024, 1.2, 1.2, 1.5, 0.95, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s1024-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 2048, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s2048-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 2048, 1.2, 1.2, 1.5, 0.95, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s2048-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 4096, 0.0, 0.0, 1.0, 0.999, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s4096-no-pen", marks=_slow),
        pytest.param((1, 8), 32000, 32, 4096, 1.2, 1.2, 1.5, 0.95, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-b32-s4096-pen", marks=_slow),
        # --- (1,8) Mistral7B v32768 ---
        pytest.param((1, 8), 32768, 32, 128, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x8-Mistral7B-v32768-b32-s128-no-pen"),
        pytest.param((1, 8), 32768, 32, 128, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x8-Mistral7B-v32768-b32-s128-pen", marks=_slow),
        pytest.param((1, 8), 32768, 32, 1024, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x8-Mistral7B-v32768-b32-s1024-no-pen", marks=_slow),
        pytest.param((1, 8), 32768, 32, 1024, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x8-Mistral7B-v32768-b32-s1024-pen", marks=_slow),
        pytest.param((1, 8), 32768, 32, 2048, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x8-Mistral7B-v32768-b32-s2048-no-pen", marks=_slow),
        pytest.param((1, 8), 32768, 32, 2048, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x8-Mistral7B-v32768-b32-s2048-pen", marks=_slow),
        pytest.param((1, 8), 32768, 32, 4096, 0.0, 0.0, 1.0, 0.999, MISTRAL_7B, id="1x8-Mistral7B-v32768-b32-s4096-no-pen", marks=_slow),
        pytest.param((1, 8), 32768, 32, 4096, 1.2, 1.2, 1.5, 0.95, MISTRAL_7B, id="1x8-Mistral7B-v32768-b32-s4096-pen", marks=_slow),
        # --- (1,8) Llama1B v128256 ---
        pytest.param((1, 8), 128256, 1, 71, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b1-s71-no-pen"),
        pytest.param((1, 8), 128256, 1, 80, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b1-s80-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 1, 115, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b1-s115-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 1, 337, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b1-s337-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 1, 512, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b1-s512-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 1, 785, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b1-s785-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 1, 16229, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b1-s16229-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 52, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s52-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 56, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s56-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 57, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s57-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 59, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s59-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 60, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s60-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 61, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s61-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 63, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s63-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 65, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s65-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 66, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s66-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 67, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s67-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 70, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s70-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 71, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s71-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 72, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s72-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 75, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s75-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 77, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s77-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 80, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s80-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 81, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s81-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 84, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s84-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 86, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s86-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 87, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s87-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 91, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s91-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 93, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s93-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 94, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s94-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 96, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s96-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 98, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s98-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 99, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s99-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 101, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s101-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 102, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s102-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 103, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s103-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 104, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s104-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 105, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s105-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 106, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s106-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 109, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s109-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 113, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s113-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 114, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s114-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 115, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s115-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 116, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s116-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 117, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s117-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 119, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s119-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 124, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s124-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 125, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s125-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 128, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s128-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 128, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s128-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 337, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s337-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 512, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s512-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 712, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s712-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 785, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s785-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 1024, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s1024-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 1024, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s1024-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 2048, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s2048-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 2048, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s2048-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 4096, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s4096-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 4096, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s4096-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 8192, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s8192-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 8192, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s8192-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 16229, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s16229-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 16384, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s16384-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 16384, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s16384-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 32768, 0.0, 0.0, 1.0, 0.999, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s32768-no-pen", marks=_slow),
        pytest.param((1, 8), 128256, 32, 32768, 1.2, 1.2, 1.5, 0.95, LLAMA_1B, id="1x8-Llama1B-v128256-b32-s32768-pen", marks=_slow),
        # --- (1,8) Qwen3-32B v151936 ---
        pytest.param((1, 8), 151936, 32, 128, 1.2, 1.2, 1.5, 0.95, QWEN3_32B, id="1x8-Qwen3-32B-v151936-b32-s128-pen"),
        # --- (1,8) Qwen2.5-72B v152064 ---
        pytest.param((1, 8), 152064, 1, 65, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b1-s65-no-pen"),
        pytest.param((1, 8), 152064, 1, 80, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b1-s80-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 1, 109, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b1-s109-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 1, 421, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b1-s421-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 1, 512, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b1-s512-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 1, 780, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b1-s780-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 48, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s48-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 50, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s50-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 51, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s51-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 54, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s54-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 55, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s55-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 56, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s56-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 57, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s57-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 59, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s59-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 60, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s60-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 61, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s61-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 64, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s64-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 65, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s65-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 68, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s68-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 69, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s69-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 71, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s71-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 74, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s74-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 75, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s75-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 76, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s76-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 80, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s80-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 81, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s81-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 85, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s85-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 87, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s87-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 88, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s88-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 90, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s90-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 92, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s92-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 93, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s93-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 95, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s95-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 96, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s96-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 97, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s97-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 98, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s98-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 99, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s99-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 100, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s100-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 101, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s101-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 102, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s102-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 103, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s103-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 107, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s107-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 108, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s108-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 109, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s109-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 110, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s110-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 111, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s111-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 113, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s113-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 118, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s118-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 119, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s119-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 128, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s128-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 128, 1.2, 1.2, 1.5, 0.95, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s128-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 421, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s421-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 512, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s512-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 707, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s707-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 780, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s780-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 1024, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s1024-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 1024, 1.2, 1.2, 1.5, 0.95, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s1024-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 2048, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s2048-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 2048, 1.2, 1.2, 1.5, 0.95, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s2048-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 4096, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s4096-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 4096, 1.2, 1.2, 1.5, 0.95, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s4096-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 8192, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s8192-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 8192, 1.2, 1.2, 1.5, 0.95, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s8192-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 16255, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s16255-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 16384, 0.0, 0.0, 1.0, 0.999, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s16384-no-pen", marks=_slow),
        pytest.param((1, 8), 152064, 32, 16384, 1.2, 1.2, 1.5, 0.95, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-b32-s16384-pen", marks=_slow),

    ]
    # fmt: on


# ==============================================================================
# Reference implementation (pure torch)
# ==============================================================================


def reference_apply_penalties(logits, prompt_mask, output_mask, output_counts, presence, frequency, repetition):
    """Pure-torch reference for penalty math, following the OpenAI API spec.

    Algorithm source: vLLM's ``apply_penalties`` in ``vllm/model_executor/layers/utils.py``
    (https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/utils.py).

    - Presence: subtract flat penalty for each token that appeared in output
    - Frequency: subtract penalty proportional to token occurrence count
    - Repetition: sign-dependent scaling for tokens in prompt OR output
      (positive logits divided by penalty, negative logits multiplied by penalty)
    """
    logits = logits.clone().float()
    output_mask_f = output_mask.float()
    output_counts_f = output_counts.float()

    # Presence: logits -= output_mask * presence  (vLLM: presence_penalties * output_mask)
    logits -= output_mask_f * presence

    # Frequency: logits -= output_counts * frequency  (vLLM: frequency_penalties * output_bin_counts)
    logits -= output_counts_f * frequency

    # Repetition: sign-dependent scaling  (vLLM: apply_repetition_penalties on combined prompt+output mask)
    combined = ((prompt_mask + output_mask) > 0).float()
    inv_rep = 1.0 / repetition
    # If logit > 0: multiply by 1/rep (shrink toward 0). If logit <= 0: multiply by rep (push away from 0).
    scale = torch.where(
        logits > 0,
        torch.where(combined.bool(), inv_rep, torch.ones_like(logits)),
        torch.where(combined.bool(), repetition, torch.ones_like(logits)),
    )
    logits *= scale
    return logits


# ==============================================================================
# Unit tests: Config and dataclasses (no device)
# ==============================================================================


class TestConfigUnit:
    def test_config_defaults(self):
        cfg = Penalties1DConfig(vocab_size=1024)
        assert cfg.max_batch_size == 32
        assert cfg.mesh_device is None
        assert cfg.sub_core_grids is None
        assert cfg.prompt_mask is None

    def test_config_not_resolved_without_mesh_device(self):
        cfg = Penalties1DConfig(vocab_size=1024)
        assert not cfg.is_resolved()

    def test_penalty_params_fields(self):
        fields = PenaltyParams.__dataclass_fields__
        assert set(fields.keys()) == {
            "prompt_mask",
            "presence_penalties",
            "frequency_penalties",
            "repetition_penalties",
            "inverse_repetition_penalties",
        }

    def test_penalty_accumulator_fields(self):
        fields = PenaltyAccumulator.__dataclass_fields__
        assert set(fields.keys()) == {"output_mask", "output_counts", "output_counts_gathered"}


# ==============================================================================
# Device tests: Config resolution and Penalties1D
# ==============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
class TestPenalties1DDevice:
    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_config(self, ttnn_mesh_device, vocab_size):
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        resolved = _resolve_penalties1d_config(cfg)
        assert resolved.is_resolved()
        assert resolved.mesh_device is ttnn_mesh_device

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_load_device_buffers(self, ttnn_mesh_device, vocab_size):
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        assert pen._device_buffers_loaded
        assert isinstance(pen._decode_src, ttnn.Tensor)
        assert isinstance(pen._zeros, ttnn.Tensor)

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_decode_forward_none_passthrough(self, ttnn_mesh_device, vocab_size):
        """forward() with None params/accum returns logits unchanged."""
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        logits_host = torch.randn(32, vocab_size, dtype=torch.bfloat16)
        logits_tt = ttnn.from_torch(logits_host, device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        result = pen.forward(logits_tt, params=None, accum=None)
        assert result is logits_tt

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_from_model_args(self, ttnn_mesh_device, vocab_size):
        """from_model_args backward compat factory."""

        class MockArgs:
            padded_vocab_size = vocab_size
            sub_core_grids = None

        pen = Penalties1D.from_model_args(ttnn_mesh_device, MockArgs())
        assert pen.config.vocab_size == vocab_size
        assert pen.config.mesh_device is ttnn_mesh_device

    def test_rejects_galaxy(self, ttnn_mesh_device):
        """from_model_args should reject 2D (Galaxy) topologies."""

        class FakeMesh:
            shape = (2, 4)

        class MockArgs:
            padded_vocab_size = 1024
            sub_core_grids = None

        with pytest.raises(ValueError, match="1D mesh topologies"):
            Penalties1D.from_model_args(FakeMesh(), MockArgs())


# ==============================================================================
# VS Reference tests — full penalty pipeline compared against pure-torch golden
# ==============================================================================


def _make_penalty_tensors_on_device(
    ttnn_mesh_device,
    B,
    *,
    prompt_mask_host,
    output_mask_host,
    output_counts_host,
    presence_val,
    frequency_val,
    repetition_val,
):
    """Helper: build PenaltyParams + PenaltyAccumulator on device from host tensors."""
    params = PenaltyParams(
        prompt_mask=ttnn.from_torch(
            prompt_mask_host,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        presence_penalties=ttnn.from_torch(
            torch.full((B, 1), presence_val),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
        frequency_penalties=ttnn.from_torch(
            torch.full((B, 1), frequency_val),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
        repetition_penalties=ttnn.from_torch(
            torch.full((B, 1), repetition_val),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
        inverse_repetition_penalties=ttnn.from_torch(
            torch.full((B, 1), 1.0 / repetition_val),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
    )
    accum = PenaltyAccumulator(
        output_mask=ttnn.from_torch(
            output_mask_host,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        output_counts=ttnn.from_torch(
            output_counts_host,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        output_counts_gathered=ttnn.from_torch(
            output_counts_host,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
    return params, accum


def _readback_logits(result_tt, ttnn_mesh_device, B, vocab_size):
    """Helper: read logits back from device to host torch tensor."""
    result_host = ttnn.to_torch(
        result_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            ttnn_mesh_device, dims=(0, 1), mesh_shape=tuple(ttnn_mesh_device.shape)
        ),
    )
    return result_host[:B, :vocab_size].float()


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
@pytest.mark.parametrize(
    "mesh_shape,vocab_size,batch_size,seq_len,presence,frequency,repetition,pcc,hf_model_name",
    _list_collected_penalty_cases(),
)
def test_penalties1d_vs_reference(
    ttnn_mesh_device,
    mesh_shape,
    vocab_size,
    batch_size,
    seq_len,
    presence,
    frequency,
    repetition,
    pcc,
    hf_model_name,
):
    """
    Test Penalties1D.decode_forward matches the pure-torch reference_apply_penalties.

    Parametrized across penalty combinations to verify each penalty type independently
    and in combination. PCC thresholds account for bfloat16 precision.
    """
    torch.manual_seed(42)
    B = 32

    pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

    # Build host tensors with realistic patterns
    logits_host = torch.randn(B, vocab_size, dtype=torch.bfloat16)

    # prompt_mask: first 50 tokens were in the prompt
    prompt_mask_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    prompt_mask_host[:, :50] = 1

    # output_mask: tokens 40-60 appeared in output (overlaps with prompt)
    output_mask_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    output_mask_host[:, 40:60] = 1

    # output_counts: tokens 40-60 appeared 1-3 times
    output_counts_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    output_counts_host[:, 40:50] = 2
    output_counts_host[:, 50:60] = 1

    # --- Reference (pure torch) ---
    expected = reference_apply_penalties(
        logits_host,
        prompt_mask_host,
        output_mask_host,
        output_counts_host,
        presence,
        frequency,
        repetition,
    )

    # --- TT device ---
    logits_tt = ttnn.from_torch(
        logits_host,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    params, accum = _make_penalty_tensors_on_device(
        ttnn_mesh_device,
        B,
        prompt_mask_host=prompt_mask_host,
        output_mask_host=output_mask_host,
        output_counts_host=output_counts_host,
        presence_val=presence,
        frequency_val=frequency,
        repetition_val=repetition,
    )

    result_tt = pen.decode_forward(logits_tt, params, accum)
    result_host = _readback_logits(result_tt, ttnn_mesh_device, B, vocab_size)

    passing, pcc_msg = comp_pcc(expected, result_host, pcc=pcc)
    assert passing, f"Penalties1D vs reference failed: {pcc_msg} (threshold={pcc})"


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
def test_penalties1d_changes_argmax(ttnn_mesh_device):
    """
    Heavy repetition penalty should change which token has the highest logit.

    Setup: token 0 has the highest logit AND appears in prompt+output.
    With repetition=5.0, the penalty should push token 0's logit down far enough
    that a different token becomes the argmax.
    """
    torch.manual_seed(123)
    B = 32
    vocab_size = 1024

    pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

    # Token 0 is the clear winner in raw logits
    logits_host = torch.randn(B, vocab_size, dtype=torch.bfloat16)
    logits_host[:, 0] = 10.0  # make token 0 dominant

    # Token 0 appears in prompt and output
    prompt_mask_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    prompt_mask_host[:, 0] = 1
    output_mask_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    output_mask_host[:, 0] = 1
    output_counts_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    output_counts_host[:, 0] = 3

    # Original argmax should be token 0
    assert logits_host[0].argmax().item() == 0

    logits_tt = ttnn.from_torch(
        logits_host,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    params, accum = _make_penalty_tensors_on_device(
        ttnn_mesh_device,
        B,
        prompt_mask_host=prompt_mask_host,
        output_mask_host=output_mask_host,
        output_counts_host=output_counts_host,
        presence_val=2.0,
        frequency_val=2.0,
        repetition_val=5.0,
    )

    result_tt = pen.decode_forward(logits_tt, params, accum)
    result_host = _readback_logits(result_tt, ttnn_mesh_device, B, vocab_size)

    # After heavy penalties, token 0 should no longer be argmax
    new_argmax = result_host[0].argmax().item()
    assert new_argmax != 0, f"Expected penalty to change argmax from 0, but it's still {new_argmax}"


# ==============================================================================
# Helper: build topology-correct PenaltyParams + PenaltyAccumulator from config
# ==============================================================================


def _make_proper_params_accum(pen: Penalties1D):
    """Build PenaltyParams + PenaltyAccumulator from the module's resolved config.

    Uses the config's LazyBuffer mesh_mappers to guarantee the correct dtype/layout/
    sharding for whatever device topology is active. This is required for methods like
    init_prompt_penalties and update_output_tokens that call _token_bin_counts_and_mask,
    which expects properly sharded output tensors.
    """
    params = PenaltyParams(
        prompt_mask=_materialize(pen.config.prompt_mask),
        presence_penalties=_materialize(pen.config.presence_penalties),
        frequency_penalties=_materialize(pen.config.frequency_penalties),
        repetition_penalties=_materialize(pen.config.repetition_penalties),
        inverse_repetition_penalties=_materialize(pen.config.inverse_repetition_penalties),
    )
    accum = PenaltyAccumulator(
        output_mask=_materialize(pen.config.output_mask),
        output_counts=_materialize(pen.config.output_counts),
        output_counts_gathered=_materialize(pen.config.output_counts_gathered),
    )
    return params, accum


# ==============================================================================
# Additional unit tests (no device)
# ==============================================================================


class TestConfigUnitMore:
    def test_buf_resolved_with_lazy_buffer(self):
        """_buf_resolved calls buf.is_resolved() for a LazyBuffer (line 97)."""
        from models.common.modules.lazy_buffer import LazyBuffer

        lb = LazyBuffer(source=torch.zeros(1), device=None)
        assert not Penalties1DConfig._buf_resolved(lb)  # is_resolved() → False (device=None)

    def test_buf_resolved_returns_false_for_none(self):
        """_buf_resolved returns False for None (baseline, complements line 95/97 tests)."""
        assert not Penalties1DConfig._buf_resolved(None)


# ==============================================================================
# Additional device tests: coverage for previously untested methods
# ==============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
class TestPenalties1DDeviceExtra:
    """Coverage for methods not exercised by the reference tests."""

    # ------------------------------------------------------------------
    # _buf_resolved ttnn.Tensor path (line 95)
    # ------------------------------------------------------------------

    def test_buf_resolved_with_tt_tensor(self, ttnn_mesh_device):
        """_buf_resolved returns True for a real ttnn.Tensor (line 95)."""
        tt = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
        )
        assert Penalties1DConfig._buf_resolved(tt)

    # ------------------------------------------------------------------
    # from_config (lines 141-145)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_from_config(self, ttnn_mesh_device, vocab_size):
        """from_config power-path classmethod (lines 141-145)."""
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen = Penalties1D.from_config(cfg)
        assert pen.config.vocab_size == vocab_size
        assert pen.config.mesh_device is ttnn_mesh_device
        assert not pen._device_buffers_loaded

    # ------------------------------------------------------------------
    # load_device_buffers idempotent guard (line 166)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_load_device_buffers_idempotent(self, ttnn_mesh_device, vocab_size):
        """Second call to load_device_buffers returns early without re-allocating (line 166)."""
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        decode_src_first = pen._decode_src
        pen.load_device_buffers()  # hits early-return at line 166
        assert pen._decode_src is decode_src_first

    # ------------------------------------------------------------------
    # init_prompt_penalties + _token_bin_counts_and_mask counts=None path
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_init_prompt_penalties(self, ttnn_mesh_device, vocab_size):
        """init_prompt_penalties scatters prompt tokens into prompt_mask."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        params, accum = _make_proper_params_accum(pen)

        prompt_tokens = torch.randint(0, vocab_size, (B, 10))
        pen.init_prompt_penalties(params, accum, prompt_tokens)

    # ------------------------------------------------------------------
    # forward() prompt init dispatch
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_forward_dispatches_to_init_prompt(self, ttnn_mesh_device, vocab_size):
        """forward() with prompt_tokens kwarg routes to init_prompt_penalties."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        params, accum = _make_proper_params_accum(pen)

        logits_tt = ttnn.from_torch(
            torch.randn(B, vocab_size, dtype=torch.bfloat16),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        prompt_tokens = torch.randint(0, vocab_size, (B, 5))
        result = pen.forward(logits_tt, params=params, accum=accum, prompt_tokens=prompt_tokens)
        assert result is logits_tt

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_forward_dispatches_to_decode(self, ttnn_mesh_device, vocab_size):
        """forward() without prompt_tokens routes to decode_forward (line 280).

        Uses unsharded (replicated) tensors — same topology as test_penalties1d_vs_reference —
        so the broadcast between logits [B, V] and penalty masks [B, V] is valid on all mesh
        shapes. The goal here is line 280 coverage, not correctness (covered elsewhere).
        """
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

        zeros_BV = torch.zeros(B, vocab_size, dtype=torch.int32)
        params, accum = _make_penalty_tensors_on_device(
            ttnn_mesh_device,
            B,
            prompt_mask_host=zeros_BV,
            output_mask_host=zeros_BV,
            output_counts_host=zeros_BV,
            presence_val=0.0,
            frequency_val=0.0,
            repetition_val=1.0,
        )
        logits_tt = ttnn.from_torch(
            torch.randn(B, vocab_size, dtype=torch.bfloat16),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        result = pen.forward(logits_tt, params=params, accum=accum)
        assert result is not None

    # ------------------------------------------------------------------
    # update_output_tokens: standard decode path (lines 290-294)
    # and _token_bin_counts_and_mask counts-not-None path (line 419)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_update_output_tokens_standard(self, ttnn_mesh_device, vocab_size):
        """update_output_tokens with standard decode-shape [1,1,1,B] (lines 290-294)."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        _, accum = _make_proper_params_accum(pen)

        # Standard sampling output: shape[-1]=B=32, shape[-2]=1 → if-branch
        tokens_tt = ttnn.from_torch(
            torch.randint(0, vocab_size, (1, 1, 1, B), dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pen.update_output_tokens(accum, tokens_tt)

    # ------------------------------------------------------------------
    # update_output_tokens: multi-token else branch (lines 296-303)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_update_output_tokens_multi_token(self, ttnn_mesh_device, vocab_size):
        """update_output_tokens with multi-token [B,S] shape triggers else-branch (lines 296-303)."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        _, accum = _make_proper_params_accum(pen)

        # shape[-1]=4 != B=32 → else-branch; src = ones(B, 4) created inline
        tokens_tt = ttnn.from_torch(
            torch.randint(0, vocab_size, (B, 4), dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pen.update_output_tokens(accum, tokens_tt)

    # ------------------------------------------------------------------
    # reset_output_tokens: tokens=None (lines 317-324) and with tokens (lines 326-346)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_reset_output_tokens_no_tokens(self, ttnn_mesh_device, vocab_size):
        """reset_output_tokens(tokens=None) zeros the accum buffers (lines 317-324)."""
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        _, accum = _make_proper_params_accum(pen)
        pen.reset_output_tokens(accum, tokens=None)

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_reset_output_tokens_with_tokens(self, ttnn_mesh_device, vocab_size):
        """reset_output_tokens(tokens=...) zeros then re-initializes from tokens (lines 326-346)."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        _, accum = _make_proper_params_accum(pen)
        tokens = torch.randint(0, vocab_size, (B, 5))
        pen.reset_output_tokens(accum, tokens=tokens)

    # ------------------------------------------------------------------
    # _pad_batch_to_max: pad (lines 399-401), truncate (402-403), ValueError (396-397)
    # ------------------------------------------------------------------

    def test_pad_batch_to_max_pads_small_batch(self, ttnn_mesh_device):
        """_pad_batch_to_max pads when B < max_batch_size (lines 399-401)."""
        pen = Penalties1D(vocab_size=1024, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        small = torch.randint(0, 100, (4, 10))
        padded = pen._pad_batch_to_max(small, pad_value=-1)
        assert padded.shape[0] == pen.config.max_batch_size
        assert (padded[4:] == -1).all()

    def test_pad_batch_to_max_truncates_large_batch(self, ttnn_mesh_device):
        """_pad_batch_to_max truncates when B > max_batch_size (lines 402-403)."""
        pen = Penalties1D(vocab_size=1024, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        large = torch.randint(0, 100, (64, 10))
        truncated = pen._pad_batch_to_max(large, pad_value=-1)
        assert truncated.shape[0] == pen.config.max_batch_size

    def test_pad_batch_to_max_raises_on_non_2d(self, ttnn_mesh_device):
        """_pad_batch_to_max raises ValueError for non-2D input (lines 396-397)."""
        pen = Penalties1D(vocab_size=1024, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        with pytest.raises(ValueError, match="Expected 2D"):
            pen._pad_batch_to_max(torch.zeros(10), pad_value=-1)

    # ------------------------------------------------------------------
    # _resolve_buf: ttnn.Tensor passthrough (lines 474-475)
    # and LazyBuffer resolve (line 476)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_buf_tensor_passthrough(self, ttnn_mesh_device, vocab_size):
        """Pre-existing ttnn.Tensor passes through _resolve_buf unchanged (lines 474-475)."""
        B = 32
        pre_tensor = ttnn.from_torch(
            torch.zeros(B, vocab_size, dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
        )
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, prompt_mask=pre_tensor)
        resolved = _resolve_penalties1d_config(cfg)
        assert resolved.prompt_mask is pre_tensor

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_buf_lazy_buffer_passthrough(self, ttnn_mesh_device, vocab_size):
        """Pre-existing LazyBuffer with device=None gets device filled in (line 476)."""
        from models.common.modules.lazy_buffer import LazyBuffer

        B = 32
        partial_lb = LazyBuffer(
            source=torch.zeros(B, vocab_size, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, prompt_mask=partial_lb)
        resolved = _resolve_penalties1d_config(cfg)
        assert isinstance(resolved.prompt_mask, LazyBuffer)
        assert resolved.prompt_mask.device is ttnn_mesh_device

    # ------------------------------------------------------------------
    # _materialize: ttnn.Tensor passthrough (line 552)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_materialize_tensor_passthrough(self, ttnn_mesh_device, vocab_size):
        """Pre-existing ttnn.Tensor as decode_src passes through _materialize (line 552)."""
        B = 32
        pre_src = ttnn.from_torch(
            torch.ones(B, 1, dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, decode_src=pre_src)
        pen = Penalties1D.from_config(cfg)
        pen.load_device_buffers()
        assert pen._decode_src is pre_src
