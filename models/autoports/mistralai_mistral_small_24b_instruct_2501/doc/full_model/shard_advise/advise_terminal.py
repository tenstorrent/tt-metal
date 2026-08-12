# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor target for the TP4 full-model terminal graph.

The advisor does not model mesh collectives, so this captures the exact
rank-local embedding and LM-head shapes.  The runtime's embedding all-gather
and sampler candidate all-gathers are audited separately on hardware.
"""

from __future__ import annotations

import os
import sys

import torch

import ttnn  # Import the advisor environment before appending tt-metal.

MODEL_ROOT = os.environ.get("SHARD_ADVISE_MODEL_ROOT", "/home/mvasiljevic/tt-metal")
HIDDEN_SIZE = 5120
VOCAB_SIZE = 131072
LOCAL_HIDDEN_SIZE = HIDDEN_SIZE // 4
LOCAL_VOCAB_SIZE = VOCAB_SIZE // 4
LM_HEAD_SPLIT = 8192

_EMBEDDING = None
_FINAL_NORM = None
_LM_HEAD_WEIGHTS = None
_LM_HEAD_INPUT_MEM_CONFIG = None
_LM_HEAD_PROGRAM_CONFIGS = None
_LM_HEAD_COMPUTE_KERNEL = None


def terminal(tokens, hidden):
    local_embedding = ttnn.embedding(
        tokens,
        _EMBEDDING,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    normed = ttnn.rms_norm(
        hidden,
        epsilon=1.0e-5,
        weight=_FINAL_NORM,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    normed = ttnn.to_memory_config(normed, _LM_HEAD_INPUT_MEM_CONFIG)
    outputs = []
    for weight, program_config in zip(_LM_HEAD_WEIGHTS, _LM_HEAD_PROGRAM_CONFIGS):
        split = ttnn.linear(
            normed,
            weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=_LM_HEAD_COMPUTE_KERNEL,
        )
        outputs.append(ttnn.sharded_to_interleaved(split, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    logits = ttnn.concat(outputs, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return local_embedding, logits


def make_inputs(device):
    if MODEL_ROOT not in sys.path:
        sys.path.append(MODEL_ROOT)
    from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.optimized_decoder import (
        _dram_matmul_program_config,
        _dram_sharded_weight_memory_config,
        _l1_width_sharded_memory_config,
    )

    global _EMBEDDING
    global _FINAL_NORM
    global _LM_HEAD_WEIGHTS
    global _LM_HEAD_INPUT_MEM_CONFIG
    global _LM_HEAD_PROGRAM_CONFIGS
    global _LM_HEAD_COMPUTE_KERNEL

    _EMBEDDING = ttnn.from_torch(
        torch.zeros((VOCAB_SIZE, LOCAL_HIDDEN_SIZE), dtype=torch.bfloat16),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _FINAL_NORM = ttnn.from_torch(
        torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _LM_HEAD_WEIGHTS = [
        ttnn.from_torch(
            torch.zeros((HIDDEN_SIZE, LM_HEAD_SPLIT), dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=_dram_sharded_weight_memory_config(device, HIDDEN_SIZE, LM_HEAD_SPLIT),
        )
        for _ in range(LOCAL_VOCAB_SIZE // LM_HEAD_SPLIT)
    ]
    _LM_HEAD_INPUT_MEM_CONFIG = _l1_width_sharded_memory_config(
        device,
        ttnn.TILE_SIZE,
        HIDDEN_SIZE,
        10,
    )
    _LM_HEAD_PROGRAM_CONFIGS = [
        _dram_matmul_program_config(
            ttnn.TILE_SIZE,
            HIDDEN_SIZE,
            LM_HEAD_SPLIT,
            10,
            64,
            max_in0_block_w=4,
        )
        for _ in _LM_HEAD_WEIGHTS
    ]
    _LM_HEAD_COMPUTE_KERNEL = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    tokens = ttnn.from_torch(
        torch.zeros((1, 1, 1, 32), dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    hidden = ttnn.from_torch(
        torch.zeros((1, 1, 32, HIDDEN_SIZE), dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tokens, hidden
