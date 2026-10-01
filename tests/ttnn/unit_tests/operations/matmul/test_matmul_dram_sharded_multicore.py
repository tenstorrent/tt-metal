# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DRAM-sharded decode matmul, multi-core variant (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
with cores_per_bank > 0).

The served decode shapes come with the layouts the models give them (activation and output shard grids,
dtypes, fidelity), recorded from tt_transformers on Blackhole (P150 and the per-chip shapes of P150x4 and
P300 meshes) and on Wormhole (N150, N300, N150x4, T3K). Each case is checked against a float64 matmul of
the on-device operands (the quantized weight as stored), so the only error left is the op's own
accumulation and output rounding; the variant keeps the whole K in fp32 Dest and must be at least as
accurate as the single-reader path on the same call.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole, is_wormhole_b0

pytestmark = pytest.mark.use_module_device

DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}

# Blackhole: (K, N, weight, activation, output dtype, activation shard grid, activation shard width (tiles),
#             output shard grid, output shard width (tiles), fidelity, fp32 Dest, stock in0_block_w, cores_per_bank)
# The compute config (fidelity, fp32 Dest) and in0_block_w are what the model passes; the stock path runs
# with them as recorded. The multi-core variant takes the fidelity and always accumulates in fp32 Dest.
# A grid is ("rect", x, y) from (0, 0) or ("rm", n): n cores row-major over the worker grid.
# fmt: off
BH_SHAPES = [
    (1280, 5120, 'bfp8', 'bf16', 'bf16', ('rect', 10, 1), 4, ('rect', 10, 1), 16, 'HiFi2', True, 4, 6),  # Qwen2.5-32B, per chip P150x4
    (1792, 3584, 'bf16', 'bf16', 'bf16', ('rm', 14), 4, ('rm', 14), 8, 'HiFi4', True, 4, 2),  # Qwen2.5-7B, per chip P300 (1x2)
    (2048, 2048, 'bfp8', 'bf16', 'bf16', ('rm', 32), 2, ('rm', 32), 2, 'HiFi2', True, 8, 3),  # Llama-3.2-1B-Instruct
    (2048, 3072, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 2, ('rm', 32), 3, 'HiFi2', True, 8, 2),  # Llama-3.2-1B-Instruct
    (2048, 5120, 'bfp8', 'bf16', 'bf16', ('rm', 16), 4, ('rm', 16), 10, 'HiFi2', True, 4, 4),  # Qwen3-32B, per chip P150x4
    (2048, 8192, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 1, ('rm', 64), 4, 'LoFi', False, 2, 5),  # Llama-3.2-1B-Instruct
    (2048, 8192, 'bfp8', 'bf16', 'bf16', ('rm', 16), 4, ('rm', 16), 16, 'HiFi2', True, 4, 4),  # Llama-3.3-70B and Qwen2.5-72B, per chip P150x4
    (2048, 16032, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 1, ('rm', 63), 8, 'HiFi2', False, 1, 4),  # Llama-3.2-1B-Instruct
    (3072, 3072, 'bfp8', 'bf16', 'bf16', ('rm', 24), 4, ('rm', 24), 4, 'HiFi2', True, 8, 3),  # Llama-3.2-3B-Instruct
    (3072, 5120, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 3, ('rm', 32), 5, 'HiFi2', True, 3, 5),  # Llama-3.2-3B-Instruct
    (3072, 8192, 'bfp4', 'bf16', 'bf16', ('rect', 8, 4), 3, ('rm', 32), 8, 'LoFi', False, 3, 5),  # Llama-3.2-3B-Instruct
    (3072, 16032, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 6), 2, ('rm', 46), 11, 'HiFi2', False, 2, 4),  # Llama-3.2-3B-Instruct
    (3584, 2304, 'bf16', 'bf16', 'bf16', ('rect', 7, 4), 4, ('rm', 24), 3, 'HiFi4', True, 8, 2),  # Qwen2.5-7B, per chip P300 (1x2)
    (3584, 9728, 'bfp8', 'bf16', 'bf16', ('rect', 8, 2), 7, ('rm', 16), 19, 'HiFi4', True, 7, 4),  # Qwen2.5-7B, per chip P300 (1x2)
    (3584, 11904, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 7), 2, ('rm', 54), 7, 'HiFi2', False, 2, 4),  # Qwen2.5-7B, per chip P300 (1x2)
    (3584, 16032, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 7), 2, ('rm', 56), 9, 'HiFi2', False, 2, 4),  # Qwen2.5-7B, per chip P300 (1x2)
    (4096, 704, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, ('rect', 11, 2), 1, 'HiFi2', False, 16, 2),  # Mistral-7B-Instruct-v0.3
    (4096, 4096, 'bfp8', 'bf16', 'bf16', ('rm', 32), 4, ('rm', 32), 4, 'HiFi2', True, 4, 2),  # Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-8B
    (4096, 6144, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 4, ('rm', 32), 6, 'HiFi2', True, 4, 3),  # Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-8B
    (4096, 7648, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, ('rm', 60), 4, 'HiFi2', False, 2, 2),  # Qwen3-8B
    (4096, 12288, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 2, ('rm', 64), 6, 'LoFi', False, 2, 4),  # Qwen3-8B
    (4096, 14336, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 2, ('rm', 64), 7, 'LoFi', False, 2, 4),  # Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
    (4096, 14336, 'bfp8', 'bf16', 'bf16', ('rect', 8, 8), 2, ('rm', 64), 7, 'HiFi2', False, 2, 4),  # Llama-3.1-8B-Instruct
    (4096, 16032, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, ('rm', 63), 8, 'HiFi2', False, 2, 4),  # Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-8B
    (5120, 1792, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 5, ('rm', 28), 2, 'HiFi2', True, 10, 2),  # Qwen2.5-32B, per chip P150x4
    (5120, 1912, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 5), 4, ('rm', 30), 2, 'HiFi2', False, 8, 2),  # Qwen3-32B, per chip P150x4
    (5120, 1944, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 5), 4, ('rm', 31), 2, 'HiFi2', False, 8, 2),  # Qwen2.5-32B, per chip P150x4
    (5120, 2560, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 5, ('rm', 27), 3, 'HiFi2', True, 10, 2),  # Qwen3-32B, per chip P150x4
    (5120, 4008, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 5), 4, ('rm', 32), 4, 'HiFi2', False, 4, 2),  # Qwen2.5-32B and Qwen3-32B, per chip P150x4
    (5120, 6400, 'bfp4', 'bf16', 'bf16', ('rect', 8, 5), 4, ('rm', 40), 5, 'LoFi', False, 4, 4),  # Qwen3-32B, per chip P150x4
    (5120, 7168, 'bfp4', 'bf16', 'bf16', ('rect', 8, 4), 5, ('rm', 32), 7, 'LoFi', False, 5, 4),  # Qwen2.5-32B, per chip P150x4
    (6400, 5120, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 5), 5, ('rm', 40), 4, 'HiFi2', False, 5, 4),  # Qwen3-32B, per chip P150x4
    (7168, 5120, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 4), 7, ('rm', 32), 5, 'HiFi2', False, 7, 3),  # Qwen2.5-32B, per chip P150x4
    (7168, 8192, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 4), 7, ('rm', 32), 8, 'HiFi2', False, 7, 4),  # Llama-3.3-70B, per chip P150x4
    (8192, 1944, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 4, ('rm', 61), 1, 'HiFi2', False, 8, 3),  # Qwen2.5-72B, per chip P150x4
    (8192, 2048, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 4, ('rm', 64), 1, 'HiFi2', False, 8, 4),  # Llama-3.2-1B-Instruct
    (8192, 2560, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 8, ('rm', 27), 3, 'HiFi2', True, 8, 2),  # Llama-3.3-70B and Qwen2.5-72B, per chip P150x4
    (8192, 3072, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 4), 8, ('rm', 32), 3, 'HiFi2', False, 8, 4),  # Llama-3.2-3B-Instruct
    (8192, 4008, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 4, ('rm', 63), 2, 'HiFi2', False, 4, 2),  # Llama-3.3-70B and Qwen2.5-72B, per chip P150x4
    (8192, 7168, 'bfp4', 'bf16', 'bf16', ('rect', 8, 4), 8, ('rm', 32), 7, 'LoFi', False, 8, 4),  # Llama-3.3-70B, per chip P150x4
    (8192, 8192, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 4, ('rm', 64), 4, 'LoFi', False, 4, 4),  # Qwen2.5-72B, per chip P150x4
    (8192, 8192, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 4, ('rm', 64), 4, 'HiFi2', False, 4, 4),  # Qwen2.5-72B, per chip P150x4
    (9728, 3584, 'bfp8', 'bf16', 'bf16', ('rect', 8, 2), 19, ('rm', 16), 7, 'HiFi4', True, 1, 4),  # Qwen2.5-7B, per chip P300 (1x2)
    (12288, 4096, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 6, ('rm', 64), 2, 'HiFi2', False, 6, 2),  # Qwen3-8B
    (14336, 4096, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 7, ('rm', 64), 2, 'HiFi2', False, 7, 2),  # Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
]
# Wormhole: (K, N, weight, activation, output dtype, activation shard grid, activation shard width (tiles),
#            per_core_N (the output config carries no shard spec), fidelity, fp32 Dest, stock in0_block_w, cores_per_bank)
WH_SHAPES = [
    (384, 3072, 'bfp8', 'bf16', 'bf16', ('rect', 3, 1), 4, 32, 'HiFi2', True, 12, 1),  # Llama-3.2-3B-Instruct/T3K
    (512, 5376, 'bfp8', 'bf16', 'bf16', ('rect', 4, 1), 4, 42, 'HiFi2', True, 4, 2),  # gemma-3-27b-it/T3K
    (640, 5120, 'bfp8', 'bf16', 'bf16', ('rect', 5, 1), 4, 32, 'HiFi2', True, 4, 3),  # QwQ-32B/T3K, Qwen2.5-Coder-32B-Instruct/T3K
    (896, 3584, 'bf16', 'bf16', 'bf16', ('rect', 7, 1), 4, 16, 'HiFi4', True, 4, 2),  # Qwen2.5-7B-Instruct/N150x4
    (1024, 1152, 'bfp8', 'bf16', 'bf16', ('rect', 4, 1), 8, 9, 'HiFi2', True, 16, 1),  # gemma-3-1b-it/N150
    (1024, 2048, 'bfp8', 'bf16', 'bf16', ('rect', 8, 2), 2, 4, 'HiFi2', True, 16, 2),  # Llama-3.2-1B-Instruct/N300, Qwen2.5-VL-3B-Instruct/N300
    (1024, 2560, 'bfp8', 'bf16', 'bf16', ('rect', 4, 1), 8, 20, 'HiFi2', True, 8, 1),  # gemma-3-4b-it/N300
    (1024, 3072, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 4), 1, 3, 'HiFi2', False, 8, 2),  # Llama-3.2-3B-Instruct/T3K
    (1024, 5120, 'bfp8', 'bf16', 'bf16', ('rect', 8, 1), 4, 20, 'HiFi2', True, 4, 3),  # Qwen3-32B/T3K, Qwen3-VL-32B-Instruct/T3K
    (1152, 1536, 'bfp8', 'bf16', 'bf16', ('rect', 6, 6), 1, 2, 'HiFi2', True, 12, 1),  # gemma-3-1b-it/N150
    (1152, 6912, 'bfp4', 'bf16', 'bf16', ('rect', 6, 6), 1, 6, 'LoFi', False, 4, 2),  # gemma-3-1b-it/N150
    (1152, 21664, 'bfp8', 'bf16', 'bfp8', ('rect', 6, 6), 1, 19, 'HiFi2', False, 1, 1),  # gemma-3-1b-it/N150
    (1152, 24048, 'bfp8', 'bf16', 'bfp8', ('rect', 6, 6), 1, 21, 'HiFi2', False, 1, 1),  # gemma-3-1b-it/N150
    (1536, 3072, 'bfp8', 'bf16', 'bf16', ('rm', 12), 4, 8, 'HiFi2', True, 12, 2),  # Llama-3.2-3B-Instruct/N300
    (1536, 4096, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 2), 3, 8, 'HiFi2', False, 6, 1),  # Qwen3-8B/T3K
    (1792, 3584, 'bf16', 'bf16', 'bf16', ('rm', 14), 4, 8, 'HiFi4', True, 8, 1),  # Qwen2.5-7B-Instruct/N300, Qwen2.5-VL-7B-Instruct/N300
    (1792, 4096, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 1), 7, 16, 'HiFi2', False, 7, 1),  # Llama-3.1-8B-Instruct/T3K, Mistral-7B-Instruct-v0.3/T3K
    (2048, 384, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 2, 1, 'HiFi2', True, 16, 1),  # Llama-3.2-1B-Instruct/T3K
    (2048, 1024, 'bfp4', 'bf16', 'bf16', ('rect', 8, 4), 2, 1, 'LoFi', False, 16, 1),  # Llama-3.2-1B-Instruct/T3K
    (2048, 1280, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 2, 2, 'HiFi2', True, 16, 1),  # Qwen2.5-VL-3B-Instruct/N300
    (2048, 1536, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 2, 2, 'HiFi2', True, 16, 1),  # Llama-3.2-1B-Instruct/N300
    (2048, 2048, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 2, 2, 'HiFi2', True, 16, 1),  # Llama-3.2-1B-Instruct/N150, Qwen2.5-VL-3B-Instruct/N150
    (2048, 2560, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 2, 3, 'HiFi2', True, 8, 1),  # Qwen2.5-VL-3B-Instruct/N150, gemma-3-4b-it/N150
    (2048, 3072, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 2, 3, 'HiFi2', True, 8, 1),  # Llama-3.2-1B-Instruct/N150
    (2048, 4096, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 1, 2, 'LoFi', False, 8, 1),  # Llama-3.2-1B-Instruct/N300
    (2048, 4096, 'bfp8', 'bf16', 'bf16', ('rect', 8, 2), 4, 8, 'HiFi2', True, 8, 1),  # Llama-3.1-8B-Instruct/N300, Mistral-7B-Instruct-v0.3/N300
    (2048, 5504, 'bfp4', 'bf16', 'bf16', ('rect', 4, 1), 16, 43, 'LoFi', False, 8, 3),  # Qwen2.5-VL-3B-Instruct/N300
    (2048, 8192, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 1, 4, 'LoFi', False, 4, 2),  # Llama-3.2-1B-Instruct/N150
    (2048, 11008, 'bfp4', 'bf16', 'bf16', ('rect', 8, 1), 8, 43, 'LoFi', False, 8, 1),  # Qwen2.5-VL-3B-Instruct/N150
    (2048, 16032, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 1, 8, 'HiFi2', False, 2, 1),  # Llama-3.2-1B-Instruct/T3K
    (2048, 21376, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 1, 11, 'HiFi2', False, 1, 1),  # Llama-3.2-1B-Instruct/N300
    (2048, 23680, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 1, 12, 'HiFi2', False, 1, 1),  # Qwen2.5-VL-3B-Instruct/N150
    (2048, 33216, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 1, 17, 'HiFi2', False, 1, 1),  # Qwen2.5-VL-3B-Instruct/N300
    (2048, 42752, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 1, 21, 'HiFi2', False, 1, 1),  # Llama-3.2-1B-Instruct/N150, Qwen2.5-VL-3B-Instruct/N150
    (2560, 2048, 'bfp8', 'bf16', 'bf16', ('rect', 8, 5), 2, 2, 'HiFi2', True, 16, 1),  # gemma-3-4b-it/N300
    (2560, 4096, 'bfp8', 'bf16', 'bf16', ('rect', 8, 5), 2, 4, 'HiFi2', True, 8, 1),  # gemma-3-4b-it/N150
    (2560, 5120, 'bfp4', 'bf16', 'bf16', ('rect', 8, 5), 2, 4, 'LoFi', False, 4, 2),  # gemma-3-4b-it/N300
    (2560, 10240, 'bfp4', 'bf16', 'bf16', ('rect', 8, 5), 2, 8, 'LoFi', False, 2, 1),  # gemma-3-4b-it/N150
    (2560, 21728, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 5), 2, 17, 'HiFi2', False, 2, 1),  # gemma-3-4b-it/N150
    (2560, 24224, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 5), 2, 19, 'HiFi2', False, 2, 1),  # gemma-3-4b-it/N300
    (2560, 26720, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 5), 2, 21, 'HiFi2', False, 2, 1),  # gemma-3-4b-it/N150, gemma-3-4b-it/N300
    (2688, 5376, 'bfp8', 'bfp8', 'bf16', ('rect', 7, 6), 2, 4, 'HiFi2', False, 6, 1),  # gemma-3-27b-it/T3K
    (3072, 640, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 3, 1, 'HiFi2', True, 12, 1),  # Llama-3.2-3B-Instruct/T3K
    (3072, 1024, 'bfp4', 'bf16', 'bf16', ('rect', 8, 4), 3, 1, 'LoFi', False, 12, 1),  # Llama-3.2-3B-Instruct/T3K
    (3072, 2560, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 3, 3, 'HiFi2', True, 12, 1),  # Llama-3.2-3B-Instruct/N300
    (3072, 3072, 'bfp8', 'bf16', 'bf16', ('rect', 8, 3), 4, 4, 'HiFi2', True, 12, 1),  # Llama-3.2-3B-Instruct/N150
    (3072, 4096, 'bfp4', 'bf16', 'bf16', ('rect', 8, 4), 3, 4, 'LoFi', False, 6, 1),  # Llama-3.2-3B-Instruct/N300
    (3072, 5120, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 3, 5, 'HiFi2', True, 6, 1),  # Llama-3.2-3B-Instruct/N150
    (3072, 8192, 'bfp4', 'bf16', 'bf16', ('rect', 8, 4), 3, 8, 'LoFi', False, 3, 2),  # Llama-3.2-3B-Instruct/N150
    (3072, 16032, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 6), 2, 11, 'HiFi2', False, 2, 1),  # Llama-3.2-3B-Instruct/T3K
    (3072, 32064, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 6), 2, 21, 'HiFi2', False, 2, 1),  # Llama-3.2-3B-Instruct/N150, Llama-3.2-3B-Instruct/N300
    (3200, 5120, 'bfp8', 'bfp8', 'bf16', ('rect', 5, 4), 5, 8, 'HiFi2', False, 5, 1),  # Qwen3-32B/T3K, Qwen3-VL-32B-Instruct/T3K
    (3456, 5120, 'bfp8', 'bfp8', 'bf16', ('rect', 4, 1), 27, 40, 'HiFi2', False, 3, 1),  # Qwen2.5-Coder-32B-Instruct/T3K
    (3584, 608, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 7), 2, 1, 'HiFi2', False, 16, 1),  # Qwen2.5-7B-Instruct/N150x4
    (3584, 1152, 'bf16', 'bf16', 'bf16', ('rect', 7, 4), 4, 2, 'HiFi4', True, 16, 1),  # Qwen2.5-7B-Instruct/N150x4
    (3584, 1216, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 7), 2, 1, 'HiFi2', False, 16, 1),  # Qwen2.5-7B-Instruct/N300, Qwen2.5-VL-7B-Instruct/N300
    (3584, 2304, 'bf16', 'bf16', 'bf16', ('rect', 7, 4), 4, 3, 'HiFi4', True, 16, 1),  # Qwen2.5-7B-Instruct/N300, Qwen2.5-VL-7B-Instruct/N300
    (3584, 5120, 'bfp8', 'bf16', 'bf16', ('rect', 8, 2), 7, 10, 'HiFi4', True, 7, 1),  # Qwen2.5-7B-Instruct/N150x4, QwQ-32B/T3K
    (3584, 8192, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 2), 7, 16, 'HiFi2', False, 7, 1),  # Llama-3.3-70B-Instruct/T3K
    (3584, 9472, 'bfp8', 'bf16', 'bf16', ('rect', 8, 1), 14, 37, 'HiFi4', True, 7, 2),  # Qwen2.5-VL-7B-Instruct/N300
    (3584, 9728, 'bfp8', 'bf16', 'bf16', ('rect', 8, 2), 7, 19, 'HiFi4', True, 7, 2),  # Qwen2.5-7B-Instruct/N300
    (3584, 37408, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 7), 2, 21, 'HiFi2', False, 2, 1),  # Qwen2.5-7B-Instruct/N150x4, Qwen2.5-7B-Instruct/N300
    (4096, 768, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 4, 1, 'HiFi2', True, 16, 1),  # Llama-3.1-8B-Instruct/T3K, Mistral-7B-Instruct-v0.3/T3K
    (4096, 1536, 'bfp4', 'bf16', 'bf16', ('rect', 8, 2), 8, 3, 'LoFi', False, 16, 1),  # Qwen3-8B/T3K
    (4096, 1792, 'bfp4', 'bf16', 'bf16', ('rect', 8, 1), 16, 7, 'LoFi', False, 16, 1),  # Llama-3.1-8B-Instruct/T3K, Mistral-7B-Instruct-v0.3/T3K
    (4096, 2048, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 2, 1, 'HiFi2', False, 16, 1),  # Llama-3.2-1B-Instruct/N300
    (4096, 3072, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 4), 4, 3, 'HiFi2', False, 8, 1),  # Llama-3.2-3B-Instruct/N300, Llama-3.1-8B-Instruct/N300
    (4096, 4096, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 4, 4, 'HiFi2', True, 8, 1),  # Llama-3.1-8B-Instruct/N150, Mistral-7B-Instruct-v0.3/N150
    (4096, 6144, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 2, 3, 'LoFi', False, 4, 1),  # Qwen3-8B/N300
    (4096, 6144, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 4, 6, 'HiFi2', True, 4, 1),  # Llama-3.1-8B-Instruct/N150, Mistral-7B-Instruct-v0.3/N150
    (4096, 7168, 'bfp4', 'bf16', 'bf16', ('rect', 8, 4), 4, 7, 'LoFi', False, 4, 1),  # Llama-3.1-8B-Instruct/N300, Mistral-7B-Instruct-v0.3/N300
    (4096, 8192, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 2, 4, 'HiFi2', False, 4, 1),  # Qwen2.5-72B-Instruct/T3K, Qwen2.5-VL-72B-Instruct/T3K
    (4096, 12288, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 2, 6, 'LoFi', False, 2, 1),  # Qwen3-8B/N150
    (4096, 14336, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 2, 7, 'LoFi', False, 2, 1),  # Llama-3.1-8B-Instruct/N150, Mistral-7B-Instruct-v0.3/N150
    (4096, 16032, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, 8, 'HiFi2', False, 2, 1),  # Llama-3.1-8B-Instruct/T3K
    (4096, 16384, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, 8, 'HiFi2', False, 2, 1),  # Mistral-7B-Instruct-v0.3/N300
    (4096, 19008, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, 10, 'HiFi2', False, 2, 1),  # Qwen3-8B/T3K
    (4096, 21376, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, 11, 'HiFi2', False, 2, 1),  # Llama-3.1-8B-Instruct/N300
    (4096, 23680, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, 12, 'HiFi2', False, 2, 1),  # Qwen3-8B/N150
    (4096, 32768, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, 16, 'HiFi2', False, 2, 1),  # Mistral-7B-Instruct-v0.3/N150
    (4096, 33216, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, 17, 'HiFi2', False, 2, 1),  # Qwen3-8B/N300
    (4096, 42752, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 2, 21, 'HiFi2', False, 2, 1),  # Llama-3.1-8B-Instruct/N150, Qwen3-8B/N150
    (5120, 896, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 5, 1, 'HiFi2', True, 10, 1),  # QwQ-32B/T3K, Qwen2.5-Coder-32B-Instruct/T3K
    (5120, 1280, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 5, 2, 'HiFi2', True, 10, 1),  # Qwen3-32B/T3K, Qwen3-VL-32B-Instruct/T3K
    (5120, 2560, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 5), 4, 2, 'HiFi2', False, 8, 1),  # gemma-3-4b-it/N300
    (5120, 3200, 'bfp4', 'bf16', 'bf16', ('rect', 5, 4), 8, 5, 'LoFi', False, 8, 1),  # Qwen3-32B/T3K, Qwen3-VL-32B-Instruct/T3K
    (5120, 3456, 'bfp4', 'bf16', 'bf16', ('rect', 4, 1), 40, 27, 'LoFi', False, 10, 1),  # Qwen2.5-Coder-32B-Instruct/T3K
    (5120, 3584, 'bfp4', 'bf16', 'bf16', ('rect', 8, 2), 10, 7, 'LoFi', False, 10, 1),  # QwQ-32B/T3K, Qwen2.5-VL-32B-Instruct/T3K
    (5120, 3584, 'bfp8', 'bf16', 'bf16', ('rect', 8, 2), 10, 7, 'HiFi4', True, 10, 2),  # Qwen2.5-7B-Instruct/N150x4
    (5120, 19008, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 5), 4, 15, 'HiFi2', False, 4, 1),  # Qwen3-32B/T3K, QwQ-32B/T3K
    (5376, 1024, 'bfp8', 'bf16', 'bf16', ('rect', 7, 4), 6, 2, 'HiFi2', True, 12, 1),  # gemma-3-27b-it/T3K
    (5376, 2688, 'bfp4', 'bf16', 'bf16', ('rect', 7, 6), 4, 2, 'LoFi', False, 12, 1),  # gemma-3-27b-it/T3K
    (5376, 32800, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 7), 3, 19, 'HiFi2', False, 3, 1),  # gemma-3-27b-it/T3K
    (5504, 2048, 'bfp8', 'bfp8', 'bf16', ('rect', 4, 1), 43, 16, 'HiFi2', False, 1, 1),  # Qwen2.5-VL-3B-Instruct/N300
    (6144, 4096, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 3, 2, 'HiFi2', False, 6, 1),  # Qwen3-8B/N300
    (6912, 1152, 'bfp8', 'bfp8', 'bf16', ('rect', 6, 6), 6, 1, 'HiFi2', False, 12, 1),  # gemma-3-1b-it/N150
    (7168, 4096, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 4), 7, 4, 'HiFi2', False, 7, 1),  # Llama-3.1-8B-Instruct/N300, Mistral-7B-Instruct-v0.3/N300
    (8192, 1280, 'bfp8', 'bf16', 'bf16', ('rect', 8, 4), 8, 2, 'HiFi2', True, 16, 1),  # Qwen2.5-72B-Instruct/T3K, Llama-3.3-70B-Instruct/T3K
    (8192, 2048, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 4, 1, 'HiFi2', False, 16, 1),  # Llama-3.2-1B-Instruct/N150
    (8192, 3072, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 4), 8, 3, 'HiFi2', False, 8, 1),  # Llama-3.2-3B-Instruct/N150
    (8192, 3584, 'bfp4', 'bf16', 'bf16', ('rect', 8, 2), 16, 7, 'LoFi', False, 8, 1),  # Llama-3.3-70B-Instruct/T3K
    (8192, 4096, 'bfp4', 'bf16', 'bf16', ('rect', 8, 8), 4, 2, 'LoFi', False, 8, 1),  # Qwen2.5-72B-Instruct/T3K, Qwen2.5-VL-72B-Instruct/T3K
    (8192, 16032, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 4, 8, 'HiFi2', False, 4, 1),  # Llama-3.3-70B-Instruct/T3K
    (8192, 19008, 'bfp8', 'bf16', 'bfp8', ('rect', 8, 8), 4, 10, 'HiFi2', False, 4, 1),  # Qwen2.5-72B-Instruct/T3K, Qwen2.5-VL-72B-Instruct/T3K
    (9472, 3584, 'bfp8', 'bf16', 'bf16', ('rect', 8, 1), 37, 14, 'HiFi4', True, 1, 2),  # Qwen2.5-VL-7B-Instruct/N300
    (9728, 3584, 'bfp8', 'bf16', 'bf16', ('rect', 8, 2), 19, 7, 'HiFi4', True, 1, 2),  # Qwen2.5-7B-Instruct/N300
    (10240, 2560, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 5), 8, 2, 'HiFi2', False, 8, 1),  # gemma-3-4b-it/N150
    (11008, 2048, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 1), 43, 8, 'HiFi2', False, 1, 1),  # Qwen2.5-VL-3B-Instruct/N150
    (12288, 4096, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 6, 2, 'HiFi2', False, 6, 1),  # Qwen3-8B/N150
    (14336, 4096, 'bfp8', 'bfp8', 'bf16', ('rect', 8, 8), 7, 2, 'HiFi2', False, 7, 1),  # Llama-3.1-8B-Instruct/N150, Mistral-7B-Instruct-v0.3/N150
]
# fmt: on


def _grid(device, layout):
    if layout[0] == "rect":
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(layout[1] - 1, layout[2] - 1))})
    return ttnn.num_cores_to_corerangeset(layout[1], device.compute_with_storage_grid_size(), row_wise=True)


def _width_sharded_l1(grid, width_tiles):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, width_tiles * 32), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _operands(device, k, n, w_dtype, x_dtype, x_layout, x_shard_w, seed=0):
    torch.manual_seed(seed)
    x_mc = _width_sharded_l1(_grid(device, x_layout), x_shard_w)
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, k), dtype=DTYPES[x_dtype], layout=ttnn.TILE_LAYOUT, device=device, memory_config=x_mc
    )
    banks = device.dram_grid_size().x
    padded_n = math.ceil(n / (32 * banks)) * 32 * banks
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    w_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (k, padded_n // banks), ttnn.ShardOrientation.ROW_MAJOR),
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, k, n), dtype=DTYPES[w_dtype], layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_mc
    )
    return x, w


def _matmul(
    x, w, *, out_mc, out_dtype, per_core_n, in0_block_w, cores_per_bank, fidelity, fp32=True, fused_activation=None
):
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fused_activation=fused_activation,
        cores_per_bank=cores_per_bank,
    )
    compute = ttnn.init_device_compute_kernel_config(
        x.device().arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )
    return ttnn.linear(
        x,
        w,
        program_config=program_config,
        memory_config=out_mc,
        dtype=DTYPES[out_dtype],
        compute_kernel_config=compute,
    )


def _reference(x, w, n):
    """float64 product of the operands as they sit on the device (the quantized weight)."""
    xt = ttnn.to_torch(x).double().reshape(32, -1)
    wt = ttnn.to_torch(w).double().reshape(xt.shape[1], -1)[:, :n]
    return xt @ wt


def _rel_err(out, ref):
    return ((out.double() - ref).norm() / ref.norm()).item()


def _check(out, ref, out_dtype):
    passed, pcc = comp_pcc(ref, out.double(), 0.9999)
    assert passed, pcc
    err = _rel_err(out, ref)
    # bf16 output rounding alone is ~0.004 relative; bfloat8_b output (the LM-head splits) ~0.008.
    assert err < (0.007 if out_dtype == "bf16" else 0.012), err
    return err


def _out(device, case_out_layout, out_shard_w):
    return _width_sharded_l1(_grid(device, case_out_layout), out_shard_w)


def _bh_case(device, case, cores_per_bank, in0_block_w=2):
    k, n, wd, xd, od, x_layout, x_sw, out_layout, out_sw, fid, fp32, stock_bw, _ = case
    x, w = _operands(device, k, n, wd, xd, x_layout, x_sw)
    out_mc = _out(device, out_layout, out_sw)
    kw = dict(out_mc=out_mc, out_dtype=od, per_core_n=out_sw, fidelity=fid, fp32=fp32)
    out = _matmul(x, w, in0_block_w=in0_block_w, cores_per_bank=cores_per_bank, **kw)
    ref = _reference(x, w, n)
    return x, w, kw, ttnn.to_torch(out).reshape(32, -1)[:, :n], ref


def _id(case):
    return f"{case[0]}x{case[1]}-{case[2]}"


# One shape per projection kind (QKV, WO, W1/W3 bfp4, W2, LM split, plus the uneven and few-shard corners) runs
# on every invocation; the rest of the served shapes are marked slow and run with `--runslow`.
_REPRESENTATIVE = {
    "4096x6144-bfp8",
    "4096x4096-bfp8",
    "4096x14336-bfp4",
    "14336x4096-bfp8",
    "4096x16032-bfp8",
    "4096x704-bfp8",
    "9728x3584-bfp8",
    "2048x5504-bfp4",
    "2048x42752-bfp8",
    "1280x5120-bfp8",
}


def _cases(shapes):
    return [pytest.param(c, id=_id(c), marks=() if _id(c) in _REPRESENTATIVE else pytest.mark.slow) for c in shapes]


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole shapes (8 DRAM banks)")
@pytest.mark.parametrize("case", _cases(BH_SHAPES))
def test_served_shapes_blackhole(device, case):
    x, w, kw, out, ref = _bh_case(device, case, case[-1])
    err = _check(out, ref, case[4])
    stock = _matmul(x, w, in0_block_w=case[11], cores_per_bank=0, **kw)
    stock_err = _rel_err(ttnn.to_torch(stock).reshape(32, -1)[:, : case[1]], ref)
    assert err <= stock_err + 5e-4, (err, stock_err)


@pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole shapes (12 DRAM banks)")
@pytest.mark.parametrize("case", _cases(WH_SHAPES))
def test_served_shapes_wormhole(device, case):
    k, n, wd, xd, od, x_layout, x_sw, pcn, fid, fp32, stock_bw, cores_per_bank = case
    x, w = _operands(device, k, n, wd, xd, x_layout, x_sw)
    kw = dict(out_mc=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, out_dtype=od, per_core_n=pcn, fidelity=fid, fp32=fp32)
    out = _matmul(x, w, in0_block_w=2, cores_per_bank=cores_per_bank, **kw)
    ref = _reference(x, w, n)
    err = _check(ttnn.to_torch(out).reshape(32, -1)[:, :n], ref, od)
    stock = _matmul(x, w, in0_block_w=stock_bw, cores_per_bank=0, **kw)
    stock_err = _rel_err(ttnn.to_torch(stock).reshape(32, -1)[:, :n], ref)
    assert err <= stock_err + 5e-4, (err, stock_err)


def _bank_case(device, k, n, wd="bfp8"):
    """A layout like the models': activation on 32 cores (8 x 4), output on per_core_N tiles per core."""
    banks = device.dram_grid_size().x
    nt = math.ceil(n / 32)
    out_sw = math.ceil(nt / 32)
    x_sw = k // 32 // 32
    return (k, n, wd, "bf16", "bf16", ("rect", 8, 4), x_sw, ("rm", math.ceil(nt / out_sw)), out_sw, "HiFi2", True, 1, 1)


@pytest.mark.parametrize(
    "k, n, cores_per_bank",
    [
        # columns split unevenly: 16 or 24 columns per bank over 3, 5, 6, 7 cores
        (4096, 4096, 1),
        (4096, 4096, 3),
        (4096, 4096, 5),
        (4096, 6144, 7),
        # several column passes per core (more than 8 columns each)
        (4096, 14336, 2),
        # more cores than columns: K row groups and an in-group reduce-scatter
        (7168, 256, 2),
        (7168, 256, 4),
        (2048, 512, 6),
    ],
)
def test_cores_per_bank(device, k, n, cores_per_bank):
    banks = device.dram_grid_size().x
    grid = device.compute_with_storage_grid_size()
    nbt = math.ceil(math.ceil(n / 32) / banks)
    column_groups = min(cores_per_bank, nbt)
    row_groups = cores_per_bank // nbt if cores_per_bank > nbt else 1
    if column_groups * row_groups * math.ceil(math.ceil(n / 32) / nbt) > grid.x * grid.y:
        pytest.skip("more cores than the worker grid")
    case = _bank_case(device, k, n)
    *_, out, ref = _bh_case(device, case, cores_per_bank)
    _check(out, ref, "bf16")


@pytest.mark.parametrize("in0_block_w", [1, 3, 5, 8])
def test_k_block_width(device, in0_block_w):
    """in0_block_w is the K tiles per streamed weight block; the last block may be short (K = 4096 is
    128 tiles, which 3 and 5 do not divide)."""
    case = _bank_case(device, 4096, 4096)
    *_, out, ref = _bh_case(device, case, 2, in0_block_w=in0_block_w)
    _check(out, ref, "bf16")


def test_program_cache_reuses_the_program(device):
    """A second call with new tensors of the same specs reuses the program and still reads the new
    addresses (the variant has no per-core runtime arguments; the tensors arrive by binding)."""
    case = _bank_case(device, 4096, 4096)
    _, _, kw, out0, ref0 = _bh_case(device, case, 2)
    entries = device.num_program_cache_entries()
    x, w = _operands(device, 4096, 4096, "bfp8", "bf16", case[5], case[6], seed=1)
    out1 = _matmul(x, w, in0_block_w=2, cores_per_bank=2, **kw)
    assert device.num_program_cache_entries() == entries
    _check(ttnn.to_torch(out1).reshape(32, -1), _reference(x, w, 4096), "bf16")
    _check(out0, ref0, "bf16")


def test_rejects_unsupported(device, expect_error):
    case = _bank_case(device, 4096, 4096)
    x, w = _operands(device, 4096, 4096, "bfp8", "bf16", case[5], case[6])
    kw = dict(out_mc=_out(device, case[7], case[8]), out_dtype="bf16", per_core_n=case[8], fidelity="HiFi2")
    with expect_error(RuntimeError, "fused activation"):
        _matmul(
            x, w, in0_block_w=2, cores_per_bank=2, fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), **kw
        )
