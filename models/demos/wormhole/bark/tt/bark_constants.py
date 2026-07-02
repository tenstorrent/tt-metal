# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small token vocabulary constants.

Single source of truth -- imported by bark_model.py, tests, and utilities.
Values originate from HuggingFace suno/bark-small generation configs:
  BarkSemanticGenerationConfig, BarkCoarseGenerationConfig.
"""

# --- Semantic stage ---
SEMANTIC_VOCAB_SIZE = 10_000
SEMANTIC_PAD_TOKEN = 10_000  # EOS for semantic stage
TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_INFER_TOKEN = 129_599

# --- Coarse stage ---
COARSE_SEMANTIC_PAD_TOKEN = 12_048
CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
COARSE_INFER_TOKEN = COARSE_SEMANTIC_PAD_TOKEN + N_COARSE_CODEBOOKS  # 12_050
