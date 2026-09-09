# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint constants of MiniMax-Music3's autoregressive stage.

Transcribed from diffusers ``modular_pipelines/minimax_music3/encoders.py`` (Apache-2.0). These
are part of the checkpoint contract: the Qwen3 backbone's vocabulary has the 16384 semantic audio
codes appended at ``AUDIO_CODE_OFFSET`` and the AR loop samples only from that range plus the end
token.
"""

AUDIO_END_TOKEN_ID = 151670
AUDIO_CFG_TOKEN_ID = 151654
AUDIO_CODE_OFFSET = 151675
SEMANTIC_VOCAB_SIZE = 16384
AR_CFG_SCALE = 1.5
AR_CFG_TOP_K = 50
AR_SAMPLING_TOP_K = 50
MAX_PROMPT_TOKENS = 5000
MAX_AUDIO_FRAMES = 9000
FRAME_RATE = 25.0

# RVQ depth decoder (``rvq_depth_decoder/config.json``): 8 codebooks, the first is the semantic code
# emitted by the Qwen3 backbone, the other 7 are residual codes with a 1024-entry vocabulary each.
NUM_CODEBOOKS = 8
AUDIO_VOCAB_SIZE = 1024

# Backbone geometry (``language_model/config.json``).
LLM_HIDDEN = 4096
LLM_VOCAB = 200000
LLM_MAX_POSITION_EMBEDDINGS = 10240
LLM_BATCH = 2  # row 0 = conditional prompt, row 1 = unconditional (CFG) prompt
