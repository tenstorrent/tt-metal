# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Llama-3.1-8B Model Configuration.

Single source of truth for model dimension constants.
Values from the HuggingFace config.json for meta-llama/Llama-3.1-8B.

Deliberately import-free: the prefill adapter names this class at class-definition time, and the
adapter must stay cheap to import for the H2D producers. Nothing here may import torch, ttnn or
transformers.

Mirrors ``deepseek_v3_d_p/reference/*_config.py``, but lives in this package rather than under
deepseek: Llama is dense GQA and shares no MoE/MLA substrate with that family.
"""


class Llama31_8BConfig:
    """Llama-3.1-8B model dimensions."""

    # Core dimensions
    EMB_SIZE = 4096  # embedding dimension (hidden_size)
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    INTERMEDIATE_SIZE = 14336  # dense FFN hidden dimension
    HEAD_DIM = 128

    # Model architecture
    NUM_LAYERS = 32
    VOCAB_SIZE = 128256  # unpadded; the device LM head pads to 129280 (decode only, prefill skips it)

    # Attention dimensions — GQA, group size 4
    NUM_ATTENTION_HEADS = 32
    NUM_KEY_VALUE_HEADS = 8

    # Dense model: no MoE. Kept as explicit zeros so a shared helper that reads these
    # does not have to special-case the absence of the block.
    NUM_ROUTED_EXPERTS = 0
    NUM_EXPERTS_PER_TOKEN = 0
    NUM_SHARED_EXPERTS = 0

    # Normalization / activation
    RMS_NORM_EPS = 1e-5
    SWIGLU_LIMIT = None  # Llama's SwiGLU is unclamped (contrast gpt-oss's 7.0)

    # RoPE — llama3 frequency scaling, NOT YaRN. The DeepSeek/Kimi/gpt-oss prefill lineage
    # implements YaRN; inheriting it here would silently diverge from HF as position advances.
    ROPE_THETA = 500000.0
    ROPE_TYPE = "llama3"
    ROPE_SCALING_FACTOR = 8.0
    ROPE_LOW_FREQ_FACTOR = 1.0
    ROPE_HIGH_FREQ_FACTOR = 4.0
    ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192

    # Architecture capability. The served context is a deployment choice (see the SP/TP notes in
    # tt/config.py); this is only what the architecture supports.
    MAX_POSITION_EMBEDDINGS = 131072

    # No attention sinks, no sliding window, no QK-norm, no attention/o biases: every one of the
    # 32 layers is plain full-causal GQA + dense MLP. This uniformity is why the migration address
    # table for this model needs no per-layer layout branch.
    SLIDING_WINDOW = None
    ATTENTION_BIAS = False
    USE_QK_NORM = False
