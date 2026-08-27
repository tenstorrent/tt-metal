# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Small-4-119B Model Configuration (text tower only).

Single source of truth for model dimension constants.
Values from HuggingFace config.json for mistralai/Mistral-Small-4-119B-2603 (text_config);
the vision tower (Pixtral) and the multimodal projector are ignored for text-only prefill.

Mistral 4 is an MLA + MoE model, so it rides the shared ttMLA / ttMoE path, but it is not a
DeepSeek-schema checkpoint. ``mistral4_hf_config`` translates it into the field names the shared
device code reads:

1. ``rope_parameters`` -> ``rope_scaling`` (transformers 5.x renamed the block; ttMLA subscripts
   ``hf_config.rope_scaling[...]`` with no defaults), and ``rope_theta`` moves out of it up to the
   top level, where ttMLA reads it. ``factor = 128`` still drives the YaRN frequency interpolation.
2. ``mla_disable_yarn_mscale = True``. HF's Mistral4Attention uses a plain ``qk_head_dim ** -0.5``,
   whereas ``tt/mla/mla.py`` and ``reference/mla_reference.py`` otherwise derive a YaRN mscale
   correction from ``factor`` and square it into the softmax scale. Both sides would inflate it
   identically, so PCC stays green while the model is wrong: the flag is not safe to drop.

KNOWN GAP (not expressible in config): Mistral also scales queries by a position-dependent
``get_llama_4_attn_scale`` = 1 + 0.1*ln(1 + floor(pos / 8192)), for which ttMLA has no equivalent. It
is exactly 1.0 for every position below 8192 and steps up at each 8192-token boundary, so comparisons
against an HF reference are only meaningful at or below 8192 until ttMLA implements it.
"""

import types


class MistralSmall4Config:
    """Mistral-Small-4-119B model dimensions (text_config)."""

    # Core dimensions
    EMB_SIZE = 4096  # hidden_size
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension (also the shared expert's)
    INTERMEDIATE_SIZE = 12288  # Dense FFN hidden dimension; unused - NUM_DENSE_LAYERS is 0

    # MoE configuration
    NUM_ROUTED_EXPERTS = 128  # n_routed_experts
    NUM_EXPERTS_PER_TOKEN = 4  # num_experts_per_tok
    NUM_SHARED_EXPERTS = 1  # n_shared_experts
    NUM_EXPERT_GROUPS = 1  # n_group
    NUM_LIMITED_GROUPS = 1  # topk_group
    ROUTE_SCALE = 1.0  # routed_scaling_factor
    NORM_TOPK_PROB = True  # norm_topk_prob

    # Model architecture
    NUM_LAYERS = 36  # num_hidden_layers
    NUM_DENSE_LAYERS = 0  # first_k_dense_replace - every layer is MoE
    VOCAB_SIZE = 131072
    MAX_POSITION_EMBEDDINGS = 1048576

    # MLA dimensions
    NUM_ATTENTION_HEADS = 32
    NUM_KEY_VALUE_HEADS = 32
    Q_LORA_RANK = 1024
    KV_LORA_RANK = 256
    QK_NOPE_HEAD_DIM = 64
    QK_ROPE_HEAD_DIM = 64
    QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # == head_dim
    V_HEAD_DIM = 128

    # Norm / RoPE
    RMS_NORM_EPS = 1e-06
    ROPE_THETA = 10000.0  # nested under rope_parameters in config.json
    ROPE_INTERLEAVE = True  # Mistral rotates interleaved pairs, matching ttMLA's own layout

    # YaRN scaling, as stated in config.json rope_parameters. The MSCALE values are passed through
    # verbatim; mla_disable_yarn_mscale suppresses the mscale softmax correction - see module docstring.
    ROPE_SCALING_FACTOR = 128.0
    ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192
    ROPE_SCALING_BETA_FAST = 32.0
    ROPE_SCALING_BETA_SLOW = 1.0
    ROPE_SCALING_MSCALE = 1.0
    ROPE_SCALING_MSCALE_ALL_DIM = 1.0
    LLAMA4_SCALING_BETA = 0.1  # llama_4_scaling_beta; no ttMLA equivalent (see module docstring)

    # Misc read by the test fixtures / reference construction
    INITIALIZER_RANGE = 0.02
    ATTENTION_BIAS = False
    ATTENTION_DROPOUT = 0.0


def mistral4_hf_config(max_seq: int = 8192):
    """HF-attribute-style config the unified ttMLA reads (Mistral 4 dims, DeepSeek field names).

    Hand-built rather than read via ``AutoConfig``: the checkpoint's config exposes
    ``rope_parameters`` (with ``rope_theta`` nested inside it) where ttMLA and the vendored DeepSeek
    reference want a top-level ``rope_theta`` plus a DeepSeek-shaped ``rope_scaling`` dict, and its
    ``quantization_config`` sits on the outer (multimodal) config, so unwrapping to ``text_config``
    drops it and the fp8 loader would refuse the tensors.
    """
    C = MistralSmall4Config
    return types.SimpleNamespace(
        vocab_size=C.VOCAB_SIZE,
        hidden_size=C.EMB_SIZE,
        intermediate_size=C.INTERMEDIATE_SIZE,
        moe_intermediate_size=C.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=C.NUM_LAYERS,
        num_attention_heads=C.NUM_ATTENTION_HEADS,
        num_key_value_heads=C.NUM_KEY_VALUE_HEADS,
        kv_lora_rank=C.KV_LORA_RANK,
        q_lora_rank=C.Q_LORA_RANK,
        qk_nope_head_dim=C.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=C.QK_ROPE_HEAD_DIM,
        qk_head_dim=C.QK_HEAD_DIM,
        v_head_dim=C.V_HEAD_DIM,
        rms_norm_eps=C.RMS_NORM_EPS,
        max_position_embeddings=C.MAX_POSITION_EMBEDDINGS,
        # The runner / tests overwrite this per run; seed it so the rope tables are built consistently.
        max_seq_len=max_seq,
        # Lifted OUT of rope_parameters: rope.py reads hf_config.rope_theta at the top level.
        rope_theta=float(C.ROPE_THETA),
        rope_interleave=C.ROPE_INTERLEAVE,
        attention_bias=C.ATTENTION_BIAS,
        attention_dropout=C.ATTENTION_DROPOUT,
        initializer_range=C.INITIALIZER_RANGE,
        hidden_act="silu",
        pretraining_tp=1,
        # Mistral's softmax scale is a plain qk_head_dim ** -0.5 - see module docstring.
        mla_disable_yarn_mscale=True,
        # rope_parameters renamed to rope_scaling, with "type" retained for the reference's
        # _init_rope dispatch.
        rope_scaling={
            "type": "yarn",
            "factor": C.ROPE_SCALING_FACTOR,
            "original_max_position_embeddings": C.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "beta_fast": C.ROPE_SCALING_BETA_FAST,
            "beta_slow": C.ROPE_SCALING_BETA_SLOW,
            "mscale": C.ROPE_SCALING_MSCALE,
            "mscale_all_dim": C.ROPE_SCALING_MSCALE_ALL_DIM,
        },
        # MoE structure read by the MoE path and the pretrained cache-build path.
        first_k_dense_replace=C.NUM_DENSE_LAYERS,
        n_routed_experts=C.NUM_ROUTED_EXPERTS,
        n_shared_experts=C.NUM_SHARED_EXPERTS,
        num_experts_per_tok=C.NUM_EXPERTS_PER_TOKEN,
        n_group=C.NUM_EXPERT_GROUPS,
        topk_group=C.NUM_LIMITED_GROUPS,
        routed_scaling_factor=C.ROUTE_SCALE,
        norm_topk_prob=C.NORM_TOPK_PROB,
        # fp8 with weight_block_size null (per-tensor) rather than the [128,128] block scheme, so
        # test_utils dispatches to the per-tensor dequant path.
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "static",
            "dequantize": False,
            "weight_block_size": None,
        },
    )
