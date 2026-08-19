# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Small-4-119B Model Configuration (text tower only).

Single source of truth for model dimension constants.
Values from HuggingFace config.json for mistralai/Mistral-Small-4-119B-2603 (text_config);
the vision tower (Pixtral) and the multimodal projector are ignored for text-only prefill.

Mistral 4 is an MLA + MoE model (q_a/q_b LoRA, kv_a_proj_with_mqa, compressed KVPE cache), so it
rides the shared ttMLA / ttMoE path. It is NOT a DeepSeek-schema checkpoint, so ``mistral4_hf_config``
translates it into the field names the shared device code reads. Three translations, all load-bearing:

1. ``rope_parameters`` -> ``rope_scaling``. transformers 5.x renamed the block; ttMLA subscripts
   ``hf_config.rope_scaling[...]`` with no defaults (tt/mla/rope.py:47-52).
2. ``rope_theta`` moves UP a level. Mistral nests it inside ``rope_parameters``; ttMLA reads
   ``hf_config.rope_theta`` at the top level (tt/mla/rope.py:45).
3. ``mscale`` / ``mscale_all_dim`` are deliberately NOT the checkpoint's 1.0 — see below.

The softmax-scale divergence (3) is the subtle one. HF's own Mistral4Attention uses a plain
``self.scaling = qk_head_dim ** -0.5`` with no YaRN mscale correction at all
(transformers/models/mistral4/modeling_mistral4.py:416). Both this package's device path and its
vendored DeepSeek reference instead derive an mscale from the YaRN factor, and they key off
DIFFERENT fields to do it:

    device    tt/mla/mla.py:382-385                   mscale     = 0.1*mscale*ln(factor) + 1
    reference reference/modeling_deepseek.py:698-704  mscale_all_dim, same formula

With the checkpoint's literal ``mscale = mscale_all_dim = 1.0`` and ``factor = 128``, both sides
compute 0.1*ln(128)+1 = 1.4852 and square it into the softmax scale — a 2.206x inflation that HF
Mistral does not apply. Because BOTH sides inflate identically, a device-vs-reference PCC test stays
green while being 2.2x wrong against the real model. Setting both to 0.0 collapses the correction to
exactly 1.0 on both paths (device: 0.1*0*ln(128)+1 = 1; reference: ``if mscale_all_dim:`` is False),
restoring Mistral's plain qk_head_dim**-0.5.

This costs nothing elsewhere: rope.py forces ``mscale_all_dim = mscale`` before building the tables
(rope.py:56), so the YaRN _mscale ratio is 1.0 for ANY value and the cos/sin tables are byte-identical.
``factor = 128`` is still carried, so the YaRN frequency interpolation — which Mistral DOES use —
is unaffected. The checkpoint's literal values are preserved on the constants class below.

KNOWN GAP (not expressible in config): Mistral additionally scales queries by a position-dependent
``get_llama_4_attn_scale`` = 1 + 0.1*ln(1 + floor(pos / 8192)) (modeling_mistral4.py:367-369). ttMLA
has no equivalent. It is exactly 1.0 for positions < 8192, so short-context runs are unaffected;
it reaches ~1.07 at 16k and ~1.19 at 56k tokens.
"""

import types


class Mistral4Small119BConfig:
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
    QK_HEAD_DIM = 128  # qk_nope_head_dim + qk_rope_head_dim; == head_dim
    V_HEAD_DIM = 128

    # Norm / RoPE
    RMS_NORM_EPS = 1e-06
    ROPE_THETA = 10000.0  # nested under rope_parameters in config.json
    ROPE_INTERLEAVE = True  # Mistral rotates interleaved pairs, matching ttMLA's own layout

    # YaRN scaling (config.json rope_parameters). MSCALE / MSCALE_ALL_DIM are recorded here as the
    # checkpoint states them; mistral4_hf_config deliberately passes 0.0 instead - see module docstring.
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


# Collapses the DeepSeek YaRN mscale correction to exactly 1.0 on both the device and the vendored
# reference, matching HF Mistral4Attention's plain qk_head_dim**-0.5. See the module docstring.
_MSCALE_DISABLED = 0.0


def mistral4_hf_config(max_seq: int = 8192):
    """HF-attribute-style config the unified ttMLA reads (Mistral 4 dims, DeepSeek field names).

    Hand-built rather than taken from ``AutoConfig``: transformers 5.12 DOES ship a loadable
    ``Mistral4Config``, but it exposes ``rope_parameters`` (with ``rope_theta`` nested inside it),
    while ttMLA and the vendored DeepSeek reference both read a top-level ``rope_theta`` plus a
    DeepSeek-shaped ``rope_scaling`` dict. This is a schema translation, not a fallback.
    """
    C = Mistral4Small119BConfig
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
        # rope_parameters renamed to rope_scaling, with "type" retained for the reference's
        # _init_rope dispatch. factor/beta_* drive the YaRN frequency interpolation (Mistral uses it);
        # mscale/mscale_all_dim are zeroed to disable the softmax correction (Mistral does not).
        rope_scaling={
            "type": "yarn",
            "factor": C.ROPE_SCALING_FACTOR,
            "original_max_position_embeddings": C.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "beta_fast": C.ROPE_SCALING_BETA_FAST,
            "beta_slow": C.ROPE_SCALING_BETA_SLOW,
            "mscale": _MSCALE_DISABLED,
            "mscale_all_dim": _MSCALE_DISABLED,
        },
        # MoE structure read by the MoE path and the pretrained cache-build path
        # (load_and_compute_layer_by_layer). first_k_dense_replace = 0: every layer is MoE.
        first_k_dense_replace=C.NUM_DENSE_LAYERS,
        n_routed_experts=C.NUM_ROUTED_EXPERTS,
        n_shared_experts=C.NUM_SHARED_EXPERTS,
        num_experts_per_tok=C.NUM_EXPERTS_PER_TOKEN,
        n_group=C.NUM_EXPERT_GROUPS,
        topk_group=C.NUM_LIMITED_GROUPS,
        routed_scaling_factor=C.ROUTE_SCALE,
        norm_topk_prob=C.NORM_TOPK_PROB,
        # Checkpoint is fp8 with activation_scheme "static" and weight_block_size null (per-tensor),
        # NOT the [128,128] block scheme the shared dequantizer implements. Carried verbatim so the
        # mismatch surfaces where dequant happens rather than as a missing attribute; this is why
        # the adapter leaves supports_pretrained off.
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "static",
            "dequantize": False,
            "weight_block_size": None,
        },
    )
