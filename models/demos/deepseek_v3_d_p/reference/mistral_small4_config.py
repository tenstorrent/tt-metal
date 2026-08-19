# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Mistral Small 4 119B Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for `mistralai/Mistral-Small-4-119B-2603`
(the ``text_config`` block; the checkpoint is a ``Mistral3ForConditionalGeneration``
wrapper carrying a Pixtral ``vision_config`` that this text-only prefill path ignores).

Architecture summary: **DeepSeek-family MLA attention with GPT-OSS-family MoE routing.**
The attention weight names are DeepSeek's exactly (``q_a_proj`` / ``q_a_layernorm`` /
``q_b_proj`` / ``kv_a_proj_with_mqa`` / ``kv_a_layernorm`` / ``kv_b_proj`` / ``o_proj``),
but the router is a plain softmax top-4 with no correction bias and no expert groups,
and the expert weights ship as ONE stacked fp8 tensor per projection with gate and up
fused, not as per-expert ``mlp.experts.{i}.*`` entries.
"""

import types


class MistralSmall4Config:
    """Mistral Small 4 119B model dimensions."""

    # Core dimensions
    EMB_SIZE = 4096  # embedding dimension (hidden_size)
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 12288  # shared-expert / dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 128
    NUM_EXPERTS_PER_TOKEN = 4
    NUM_SHARED_EXPERTS = 1
    # n_group / topk_group are both 1: no expert-group routing, the gate collapses to plain top-k.
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    # routed_scaling_factor = 1.0 (DeepSeek-V3 uses 2.5, GLM 2.5, V4-Flash 1.5).
    ROUTE_SCALE = 1.0
    # ⚠ NOT USED by the gate mode this model must run under, and deliberately left at the family
    # default rather than set to "softmax". The grouped-topk / hash-gate kernels accept ONLY
    # "sigmoid" or "sqrtsoftplus" (moe_grouped_topk.cpp:17-23 TT_THROWs on anything else) — there
    # is no softmax score_func. Mistral's router (softmax over all experts -> top-4 -> renormalize,
    # `norm_topk_prob: true`) is mathematically IDENTICAL to GPT-OSS routing (top-k on raw logits ->
    # softmax over just the selected k), because softmax is monotonic so the top-k indices agree and
    # the renormalization is exactly the softmax over the selected subset. So Mistral runs
    # GateComputeMode.GPT_DEVICE, where score_func is never read.
    # If someone runs this model under DEVICE / DEVICE_FP32 instead, they get a SIGMOID router
    # affinity silently — wrong numbers, not a crash. That is the tripwire this comment exists for.
    SCORE_FUNC = "sigmoid"

    # Model architecture
    NUM_LAYERS = 36
    # first_k_dense_replace = 0 -> there are NO leading dense-FFN layers. Every one of the 36 layers
    # is MoE. DeepSeek-V3 and GLM-5.1 both have 3; this is the first resident with zero.
    NUM_DENSE_LAYERS = 0
    VOCAB_SIZE = 131072
    MAX_POSITION_EMBEDDINGS = 1048576  # 1M context

    # MLA dimensions
    NUM_ATTENTION_HEADS = 32
    NUM_KEY_VALUE_HEADS = 32
    Q_LORA_RANK = 1024
    # ⚠ 256 is unprecedented in this family (DeepSeek-V3 / Kimi / GLM all use 512). It makes the
    # packed-FP8 KV cache's rope_offset_bytes = 264, which is not 16-byte aligned and fails
    # MlaKvCacheFormat.SCALED_FP8's validate_scaled(). Dense MLA's allocate_kv_cache does not select
    # a packed format by default, so this only bites when SCALED_FP8 is opted into.
    KV_LORA_RANK = 256
    QK_NOPE_HEAD_DIM = 64
    QK_ROPE_HEAD_DIM = 64
    QK_HEAD_DIM = 128  # = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
    V_HEAD_DIM = 128
    HEAD_DIM = 128

    # Other
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 10000.0
    INITIALIZER_RANGE = 0.02


def mistral4_hf_config(max_seq: int = 8192):
    """HF-attribute-style config the unified ttMLA reads (Mistral Small 4 dims, YaRN factor 128).

    Hand-built for the same reason GLM's is (see ``glm_5_1_config.glm_hf_config``): ttMLA, rope and
    the cache-build path read a *curated field set*, and Mistral's own ``config.json`` does not
    present it in the shape they index. Specifically:

      * ``rope_parameters`` -> ``rope_scaling``. transformers 5.x renamed the block; ``tt/mla/rope.py``
        indexes ``hf_config.rope_scaling[...]``. This is the known silent-wrong-answer gotcha.
      * ``rope_theta`` must be **hoisted**. Mistral nests it INSIDE ``rope_parameters``;
        ``rope.py`` reads it as a *top-level* ``hf_config.rope_theta``. So the mapping is a rename
        PLUS a hoist, not a rename alone.
      * ``max_seq_len`` is a top-level field this family uses and no HF config has.

    Field set actually consumed (verified by grep, not assumed):
      * ``tt/mla/mla.py``  -> hidden_size, num_attention_heads, kv_lora_rank, q_lora_rank,
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim, rms_norm_eps, rope_scaling["factor"|"mscale"]
      * ``tt/mla/rope.py`` -> max_seq_len, qk_rope_head_dim, rope_theta, and rope_scaling's
        factor / original_max_position_embeddings / beta_fast / beta_slow / mscale / mscale_all_dim

    YaRN: ``factor=128.0`` with ``mscale=1.0`` gives an attention softmax scale of
    ``qk_head_dim**-0.5 * m**2`` where ``m = 0.1*mscale*ln(128) + 1 = 1.4852`` (``mla.py:382-385``),
    i.e. ``128**-0.5 * 2.2058 = 0.19497`` rather than the bare ``0.08839``. Note ``mla.py`` hardcodes
    that ``0.1``; Mistral's config exposes it as ``llama_4_scaling_beta: 0.1``. The two agree TODAY,
    so no code change is needed — but a future Mistral shipping a different beta would be silently
    mis-scaled. See the open questions in the training log.
    """
    return types.SimpleNamespace(
        vocab_size=MistralSmall4Config.VOCAB_SIZE,
        hidden_size=MistralSmall4Config.EMB_SIZE,
        intermediate_size=MistralSmall4Config.INTERMEDIATE_SIZE,
        moe_intermediate_size=MistralSmall4Config.MOE_INTERMEDIATE_SIZE,
        num_attention_heads=MistralSmall4Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=MistralSmall4Config.NUM_KEY_VALUE_HEADS,
        num_hidden_layers=MistralSmall4Config.NUM_LAYERS,
        # --- MLA ---
        kv_lora_rank=MistralSmall4Config.KV_LORA_RANK,
        q_lora_rank=MistralSmall4Config.Q_LORA_RANK,
        qk_nope_head_dim=MistralSmall4Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=MistralSmall4Config.QK_ROPE_HEAD_DIM,
        v_head_dim=MistralSmall4Config.V_HEAD_DIM,
        rms_norm_eps=MistralSmall4Config.RMS_NORM_EPS,
        attention_bias=False,
        # --- fields the CPU *reference* needs that ttMLA itself does not ---
        # `create_mla_reference` instantiates deepseek_v3/reference/modeling_deepseek.py's
        # DeepseekV3Attention, which reads a wider field set than the device path. Omitting any of
        # these is an AttributeError on a SimpleNamespace, ~11 s into the test, before the device is
        # touched. Found by running it, not by reading:
        #   attention_dropout        modeling_deepseek.py:657
        #   max_position_embeddings  modeling_deepseek.py:660
        #   rope_scaling["type"]     modeling_deepseek.py:714  (_init_rope dispatches on it)
        attention_dropout=0.0,
        max_position_embeddings=MistralSmall4Config.MAX_POSITION_EMBEDDINGS,
        # Required by the `random_weights` test fixture (conftest.py:932), which scales every random
        # MLA tensor by it. Not read by ttMLA itself, but a hand-built config that omits it makes
        # every random-weight test fail with AttributeError before reaching the device.
        initializer_range=MistralSmall4Config.INITIALIZER_RANGE,
        # --- rope: rope_parameters -> rope_scaling, with rope_theta hoisted out ---
        max_seq_len=max_seq,
        rope_theta=float(MistralSmall4Config.ROPE_THETA),
        rope_scaling={
            # config.json carries BOTH "rope_type" and "type" (= "yarn"). The device path never reads
            # either, but the reference's _init_rope dispatches on rope_scaling["type"], so it must be
            # present or the reference silently is not YaRN. Keep both, as the checkpoint does.
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 128.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            # NOTE: config.json says 8192 — the PRE-extension training length, not max_seq. GLM's
            # builder passes its own max_seq here because GLM's factor is 1.0 (YaRN disabled) so the
            # value is inert. For Mistral factor=128 makes YaRN live, and the ramp
            # (beta_fast/beta_slow -> low/high correction dims) is computed against THIS number.
            # Substituting max_seq here would change the frequency ramp -> wrong rope. Keep 8192.
            "original_max_position_embeddings": 8192,
        },
        # --- MoE structure read by the pretrained cache-build path ---
        first_k_dense_replace=MistralSmall4Config.NUM_DENSE_LAYERS,
        n_routed_experts=MistralSmall4Config.NUM_ROUTED_EXPERTS,
        num_experts_per_tok=MistralSmall4Config.NUM_EXPERTS_PER_TOKEN,
        n_shared_experts=MistralSmall4Config.NUM_SHARED_EXPERTS,
        n_group=MistralSmall4Config.NUM_EXPERT_GROUPS,
        topk_group=MistralSmall4Config.NUM_LIMITED_GROUPS,
        norm_topk_prob=True,
        routed_scaling_factor=MistralSmall4Config.ROUTE_SCALE,
        # --- quantization ---
        # ⚠ Deliberately NOT the [128,128] block shape every other resident carries. Mistral ships
        # PER-TENSOR fp8: `weight_block_size` is null in config.json, dense weights carry a rank-0
        # scalar `*_scale_inv`, and the stacked expert tensors carry `[128,1,1]` (one scale per
        # expert). The shared dequantizer (deepseek_v3/utils/hf_model_utils.py:208) asserts
        # `tensor.ndim == inv_scale.ndim` and `len(block_shape) == tensor.ndim`, so it RAISES on both
        # of Mistral's shapes rather than silently mis-scaling. Passing the honest value through
        # keeps that a loud failure until a Mistral-specific dequant path exists.
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "static",
            "weight_block_size": None,
        },
    )
