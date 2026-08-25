# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi K3 Model Configuration (text tower only).

Single source of truth for model dimension constants.
Values from HuggingFace config.json for Kimi-K3 (``text_config``), whose ``model_type`` is
``kimi_linear``. The top-level checkpoint config is a multimodal wrapper
(``KimiK3ForConditionalGeneration``) holding ``text_config`` + ``vision_config``; every LM field
the TT stack reads lives under ``text_config``.

K3 is a **hybrid**: of its 93 layers only 24 are full-attention (MLA) layers, the rest are KDA
linear-attention layers. Only the MLA side is modelled here.

MLA deltas vs Kimi-K2.6:
  * 96 attention heads (K2.6: 64)
  * **NoPE**: ``mla_use_nope`` -> no rotary embedding at all, so no ``rope_theta`` and no
    ``rope_scaling``/YaRN. The 64 ``qk_rope_head_dim`` columns still exist and are still cached
    (per-token latent stays 576 wide); they are simply never rotated.
  * softmax scale is plain ``qk_head_dim**-0.5`` -- K2.6 multiplies by ``mscale**2`` (~2.0), so
    reusing K2.6's scale here is a silent 2x error.
  * **output gate**: ``mla_use_output_gate`` -> a full-rank ``g_proj`` (hidden -> num_heads *
    v_head_dim) whose sigmoid multiplies the attention output before ``o_proj``.

Note on quantization: the checkpoint is MXFP4 (``compressed-tensors``, 4-bit, group_size 32), but
``quantization_config.ignore`` includes ``re:.*self_attn.*`` -- every MLA weight, ``g_proj``
included, is plain bf16. Only the MoE routed experts are quantized.
"""

import types


class KimiK3Config:
    """Kimi K3 model dimensions."""

    # Core dimensions
    EMB_SIZE = 7168  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 3072  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 33792  # Dense FFN hidden dimension

    # MoE configuration (HF names: num_experts / num_experts_per_token / num_shared_experts)
    NUM_ROUTED_EXPERTS = 896
    NUM_EXPERTS_PER_TOKEN = 16
    NUM_SHARED_EXPERTS = 2
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    ROUTE_SCALE = 1.0  # routed_scaling_factor
    ROUTED_EXPERT_HIDDEN_SIZE = 3584  # LatentMoE: routed experts run at a reduced hidden dim

    # Above this, moe_grouped_topk's circular buffers (sized from NUM_ROUTED_EXPERTS/32) no longer fit
    # L1 alongside the height-sharded gate input, and the program fails to validate. Enforced by
    # TtMoEGateConfig as both the default per-chip depth and a ceiling on any explicit sp_dim.
    MAX_GATE_SEQ_LEN_PER_CHIP = 3200

    # Gate-test device-mode scores bar, relaxing the shared 0.93; see #52569. 896 experts under sigmoid
    # near-tie the 16th and 17th scores often enough that device precision swaps a pick, and the
    # spread across Blackhole Galaxies (0.886 - 0.952) straddles the shared bar.
    GATE_SCORES_PCC_DEVICE = 0.87
    # Upstream KimiSparseMoeBlock builds ONE KimiMLP for the shared expert, not num_shared_experts of
    # them: shared_experts.gate_proj.weight is [6144, 7168].
    SHARED_EXPERT_INTERMEDIATE_SIZE = MOE_INTERMEDIATE_SIZE * NUM_SHARED_EXPERTS  # 6144

    # Model architecture
    NUM_LAYERS = 93
    NUM_DENSE_LAYERS = 1  # first_k_dense_replace
    VOCAB_SIZE = 163840

    # MLA dimensions
    NUM_ATTENTION_HEADS = 96
    NUM_KEY_VALUE_HEADS = 96
    Q_LORA_RANK = 1536
    KV_LORA_RANK = 512
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    V_HEAD_DIM = 128

    # MLA behaviour flags (K2.6 has neither)
    USE_NOPE = True  # mla_use_nope: rotary_emb is None; the 64 rope dims pass through unrotated
    USE_OUTPUT_GATE = True  # mla_use_output_gate: sigmoid(g_proj(x)) gates attn_out before o_proj

    # Norm / context
    RMS_NORM_EPS = 1e-5
    MAX_POSITION_EMBEDDINGS = 1048576
    # Deliberately NO ROPE_THETA / ROPE_SCALING_*: K3 has neither. Inventing them is exactly how the
    # softmax scale silently picks up a 2x mscale factor (see the module docstring).

    # 1-indexed, as in the HF config. 92 and 93 are adjacent, breaking the 3 KDA : 1 MLA pattern.
    # fmt: off
    FULL_ATTN_LAYERS_1BASED = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
                               52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93]
    # fmt: on

    # KDA (linear attention) sizing, recorded for completeness; no TT implementation exists yet.
    KDA_NUM_HEADS = 96
    KDA_HEAD_DIM = 128
    KDA_SHORT_CONV_KERNEL_SIZE = 4
    KDA_GATE_LOWER_BOUND = -5.0

    # AttnRes (attention-side, out of scope here; recorded so the delta is not lost)
    ATTN_RES_BLOCK_SIZE = 12

    LATENT_MOE_USE_NORM = True
    # The routed experts run the checkpoint's SiTU-GLU on device (#51351). Spelled as a string
    # because this config is torch-only; ROUTED_EXPERT_ACTIVATION_BY_NAME maps it onto the kernel
    # enum. Routed only: the shared expert and the dense FFN have no SiTU kernel and stay on SiLU.
    ROUTED_EXPERT_ACTIVATION = "situ"
    # Must match SituGluConfigKimi, which the fused routed-expert kernel bakes in.
    ACTIVATION_SITU_BETA = 4.0
    ACTIVATION_SITU_LINEAR_BETA = 25.0

    @classmethod
    def mla_layer_ids(cls) -> list[int]:
        """0-indexed model-layer indices of the full-attention (MLA) layers.

        ``FULL_ATTN_LAYERS_1BASED`` is 1-indexed as it appears in the HF config, so this is the
        form every 0-indexed consumer (layer loops, kv-slot maps, ``layer_split_boundaries``)
        wants: ``[3, 7, 11, ..., 87, 91, 92]``.
        """
        return sorted(layer - 1 for layer in cls.FULL_ATTN_LAYERS_1BASED)

    @classmethod
    def mla_kv_slot(cls, layer_idx: int) -> int:
        """KV-cache slot for a 0-indexed *model* layer index.

        The KVPE cache holds one slot per MLA layer (24), not per model layer (93), so a model
        layer index cannot be used as a cache index directly. Raises for a non-MLA layer rather
        than returning a plausible-but-wrong slot.
        """
        ids = cls.mla_layer_ids()
        if layer_idx not in ids:
            raise ValueError(f"model layer {layer_idx} is a KDA layer, not an MLA layer; MLA layers are {ids}")
        return ids.index(layer_idx)


def kimi_k3_hf_config(max_seq: int = 8192):
    """HF-attribute-style config for the Kimi-K3 MLA and MoE paths.

    Hand-built rather than loaded via ``AutoConfig`` because upstream ``modeling_kimi_linear.py``
    raises ``ImportError`` at module import without ``fla-core``, which is not installed here.

    ``rope_scaling=None`` is the real K3 value and exercises ``ttMLA``'s guard; do not substitute
    ``{"factor": 1.0, ...}`` as ``glm_5_2_hf_config`` does, since K3 has no rotary embedding at all.
    ``max_position_embeddings`` is capped at ``max_seq`` because the CPU reference eagerly builds
    ``[max_position_embeddings, qk_rope_head_dim]`` cos/sin buffers -- 512 MB at K3's true 1M.
    """
    return types.SimpleNamespace(
        vocab_size=KimiK3Config.VOCAB_SIZE,
        hidden_size=KimiK3Config.EMB_SIZE,
        intermediate_size=KimiK3Config.INTERMEDIATE_SIZE,
        moe_intermediate_size=KimiK3Config.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=KimiK3Config.NUM_LAYERS,
        num_attention_heads=KimiK3Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=KimiK3Config.NUM_KEY_VALUE_HEADS,
        kv_lora_rank=KimiK3Config.KV_LORA_RANK,
        q_lora_rank=KimiK3Config.Q_LORA_RANK,
        qk_nope_head_dim=KimiK3Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=KimiK3Config.QK_ROPE_HEAD_DIM,
        v_head_dim=KimiK3Config.V_HEAD_DIM,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
        max_seq_len=max_seq,
        initializer_range=0.02,
        # --- the two K3 MLA flags ttMLA and MLAReference branch on ---
        mla_use_nope=KimiK3Config.USE_NOPE,
        mla_use_output_gate=KimiK3Config.USE_OUTPUT_GATE,
        # K3 has no rotary embedding: scale is plain qk_head_dim**-0.5, no mscale.
        rope_scaling=None,
        # Inert under NoPE; present only so the CPU reference can construct (see docstring).
        rope_theta=10000.0,
        max_position_embeddings=max_seq,
        attention_bias=False,
        attention_dropout=0.0,
        # MoE name bridge: the TT stack reads the DeepSeek spellings, the vendored
        # KimiSparseMoeBlock / KimiMoEGate read K3's own, so both are supplied where they differ.
        first_k_dense_replace=KimiK3Config.NUM_DENSE_LAYERS,
        n_routed_experts=KimiK3Config.NUM_ROUTED_EXPERTS,
        num_experts_per_tok=KimiK3Config.NUM_EXPERTS_PER_TOKEN,
        n_shared_experts=KimiK3Config.NUM_SHARED_EXPERTS,
        num_experts=KimiK3Config.NUM_ROUTED_EXPERTS,
        num_experts_per_token=KimiK3Config.NUM_EXPERTS_PER_TOKEN,
        num_shared_experts=KimiK3Config.NUM_SHARED_EXPERTS,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_expert_group=KimiK3Config.NUM_EXPERT_GROUPS,
        # Grouped routing is a no-op at 1/1, but ``run_model`` reads these unconditionally.
        n_group=KimiK3Config.NUM_EXPERT_GROUPS,
        topk_group=KimiK3Config.NUM_LIMITED_GROUPS,
        routed_scaling_factor=KimiK3Config.ROUTE_SCALE,
        norm_topk_prob=True,  # moe_renormalize
        scoring_func="sigmoid",  # moe_router_activation_func
        topk_method="noaux_tc",
        # LatentMoE: the routed experts' reduced hidden dim, and the latent RMSNorm flag.
        routed_expert_hidden_size=KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        latent_moe_use_norm=KimiK3Config.LATENT_MOE_USE_NORM,
        # The checkpoint says "situ", but consumers of this field index ACT2FN with it, which has
        # no "situ" entry. The routed expert selects SiTU through RoutedExpertActivation instead.
        hidden_act="silu",
        activation_situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
        activation_situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
    )
