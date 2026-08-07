# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GLM 5.2 Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for GLM-5.2 (model_type ``glm_moe_dsa``).

Geometry is identical to GLM-5.1 (attention, MoE, indexer sizing, layer count). The 5.2
deltas: longer context (rope_theta 8e6, 1M positions) and cross-layer DSA indexer reuse,
where only ``full`` layers run the lightning indexer and ``shared`` layers reuse the most
recent full layer's top-k selection. The full/shared map is ``indexer_types``.
"""

import types


class GLM52Config:
    """GLM 5.2 model dimensions."""

    # Core dimensions
    EMB_SIZE = 6144  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 12288  # Dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
    NUM_SHARED_EXPERTS = 1
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1

    # Model architecture
    NUM_LAYERS = 78
    NUM_DENSE_LAYERS = 3  # first_k_dense_replace
    VOCAB_SIZE = 154880
    MAX_POSITION_EMBEDDINGS = 1048576  # 5.2: 1M context (5.1 was 202752)

    # MLA dimensions
    NUM_ATTENTION_HEADS = 64
    Q_LORA_RANK = 2048
    KV_LORA_RANK = 512
    QK_NOPE_HEAD_DIM = 192
    QK_ROPE_HEAD_DIM = 64
    V_HEAD_DIM = 256

    # Indexer / sparse attention
    INDEX_TOPK = 2048
    INDEX_HEAD_DIM = 128
    INDEX_N_HEADS = 32

    # Indexer reuse (5.2): a ``full`` layer runs the indexer; the following ``shared`` layers reuse
    # its top-k indices. The per-layer map is derived from freq/offset (HF __post_init__ formula):
    # full if (max(i - offset + 1, 0) % freq) == 0 else shared -> full at {0,1,2,6,10,...,74}.
    # A checkpoint only carries indexer weights for ``full`` layers, so reuse is required, not optional.
    INDEX_TOPK_FREQ = 4
    INDEX_SKIP_TOPK_OFFSET = 3
    INDEX_TOPK_PATTERN = None  # optional explicit "F"/"S" string override; None -> derive from freq/offset

    # Other
    RMS_NORM_EPS = 1e-5
    ROUTE_SCALE = 2.5
    ROPE_THETA = 8000000  # 5.2: raised from 1e6 to support 1M context

    @classmethod
    def indexer_types(cls, num_layers: int | None = None):
        """Per-layer indexer mode list (``"full"`` / ``"shared"``), length ``num_layers``."""
        n = cls.NUM_LAYERS if num_layers is None else num_layers
        if cls.INDEX_TOPK_PATTERN is not None:
            m = {"F": "full", "S": "shared"}
            return [m[c] if isinstance(c, str) else c for c in cls.INDEX_TOPK_PATTERN][:n]
        freq = max(cls.INDEX_TOPK_FREQ, 1)
        off = cls.INDEX_SKIP_TOPK_OFFSET
        return ["full" if (max(i - off + 1, 0) % freq) == 0 else "shared" for i in range(n)]


def glm_5_2_hf_config(max_seq: int = 8192):
    """HF-attribute-style config the unified ttMLA reads (GLM-5.2 dims, no YaRN).

    Mirrors ``glm_5_1_config.glm_hf_config`` (same curated field set device + CPU-reference read),
    with the 5.2 deltas: ``rope_theta=8e6`` and the indexer-reuse fields (``indexer_types`` map plus
    the freq/offset it derives from). ``rope_scaling.factor=1.0`` disables YaRN -> plain RoPE at θ=8e6.
    The four ``index_*`` attrs size the DSA indexer (GLM's indexer RoPE is interleaved).
    """
    return types.SimpleNamespace(
        vocab_size=GLM52Config.VOCAB_SIZE,
        hidden_size=GLM52Config.EMB_SIZE,
        intermediate_size=GLM52Config.INTERMEDIATE_SIZE,  # dense-FFN (layers 0-2) hidden dim = 12288
        num_attention_heads=GLM52Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=GLM52Config.NUM_ATTENTION_HEADS,
        kv_lora_rank=GLM52Config.KV_LORA_RANK,
        q_lora_rank=GLM52Config.Q_LORA_RANK,
        qk_nope_head_dim=GLM52Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=GLM52Config.QK_ROPE_HEAD_DIM,
        v_head_dim=GLM52Config.V_HEAD_DIM,
        rms_norm_eps=GLM52Config.RMS_NORM_EPS,
        max_seq_len=max_seq,
        rope_theta=float(GLM52Config.ROPE_THETA),
        attention_bias=False,
        rope_scaling={
            "factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": max_seq,
        },
        index_n_heads=GLM52Config.INDEX_N_HEADS,
        index_head_dim=GLM52Config.INDEX_HEAD_DIM,
        index_topk=GLM52Config.INDEX_TOPK,
        index_rope_interleave=True,
        # Indexer reuse: the per-layer full/shared map (length NUM_LAYERS) plus the params it derives
        # from. Consumers read `indexer_types` by layer index; absent on GLM-5.1 -> all layers full.
        indexer_types=GLM52Config.indexer_types(),
        index_topk_freq=GLM52Config.INDEX_TOPK_FREQ,
        index_skip_topk_offset=GLM52Config.INDEX_SKIP_TOPK_OFFSET,
        first_k_dense_replace=GLM52Config.NUM_DENSE_LAYERS,
        n_routed_experts=GLM52Config.NUM_ROUTED_EXPERTS,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )
