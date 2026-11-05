# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi K3.1 Model Configuration (text tower, sparse-MLA / DSA view).

K3.1 (``moonshotai/Kimi-K3.1-Pro``, top-level ``model_type: kimi_k31``, ``text_config.model_type: kimi_linear``)
is a post-trained K3: every MLA / KDA / MoE dimension is unchanged (see ``kimi_k3_config``). The one
architectural delta is that its 24 full-attention layers are now **DSA** -- MLA plus a lightning indexer
that picks the top-``index_topk`` keys per query. ``text_config.dsa_config``:

    dsa_layers          == linear_attn_config.full_attn_layers (1-based, 24 layers)
    index_topk          2048
    index_head_dim      128
    index_num_heads     32
    index_rope_dim      0          -> the indexer is NoPE too (no rotation of q/k index heads)
    index_k_norm_type   layernorm  -> LayerNorm with bias, as DeepSeek-V3.2 / GLM
    index_topk_freq     1
    index_skip_topk_offset 1

Facts confirmed from the released checkpoint (no modeling_*.py is shipped, so these are read from the
safetensors headers), for every DSA layer, under ``language_model.model.layers.{i}.self_attn.indexer.``:
``wq_b`` [32*128, 1536], ``wk`` [128, 7168], ``weights_proj`` [32, 7168], ``k_norm.weight`` / ``k_norm.bias``
[128] -- all bf16 (``quantization_config.ignore`` covers ``self_attn``). That is exactly the DeepSeek-V3.2 /
GLM indexer parameter set, so the indexer math is taken to be the DS-V3.2 lightning indexer (ReLU score
weighted by ``weights_proj``) with the rope step removed.

``index_topk_freq`` / ``index_skip_topk_offset`` are the same field names GLM-5.3 uses for cross-layer
indexer reuse (``full`` iff ``max(i - offset + 1, 0) % freq == 0``). With freq 1 every DSA layer is ``full``,
so they are recorded here as documentation only. Deliberately NO ``indexer_types`` is emitted: absent, the
indexer uses its non-compact slot path (slot = ``cache_layer_idx``, stride = ``layer_num``), which is the
same 24-slot MLA-ordinal space the KVPE cache uses (``KimiK3Config.mla_kv_slot``). ``["full"] * 93``
would size 93 slots; ``["full"] * 24`` would overflow at model layer >= 24.
"""

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config


class KimiK31Config(KimiK3Config):
    """Kimi K3.1 dimensions: K3 plus the DSA indexer."""

    # Indexer / sparse attention (text_config.dsa_config)
    INDEX_TOPK = 2048
    INDEX_HEAD_DIM = 128
    INDEX_N_HEADS = 32
    INDEX_ROPE_DIM = 0  # NoPE indexer
    # Documentation only -- see the module docstring (freq 1 -> every DSA layer runs its own indexer).
    INDEX_TOPK_FREQ = 1
    INDEX_SKIP_TOPK_OFFSET = 1

    # 1-indexed as in the HF config; identical to the full-attention layers.
    DSA_LAYERS_1BASED = list(KimiK3Config.FULL_ATTN_LAYERS_1BASED)


assert KimiK31Config.DSA_LAYERS_1BASED == KimiK31Config.FULL_ATTN_LAYERS_1BASED


def kimi_k3_1_hf_config(max_seq: int = 8192):
    """HF-attribute-style config for the Kimi-K3.1 sparse-MLA path: ``kimi_k3_hf_config`` plus the indexer.

    Every MLA field (NoPE, output gate, ``rope_scaling=None``) is inherited from K3 unchanged. The indexer
    fields follow the DS-V3.2 / GLM-5.3 spellings that ``TtIndexer`` and the sparse CPU reference read.
    ``index_rope_interleave`` is inert under ``index_rope_dim=0`` but ``TtIndexer`` requires the key.
    """
    cfg = kimi_k3_hf_config(max_seq)
    cfg.index_n_heads = KimiK31Config.INDEX_N_HEADS
    cfg.index_head_dim = KimiK31Config.INDEX_HEAD_DIM
    cfg.index_topk = KimiK31Config.INDEX_TOPK
    cfg.index_rope_dim = KimiK31Config.INDEX_ROPE_DIM
    cfg.index_rope_interleave = True
    # Full indexer layers overlap their local TopK with the sparse-KV SP gather (as GLM-5.3). As a model
    # default, "auto" runs the serial schedule on SP=1 meshes (ttMLA), where no overlap is possible.
    cfg.sparse_mla_overlap_profile = "auto"
    return cfg
