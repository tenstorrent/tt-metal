# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 (``moonshotai/Kimi-K3.1-Pro``): K3 with its 24 full-attention MLA layers turned into DSA layers.

``text_config.dsa_config``: top-2048, 32 index heads x 128, ``index_rope_dim`` 0 (NoPE indexer), LayerNorm
k_norm with bias. The indexer weights match DeepSeek-V3.2 / GLM's, so the indexer is DS-V3.2's without RoPE and
without its q/k Hadamard rotation.
No ``indexer_types``: every DSA layer owns its index slot, in the same 24-slot space as the KVPE cache.
"""

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config


class KimiK31Config(KimiK3Config):
    """Kimi K3.1 dimensions: K3 plus the DSA indexer."""

    # Indexer / sparse attention (text_config.dsa_config)
    INDEX_TOPK = 2048
    INDEX_HEAD_DIM = 128
    INDEX_N_HEADS = 32
    INDEX_ROPE_DIM = 0  # NoPE indexer
    # Documentation only (not emitted): freq 1, so every DSA layer runs its own indexer.
    INDEX_TOPK_FREQ = 1
    INDEX_SKIP_TOPK_OFFSET = 1

    # K3's 1536 B/bank is spent by KDA's long-lived semaphores, MoE routing and dense MLA; a DSA layer's
    # early-Q all-to-all then needs one more 16 B slot per bank (OOM at 1536 with KDA layers 1-2 and DSA
    # layer 3 in one forward, test_kimi_k3_1_dsa_trace.py hybrid_L0_3 window).
    L1_SMALL_SIZE = 2048

    # 1-indexed as in the HF config; identical to the full-attention layers.
    DSA_LAYERS_1BASED = list(KimiK3Config.FULL_ATTN_LAYERS_1BASED)


assert KimiK31Config.DSA_LAYERS_1BASED == KimiK31Config.FULL_ATTN_LAYERS_1BASED


def kimi_k3_1_hf_config(max_seq: int = 8192):
    """``kimi_k3_hf_config`` plus the indexer fields (``index_rope_interleave`` is inert but required)."""
    cfg = kimi_k3_hf_config(max_seq)
    cfg.index_n_heads = KimiK31Config.INDEX_N_HEADS
    cfg.index_head_dim = KimiK31Config.INDEX_HEAD_DIM
    cfg.index_topk = KimiK31Config.INDEX_TOPK
    cfg.index_rope_dim = KimiK31Config.INDEX_ROPE_DIM
    cfg.index_rope_interleave = True
    # vLLM's KimiK3DSAIndexerKNorm (LayerNorm) takes config.rms_norm_eps, not DS-V3.2's 1e-6.
    cfg.index_k_norm_eps = cfg.rms_norm_eps
    # No Hadamard on the indexer q/k (vLLM PR 65685 and K3.1 decode): the index cache holds natural-basis keys
    # (DS-V3.2 / GLM keep the default, rotated).
    cfg.index_hadamard = False
    # Overlap the indexer TopK with the sparse-KV gather, as GLM-5.3 (serial on SP=1).
    cfg.sparse_mla_overlap_profile = "auto"
    # Indexer score matmul at HiFi2, not the op's LoFi default for bf8 x bf8: LoFi was the
    # largest device score error at DSA layers 3/7. Top-k recall vs the trace L3/L7 +0.015/+0.016 at 10K,
    # +0.033/+0.038 at 128K, for +0.2% / +0.9% scenario TTFT (stage 2, indicative 75 W).
    cfg.index_score_math_fidelity = "HiFi2"
    return cfg
