# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Layer schedule of DeepSeekV41FlashConfig against the hand-written V4.1 table (no device)."""

import json
from pathlib import Path

from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import V41BlockType as T

VENDORED_CONFIG = Path(__file__).parents[2] / "reference" / "deepseek_v41" / "config.json"

# discovery.md §2, written out by hand from the six block types.
EXPECTED = {
    0: (T.SWA_ONLY, None, None),
    1: (T.SWA_ONLY, None, None),
    2: (T.KV_INDEX_SOURCE, 2, 2),
    3: (T.CONSUMER_RATIO2, 2, 2),
    7: (T.CONSUMER_RATIO2, 2, 2),
    8: (T.KV_INDEX_SOURCE, 8, 8),
    13: (T.CONSUMER_RATIO2, 8, 8),
    14: (T.KV_INDEX_SOURCE, 14, 14),
    19: (T.CONSUMER_RATIO2, 14, 14),
    20: (T.CANDIDATE_SOURCE, 20, 20),
    21: (T.CONSUMER_RATIO1, 20, 20),
    23: (T.CONSUMER_RATIO1, 20, 20),
    24: (T.CANDIDATE_INDEX_SOURCE, 20, 24),
    27: (T.CONSUMER_RATIO1, 20, 24),
    36: (T.CANDIDATE_INDEX_SOURCE, 20, 36),
    39: (T.CONSUMER_RATIO1, 20, 36),
}


def test_block_type_and_sources(expect_error):
    for layer, (block_type, kv_source, index_source) in EXPECTED.items():
        assert C.block_type(layer) == block_type, layer
        if kv_source is None:
            with expect_error(ValueError, "does not attend over compressed KV"):
                C.kv_source(layer)
        else:
            assert C.kv_source(layer) == kv_source, layer
            assert C.index_source(layer) == index_source, layer


def test_block_type_counts():
    counts = {}
    for layer in range(C.NUM_LAYERS):
        counts[C.block_type(layer)] = counts.get(C.block_type(layer), 0) + 1
    assert counts == {
        T.SWA_ONLY: 2,
        T.KV_INDEX_SOURCE: 3,
        T.CONSUMER_RATIO2: 15,
        T.CANDIDATE_SOURCE: 1,
        T.CANDIDATE_INDEX_SOURCE: 4,
        T.CONSUMER_RATIO1: 15,
    }


def test_matches_pinned_reference_config():
    """Constants equal the vendored upstream inference/config.json."""
    cfg = json.loads(VENDORED_CONFIG.read_text())
    pairs = {
        "dim": C.EMB_SIZE,
        "moe_inter_dim": C.MOE_INTERMEDIATE_SIZE,
        "n_layers": C.NUM_LAYERS,
        "n_mtp_layers": C.NUM_DSPARK_LAYERS,
        "n_heads": C.NUM_ATTENTION_HEADS,
        "n_routed_experts": C.NUM_ROUTED_EXPERTS,
        "n_activated_experts": C.NUM_EXPERTS_PER_TOKEN,
        "q_lora_rank": C.Q_LORA_RANK,
        "o_lora_rank": C.O_LORA_RANK,
        "o_groups": C.O_GROUPS,
        "head_dim": C.HEAD_DIM,
        "rope_head_dim": C.QK_ROPE_HEAD_DIM,
        "norm_eps": C.RMS_NORM_EPS,
        "window_size": C.SLIDING_WINDOW,
        "index_n_heads": C.INDEX_N_HEADS,
        "index_head_dim": C.INDEX_HEAD_DIM,
        "index_topk": C.INDEX_TOPK,
        "candidate_source_layer": C.CANDIDATE_SOURCE_LAYER,
        "candidate_topk_blocks": C.CANDIDATE_TOPK_BLOCKS,
        "candidate_block_size": C.CANDIDATE_BLOCK_SIZE,
        "route_scale": C.ROUTE_SCALE,
        "swiglu_limit": C.SWIGLU_LIMIT,
        "compress_rope_theta": C.COMPRESS_ROPE_THETA,
        "rope_factor": C.ROPE_SCALING_FACTOR,
        "original_seq_len": C.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
        "hc_mult": C.HC_MULT,
        "hc_sinkhorn_iters": C.HC_SINKHORN_ITERS,
        "hc_eps": C.HC_EPS,
        "vocab_size": C.VOCAB_SIZE,
    }
    for key, value in pairs.items():
        assert cfg[key] == value, key
    assert tuple(cfg["compress_ratios"]) == C.COMPRESS_RATIOS
    assert tuple(cfg["kv_source_layers"]) == C.KV_SOURCE_LAYERS
    assert tuple(cfg["index_source_layers"]) == C.INDEX_SOURCE_LAYERS
    assert tuple(cfg["engram_layer_ids"]) == C.ENGRAM_LAYER_IDS
    assert tuple(cfg["engram_num_embeddings"]) == C.ENGRAM_NUM_EMBEDDINGS
    assert tuple(cfg["dspark_target_layer_ids"]) == C.DSPARK_TARGET_LAYER_IDS
