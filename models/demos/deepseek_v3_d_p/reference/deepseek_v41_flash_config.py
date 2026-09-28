# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V4.1 Flash Model Configuration.

Single source of truth for model dimension constants and the per-layer attention schedule.
Values from ``inference/config.json`` of HuggingFace ``deepseek-ai/DeepSeek-V4.1-Flash`` (revision
dba1be0a40aa). Layer roles are derived from the reference's source lists exactly as
``inference/model.py`` wires them: a consumer reads the compressed KV / index keys / top-k of the most
recent source executed before it (``SharedAttentionRuntime``).
"""

from enum import Enum


class V41BlockType(Enum):
    """The six V4.1 decoder block types, by attention role."""

    SWA_ONLY = "swa_only"  # sliding window only (layers 0, 1)
    KV_INDEX_SOURCE = "kv_index_source"  # compresses its KV, owns index keys, runs its indexer
    CONSUMER_RATIO2 = "consumer_ratio2"  # reads the latest ratio-2 source's compressed KV and top-k
    CANDIDATE_SOURCE = "candidate_source"  # KV + index source that also publishes candidate blocks
    CANDIDATE_INDEX_SOURCE = "candidate_index_source"  # own indexer, masked to the published candidates
    CONSUMER_RATIO1 = "consumer_ratio1"  # reads layer 20's ratio-1 KV and the latest index source's top-k


class DeepSeekV41FlashConfig:
    """DeepSeek V4.1 Flash model dimensions and layer schedule."""

    # Core dimensions
    EMB_SIZE = 5120
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2304
    HEAD_DIM = 512

    # MoE configuration
    NUM_ROUTED_EXPERTS = 384
    NUM_EXPERTS_PER_TOKEN = 6
    NUM_SHARED_EXPERTS = 1
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    SCORE_FUNC = "sqrtsoftplus"
    ROUTE_SCALE = 1.5
    ROUTED_EXPERT_ACTIVATION = "clamped_silu_glu"
    SHARED_EXPERT_ACTIVATION = "clamped_silu_glu"
    SWIGLU_LIMIT = 10.0

    # Model architecture
    NUM_LAYERS = 40  # backbone; DSpark layers follow at indices 40..42
    NUM_DSPARK_LAYERS = 3
    NUM_DENSE_LAYERS = 0
    VOCAB_SIZE = 129280
    SLIDING_WINDOW = 128

    # MLA dimensions
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 1
    Q_LORA_RANK = 1280
    O_LORA_RANK = 1024
    O_GROUPS = 8
    QK_ROPE_HEAD_DIM = 64

    # Indexer / sparse attention
    INDEX_N_HEADS = 32
    INDEX_HEAD_DIM = 128
    INDEX_TOPK = 512
    CANDIDATE_TOPK_BLOCKS = 2048
    CANDIDATE_BLOCK_SIZE = 8

    # Per-layer attention schedule, DSpark layers included (config.json compress_ratios).
    COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20 + (0,) * 3
    KV_SOURCE_LAYERS = (2, 8, 14, 20)
    INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
    CANDIDATE_SOURCE_LAYER = 20

    # RoPE: ratio-0 layers use ROPE_THETA without YaRN; compressed layers use COMPRESS_ROPE_THETA + YaRN.
    ROPE_THETA = 10000
    COMPRESS_ROPE_THETA = 160000.0
    ROPE_SCALING_FACTOR = 16
    ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS = 65536
    ROPE_SCALING_BETA_FAST = 32
    ROPE_SCALING_BETA_SLOW = 1

    # Hyper-connections
    HC_MULT = 4
    HC_SINKHORN_ITERS = 20
    HC_EPS = 1.0e-6

    # Every RMSNorm, including the hc_mixes normalization (the vision tower keeps its own 1e-6).
    RMS_NORM_EPS = 1e-20

    # Engram
    ENGRAM_LAYER_IDS = (1, 14)
    ENGRAM_NUM_EMBEDDINGS = (384006168, 384016682)
    ENGRAM_VOCAB_SIZE = 16000000
    ENGRAM_MAX_NGRAM_SIZE = 4
    ENGRAM_N_HEADS = 8
    ENGRAM_HEAD_DIM = 256
    ENGRAM_PAD_ID = 2
    ENGRAM_COMPRESSED_VOCAB_SIZE = 99092

    # DSpark (prefill only seeds the DSpark window caches from these taps)
    DSPARK_TARGET_LAYER_IDS = (37, 38, 39)

    # Vision
    VISION_N_LAYERS = 32
    VISION_DIM = 1024
    VISION_N_HEADS = 16
    VISION_INTER_DIM = 2816
    VISION_PATCH_SIZE = 14
    VISION_DOWNSAMPLE_RATIO = 3
    VISION_MAX_N_TOKEN = 1024
    VISION_ROPE_THETA = 10000
    VISION_NORM_EPS = 1e-6
    IMAGE_TOKEN_ID = 129264

    MAX_POSITION_EMBEDDINGS = 1048576

    @classmethod
    def compress_ratio(cls, layer: int) -> int:
        return cls.COMPRESS_RATIOS[layer]

    @classmethod
    def _latest_source(cls, layer: int, sources: tuple[int, ...]) -> int:
        """The most recent source executed at or before ``layer``: the one whose shared state it reads."""
        candidates = [s for s in sources if s <= layer]
        if not candidates:
            raise ValueError(f"layer {layer} has no preceding source in {sources}")
        return candidates[-1]

    @classmethod
    def kv_source(cls, layer: int) -> int:
        """The layer that produced the compressed KV and index keys ``layer`` attends over."""
        if layer >= cls.NUM_LAYERS or cls.compress_ratio(layer) == 0:
            raise ValueError(f"layer {layer} does not attend over compressed KV")
        source = cls._latest_source(layer, cls.KV_SOURCE_LAYERS)
        if cls.compress_ratio(source) != cls.compress_ratio(layer):
            raise ValueError(f"layer {layer} and its KV source {source} have different compress ratios")
        return source

    @classmethod
    def index_source(cls, layer: int) -> int:
        """The layer whose top-k selection ``layer`` uses."""
        if layer >= cls.NUM_LAYERS or cls.compress_ratio(layer) == 0:
            raise ValueError(f"layer {layer} does not use a top-k selection")
        return cls._latest_source(layer, cls.INDEX_SOURCE_LAYERS)

    @classmethod
    def block_type(cls, layer: int) -> V41BlockType:
        if not 0 <= layer < cls.NUM_LAYERS:
            raise ValueError(f"layer {layer} is not a backbone layer")
        ratio = cls.compress_ratio(layer)
        if ratio == 0:
            return V41BlockType.SWA_ONLY
        if layer == cls.CANDIDATE_SOURCE_LAYER:
            return V41BlockType.CANDIDATE_SOURCE
        if layer in cls.KV_SOURCE_LAYERS:
            return V41BlockType.KV_INDEX_SOURCE
        if layer in cls.INDEX_SOURCE_LAYERS:
            if not 0 <= cls.CANDIDATE_SOURCE_LAYER < layer:
                raise ValueError(f"index source {layer} owns no keys and has no candidate source before it")
            return V41BlockType.CANDIDATE_INDEX_SOURCE
        return V41BlockType.CONSUMER_RATIO2 if ratio == 2 else V41BlockType.CONSUMER_RATIO1
