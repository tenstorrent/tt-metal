"""Model constants for Fish Audio S2 Pro, read from the HF snapshot's config.json when available.

All numbers below are verified against fishaudio/s2-pro @ 1de9996b (see doc/ and tests/test_weights.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

HF_REPO_ID = "fishaudio/s2-pro"
HF_REVISION = "1de9996b6be38b745688de084d87a5633f714e4e"

SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = 2048  # codec hop 512 x quantizer downsample 4  -> 21.53 frames/s
FRAMES_PER_SECOND = SAMPLE_RATE / SAMPLES_PER_FRAME

# Special token ids (tokenizer.json of the pinned revision)
IM_START_ID = 151644
IM_END_ID = 151645
SEMANTIC_BEGIN_ID = 151678
SEMANTIC_END_ID = 155773
VOICE_MODALITY_ID = 151673
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
VOICE_TOKEN = "<|voice|>"

# Decoding defaults (fish-speech inference.py)
DEFAULT_TOP_K = 30
RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9
CODEBOOK_SCALE = 1.0 / (11**0.5)  # 1/sqrt(num_codebooks + 1), scale_codebook_embeddings=True


@dataclass(frozen=True)
class TowerConfig:
    n_layer: int
    dim: int
    n_head: int
    n_local_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rope_base: float
    norm_eps: float
    attention_qk_norm: bool
    tie_word_embeddings: bool
    max_seq_len: int

    @property
    def q_dim(self):
        return self.n_head * self.head_dim

    @property
    def kv_dim(self):
        return self.n_local_heads * self.head_dim


@dataclass(frozen=True)
class S2Config:
    slow: TowerConfig
    fast: TowerConfig
    num_codebooks: int = 10
    codebook_size: int = 4096  # fast head width and codebook-0 cardinality
    residual_codebook_size: int = 1024  # real cardinality of codebooks 1..9 (codec clamps)
    semantic_begin_id: int = SEMANTIC_BEGIN_ID
    semantic_end_id: int = SEMANTIC_END_ID
    im_end_id: int = IM_END_ID

    @staticmethod
    def default() -> "S2Config":
        slow = TowerConfig(36, 2560, 32, 8, 128, 9728, 155776, 1e6, 1e-6, True, True, 32768)
        fast = TowerConfig(4, 2560, 32, 8, 128, 9728, 4096, 1e6, 1e-6, False, False, 11)
        return S2Config(slow=slow, fast=fast)

    @staticmethod
    def from_snapshot(snapshot: str | os.PathLike) -> "S2Config":
        cfg = json.load(open(Path(snapshot) / "config.json"))
        assert cfg.get("model_type") == "fish_qwen3_omni", cfg.get("model_type")
        tc, ac = cfg["text_config"], cfg["audio_decoder_config"]

        def tower(c, tie, max_seq_len):
            return TowerConfig(
                c["n_layer"],
                c["dim"],
                c["n_head"],
                c["n_local_heads"],
                c["head_dim"],
                c["intermediate_size"],
                c["vocab_size"],
                float(c["rope_base"]),
                float(c["norm_eps"]),
                bool(c.get("attention_qk_norm", False)),
                tie,
                max_seq_len,
            )

        s2 = S2Config(
            slow=tower(tc, bool(tc.get("tie_word_embeddings", True)), int(tc["max_seq_len"])),
            fast=tower(ac, False, int(ac.get("max_seq_len", 11))),
            num_codebooks=int(ac["num_codebooks"]),
            codebook_size=int(ac["vocab_size"]),
            semantic_begin_id=int(cfg["semantic_start_token_id"]),
            semantic_end_id=int(cfg["semantic_end_token_id"]),
            im_end_id=int(cfg.get("eos_token_id", IM_END_ID)),
        )
        assert s2.fast.dim == s2.slow.dim, "fast_project_in must be identity (fast dim == slow dim)"
        return s2

    def as_dict(self):
        return asdict(self)
