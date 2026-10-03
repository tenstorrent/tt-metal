"""Model constants and sub-model configs for MiniMax Music 3 (MiniMaxAI/MiniMax-Music3 @ fbdf52fb).

Every number here is part of the checkpoint's inference contract (diffusers PR #14456, encoders.py / before_denoise.py /
denoise.py / decoders.py) or read from the sub-folder config.json files. Changing any of them changes the audio.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

HF_REPO_ID = "MiniMaxAI/MiniMax-Music3"
HF_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
# subfolders of the snapshot we use; everything else (originals, assets) is skipped by the weights pointer
HF_IGNORE_PATTERNS = ["flowmatching_vae.pth", "dav.pth", "qwen_7B/*", "assets/*", "figures/*", "scripts/*"]

# ---- prompt / token contract (encoders.py) --------------------------------------------------------------
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
CAPTION_START, CAPTION_END = "<|caption_start|>", "<|caption_end|>"
LYRICS_START, LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END_TOKEN_ID = 151670
AUDIO_CFG_TOKEN_ID = 151654
AUDIO_CODE_OFFSET = 151675
SEMANTIC_VOCAB_SIZE = 16384
MAX_PROMPT_TOKENS = 5_000
MAX_AUDIO_FRAMES = 9_000
# the c0 head only ever sees the 16 384 semantic rows + the end-of-audio row: [151675, 168059) and 151670
SLICED_VOCAB = SEMANTIC_VOCAB_SIZE + 1  # row SEMANTIC_VOCAB_SIZE (16384) of the sliced head == AUDIO_END_TOKEN_ID
SLICED_VOCAB_PADDED = 16416  # next multiple of 32

# ---- autoregressive sampling (fixed by the reference recipe) -------------------------------------------
AR_CFG_SCALE = 1.5
AR_CFG_TOP_K = 50
AR_SAMPLING_TOP_K = 50
NUM_CODEBOOKS = 8
AUDIO_VOCAB_SIZE = 1024
FRAME_RATE = 25.0  # frames per second of audio (= 24000 / 960)

# ---- flow matching (before_denoise.py / denoise.py / decoders.py) --------------------------------------
CHUNK_FRAMES = 200
CHUNK_HOP = 100
OVERLAP_LATENT_LENGTH = 172
CROP_LEFT_LATENT = 86
CROP_RIGHT_LATENT = 344 - 86
DIT_GUIDANCE_SCALE = 1.7
DIT_NUM_STEPS = 30
LATENT_CHANNELS = 128
LATENT_HOP = 512
VOCODER_SAMPLE_RATE = 44100
OUTPUT_SAMPLE_RATE = 32000  # the reference server resamples 44.1k -> 32k


def latent_length_for_frames(num_frames: int) -> int:
    """condition_encoder: latents per window = int(frames * 44100/24000 * 960/512) (3.4453125 per frame)."""
    return max(1, int(num_frames * 44100 / 24000 * 960 / 512))


def chunk_starts(num_frames: int):
    return [0] if num_frames <= CHUNK_FRAMES else list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))


@dataclass(frozen=True)
class LLMConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 200000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 10240


@dataclass(frozen=True)
class DepthConfig:
    hidden_size: int = 4096
    num_layers: int = 4
    num_attention_heads: int = 16
    intermediate_size: int = 6144
    audio_vocab_size: int = 1024
    num_codebooks: int = 8
    max_position_embeddings: int = 16

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True)
class DiTConfig:
    in_channels: int = 128
    condition_dim: int = 2048
    num_layers: int = 36
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    ff_inner_dim: int = 8192
    rotary_dim: int = 32
    fourier_embedding_dim: int = 256

    @property
    def inner_dim(self):
        return self.num_attention_heads * self.attention_head_dim

    @property
    def concat_channels(self):
        return 2 * self.in_channels + self.condition_dim


@dataclass(frozen=True)
class CondConfig:
    condition_hidden_dim: int = 4096
    num_condition_layers: int = 8
    out_dim: int = 2048
    input_sampling_rate: int = 24000
    input_hop_length: int = 960
    output_sampling_rate: int = 44100
    output_hop_length: int = 512


@dataclass(frozen=True)
class VocoderConfig:
    latent_channels: int = 128
    decoder_input_dim: int = 1024
    decoder_hidden_dim: int = 1536
    upsampling_ratios: tuple = (8, 8, 4, 2)
    sampling_rate: int = 44100


@dataclass(frozen=True)
class Music3Config:
    llm: LLMConfig
    depth: DepthConfig
    dit: DiTConfig
    cond: CondConfig
    vocoder: VocoderConfig

    @staticmethod
    def default() -> "Music3Config":
        return Music3Config(LLMConfig(), DepthConfig(), DiTConfig(), CondConfig(), VocoderConfig())

    @staticmethod
    def from_snapshot(snapshot: str | os.PathLike) -> "Music3Config":
        s = Path(snapshot)
        j = lambda sub: json.load(open(s / sub / "config.json"))
        lm, dd, dt, ce, vc = (
            j("language_model"),
            j("rvq_depth_decoder"),
            j("transformer"),
            j("condition_encoder"),
            j("vocoder"),
        )
        assert lm["model_type"] == "qwen3" and lm["architectures"] == ["Qwen3ForCausalLM"], lm.get("architectures")
        rope = lm.get("rope_parameters") or {}
        cfg = Music3Config(
            llm=LLMConfig(
                lm["hidden_size"],
                lm["intermediate_size"],
                lm["num_hidden_layers"],
                lm["num_attention_heads"],
                lm["num_key_value_heads"],
                lm["head_dim"],
                lm["vocab_size"],
                float(lm["rms_norm_eps"]),
                float(rope.get("rope_theta", lm.get("rope_theta", 1e6))),
                int(lm["max_position_embeddings"]),
            ),
            depth=DepthConfig(
                dd["hidden_size"],
                dd["num_layers"],
                dd["num_attention_heads"],
                dd["intermediate_size"],
                dd["audio_vocab_size"],
                dd["num_codebooks"],
                dd["max_position_embeddings"],
            ),
            dit=DiTConfig(
                dt["in_channels"],
                dt["condition_dim"],
                dt["num_layers"],
                dt["num_attention_heads"],
                dt["attention_head_dim"],
                dt["ff_inner_dim"],
                dt["rotary_dim"],
                dt["fourier_embedding_dim"],
            ),
            cond=CondConfig(
                ce["condition_hidden_dim"],
                ce["num_condition_layers"],
                ce["out_dim"],
                ce["input_sampling_rate"],
                ce["input_hop_length"],
                ce["output_sampling_rate"],
                ce["output_hop_length"],
            ),
            vocoder=VocoderConfig(
                vc["latent_channels"],
                vc["decoder_input_dim"],
                vc["decoder_hidden_dim"],
                tuple(vc["upsampling_ratios"]),
                vc["sampling_rate"],
            ),
        )
        assert cfg.llm.vocab_size > AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE, cfg.llm.vocab_size
        assert cfg.depth.num_codebooks == NUM_CODEBOOKS and cfg.depth.audio_vocab_size == AUDIO_VOCAB_SIZE
        assert cfg.cond.num_condition_layers == NUM_CODEBOOKS and cfg.dit.in_channels == LATENT_CHANNELS
        return cfg

    def as_dict(self):
        return asdict(self)


def resolve_snapshot(repo_id: str | None = None, revision: str | None = None, allow_download: bool = True) -> Path:
    """Locate the HF snapshot dir. Offline-first: the tt-model container mounts the host HF cache; MUSIC3_SNAPSHOT
    or a directory in HF_MODEL short-circuits."""
    p = os.environ.get("MUSIC3_SNAPSHOT")
    if p and os.path.isdir(p):
        return Path(p)
    repo_id = repo_id or os.environ.get("HF_MODEL") or HF_REPO_ID
    if os.path.isdir(repo_id):
        return Path(repo_id)
    revision = revision or os.environ.get("MUSIC3_WEIGHTS_REVISION") or HF_REVISION
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(repo_id, revision=revision, local_files_only=True, ignore_patterns=HF_IGNORE_PATTERNS)
        )
    except Exception:
        if not allow_download:
            raise
        return Path(snapshot_download(repo_id, revision=revision, ignore_patterns=HF_IGNORE_PATTERNS))
