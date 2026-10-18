# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
HF_REPO_ID = "coqui/XTTS-v2"
CHECKPOINT_FILE = "model.pth"
VOCAB_FILE = "vocab.json"
# Pin so PCC/perf stay reproducible — unpinned downloads follow the repo default branch.
HF_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"

# ---------------------------------------------------------------------------
# GPT-2 backbone (coqui/XTTS-v2 config.json)
# ---------------------------------------------------------------------------
NUM_LAYERS = 30
HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
FFN_SIZE = 4 * HIDDEN_SIZE
LAYER_NORM_EPS = 1e-5

# Learned position tables: gpt.text_pos_embedding=404, gpt.mel_pos_embedding=608.
MAX_TEXT_POS = 404  # gpt_max_text_tokens (402) + 2
MAX_MEL_POS = 608  # gpt_max_audio_tokens (605) + 3
MAX_GPT_SEQ_LEN = MAX_TEXT_POS + MAX_MEL_POS
MAX_POSITIONS = MAX_GPT_SEQ_LEN

NUM_TEXT_TOKENS = 6681
NUM_AUDIO_TOKENS = 1026

# ---------------------------------------------------------------------------
# Special tokens
# Text is wrapped [START(261)] + ([lang] + tokens) + [STOP(0)].
# ---------------------------------------------------------------------------
START_TEXT_TOKEN = 261
STOP_TEXT_TOKEN = 0
START_AUDIO_TOKEN = 1024
STOP_AUDIO_TOKEN = 1025
MAX_AUDIO_TOKENS = 605

# ---------------------------------------------------------------------------
# Conditioning encoder + perceiver resampler
# ---------------------------------------------------------------------------
COND_N_MELS = 80  # GPT branch; speaker encoder is 64-mel
NUM_ATTN_HEADS = 16
NUM_LATENTS = 32
GROUP_NORM_GROUPS = 32
GROUP_NORM_EPS = 1e-5
ENC_HEAD_DIM = HIDDEN_SIZE // NUM_ATTN_HEADS
PERCEIVER_HEADS = 8
PERCEIVER_HEAD_DIM = 64
PERCEIVER_DEPTH = 2
PERCEIVER_FF_MULT = 4
PERCEIVER_INNER = PERCEIVER_HEADS * PERCEIVER_HEAD_DIM

# ---------------------------------------------------------------------------
# Conditioning mel — 22.05 kHz
# ---------------------------------------------------------------------------
MEL_N_FFT = 2048
MEL_HOP = 256
MEL_WIN = 1024
MEL_SR = 22050
MEL_FMIN = 0
MEL_FMAX = 8000

# Style embeddings are averaged over gpt_cond_chunk_len windows up to gpt_cond_len.
GPT_COND_LEN_SEC = 30
GPT_COND_CHUNK_SEC = 4
COND_CHUNK_SEC = 6  # legacy load_reference_audio default
COND_MIN_CHUNK_FRAMES = 32  # drop a tiny trailing chunk
COND_CHUNK_FRAMES = int(round(GPT_COND_CHUNK_SEC * MEL_SR / MEL_HOP))
COND_MAX_FRAMES = int(round(GPT_COND_LEN_SEC * MEL_SR / MEL_HOP))
COND_CHUNK_SAMPLES = int(round(GPT_COND_CHUNK_SEC * MEL_SR))
COND_MIN_CHUNK_SAMPLES = COND_MIN_CHUNK_FRAMES * MEL_HOP
COND_MAX_SAMPLES = int(round(GPT_COND_LEN_SEC * MEL_SR))

COQUI_TESTS_WAV_URL = "https://raw.githubusercontent.com/coqui-ai/TTS/dev/tests/data/ljspeech/wavs"

# ---------------------------------------------------------------------------
# Speaker-encoder mel — 16 kHz
# ---------------------------------------------------------------------------
SPK_SAMPLE_RATE = 16000
SPK_N_FFT = 512
SPK_HOP_LENGTH = 160
SPK_WIN_LENGTH = 400
SPK_N_MELS = 64
SPK_POWER = 2.0
SPK_PREEMPH = 0.97
SPK_FRONTEND_PREFIX = "hifigan_decoder.speaker_encoder.torch_spec."

# SE-ResNet-34
SPK_INPUT_DIM = SPK_N_MELS
SPK_PROJ_DIM = 512
SPK_LAYERS = [3, 4, 6, 3]
SPK_NUM_FILTERS = [32, 64, 128, 256]
SPK_REDUCTION = 8
SPK_LOG_INPUT = True
SPK_OUTMAP_SIZE = SPK_INPUT_DIM // 8
SPK_ASP_DIM = SPK_OUTMAP_SIZE * SPK_NUM_FILTERS[3]
SPK_BN_EPS = 1e-5
SPK_INSTANCENORM_EPS = 1e-5
SPK_ASP_EPS = 1e-5

# ---------------------------------------------------------------------------
# HiFi-GAN generator
# ---------------------------------------------------------------------------
DECODER_INPUT_DIM = 1024
UPSAMPLE_INITIAL_CHANNEL = 512
UPSAMPLE_RATES = [8, 8, 2, 2]  # product = 256 = output_hop_length
UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
COND_CHANNELS = 512
OUT_CHANNELS = 1
LRELU_SLOPE = 0.1
FINAL_LRELU_SLOPE = 0.01  # F.leaky_relu default, used before conv_post

AR_MEL_LENGTH_COMPRESSION = 1024
OUTPUT_HOP_LENGTH = 256
INPUT_SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 24000
LATENT_SCALE = AR_MEL_LENGTH_COMPRESSION / OUTPUT_HOP_LENGTH
SR_SCALE = OUTPUT_SAMPLE_RATE / INPUT_SAMPLE_RATE

# ---------------------------------------------------------------------------
# Device / tiling
# ---------------------------------------------------------------------------
TILE = 32
NEG_INF = -1e30

L1_SMALL_SIZE = 65536
# Chunked takes hold setup + decode + vocoder traces at once (one-shot releases each first).
SESSION_TRACE_REGION = 157286400  # 150 MB

# ---------------------------------------------------------------------------
# Text / language
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE = "en"
# Final "." is its own token (id 9) and the model tends to say "dot"; internal commas stay.
SENTENCE_FINAL_PUNCT_RE = r"[.!?]+\s*$"
SENTENCE_SPLIT_RE = r"(?<=[.!?])\s+"
CLAUSE_SPLIT_RE = r"(?<=[,;:])\s+"
COQUI_CLIP_RE = r"^LJ\d{3}-\d{4}\.wav$"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    """Sampling knobs for the autoregressive GPT decode."""

    temperature: float = 0.65  # 0 = greedy
    top_k: int = 50
    top_p: float = 0.85
    repetition_penalty: float = 5.0
    # Traced decode replays a fixed step count and trims STOP afterwards.
    max_tokens: int = 240
    # 0 = disabled (HF default). Negative = auto (min_tokens_auto_factor × wrapped text length).
    min_tokens: int = 0
    min_tokens_auto_factor: float = 2.0
    num_outputs: int = 1  # >1 = best-of-N by CER
    seed: int | None = None


GENERATION = GenerationConfig()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkingConfig:
    """Text-split budgets for single-pass vs chunked synthesis."""

    max_text_ids: int = 352  # under MAX_TEXT_POS (404) with headroom
    max_single_pass_codes: int = 205  # above this, split into chunks
    # Must stay under chunk_max_tokens / ~1.2 so a chunk can still emit STOP.
    max_chunk_codes: int = 205
    codes_per_id: float = 156 / 71.0  # measured: 71 text ids -> 156 audio codes
    # Shared vocoder capture length (tile-aligned). 256 avoids most cap-overruns; 288 costs ~11% RTF.
    chunk_max_tokens: int = 256
    chunk_retries: int = 2  # redraws if a chunk hits the cap without STOP


CHUNKING = ChunkingConfig()


# ---------------------------------------------------------------------------
# Audio post-processing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioPostConfig:
    """Onset/offset fade and silence padding for vocoder output."""

    fade_seconds: float = 0.015
    pad_seconds: float = 0.06
    chunk_gap_seconds: float = 0.12


AUDIO_POST = AudioPostConfig()


# ---------------------------------------------------------------------------
# Take scoring (num_outputs > 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringConfig:
    """ASR settings for CER-based best-of-N take ranking."""

    asr_model_id: str = "openai/whisper-base.en"
    asr_device: str = "cpu"
    asr_sample_rate: int = 16000


SCORING = ScoringConfig()

# Used by eval/xtts_eval.py. Revisions pinned like HF_REVISION.
EVAL_WHISPER_MODEL_ID = "openai/whisper-large-v3"
EVAL_WHISPER_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
EVAL_WHISPER_SR = 16000
EVAL_UTMOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
EVAL_UTMOS_SR = 16000
EVAL_ECAPA2_REPO_ID = "Jenthe/ECAPA2"
EVAL_ECAPA2_REVISION = "207cb6d137c671a12ba820ebec3b719549b06c0f"
EVAL_ECAPA2_SR = 16000


# ---------------------------------------------------------------------------
# Demo defaults
# ---------------------------------------------------------------------------
@dataclass
class DemoConfig:
    """Defaults for the TTNN XTTS demo (demo/xtts_demo.py)."""

    text: str = "Voice synthesis has come a long way, and modern systems can already generate natural sounding speech with remarkable accuracy. Hey, how are you doing? "
    ref_audio: str = "LJ001-0001.wav+LJ001-0003.wav+LJ001-0004.wav+LJ001-0005.wav"
    language: str = DEFAULT_LANGUAGE
    output: str = "generated/xtts_demo/xtts_demo.wav"
    write_torch_ref: bool = False

    ref_seconds: int = GPT_COND_LEN_SEC
    # Co-resident with the traced model, L1 clashes above ~20 s. GPT conditioning still uses 30 s.
    spk_seconds: int = 16

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    audio_post: AudioPostConfig = field(default_factory=AudioPostConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    device_id: int = 0
    l1_small_size: int = L1_SMALL_SIZE
    session_trace_region: int = SESSION_TRACE_REGION


DEMO = DemoConfig()


@dataclass
class ReferenceDemoConfig:
    """Defaults for the host-only XTTS reference demo."""

    text: str = DemoConfig.text
    ref_audio: str = DemoConfig.ref_audio
    language: str = DEFAULT_LANGUAGE
    output: str = "generated/xtts_reference_demo/xtts_reference_demo.wav"

    ref_seconds: int = GPT_COND_LEN_SEC
    spk_seconds: int = GPT_COND_LEN_SEC

    # CPU path can stop at STOP, so a larger cap is free when unused.
    generation: GenerationConfig = field(default_factory=lambda: replace(GENERATION, max_tokens=400))
    audio_post: AudioPostConfig = field(default_factory=AudioPostConfig)

    max_text_ids: int = MAX_TEXT_POS - 52
    max_pass_codes: int = 560  # under MAX_AUDIO_TOKENS, with overshoot margin
    codes_per_id: float = CHUNKING.codes_per_id

    threads: int = 4  # 0 = torch default (one per core); oversubscribes on a shared host
    decode_threads: int = 2  # AR loop is launch-bound; 0 = use threads


REFERENCE_DEMO = ReferenceDemoConfig()
