# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------------------
# Checkpoint (checkpoint facts)
# ---------------------------------------------------------------------------
HF_REPO_ID = "coqui/XTTS-v2"
CHECKPOINT_FILE = "model.pth"
VOCAB_FILE = "vocab.json"  # XTTS-v2 BPE tokenizer, alongside model.pth in the HF repo
# Pinned so PCC/perf numbers stay reproducible: unpinned downloads follow the repo's default
# branch, and an upstream re-upload would move them silently. Every download from HF_REPO_ID
# (checkpoint, vocab.json, samples/*.wav) passes this.
HF_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"

# ---------------------------------------------------------------------------
# GPT-2 backbone (checkpoint facts)
# Read off coqui/XTTS-v2 config.json: model_args.gpt_layers /
# gpt_n_model_channels / gpt_n_heads.
# ---------------------------------------------------------------------------
NUM_LAYERS = 30
HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64
FFN_SIZE = 4 * HIDDEN_SIZE  # 4096 (GPT2 n_inner default)
LAYER_NORM_EPS = 1e-5

# Sequence-length limits, read off the checkpoint's learned position embeddings
# (gpt.text_pos_embedding=404, gpt.mel_pos_embedding=608). At inference the GPT
# runs on the concatenated [text] + [mel] stream, so coqui sizes the GPT-2
# causal backbone to n_positions = text + mel.
MAX_TEXT_POS = 404  # gpt_max_text_tokens (402) + 2
MAX_MEL_POS = 608  # gpt_max_audio_tokens (605) + 3
MAX_GPT_SEQ_LEN = MAX_TEXT_POS + MAX_MEL_POS  # 1012 — full GPT context
MAX_POSITIONS = MAX_GPT_SEQ_LEN  # sizes the causal mask; must cover any tested seq_len

# Vocab sizes, read off the checkpoint embedding/head tensors
# (gpt.text_embedding=6681, gpt.mel_embedding=1026).
NUM_TEXT_TOKENS = 6681
NUM_AUDIO_TOKENS = 1026

# ---------------------------------------------------------------------------
# Special tokens (checkpoint facts)
# Text is wrapped [START(261)] + ([lang] + tokens) + [STOP(0)]. config.json
# carries gpt_start/stop_text_token=None; the coqui GPT constructor defaults are
# 261/0, exactly the [START]/[STOP] ids in vocab.json.
# ---------------------------------------------------------------------------
START_TEXT_TOKEN = 261  # [START] in vocab.json
STOP_TEXT_TOKEN = 0  # [STOP]
START_AUDIO_TOKEN = 1024  # gpt_start_audio_token
STOP_AUDIO_TOKEN = 1025  # gpt_stop_audio_token
MAX_AUDIO_TOKENS = 605  # gpt_max_audio_tokens

# ---------------------------------------------------------------------------
# Conditioning encoder + perceiver resampler (checkpoint facts)
# gpt.conditioning_encoder.* and gpt.conditioning_perceiver.*
# ---------------------------------------------------------------------------
COND_N_MELS = 80  # conditioning mel bands (the GPT branch; NOT the speaker encoder's 64)
NUM_ATTN_HEADS = 16  # GPT.__init__ passes heads (=16) to ConditioningEncoder
NUM_LATENTS = 32  # perceiver latents = the GPT prompt length contributed by the audio
GROUP_NORM_GROUPS = 32
GROUP_NORM_EPS = 1e-5
ENC_HEAD_DIM = HIDDEN_SIZE // NUM_ATTN_HEADS  # 64
PERCEIVER_HEADS = 8
PERCEIVER_HEAD_DIM = 64
PERCEIVER_DEPTH = 2
PERCEIVER_FF_MULT = 4  # coqui PerceiverResampler feed-forward multiplier
PERCEIVER_INNER = PERCEIVER_HEADS * PERCEIVER_HEAD_DIM  # 512

# ---------------------------------------------------------------------------
# Conditioning mel — 22.05 kHz branch (checkpoint facts)
# coqui get_gpt_cond_latents' mel frontend.
# ---------------------------------------------------------------------------
MEL_N_FFT = 2048
MEL_HOP = 256
MEL_WIN = 1024
MEL_SR = 22050
MEL_FMIN = 0
MEL_FMAX = 8000

# Conditioning windows (upstream defaults). coqui conditions on up to
# gpt_cond_len=30 s of reference audio, split into gpt_cond_chunk_len=4 s
# windows, running get_style_emb per chunk and AVERAGING the 32-latent style
# embeddings.
GPT_COND_LEN_SEC = 30  # gpt_cond_len / max_ref_len
GPT_COND_CHUNK_SEC = 4  # gpt_cond_chunk_len
COND_CHUNK_SEC = 6  # legacy single-window length (load_reference_audio default)
COND_MIN_CHUNK_FRAMES = 32  # drop a tiny trailing chunk (also keeps lengths tile-sane)
COND_CHUNK_FRAMES = int(round(GPT_COND_CHUNK_SEC * MEL_SR / MEL_HOP))  # ~344 mel frames / chunk
COND_MAX_FRAMES = int(round(GPT_COND_LEN_SEC * MEL_SR / MEL_HOP))  # gpt_cond_len as mel frames
COND_CHUNK_SAMPLES = int(round(GPT_COND_CHUNK_SEC * MEL_SR))  # 88200 samples / 4 s window
COND_MIN_CHUNK_SAMPLES = COND_MIN_CHUNK_FRAMES * MEL_HOP
COND_MAX_SAMPLES = int(round(GPT_COND_LEN_SEC * MEL_SR))  # gpt_cond_len: 30 s of reference audio

# Single-speaker LJSpeech clips shipped as test data in the upstream coqui repo, already at MEL_SR.
COQUI_TESTS_WAV_URL = "https://raw.githubusercontent.com/coqui-ai/TTS/dev/tests/data/ljspeech/wavs"

# ---------------------------------------------------------------------------
# Speaker-encoder mel frontend — 16 kHz branch (checkpoint facts)
# coqui: nn.Sequential(PreEmphasis(0.97), torchaudio.MelSpectrogram(...)).
# ---------------------------------------------------------------------------
SPK_SAMPLE_RATE = 16000
SPK_N_FFT = 512
SPK_HOP_LENGTH = 160
SPK_WIN_LENGTH = 400
SPK_N_MELS = 64
SPK_POWER = 2.0
SPK_PREEMPH = 0.97
SPK_FRONTEND_PREFIX = "hifigan_decoder.speaker_encoder.torch_spec."

# Speaker-encoder body — coqui ResNetSpeakerEncoder (SE-ResNet-34).
SPK_INPUT_DIM = SPK_N_MELS  # 64
SPK_PROJ_DIM = 512  # speaker embedding dim (== d_vector_dim)
SPK_LAYERS = [3, 4, 6, 3]
SPK_NUM_FILTERS = [32, 64, 128, 256]
SPK_REDUCTION = 8
SPK_LOG_INPUT = True
SPK_OUTMAP_SIZE = SPK_INPUT_DIM // 8  # freq dim after 3 stride-2 downsamples = 8
SPK_ASP_DIM = SPK_OUTMAP_SIZE * SPK_NUM_FILTERS[3]  # 2048
SPK_BN_EPS = 1e-5
SPK_INSTANCENORM_EPS = 1e-5
SPK_ASP_EPS = 1e-5

# ---------------------------------------------------------------------------
# HiFi-GAN generator — waveform_decoder (checkpoint facts)
# coqui/XTTS-v2 config, model_args.
# ---------------------------------------------------------------------------
DECODER_INPUT_DIM = 1024  # GPT latent dim fed to conv_pre
UPSAMPLE_INITIAL_CHANNEL = 512
UPSAMPLE_RATES = [8, 8, 2, 2]  # product = 256 = output_hop_length
UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
COND_CHANNELS = 512  # d_vector_dim (speaker embedding)
OUT_CHANNELS = 1
LRELU_SLOPE = 0.1
FINAL_LRELU_SLOPE = 0.01  # coqui's pre-conv_post activation uses the F.leaky_relu default

# HifiDecoder latent pre-upsampling rates.
AR_MEL_LENGTH_COMPRESSION = 1024
OUTPUT_HOP_LENGTH = 256
INPUT_SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 24000
LATENT_SCALE = AR_MEL_LENGTH_COMPRESSION / OUTPUT_HOP_LENGTH  # 4.0
SR_SCALE = OUTPUT_SAMPLE_RATE / INPUT_SAMPLE_RATE  # 160/147 ≈ 1.08844

# ---------------------------------------------------------------------------
# Device / tiling
# ---------------------------------------------------------------------------
TILE = 32  # ttnn tile height/width; all padded lengths round up to this
NEG_INF = -1e30  # additive attention-mask fill for masked-out positions

L1_SMALL_SIZE = 65536  # l1_small_size for ttnn.open_device on this model
# A chunked take holds the setup + decode + vocoder traces LIVE at the same time (the one-shot
# path releases each before capturing the next), so it needs a trace region for all three.
SESSION_TRACE_REGION = 157286400  # 150 MB

# ---------------------------------------------------------------------------
# Text / language
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE = "en"
# Strips sentence-final punctuation: the final "." is its own token (id 9) and the model
# tends to VERBALIZE it as "dot" at the tail. Internal commas (prosody) are kept.
SENTENCE_FINAL_PUNCT_RE = r"[.!?]+\s*$"
SENTENCE_SPLIT_RE = r"(?<=[.!?])\s+"
# Internal prosodic boundaries. A sentence too long for one pass is broken here first, so the
# seam lands where a speaker would pause anyway (upstream coqui hard-wraps such sentences).
CLAUSE_SPLIT_RE = r"(?<=[,;:])\s+"
COQUI_CLIP_RE = r"^LJ\d{3}-\d{4}\.wav$"  # coqui-ai/TTS tests/data/ljspeech/wavs clip names


# ---------------------------------------------------------------------------
# Generation defaults (upstream defaults)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    """Sampling knobs for the autoregressive GPT decode."""

    temperature: float = 0.65  # 0 = greedy; 0.65 = cleanest single take
    top_k: int = 50
    top_p: float = 0.85  # nucleus cutoff (XTTS uses 0.85)
    repetition_penalty: float = 5.0  # XTTS uses 5.0
    # Traced decode replays a fixed max_tokens steps and trims STOP afterwards. 240 sits above
    # max_single_pass_codes (205) with sampler overshoot room.
    max_tokens: int = 240
    # STOP-suppression floor in audio codes. 0 = disabled (HF default). Negative = auto
    # (min_tokens_auto_factor × wrapped text length). Useful for long prompts; on short ones it
    # can force an invented tail.
    min_tokens: int = 0
    min_tokens_auto_factor: float = 2.0  # auto floor = factor x padded text length
    num_outputs: int = 1  # takes to generate; >1 = best-of-N by CER (coqui num_gpt_outputs=1)
    seed: int | None = None  # None = unseeded (ttnn sampling isn't bit-exact across runs anyway)


GENERATION = GenerationConfig()


# ---------------------------------------------------------------------------
# Chunking budgets (port budgets)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkingConfig:
    """Text-split budgets for single-pass vs chunked synthesis."""

    max_text_ids: int = 352  # keep the padded text under MAX_TEXT_POS (404) with headroom
    max_single_pass_codes: int = 205  # above this, split into chunks
    # Per chunk once splitting; lower than single-pass on purpose. codes_per_id is a linear fit,
    # and a real take runs up to ~1.2x it (measured over a 37-chunk paragraph), so this must stay
    # under chunk_max_tokens / 1.2 — a chunk that outgrows its budget never reaches STOP and its
    # tail comes out as noise.
    max_chunk_codes: int = 155
    codes_per_id: float = 156 / 71.0  # measured: 71 text ids -> 156 audio codes
    # Chunked takes share one capture, so the vocoder always runs this many latent frames
    # (zero-padded). Tile-aligned, above max_chunk_codes, below the ~205 L1 clash.
    chunk_max_tokens: int = 192
    # Redraws allowed for a chunk that reaches the code cap without emitting STOP (an unfinished,
    # usually noisy tail). Sampling is stochastic, so a redraw is normally enough; each costs one
    # trace replay.
    chunk_retries: int = 2


CHUNKING = ChunkingConfig()


# ---------------------------------------------------------------------------
# Audio post-processing (upstream defaults)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioPostConfig:
    """Onset/offset fade and silence padding for vocoder output."""

    fade_seconds: float = 0.015  # ~15 ms raised-cosine fade in/out
    pad_seconds: float = 0.06  # ~60 ms of silence as lead-in/out
    chunk_gap_seconds: float = 0.12  # ~120 ms of silence joining chunk waveforms


AUDIO_POST = AudioPostConfig()


# ---------------------------------------------------------------------------
# Take scoring — best-of-N ranking (only used when num_outputs > 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringConfig:
    """ASR settings for CER-based best-of-N take ranking."""

    asr_model_id: str = "openai/whisper-base.en"
    asr_device: str = "cpu"
    asr_sample_rate: int = 16000


SCORING = ScoringConfig()

# Heavier evaluation models used by ``eval/xtts_eval.py`` (not the demo's quick CER).
# Revisions are pinned for the same reason as HF_REVISION: unpinned downloads follow the
# default branch, so an upstream re-upload would move recorded metrics with no warning.
EVAL_WHISPER_MODEL_ID = "openai/whisper-large-v3"
EVAL_WHISPER_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
EVAL_WHISPER_SR = 16000
EVAL_UTMOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"  # torch.hub tag (not a commit)
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

    text: str = (
        "Voice synthesis has come a long way, and modern systems can already generate "
        "natural sounding speech with remarkable accuracy. Hey how are you doing?."
    )
    # Four LJSpeech clips joined to ~32.6 s, clipped to gpt_cond_len (30 s) = 8 conditioning windows.
    ref_audio: str = "LJ001-0001.wav+LJ001-0003.wav+LJ001-0004.wav+LJ001-0005.wav"
    language: str = DEFAULT_LANGUAGE
    output: str = "generated/xtts_demo/xtts_demo.wav"
    write_torch_ref: bool = False

    ref_seconds: int = GPT_COND_LEN_SEC  # conditioning window (coqui gpt_cond_len)
    # Speaker-embedding window. Upstream uses the whole reference (up to 30 s). Co-resident with the
    # rest of the traced model this one clashes L1 above ~20 s, so it keeps a margin below that.
    # Longer is better but flattens out: ECAPA2 similarity to the reference measures 0.694 / 0.715 /
    # 0.731 / 0.736 / 0.743 at 4 / 8 / 12 / 16 / 20 s, for +0.3 ms of setup replay per second of
    # window. GPT conditioning uses the full 30 s regardless — this window only feeds the speaker
    # vector, which conditions the HiFi-GAN and leaves the generated codes untouched.
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

    ref_seconds: int = GPT_COND_LEN_SEC  # conditioning window (gpt_cond_len)
    # Speaker-embedding window. Defaults to the WHOLE reference (coqui max_ref_length) — unlike
    # the device demo, which keeps a margin under the L1 ceiling.
    spk_seconds: int = GPT_COND_LEN_SEC

    # Sampling mirrors the device demo, but the code cap is the CPU one: STOP genuinely ends the
    # loop here (no fixed-length traced replay), so a generous cap costs nothing when unused.
    generation: GenerationConfig = field(default_factory=lambda: replace(GENERATION, max_tokens=400))
    audio_post: AudioPostConfig = field(default_factory=AudioPostConfig)

    # Single-pass budgets. NOT L1 limits — on CPU the only walls are the checkpoint's learned
    # position tables: text_pos_embedding (MAX_TEXT_POS rows) and mel_pos_embedding
    # (MAX_MEL_POS rows, i.e. MAX_AUDIO_TOKENS codes). Both are left with headroom.
    max_text_ids: int = MAX_TEXT_POS - 52  # 352 wrapped text ids
    max_pass_codes: int = 560  # below the 605-code mel budget, with margin for sampler overshoot
    codes_per_id: float = CHUNKING.codes_per_id

    # torch CPU threads for the big-tensor stages. 0 = leave torch's default (one per core), which
    # oversubscribes and thrashes on a shared host.
    threads: int = 4
    # Threads for the autoregressive loop only. A single-token step is many tiny GEMMs, so it is
    # launch-bound: fewer threads is much faster. 0 = use ``threads``.
    decode_threads: int = 2


REFERENCE_DEMO = ReferenceDemoConfig()
