# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for every XTTS-v2 configuration value.

Everything that is a *number the model is defined by* — checkpoint coordinates,
GPT-2 backbone dims, sequence limits, token ids, mel/STFT parameters, the
speaker-encoder and HiFi-GAN topologies, the sampling defaults, and the
device-side budgets the demo runs against — lives here and nowhere else. The
reference modules, the TTNN port and the demo all import from this file, so a
value is changed in exactly one place.

Values fall into three kinds; they are labelled per section:

  * **checkpoint facts** — fixed by ``coqui/XTTS-v2`` ``model.pth`` /
    ``config.json``. Changing one makes the weights stop loading (or load and
    produce garbage). Do not "tune" these.
  * **upstream defaults** — coqui's inference-time choices (sampling
    temperature, conditioning window lengths). Safe to change; changes what the
    audio sounds like.
  * **port budgets** — limits this TTNN port measured on hardware (L1 clashes,
    code budgets, trace region size). Safe to change, but they were measured;
    see the comment on each before moving one.

What is deliberately NOT here: per-op TTNN tuning inside ``tt/`` (memory
configs, math fidelity, shard-strategy flags, program configs). Those are
properties of a specific kernel implementation rather than of the model, and
they only make sense next to the code that uses them.
"""

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
COQUI_CLIP_RE = r"^LJ\d{3}-\d{4}\.wav$"  # coqui-ai/TTS tests/data/ljspeech/wavs clip names


# ---------------------------------------------------------------------------
# Generation defaults (upstream defaults)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    """Sampling knobs for the autoregressive GPT decode.

    ``temperature=0`` is greedy (deterministic — the validated correctness path);
    the non-zero defaults are XTTS's own, which give more natural prosody.
    """

    temperature: float = 0.65  # 0 = greedy; 0.65 = cleanest single take
    top_k: int = 50
    top_p: float = 0.85  # nucleus cutoff (XTTS uses 0.85)
    repetition_penalty: float = 5.0  # XTTS uses 5.0
    # Cap on audio codes. This is NOT a "stop earlier if you can" budget: the traced decode loop
    # replays a fixed max_tokens steps and treats STOP as a post-loop trim (a captured trace cannot
    # branch), so every step above what the text actually needs is ~9.4 ms of pure waste — 400 cost
    # ~2 s per pass while real single-pass generations land at 160-210 codes. 240 sits above
    # MAX_SINGLE_PASS_CODES (205) with margin for the overshoot the sampler is entitled to, and it
    # also shrinks the KV cache (max_seq scales with it), so nothing that fits the pass is truncated.
    max_tokens: int = 240
    # STOP-suppression floor in audio codes. 0 = disabled, matches HF. Negative = auto
    # (MIN_TOKENS_AUTO_FACTOR x the wrapped text length). A floor is only right for *long*
    # prompts. Greedy, on the default text (96 wrapped tokens, needs ~196 codes): 0 stops at
    # 152 codes and transcribes at CER 0.151, cut after "remarkable accuracy"; auto (floor 192)
    # reaches 181 codes at CER 0.000. On a short prompt it inverts — "Hello from Tenstorrent."
    # goes CER 0.273 -> 0.591, the floor forcing 12 extra codes of invented tail.
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
    """How much text fits one pass, and how text over that is split.

    TWO code budgets, because the wall behaves differently in the two cases. It is an ALLOCATION
    COLLISION ("Statically allocated circular buffers ... clash with L1 buffers"), not a clean size
    limit, so it degrades as device open/close cycles accumulate in a process. Measured on p150 by
    running the demo:

      ONE pass, fresh device (words -> codes):  20->175 PASS | 23->177,182,184 PASS | 22->192 PASS
                                                25->203 PASS | 24->207 PASS  <-- highest seen to pass
                                                27-> FAIL cb-clash | 29-> FAIL
      Nth pass, same process:                   a chunk estimated at ~204 codes FAILED as the 5th
                                                cycle, while 207 passed as the 1st -> per-chunk
                                                headroom SHRINKS with chunk count, so the chunk
                                                budget must be lower.

    Note the same text varies +/-4% run to run (23 words gave 177/182/184), so leave margin.
    """

    max_text_ids: int = 352  # keep the padded text under MAX_TEXT_POS (404) with headroom
    max_single_pass_codes: int = 205  # above this, split into chunks
    max_chunk_codes: int = 165  # per chunk once splitting; lower than single-pass on purpose
    codes_per_id: float = 156 / 71.0  # measured: 71 text ids -> 156 audio codes
    # Chunked takes share ONE capture, so the decode budget is baked into it: the vocoder always
    # runs on this many latent frames (zero-padded past the codes actually generated) instead of on
    # the exact length. It therefore has to sit above max_chunk_codes with margin for sampler
    # overshoot, and below the ~205 codes where the vocoder's circular buffers start clashing with
    # L1. 192 is a multiple of TILE (the decode accumulators tile cleanly) — the same budget the
    # trace test uses.
    chunk_max_tokens: int = 192
    # Words in the longest sentence that still fits a pass. A sentence is never split, so a longer
    # one cannot be made to fit and the caller is warned instead. ~3.7 text ids per word, measured.
    ids_per_word: float = 3.7


CHUNKING = ChunkingConfig()


# ---------------------------------------------------------------------------
# Audio post-processing (upstream defaults)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioPostConfig:
    """Onset/offset cleanup on the vocoder output.

    The vocoder starts at the first content code with no natural lead-in, which reads as an
    abrupt ("crimped") onset — a short raised-cosine fade plus a little silence fixes it.
    """

    fade_seconds: float = 0.015  # ~15 ms raised-cosine fade in/out
    pad_seconds: float = 0.06  # ~60 ms of silence as lead-in/out
    chunk_gap_seconds: float = 0.12  # ~120 ms of silence joining chunk waveforms


AUDIO_POST = AudioPostConfig()


# ---------------------------------------------------------------------------
# Take scoring — best-of-N ranking (only used when num_outputs > 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringConfig:
    """Primary metric is CER (ASR transcription vs the input text — directly measures
    "does the audio say the text"), with a code-diversity fallback when the ASR backends
    are unavailable."""

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
    """Everything ``demo/xtts_demo.py`` runs with.

    Only ``text`` / ``ref_audio`` / ``min_tokens`` are exposed on the command line; the rest
    is fixed to the tuned XTTS-v2 defaults so the demo runs full-model-traced with no other
    knobs. Change a default here rather than in the demo.
    """

    # "can already" (not "can now"): "can now" is a /n/#/n/ nasal collision the vocoder
    # merges into "cannow/cannot" — "already" starts with a vowel and transcribes cleanly (CER 0.008).
    text: str = (
        "Voice synthesis has come a long way, and modern systems can already generate "
        "natural sounding speech with remarkable accuracy. Hey how are you doing?."
    )
    # DOWNLOADABLE by default (cached under torch.hub): four single-speaker coqui-ai/TTS LJSpeech
    # test clips joined to ~32.6 s, clipped to gpt_cond_len (30 s) = 8 conditioning windows — the
    # shapes this demo is tuned for. A single HF sample (en_sample.wav) is ~3 s / ONE window.
    ref_audio: str = "LJ001-0001.wav+LJ001-0003.wav+LJ001-0004.wav+LJ001-0005.wav"
    language: str = DEFAULT_LANGUAGE
    output: str = "generated/xtts_demo/xtts_demo.wav"
    write_torch_ref: bool = False

    ref_seconds: int = GPT_COND_LEN_SEC  # conditioning window (coqui gpt_cond_len)
    # Speaker-embedding window. coqui uses the whole reference (max_ref_length=30 s) here, but the
    # speaker ENCODER does not fit one: 30 s clashes L1 ("circular buffers ... end at 1184832"
    # against a buffer at 986496) in the ResNet, not in the mel frontend, which now chunks its
    # framing and takes any length. 8 s runs, and a speaker embedding is pooled over time anyway,
    # so it saturates well before this. The GPT conditioning above uses the FULL 30 s.
    spk_seconds: int = 8

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
    """Everything ``demo/xtts_reference_demo.py`` (the host-only CPU twin) runs with.

    Same text/voice defaults as :class:`DemoConfig` so the two demos A/B directly, but its
    budgets are different: on CPU there are no L1 walls, only the checkpoint's learned position
    tables, so a pass is much longer.
    """

    text: str = DemoConfig.text
    ref_audio: str = DemoConfig.ref_audio
    language: str = DEFAULT_LANGUAGE
    output: str = "generated/xtts_reference_demo/xtts_reference_demo.wav"

    ref_seconds: int = GPT_COND_LEN_SEC  # conditioning window (gpt_cond_len)
    # Speaker-embedding window. Defaults to the WHOLE reference (coqui max_ref_length) — unlike
    # the device demo, which is capped at 8 s by L1.
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
