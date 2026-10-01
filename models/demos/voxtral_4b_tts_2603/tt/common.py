# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the `mistralai/Voxtral-4B-TTS-2603` end-to-end TTNN package.

Everything here is SETUP, never forward-path compute: the PyTorch reference loader, the real
tokenizer, the speech-prompt and batch builders, the decode safety cap, a PCC helper and the
on-disk cache for the CPU reference results.

The hub repo ships `params.json`, `consolidated.safetensors` and `tekken.json` -- no
`config.json`, so `AutoConfig`/`AutoModel` raise. `reference/reference_loader.py` rebuilds the
reference model from the native checkpoint, and `load_reference_model()` delegates to it.
"""
from __future__ import annotations

import base64
import importlib
import json
import os
from functools import lru_cache

import torch

HF_MODEL_ID = "mistralai/Voxtral-4B-TTS-2603"
# The checkpoint snapshot every number in the README was measured on.
HF_REVISION = os.environ.get("VOXTRAL_HF_REVISION", "b81be46c3777f88621676791b512bb01dc1cb970")


def _revision(model_id: str):
    return HF_REVISION if model_id == HF_MODEL_ID else None


# BATCH=32: 32 independent samples per pipeline call. A single sample wastes 31/32 of a
# 32-row matmul tile; filling it raises aggregate throughput ~32x at unchanged per-sample latency.
DEFAULT_BATCH = 32

# Real tokens per prompt for `build_batch_inputs` -- the plain-text prompts the acoustic and vocode
# trace hooks and the golden self-check feed the reference backbone (not the speech requests).
DEFAULT_SEQ_LEN = 32


# --------------------------------------------------------------------------------------
# The reference model and its config
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _reference_loader_module():
    return importlib.import_module("models.demos.voxtral_4b_tts_2603.reference.reference_loader")


def load_reference_model(model_id: str = HF_MODEL_ID):
    """The HF golden: `MistralForCausalLM` rebuilt from the native checkpoint."""
    return _reference_loader_module().load_reference_model(model_id)


def resolve_repo(model_id: str = HF_MODEL_ID) -> str:
    """Local directory holding params.json / consolidated.safetensors / tekken.json."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id,
        revision=_revision(model_id),
        allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
    )


def load_params(model_id: str = HF_MODEL_ID) -> dict:
    with open(os.path.join(resolve_repo(model_id), "params.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# The real tokenizer (tekken.json -> tiktoken)
# --------------------------------------------------------------------------------------


class TekkenTokenizer:
    """Mistral's Tekken v7 tokenizer, rebuilt on `tiktoken`.

    `mistral_common` is not installed in tt-metal's `python_env` (and the env has no pip), and
    `AutoTokenizer` cannot help because the repo ships no `tokenizer_config.json`. `tekken.json`
    carries everything needed: the split pattern, the ranked merge table, and the convention that a
    real token id is `rank + num_special_tokens`.
    """

    def __init__(self, tekken_path: str) -> None:
        import tiktoken

        with open(tekken_path) as f:
            spec = json.load(f)
        cfg = spec["config"]
        self.num_special = int(cfg["default_num_special_tokens"])
        self.vocab_size = int(cfg["default_vocab_size"])
        n_merges = self.vocab_size - self.num_special
        mergeable = {base64.b64decode(e["token_bytes"]): e["rank"] for e in spec["vocab"][:n_merges]}
        self._enc = tiktoken.Encoding(
            name="tekken_v7",
            pat_str=cfg["pattern"],
            mergeable_ranks=mergeable,
            special_tokens={},
        )
        self.bos_id = 1
        self.eos_id = 2

    def encode(self, text: str, bos: bool = True) -> list[int]:
        ids = [r + self.num_special for r in self._enc.encode(text, allowed_special=set())]
        return ([self.bos_id] + ids) if bos else ids

    def decode(self, ids) -> str:
        ranks = [int(i) - self.num_special for i in ids if int(i) >= self.num_special]
        return self._enc.decode(ranks) if ranks else ""


@lru_cache(maxsize=1)
def load_tokenizer(model_id: str = HF_MODEL_ID) -> TekkenTokenizer:
    return TekkenTokenizer(os.path.join(resolve_repo(model_id), "tekken.json"))


# --------------------------------------------------------------------------------------
# The real input: 32 INDEPENDENT prompts
# --------------------------------------------------------------------------------------

# 32 distinct English texts. Each is long enough that truncating to DEFAULT_SEQ_LEN real tokens
# never needs padding, so all 32 rows are genuine content and causality is untouched.
PROMPT_TEXTS = [
    "The lighthouse keeper wrote in his journal every evening, noting the colour of the water and the direction of the wind across the bay. Nothing else in the log ever varied, and that was the point of keeping it.",
    "In the quiet hours before dawn the bakery ovens were already warm, and the smell of bread drifted down the empty cobbled street outside. By six o'clock the first customers were queueing outside in the cold.",
    "She tuned the old radio slowly, sliding past static and foreign voices until a piano concerto emerged clearly from the crackling speaker. She left the dial exactly there for the rest of the evening.",
    "The expedition mapped the cave system for three weeks, marking every passage with chalk and recording the temperature at each junction. The final map took another two months to draw properly.",
    "Autumn arrived early that year, and the maple trees along the river turned a deep copper colour almost overnight, surprising everyone. Nobody could remember a season that had turned so quickly before.",
    "He repaired bicycles in a narrow shop behind the station, where spare wheels hung from the ceiling like the rings of a strange planet. Most of them had been hanging there since long before he bought the place.",
    "The letter had travelled for six months across two oceans before it finally reached the small farmhouse at the end of the gravel road. The stamps alone were worth more than the paper they were stuck to.",
    "Astronomers watched the comet approach for weeks, adjusting their instruments each night as its tail grew brighter against the dark sky. By the third week they could predict its position to within a degree.",
    "A violinist practised the same difficult passage for hours, stopping only when the light faded and the room became too dim to read music. The neighbours never once complained about the repetition.",
    "The archive basement held thousands of photographs, each one labelled in faded pencil with a date, a place, and sometimes only a first name. Sorting them properly would clearly take the whole of the coming winter.",
    "Rain fell steadily on the greenhouse roof while the gardener transplanted seedlings into deeper pots, humming an old song under her breath. The glass fogged over completely before she had finished the last tray.",
    "The ferry crossed the strait twice each morning, carrying commuters, crates of fish, and occasionally a nervous dog on a short leash. The crossing took nineteen minutes in good weather and rather longer otherwise.",
    "Engineers tested the bridge cables with careful instruments, measuring how the steel responded to the weight of traffic and to the cold. Their report ran to four hundred pages and recommended almost no changes.",
    "In the museum's back room a conservator removed varnish from a painting, revealing colours that nobody had seen for almost two centuries. The blue underneath turned out to be far brighter than anyone expected.",
    "The village held a festival every summer, with paper lanterns strung between the houses and long tables set out in the central square. Musicians came down from the city for it, and stayed until the morning.",
    "A cartographer redrew the coastline after the storm, because the sandbar had shifted nearly a hundred metres further east than before. The old chart was kept anyway, pinned to the wall beside the new one.",
    "He learned to cook from his grandmother, who measured nothing and explained everything twice, always in the middle of doing something else. He still cannot make the soup taste the way hers did.",
    "The observatory sat above the treeline, and on clear nights the students climbed the winding road carrying thermoses of strong coffee. The telescope itself was older than the building that housed it.",
    "Librarians catalogued the donated collection for months, discovering pressed flowers and train tickets tucked between the pages of novels. Someone had clearly used the books as a filing cabinet for decades.",
    "The potter worked quickly once the clay was centred, drawing the walls upward with steady pressure while the wheel turned beneath her hands. Each pot came off the wheel looking almost exactly like the last.",
    "Snow closed the mountain pass for a week, so the mail was carried in on skis by a postman who had done the same route for twenty years. He said the quiet was the only part he would genuinely miss.",
    "A biologist counted the nesting pairs along the cliff face, recording each sighting in a waterproof notebook tied to her wrist by string. The numbers were lower than last year, which worried her considerably.",
    "The orchestra rehearsed in an unheated hall, and between movements the musicians blew on their fingers and laughed about the temperature. By the final run-through nobody mentioned the cold at all.",
    "Workers restored the clock tower over two summers, replacing the escapement, regilding the hands, and finally ringing the bell at noon. The whole town came out into the square to hear it strike.",
    "She kept a small boat moored at the pier and sailed it alone on weekends, following the shoreline until the town disappeared behind her. On the way back the wind was always against her, which she preferred.",
    "The bookshop occupied three floors of a crooked building, and every staircase creaked in a slightly different key as customers climbed it. Regular visitors learned which steps to avoid if they wanted to browse unnoticed.",
    "Geologists drilled a core sample from the lakebed, reading centuries of pollen and ash in the thin bands of sediment they brought up. One dark band marked a fire that no written record had preserved.",
    "A blacksmith demonstrated the old techniques at the fair, and children watched the sparks leap while the metal changed from red to grey. The noise of the hammer carried right across the showground all afternoon.",
    "The train ran through farmland for an hour before the first suburbs appeared, low houses with gardens backing directly onto the tracks. Passengers who made the trip daily stopped noticing the view entirely.",
    "Translators argued for days about a single line of the poem, because the original word meant both a departure and a kind of forgiveness. In the end they printed both versions and let the reader decide.",
    "The bee keeper opened each hive slowly, reading the temper of the colony from its sound before lifting a single frame into the sunlight. The hum told her more than any inspection of the frames could have.",
    "Divers surveyed the wreck at forty metres, photographing the hull in overlapping strips so the whole ship could be reassembled on screen. Visibility was poor, so every strip had to be shot twice to be safe.",
]


# 32 distinct one-sentence texts for the SPEECH task, each exactly 18 tekken tokens, so the batch
# is rectangular with NO padding. It has to be: the prefill runs causal attention without a per-row
# padding mask, and the previous layout -- BOS tokens left-padded into the text segment -- was
# measured to wreck the conditioning. On device, rows needing 0-1 pad tokens came back at Whisper
# WER 0.00-0.03 while rows needing 4-11 came back as babble that never emitted end_audio (corpus
# WER 1.50, 29 of 32 rows ran to the 256-frame cap). Equal lengths are asserted, not assumed.
SPEECH_TEXTS = [
    "The lighthouse keeper wrote in his journal every evening, noting the colour of the water.",
    "In the quiet hours before dawn the bakery ovens were already warm and waiting inside.",
    "She tuned the old radio slowly until a gentle piano concerto finally emerged from the small speaker.",
    "The expedition mapped the cave system for three long weeks, marking every passage with white chalk.",
    "Autumn arrived early that year, and the tall maple trees turned copper almost overnight.",
    "He repaired bicycles in a narrow shop behind the station, where spare wheels hung everywhere.",
    "The letter had travelled for six months across two oceans before it reached the small farmhouse.",
    "Astronomers watched the comet approach for weeks, adjusting their telescopes every night.",
    "A young violinist practised the same difficult passage for hours until her fingers ached.",
    "The archive basement held thousands of old photographs, each one carefully labelled by hand in ink.",
    "Rain fell steadily on the glass greenhouse roof while the gardener repotted the orchids.",
    "The ferry crossed the strait each morning, carrying commuters and crates of fish.",
    "Engineers tested the bridge cables with careful instruments for a month before the road finally reopened.",
    "In the museum's back room a conservator removed varnish from an old painting.",
    "The village held a festival every summer, with paper lanterns strung across the square.",
    "A cartographer redrew the coastline after the storm, because the old maps were wrong.",
    "He learned to cook from his grandmother, who measured nothing at all and tasted everything twice.",
    "The observatory sat above the treeline, and on clear nights the stars seemed close.",
    "Librarians catalogued the donated collection for many months, discovering several very rare books.",
    "The potter worked fast once the clay was centred, drawing the walls up at once.",
    "Snow closed the pass for a week, so the mail was carried in on skis.",
    "A biologist counted the nesting pairs along the tall cliff face every spring for a decade.",
    "The orchestra rehearsed in a cold hall, and the players wore thick gloves between movements.",
    "Workers slowly restored the old clock tower over two summers, replacing every rusted gear.",
    "She kept a small wooden boat moored at the pier and sailed it alone on weekends.",
    "The bookshop occupied three floors of a crooked stone building near the busy old harbour.",
    "Geologists drilled a core sample from the lakebed, reading centuries of climate history.",
    "A blacksmith demonstrated the old techniques at the summer fair, and the children watched closely.",
    "The slow train ran through farmland for an hour before the first grey houses finally appeared.",
    "Translators argued for days about a single line of the poem and never fully agreed.",
    "The bee keeper opened each hive slowly, reading the mood of the whole colony first.",
    "Divers surveyed the wreck at forty metres, photographing the hull before the light faded.",
]

# THE VOICE THE MODEL SPEAKS IN, which it cannot invent for itself.
#
# Voxtral TTS is voice-PROMPTED: the prompt carries a block of `[AUDIO]` placeholder tokens whose
# embeddings are replaced by a speaker's voice embedding, and the backbone conditions on those. Run
# without one the model has no voice to imitate and emits voice-shaped babble -- every stage
# numerically correct, the output unintelligible.
#
# The 20 presets ship in the checkpoint repo as `voice_embedding/<id>.pt` and were simply never
# fetched: a local snapshot holds only the four files the loader asked for, and reading absence off
# that snapshot is how this was recorded as "the repo does not ship voice prompts".
#
# Layout and ids are the tokenizer's own, verified against mistral_common's
# `encode_speech_request`, which is what vLLM-Omni calls:
#
#     [BOS] [BEGIN_AUDIO] [AUDIO]*N [NEXT_AUDIO_TEXT] <text> [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]
#
# N is the voice's own token count, published per voice in tekken.json's
# `audio.voice_num_audio_tokens`, and equals the row count of its embedding -- the two are asserted
# against each other rather than assumed.
# The preset the package speaks in by default; one of the 20 in tekken.json's voice list.
DEFAULT_VOICE = "casual_male"

_BEGIN_AUDIO_ID = 25
_AUDIO_ID = 24
_NEXT_AUDIO_TEXT_ID = 36
_REPEAT_AUDIO_TEXT_ID = 35


def load_voice_embedding(voice: str, model_id: str = HF_MODEL_ID):
    """The speaker's voice embedding, `[N, hidden]`, from the checkpoint repo."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, f"voice_embedding/{voice}.pt", revision=_revision(model_id))
    emb = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(emb) or emb.dim() != 2:
        raise ValueError(f"voice {voice!r}: expected a 2-D embedding, got {type(emb).__name__}")
    return emb.to(torch.float32)


def available_voices(model_id: str = HF_MODEL_ID) -> dict:
    """`{voice: n_audio_tokens}` as the tokenizer publishes it."""
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(model_id, "tekken.json", revision=_revision(model_id)), encoding="utf-8") as fh:
        meta = json.load(fh)
    return dict(((meta.get("audio") or {}).get("voice_num_audio_tokens") or {}))


def build_voice_prompt(texts, voice: str, model_id: str = HF_MODEL_ID):
    """`(input_ids [B, S], audio_mask [B, S] bool, voice_embedding [N, hidden])` for `texts`.

    Every row carries the same voice, so the placeholder block is the same width on each and the
    batch stays rectangular without padding.
    """
    tok = load_tokenizer(model_id)
    emb = load_voice_embedding(voice, model_id)
    declared = available_voices(model_id).get(voice)
    if declared is not None and int(declared) != int(emb.shape[0]):
        raise ValueError(
            f"voice {voice!r}: tekken.json declares {declared} audio tokens but the embedding has "
            f"{emb.shape[0]} rows"
        )
    n_audio = int(emb.shape[0])

    bos = tok.bos_id if hasattr(tok, "bos_id") else 1
    bodies = [tok.encode(text, bos=False) for text in texts]

    # NO PADDING. The prefill has no per-row padding mask, so a pad token is CONTENT to the model --
    # and a BOS run in the middle of the prompt measurably destroys the conditioning (see
    # SPEECH_TEXTS). A ragged batch is refused rather than silently degraded.
    widths = sorted({len(b) for b in bodies})
    if len(widths) != 1:
        raise ValueError(f"speech texts must tokenize to one common length for an unpadded batch; got widths {widths}")
    rows = [
        [bos, _BEGIN_AUDIO_ID]
        + [_AUDIO_ID] * n_audio
        + [_NEXT_AUDIO_TEXT_ID]
        + body
        + [_REPEAT_AUDIO_TEXT_ID, _BEGIN_AUDIO_ID]
        for body in bodies
    ]
    input_ids = torch.tensor(rows, dtype=torch.long)
    return input_ids, input_ids == _AUDIO_ID, emb


def build_batch_inputs(batch: int = DEFAULT_BATCH, seq_len: int = DEFAULT_SEQ_LEN, model_id: str = HF_MODEL_ID):
    """The REAL pipeline input: `batch` independent prompts, each exactly `seq_len` real tokens.

    Returns ``(input_ids [batch, seq_len] int64, texts)``. No padding: every prompt text is long
    enough to truncate to `seq_len`, so every row is genuine content and the causal mask is the
    plain lower-triangular one. The rows are asserted pairwise distinct -- a batch of 32 copies
    would make the PCC gate meaningless.
    """
    tok = load_tokenizer(model_id)
    if batch > len(PROMPT_TEXTS):
        raise ValueError(f"only {len(PROMPT_TEXTS)} distinct prompts are defined, asked for {batch}")
    rows, texts = [], []
    for text in PROMPT_TEXTS[:batch]:
        ids = tok.encode(text, bos=True)
        if len(ids) < seq_len:
            raise ValueError(f"prompt tokenizes to {len(ids)} < seq_len={seq_len}: {text[:48]!r}")
        ids = ids[:seq_len]
        rows.append(ids)
        texts.append(tok.decode(ids[1:]))
    input_ids = torch.tensor(rows, dtype=torch.long)
    if len({tuple(r) for r in rows}) != batch:
        raise AssertionError("batch rows are not pairwise distinct")
    return input_ids, texts


# --------------------------------------------------------------------------------------
# Decode horizon: each row runs until its semantic head emits end_audio (`resolve_max_frames` is only a cap)
# --------------------------------------------------------------------------------------


def begin_audio_token_id(model_id: str = HF_MODEL_ID) -> int:
    """`[BEGIN_AUDIO]` (25) -- the token that tells the backbone to start emitting audio frames.

    It lives under `multimodal.audio_model_args`, not directly under `multimodal` (which carries
    only `bos_token_id`), and `tekken.json` confirms rank 25 is `[BEGIN_AUDIO]`.
    """
    return int(load_params(model_id)["multimodal"]["audio_model_args"]["begin_audio_token_id"])


def audio_stop_token_id(hf_model) -> int:
    """`AudioSpecialTokens.end_audio`, read off the reference rather than written as a literal.

    This is the model's real stop signal for the TTS chain: the acoustic transformer's semantic
    head predicts it, and the reference's own `decode_one_frame` tests `semantic_code != this`
    to decide whether a frame is still speech.
    """
    return int(hf_model.acoustic_transformer._end_audio_token_id)


def audio_empty_token_id(hf_model) -> int:
    return int(hf_model.acoustic_transformer._empty_audio_token_id)


def n_audio_special_tokens(hf_model) -> int:
    """The offset between an emitted audio token and the codec's own code space.

    The acoustic transformer emits codes shifted up by the number of audio special tokens, and
    `MultiVocabEmbeddings` expects them in that shifted space; the codec's quantizer expects them
    unshifted. Subtracting this is the only conversion between the two.
    """
    from enum import Enum  # noqa: F401 - the enum lives in the reference loader's namespace

    at = hf_model.acoustic_transformer
    return int(max(at._end_audio_token_id, at._empty_audio_token_id) + 1)


def resolve_max_frames(hf_model) -> tuple:
    """The SAFETY CAP on the decode loop -- a backstop, never the horizon.

    The horizon is the model's own stop rule (`audio_stop_token_id`, per row). This only bounds a
    run that never emits it, and a correctness run that ends HERE is a failure, not a pass: the e2e
    test asserts the run ended on the stop rule. The cap is the tighter of the model's own context
    (`max_position_embeddings`) and the port's real ceiling, the codec's prebuilt ALiBi mask
    (2048 rows / 8x upsampling = 256 frames = 20.5 s of audio).
    """
    env = os.environ.get("VOXTRAL_MAX_FRAMES")
    if env:
        return int(env), "env VOXTRAL_MAX_FRAMES"
    frame_rate = float(hf_model.audio_tokenizer.frame_rate)
    context = int(getattr(hf_model.config, "max_position_embeddings", CODEC_MAX_FRAMES) or CODEC_MAX_FRAMES)
    frames = min(CODEC_MAX_FRAMES, context)
    return frames, (
        f"safety cap = min(codec ALiBi ceiling {CODEC_MAX_FRAMES}, max_position_embeddings {context}) = "
        f"{frames} frames ({frames / frame_rate:.1f} s at {frame_rate} Hz)"
    )


# The codec (`tt/modules/voxtral_t_t_s_audio_tokenizer.py`) prebuilds its sliding-window ALiBi mask
# to 2048 rows and the decoder upsamples 8x, so the widest input it can take is 2048 / 8 frames.
# Raising this means raising `_MASK_MAX_SEQ` there -- the mask cannot be rebuilt inside the forward.
CODEC_MAX_FRAMES = 2048 // 8


def eos_token_id(hf_model):
    gc = getattr(hf_model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if eos is None:
        eos = getattr(hf_model.config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        return int(eos[0])
    return None if eos is None else int(eos)


# --------------------------------------------------------------------------------------
# PCC
# --------------------------------------------------------------------------------------


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors, flattened, in float64."""
    x = a.detach().to(torch.float64).flatten()
    y = b.detach().to(torch.float64).flatten()
    if x.numel() != y.numel():
        raise ValueError(f"shape mismatch for PCC: {tuple(a.shape)} vs {tuple(b.shape)}")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return 1.0 if torch.equal(x, y) else 0.0
    return float((x * y).sum() / denom)


# --------------------------------------------------------------------------------------
# Golden caching and host threads
# --------------------------------------------------------------------------------------


def use_all_cpu_threads() -> int:
    """Give torch every core for the HF golden, which is the slow side of the gate.

    The golden is a 3.4 B model on CPU; torch defaults to half the cores here.
    """
    import torch as _torch

    _torch.set_num_threads(os.cpu_count() or 1)
    return _torch.get_num_threads()


def golden_cache_path(key: str) -> str:
    # A per-user cache directory, not a fixed world-writable one under /tmp: these files are loaded back.
    default = os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
        "voxtral_4b_tts_2603_golden",
    )
    root = os.environ.get("VOXTRAL_GOLDEN_CACHE", default)
    os.makedirs(root, mode=0o700, exist_ok=True)
    return os.path.join(root, f"{key}.pt")


def golden_key(**parts) -> str:
    """A cache key over everything the golden depends on.

    The reference loader's CONTRACT number is part of it: a loader that starts covering more of
    the checkpoint must invalidate every cached golden rather than be compared against a stale
    one (a cached golden that outlives its inputs is the failure mode that had a PCC gate
    silently testing another checkout for a whole run).
    """
    import hashlib

    loader = _reference_loader_module()
    parts["loader_contract"] = getattr(loader, "REFERENCE_LOADER_CONTRACT", 0)
    # The weights and the reference CODE are inputs too: a different checkpoint revision, or a fix on
    # the reference side alone (which leaves the TT codes and hiddens bit-identical), must not be served
    # a golden computed before it.
    parts["hf_revision"] = HF_REVISION
    parts["reference_source"] = _reference_source_hash()
    blob = json.dumps({k: _hashable(v) for k, v in sorted(parts.items())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


@lru_cache(maxsize=1)
def _reference_source_hash() -> str:
    import hashlib

    ref_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference")
    digest = hashlib.sha256()
    for name in ("golden.py", "reference_loader.py"):
        with open(os.path.join(ref_dir, name), "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()[:24]


def _hashable(value):
    if isinstance(value, torch.Tensor):
        import hashlib

        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:32]
    return value


def cached_golden(key: str, compute):
    """Return `compute()`, memoised on disk under `key`.

    The gate is iterated many times against an unchanging reference, and the golden costs minutes.
    Nothing device-side is cached -- only the torch reference output.
    """
    path = golden_cache_path(key)
    if os.path.exists(path) and not os.environ.get("VOXTRAL_GOLDEN_REFRESH"):
        return torch.load(path, weights_only=True)
    value = compute()
    # Written to a temporary name and moved into place, so an interrupted save (Ctrl-C, a timeout) or a
    # second process never leaves a truncated file that every later run would fail to load.
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(value, tmp)
    os.replace(tmp, path)
    return value
