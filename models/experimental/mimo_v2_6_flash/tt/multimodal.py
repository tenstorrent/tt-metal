# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal input (image, video, audio) for the MiMo-V2.6-Flash prefill: host-side bookkeeping.

The LLM is unchanged: an item's embeddings replace the embeddings of its placeholder tokens (doc/multimodal_plan.md).
This module turns a prefill row's items into the host rows of one chunk (``chunk_embeds``), which
TtMiMoModel.prefill_enqueue(embeds=, embed_rows=) merges into the chunk on device:

* ``MMItem``: one item of the plugin contract (``prefill_forward``'s ``mm_items``): modality, vLLM identifier (its mm
  hash), PlaceholderRange ``offset`` / ``length`` / ``is_embed`` in the full prompt, and the processor fields
  (``data``; None when vLLM stripped them because the item lies in an APC-cached prefix).
* ``EncoderBackend``: what turns an item's processor fields into its embeddings [num_embeds, hidden] (host, bf16).
  ``CompositeBackend`` serves images / videos with the device ViT (tt/vision/) and audio on the host (tt/audio/);
  ``HFHostBackend`` (debug only) runs the checkpoint's own HF modules on the host.
* ``EmbeddingCache``: an LRU of item embeddings keyed by identifier, so a later chunk, a chunk start moved back into an
  earlier chunk's item, or another request with the same item does not encode it again.
* ``chunk_embeds``: scatter of the items' embeddings into the rows of a chunk [start, end) of the prompt: an item may
  start before the chunk (only its rows in the chunk are used), end before it (skipped), carry an ``is_embed`` mask
  (video: timestamp and frame-separator tokens inside the range keep their token embeddings).

Errors in a request's multimodal input raise ``MMInputError``; the vLLM adapter turns them into a logged error and
the placeholder tokens' own embeddings for that item, never an exception out of prefill (which kills the engine).
"""

import collections
import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from loguru import logger

MODALITIES = ("image", "video", "audio")

# Processor fields an item needs to be encoded (vLLM's MiMoV2OmniMultiModalProcessor._get_mm_fields_config).
REQUIRED_FIELDS = {
    "image": ("pixel_values", "image_grid_thw"),
    "video": ("pixel_values_videos", "video_grid_thw"),
    # the mel (or its exact fp32 bits), its RVQ codes, or the LLM embeddings the front end made (tt/audio/processor.py)
    "audio": ("audio_features|audio_mel_bits|audio_codes|audio_embeds",),
}


class MMInputError(ValueError):
    """A request's multimodal input cannot be served (malformed, unsupported or missing data)."""


@dataclass
class MMItem:
    modality: str
    identifier: Optional[str]
    offset: int
    length: int
    is_embed: Optional[torch.Tensor] = None  # bool [length]; None: every row of the range is an embedding
    data: Optional[dict] = None

    @property
    def num_embeds(self):
        return self.length if self.is_embed is None else int(self.is_embed.sum())

    @property
    def end(self):
        return self.offset + self.length

    def embed_positions(self):
        """Prompt positions of the item's embedding rows, in order (int64 [num_embeds])."""
        positions = torch.arange(self.offset, self.end, dtype=torch.int64)
        return positions if self.is_embed is None else positions[self.is_embed]


def _field(data, name):
    """A processor field's tensor: vLLM hands MultiModalFieldElem-like objects (``.data``) or plain tensors."""
    value = data.get(name)
    return getattr(value, "data", value)


def parse_item(entry, modalities=MODALITIES):
    """One ``mm_items`` entry (a dict of the plugin contract, or an MMItem) -> MMItem; raises MMInputError."""
    if isinstance(entry, MMItem):
        item = entry
    else:
        if not isinstance(entry, dict):
            raise MMInputError(f"a multimodal item must be a dict, got {type(entry).__name__}")
        try:
            item = MMItem(
                modality=str(entry["modality"]),
                identifier=None if entry.get("identifier") is None else str(entry["identifier"]),
                offset=int(entry["offset"]),
                length=int(entry["length"]),
                is_embed=entry.get("is_embed"),
                data=entry.get("data"),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise MMInputError(f"malformed multimodal item {sorted(entry)}: {e}") from e
    if item.modality not in modalities:
        raise MMInputError(f"modality {item.modality!r} is not served (served: {', '.join(modalities)})")
    if item.offset < 0 or item.length <= 0:
        raise MMInputError(f"{item.modality} item at offset {item.offset} with length {item.length}")
    if item.is_embed is not None:
        is_embed = torch.as_tensor(item.is_embed).to(torch.bool).reshape(-1)
        if is_embed.numel() != item.length:
            raise MMInputError(f"{item.modality} item: is_embed has {is_embed.numel()} rows, length {item.length}")
        item.is_embed = None if bool(is_embed.all()) else is_embed
    if item.data is not None:
        missing = [f for f in REQUIRED_FIELDS[item.modality] if all(_field(item.data, g) is None for g in f.split("|"))]
        if missing:
            raise MMInputError(f"{item.modality} item {item.identifier}: missing processor fields {missing}")
    return item


def parse_items(entries, modalities=MODALITIES):
    """A row's ``mm_items`` entry (None or a list) -> [MMItem] sorted by offset, non-overlapping."""
    if not entries:
        return []
    items = sorted((parse_item(e, modalities) for e in entries), key=lambda it: it.offset)
    for a, b in zip(items, items[1:]):
        if b.offset < a.end:
            raise MMInputError(f"multimodal items overlap: [{a.offset}, {a.end}) and [{b.offset}, {b.end})")
    return items


def items_overlapping(items, start, end):
    """The items with an embedding row in prompt positions [start, end)."""
    out = []
    for item in items:
        if item.end <= start or item.offset >= end:
            continue
        pos = item.embed_positions()
        if bool(((pos >= start) & (pos < end)).any()):
            out.append(item)
    return out


def chunk_embeds(items, start, end, embeddings_of, hidden):
    """Host rows of the prompt chunk [start, end): (embeds bf16 [end - start, hidden], rows bool [end - start]) for
    TtMiMoModel.prefill_enqueue(embeds=, embed_rows=), or (None, None) when no item has a row in it.
    ``embeddings_of(item)`` -> the item's embeddings [num_embeds, hidden], or None to leave its rows to their tokens
    (an item that could not be encoded)."""
    n = end - start
    embeds = rows = None
    for item in items_overlapping(items, start, end):
        emb = embeddings_of(item)
        if emb is None:
            continue
        if tuple(emb.shape) != (item.num_embeds, hidden):
            raise MMInputError(
                f"{item.modality} item {item.identifier}: embeddings {tuple(emb.shape)}, expected "
                f"({item.num_embeds}, {hidden})"
            )
        pos = item.embed_positions()
        keep = (pos >= start) & (pos < end)
        if embeds is None:
            embeds = torch.zeros(n, hidden, dtype=torch.bfloat16)
            rows = torch.zeros(n, dtype=torch.bool)
        embeds[pos[keep] - start] = emb[keep].to(torch.bfloat16)
        rows[pos[keep] - start] = True
    return embeds, rows


def placeholder_rows(tokens, token_id):
    """Runs of ``token_id`` in ``tokens`` (1D): [(offset, length)] in order."""
    hits = (torch.as_tensor(tokens).reshape(-1) == token_id).nonzero().flatten().tolist()
    runs = []
    for p in hits:
        if runs and runs[-1][0] + runs[-1][1] == p:
            runs[-1][1] += 1
        else:
            runs.append([p, 1])
    return [tuple(r) for r in runs]


def content_id(*tensors):
    """A stable identifier of an item's processor tensors (the image-only fallback has no vLLM identifier)."""
    h = hashlib.sha1()
    for t in tensors:
        t = torch.as_tensor(t).detach().cpu().contiguous()
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return "sha1:" + h.hexdigest()


def items_from_image_lists(tokens, end, pixel_values, grids, image_token_id, merge_size=2):
    """The plugin's image-only kwargs of one row (``pixel_values`` / ``image_grid_thw``: per image a tensor or None)
    -> [MMItem]. The prompt's k-th run of ``image_token_id`` in tokens[:end] is image k; images whose run starts at or
    past ``end`` are not in the prompt yet and are left out. A run that ends at ``end`` may be cut there."""
    if grids is None:
        return []
    pixel_values = list(pixel_values) if pixel_values is not None else [None] * len(grids)
    if len(pixel_values) != len(grids):
        raise MMInputError(f"{len(pixel_values)} pixel entries for {len(grids)} image grids")
    runs = placeholder_rows(torch.as_tensor(tokens).reshape(-1)[:end], image_token_id)
    items = []
    for k, (pv, grid) in enumerate(zip(pixel_values, grids)):
        if k >= len(runs):
            break
        if grid is None:
            raise MMInputError(f"image {k} has no image_grid_thw")
        grid = torch.as_tensor(grid).reshape(-1)
        if grid.numel() != 3:
            raise MMInputError(f"image {k}: image_grid_thw {tuple(grid.shape)}, expected 3 values")
        length = int(grid.prod()) // merge_size**2
        offset, run = runs[k]
        if run != length and not (run < length and offset + run == end):
            raise MMInputError(
                f"image {k}: {run} placeholder tokens at {offset}, its grid {grid.tolist()} has {length}"
            )
        data = None if pv is None else {"pixel_values": pv, "image_grid_thw": grid}
        identifier = None if pv is None else content_id(pv, grid)
        items.append(MMItem("image", identifier, offset, length, None, data))
    return items


class EncoderBackend(Protocol):
    """Item embeddings on the host (bf16), one row per placeholder token the item's embedding rows replace."""

    def embed_image(self, pixel_values, grid_thw) -> torch.Tensor:  # [sum(t * h * w) / 4, hidden]
        ...

    def embed_video(self, pixel_values_videos, grid_thw) -> torch.Tensor:  # [sum(t * h * w) / 4, hidden]
        ...

    def embed_audio(self, mel) -> torch.Tensor:  # [n_tokens, hidden]
        ...


def mel_from_bits(bits, what):
    """An item's fp32 mel sent as its int32 bits (``audio_mel_bits`` / ``va_audio_mel_bits``: vLLM casts floating
    processor outputs to the model dtype, integers never) -> float32 [T, 128], bit for bit."""
    bits = torch.as_tensor(bits)
    if bits.dtype != torch.int32:
        raise MMInputError(f"{what}: mel bits must be int32, got {bits.dtype}")
    return bits.contiguous().view(torch.float32)


class TrackCodes:
    """RVQ codes of video audio tracks the engine tokenized (``va_audio_mel_bits``), by item identifier, least
    recently used first out past MIMO_MM_TRACK_CODES_MB (default 64; int16, 1.6 MB for a 27 min track). A video whose
    embeddings left the EmbeddingCache (its next question, or a chunk start moved back into it) re-runs its ViT
    passes but not its tokenizer encoder, as with codes from the front end (MIMO_AUDIO_TOKENIZER=frontend)."""

    def __init__(self, budget_bytes=None):
        if budget_bytes is None:
            budget_bytes = int(float(os.environ.get("MIMO_MM_TRACK_CODES_MB", "64")) * 2**20)
        self.budget, self.used = budget_bytes, 0
        self.entries = collections.OrderedDict()
        self.lock = threading.Lock()

    def get(self, identifier):
        with self.lock:
            codes = self.entries.get(identifier)
            if codes is not None:
                self.entries.move_to_end(identifier)
            return codes

    def put(self, identifier, codes):
        codes = codes.to(torch.int16)  # codebooks of at most 1024 entries
        size = codes.numel() * codes.element_size()
        with self.lock:
            old = self.entries.pop(identifier, None)
            if old is not None:
                self.used -= old.numel() * old.element_size()
            if size > self.budget:
                return
            self.entries[identifier] = codes
            self.used += size
            while self.used > self.budget:
                _, out = self.entries.popitem(last=False)
                self.used -= out.numel() * out.element_size()


TRACK_CODES = TrackCodes()


def video_audio_codes(backend, item):
    """A video item's audio track as RVQ codes [n_codes, 20]: ``va_audio_codes`` from the front end
    (MIMO_AUDIO_TOKENIZER=frontend), else its fp32 mel (``va_audio_mel_bits``, the default) through
    ``backend.audio_codes`` (the row's device tokenizer encoder and the host RVQ), remembered in TRACK_CODES."""
    codes = _field(item.data, "va_audio_codes")
    if codes is not None:
        return torch.as_tensor(codes)
    bits = _field(item.data, "va_audio_mel_bits")
    if bits is None:
        raise MMInputError(f"video item {item.identifier}: an audio track without va_audio_codes / va_audio_mel_bits")
    if item.identifier is not None:
        codes = TRACK_CODES.get(item.identifier)
        if codes is not None:
            return codes
    if not hasattr(backend, "audio_codes"):
        raise MMInputError("this backend has no audio tokenizer encoder (audio_codes)")
    codes = backend.audio_codes(mel_from_bits(bits, f"video item {item.identifier}"))
    if item.identifier is not None:
        TRACK_CODES.put(item.identifier, codes)
    return codes


def interleave_video_audio(backend, item, frames):
    """A video with its audio track (tt/video_audio.py): the frames' rows ``frames`` [T * n_per_grid, hidden] and the
    audio's (its codes, ``video_audio_codes``, through ``backend.embed_audio_codes``) in the order of the item's
    embedding rows, as vLLM 0.31's MiMoV2OmniForCausalLM.embed_multimodal lays them out: per group of T / n_segs grids,
    the group's frames, then the group's audio segment (``video_audio_seg_lens``), the segments taken from the start of
    the audio in turn. Where the codes come from (the front end, or the engine's tokenizer from the mel) changes
    nothing here.
    """
    data = item.data
    grid = torch.as_tensor(_field(data, "video_grid_thw")).reshape(-1, 3)
    if grid.shape[0] != 1:
        raise MMInputError(f"video item {item.identifier}: {grid.shape[0]} grids with an audio track")
    n_segs = int(torch.as_tensor(_field(data, "video_audio_n_segs")).reshape(-1)[0])
    seg_lens = [int(x) for x in torch.as_tensor(_field(data, "video_audio_seg_lens")).reshape(-1)[:n_segs]]
    if not hasattr(backend, "embed_audio_codes"):
        raise MMInputError("this backend has no audio patch encoder (embed_audio_codes)")
    audio = backend.embed_audio_codes(video_audio_codes(backend, item))
    t = int(grid[0, 0])
    if n_segs <= 0 or t % n_segs or frames.shape[0] % t:
        raise MMInputError(f"video item {item.identifier}: {t} grids in {n_segs} audio groups")
    per_group = frames.shape[0] // t * (t // n_segs)
    if sum(seg_lens) > audio.shape[0]:
        raise MMInputError(
            f"video item {item.identifier}: audio segments of {sum(seg_lens)} tokens, the track has {audio.shape[0]}"
        )
    rows, a = [], 0
    for g, n in enumerate(seg_lens):
        rows.append(frames[g * per_group : (g + 1) * per_group])
        rows.append(audio[a : a + n].to(frames.dtype))
        a += n
    return torch.cat(rows)


def encode_item(backend, item):
    """An item's embeddings [num_embeds, hidden] bf16 from its processor fields (raises MMInputError)."""
    if item.data is None:
        raise MMInputError(f"{item.modality} item {item.identifier} has no data (stripped) and is not cached")
    if item.modality == "image":
        out = backend.embed_image(
            _field(item.data, "pixel_values"), torch.as_tensor(_field(item.data, "image_grid_thw")).reshape(-1, 3)
        )
    elif item.modality == "video":
        out = backend.embed_video(
            _field(item.data, "pixel_values_videos"),
            torch.as_tensor(_field(item.data, "video_grid_thw")).reshape(-1, 3),
        )
        n_segs = _field(item.data, "video_audio_n_segs")
        if n_segs is not None and int(torch.as_tensor(n_segs).sum()):
            out = interleave_video_audio(backend, item, out)
    elif _field(item.data, "audio_embeds") is not None:
        out = torch.as_tensor(_field(item.data, "audio_embeds"))  # encoded in the front end: upload only
        if out.dim() == 3 and out.shape[0] == 1:
            out = out[0]
        if not out.is_floating_point() or not bool(torch.isfinite(out).all()):
            raise MMInputError(f"audio item {item.identifier}: audio_embeds must be finite floats")
    elif _field(item.data, "audio_codes") is not None:
        out = backend.embed_audio_codes(_field(item.data, "audio_codes"))
    elif _field(item.data, "audio_mel_bits") is not None:  # the fp32 mel as int32 bits (MIMO_AUDIO_TOKENIZER=device)
        out = backend.embed_audio(mel_from_bits(_field(item.data, "audio_mel_bits"), f"audio item {item.identifier}"))
    else:
        out = backend.embed_audio(_field(item.data, "audio_features"))
    if out.dim() != 2 or out.shape[0] != item.num_embeds:
        raise MMInputError(
            f"{item.modality} item {item.identifier}: the encoder gave {tuple(out.shape)} for {item.num_embeds} "
            "placeholder rows"
        )
    return out.to(torch.bfloat16)


class EmbeddingCache:
    """Item embeddings by identifier, least recently used first out past ``budget_bytes`` (MIMO_MM_CACHE_MB, default
    4096). Thread-safe (the plugin may call the adapter from several threads)."""

    def __init__(self, budget_bytes=None):
        if budget_bytes is None:
            budget_bytes = int(float(os.environ.get("MIMO_MM_CACHE_MB", "4096")) * 2**20)
        self.budget = budget_bytes
        self.used = 0
        self.entries = collections.OrderedDict()
        self.hits = self.misses = 0
        self.lock = threading.Lock()

    def get(self, identifier):
        if identifier is None:
            return None
        with self.lock:
            emb = self.entries.get(identifier)
            if emb is None:
                self.misses += 1
                return None
            self.entries.move_to_end(identifier)
            self.hits += 1
            return emb

    def put(self, identifier, emb):
        if identifier is None:
            return
        size = emb.numel() * emb.element_size()
        with self.lock:
            old = self.entries.pop(identifier, None)
            if old is not None:
                self.used -= old.numel() * old.element_size()
            if size > self.budget:
                return
            self.entries[identifier] = emb
            self.used += size
            while self.used > self.budget:
                _, out = self.entries.popitem(last=False)
                self.used -= out.numel() * out.element_size()

    def discard(self, identifiers):
        """Drop these identifiers' entries (a request that is done with them)."""
        with self.lock:
            for identifier in identifiers:
                out = self.entries.pop(identifier, None)
                if out is not None:
                    self.used -= out.numel() * out.element_size()

    def __contains__(self, identifier):
        with self.lock:
            return identifier in self.entries

    def __len__(self):
        return len(self.entries)


class ItemEncoder:
    """Embeddings of a row's items through the cache, encoding a miss with ``backend`` (what chunk_embeds'
    ``embeddings_of`` takes). Last resort only: an item that cannot be encoded logs an ERROR (every time, with its
    identifier and the running count ``failures``) and keeps its placeholder tokens' embeddings (None), so the engine
    survives, but that request's answer is wrong. Validate requests before they reach prefill wherever possible."""

    def __init__(self, backend, cache=None):
        self.backend = backend
        self.cache = cache if cache is not None else EmbeddingCache()
        self.failed = set()
        self.failures = 0

    def __call__(self, item, context=""):
        emb = self.cache.get(item.identifier)
        if emb is not None and emb.shape[0] == item.num_embeds:
            return emb
        try:
            emb = encode_item(self.backend, item)
        except Exception as e:  # noqa: BLE001 - any encoder failure must not kill the engine
            self.failed.add(item.identifier or id(item))
            self.failures += 1
            logger.error(
                f"MiMo multimodal: ENCODER FAILURE #{self.failures}{context}: {item.modality} item "
                f"{item.identifier} at prompt [{item.offset}, {item.end}) was not encoded; its placeholder tokens "
                f"stand in, so this request's answer is degraded: {type(e).__name__}: {e}"
            )
            return None
        self.cache.put(item.identifier, emb)
        return emb


class CompositeBackend:
    """An EncoderBackend from one backend per modality family: ``vision`` (embed_image / embed_video; the serving
    default is tt/vision/encoder.MiMoVisionEncoder on the request's row) and ``audio`` (embed_audio; the serving
    default is tt/audio/host_backend.HostAudioBackend). A missing one raises MMInputError for its modalities."""

    def __init__(self, vision=None, audio=None):
        self.vision, self.audio = vision, audio

    def embed_image(self, pixel_values, grid_thw):
        if self.vision is None:
            raise MMInputError("this backend serves no images")
        return self.vision.embed_image(pixel_values, grid_thw)

    def embed_video(self, pixel_values_videos, grid_thw):
        if self.vision is None:
            raise MMInputError("this backend serves no videos")
        return self.vision.embed_video(pixel_values_videos, grid_thw)

    def embed_audio(self, mel):
        if self.audio is None:
            raise MMInputError("this backend serves no audio")
        return self.audio.embed_audio(mel)

    def __repr__(self):
        return f"CompositeBackend(vision={type(self.vision).__name__}, audio={type(self.audio).__name__})"


# --------------------------------------------------------------------------------------------------------------------
# Debug only (MIMO_MM_ENCODER=hf_host): the checkpoint's own HF modules (modeling_mimo_v2.py) on the host. The serving
# encoders are the device ViT (tt/vision/) and the host audio path (tt/audio/); the references are reference/vision.py
# and reference/audio.py.
# --------------------------------------------------------------------------------------------------------------------


def _hf_module(model_path):
    """The checkpoint's modeling_mimo_v2 module (remote code, loaded the way transformers loads it)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("modeling_mimo_v2.MiMoVisionTransformer", model_path)
    import sys

    return sys.modules[cls.__module__]


def _load_prefixed(module, path, prefix):
    from safetensors import safe_open

    state = {}
    with safe_open(path, "pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = f.get_tensor(key)
    missing, unexpected = module.load_state_dict(state, strict=False)
    return missing, unexpected


class HFHostBackend:
    """Debug only: image / video / audio embeddings from the checkpoint's HF modules on the host (torch, ``dtype``
    MIMO_MM_HOST_DTYPE: float32 by default). One instance per process (``shared``)."""

    _shared = {}
    _lock = threading.Lock()

    @classmethod
    def shared(cls, model_path):
        with cls._lock:
            if model_path not in cls._shared:
                cls._shared[model_path] = cls(model_path)
            return cls._shared[model_path]

    def __init__(self, model_path, dtype=None, modalities=MODALITIES):
        import json

        if dtype is None:
            dtype = getattr(torch, os.environ.get("MIMO_MM_HOST_DTYPE", "float32"))
        self.model_path, self.dtype = model_path, dtype
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
        self.hidden = config["hidden_size"]
        hf = _hf_module(model_path)
        self.hf = hf
        shard = os.path.join(model_path, "model_pp0_ep0_shard0.safetensors")
        self.visual = self.audio = None
        self.lock = threading.Lock()  # torch modules are not re-entrant across the plugin's threads
        if "image" in modalities or "video" in modalities:
            self.visual = hf.MiMoVisionTransformer(hf._as_namespace(config["vision_config"]))
            merger = self.visual.merger
            # The checkpoint's merger has no biases and one norm weight. modeling_mimo_v2.py builds a LayerNorm and
            # biased Linears (whose missing biases would keep their random init); vLLM's mimo_v2_omni.py an RMSNorm
            # and bias-free Linears. Follow vLLM unless MIMO_MM_MERGER_NORM=layer.
            if os.environ.get("MIMO_MM_MERGER_NORM", "rms") == "rms":
                merger.ln_q = torch.nn.RMSNorm(merger.ln_q.normalized_shape, eps=1e-6)
            for linear in (merger.mlp[0], merger.mlp[2]):
                linear.bias = None
            missing, _ = _load_prefixed(self.visual, shard, "visual.")
            missing = [k for k in missing if k != "merger.ln_q.bias"]  # LayerNorm without a bias: zeros
            if missing:
                raise RuntimeError(f"ViT weights missing from {shard}: {missing[:5]}")
            self.visual.to(dtype).eval()
            self.merge = config["vision_config"].get("spatial_merge_size", 2)
        if "audio" in modalities and config.get("audio_config"):
            audio_cfg = hf._as_namespace(config["audio_config"])
            self.speech_embeddings = hf._build_speech_embeddings(audio_cfg)
            _load_prefixed(self.speech_embeddings, shard, "speech_embeddings.")
            self.audio = hf.MiMoAudioEncoder(audio_cfg)
            missing, _ = _load_prefixed(self.audio, shard, "audio_encoder.")
            missing = [k for k in missing if "embed_tokens" not in k]  # unused by inputs_embeds
            if missing:
                raise RuntimeError(f"audio encoder weights missing from {shard}: {missing[:5]}")
            tok_dir = os.path.join(model_path, "audio_tokenizer")
            with open(os.path.join(tok_dir, "config.json")) as f:
                tok_cfg = json.load(f)
            self.audio_tokenizer = hf.MiMoAudioTokenizer(hf.MiMoAudioTokenizerConfig(**tok_cfg))
            _load_prefixed(self.audio_tokenizer, os.path.join(tok_dir, "model.safetensors"), "")
            for m in (self.speech_embeddings, self.audio, self.audio_tokenizer):
                m.to(dtype).eval()
        logger.info(
            f"MiMo multimodal: HF host encoders (debug backend) loaded ({dtype}; vision {self.visual is not None}, audio "
            f"{self.audio is not None})"
        )

    @torch.no_grad()
    def embed_image(self, pixel_values, grid_thw):
        if self.visual is None:
            raise MMInputError("this backend serves no images")
        grid_thw = torch.as_tensor(grid_thw).reshape(-1, 3).to(torch.int64)
        with self.lock:
            out = self.visual(torch.as_tensor(pixel_values).to(self.dtype), grid_thw)
        return out.to(torch.bfloat16)

    embed_video = embed_image

    @torch.no_grad()
    def embed_audio(self, mel):
        if self.audio is None:
            raise MMInputError("this backend serves no audio")
        if isinstance(mel, (list, tuple)):
            mels = [torch.as_tensor(m) for m in mel]
        else:
            mel = torch.as_tensor(mel)
            mels = list(mel) if mel.dim() == 3 else [mel]
        with self.lock:
            out = self.audio.get_audio_feature(
                [m.to(self.dtype) for m in mels], self.speech_embeddings, self.audio_tokenizer.encoder
            )
        return out.to(torch.bfloat16)
