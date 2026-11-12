# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A video with its audio track, as the checkpoint's processor interleaves them (front end of the TT path).

The checkpoint's processor (``MiMoVLProcessor``, ``video_audio`` items) cuts a video's audio into one segment per group
of frames (``video_audio_interleave_length`` 0: one group per temporal grid, i.e. two frames) at 6.25 tokens a second,
by the frames' times, and writes ``<|mimo_video_start|> (timestamp <|vision_start|> pads <|vision_end|>
<|mimo_audio_start|> audio pads <|mimo_audio_end|>)* <|mimo_video_end|>``; vLLM 0.31's ``get_video_replacement`` writes
the same from the item's ``video_audio_n_segs`` / ``video_audio_seg_lens``. vLLM's API has no way to ask for it (its
``use_audio_in_video`` adds the audio as a separate item and placeholder), so here:

* ``VideoAudioMixin.apply``: with MIMO_VIDEO_AUDIO=1 (the default when audio is served, MIMO_MM_MODALITIES; a request's
  / the server's ``mm_processor_kwargs`` ``{"mimo_video_audio": true|false}`` overrides it) each video's audio track is decoded from the video's own bytes (vLLM's ``load_audio``),
  resampled to 24 kHz mono (vLLM's resampler, as its parser does for audio items) and kept in the video's loader
  metadata (``mimo_audio``), which goes into the item's hash and survives vLLM's processor cache with the frames
  (tt/video_input.py). A video without an audio track (or whose track fails to decode) stays a plain video (logged).
* ``_get_hf_mm_inputs``: a call's videos go to the checkpoint's processor as ``video_audio`` items when every one of
  them has its audio (the processor lists plain videos before video_audio ones, so a mix would reorder the items).
* ``_apply_hf_processor_main``: each item's mel is recomputed in fp32 (vLLM casts the processor's to bf16) and, as
  tt/audio/processor.py does for audio items, follows MIMO_AUDIO_TOKENIZER: with ``device`` (the default) the item
  carries the whole track's fp32 mel as its int32 bits, ``va_audio_mel_bits`` [T, 128] (a video field; 51 KB a
  second), and the engine tokenizes it on the request's row (the device tokenizer encoder, the host RVQ, then the
  patch encoder; tt/multimodal.py ``video_audio_codes``); with ``frontend`` the fp32 host tokenizer encoder turns it
  into RVQ codes here, ``va_audio_codes`` int32 [n_codes, 20]. Either replaces upstream's ``va_audio_features`` (a
  field of a "va_audio" modality that no placeholder belongs to). The engine (tt/multimodal.py ``encode_item``)
  embeds frames and codes and interleaves them per group, as vLLM's ``embed_multimodal`` does. The front-end
  tokenizer took 56-204 s of a 10-40 min video in vLLM's single multimodal worker (75-84 % of the front end).

The timestamps the segments are cut by are the frames' real times (tt/video_input.py, MIMO_VIDEO_TIMESTAMPS=loader);
with upstream's 1 fps labels the audio would be cut at the wrong places.
"""

import collections
import hashlib
import os
import threading
from io import BytesIO

import numpy as np
import torch
from loguru import logger


def _default_video_audio():
    """On (Xiaomi's API takes a video's audio track: +6.25 tokens a second) when the model serves audio
    (MIMO_MM_MODALITIES), whose tokenizer and patch encoder the track needs. Video-MME short docs 0-59, 2 fps, 280 tokens
    a frame pair: 70.0 / 68.3 % with the track, 63.3 / 65.0 / 65.0 % without (Xiaomi's API: 75.0 %)."""
    modalities = os.environ.get("MIMO_MM_MODALITIES", "image,video,audio").split(",")
    return "1" if "audio" in modalities else "0"


VIDEO_AUDIO = os.environ.get("MIMO_VIDEO_AUDIO", _default_video_audio()) == "1"
AUDIO_KEY = "mimo_audio"  # 24 kHz mono float32 waveform of the video's audio track, in its loader metadata
SAMPLE_RATE = 24000
# The request / server mm_processor_kwargs key (not ``video_audio``: vLLM passes the kwargs on to the checkpoint's
# processor, whose ``video_audio`` argument holds the items).
REQUEST_KEY = "mimo_video_audio"


def wanted(hf_processor_mm_kwargs):
    """Whether a request's videos take their audio: its / the server's mm_processor_kwargs ``mimo_video_audio``, else
    MIMO_VIDEO_AUDIO."""
    value = (hf_processor_mm_kwargs or {}).get(REQUEST_KEY)
    return VIDEO_AUDIO if value is None else bool(value)


_TRACKS = collections.OrderedDict()  # sha256 of a video file -> its decoded track (or None), least recent first
_TRACKS_LOCK = threading.Lock()
TRACK_CACHE_BYTES = int(float(os.environ.get("MIMO_VIDEO_AUDIO_CACHE_MB", "2048")) * 2**20)


def cached_audio_track(data):
    """decode_audio_track, remembered for the last few files (MIMO_VIDEO_AUDIO_CACHE_MB, default 2048): the second and
    third question on a video skip the decode (vLLM's processor cache skips the rest)."""
    key = hashlib.sha256(data).hexdigest()
    with _TRACKS_LOCK:
        if key in _TRACKS:
            _TRACKS.move_to_end(key)
            return _TRACKS[key]
    wave = decode_audio_track(data)
    with _TRACKS_LOCK:
        _TRACKS[key] = wave
        while sum(w.nbytes for w in _TRACKS.values() if w is not None) > TRACK_CACHE_BYTES and len(_TRACKS) > 1:
            _TRACKS.popitem(last=False)
    return wave


def decode_audio_track(data):
    """A video file's audio track -> float32 [samples] mono at 24 kHz, or None when it has none."""
    from vllm.multimodal.audio import AudioResampler
    from vllm.multimodal.media.audio import load_audio

    try:
        wave, sr = load_audio(BytesIO(data), sr=None)
    except Exception as e:  # noqa: BLE001 - no track, or a codec the backends do not read
        logger.warning(f"MiMo video audio: no audio track decoded ({type(e).__name__}: {e}); the video goes alone")
        return None
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 2:
        wave = wave.mean(axis=0)
    if wave.size == 0:
        return None
    return np.asarray(AudioResampler(SAMPLE_RATE).resample(wave, orig_sr=sr), dtype=np.float32)


TOKENS_PER_SECOND = 6.25  # audio tokens (25 Hz codes in groups of 4)
SAMPLES_PER_TOKEN = int(SAMPLE_RATE / TOKENS_PER_SECOND)


def cover_frames(wave, metadata, num_frames):
    """``wave``, zero-padded when it ends before the video's last grid starts. The checkpoint's processor cuts a
    segment per grid, from the grid's time to the next one's (the last one to the last frame plus one interval, cut at
    the track's end), and asserts every segment is non-empty: a track that ends before the last grid's time fails the
    request (seen on Video-MME clips). The missing tail becomes silence, up to two tokens past that time: a track that
    only ends inside the last grid is left as it is (its last segment is shorter, as Xiaomi's API's token counts show).
    """
    from .video_input import VIDEO_TIMESTAMPS, loader_timestamps, upstream_timestamps

    times = loader_timestamps(metadata, num_frames) if VIDEO_TIMESTAMPS == "loader" else None
    if times is None:
        times = upstream_timestamps(num_frames)
    last_grid = float(times[(num_frames - 1) // 2 * 2])  # the processor pads an odd frame count with the last frame
    need = (int(last_grid * TOKENS_PER_SECOND) + 2) * SAMPLES_PER_TOKEN
    return wave if wave.size >= need else np.pad(wave, (0, need - wave.size))


def attach_audio(video_items):
    """Decode each video's audio track into its metadata (``AUDIO_KEY``) from the video's bytes (vLLM keeps them on the
    item: MediaWithBytes), padded to cover every frame. Items without bytes or metadata (dummy videos, cached items)
    are left alone."""
    if video_items is None:
        return
    for k, entry in enumerate(getattr(video_items, "data", [])):
        if not (isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict)):
            continue
        media, meta = entry
        if AUDIO_KEY in meta:
            continue
        data = getattr(media, "original_bytes", None)
        if data is None:
            continue
        wave = cached_audio_track(data)
        if wave is not None:
            frames = getattr(media, "media", media)
            meta[AUDIO_KEY] = cover_frames(wave, meta, len(frames))


def _video_audio_input(video, wave):
    from vllm.transformers_utils.processors.mimo_v2_omni import VideoAudioInput

    return VideoAudioInput(video=video, audio=(torch.from_numpy(wave), SAMPLE_RATE))


def _waves(mm_items, n):
    from .video_input import _metadata_list

    metas = _metadata_list(mm_items.get("video"), n)
    return [m.get(AUDIO_KEY) if isinstance(m, dict) else None for m in metas]


class VideoAudioMixin:
    """For a MiMoV2OmniMultiModalProcessor subclass placed before VideoTimestampsMixin (module docstring)."""

    def apply(self, inputs, *args, **kwargs):
        if wanted(getattr(inputs, "hf_processor_mm_kwargs", None)):
            attach_audio(inputs.mm_data_items.get("video"))
        return super().apply(inputs, *args, **kwargs)

    def _get_hf_mm_inputs(self, mm_items, hf_kwargs):
        inputs = super()._get_hf_mm_inputs(mm_items, hf_kwargs)  # VideoTimestampsMixin: the videos as (frames, times)
        data = inputs.hf_data
        videos = data.get("videos")
        if not videos:
            return inputs
        waves = _waves(mm_items, len(videos))
        if any(w is None for w in waves):
            if any(w is not None for w in waves):
                logger.warning("MiMo video audio: some videos of the request have no audio track; none takes its audio")
            return inputs
        del data["videos"]
        data["video_audio"] = [_video_audio_input(v, w) for v, w in zip(videos, waves)]
        return inputs

    def _apply_hf_processor_main(self, mm_items, hf_kwargs):
        out = super()._apply_hf_processor_main(mm_items, hf_kwargs)
        if "va_audio_features" not in out:
            return out
        from ..reference import audio as ref
        from .audio import processor as audio_proc

        mels = out.pop("va_audio_features")
        if isinstance(mels, torch.Tensor):
            mels = list(mels) if mels.dim() == 3 else [mels]
        waves = [w for w in _waves(mm_items, len(mels)) if w is not None]
        path = audio_proc.audio_checkpoint_path(self.info.ctx.model_config)
        cfg = ref.MiMoAudioConfig.from_model_path(path)
        mels32 = []
        for k, mel in enumerate(mels):
            mel32 = ref.log_mel_spectrogram(torch.from_numpy(waves[k]), cfg) if k < len(waves) else None
            if mel32 is None or mel32.shape != mel.shape or not torch.equal(mel32.to(mel.dtype), mel):
                logger.warning(f"MiMo video audio: item {k}: tokenizing the processor's {mel.dtype} mel")
                mel32 = mel.float()
            mels32.append(mel32)
        if audio_proc.DEVICE_TOKENIZER:  # the engine's row tokenizes it: the fp32 mel, exactly (int32 bits: never cast)
            out["va_audio_mel_bits"] = [m.contiguous().view(torch.int32) for m in mels32]
        else:
            out["va_audio_codes"] = audio_proc.tokenize_mels(mels32, path)
        return out

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        from vllm.multimodal.inputs import MultiModalFieldConfig

        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        fields.pop("va_audio_features", None)
        fields["va_audio_codes"] = MultiModalFieldConfig.batched("video")  # fields missing from the output are skipped
        fields["va_audio_mel_bits"] = MultiModalFieldConfig.batched("video")
        return fields
