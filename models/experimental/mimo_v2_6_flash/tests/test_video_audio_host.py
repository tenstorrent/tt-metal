# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A video with its audio track (tt/video_audio.py, tt/multimodal.py; host, needs vLLM 0.31.0, PyAV, torchaudio and a
checkpoint with ``audio_tokenizer/``).

A synthetic 6 s clip (4 fps, AAC audio) goes through vLLM's media loader and the TT processor with
``mm_processor_kwargs={"mimo_video_audio": true}``:

* the prompt is the checkpoint processor's interleaving: per temporal grid (two frames, 0.5 s) its timestamp, frames
  and an audio segment of 6.25 tokens a second (``video_audio_seg_lens``), all inside one video item;
* the item carries the audio as the track's fp32 mel bit for bit (``va_audio_mel_bits``, MIMO_AUDIO_TOKENIZER=device,
  the default: the engine's row tokenizes it) or as its RVQ codes (``va_audio_codes``, MIMO_AUDIO_TOKENIZER=frontend):
  the codes are the fp32 host tokenizer's on exactly that mel, and both items have the same prompt and other fields;
* the engine's rows (``interleave_video_audio``, scattered by ``chunk_embeds``) put each frame row on its video pad and
  each audio row on its audio pad, in the order vLLM's ``embed_multimodal`` lays them out, whichever field carries the
  audio: given the same codes the mel item's rows equal the codes item's; a track's codes are remembered by item
  identifier (``TRACK_CODES``), so a video encoded again does not run the tokenizer again;
* without the switch, or for a clip without audio, the video goes alone.
"""

import base64
import collections
import io
import os

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")
av = pytest.importorskip("av")
pytest.importorskip("torchaudio")

from models.experimental.mimo_v2_6_flash.reference import audio as ref  # noqa: E402
from models.experimental.mimo_v2_6_flash.reference import vision_inputs as vi  # noqa: E402
from models.experimental.mimo_v2_6_flash.tests.common import model_path  # noqa: E402
from models.experimental.mimo_v2_6_flash.tt import multimodal as mm  # noqa: E402

PROMPT = "<|im_start|>user\n" + vi.VIDEO_PLACEHOLDER + "What is said?<|im_end|>\n<|im_start|>assistant\n"
SECONDS, FPS, SR = 6, 4, 44100


def clip(with_audio=True, audio_seconds=SECONDS):
    """An mp4: SECONDS s of panning frames at FPS, and a 440 Hz tone rising in loudness (AAC, SR)."""
    frames = vi.sample_video(num_frames=SECONDS * FPS, size=(320, 192))
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        vs = c.add_stream("libx264", rate=FPS)
        vs.width, vs.height, vs.pix_fmt = 320, 192, "yuv420p"
        a_s = c.add_stream("aac", rate=SR) if with_audio else None
        if a_s is not None:
            a_s.layout = "mono"
        for f in frames:
            for pkt in vs.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(pkt)
        for pkt in vs.encode():
            c.mux(pkt)
        if a_s is not None:
            t = np.arange(int(audio_seconds * SR)) / SR
            wave = (0.3 * np.sin(2 * np.pi * 440 * t) * (t / SECONDS)).astype(np.float32)
            step = 1024
            for k in range(0, wave.size, step):
                frame = av.AudioFrame.from_ndarray(wave[None, k : k + step], format="flt", layout="mono")
                frame.sample_rate = SR
                frame.pts = k
                for pkt in a_s.encode(frame):
                    c.mux(pkt)
            for pkt in a_s.encode():
                c.mux(pkt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def processor():
    path = model_path()
    if not os.path.exists(os.path.join(path, "audio_tokenizer", "model.safetensors")):
        pytest.skip(f"no audio_tokenizer/ in {path}")
    from vllm.model_executor.models.mimo_v2_omni import (
        MiMoV2OmniDummyInputsBuilder,
        MiMoV2OmniMultiModalProcessor,
        MiMoV2OmniProcessingInfo,
    )

    from models.experimental.mimo_v2_6_flash.tt.video_audio import VideoAudioMixin
    from models.experimental.mimo_v2_6_flash.tt.video_input import (
        VideoDummyMixin,
        VideoInfoMixin,
        VideoTimestampsMixin,
    )

    class Processor(VideoAudioMixin, VideoTimestampsMixin, MiMoV2OmniMultiModalProcessor):
        pass

    class Info(VideoInfoMixin, MiMoV2OmniProcessingInfo):
        pass

    class Dummy(VideoDummyMixin, MiMoV2OmniDummyInputsBuilder):
        pass

    upstream = vi.vllm_processor(path)
    info = Info(upstream.info.ctx)
    return Processor(info, Dummy(info))


@pytest.fixture(scope="module")
def cached_processor(processor):
    """The processor with vLLM's processor cache, as the API server runs it."""
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache

    cache = MultiModalProcessorOnlyCache(processor.info.ctx.model_config)

    class Cached(type(processor)):
        """vLLM 0.31 hands the processor its cache with each call (its renderer does)."""

        def __call__(self, *args, **kwargs):
            return super().__call__(*args, cache=cache, **kwargs)

    return Cached(processor.info, processor.dummy_inputs)


@pytest.fixture(scope="module")
def items(processor):
    """The clip through the processor with the tokenizer encoder in each place (MIMO_AUDIO_TOKENIZER):
    {"device" | "frontend": (prompt ids, placeholder, item)}."""
    from models.experimental.mimo_v2_6_flash.tt.audio import processor as audio_proc

    data, out = clip(), {}
    saved = audio_proc.DEVICE_TOKENIZER
    try:
        for where in ("device", "frontend"):
            audio_proc.DEVICE_TOKENIZER = where == "device"
            out[where] = run(processor, data, mimo_video_audio=True)
    finally:
        audio_proc.DEVICE_TOKENIZER = saved
    return out


@pytest.fixture(scope="module")
def host_audio():
    """The fp32 host tokenizer encoder (the front end's: MIMO_AUDIO_TOKENIZER=frontend) and the bf16 patch encoder."""
    from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import HostAudioBackend

    return HostAudioBackend(model_path())


def as_entry(ph, item, identifier="clip"):
    """A plugin-contract entry for the processor's item (as the plugin sends it)."""
    return {
        "modality": "video",
        "identifier": identifier,
        "offset": ph.offset,
        "length": ph.length,
        "is_embed": ph.is_embed,
        "data": {k: v.data for k, v in item.items()},
    }


def run(proc, data, uuids=None, **kwargs):
    """``data``: the video file's bytes, fetched as vLLM's API server does; None: a uuid-only reference (``uuids``)."""
    from vllm.multimodal.media import MediaConnector

    video = None
    if data is not None:
        video = MediaConnector(media_io_kwargs={"video": {"num_frames": -1}}).fetch_video(
            "data:video/mp4;base64," + base64.b64encode(data).decode()
        )
    out = proc(PROMPT, proc.info.parse_mm_data({"video": [video]}), uuids, hf_processor_mm_kwargs=kwargs)
    (ph,) = out["mm_placeholders"]["video"]
    (item,) = out["mm_kwargs"]["video"]
    return out["prompt_token_ids"], ph, item


def tagged(tags):
    """Rows tagged by integers, as (tag // 64, tag % 64): exact in bf16."""
    return torch.stack([torch.div(tags, 64, rounding_mode="floor"), tags % 64], dim=1).to(torch.bfloat16)


def untag(rows):
    return (rows[:, 0].float() * 64 + rows[:, 1].float()).to(torch.int64)


class TaggedBackend:
    """Frame rows tagged by their row index, audio rows by -1 - their index; ``audio_codes`` gives the code frames a
    mel of that length has (and counts its calls)."""

    def __init__(self):
        self.tokenized = 0

    def embed_video(self, pixel_values, grid):
        return tagged(torch.arange(int(torch.as_tensor(grid).prod()) // 4))

    def embed_audio_codes(self, codes):
        return tagged(-1 - torch.arange(-(-codes.shape[0] // 4)))

    def audio_codes(self, mel):
        self.tokenized += 1
        return torch.zeros(ref.num_codes(mel.shape[0]), 20, dtype=torch.int64)


@pytest.mark.parametrize("where", ["device", "frontend"])
def test_video_with_its_audio_track(processor, items, where):
    vocab = processor.info.get_tokenizer().get_vocab()
    ids, ph, item = items[where]
    block = ids[ph.offset : ph.offset + ph.length]
    t, h, w = item["video_grid_thw"].data.tolist()
    n_segs = int(item["video_audio_n_segs"].data)
    seg_lens = item["video_audio_seg_lens"].data.reshape(-1)[:n_segs].tolist()
    assert t == SECONDS * FPS // 2 and n_segs == t  # one audio segment per grid (interleave length 0)
    assert block.count(vocab["<|mimo_audio_start|>"]) == n_segs == block.count(vocab["<|vision_start|>"])
    pads = block.count(vocab["<|audio_pad|>"])
    assert pads == sum(seg_lens) and abs(pads - SECONDS * 6.25) <= 4, (pads, seg_lens)
    assert all(abs(n - 0.5 * 6.25) <= 1 for n in seg_lens[:-1]), seg_lens
    if where == "frontend":
        codes = item["va_audio_codes"].data
        assert "va_audio_mel_bits" not in item
    else:
        bits = item["va_audio_mel_bits"].data
        assert "va_audio_codes" not in item and bits.dtype == torch.int32 and bits.shape[1] == 128, bits.shape
        codes = torch.zeros(ref.num_codes(bits.shape[0]), 20, dtype=torch.int32)
    assert codes.dtype == torch.int32 and codes.shape[1] == 20 and -(-codes.shape[0] // 4) >= sum(seg_lens)
    is_embed = torch.as_tensor(ph.is_embed)
    pad_ids = (vocab["<|video_pad|>"], vocab["<|audio_pad|>"])
    assert torch.equal(is_embed, torch.tensor([x in pad_ids for x in block]))

    # The engine's rows: frame rows tagged by their row index, audio rows by -1 - their index.
    (it,) = mm.parse_items([as_entry(ph, item, f"clip-order-{where}")])
    emb = mm.encode_item(TaggedBackend(), it)
    embeds, rows = mm.chunk_embeds([it], 0, len(ids), lambda x: emb, 2)
    got = untag(embeds[rows])
    # vLLM's embed_multimodal order: per group the group's frames, then the group's audio segment.
    n_per = h * w // 4
    expect, a = [], 0
    for g in range(n_segs):
        expect += list(range(g * n_per, (g + 1) * n_per))
        expect += [-1 - k for k in range(a, a + seg_lens[g])]
        a += seg_lens[g]
    assert got.tolist() == expect
    pos = torch.nonzero(rows).flatten()
    assert all(ids[p] == vocab["<|video_pad|>"] for p, v in zip(pos.tolist(), got.tolist()) if v >= 0)
    assert all(ids[p] == vocab["<|audio_pad|>"] for p, v in zip(pos.tolist(), got.tolist()) if v < 0)


def test_device_tokenizer_sends_the_track_mel(items, host_audio):
    """MIMO_AUDIO_TOKENIZER=device: the same prompt and fields as with the front end's tokenizer, except the audio,
    which is the fp32 mel the front end tokenizes, bit for bit (its codes are the fp32 host tokenizer's on it)."""
    (ids_d, ph_d, item_d), (ids_f, ph_f, item_f) = items["device"], items["frontend"]
    assert ids_d == ids_f and (ph_d.offset, ph_d.length) == (ph_f.offset, ph_f.length)
    assert torch.equal(torch.as_tensor(ph_d.is_embed), torch.as_tensor(ph_f.is_embed))
    assert set(item_d) - {"va_audio_mel_bits"} == set(item_f) - {"va_audio_codes"}
    for k in set(item_d) - {"va_audio_mel_bits"}:
        assert torch.equal(item_d[k].data, item_f[k].data), k
    mel = mm.mel_from_bits(item_d["va_audio_mel_bits"].data, "clip")
    assert mel.dtype == torch.float32 and bool(torch.isfinite(mel).all())
    assert torch.equal(host_audio.codes(mel), item_f["va_audio_codes"].data.long())


def test_device_path_rows_equal_host_path_rows(items, host_audio, monkeypatch):
    """Given the same codes, the mel item's engine rows (``va_audio_mel_bits`` through the backend's ``audio_codes``)
    equal the codes item's (``va_audio_codes``) bit for bit, with the real bf16 patch encoder; a track's codes are
    remembered by identifier (an item encoded again does not tokenize again); and an AudioCodesBackend without a
    device tokenizer tokenizes on the host (the front end's codes)."""
    from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import AudioCodesBackend

    monkeypatch.setattr(mm, "TRACK_CODES", mm.TrackCodes())
    (_, ph, item_d), (_, _, item_f) = items["device"], items["frontend"]

    class Backend(TaggedBackend):
        def embed_video(self, pixel_values, grid):  # tagged frame rows, as wide as the patch encoder's
            return torch.nn.functional.pad(super().embed_video(pixel_values, grid), (0, 4094))

        def embed_audio_codes(self, codes):
            return host_audio.embed_audio_codes(codes)

        def audio_codes(self, mel):
            self.tokenized += 1
            return host_audio.codes(mel)

    backend = Backend()
    (it_f,) = mm.parse_items([as_entry(ph, item_f, "codes-item")])
    (it_d,) = mm.parse_items([as_entry(ph, item_d, "mel-item")])
    host_rows = mm.encode_item(backend, it_f)
    assert backend.tokenized == 0 and host_rows.shape[0] == it_f.num_embeds
    assert torch.equal(mm.encode_item(backend, it_d), host_rows) and backend.tokenized == 1
    assert torch.equal(mm.encode_item(backend, it_d), host_rows) and backend.tokenized == 1  # remembered codes
    (it_other,) = mm.parse_items([as_entry(ph, item_d, "another-mel-item")])
    assert torch.equal(mm.encode_item(backend, it_other), host_rows) and backend.tokenized == 2
    engine = AudioCodesBackend(backend, model_path())  # no device tokenizer: the host one, loaded on first use
    mel = mm.mel_from_bits(item_d["va_audio_mel_bits"].data, "clip")
    assert torch.equal(engine.audio_codes(mel), item_f["va_audio_codes"].data.long())


def test_track_codes_cache_is_bounded():
    """TRACK_CODES keeps the most recently used tracks' codes (int16) up to its budget."""
    cache = mm.TrackCodes(budget_bytes=3 * 100 * 20 * 2)  # three 100-frame tracks
    for name in ("a", "b", "c"):
        cache.put(name, torch.full((100, 20), 7, dtype=torch.int64))
    cache.get("a")
    cache.put("d", torch.zeros(100, 20, dtype=torch.int64))
    assert cache.get("b") is None and all(cache.get(k) is not None for k in "acd")
    assert cache.get("a").dtype == torch.int16 and int(cache.get("a").max()) == 7
    cache.put("e", torch.zeros(1000, 20, dtype=torch.int64))  # larger than the budget: not kept
    assert cache.get("e") is None and cache.used == 3 * 100 * 20 * 2


def test_without_audio(processor):
    vocab = processor.info.get_tokenizer().get_vocab()
    for data, kwargs in ((clip(), {"mimo_video_audio": False}), (clip(with_audio=False), {"mimo_video_audio": True})):
        ids, ph, item = run(processor, data, **kwargs)
        assert vocab["<|mimo_audio_start|>"] not in ids[ph.offset : ph.offset + ph.length]
        assert "va_audio_codes" not in item and "va_audio_mel_bits" not in item


def test_a_videos_audio_does_not_depend_on_the_processor_cache(cached_processor):
    """Whether a video takes its audio is decided on its request (all of the request's videos have a track, or none
    takes it), before vLLM hashes the items. The processor cache then holds a video with and without its track as
    two entries, so the same request gets the same answer whatever was asked before: a video first seen next to a
    silent one is not served without its track when asked alone, and one cached with its track is not mixed into a
    request with a silent video."""
    from vllm.multimodal.media import MediaConnector

    connector = MediaConnector(media_io_kwargs={"video": {"num_frames": -1}})
    talk, silent = clip(audio_seconds=SECONDS - 0.2), clip(with_audio=False)  # bytes no other test has cached

    def with_audio(*videos):
        fetched = [connector.fetch_video("data:video/mp4;base64," + base64.b64encode(v).decode()) for v in videos]
        prompt = PROMPT.replace(vi.VIDEO_PLACEHOLDER, vi.VIDEO_PLACEHOLDER * len(videos))
        out = cached_processor(
            prompt,
            cached_processor.info.parse_mm_data({"video": fetched}),
            None,
            hf_processor_mm_kwargs={"mimo_video_audio": True},
        )
        return ["va_audio_mel_bits" in item or "va_audio_codes" in item for item in out["mm_kwargs"]["video"]]

    assert with_audio(talk, silent) == [False, False]
    assert with_audio(talk) == [True]  # not the plain entry the mixed request cached
    assert with_audio(talk, silent) == [False, False]  # not the entry with audio next to a plain video
    assert with_audio(silent, talk) == [False, False]
    assert with_audio(talk) == [True] and with_audio(silent) == [False]
    assert with_audio(talk, talk) == [True, True]


def test_uniform_audio_and_strip():
    from models.experimental.mimo_v2_6_flash.tt import video_audio as va

    class Items:
        def __init__(self, metas):
            self.data = [(object(), meta) for meta in metas]

    wave = np.zeros(8, dtype=np.float32)
    mixed = Items([{va.AUDIO_KEY: wave}, {}, {va.AUDIO_KEY: wave}])
    assert va.uniform_audio(mixed) is False and all(va.AUDIO_KEY not in meta for _, meta in mixed.data)
    full = Items([{va.AUDIO_KEY: wave}, {va.AUDIO_KEY: wave}])
    assert va.uniform_audio(full) is True and all(va.AUDIO_KEY in meta for _, meta in full.data)
    # An item without loader metadata (frames handed in directly) has no track: it counts as a silent video.
    bare = Items([{va.AUDIO_KEY: wave}])
    bare.data.append(np.zeros((2, 4, 4, 3), dtype=np.uint8))
    assert va.uniform_audio(bare) is False and va.AUDIO_KEY not in bare.data[0][1]
    assert va.uniform_audio(Items([{}, {}])) is False and va.uniform_audio(None) is False
    va.strip_audio(full)
    assert all(va.AUDIO_KEY not in meta for _, meta in full.data)


def test_audio_track_shorter_than_the_picture(processor):
    """A track that ends 0.6 s before the last frame (seen on real clips: the processor's last segment came out empty
    and the request failed) is padded with silence: every grid gets a non-empty segment, the last one included."""
    vocab = processor.info.get_tokenizer().get_vocab()
    ids, ph, item = run(processor, clip(audio_seconds=SECONDS - 0.6), mimo_video_audio=True)
    n_segs = int(item["video_audio_n_segs"].data)
    seg_lens = item["video_audio_seg_lens"].data.reshape(-1)[:n_segs].tolist()
    assert n_segs == SECONDS * FPS // 2 and min(seg_lens) > 0, seg_lens
    assert ids[ph.offset : ph.offset + ph.length].count(vocab["<|audio_pad|>"]) == sum(seg_lens)


@pytest.mark.parametrize("where", ["device", "frontend"])
def test_next_question_on_a_video_redoes_nothing(cached_processor, monkeypatch, where):
    """The 2nd / 3rd question on a video, with vLLM's processor cache: the same bytes again neither decode the audio
    track again (cached_audio_track) nor resize frames, recompute the mel or run the audio tokenizer (which the front
    end runs only with MIMO_AUDIO_TOKENIZER=frontend); a uuid-only reference (the request's
    ``{"type": "video_url", "video_url": null, "uuid": id}`` after a first request with that uuid) also skips vLLM's
    fetch and decode. Both give the first request's prompt and item."""
    from models.experimental.mimo_v2_6_flash.tt import video_audio, video_input
    from models.experimental.mimo_v2_6_flash.tt.audio import processor as audio_proc

    monkeypatch.setattr(video_audio, "_TRACKS", collections.OrderedDict())
    monkeypatch.setattr(audio_proc, "DEVICE_TOKENIZER", where == "device")
    calls = collections.Counter()
    for module, name in (
        (video_audio, "decode_audio_track"),
        (video_input, "_resized_frames"),
        (ref, "log_mel_spectrogram"),
        (audio_proc, "tokenize_mels"),
    ):
        fn = getattr(module, name)
        monkeypatch.setattr(module, name, lambda *a, _fn=fn, _name=name, **k: (calls.update([_name]), _fn(*a, **k))[1])
    data = clip()
    uuids = {"video": [f"test-next-question-on-a-video-{where}"]}  # the processor cache does not key on the placement
    work = {"decode_audio_track": 1, "_resized_frames": 1, "log_mel_spectrogram": 1}
    if where == "frontend":
        work["tokenize_mels"] = 1
    first = run(cached_processor, data, uuids, mimo_video_audio=True)
    assert ("va_audio_mel_bits" if where == "device" else "va_audio_codes") in first[2]
    assert calls == work, calls
    again = run(cached_processor, data, uuids, mimo_video_audio=True)
    reference = run(cached_processor, None, uuids, mimo_video_audio=True)
    assert calls == work, calls
    for ids, ph, item in (again, reference):
        assert ids == first[0] and (ph.offset, ph.length) == (first[1].offset, first[1].length)
        assert item.keys() == first[2].keys() and all(torch.equal(item[k].data, first[2][k].data) for k in item)


def test_audio_track_cache_is_bounded(monkeypatch):
    """cached_audio_track keeps the most recently used tracks up to MIMO_VIDEO_AUDIO_CACHE_MB."""
    from models.experimental.mimo_v2_6_flash.tt import video_audio

    monkeypatch.setattr(video_audio, "_TRACKS", collections.OrderedDict())
    monkeypatch.setattr(video_audio, "TRACK_CACHE_BYTES", 3 * 4000)  # three 1000-sample float32 tracks
    decoded = []
    monkeypatch.setattr(video_audio, "decode_audio_track", lambda d: decoded.append(d) or np.zeros(1000, np.float32))
    for data in (b"a", b"b", b"a", b"c", b"d", b"a", b"b"):
        video_audio.cached_audio_track(data)
    assert decoded == [b"a", b"b", b"c", b"d", b"b"]  # "d" evicted the least recently used "b", not "a"


def test_track_ending_inside_the_last_grid_is_not_padded():
    """Only a track that ends before the last grid starts is padded (to two tokens past it); one that ends inside the
    last grid keeps its length (the last segment is shorter, as Xiaomi's API counts it)."""
    from models.experimental.mimo_v2_6_flash.tt.video_audio import SAMPLE_RATE, cover_frames
    from models.experimental.mimo_v2_6_flash.tt.video_input import dummy_metadata

    meta = dict(dummy_metadata(24), fps=4.0)  # 24 frames 0.25 s apart: the last grid starts at 5.5 s
    inside = np.zeros(int(5.9 * SAMPLE_RATE), dtype=np.float32)  # ends inside the last grid [5.5, 6.0)
    assert cover_frames(inside, meta, 24) is inside
    short = np.zeros(int(5.0 * SAMPLE_RATE), dtype=np.float32)
    padded = cover_frames(short, meta, 24)
    assert padded.size == (int(5.5 * 6.25) + 2) * SAMPLE_RATE * 4 // 25 and not padded[short.size :].any()


def test_switch(monkeypatch):
    """MIMO_VIDEO_AUDIO (on by default when audio is served) is the default; a request's or the server's
    mm_processor_kwargs ``mimo_video_audio`` overrides it either way."""
    from models.experimental.mimo_v2_6_flash.tt import video_audio

    for default in (True, False):
        monkeypatch.setattr(video_audio, "VIDEO_AUDIO", default)
        assert video_audio.wanted({}) is default and video_audio.wanted(None) is default
        assert video_audio.wanted({"mimo_video_audio": True}) and not video_audio.wanted({"mimo_video_audio": False})
    monkeypatch.setenv("MIMO_MM_MODALITIES", "image,video")
    assert video_audio._default_video_audio() == "0"
    monkeypatch.setenv("MIMO_MM_MODALITIES", "image,video,audio")
    assert video_audio._default_video_audio() == "1"
