# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A video with its audio track (tt/video_audio.py, tt/multimodal.py; host, needs vLLM 0.30.0, PyAV, torchaudio and a
checkpoint with ``audio_tokenizer/``).

A synthetic 6 s clip (4 fps, AAC audio) goes through vLLM's media loader and the TT processor with
``mm_processor_kwargs={"mimo_video_audio": true}``:

* the prompt is the checkpoint processor's interleaving: per temporal grid (two frames, 0.5 s) its timestamp, frames
  and an audio segment of 6.25 tokens a second (``video_audio_seg_lens``), all inside one video item;
* the item carries the audio as RVQ codes (``va_audio_codes``), the same codes the front end makes for the track as an
  audio item;
* the engine's rows (``interleave_video_audio``, scattered by ``chunk_embeds``) put each frame row on its video pad and
  each audio row on its audio pad, in the order vLLM's ``embed_multimodal`` lays them out;
* without the switch, or for a clip without audio, the video goes alone.
"""

import base64
import io
import os

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")
av = pytest.importorskip("av")
pytest.importorskip("torchaudio")

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


def run(proc, data, **kwargs):
    from vllm.multimodal.media import MediaConnector

    video = MediaConnector(media_io_kwargs={"video": {"num_frames": -1}}).fetch_video(
        "data:video/mp4;base64," + base64.b64encode(data).decode()
    )
    out = proc(PROMPT, proc.info.parse_mm_data({"video": [video]}), hf_processor_mm_kwargs=kwargs)
    (ph,) = out["mm_placeholders"]["video"]
    (item,) = out["mm_kwargs"]["video"]
    return out["prompt_token_ids"], ph, item


def test_video_with_its_audio_track(processor):
    vocab = processor.info.get_tokenizer().get_vocab()
    ids, ph, item = run(processor, clip(), mimo_video_audio=True)
    block = ids[ph.offset : ph.offset + ph.length]
    t, h, w = item["video_grid_thw"].data.tolist()
    n_segs = int(item["video_audio_n_segs"].data)
    seg_lens = item["video_audio_seg_lens"].data.reshape(-1)[:n_segs].tolist()
    assert t == SECONDS * FPS // 2 and n_segs == t  # one audio segment per grid (interleave length 0)
    assert block.count(vocab["<|mimo_audio_start|>"]) == n_segs == block.count(vocab["<|vision_start|>"])
    pads = block.count(vocab["<|audio_pad|>"])
    assert pads == sum(seg_lens) and abs(pads - SECONDS * 6.25) <= 4, (pads, seg_lens)
    assert all(abs(n - 0.5 * 6.25) <= 1 for n in seg_lens[:-1]), seg_lens
    codes = item["va_audio_codes"].data
    assert codes.dtype == torch.int32 and codes.shape[1] == 20 and -(-codes.shape[0] // 4) >= sum(seg_lens)
    is_embed = torch.as_tensor(ph.is_embed)
    pad_ids = (vocab["<|video_pad|>"], vocab["<|audio_pad|>"])
    assert torch.equal(is_embed, torch.tensor([x in pad_ids for x in block]))

    # The engine's rows: frame rows tagged by their row index, audio rows by -1 - their index (as (tag // 64, tag % 64):
    # exact in bf16).
    def tagged(tags):
        return torch.stack([torch.div(tags, 64, rounding_mode="floor"), tags % 64], dim=1).to(torch.bfloat16)

    class Backend:
        def embed_video(self, pixel_values, grid):
            return tagged(torch.arange(int(torch.as_tensor(grid).prod()) // 4))

        def embed_audio_codes(self, codes):
            return tagged(-1 - torch.arange(-(-codes.shape[0] // 4)))

    entry = {
        "modality": "video",
        "identifier": "clip",
        "offset": ph.offset,
        "length": ph.length,
        "is_embed": ph.is_embed,
        "data": {k: v.data for k, v in item.items()},
    }
    (it,) = mm.parse_items([entry])
    emb = mm.encode_item(Backend(), it)
    embeds, rows = mm.chunk_embeds([it], 0, len(ids), lambda x: emb, 2)
    got = (embeds[rows][:, 0].float() * 64 + embeds[rows][:, 1].float()).to(torch.int64)
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


def test_without_audio(processor):
    vocab = processor.info.get_tokenizer().get_vocab()
    for data, kwargs in ((clip(), {"mimo_video_audio": False}), (clip(with_audio=False), {"mimo_video_audio": True})):
        ids, ph, item = run(processor, data, **kwargs)
        assert vocab["<|mimo_audio_start|>"] not in ids[ph.offset : ph.offset + ph.length]
        assert "va_audio_codes" not in item


def test_audio_track_shorter_than_the_picture(processor):
    """A track that ends 0.6 s before the last frame (seen on real clips: the processor's last segment came out empty
    and the request failed) is padded with silence: every grid gets a non-empty segment, the last one included."""
    vocab = processor.info.get_tokenizer().get_vocab()
    ids, ph, item = run(processor, clip(audio_seconds=SECONDS - 0.6), mimo_video_audio=True)
    n_segs = int(item["video_audio_n_segs"].data)
    seg_lens = item["video_audio_seg_lens"].data.reshape(-1)[:n_segs].tolist()
    assert n_segs == SECONDS * FPS // 2 and min(seg_lens) > 0, seg_lens
    assert ids[ph.offset : ph.offset + ph.length].count(vocab["<|audio_pad|>"]) == sum(seg_lens)


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
