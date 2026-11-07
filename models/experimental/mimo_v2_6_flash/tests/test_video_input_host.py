# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Video input fidelity (tt/video_input.py; host, needs vLLM 0.30.0 and the checkpoint).

* Timestamps: a video whose loader metadata says its 32 frames were sampled over 74 s is announced with those times
  (00:00, 00:04, ... 01:11), not upstream's 00:00 .. 00:30, also through vLLM's processor cache (which re-parses the
  items it processes, so the metadata must travel with the frames); the video block vLLM writes (from the item's
  fields) equals the checkpoint processor's own for the same frame times, token for token; MIMO_VIDEO_TIMESTAMPS=vllm
  gives upstream's prompt; profiling's dummy videos get metadata and go through.
* Pixel caps: MIMO_VIDEO_MAX_PIXELS caps every frame's merged tokens; MIMO_VIDEO_TOTAL_MAX_PIXELS a clip's.
"""

import os
import re

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from models.experimental.mimo_v2_6_flash.reference import vision_inputs as vi  # noqa: E402
from models.experimental.mimo_v2_6_flash.tests.common import model_path  # noqa: E402

PROMPT = "<|im_start|>user\n" + vi.VIDEO_PLACEHOLDER + "What happens in the video?<|im_end|>\n<|im_start|>assistant\n"
SOURCE_FRAMES, SOURCE_FPS = 2227, 29.97  # a 74.3 s clip
TIMESTAMP = re.compile(r"\d\d:\d\d")


@pytest.fixture(scope="module")
def processors():
    path = model_path()
    if not os.path.exists(os.path.join(path, "config.json")):
        pytest.skip(f"no checkpoint at {path}")
    from vllm.model_executor.models.mimo_v2_omni import (
        MiMoV2OmniDummyInputsBuilder,
        MiMoV2OmniMultiModalProcessor,
        MiMoV2OmniProcessingInfo,
    )
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache

    from models.experimental.mimo_v2_6_flash.tt.video_input import (
        VideoDummyMixin,
        VideoInfoMixin,
        VideoTimestampsMixin,
    )

    class Processor(VideoTimestampsMixin, MiMoV2OmniMultiModalProcessor):
        pass

    class Info(VideoInfoMixin, MiMoV2OmniProcessingInfo):
        pass

    class Dummy(VideoDummyMixin, MiMoV2OmniDummyInputsBuilder):
        pass

    upstream = vi.vllm_processor(path)
    info = Info(upstream.info.ctx)
    cache = MultiModalProcessorOnlyCache(upstream.info.ctx.model_config)
    return {
        "upstream": upstream,
        "ours": Processor(info, Dummy(info)),
        "cached": Processor(info, Dummy(info), cache=cache),
    }


def loader_metadata(num_frames):
    """What vLLM's opencv loader returns for frames sampled uniformly over the clip (VideoLoader.create_hf_metadata)."""
    indices = np.linspace(0, SOURCE_FRAMES - 1, num_frames, dtype=int).tolist()
    return {
        "total_num_frames": SOURCE_FRAMES,
        "fps": SOURCE_FPS,
        "duration": SOURCE_FRAMES / SOURCE_FPS,
        "video_backend": "opencv",
        "frames_indices": indices,
        "do_sample_frames": False,
    }


def run(proc, video):
    out = proc(PROMPT, proc.info.parse_mm_data({"video": [video]}), hf_processor_mm_kwargs={})
    (ph,) = out["mm_placeholders"]["video"]
    (item,) = out["mm_kwargs"]["video"]
    return out["prompt_token_ids"], ph, item


def timestamps_in(proc, ids, ph):
    text = proc.info.get_tokenizer().decode(ids[ph.offset : ph.offset + ph.length])
    return TIMESTAMP.findall(text)


@pytest.mark.parametrize("which", ["ours", "cached"])
def test_loader_times(processors, which, monkeypatch):
    from models.experimental.mimo_v2_6_flash.tt import video_input
    from models.experimental.mimo_v2_6_flash.tt.video_input import loader_timestamps

    monkeypatch.setattr(video_input, "VIDEO_TIMESTAMPS", "loader")
    upstream, ours = processors["upstream"], processors[which]
    frames = vi.sample_video(num_frames=32, size=(320, 192))
    meta = loader_metadata(32)
    times = loader_timestamps(meta, 32)
    true = torch.tensor(meta["frames_indices"], dtype=torch.float64) / SOURCE_FPS
    assert float((times.double() - true).abs().max()) < 0.004 * float(true[-1]) + 0.05

    for _ in range(2):  # the second request hits the processor cache (when there is one)
        ids, ph, item = run(ours, (frames, meta))
        stamps = timestamps_in(ours, ids, ph)
        expect = [f"{int(t // 60):02d}:{int(t % 60):02d}" for t in times[::2].tolist()]
        assert stamps == expect and stamps[-1] == "01:11", stamps
    # vLLM wrote those tokens from the item's fields; the checkpoint's processor writes the same video block (its
    # <|mimo_video_start|> .. <|mimo_video_end|>) from the frame times.
    hf = ours.info.get_hf_processor()
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    hf_ids = hf(text=PROMPT, videos=[(t, times)], return_tensors="pt")["input_ids"].reshape(-1).tolist()
    vocab = ours.info.get_tokenizer().get_vocab()
    first, last = hf_ids.index(vocab["<|mimo_video_start|>"]), hf_ids.index(vocab["<|mimo_video_end|>"])
    assert ids[ph.offset : ph.offset + ph.length] == hf_ids[first : last + 1]
    assert abs(float(item["second_per_grid_ts"].data) - 2 * float(times[1] - times[0])) < 1e-6

    # Upstream: the same frames announced as 00:00 .. 00:30, the rest of the prompt identical.
    up_ids, up_ph, _ = run(upstream, frames)
    assert timestamps_in(upstream, up_ids, up_ph) == [f"00:{2 * j:02d}" for j in range(16)]
    assert len(up_ids) == len(ids)


def test_vllm_timestamps_switch(processors, monkeypatch):
    from models.experimental.mimo_v2_6_flash.tt import video_input

    frames = vi.sample_video(num_frames=8, size=(320, 192))
    monkeypatch.setattr(video_input, "VIDEO_TIMESTAMPS", "vllm")
    assert run(processors["ours"], (frames, loader_metadata(8)))[0] == run(processors["upstream"], frames)[0]
    # Metadata that does not describe the frames: upstream's labels too.
    monkeypatch.setattr(video_input, "VIDEO_TIMESTAMPS", "loader")
    assert run(processors["ours"], (frames, loader_metadata(9)))[0] == run(processors["upstream"], frames)[0]


def test_dummy_videos_have_metadata(processors):
    ours = processors["cached"]
    data = ours.dummy_inputs.get_dummy_mm_data(4096, {"video": 1}, {})
    (video,) = data["video"]
    assert isinstance(video, tuple) and video[1]["frames_indices"] == list(range(len(video[0])))
    out = ours(vi.VIDEO_PLACEHOLDER, ours.info.parse_mm_data(data), hf_processor_mm_kwargs={})
    assert len(out["mm_placeholders"]["video"]) == 1


@pytest.mark.parametrize("frame_cap, total_cap", [(286720, None), (None, 16384 * 1024)])
def test_pixel_caps(processors, monkeypatch, frame_cap, total_cap):
    from models.experimental.mimo_v2_6_flash.tt import video_input

    frames = vi.sample_video(num_frames=64, size=(1280, 720))
    meta = loader_metadata(64)
    full = run(processors["ours"], (frames, meta))[2]["video_grid_thw"].data.tolist()
    assert full == [32, 44, 80]  # 880 merged tokens a grid
    monkeypatch.setattr(video_input, "VIDEO_MAX_PIXELS", frame_cap)
    monkeypatch.setattr(video_input, "VIDEO_TOTAL_MAX_PIXELS", total_cap)
    t, h, w = run(processors["ours"], (frames, meta))[2]["video_grid_thw"].data.tolist()
    per_grid = h * w // 4
    if frame_cap:
        assert per_grid * 1024 <= frame_cap and per_grid >= 200, (t, h, w)
    else:
        assert t * per_grid * 1024 <= total_cap and t * per_grid >= 0.8 * total_cap / 1024, (t, h, w)
