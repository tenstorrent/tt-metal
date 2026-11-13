# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Video input fidelity (tt/video_input.py; host, needs vLLM 0.31.0 and the checkpoint).

* Timestamps: a video whose loader metadata says its 32 frames were sampled over 74 s is announced with those times
  (00:00, 00:04, ... 01:11), not upstream's 00:00 .. 00:30, also through vLLM's processor cache (which re-parses the
  items it processes, so the metadata must travel with the frames); the video block vLLM writes (from the item's
  fields) equals the checkpoint processor's own for the same frame times, token for token; MIMO_VIDEO_TIMESTAMPS=vllm
  gives upstream's prompt; profiling's dummy videos get metadata and go through.
* Pixel caps: MIMO_VIDEO_MAX_PIXELS caps every frame; MIMO_VIDEO_TOTAL_MAX_PIXELS a clip (spread over its frames);
  the defaults give Xiaomi's API's frame sizes; frames are resized before the processor exactly as it would.
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

    class Cached(Processor):
        """With vLLM's processor cache, which vLLM 0.31 hands the processor with each call (its renderer does)."""

        def __call__(self, *args, **kwargs):
            return super().__call__(*args, cache=cache, **kwargs)

    return {
        "upstream": upstream,
        "ours": Processor(info, Dummy(info)),
        "cached": Cached(info, Dummy(info)),
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


@pytest.mark.parametrize(
    "fps, frame_step, want_late_s",
    [
        (30000 / 1001, 15, 2.4),  # NTSC at 2 fps: a true interval of 0.5005 s, labelled 0.5 s
        (30.0, 15.5, 5.0),  # 0.51667 s, labelled 0.515625 s
        (25.0, 12.5, 0.0),  # 0.5 s: exact
    ],
)
def test_loader_times_drift_is_bounded(fps, frame_step, want_late_s):
    """The labels are uniform (vLLM rebuilds them from a start and an interval) with the interval rounded to bf16, so
    frame j's label is off by j times the interval's rounding error: it grows along the clip and stays within bf16's
    half unit in the last place, 0.39 % of the elapsed time (tt/video_input.py's docstring). 4800 frames: a 40-minute
    clip at 2 fps."""
    from models.experimental.mimo_v2_6_flash.tt.video_input import loader_timestamps

    n = 4800
    indices = np.floor(np.arange(n) * frame_step).astype(int)
    times = loader_timestamps({"frames_indices": indices.tolist(), "fps": fps}, n).double()
    true = torch.from_numpy(indices / fps)
    steps = times[1:] - times[:-1]
    assert float(steps.max() - steps.min()) == 0.0  # uniform, and exact in fp32: every label is start + j * step
    late = true - times
    assert abs(float(late[-1]) - want_late_s) < 0.1, float(late[-1])
    assert float(late.abs().max()) <= 2**-8 * float(true[-1]) + 0.05
    # The error is the interval's, times the frame index: no worse anywhere than at the end.
    assert float(late.abs().max()) <= abs(float(late[-1])) + 0.05


def test_frame_cap(monkeypatch, expect_error):
    """A clip sampled into more frames than MIMO_VIDEO_MAX_FRAMES is refused with a message: by vLLM's loader as soon
    as it has chosen the frame indices (before it decodes a frame), and by ``mimo_video`` for frames from elsewhere."""
    from vllm.multimodal.video import VideoBackend, VideoSourceMetadata, VideoTargetMetadata

    from models.experimental.mimo_v2_6_flash.tt import video_input

    assert video_input.FRAME_CAP_INSTALLED and video_input.install_frame_cap()  # installed once
    assert video_input.VIDEO_MAX_FRAMES == 7200
    source = VideoSourceMetadata(total_frames_num=3000, original_fps=30.0, duration=100.0)

    def sample(num_frames, fps):
        target = VideoTargetMetadata(num_frames=num_frames, fps=fps, max_duration=300)
        return VideoBackend.compute_frames_index_to_sample(source, target)

    assert len(sample(-1, 2)) == 200
    monkeypatch.setattr(video_input, "VIDEO_MAX_FRAMES", 64)
    with expect_error(ValueError, r"\(100 s long\) is sampled into 200 frames, more than the 64"):
        sample(-1, 2)
    assert len(sample(64, 2)) == 64 and len(sample(-1, 0.5)) == 50
    frames = np.zeros((66, 8, 8, 3), dtype=np.uint8)
    with expect_error(ValueError, "sampled into 66 frames"):
        video_input.mimo_video(frames, None)
    assert video_input.mimo_video(frames[:64], None)[0].shape[0] == 64
    monkeypatch.setattr(video_input, "VIDEO_MAX_FRAMES", 0)  # no limit
    assert len(sample(-1, 2)) == 200 and video_input.mimo_video(frames, None)[0].shape[0] == 66


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
    monkeypatch.setattr(video_input, "VIDEO_MAX_PIXELS", None)
    monkeypatch.setattr(video_input, "VIDEO_TOTAL_MAX_PIXELS", None)
    full = run(processors["ours"], (frames, meta))[2]["video_grid_thw"].data.tolist()
    assert full == [32, 44, 80]  # the checkpoint's sizes: 880 merged tokens a grid
    monkeypatch.setattr(video_input, "VIDEO_MAX_PIXELS", frame_cap)
    monkeypatch.setattr(video_input, "VIDEO_TOTAL_MAX_PIXELS", total_cap)
    t, h, w = run(processors["ours"], (frames, meta))[2]["video_grid_thw"].data.tolist()
    frame_px = h * w * 256
    cap = frame_cap or total_cap // 64  # the clip total is spread over its frames
    assert frame_px <= cap and frame_px >= 0.8 * cap, (t, h, w)


def test_xiaomi_frame_sizes(processors):
    """The defaults are Xiaomi's API's: a 1280x720 frame is 736x384 (276 merged tokens a grid), and a 523 s clip
    sampled at 2 fps (1046 frames) gets 268435456 / 1046 pixels a frame (231 tokens a grid), as its token counts show.
    """
    from models.experimental.mimo_v2_6_flash.tt.video_input import processor_frame_size

    p = processors["ours"].info.get_hf_processor().mimo_processor
    resize = (p._vid_kw, p.patch_size * p.merge_size)
    assert processor_frame_size(*resize, 720, 1280, 64) == (384, 736)
    h, w = processor_frame_size(*resize, 720, 1280, 1046)
    assert h * w // 1024 == 231, (h, w)
    _, _, item = run(processors["ours"], (vi.sample_video(num_frames=64, size=(1280, 720)), loader_metadata(64)))
    t, gh, gw = item["video_grid_thw"].data.tolist()
    assert (t, gh * gw // 4) == (32, 276)


def test_odd_frame_count_keeps_whole_pairs(processors, monkeypatch):
    """33 sampled frames give 16 grids, as Xiaomi's API counts them (its 523.6 s clip at 2 fps: 523 grids), not the 17
    the checkpoint's processor makes by repeating the last frame; MIMO_VIDEO_EVEN_FRAMES=0 keeps the repeat."""
    from models.experimental.mimo_v2_6_flash.tt import video_input

    frames = vi.sample_video(num_frames=33, size=(320, 192))
    assert run(processors["ours"], (frames, loader_metadata(33)))[2]["video_grid_thw"].data.tolist()[0] == 16
    monkeypatch.setattr(video_input, "EVEN_FRAMES", False)
    assert run(processors["ours"], (frames, loader_metadata(33)))[2]["video_grid_thw"].data.tolist()[0] == 17


def test_video_items_bounded_only_by_the_context(processors):
    """vLLM sizes its encoder cache from the processing info's per-item bounds and refuses a longer item; upstream's
    video bound (114688) refused a VGI-Bench clip of 124078 embedding rows. A video's bound is the context."""
    upstream = processors["upstream"].info.get_mm_max_tokens_per_item(1048576, {"video": 1})
    ours = processors["ours"].info.get_mm_max_tokens_per_item(1048576, {"video": 1})
    assert upstream["video"] < 124078 and ours["video"] == 1048576 and ours["image"] == upstream["image"]


def test_cache_hit_converts_nothing(processors, monkeypatch):
    """vLLM's processor cache: the second request on a video (the next question) neither resizes nor processes its
    frames."""
    from models.experimental.mimo_v2_6_flash.tt import video_input

    calls = []
    resized = video_input._resized_frames
    monkeypatch.setattr(video_input, "_resized_frames", lambda *a: calls.append(1) or resized(*a))
    frames = vi.sample_video(num_frames=16, size=(640, 352))
    meta = loader_metadata(16)
    first = run(processors["cached"], (frames, meta))
    second = run(processors["cached"], (frames, meta))
    assert len(calls) == 1 and first[0] == second[0]


@pytest.mark.parametrize("resize", ["torch", "cv2"])
def test_frames_resized_before_the_processor_as_it_would(processors, monkeypatch, resize):
    """mimo_video resizes the frames to the processor's size first (a long clip's frames never exist as full-size
    floats); the processor's own resize is then the identity. With torch (the processor's call) its pixels equal what
    it makes from the full-size frames bit for bit; with OpenCV (the default, faster) they differ by at most one bf16
    step of the normalised pixel."""
    from models.experimental.mimo_v2_6_flash.tt import video_input

    monkeypatch.setattr(video_input, "RESIZE", resize)
    frames = vi.sample_video(num_frames=40, size=(1280, 720))
    meta = loader_metadata(40)
    _, _, item = run(processors["ours"], (frames, meta))
    hf = processors["ours"].info.get_hf_processor()
    full = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    ref = hf(videos=[(full, video_input.loader_timestamps(meta, 40))], return_tensors="pt")
    got, want = item["pixel_values_videos"].data.float(), ref["pixel_values_videos"].to(torch.bfloat16).float()
    if resize == "torch":
        assert torch.equal(got, want)
    else:
        assert got.shape == want.shape and float((got - want).abs().max()) <= 2**-5, float((got - want).abs().max())
        assert float((got != want).float().mean()) < 0.1
