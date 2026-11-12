# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Video input as the checkpoint's processor expects it: real frame times and Xiaomi-like frame sizes.

vLLM's video loader samples the frames (``--media-io-kwargs '{"video": {...}}'`` or a request's ``media_io_kwargs``:
32 frames spread over the whole clip by default; ``{"num_frames": N, "fps": F}`` samples at F frames per second, at
most N frames) and returns them with their source indices and fps (``frames_indices``, ``fps``). vLLM 0.31's
``MiMoV2OmniMultiModalProcessor`` drops those and labels the frames 0, 1, 2, ... seconds (``_INPUT_FPS = 1``), so a
2-minute clip's 32 frames are announced as 00:00 .. 00:30 and every timestamp token the model reads is wrong.

* ``VideoInfoMixin`` (processing info): the data parser keeps each video's loader metadata with its frames
  (``video_needs_metadata``), so it survives vLLM's processor cache, which re-parses the items it has to process from
  their frames alone; ``VideoDummyMixin`` gives profiling's dummy videos metadata (1 fps), as the parser then requires.
* ``VideoTimestampsMixin`` (processor; MIMO_VIDEO_TIMESTAMPS=loader, the default): the frames go to the checkpoint's
  processor with the loader's times. vLLM writes the timestamp tokens again from the item's ``video_start_times`` and
  ``second_per_grid_ts`` (``get_video_replacement``: ``start + j * spt``), fields it may hold in bf16, while the
  checkpoint's processor writes them from the per-frame times; the two must give the same tokens. So the times are
  made uniform: the first sampled frame's time and the sampled frames' mean interval, each rounded so it and a
  grid's span (two frames) are exact in bf16 (and so in every fp32 / fp64 sum both sides make): within 0.4 % of the
  true times. ``vllm`` keeps upstream's 1 fps labels.
* ``VideoInfoMixin`` also applies Xiaomi's API's frame sizes (measured on its token counts, which these reproduce
  exactly): every frame at most MIMO_VIDEO_MAX_PIXELS (307200: a 1280x720 frame becomes 736x384, 276 merged tokens a
  frame pair; the checkpoint allows 8388608, 880) and at most MIMO_VIDEO_TOTAL_MAX_PIXELS / frames (268435456, the
  checkpoint's clip total: a 523 s clip sampled at 2 fps gets 231 tokens a pair). The checkpoint's processor divides
  its clip total by frame pairs, not frames, so it gets half the value. ``0`` turns a cap off (the checkpoint's).
* ``mimo_video`` resizes the frames to the size the processor will give them before turning them into floats, a few
  at a time, with the processor's own interpolation (the processor's resize is then the identity): a long clip's
  frames as floats at full size (4800 frames of a 40-minute 720p clip at 2 fps: 53 GB) never exist.

Frame sampling is vLLM's loader's: Xiaomi's is ``--media-io-kwargs '{"video": {"num_frames": -1, "fps": 2}}'``.
"""

import os

import numpy as np
import torch

VIDEO_TIMESTAMPS = os.environ.get("MIMO_VIDEO_TIMESTAMPS", "loader")
if VIDEO_TIMESTAMPS not in ("loader", "vllm"):
    raise ValueError(f"MIMO_VIDEO_TIMESTAMPS={VIDEO_TIMESTAMPS!r}: loader or vllm")


def _env_pixels(name, default):
    value = os.environ.get(name)
    value = int(float(value)) if value else default
    return value or None


# Xiaomi's API's frame sizes (module docstring); 0 keeps the checkpoint's.
VIDEO_MAX_PIXELS = _env_pixels("MIMO_VIDEO_MAX_PIXELS", 307200)
VIDEO_TOTAL_MAX_PIXELS = _env_pixels("MIMO_VIDEO_TOTAL_MAX_PIXELS", 268435456)
RESIZE_CHUNK = 32  # frames turned into floats and resized at a time (torch)
RESIZE = os.environ.get("MIMO_VIDEO_RESIZE", "cv2")
RESIZE_THREADS = int(os.environ.get("MIMO_VIDEO_RESIZE_THREADS", "8"))
EVEN_FRAMES = os.environ.get("MIMO_VIDEO_EVEN_FRAMES", "1") == "1"


def loader_timestamps(metadata, num_frames):
    """Times (s, float32 [num_frames]) of the frames vLLM's loader sampled, from its metadata (``frames_indices`` into
    the source at ``fps``), uniform and rounded so vLLM and the checkpoint's processor write the same timestamp tokens
    (module docstring). None when the metadata does not describe these frames."""
    if not isinstance(metadata, dict):
        return None
    indices, fps = metadata.get("frames_indices"), metadata.get("fps")
    if indices is None or not fps or fps <= 0 or len(indices) != num_frames or num_frames == 0:
        return None
    start = _bf16(float(indices[0]) / fps)
    if num_frames == 1:
        return torch.tensor([start], dtype=torch.float32)
    mean_step = (float(indices[-1]) - float(indices[0])) / fps / (num_frames - 1)
    step = _bf16(2 * mean_step) / 2  # a grid's span (2 frames) exact in bf16
    if step <= 0:
        return None
    return (start + torch.arange(num_frames, dtype=torch.float64) * step).to(torch.float32)


def _bf16(x):
    return float(torch.tensor(x, dtype=torch.float32).to(torch.bfloat16))


def _frames_tchw(video):
    """[T, H, W, C] (numpy or torch, vLLM's loader) -> float [T, C, H, W], as upstream converts them."""
    frames = torch.from_numpy(np.asarray(video)) if not isinstance(video, torch.Tensor) else video
    if frames.ndim == 4 and frames.shape[-1] in (1, 3, 4):
        return frames.permute(0, 3, 1, 2).float()
    return frames.float()


def _metadata_list(items, n):
    metadata = getattr(items, "metadata", None)
    if isinstance(metadata, dict):
        metadata = [metadata] if n == 1 else None
    if not isinstance(metadata, list) or len(metadata) != n:
        return [None] * n
    return metadata


def upstream_timestamps(num_frames):
    """vLLM 0.31's labels: the frames 1 s apart (MiMoV2OmniMultiModalProcessor._INPUT_FPS)."""
    return torch.arange(num_frames, dtype=torch.float32)


def processor_frame_size(vid_kw, factor, height, width, num_frames, temporal_patch=2):
    """(h, w) the checkpoint's processor gives every frame of a ``num_frames`` clip (MiMoVLProcessor.process_video)."""
    from vllm.transformers_utils.processors.mimo_v2_omni import _smart_resize

    max_px = max(
        vid_kw["min_pixels"],
        min(vid_kw["total_max_pixels"] * temporal_patch // max(num_frames, 1), vid_kw["max_pixels"]),
    )
    return _smart_resize(height, width, factor, vid_kw["min_pixels"], max_px)


def _resized_frames(video, size):
    """[T, H, W, C] uint8 frames -> float [T, C, h, w] at ``size``: the processor's interpolation (_transform_batch:
    bilinear, align_corners=False), which maps each frame on its own. With OpenCV (MIMO_VIDEO_RESIZE=cv2, default),
    float32 cv2.resize INTER_LINEAR (the same pixel-centre bilinear map) on MIMO_VIDEO_RESIZE_THREADS threads: within
    0.03 (on the 0-255 scale) of the processor's values, below bf16's resolution after its normalisation, and 5-8x
    faster on 720p VGI videos than torch's (``torch``: the processor's own call, RESIZE_CHUNK frames at a time)."""
    if (
        RESIZE == "cv2"
        and isinstance(video, np.ndarray)
        and video.ndim == 4
        and video.shape[-1] == 3
        and video.dtype == np.uint8
    ):
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None:
            return _resized_frames_cv2(cv2, video, size)
    import torch.nn.functional as F

    frames = torch.from_numpy(np.asarray(video)) if not isinstance(video, torch.Tensor) else video
    if not (frames.ndim == 4 and frames.shape[-1] in (1, 3, 4)):
        return _frames_tchw(frames)
    out = []
    for k in range(0, frames.shape[0], RESIZE_CHUNK):
        chunk = frames[k : k + RESIZE_CHUNK].permute(0, 3, 1, 2).float()
        if tuple(chunk.shape[-2:]) != tuple(size):
            chunk = F.interpolate(chunk, tuple(size), mode="bilinear", align_corners=False)
        out.append(chunk)
    return torch.cat(out)


def even_frames(video, metadata):
    """Drop the last of an odd number of loader-sampled frames (MIMO_VIDEO_EVEN_FRAMES=1, default). The checkpoint's
    processor would repeat it to fill the last frame pair; Xiaomi's API keeps whole pairs only (a 523.6 s clip at 2 fps:
    its 1047 frames give it 523 grids, the repeat 524, 238 more prompt tokens)."""
    indices = metadata.get("frames_indices") if isinstance(metadata, dict) else None
    n = len(video) if hasattr(video, "__len__") else 0
    if not EVEN_FRAMES or indices is None or n != len(indices) or n < 3 or n % 2 == 0:
        return video, metadata
    return video[:-1], dict(metadata, frames_indices=list(indices[:-1]))


def _resized_frames_cv2(cv2, video, size):
    from concurrent.futures import ThreadPoolExecutor

    h, w = size
    out = torch.empty((video.shape[0], 3, h, w), dtype=torch.float32)
    same = tuple(video.shape[1:3]) == (h, w)

    def one(k):
        frame = video[k].astype(np.float32)
        if not same:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        out[k] = torch.from_numpy(frame).permute(2, 0, 1)

    with ThreadPoolExecutor(RESIZE_THREADS) as pool:
        list(pool.map(one, range(video.shape[0])))
    return out


def mimo_video(video, metadata=None, resize=None):
    """A video as the checkpoint's processor takes it: (frames float [T, C, H, W], times float32 [T]), the loader's
    times with MIMO_VIDEO_TIMESTAMPS=loader and metadata that describes the frames, else upstream's. ``resize``:
    (vid_kw, factor) of the processor, to resize the frames to its size first (module docstring)."""
    if isinstance(video, tuple) and len(video) == 2 and all(isinstance(v, torch.Tensor) for v in video):
        return video
    if isinstance(video, tuple) and len(video) == 2 and (video[1] is None or isinstance(video[1], dict)):
        video, metadata = video
    video, metadata = even_frames(video, metadata)
    shape = tuple(getattr(video, "shape", ()))
    if resize is not None and len(shape) == 4 and shape[-1] in (1, 3, 4):
        size = processor_frame_size(*resize, shape[1], shape[2], shape[0])
        frames = _resized_frames(video, size)
    else:
        frames = _frames_tchw(video)
    times = loader_timestamps(metadata, frames.shape[0]) if VIDEO_TIMESTAMPS == "loader" else None
    return frames, (upstream_timestamps(frames.shape[0]) if times is None else times)


class VideoTimestampsMixin:
    """For a MiMoV2OmniMultiModalProcessor subclass: hand the checkpoint's processor the loader's frame times
    (MIMO_VIDEO_TIMESTAMPS=loader); every video goes as (frames, times)."""

    def _get_hf_mm_inputs(self, mm_items, hf_kwargs):
        """The items' data for the checkpoint's processor, with every video converted here (``mimo_video``).

        vLLM 0.31's MiMoV2OmniMultiModalProcessor converts the videos in its own ``_get_hf_mm_inputs``, on the base
        class's data, before a subclass sees them: it cannot take the (frames, metadata) items of the metadata-keeping
        parser, and it turns the frames into floats at full size. So this takes the base class's data instead of
        upstream's; converting the videos is all upstream's override does. vLLM calls it only for the items its
        processor cache misses (``_apply_hf_processor_main``), so a cache hit converts nothing."""
        from vllm.multimodal.processing.processor import BaseMultiModalProcessor

        inputs = BaseMultiModalProcessor._get_hf_mm_inputs(self, mm_items, hf_kwargs)
        videos = inputs.hf_data.get("videos")
        if videos:
            metadata = _metadata_list(mm_items.get("video"), len(videos))
            processor = self.info.get_hf_processor().mimo_processor
            resize = (processor._vid_kw, processor.patch_size * processor.merge_size)
            videos = [None if v is None else mimo_video(v, m, resize) for v, m in zip(videos, metadata)]
            inputs.hf_data["videos"] = videos
        return inputs


class VideoInfoMixin:
    """For a MiMoV2OmniProcessingInfo subclass: a data parser that keeps the loader metadata with the frames, and
    MIMO_VIDEO_MAX_PIXELS / MIMO_VIDEO_TOTAL_MAX_PIXELS (module docstring)."""

    def get_data_parser(self):
        from vllm.multimodal.parse import MultiModalDataParser

        return MultiModalDataParser(target_sr=24000.0, video_needs_metadata=True)

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        """Upstream's bounds, except a video's: its embedding rows grow with its length (2 fps, frames down to the
        processor's minimum size, its audio track at 6.25 tokens a second), so only the context bounds it. vLLM sizes
        its encoder cache from these bounds and refuses a longer item (upstream's 114688 refused a VGI-Bench
        clip, 124078 rows); the TT model encodes items itself (tt/multimodal.py), so that cache only counts here."""
        out = dict(super().get_mm_max_tokens_per_item(seq_len, mm_counts))
        out["video"] = max(out.get("video", 0), int(seq_len))
        return out

    def get_hf_processor(self, **kwargs):
        processor = super().get_hf_processor(**kwargs)
        p = processor.mimo_processor
        kw = p._vid_kw
        if VIDEO_MAX_PIXELS:
            kw["max_pixels"] = VIDEO_MAX_PIXELS
        if VIDEO_TOTAL_MAX_PIXELS:  # the processor divides it by frame pairs (process_video): half for "per frame"
            kw["total_max_pixels"] = VIDEO_TOTAL_MAX_PIXELS // (p.temporal_patch_size * p.temporal_compression_ratio)
        return processor


class VideoDummyMixin:
    """For a MiMoV2OmniDummyInputsBuilder subclass: dummy videos with loader metadata (1 fps, every frame), which the
    metadata-keeping parser requires."""

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        data = super().get_dummy_mm_data(seq_len, mm_counts, mm_options)
        videos = data.get("video")
        if videos:
            data = dict(data)
            data["video"] = [v if isinstance(v, tuple) else (v, dummy_metadata(len(v))) for v in videos]
        return data


def dummy_metadata(num_frames):
    return {
        "total_num_frames": num_frames,
        "fps": 1.0,
        "duration": float(num_frames),
        "video_backend": "opencv",
        "frames_indices": list(range(num_frames)),
        "do_sample_frames": False,
    }
