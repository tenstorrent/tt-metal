# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks on `MiniMaxH3Vae._decode`'s temporal assembly: the batch path gives the frames
the inline split did before `split_clip`, and the streamed path wires its assembly to the device decode."""

import numpy as np
import pytest
import torch

from ....models.vae.minimax_h3 import vae_minimax_h3 as vm
from ....models.vae.minimax_h3.temporal_minimax_h3 import (
    assemble_clip_parts,
    clip_frames,
    clip_num_frames,
)


def _vae(decode_clips, assemble_beside_decode=False):
    vae = object.__new__(vm.MiniMaxH3Vae)
    vae.config = vm.MiniMaxH3VaeConfig()
    vae.decoder = None
    vae.device_stitch = True
    vae.assemble_beside_decode = assemble_beside_decode
    vae._assembled_shapes = {}
    vae._ensure_loaded = lambda module, subfolder: None
    vae._decoder_subfolder = lambda: "vae"
    vae.reserve_decode_output = lambda *latent_key: None
    vae._report_profile = lambda total: None
    vae._decode_clips_device_stitched = decode_clips
    return vae


def _chunk_frames(config) -> int:
    return config.tokens_chunk_size * config.temporal_compression_ratio


def _fake_clips(output_type, num_frames, seed):
    rng = np.random.default_rng(seed)

    def decode_clips(chunk_latents, output_type_arg="float", on_clip=None):
        assert output_type_arg == output_type
        clips = []
        for _ in chunk_latents:
            if output_type == "yuv420":
                clip = rng.integers(0, 256, size=(num_frames, 6, 4), dtype=np.uint8)
            else:
                clip = torch.from_numpy(rng.standard_normal((1, 3, num_frames, 4, 4)).astype(np.float32))
            clips.append(clip)
            if on_clip is not None:
                on_clip(clip)
        return clips

    return decode_clips


def _reference(clips, config, latent_frames):
    """The batch assembly as written before `split_clip`."""
    chunk_num_frames = _chunk_frames(config)
    parts, overlap = [], None
    for clip in clips:
        for j in range(int(config.token_drop > 0) + 1):
            frame_start = j * chunk_num_frames
            chunk = clip_frames(clip, frame_start, frame_start + chunk_num_frames)
            chunk = clip_frames(chunk, config.frame_pre_padding, clip_num_frames(chunk))
            if j == 0:
                parts.append((chunk, overlap))
            else:
                overlap = chunk
    if overlap is not None:
        parts.append((overlap, None))
    result = assemble_clip_parts(parts, config.frame_overlap)
    chunk_size, temporal_ratio = config.tokens_chunk_size, config.temporal_compression_ratio
    pad_tokens = (-(latent_frames + config.token_drop)) % chunk_size
    intra_tail = config.clip_length % temporal_ratio
    pad_frames = sum(
        intra_tail if intra_tail and (latent_frames + k) % chunk_size == 0 else temporal_ratio
        for k in range(pad_tokens)
    )
    return clip_frames(result, 0, clip_num_frames(result) - pad_frames)


@pytest.mark.parametrize("output_type", ["yuv420", "float"])
@pytest.mark.parametrize("latent_frames", [7, 12, 16, 31])
def test_the_batch_assembly_is_unchanged(output_type, latent_frames):
    config = vm.MiniMaxH3VaeConfig()
    recorded = []
    fake = _fake_clips(output_type, 2 * _chunk_frames(config), seed=latent_frames)

    def decode_clips(chunk_latents, output_type_arg="float", on_clip=None):
        assert on_clip is None
        clips = fake(chunk_latents, output_type_arg)
        recorded.extend(clips)
        return clips

    vae = _vae(decode_clips)
    result = vae._decode(torch.zeros(1, 1, latent_frames, 2, 2), output_type)
    expected = _reference(recorded, config, latent_frames)
    if output_type == "yuv420":
        np.testing.assert_array_equal(result, expected)
    else:
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("assemble_beside_decode", [True, False])
@pytest.mark.parametrize("latent_frames", [7, 16, 31])
def test_a_decode_with_no_consumer_can_assemble_beside_the_device_decode(latent_frames, assemble_beside_decode):
    config = vm.MiniMaxH3VaeConfig()
    recorded = []
    fake = _fake_clips("yuv420", 2 * _chunk_frames(config), seed=latent_frames)

    def decode_clips(chunk_latents, output_type_arg="float", on_clip=None):
        assert (on_clip is not None) == assemble_beside_decode  # beside: each clip reaches the assembly as it lands
        clips = fake(chunk_latents, output_type_arg, on_clip)
        recorded.extend(clips)
        return clips

    vae = _vae(decode_clips, assemble_beside_decode)
    result = vae._decode(torch.zeros(1, 1, latent_frames, 2, 2), "yuv420")
    np.testing.assert_array_equal(result, _reference(recorded, config, latent_frames))


class _FakeStream:
    made: list = []

    def __init__(self, num_clips, **kwargs):
        self.num_clips = num_clips
        self.kwargs = kwargs
        self.added = []
        self.closed = False
        _FakeStream.made.append(self)

    def add(self, clip):
        self.added.append(clip)

    def result(self):
        return np.zeros((40, 6, 4), dtype=np.uint8)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_stream(monkeypatch):
    _FakeStream.made = []
    monkeypatch.setattr(vm, "StreamingClipAssembly", _FakeStream)
    return _FakeStream


def test_the_streamed_decode_hands_each_clip_to_the_assembly(fake_stream):
    config = vm.MiniMaxH3VaeConfig()
    seen = {}

    def decode_clips(chunk_latents, output_type="float", on_clip=None):
        seen["on_clip"] = on_clip
        return _fake_clips("yuv420", 2 * _chunk_frames(config), seed=0)(chunk_latents, output_type, on_clip)

    vae = _vae(decode_clips)
    on_frames = lambda frames: None
    # 16 latents plus the 3 dropped tokens pad to 20: one padding token, 4 frames cut off the end.
    vae._decode(torch.zeros(1, 1, 16, 2, 2), "yuv420", on_frames=on_frames)

    (stream,) = fake_stream.made
    assert seen["on_clip"] == stream.add
    assert stream.kwargs["trim"] == 4
    assert stream.kwargs["on_frames"] is on_frames
    assert stream.num_clips == len(stream.added) == 3
    assert stream.closed


def test_the_streamed_decode_closes_the_assembly_when_the_device_raises(fake_stream):
    def decode_clips(chunk_latents, output_type="float", on_clip=None):
        raise RuntimeError("device gone")

    vae = _vae(decode_clips)
    with pytest.raises(RuntimeError, match="device gone"):
        vae._decode(torch.zeros(1, 1, 16, 2, 2), "yuv420", on_frames=lambda frames: None)
    (stream,) = fake_stream.made
    assert stream.closed
