# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The codec decoder fed vLLM-Omni's streamed chunks, against one decode of the whole clip.

Chunks follow the PoC deploy's schedule: a ramp from 2 frames up to the steady 25, then a
partial last chunk. Each is decoded behind up to `STREAM_WINDOW` frames of its own history,
which approximates the one-shot decode rather than reproducing it (see `STREAM_WINDOW`).
The one-shot decode runs eagerly, so it runs before the streaming decoder captures its
traces.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/pcc/test_streaming_codec.py
"""

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.audio.qwen3_tts.tests.reference_helpers import codebook_size
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import (
    STREAM_WINDOW,
    StreamingCodecDecoder,
    preprocess_codec_parameters,
)

RAMP = [2, 2, 3, 4, 6, 8, 11, 15, 20, 25]
STEADY = 25
FRAMES = 200
# Measured 0.99873 on Wormhole with random codes. This decoder against the fp32 reference
# scores 0.995 on random codes (`test_codec_pcc.py`), so the window's error is the smaller.
STREAM_PCC = 0.998


def _schedule(total):
    sizes = []
    for size in RAMP + [STEADY] * total:
        if sum(sizes) >= total:
            break
        sizes.append(min(size, total - sum(sizes)))
    return sizes


@pytest.fixture(scope="module")
def codes():
    return torch.randint(0, codebook_size(), (1, 16, FRAMES), generator=torch.Generator().manual_seed(0))


DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 50_000_000}]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_streamed_chunks_track_one_decode_of_the_clip(device, codes):
    stream = StreamingCodecDecoder(device, preprocess_codec_parameters(device))
    whole = stream.codec.decode(codes, bucket=1).reshape(-1)
    stream.warmup()

    state, pieces, start = {}, [], 0
    for size in _schedule(FRAMES):
        pieces.append(stream.decode(codes[..., start : start + size], state))
        start += size
    stream.release()
    streamed = torch.cat(pieces)

    assert streamed.numel() == whole.numel()
    passed, pcc = comp_pcc(whole, streamed, STREAM_PCC)
    print(f"\n  {len(pieces)} chunks, window {STREAM_WINDOW}: PCC {pcc}")
    assert passed, f"streamed PCC {pcc} < {STREAM_PCC}"
    assert state["codes"].shape[-1] == STREAM_WINDOW


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_reference_frames_are_context_only(device, codes, expect_error):
    stream = StreamingCodecDecoder(device, preprocess_codec_parameters(device))
    stream.warmup()
    first = codes[..., :12]
    full = stream.decode(first, {})
    skipped = stream.decode(first, {}, skip=10)
    assert torch.equal(skipped, full[-2 * stream.upsample :])
    assert stream.decode(first, {}, skip=12).numel() == 0

    with expect_error(ValueError, "skip <= frames"):
        stream.decode(first, {}, skip=13)
    with expect_error(ValueError, "skip <= frames"):
        stream.decode(codes[..., : STREAM_WINDOW + 1], {})
    stream.release()
