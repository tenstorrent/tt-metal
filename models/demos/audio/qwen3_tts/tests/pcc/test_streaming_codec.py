# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The codec decoder fed vLLM-Omni's streamed chunks, against one decode of the whole clip.

Chunks follow the PoC deploy's schedule: a ramp from 2 frames up to the steady 25, then a
partial last chunk. Each is decoded behind up to `STREAM_WINDOW` frames of its own history,
which approximates the one-shot decode rather than reproducing it (see `STREAM_WINDOW`).
The one-shot decode runs eagerly, so it runs before the streaming decoder captures its
traces. The last test runs the vLLM-Omni Code2Wav stage class through a request's end.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/pcc/test_streaming_codec.py
"""

from types import SimpleNamespace

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


def _code2wav_stage(device):
    from vllm.config import LoadConfig

    from models.demos.audio.qwen3_tts import weights
    from models.demos.audio.qwen3_tts.tt.code2wav_vllm import Qwen3TTSCode2WavForTT

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=weights.checkpoint_dir(),
            async_chunk=True,
            revision=None,
            dtype=torch.float32,
            stage_connector_config={"extra": {}},
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        load_config=LoadConfig(),
    )
    return Qwen3TTSCode2WavForTT(vllm_config=vllm_config, mesh_device=device)


def _step(stage, chunk, meta=None):
    """One runner step for request "r": `chunk` [1, 16, F] as Omni ships it, flat and
    quantizer-major."""
    flat = chunk.reshape(-1)
    output = stage(
        input_ids=flat,
        positions=None,
        runtime_additional_information=[{"meta": meta or {}}],
        seq_token_counts=[flat.numel()],
        request_ids=["r"],
    )
    return output.multimodal_outputs["model_outputs"][0]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_a_request_ending_on_a_chunk_boundary_finishes_on_an_empty_chunk(device, codes):
    """vLLM-Omni ends such a request with a chunk of no frames marked finished, and the
    runner reports the request finished. The request's decode history must go, so a later
    request reusing its id starts from nothing."""
    stage = _code2wav_stage(device)
    stream = stage.decoder.stream
    stream.warmup()
    sizes = RAMP[:4]
    edges = [sum(sizes[:index]) for index in range(len(sizes) + 1)]
    chunks = [codes[..., start:end] for start, end in zip(edges, edges[1:])]

    streamed = [_step(stage, chunk) for chunk in chunks]
    final = _step(stage, codes[..., :0], {"finished": True})
    stage.on_requests_finished({"r"})
    reused = _step(stage, chunks[0])

    state = {}
    direct = [stream.decode(chunk, state) for chunk in chunks]
    stream.release()

    assert [piece.numel() for piece in streamed] == [size * stream.upsample for size in sizes]
    for piece, expected in zip(streamed, direct):
        assert torch.equal(piece, expected)
    assert final.numel() == 0
    assert torch.equal(reused, direct[0])
