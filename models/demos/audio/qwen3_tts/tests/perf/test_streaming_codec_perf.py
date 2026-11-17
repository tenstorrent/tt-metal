# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""What one streamed Code2Wav chunk costs to decode, warm, behind a full window of history.

The Code2Wav stage decodes a step's requests one after another, so a chunk's decode time
times the streams on the device is how long each stream waits for its next chunk. Chunk
sizes are the PoC deploy's: its ramp's ends, a mid-ramp size and the steady 25.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/perf/test_streaming_codec_perf.py
"""

import time

import pytest
import torch

from models.demos.audio.qwen3_tts.tests.reference_helpers import codebook_size
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import STREAM_WINDOW, StreamingCodecDecoder

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 50_000_000}]
REPEATS = 8

# Ceilings in ms, ~30% over Wormhole medians (37, 52, 62, 75), loose because the card is
# shared. At the steady 25 frames, 8 streams on one device take ~0.6 s per 2 s of audio.
# Convolutions over the whole window cost ~213 ms here at every chunk size.
MAX_CHUNK_MS = {3: 50.0, 6: 70.0, 15: 85.0, 25: 100.0}


@pytest.mark.slow
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_streamed_chunk_decode(device):
    stream = StreamingCodecDecoder(device)
    stream.warmup()
    codes = torch.randint(0, codebook_size(), (1, 16, STREAM_WINDOW + 25), generator=torch.Generator().manual_seed(0))

    rows, failed = [], []
    for frames, ceiling in MAX_CHUNK_MS.items():
        times = []
        for _ in range(REPEATS):
            state = {"codes": codes[..., :STREAM_WINDOW]}
            start = time.perf_counter()
            stream.decode(codes[..., STREAM_WINDOW : STREAM_WINDOW + frames], state)
            times.append(1000 * (time.perf_counter() - start))
        median = sorted(times)[REPEATS // 2]
        rows.append(f"  {frames:6d} {median:9.1f} {ceiling:9.1f}")
        if median > ceiling:
            failed.append(f"{frames} frames: {median:.1f} ms > {ceiling} ms")
    stream.release()

    print(f"\n  {'frames':>6s} {'median':>9s} {'ceiling':>9s}   (ms, history {STREAM_WINDOW} frames)")
    print("\n".join(rows))
    assert not failed, "; ".join(failed)
