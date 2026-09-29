# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""warmup() must leave nothing for a request to compile: every prefill shape and codec bucket.

Its own module on purpose: the test opens a fresh device so the reported warm-up time is not
inherited from another test's program cache, and a second device open while test_perf.py's
module-scoped pipeline still holds the chip leaves the device unrecoverable (STATUS 6.81).

Run:
    pytest -svv models/experimental/voxtral_tts/tests/perf/test_warmup.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device  # noqa: E402


@pytest.mark.slow
def test_warmup_compiles_every_prefill_shape_and_codec_bucket():
    """Warmup must leave nothing for a request to compile: every prefill shape and codec bucket.

    Opens its own device so the reported time is not inherited from another test's warm cache.
    """
    from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt

    d = open_device()
    try:
        p = TtVoxtralPipeline(d)
        assert p.warmed == {}, "warmed should be empty before warmup()"
        p.warmup(verbose=True)
        w = p.warmed
        step = gpt.PREFILL_MULTIPLE
        expected_shapes = list(range(step, p.backbone.max_seq_len + 1, step))
        print(
            f"\n  warmup {w['seconds']:.1f}s: {len(w['prefill_shapes'])} prefill shapes, "
            f"{len(w['codec_buckets'])} codec buckets, traced={w['traced']}"
        )
        assert w["prefill_shapes"] == expected_shapes, (
            f"warmup compiled {len(w['prefill_shapes'])} of {len(expected_shapes)} prefill shapes: "
            f"missing {sorted(set(expected_shapes) - set(w['prefill_shapes']))}"
        )
        assert w["codec_buckets"], "no codec bucket compiled"
        assert w["codec_buckets"][0] == (p.codec.bucket or 1)
        assert w["traced"], "the frame-loop trace was not captured, so generate() still pays it"
        p.close()
    finally:
        ttnn.close_device(d)
