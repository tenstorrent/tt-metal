# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the opt-in traced conv VAE decode and its stage timer. No device needed."""

import pytest
import torch
from loguru import logger

from models.tt_dit.models.vae import vae_ltx
from models.tt_dit.models.vae.vae_ltx import LTXVideoDecoder


class _FakeTracer:
    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.fn(*args)


def _decoder(monkeypatch, trace_env):
    if trace_env is None:
        monkeypatch.delenv("LTX_VIDEO_VAE_TRACE", raising=False)
    else:
        monkeypatch.setenv("LTX_VIDEO_VAE_TRACE", trace_env)
    monkeypatch.setattr(vae_ltx, "Tracer", _FakeTracer)
    monkeypatch.setattr(vae_ltx.ttnn, "synchronize_device", lambda device: None)
    # Only the attributes forward reads are set; the decode, upload and readback are stubbed below.
    dec = LTXVideoDecoder.__new__(LTXVideoDecoder)
    dec.trace_decode = vae_ltx.os.environ.get("LTX_VIDEO_VAE_TRACE", "0") == "1"
    dec._vae_traced = False
    dec._decode_tracer = None
    dec.mesh_device = object()
    dec.eager_calls = 0

    def decode_device(x, h, w):
        dec.eager_calls += 1
        return x + 1

    dec.decode_device = decode_device
    dec._upload = lambda sample: (sample, 4, 6)
    dec._to_host = lambda sample_tt, output_type: sample_tt * 2
    return dec


def test_trace_flag_defaults_off_in_constructor_source():
    # The constructor reads the flag with a "0" default, so an unset variable keeps the eager decode.
    import inspect

    src = inspect.getsource(LTXVideoDecoder.__init__)
    assert 'os.environ.get("LTX_VIDEO_VAE_TRACE", "0") == "1"' in src
    assert "self._vae_traced = False" in src


@pytest.mark.parametrize(
    "trace_env,warm,expect_traced",
    [
        (None, True, False),  # flag unset: eager even when the pipeline marked the decoder warm
        ("1", False, False),  # flag set, decoder not yet warm (warmup/compile pass): eager
        ("1", True, True),  # flag set and warm: the decode goes through the tracer
    ],
)
def test_forward_traces_only_when_flagged_and_warm(monkeypatch, trace_env, warm, expect_traced):
    monkeypatch.delenv("LTX_TIME_STAGES", raising=False)
    dec = _decoder(monkeypatch, trace_env)
    dec._vae_traced = warm
    out = dec.forward(torch.zeros(2), output_type="yuv")
    assert torch.equal(out, torch.full((2,), 2.0))
    assert (dec._decode_tracer is not None) is expect_traced
    # The fake tracer calls through to decode_device, so the eager counter moves on both paths.
    assert dec.eager_calls == 1
    if expect_traced:
        assert dec._decode_tracer.calls == 1
        dec.forward(torch.zeros(2), output_type="yuv")
        assert dec._decode_tracer.calls == 2  # the tracer is built once and replayed


def test_explicit_traced_argument_still_wins(monkeypatch):
    monkeypatch.delenv("LTX_TIME_STAGES", raising=False)
    dec = _decoder(monkeypatch, None)
    dec.forward(torch.zeros(2), output_type="yuv", traced=True)
    assert dec._decode_tracer is not None and dec._decode_tracer.calls == 1


@pytest.mark.parametrize("time_stages", [None, "1"])
def test_stage_split_logged_only_under_time_stages(monkeypatch, time_stages):
    if time_stages is None:
        monkeypatch.delenv("LTX_TIME_STAGES", raising=False)
    else:
        monkeypatch.setenv("LTX_TIME_STAGES", time_stages)
    dec = _decoder(monkeypatch, "1")
    dec._vae_traced = True
    syncs = []
    monkeypatch.setattr(vae_ltx.ttnn, "synchronize_device", lambda device: syncs.append(device))
    lines = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="INFO")
    try:
        dec.forward(torch.zeros(2), output_type="yuv")
    finally:
        logger.remove(sink)
    split = [line for line in lines if line.startswith("VAE_DECODE_SPLIT")]
    if time_stages is None:
        assert split == [] and syncs == []
    else:
        assert len(split) == 1 and len(syncs) == 4
        assert split[0].startswith("VAE_DECODE_SPLIT traced=1 upload=")
        for key in ("decode=", "output=", "total="):
            assert key in split[0]
