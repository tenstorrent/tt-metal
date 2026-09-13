"""S2Engine streaming contract without a device: the codec worker thread (_StreamDecoder) must emit segments in
order whose concatenation equals the full-clip decode, coalesce when the codec is slower than generation, and
shut down cleanly on cancel / codec errors. Uses a fake generator and a fake causal codec (CPU only)."""
import threading
import time

import numpy as np
import pytest
import torch

from models.autoports.fishaudio_s2_pro.server import engine as E

SPF = 16  # fake samples per frame


class FakeCodec:
    """Causal by construction: sample k of frame t depends only on frame t's codes."""

    def __init__(self, delay=0.0, fail_at=None):
        self.delay, self.fail_at, self.calls = delay, fail_at, []

    def decode(self, codes: torch.Tensor) -> np.ndarray:
        self.calls.append(int(codes.shape[1]))
        if self.fail_at is not None and codes.shape[1] >= self.fail_at:
            raise RuntimeError("codec boom")
        time.sleep(self.delay)
        if codes.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        per = codes.float().sum(0) / 1e4  # (T,)
        return (per[:, None] + torch.arange(SPF)[None, :] / 1e5).reshape(-1).numpy().astype(np.float32)


class FakeStats:
    frames = prompt_len = 0
    stopped_on_im_end = True
    prefill_s = decode_s = 0.0
    frames_per_s = 0.0


class FakeGen:
    def __init__(self, n_frames, step_delay=0.0, cancel_after=None, job=None):
        self.n, self.step_delay, self.cancel_after, self.job = n_frames, step_delay, cancel_after, job

    def generate(self, text, on_frame=None, **kw):
        frames = []
        for i in range(self.n):
            frame = [1000 + i, i % 7, *[(i * 3 + k) % 11 for k in range(9)]]
            frames.append(frame)
            if self.cancel_after is not None and i == self.cancel_after:
                self.job.cancel.set()
            on_frame(i, frame)
            time.sleep(self.step_delay)
        st = FakeStats()
        st.frames = len(frames)
        return torch.tensor(frames, dtype=torch.int64).T[1:], st


def _engine(codec, gen, first=4, chunk=8):
    e = E.S2Engine.__new__(E.S2Engine)
    e.log = lambda *a, **k: None
    e.codec, e.gen = codec, gen
    e.first_chunk_frames, e.chunk_frames = first, chunk
    e._references = lambda req: (None, None)
    return e


def _req(**kw):
    from models.autoports.fishaudio_s2_pro.server.schemas import ServeTTSRequest

    return ServeTTSRequest(text="hello", **kw)


def _drain(job):
    events = []
    while True:
        kind, payload = job.out.get(timeout=30)
        events.append((kind, payload))
        if kind in ("final", "error", "cancelled"):
            return events


@pytest.mark.parametrize(
    "n_frames,codec_delay,step_delay", [(37, 0.0, 0.0), (40, 0.05, 0.002), (3, 0.0, 0.0), (0, 0.0, 0.0)]
)
def test_stream_segments_concatenate_to_full_decode(n_frames, codec_delay, step_delay):
    codec = FakeCodec(delay=codec_delay)
    e = _engine(codec, FakeGen(n_frames, step_delay))
    job = E.Job(_req(streaming=True))
    e._serve(job)
    ev = _drain(job)
    assert ev[0][0] == "header" and ev[-1][0] == "final"
    segs = [p for k, p in ev if k == "segment"]
    wav, stats = ev[-1][1]
    ref = (
        FakeCodec().decode(
            torch.tensor(
                [[1000 + i, i % 7, *[(i * 3 + k) % 11 for k in range(9)]] for i in range(n_frames)], dtype=torch.int64
            ).T[1:]
        )
        if n_frames
        else np.zeros(0, dtype=np.float32)
    )
    streamed = np.concatenate(segs) if segs else np.zeros(0, dtype=np.float32)
    assert len(streamed) == n_frames * SPF == len(wav)
    assert np.array_equal(streamed, ref) and np.array_equal(wav, ref)
    assert all(len(s) > 0 for s in segs)
    if n_frames:
        assert codec.calls[-1] == n_frames  # the last decode is the full clip
        assert codec.calls == sorted(codec.calls)  # prefixes only grow
    if n_frames >= 4:
        assert segs[0].shape[0] == 4 * SPF  # first chunk = first_chunk_frames
    if codec_delay:
        # generation (40 x 2 ms) is much faster than the codec (50 ms/decode): boundaries must have been coalesced
        assert len(codec.calls) < 1 + (n_frames - 4) // 8 + 1
    assert stats["stream_decodes"] == len(codec.calls)
    assert threading.active_count() == 1 or not any(
        t.name == "s2-codec" and t.is_alive() for t in threading.enumerate()
    )


def test_non_streaming_unchanged():
    codec = FakeCodec()
    e = _engine(codec, FakeGen(20))
    job = E.Job(_req(streaming=False))
    e._serve(job)
    ev = _drain(job)
    assert [k for k, _ in ev] == ["final"] and codec.calls == [20]


def test_cancel_mid_stream_stops_worker():
    codec = FakeCodec(delay=0.02)
    job = E.Job(_req(streaming=True))
    e = _engine(codec, FakeGen(50, 0.001, cancel_after=20, job=job))
    raised = None
    try:
        e._serve(job)
    except E.Cancelled as ex:
        raised = ex
    assert isinstance(raised, E.Cancelled)
    assert not any(t.name == "s2-codec" and t.is_alive() for t in threading.enumerate())


def test_codec_error_surfaces_on_generation_thread():
    codec = FakeCodec(fail_at=12)
    e = _engine(codec, FakeGen(30))
    job = E.Job(_req(streaming=True))
    raised = None
    try:
        e._serve(job)
    except RuntimeError as ex:
        raised = ex
    assert raised is not None and "codec boom" in str(raised)
    assert not any(t.name == "s2-codec" and t.is_alive() for t in threading.enumerate())
