# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Streaming verified on **content**.

The bring-up scope is explicit about what this must not be:

> Streaming must compare *content*: concatenated streamed audio vs non-streamed
> audio for the same text and seed, gated on sample-level correlation and
> mel-space PCC — not chunk count.

Counting chunks proves the loop ran. It proves nothing about whether the audio is
the same speech, which is the thing that was actually in doubt.

**The plan asks for a sample-level gate and that gate is not achievable**, by this
port or by the original. Measured on the PyTorch reference's own streamed and
non-streamed audio for the same text and seed:

    sample corr  -0.0260      mel-space  0.6689      envelope  0.6562

The cause is f0-into-phase integration again. Chunked decoding produces a slightly
different mel; f0
derives from that mel; and f0 error *integrates* into excitation phase over tens
of thousands of samples. Holding the phase vector fixed across both runs -- which
this test does -- does not rescue it, because the divergence is in f0, not in the
initial phase.

So sample correlation is **reported next to the reference's own number** rather
than gated, and three things are gated instead:

* **mel-space PCC** — the content gate. Does the streamed audio say the same
  thing, with the same prosody?
* **envelope PCC** — the same question in the energy domain.
* **seam continuity** — the first difference of the waveform near a chunk
  boundary, against the utterance's own p99.9. A phase discontinuity at a seam is
  a step, and a step is an outlier here. This is what the three caches exist to
  prevent, and it is the only one of the three that tests them directly.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden, pcc
from models.demos.cosyvoice.tt.streaming import StreamConfig, hamming
from models.demos.cosyvoice.tt.weights import default_weights_path

HIFT_WEIGHTS = default_weights_path()
FLOW_WEIGHTS = HIFT_WEIGHTS.replace("hift_", "flow_")

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 131072}], indirect=True)
needs_weights = pytest.mark.skipif(
    not (os.path.exists(HIFT_WEIGHTS) and os.path.exists(FLOW_WEIGHTS)), reason="export flow and hift weights first"
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "e2e.npz")), reason="generate goldens first"
)


# --------------------------------------------------------------------------
# host tier
# --------------------------------------------------------------------------
def test_incremental_session_cuts_the_same_chunks_as_the_batch_loop():
    """Pushing tokens one at a time must schedule exactly the batch loop's chunks.

    This is what lets the pipelined path inherit the content guarantee below without
    re-proving it on device. `synthesize` and `synthesize_streaming` share one
    scheduler (`StreamSession`), and the claim that they cut identically is a pure
    property of the token count -- so it is checked here, on the host, over many
    lengths, instead of once on hardware at whatever length the golden happens to be.

    The synthesizer itself is stubbed: this is a test of the schedule, not of the
    flow decoder. What it must catch is an off-by-one in the hop or the overlap,
    which would leave the two paths producing different audio for the same tokens.
    """
    from models.demos.cosyvoice.tt.streaming import StreamConfig, StreamSession

    cfg = StreamConfig()

    class FakeSynth:
        """Records the token spans it is asked to synthesise."""

        def __init__(self):
            self.cfg = cfg
            self.spans = []

        def _one(self, chunk_tokens, ctx, state, rng, finalize):
            self.spans.append((tuple(chunk_tokens), finalize))
            return None, 0

    def batch_spans(tokens):
        """The loop `synthesize` used before the scheduler was extracted."""
        spans, i, hop = [], 0, cfg.token_hop_len
        while i + hop + cfg.token_overlap_len <= len(tokens):
            spans.append((tuple(tokens[i : i + hop + cfg.token_overlap_len]), False))
            i += hop
            hop = min(cfg.token_max_hop_len, int(hop * cfg.stream_scale_factor))
        spans.append((tuple(tokens[i:]), True))
        return spans

    # Lengths either side of every boundary the scheduler can trip on: shorter than
    # one chunk, exactly one chunk, one token over, and several chunks deep.
    for n in (0, 1, 119, 120, 121, 164, 219, 220, 221, 400, 1000):
        tokens = list(range(n))
        synth = FakeSynth()
        session = StreamSession(synth, ctx=None, rng_for_chunk=None)
        for t in tokens:
            list(session.push(t))
        session.finish()
        session.close()
        assert synth.spans == batch_spans(tokens), f"schedules diverge at {n} tokens"


def test_session_finish_is_not_reusable(expect_error):
    """A second `finish()` would re-vocode the tail against caches the first call
    already consumed, which is a use-after-free rather than a duplicate chunk."""
    from models.demos.cosyvoice.tt.streaming import StreamConfig, StreamSession

    class FakeSynth:
        cfg = StreamConfig()

        def _one(self, *a, **k):
            return None, 0

    session = StreamSession(FakeSynth(), ctx=None, rng_for_chunk=None)
    session.finish()
    with expect_error(RuntimeError, "called twice"):
        session.finish()


def test_window_is_symmetric_hamming_not_periodic():
    """`np.hamming` is the **symmetric** form; `scipy.signal.get_window` returns the
    periodic one by default. The discriminator is palindromy: the symmetric window
    satisfies `w[i] == w[N-1-i]`, the periodic one does not.

    Getting this wrong shifts every crossfade by a fraction of a frame, which is
    inaudible in isolation and accumulates across seams.
    """
    w = hamming(8)
    assert torch.allclose(w, w.flip(0), atol=1e-9), "symmetric window must be palindromic"
    assert abs(float(w[0]) - 0.08) < 1e-6, float(w[0])
    periodic = 0.54 - 0.46 * torch.cos(2 * 3.141592653589793 * torch.arange(8) / 8)
    assert not torch.allclose(periodic, periodic.flip(0), atol=1e-9), "the periodic form is not"


def test_the_reference_crossfade_does_not_preserve_level():
    """CosyVoice's crossfade sums to **1.06–1.08**, not 1.

    `fade_in_out` weights the incoming signal by `w[:n]` and the outgoing by
    `w[n:]`, and for a Hamming window those do not form a complementary pair -- the
    overlap region is boosted by up to 8 %. A Hann window would sum to unity, and
    substituting one would be an obvious "fix" that silently changes the output.

    Pinned rather than corrected: matching the reference is the requirement.
    """
    n = StreamConfig().mel_overlap_len
    w = hamming(2 * n)
    s = w[:n] + w[n:]
    assert float(s.min()) > 1.05 and float(s.max()) < 1.09, (float(s.min()), float(s.max()))
    assert not torch.allclose(s, torch.ones(n), atol=1e-2), "if this ever sums to 1, the window changed"


def test_stream_constants_match_the_reference():
    """Derived, not copied, so a change to the frame rate propagates."""
    cfg = StreamConfig()
    assert cfg.token_hop_len == 100 and cfg.token_max_hop_len == 200
    assert cfg.token_overlap_len == 20
    assert cfg.mel_overlap_len == 34, cfg.mel_overlap_len  # int(20/50 * 22050/256)
    assert cfg.mel_cache_len == 20
    assert cfg.source_cache_len == 5120  # 20 mel frames x hop 256
    assert cfg.chunk_size() == 120


@needs_golden
def test_the_golden_utterance_is_long_enough_to_chunk():
    """164 generated tokens against a 120-token chunk: two chunks, one seam.

    Stated because a test that silently produced a single chunk would pass every
    gate below while exercising none of the caching this file exists to check.
    """
    lr = load_golden("flow.length_regulator")
    n_generated = as_torch(lr["call0.in_x2"]).shape[1]
    cfg = StreamConfig()
    assert n_generated > cfg.chunk_size(), (n_generated, cfg.chunk_size())
    n_chunks = 1 + max(0, (n_generated - cfg.chunk_size()) // cfg.token_hop_len + 1)
    assert n_chunks >= 2, n_chunks


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
def _mel_of(wav: torch.Tensor, n_fft=1024, hop=256, n_mels=80) -> torch.Tensor:
    """A plain log-magnitude spectrogram for the mel-space gate.

    Not the model's own mel front-end: the point is to compare two *waveforms* in a
    perceptually weighted space, and any fixed linear-frequency magnitude basis
    does that. Using the model's filterbank would add a dependency for no extra
    discrimination.
    """
    w = torch.hann_window(n_fft)
    spec = torch.stft(wav.flatten(), n_fft=n_fft, hop_length=hop, window=w, return_complex=True)
    return torch.log(spec.abs().clamp_min(1e-5))


@needs_weights
@needs_golden
@needs_l1_small
def test_device_streamed_matches_non_streamed(device):
    """The same tokens and the same seed, chunked and whole, compared as audio.

    Both runs go through the identical flow decoder and vocoder on device; the only
    difference is that one is fed in 120-token chunks with the three caches carried
    across seams. The RNG is drawn per *mel frame count* from a fixed seed so the
    two runs see comparable noise -- they cannot see identical noise, because the
    chunked run vocodes different spans.
    """
    import ttnn
    from models.demos.cosyvoice.tt.flow.model import TtMaskedDiffWithXvec
    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator
    from models.demos.cosyvoice.tt.streaming import TtStreamingSynthesizer
    from models.demos.cosyvoice.tt.weights import WeightBag

    emb_g = load_golden("flow.input_embedding")
    lr_g = load_golden("flow.length_regulator")
    cfm_g = load_golden("flow.cfm")
    spk_g = load_golden("flow.spk_embed_affine")

    all_tokens = torch.from_numpy(emb_g["call0.in_tokens"]).to(torch.int32)
    token_len1 = as_torch(lr_g["call0.in_x1"]).shape[1]
    mel_len1 = int(lr_g["call0.in_mel_len1"])
    prompt_tokens = all_tokens[:, :token_len1]
    generated = all_tokens[0, token_len1:].tolist()
    prompt_feat = as_torch(cfm_g["call0.in_cond"])[:, :, :mel_len1].permute(0, 2, 1).contiguous()
    embedding = as_torch(spk_g["call0.in_x"]).reshape(1, 1, -1)

    flow_bag = WeightBag.load(FLOW_WEIGHTS)
    flow = TtMaskedDiffWithXvec(device, flow_bag, flow_bag.meta)
    hift = TtHiFTGenerator(device, WeightBag.load(HIFT_WEIGHTS))

    def dev(v, dtype=ttnn.bfloat16):
        return ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # ONE phase vector, shared by both runs and by every chunk. Seeding per chunk
    # would give the two runs different phases, and a different phase gives
    # ~0 sample correlation by construction -- the gate would then measure the RNG
    # rather than the streaming. Holding it fixed makes the sample gate a real test
    # of the excitation splice: with the splice the phase runs continuously across
    # a seam, without it every chunk restarts at this same offset and correlation
    # collapses.
    _g = torch.Generator().manual_seed(1986)
    _phase = torch.empty(1, 1, 9).uniform_(-3.141592653589793, 3.141592653589793, generator=_g)
    _phase[0, 0, 0] = 0.0

    def rng(mel_frames, seed=1986):
        g = torch.Generator().manual_seed(seed + mel_frames)
        return _phase, torch.randn(1, mel_frames * 256, 9, generator=g)

    def flow_chunk(tokens):
        """Flow-decode one token chunk with the shared prompt."""
        toks = torch.cat([prompt_tokens, torch.tensor(tokens, dtype=torch.int32).reshape(1, -1)], dim=1)
        mel_len2 = TtMaskedDiffWithXvec.mel_len_for(len(tokens))
        g = torch.Generator().manual_seed(1986 + len(tokens))
        z = torch.randn(1, mel_len1 + mel_len2, 80, generator=g)
        mel = flow.inference(
            ttnn.from_torch(toks, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            token_len1,
            mel_len1,
            mel_len2,
            dev(prompt_feat),
            dev(embedding),
            dev(z),
        )
        return mel, mel_len2

    # SimpleNamespace, not a class body: a class body does not close over the
    # enclosing function's scope, so `flow_chunk` would be undefined inside it.
    ctx = SimpleNamespace(flow_chunk=flow_chunk)

    # ---- non-streamed
    mel, frames = flow_chunk(generated)
    phase, noise = rng(frames)
    whole, n_whole, src = hift.inference(
        mel, frames, phase_vec=dev(phase, ttnn.float32), sine_noise_unit=dev(noise, ttnn.float32)
    )
    offline = ttnn.to_torch(whole).float().reshape(1, -1)
    for t in (mel, whole, src):
        ttnn.deallocate(t)

    # ---- streamed
    synth = TtStreamingSynthesizer(device, flow, hift)
    chunks = synth.synthesize(generated, ctx, rng)
    pieces = [ttnn.to_torch(c).float().reshape(1, -1) for c in chunks]
    for c in chunks:
        ttnn.deallocate(c)
    streamed = torch.cat(pieces, dim=1)

    n = min(offline.shape[1], streamed.shape[1])
    sample_corr = pcc(streamed[:, :n], offline[:, :n])
    mel_corr = pcc(_mel_of(streamed[:, :n]), _mel_of(offline[:, :n]))
    win = 256
    env = [x[:, :n][:, : n // win * win].reshape(-1, win).pow(2).mean(1).sqrt() for x in (streamed, offline)]
    env_corr = pcc(*env)

    # Seam continuity: the thing the three caches exist to produce. A phase
    # discontinuity at a chunk boundary is a step in the waveform, so it shows up
    # as an outlier in the first difference. Compared against the utterance's own
    # distribution rather than an absolute threshold, since speech is not smooth.
    d = (streamed[0, 1:] - streamed[0, :-1]).abs()
    seam = len(pieces[0][0])
    lo, hi = max(0, seam - 64), min(d.shape[0], seam + 64)
    seam_max = float(d[lo:hi].max())
    global_p999 = float(d.quantile(0.999))

    print(f"\n  streaming: {len(generated)} tokens -> {len(pieces)} chunks")
    print(f"    lengths  streamed {streamed.shape[1]}  non-streamed {offline.shape[1]}")
    print(f"    mel-space PCC        {mel_corr:.6f}   (the content gate)")
    print(f"    envelope  PCC        {env_corr:.6f}")
    print(f"    sample correlation   {sample_corr:.6f}   (reported, not gated -- see below)")
    print(f"    seam |diff| max      {seam_max:.5f}  vs utterance p99.9 {global_p999:.5f}")
    print(
        f"    RMS  streamed {float(streamed.pow(2).mean().sqrt()):.5f}"
        f"   non-streamed {float(offline.pow(2).mean().sqrt()):.5f}"
    )
    print("    PyTorch reference, same comparison: sample -0.0260, mel 0.6689, envelope 0.6562")

    assert len(pieces) >= 2, "the run produced a single chunk; nothing was streamed"
    assert streamed.shape[1] == offline.shape[1], (streamed.shape, offline.shape)
    assert mel_corr >= 0.85, mel_corr
    assert env_corr >= 0.85, env_corr
    # No step at the seam: the crossfades and the excitation splice did their job.
    assert seam_max <= 4 * global_p999, (seam_max, global_p999)
    # `sample_corr` is deliberately NOT gated. The plan asks for it, and it is not
    # achievable by any implementation of this architecture -- the PyTorch reference
    # scores -0.026 on the same comparison. Chunked decoding yields a slightly
    # different mel, f0 derives from that mel, and f0 error integrates into
    # excitation phase over tens of thousands of samples. Holding the phase
    # vector fixed across both runs, as this test does, does not rescue it. Reported
    # rather than dropped, and reported next to the reference's own number so the
    # comparison is legible.
    assert sample_corr > -0.5, sample_corr
