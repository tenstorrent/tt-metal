# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device Kokoro ISTFTNet back-half (decoder + generator + iSTFT) validation.

``tt/device_pipeline.py`` ports the acoustic decoder / vocoder to TTNN. These
tests run it on device and compare the produced 24 kHz waveform against the torch
reference (the ``kokoro`` ``KModel`` decoder), given identical acoustic-feature
inputs.

Correctness is measured in the **STFT log-magnitude domain**, not on the raw
waveform. The on-device ``sinegen_device`` uses a deterministic harmonic-phase
model that intentionally omits the reference ``SineGen``'s random initial phase
and voiced/unvoiced phase-reset (istftnet.py), so the two waveforms are
perceptually/spectrally equivalent but phase-decorrelated — raw-waveform PCC is
~0.12 and meaningless here, while STFT log-magnitude PCC is ~0.98.

    pytest --disable-warnings models/demos/audio/kokoro/tests/test_device_pipeline.py
"""
import contextlib
import sys
import types

import numpy as np
import pytest
import torch

# Stub spaCy before importing kokoro (misaki.en imports spacy; numpy-2 ABI vs the
# numpy 1.26 TTNN is built against). Only the espeak G2P path is used.
if "spacy" not in sys.modules:
    sys.modules["spacy"] = types.ModuleType("spacy")

from models.demos.audio.kokoro.tt.device_pipeline import KokoroDevicePipeline

MODEL_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
TEXT = "Kokoro runs on Tenstorrent."
SPEC_PCC_BAR = 0.95  # STFT log-magnitude PCC vs torch (measured ~0.977)


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    return float(np.corrcoef(a, b)[0, 1])


def _logmag_pcc(a, b, n_fft=512, hop=160):
    """Phase-invariant spectral similarity: PCC of log-magnitude STFTs."""
    a = torch.tensor(np.asarray(a), dtype=torch.float32).ravel()
    b = torch.tensor(np.asarray(b), dtype=torch.float32).ravel()
    n = min(a.numel(), b.numel())
    win = torch.hann_window(n_fft)
    la = torch.log(torch.stft(a[:n], n_fft, hop, window=win, return_complex=True).abs() + 1e-5)
    lb = torch.log(torch.stft(b[:n], n_fft, hop, window=win, return_complex=True).abs() + 1e-5)
    return float(np.corrcoef(la.numpy().ravel(), lb.numpy().ravel())[0, 1])


@contextlib.contextmanager
def _deterministic_source():
    """Make the reference ISTFTNet source deterministic for apples-to-apples PCC.

    ``SineGen._f02sine`` adds a random initial phase (``torch.rand``) to the
    harmonics and the source adds ``torch.randn`` noise; the on-device
    ``sinegen_device`` is deterministic (zero phase, no added noise). Without this
    the waveforms are perceptually equivalent but phase-decorrelated, so raw
    waveform PCC is meaningless. Zeroing both here aligns the reference with the
    device so PCC measures the decoder's numeric fidelity.
    """
    real_rand, real_randn = torch.rand, torch.randn
    torch.rand = lambda *s, **k: torch.zeros(*s, **k)
    torch.randn = lambda *s, **k: torch.zeros(*s, **k)
    try:
        yield
    finally:
        torch.rand, torch.randn = real_rand, real_randn


@pytest.fixture(scope="module")
def kmodel():
    from kokoro.model import KModel

    return KModel(repo_id=MODEL_ID).eval()


@pytest.fixture(scope="module")
def inputs(kmodel):
    """Return (input_ids[1,S], ref_s[1,256]) for a fixed utterance."""
    from huggingface_hub import hf_hub_download
    from misaki.espeak import EspeakFallback

    class _Utt:
        def __init__(self, text):
            self.text = text

    g2p = EspeakFallback(british=False)
    phonemes, _ = g2p(_Utt(TEXT))
    ids = [i for i in (kmodel.vocab.get(p) for p in phonemes) if i is not None]
    input_ids = torch.LongTensor([[0, *ids, 0]])
    pack = torch.load(hf_hub_download(MODEL_ID, f"voices/{DEFAULT_VOICE}.pt"), weights_only=True)
    ref_s = pack[len(ids) - 1]  # [1,256]
    return input_ids, ref_s


def _reference_decoder_inputs(kmodel, input_ids, ref_s):
    """Run the host front half, capture the decoder args + reference audio.

    ``KModel.decoder`` is an ``nn.Module`` so it can't be reassigned to a plain
    function; a forward pre-hook captures its positional inputs (asr, F0, N, s)
    without perturbing the reference forward.
    """
    captured = {}
    handle = kmodel.decoder.register_forward_pre_hook(lambda m, args: captured.__setitem__("args", args))
    try:
        with _deterministic_source():
            ref_audio, _ = kmodel.forward_with_tokens(input_ids, ref_s, 1.0)
    finally:
        handle.remove()
    return captured["args"], ref_audio.detach().cpu().numpy()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_decoder_pcc(device, kmodel, inputs):
    """On-device decoder/generator/iSTFT vs torch reference (identical inputs)."""
    input_ids, ref_s = inputs
    (asr, F0, N, s), ref_audio = _reference_decoder_inputs(kmodel, input_ids, ref_s)

    pipe = KokoroDevicePipeline(kmodel, device)
    dev_audio = pipe.decoder(asr, F0, N, s).detach().cpu().numpy()

    assert dev_audio.size > 0
    # length should match the reference decoder to within a couple of STFT hops
    assert abs(dev_audio.size - ref_audio.size) <= 2 * 5, (dev_audio.size, ref_audio.size)
    spec = _logmag_pcc(ref_audio, dev_audio)
    print(
        f"device-decoder vs torch: log-mag PCC={spec:.5f} raw-waveform PCC={_pcc(ref_audio, dev_audio):.5f} "
        f"(ref {ref_audio.size}, dev {dev_audio.size})"
    )
    assert spec >= SPEC_PCC_BAR, f"log-mag PCC {spec:.5f} < {SPEC_PCC_BAR}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_synthesize_end_to_end(device, kmodel, inputs):
    """The synthesize() entrypoint produces non-silent audio matching the reference."""
    input_ids, ref_s = inputs
    with _deterministic_source():
        ref_audio, _ = kmodel.forward_with_tokens(input_ids, ref_s, 1.0)
    ref_audio = ref_audio.detach().cpu().numpy()

    pipe = KokoroDevicePipeline(kmodel, device)
    dev_audio = pipe.synthesize(input_ids, ref_s, 1.0).detach().cpu().numpy()

    dur = dev_audio.size / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(dev_audio.astype(np.float64) ** 2)))
    spec = _logmag_pcc(ref_audio, dev_audio)
    print(f"synthesize: {dur:.2f}s, rms={rms:.4f}, log-mag PCC vs torch={spec:.5f}")
    assert dur > 0.3, f"implausibly short audio: {dur:.2f}s"
    assert rms > 0.01, f"near-silent audio: rms={rms:.4f}"
    assert spec >= SPEC_PCC_BAR, f"log-mag PCC {spec:.5f} < {SPEC_PCC_BAR}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_front_half_device(device, kmodel, inputs):
    """Acoustic front half on device vs torch reference (identical plbert input)."""
    input_ids, ref_s = inputs
    cap = {}
    handle = kmodel.decoder.register_forward_pre_hook(lambda m, a: cap.__setitem__("args", a))
    try:
        _, ref_dur = kmodel.forward_with_tokens(input_ids, ref_s, 1.0)
    finally:
        handle.remove()
    asr_r, F0_r, N_r, _ = cap["args"]

    pipe = KokoroDevicePipeline(kmodel, device)
    # host plbert here (tt_plbert=False) isolates the ported front half for a tight
    # apples-to-apples PCC; the all-device path (TT plbert) is covered end-to-end below.
    asr_d, F0_d, N_d, _, dur_d = pipe.front_half_device(input_ids, ref_s, 1.0, tt_plbert=False)

    assert torch.equal(dur_d, ref_dur.reshape(-1)), "predicted durations diverged (alignment would differ)"
    pa, pf, pn = _pcc(asr_r, asr_d), _pcc(F0_r, F0_d), _pcc(N_r, N_d)
    print(f"front-half PCC vs torch: asr={pa:.5f} F0={pf:.5f} N={pn:.5f}")
    assert min(pa, pf, pn) >= 0.99, f"front-half PCC too low: asr={pa:.5f} F0={pf:.5f} N={pn:.5f}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_synthesize_device_end_to_end(device, kmodel, inputs):
    """Fully-on-device pipeline (TT plbert + front half + back half).

    Two checks, because durations are quantized (round) and TT plbert's ~0.1% numeric
    error can flip a single duration by +/-1 frame — which shifts the whole alignment
    and makes a raw frame-aligned comparison unfair:
    1. Fidelity: pin the alignment to the reference durations so the audio is
       frame-aligned, then require STFT log-magnitude PCC >= bar. This validates the
       all-device compute (TT plbert included) end to end.
    2. Free-running: predict durations on device too; require valid audio and a length
       within 10% of the reference (durations must stay within a frame or two).
    """
    input_ids, ref_s = inputs
    with _deterministic_source():
        ref_audio, ref_dur = kmodel.forward_with_tokens(input_ids, ref_s, 1.0)
    ref_audio = ref_audio.detach().cpu().numpy()
    ref_dur = ref_dur.reshape(-1)

    pipe = KokoroDevicePipeline(kmodel, device)

    aligned = pipe.synthesize_device(input_ids, ref_s, 1.0, pred_dur=ref_dur).detach().cpu().numpy()
    spec = _logmag_pcc(ref_audio, aligned)
    print(f"all-device (aligned): {aligned.size / SAMPLE_RATE:.2f}s log-mag PCC={spec:.5f}")
    assert spec >= SPEC_PCC_BAR, f"all-device log-mag PCC {spec:.5f} < {SPEC_PCC_BAR}"

    free = pipe.synthesize_device(input_ids, ref_s, 1.0).detach().cpu().numpy()
    rms = float(np.sqrt(np.mean(free.astype(np.float64) ** 2)))
    len_err = abs(free.size - ref_audio.size) / ref_audio.size
    print(f"all-device (free-running): {free.size / SAMPLE_RATE:.2f}s rms={rms:.4f} len_err={len_err:.3f}")
    assert rms > 0.01, f"near-silent audio: rms={rms:.4f}"
    assert len_err < 0.10, f"free-running length off by {len_err:.1%} (durations diverged)"
