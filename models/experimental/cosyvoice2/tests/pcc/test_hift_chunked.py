# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chunked HiFT (tt/hifigan/chunking.py): the schedule and the stitching on the host, and the seam gate on device.

The seam gate (`COSYVOICE2_HIFT_STREAM_REF`, scripts/hift_streaming_reference.py's --out-dir; skipped without it):
upstream's own HiFT ran each case on the same schedule, with its streaming cache, its crossfade and one fixed
sine-noise draw. TT runs the same mel with the same noise three ways:
- **mechanism**: upstream's F0 for each call injected, so what is left is the chunking itself (source carry-over,
  crossfade) plus the port's own HiFT error. Gated at every seam on the error relative to the signal over the
  crossfade, and on PCC around it; over the whole signal on PCC. (max |diff| is printed, not gated: it scales with
  the signal, and voiced seams are loud.)
- **negative control**: the same, without the crossfade (a hard switch at each seam). It must fail the seam gate at
  every seam. The cases put every crossfade in voiced speech, where a missing crossfade has something to show.
  Where upstream's own two calls already agree over the crossfade, the control fails mostly on the crossfade's gain
  (its Hamming halves sum to about 1.08); the table prints that agreement for each seam.
- **own F0**: TT's F0 predictor, as the pipeline runs it. F0 differences drift the sine phase, so it is gated on
  log-mel L1, over the whole signal and around each seam.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch

from models.experimental.cosyvoice2.tt.hifigan.chunking import (
    CHUNK_FRAMES,
    HOP,
    OVERLAP_FRAMES,
    chunk_schedule,
    speech_window,
    stitch,
)


def test_chunk_schedule_covers_every_frame_once(expect_error):
    for frames in (512, 513, 600, 1015, 1016, 1017, 1519, 1520, 1521, 3000, 5120):
        sched = chunk_schedule(frames)
        # every call is exactly one chunk, inside the mel, and the last one ends at the mel's end
        assert all(c.start >= 0 and c.start + CHUNK_FRAMES <= frames for c in sched)
        assert sched[-1].start + CHUNK_FRAMES == frames
        # the emitted spans tile [0, frames) exactly, in order
        spans = [(c.start + c.fade_from, c.start + c.emit_to) for c in sched]
        assert spans[0][0] == 0 and spans[-1][1] == frames
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
        # every later call carries the source over its whole overlap with the previous one (at least upstream's 8
        # frames), and crossfades over the previous call's held-back last 8 frames
        for prev, c in zip(sched, sched[1:]):
            assert c.carry >= OVERLAP_FRAMES and c.start + c.carry == prev.start + CHUNK_FRAMES
            assert c.fade_from == c.carry - OVERLAP_FRAMES and prev.emit_to == CHUNK_FRAMES - OVERLAP_FRAMES
    assert [(c.start, c.carry) for c in chunk_schedule(1016)] == [(0, 0), (504, 8)]  # lines up: no anchoring
    assert [(c.start, c.carry) for c in chunk_schedule(600)] == [(0, 0), (88, 424)]  # the last call anchored
    assert len(chunk_schedule(512)) == 1
    with expect_error(ValueError, "shorter than one chunk"):
        chunk_schedule(511)


def test_stitch_is_upstreams_crossfade():
    """Calls cut from one continuous signal stitch back to it: exactly outside the crossfades, and inside each one
    to upstream's `fade_in_out` of the two copies. With crossfade=False (the negative control) the seam is a hard
    switch."""
    frames = 1300
    x = torch.from_numpy(np.random.default_rng(0).standard_normal(frames * HOP).astype(np.float32))
    sched = chunk_schedule(frames)
    outs = [x[c.start * HOP : (c.start + CHUNK_FRAMES) * HOP].clone() for c in sched]
    got = stitch(outs, sched)
    assert got.shape == x.shape
    w = speech_window()
    n = len(w) // 2
    gain = torch.from_numpy(w[:n] + w[n:])  # upstream's Hamming crossfade sums to ~1.08, not 1
    fades = [(c.start + c.fade_from) * HOP for c in sched[1:]]
    mask = torch.ones_like(x, dtype=torch.bool)
    for f in fades:
        mask[f : f + n] = False
        assert torch.allclose(got[f : f + n].double(), x[f : f + n].double() * gain, atol=1e-6)
    assert torch.equal(got[mask], x[mask])
    # hard switch: identical copies make a seamless (unscaled) signal
    assert torch.equal(stitch(outs, sched, crossfade=False), x)


REF_DIR = os.environ.get("COSYVOICE2_HIFT_STREAM_REF", "")
SEAM_PAD = 960  # 40 ms either side of the 160 ms crossfade
# The gate (docs/VALIDATION.md, "Chunked HiFT"), on the six voiced-seam cases (2026-09-29):
# - mechanism, over each crossfade (3,840 samples): ||TT - upstream|| / ||upstream|| <= 0.10 (measured 0.041-0.078);
#   the no-crossfade control must exceed it at every seam (measured 0.108-0.473);
# - mechanism, per seam window (the crossfade +-40 ms): PCC >= 0.995 (measured min 0.99763);
# - mechanism, whole signal: PCC >= 0.995 (measured min 0.99641, a high-F0 voice: the port's own HiFT error with F0
#   injected, not the chunking, whose seam there measures 0.044);
# - own F0: whole-signal log-mel L1 <= 0.13 (measured 0.076-0.097), and no seam window above 1.5x its utterance's
#   whole-signal L1 (measured ratio at most 1.24).
SEAM_REL_ERR, SEAM_PCC, WHOLE_PCC = 0.10, 0.995, 0.995
OWN_F0_LOGMEL_L1_WHOLE, OWN_F0_SEAM_OVER_WHOLE = 0.13, 1.5


def _logmel(y: torch.Tensor) -> torch.Tensor:
    """80-bin log-mel at the model's own frame rate (1920/480, 0-8 kHz): `[80, frames]`."""
    from librosa.filters import mel as librosa_mel_fn

    basis = torch.from_numpy(librosa_mel_fn(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)).float()
    y = torch.nn.functional.pad(y.reshape(1, 1, -1), (720, 720), mode="reflect").reshape(1, -1)
    spec = torch.stft(y, 1920, hop_length=480, win_length=1920, window=torch.hann_window(1920), center=False,
                      return_complex=True)  # fmt: skip
    return torch.log(torch.clamp(basis @ spec.abs(), min=1e-5))[0]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(np.corrcoef(a.double().numpy(), b.double().numpy())[0, 1])


@pytest.mark.skipif(
    not REF_DIR, reason="set COSYVOICE2_HIFT_STREAM_REF to scripts/hift_streaming_reference.py's out dir"
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.timeout(0)  # a device job is never killed mid-op (pytest.ini sets 300 s)
def test_device_chunked_hift_seams_match_upstream_streaming(device):
    import ttnn
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
    from models.experimental.cosyvoice2.tt.hifigan.conv import config_tensors_in_dram_override
    from models.experimental.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
    from models.experimental.cosyvoice2.tt.hifigan.generator import (
        TorchHiFTDecodeRef,
        TorchHiFTGeneratorInferenceRef,
        TtHiFTDecoder,
        TtHiFTGenerator,
    )

    cases = sorted(glob.glob(os.path.join(REF_DIR, "*.npz")))
    assert cases, REF_DIR
    with config_tensors_in_dram_override(True):  # as the pipeline builds it
        hift_sd = load_checkpoint_file("hift.pt")
        decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
        ref = TorchHiFTGeneratorInferenceRef(
            decode_ref,
            TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor.")),
            hift_sd["m_source.l_linear.weight"],
            hift_sd["m_source.l_linear.bias"],
        )
        gen = TtHiFTGenerator(device, ref, TtHiFTDecoder(device, decode_ref, dtype=ttnn.float32), dtype=ttnn.float32)

    print(
        "\n| case | seam (sample) | arm | rel. error over the crossfade | PCC | max\\|diff\\| | upstream's calls agree to |"
        "\n|---|---|---|---|---|---|---|"
    )
    failures, control_failed, control_passed = [], [], []
    for path in cases:
        case = os.path.basename(path)[: -len(".npz")]
        d = np.load(path)
        mel, noise = torch.from_numpy(d["mel"]), torch.from_numpy(d["noise"])
        want = torch.from_numpy(d["chunked"])
        f0s = [torch.from_numpy(f).reshape(1, CHUNK_FRAMES) for f in d["f0"]]
        schedule = chunk_schedule(mel.shape[1])
        assert [c.start for c in schedule] == d["starts"].tolist(), case
        arms = {
            "mechanism": gen.inference_chunked(mel, noise, f0s=f0s),
            "no crossfade (control)": gen.inference_chunked(mel, noise, f0s=f0s, crossfade=False),
        }
        own = gen.inference_chunked(mel, noise)
        n = OVERLAP_FRAMES * HOP
        seams = [(c.start + c.fade_from) * HOP for c in schedule[1:]]
        agreement = d["seam_agreement"].tolist() if "seam_agreement" in d.files else [float("nan")] * len(seams)
        for seam, agree in zip(seams, agreement):
            lo, hi = seam - SEAM_PAD, seam + n + SEAM_PAD
            for arm, got in arms.items():
                pcc, err = _pcc(got[lo:hi], want[lo:hi]), float((got[lo:hi] - want[lo:hi]).abs().max())
                rel = float((got[seam : seam + n] - want[seam : seam + n]).norm() / want[seam : seam + n].norm())
                print(f"| {case} | {seam} | {arm} | {rel:.3f} | {pcc:.5f} | {err:.4f} | {agree:.3f} |", flush=True)
                ok = rel <= SEAM_REL_ERR and pcc >= SEAM_PCC
                if arm == "mechanism" and not ok:
                    failures.append(f"{case} seam {seam}: rel. error {rel:.3f}, PCC {pcc:.5f}, max|diff| {err:.4f}")
                if arm != "mechanism":
                    (control_passed if ok else control_failed).append(f"{case} seam {seam}")
        whole = _pcc(arms["mechanism"], want)
        print(
            f"| {case} | whole signal | mechanism | {whole:.5f} | {float((arms['mechanism'] - want).abs().max()):.4f} |"
        )
        if whole < WHOLE_PCC:
            failures.append(f"{case} whole-signal PCC {whole:.5f}")
        # own F0: spectral, whole and within 5 frames (100 ms) of each seam
        lm_got, lm_want = _logmel(own), _logmel(want)
        l1 = (lm_got - lm_want).abs()
        whole_l1 = float(l1.mean())
        seam_l1 = [float(l1[:, max(0, s // HOP - 5) : s // HOP + OVERLAP_FRAMES + 5].mean()) for s in seams]
        upstream_chunking = float((_logmel(torch.from_numpy(d["single"])) - lm_want).abs().mean())
        print(
            f"  {case} own F0: log-mel L1 whole {whole_l1:.4f}, around each seam {[round(x, 4) for x in seam_l1]}; "
            f"for scale, upstream chunked vs its own single pass {upstream_chunking:.4f}"
        )
        if whole_l1 > OWN_F0_LOGMEL_L1_WHOLE or max(seam_l1) > OWN_F0_SEAM_OVER_WHOLE * whole_l1:
            failures.append(f"{case} own-F0 log-mel L1 whole {whole_l1:.4f}, seams {seam_l1}")
    print(
        f"  the no-crossfade control failed the seam gate at {len(control_failed)} seams, passed it at: {control_passed}"
    )
    assert not failures, failures
    assert not control_passed, f"the no-crossfade control passed the seam gate at {control_passed}: blind there"
