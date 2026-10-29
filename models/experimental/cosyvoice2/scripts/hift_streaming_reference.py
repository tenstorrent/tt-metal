# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Upstream's own HiFT on the chunked schedule: the reference for chunked HiFT's seam gate.

RUN IN THE REFERENCE VENV (see requirements-reference*.txt and scripts/reference_env.py):

    COSYVOICE2_REPO=<upstream checkout> LIBRISPEECH_ROOT=<dir containing LibriSpeech/> \\
        $COSYVOICE2_REF_ENV/bin/python hift_streaming_reference.py --out-dir <dir>

Each case is a real LibriSpeech test-clean utterance's mel, from upstream's own frontend feature extractor, cut to a
fixed span: six speakers, three female and three male, and nine seams. Every cut puts each seam's crossfade in voiced
speech: upstream's own F0 predictor gives F0 above its 10 Hz voicing threshold over every crossfade frame, +-4
frames (checked here; the selection rule is in the notes branch, scripts/2026-09-29/r2_select_seam_mels.py). A seam
in silence can't show a missing crossfade. The lengths cover an anchored last call with a long overlap (600,
1,100), a short one (800, 1,300), and calls that line up (1,016, 1,520), with two calls and with three. Upstream's
`HiFTGenerator`, the checkpoint the pipeline uses, vocodes each with one fixed sine-noise draw, injected in place of
SineGen2's own `randn_like` (its other draw, the initial phase, never reaches the output: tt/hifigan/source.py):
- **chunked:** tt/hifigan/chunking.py's schedule (loaded from the TT tree by path). Each call goes through upstream's
  `inference(speech_feat, cache_source)`, and the calls are stitched with upstream's own `fade_in_out` and
  `np.hamming` window. The stitch is also checked against chunking.py's `stitch` on the same calls.
- **no crossfade:** the same calls, switched hard at each seam (for scale).
- **single pass:** the whole mel in one `inference` (for scale: what chunking changes).
Upstream's F0 for each call is recorded, for the TT side to inject. So is, per seam, how far upstream's own two
calls already agree over the crossfade (relative L2 of the old call's held-back tail against the new call's head):
where they agree, a hard switch has little discontinuity to show, and the control fails mostly on the crossfade's
gain (its Hamming halves sum to about 1.08).

Writes `<case>.npz`: `mel` [1, T, 80], `noise` [1, T x 480, 9], `starts` [calls], `f0` [calls, 512],
`seam_agreement` [seams], `seam_f0_min` [seams], and 1-D audio `chunked`, `no_crossfade`, `single`.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_env  # noqa: E402

CASES = {  # utterance id -> (first mel frame, mel frames): six speakers, nine seams, every crossfade voiced
    "121-123852-0000": (9, 600),  # F, 2 calls
    "260-123288-0015": (10, 800),  # M, 2 calls
    "1221-135766-0011": (15, 1016),  # F, 2 calls
    "672-122797-0008": (56, 1100),  # M, 3 calls
    "1995-1836-0004": (17, 1300),  # F, 3 calls
    "908-157963-0007": (30, 1520),  # M, 3 calls
}
VOICED_HZ, VOICED_MARGIN = 10.0, 4  # upstream's nsf_voiced_threshold; frames checked either side of each crossfade
SEED = 1986


def load_chunking():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tt", "hifigan", "chunking.py")
    spec = importlib.util.spec_from_file_location("chunking", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclass resolves annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def injected_sine_noise(torch, noise):
    """SineGen2 draws `torch.randn_like(sine_waves)`, `[1, L, 9]`; hand it `noise` instead. Other draws pass."""
    original = torch.randn_like

    def randn_like(t, *a, **k):
        return noise.to(t.dtype) if tuple(t.shape) == tuple(noise.shape) else original(t, *a, **k)

    torch.randn_like = randn_like
    try:
        yield
    finally:
        torch.randn_like = original


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch

    chunking = load_chunking()
    model = reference_env.load_upstream()  # puts the upstream checkout on sys.path
    from cosyvoice.utils.common import fade_in_out

    hift = model.model.hift
    f0_calls = []
    f0_forward = hift.f0_predictor.forward

    def recording_f0(x):
        f0 = f0_forward(x)
        f0_calls.append(f0.detach().clone())
        return f0

    hift.f0_predictor.forward = recording_f0
    C, HOP = chunking.CHUNK_FRAMES, chunking.HOP
    window = np.hamming(2 * chunking.OVERLAP_FRAMES * HOP)
    assert np.array_equal(window, chunking.speech_window())

    for utt, (first, frames) in CASES.items():
        spk, chapter, _ = utt.split("-")
        wav_path = os.path.join(
            reference_env.librispeech_root(), "LibriSpeech", "test-clean", spk, chapter, f"{utt}.flac"
        )
        feat, _ = model.frontend._extract_speech_feat(wav_path)  # [1, T, 80]
        mel = feat[:, first : first + frames].float().contiguous()
        assert mel.shape[1] == frames, (utt, feat.shape)
        noise = torch.randn(1, frames * HOP, 9, generator=torch.Generator().manual_seed(SEED))
        schedule = chunking.chunk_schedule(frames)
        # every crossfade in voiced speech, by upstream's own F0 over the whole utterance (unrecorded: f0_forward)
        with torch.inference_mode():
            f0_utt = f0_forward(feat.transpose(1, 2).float()).reshape(-1)
        n_fade = chunking.OVERLAP_FRAMES
        seam_frames = [first + c.start + c.fade_from for c in schedule[1:]]
        seam_f0_min = [float(f0_utt[f - VOICED_MARGIN : f + n_fade + VOICED_MARGIN].min()) for f in seam_frames]
        assert all(f0 > VOICED_HZ for f0 in seam_f0_min), (utt, seam_f0_min)
        outs, prev_source, f0s = [], None, []
        with torch.inference_mode():
            for c in schedule:
                cache = (
                    torch.zeros(1, 1, 0)
                    if c.carry == 0
                    else chunking.carried_source(prev_source, c).reshape(1, 1, -1).clone()
                )
                f0_calls.clear()
                with injected_sine_noise(torch, noise[:, c.start * HOP : (c.start + C) * HOP]):
                    speech, source = hift.inference(
                        speech_feat=mel[:, c.start : c.start + C].transpose(1, 2).contiguous(), cache_source=cache
                    )
                outs.append(speech.reshape(-1).float())
                prev_source = source.reshape(-1).float()
                f0s.append(f0_calls[-1].reshape(-1))
            # upstream's own crossfade, exactly as token2wav applies it, then the same with chunking.py's stitch
            pieces, held = [], None
            for c, out in zip(schedule, outs):
                emit = out[c.fade_from * HOP : c.emit_to * HOP].reshape(1, -1)
                if held is not None:
                    emit = fade_in_out(emit, held.reshape(1, -1), window)
                pieces.append(emit.reshape(-1))
                held = out[c.emit_to * HOP :] if c.emit_to < C else None
            chunked = torch.cat(pieces)
            # how far upstream's own two calls already agree over each crossfade: the old call's held-back tail
            # against the new call's head
            seam_agreement = []
            for prev_c, prev_out, c, out in zip(schedule, outs, schedule[1:], outs[1:]):
                tail = prev_out[prev_c.emit_to * HOP :]
                head = out[c.fade_from * HOP : c.fade_from * HOP + len(tail)]
                seam_agreement.append(float((tail - head).norm() / head.norm()))
            ours = chunking.stitch(outs, schedule)
            stitch_diff = float((chunked - ours).abs().max())
            no_crossfade = chunking.stitch(outs, schedule, crossfade=False)
            with injected_sine_noise(torch, noise):
                single, _ = hift.inference(speech_feat=mel.transpose(1, 2).contiguous())
            single = single.reshape(-1).float()
        np.savez(
            os.path.join(args.out_dir, f"{utt}.npz"),
            mel=mel.numpy(),
            noise=noise.numpy(),
            starts=np.array([c.start for c in schedule], dtype=np.int32),
            f0=torch.stack(f0s).numpy(),
            seam_agreement=np.array(seam_agreement),
            seam_f0_min=np.array(seam_f0_min),
            chunked=chunked.numpy(),
            no_crossfade=no_crossfade.numpy(),
            single=single.numpy(),
        )
        seams = [(c.start + c.fade_from) * HOP for c in schedule[1:]]
        print(
            f"  {utt}: {frames} frames, {len(schedule)} calls (starts {[c.start for c in schedule]}), seams at samples "
            f"{seams}; stitch vs chunking.py max|diff| {stitch_diff:.3g}; chunked vs single pass PCC "
            f"{float(np.corrcoef(chunked.numpy(), single.numpy())[0, 1]):.5f}; upstream's calls agree over each "
            f"crossfade to {[round(a, 3) for a in seam_agreement]}; min F0 there {[round(f, 1) for f in seam_f0_min]} Hz",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
