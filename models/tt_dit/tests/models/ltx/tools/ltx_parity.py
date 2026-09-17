# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Frame + audio parity between two LTX exports (run both with LTX_EXPORT_LOSSLESS=1).

    python -m models.tt_dit.tests.models.ltx.tools.ltx_parity --ref ref.mp4 --new new.mp4 [--psnr 40] [--corr 0.99]

Decodes both mp4s to yuv420p, reports per-frame PSNR of the Y plane (min / mean) and of Cb/Cr, and the
zero-lag normalised correlation of the two waveforms (the ``.audio.npy`` sidecars the lossless export writes,
falling back to the AAC tracks). Prints exactly one verdict line:

    PARITY OK psnr_y=<min> psnr_y_mean=<mean> psnr_c=<min> audio_corr=<c> frames=<n>
    PARITY FAIL ...
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _frames_yuv(path: str) -> np.ndarray:
    import av

    out = []
    with av.open(path) as c:
        stream = c.streams.video[0]
        for frame in c.decode(stream):
            out.append(frame.reformat(format="yuv420p").to_ndarray())  # (H*3//2, W) uint8
    return np.stack(out)


def _audio(path: str) -> np.ndarray | None:
    side = path + ".audio.npy"
    if os.path.exists(side):
        a = np.load(side).astype(np.float64)
        return a.reshape(-1) if a.ndim == 1 else a.mean(axis=0) if a.shape[0] <= 2 else a.reshape(-1)
    import av

    with av.open(path) as c:
        if not c.streams.audio:
            return None
        chunks = [f.to_ndarray().astype(np.float64) for f in c.decode(c.streams.audio[0])]
    if not chunks:
        return None
    a = np.concatenate(chunks, axis=-1)
    return a.mean(axis=0) if a.ndim == 2 else a


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--psnr", type=float, default=40.0, help="min per-frame Y PSNR (dB)")
    ap.add_argument("--corr", type=float, default=0.99, help="min audio correlation")
    args = ap.parse_args()

    ref, new = _frames_yuv(args.ref), _frames_yuv(args.new)
    if ref.shape != new.shape:
        print(f"PARITY FAIL shape ref={ref.shape} new={new.shape}")
        return 1
    n, h32, w = ref.shape
    h = h32 * 2 // 3
    ys = [_psnr(ref[i, :h], new[i, :h]) for i in range(n)]
    cs = [_psnr(ref[i, h:], new[i, h:]) for i in range(n)]
    ra, na = _audio(args.ref), _audio(args.new)
    if ra is None or na is None:
        corr = float("nan")
    else:
        m = min(len(ra), len(na))
        ra, na = ra[:m] - ra[:m].mean(), na[:m] - na[:m].mean()
        den = np.sqrt((ra**2).sum() * (na**2).sum())
        corr = float((ra * na).sum() / den) if den > 0 else float("nan")
    ok = min(ys) >= args.psnr and (corr != corr or corr >= args.corr)  # nan corr = no audio on either side
    verdict = "PARITY OK" if ok else "PARITY FAIL"
    print(
        f"{verdict} psnr_y={min(ys):.2f} psnr_y_mean={float(np.mean(ys)):.2f} psnr_c={min(cs):.2f} "
        f"audio_corr={corr:.4f} frames={n}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
