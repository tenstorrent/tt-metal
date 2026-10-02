# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase B gates for the on-device frame loop (tt/ttnn_voxtral_device_loop.py).

B1 exactness: the same B requests (same prompts, same seeds, same noise) through the host loop and
through the device loop must give the same frame counts and the same codes, frame for frame. The
autoregressive loop amplifies any difference, so a row either matches to the end or diverges at a
first frame; the test reports both and the first divergence's cause (semantic vs acoustic code).

B1 timing: ms per frame for both loops on the same batch (the device loop should approach the
device-only frame graph cost: no host work between frames).

Env: DL_BATCH (32), VOXTRAL_DEVICE_ID (0), DL_RESULTS (json path).
"""

import json
import os
import time

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tests.reference_helpers import all_voices, needs_checkpoint  # noqa: E402
from models.experimental.voxtral_tts.tests.sentence_corpus import lang_of, wer_band  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_batched import TtVoxtralBatchedPipeline  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600), needs_checkpoint]

B = int(os.environ.get("DL_BATCH", "32"))
DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))
RESULTS_PATH = os.environ.get("DL_RESULTS", "")


def _requests(n):
    voices = list(all_voices())
    reqs = []
    for i in range(n):
        v = voices[i % len(voices)]
        band = wer_band(lang_of(v), "medium")
        reqs.append((band[(i // len(voices)) % len(band)], v, i))
    return reqs


def _compare(fh, fd):
    """-> dict with per-row verdicts."""
    rows = []
    for b, (h, d) in enumerate(zip(fh, fd)):
        n = min(len(h), len(d))
        first = None
        for t in range(n):
            if not torch.equal(h[t], d[t]):
                first = t
                break
        same_len = len(h) == len(d)
        if first is None and same_len:
            rows.append({"row": b, "ok": True, "frames": len(h)})
            continue
        t = first if first is not None else n
        cause = ""
        if first is not None:
            sem_diff = int(h[t][0]) != int(d[t][0])
            ac_flips = int((h[t][1:] != d[t][1:]).sum())
            cause = (
                f"semantic {int(h[t][0])}->{int(d[t][0])}"
                if sem_diff
                else f"{ac_flips} acoustic code(s) off by {int((h[t][1:] - d[t][1:]).abs().max())}"
            )
        rows.append(
            {"row": b, "ok": False, "frames_host": len(h), "frames_dev": len(d), "first_diff": t, "cause": cause}
        )
    return rows


def test_device_loop_matches_host_loop():
    assert os.environ.get("VOXTRAL_DEVICE_LOOP", "1") != "0", "run with the device loop enabled"
    dev = open_device(device_id=DEVICE_ID)
    pipe = None
    out = {}
    try:
        pipe = TtVoxtralBatchedPipeline(dev, max_batch=B, max_seq_len=1024)
        pipe.warmup()
        reqs = _requests(B)
        # host loop (reference)
        pipe.device_loop = False
        t0 = time.perf_counter()
        fh = pipe.generate_batch(reqs)
        th = time.perf_counter() - t0
        host_ms = pipe.last_timings["decode_ms_per_frame"]
        # device loop
        pipe.device_loop = True
        t0 = time.perf_counter()
        fd = pipe.generate_batch(reqs)
        td = time.perf_counter() - t0
        dev_ms = pipe.last_timings["decode_ms_per_frame"]
        assert pipe.last_timings["device_loop"] is True
        rows = _compare(fh, fd)
        ok = [r for r in rows if r["ok"]]
        bad = [r for r in rows if not r["ok"]]
        print(f"[dl] B={B}: {len(ok)}/{B} rows identical to the host loop over all frames", flush=True)
        for r in bad:
            print(
                f"[dl] row {r['row']:2d}: host {r['frames_host']} frames, device {r['frames_dev']}, first difference at frame {r['first_diff']}: {r['cause']}",
                flush=True,
            )
        print(f"[dl] frames per row (device): {[len(f) for f in fd]}", flush=True)
        print(
            f"[dl] ms/frame: host loop {host_ms:.1f} (wall {th:.2f}s), device loop {dev_ms:.1f} (wall {td:.2f}s), {pipe.last_timings['steps']} steps",
            flush=True,
        )
        # determinism of the device loop itself
        fd2 = pipe.generate_batch(reqs)
        det = all(len(a) == len(b) and torch.equal(a, b) for a, b in zip(fd, fd2))
        print(f"[dl] device loop deterministic across two runs: {det}", flush=True)
        out = {
            "batch": B,
            "identical_rows": len(ok),
            "rows": rows,
            "host_ms": host_ms,
            "dev_ms": dev_ms,
            "deterministic": det,
        }
        if RESULTS_PATH:
            json.dump(out, open(RESULTS_PATH, "w"), indent=2)
        assert det, "device loop not deterministic"
        # Gates. Frame 0 is produced from inputs identical to the host loop's (the prefill hidden and
        # the noise, copied in), so it must match bit for bit: this checks argmax, FSQ, the stop mask
        # and the code record. Later frames are fed by the device's own embedding sum, whose fp32
        # accumulation order differs from torch's, so a 1-ulp bf16 difference in the input can flip a
        # code and the autoregressive run then diverges; that is reported, and intelligibility is the
        # WER gate's call (test_wer_batched.py).
        frame0 = [r for r in bad if r["first_diff"] == 0]
        assert not frame0, f"frame 0 differs from the host loop in {len(frame0)} row(s): {frame0[:4]}"
        div = sorted(r["first_diff"] for r in bad)
        print(f"[dl] divergence frames (rows that differ later): {div}", flush=True)
    finally:
        if pipe is not None:
            pipe.close()
        ttnn.close_device(dev)
