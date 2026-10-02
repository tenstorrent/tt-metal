# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gate for TtVoxtralGPT.prefill_batched (all users in one right-padded pass, 2D-multicast matmul
configs, mixed voices and lengths) against the per-user prefill it replaces.

P1 hidden: the last-position hidden of every user, one pass vs per user (same prompts): per-row PCC.
P2 codes: a full batched generation with each prefill path; rows identical / first differing frame.
P3 time: prefill seconds, both paths.

Env: BP_BATCH (32), VOXTRAL_DEVICE_ID (0), BP_RESULTS (json path).
"""

import json
import os
import time

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts import frontend  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import all_voices, needs_checkpoint  # noqa: E402
from models.experimental.voxtral_tts.tests.sentence_corpus import lang_of, wer_band  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_batched import TtVoxtralBatchedPipeline  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600), needs_checkpoint]

B = int(os.environ.get("BP_BATCH", "32"))
DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))
RESULTS_PATH = os.environ.get("BP_RESULTS", "")


def _requests(n):
    voices = list(all_voices())
    reqs = []
    for i in range(n):
        v = voices[i % len(voices)]
        band = wer_band(lang_of(v), "medium" if i % 3 else "long")  # mixed lengths on purpose
        reqs.append((band[(i // len(voices)) % len(band)], v, i))
    return reqs


def test_batched_prefill_matches_per_user():
    dev = open_device(device_id=DEVICE_ID)
    pipe = None
    try:
        pipe = TtVoxtralBatchedPipeline(dev, max_batch=B, max_seq_len=1024)
        pipe.warmup()
        bb = pipe.backbone
        reqs = _requests(B)
        embeds = [frontend.build_prompt_embeds(t, v, pipe.wb, model_dir=pipe.model_dir) for t, v, _ in reqs]
        lens = [e.shape[1] for e in embeds]
        print(
            f"[bp] B={B}: prompt lengths {min(lens)}..{max(lens)} tokens, {len(set(v for _, v, _ in reqs))} voices",
            flush=True,
        )
        t0 = time.perf_counter()
        h_user = torch.cat([bb.prefill(embeds[b], last_only=True, user=b) for b in range(B)], dim=1)[0]
        t_user = time.perf_counter() - t0
        bb.prefill_batched(embeds)  # first call may compile new shapes; time the steady state
        t0 = time.perf_counter()
        h_batch = bb.prefill_batched(embeds)
        t_batch = time.perf_counter() - t0
        rows = [pcc(h_batch[b], h_user[b]) for b in range(B)]
        maxdiff = (h_batch - h_user).abs().max().item()
        print(
            f"[bp] P1 hidden: per-row PCC min {min(rows):.6f} mean {sum(rows) / B:.6f}, max |diff| {maxdiff:.3e}",
            flush=True,
        )
        print(
            f"[bp] P3 prefill time: per user {t_user:.2f}s ({t_user / B * 1e3:.0f} ms each), one pass {t_batch:.2f}s -> {t_user / t_batch:.1f}x",
            flush=True,
        )
        pipe.batched_prefill = False
        fu = pipe.generate_batch(reqs)
        tu = pipe.last_timings["prefill_s"]
        pipe.batched_prefill = True
        pipe.generate_batch(reqs)  # compiles the group shapes if warmup did not
        fb = pipe.generate_batch(reqs)
        tb = pipe.last_timings["prefill_s"]
        same, firsts = 0, []
        for a, c in zip(fu, fb):
            n = min(len(a), len(c))
            d = next((t for t in range(n) if not torch.equal(a[t], c[t])), None)
            if d is None and len(a) == len(c):
                same += 1
            else:
                firsts.append(d if d is not None else n)
        print(
            f"[bp] P2 codes: {same}/{B} rows identical over all frames; first differing frames of the rest: {sorted(firsts)}",
            flush=True,
        )
        print(
            f"[bp] P2 prefill_s inside generate_batch: per user {tu:.2f}s, one pass {tb:.2f}s; decode {pipe.last_timings['decode_ms_per_frame']:.1f} ms/frame",
            flush=True,
        )
        out = {
            "batch": B,
            "pcc_min": min(rows),
            "pcc_mean": sum(rows) / B,
            "maxdiff": maxdiff,
            "t_user": t_user,
            "t_batch": t_batch,
            "rows_identical": same,
            "first_diffs": sorted(firsts),
        }
        if RESULTS_PATH:
            json.dump(out, open(RESULTS_PATH, "w"), indent=2)
        assert min(rows) >= 0.999, f"one-pass prefill hidden differs from per-user prefill: min row PCC {min(rows):.5f}"
        assert t_batch < t_user, "one-pass prefill is not faster than per-user prefill"
    finally:
        if pipe is not None:
            pipe.close()
        ttnn.close_device(dev)
