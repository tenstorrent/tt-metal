# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF fp32 references for the prefill contract (tests/perf/prefill.py), cached on disk.

One reference per (bucket, logical length): the unmodified HF `EmbeddingGemma2TextModel` in fp32 on
the CPU (`reference/hf_reference.py`, sdpa attention, the method of tests/pcc/test_text_model.py),
run on the unpadded [1, L] input from `tests/common.make_batch`. Saved: last hidden [L, 768] fp32,
the pooled + normalized embedding [768], and the cost of producing it (wall s, peak RSS).

Long references are costly (E5a: 24 GB RSS at 32768 tokens), so build them once, each in its own
process so the peak RSS is that length's own, and point EMBEDDING_GEMMA2_PREFILL_REFS at the cache:

    python -m models.demos.embedding_gemma2.tests.perf.prefill_refs --bucket 32768 \
        --out /path/to/refs

Optional `--rss-cap-gb N`: a watchdog thread exits the process (code 3) when its RSS passes N GB,
so an infeasible length fails with a measured number instead of pushing the host into swap.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import threading
import time
from pathlib import Path

import torch

MODEL_ID = os.environ.get("HF_MODEL", "google/embeddinggemma-2")


def logical_len(bucket: int) -> int:
    """Logical length of the contract input in `bucket`: 3/4 full, so 1/4 of the rows are key padding
    (the masked path with real inputs padded into the bucket)."""
    return bucket * 3 // 4


def ref_path(out_dir: Path, bucket: int, length: int) -> Path:
    return Path(out_dir) / f"hf_fp32_b1_s{bucket}_L{length}.pt"


def load_ref(out_dir: Path, bucket: int, length: int):
    p = ref_path(out_dir, bucket, length)
    if not p.exists():
        return None
    return torch.load(p, weights_only=False)


def _rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return 0.0


def _watchdog(cap_gb: float, log: list):
    peak = 0.0
    while True:
        r = _rss_gb()
        peak = max(peak, r)
        log[:] = [peak]
        if r > cap_gb:
            print(f"RSS_CAP_EXCEEDED rss {r:.1f} GB > cap {cap_gb:.1f} GB", flush=True)
            os._exit(3)
        time.sleep(0.2)


def build_ref(out_dir: Path, bucket: int, length: int | None = None, rss_cap_gb: float = 0.0) -> dict:
    from models.demos.embedding_gemma2.reference.hf_reference import load_reference_text_model, run_reference
    from models.demos.embedding_gemma2.tests.common import make_batch
    from models.demos.embedding_gemma2.tt.weights import load_text_state_dict, resolve_snapshot

    length = logical_len(bucket) if length is None else length
    peak_log = [0.0]
    if rss_cap_gb > 0:
        threading.Thread(target=_watchdog, args=(rss_cap_gb, peak_log), daemon=True).start()
    snapshot = resolve_snapshot(MODEL_ID)
    t0 = time.perf_counter()
    sd = load_text_state_dict(snapshot)
    model, _ = load_reference_text_model(snapshot, sd)
    del sd
    t_load = time.perf_counter() - t0
    ids, mask = make_batch(snapshot, [length])
    t1 = time.perf_counter()
    out = run_reference(model, ids, mask, capture_layers=())
    t_fwd = time.perf_counter() - t1
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    rec = {
        "bucket": bucket,
        "length": length,
        "input_ids": ids[0].to(torch.int32),
        "last_hidden": out.last_hidden[0].float().contiguous(),
        "embedding": out.embedding[0].float().contiguous(),
        "meta": {
            "snapshot": str(snapshot),
            "dtype": "fp32",
            "attn_implementation": "sdpa",
            "load_s": t_load,
            "forward_s": t_fwd,
            "peak_rss_gb": peak_rss_gb,
            "torch_threads": torch.get_num_threads(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rec, ref_path(out_dir, bucket, length))
    print("REF " + json.dumps({"bucket": bucket, "length": length, **rec["meta"]}), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", type=int, required=True)
    ap.add_argument("--length", type=int, default=None, help="logical length (default 3/4 of the bucket)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rss-cap-gb", type=float, default=0.0)
    a = ap.parse_args()
    build_ref(a.out, a.bucket, a.length, a.rss_cap_gb)


if __name__ == "__main__":
    main()
