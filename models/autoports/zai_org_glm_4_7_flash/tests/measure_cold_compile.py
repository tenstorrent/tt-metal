# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure what a cold TTNN kernel cache costs the full model's TTFT.

Prefill length bucketing exists because every distinct physical prefill length
is its own set of TTNN programs. Two different costs hide behind that:

* **cold on-disk JIT cache** - the kernels have to be compiled from source;
* **warm on-disk cache, cold process** - only the in-process program cache is
  repopulated.

Both are setup costs a serving deployment pays once, but only if it warms up;
otherwise the first request of an unseen shape pays them inside its TTFT. The
JIT cache root is ``$HOME/.cache/tt-metal-cache`` (``tt_metal/jit_build/build.cpp``
``get_default_root_path``), so a throwaway ``HOME`` is what makes it cold:

    HOME=/tmp/glm47_coldhome \\
      GLM47_FLASH_SNAPSHOT=<hf snapshot dir> \\
      python models/autoports/zai_org_glm_4_7_flash/tests/measure_cold_compile.py

Check the run log for ``JIT cache stats: 0/N hits (0.0%)`` to confirm it really
was cold.

Writes ``doc/full_model/compile_cost.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.model import source_manifest

MODEL_DIR = Path(__file__).resolve().parents[1]
OUT = MODEL_DIR / "doc" / "full_model" / "compile_cost.json"


def _ids(gen, seq):
    text = "Tenstorrent builds AI accelerators. A plain in-distribution prompt for the compile-cost probe. " * 400
    ids = gen.tokenizer.encode(text, add_special_tokens=True)
    while len(ids) < seq:
        ids = ids + ids
    return ids[:seq]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=4, help="warm repeats per prompt after the first call")
    args = ap.parse_args()
    cache = str(Path(os.environ.get("HOME", "~")).expanduser() / ".cache" / "tt-metal-cache")
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    marks = {}

    def note(message: str):
        """Keep only the timed build phases ("<phase> in <n>s")."""
        text = message.strip()
        if " in " in text and text.endswith("s"):
            phase, _, seconds = text.rpartition(" in ")
            try:
                marks[phase.strip().replace(" ", "_") + "_s"] = float(seconds.rstrip("s"))
            except ValueError:
                pass

    try:
        t0 = time.perf_counter()
        gen = build_generator(MODEL_DIR, dev, progress=note)
        marks["build_total_s"] = round(time.perf_counter() - t0, 2)

        result = {
            "source_manifest": source_manifest([__file__]),
            "kernel_cache": cache,
            "build": marks,
            "prefill": {},
        }

        def timed_prefill(ids, seq):
            gen.reset()
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            logits, _ = gen.model.prefill_forward_last_logits_device(
                ids, kv_cache=gen._kv_cache, page_table=gen._page_table_dev, seq_len=seq
            )
            # prefill_forward_last_logits_device returns a *device* tensor and
            # deallocate does not block, so without this the timer measures host
            # enqueue, not prefill: the first call returns while the device is
            # still draining and every later call is device-bound behind it.
            ttnn.synchronize_device(dev)
            elapsed = (time.perf_counter() - t0) * 1000
            ttnn.deallocate(logits)
            return elapsed

        for seq in (128, 3000):
            ids = _ids(gen, seq)
            first = timed_prefill(ids, seq)
            # Several repeats: a single "second call" is not separable from
            # prefill's run-to-run spread, which this records explicitly.
            repeats = [timed_prefill(ids, seq) for _ in range(args.repeats)]
            mean = sum(repeats) / len(repeats)
            result["prefill"][f"prompt_{seq}"] = {
                "physical_len": gen.model.prefill_physical_len(seq),
                "warmed_at_build": seq <= 2048,
                "first_call_ms": round(first, 1),
                "repeat_calls_ms": [round(v, 1) for v in repeats],
                "repeat_mean_ms": round(mean, 1),
                "repeat_spread_pct": round((max(repeats) - min(repeats)) / mean * 100, 1),
                "first_minus_repeat_mean_ms": round(first - mean, 1),
            }
            print(seq, json.dumps(result["prefill"][f"prompt_{seq}"]), flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
