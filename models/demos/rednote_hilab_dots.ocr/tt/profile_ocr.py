# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy/perf harness for the dots.ocr ``ocr`` use case (skills/perf).

1 warmup ocr() call (compiles kernels; captures the CROSS-CALL decode trace
when --traced) then N timed calls (pure replay). Per-step latencies are
bucketed by kind {"prefill", "decode", "capture", "replay"}; the steady-state
report is the median of replay steps (--traced) or decode steps.

Untraced baseline:
    python models/demos/rednote_hilab_dots.ocr/tt/profile_ocr.py

Traced (sub-pass 1) + tracy (sub-pass 2):
    python -m tracy -p -v -r --op-support-count 20000 -n ocr_traced \
        models/demos/rednote_hilab_dots.ocr/tt/profile_ocr.py --traced

Wall-clock steady-state A/B (e.g. weight-dtype experiments): time N pure
trace replays back-to-back at advancing cache slots, host tiles prebuilt
so every timed step is exactly the production hot path (4 H2D copies +
execute_trace + 1-int readback):
    python models/demos/rednote_hilab_dots.ocr/tt/profile_ocr.py \
        --traced --bench-replays 200
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from statistics import median

import torch  # noqa: F401  (keeps torch initialized before ttnn)

import ttnn

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent


def _load_by_path(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=str(MODEL_DIR / "demo" / "inputs" / "invoice_total.png"))
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--num-timed", type=int, default=2)
    p.add_argument("--traced", action="store_true")
    p.add_argument("--bench-replays", type=int, default=0, help="time N pure trace replays (requires --traced)")
    args = p.parse_args()
    assert not args.bench_replays or args.traced, "--bench-replays requires --traced"

    from PIL import Image

    demo = _load_by_path("dots_ocr_demo_ocr", MODEL_DIR / "demo" / "demo_ocr.py")
    ocr_mod = _load_by_path("dots_ocr_tt_ocr_model", MODEL_DIR / "tt" / "ocr_model.py")
    tokenizer, image_processor, chat_template = demo.load_host_processors()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        trace_region_size=256_000_000 if args.traced else 0,
    )
    model = None
    try:
        model = ocr_mod.TtOCRModel(mesh_device, tokenizer, image_processor, chat_template)
        image = Image.open(args.image).convert("RGB")

        steps = []
        cb = lambda pos, ms, kind: steps.append((pos, ms, kind))  # noqa: E731

        # Warmup call: compiles kernels (and proves trace capture when --traced).
        t0 = time.perf_counter()
        warm_text = model.ocr(image, max_new_tokens=args.max_new_tokens, use_trace=args.traced, step_callback=cb)
        warm_ms = (time.perf_counter() - t0) * 1000.0
        print(f"warmup total_ms={warm_ms:.1f} text={warm_text!r}")

        if args.bench_replays:
            # Steady-state wall-clock bench: the warmup call captured the
            # trace; drive N replays at fresh advancing slots. Prebuild the
            # per-slot host tiles OUTSIDE the timed loop so each timed step
            # is exactly the production hot path.
            hidden, tok, start_slot = 1536, 100, 256
            for i in range(args.bench_replays):
                model._host_decode_inputs(start_slot + i)
            lat = []
            for i in range(args.bench_replays):
                t0 = time.perf_counter()
                tok_out, kind = model._decode_step_traced(tok, start_slot + i, hidden)
                lat.append((time.perf_counter() - t0) * 1000.0)
                assert kind == "replay", f"bench step {i} was {kind}, not replay"
            lat_sorted = sorted(lat)
            n = len(lat_sorted)
            print(
                f"BENCH replays={n} median_ms={median(lat):.2f} mean_ms={sum(lat)/n:.2f} "
                f"p10_ms={lat_sorted[n//10]:.2f} p90_ms={lat_sorted[(9*n)//10]:.2f} "
                f"min_ms={lat_sorted[0]:.2f} max_ms={lat_sorted[-1]:.2f}"
            )
            return

        totals, texts = [], []
        for i in range(args.num_timed):
            steps.clear()
            t0 = time.perf_counter()
            text = model.ocr(image, max_new_tokens=args.max_new_tokens, use_trace=args.traced, step_callback=cb)
            totals.append((time.perf_counter() - t0) * 1000.0)
            texts.append(text)
            steady = [ms for _, ms, kind in steps if kind == ("replay" if args.traced else "decode")]
            cap = [ms for _, ms, kind in steps if kind == "capture"]
            prefill = [ms for _, ms, kind in steps if kind == "prefill"]
            print(
                f"call[{i}] total_ms={totals[-1]:.1f} steps={len(steps)} prefill_ms={prefill[0] if prefill else 0:.1f} "
                f"steady_step_ms={median(steady):.2f} capture_ms={cap[0] if cap else 0:.1f} text={text!r}"
            )

        steady = [ms for _, ms, kind in steps if kind == ("replay" if args.traced else "decode")]
        print(
            f"SUMMARY traced={args.traced} steady_step_ms={median(steady):.2f} "
            f"total_ms={median(totals):.1f} consistent={len(set(texts + [warm_text])) == 1}"
        )
    finally:
        if args.traced and model is not None:
            model.release_decode_trace()
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
