# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which device buffers are unsafe for the captured decode traces?

Every run log carries one line of

    Allocating device buffers is potentially unsafe due to the existence of an
    active trace

and that line says almost nothing on its own: `allocator.cpp` emits it at most
once per host thread for the process lifetime, behind a `thread_local static
bool`, so one line is compatible with one unsafe allocation or a thousand.
`mesh_device.cpp` registers a trace as active at `end_mesh_trace` and keeps
allocations unsafe until it is released, so the window is the whole life of the
retained model and sampling traces, not just the capture.

This probe replaces that inference with the accounting Metal itself provides.
Run it with the tracker on:

    TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \\
      python models/autoports/zai_org_glm_4_7_flash/probe/trace_alloc_probe.py

With the tracker enabled, `ttnn.execute_trace` verifies before every replay and
raises if a buffer allocated while a trace was active is still alive, so a
clean run of the generator's own decode loop *is* the gate. This probe adds the
per-phase detail (`get_unsafe_tracked_ids` for every live trace id) and runs
four arms:

1. the shipped path at a warmed single-chunk prompt;
2. the shipped path at a first-use multi-chunk prompt, where the
   chunk-offset-dependent programs compile *after* capture and the generator's
   ``_maybe_recapture_after_compile`` hook must fire;
3. the same compile with the hook bypassed, which is what the hazard looks like
   when nothing handles it;
4. ``recapture_decode_traces()`` over that state, which must clear it.

Reduced build (HF layers 0 and 1) on purpose: the allocation lifetimes this
looks at are model-level, and the reduced build has all of them.

Writes ``doc/full_model/trace_alloc.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.model import source_manifest

MODEL_DIR = Path(__file__).resolve().parents[1]
OUT = MODEL_DIR / "doc" / "full_model" / "trace_alloc.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1", help="HF layer indices, or 'all'")
    ap.add_argument("--seq-cap", type=int, default=8192)
    ap.add_argument("--warmed-prompt", type=int, default=37, help="single chunk, bucket warmed at build")
    ap.add_argument("--multi-chunk-prompt", type=int, default=3000, help="first use, compiles after capture")
    ap.add_argument("--second-multi-chunk-prompt", type=int, default=5000, help="another unwarmed chunk depth")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    tracking = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    layers = None if args.layers == "all" else [int(v) for v in args.layers.split(",")]

    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    arms = []
    try:
        gen = build_generator(
            MODEL_DIR, dev, max_batch_size=1, max_seq_len=args.seq_cap, layer_indices=layers, progress=print
        )
        model = gen.model

        def unsafe_ids():
            """Every live trace id -> {buffer id: allocating op}."""
            from ttnn._ttnn.operations.trace import get_unsafe_tracked_ids

            out = {}
            if gen._decode_trace_id is not None:
                out[f"model:{gen._decode_trace_id}"] = {
                    str(k): v for k, v in sorted(get_unsafe_tracked_ids(dev, gen._decode_trace_id).items())
                }
            sampling = gen.sampling
            for slot in getattr(sampling, "_trace_states", {}).values() if sampling is not None else []:
                tid = slot.get("id")
                if tid:
                    out[f"sampling:{tid}"] = {str(k): v for k, v in sorted(get_unsafe_tracked_ids(dev, tid).items())}
            return out

        def record(name, note, *, replay=True, warm_at=0):
            ids = unsafe_ids()
            counts = {k: len(v) for k, v in ids.items()}
            row = {
                "arm": name,
                "note": note,
                "unsafe_by_trace": counts,
                "unsafe_total": sum(counts.values()),
                "program_cache_entries": dev.num_program_cache_entries(),
                "program_cache_entries_at_capture": gen._program_cache_entries_at_capture,
                "trace_recaptures": gen.counters["trace_recaptures"],
                "allocating_ops": sorted({v.split()[1] for m in ids.values() for v in m.values() if " " in v}),
            }
            if replay:
                gen.set_decode_positions([warm_at] * gen.max_batch_size)
                try:
                    gen.replay_decode_trace()
                    ttnn.synchronize_device(dev)
                    row["replay"] = "ok"
                except RuntimeError as exc:
                    row["replay"] = "REFUSED by the tracker"
                    row["replay_error_head"] = str(exc).splitlines()[0]
            arms.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "allocating_ops"}), flush=True)

        def ids_for(seq):
            return list(range(1000, 1000 + seq))

        # 1. shipped path, warmed single-chunk prompt
        gen.reset()
        gen.generate(ids_for(args.warmed_prompt), args.tokens, enable_trace=True, stop_on_eos=False)
        record(
            "shipped_path_warmed_single_chunk",
            "warmup_prefill compiled every bucket shape and warmup_terminal_shapes every terminal tile "
            "offset before capture, so a prompt inside one chunk allocates nothing in the unsafe window "
            "and needs no recapture",
            warm_at=args.warmed_prompt,
        )

        # 2. shipped path, first-use multi-chunk prompt: the hook must fire
        gen.reset()
        gen.generate(ids_for(args.multi_chunk_prompt), args.tokens, enable_trace=True, stop_on_eos=False)
        record(
            "shipped_path_first_use_multi_chunk",
            "the chunk-offset-dependent programs compile after capture; "
            "_maybe_recapture_after_compile must fire and leave nothing unsafe",
            warm_at=args.multi_chunk_prompt,
        )

        # 3. the same compile with the hook bypassed
        before = gen.counters["trace_recaptures"]
        gen.reset()
        logits, _ = model.prefill_forward_last_logits_device(
            ids_for(args.second_multi_chunk_prompt),
            kv_cache=gen._kv_cache,
            page_table=gen._page_table_dev,
            seq_len=args.second_multi_chunk_prompt,
        )
        ttnn.deallocate(logits)
        assert gen.counters["trace_recaptures"] == before, "model-level prefill must not recapture"
        record(
            "hook_bypassed_first_use_multi_chunk",
            "the model-level prefill entry point has no hook, so this is the untreated hazard: one "
            "device buffer per newly cached program, alive for the process lifetime",
            warm_at=args.second_multi_chunk_prompt,
        )

        # 4. recapture clears it
        gen.recapture_decode_traces()
        record(
            "after_explicit_recapture",
            "recapture_decode_traces() puts the trace intermediates back on the safe side of those " "program buffers",
            warm_at=args.second_multi_chunk_prompt,
        )
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)

    by_arm = {a["arm"]: a for a in arms}
    expected_clean = [
        "shipped_path_warmed_single_chunk",
        "shipped_path_first_use_multi_chunk",
        "after_explicit_recapture",
    ]
    clean = all(by_arm[name]["unsafe_total"] == 0 and by_arm[name].get("replay") == "ok" for name in expected_clean)
    hazard_reproduced = by_arm["hook_bypassed_first_use_multi_chunk"]["unsafe_total"] > 0
    single_chunk_free = by_arm["shipped_path_warmed_single_chunk"]["trace_recaptures"] == 0
    payload = {
        "source_manifest": source_manifest([__file__]),
        "tracking_enabled": tracking,
        "tracking_note": (
            "Without TT_METAL_TRACE_ALLOC_TRACKING=1 the tracker is a no-op, every id map is empty and "
            "the run proves nothing. Check tracking_enabled before reading the result."
        ),
        "layers": args.layers,
        "context_allocated": args.seq_cap,
        "verdict": "clean" if (clean and tracking and single_chunk_free) else "unsafe_buffers_survive",
        "shipped_paths_clean": clean,
        "single_chunk_prompt_needs_no_recapture": single_chunk_free,
        "untreated_hazard_reproduced": hazard_reproduced,
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out, "verdict:", payload["verdict"])
    return 0 if payload["verdict"] == "clean" else 1


if __name__ == "__main__":
    raise SystemExit(main())
