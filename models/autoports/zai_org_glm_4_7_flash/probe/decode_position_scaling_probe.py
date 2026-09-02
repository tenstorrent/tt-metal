# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How does one traced decode step scale with the decode position?

A 202751-token full-model run stalled in a device readback behind traced decode
replays at position 202751. This isolates the cause cheaply: the flash MLA
decode reads whatever is in the cache, so the cache does not have to be
*filled* to time a step at a large ``cur_pos`` - only allocated. That turns a
37-minute prefill into a two-second setup and lets the whole position ladder be
swept on the reduced 2-layer probe.

Each position is timed with a hard wall-clock budget so a genuine stall is
reported instead of hanging the probe.

    python models/autoports/zai_org_glm_4_7_flash/probe/decode_position_scaling_probe.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import GLM47FlashGenerator
from models.autoports.zai_org_glm_4_7_flash.tt.model import GLM47FlashModel

MODEL_DIR = Path(__file__).resolve().parents[1]
OUT = MODEL_DIR / "doc" / "full_model" / "decode_position_scaling.json"
POSITIONS = (128, 2048, 8192, 32768, 65536, 131072, 202751)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1", help="HF layer indices, or 'all'")
    ap.add_argument("--seq-cap", type=int, default=202752)
    ap.add_argument("--positions", default=",".join(str(p) for p in POSITIONS))
    ap.add_argument("--budget-s", type=float, default=120.0, help="per-position wall budget")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    layers = None if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    rows = []
    try:
        model = GLM47FlashModel.from_pretrained(
            dev, max_batch_size=1, max_seq_len=args.seq_cap, layer_indices=layers, progress=print
        )
        gen = GLM47FlashGenerator(model)
        gen._ensure_owned_state()
        gen.capture_decode_trace()
        gen.reset()

        for pos in [int(p) for p in args.positions.split(",")]:
            if pos >= model.max_seq_len:
                pos = model.max_seq_len - 1
            row = {"position": pos}
            for label, run in (
                ("eager_ms", lambda: _eager(gen)),
                ("traced_ms", lambda: _traced(gen)),
            ):
                gen.set_decode_tokens([11])
                gen.set_decode_positions([pos])
                t0 = time.perf_counter()
                try:
                    run()
                    ttnn.synchronize_device(dev)
                    row[label] = round((time.perf_counter() - t0) * 1000, 2)
                except Exception as exc:  # noqa: BLE001 - the failure is the datum
                    row[label] = f"FAILED: {str(exc).splitlines()[0][:160]}"
                    break
                if time.perf_counter() - t0 > args.budget_s:
                    row[label + "_over_budget"] = True
            rows.append(row)
            print(json.dumps(row), flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)

    payload = {
        "layers": args.layers,
        "cache_context": args.seq_cap,
        "note": (
            "The cache is allocated but not filled; the flash MLA decode reads it regardless, so these "
            "are honest timings of the op's work at each cur_pos even though the contents are zeros."
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out)


def _eager(gen):
    logits = gen._decode_logits_device()
    ttnn.deallocate(logits)


def _traced(gen):
    ttnn.execute_trace(gen.mesh_device, gen._decode_trace_id, cq_id=0, blocking=False)


if __name__ == "__main__":
    main()
