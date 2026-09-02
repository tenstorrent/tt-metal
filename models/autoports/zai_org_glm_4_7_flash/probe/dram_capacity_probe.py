# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How much DRAM can device 0 actually hand out?

Every capacity claim in the full-model stage is measured against this number,
not against the board's nameplate. Allocates DRAM tensors in fixed chunks until
the allocator refuses, and writes the result next to the stage evidence.

    python models/autoports/zai_org_glm_4_7_flash/probe/dram_capacity_probe.py
"""

import argparse
import json
from pathlib import Path

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.model import source_manifest

OUT = Path(__file__).resolve().parents[1] / "doc" / "full_model" / "dram_capacity.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-mib", type=int, default=512)
    ap.add_argument("--limit-gib", type=int, default=40)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    held, total_mib, failure = [], 0, None
    try:
        rows = args.chunk_mib * 1024 * 1024 // 2 // 1024  # bf16, width 1024
        while total_mib < args.limit_gib * 1024:
            try:
                held.append(
                    ttnn.zeros(
                        (1, 1, rows, 1024),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
            except Exception as exc:  # allocator refusal is the measurement
                failure = str(exc).splitlines()[2] if len(str(exc).splitlines()) > 2 else str(exc)[:300]
                break
            total_mib += args.chunk_mib
            if total_mib % 4096 == 0:
                print(f"allocated {total_mib / 1024:.1f} GiB", flush=True)
        for tensor in held:
            ttnn.deallocate(tensor)
    finally:
        ttnn.close_device(dev)

    payload = {
        "source_manifest": source_manifest([__file__]),
        "device_id": 0,
        "chunk_mib": args.chunk_mib,
        "allocatable_mib": total_mib,
        "allocatable_gib": round(total_mib / 1024, 3),
        "allocatable_bytes": total_mib * 1024 * 1024,
        "first_refusal": failure,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
