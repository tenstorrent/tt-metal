# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only standard-library audit of the frozen early-guard evidence."""

import hashlib
import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    header = here / "identity_early/compute_streaming.hpp"
    header_key = str(header.relative_to(root))
    header_hash = hashlib.sha256(header.read_bytes()).hexdigest()
    paths = sorted(set(here.glob("*-identity-early-*.json")) |
                   set((here / "identity-early-qualification").glob("*.json")) |
                   set((here / "identity-early-boundary-qualification").glob("*/*.json")) |
                   set((here / "identity-early-distinct-timings").glob("*.json")))
    assert paths, "No evidence found"
    for path in paths:
        record = json.loads(path.read_text())
        assert record["source_sha256"][header_key] == header_hash, path
        for field in ("source_stable", "baseline_candidate_equal", "trace_equal",
                      "original_host_immutable", "original_device_immutable",
                      "prepared_device_immutable"):
            assert record[field] is True, (path, field)
        assert record["preprocessing_mismatches"] == [0, 0, 0], path
        assert record["mandatory_trace_replays_per_kernel"] == 2, path
        assert record["output_sha256"] == record["baseline_output_sha256"], path
        if record["arguments"]["mode"] == "distinct":
            assert record["canonical_adapter_equal"] is True, path
        left, right = record["baseline_metadata"], record["candidate_metadata"]
        for field in ("fidelity", "fp32_dst", "input_slots", "cb_specs", "cb_bytes"):
            assert left[field] == right[field], (path, field)
        ld, rd = dict(left["defines"]), dict(right["defines"])
        del ld["SDPA_SPRINT_CANDIDATE_HEADER"]
        del rd["SDPA_SPRINT_CANDIDATE_HEADER"]
        assert ld == rd, (path, "numerical defines")
        assert right["input_slots"] == 2 and right["fp32_dst"] is False, path
    print(f"PASS {len(paths)} frozen-candidate records: raw bits, canonical distinct outputs, "
          "two replays, preparation, immutability, numerical config and CB geometry.")
    print(f"Candidate SHA256 {header_hash}")
    print("Selected source manifest only; not a compiler/firmware dependency closure.")


if __name__ == "__main__":
    main()
