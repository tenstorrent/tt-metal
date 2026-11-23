# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Complete a GLM-5.4 whole-trunk + MTP stage JSON with G3's period with vs without MTP (CPU only, q171.2.10).

    python glm54_mtp_result.py <mtp.json> <no_mtp.json>

<mtp.json> is make_glm54_mtp_stages.sh's output (G3 = passes 82-105 + MTP rows, glm54_trunk_result.py checks done);
<no_mtp.json> the same trunk without MTP (make_glm54_trunk_stages.sh). Adds to <mtp.json>, in place:
  g3_mtp_vs_no_mtp  per probe: G3 period_ms with MTP (its passes and MTP rows), G3's trunk passes alone in the same
                    run, and G3 without MTP (the no-MTP run), and the differences
  mtp_checks        per MTP run: the MTP input verdict, MTP's KV slot digests (finite, rows written), and whether the
                    trunk slots' digests equal the no-MTP run's (MTP leaves the trunk's KV bit-identical)
"""

import json
import sys
from pathlib import Path


def main() -> None:
    out_path, base_path = sys.argv[1:]
    result = json.loads(Path(out_path).read_text())
    base = json.loads(Path(base_path).read_text())

    with_mtp, trunk_only = result["at_glm54_ranges"]["G3"], result["g3_trunk_passes_in_mtp_run"]["G3"]
    without = base["at_glm54_ranges"]["G3"]
    comparison = {}
    for probe, row in with_mtp.items():
        if not isinstance(row, dict) or probe not in without:
            continue
        w, t, n = row["period_ms"], trunk_only[probe]["period_ms"], without[probe]["period_ms"]
        comparison[probe] = {
            "with_mtp_period_ms": w,
            "without_mtp_period_ms": n,
            "delta_ms": round(w - n, 2),
            "ratio": round(w / n, 3),
            "g3_trunk_passes_in_mtp_run_period_ms": t,
            "mtp_rows_ms": round(w - t, 2),
            "chunk": row["chunk"],
        }
    result["g3_mtp_vs_no_mtp"] = {
        "definition": "G3 period_ms (stage_probe_summary.py) with MTP = passes 82-105 + the MTP ack rows of the MTP run; "
        "without = passes 82-105 of the no-MTP run (same head, request and method); mtp_rows_ms = with MTP minus G3's "
        "trunk passes in the same run (MTP's own time incl. the trunk norm)",
        "no_mtp_runs": list(base.get("runs", {})),
        **comparison,
    }

    base_digests = [c["kv_digest"] for c in base.get("checks", {}).values()]
    checks = {}
    for name, c in result.get("checks", {}).items():
        digests = c["kv_digest"]
        mtp = {k: v for k, v in digests.items() if k.endswith("mtp")}
        trunk = {k: v["sha256"] for k, v in digests.items() if not k.endswith("mtp")}
        same = [all(b.get(k, {}).get("sha256") == sha for k, sha in trunk.items()) for b in base_digests]
        checks[name] = {
            "mtp_input": c.get("mtp_input"),
            "mtp_slot_finite": bool(mtp) and all(v["finite"] for v in mtp.values()),
            "mtp_slot_nonzero_rows": {k: v["nonzero_rows"] for k, v in mtp.items()},
            "trunk_kv_equals_no_mtp_run": bool(same) and all(same),
        }
    result["mtp_checks"] = checks
    Path(out_path).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("g3_mtp_vs_no_mtp", "mtp_checks")}, indent=1))


if __name__ == "__main__":
    main()
