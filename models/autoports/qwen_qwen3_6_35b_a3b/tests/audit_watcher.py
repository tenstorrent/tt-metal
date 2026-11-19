# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Check completed watcher runs and preserve hashes of their raw logs."""

import hashlib
import json
import os
import re
from pathlib import Path


def main():
    doc = (
        Path(__file__).parents[1]
        / "doc"
        / {
            "functional": "functional_decoder",
            "fused": "fused_decoder",
            "optimized": "optimized_decoder",
        }[os.environ.get("QWEN_DECODER", "functional")]
    )
    rows = []
    failure = re.compile(
        r"Watcher (?:detected|found corruption|data corruption|unexpected|stopped the device)|"
        r"tripped assert|corrupted noc sanitization|TT_FATAL",
        re.IGNORECASE,
    )
    for idx in (0, 3):
        result = json.loads((doc / f"trace_reuse_{idx}.json").read_text())
        provenance = json.loads((doc / f"trace_reuse_{idx}.json.provenance.json").read_text())
        assert provenance["env"]["TT_METAL_WATCHER"] == "10"
        assert not provenance["env"]["TT_METAL_DEVICE_PROFILER"]
        assert len(result["results"]) == 9 and result["repeated_request_bit_equal"]
        assert all(row["pcc"] >= 0.995 for row in result["results"])
        log = doc / f"watcher/layer_{idx}/generated/watcher/watcher.log"
        raw = log.read_bytes()
        text = raw.decode()
        dumps = len(re.findall(r"Dump #\d+ completed", text))
        assert dumps > 0
        console = (doc / f"logs/trace_reuse_watcher_{idx}.log").read_text()
        matches = failure.findall(text + console)
        assert not matches, matches
        rows.append(
            dict(
                layer=idx,
                log=str(log.relative_to(doc)),
                sha256=hashlib.sha256(raw).hexdigest(),
                completed_dumps=dumps,
                hardware_errors=matches,
                trace_reuse_pass=True,
            )
        )
    (doc / "watcher_audit.json").write_text(
        json.dumps(
            dict(
                clean=True,
                runs=rows,
                classified_harness_issue=(
                    None
                    if os.environ.get("QWEN_DECODER") in ("fused", "optimized")
                    else "First linear run passed device checks but failed MeshTraceId JSON serialization; fixed and rerun."
                ),
                positive_control="Deliberate post-capture allocation is rejected by allocation tracker; this is not a watcher hardware error.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("WATCHER_AUDIT_CLEAN", len(rows), "layer kinds")


if __name__ == "__main__":
    main()
