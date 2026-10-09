# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render the PCC log written by tests/test_utils.check_pcc as a markdown table (module/layer x seq_len).

    python models/demos/pplx_decider_v1_27b/tests/pcc_report.py [pcc_results.jsonl]
"""

import json
import os
import sys
from pathlib import Path

DEFAULT_LOG = Path(
    os.environ.get("PPLX_DECIDER_PCC_LOG", "/local/ttuser/gtobar/artifacts/pplx_decider/logs/pcc_results.jsonl")
)


def render(log: Path) -> str:
    latest = {}
    for line in log.read_text().splitlines():
        r = json.loads(line)
        latest[(r["module"], r["layer"], r["seq_len"])] = r
    seq_lens = sorted({k[2] for k in latest})
    rows = sorted({(k[0], k[1]) for k in latest}, key=lambda m: (m[0], -1 if m[1] is None else m[1]))
    policies = sorted({r["policy"] for r in latest.values()})
    out = [f"Policy: {', '.join(policies)}. Threshold 0.995. Cells: PCC (FAIL marks < threshold).", ""]
    out.append("| module | layer | kind | " + " | ".join(f"S={s}" for s in seq_lens) + " |")
    out.append("|---|---|---|" + "---:|" * len(seq_lens))
    for module, layer in rows:
        kind = next((r["kind"] for k, r in latest.items() if k[:2] == (module, layer)), None) or "-"
        cells = []
        for s in seq_lens:
            r = latest.get((module, layer, s))
            cells.append("" if r is None else f"{r['pcc']:.6f}" + ("" if r["passed"] else " FAIL"))
        out.append(f"| {module} | {'-' if layer is None else layer} | {kind} | " + " | ".join(cells) + " |")
    worst = min(latest.values(), key=lambda r: r["pcc"])
    out += [
        "",
        f"Cases: {len(latest)}; failures: {sum(not r['passed'] for r in latest.values())}; "
        f"min PCC {worst['pcc']:.6f} ({worst['module']} L{worst['layer']} S{worst['seq_len']}).",
    ]
    tok0 = [r["pcc_wo_tok0"] for r in latest.values() if r.get("pcc_wo_tok0") is not None]
    if tok0:
        out.append(f"Diagnostic (not gated): min PCC excluding token 0 = {min(tok0):.6f}.")
    return "\n".join(out)


if __name__ == "__main__":
    print(render(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG))
