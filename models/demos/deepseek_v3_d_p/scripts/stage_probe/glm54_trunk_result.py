# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Complete a GLM-5.4 whole-trunk stage JSON (stage_probe_summary.py output) with its checks and verdicts (CPU only).

    python glm54_trunk_result.py <trunk.json> <glm53_stage_ref.json> <run_dir> [<run_dir> ...]

Adds to <trunk.json>, in place:
  checks.<run>      every run's trunk_probe_runner lines: the pass-2 push-reuse verdict and the KVPE digests (finite)
  determinism       whether every run logged the same digest per chunk label (bit-identical KV across runs)
  repeat_spread     per stage and probe, the largest minus smallest period_ms over the runs' splits
  verdict           per probe: GLM-5.4 slowest stage period_ms (first run) vs the GLM-5.3 reference's slowest stage
                    period_ms x threshold_factor (1.54, epic APPROVED ACCEPTANCE), met / unmet; a probe named in
                    TRUNK_PROVISIONAL (newline-separated <probe>=<why> lines) gets provisional: true and that reason
"""

import json
import os
import re
import sys
from pathlib import Path

THRESHOLD = 1.54
_REUSE = re.compile(r"TRUNK_ASSERT push_reuse (PASS|FAIL): (.*)$", re.M)
_DIGEST = re.compile(r"TRUNK_KV_DIGEST chunk=(\w+) sha256=(\w+) finite=(\w+) nonzero_rows=(\d+) rows=(\d+)")
_DIGEST_FAIL = re.compile(r"TRUNK_KV_DIGEST chunk=(\w+) failed: (.*)$", re.M)


def run_checks(run_dir: Path) -> dict:
    text = (run_dir / "runner.log").read_text(errors="replace")
    reuse = _REUSE.search(text)
    digests = {
        m.group(1): {
            "sha256": m.group(2),
            "finite": m.group(3) == "True",
            "nonzero_rows": int(m.group(4)),
            "rows": int(m.group(5)),
        }
        for m in _DIGEST.finditer(text)
    }
    failed = {m.group(1): m.group(2).strip() for m in _DIGEST_FAIL.finditer(text)}
    return {
        "push_reuse": {"verdict": reuse.group(1), "detail": reuse.group(2).strip()} if reuse else "not logged",
        "kv_digest": digests,
        "kv_digest_failed": failed,
        "finite": bool(digests) and not failed and all(d["finite"] for d in digests.values()),
    }


def main() -> None:
    out_path, ref_path, *run_dirs = sys.argv[1:]
    provisional = dict(
        line.split("=", 1) for line in os.environ.get("TRUNK_PROVISIONAL", "").splitlines() if line.strip()
    )
    result = json.loads(Path(out_path).read_text())
    ref = json.loads(Path(ref_path).read_text())["reference"]
    runs = [Path(d) for d in run_dirs]

    checks = {d.name: run_checks(d) for d in runs}
    result["checks"] = checks
    per_run = [c["kv_digest"] for c in checks.values()]
    labels = sorted(set().union(*per_run)) if per_run else []
    result["determinism"] = {
        "runs": [d.name for d in runs],
        "chunks": labels,
        "bit_identical": len(runs) >= 2
        and bool(labels)
        and all(len({tuple(sorted(r.get(c, {}).items())) for r in per_run}) == 1 for c in labels)
        and all(c in r for r in per_run for c in labels),
        "definition": "sha256 of user 0's KVPE rows after the request, per probe chunk (all 78 cache layers, every "
        "chip) and per pass-2 window ring chunk (loop layers 26-53), equal across runs that fed the same tokens",
    }

    splits = [k for k in ("at_glm54_ranges", "repeat") if k in result]
    if len(splits) == 2:
        first, second = result["at_glm54_ranges"], result["repeat"]["at_glm54_ranges"]
        result["repeat_spread"] = {
            "definition": "|period_ms(run 1) - period_ms(run 2)| per stage and probe",
            **{
                s: {
                    p: round(abs(first[s][p]["period_ms"] - second[s][p]["period_ms"]), 2)
                    for p in first[s]
                    if p in second[s]
                }
                for s in ("G0", "G1", "G2", "G3")
            },
        }

    comp = result["at_glm54_ranges"]["composed"]
    verdict = {}
    for probe, ref_ms in ref["slowest_stage_period_ms"].items():
        if probe not in comp:
            verdict[probe] = {"status": "not measured"}
            continue
        target = round(ref_ms * THRESHOLD, 2)
        ms = comp[probe]["slowest_period_ms"]
        verdict[probe] = {
            "glm54_slowest_stage": comp[probe]["slowest_stage_by_period"],
            "glm54_slowest_period_ms": ms,
            "glm53_reference_slowest_period_ms": ref_ms,
            "glm53_reference_slowest_stage": ref["slowest_stage"][probe],
            "target_max_ms": target,
            "ratio_to_glm53": round(ms / ref_ms, 3),
            "status": "met" if ms <= target else "unmet",
        }
        if probe in provisional:
            verdict[probe].update(provisional=True, provisional_reason=provisional[probe])
    result["verdict"] = {
        "rule": f"GLM-5.4 slowest stage period_ms <= {THRESHOLD} x GLM-5.3 reference slowest stage period_ms "
        "(glm53_default_split, glm53_stage_ref.json reference), per probe",
        **verdict,
    }
    Path(out_path).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("verdict", "determinism") if k in result}, indent=1))
    for name, c in checks.items():
        print(
            name,
            "push_reuse:",
            c["push_reuse"] if isinstance(c["push_reuse"], str) else c["push_reuse"]["verdict"],
            "finite:",
            c["finite"],
        )


if __name__ == "__main__":
    main()
