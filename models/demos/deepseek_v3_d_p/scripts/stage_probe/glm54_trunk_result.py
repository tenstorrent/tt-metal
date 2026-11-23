# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Complete a GLM-5.4 whole-trunk stage JSON (stage_probe_summary.py output) with its checks and verdicts (CPU only).

    python glm54_trunk_result.py <trunk.json> <glm53_stage_ref.json> <run_dir> [<run_dir> ...]

Adds to <trunk.json>, in place:
  checks.<run>      every run's trunk_probe_runner lines: the pass-2 push-reuse verdict and the KVPE digests (finite)
  determinism       whether every run logged the same digest per layer (bit-identical KV across runs)
  repeat_spread     per stage and probe, the largest minus smallest period_ms over the runs' splits
  verdict           per probe: GLM-5.4 slowest stage period_ms (first run) vs the GLM-5.3 reference's slowest stage
                    period_ms x threshold_factor (1.54, epic APPROVED ACCEPTANCE), met / unmet
"""

import json
import re
import sys
from pathlib import Path

THRESHOLD = 1.54
_REUSE = re.compile(r"TRUNK_ASSERT push_reuse (PASS|FAIL): (.*)$")
_DIGEST = re.compile(r"TRUNK_KV_DIGEST layer=(\d+) sha256=(\w+) finite=(\w+) nonzero_rows=(\d+) rows=(\d+)")
_DIGEST_FAIL = re.compile(r"TRUNK_KV_DIGEST layer=(\d+) failed: (.*)$")


def run_checks(run_dir: Path) -> dict:
    text = (run_dir / "runner.log").read_text(errors="replace")
    reuse = _REUSE.search(text)
    digests = {
        int(m.group(1)): {
            "sha256": m.group(2),
            "finite": m.group(3) == "True",
            "nonzero_rows": int(m.group(4)),
            "rows": int(m.group(5)),
        }
        for m in _DIGEST.finditer(text)
    }
    failed = {int(m.group(1)): m.group(2).strip() for m in _DIGEST_FAIL.finditer(text)}
    return {
        "push_reuse": {"verdict": reuse.group(1), "detail": reuse.group(2).strip()} if reuse else "not logged",
        "kv_digest": digests,
        "kv_digest_failed": failed,
        "finite": bool(digests) and not failed and all(d["finite"] for d in digests.values()),
    }


def main() -> None:
    out_path, ref_path, *run_dirs = sys.argv[1:]
    result = json.loads(Path(out_path).read_text())
    ref = json.loads(Path(ref_path).read_text())["reference"]
    runs = [Path(d) for d in run_dirs]

    checks = {d.name: run_checks(d) for d in runs}
    result["checks"] = checks
    per_run = [c["kv_digest"] for c in checks.values()]
    layers = sorted(set().union(*per_run)) if per_run else []
    result["determinism"] = {
        "runs": [d.name for d in runs],
        "layers": layers,
        "bit_identical": len(runs) >= 2
        and bool(layers)
        and all(len({tuple(sorted(r.get(l, {}).items())) for r in per_run}) == 1 for l in layers)
        and all(l in r for r in per_run for l in layers),
        "definition": "sha256 of user 0's KVPE cache slot (every chip, every row incl. the loop layers' pass-2 "
        "window ring) after the request, equal across runs that fed the same tokens",
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
