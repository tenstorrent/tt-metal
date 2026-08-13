# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the measured fields of ``doc/context_contract.json`` from the suite run.

Same idea as every earlier stage's version, retargeted at this stage's labels and
at its **three** PCC populations: multichip-vs-single-chip TTNN (bar 0.999, the
only one that sees the fracture), the released checkpoint (bar 0.995) and the
i.i.d.-Gaussian synthetic harness.  Transcribing these by hand is how they go
stale, so this does it and ``--check`` turns staleness into a non-zero exit
rather than a review finding.

Usage::

    python refresh_context_contract.py            # rewrite the JSON in place
    python refresh_context_contract.py --check    # exit 1 if anything would change

Prose fields, the capability contract itself, the byte budget and the performance
block are left alone: this only touches values with a single source of truth in
the run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
CONTRACT = ROOT / "doc/context_contract.json"
#: The suite is two pytest invocations -- the main module holds a session-scoped
#: 1x4 mesh, and the comparison module opens a 1x1 then a 1x4 -- so both junits and
#: both logs are read.
JUNITS = (
    ROOT / "doc/multichip_decoder/test_results.xml",
    ROOT / "doc/multichip_decoder/test_results_vs_single_chip.xml",
)
SUITE_LOGS = (
    ROOT / "doc/multichip_decoder/logs/full_test_run.log",
    ROOT / "doc/multichip_decoder/logs/vs_single_chip_run.log",
)

#: contract key -> the ``assert_pcc`` label prefix that carries it
PREFILL_PCC = {
    "sliding@131072_last32rows": "multichip prefill[sliding] full-context seq_len=131072 (last 32 rows)",
    "full@131072_last32rows": "multichip prefill[full] full-context seq_len=131072 (last 32 rows)",
    "sliding@130073_last32rows": "multichip prefill[sliding] full-context seq_len=130073 (last 32 rows)",
    "full@130073_last32rows": "multichip prefill[full] full-context seq_len=130073 (last 32 rows)",
}
DECODE_PCC = {
    "sliding@pos131071": "multichip decode[sliding] full-context pos=131071",
    "full@pos131071": "multichip decode[full] full-context pos=131071",
}
SLOT_PCC = {
    "prefill_sliding": "multichip multi-chunk prefill[sliding] user_id=2 seq_len=12345",
    "prefill_full": "multichip multi-chunk prefill[full] user_id=2 seq_len=12345",
    "decode_sliding": "multichip decode[sliding] user_id=2 pos=12345",
    "decode_full": "multichip decode[full] user_id=2 pos=12345",
}
REAL_PCC = {
    "prefill_sliding@12345": "multichip real-weight prefill[sliding] seq_len=12345",
    "prefill_full@12345": "multichip real-weight prefill[full] seq_len=12345",
    "prefill_sliding@1": "multichip real-weight prefill[sliding] seq_len=1",
    "prefill_full@1": "multichip real-weight prefill[full] seq_len=1",
    "traced_decode_sliding_batch8": "multichip real-weight traced decode[sliding] batch=8",
    "traced_decode_full_batch8": "multichip real-weight traced decode[full] batch=8",
}
VS_SINGLE_PCC = {
    "prefill_sliding": "@worst:multichip vs single-chip TTNN prefill",
    "prefill_full": "@worst:multichip vs single-chip TTNN prefill",
    # The decode labels carry a step index, so ``resolve`` would find four; the
    # contract takes the worst of them, which is what a bar is about.
    "decode_sliding": "@worst:multichip vs single-chip TTNN decode",
    "decode_full": "@worst:multichip vs single-chip TTNN decode",
}
#: ``VS_SINGLE_PCC`` keys ending in a kind pick that kind's labels.
VS_SINGLE_KIND = {
    "prefill_sliding": "sliding",
    "prefill_full": "full",
    "decode_sliding": "sliding",
    "decode_full": "full",
}


def read_pccs() -> dict[str, float]:
    """``{assert_pcc label: value}`` from the committed suite log."""
    out: dict[str, float] = {}
    for log in SUITE_LOGS:
        for line in log.read_text(errors="ignore").splitlines():
            match = re.search(
                r"(?:assert_pcc|test_multichip_matches_single_chip):\d+ - (.+?): ([0-9.]+)$", line.strip()
            )
            if match:
                out[match.group(1)] = float(match.group(2))
    return out


def resolve(pccs: dict[str, float], label: str) -> float:
    """Exact label, else the unique label starting with it (window suffixes vary)."""
    if label in pccs:
        return pccs[label]
    hits = [value for key, value in pccs.items() if key.startswith(label)]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} log lines match {label!r}; cannot refresh unambiguously")
    return hits[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 if the contract is stale")
    args = parser.parse_args()

    contract = json.loads(CONTRACT.read_text())
    before = json.dumps(contract, indent=2, sort_keys=True)
    pccs = read_pccs()
    total = failures = skipped = 0
    for path in JUNITS:
        suite = ET.parse(path).getroot()
        if suite.tag != "testsuite":
            suite = suite.find("testsuite")
        total += int(suite.get("tests"))
        failures += int(suite.get("failures", 0)) + int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))

    tested = contract["tested"]
    for key, label in PREFILL_PCC.items():
        tested["prefill"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in DECODE_PCC.items():
        tested["decode"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in SLOT_PCC.items():
        tested["nonzero_cache_slot"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in REAL_PCC.items():
        tested["real_weights"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in VS_SINGLE_PCC.items():
        if label.startswith("@worst:"):
            prefix, kind = label[len("@worst:") :], VS_SINGLE_KIND[key]
            # Labels carry the shape they were measured at ("[full seq_len=12345
            # batch=4]"), and the contract records the worst of them per kind.
            hits = [
                v
                for name, v in pccs.items()
                if name.startswith(prefix) and (f"[{kind}]" in name or f"[{kind} " in name)
            ]
            if not hits:
                raise SystemExit(f"no log lines match {prefix!r} for kind {kind!r}")
            tested["vs_single_chip"]["pcc"][key] = round(min(hits), 6)
        else:
            tested["vs_single_chip"]["pcc"][key] = round(resolve(pccs, label), 6)

    versus = {k: v for k, v in pccs.items() if "vs single-chip" in k}
    # The two-layer chain has its own bar (0.96) because it composes two layers'
    # precision error; it is counted separately so the single-layer worst case is
    # not reported as its value.
    stacked = {k: v for k, v in pccs.items() if "two-layer stack" in k}
    rest = {k: v for k, v in pccs.items() if k not in versus and k not in stacked}
    real = {k: v for k, v in rest.items() if "real-weight" in k}
    synthetic = {k: v for k, v in rest.items() if "real-weight" not in k}
    tests = contract["tests"]
    tests["total"] = total
    tests["passed"] = total - failures - skipped
    tests["skipped"] = skipped
    tests["asserted_pcc_checks"] = len(pccs)
    tests["vs_single_chip_checks"] = len(versus)
    tests["min_pcc_vs_single_chip"] = round(min(versus.values()), 6)
    tests["real_weight_checks"] = len(real)
    tests["min_pcc_real_weights"] = round(min(real.values()), 6)
    tests["synthetic_checks"] = len(synthetic)
    tests["min_pcc_synthetic"] = round(min(synthetic.values()), 6)
    tests["two_layer_stack_checks"] = len(stacked)
    tests["min_pcc_two_layer_stack"] = round(min(stacked.values()), 6)

    full_context = [v for k, v in pccs.items() if "full-context" in k]
    contract["multichip_decoder_capacity_evidence"]["result"] = (
        f"{len(full_context)} full-context checks, all >= {min(full_context):.6f} "
        "(see tested.prefill.pcc / tested.decode.pcc)"
    )

    after = json.dumps(contract, indent=2, sort_keys=True)
    if args.check:
        if before != after:
            print("context_contract.json is stale against the committed suite run", file=sys.stderr)
            return 1
        print("context_contract.json matches the committed suite run")
        return 0
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")
    print(f"refreshed {CONTRACT} from {total} tests and {len(pccs)} asserted PCC checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
