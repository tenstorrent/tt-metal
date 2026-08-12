# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the measured fields of the ``functional_decoder`` block of
``doc/context_contract.json``.

The contract file is shared by every bringup stage: its top level belongs to the
newest stage, and each earlier stage keeps its own block.  This script owns
``contract["functional_decoder"]`` and nothing else, so it can run after later
stages have rewritten the top level.

Every number it writes is a transcription of
``doc/functional_decoder/test_results.xml``,
``doc/functional_decoder/logs/full_test_run.log``, ``logs/watcher_run.log`` and the committed
``doc/functional_decoder/tracy/**/**_perf_report.csv`` windows.  Transcribing
them by hand is how such numbers go stale, so this script does it and
``--check`` makes the staleness a non-zero exit instead of a review finding.

Prose fields, the capability contract itself, coverage lists and the byte budget
are left alone: this only touches values that have a single source of truth in a
committed run.

Usage::

    python refresh_context_contract.py            # rewrite the JSON in place
    python refresh_context_contract.py --check    # exit 1 if anything would change
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from summarize_perf import WINDOWS, measure, replay_count  # noqa: E402

STAGE = pathlib.Path(__file__).resolve().parents[1]  # doc/functional_decoder/
ROOT = STAGE.parents[1]  # models/autoports/<model>/
CONTRACT = ROOT / "doc/context_contract.json"
JUNIT = STAGE / "test_results.xml"
SUITE_LOG = STAGE / "logs/full_test_run.log"
WATCHER_LOG = STAGE / "logs/watcher_run.log"

#: contract key -> the ``assert_pcc`` label that carries it
PREFILL_PCC = {
    "sliding@131072_last32rows": "prefill[sliding] full-context seq_len=131072 (last 32 rows)",
    "full@131072_last32rows": "prefill[full] full-context seq_len=131072 (last 32 rows)",
    "sliding@130073_last32rows": "prefill[sliding] full-context seq_len=130073 (last 32 rows)",
    "full@130073_last32rows": "prefill[full] full-context seq_len=130073 (last 32 rows)",
    "sliding@131072_interior65536_32rows": "prefill[sliding] full-context seq_len=131072 (interior @65536 32 rows)",
    "full@131072_interior65536_32rows": "prefill[full] full-context seq_len=131072 (interior @65536 32 rows)",
}
DECODE_PCC = {
    "sliding@pos131071": "decode[sliding] full-context pos=131071",
    "full@pos131071": "decode[full] full-context pos=131071",
}
SLOT_PCC = {
    "prefill_sliding": "multi-chunk prefill[sliding] user_id=2 seq_len=12345",
    "prefill_full": "multi-chunk prefill[full] user_id=2 seq_len=12345",
    "decode_sliding": "decode[sliding] user_id=2 pos=12345",
    "decode_full": "decode[full] user_id=2 pos=12345",
}
REAL_WEIGHT_PCC = {
    "prefill_sliding": "real-weights prefill[sliding] seq_len=2049",
    "prefill_full": "real-weights prefill[full] seq_len=2049",
    "decode_sliding": "real-weights decode[sliding] pos=2049",
    "decode_full": "real-weights decode[full] pos=2049",
}
FP32_CONTROL_PCC = {
    "prefill_sliding": "prefill[sliding] vs FP32 HF reference seq_len=2049",
    "prefill_full": "prefill[full] vs FP32 HF reference seq_len=2049",
    "decode_sliding": "decode[sliding] vs FP32 HF reference pos=2049",
    "decode_full": "decode[full] vs FP32 HF reference pos=2049",
}
#: contract performance key -> window label in ``summarize_perf.WINDOWS``
PREFILL_MS = {
    "sliding": "prefill 8192 tok, batch 1  [sliding]",
    "full": "prefill 8192 tok, batch 1  [full]",
}
DECODE_MS = {
    "sliding@2048": "traced decode @2048        [sliding]",
    "sliding@131071": "traced decode @131071      [sliding]",
    "full@2048": "traced decode @2048        [full]",
    "full@131071": "traced decode @131071      [full]",
}
OPS_PER_ITER = {
    "prefill_sliding": "prefill 8192 tok, batch 1  [sliding]",
    "prefill_full": "prefill 8192 tok, batch 1  [full]",
    "decode_sliding": "traced decode @2048        [sliding]",
    "decode_full": "traced decode @2048        [full]",
}


def read_pccs() -> dict[str, float]:
    """``{assert_pcc label: value}`` from the committed suite log."""
    out: dict[str, float] = {}
    for line in SUITE_LOG.read_text(errors="ignore").splitlines():
        m = re.search(r"assert_pcc:\d+ - (.+?): ([0-9.]+)$", line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def resolve(pccs: dict[str, float], label: str) -> float:
    if label not in pccs:
        raise SystemExit(f"no suite-log line for {label!r}")
    return pccs[label]


def watcher_tests() -> int:
    """Tests inside the committed watcher console log.

    Hand-maintaining this is how it drifted once already: the watcher block used
    to say 14 while the committed run was 15 tests.
    """
    text = WATCHER_LOG.read_text(errors="ignore")
    hits = re.findall(r"^=+ (\d+) passed[^\n]*$", text, re.MULTILINE)
    if len(hits) != 1:
        raise SystemExit(f"{WATCHER_LOG} has {len(hits)} pytest summary lines; cannot refresh unambiguously")
    if re.search(r"\d+ (failed|error)", text):
        raise SystemExit(f"{WATCHER_LOG} reports failures; the watcher run is not clean")
    return int(hits[0])


def perf_windows() -> dict[str, dict[str, object]]:
    """``{window label: measurement}`` for the six committed Tracy windows."""
    out = {}
    for label, rel_csv, rel_log, _rel_raw, _signpost in WINDOWS:
        iters = replay_count(STAGE / rel_log)
        out[label] = measure(STAGE / rel_csv, iters)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the contract is stale")
    args = ap.parse_args()

    contract = json.loads(CONTRACT.read_text())
    before = json.dumps(contract, indent=2, sort_keys=True)
    if "functional_decoder" not in contract:
        print(f"{CONTRACT} has no functional_decoder block to refresh", file=sys.stderr)
        return 2

    pccs = read_pccs()
    suite = ET.parse(JUNIT).getroot()
    if suite.tag != "testsuite":
        suite = suite.find("testsuite")
    total = int(suite.get("tests"))
    failures = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))

    block = contract["functional_decoder"]
    tested = block["tested"]
    for key, label in PREFILL_PCC.items():
        tested["prefill"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in DECODE_PCC.items():
        tested["decode"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in SLOT_PCC.items():
        tested["nonzero_cache_slot"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in REAL_WEIGHT_PCC.items():
        tested["real_weights"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in FP32_CONTROL_PCC.items():
        tested["fp32_reference_control"]["pcc"][key] = round(resolve(pccs, label), 6)

    tested["watcher"]["tests"] = watcher_tests()

    worst_label, worst = min(pccs.items(), key=lambda kv: kv[1])
    tests = block["tests"]
    tests["total"] = total
    tests["passed"] = total - failures - skipped
    tests["pcc_checks"] = len(pccs)
    tests["min_pcc"] = round(worst, 6)
    tests["min_pcc_check"] = worst_label

    windows = perf_windows()
    perf = block["performance"]
    for key, label in PREFILL_MS.items():
        perf["prefill_8192_tokens_batch1_ms"][key] = round(windows[label]["device_ms"], 3)
    for key, label in DECODE_MS.items():
        perf["traced_decode_ms_per_token"][key] = round(windows[label]["device_ms"], 3)
    for key, label in OPS_PER_ITER.items():
        perf["device_ops_per_iteration"][key] = windows[label]["ops_per_iter"]

    after = json.dumps(contract, indent=2, sort_keys=True)
    if args.check:
        if before != after:
            print(
                "context_contract.json functional_decoder block is stale against the committed run",
                file=sys.stderr,
            )
            return 1
        print("context_contract.json functional_decoder block matches the committed run")
        return 0
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")
    print(
        f"refreshed {CONTRACT} functional_decoder block from {total} tests, "
        f"{len(pccs)} PCC checks (worst {worst:.6f}) and {len(windows)} perf windows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
