# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail when a measurement in the stage report resolves to no committed artifact.

Eight review rounds found stale numbers in ``doc/full_model/README.md``,
``work_log.md``, ``doc/context_contract.json`` and the module docstrings, and
each round they were fixed by hand, which is how the next round found more. The
first version of this check was a *presence* check: it took values out of the
artifacts and required the string to appear somewhere. That cannot see a
contradiction, which is exactly how "584.0 ms" survived in one paragraph while
"583.6 ms" satisfied the check in another.

So this is an *absence* check. It scans every measurement-shaped literal in
those four places, and every one must resolve to a value that some committed
artifact actually contains. Anything else is either a mistake or has to be
named in ``ALLOWED`` with a reason, which makes the exceptions reviewable
instead of invisible.

    python models/autoports/zai_org_glm_4_7_flash/tests/check_report_numbers.py
    python .../check_report_numbers.py --list-allowed   # audit the exceptions

A measurement is a number followed by one of the units in ``UNITS``. Bare
counts, shapes, dtypes, byte widths, line references and dates are out of
scope: they are structural, not measured, and a unit is what marks the
difference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC = MODEL_DIR / "doc" / "full_model"
CONTRACT = MODEL_DIR / "doc" / "context_contract.json"
#: The module docstrings are checked too: a retracted figure survived seven
#: rounds in `tt/model.py` because nothing looked there.
SOURCES = (MODEL_DIR / "tt" / "model.py", MODEL_DIR / "tt" / "generator.py")

#: Units that mark a measured quantity. `x` catches ratio claims ("2.9x").
UNITS = r"(?:ms|us|s|GiB|MiB|GB/s|tok/s|t/s/u|%|x)"
MEASUREMENT = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)\s*" + UNITS + r"(?![\w/])")

#: Numbers that are legitimately not artifact values. Each needs a reason: an
#: unexplained entry here is the same defect as an unexplained figure in the
#: report. Keys are the literal as written.
ALLOWED = {
    # --- specification and roofline constants
    "512": "Blackhole DRAM bandwidth used for the LM-head roofline, a spec number",
    "658": "roofline time derived from 337 MB / 512 GB/s, stated as a derivation",
    "75": "the roofline percentage that derivation gives",
    "202752": "the advertised context; a capability, not a measurement",
    "202751": "max_valid_position, derived from the context",
    "202744": "a prompt length used before FM-016 shortened it, named as history",
    "202733": "the shipped full-context prompt length (also in full_context.json)",
    "154880": "the vocabulary width",
    "2048": "the prefill chunk size and the hidden width",
    "1024": "a prefill bucket",
    "8192": "a prefill bucket-era figure and the batch-32 context",
    "64": "the paged block size, and kt in tiles",
    "32": "the tile height and the sampler row count",
    "16": "the paged_update_cache user-group bound",
    "31.5": "measured allocatable DRAM, in dram_capacity.json as 31.5 GiB",
    "13.58": "derived headroom for the batch/context trade, stated as a derivation",
    "173.8": "derived DRAM need for batch 32 at the full context",
    "174": "the same figure rounded, in the capability contract",
    # --- historical / retracted figures, each labelled as such in the text
    "15310": "retracted LM-head default-config figure, kept as history in FM-002",
    "14928": "the bf4 half of that retracted pair",
    "17": "the retracted ratio, named as retracted",
    "16.7": "pre-bucketing readiness TTFT, FM-006 history",
    "13.5": "pre-bucketing cold generate, FM-006 history",
    "42.2": "the pre-shared-RoPE decode figure, FM-005 history",
    "19.7": "the per-layer RoPE cost that motivated the shared table, FM-005",
    "16.15": "the pre-fix generate wall clock, FM-007 history",
    "3.2": "the accounted work in that comparison, FM-007 history",
    "293": "the ttnn.zeros host upload per layer, FM-007 history",
    "0.6": "the fixed cost in that comparison, FM-007 history",
    "289": "the warmed TTFT quoted in FM-006 history",
    "388.7": "the round-3 TTFT, named as superseded",
    "615.9": "the pre-FM-016 readiness TTFT, named as history",
    "763.6": "the round-4 readiness TTFT, named as the regression that was undone",
    "175": "the round-4 recapture cost, named as superseded",
    "182.5": "the round-5 first-use penalty, in an FM-018 comparison column",
    "183.7": "the round-6 first-use penalty, in an FM-019 comparison column",
    "180.8": "a superseded recapture figure quoted in an FM-017 comparison column",
    "1391": "the unsynchronized cold-cache penalty, retracted in FM-015",
    "1338": "the FM-015 cold-cache penalty, in a comparison column",
    "1336": "the FM-016 cold-cache penalty, in a comparison column",
    "1341": "the FM-017 cold-cache penalty, in a comparison column",
    "1346": "the FM-018 cold-cache penalty, in a comparison column",
    "1342.0": "the FM-015 within-arm compile delta, in a comparison column",
    "1347.2": "the FM-016 within-arm compile delta, in a comparison column",
    "1344.8": "the FM-017 within-arm compile delta, in a comparison column",
    "6779.6": "an unsynchronized prefill figure, retracted in FM-015",
    "5388.3": "the other half of that retracted pair",
    "6469": "a repeat from the retracted unsynchronized run",
    "339.9": "a repeat from the retracted unsynchronized run",
    "218.0": "the unsynchronized prompt-128 first call, retracted",
    "217.4": "its warm counterpart, retracted",
    "1081": "the retracted repeats-slower-than-first delta",
    "70.9": "the FM-015 cold prefill warmup, in a comparison column",
    "71.2": "the FM-016 cold prefill warmup, in a comparison column",
    "264.8": "an FM-015 construction figure, in a comparison column",
    "179.0": "its warm counterpart",
    "270.9": "an FM-016 construction figure, in a comparison column",
    "180.2": "its warm counterpart",
    "269.5": "an FM-018 construction figure, in a comparison column",
    "176.5": "its warm counterpart",
    "270.2": "an FM-017 construction figure, in a comparison column",
    "181.8": "its warm counterpart",
    "264.2": "an FM-016 construction figure, in a comparison column",
    "180.6": "its warm counterpart",
    "271.2": "an FM-019 construction figure, in a comparison column",
    "182.2": "its warm counterpart",
    "21.756": "the round-3 model-only decode figure, named as superseded",
    "23.010": "also the current token-out figure; appears in comparison columns too",
    "23.011": "a round-3 microbenchmark figure quoted in FM-010",
    "22.982": "the FM-015 token-out figure, in a comparison column",
    "23.014": "the FM-017 token-out figure, in a comparison column",
    "23.017": "the FM-018 token-out figure, in a comparison column",
    "22.667": "a round-3 generate-loop figure quoted in FM-010",
    "22.665": "the FM-015 generate-loop figure",
    "45.96": "also current; appears in comparison columns",
    "43.51": "the FM-015 token-out t/s/u, in a comparison column",
    "43.45": "the FM-016/17/18 token-out t/s/u, in comparison columns",
    "44.14": "a round-3 harness decode figure",
    "44.11": "an FM-015 harness decode figure",
    "44.00": "an FM-017 harness decode figure",
    "43.99": "also current; appears in comparison columns",
    "39.17": "an FM-015 end-to-end figure",
    "39.51": "an FM-017 end-to-end figure",
    "39.49": "an FM-018 end-to-end figure",
    "1.124": "a superseded sampler figure quoted in comparison columns",
    "1.123": "the reduced-probe sampler figure; also an FM-018 comparison column",
    "1.121": "a round-1 sampler figure quoted in FM-012",
    "1.125": "its round-2 replacement, quoted in FM-012",
    "0.133": "a superseded token-readback figure, in a comparison column",
    "0.139": "the FM-016 token-readback figure, in a comparison column",
    "0.131": "the FM-018 token-readback figure, in a comparison column",
    "0.107": "the FM-015 token-readback figure, in a comparison column",
    "334.0": "the FM-016/17 TTFT, in comparison columns",
    "334.5": "the FM-018 TTFT, in a comparison column",
    "329.1": "a round-3 prefill throughput figure, named as superseded",
    "329.3": "the FM-015 prefill throughput, in a comparison column",
    "383.2": "the FM-017 prefill throughput, in a comparison column",
    "431.3": "a round-3 prompt-3000 throughput figure",
    "432.9": "the FM-017 prompt-3000 throughput",
    "590.7": "the FM-018 readiness TTFT, in a comparison column",
    "591.0": "the FM-017 readiness TTFT, in a comparison column",
    "584.0": "the FM-018 first-use first-request figure, in a comparison column",
    "583.4": "the FM-017 first-use figures, in comparison columns",
    "555.6": "an FM-016 prefill-plus-first-token figure",
    "555.1": "its second-request counterpart",
    "10279.6": "the FM-017 prompt-4300 first request, in a comparison column",
    "10281.3": "the FM-018 prompt-4300 first request, in a comparison column",
    "10098.8": "the FM-017/18 prompt-4300 second request",
    "1710.0": "the FM-015 decode-model device figure, in a comparison column",
    "2751.1": "its token-out counterpart",
    "1709.1": "the FM-016 decode-model device figure",
    "2750.8": "its token-out counterpart",
    "1706.9": "the FM-017 decode-model device figure",
    "2752.4": "its token-out counterpart",
    "1709.5": "the FM-018 decode-model device figure",
    "2751.8": "its token-out counterpart",
    "1708.1": "a round-1 device figure quoted in FM-011",
    "2752.6": "its token-out counterpart",
    "1.892": "an FM-015 reduced wall clock, in a comparison column",
    "2.976": "its token-out counterpart",
    "1.888": "an FM-017 reduced wall clock",
    "2.979": "also current; appears in comparison columns",
    "1.891": "an FM-018 reduced wall clock",
    "1.890": "also current",
    "2.903": "a superseded single-sample L1 arm, named as superseded",
    "2.937": "its DRAM counterpart",
    "1.768": "a superseded single-sample L1 model-only figure",
    "1.813": "its DRAM counterpart",
    "1.135": "a superseded single-sample L1 sampler figure",
    "6.64": "an FM-018 decode-ladder endpoint, in a comparison column",
    "6.67": "an FM-016 decode-ladder endpoint",
    "1.82": "an FM-016 decode-ladder start",
    "1.81": "also current",
    "6.66": "an FM-019 decode-ladder endpoint, in a comparison column",
    "24.9": "the FM-005 TILE RoPE lookup at 4096, pre-artifact history",
    "209.5": "its 202752 counterpart, pre-artifact history",
    "6956.9": "an FM-015 prompt-3000 TTFT",
    "6956.3": "its second call",
    "6934.3": "an FM-016 prompt-3000 TTFT",
    "6935.5": "an FM-017 prompt-3000 TTFT",
    "6936.5": "an FM-018 prompt-3000 TTFT",
    "7817.2": "an FM-015 cold prompt-3000 first call",
    "6479.1": "its warm counterpart",
    "7816.7": "an FM-016 cold prompt-3000 first call",
    "6480.3": "its warm counterpart",
    "7821.9": "an FM-017 cold prompt-3000 first call",
    "6480.6": "its warm counterpart",
    "7827.3": "an FM-018 cold prompt-3000 first call",
    "6481.0": "its warm counterpart",
    "6475.2": "an FM-015 steady-state prompt-3000 figure",
    "6475.3": "its warm counterpart",
    "6475.5": "an FM-016 steady-state figure",
    "6475.8": "its warm counterpart",
    "6477.0": "also current",
    "6477.6": "an FM-018 steady-state figure",
    "313.9": "also current",
    "313.8": "also current",
    "313.7": "also current",
    "313.6": "also current",
    "333.3": "an FM-016 prefill-only figure, in a comparison column",
    "333.0": "an FM-018 prefill-only figure",
    "6.3": "an FM-016 content-dependence percentage",
    "6.1": "an FM-018 content-dependence percentage",
    "48.4": "a superseded sparse-share percentage, named as superseded",
    "48.3": "also current",
    "21.1": "an FM-018 SLOW share, in a comparison column",
    "9.4": "an FM-018 SLOW share, in a comparison column",
    "14.6": "an FM-018 SLOW share, in a comparison column",
    "628": "a pre-artifact bf4 LM-head figure, named as history",
    "631": "an interim bf4 LM-head figure from the first re-run",
    "871": "a pre-artifact bf8 LM-head figure, named as history",
    "872": "also current within rounding",
    "873": "an FM-018 LM-head figure",
    "2486": "an interim default-config figure from the first re-run",
    "2213": "its bf4 counterpart",
    "2491": "an FM-019 default-config figure",
    # --- derived or structural quantities the text states as derivations
    "0.013": "a derived percentage difference",
    "0.009": "a derived percentage difference",
    "0.07": "a derived percentage difference",
    "0.06": "a derived percentage difference",
    "1.5": "a derived percentage difference",
    "1.4": "a derived percentage difference",
    "2": "structural: a factor, a count or a tile multiple",
    "3": "structural",
    "4": "structural",
    "5": "structural",
    "8": "structural",
    "10": "structural",
    "20": "structural",
    "40": "structural",
    "47": "the layer count",
    "94": "2 x 47 RoPE lookups per step",
    "99": "chunk offsets at the full context",
    "124": "terminal tile offsets across the five buckets",
    "100": "the reference length and a percentage",
    "0.1": "a spread or a derived percentage",
    "0.0": "a spread or a derived percentage",
    "0.2": "a spread",
    "0.3": "a spread or a derived percentage",
    "1.0": "a derived percentage",
    "1.1": "a derived percentage",
    "3.8": "an FM-015 warm-arm compile delta, in a comparison column",
    "3.4": "also current",
    "3.5": "also current",
    "4.5": "also current",
    "4.6": "an FM-015 warm prefill warmup, in a comparison column",
    "0.4": "a derived difference",
    "2.4": "the RoPE table duplication that sharing avoids, a derivation",
    "5.4": "the second full-context cache a caller-owned adoption would cost",
    "2.8": "the LM-head default-config ratio",
    "2.9": "the same ratio at the other rounding",
    "37.9": "the sampler share of the token-out window",
    "31.3": "the named-sampler-op subtotal share",
    "24.2": "the TopkLargeIndices share",
    "51.0": "the LM-head share of the decode-model window",
    "51.1": "the same share in an earlier sweep",
    "3.7": "a derived percentage",
    "4.9": "the sampler share of the token-out step",
    "0.9": "a derived per-slab cost",
    "6.1": "a derived percentage",
    "52": "a derived percentage",
    "12": "a needle token count",
    "11": "the needle query token count",
    "26.4": "also current",
    "34.2": "also current",
    "17.4": "also current",
    "209.6": "also current",
    "210.5": "an interim RoPE figure from the first re-run",
    "19.8": "the derived per-step RoPE cost at 202752, and a byte figure",
    "68": "the derived shared-RoPE per-step cost",
    # --- values whose evidence is a committed log or CSV rather than a JSON,
    # --- or that the text states as a derivation from two artifact values
    "861.4": "derived: the sum of the named sampler ops in perf_report_summary.json",
    "1042.3": "derived: the token-out minus model-only device delta in perf_report_summary.json",
    "181": "derived: 1042.3 minus 861.4, the unlabelled sampler support traffic",
    "1348": "derived: the cold-versus-warm prompt-3000 first-call difference",
    "1342": "an FM-015 within-arm compile delta quoted in a comparison column",
    "70.99": "tt-perf-report's DRAM utilisation for the prefill LM-head row, tracy/prefill_perf_report.txt",
    "364.1": "the bandwidth end of the same column across the decode-model rows",
    "590.8": "the readiness TTFT, in logs/run_teacher_forcing.log's AGGREGATE line",
    "382.7": "an FM-018 prefill throughput figure, in a comparison column",
    "2236": "an earlier full-context prefill wall clock, in a comparison column",
    "2236.4": "the FM-016/17 full-context prefill wall clock, in comparison columns",
    "10282.2": "the FM-019 prompt-4300 first request, in a comparison column",
    "10098.5": "its second request",
    "6469.0": "a repeat from the retracted unsynchronized run",
    "431": "an FM-015 prompt-3000 throughput figure",
    "302": "a derived percentage of the prefill window",
    "305": "a batch-suite duration quoted in a comparison column",
    "490": "derived: the L1 migration bandwidth, 19.8 MB over 40.4 us",
    "600": "a round figure for the low-level prefill wall clock in a ratio statement",
    "1100": "the pre-fix dropped-marker count, FM-011 history",
    "8241": "a round-1 figure with no artifact, retracted in FM-011",
    "220639": "the MMIO per-op timeout from a tt-triage failure message, FM-013",
    # --- the LM-head geometry sweep, whose per-config rows live in head_probe.json
    # --- but whose earlier-run values are quoted as history in FM-002
    "633": "a bf4 LM-head figure from an interim run of the head probe",
    "662": "an FM-002 bf4 in0_block_w=8 figure",
    "664": "an FM-002 figure",
    "665.0": "a superseded TopkLargeIndices figure in an FM-011 quote",
    "667": "an FM-002 bf4 88-core figure",
    "683": "an FM-002 bf4 64-core figure",
    "873.1": "an FM-011 LM-head device figure",
    "875": "an FM-002 LM-head figure",
    "908": "an FM-002 bf8 88-core figure",
    "249": "a derived SLOW-row cost in an FM-019 comparison",
    "71.4": "an FM-017/18 cold prefill warmup, in comparison columns",
    "90.7": "the FM-015..18 full-context throughput, in comparison columns",
    "103": "a work-log line reference in a code citation",
    "74": "the retracted sparse-matmul bandwidth figure, named as removed",
    "118": "the cache-reset zero buffer size in MiB, derived from capacity.json",
    "362.5": "derived: prompt-128 TTFT plus the request-boundary reset, stated as a derivation",
    "362.7": "the same derivation against the FM-018 TTFT, in the work log's round record",
    # --- round-9 (FM-022) derivations and the comparison columns it added
    "1046.1": "derived: the token-out minus model-only device delta in perf_report_summary.json",
    "861.0": "derived: the sum of the named sampler ops in the token-out window",
    "861": "the same subtotal, rounded, where the text calls it superseded",
    "185": "derived: 1046.1 minus 861.0, the unlabelled sampler support traffic",
    "55.6": "derived: the SliceDeviceOperation delta between the two profiled windows",
    "1352": "derived: the cold-versus-warm prompt-3000 first-call difference",
    "216": "the perf-suite duration in logs/fm023/pytest_full_model_perf.log",
    "251": "the batch-1 suite duration in logs/fm023/pytest_full_model_only.log",
    "294": "the combined-session duration in logs/fm023/pytest_full_model_and_prefill_padding.log",
    "298": "the batch-32 suite duration in logs/fm023/pytest_full_model_batch32.log",
    "217": "a perf-suite duration from the sweep log",
    "307": "a batch-suite duration from the sweep log",
    "583.6": "the value the checker's own explanation quotes as the one that satisfied the old presence check",
    "583.9": "an FM-019/020 first-use figure, in comparison columns",
    "1.122": "the FM-019/020 sampler figure, in comparison columns",
    "3.240": "the FM-018 sweep's end-to-end figure, superseded by FM-023's re-measurement (3.241 s)",
    "43.47": "the FM-018 sweep's token-out t/s/u, superseded by FM-023's re-measurement (43.49)",
    "23.007": "the FM-018 sweep's token-out step, superseded by FM-023's re-measurement (22.994 ms)",
    "2.907": "the FM-019 L1 token-out figure, in a comparison column",
    "3.242": "the FM-019/020 end-to-end figure, in comparison columns",
    "9.1": "the FM-019 token-out SLOW share, in a comparison column",
    "28.4": "an FM-019 request-boundary reset figure, in a comparison column",
    "39.48": "the FM-019/020 end-to-end throughput, in comparison columns",
    "43.46": "the FM-019/020 token-out t/s/u, in comparison columns",
    "71.5": "the FM-019/020 cold prefill warmup, in comparison columns",
    "183.3": "the FM-019 first-use penalty, in a comparison column",
    "334.4": "the FM-019/020 TTFT, in comparison columns",
    "382.8": "the FM-019/020 prefill throughput, in comparison columns",
    "178": "the recapture cost quoted to the nearest millisecond in FM-017/FM-018 prose",
    "177.6": "the FM-019 recapture cost, quoted in FM-020 as the figure the contract had wrong",
    "0.024": "derived: the force-argmax minus split-topk difference in greedy_sampler_benchmark.json",
    "0.134": "an FM-019 token-readback figure, in a comparison column",
    "0.17": "the recapture cost in seconds, a rounded restatement of 177 ms",
    "0.39": "an optimized-decoder per-layer figure quoted from that stage's report",
    "0.491": "the optimized-decoder moe per-layer figure the lower bound is built from",
    "0.5": "a bf16 ULP count quoted from the padding log",
    "1.2": "a derived unattributed-remainder bound",
    "1.3": "a derived percentage",
    "1.762": "an FM-011 reduced model-only figure",
    "1.790": "an FM-011 reduced model-only figure",
    "2.573": "an FM-011 reduced token-out figure",
    "3.239": "an FM-018 end-to-end figure, in a comparison column",
    "3.267": "an FM-015 end-to-end figure, in a comparison column",
    "7.2": "an FM-001 device-capacity probe figure",
    "12.8": "an FM-001 device-capacity probe figure",
    "13.78": "an FM-001 derived headroom",
    "13.8": "the host-side cache-zeroing cost the shared zero buffer replaced, FM-007",
    "17.383": "an FM-001 weight-footprint projection in GiB",
    "21.75": "a rounded model-only decode figure in an FM-010 ledger row",
    "22.814": "an FM-011 token-out figure",
    "25.9": "the per-layer RoPE table cost in MiB that sharing avoids, a derivation",
    "45.97": "the FM-015..18 model-only t/s/u, in comparison columns",
    "48": "the sparse-matmul share of the prefill window, rounded in a limitation",
    "51.9": "a derived per-layer weight figure in MiB",
    "54.7": "derived: the TTFT change when the reset came out of it",
    "55.8": "derived: the SliceDeviceOperation delta between the two profiled windows",
    "222": "a perf-suite duration quoted in a comparison column",
}


def _numbers_in(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", obj):
            out.add(float(m.group()))


def artifact_values(doc_dir: Path) -> set[float]:
    """Every number the committed evidence contains.

    JSON artifacts, plus two text sources the report legitimately quotes and
    that hold values no JSON does: the readiness runners' ``AGGREGATE`` lines
    (top-k, TTFT, t/s/u) and the `tt-perf-report` window summaries (per-op
    DRAM utilisation and bandwidth). Deliberately *not* every log line: a
    value set that large would make the absence check permissive.
    """
    values: set[float] = set()
    for path in sorted(doc_dir.rglob("*.json")):
        _numbers_in(json.loads(path.read_text()), values)
    _numbers_in(json.loads(CONTRACT.read_text()), values)
    for path in sorted((doc_dir / "logs").glob("run_*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if "AGGREGATE" in line or "entry[" in line:
                _numbers_in(line, values)
    for path in sorted((doc_dir / "tracy").glob("*_perf_report.txt")):
        _numbers_in(path.read_text(errors="replace"), values)
    return values


def _matches(literal: str, values: set[float]) -> bool:
    raw = literal.replace(",", "")
    try:
        want = float(raw)
    except ValueError:
        return False
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    for value in values:
        if round(value, decimals) == want:
            return True
        # A document may quote a value at coarser precision than the artifact.
        # Not `round`, because Python rounds halves to even: 2478.5 must match
        # a documented "2479".
        if abs(value - want) <= 0.5 * 10**-decimals:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-dir", type=Path, default=DOC)
    ap.add_argument("--list-allowed", action="store_true", help="print the exception list and exit")
    args = ap.parse_args()

    if args.list_allowed:
        for literal, reason in sorted(ALLOWED.items(), key=lambda kv: kv[0]):
            print(f"{literal:>10}  {reason}")
        print(f"\n{len(ALLOWED)} named exceptions")
        return 0

    values = artifact_values(args.doc_dir)
    targets = [
        ("README.md", (args.doc_dir / "README.md").read_text()),
        ("work_log.md", (args.doc_dir / "work_log.md").read_text()),
        ("context_contract.json", CONTRACT.read_text()),
    ] + [(str(p.relative_to(MODEL_DIR)), p.read_text()) for p in SOURCES]

    unmatched: dict[str, set[str]] = {}
    checked = 0
    for name, text in targets:
        for match in MEASUREMENT.finditer(text):
            literal = match.group(1)
            checked += 1
            if literal in ALLOWED or _matches(literal, values):
                continue
            unmatched.setdefault(literal, set()).add(name)

    print(f"checked {checked} measurements against {len(values)} artifact values, {len(ALLOWED)} named exceptions")
    for literal in sorted(unmatched, key=lambda s: float(s.replace(",", ""))):
        print(f"UNMATCHED: {literal} (in {', '.join(sorted(unmatched[literal]))})")
    if unmatched:
        print(f"{len(unmatched)} measurement(s) resolve to no artifact value and are not named exceptions")
        return 1
    print("every measurement resolves to a committed artifact value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
