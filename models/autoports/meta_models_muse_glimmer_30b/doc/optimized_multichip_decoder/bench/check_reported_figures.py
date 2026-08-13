# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive the reported figures in this stage's documents, claim by claim.

Four rounds of `$stage-review` on this stage found the same class of defect: the
code and the measurements were right, and a number in `README.md`,
`work_log.md` or `context_contract.json` was from a superseded run.

The first version of this file asked the wrong question -- "does this artifact's
value appear *somewhere* in either document" -- which a review showed is nearly
vacuous, because the two documents duplicate figures and either one can rot
alone.  This version goes the other way: it **parses each claim where it is
written**, anchored to its own table row, and re-derives that specific number
from the artifact the claim cites.  Corrupting one cell in one document fails it.

It also checks the derived columns (percentage deltas, op counts), because those
are what a reader acts on and they were wrong once.

    python .../bench/check_reported_figures.py
"""

from __future__ import annotations

import collections
import csv
import json
import pathlib
import re
import sys

DOC = pathlib.Path(__file__).resolve().parents[1]
BASELINE = DOC.parent / "multichip_decoder"
DECODE_REPLAYS = 8

#: Op codes that are collectives.  ``LayerNormPreAllGather`` /
#: ``LayerNormPostAllGather`` are **not**: they are the distributed norm's two
#: compute halves and they belong to the norm total.  Substring-matching
#: "AllGather" swept them into the collective group in the first version of this
#: file, which double-counted 372.0 us and understated the collective saving by
#: 2.6x -- the exact defect this file exists to prevent.
COLLECTIVE_OPS = (
    "ReduceScatterDeviceOperation",
    "AllGatherDeviceOperation",
    "AllReduceDeviceOperation",
    "ReduceScatterMinimalAsyncDeviceOperation",
    "AllGatherAsyncDeviceOperation",
)
NORM_OPS = (
    "LayerNormDeviceOperation",
    "LayerNormPreAllGatherDeviceOperation",
    "LayerNormPostAllGatherDeviceOperation",
    "RMSNormDeviceOperation",
)

failures: list[str] = []
checks = 0


def near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def expect(ok: bool, where: str, detail: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{where}: {detail}")


def capture(root: pathlib.Path, kind: str, tag: str, replays: int = 1):
    """``(total_us_per_iteration, {op_code: us}, ops_per_iteration)``."""
    rows = list(csv.DictReader((root / "tracy" / kind / f"{tag}_perf_report.csv").open()))
    key = next(k for k in rows[0] if k.strip().lower().startswith("device time"))
    code = next(k for k in rows[0] if "OP CODE" in k.upper() or k.strip() == "Op Code")
    per: dict[str, float] = collections.defaultdict(float)
    for row in rows:
        if row[key].strip():
            per[row[code]] += float(row[key])
    return sum(per.values()) / replays, {k: v / replays for k, v in per.items()}, len(rows) // replays


def group(per_op: dict[str, float], names: tuple[str, ...]) -> float:
    return sum(v for k, v in per_op.items() if any(k.startswith(n) for n in names))


def ab_rows(path: pathlib.Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    pattern = re.compile(r"^AB\S*\s+(\S+)\s+kind=(\S+).*?prefill\d+=\s*([\d.]+|nan) ms\s+traced_decode@\d+=\s*([\d.]+)")
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.match(line.strip())
        if m:
            out[(m.group(1), m.group(2))] = {"prefill_ms": float(m.group(3)), "decode_ms": float(m.group(4))}
    return out


def find_row(text: str, label: str) -> list[str] | None:
    """The cells of the row labelled ``label``, or ``None``.  Records nothing."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells and cells[0].replace("**", "") == label:
            return cells
    return None


def row_cells(text: str, label: str, doc: str) -> list[str] | None:
    """The cells of the markdown table row whose first cell is ``label``."""
    cells = find_row(text, label)
    if cells is None:
        failures.append(f"{doc}: no table row labelled {label!r}")
    return cells


def numbers(cell: str) -> list[float]:
    return [float(v) for v in re.findall(r"-?\d+\.\d+|(?<![\d.])-?\d+(?![\d.])", cell.replace("−", "-"))]


def main() -> int:
    readme = (DOC / "README.md").read_text()
    work_log = (DOC / "work_log.md").read_text()
    contract = json.loads((DOC.parent / "context_contract.json").read_text())
    stage = contract["optimized_multichip_decoder"]
    # The contract carries this stage's facts twice: at the top level, which is
    # what a reader sees first, and nested under the stage name, which is what the
    # next stage's own block will nest in turn.  Both are checked, and they are
    # required to agree -- editing one and not the other is exactly how the
    # top-level block went stale two rounds ago.
    for field in ("performance", "tests"):
        expect(contract[field] == stage[field], f"contract top-level {field}",
               "differs from optimized_multichip_decoder." + field)

    # ---------------------------------------------------------------- device time
    dev: dict[tuple[str, str], tuple] = {}
    for kind in ("sliding", "full"):
        for tag, replays in (("decode_2048", DECODE_REPLAYS), ("decode_131071", DECODE_REPLAYS),
                             ("prefill_128", 1), ("prefill_8192", 1)):
            after, after_ops, after_n = capture(DOC, kind, tag, replays)
            before, before_ops, before_n = capture(BASELINE, kind, tag, replays)
            dev[(kind, tag)] = (before, after, before_n, after_n, before_ops, after_ops)

    # Both documents carry the same four-row device table; each row pairs the two
    # layer kinds, so one row asserts eight numbers plus its deltas and op counts.
    device_rows = (
        ("decode sliding / full @2048", "decode_2048", True),
        ("decode sliding / full @131071", "decode_131071", True),
        ("prefill 8192 sliding / full", "prefill_8192", True),
        ("prefill 128 sliding / full", "prefill_128", True),
    )
    for doc_name, text in (("README.md", readme), ("work_log.md", work_log)):
        for label, tag, has_ops in device_rows:
            cells = row_cells(text, label, doc_name)
            if cells is None:
                continue
            before_s, after_s, delta_s = numbers(cells[1]), numbers(cells[2]), numbers(cells[3])
            for i, kind in enumerate(("sliding", "full")):
                b, a, bn, an, _, _ = dev[(kind, tag)]
                expect(len(before_s) > i and near(before_s[i], b, 0.15), f"{doc_name} '{label}' before[{kind}]",
                       f"says {before_s[i:i+1]}, CSV gives {b:.1f}")
                expect(len(after_s) > i and near(after_s[i], a, 0.15), f"{doc_name} '{label}' after[{kind}]",
                       f"says {after_s[i:i+1]}, CSV gives {a:.1f}")
                pct = 100.0 * (a - b) / b
                expect(len(delta_s) > i and near(delta_s[i], pct, 0.02), f"{doc_name} '{label}' delta[{kind}]",
                       f"says {delta_s[i:i+1]}, CSV gives {pct:+.2f} %")
                if has_ops and len(cells) > 4:
                    ops = numbers(cells[4])
                    if len(ops) >= 2 * (i + 1) and "unchanged" not in cells[4]:
                        expect(near(ops[2 * i], bn, 0.5) and near(ops[2 * i + 1], an, 0.5),
                               f"{doc_name} '{label}' ops[{kind}]", f"says {ops[2*i:2*i+2]}, CSV gives {bn}->{an}")

    for field, kind, tag in (("sliding@2048", "sliding", "decode_2048"), ("full@2048", "full", "decode_2048"),
                             ("sliding@131071", "sliding", "decode_131071"), ("full@131071", "full", "decode_131071")):
        for where, block in (("stage", stage["performance"]), ("top-level", contract["performance"])):
            expect(near(block["traced_decode_us_device"][field], dev[(kind, tag)][1], 0.05),
                   f"contract {where} traced_decode_us_device[{field}]",
                   f"says {block['traced_decode_us_device'][field]}, CSV gives {dev[(kind, tag)][1]:.1f}")
    for field, kind, tag in (("8192_sliding", "sliding", "prefill_8192"), ("8192_full", "full", "prefill_8192"),
                             ("128_sliding", "sliding", "prefill_128"), ("128_full", "full", "prefill_128")):
        for where, block in (("stage", stage["performance"]), ("top-level", contract["performance"])):
            expect(near(block["prefill_us_device"][field], dev[(kind, tag)][1], 0.05),
                   f"contract {where} prefill_us_device[{field}]",
                   f"says {block['prefill_us_device'][field]}, CSV gives {dev[(kind, tag)][1]:.1f}")

    # ------------------------------------------- the fractured-norm accounting
    _, before_ops, _ = capture(BASELINE, "sliding", "prefill_8192")
    _, after_ops, _ = capture(DOC, "sliding", "prefill_8192")
    totals = {
        "six RMSNorms": (group(before_ops, NORM_OPS), group(after_ops, NORM_OPS)),
        "collectives": (group(before_ops, COLLECTIVE_OPS), group(after_ops, COLLECTIVE_OPS)),
    }
    #: The same two totals are tabulated in both documents under slightly
    #: different row labels; try each spelling and fail once if none is present.
    aliases = {
        "six RMSNorms": ("six RMSNorms", "six prefill RMSNorms (8192, sliding)"),
        "collectives": ("collectives", "prefill collectives"),
    }
    for doc_name, text in (("README.md", readme), ("work_log.md", work_log)):
        for label, (before, after) in totals.items():
            cells = None
            for candidate in aliases[label]:
                found = find_row(text, candidate)
                if found:
                    cells = found
                    break
            if cells is None:
                expect(False, f"{doc_name} '{label}'", f"no row labelled any of {aliases[label]}")
                continue
            got_before, got_after = numbers(cells[1]), numbers(cells[2])
            expect(bool(got_before) and near(got_before[0], before, 0.15), f"{doc_name} '{cells[0]}' before",
                   f"says {got_before[:1]}, CSV gives {before:.1f}")
            expect(bool(got_after) and near(got_after[0], after, 0.15), f"{doc_name} '{cells[0]}' after",
                   f"says {got_after[:1]}, CSV gives {after:.1f}")
    expect(near(stage["performance"]["prefill_norm_us_device"]["8192_sliding"], totals["six RMSNorms"][1], 0.15),
           "contract prefill_norm_us_device", f"CSV gives {totals['six RMSNorms'][1]:.1f}")

    # ------------------------------------------------------------ the whole-layer A/B
    ab = ab_rows(DOC / "logs" / "final_layer_ab.log")
    single = ab_rows(DOC / "logs" / "final_layer_ab_single.log")
    for key, (name, kind, field) in {
        ("traced decode, sliding @2048", "decode"): ("tp4", "sliding", "decode_ms"),
        ("traced decode, full @2048", "decode"): ("tp4", "full", "decode_ms"),
    }.items():
        label = key[0]
        cells = row_cells(readme, label, "README.md")
        if cells is None:
            continue
        before_vals, after_vals = numbers(cells[1]), numbers(cells[2])
        for candidate, vals in (("before", before_vals), ("tp4", after_vals)):
            got = [ab[(n, kind)][field] for n in (("before", "beforeb") if candidate == "before" else ("tp4", "tp4b", "tp4c")) if (n, kind) in ab]
            expect(sorted(round(v, 4) for v in vals) == sorted(round(v, 4) for v in got),
                   f"README.md '{label}' {candidate}", f"says {sorted(vals)}, log gives {sorted(got)}")
        mean_before = sum(ab[(n, kind)][field] for n in ("before", "beforeb")) / 2
        mean_after = sum(ab[(n, kind)][field] for n in ("tp4", "tp4b", "tp4c")) / 3
        pct = 100.0 * (mean_after - mean_before) / mean_before
        expect(bool(numbers(cells[3])) and near(numbers(cells[3])[0], pct, 0.02), f"README.md '{label}' delta",
               f"says {numbers(cells[3])[:1]}, log gives {pct:+.2f} %")
    for field, name, kind in (("sliding@2048", "tp4", "sliding"), ("full@2048", "tp4", "full")):
        expect(near(stage["performance"]["traced_decode_ms_per_token_e2e"][field], ab[(name, kind)]["decode_ms"], 5e-4),
               f"contract traced_decode_ms_per_token_e2e[{field}]", f"log gives {ab[(name, kind)]['decode_ms']}")
    for field, kind in (("traced_decode_sliding@2048", "sliding"), ("traced_decode_full@2048", "full")):
        expect(near(stage["performance"]["single_chip_baseline_e2e"][field], single[("single", kind)]["decode_ms"], 5e-4),
               f"contract single_chip_baseline_e2e[{field}]", f"log gives {single[('single', kind)]['decode_ms']}")

    # -------------------------------------------------------------- correctness
    worst = dict(re.findall(r"worst\[([^\]]+)\]: ([\d.]+)", (DOC / "logs" / "vs_single_chip_run.log").read_text(errors="ignore")))
    label_of = {
        "sliding seq_len=2049 batch=1": "sliding, 2049, batch 1",
        "full seq_len=2049 batch=1": "full, 2049, batch 1",
        "sliding seq_len=12345 batch=4": "sliding, 12345, batch 4",
        "full seq_len=12345 batch=4": "full, 12345, batch 4",
    }
    for log_label, value in worst.items():
        cells = row_cells(readme, label_of[log_label], "README.md")
        if cells is None:
            continue
        expect(bool(numbers(cells[1])) and near(numbers(cells[1])[0], float(value), 5e-7),
               f"README.md vs-single-chip '{label_of[log_label]}'", f"says {numbers(cells[1])[:1]}, log gives {value}")
    # The baseline's values are legitimate in a "before" column and in prose that
    # names them as the baseline's; what must not happen is one of them appearing
    # as *this stage's* worst value, which is the first cell of those rows.
    for log_label, value in worst.items():
        cells = row_cells(readme, label_of[log_label], "README.md")
        if cells and len(cells) > 2:
            expect(numbers(cells[1])[0] != numbers(cells[2])[0] or log_label.startswith("full seq_len=12345"),
                   f"README.md vs-single-chip '{label_of[log_label]}'",
                   "this stage's value equals the baseline's; only the decode row should")

    suite = (DOC / "logs" / "full_test_run.log").read_text(errors="ignore")
    passed = re.search(r"(\d+) passed", suite)
    expect(passed is not None, "suite log", "no pass count")
    if passed:
        for doc_name, text in (("README.md", readme), ("work_log.md", work_log)):
            expect(f"**{passed.group(1)} passed**" in text, f"{doc_name} suite pass count",
                   f"'{passed.group(1)} passed' not stated")
        expect(stage["tests"]["passed"] == int(passed.group(1)) + 4, "contract tests.passed",
               f"says {stage['tests']['passed']}, logs give {int(passed.group(1)) + 4}")

    # --------------------------------- probe figures: in their log and in the docs
    named = (
        ("1348.0", "prefill_ccl_probe.log"), ("1588.7", "prefill_ccl_probe.log"),
        ("2606.3", "prefill_ccl_probe.log"), ("2086.6", "prefill_ccl_probe.log"),
        ("4443.9", "fractured_prefill_probe.log"), ("5902.1", "fractured_prefill_probe.log"),
        ("44.91", "fused_ccl_probe.log"), ("87.47", "fused_ccl_probe.log"),
        ("64.74", "fused_ccl_gathered_input.log"), ("65.84", "fused_ccl_gathered_input.log"),
        ("40.50", "packing_probe.log"), ("41.05", "packing_probe.log"),
        ("142.96", "packing_probe.log"), ("145.66", "packing_probe.log"),
        ("0.739526", "regression_bisect.log"), ("0.774936", "regression_bisect.log"),
        ("1.34", "ab_frac_norm_gate.log"), ("1.18", "ab_frac_norm_gate.log"),
    )
    for value, log_name in named:
        body = (DOC / "logs" / log_name).read_text(errors="ignore")
        expect(value in body, f"probe figure {value}", f"not in logs/{log_name}")
        expect(value in readme or value in work_log, f"probe figure {value}", "not quoted in either document")

    # ------------------ no figure quoted in shipped source may lack an artifact
    for source in (DOC.parent.parent / "tt" / "multichip_decoder.py", DOC.parent.parent / "tests" / "test_multichip_decoder.py"):
        text = source.read_text()
        # Inherited docstrings legitimately quote the multichip stage's own
        # measurements, so its logs count as artifacts here too.
        logs = "\n".join(
            q.read_text(errors="ignore")
            for q in list((DOC / "logs").glob("*.log")) + list((BASELINE / "logs").glob("*.log"))
        )
        for quoted in re.findall(r"(?<![\w.])0\.4[0-9]{3}(?![\w])", text):
            expect(quoted in logs, f"{source.name} quotes {quoted}", "no committed log contains it")

    print(f"checked {checks} claims against committed artifacts")
    for failure in failures:
        print(f"  STALE  {failure}")
    if failures:
        print(f"{len(failures)} claim(s) do not match the artifacts they cite")
        return 1
    print("every checked claim matches the artifact it cites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
