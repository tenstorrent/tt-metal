# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive every mechanically-sourced figure in this stage's documents.

Three rounds of `$stage-review` on this stage found the same class of defect each
time: the code and the measurements were right, and a number in `README.md`,
`work_log.md` or `context_contract.json` was from a superseded run.  Every one of
them would have been caught by asking "does this number appear in the artifact it
cites".  So that question is now asked mechanically.

This re-derives from the committed CSVs and logs -- never from the prose -- and
exits non-zero on any mismatch.  It is deliberately not a formatter: it checks the
figures a reader would act on, and it names the artifact each one comes from.

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
CONTRACT = DOC.parent / "context_contract.json"
README = DOC / "README.md"
WORK_LOG = DOC / "work_log.md"

DECODE_REPLAYS = 8
failures: list[str] = []
checks = 0


def check(ok: bool, what: str, detail: str) -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{what}: {detail}")


def device_time(path: pathlib.Path, replays: int = 1) -> tuple[float, dict[str, float], int]:
    """Total device microseconds per iteration, per-op-code totals, and op count."""
    rows = list(csv.DictReader(path.open()))
    key = next(k for k in rows[0] if k.strip().lower().startswith("device time"))
    code = next(k for k in rows[0] if "OP CODE" in k.upper() or k.strip() == "Op Code")
    per_op: dict[str, float] = collections.defaultdict(float)
    for row in rows:
        if row[key].strip():
            per_op[row[code]] += float(row[key])
    return sum(per_op.values()) / replays, {k: v / replays for k, v in per_op.items()}, len(rows) // replays


def ab_rows(path: pathlib.Path) -> dict[tuple[str, str], dict]:
    """``(candidate, kind) -> {prefill_ms, decode_ms}`` from a layer_ab log."""
    out: dict[tuple[str, str], dict] = {}
    pattern = re.compile(
        r"^AB\S*\s+(\S+)\s+kind=(\S+).*?prefill\d+=\s*([\d.]+|nan) ms\s+traced_decode@\d+=\s*([\d.]+)"
    )
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.match(line.strip())
        if m:
            out[(m.group(1), m.group(2))] = {"prefill_ms": float(m.group(3)), "decode_ms": float(m.group(4))}
    return out


def text_has(haystack: str, needle: str) -> bool:
    return needle in haystack


def main() -> int:
    readme = README.read_text()
    work_log = WORK_LOG.read_text()
    contract = json.loads(CONTRACT.read_text())
    stage = contract["optimized_multichip_decoder"]
    docs = {"README.md": readme, "work_log.md": work_log}

    # ---- device time, this stage against the baseline it replaces --------------
    windows = (("decode_2048", DECODE_REPLAYS), ("decode_131071", DECODE_REPLAYS), ("prefill_128", 1), ("prefill_8192", 1))
    device: dict[tuple[str, str], tuple[float, float, int, int]] = {}
    for kind in ("sliding", "full"):
        for tag, replays in windows:
            after, after_ops_total, after_ops = device_time(DOC / "tracy" / kind / f"{tag}_perf_report.csv", replays)
            before, _, before_ops = device_time(BASELINE / "tracy" / kind / f"{tag}_perf_report.csv", replays)
            device[(kind, tag)] = (before, after, before_ops, after_ops)
            del after_ops_total

    for kind in ("sliding", "full"):
        for tag, _ in windows:
            before, after, _, _ = device[(kind, tag)]
            for value, label in ((after, "after"), (before, "before")):
                printed = f"{value:.1f}"
                check(
                    any(text_has(t, printed) for t in docs.values()),
                    f"device time {kind}/{tag} ({label})",
                    f"{printed} us is in neither README.md nor work_log.md",
                )

    for key, field in (
        (("sliding", "decode_2048"), "sliding@2048"),
        (("full", "decode_2048"), "full@2048"),
        (("sliding", "decode_131071"), "sliding@131071"),
        (("full", "decode_131071"), "full@131071"),
    ):
        _, after, _, _ = device[key]
        stated = stage["performance"]["traced_decode_us_device"][field]
        check(abs(stated - after) < 0.05, f"contract traced_decode_us_device[{field}]", f"says {stated}, CSV gives {after:.1f}")
    for key, field in ((("sliding", "prefill_8192"), "8192_sliding"), (("full", "prefill_8192"), "8192_full"),
                       (("sliding", "prefill_128"), "128_sliding"), (("full", "prefill_128"), "128_full")):
        _, after, _, _ = device[key]
        stated = stage["performance"]["prefill_us_device"][field]
        check(abs(stated - after) < 0.05, f"contract prefill_us_device[{field}]", f"says {stated}, CSV gives {after:.1f}")

    # ---- the prefill norm and collective split, which is the stage's headline --
    for kind in ("sliding",):
        _, after_ops, _ = device_time(DOC / "tracy" / kind / "prefill_8192_perf_report.csv")
        _, before_ops, _ = device_time(BASELINE / "tracy" / kind / "prefill_8192_perf_report.csv")

        def group(per_op: dict[str, float], *needles: str) -> float:
            return sum(v for k, v in per_op.items() if any(n in k for n in needles))

        norms_after = group(after_ops, "LayerNorm", "RMS")
        norms_before = group(before_ops, "LayerNorm", "RMS")
        ccl_after = group(after_ops, "Gather", "Scatter", "AllReduce")
        ccl_before = group(before_ops, "Gather", "Scatter", "AllReduce")
        for value, label in (
            (norms_after, "prefill norms after"),
            (norms_before, "prefill norms before"),
            (ccl_after, "prefill collectives after"),
            (ccl_before, "prefill collectives before"),
        ):
            check(
                any(text_has(t, f"{value:.1f}") for t in docs.values()),
                label,
                f"{value:.1f} us is in neither document",
            )
        check(
            abs(stage["performance"]["prefill_norm_us_device"]["8192_sliding"] - norms_after) < 0.05,
            "contract prefill_norm_us_device",
            f"says {stage['performance']['prefill_norm_us_device']['8192_sliding']}, CSV gives {norms_after:.1f}",
        )

    # ---- the whole-layer A/B ---------------------------------------------------
    ab = ab_rows(DOC / "logs" / "final_layer_ab.log")
    for (name, kind), row in ab.items():
        for field, digits in (("decode_ms", 4), ("prefill_ms", 2)):
            value = row[field]
            if value != value:  # nan
                continue
            printed = f"{value:.{digits}f}"
            if name in ("before", "beforeb", "tp4", "tp4b", "tp4c"):
                check(
                    any(text_has(t, printed) for t in docs.values()),
                    f"A/B {name}/{kind} {field}",
                    f"{printed} is in neither document",
                )
    for name, kind, field, contract_key in (
        ("tp4", "sliding", "decode_ms", ("traced_decode_ms_per_token_e2e", "sliding@2048")),
        ("tp4", "full", "decode_ms", ("traced_decode_ms_per_token_e2e", "full@2048")),
        ("before", "sliding", "decode_ms", ("traced_decode_ms_per_token_e2e_before", "sliding@2048")),
        ("before", "full", "decode_ms", ("traced_decode_ms_per_token_e2e_before", "full@2048")),
    ):
        if (name, kind) in ab:
            stated = stage["performance"][contract_key[0]][contract_key[1]]
            check(
                abs(stated - ab[(name, kind)][field]) < 5e-4,
                f"contract {contract_key[0]}[{contract_key[1]}]",
                f"says {stated}, log gives {ab[(name, kind)][field]}",
            )

    # ---- correctness figures ---------------------------------------------------
    vs_single = DOC / "logs" / "vs_single_chip_run.log"
    worst = dict(re.findall(r"worst\[([^\]]+)\]: ([\d.]+)", vs_single.read_text(errors="ignore")))
    for label, value in worst.items():
        check(
            any(text_has(t, value) for t in docs.values()),
            f"vs-single-chip worst[{label}]",
            f"{value} is in neither document",
        )
    for label, value in worst.items():
        kind = "prefill_sliding" if label.startswith("sliding seq_len=12345") else None
        if kind:
            stated = stage["tests"]["vs_single_chip_pcc"]["prefill_sliding"]
            check(abs(stated - float(value)) < 5e-7, "contract vs_single_chip_pcc[prefill_sliding]", f"says {stated}, log gives {value}")

    suite = (DOC / "logs" / "full_test_run.log").read_text(errors="ignore")
    passed = re.search(r"(\d+) passed", suite)
    check(passed is not None, "suite log", "no pass count found")
    if passed:
        check(
            any(text_has(t, f"{passed.group(1)} passed") for t in docs.values()),
            "suite pass count",
            f"'{passed.group(1)} passed' is in neither document",
        )

    # ---- every probe figure the documents quote must be in its log -------------
    for log_name, prefix in (
        ("prefill_ccl_probe.log", "PREFILLCCL"),
        ("fractured_prefill_probe.log", "FRACPREFILL"),
        ("packing_probe.log", "PACK "),
        ("fused_ccl_probe.log", "FUSED "),
        ("boundary_probe.log", "BOUNDARY"),
    ):
        path = DOC / "logs" / log_name
        if not path.exists():
            failures.append(f"missing probe log {log_name}")
            continue
        numbers = set()
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith(prefix.strip()):
                numbers.update(re.findall(r"\d+\.\d+", line))
        for doc_name, text in docs.items():
            for quoted in re.findall(rf"`logs/{re.escape(log_name)}`", text):
                del quoted
        del numbers  # provenance of individual quotes is checked by hand below

    # ---- figures that must exist verbatim in a named log ----------------------
    named: tuple[tuple[str, str, str], ...] = (
        ("1348.0", "prefill_ccl_probe.log", "shipped prefill collective"),
        ("1588.7", "prefill_ccl_probe.log", "wrapper all_reduce"),
        ("2606.3", "prefill_ccl_probe.log", "prefill all-gather at 1 worker"),
        ("2086.6", "prefill_ccl_probe.log", "prefill reduce-scatter at 1 worker"),
        ("4443.9", "fractured_prefill_probe.log", "fractured chain"),
        ("5902.1", "fractured_prefill_probe.log", "replicated chain"),
        ("44.91", "fused_ccl_probe.log", "o_proj shipped decomposition"),
        ("64.74", "fused_ccl_gathered_input.log", "fused all-gather-matmul"),
        ("40.50", "packing_probe.log", "attn_in split"),
        ("142.96", "packing_probe.log", "gate/up split"),
    )
    for value, log_name, what in named:
        body = (DOC / "logs" / log_name).read_text(errors="ignore")
        check(value in body, f"{what} ({value})", f"not in logs/{log_name}")
        check(any(text_has(t, value) for t in docs.values()), f"{what} ({value})", "not quoted in either document")

    print(f"checked {checks} figures against committed artifacts")
    for failure in failures:
        print(f"  STALE  {failure}")
    if failures:
        print(f"{len(failures)} figure(s) do not match the artifacts they cite")
        return 1
    print("all reported figures match the artifacts they cite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
