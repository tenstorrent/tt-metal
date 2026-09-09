#!/usr/bin/env python3
"""Regenerate the derived sanity reports from sanity_test_suite.json.

sanity_test_suite.json is the authoritative manifest: the runner reads it and
every count, lane, gap and per-kernel row below is projected from it. The
markdown, HTML and CSV are outputs only; edit the manifest, then rerun this.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

REPORT_DIR = Path("ttnn/cpp/ttnn/kernel_lib/reduce_migration_inventory_2026-09-08_f808380a87b")

COVERED_STATUSES = ("selected", "covered")

KIND_LABELS = {
    "compute": "Compute",
    "dataflow": "Dataflow calling old helpers",
    "manual auxiliary reader": "Manual auxiliary readers",
}

LANE_REQUIREMENTS = {
    "common": "One device; shared Wormhole/Blackhole test sources. Individual core-grid and debug-mode constraints still apply.",
    "wormhole": "One Wormhole device: two Moreh layernorm backward cases, the Falcon causal-mask operation case and the BGE encoder SDPA case.",
    "blackhole": "One Blackhole device: KDA, indexer and sparse attention cases.",
    "quasar": "Single logical Quasar device/emulator; requires that environment and build.",
    "fabric-1x4": "Four devices, 1x4 mesh with 2D fabric; requires unit_tests_ttnn_udm.",
    "wormhole-n300": "Exactly two Wormhole devices (N300), 2x1 mesh.",
    "blackhole-2x4": "Exactly eight Blackhole devices in a 2x4 box.",
    "galaxy": "Existing DiT fixture opens a 4x8 Galaxy mesh (32 devices), then TP1/TP2 submeshes.",
    "wormhole-t3k": "Eight Wormhole devices (2x4 T3K).",
    "blackhole-galaxy": "Blackhole Galaxy cluster; the selected experimental ring case uses a 1x4 submesh.",
}

RUN_COMMANDS = """# Inspect the selection without importing TTNN or opening devices.
python scripts/run_reduce_migration_sanity.py --list

# Usual single-device selections; choose the line for your architecture.
python scripts/run_reduce_migration_sanity.py --lane common --lane wormhole
python scripts/run_reduce_migration_sanity.py --lane common --lane blackhole

# Other lanes run separately on the matching hardware, for example:
python scripts/run_reduce_migration_sanity.py --lane wormhole-n300

# Inspect/collect the whole suite, or run a kernel's primary case.
python scripts/run_reduce_migration_sanity.py --dry-run
python scripts/run_reduce_migration_sanity.py --collect-only
python scripts/run_reduce_migration_sanity.py --kernel S071

# Select every case, including those needing other hardware.
python scripts/run_reduce_migration_sanity.py"""

SUBSET_NOTE = (
    "This is a subset of the existing [full unit suite](unit_test_suite.html), which collected 18,452 cases. "
    "The original inventory counts source kernels, including three compute/dataflow pairs embedded in Python. "
    "A kernel can run incidentally in additional cases; one primary case is assigned to it in the coverage map. "
    "This does not cover every template instantiation, layout, accumulation mode or helper call site."
)

MD_NOTES = [
    "Every invocation goes through `scripts/run_safe_pytest.sh`, with serial execution, `--run-all` and "
    "`--no-precompile` by default. The C++ adapter runs one named GTest per pytest case. Required binaries are "
    "`build/test/ttnn/unit_tests_ttnn` and, for UDM, `build/test/ttnn/unit_tests_ttnn_udm`. The build must match "
    "the checkout being tested.",
    "Each group must collect exactly one case. During execution, a skip, xfail, missing test or any failure makes "
    "the runner return nonzero; it continues other groups and saves their results. Collection does not establish "
    "that a case will run on the current hardware. Hardware lanes are explicit selectors, not hardware discovery "
    "or a guarantee that every SKU/debug configuration is supported. Running all lanes on one machine is generally "
    "not possible.",
    "The `{{arch}}` token in a few node IDs substitutes only the `silicon_arch_name` fixture from pytest's "
    "`--tt-arch`; all numerical parameters remain fixed. The plugin selects the resulting exact node and the runner "
    "checks its cardinality. For a collection targeting another build, pass e.g. `-- --tt-arch=blackhole`.",
    "The inline examples retain their existing fixed width loops (compute fusion: 2 tile counts; reduce accumulate: "
    "4; row reduce accumulate: 6), but the runner pins their supported environment selectors to the helper variants. "
    "DiT's correctness sweep is pinned to `cross_k_prompt_L512` and `CORR_DET_REPEATS=0`, keeping its numerical check "
    "and removing extra determinism launches. Moreh callback tests and some other tests also make several calls "
    "internally. **{cases} is a pytest-case count, not a device-launch count.**",
    "Results go to a new directory under `generated/test_reports/`, with per-case logs, JUnit XML, collection "
    "metadata and `summary.json`. `--output-dir` must name a new directory, so previous results are preserved.",
]

HTML_RUNNER_NOTE = (
    "All cases use <code>run_safe_pytest.sh</code>. Each must collect exactly once and pass; skips, xfails, missing "
    "cases and failures return nonzero. The runner continues the other groups. No arguments select every hardware "
    "lane; use lanes for the machine you have. The C++ build must match the checkout."
)

HTML_RESULTS_NOTE = (
    "Results are saved in a new <code>generated/test_reports/</code> directory with logs, JUnit XML and summary "
    "JSON. The inline example selectors are pinned to helper variants; their fixed width loops remain. DiT selects "
    "one numerical configuration and no extra determinism repetitions. Architecture fixture tokens adapt to "
    "<code>--tt-arch</code>; numerical parameters stay fixed."
)

SELECTION_EVIDENCE = (
    "Selection was checked against factory/test sources at `{commit}`. Python/script-only additions require no "
    "C++ build."
)

HTML_SELECTION_EVIDENCE = (
    "Selection evidence comes from the factory and test sources at <code>{commit}</code>. This is a "
    "Python/script-only change, requiring no C++ build."
)

LANE_NOTE = (
    "Common denotes shared single-device test sources, not verification on every SKU or debug configuration. "
    "Running every lane on one machine is generally not possible."
)

GAP_INTRO = "These are missing executable paths, not cases silently skipped by the suite."

GAP_NOTE = (
    "The disabled general large-H softmax factory does not add a kernel gap: `SM036` explicitly runs Moreh "
    "`LARGE_H`, which uses the same compute and reader sources (`S071`, `DF029`)."
)

HTML_GAP_NOTE = (
    "The disabled general large-H softmax factory adds no kernel gap: SM036 runs the shared S071/DF029 kernels "
    "using explicit Moreh LARGE_H."
)

HTML_HEAD = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reduce migration sanity suite</title><style>body{margin:0;background:#f6f8fc;color:#182333;font:15px/1.6 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:32px}a{color:#2257a0}h1{font-size:30px}h2{margin-top:36px}h3{margin-top:0}code{font-size:12px;overflow-wrap:anywhere}pre{background:#eaf0f7;border-radius:6px;padding:16px;overflow:auto;white-space:pre-wrap}table{width:100%;border-collapse:collapse;background:white;margin:16px 0}th,td{text-align:left;border-bottom:1px solid #dbe2ec;padding:10px;vertical-align:top}th{background:#e7eef8}td.path{max-width:480px;overflow-wrap:anywhere;font-size:12px}.card{background:white;border:1px solid #dbe2ec;border-radius:8px;margin:16px 0;padding:18px}.counts{font-size:20px;font-weight:600}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}input,select{font:inherit;padding:8px;border:1px solid #a9b5c5;border-radius:5px}input{flex:1;min-width:250px}.gap{background:#fff4e5}.muted{color:#536176}.tablewrap{overflow:auto}nav{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:700px){main{padding:18px}th,td{padding:6px}}</style><main>"""

HTML_SCRIPT = """<script>const q=document.querySelector("#q"),lane=document.querySelector("#lane"),cards=[...document.querySelectorAll(".case")];function filter(){let count=0;for(const card of cards){const show=(!lane.value||card.dataset.lane===lane.value)&&card.textContent.toLowerCase().includes(q.value.toLowerCase());card.hidden=!show;if(show)count++;}document.querySelector("#visible").textContent=`${count} of ${cards.length} cases`;}q.addEventListener("input",filter);lane.addEventListener("change",filter);filter();</script></html>"""


def nodeid(group: dict) -> str:
    tests = group.get("tests") or []
    return f"{group['source']}::{tests[0]}" if tests else group["source"]


def environment_pairs(group: dict) -> list[str]:
    return [f"{key}={value}" for key, value in (group.get("environment") or {}).items()]


def counts(manifest: dict) -> dict:
    groups = manifest["groups"]
    kernels = manifest["kernels"]
    covered = [k for k in kernels if k["status"] in COVERED_STATUSES]
    return {
        "cases": len(groups),
        "python": sum(1 for g in groups if g["kind"] == "pytest"),
        "cpp": sum(1 for g in groups if g["kind"] == "gtest"),
        "kernels": len(kernels),
        "covered": len(covered),
        "uncovered": len(kernels) - len(covered),
    }


def category_rows(manifest: dict) -> list[tuple[str, int, int]]:
    rows = []
    for kind, label in KIND_LABELS.items():
        entries = [k for k in manifest["kernels"] if k["kind"] == kind]
        covered = [k for k in entries if k["status"] in COVERED_STATUSES]
        rows.append((label, len(covered), len(entries)))
    return rows


def lane_rows(manifest: dict) -> list[tuple[str, int, str]]:
    order = list(LANE_REQUIREMENTS)
    seen = {}
    for group in manifest["groups"]:
        seen[group["lane"]] = seen.get(group["lane"], 0) + 1
    unknown = sorted(set(seen) - set(order))
    if unknown:
        raise SystemExit(f"lanes missing a documented requirement: {unknown}")
    return [(lane, seen[lane], LANE_REQUIREMENTS[lane]) for lane in order if lane in seen]


def verification_sentence(collection: dict) -> str:
    failed = collection.get("failed_groups") or []
    arch = collection.get("architecture_template_check") or {}
    arch_count = arch.get("collected_cases")
    arch_target = arch.get("tt_arch", "blackhole")
    sentence = (
        f"Collected {collection['collected_cases']} cases across {collection['groups_completed']} groups "
        f"with {len(failed)} failed groups. No on-device test bodies were run. Host checks verified failure "
        "propagation for skips, xfails, assertions and case-count drift, plus continuation to a later passing case."
    )
    if arch_count:
        sentence += (
            f" {arch_count} architecture-dependent selections also collected exactly once with --tt-arch={arch_target}."
        )
    return sentence


def write_markdown(path: Path, manifest: dict, collection: dict) -> None:
    c = counts(manifest)
    out = [
        "# Reduce-helper migration sanity suite",
        "",
        f"**{c['cases']} exact test cases: {c['python']} Python cases and {c['cpp']} C++ GTests.** "
        f"They provide one primary numerical case for each of **{c['covered']} of {c['kernels']} kernel entries**. "
        "Shared coverage is deduplicated: one case can cover compute plus reader/writer kernels. "
        f"The {c['uncovered']} uncovered entries are listed below.",
        "",
        "| Kernel category | Covered | Inventory total |",
        "| --- | ---: | ---: |",
    ]
    for label, covered, total in category_rows(manifest):
        out.append(f"| {label} | {covered} | {total} |")
    out.append(f"| Total | {c['covered']} | {c['kernels']} |")
    out += ["", SUBSET_NOTE, "", "```bash", RUN_COMMANDS, "```", ""]
    for note in MD_NOTES:
        out += [note.format(cases=c["cases"]), ""]

    out += ["## Hardware lanes", "", "| Lane | Cases | Requirements |", "| --- | ---: | --- |"]
    for lane, cases, requirement in lane_rows(manifest):
        out.append(f"| `{lane}` | {cases} | {requirement} |")
    out += ["", LANE_NOTE, "", "## Coverage gaps", "", GAP_INTRO, ""]
    for gap in manifest["gaps"]:
        ids = ", ".join(f"`{k}`" for k in gap["kernels"])
        out.append(f"- {ids}: {gap['reason']}")
    verification = f"{verification_sentence(collection)} {SELECTION_EVIDENCE.format(commit=manifest['commit'])}"
    out += ["", GAP_NOTE, "", "## Verification", "", verification, ""]
    out += [
        "## Exact selections and kernel map",
        "",
        "[Collection evidence](sanity_test_collection.json). See the [HTML report](sanity_test_suite.html), "
        "[manifest](sanity_test_suite.json), and [one-row-per-kernel CSV](sanity_kernel_coverage.csv).",
        "",
    ]
    for group in manifest["groups"]:
        kernels = ", ".join(f"`{k}`" for k in group["kernel_ids"])
        out += [
            f"### {group['id']} — {group['family']}",
            "",
            f"Lane: `{group['lane']}`. Primary kernels: {kernels}.",
            "",
            group["selection"],
            "",
            "```text",
            nodeid(group),
            "```",
            "",
        ]
        pinned = environment_pairs(group)
        if pinned:
            out += ["Pinned environment: " + ", ".join(f"`{pair}`" for pair in pinned) + ".", ""]
    path.write_text("\n".join(out).rstrip("\n") + "\n")


def write_html(path: Path, manifest: dict, collection: dict) -> None:
    c = counts(manifest)
    groups = {g["id"]: g for g in manifest["groups"]}
    e = html.escape
    blob = f"https://github.com/tenstorrent/tt-metal/blob/{manifest['commit']}/"
    out = [
        HTML_HEAD,
        "<h1>Reduce-helper migration sanity suite</h1>",
        '<nav><a href="index.html">Migration inventory</a><a href="unit_test_suite.html">Full unit suite</a>'
        '<a href="sanity_test_suite.json">Manifest</a><a href="sanity_kernel_coverage.csv">Kernel coverage CSV</a>'
        '<a href="sanity_test_suite.md">Markdown</a></nav>',
        f'<p class="counts">{c["cases"]} cases · {c["covered"]} kernels covered · {c["uncovered"]} gaps</p>',
        f"<p>{c['python']} exact Python parameter cases and {c['cpp']} named C++ GTests. Each covered kernel has one "
        "primary numerical case; shared compute/reader/writer coverage is deduplicated. Some cases make multiple "
        "device calls internally. Coverage is per source kernel, not every helper call or template variant.</p>",
    ]
    table = "<table><tr><th>Kernel category</th><th>Covered</th><th>Inventory total</th></tr>"
    for label, covered, total in category_rows(manifest):
        table += f"<tr><td>{label}</td><td>{covered}</td><td>{total}</td></tr>"
    table += f"<tr><th>Total</th><th>{c['covered']}</th><th>{c['kernels']}</th></tr></table>"
    out.append(table)
    out.append(f"<h2>Run the suite</h2><pre><code>{e(RUN_COMMANDS, quote=True)}\n</code></pre>")
    out.append(f"<p>{HTML_RUNNER_NOTE}</p>")
    out.append(f"<p>{HTML_RESULTS_NOTE}</p>")
    lanes = lane_rows(manifest)
    out.append("<h2>Hardware lanes</h2><table><tr><th>Lane</th><th>Cases</th><th>Requirements</th></tr>")
    for lane, cases, requirement in lanes:
        out.append(f"<tr><td><code>{lane}</code></td><td>{cases}</td><td>{e(requirement)}</td></tr>")
    out.append(f'</table><p class="muted">{e(LANE_NOTE)}</p><h2>Coverage gaps</h2>')
    for gap in manifest["gaps"]:
        ids = ", ".join(gap["kernels"])
        out.append(f'<div class="card gap"><b>{ids}</b><p>{e(gap["reason"])}</p></div>')
    out.append(f"<p>{e(HTML_GAP_NOTE)}</p>")
    out.append(
        f"<h2>Verification</h2><p>{e(verification_sentence(collection))}</p>"
        '<p><a href="sanity_test_collection.json">Collection evidence</a></p>'
        f"<p>{HTML_SELECTION_EVIDENCE.format(commit=manifest['commit'])}</p>"
    )
    out.append(
        '<h2>Exact cases</h2><div class="controls"><label for="q">Search</label><input id="q" type="search" '
        'placeholder="Kernel, test, factory or shape"><label for="lane">Lane</label>'
        '<select id="lane"><option value="">All lanes</option>'
    )
    for lane, _, _ in lanes:
        out.append(f"<option>{lane}</option>")
    out.append('</select><span id="visible" aria-live="polite"></span></div><div id="cases">')
    for group in manifest["groups"]:
        factories = ", ".join(f for f in group.get("factories", []) if f)
        out.append(
            f'<article class="card case" id="{group["id"]}" data-lane="{group["lane"]}">'
            f'<h3>{group["id"]} — {e(group["family"])}</h3>'
        )
        meta = (
            f'<p><b>Primary kernels:</b> {", ".join(group["kernel_ids"])} · <b>Lane:</b> <code>{group["lane"]}</code>'
        )
        if factories:
            meta += f" · <b>Factories:</b> {factories}"
        out.append(meta + "</p>")
        out.append(f'<p>{e(group["selection"])}</p><p>{e(group["hardware"])}</p>')
        out.append(f"<pre><code>{e(nodeid(group), quote=True)}</code></pre>")
        pinned = environment_pairs(group)
        if pinned:
            rendered = ", ".join(f"<code>{e(pair)}</code>" for pair in pinned)
            out.append(f"<p>Pinned environment: {rendered}</p>")
        evidence = group.get("test_evidence")
        if evidence:
            out.append(f'<p class="muted">{e(evidence)}</p>')
        out.append("</article>")
    out.append(
        '</div><h2>One row per kernel</h2><div class="tablewrap"><table><tr><th>ID</th><th>Kind</th>'
        "<th>Source</th><th>Primary case</th><th>Selection / gap</th></tr>"
    )
    for kernel in manifest["kernels"]:
        case = kernel.get("case") or ""
        case_cell = f'<a href="#{case}">{case}</a>' if case else "—"
        selection = (groups.get(case) or kernel).get("selection") or ""
        row_class = "" if kernel["status"] in COVERED_STATUSES else ' class="gap"'
        out.append(
            f'<tr{row_class}><td><code>{kernel["id"]}</code></td><td>{e(kernel["kind"])}</td>'
            f'<td class="path"><a href="{blob}{kernel["path"]}">{e(kernel["path"])}</a></td>'
            f"<td>{case_cell}</td><td>{e(selection)}</td></tr>"
        )
    out.append("</table></div></main>" + HTML_SCRIPT)
    path.write_text("\n".join(out) + "\n")


def check_kernel_group_agreement(manifest: dict) -> None:
    """The group owns every case-derived field; kernels[] keeps a copy for lookup."""
    groups = {g["id"]: g for g in manifest["groups"]}
    drift = []
    for kernel in manifest["kernels"]:
        group = groups.get(kernel.get("case") or "")
        if not group:
            continue
        for field in ("selection", "test_evidence"):
            if (kernel.get(field) or "") != (group.get(field) or ""):
                drift.append(f"{kernel['id']} ({group['id']}) {field}")
    if drift:
        raise SystemExit(
            "kernels[] disagrees with its group in sanity_test_suite.json; sync the kernel copies "
            "from the group entries first:\n  " + "\n  ".join(drift)
        )


def write_csv(path: Path, manifest: dict) -> None:
    groups = {g["id"]: g for g in manifest["groups"]}
    columns = [
        "kernel_id",
        "kind",
        "kernel_path",
        "status",
        "case_id",
        "lane",
        "test",
        "test_evidence",
        "factory_ids",
        "selection",
        "kernel_or_factory_evidence",
        "hardware",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for kernel in manifest["kernels"]:
            group = groups.get(kernel.get("case") or "")
            writer.writerow(
                {
                    "kernel_id": kernel["id"],
                    "kind": kernel["kind"],
                    "kernel_path": kernel["path"],
                    "status": kernel["status"],
                    "case_id": kernel.get("case") or "",
                    "lane": group["lane"] if group else "",
                    "test": nodeid(group) if group else "",
                    "test_evidence": group.get("test_evidence", "") if group else "",
                    "factory_ids": "; ".join(f for f in kernel.get("factories", []) if f),
                    "selection": (group or kernel).get("selection") or "",
                    "kernel_or_factory_evidence": kernel.get("evidence") or "",
                    "hardware": group["hardware"] if group else "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--collection", type=Path, help="Collection evidence JSON (default: report dir copy)")
    args = parser.parse_args()

    report_dir = args.report_dir
    manifest = json.loads((report_dir / "sanity_test_suite.json").read_text())
    collection = json.loads((args.collection or report_dir / "sanity_test_collection.json").read_text())

    if collection["groups_completed"] != len(manifest["groups"]):
        raise SystemExit(
            f"collection evidence covers {collection['groups_completed']} groups but the manifest has "
            f"{len(manifest['groups'])}; rerun --collect-only before regenerating"
        )

    check_kernel_group_agreement(manifest)
    write_markdown(report_dir / "sanity_test_suite.md", manifest, collection)
    write_html(report_dir / "sanity_test_suite.html", manifest, collection)
    write_csv(report_dir / "sanity_kernel_coverage.csv", manifest)

    c = counts(manifest)
    print(
        f"Regenerated md/html/csv: {c['cases']} cases ({c['python']} Python + {c['cpp']} C++), "
        f"{c['covered']}/{c['kernels']} kernel entries covered, {c['uncovered']} gaps."
    )


if __name__ == "__main__":
    main()
