# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive every headline number in README.md from the artifacts, and check it is there.

A README is a claim until something re-computes it.  This gate reads the JSON and
log artifacts, formats each figure exactly as the document quotes it, and fails if
the string is absent -- so a number that goes stale after a re-measurement is a
failing check rather than a sentence nobody re-read.

It deliberately checks *rendered strings* rather than parsing prose: the failure
mode being guarded is "the artifact changed and the document did not", and a
missing string catches that regardless of the sentence around it.

Usage::

    python doc/datatype_sweep/bench/check_reported_figures.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
D = ROOT / "doc/datatype_sweep"
PREV = ROOT / "doc/optimized_full_model"
#: The stage whose HF qualitative control this stage reuses.
PREV_FULL_QUAL = ROOT / "doc/full_model/qualitative"


def load(path: pathlib.Path):
    return json.loads(path.read_text())


def main() -> int:
    readme = (D / "README.md").read_text()
    work_log = (D / "work_log.md").read_text()
    results = load(D / "sweep_results.json")
    rows = {c["config_id"]: c for c in results["configs"]}
    selected_id = results["selected"]["config_id"]
    selected = rows[selected_id]
    baseline = rows["c00-baseline-attn8-mlp4-kv8-lofi"]
    perf = load(D / "evidence_perf.json")["performance"]
    prev_perf = load(PREV / "evidence_perf.json")["performance"]
    accuracy = load(D / "evidence_accuracy.json")
    config = load(D / "selected_precision_config.json")
    contract = load(ROOT / "doc/context_contract.json")

    figures: dict[str, str] = {}

    def add(name: str, text: str) -> None:
        figures[name] = text

    # ---- the selected config and its gate
    add("selected id", selected_id)
    add("selected top1", f"{selected['accuracy']['top1']:.3f}")
    add("selected top5", f"{selected['accuracy']['top5']:.3f}")
    add(
        "selected teacher-forcing median",
        f"{selected['performance']['teacher_forcing_decode_tok_s_u_median']:.3f} t/s/u",
    )
    tf_rounds = sorted((selected["performance"]["teacher_forcing_decode_tok_s_u_rounds"] or [])[1:])
    add("selected teacher-forcing range", f"{tf_rounds[0]:.3f}–{tf_rounds[-1]:.3f}")
    add("selected rounds", f"median of {len(selected['performance']['teacher_forcing_decode_tok_s_u_rounds'])}")

    # ---- post-selection token-out, and the previous stage's, in the same regime
    add(
        "token-out after",
        f"{perf['token_out_decode_ms_per_token']['min']:.3f} ms/token · "
        f"{perf['token_out_decode_tok_s_u']:.2f} t/s/u",
    )
    add(
        "token-out before",
        f"{prev_perf['token_out_decode_ms_per_token']['min']:.3f} ms/token · "
        f"{prev_perf['token_out_decode_tok_s_u']:.2f} t/s/u",
    )
    delta = (perf["token_out_decode_ms_per_token"]["min"] / prev_perf["token_out_decode_ms_per_token"]["min"] - 1) * 100
    add("token-out delta", f"{delta:.2f} %")
    add(
        "logits-only after",
        f"{perf['traced_decode_logits_only_ms_per_token']['min']:.3f} ms/token · "
        f"{perf['traced_decode_logits_only_tok_s_u']:.2f} t/s/u",
    )
    add(
        "logits-only before",
        f"{prev_perf['traced_decode_logits_only_ms_per_token']['min']:.3f} ms/token · "
        f"{prev_perf['traced_decode_logits_only_tok_s_u']:.2f} t/s/u",
    )
    logits_delta = (
        perf["traced_decode_logits_only_ms_per_token"]["min"]
        / prev_perf["traced_decode_logits_only_ms_per_token"]["min"]
        - 1
    ) * 100
    add("logits-only delta", f"{logits_delta:.2f} %")
    add("post-selection ttft", f"{perf['ttft_ms']['min']:.2f} ms")
    add("sampling trace", f"{perf['sampling_trace_ms_per_token']['min']:.3f} ms")

    # ---- baseline teacher-forcing, and the teacher-forcing delta
    add(
        "baseline teacher-forcing",
        f"{baseline['performance']['teacher_forcing_decode_tok_s_u_median']:.3f} t/s/u",
    )
    base_rounds = sorted((baseline["performance"]["teacher_forcing_decode_tok_s_u_rounds"] or [])[1:])
    add("baseline teacher-forcing range", f"{base_rounds[0]:.3f}–{base_rounds[-1]:.3f}")
    tf_delta = (
        selected["performance"]["teacher_forcing_decode_tok_s_u_median"]
        / baseline["performance"]["teacher_forcing_decode_tok_s_u_median"]
        - 1
    ) * 100
    add("teacher-forcing delta", f"{tf_delta:.2f} %")

    # ---- long-lived DRAM, before and after
    add("dram after", f"{accuracy['capacity']['per_device_total_bytes'] / 1e9:.3f} GB")
    prev_accuracy = load(PREV / "evidence_accuracy.json")
    add("dram before", f"{prev_accuracy['capacity']['per_device_total_bytes'] / 1e9:.3f} GB")
    add(
        "dram delta",
        f"{(prev_accuracy['capacity']['per_device_total_bytes'] - accuracy['capacity']['per_device_total_bytes']) / 1e6:.0f} MB",
    )

    # ---- capability
    add("context", str(contract["current_supported_context"]))
    add("capability reduction", contract["capability_reduction"])
    kv = contract["kv_cache_dtype_capacity"]["by_dtype"]
    add("kv bfp8 fits", str(kv["bfloat8_b"]["full_context_sequences_that_fit"]))
    add("kv bfp4 fits", str(kv["bfloat4_b"]["full_context_sequences_that_fit"]))

    # ---- the fastest-but-rejected candidate
    c08 = rows["c08-attn4-kv4-cclbfp8"]
    c08_rounds = sorted((c08["performance"]["teacher_forcing_decode_tok_s_u_rounds"] or [])[1:])
    add("c08 median", f"{c08['performance']['teacher_forcing_decode_tok_s_u_median']:.3f}")
    add("c08 range", f"{c08_rounds[0]:.3f}–{c08_rounds[-1]:.3f}")
    add("c08 top1", f"{c08['accuracy']['top1']:.3f}")

    # ---- the candidate table itself.  Round 1 of the stage review found six
    # count errors in prose; a re-measurement can just as easily leave a stale
    # *row*, so every row of the README's candidate table is re-derived from the
    # results rather than spot-checked.
    table_problems: list[str] = []
    seen_rows: set[str] = set()
    for line in readme.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6 or not re.fullmatch(r"`c\d\d`", cells[0]):
            continue
        short = cells[0].strip("`")
        try:
            # Cells may be bolded (``**0.970**``); the figure is the number.
            top1, tf, lo = (float(cells[i].strip("*")) for i in (2, 3, 4))
        except ValueError:
            continue
        match = [c for c in results["configs"] if c["config_id"].startswith(short + "-")]
        if not match:
            table_problems.append(f"README table row {short} has no matching config")
            continue
        row = match[0]
        seen_rows.add(row["config_id"])
        expected = (
            row["accuracy"]["top1"],
            row["performance"]["teacher_forcing_decode_tok_s_u_median"],
            row["performance"]["traced_logits_only_tok_s_u"],
        )
        if any(e is None for e in expected):
            table_problems.append(f"README table row {short} quotes numbers for a config with none")
            continue
        for label, got, want, tol in (
            ("top-1", top1, expected[0], 1e-9),
            ("teacher-forcing", tf, expected[1], 5e-4),
            ("logits-only", lo, expected[2], 5e-4),
        ):
            if abs(got - want) > tol:
                table_problems.append(f"README table row {short} {label}: says {got}, artifact says {want:.3f}")

    # ---- the Pareto frontier the charts draw, named in the README.  Round 1 of
    # the stage review found the document naming the selected config as a frontier
    # point when it is not one, so the membership is re-derived here.
    def frontier(metric: str) -> list[str]:
        points = [
            (c["accuracy"][metric], c["performance"]["teacher_forcing_decode_tok_s_u_median"], c["config_id"])
            for c in results["configs"]
            if c["accuracy"][metric] is not None
            and c["performance"]["teacher_forcing_decode_tok_s_u_median"] is not None
        ]
        return sorted(p[2] for p in points if not any(o[0] >= p[0] and o[1] >= p[1] and o[:2] != p[:2] for o in points))

    top1_front = frontier("top1")
    add("top-1 frontier size", {1: "one point", 2: "exactly two points"}.get(len(top1_front), str(len(top1_front))))
    for config_id in top1_front:
        add(f"top-1 frontier member {config_id}", f"`{config_id.split('-', 1)[0]}`")

    # ---- coverage.  Round 1 of the stage review found six count errors and a
    # false "every candidate was smoketested" claim, none of which this gate
    # looked at: it checked rendered numbers and not what they were counting.
    smoketest = load(D / "smoketest.json")
    # Every candidate must have *been* smoketested; a candidate whose smoketest
    # recorded the op-contract blocker is covered, not uncovered -- that is the
    # smoketest doing its job, and it is why the blockers below are quoted from a
    # two-layer build rather than only from the 52-layer sweep.
    smoketested = set(smoketest["results"])
    smoketest_blocked = {k for k, v in smoketest["results"].items() if "error" in v}
    configs = {path.stem for path in (D / "configs").glob("*.json")}
    measured = {c["config_id"] for c in results["configs"] if c["status"] == "ok"}
    blocked = {c["config_id"] for c in results["configs"] if c["status"] != "ok"}
    add("candidate count", str(len(configs)))
    add("measured count", str(len(measured)))
    add("blocked count", str(len(blocked)))

    # The same counts, spelled out, must not contradict them anywhere in either
    # document.  Round 2 of the stage review found the corrected counts in the
    # README and the *uncorrected* ones still in the work log, because this gate
    # only ever read the README.
    #
    # The rule is deliberately blunt -- a wrong spelled-out count fails even when
    # a document is quoting it as a historical error -- because the alternative is
    # teaching a gate to parse quotation, and describing a past mistake instead of
    # reproducing its exact words costs a sentence.
    words = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
        10: "Ten",
        11: "Eleven",
        12: "Twelve",
        13: "Thirteen",
        14: "Fourteen",
        15: "Fifteen",
        16: "Sixteen",
        17: "Seventeen",
        18: "Eighteen",
        19: "Nineteen",
        20: "Twenty",
    }
    wrong_counts: list[str] = []
    for document, name in ((readme, "README.md"), (work_log, "work_log.md")):
        lowered = document.lower()
        for actual, phrase in (
            (len(blocked), "candidates produced no number"),
            (len(blocked), "candidates did not produce a number"),
        ):
            for number, word in words.items():
                if number != actual and f"{word.lower()} {phrase}" in lowered:
                    wrong_counts.append(f"{name}: '{word} {phrase}' but {actual} are blocked")
        for number, word in words.items():
            if number != 17 and f"{word.lower()} are host-only" in lowered:
                wrong_counts.append(f"{name}: '{word} are host-only' but there are 17")

    # ---- the blocked candidates' own numbers.  Three of the five complete a full
    # prefill accuracy pass before failing at decode-trace capture, and the README
    # states both the split and the triples; re-derive both.
    for row in results["configs"]:
        if row["status"] == "ok":
            continue
        run = load(REPO / row["run_artifact"])
        prefill = (run.get("prefill_check") or {}).get("per_entry")
        short = row["config_id"].split("-", 1)[0]
        if prefill:
            entry = prefill[0]
            add(
                f"{short} blocked prefill triple",
                f"| `{short}` | decode-trace capture | {entry['top1']:.3f} / {entry['top5']:.3f} / "
                f"{entry['top100']:.3f} |",
            )
        else:
            add(f"{short} blocked in prefill", f"| `{short}` | prefill | — |")

    # ---- the qualitative arms must have run the same prompt token ids.  The
    # suite's system message embeds the current date, so a reused HF control plus
    # a re-rendered TT prompt silently puts the two arms on different inputs and
    # makes ``first_divergence_from_hf`` a measure of the calendar.  Round 2 of
    # the stage review found exactly that.
    qual = D / "qualitative"
    prompts = {i["id"]: i for i in load(qual / "qualitative_prompts.json")}
    control_prompts = {i["id"]: i for i in load(PREV_FULL_QUAL / "qualitative_prompts.json")}
    prompt_problems = [
        f"qualitative prompt {key} does not match the reused HF control's token ids"
        for key in sorted(prompts)
        if key not in control_prompts or prompts[key]["token_ids"] != control_prompts[key]["token_ids"]
    ]
    comparison = load(qual / "qualitative_comparison_chat.json")
    add("qualitative divergences", ", ".join(str(r["first_divergence_from_hf"]) for r in comparison))
    add("qualitative worst adjacent dup", f"{max(r['tt_adjacent_dup'] for r in comparison):.4f}")

    # ---- the cross-check spread quoted in the regimes table
    ratios = []
    for row in results["configs"]:
        run = load(REPO / row["run_artifact"])
        rounds = sorted((run.get("traced_logits_only") or {}).get("rounds_ms") or [])
        if len(rounds) >= 2:
            ratios.append((rounds[1] / rounds[0] - 1) * 100)
    if ratios:
        add("cross-check best-two spread", f"{min(ratios):.3f}–{max(ratios):.3f} %")
    # ...and the third round's systematic penalty against the mean of the first
    # two, which is the figure the regimes table quotes as the reason the metric
    # takes a min.  Round 3 found it stale after a candidate was added.
    penalties = []
    for row in results["configs"]:
        run = load(REPO / row["run_artifact"])
        rounds = sorted((run.get("traced_logits_only") or {}).get("rounds_ms") or [])
        if len(rounds) == 3:
            penalties.append((rounds[2] / ((rounds[0] + rounds[1]) / 2) - 1) * 100)
    if penalties:
        add("cross-check third-round penalty", f"{min(penalties):.2f}–{max(penalties):.2f} %")

    # ---- figures the work log quotes that the artifacts own.  Round 3 found the
    # work log carrying a superseded per-candidate throughput and two stale
    # counts, none of which this gate looked at because it only read prose
    # *phrases* there.  Numbers get the same treatment as the README's table.
    work_log_problems: list[str] = []
    junit = (D / "test_results_precision_config.xml").read_text()
    total_tests = int(re.search(r'tests="(\d+)"', junit).group(1))
    device_tests = len(re.findall(r'time="(?:1[0-9]|[2-9][0-9])\.', junit))
    for config_id, tf in (
        (row["config_id"], row["performance"]["teacher_forcing_decode_tok_s_u_median"])
        for row in results["configs"]
        if row["status"] == "ok"
    ):
        short = config_id.split("-", 1)[0]
        for stale in re.findall(rf"`{short}`[^\n]*?it measures ([0-9]+\.[0-9]+)", work_log):
            if abs(float(stale) - tf) > 5e-4:
                work_log_problems.append(f"work_log.md quotes {short} at {stale} but the artifact says {tf:.3f}")
    pass_one = len(list((D / "runs_pass1_rounds5").glob("*.json")))
    for number, word in words.items():
        if number != pass_one and f"five rounds, {word.lower()} candidates" in work_log.lower():
            work_log_problems.append(
                f"work_log.md says pass 1 ran {word.lower()} candidates but runs_pass1_rounds5/ holds {pass_one}"
            )
        if number != total_tests - 17 and f"{word.lower()} are device cases" in work_log.lower():
            work_log_problems.append(
                f"work_log.md says {word.lower()} device cases but the junit has {total_tests - 17}"
            )

    # ---- the trace-verification counter, and the accuracy references
    replays = selected["performance"]["trace_replays_per_round"]
    if set(replays) != {99}:
        print(f"FIGURE_GATE trace_replays_per_round is {sorted(set(replays))}, expected all 99", file=sys.stderr)
        return 1
    add("trace replays", "**99**")

    def normalise(text: str) -> str:
        """Typographic dashes in prose must not fail a numeric check.

        The document uses U+2212 MINUS and U+2013 EN DASH where a formatter
        produces ASCII ``-``; the figure being checked is the number, not the
        glyph, so both sides are folded to ASCII before the lookup.
        """
        return text.replace("\u2212", "-").replace("\u2013", "-").replace("\u00a0", " ")

    haystack = normalise(readme)
    problems = [name for name, text in figures.items() if normalise(text) not in haystack]

    # ---- structural checks that are not string lookups
    structural: list[str] = []
    if config["config_id"] != selected_id:
        structural.append(
            f"selected_precision_config.json is {config['config_id']!r}, the sweep selected {selected_id!r}"
        )
    if accuracy["capacity"].get("precision_config_id") != selected_id:
        structural.append(
            "evidence_accuracy.json was not produced by the selected config: "
            f"{accuracy['capacity'].get('precision_config_id')!r}"
        )
    if accuracy.get("build_kwargs"):
        structural.append(
            f"evidence_accuracy.json build_kwargs is {accuracy['build_kwargs']!r}, expected {{}} -- the "
            "post-selection evidence must come from a build with no precision knobs passed"
        )
    for name in ("top1_perf_pareto.png", "top5_perf_pareto.png"):
        if not (D / name).is_file():
            structural.append(f"missing chart {name}")
        if name not in readme:
            structural.append(f"{name} is not shown in the README")
    for row in results["configs"]:
        if row["status"] == "ok" and not row["propagation_verified"]:
            structural.append(f"{row['config_id']} was measured without a verified precision propagation")
    # every config id the README names must exist in the results
    for cited in sorted(set(re.findall(r"`(c\d\d)`", readme))):
        if not any(c.startswith(cited) for c in rows):
            structural.append(f"README cites {cited} but no such config was evaluated")
    # coverage: every artifact has a run, every candidate was smoketested, and
    # every measured candidate is named in the README's table
    for missing in sorted(configs - set(rows)):
        structural.append(f"{missing} has a config artifact but no run artifact")
    for missing in sorted(set(rows) - configs):
        structural.append(f"{missing} has a run artifact but no config artifact")
    for missing in sorted(configs - smoketested):
        structural.append(
            f"{missing} was never smoketested: $datatype-sweep requires the one-decoder smoketest "
            "before a candidate is used or rejected in the sweep"
        )
    for config_id in sorted(measured):
        if f"`{config_id.split('-', 1)[0]}`" not in readme:
            structural.append(f"{config_id} was measured but is not named in the README")
    for config_id in sorted(blocked):
        if f"`{config_id.split('-', 1)[0]}`" not in readme:
            structural.append(f"{config_id} was blocked but is not named in the README")
    # A candidate blocked in the 52-layer sweep must be blocked in the two-layer
    # smoketest too, and vice versa: a divergence means one of the two is not
    # measuring the policy the other is.
    if selected_id in top1_front and "the selected point is not one" in readme:
        structural.append(
            "the README says the selected point is not on the top-1 frontier, but it is: " f"{top1_front}"
        )
    if selected_id not in top1_front and "the selected point is not one" not in readme:
        structural.append(
            f"the selected config is not on the top-1 Pareto frontier ({top1_front}) and the README " "does not say so"
        )
    structural.extend(table_problems)
    structural.extend(wrong_counts)
    structural.extend(prompt_problems)
    structural.extend(work_log_problems)
    # Accuracy stability is claimed for the whole triple, so check the triple from
    # the raw rounds rather than trusting the derived flag -- older run artifacts
    # computed it over (top-1, top-5) only.
    for row in results["configs"]:
        if row["status"] != "ok":
            continue
        run = load(REPO / row["run_artifact"])
        triples = {
            (r["per_entry"][0]["top1"], r["per_entry"][0]["top5"], r["per_entry"][0]["top100"])
            for r in run["teacher_forcing_rounds"]
        }
        if len(triples) != 1:
            structural.append(f"{row['config_id']} accuracy is not stable across rounds: {sorted(triples)}")

    # every measured candidate's teacher-forcing run must be traced, not just the
    # selected one -- the README's claim is about the set
    for row in results["configs"]:
        if row["status"] != "ok":
            continue
        counts = set(row["performance"]["trace_replays_per_round"] or [])
        if counts != {99}:
            structural.append(f"{row['config_id']} trace_replays_per_round is {sorted(counts)}, expected all 99")
    # The selected artifact's provenance must describe the sweep that selected it.
    provenance = config.get("provenance") or {}
    if provenance.get("selection_reason") != results["selected"]["reason"]:
        structural.append(
            "selected_precision_config.json:provenance.selection_reason does not match "
            "sweep_results.json:selected.reason -- the artifact describes a different sweep"
        )
    if provenance.get("tied_within_tolerance") != results["selected"]["tied_within_tolerance"]:
        structural.append(
            "selected_precision_config.json:provenance.tied_within_tolerance does not match "
            "sweep_results.json:selected"
        )
    for config_id in sorted(measured - seen_rows):
        structural.append(f"{config_id} was measured but has no row in the README candidate table")
    if smoketest_blocked != blocked:
        structural.append(f"smoketest blocked {sorted(smoketest_blocked)} but the sweep blocked {sorted(blocked)}")

    for name in problems:
        print(f"FIGURE_GATE missing from README.md: {name} = {figures[name]!r}", file=sys.stderr)
    for problem in structural:
        print(f"FIGURE_GATE {problem}", file=sys.stderr)
    if problems or structural:
        print(f"FIGURE_GATE_FAILED {len(problems)} missing figure(s), {len(structural)} structural", file=sys.stderr)
        return 1
    print(f"FIGURE_GATE_OK {len(figures)} figures re-derived and found in README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
