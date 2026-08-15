# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive every mechanically-sourced figure and path in this stage's reports.

Nine consecutive review rounds on this stage found the same defect class and little
else: a document claiming more than the artifact it cited actually showed. Several
times the offending text was written to *fix* the previous instance. Prose cannot be
linted, but the three things that went wrong every time can be:

  * a **number** in `README.md` / `work_log.md` that no longer matches the JSON or
    log it came from,
  * a **path** cited by a report that does not exist, and
  * a **claim** a previous round already corrected, reasserted somewhere else.

This is the same instrument `doc/multichip_decoder/bench/check_reported_figures.py`
is for the decoder stage, pointed at the serving artifacts. It checks against the
committed artifacts, never against itself:

  * the headline single-user metrics, from `readiness_vllm/vllm_benchmark.json`;
  * the CI serving-burst metrics, from `vllm_ci_serving_benchmark.json`;
  * TPOT-derived decode t/s/u, recomputed as `1000 / mean_tpot_ms`;
  * the sampling suite's pass/fail/skip counts, parsed from `sampling_tests.log`,
    with a row required for every file and every failure named in the README;
  * the failure taxonomy: the 3/7 split, its sum, and which group is "resolved";
  * the qualitative comparison, raw and with the API-stripped control token removed,
    per prompt, plus duplication, non-ASCII, trigram bands and their direction;
  * the p1 channel-margin dismissal, from the datatype-sweep stage's probe;
  * the worst adjacent duplication, parsed from `logs/degenerate_check_all.log`;
  * the served context against `doc/context_contract.json`;
  * the KV pool block count and byte figure;
  * the logit-determinism candidate count;
  * every `doc/...`, `readiness_vllm/...` or `logs/...` path cited in the reports,
    and every bare `*.json` / `*.log` filename cited outside a fenced block;
  * the stale-claim guards (see `_unquoted` and `_sentences`), which forbid a
    document from *asserting* a claim a review round already corrected, while
    leaving it free to quote one.

Every check is verified by perturbation, and each round's entry in `work_log.md`
records the perturbation that would catch its regression.

    python check_reported_figures.py            # report
    python check_reported_figures.py --check    # exit 1 on any mismatch
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/vllm_integration/
MODEL = ROOT.parent.parent  # models/autoports/<model>/
README = ROOT / "README.md"
WORK_LOG = ROOT / "work_log.md"
AUTOFIX = ROOT / "AUTOFIX.md"
AUTODEBUG = ROOT / "AUTODEBUG.md"

TOLERANCE = 0.005  # 0.5 % on measured quantities


class Failures(list):
    def check(self, label: str, reported, derived, *, exact: bool = False) -> None:
        if exact:
            ok = reported == derived
        else:
            try:
                ok = abs(float(reported) - float(derived)) <= TOLERANCE * max(abs(float(derived)), 1e-9)
            except (TypeError, ValueError):
                ok = reported == derived
        status = "ok " if ok else "BAD"
        print(f"  [{status}] {label}: reported {reported!r} vs derived {derived!r}")
        if not ok:
            self.append(f"{label}: reported {reported!r}, derived {derived!r}")

    def require(self, label: str, condition: bool, detail: str = "") -> None:
        print(f"  [{'ok ' if condition else 'BAD'}] {label}{(' — ' + detail) if detail and not condition else ''}")
        if not condition:
            self.append(f"{label}{(': ' + detail) if detail else ''}")


def _unquoted(text: str) -> str:
    """The document's own assertions, with quotations of other text removed.

    The stale-claim scan below has to let the review-history sections reproduce a
    corrected defect verbatim -- that is what they are for -- without letting a
    stale claim be *asserted* anywhere.

    The first attempt at that scoped by section: cut the work log at the first
    ``## N. Stage review`` heading and scan only what came before. That is a blind
    spot, not a rule. It excluded ``## 7.1``, which is live technical content, and
    it excluded every section the stage has yet to write, since new sections are
    appended after the review history. Both were confirmed by planting the round-4
    sentence in each and watching the checker pass.

    The rule here is about form instead of position, so it has no such gap: a
    review section may **quote** the defective sentence -- inside quotation marks, a
    fence or a blockquote -- and may not state it as the document's own claim. That
    is the discipline the prose already follows.

    Markdown formatting is **normalised, not deleted.** Deleting inline code spans
    was the first attempt and it was far too permissive: these documents write every
    identifier and most numbers in backticks, so ``All `21` logprobs tests pass``
    and ``All 21 `logprobs` tests pass`` both slipped through, and the isolation
    guard could never see a sentence naming ``test_request_isolation.py`` the way
    every other sentence in these documents names it. Backticks, asterisks and
    underscores are stripped as characters instead, so emphasis cannot break a
    claim into invisible pieces either.

    Whitespace is collapsed **before** quotations are removed, and the returned text
    is collapsed too. Both matter: these documents are hard-wrapped, so a quotation
    and a claim alike routinely straddle a line break, and a scan that stops at `\n`
    silently misses either one.

    Quote pairing is **positional**: a `"` counts as an opener only after a space or
    an opening bracket, and as a closer only before a space or punctuation. Naive
    pairing is parity-dependent, and round 9 found the README carries an odd number
    of unfenced double quotes -- one of them inside `` `!"#` `` -- so the same
    sentence was exempted in one document and scanned as an assertion in the other.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # fenced blocks
    text = re.sub(r"^\s*>.*$", " ", text, flags=re.M)  # blockquotes (line-structured)
    text = re.sub(r"\s+", " ", text)  # hard wraps, before anything spans one
    # The bracket sets include the markdown markers, since a quotation is routinely
    # wrapped in emphasis (`**"..."**`) and those are only stripped on the next line.
    text = re.sub(r"(?<=[\s({\[*`_—-])[\"“][^\"“”]*?[\"”](?=[\s.,;:)\]}*`_]|$)", " ", text)  # quotations
    return re.sub(r"[`*_]", "", text)  # emphasis/code markers, as characters


def _sentences(text: str) -> list[str]:
    """Sentence-ish spans, with filename dots protected so they do not split one.

    The stale-claim guards below scope to a sentence rather than to a character
    window. Windows were what rounds 7-9 kept defeating in both directions: too
    narrow and a claim escapes by being worded longer, too wide and a true statement
    two clauses away gets flagged.
    """
    protected = re.sub(r"\.(py|json|log|sh|md|txt)\b", r"_\1", text)
    # Abbreviations too: round 10 got "All 21 logprobs tests, e.g. the chat ones,
    # pass." through by splitting the claim in half at "e.g.".
    protected = re.sub(r"\b(e|i)\.(g|e)\.", r"\1_\2_", protected)
    return re.split(r"(?<=[.;:!?])\s+", protected)


def _num(text: str, pattern: str):
    m = re.search(pattern, text)
    return float(m.group(1).replace(",", "")) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if anything mismatches")
    args = ap.parse_args()

    readme = README.read_text()
    work_log = WORK_LOG.read_text()
    fails = Failures()

    # --- headline single-user profile ---------------------------------------
    print("headline single-user 128/128/1:")
    bench = json.loads((MODEL / "readiness_vllm/vllm_benchmark.json").read_text())
    fails.check("TTFT p50 (ms)", _num(readme, r"\*\*TTFT\*\*.*?\|\s*\*\*([0-9.]+) ms"), bench["ttft_ms"]["p50"])
    fails.check(
        "decode t/s/u",
        _num(readme, r"\*\*Decode t/s/u\*\*.*?\|\s*\*\*([0-9.]+) t/s/u"),
        1000.0 / bench["tpot_ms"]["mean"],
    )
    fails.check("TPOT mean (ms)", _num(readme, r"\| TPOT mean / p99 \| ([0-9.]+) ms"), bench["tpot_ms"]["mean"])
    fails.check("ITL p50 (ms)", _num(readme, r"\| ITL p50 / p99 \| ([0-9.]+) ms"), bench["itl_ms"]["p50"])
    fails.check(
        "output throughput",
        _num(readme, r"\| Aggregate output throughput \| ([0-9.]+) tok/s"),
        bench["output_throughput_tok_per_s"],
    )
    fails.require(
        "primary profile completed every request",
        bench["completed_requests"] == bench["config"]["num_requests"] and bench["missing_output_tokens"] == 0,
    )

    # --- CI serving burst ----------------------------------------------------
    print("CI serving-burst 100/100/32:")
    ci = json.loads((MODEL / "readiness_vllm/vllm_ci_serving_benchmark.json").read_text())
    fails.check(
        "burst aggregate throughput",
        _num(readme, r"\| Aggregate output throughput \| \*\*([0-9.]+) tok/s"),
        ci["output_throughput_tok_per_s"],
    )
    fails.check(
        "burst decode t/s/u",
        _num(readme, r"\| Decode t/s/u from mean TPOT \| ([0-9.]+) t/s/u"),
        1000.0 / ci["tpot_ms"]["mean"],
    )
    fails.require("burst completed 32/32", ci["completed_requests"] == 32 and ci["missing_output_tokens"] == 0)

    # --- sampling suite counts ----------------------------------------------
    print("sampling suite:")
    log = (MODEL / "readiness_vllm/sampling_tests.log").read_text(errors="replace")
    plain = re.sub(r"\x1b\[[0-9;]*m", "", log)
    m = re.search(r"(\d+) failed, (\d+) passed, (\d+) skipped", plain)
    derived = (int(m.group(2)), int(m.group(1)), int(m.group(3))) if m else None
    reported = re.search(r"(\d+) passed, (\d+) failed, (\d+) skipped", readme)
    fails.check(
        "sampling passed/failed/skipped",
        tuple(int(g) for g in reported.groups()) if reported else None,
        derived,
        exact=True,
    )
    per_file: dict[str, dict[str, int]] = {}
    for path, status in re.findall(r"tests/tt/([a-z_0-9]+\.py)::\S+ (PASSED|FAILED|SKIPPED)", plain):
        per_file.setdefault(path, {}).setdefault(status, 0)
        per_file[path][status] += 1
    # Every file in the log must have a row. An `if row:` guard here would let the
    # table shrink silently, which is precisely the omission this check exists to
    # catch: round 4's defect was a *missing* mention of the one failing file.
    for f, counts in sorted(per_file.items()):
        row = re.search(rf"\| `{re.escape(f)}` \| (\d+) \| \*?\*?(\d+)\*?\*? \| (\d+) \|", readme)
        fails.require(f"per-file row present for {f}", row is not None, "no row in the README table")
        if row:
            fails.check(
                f"per-file {f}",
                tuple(int(g) for g in row.groups()),
                (counts.get("PASSED", 0), counts.get("FAILED", 0), counts.get("SKIPPED", 0)),
                exact=True,
            )
    fails.require(
        "the one skip is named",
        "test_chat_logprobs_all_vocab" in readme and per_file.get("test_logprobs.py", {}).get("SKIPPED") == 1,
    )
    # Round 8: the summary line folded all ten failures into the *waivable* class
    # while the section below it split them 7/3. The split is load-bearing -- only
    # the reproducibility-only class may be classified rather than fixed -- so the
    # two numbers, their sum, and the named failures are all derived.
    failed_names = [t for _, t in re.findall(r"tests/tt/([a-z_0-9]+\.py)::(\S+) FAILED", plain)]
    split = re.search(r"(\d+) correctness-class, all resolved, plus (\d+)\s+reproducibility-only", readme)
    fails.require("README states the failure taxonomy split", split is not None)
    if split:
        correctness, repro = (int(g) for g in split.groups())
        # Round 9: the round-8 replacement said all ten were "classified below and
        # all resolved". Only the three correctness-class ones are resolved; the
        # seven reproducibility-only ones are classified, and are still open in
        # Limitations. "Resolved" must not distribute over the whole ten.
        loose = [
            m.group(0)
            for m in re.finditer(r".{0,30}all resolved", _unquoted(readme))
            if "correctness-class" not in m.group(0)
        ]
        fails.require(
            "'all resolved' is attached to the correctness-class group only",
            not loose,
            f"the 7 reproducibility-only failures are classified, not resolved: {loose}",
        )
        fails.check("failure taxonomy sums to the log's failures", repro + correctness, len(failed_names), exact=True)
        # The three correctness-class failures are the ones the stage had to resolve
        # rather than classify, so they are pinned by name, not by count.
        named_correctness = [n for n in failed_names if "presence_penalt" in n or "allowed_token_ids" in n]
        fails.check("correctness-class count", correctness, len(named_correctness), exact=True)
    # `(?!\w)` so a name is not satisfied by being a prefix of something else: round 9
    # showed `test_seeding` was "found" inside `test_seeding_and_variety.py`, the
    # filename another check already requires, making its row vacuous.
    missing_names = [
        n for n in failed_names if not re.search(rf"{re.escape(n.split('::')[-1].split('[')[0])}(?!\w)", readme)
    ]
    fails.require("every failed test is named in the README", not missing_names, f"missing: {missing_names}")
    # Round 5 found the round-4 sentence uncorrected in work_log.md while the README
    # had been fixed, so both documents are scanned for stale count claims -- in full,
    # minus quotations (see `_unquoted`), so no section is out of scope.
    for doc_name, doc in (("README.md", _unquoted(readme)), ("work_log.md", _unquoted(work_log))):
        # Both guards scope to a sentence and test a *property* of it, rather than
        # matching a phrase inside a character window. Rounds 7-9 defeated the window
        # form four separate ways -- periods in `test_logprobs.py`, reversed word
        # order, longer wording, paraphrase -- while it simultaneously rejected true
        # sentences like "Of the 21 logprobs tests, 20 pass and one is skipped".
        # A window matches proximity; these match the claim.
        passed = per_file.get("test_logprobs.py", {}).get("PASSED", 0)
        pass_word = re.compile(r"\bpass(es|ed|ing)?\b")

        # Rule: a sentence that says logprobs tests pass must state the number that
        # actually passed. It may say anything else it likes about the other counts.
        stale = [
            s
            for s in _sentences(doc)
            if re.search(r"logprobs tests?\b", s) and pass_word.search(s) and not re.search(rf"\b{passed}\b", s)
        ]
        fails.require(
            f"{doc_name} makes no stale logprobs-count claim",
            not stale,
            f"a sentence says logprobs tests pass without naming the {passed} that do: {stale[:1]}",
        )
        # Rule: a sentence that mentions the isolation file *and* passing must also
        # say it fails. Order-free, so round 9's "All tests pass in
        # test_request_isolation.py" and "Everything passed in the request-isolation
        # file" are caught, while "test_request_isolation.py fails; every other file
        # passed" and "0 passed, 1 failed" -- both true, both previously rejected --
        # are not.
        clean_isolation = [
            s
            for s in _sentences(doc)
            if "isolation" in s and pass_word.search(s) and not re.search(r"\bfail(s|ed|ing|ure|ures)?\b", s)
        ]
        fails.require(
            f"{doc_name} does not call isolation a clean file",
            not clean_isolation,
            f"test_request_isolation.py has a failing test; found: {clean_isolation[:1]}",
        )

    # --- qualitative comparison ----------------------------------------------
    # Round 6's P1 sat in this section, which nothing here covered: the README
    # explained the two comparison files' `identical: false` with a cause that held
    # for one of them and not the other. Every number in that explanation is
    # mechanically derivable, so all of it is checked.
    print("qualitative comparison:")
    qual = ROOT / "qualitative"
    strip = json.loads((qual / "qualitative_stripped_divergence_chat.json").read_text())
    cmp_hf = {r["id"]: r for r in json.loads((qual / "qualitative_comparison_chat.json").read_text())}
    cmp_sweep = {r["id"]: r for r in json.loads((qual / "qualitative_vllm_vs_datatype_sweep_chat.json").read_text())}
    fails.require(
        "served == standalone once the control token is stripped",
        strip["verdict"]["served_matches_standalone_exactly_when_stripped"],
    )
    fails.require(
        "no serving-introduced early HF divergence",
        not strip["verdict"]["serving_introduced_early_hf_divergences"],
        f"prompts: {strip['verdict']['serving_introduced_early_hf_divergences']}",
    )
    # p1's channel divergence is dismissed as a precision-policy tie inherited from
    # the datatype-sweep stage. Both halves of that -- the margin and the policy it
    # was measured under -- come from that stage's probe, so both are re-derived.
    chan = json.loads((ROOT.parent / "datatype_sweep/channel_margin_probe.json").read_text())
    shipped = chan["configs"]["c14-attn4-cclbfp8-kv8"]
    for arm, pat in (
        ("prefill", r"\*\*([0-9.]+) logits\*\* in prefill"),
        ("decode", r"`=user` by ([0-9.]+) in decode"),
    ):
        top = shipped[arm]["top"]
        fails.check(
            f"p1 channel margin ({arm})",
            _num(readme.replace("\n", " "), pat),
            round(top[0]["logit"] - top[1]["logit"], 6),
        )
    fails.require(
        "the channel probe scored p1 and the shipped policy",
        chan["prompt_id"] == "p1" and shipped["decode"]["top"][0]["text"] == "=user",
    )
    # Both table cells list one value per prompt in p0..p5 order, so every cell is
    # compared elementwise against the artifacts. Round 7 killed the previous form
    # of this check: the cells read "2 on five, 1 on p1" and the checker scraped the
    # two integers out of that prose, so "2 on three" and "1 on p3" both still
    # passed -- the words carrying the actual claim were never parsed.
    ids = sorted(cmp_hf)
    hf_row = re.search(r"\| served vs HF control \| ([^|]+) \| ([^|]+) \|", readme)
    sweep_row = re.search(r"\| served vs datatype-sweep standalone \| ([^|]+) \| ([^|]+) \|", readme)
    fails.require("README carries both stripped-divergence table rows", hf_row is not None and sweep_row is not None)
    if hf_row and sweep_row:
        fails.check(
            "raw divergence vs HF, per prompt",
            [int(x) for x in re.findall(r"\d+", hf_row.group(1))],
            [cmp_hf[k]["first_divergence_from_hf"] for k in ids],
            exact=True,
        )
        fails.check(
            "stripped divergence vs HF, per prompt",
            [int(x) for x in re.findall(r"\d+", hf_row.group(2))],
            [strip["verdict"]["stripped_first_divergence_vs_hf"][k] for k in ids],
            exact=True,
        )
        fails.check(
            "raw divergence vs standalone, per prompt",
            [int(x) for x in re.findall(r"\d+", sweep_row.group(1))],
            [cmp_sweep[k]["first_divergence"] for k in ids],
            exact=True,
        )
        rows = {r["id"]: r for r in strip["rows"]}
        # "none -- identical over the full 127-token common prefix, 6/6": the token
        # count and both halves of the fraction are derived, not scraped-and-agreed.
        length, matched, total = (int(x) for x in re.findall(r"\d+", sweep_row.group(2)))
        fails.check(
            "stripped common-prefix length",
            length,
            len(json.loads((qual / "qualitative_tt_chat.json").read_text())[0]["token_ids"]),
            exact=True,
        )
        fails.check(
            "prompts identical to standalone once stripped",
            (matched, total),
            (
                sum(1 for k in ids if rows[k]["stripped_identical_to_standalone_over_common_prefix"]),
                len(ids),
            ),
            exact=True,
        )
    fails.check(
        "worst adjacent duplication in the chat arm",
        _num(readme.replace("\n", " "), r"worst adjacent\s+duplication ([0-9.]+) across the six"),
        max(r["tt_adjacent_dup"] for r in cmp_hf.values()),
    )
    fails.check(
        "worst non-ASCII fraction",
        _num(readme, r"non-ASCII ≤ ([0-9.]+)"),
        max(r["tt_non_ascii"] for r in cmp_hf.values()),
    )
    band = re.search(r"TT ([0-9.]+)-([0-9.]+), HF ([0-9.]+)-([0-9.]+)", readme)
    fails.require("README states the trigram-loop bands", band is not None)
    if band:
        tt = [r["tt_trigram_loop"] for r in cmp_hf.values()]
        hf_tri = [r["hf_trigram_loop"] for r in cmp_hf.values()]
        fails.check(
            "trigram-loop bands",
            tuple(float(g) for g in band.groups()),
            (min(tt), max(tt), min(hf_tri), max(hf_tri)),
            exact=True,
        )
        # Round 7 found the *direction* wrong on p0 in the sentence written to fix
        # round 6's wording, and the band endpoints alone could not see it. The
        # per-prompt claim is now derived too.
        flat = readme.replace("\n", " ")
        higher = re.search(r"TT runs higher on ((?:p\d/?)+) and lower on ((?:p\d/?)+)", flat)
        fails.require("README states the per-prompt trigram direction", higher is not None)
        if higher:
            fails.check(
                "prompts where TT trigram > HF",
                sorted(higher.group(1).split("/")),
                sorted(k for k in cmp_hf if cmp_hf[k]["tt_trigram_loop"] > cmp_hf[k]["hf_trigram_loop"] + 0.005),
                exact=True,
            )
            fails.check(
                "prompts where TT trigram < HF",
                sorted(higher.group(2).split("/")),
                sorted(k for k in cmp_hf if cmp_hf[k]["tt_trigram_loop"] < cmp_hf[k]["hf_trigram_loop"] - 0.005),
                exact=True,
            )
        # Round 6: "matching the HF control's" was asserted while 4 of 6 prompts
        # differ. The README must not claim the two arms match outright.
        # Round 7: the round-6 wording survived in work_log.md with "HF" dropped,
        # so this guard is document-agnostic and matches the claim, not one phrasing.
        differ = sum(1 for k in cmp_hf if cmp_hf[k]["tt_trigram_loop"] != cmp_hf[k]["hf_trigram_loop"])
        for doc_name, doc in (("README.md", _unquoted(readme)), ("work_log.md", _unquoted(work_log))):
            claim = re.search(r"trigram-loop fractions match\w* the (HF )?control", doc)
            fails.require(
                f"{doc_name} makes no unqualified 'trigram fractions match the control' claim",
                claim is None,
                f"{differ}/6 prompts differ",
            )

    # --- degenerate output ---------------------------------------------------
    print("degenerate-output check:")
    deg = (ROOT / "logs/degenerate_check_all.log").read_text(errors="replace")
    dups = [float(x) for x in re.findall(r"'adjacent_duplication': ([0-9.]+)", deg)]
    fails.check(
        "worst adjacent duplication",
        _num(readme.replace("\n", " "), r"worst adjacent duplication is\s+\*?\*?([0-9.]+)"),
        max(dups),
    )
    fails.check(
        "measurement count", _num(readme.replace("\n", " "), r"over all (\d+)\s+measurements"), float(len(dups))
    )
    fails.require("checker reported clean", "No degenerate output detected." in deg)

    # --- context contract ----------------------------------------------------
    print("capability contract:")
    contract = json.loads((MODEL / "doc/context_contract.json").read_text())
    fails.check(
        "served max_model_len",
        _num(readme, r"--max-model-len (\d+)"),
        float(contract["current_supported_context"]),
        exact=True,
    )
    fails.require("contract records no reduction", contract["capability_reduction"] == "none")

    # --- KV pool -------------------------------------------------------------
    print("KV pool:")
    probe = json.loads((ROOT / "probe_full_fixed.json").read_text())
    blocks = probe["kv"]["model_max_num_blocks"]
    fails.check("KV pool blocks", _num(readme, r"52 layers x 2 x `\((\d+),"), float(blocks), exact=True)
    kv = json.loads((ROOT / "kv_budget_probe.json").read_text())
    fails.require("measured ceiling is a lower bound", kv.get("is_proven_lower_bound_not_ceiling") is True)
    fails.check(
        "measured feasible blocks in work log",
        _num(work_log, r"\*\*(\d+) blocks / [0-9,]+ tokens\*\*"),
        float(kv["largest_feasible_blocks"]),
    )

    # --- logit determinism ---------------------------------------------------
    print("logit determinism:")
    ld = json.loads((ROOT / "logit_determinism.json").read_text())
    fails.check(
        "candidates compared",
        _num(readme.replace("\n", " "), r"([0-9]+)\s+candidate logprobs"),
        float(ld["run_to_run"]["candidates_compared"]),
    )
    fails.require(
        "both determinism verdicts true",
        all(ld["verdict"].values()) and ld["run_to_run"]["max_abs_logprob_delta"] == 0.0,
    )

    # --- every cited path exists --------------------------------------------
    print("cited paths:")
    cited: set[str] = set()
    for text in (readme, work_log, AUTOFIX.read_text(), AUTODEBUG.read_text()):
        cited |= set(re.findall(r"`((?:doc/|readiness_vllm/|logs/)[A-Za-z0-9_./*-]+)`", text))
    missing = []
    for rel in sorted(cited):
        if "*" in rel:
            continue
        base = MODEL if rel.startswith(("doc/", "readiness_vllm/")) else ROOT
        if not (base / rel).exists():
            missing.append(rel)
    fails.require("no dangling citations", not missing, f"missing: {missing}")

    # Bare filenames too. Round 9 found `probe_full.json` cited in AUTODEBUG.md for a
    # run that hung before it could write any JSON -- invisible to the check above,
    # which only resolved `doc/`, `readiness_vllm/` and `logs/` prefixes, and these
    # documents cite most artifacts by bare name.
    known = {p.name for p in ROOT.rglob("*") if p.is_file()} | {p.name for p in MODEL.rglob("*") if p.is_file()}
    # A name may be absent if every mention says so -- two probes here exit at a guard
    # before writing their JSON, and recording that is the point. The negation has to
    # sit *next to* the name: scoping it to the sentence was too loose, since a
    # "never" anywhere in a long sentence excused a citation elsewhere in it.
    bare: dict[str, list[bool]] = {}
    for text in (readme, work_log, AUTOFIX.read_text(), AUTODEBUG.read_text()):
        prose = re.sub(r"\s+", " ", re.sub(r"```.*?```", " ", text, flags=re.S))  # commands may name future outputs
        # Quotations are exempt here for the same reason they are in `_unquoted`: a
        # review section quoting a dangling citation is describing it, not making it.
        prose = re.sub(r"(?<=[\s({\[*`_—-])[\"“][^\"“”]*?[\"”](?=[\s.,;:)\]}*`_]|$)", " ", prose)
        for m in re.finditer(r"`([A-Za-z0-9_-]+\.(?:json|log))`", prose):
            # The negation must *determine* the name -- "no `x.json`" or
            # "`x.json` does not exist" -- not merely share a neighbourhood with it.
            # A proximity window still let "though the run never finished, `x.json`
            # records it" through.
            before = re.sub(r"[`*_]+$", "", prose[max(0, m.start() - 20) : m.start()])
            after = re.sub(r"^[`*_]+", "", prose[m.end() : m.end() + 30])
            declared_absent = bool(
                re.search(r"\b(no|not|without)\s*$", before)
                or re.match(
                    r"[,;]?\s*(which |that )?(does not exist|is (not|never) (produced|written)|was never written)",
                    after,
                )
            )
            bare.setdefault(m.group(1), []).append(declared_absent)
    missing_bare = sorted(n for n, flags in bare.items() if n not in known and not all(flags))
    fails.require("no dangling bare-filename citations", not missing_bare, f"missing: {missing_bare}")

    print()
    if fails:
        print(f"{len(fails)} MISMATCH(ES):")
        for f in fails:
            print(f"  - {f}")
        return 1 if args.check else 0
    print("all reported figures and cited paths re-derived from the artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
