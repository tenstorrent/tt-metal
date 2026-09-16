"""CLI entry point for issue_verifier.

Invoked via: python .github/issue_verifier/run_issue_verifier.py [args]

Two stages, because they belong on different runners. `plan` reads the issue on
a cheap host and decides which pool can settle it; `probe` runs the experiment
on whatever pool `plan` picked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from loguru import logger

from issue_verifier import skus
from issue_verifier.agent import PLANNER_TOOLS, PROBER_TOOLS, AgentFailed, load_prompt, run_session
from issue_verifier.issues import IssueFetchFailed, fetch_issue
from issue_verifier.report import render
from issue_verifier.verdict import Verdict, assess

JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)

# Only a reproduced bug should look like a pass. Everything else is either a
# rejection or an unfinished job, and neither should read as "verified".
EXIT_CODES = {
    Verdict.LIKELY_VALID: 0,
    Verdict.CLAIM_UNSOUND: 3,
    Verdict.NOT_REPRODUCED: 4,
    Verdict.NEEDS_HARDWARE: 5,
    Verdict.NEEDS_HUMAN: 6,
}


def _extract_json(text: str, what: str) -> dict:
    blocks = JSON_BLOCK.findall(text)
    if not blocks:
        raise ValueError(f"{what}: no ```json block in session output:\n{text[-2000:]}")
    try:
        return json.loads(blocks[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what}: malformed JSON block: {exc}\n{blocks[-1][:2000]}") from exc


def _emit_outputs(**values: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def _append_summary(markdown: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a") as handle:
        handle.write(markdown)


def stage_plan(number: int, repo: str, workdir: Path, outdir: Path, model: str | None) -> int:
    issue = fetch_issue(number, repo)
    logger.info(f"planning verification for #{number}: {issue['title'][:80]}")

    prompt = load_prompt(
        "plan.md",
        number=str(issue["number"]),
        author=issue["author"],
        body=issue["body"],
        sku_choices=skus.describe_choices(),
        host_only_sku=skus.HOST_ONLY_SKU,
    )
    result = run_session(prompt, cwd=workdir, tools=PLANNER_TOOLS, model=model, timeout_s=900)
    logger.info(f"planning session done in {result.turns} turns (${result.cost_usd:.2f})")

    plan = _extract_json(result.text, "planner")

    sku = plan.get("sku") or skus.DEFAULT_SKU
    try:
        runs_on = skus.load_runs_on(sku)
    except skus.UnknownSku as exc:
        logger.warning(f"{exc}; falling back to {skus.DEFAULT_SKU}")
        sku = skus.DEFAULT_SKU
        runs_on = skus.load_runs_on(sku)
        plan["sku_rationale"] = f"{plan.get('sku_rationale', '')} (overridden: requested SKU not allowlisted)"
    plan["sku"] = sku

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "issue.json").write_text(json.dumps(issue, indent=2))
    (outdir / "plan.json").write_text(json.dumps(plan, indent=2))

    _emit_outputs(
        sku=sku,
        runs_on=json.dumps(runs_on),
        verifiable=str(plan.get("verifiable", False)).lower(),
        op=str(plan.get("op") or ""),
    )
    logger.info(f"plan: verifiable={plan.get('verifiable')} sku={sku} runs_on={runs_on}")

    if not plan.get("verifiable", False):
        # Render the bail-out now: the probe job will be skipped, and a job
        # summary that says nothing is worse than one that says why.
        assessment = assess(plan, {})
        markdown = render(issue=issue, plan=plan, measurements={}, assessment=assessment, sku=sku)
        (outdir / "report.md").write_text(markdown)
        _append_summary(markdown)

    return 0


def stage_probe(workdir: Path, outdir: Path, model: str | None, timeout_s: int) -> int:
    issue = json.loads((outdir / "issue.json").read_text())
    plan = json.loads((outdir / "plan.json").read_text())
    sku = plan.get("sku", skus.DEFAULT_SKU)

    if not plan.get("verifiable", False):
        logger.warning("plan marked the issue unverifiable; nothing to probe")
        assessment = assess(plan, {})
        markdown = render(issue=issue, plan=plan, measurements={}, assessment=assessment, sku=sku)
        (outdir / "report.md").write_text(markdown)
        _append_summary(markdown)
        return EXIT_CODES[assessment.verdict]

    hardware_note = (
        "a Tenstorrent device is attached; Gate C must run"
        if skus.needs_hardware(sku)
        else "no Tenstorrent device on this runner; skip Gate C"
    )
    prompt = load_prompt(
        "probe.md",
        plan=json.dumps(plan, indent=2),
        sku=sku,
        hardware_note=hardware_note,
        host_only_sku=skus.HOST_ONLY_SKU,
    )

    measurements: dict = {}
    try:
        result = run_session(prompt, cwd=workdir, tools=PROBER_TOOLS, model=model, timeout_s=timeout_s)
        logger.info(f"probe session done in {result.turns} turns (${result.cost_usd:.2f})")
        measurements = _extract_json(result.text, "prober")
    except (AgentFailed, ValueError) as exc:
        logger.error(f"probe failed: {exc}")
        measurements = {"gate_a": {"ran": False, "error": str(exc)[:1000]}}

    (outdir / "measurements.json").write_text(json.dumps(measurements, indent=2))

    assessment = assess(plan, measurements)
    (outdir / "verdict.json").write_text(
        json.dumps(
            {
                "issue": issue["number"],
                "verdict": assessment.verdict.value,
                "headline": assessment.headline,
                "reasons": assessment.reasons,
                "flags": assessment.flags,
                "sku": sku,
            },
            indent=2,
        )
    )

    markdown = render(issue=issue, plan=plan, measurements=measurements, assessment=assessment, sku=sku)
    (outdir / "report.md").write_text(markdown)
    _append_summary(markdown)

    logger.info(f"verdict: {assessment.verdict.value} — {assessment.headline}")
    print(markdown)
    return EXIT_CODES[assessment.verdict]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="issue_verifier",
        description="Verify a tt-metal bug report by running it, not by reading it.",
    )
    parser.add_argument("--issue", type=int, help="Issue number (required for --stage plan/all).")
    parser.add_argument("--repo", default="tenstorrent/tt-metal", help="Target repository.")
    parser.add_argument(
        "--stage",
        choices=["plan", "probe", "all"],
        default="all",
        help="plan: extract the experiment and pick a SKU. probe: run it. all: both (local use).",
    )
    parser.add_argument("--outdir", type=Path, default=Path("issue-verifier-out"), help="Artifact directory.")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="tt-metal checkout the agent works in.")
    parser.add_argument("--model", default=None, help="Model override passed to the Claude CLI.")
    parser.add_argument("--timeout", type=int, default=2700, help="Probe session timeout in seconds.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO", format="<level>{level: <8}</level> {message}")

    if args.stage in ("plan", "all") and not args.issue:
        parser.error("--issue is required for --stage plan and --stage all.")

    outdir = args.outdir.resolve()
    workdir = args.workdir.resolve()

    try:
        if args.stage in ("plan", "all"):
            stage_plan(args.issue, args.repo, workdir, outdir, args.model)
        if args.stage in ("probe", "all"):
            return stage_probe(workdir, outdir, args.model, args.timeout)
    except (IssueFetchFailed, AgentFailed, ValueError) as exc:
        logger.error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
