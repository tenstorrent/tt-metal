# Create CI Tickets

## Overview
Create deterministic CI maintenance tickets for repeated failures on `main` in `tenstorrent/tt-metal`. **Quality over quantity:** only create issues you can fully validate. There is no minimum number of issues to create.

## Steps
0. **Start with a clean slate**
   - Delete stale files in `build_ci/ci_ticketing/create_tickets` so you never rely on pre-existing outputs. **Exception:** if the user specified they already ran `tools/ci/extract_failing_jobs.py`, delete everything in that folder **except** `build_ci/ci_ticketing/create_tickets/failing_jobs.json` (that file is the input for this run). Otherwise delete everything (including any old `failing_jobs.json`) so that step 1 produces a fresh `failing_jobs.json`.
   - In all cases, remove old `created_issues.jsonl`, `created_tickets_report.json`, and contents of `downloaded_logs` so this run’s outputs are not mixed with previous runs.

1. **Prepare candidate failures**
   - Enter the virtual environment defined locally in tt-metal (or tell the user to create a virtual environment if you can't find one locally in tt-metal)
   - Run `python tools/ci/extract_failing_jobs.py` (unless the user specified they already ran it—then skip this and use the existing `failing_jobs.json`).
   - Optional: filter one workflow with `python tools/ci/extract_failing_jobs.py <workflow-name>`
   - Confirm candidates exist in `build_ci/ci_ticketing/create_tickets/failing_jobs.json`

2. **Validate each candidate before creating any issue**
   - Follow `.cursor/rules/ci-create-tickets.mdc` in full.
   - For **each** candidate you might create an issue for:
     - Download the job logs for **all three** failing runs (e.g. via `gh run view <run_id> --job <job_id> --log-failed` into `build_ci/ci_ticketing/create_tickets/downloaded_logs`).
     - Read the downloaded logs and confirm the **same** error (or same failure signature) appears in all three runs. If it does not, do not create an issue for that candidate.
     - Extract the **actual** error excerpt from the logs (e.g. `##[error]` lines, `[  FAILED  ]` lines, or the relevant exception/failure message). Never use a generic placeholder like "Job failed in the last 3 runs" in the issue body.
     - Delete each downloaded log file after you finish using it.
   - Only create as many issues as you have capacity to validate in this way. Do not create issues for candidates you have not validated with downloaded logs.

3. **Create issues explicitly**
   - Create each issue with `gh issue create` (no bulk automation).
   - Add label `glean CI maintenance`.
   - Never assign issues directly to developers.
   - Respect user cap (e.g. "create up to N issues").
   - Every issue body must include the **real** error excerpt from the logs, not a generic statement.
   - **Reproduction steps** must describe how to reproduce the failure **locally** on a machine with the necessary hardware (e.g. which command or test to run, from repo root). Do not write "re-run on main" or "re-run the workflow"—that is redundant; the issue already implies it fails on main.

4. **Write outputs**
   - Append created items to `build_ci/ci_ticketing/create_tickets/created_issues.jsonl`
   - Write summary to `build_ci/ci_ticketing/create_tickets/created_tickets_report.json`
