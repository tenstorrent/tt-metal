# Create CI Tickets

## Overview
Create deterministic CI maintenance tickets for repeated failures on `main` in `tenstorrent/tt-metal`.

## Steps
1. **Prepare candidate failures**
   - Enter the virtual environment defined locally in tt-metal (or tell the user to create a virtual environment if you can't find one locally in tt-metal)
   - Run `python tools/ci/extract_failing_jobs.py`
   - Optional: filter one workflow with `python tools/ci/extract_failing_jobs.py <workflow-name>`
   - Confirm candidates exist in `build_ci/ci_ticketing/create_tickets/failing_jobs.json`
   - If the user specified that they already ran extract_failing_jobs.py, just read the output without running it again.

2. **Apply create-ticket rule**
   - Follow `.cursor/rules/ci-create-tickets.mdc`
   - Build candidates from `failing_jobs.json`
   - Deduplicate against open issues with label `glean CI maintenance`
   - Validate failures from logs using `gh api`

3. **Create issues explicitly**
   - Create each issue with `gh issue create` (no bulk automation)
   - Add label `glean CI maintenance`
   - Never assign issues directly to developers
   - Respect user cap (for example, "create up to N issues")

4. **Write outputs**
   - Append created items to `build_ci/ci_ticketing/create_tickets/created_issues.jsonl`
   - Write summary to `build_ci/ci_ticketing/create_tickets/created_tickets_report.json`
