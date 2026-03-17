# Close CI Tickets

## Overview
Review open CI maintenance issues and close only those no longer relevant based on the latest completed run on `main`.

## Steps
0. **Start with a clean slate**
   - Delete any stale files in `build_ci/ci_ticketing/close_tickets` (e.g. old `closed_tickets.json` and contents of `glean_review_logs`) so you never rely on pre-existing outputs. All inputs for this run must be produced or re-fetched during this run.

1. **Load candidate issues**
   - Query open issues with label `glean CI maintenance`
   - Evaluate all candidates before applying any close limit

2. **Apply close-ticket rule**
   - Follow `.cursor/rules/ci-close-tickets.mdc`
   - (Stale logs under `build_ci/ci_ticketing/close_tickets/glean_review_logs` should already have been cleared in step 0.)
   - For each issue, compare ticket failure vs latest completed job run on `main`

3. **Decide closures**
   - Close if exact failing test now passes, or if root cause is clearly different
   - Do not close for cosmetic differences, infra-only noise, or same root cause
   - Apply user cap (for example, "close up to N issues") only at final close step
   - Always review the failure logs yourself. Never write a script to close tickets for you

4. **Write outputs**
   - Record closed issues in `build_ci/ci_ticketing/close_tickets/closed_tickets.json`
   - Include `issue_url`, `old_failure_message`, and `new_failure_message`
   - Write a comment to each issue you close stating why the issue was closed and linking the run of the specific job that made you decide. The format should be something like: https://github.com/tenstorrent/tt-metal/actions/runs/<run-id>>/job/<job-id>
