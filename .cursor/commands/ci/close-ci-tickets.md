# Close CI Tickets

## Overview
Review open CI maintenance issues and close only those no longer relevant based on the latest completed run on `main`.

## Steps
1. **Load candidate issues**
   - Query open issues with label `glean CI maintenance`
   - Evaluate all candidates before applying any close limit

2. **Apply close-ticket rule**
   - Follow `.cursor/rules/ci-close-tickets.mdc`
   - Clear stale logs under `.auto_triage/output/ci_ticketing/close_tickets/glean_review_logs`
   - For each issue, compare ticket failure vs latest completed job run on `main`

3. **Decide closures**
   - Close if exact failing test now passes, or if root cause is clearly different
   - Do not close for cosmetic differences, infra-only noise, or same root cause
   - Apply user cap (for example, "close up to N issues") only at final close step

4. **Write outputs**
   - Record closed issues in `.auto_triage/output/ci_ticketing/close_tickets/closed_tickets.json`
   - Include `issue_url`, `old_failure_message`, and `new_failure_message`
