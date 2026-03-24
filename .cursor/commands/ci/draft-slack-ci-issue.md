# Draft Slack message for CI issue

## Overview
Produce a copy-paste-ready Slack message for pipelines triage, using a CI issue plus recent auto-triage evidence from exported Slack data when available.

## Input
- **Required:** One GitHub issue in `tenstorrent/tt-metal` — provide as issue number (e.g. `39837`) or full URL.
- **Assumed pre-step:** User already ran `python tools/ci/export_slack_channel.py` (or equivalent) and produced `build_ci/raw_data/slack_C08SJ7MGESY_last_30_days.json`.

## Steps
1. **Resolve the issue**
   - If given a number, use `gh issue view <number> --repo tenstorrent/tt-metal --json title,body,labels,url`.
   - If given a URL, parse issue number and run the same.
   - If issue is not about a CI pipeline failure, stop and say no Slack draft is needed.

2. **Extract baseline CI context from issue + logs**
   - Workflow name and job name (from title or body).
   - Short descriptor for the heads-up line (e.g. "llama3.3-70b test", "Blackhole demo perf").
   - One concise root-cause or area sentence from the failure/body.
   - The main failure line or a short error excerpt.
   - Identify the latest failing **job** run on `main` for the same job (link must be `.../actions/runs/<run_id>/job/<job_id>`, not workflow-only run link).
   - Read latest failing job logs to confirm failure details and use those details in judgment.

3. **Mine auto-triage evidence from exported Slack JSON**
   - Use only `build_ci/raw_data/slack_C08SJ7MGESY_last_30_days.json` (channel `C08SJ7MGESY` / `#metal-infra-pipeline-status`).
   - Never use any other Slack channel in this workflow.
   - Do **not** read the full JSON file directly. Query it with targeted filters only (`jq`/`rg`) and inspect only small matched slices.
   - Search for messages/replies relevant to the same failure (matching job/workflow/test path/run links/signature).
   - Prefer "FULL REPORT" style replies when present.
   - Treat auto-triage as reliable only when it explicitly includes `HIGH CONFIDENCE`.
   - If no relevant high-confidence triage is found in that month, continue without triage attribution.
   - If relevant triage is found, capture a Slack permalink to include in the draft.

4. **Suggest mentions**
   - Use issue context, latest job logs, `.github/CODEOWNERS`, and recent commit history to suggest 2-4 people best suited to fix.
   - Prefer specific individuals over broad team handles.
   - Resolve each selected GitHub login to a human name with `gh api users/<login>` and use that name in mentions.
   - Mentions must be `@<human name>` (for example, `@salar hosseini`), not GitHub handles like `@skhorasganiTT` and not Slack IDs.
   - Do not over-weight low-confidence auto-triage suggestions.

5. **Check for linked PR**
   - If a draft/open PR exists that would close the issue, mention it in the Slack draft.

6. **Draft the message**
   - Follow `.cursor/rules/slack-ci-issue-draft.mdc`.
   - Include: issue link, failure summary, latest failing job link, likely owners to ping, optional relevant auto-triage Slack permalink, optional linked PR note, message that test will be disabled if not fixed by the end of the next business day (so tomorrow on monday through thursday and monday on friday).

7. **Output for copy-paste**
   - Emit the full message in one easy-to-copy block.
   - Format with an extra blank line between logical lines/sections (double-newline style for Slack readability).
