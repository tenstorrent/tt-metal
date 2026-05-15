# Draft Slack message for CI issue(s)

## Overview
Produce a copy-paste-ready Slack message for pipelines triage, using one or many CI issues plus recent auto-triage evidence from exported Slack data when available.

## Input
- **Required:** One or many GitHub issues in `tenstorrent/tt-metal` — provide as issue number(s) (e.g. `39837`) and/or full URL(s).
- **Batch behavior:** If multiple issues are provided, produce **one unified Slack message**. Prefer one shared error summary ("error kind") plus one list of issue URLs; only split into multiple groups when failures are materially different.
- **Assumed pre-step:** User already ran `python tools/ci/export_slack_channel.py` (or equivalent) and produced `build_ci/raw_data/slack_C08SJ7MGESY_last_30_days.json`.

## Steps
1. **Resolve issue(s)**
   - If given a number, use `gh issue view <number> --repo tenstorrent/tt-metal --json title,body,labels,url`.
   - If given a URL, parse issue number and run the same.
   - If an issue is not about a CI pipeline failure, exclude it from grouping.
   - If none are CI pipeline failures, stop and say no Slack draft is needed.

2. **Extract baseline CI context from issue(s) + logs**
   - For each issue: workflow name, job name, short descriptor, root-cause area sentence, main failure line/error excerpt.
   - Identify the latest failing **job** run on `main` for the same job (link must be `.../actions/runs/<run_id>/job/<job_id>`, not workflow-only run link).
   - Read latest failing job logs to confirm failure details and use those details in judgment.
   - If multiple issues are provided, cluster them into similarity groups using:
     - Same workflow + same/similar job family, and
     - Same terminal failure signature/error pattern.
   - Default to one combined bucket when failures are the same kind; keep one representative error excerpt and 1-2 representative job links total.
   - Only create separate groups when failure signatures are materially different.

3. **Mine auto-triage evidence from exported Slack JSON**
   - Use only `build_ci/raw_data/slack_C08SJ7MGESY_last_30_days.json` (channel `C08SJ7MGESY` / `#metal-infra-pipeline-status`).
   - Never use any other Slack channel in this workflow.
   - Do **not** read the full JSON file directly. Query it with targeted filters only (`jq`/`rg`) and inspect only small matched slices.
   - Search for messages/replies relevant to each issue group (matching job/workflow/test path/run links/signature).
   - Prefer "FULL REPORT" style replies when present.
   - Treat auto-triage as reliable only when it explicitly includes `HIGH CONFIDENCE`.
   - If no relevant high-confidence triage is found in that month, **omit auto-triage entirely from the Slack draft**—do not tell the channel that nothing was found or that triage was searched.
   - If relevant triage is found, capture a Slack permalink to include in the draft.

4. **Suggest mentions**
   - Use grouped issue context, latest job logs, `.github/CODEOWNERS`, and recent commit history to suggest a compact owner set.
   - If one shared failure kind covers all issues, provide one mention list (typically 2-4 people total), not per-issue mentions.
   - If the failure is **infra-dominated** (read-only FS / shared mount / missing cache on RO volume / runner disk or image—see `.cursor/rules/ci-slack-issue-draft.mdc`), **do not** default to workflow or model CODEOWNERS; prefer infra/runner ownership (issue-named contacts, recent authors on relevant workflow or `infra/` changes). Add product/workflow owners only when a code or workflow/env change is clearly needed too.
   - If there are multiple groups, keep total mentions bounded (typically 3-6 unique people overall). Avoid repeating the same mention list.
   - Prefer specific individuals over broad team handles.
   - Resolve each selected GitHub login to a human name with `gh api users/<login>` and use that name in mentions.
   - Mentions must be `@<human name>` (for example, `@salar hosseini`), not GitHub handles like `@skhorasganiTT` and not Slack IDs.
   - Do not over-weight low-confidence auto-triage suggestions.

5. **Check for linked PR(s)**
   - If draft/open PRs exist that would close one or more grouped issues, mention them in the Slack draft.

6. **Draft the message**
   - Follow `.cursor/rules/ci-slack-issue-draft.mdc`.
   - In single-issue mode: include issue link, failure summary, latest failing job link, likely owners to ping, optional linked PR note, and the disable deadline line.
   - In multi-issue mode (default compact form): include
     - one concise heads-up summary,
     - one shared failure/error-kind line,
     - one combined list of issue links,
     - 1-2 representative latest failing job link(s),
     - one owner mention list.
   - Only use multiple grouped sections when there are genuinely different failure kinds.
   - Add auto-triage permalink line(s) **only** when step 3 found relevant `HIGH CONFIDENCE` material; otherwise say nothing about auto-triage in the message body.
   - Disable deadline line rule: **Mon–Thu** → "by the end of **tomorrow**"; **Fri–Sun** → "by the end of **Monday**" (never "next business day" / "end of business the next weekday").

7. **Output for copy-paste**
   - Emit the full message in one easy-to-copy block as **plain text** (not wrapped in markdown ` ``` ` code fences—Slack does not hyperlink inside code blocks).
   - Use **raw https URLs only** for the GitHub issue, Actions job link, any PR, and any auto-triage permalink. **Do not** use Slack `<url|label>` mrkdwn—it often renders as literal angle brackets.
   - Format with an extra blank line between logical lines/sections (double-newline style for Slack readability).
   - Even for many issues, produce **one** unified Slack draft (not one draft per issue).

## Example multi-issue usage

- **Input example:** `40742 40743 https://github.com/tenstorrent/tt-metal/issues/40739`
- **Expected behavior:** one Slack draft with grouped sections, not three separate drafts.
- **Expected behavior:** one Slack draft with a shared error summary and one issue list (group sections only if failures differ).

Expected output shape (plain text, illustrative):

Heads up on a batch of similar deterministic CI failures across Blackhole post-commit jobs.

Issues: https://github.com/tenstorrent/tt-metal/issues/40742, https://github.com/tenstorrent/tt-metal/issues/40743, https://github.com/tenstorrent/tt-metal/issues/40739

failure kind: Ethernet/system-health gate fails before tests start (`tt-smi -r` + `test_system_health` loop).

failure: Health checks failed after 10 attempts; ##[error]Process completed with exit code 1.

Representative latest failing jobs: https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>/job/<job_id>, https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>/job/<job_id>

Likely owners: @first last, @first last, @first last

This auto-triage report may be of relevance: https://tenstorrent.slack.com/archives/C08SJ7MGESY/p<timestamp>

If this isn't fixed by the end of tomorrow, we'll plan to disable the affected test/job scope(s) to keep the pipeline green.
