# Draft Slack message for CI issue

## Overview
Produce a copy-paste-ready Slack message for a pipelines channel announcing a filed CI issue and (optionally) a PR. User provides the issue; user may provide the PR number or you infer it from the issue.

## Input
- **Required:** One GitHub issue in `tenstorrent/tt-metal` — provide as issue number (e.g. `39837`) or full URL.
- **Optional:** PR number if there is an auto-generated PR (e.g. `39980`). If not provided, check issue body/linked PRs and omit PR line if none.

## Steps
1. **Resolve the issue**
   - If given a number, use `gh issue view <number> --repo tenstorrent/tt-metal --json title,body,labels,url`.
   - If given a URL, parse issue number and run the same.

2. **Extract from the issue**
   - Workflow name and job name (from title or body).
   - Short descriptor for the heads-up line (e.g. "llama3.3-70b test", "Blackhole demo perf").
   - One concise root-cause or area sentence from the failure/body.
   - The main failure line or a short error excerpt.

3. **Suggest mentions**
   - Use `.github/CODEOWNERS` and the issue (workflow, paths in body, labels) to suggest 2–4 relevant people. Prefer individuals over team handles. Output as names or usernames; user will add Slack @mentions.

4. **Draft the message**
   - Follow `.cursor/rules/slack-ci-issue-draft.mdc`: heads-up line, context, failure excerpt, optional PR + "Appreciate an extra set of eyes", then suggested mentions.

5. **Output for copy-paste**
   - Emit the full message in a single fenced block (e.g. markdown code block) or under a clear "Copy below" section so the user can paste directly into Slack and tweak as needed.
