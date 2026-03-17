# Solve CI Ticket

## Overview
Take one CI issue ticket in `tenstorrent/tt-metal`, attempt a fix on a new branch from `main`, push a commit, and open a **draft PR** linked to the ticket. Stop for explicit user approval before running any workflows.

## Input
- **Required:** one issue URL or issue number (example: `https://github.com/tenstorrent/tt-metal/issues/40111` or `40111`).
- **Optional:** one auto-triage run URL (example: `https://github.com/tenstorrent/tt-metal/actions/runs/23179375291`).

If no input is provided, stop and ask for the ticket.

## Steps
1. **Load ticket context**
   - Resolve and inspect issue with `gh issue view`.
   - Capture workflow name, job name, failing tests, and error excerpt.

2. **Optionally load auto-triage context**
   - If an auto-triage run URL is provided, review its summary report.
   - Extract useful hints (suspected culprit commit, impacted area/files, possible owners).
   - Treat triage output as hints only; validate against job logs and code before acting.

3. **Create branch from main**
   - `git checkout main`
   - `git pull --ff-only origin main`
   - `git checkout -b fix/ci-<issue-number>-<short-slug>`

4. **Investigate and propose fix**
   - Follow `.cursor/rules/ci-solve-ticket.mdc`.
   - Reproduce locally when possible.
   - If hardware is unavailable, do CI/log-driven diagnosis and document constraints.
   - Implement the smallest credible fix with targeted validation.

5. **Validate, commit, and push**
   - Run local checks that are available and relevant.
   - Commit focused changes.
   - Push branch to origin.

6. **Create draft PR linked to issue**
   - Use `gh pr create --draft`.
   - Link ticket in title/body (`Refs #<issue>` or `Closes #<issue>`).
   - Include: root-cause hypothesis, change summary, validation done, limitations, CI plan, and (if applicable) whether auto-triage hints were confirmed.

7. **Mandatory safety gate before workflow runs**
   - Stop execution and ask the user exactly:
     - `do you want me to kick off the workflow runs to test my changes now?`
   - Do not run any workflow unless user explicitly approves.

8. **If user approves workflow runs**
   - Trigger selected workflows with `gh workflow run ... --ref <branch>`.
   - Gather run URLs.
   - Update PR description to include links to currently executing runs.

9. **If fix is not credible solo, hand off**
   - Explicitly explain why the agent is giving up.
   - Recommend owners using `.github/CODEOWNERS`, file history, and workflow ownership context.
   - Provide an actionable handoff summary.

## Output
- Branch name and latest commit hash.
- Draft PR URL.
- Whether workflows were triggered (must be user-approved first).
- If handed off: recommended owner(s) with rationale.
