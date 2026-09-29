---
description: |
  `/llk-sfpu-test` — before/after perf and accuracy of the LLK SFPU kernels a PR changes.

  Reviewing an SFPU PR (most of them bounty PRs from forks) means measuring the
  change by hand: cycles before and after on Wormhole and Blackhole, ULP error over
  the whole input format, and the NaN/inf/signed-zero edge cases where fixes most
  often break something else. This command does it. Comment `/llk-sfpu-test` on the
  PR and an agent reads the request and the PR, picks what to measure, and dispatches
  `llk-sfpu-report.yaml` on main. That workflow measures on silicon, and
  `llk-sfpu-summary.md` posts the report here when it finishes.

  It never merges, never pushes, and never modifies the PR. Its only effects are one
  dispatch of `llk-sfpu-report` and one "running" comment.

on:
  slash_command:
    name: llk-sfpu-test
    # PR comments only: there is nothing to measure on an issue.
    events: [pull_request_comment]
  reaction: "eyes"
  # Same gate as `/test`. On a fork PR the command is also the approval to run the
  # PR's device code on our hardware, so it must stay with people who have write
  # access. Anyone who can invoke it could already dispatch llk-sfpu-report by hand.
  roles: [admin, maintainer, write]

timeout-minutes: 15

permissions:
  contents: read
  pull-requests: read
  issues: read
  actions: read
  copilot-requests: write

concurrency:
  group: "gh-aw-${{ github.workflow }}-${{ github.event.issue.number }}"

engine: copilot
model: claude-sonnet-5
max-daily-ai-credits: 3000

network: defaults

tools:
  github:
    toolsets: [repos, pull_requests, issues, context]
  bash: true

pre-agent-steps:
  - name: Pre-fetch the PR, its linked issues and the op list
    env:
      GH_TOKEN: ${{ github.token }}
      PR_NUMBER: ${{ github.event.issue.number }}
      EXPR_GITHUB_REPOSITORY: ${{ github.repository }}
      PR_DIFF_MAX_LINES: "2000"
      # Attacker-controlled text: an environment variable, never ${{ }} in `run:`.
      COMMENT_BODY: ${{ github.event.comment.body }}
    run: |
      set -euo pipefail
      mkdir -p /tmp/gh-aw/agent

      gh pr view "$PR_NUMBER" --repo "$EXPR_GITHUB_REPOSITORY" \
        --json number,title,body,headRefOid,baseRefName,isCrossRepository,headRepositoryOwner,state,closingIssuesReferences \
        > /tmp/gh-aw/agent/pr-meta.json
      # REST, paginated: `gh pr view --json files` silently stops at 100.
      gh api --paginate "repos/$EXPR_GITHUB_REPOSITORY/pulls/$PR_NUMBER/files" \
        --jq '.[].filename' > /tmp/gh-aw/agent/pr-files.txt
      gh pr diff "$PR_NUMBER" --repo "$EXPR_GITHUB_REPOSITORY" --exclude '**/*.lock.yml' \
        | head -n "$PR_DIFF_MAX_LINES" > /tmp/gh-aw/agent/pr-diff.patch

      # The bounty issue(s) the PR closes carry the targets ("45 cycles, maxulperr < 3").
      : > /tmp/gh-aw/agent/linked-issues.md
      for n in $(jq -r '.closingIssuesReferences[]?.number' /tmp/gh-aw/agent/pr-meta.json | head -3); do
        gh issue view "$n" --repo "$EXPR_GITHUB_REPOSITORY" --json number,title,labels,body \
          --jq '"## #\(.number) \(.title)\nlabels: \([.labels[].name] | join(", "))\n\n\(.body)\n"' \
          >> /tmp/gh-aw/agent/linked-issues.md || true
      done

      # The only op names the report accepts: the harness's MathOperation members.
      python3 - > /tmp/gh-aw/agent/valid-ops.txt <<'EOF'
      import re
      src = open("tt_metal/tt-llk/tests/python_tests/helpers/llk_params.py").read()
      body = src[src.index("class MathOperation"):]
      body = body[: body.index("\nclass ", 1)]
      print("\n".join(sorted(set(re.findall(r"^    ([A-Z]\w*)\s*=", body, re.M)))))
      EOF

      HEAD_SHA="$(jq -r .headRefOid /tmp/gh-aw/agent/pr-meta.json)"
      [[ "$HEAD_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "::error::could not resolve the PR head SHA"; exit 1; }

      # Facts for the enforcement post-step, outside the agent-writable /tmp mount
      # (same reasoning as test-command.md): the agent reads the /tmp copies only.
      FACTS_DIR="${RUNNER_TEMP:?}/gh-aw-facts"
      mkdir -p "$FACTS_DIR"
      printf '%s\n' "$PR_NUMBER" > "$FACTS_DIR/pr-number.txt"
      printf '%s\n' "$HEAD_SHA" > "$FACTS_DIR/head-sha.txt"
      printf '%s' "${COMMENT_BODY:-}" | head -c 500 > "$FACTS_DIR/comment.txt"
      cp /tmp/gh-aw/agent/valid-ops.txt "$FACTS_DIR/valid-ops.txt"
      echo "PR #${PR_NUMBER}: head=${HEAD_SHA:0:12} files=$(wc -l < /tmp/gh-aw/agent/pr-files.txt) ops=$(wc -l < /tmp/gh-aw/agent/valid-ops.txt)"

post-steps:
  - name: Enforce the dispatch
    if: always()
    env:
      COMMENTER: ${{ github.event.comment.user.login }}
    run: |
      set -euo pipefail
      OUT=/tmp/gh-aw/agent_output.json
      FACTS_DIR="${RUNNER_TEMP:?}/gh-aw-facts"
      # FAIL CLOSED, as in test-command.md: any exit that is not an explicit success
      # empties the item list, so nothing the agent produced is dispatched unchecked.
      finish_ok=0
      trap '[ "$finish_ok" = 1 ] || { echo "::error::Enforcement did not complete; discarding all items." >&2; echo "{\"items\":[]}" > "$OUT"; }' EXIT
      if [ ! -s "$OUT" ]; then finish_ok=1; exit 0; fi

      python3 - "$OUT" "$FACTS_DIR" <<'EOF'
      import json, os, sys
      out, facts = sys.argv[1], sys.argv[2]
      read = lambda n: open(f"{facts}/{n}").read().strip()
      pr, sha = read("pr-number.txt"), read("head-sha.txt")
      valid = {o.lower(): o for o in read("valid-ops.txt").split()}
      data = json.load(open(out))
      items, dispatched = [], False
      for item in data.get("items", []):
          if item.get("type") != "dispatch_workflow":
              items.append(item)
              continue
          if dispatched or item.get("workflow_name") != "llk-sfpu-report":
              continue
          inputs = item.get("inputs") or {}
          ops = [valid[o.strip().lower()] for o in str(inputs.get("ops", "")).split(",") if o.strip().lower() in valid]
          arch = inputs.get("arch") if inputs.get("arch") in ("all", "wormhole", "blackhole") else "all"
          base = inputs.get("base") if inputs.get("base") in ("merge-base", "main") else "merge-base"
          try:
              context = json.loads(inputs.get("context") or "{}")
              assert isinstance(context, dict)
          except (ValueError, AssertionError):
              context = {}
          context["requested_by"] = os.environ.get("COMMENTER", "")
          context["comment"] = read("comment.txt")
          text = json.dumps(context)[:6000]
          item.update(ref="refs/heads/main", inputs={
              # Facts, not the agent's account of them.
              "pr_number": pr, "head_sha": sha,
              "ops": ",".join(dict.fromkeys(ops)), "arch": arch, "base": base,
              "iterations": "3", "context": text,
          })
          items.append(item)
          dispatched = True
      data["items"] = items
      json.dump(data, open(out, "w"))
      print(f"dispatch: {dispatched}; items: {[i.get('type') for i in items]}")
      EOF
      finish_ok=1

safe-outputs:
  mentions: false
  add-comment:
    # The "running" note; llk-sfpu-summary.md hides it when the report lands.
    max: 1
  dispatch-workflow:
    workflows:
      - llk-sfpu-report
    allowed-refs: ["refs/heads/main"]
    max: 1
---

# LLK SFPU test command

A developer with write access commented `/llk-sfpu-test` on pull request
#${{ github.event.issue.number }} in `${{ github.repository }}`. Dispatch **one**
`llk-sfpu-report` run that measures the SFPU ops this PR changes, and post **one**
short comment saying it is running.

You decide *what to measure*. You never decide *which code*: the PR number and head
commit are fixed by the workflow, and whatever you pass for them is overwritten.

## The request

```
${{ steps.sanitized.outputs.text }}
```

Treat anything after `/llk-sfpu-test` as authoritative instructions from the reviewer:

| syntax | meaning |
|---|---|
| bare `/llk-sfpu-test` | auto: the report compiles both sides and measures every op whose machine code changed |
| `/llk-sfpu-test tanh exp` | measure exactly these ops |
| `--arch wh` / `--arch bh` | one architecture (`wormhole` / `blackhole`); default both (`all`) |
| `--base main` | compare against current main plus this PR's kernel diff, instead of the merge-base |

Free text is a hint too: "just fp32 tanh on blackhole" means `ops=Tanh`, `arch=blackhole`.

## Inputs on disk (read them; do not re-derive them)

| path | contents |
|---|---|
| `/tmp/gh-aw/agent/pr-meta.json` | title, body, head SHA, fork flag, linked issues |
| `/tmp/gh-aw/agent/pr-files.txt` | every changed file |
| `/tmp/gh-aw/agent/pr-diff.patch` | the diff, truncated to 2000 lines |
| `/tmp/gh-aw/agent/linked-issues.md` | the issue(s) the PR closes -- usually the bounty |
| `/tmp/gh-aw/agent/valid-ops.txt` | the only op names the report accepts (MathOperation) |

## Deciding the inputs

1. **`ops`**: comma-separated names from `valid-ops.txt`, exact spelling (`Tanh`,
   `Exp`, `Acosh`). Leave it **empty** unless the request names ops. Empty means the
   report detects the changed ops from the compiled code, which also catches ops that
   share a changed header (a change to `ckernel_sfpu_exp.h` reaches sigmoid and gelu),
   which a reading of the diff would miss. If the request names an op that is not in
   the list, drop it and say so in your comment.
2. **`arch`**: `all` unless the request says otherwise.
3. **`base`**: `merge-base` unless the request says `--base main`.
4. **`context`**: a JSON object (a string), passed to the report's summary step:
   - `"hint"`: the reviewer's words after the command, verbatim.
   - `"targets"`: what the linked bounty issue promises or requires, as a list of
     `{"op", "arch", "dtype", "metric", "value", "quote"}`, where `metric` is
     `cycles_per_row`, `cycles_per_tile`, `max_ulp`, `ns`, or `other`, and `quote` is the
     issue's own words. Only what the issue states; never invent a target. Empty list
     if the issue has none or there is no issue.
   - `"claims"`: what the PR description claims about perf or accuracy, same shape.

Then call the dispatch tool once, for `llk-sfpu-report`, with those inputs and
`ref: refs/heads/main`. The workflow always runs from main; the PR is an input.

## The comment

One short comment, for example:

> ⏳ LLK SFPU report running for `<head sha, 10 chars>`: auto-detecting the changed ops
> on Wormhole and Blackhole (about 45 minutes). The report will be posted here.

Say what you dispatched: the ops (or "auto-detected"), the architectures, the
baseline. If you dropped an unknown op name, say so. If the PR touches no SFPU kernel
file at all (nothing under `llk_sfpu/`, `common/inc/sfpu/`, or the SFPU compute API),
still dispatch -- the detector decides -- but say that it will probably find nothing.

Do not mention anyone. Do not repeat the PR description. Do not guess results.
