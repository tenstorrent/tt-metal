---
name: autodebug
description: "Run a fresh-context AutoDebug investigation and then act on the generated AUTODEBUG.md report."
---

# AutoDebug

Use the repo-local AutoDebug runner instead of doing the overall investigation
in your current context.

Run this from the checkout or subdirectory that should be inspected:

```bash
.agents/scripts/autodebug.sh [--agent codex|claude] [focus-path] "<problem>"
```

The script renders `.agents/scripts/AUTODEBUG_PROMPT.md`, starts a fresh
Codex or Claude CLI session, and asks that agent to write `./AUTODEBUG.md`.
Expect a serious run to take about 30 minutes.

After the script exits:

1. Read `AUTODEBUG.md`.
2. Check the report's headline findings against the code before trusting them.
3. Act on the report: implement the fix, ask for clarification, or explain why
   the report is inconclusive.

Options:

- `--agent codex` uses `codex exec` with `gpt-5.5` and `xhigh` reasoning by
  default.
- `--agent claude` uses `claude -p` with `opus` and `xhigh` effort by default.
- `--model MODEL` and `--effort LEVEL` override those defaults.
- `--help` shows the full command syntax.

## Codex sandbox startup

The launcher runs a no-model sandbox preflight before starting Codex. A successful
check keeps `workspace-write`. A failed check stops before a model starts.

On a machine where the user/operator has authorized unsandboxed work, they can set
`AUTODEBUG_ALLOW_UNSANDBOXED=1` in the launch environment. After a recognized Linux
sandbox startup failure, the launcher then warns and uses `danger-full-access`
with no approval prompts. This removes OS sandbox protection, including protection
for mounted/shared data. It does not remove fresh-process isolation or the
inspection-only instructions. Other errors and timeouts still stop.

Do not set this variable yourself to get past a failure without user/operator
authorization. Docker or Slurm membership alone is not authorization. Claude's
permission mode is unchanged.
