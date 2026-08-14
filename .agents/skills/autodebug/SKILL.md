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
