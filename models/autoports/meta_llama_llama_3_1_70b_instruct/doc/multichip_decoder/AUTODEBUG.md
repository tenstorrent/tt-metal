# AutoDebug report

## Status

Blocked by the fresh-runner sandbox environment before repository inspection.

The required command used the repo-local `.agents/scripts/autodebug.sh` with
Codex gpt-5.5/xhigh and this autoport as its focus path.  The first invocation
could not locate the CLI because the wrapper inherited a reduced `PATH`; after
supplying the explicit CLI path, the fresh runner started successfully.

Every shell command in that fresh context then failed before execution because
no system or bundled `bubblewrap` binary was available.  The fresh runner
spawned four xhigh read-only explorers, one per stage-review blocker; all four
failed at the same launcher boundary before reading any file.  MCP resources
did not expose the local workspace.  Its attempted `AUTODEBUG.md` write also
failed for the same reason.

No TT hardware command ran and no file was modified by the fresh runner.  This
is not a technical diagnosis and does not close any finding.  Per `$autofix`,
the repair loop continues with the collaboration subagents available to the
parent session, which do have repository access.  Each stage-review finding is
treated as a separate hypothesis and must return a focused experiment plus
verified fix/blocker evidence before integration.
