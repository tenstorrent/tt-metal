---
name: autodebug
description: Run a fresh-context Cursor Subagent investigation, verify its AUTODEBUG.md findings, and then act on proven root causes.
---

# AutoDebug for Cursor

Use Cursor's `Subagent` tool with `subagent_type="generalPurpose"` and
`readonly=true` to start a fresh investigation. Do not use Claude Code's
Task/Agent vocabulary.

1. Read `.cursor/skills/autodebug/AUTODEBUG_PROMPT.md`.
2. Append the concrete problem, failing command/logs, checkout root, and focus
   paths.
3. Ask the subagent to investigate without editing implementation files and to
   return a root-cause report with cited evidence. A report file
   `./AUTODEBUG.md` is optional; the subagent final response is sufficient.
4. Independently verify headline findings against source and runtime evidence.
5. Implement only verified fixes, or explain why the report is inconclusive.

If a subagent cannot be launched, perform the same read-only investigation
serially and mark it `serial-cursor`, rather than waiting for a nonexistent
tool.

For DiffusionGemma stage sequencing, read the applicable
`.cursor/commands/dg-NN-*.md` command and load `diffusion-gemma` first.
