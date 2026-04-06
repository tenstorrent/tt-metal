# Workspace Detection

Figure out where you are and what's available before executing anything.
If something is missing, tell the developer — don't fix it silently.

## What to determine

1. **Workspace root** — the parent directory containing repo checkouts.
   Heuristic: walk up from cwd looking for a directory with `tt-metal/` as
   a child, or derive from `$TT_METAL_HOME` if set.

2. **Available repos** — which subdirectories of the workspace root are git
   repos, and which have matching recipes in `knowledge/recipes/`.

3. **Readiness** — is the venv active, is tt-metal built, is tt-device-mcp
   available? Report what's missing with pointers to the relevant recipe.

4. **Env values for job composition** — `$USER`, `$HF_HOME`, whether
   `$HF_TOKEN` is set. These feed into `inherited_env` when submitting
   device jobs (see `execution.md`).

## Output

Summarize findings and any issues. Stop and report if there are blockers.
