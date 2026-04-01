# Contributing to tt-agent

## Adding a New Skill

1. **Decide the layer.** Which directory does your skill belong in?
   - `skills/orchestration/` — routes and dispatches requests
   - `skills/workflows/` — autonomous loops with convergence criteria
   - `skills/tools/` — single-purpose capabilities used during execution
   - `skills/meta/` — system-level utilities (extending or learning from the system)

2. **Use `/tt:skill-creator`.** It wraps `/skill-creator` and applies tt-agent
   conventions automatically.

3. **Follow the golden rules:**
   - Skills describe *how to do something*, not *what the API is*
   - Point to code locations, never inline API signatures: `see tt_metal/hw/inc/api/dataflow/`
   - Keep SKILL.md ≤ 150 lines; move domain content to sub-files
   - Every SKILL.md must have valid YAML frontmatter with `name` and `description`
   - `name` must match the directory name exactly

4. **Run the validation test:**
   ```bash
   pytest tt-agent/tests/test_skill_frontmatter.py -v
   ```

## Adding to knowledge/

`knowledge/hardware/` — only for silicon-stable facts (Tensix architecture, NOC topology,
tile granularity). If it could change in a software release, it does not go here.

`knowledge/references/` — curated pointers to canonical examples, one file per topic.
Format: path + one-line description. No content inlined. Update paths when examples move.

**Do not add:** API signatures, function names, implementation patterns, op lists.
These belong to `tt-learn` (fetched fresh from source), not to static files.

## Adding a Platform Adapter

Create `adapters/<platform-name>/` and add the platform's entrypoint file
(`CLAUDE.md` for Claude Code, `AGENTS.md` for Codex, etc.). Reference the same
`skills/` and `knowledge/` directories — content does not change per platform.

## PR Conventions

- One skill per PR when possible
- Include a brief note on which layer the skill belongs to and why
- Run `pytest tt-agent/tests/` before submitting
- Update `knowledge/references/` if your skill introduces canonical examples worth pointing to
