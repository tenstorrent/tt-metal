# Contributing to tt-agent

## Adding a New Skill

1. **Decide the layer.** Set `metadata.layer` in YAML frontmatter:
   - `orchestration` — routes and dispatches requests
   - `workflow` — autonomous loops with convergence criteria
   - `tool` — single-purpose capabilities used during execution
   - `meta` — system-level utilities (extending or learning from the system)

2. **Use `/tt:skill-creator`.** It wraps `/skill-creator` and applies tt-agent
   conventions automatically.

3. **Follow the golden rules:**
   - Skills describe *how to do something*, not *what the API is*
   - Point to code locations, never inline API signatures: `see tt_metal/hw/inc/api/dataflow/`
   - Keep SKILL.md ≤ 150 lines; move domain content to sub-files
   - Every SKILL.md must have valid YAML frontmatter with `name` and `description`
   - `name` must match the directory name exactly

4. **Workflow skills must declare phases:**
   - Each phase specifies Loads (knowledge/skill files) and Produces (output)
   - Include a phase table in SKILL.md
   - Every file referenced in Loads must exist on disk (enforced by tests)

5. **Run the validation test:**
   ```bash
   pytest tt-agent/tests/test_skill_frontmatter.py -v
   ```

## Adding Recipes (Repo Engineers)

Recipes describe how to build, test, and run in a specific repo. Any repo engineer
can contribute without understanding skills.

1. Create `knowledge/recipes/<repo>/` with:
   - `index.md` — 5-line TOC pointing to sibling files
   - `build.md` — build commands, submodules, clean build
   - `test.md` — test invocation, common fixtures
   - `env.md` — required environment variables
   - Additional files as needed (e.g., `server.md` for vLLM)

2. **Format rules:**
   - Plain markdown, no frontmatter
   - Self-contained per file (each file readable standalone)
   - Under 60 lines per file
   - Focus on *what commands to run*, not *why they work*

3. **Adding a new repo:** Create the directory, add `index.md` + relevant files.
   No skill changes required — tt-run discovers recipes by matching the detected
   repo name to a recipe directory.

## Adding Domain Knowledge (Domain Experts)

Domain experts contribute stable knowledge about disciplines like profiling or
debugging. This knowledge is loaded by workflow skills during relevant phases.

1. Add files to `knowledge/<domain>/` (e.g., `knowledge/profiling/`,
   `knowledge/debugging/`).

2. **Format rules:**
   - Plain markdown, under 80 lines per file
   - Describe patterns, methodologies, and interpretation guides
   - Not step-by-step procedures (those are skills)
   - Stable enough to maintain as static files (unlike APIs)

3. **No wiring required.** New knowledge files don't need to be referenced by a
   skill immediately. They can exist unwired — the agent team wires them into
   skill phase tables when building or updating workflow skills.

## Adding to knowledge/hardware/ and knowledge/references/

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
