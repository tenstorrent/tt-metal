# LLK CodeGen

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`). **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git.

---

## Request Routing

### 1. Generate Kernel (direct request)

When a user asks to **"generate {kernel} for {target_arch}"**, read and follow `codegen/agents/{target_arch}/orchestrator.md`.

If the target architecture is not specified, default to **quasar**.

### 2. Solve a GitHub Issue

When a user references a **GitHub issue** (e.g., "solve issue #123", "fix #456", "work on issue 789"), follow this routing process:

#### Step A: Fetch the Issue

Fetch **all** issue data — title, body, comments, and labels:

```bash
gh issue view {number} --json number,title,body,labels,comments
```

Extract and store verbatim:
- `ISSUE_NUMBER` — the issue number
- `ISSUE_TITLE` — the issue title, unmodified
- `ISSUE_BODY` — the full issue description, unmodified (includes error messages, reproduction steps, code snippets, etc.)
- `ISSUE_LABELS` — all labels as a list
- `ISSUE_COMMENTS` — all comments in full, unmodified (includes follow-up context, clarifications, stack traces, etc.)

**CRITICAL: Never alter, summarize, paraphrase, or truncate any issue content.** The raw title, body, and comments must be passed as-is to every subagent. Agents depend on exact error messages, code snippets, and reproduction steps from the issue to do their work correctly.

#### Step B: Determine Architecture

1. **Check labels** — look for `blackhole`, `quasar`, `wormhole` in the issue labels.
2. **Fallback: scan content** — if no architecture label found, scan the issue title and body for:
   - `blackhole`, `bh`, `tt_llk_blackhole` → **blackhole**
   - `quasar`, `qs`, `tt_llk_quasar`, `trinity` → **quasar**
3. **Default** — if still ambiguous, default to **blackhole**.

#### Step C: Determine Task Type

1. **Check labels** — look for:
   - Creation: `new-kernel`, `enhancement`, `feature`, `implement`, `port`
   - Issue fix: `bug`, `fix`, `defect`, `regression`, `compile-error`, `test-failure`
2. **Fallback: keyword heuristics** — if labels are inconclusive, scan title and body for:
   - **New kernel** signals: "implement", "add", "create", "port", "new kernel", "missing", "generate"
   - **Issue fix** signals: "fix", "broken", "error", "fail", "crash", "wrong", "incorrect", "regression", "compile"
3. **Default** — if still ambiguous, treat as **issue fix**.

#### Step D: Route to Orchestrator

| Architecture | Task Type | Orchestrator |
|-------------|-----------|-------------|
| **blackhole** | issue fix | `codegen/agents/blackhole/bh-orchestrator.md` |
| **quasar** | new kernel | `codegen/agents/quasar/orchestrator.md` |
| **quasar** | issue fix | Not yet supported — inform the user |
| **blackhole** | new kernel | Not yet supported — inform the user |

Pass **all** fetched issue context verbatim to the selected orchestrator: `ISSUE_NUMBER`, `ISSUE_TITLE`, `ISSUE_BODY`, `ISSUE_LABELS`, `ISSUE_COMMENTS`. Never summarize or alter any of these fields — agents need the raw content to parse error messages, stack traces, and reproduction steps.

---

## Architecture-Based Orchestrators

Each target architecture has its own orchestrator and agent playbooks under `codegen/agents/{arch}/`:

| Architecture | Orchestrator | Agents |
|--------------|-------------|--------|
| **quasar** | `codegen/agents/quasar/orchestrator.md` | `codegen/agents/quasar/llk-*.md` |
| **blackhole** | `codegen/agents/blackhole/bh-orchestrator.md` | `codegen/agents/blackhole/bh-*.md` |
