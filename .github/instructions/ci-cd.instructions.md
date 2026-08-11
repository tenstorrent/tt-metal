---
description: 'PR review for GitHub Actions workflows and CI infrastructure — security, runner hygiene, time budget, and pipeline structure'
applyTo: '.github/workflows/**,.github/actions/**'
excludeAgent: "cloud-agent"
---

# CI/CD Workflow Review

## 🔴 CRITICAL

- **Mutable action ref**: every `uses:` must be pinned to a full commit SHA, not a tag (`@v4`) or branch (`@main`). Tags are mutable and can be silently redirected to malicious commits. Add the version as a comment: `uses: actions/checkout@abc123 # v4.3.1`
- **Secret in env or run**: secrets must be referenced via `${{ secrets.NAME }}` only. Never echo, print, or interpolate a secret into a log line — even masked secrets can leak via timing.
- **`GITHUB_TOKEN` over-permission**: default `GITHUB_TOKEN` permissions are broad. Set `permissions:` at workflow level to `contents: read` and grant write only where explicitly needed.
- **Self-hosted runner label changes**: modifying `runs-on:` labels for Tenstorrent hardware runners can silently route jobs to wrong hardware. Flag any label change that doesn't match an existing runner pool.

## 🟡 IMPORTANT

- **Indirect runner reassignment**: a change can silently move a job to a different runner even when no `runs-on:` label is edited directly — for example, edits to the pipeline reorg YAML files (`tests/pipeline_reorg/`) or to `sku_config.yaml` that alter how a `sku` resolves to a runner pool. The reorg YAML tooling (@roseli-TT's work) generates the effective `runs-on:` assignment, so a small config change can re-route hardware. Flag any change that touches runner-routing config and verify the effective runner assignment via the pipeline reorg tooling before approving.
- **`time_budget.yaml` changes**: any edit to `.github/time_budget.yaml` must include a justification comment explaining the new budget and which jobs drove the change.
- **`fetch-depth`**: use `fetch-depth: 1` unless full history is explicitly needed (e.g., release tagging, `git describe`). Large repo + full history = slow CI.
- **Caching**: if a workflow installs Python or C++ dependencies, it should cache them with a key based on `hashFiles('**/requirements*.txt')` or `hashFiles('**/CMakeLists.txt')`.
- **Concurrency**: workflows that run on every push to `main` should set `concurrency:` to cancel in-progress runs when a new commit arrives.

## 🟢 SUGGESTION

- Reusable workflows (`workflow_call`) for patterns shared across 3+ workflow files — reduces duplication and makes updates atomic.
- Step names should be descriptive enough to identify failures in the GitHub UI without opening logs.
- `timeout-minutes` on any job that could hang (hardware-in-loop tests, long compile steps).

## Test Yaml Validation (`tests/pipeline_reorg/`)

Test yamls define what runs in CI. Every test entry must include all required keys:

| Key | Purpose |
|-----|---------|
| `name` | Display name in GitHub Actions UI |
| `cmd` | Exact command to run |
| `sku` | Machine type target — must match an entry in `.github/sku_config.yaml` |
| `owner_id` | Slack user ID for failure notification |
| `team` | Owning team (used for time budget allocation) |
| `timeout` | Maximum job runtime in minutes |

Flag any test entry missing a required key, or using a `sku` value not defined in `sku_config.yaml`.

**Not all pipelines have been through the pipeline reorg sanitization process.** Some pre-reorg pipelines still exist and may not conform to the rules above. When reviewing, flag pipelines that appear to be pre-reorg — e.g. missing required keys, using legacy/hard-coded runner labels instead of `sku`-based routing, or not following the pipeline level semantics below — so they can be migrated rather than silently accepted as compliant.

## Pipeline Level Semantics

Tests must be placed in the pipeline matching their intent and runtime:

| Level | Intent | Runtime expectation |
|-------|--------|-------------------|
| `smoke` | Basic functionality sanity on merge attempt | Seconds (< 1 min per test) |
| `sanity` | Happy-path coverage, post-commit | Short (< 5 min per test) |
| `unit` | Single component (one op, one module) | Short |
| `integration` | Multiple components interacting | Medium |
| `e2e` | End-to-end system | Long |
| `performance` | Perf measurement (performance mode enabled) | Variable |
| `stress` | Repeated execution, no perf measurement | Long, infrequent |
| `sweep` | Parameter sweep/schmoo | Very long, infrequent |

Flag a test that violates its pipeline's runtime contract (e.g., a 10-minute test in a `smoke` yaml).

## Review Checklist

- [ ] All `uses:` pinned to full SHA with version comment
- [ ] No secrets echoed or interpolated into run commands
- [ ] `permissions:` block present and minimal
- [ ] Runner-routing changes (direct `runs-on:` edits or indirect reorg/`sku_config.yaml` changes) verified against the pipeline reorg tooling
- [ ] `time_budget.yaml` changes justified
- [ ] `fetch-depth: 1` unless full history required
- [ ] Test yaml entries have all required keys (`name`, `cmd`, `sku`, `owner_id`, `team`, `timeout`)
- [ ] `sku` values match `.github/sku_config.yaml`
- [ ] Tests are in a pipeline level appropriate for their runtime
