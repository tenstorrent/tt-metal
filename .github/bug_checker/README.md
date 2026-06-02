# PR Bug Checker

LLM-powered bug pattern detection for tt-metal PRs. Scans PR diffs against a library of known bug patterns using Claude, then reports findings as PR comments, SARIF output, and CLI output.

## How It Works

1. A PR is targeted (via `/bug-check run` comment or local CLI invocation).
2. The tool loads all rules from `.github/bug_checker/manifest.yaml`.
3. Rules are filtered to only those matching the PR's changed files (path globs) or labels.
4. Each matching rule is sent to Claude along with the PR diff. Claude analyzes the diff against the bug pattern described in the rule's markdown file.
5. Findings are reported in three formats: CLI stdout, inline PR comments, and SARIF for GitHub Code Scanning.

## Usage

### GitHub Actions (primary)

Comment `/bug-check run` on any PR. The workflow at `.github/workflows/bug-check.yaml` runs the checker and posts findings as inline comments. Comment `/bug-check` (bare) to see all available subcommands.

## Subcommands

| Command | Description |
|---------|-------------|
| `/bug-check` | Show this help message |
| `/bug-check run` | Run all matching rules against the PR |
| `/bug-check list-rules` | List all rules with their severity, paths, and labels |
| `/bug-check check-rule <id>` | Run only the named rule against the PR |
| `/bug-check dry-run` | Show which rules match and what diff sections each would analyze (no LLM calls) |

## Creating a New Rule

1. **Write the rule markdown** in `.github/bug_checker/rules/your-rule-name.md`. Copy `rules/template-rule.md` and fill in each section:

````markdown
# Rule Title

## Description
What the bug is, why it happens, what to look for.

## What to Look For
1. **Pattern 1**: Specific code pattern that indicates this bug.
2. **Pattern 2**: Another variant.

## Bad Code Examples
```cpp
// BUG: explain why
bad_code();
```

## Good Code Examples
```cpp
// GOOD: explain why
good_code();
```
````

2. **Add an entry to `.github/bug_checker/manifest.yaml`**:

   ```yaml
   rules:
     your-rule-name:
       file: your-rule-name.md
       severity: warning       # "blocking" fails the check, "warning" is informational
       suggest_fix: false      # true = LLM includes suggested code fixes
       model: null             # null = use default, or e.g. "claude-sonnet-4-6"
       paths:                  # rule runs if any changed file matches these globs
         - "path/to/relevant/code/**"
       labels:                 # rule runs if any PR label matches
         - "area:your-area"
   ```

3. **Test locally** against a PR that should trigger the rule:

   ```bash
   python .github/bug_checker/run_bug_checker.py --pr <number> --verbose
   ```

## Manifest Options Reference

| Field | Type | Description |
|-------|------|-------------|
| `file` | string | Markdown filename in `.github/bug_checker/rules/` |
| `severity` | `"blocking"` \| `"warning"` | Blocking findings fail the check and are marked prominently |
| `suggest_fix` | bool | Whether the LLM should include suggested code fixes |
| `model` | string \| null | Claude model override; null uses the global default |
| `paths` | list of globs | Rule runs if any changed file matches |
| `labels` | list of strings | Rule runs if any PR label matches |

A rule is selected if **either** a path or a label matches.

### Local CLI

Install the local dependencies and provide a Claude API key:

```bash
pip install -r .github/bug_checker/requirements-bug-checker.txt
export BUG_CHECKER_API_KEY=<your-api-key>  # or ANTHROPIC_API_KEY
```

`BUG_CHECKER_MODEL` can be set to override the default model (`claude-sonnet-4-6`).

Run the checker against a branch diff or PR, optionally selecting a subcommand:

```bash
# Analyze the current branch diff against main
python .github/bug_checker/run_bug_checker.py --branch --verbose

# Analyze against a different base branch
python .github/bug_checker/run_bug_checker.py --branch origin/release-1.0

# Analyze a PR by number (requires gh CLI auth)
python .github/bug_checker/run_bug_checker.py --pr 39432 --verbose

# List rules without analyzing a target
python .github/bug_checker/run_bug_checker.py --subcommand list-rules

# Run one rule against a PR
python .github/bug_checker/run_bug_checker.py --pr 12345 --subcommand check-rule --rule-id ccl-ring-buffer-mismatch

# Preview matching rules and diff sections without LLM calls
python .github/bug_checker/run_bug_checker.py --branch --subcommand dry-run
```

## Component Reference

### CI Path — `github_client.py` : `fetch_pr_info()`

- Triggered by `--pr`
- Calls `gh pr view` and `gh pr diff` via the GitHub API
- Gets the real PR title, labels, SHAs, diff, and changed file list

### Local Path — `github_client.py` : `fetch_branch_diff()`

- Triggered by `--branch`
- Runs `git merge-base` + `git diff` locally
- PR number is set to `0`, labels default to `[]` unless `--labels` is passed

### PRInfo dataclass — `github_client.py`

- The shared data structure both input paths produce
- Fields: `number`, `title`, `base_sha`, `head_sha`, `diff`, `changed_files`, `labels`
- Everything downstream operates on this single type regardless of input source
- Diffs exceeding 8000 lines (`MAX_DIFF_LINES`) are truncated with a warning

### Orchestrator — `orchestrator.py` : `run_bug_check()`

- Central coordinator that sequences the entire pipeline:
  load rules → select matching → per-rule LLM analysis → collect findings → dispatch outputs
- Fails closed — if LLM setup or rule analysis fails, the PR gets a failure summary comment and the command exits non-zero

### Rule Engine — `rules.py` + `manifest.yaml` + `rules/*.md`

- `load_rules()` reads `manifest.yaml` and the markdown content of each rule
- `select_rules()` filters by file-glob and label match against `PRInfo`
- Current rules:
  - `ccl-ring-buffer-mismatch` — blocking, paths: `tt_metal/impl/ccl/**`, `ttnn/cpp/ttnn/operations/ccl/**`, labels: `area:ccl`
  - `reshape-dim-check` — warning, paths: `ttnn/cpp/ttnn/operations/data_movement/**`, labels: `area:ops`

### LLM Analysis — `llm.py` : `LLMSession.analyze_rule()`

- Creates an Anthropic client (`claude-sonnet-4-6`, `max_tokens: 4096`, `temperature: 0`)
- Per matched rule:
  1. `_filter_diff_for_rule()` narrows the diff to files matching the rule's path globs; for label-only matches (no path globs matched), the full diff is passed through
  2. Builds a system prompt requiring structured `` ```finding `` blocks
  3. Builds a user message with the rule's markdown + filtered diff
  4. Calls `client.messages.create()`
  5. Parses the response into `Finding` objects

### Output Dispatch — `output.py` + `github_client.py`

- **CLI stdout** — `print_findings()` — always runs, colorized with loguru
- **SARIF file** — `write_sarif()` — if `--sarif`, for GitHub Code Scanning
- **PR comments** — `post_pr_comment()` — if `--post-comments`, posts inline review comments + a summary comment

### Exit Code — `__main__.py`

- `exit 1` if any finding has `severity == "blocking"` or LLM analysis fails, `exit 0` otherwise

Each rule runs in its own isolated LLM session — no state is shared between rules.

## Data Handling

This tool sends PR diffs to the Anthropic Claude API for analysis. Do not run it on PRs containing secrets or sensitive data without appropriate review.

## Running Tests

```bash
python -m pytest .github/bug_checker/tests/ -v
```
