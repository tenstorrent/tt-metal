# PR Bug Checker

LLM-powered bug pattern detection for tt-metal PRs. Scans PR diffs against a library of known bug patterns using Claude, then reports findings as PR comments, SARIF output, and CLI output.

## How It Works

1. A PR is targeted (via `/bug-check` comment or local CLI invocation).
2. The tool loads all rules from `.github/bug_checker/manifest.yaml`.
3. Rules are filtered to only those matching the PR's changed files (path globs) or labels.
4. Each matching rule is sent to Claude along with the PR diff. Claude analyzes the diff against the bug pattern described in the rule's markdown file.
5. Findings are reported in three formats: CLI stdout, inline PR comments, and SARIF for GitHub Code Scanning.

Rules run in isolation by default (separate LLM session each). Rules with the same `group` in the manifest share a conversation, so later rules see earlier analysis.

The tool **fails open** — if an LLM call fails, the rule is skipped and the overall check still passes. PRs are never blocked by infrastructure failures.

## Usage

### GitHub Actions (primary)

Comment `/bug-check` on any PR. The workflow at `.github/workflows/bug-check.yaml` runs the checker and posts findings as inline comments.

### Local CLI

```bash
# Analyze current branch diff against main
python .github/bug_checker/run_bug_checker.py --branch --verbose

# Analyze against a different base branch
python .github/bug_checker/run_bug_checker.py --branch origin/release-1.0

# Analyze a PR by number (requires gh CLI auth)
python .github/bug_checker/run_bug_checker.py --pr 39432 --verbose
```

**Requirements**: `pip install anthropic pyyaml loguru`

**Environment variables**:
- `BUG_CHECKER_API_KEY` or `ANTHROPIC_API_KEY` — Claude API key
- `BUG_CHECKER_MODEL` — Override the default model (default: `claude-sonnet-4-20250514`)

## Adding a New Rule

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
       model: null             # null = use default, or e.g. "claude-sonnet-4-20250514"
       group: null             # null = isolated, or a group name to share LLM context
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
| `group` | string \| null | Group name for shared LLM context; null runs in isolation |
| `paths` | list of globs | Rule runs if any changed file matches |
| `labels` | list of strings | Rule runs if any PR label matches |

A rule is selected if **either** a path or a label matches.

## Data Handling

This tool sends PR diffs to the Anthropic Claude API for analysis. Do not run it on PRs containing secrets or sensitive data without appropriate review.

## Running Tests

```bash
python -m pytest .github/bug_checker/tests/ -v
```
