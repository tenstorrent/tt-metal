# Bug Checker: Known Complexity Issues

A reference for tackling inherent complexity in the LLM-powered PR bug checker, one issue at a time.

---

## ~~1. LLM Output Parsing is Fragile~~ ✓ RESOLVED

**Resolution:** Replaced regex text parsing with Claude tool use (`tool_choice: {type: tool, name: report_findings}`). Claude is now forced to return a validated JSON schema object. `_parse_findings` and `_parse_finding_block` removed; replaced by `_build_findings` which iterates the structured dict directly.

---

## ~~2. Diff Filtering Logic is Load-Bearing~~ ✓ RESOLVED

**Resolution:** Replaced `re.split` lookahead with a line-by-line state machine parser — explicit, no regex lookahead, correctly flushes the final section. Added early-skip guard in `run_bug_check` so an empty filtered diff logs `"no matching diff sections — skipping LLM call"` and short-circuits rather than firing an LLM call on empty input. Added `tests/test_orchestrator.py` covering all edge cases (no match, partial match, multi-file, single-file, last-section flush, empty diff).

---

## ~~3. Rule Targeting Has Multiple Axes~~ ✓ RESOLVED

**Resolution:** Added `Rule.match_reason(files, labels) -> str | None` — returns a human-readable string describing the first match (e.g. `"path 'foo.cpp' matches glob 'foo/**'"` or `"label 'area:ccl'"`) or None. `matches_pr` now delegates to it, eliminating duplicated logic. `select_rules` logs the reason per matched rule at DEBUG level, making selection auditable. `load_rules` warns on orphan rules (no paths and no labels) that can never be selected.

---

## ~~4. Stateful LLM Sessions for Grouped Rules~~ ✓ RESOLVED

**Resolution:** Removed `messages` state from `LLMSession` entirely. `analyze_rule` now builds a fresh single-turn `[{"role": "user", ...}]` list per call — no history is retained or shared between calls. The orchestrator loop was flattened: instead of one session per group, each rule gets its own `LLMSession`. The `group` field remains in the manifest schema (no breaking change) but has no runtime effect. `test_session_has_no_messages_state` guards against re-introduction.

---

## ~~5. Inline Comment Line Validation~~ ✓ RESOLVED

**Resolution:** Added `diff_line_numbers(diff) -> dict[str, set[int]]` to `github_client.py` — parses unified diff hunks to produce the set of valid RIGHT-side line numbers (added `+` and context ` ` lines; removed `-` lines excluded) per file. `_post_findings_as_comments` now calls this once up front and routes each finding deterministically: valid line → inline comment, invalid line → general comment with a DEBUG log. Removed the fail-and-catch cycle for invalid lines. `format_summary_comment` parameter renamed from `inline_failed` to `comment_failures` with accurate note text (API errors only, not expected line-not-in-diff cases).

---

## ~~6. Fail-Open Swallows Infrastructure Errors~~ ✓ RESOLVED

**Resolution:** Added `skipped_rules: list[str]` tracking in `run_bug_check` — populated in the `except` block with the failing rule's ID. After the rule loop, logs a `WARNING` with the count and names if any were skipped. `_post_findings_as_comments` now accepts `skipped_rules` and passes it to `format_summary_comment`. `format_summary_comment` gained a `skipped_rules` parameter that appends a `> Warning:` block listing the affected rule IDs — visible to PR authors even when there are no findings. The skipped-rules note appears regardless of whether findings were produced.

---

## ~~7. Diff Truncation Produces Silent Incomplete Analysis~~ ✓ RESOLVED

**Resolution:** Added `diff_file_paths(diff) -> set[str]` to scan `diff --git` headers. Extracted `_truncate_diff(diff, changed_files) -> (diff, truncated_files)` — a single canonical helper that performs truncation and immediately identifies which files from `changed_files` are absent from the result. Both `fetch_pr_info` and `fetch_branch_diff` now use it (eliminating duplicate truncation logic) and populate `PRInfo.truncated_files`. The orchestrator distinguishes "no matching diff sections" (rule doesn't apply) from "matched files were truncated" (analysis skipped due to large diff) — the latter populates `truncated_rules`. Truncated rules are logged as warnings and appear in the PR summary comment with a note to break the PR into smaller pieces.

---

## ~~8. External Tool Dependencies Not Validated Upfront~~ ✓ RESOLVED

**Resolution:** Added `check_prerequisites(*, need_gh, need_git)` to `github_client.py` — runs before any API or subprocess calls. `need_gh=True` checks `gh --version` (installed) then `gh auth status` (authenticated), raising `RuntimeError` with a clear actionable message on the first failure. `need_git=True` checks `git --version`. `__main__.py` calls it immediately after argument parsing: `--pr` → `check_prerequisites(need_gh=True)`, `--branch` → `check_prerequisites(need_git=True)`. Tests cover the three failure modes (gh not installed, gh not authenticated, git not installed) and the no-op case via `unittest.mock.patch`.
