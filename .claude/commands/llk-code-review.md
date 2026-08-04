---
allowed-tools: Bash(gh issue view:*), Bash(gh search:*), Bash(gh issue list:*), Bash(gh pr comment:*), Bash(gh pr diff:*), Bash(gh pr view:*), Bash(gh pr list:*), Bash(gh api:*)
description: Code review a pull request
---

Provide a code review for the given pull request.

The workflow places authoritative LLK review knowledge in this directory,
relative to the repository root:

`.llk-review-knowledge-src/dashboard/pr_review/knowledge`

In the steps below, `K` denotes that directory. Expand `K` to the literal path
above in every Task prompt so each subagent receives an exact readable path.

## Operating rules

- Task subagents have isolated contexts. They do not inherit this command, the
  parent's appended system prompt, conversation history, or prior tool results.
  Every Task prompt MUST therefore tell the subagent which exact knowledge files
  to read and include the PR title, description, change summary, and existing-
  discussion digest. Refer to files by path; do not paste the full corpus.
- Tools are functional. Do not test them or make exploratory calls. Use a tool
  only when it is required for the review.
- Use `gh` for GitHub data. Do not use web fetch. Do not run builds or tests and
  do not download a compiler/toolchain during the review.
- Use an MCP source only to settle a specific LLK/architecture fact that the
  repository cannot settle; do not perform broad documentation searches.
- Review only issues introduced by this PR, but read surrounding and cross-file
  context when needed. LLK hazards often span init/calculate paths, architecture
  variants, and the tt-llk/tt-metal API boundary.
- Prefer recall for LLK-specific problems. A grounded conditional or uncertain
  issue is worth returning as `plausible`; include the evidence, confidence, and
  one concrete fact that would confirm or refute it. Do not manufacture generic
  concerns merely to produce comments.

Create a todo list, then follow these steps precisely:

1. Launch a Haiku agent to check whether the pull request is closed. If it is
   closed, post a short issue-level skip comment with `gh pr comment` when
   `--comment` was provided, then stop. Review every open PR that was explicitly
   dispatched, including draft, automated, Claude-generated, and trivial PRs.

2. Launch a Haiku agent to return only the paths of applicable `CLAUDE.md` files:
   the repository root file if present, plus files in or above directories touched
   by the PR. A `CLAUDE.md` rule applies only to files under its directory.

3. Launch a Sonnet intake agent to inspect the PR and return:
   - title, description, base/head SHAs, and changed-file list;
   - a concise change summary and which architecture/code paths are affected;
   - booleans for `touches_test_or_golden`, `touches_sfpi_or_raw_tti`, and
     `changes_llk_api_or_behavior`;
   - a concise digest of existing issue comments, inline review comments, and
     reviews, including their paths/lines, so later agents avoid duplicates.

4. Launch these four review agents in parallel. Every task prompt must include
   the shared PR context from step 3, the applicable `CLAUDE.md` paths from step
   2, the recall policy above, and its role-specific Read instructions:

   - **Agent 1 — Sonnet, mandatory team-policy pass.** Read
     `K/pinned-rules.md` and `K/learnings.md`. Actively check every pinned rule
     against the diff and needed context; do not sample the list. Also read
     applicable `CLAUDE.md` files.
   - **Agent 2 — Sonnet, API/parity/test pass.** Read `K/review-rubric.md` and
     `K/learnings.md`. Check architecture parity, metal propagation, breaking API
     changes, guards, cleanup/style rules, and PR scope. Only when
     `touches_test_or_golden` is true, also read `K/golden-review.md` and audit the
     test/reference-model changes.
   - **Agent 3 — Opus, correctness/hazard pass.** Read `K/review-rubric.md` and
     `K/learnings.md`. Trace changed behavior through surrounding code, init versus
     calculate paths, callers, formats/modes, and relevant WH/BH/QSR variants.
     Focus on correctness, HW state, ordering, and edge cases.
   - **Agent 4 — Opus, independent correctness/performance pass.** Read
     `K/review-rubric.md` and `K/learnings.md` and independently hunt for introduced
     bugs. Only when `touches_sfpi_or_raw_tti` is true, also read
     `K/performance-audit.md`; apply its provenance and false-positive guards. Since
     this run does not build/disassemble, label claims requiring assembly evidence
     as plausible suggestions rather than proven performance wins.

   Each agent must return zero or more candidates with this exact information:
   `path`, right-side changed `line`, `category`, `confidence` (`confirmed` or
   `plausible`), concise mechanism and consequence, concrete evidence, suggested
   fix/check, `knowledge_sources` (exact file and rule/section), and whether an
   existing comment already covers it. A candidate must be actionable and tied to
   changed code. For a plausible candidate, also return its falsifier. Do not
   return pre-existing issues, duplicates, explicitly silenced rules, generic
   lint/compiler findings, or a concern disproved by the code.

5. Merge exact duplicates before validation. For each unique candidate, launch a
   focused validation subagent in parallel. Pass only the PR metadata, that single
   candidate, and the relevant diff/context location. Tell the validator to Read
   only the candidate's cited `knowledge_sources` and applicable `CLAUDE.md`, not
   the entire knowledge directory. Use Opus for correctness, hazard, and
   performance candidates; use Sonnet for policy, parity, propagation, style,
   cleanup, and test candidates.

   Each validator must return one verdict:
   - `confirmed`: the evidence establishes the issue;
   - `plausible`: the changed code creates a grounded risk, but one stated fact is
     needed to settle it;
   - `rejected`: the code or rule does not support it;
   - `pre_existing`; or
   - `duplicate`, identifying the existing comment.

6. Keep `confirmed` and genuinely grounded `plausible` findings. Drop `rejected`,
   `pre_existing`, and `duplicate` findings. Merge findings with the same root
   cause. Before writing comments, Read `K/conventions.md` exactly once and apply it
   to every finding. In particular: use terse senior-reviewer language, prefix
   nits with `nit:`, phrase uncertainty as a question with its confidence and
   falsifier, and never use GitHub suggestion blocks. Do not expose internal
   severity labels or validation narration in comments.

7. Output a concise findings summary to the terminal. If `--comment` was not
   provided, stop without writing to GitHub. If it was provided and no findings
   survived, post this issue-level comment and stop:

   ```markdown
   ## Code review

   No issues found. Checked LLK correctness, hazards, team rules, and applicable
   architecture/test context.
   ```

8. If findings survived, refresh the PR's issue comments and inline review
   comments immediately before posting. Remove any finding another reviewer has
   already covered since step 3. Prepare one self-contained comment per unique
   issue; do not publish the preparation list.

9. Post each finding as an inline comment with `gh api`. Do not call
   `mcp__github_inline_comment__create_inline_comment`: claude-code-action does
   not install that server for `workflow_dispatch` runs.

   Get the current head SHA immediately before posting:

   ```bash
   gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}" --jq '.head.sha'
   ```

   Then post using the changed-file path and a valid new-side diff line:

   ```bash
   gh api --method POST "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}/comments" \
     --raw-field body="COMMENT_BODY" \
     --raw-field commit_id="HEAD_SHA" \
     --raw-field path="CHANGED_FILE_PATH" \
     --field line=NEW_SIDE_LINE_NUMBER \
     --raw-field side="RIGHT"
   ```

   Verify every successful write returns an `html_url`. If an inline anchor is
   rejected, post that finding with `gh pr comment`, prefixing it with
   `` `path:line` ``, so the location is preserved. A fallback failure is an
   error. Never finish a `--comment` run without at least one successful GitHub
   write.

When a comment relies on a documented repository rule, identify that rule and
link it using the reviewed repository and full head SHA. Do not add redundant
code links to ordinary inline comments—the inline anchor already supplies the
location.
