# LLK pull-request review directive

Provide a code review for the pull request named in the user prompt.

Parse the pull-request reference and whether the literal `--comment` flag is
present before doing anything else. Record that boolean once and preserve it
through the entire run. A successful run with `--comment` MUST end with at least
one GitHub write whose command result is a non-empty `html_url`; otherwise report
an error instead of claiming that the review completed.

The workflow places authoritative LLK review knowledge in this directory,
relative to the repository root:

`.llk-review-knowledge-src/dashboard/pr_review/knowledge`

In the steps below, `K` denotes that directory. Expand `K` to the literal path
above in every Task prompt so each subagent receives an exact readable path.

The workflow also materializes the exact PR revisions in these directories:

- `H` = `.llk-review-pr-head` (the exact reviewed head commit)
- `B` = `.llk-review-pr-base` (the exact PR merge-base comparison commit)

Expand `H` and `B` to those literal paths in every Task prompt. The checkout root
contains review orchestration and MUST NOT be used as the source of PR code.

## Operating rules

- Task subagents have isolated contexts. They do not inherit this command, the
  parent's appended system prompt, conversation history, or prior tool results.
  Every Task prompt MUST repeat the shell/filesystem constraints and name the
  literal `H` and `B` paths. After intake, every review/validation Task prompt
  MUST also name its exact knowledge files and include the PR title, concise
  body/change summary, exact base-tip/comparison-base/head SHAs, relevant
  changed-file groups, and existing-discussion digest. Refer to files by path;
  do not paste the full corpus.
- Every delegated Agent call MUST set `run_in_background: false`. Never launch a
  background agent. For a phase with multiple independent agents, issue all of
  that phase's Agent tool calls together in one assistant message so they execute
  concurrently while remaining blocking. Do not advance to validation, posting,
  or a final response until every Agent tool result for the current phase has
  been received. A progress statement such as "agents are running" is never a
  valid final response.
- Tools are functional. Do not test them or make exploratory calls. Use a tool
  only when it is required for the review.
- Keep all worktrees read-only. Never call Write, Edit, or NotebookEdit, and
  never save PR data, diffs, summaries, plans, comments, or intermediate results
  to a file. Keep coordination and comment preparation in context.
- Read current code and surrounding context only under `H`; read baseline code
  only under `B`. Never read repository implementation from the checkout root,
  another branch, a Claude execution file, or a tool-result cache.
- Use Read, Grep, and Glob for files under `H`, `B`, and `K`. Use Bash only for
  one direct, single-line `gh api` command, the intake agent's one `gh pr view`
  metadata call, one complete `gh pr diff` call, or one read-only `git diff`,
  `git show`, `git log`, or `git rev-parse` command per tool call. Outside intake,
  scope diffs to a relevant changed path and use the exact comparison-base/head
  SHAs. Never run `find /`.
- Run read-only git commands directly from the orchestration checkout root, whose
  object database contains both exact commits. A valid diff starts literally
  with `git diff COMPARISON_BASE_SHA HEAD_SHA -- CHANGED_PATH`. Never prepend
  `git -C`, `cd`, an absolute worktree path, or any other command/prefix. `H` and
  `B` are for Read/Grep/Glob file access, not for changing Bash's working directory.
- Do not use pipes, `&&`, `||`, `;`, `&`, redirects, heredocs, command/process
  substitution, inline environment assignments, or multiple commands. Use
  separate tool calls when more than one query is needed.
- Use `gh`'s native `--json` and `--jq` flags instead of an external parser. Use
  literal repository, PR, path, line, and SHA values learned from prior tool
  results rather than shell variables. Pass comment bodies as arguments to the
  direct `gh` call; do not use stdin or temporary files.
- Do not use web fetch. Do not run builds or tests and do not download a
  compiler/toolchain during the review.
- Use an MCP source only to settle a specific LLK/architecture fact that the
  repository cannot settle; do not perform broad documentation searches.
- Review only issues introduced by this PR, but read surrounding and cross-file
  context when needed. LLK hazards often span init/calculate paths, architecture
  variants, and the tt-llk/tt-metal API boundary.
- Prefer recall for LLK-specific problems. A grounded conditional or uncertain
  issue is worth returning as `plausible`; include the evidence, confidence, and
  one concrete fact that would confirm or refute it. Do not manufacture generic
  concerns merely to produce comments.
- Delegated agents return candidates or verdicts only. They MUST NOT post to
  GitHub or call `ReportFindings`; the parent command is the sole posting owner.

Follow these steps precisely:

1. Launch a Haiku agent to check whether the pull request is closed. Give it the
   parsed PR reference and the shell/filesystem constraints; it may inspect PR
   state only and must not inspect code or post. If the PR is closed, the parent
   posts a short issue-level skip comment through the issue-comments `gh api`
   endpoint when `--comment` was provided, selects `.html_url`, verifies that it
   is non-empty, then stops. Review every open
   PR that was explicitly dispatched, including draft, automated,
   Claude-generated, and trivial PRs.

2. Launch a Haiku agent to return only the paths of applicable `CLAUDE.md` files:
   the exact-head root file under `H` if present, plus files under `H` in or above
   directories touched by the PR. A `CLAUDE.md` rule applies only to files under
   its directory. It must not inspect `CLAUDE.md` or code from the orchestration
   checkout root. It may query the changed-file list through bounded PR metadata
   endpoints, but must not request patches or the whole diff.

3. Launch a Sonnet intake agent to inspect the entire PR, including the complete
   PR diff, metadata, and discussion, so the shared context covers every changed
   file and cross-file relationship. It may make one direct `gh pr diff` call for
   this purpose. It must return:
   - title, description, exact base-branch-tip/head SHAs, and changed-file list;
   - a concise change summary and which architecture/code paths are affected;
   - compact changed-file groups that later agents can use for path-scoped review;
   - booleans for `touches_test_or_golden`, `touches_sfpi_or_raw_tti`, and
     `changes_llk_api_or_behavior`;
   - a concise digest of existing issue comments, inline review comments, and
     reviews, including their paths/lines, so later agents avoid duplicates.

   The intake agent should synthesize the complete diff rather than echoing large
   patches in its response, but must not omit a changed file from its analysis.
   The parent verifies that the returned base-tip/head SHAs match the exact
   revisions named in the coordination system prompt before launching code
   reviewers; a mismatch is an error and must stop the review. The parent adds
   the exact PR merge-base comparison SHA from that system prompt to the shared
   context.

4. Launch these four review agents concurrently and synchronously. In one
   assistant message, emit exactly four Agent tool calls, each with
   `run_in_background: false`; do not launch them one per turn and do not use
   background mode. Wait for all four tool results before doing anything in step
   5. Every task prompt must include
   the shared PR context from step 3, the applicable `CLAUDE.md` paths from step
   2, the recall policy above, and its role-specific Read instructions. It must
   also tell the agent to compare the exact merge-base SHA to the exact head SHA
   with separate path-scoped commands run directly from the current checkout root:
   `git diff COMPARISON_BASE_SHA HEAD_SHA -- CHANGED_PATH`. Tell it to read
   surrounding current/baseline code only from the literal `H`/`B` paths.
   No agent may request or return the whole PR diff, inspect implementation code
   in the checkout root, post to GitHub, or call `ReportFindings`:

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
   `path`, right-side changed `line` (the last line of the anchor), `start_line`
   (the first line for a multi-line anchor, otherwise `null`), `category`,
   `confidence` (`confirmed` or `plausible`), concise mechanism and consequence,
   concrete evidence, suggested fix/check, `knowledge_sources` (exact file and
   rule/section), and whether an existing comment already covers it. Select the
   smallest useful anchor: keep `start_line: null` when one changed line identifies
   the issue; use a range only when two or more contiguous right-side diff lines
   are all necessary to show the implicated block. Never pad every comment to a
   fixed-size range. A candidate must be actionable and tied to changed code. For
   a plausible candidate, also return its falsifier. Do not return pre-existing
   issues, duplicates, explicitly silenced rules, generic lint/compiler findings,
   or a concern disproved by the code.

   Each agent is responsible for every changed-file group relevant to its role,
   but should open only the individual patches and surrounding regions required
   for that role. It must summarize evidence rather than echoing large patches.

   After the four calls return, verify that four complete candidate responses
   were received. If any result is missing, do not summarize progress or finish;
   synchronously retry only the missing role and wait for its result.

5. Merge exact duplicates before validation. For each unique candidate, launch a
   focused validation subagent concurrently with the others. Emit all validator
   Agent calls for this phase together in one assistant message, set
   `run_in_background: false` on every call, and wait for every tool result before
   continuing. Pass only the PR metadata, that single
   candidate, exact comparison-base/head SHAs, literal `H`/`B` paths, and the
   relevant changed path/context location. Tell the validator to use a path-
   scoped exact-SHA diff, read current/baseline code only from `H`/`B`, and Read
   only the candidate's cited `knowledge_sources` and applicable `CLAUDE.md`, not
   the entire knowledge directory. It must not use the checkout-root implementation,
   temporary files, `find /`, GitHub writes, or `ReportFindings`. Use Opus for
   correctness, hazard, and performance candidates; use Sonnet for policy,
   parity, propagation, style, cleanup, and test candidates.

   If any validator result is missing, synchronously retry only that validator
   and wait. Never continue to filtering or posting while a validator is running.

   Each validator must verify that `path`, `start_line`, and `line` identify the
   minimal relevant right-side range in the PR diff, correcting the anchor when
   needed. It must then return the validated anchor and one verdict:
   - `confirmed`: the evidence establishes the issue;
   - `plausible`: the changed code creates a grounded risk, but one stated fact is
     needed to settle it;
   - `rejected`: the code or rule does not support it;
   - `pre_existing`; or
   - `duplicate`, identifying the existing comment.

6. Keep `confirmed` and genuinely grounded `plausible` findings. Drop `rejected`,
   `pre_existing`, and `duplicate` findings. Merge findings with the same root
   cause and retain the smallest validated anchor that covers that root cause.
   Before writing comments, Read `K/conventions.md` exactly once and apply it to
   every finding. In particular: use terse senior-reviewer language, prefix nits
   with `nit:`, phrase uncertainty as a question with its confidence and falsifier,
   and never use GitHub suggestion blocks. Do not expose internal severity labels
   or validation narration in comments.

7. If `--comment` was not provided, return a concise findings summary in your
   response and stop without writing to GitHub. Do not use Bash, a file, or
   `ReportFindings` for the summary.

   If `--comment` was provided, do not return the final response yet. Immediately
   before any review-result write, query the PR's current `.head.sha` with a
   direct `gh api` call and compare it to the exact reviewed head SHA from step 3.
   If they differ, do not post stale findings or a no-findings summary; report
   that the PR changed during review and stop with an error.

   If no findings survived, post this issue-level comment with one direct issue-
   comments `gh api` call using `--raw-field body=... --jq '.html_url'`:

   ```markdown
   ## Code review

   No issues found. Checked LLK correctness, hazards, team rules, and applicable
   architecture/test context.
   ```

   Verify that the command result is a non-empty URL. Only then return the concise
   no-findings summary and the posted URL in the final response.

8. If findings survived, refresh the PR's issue comments and inline review
   comments immediately before posting. Remove any finding another reviewer has
   already covered since step 3. Prepare one self-contained comment per unique
   issue in context; do not create a file or publish the preparation list. Check
   each final `path`, `start_line`, and `line` against the refreshed diff. Use
   `start_line: null` for a single-line anchor; for a range, require
   `start_line < line` and keep both endpoints on the right side of the same diff
   hunk.

9. Post each finding as an inline comment with `gh api`. Do not call
   `mcp__github_inline_comment__create_inline_comment`: claude-code-action does
   not install that server for `workflow_dispatch` runs.

   Re-read the current head SHA immediately before posting each batch:

   ```bash
   gh api "repos/OWNER/REPOSITORY/pulls/PR_NUMBER" --jq '.head.sha'
   ```

   Confirm again that it equals the exact reviewed head SHA, then place it
   literally in the separate posting call. Do not assign a shell variable or
   combine the calls. A mismatch is an error and no stale comment may be posted.

   For a single-line anchor (`start_line: null`), post using the changed-file path
   and valid new-side diff line:

   ```bash
   gh api --method POST 'repos/OWNER/REPOSITORY/pulls/PR_NUMBER/comments' --raw-field body='COMMENT_BODY' --raw-field commit_id='HEAD_SHA' --raw-field path='CHANGED_FILE_PATH' --field line=NEW_SIDE_LINE_NUMBER --raw-field side='RIGHT' --jq '.html_url'
   ```

   For a genuine multi-line anchor, add GitHub's `start_line` and `start_side`;
   `line` remains the last line of the range:

   ```bash
   gh api --method POST 'repos/OWNER/REPOSITORY/pulls/PR_NUMBER/comments' --raw-field body='COMMENT_BODY' --raw-field commit_id='HEAD_SHA' --raw-field path='CHANGED_FILE_PATH' --field start_line=FIRST_NEW_SIDE_LINE --raw-field start_side='RIGHT' --field line=LAST_NEW_SIDE_LINE --raw-field side='RIGHT' --jq '.html_url'
   ```

   Verify every successful write returns a non-empty `html_url`. If an inline
   anchor is rejected, post that finding through the issue-comments `gh api`
   endpoint with `--jq '.html_url'`, prefixing its body with
   `` `path:line` `` for a single line or `` `path:start_line-line` `` for a
   range, so the location is preserved. A fallback failure is an error. Never
   finish a `--comment` run without at least one successful GitHub write.

   Only after all required writes have returned non-empty URLs may you return the
   concise findings summary and posted URLs in the final response.

When a comment relies on a documented repository rule, identify that rule and
link it using the reviewed repository and full head SHA. Do not add redundant
code links to ordinary inline comments—the inline anchor already supplies the
location.
