# Tier 1 Ease-of-Use Issue — Agent Prompt Template

> **How to use:** copy this file, rename it `eou_<issue_num>_prompt.md`, fill in every
> `{{PLACEHOLDER}}` from the live GitHub issue, and hand the filled-in file to the agent.
>
> Fields to fill:
> | Placeholder | Example |
> |---|---|
> | `{{REPO}}` | `tt-metal` or `tt-llk` |
> | `{{ISSUE_NUMBER}}` | `34587` |
> | `{{ISSUE_URL}}` | `https://github.com/tenstorrent/tt-metal/issues/34587` |
> | `{{ISSUE_TITLE}}` | `Pack tilize/untilize bools → PackMode enum` |
> | `{{BRANCH_SLUG}}` | `pack-mode-enum` (short, kebab-case, descriptive) |
> | `{{ISSUE_BODY}}` | paste the full issue body verbatim in section 2 |

---

## 1. Context and mission

You are implementing a scoped C++ refactoring issue from the tt-metal ease-of-use epic.
The issue is tracked as **{{REPO}}#{{ISSUE_NUMBER}} — {{ISSUE_TITLE}}**.
Live issue: {{ISSUE_URL}}

The goals of the epic are: improve readability, eliminate latent bugs from stringly-typed or
multi-boolean APIs, and make the LLK layer self-documenting. Your job is to implement the change
described in section 2, verify it builds and tests pass, then open a pull request.

Repository layout cheat-sheet:
- Internal LLK (main change target): `tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/`
- LLK API wrappers (translation layer): `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/`
- Public compute kernel API (do **not** modify): `tt_metal/include/compute_kernel_api/`
- LLK standalone tests: `tt_metal/tt-llk/tests/sources/`
- TTNN (do **not** modify unless the issue explicitly says so): `ttnn/`

> **Note on `tt_metal/tt-llk/`:** this directory is a plain source tree in `tt-metal`. Edit files
> in-place; there is no submodule commit step required.

---

## 2. Issue body (paste verbatim)

> Replace this block with the full text of {{ISSUE_URL}}.
> Include all sections the issue author wrote: problem statement, motivation, proposed API,
> scope notes, out-of-scope items, and any attached plans or links.

{{ISSUE_BODY}}

---

## 3. Mandatory execution rules

Read every rule before writing a single line of code.

### 3.1 Branch — always base off fresh main

```bash
cd /localdev/ncvetkovic/work/tt-metal
git fetch origin main
git checkout -b ncvetkovic/{{ISSUE_NUMBER}}-{{BRANCH_SLUG}} FETCH_HEAD
```

Do this first, before any edits. The branch **must** be one commit (or a small cohesive series)
ahead of the current tip of `origin/main`. Never stack on top of another feature branch or on
top of any `eou_triage/` commit.

If the issue itself explicitly states a different base (e.g. "rebase on top of PR #XXXXX"), follow
that instruction instead and document why in the PR description.

### 3.2 Scope discipline

- Touch only files the issue describes.
- Before editing any file, ask yourself: "does the issue text, or the out-of-scope list in section 2,
  say anything about this file?" If in doubt, do not edit it and add a note in the PR.
- Never modify files under `tt_metal/include/compute_kernel_api/` unless the issue explicitly
  asks for it.
- Never modify files under `tt_metal/tests/tt_metal/**/test_kernels/` — those use the public
  API which stays on bools/ints.
- Never modify TTNN unless the issue explicitly says so.

### 3.3 Verification before committing

Run at minimum:

```bash
# 1. Build (picks up arch from env; default is Blackhole on this machine)
./build_metal.sh --build-tests

# 2. Invariant grep — adapt the pattern to this specific issue
#    (example for a bool→enum refactor; replace with whatever invariant the issue defines)
grep -rn "<PATTERN_THAT_SHOULD_BE_ZERO_AFTER_REFACTOR>" tt_metal/tt-llk/
```

If the issue has a CI test command, run it. If it doesn't, run the LLK standalone test suite:

```bash
cd tt_metal/tt-llk/tests
pytest --compile-producer -n 8 -x python_tests/
pytest --compile-consumer -x python_tests/
```

If a hang occurs: `pkill -9 -f pytest && tt-smi -r`

Do not commit if any of the above fail. Fix first.

### 3.4 Commit message

Use this format (fill in from the issue):

```
refactor(llk): {{ISSUE_TITLE}}

<2–4 sentences explaining the motivation and what changed at a high level.>

Resolves {{REPO}}#{{ISSUE_NUMBER}}
```

One commit per logical phase is fine (e.g. one per arch). Do not squash phases that touch
different architectures — reviewers appreciate the separation. Keep each commit self-contained
and buildable.

### 3.5 Push and open the PR

```bash
git push origin ncvetkovic/{{ISSUE_NUMBER}}-{{BRANCH_SLUG}}

cd /localdev/ncvetkovic/work/tt-metal
gh pr create \
  --base main \
  --head ncvetkovic/{{ISSUE_NUMBER}}-{{BRANCH_SLUG}} \
  --title "refactor(llk): {{ISSUE_TITLE}}" \
  --body "$(cat <<'PRBODY'
<fill in using the template in section 4>
PRBODY
)"
```

---

## 4. PR description template

Use the template below **verbatim as structure** — do not omit sections. The description should
be long enough that a reviewer who has never read the issue can understand the motivation,
what changed, what was intentionally left alone, and what to focus on. Aim for 300–600 words in
the body (excluding checklists). Short PRs get ignored; complete PRs get merged.

```markdown
### Summary

<!-- One paragraph: what problem does this solve and why does it matter?
     Link the issue with: Closes {{REPO}}#{{ISSUE_NUMBER}} -->

<Explain the problem that existed before this PR — e.g. "Two booleans `untilize` and `tilize`
encoded three valid packer states; the fourth combination was silently accepted but meaningless.
Callsites like `_llk_pack_init_<false, false, false>(fmt)` were unreadable — callers had to
memorize parameter order."  Then explain what this PR does to fix it.>

Closes {{REPO}}#{{ISSUE_NUMBER}}

### What's changed

<!-- Bullet list of concrete changes. Be specific: file paths, function names, enum values.
     Group by architecture or layer if applicable. -->

- **`<file path>`** — <what changed and why>
- **`<file path>`** — <what changed and why>
- …

<!-- If the change spans multiple architectures, use sub-bullets or sub-headers:
     **Blackhole** / **Wormhole B0** -->

### What's intentionally unchanged

<!-- Equally important. Call out every major boundary you held. -->

- Public compute kernel API (`tt_metal/include/compute_kernel_api/`) — no changes; wrappers
  translate at the boundary so callers are unaffected.
- Test kernels under `tt_metal/tests/tt_metal/**/test_kernels/` — unchanged; they use the
  public API which keeps booleans.
- <Any other explicitly out-of-scope area the issue mentions>

### Architecture-specific notes

<!-- Only include this section if there are arch differences worth calling out.
     Delete if not applicable. -->

- **Blackhole** supports <X>; **Wormhole B0** does not — guarded with `static_assert(...)`.
- <Any other arch asymmetry>

### Notes for reviewers

<!-- Where should reviewers focus? Point to the non-obvious decisions.
     Call out tradeoffs or alternatives you considered and rejected. -->

- Focus on `<file>:<line range>` — this is where <subtle decision> was made.
- Alternative considered: <describe alternative> — rejected because <reason>.
- The compile-time invariant that verifies correctness:
  ```bash
  grep -rn "<INVARIANT_PATTERN>" tt_metal/tt-llk/   # should return zero hits
  ```

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [x] Refactoring

### Checklist

- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)
```

---

## 5. Definition of Done

The issue is complete when **all** of the following hold:

- [ ] Branch is exactly N commits ahead of the tip of `origin/main` at the time of the PR
      (zero eou_triage files, zero unrelated changes).
- [ ] The invariant the issue defines (e.g. "grep returns zero hits") passes.
- [ ] `./build_metal.sh --build-tests` succeeds for all architectures touched.
- [ ] LLK standalone tests pass (or the relevant subset per the issue).
- [ ] PR is open with the full description from section 4 (no placeholder text left in).
- [ ] PR title follows `refactor(llk): <ISSUE_TITLE>` (or `fix(llk):` / `docs(llk):` as appropriate).
- [ ] No files outside the declared scope were modified.
- [ ] No new compiler warnings introduced.

---

## 6. Common pitfalls

1. **Stacking on the wrong base.** Always branch from `FETCH_HEAD` of `origin/main`, not from
   any local branch or worktree that has unrelated eou/codegen commits.

2. **Template parameter default order.** If a new enum param replaces bools, defaults must come
   last: `template <PackMode mode = PackMode::Default, bool zero_output = false>` — not the reverse.

3. **`if constexpr` translations.** Replace `if constexpr (bool_a || bool_b)` with the equivalent
   enum comparison; don't guess — trace every boolean's meaning first.

4. **Missing include.** Adding a new type (enum class, etc.) in one header doesn't mean it's
   visible everywhere. If the compiler says "not declared", add the include.

5. **Forgetting arch asymmetry.** Wormhole B0 and Blackhole often have different function
   signatures. Check both before assuming the pattern is symmetric.

6. **Pushing to the wrong branch.** Double-check `git log --oneline -5` and `git branch`
   before every push.

7. **PR with placeholder text.** Fill in every `<…>` in the PR body. Reviewers seeing
   `<!-- explain here -->` will request changes before reading the diff.
