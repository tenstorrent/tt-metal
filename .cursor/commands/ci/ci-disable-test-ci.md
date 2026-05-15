# CI Disable Test (Automation Mode)

## Purpose
Apply the smallest safe CI-only disable for a failing signal, but do not perform git/gh operations.

## Input
- Required: one GitHub issue URL/number in `ebanerjeeTT/issue_dump` during testing, or in `tenstorrent/tt-metal` after production promotion.
- Optional: one or more job URLs for stronger evidence.

## Hard Constraints
- You may read files and edit files in this repository.
- You may use `gh` only for read operations needed to inspect issue/run context and logs.
- Do not create branches.
- Do not run `git add`, `git commit`, `git push`, or `gh pr create`.
- Do not dispatch workflows.

## Procedure
1. Resolve issue/job context from provided input.
2. Download and inspect failed logs in `build_ci/disabling`.
3. Identify the narrowest disable scope that is supported by evidence.
4. Apply only minimal code/workflow edits needed to disable the failing target.
5. Add a TODO comment near the disable with non-closing issue reference.
6. Clean transient logs from `build_ci/disabling`.

## Required Final Output
At the end of your response, print this exact marker on its own line:
`===FINAL_DISABLE_EDIT_SUMMARY===`

After the marker, print only compact JSON with this schema:
```json
{
  "issue_number": 0,
  "disable_scope": "short description",
  "files_modified": ["path1", "path2"],
  "notes": "optional short note"
}
```
