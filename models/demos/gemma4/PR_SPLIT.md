# Gemma 4 prefill PR split

The original combined branch, `svuckovic/gemma4-prefill-pr`, was rebased onto
`origin/main` at `28238f903b3`. That version of main includes the SDPA changes
from `svuckovic/gemma4-prefill-pr-sdpa`, so the split branches use main's SDPA
implementation rather than carrying a separate SDPA source delta.

The work was split into these local branches:

1. `svuckovic/gemma4-prefill-tensor-repr` (`5478e677fd9`)
   - Standalone on `origin/main`.
   - Contains the tensor representation debug-performance fix.
2. `svuckovic/gemma4-prefill-model` (`381055f9f38`)
   - Standalone on `origin/main`.
   - Contains the Gemma 4 context-parallel prefill model support.
3. `svuckovic/gemma4-prefill-service` (`2b1e947c419`)
   - Stacked on `svuckovic/gemma4-prefill-model` because the service depends on
     the model support.
   - Contains the multi-slot prefill service and common prefill infrastructure.

## Verification

- `./build_metal.sh -ce` passed on the model branch.
- The hardware-independent service test selection passed: 23 tests.
- `git diff --check` passed for all three branches.
- The model branch has no changes relative to main under the core SDPA and ring
  attention all-gather source directories.
- No branches were pushed.

Hardware-dependent behavior was not tested because the development environment
does not have a Tenstorrent accelerator attached.

## Backups and workspace state

- `backup/gemma4-prefill-pr-before-sdpa-rebase-20260904` preserves the combined
  branch before rebasing onto the SDPA PR branch.
- `backup/gemma4-prefill-pr-before-main-rebase-20260906` preserves it before
  rebasing onto the latest main branch.
- The unrelated untracked nested directory
  `models/demos/t3000/llama2_70b/` was left untouched.
