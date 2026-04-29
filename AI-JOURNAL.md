# Branch: brain/artifact-reuse-by-sha

## Goal
Eliminate redundant C++ builds when multiple workflows trigger for the same commit SHA.

## Change
Added `find-existing-build` job to `build-artifact.yaml` that:
1. Queries GitHub API for prior successful runs of build-artifact.yaml on the same commit SHA
2. Checks if required artifacts exist (non-expired) in those runs, matching tracy/non-tracy variant
3. If found: feeds the run ID into the existing `download-artifacts` job which downloads + re-uploads them
4. If not found: normal build proceeds as before

Key integration point: reused the existing `use-artifacts-from-run` / `download-artifacts` infrastructure
rather than building a parallel mechanism. The `download-artifacts` job now triggers on EITHER an explicit
`use-artifacts-from-run` input OR the auto-detected `find-existing-build.outputs.run-id`.

## Expected Impact
- Each commit SHA is built at most once, regardless of how many workflows trigger
- PR CI + merge-queue CI + nightly pipelines all benefit automatically
- No changes needed to the 66 caller workflows
- `find-existing-build` runs on ubuntu-latest (free, ~5-10 seconds of API calls)

## Key Design Decisions
- Integrated with existing `download-artifacts` job instead of building a new reuse path
- `find-existing-build` is skipped when `use-artifacts-from-run` is explicitly provided
- Added `always() && !failure() && !cancelled()` to `download-artifacts` so it runs even when
  `find-existing-build` is skipped (explicit run ID case)
- Tracy/non-tracy matching: checks artifact name for `_profiler_` suffix
- Race condition (two pipelines start simultaneously): both see nothing, both build — same as today, acceptable
- Artifact expiry is checked via `.expired` field to avoid downloading expired artifacts
- Added `actions: read` permission for the API calls
