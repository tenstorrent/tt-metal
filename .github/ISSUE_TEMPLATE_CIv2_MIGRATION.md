---
name: Migrate CI Runners to CIv2
about: Track the migration of all remaining CIv1 runners to CIv2 format
title: '[CI] Migrate all CI runners to CIv2'
labels: ['ci-cd', 'infrastructure', 'enhancement']
assignees: ''
---

## Summary
Migrate all remaining CIv1 CI runners to the CIv2 format to standardize our CI infrastructure and improve maintainability.

## Background
Currently, the repository has a mix of CIv1 and CIv2 runners:
- **CIv1 runners**: Use array-based labels like `["pipeline-perf", "P150", "in-service"]`
- **CIv2 runners**: Use string labels like `tt-ubuntu-2204-p150b-stable`

As of February 2026:
- **CIv2 runners**: ~25 instances across 20+ workflows (already migrated)
- **CIv1 runners**: ~24 instances across 6 workflows (need migration)

## Motivation
1. **Consistency**: Standardize all CI infrastructure on CIv2 format
2. **Maintainability**: Simpler label format reduces complexity
3. **Reliability**: CIv2 infrastructure provides better stability
4. **Future-proofing**: CIv1 may be deprecated in the future

## Files Requiring Migration

### 1. `.github/workflows/fast-dispatch-frequent-tests-impl.yaml`
- [ ] Line 43: `["pipeline-perf", "P150", "in-service"]` → `tt-ubuntu-2204-p150b-stable`
- [ ] Line 51: `["pipeline-perf", "P150", "in-service"]` → `tt-ubuntu-2204-p150b-stable`

### 2. `.github/workflows/galaxy-unit-tests-impl.yaml`
- [ ] Lines 37, 44: `["arch-wormhole_b0", ...]` → Dynamic CIv2 label

### 3. `.github/workflows/metal-run-microbenchmarks-impl.yaml`
- [ ] Line 26: N300 bare-metal → `tt-ubuntu-2204-n300-stable`
- [ ] Lines 32, 38, 70: T3000 config → `tt-ubuntu-2204-t3000-stable`
- [ ] Line 44: N300 bare-metal → `tt-ubuntu-2204-n300-stable`
- [ ] Line 50: BH P150 → `tt-ubuntu-2204-p150b-stable`
- [ ] Lines 56, 76: BH general → `tt-ubuntu-2204-p150b-stable`

### 4. `.github/workflows/perf-models-impl.yaml`
- [ ] Lines 32-34: N300 runners → Dynamic CIv2 with N300
- [ ] Lines 36-37: P150 runners → Dynamic CIv2 with P150

### 5. `.github/workflows/publish-release-image.yaml`
- [ ] Line 96: N150 VM → `tt-ubuntu-2204-n150-stable`
- [ ] Line 101: N300 VM → `tt-ubuntu-2204-n300-stable`

### 6. `.github/workflows/tm-data-movement-perf-impl.yaml`
- [ ] Line 26: N300 bare-metal → `tt-ubuntu-2204-n300-stable`
- [ ] Line 32: BH → `tt-ubuntu-2204-p150b-stable`

## CIv1 → CIv2 Label Mappings

| CIv1 Label Pattern | CIv2 Label | Device |
|-------------------|------------|--------|
| `["pipeline-perf", "P150", "in-service"]` | `tt-ubuntu-2204-p150b-stable` | Blackhole P150 |
| `["BH", "pipeline-perf", "in-service"]` | `tt-ubuntu-2204-p150b-stable` | Blackhole |
| `["N300", "pipeline-perf", "bare-metal", "in-service"]` | `tt-ubuntu-2204-n300-stable` | Wormhole N300 |
| `["arch-wormhole_b0", "pipeline-perf", "config-t3000", "in-service"]` | `tt-ubuntu-2204-t3000-stable` | T3000 |
| `["cloud-virtual-machine", "N150", "in-service"]` | `tt-ubuntu-2204-n150-stable` | N150 VM |
| `["cloud-virtual-machine", "N300", "in-service"]` | `tt-ubuntu-2204-n300-stable` | N300 VM |

## Implementation Plan

### Phase 1: Simple Static Migrations
Migrate static labels in 4 files with low risk:
- `fast-dispatch-frequent-tests-impl.yaml`
- `metal-run-microbenchmarks-impl.yaml`
- `publish-release-image.yaml`
- `tm-data-movement-perf-impl.yaml`

### Phase 2: Dynamic Label Migrations
Handle workflows with dynamic inputs (higher complexity):
- `galaxy-unit-tests-impl.yaml`
- `perf-models-impl.yaml`

### Phase 3: Testing & Validation
- Verify all workflows run successfully
- Monitor for runner availability issues
- Check for performance regressions

## Acceptance Criteria
- [ ] All 6 workflow files migrated to CIv2 labels
- [ ] No CIv1 array-based labels with "in-service" or "pipeline-*" remain
- [ ] All migrated workflows pass in CI
- [ ] No performance regressions observed

## Related Work
- v0.63.0: Initial CIv2 migrations for model jobs, N300 runners
- v0.64.0: Moved APC and BHPC to vIOMMU in CIv2
- v0.64.4: Updates for cleanup jobs on CIv2

## Additional Resources
See `.github/MIGRATION_ISSUE_CIv2_RUNNERS.md` for detailed migration guide and examples.
