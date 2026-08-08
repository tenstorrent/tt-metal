# Migration Issue: Migrate All CI Runners to CIv2

## Summary
Migrate all remaining CIv1 CI runners to the CIv2 format to standardize our CI infrastructure and improve maintainability.

## Background
Currently, the repository has a mix of CIv1 and CIv2 runners:
- **CIv1 runners**: Use array-based labels like `["pipeline-perf", "P150", "in-service"]`
- **CIv2 runners**: Use string labels like `tt-ubuntu-2204-p150b-stable`

As of this analysis:
- **CIv2 runners**: ~25 instances across 20+ workflows (already migrated)
- **CIv1 runners**: ~24 instances across 6 workflows (need migration)

## Motivation
1. **Consistency**: Standardize all CI infrastructure on CIv2 format
2. **Maintainability**: Simpler label format reduces complexity
3. **Reliability**: CIv2 infrastructure provides better stability
4. **Future-proofing**: CIv1 may be deprecated in the future

## Scope

### Files Requiring Migration (6 files)

#### 1. `.github/workflows/fast-dispatch-frequent-tests-impl.yaml`
**Instances to migrate**: 2 (lines 43, 51)
- Current: `["pipeline-perf", "P150", "in-service"]`
- Target: `tt-ubuntu-2204-p150b-stable`

#### 2. `.github/workflows/galaxy-unit-tests-impl.yaml`
**Instances to migrate**: 2 (lines 37, 44)
- Current: `["arch-wormhole_b0", "${{ inputs.topology }}", "${{ inputs.extra-tag }}", "bare-metal", "pipeline-functional"]`
- Target: Need to determine appropriate CIv2 label based on topology input

#### 3. `.github/workflows/metal-run-microbenchmarks-impl.yaml`
**Instances to migrate**: 8 (lines 26, 32, 38, 44, 50, 56, 70, 76)
- `["N300", "pipeline-perf", "bare-metal", "in-service"]` → `tt-ubuntu-2204-n300-stable`
- `["arch-wormhole_b0", "pipeline-perf", "config-t3000", "in-service"]` → `tt-ubuntu-2204-t3000-stable`
- `["BH", "pipeline-perf", "P150", "in-service"]` → `tt-ubuntu-2204-p150b-stable`
- `["BH", "pipeline-perf", "in-service"]` → `tt-ubuntu-2204-p150b-stable`
- Note: One instance (line 64) already migrated to `tt-ubuntu-2204-p150b-viommu-stable`

#### 4. `.github/workflows/perf-models-impl.yaml`
**Instances to migrate**: 6 (lines 32-37)
- `["N300", "pipeline-perf", "bare-metal", "${{ inputs.extra-tag }}"]` → Dynamic CIv2 label with N300
- `["P150", "pipeline-functional", "cloud-virtual-machine", "${{ inputs.extra-tag }}"]` → Dynamic CIv2 label with P150
- `["P150", "pipeline-perf", "bare-metal", "${{ inputs.extra-tag }}"]` → Dynamic CIv2 label with P150

#### 5. `.github/workflows/publish-release-image.yaml`
**Instances to migrate**: 2 (lines 96, 101)
- `["cloud-virtual-machine", "N150", "in-service"]` → `tt-ubuntu-2204-n150-stable`
- `["cloud-virtual-machine", "N300", "in-service"]` → `tt-ubuntu-2204-n300-stable`

#### 6. `.github/workflows/tm-data-movement-perf-impl.yaml`
**Instances to migrate**: 2 (lines 26, 32)
- `["N300", "pipeline-perf", "bare-metal", "in-service"]` → `tt-ubuntu-2204-n300-stable`
- `["BH", "pipeline-perf", "in-service"]` → `tt-ubuntu-2204-p150b-stable`

## Proposed CIv1 → CIv2 Label Mappings

| CIv1 Label Pattern | CIv2 Label | Notes |
|-------------------|------------|-------|
| `["pipeline-perf", "P150", "in-service"]` | `tt-ubuntu-2204-p150b-stable` | Blackhole P150 perf runners |
| `["BH", "pipeline-perf", "in-service"]` | `tt-ubuntu-2204-p150b-stable` | Blackhole runners (general) |
| `["BH", "pipeline-perf", "P150", "in-service"]` | `tt-ubuntu-2204-p150b-stable` | Blackhole P150 specific |
| `["N300", "pipeline-perf", "bare-metal", "in-service"]` | `tt-ubuntu-2204-n300-stable` | Wormhole N300 bare-metal |
| `["arch-wormhole_b0", "pipeline-perf", "config-t3000", "in-service"]` | `tt-ubuntu-2204-t3000-stable` | T3000 configuration |
| `["cloud-virtual-machine", "N150", "in-service"]` | `tt-ubuntu-2204-n150-stable` | N150 VM runners |
| `["cloud-virtual-machine", "N300", "in-service"]` | `tt-ubuntu-2204-n300-stable` | N300 VM runners |

### Dynamic Labels (with inputs)
For workflows using dynamic tags like `${{ inputs.extra-tag }}` or `${{ inputs.topology }}`:
- Need to construct CIv2 labels dynamically using string interpolation
- Example: `${{ format('tt-ubuntu-2204-{0}-stable', inputs.runner-label) }}`

## Implementation Steps

1. **Phase 1: Simple Migrations** (Low Risk)
   - Migrate static label instances in:
     - `fast-dispatch-frequent-tests-impl.yaml`
     - `metal-run-microbenchmarks-impl.yaml`
     - `publish-release-image.yaml`
     - `tm-data-movement-perf-impl.yaml`

2. **Phase 2: Dynamic Label Migrations** (Medium Risk)
   - Handle workflows with dynamic inputs:
     - `galaxy-unit-tests-impl.yaml`
     - `perf-models-impl.yaml`

3. **Phase 3: Testing & Verification**
   - Ensure all migrated workflows run successfully
   - Monitor for any runner availability issues
   - Verify performance consistency

## Risks & Considerations

1. **Runner Availability**: Ensure CIv2 runners are available and configured for all required device types
2. **Performance Changes**: Monitor for any performance differences between CIv1 and CIv2 infrastructure
3. **Job Failures**: Some jobs may need adjustments if CIv2 environment differs from CIv1
4. **Rollback Plan**: Keep original configurations documented for quick rollback if needed

## Acceptance Criteria

- [ ] All 6 workflow files migrated to use CIv2 labels
- [ ] No CIv1 array-based labels with "in-service" or "pipeline-*" tags remain
- [ ] All migrated workflows pass successfully in CI
- [ ] Documentation updated (if applicable)
- [ ] No performance regressions observed

## Related Work

- Previous CIv2 migrations documented in changelogs:
  - v0.63.0: Initial CIv2 migrations for model jobs, N300 runners
  - v0.64.0: Moved APC and BHPC to vIOMMU in CIv2
  - v0.64.4: Updates for cleanup jobs on CIv2

## References

- CIv2 examples: See `clang-tidy-reusable.yaml`, `build-artifact.yaml`, `ops-post-commit.yaml`
- Changelog entries: `releases/changelog/changelog_release_v0.63.0.md`, `releases/changelog/changelog_release_v0.64.0.md`
