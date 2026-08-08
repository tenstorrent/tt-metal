# CIv2 Migration Documentation

This directory contains documentation for migrating CI runners from CIv1 to CIv2 format.

## Files Created

### 1. `MIGRATION_ISSUE_CIv2_RUNNERS.md`
**Purpose**: Comprehensive technical migration guide

**Contents**:
- Complete analysis of current state (25 CIv2 vs 24 CIv1 runners)
- Detailed scope with exact file names and line numbers
- CIv1 → CIv2 label mapping table
- Phased implementation plan with risk assessment
- Acceptance criteria and references

**Use case**: Technical reference for engineers performing the migration

### 2. `ISSUE_TEMPLATE_CIv2_MIGRATION.md`
**Purpose**: Ready-to-use GitHub issue template

**Contents**:
- Executive summary
- Actionable checklist for all 24 instances
- Label mapping table
- Implementation phases
- Acceptance criteria

**Use case**: Copy-paste into GitHub to create tracking issue

## How to Create the GitHub Issue

Since GitHub Copilot cannot directly create issues, follow these steps:

1. Go to https://github.com/tenstorrent/tt-metal/issues/new
2. Click "New Issue"
3. Copy the content from `ISSUE_TEMPLATE_CIv2_MIGRATION.md`
4. Paste into the issue description
5. Set title: `[CI] Migrate all CI runners to CIv2`
6. Add labels: `ci-cd`, `infrastructure`, `enhancement`
7. Click "Submit new issue"

## Quick Stats

- **Total CIv1 instances to migrate**: 24
- **Workflow files affected**: 6
- **Complexity levels**: 
  - Low: 18 instances (static labels)
  - Medium: 6 instances (dynamic labels with inputs)

## Migration Order (Recommended)

1. **Phase 1** (Low risk): Migrate static labels
   - `fast-dispatch-frequent-tests-impl.yaml` (2 instances)
   - `metal-run-microbenchmarks-impl.yaml` (8 instances)  
   - `publish-release-image.yaml` (2 instances)
   - `tm-data-movement-perf-impl.yaml` (2 instances)

2. **Phase 2** (Medium risk): Handle dynamic labels
   - `galaxy-unit-tests-impl.yaml` (2 instances)
   - `perf-models-impl.yaml` (6 instances)

3. **Phase 3**: Testing and validation
   - Verify workflows pass
   - Monitor performance
   - Check runner availability

## Key Label Mappings

| CIv1 | CIv2 | Device |
|------|------|--------|
| `["pipeline-perf", "P150", "in-service"]` | `tt-ubuntu-2204-p150b-stable` | Blackhole P150 |
| `["N300", "pipeline-perf", "bare-metal", "in-service"]` | `tt-ubuntu-2204-n300-stable` | Wormhole N300 |
| `["arch-wormhole_b0", "pipeline-perf", "config-t3000", "in-service"]` | `tt-ubuntu-2204-t3000-stable` | T3000 |
| `["cloud-virtual-machine", "N150", "in-service"]` | `tt-ubuntu-2204-n150-stable` | N150 VM |

## Questions?

Refer to:
- `MIGRATION_ISSUE_CIv2_RUNNERS.md` for technical details
- Changelog entries in `releases/changelog/` for historical context
- Existing CIv2 workflows for examples: `clang-tidy-reusable.yaml`, `build-artifact.yaml`
