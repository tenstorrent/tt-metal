# Metalium 2.0 convolution migration: Fable review

Review target: rewritten branch stack based on `e701eff0689`, reviewed on 2026-08-26.

Review constraints:

- Preserve one source commit per migrated operation and isolate shared Metal infrastructure.
- Do not add circular buffers, backing buffers, staging copies, or equivalent data movement.
- Do not add kernel feature-selection macros; use named constexpr or NTTP arguments where specialization is needed.
- Do not change Quasar behavior or validate Quasar-specific paths as part of the standard migrations.
- Do not compromise device performance or numerical accuracy.

## Per-commit results

| Commit | Scope | Result | Notes |
| --- | --- | --- | --- |
| `d14d860ce8d` | Metalium migration infrastructure | Approved | Shared contracts and Pool-family kernels preserve topology and architecture-correct synchronization without new storage or copies. |
| `696fa693cb7` | Fold | Approved | Named bindings preserve the legacy ABI and optional aligned-path resources are constexpr-discarded. |
| `2c385314858` | ConvertToCHW | Approved | Fixed BF16 contract and Gen1 output self-loop preserve the existing kernel protocol. |
| `7645a320aec` | Rotate | Approved | Runtime rebinding, DFB placement, and compile-time specialization remain intact. |
| `4ec418e4d80` | GridSample | Approved | Bilinear and nearest paths preserve interleaved/sharded topology and cache-hit updates. |
| `71bdc7a3275` | ConvertToHWC | Finding resolved | Removed the migration-added standard-op Quasar rejection, keeping Quasar policy outside this migration. |
| `91166126288` | PaddedSlice | Approved | Quasar change is limited to an independent, semantically identical compatibility kernel copy. |
| `e17c4785729` | SliceWrite | Approved | All factories preserve geometry, synchronization, and cache-hit address/page-size rebinding. |
| `19db11d2194` | Upsample | Approved | DFB topology and accuracy configuration are preserved; reduced entry size removes only legacy over-allocation. |
| `0598e002486` | Halo | Approved | Existing alternating untilize DFBs and gather synchronization are preserved without a new data stage. |
| `51ec453975d` | Pool2D | Approved | Split-reader/MPWI topology, aliases, op-owned tensors, trace ownership, and diagnostics remain intact. |
| `c7c789dd784` | Conv3D | Approved | Multicast, reduction synchronization, optional bindings, and runtime varargs match legacy modes. |
| `511820a491c` | Conv2D | Approved | Gen1 placement, shared/borrowed DFB topology, cache behavior, and post-realization diagnostics are preserved. |
| final commit | Beads | Approved | Commit contains only `.beads` project and interaction data. |

## Actionable finding

### ConvertToHWC migration-added Quasar rejection

- Severity: medium
- Location: `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/convert_to_hwc_program_factory.cpp`
- Evidence: the originally reviewed ConvertToHWC commit added an architecture guard rejecting `tt::ARCH::QUASAR`; its parent had no equivalent policy.
- Resolution: removed the guard in the rewritten ConvertToHWC commit `71bdc7a3275`, retained the architecture-neutral typed schema, and left Quasar support or isolation to its independent path.
- Bead: `tt-metal-jit.1`
- Validation: Release `ttnn` target built successfully; all 265 focused ConvertToHWC tests passed locally with safe watcher and triage handling.

## Integrated assessment

After resolving the finding above, no actionable correctness, ABI, DFB synchronization, cache/trace, performance, accuracy, commit-isolation, macro, copy, or buffer issues remain. Static boundary and `git diff --check` audits passed. Previously completed WH/BH sanity, Conv+Pool L2 Nightly, ResNet50, and SDXL validation remain the device-level evidence for the otherwise byte-identical approved portions of the stack.
