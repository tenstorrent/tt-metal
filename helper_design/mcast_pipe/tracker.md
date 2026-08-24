# Multicast family and chain-forwarding tracker

Plan: [`plan.md`](plan.md)
Status: Stages 1–4 complete; Stage 5 pending
Last updated: 2026-08-24

This is the execution record for the `McastFamily`/`McastGroup`, exact
multi-rectangle, GroupNorm, chain-forwarding, and Conv3D work. Keep results and
scope changes here as the implementation progresses.

Legend: `[ ]` pending, `[~]` in progress, `[x]` complete, `[!]` blocked or
failed.

## Fixed decisions

- [x] One `McastFamily` represents one semantic multicast stream.
- [x] A family contains independent groups with exact static receiver sets and
  fixed or rotating sender schedules.
- [x] Cross-group disjointness is checked over receiver sets plus every
  scheduled sender; sender/receiver overlap within one group is valid.
- [x] Chain order is helper-derived logical row-major `(y, x)`; there is no
  explicit chain-order API.
- [x] Receiver sets dense in virtual wire coordinates always use hardware
  multicast; logical density across a physical disjoint worker region is
  decomposed or chained.
- [x] Irregular sets default to exact N-rectangle multicast; chain forwarding
  requires an explicit host flag.
- [x] A single logical send covers all rectangles and each receiver calls
  receive once.
- [x] Payload receivers always pass destination L1 and byte count.
- [x] Payload and Flag/Counter signal protocols are covered by both transports.
- [x] Rotating multi-rectangle multicast is supported; rotating irregular chain
  forwarding is rejected in phase one.
- [x] `Mcast1D` and `Mcast2D` remain convenience constructors.
- [x] GroupNorm is the multi-rectangle POC and must pass before chain work
  starts; Conv3D is the chain-forwarding POC.
- [x] API source migration is limited to helper tests, GroupNorm, and Conv3D;
  the full historical helper fleet is not reconciled in this rollout.

## Stage 1 — family/group and multi-rectangle vertical slice

Stage started from commit `b69341112dd`.

### Host model

- [x] Add `McastGroup` with exact receiver set, sender schedule, and explicit
  `use_chain_forwarding` flag defaulting to false.
- [x] Add `McastFamily` ownership of protocol, semaphores, coordinate conversion,
  validation, and serialization.
- [x] Validate disjoint group footprints, including rotating sender schedules.
- [x] Validate wire compatibility and rotation span across groups.
- [x] Implement per-core group lookup and role selection.
- [x] Implement exact deterministic rectangle decomposition for 1, 2, 3, and N
  rectangles.
- [x] Rebuild `Mcast1D` as one group per row/column.
- [x] Rebuild `Mcast2D` as a one-group family.

### Wire and device pipe

- [x] Extend `McastArgs` without breaking ordinary dense use or operation-tail
  offset chaining.
- [x] Create the selected sender/receiver pipe from family runtime arguments.
- [x] Change payload receiver API to require `dst_l1` and `size_bytes`.
- [x] Migrate helper-test payload `receive()` call sites; defer production
  source migration to GroupNorm and Conv3D stages only.
- [x] Implement one readiness wait for one logical multi-rectangle send.
- [x] Send the payload to all exact rectangles and complete/fence the source
  once.
- [x] Implement multi-rectangle `send_signal()`/`receive_signal()` for Flag and
  Counter protocols.

### Stage 1 validation gate

- [x] Host tests: group validation, decomposition, wrappers, offsets, NoCs, and
  coordinate virtualization.
- [x] Device tests: payload and signals over 1, 2, 3, and N rectangles.
- [x] Device tests: concurrent disjoint groups with unequal rectangle counts.
- [x] Device tests: rotating multi-rectangle groups.
- [x] Device tests: repeated sends, dynamic pointers/sizes, aliases, and source
  lifetime.
- [x] `./build_metal.sh` passes.
- [x] Applicable helper host/device and focused source-audit suites pass.

Gate: do not integrate GroupNorm until every applicable Stage 1 item passes.

## Stage 2 — GroupNorm exact multi-rectangle proof

Stage started from commit `afaff417f1c`.

### Integration

- [x] Model the GroupNorm statistics stream as one family across exact groups.
- [x] Set `use_chain_forwarding == false`.
- [x] Replace three `Mcast2D` argument blocks with one family argument block.
- [x] Replace three sender pipes/sends with one selected pipe/send.
- [x] Remove fake singleton padding for absent first/last rectangles.
- [x] Pass global-statistics destination and exact byte count to `receive()`.
- [x] Preserve the early/manual readiness ACK protecting remote Welford reads.
- [x] Cover legacy and Welford routes.

### Stage 2 validation gate

- [x] One legacy parametrization passes through the safe pytest wrapper.
- [x] One Welford parametrization passes through the safe pytest wrapper.
- [x] Focused GroupNorm POC tests pass.
- [x] Full GroupNorm unit suite passes.
- [x] Full GroupNorm nightly suite passes.
- [x] Relevant helper regressions and source audits pass.
- [x] GroupNorm performance baseline is recorded.

Gate: do not begin Stage 3 until Stage 2 is fully green.

## Stage 3 — row-major chain transport

Stage started from commit `c6a19c93f0c`.

### Implementation

- [x] Derive logical row-major topology with the active sender as head.
- [x] Serialize/select predecessor and successor without exposing them to the
  operation.
- [x] Implement head payload injection with per-hop readiness.
- [x] Implement middle receive-then-forward from caller-supplied destination.
- [x] Implement tail receive-only behavior.
- [x] Relay the same Flag value through middle nodes.
- [x] Relay one Counter event through middle nodes.
- [x] Keep dense groups on hardware multicast when chain is enabled.
- [x] Reject rotating irregular chain groups at host construction.

### Stage 3 validation gate

- [x] Two-core payload chain passes.
- [x] Multi-hop payload chain passes.
- [x] Repeated sends with changing destination and size pass.
- [x] Flag and Counter chain signal tests pass.
- [x] Dense-fallback test passes.
- [x] Rotating-irregular rejection test passes.
- [x] Full helper host/device suites and build pass.

Gate: do not integrate Conv3D until every applicable Stage 3 item passes.

## Stage 4 — Conv3D dense/chain proof

Stage started from commit `e4828229deb`.

### Integration

- [x] Model the weights stream as one family over exact work groups.
- [x] Enable chain forwarding for the family/groups.
- [x] Confirm dense groups select multicast and irregular groups select chain.
- [x] Remove operation-owned predecessor/successor runtime arguments.
- [x] Consolidate separate Chain/Mcast roles where replaced by the helper.
- [x] Remove passive bounding-box participants from migrated paths.
- [x] Receive into the weights circular-buffer write pointer using the exact
  weight-block byte count.

### Stage 4 validation gate

- [x] One dense Conv3D parametrization passes through the safe pytest wrapper.
- [x] One irregular Conv3D parametrization passes through the safe pytest
  wrapper.
- [x] Focused Conv3D POC tests pass.
- [x] Full Conv3D unit suite passes.
- [x] Full Conv3D nightly suite passes.
- [x] Relevant helper regressions and source audits pass.
- [x] Dense and irregular Conv3D performance is recorded and accepted.

## Stage 5 — combined closeout

- [ ] Run the complete helper host suite.
- [ ] Run the complete helper device/wire suite sequentially.
- [ ] Run final GroupNorm unit and nightly suites.
- [ ] Run final Conv3D unit and nightly suites.
- [ ] Confirm fresh JIT/source-audit coverage for all changed kernel faces.
- [ ] Confirm no migrated operation still owns rectangle decomposition or chain
  topology.
- [ ] Update helper API/version documentation and migration records.
- [ ] Record final performance comparisons and any accepted variance.
- [ ] Mark this tracker complete with final commit and evidence links.

## Test evidence

Add one row per meaningful command. Do not mark a gate complete from an
unrecorded partial run.

| Date | Stage | Command/test | Result | Notes or artifact |
|---|---|---|---|---|
| 2026-08-24 | 1 | `build_Release/test/ttnn/unit_tests_ttnn --gtest_filter='McastHostFixture.*:GroupNormMcastGeometry.*'` | PASS | 45/45 focused host and geometry tests. |
| 2026-08-24 | 1 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe_source_audit.py -k 'mcast_args_owns_its_compile_time_presence_tag or mcast_args_has_one_template_owned_runtime_base' -q` | PASS | 2/2 applicable helper-wire source audits. |
| 2026-08-24 | 1 | `./build_metal.sh` | PASS | Release host build and Python binding install. |
| 2026-08-24 | 1 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py --dev -q` | PASS | 91/91 helper device tests, including the complete family matrix. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe_source_audit.py -k 'groupnorm_uses_one_family_wire' -q` | PASS | Focused source audit confirms one family wire across all GroupNorm host and kernel routes. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_group_norm.py --dev -k 'test_group_norm_with_block_sharded_v2_8x4_grid and legacy and 1280' -q` | PASS | Focused legacy compile and device case. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_group_norm.py --dev -k 'test_group_norm_with_block_sharded_v2_8x4_grid and welford and 1280' -q` | PASS | Focused Welford compile and device case. |
| 2026-08-24 | 2 | `./build_metal.sh` | PASS | Full host build after all GroupNorm factory changes. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_group_norm.py -q` | PASS | 345 passed, 10 expected platform skips. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/fused/test_group_norm_DRAM.py -q` | PASS | 181 passed, 5 skipped, 1 expected xfail. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/fused/test_group_norm.py -q` | PASS | 203 passed, 6 platform skips. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/fused/test_group_norm_DRAM.py -q` | PASS | 257 passed, 111 unsupported/platform skips. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations_compute_only/fused/test_group_norm.py -q` | PASS | 8/8 compute-only validation tests. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/perf_tests/operations/mcast/test_groupnorm_chain_vs_rectangles.py -q` | PASS | Exact five-group POC uses rectangle counts `[2,3,3,3,2]`; 7-trial median 22,087 ns, 0.15% standard deviation. |
| 2026-08-24 | 2 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py --dev -q` | PASS | 91/91 helper device regression tests after GroupNorm integration. |
| 2026-08-24 | 3 | `./build_metal.sh` | PASS | Full host build with row-major chain serialization and runtime pipe selection. |
| 2026-08-24 | 3 | `build_Release/test/ttnn/unit_tests_ttnn --gtest_filter='McastHostFixture.*:GroupNormMcastGeometry.*'` | PASS | 48/48 host/helper geometry tests, including row-major topology, dense fallback, and rejection cases. |
| 2026-08-24 | 3 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py --dev -k 'payload_chain or multi_rectangle_control_signal or repeated_dynamic_destination_and_size or dense_chain_request' -q` | PASS | 10/10 focused payload, Flag/Counter, dynamic-size, and dense-fallback cases. |
| 2026-08-24 | 3 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py --dev -k 'multi_hop_payload_chain or selects_dense_multicast' -q` | PASS | 5/5 NoC0/NoC1 Flag/Counter multi-hop and mixed per-group transport cases. |
| 2026-08-24 | 3 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py --dev -q` | PASS | Full helper device/wire suite: 101/101 passed. |
| 2026-08-24 | 3 | `GN_CHAIN_TRIALS=1 scripts/run_safe_pytest.sh tests/ttnn/perf_tests/operations/mcast/test_groupnorm_chain_vs_rectangles.py -q` | PASS | Exact GroupNorm multi-rectangle regression remains green after unified pipe transport selection. |
| 2026-08-24 | 3 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe_source_audit.py -k 'family_chain_topology or mcast_args_owns_its_compile_time_presence_tag or mcast_args_has_one_template_owned_runtime_base or groupnorm_uses_one_family_wire' -q` | PASS | 4/4 focused helper ownership and GroupNorm source audits. |
| 2026-08-24 | 4 | `./build_metal.sh` | PASS | Full host build after Conv3D family integration and virtual-wire rectangle decomposition. |
| 2026-08-24 | 4 | `build_Release/test/ttnn/unit_tests_ttnn --gtest_filter='McastHostFixture.*:GroupNormMcastGeometry.*'` | PASS | 49/49 host/helper geometry tests; includes Blackhole disjoint-worker transport selection. |
| 2026-08-24 | 4 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py --dev -q` | PASS | Full helper device/wire suite: 101/101, including rotating and fixed 8-wide lines across disjoint worker columns. |
| 2026-08-24 | 4 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv3d.py -k 'k111_s111_g1_zeros_c64_c48' -q` | PASS | Production-shaped Conv3D case that previously hung now passes; PCC 0.999991951. |
| 2026-08-24 | 4 | `scripts/run_safe_pytest.sh 'tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_weight_mcast_family[dense_weight_group]' -q` | PASS | Physically dense five-core weight group selects hardware multicast; PCC 0.999991111. |
| 2026-08-24 | 4 | `scripts/run_safe_pytest.sh 'tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_weight_mcast_family[irregular_weight_groups]' -q` | PASS | Ragged 10x9 work geometry selects helper-owned chains; PCC 0.999991414. |
| 2026-08-24 | 4 | `CONV3D_MCAST_TRIALS=7 scripts/run_safe_pytest.sh tests/ttnn/perf_tests/operations/mcast/test_conv3d_chain_vs_three_rectangles.py -q` | PASS | Focused irregular Conv3D POC correctness passed; seven-trial chain median 12,206 ns with 0.63% spread. |
| 2026-08-24 | 4 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv3d.py -q` | PASS | Full unit suite: 29 passed and 1 expected Blackhole skip. |
| 2026-08-24 | 4 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv3d.py -k 'k333_s111_g1_zeros_c64_c64' -q` | PASS | Production-shaped case reproducing the original long-chain hang now passes; PCC 0.999991419. Root cause was empty-spatial cores entering weight transport despite not belonging to the exact receiver family. |
| 2026-08-24 | 4 | `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py -q` | PASS | Full nightly shape sweep: 1,606 passed, 5 skipped, and 2 expected xfails. |
| 2026-08-24 | 4 | `build_Release/test/ttnn/unit_tests_ttnn --gtest_filter='McastHostFixture.*:GroupNormMcastGeometry.*'` | PASS | Updated host/helper gate: 50/50, including two exact 54-core ragged row-major groups. |
| 2026-08-24 | 4 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py --dev -q` | PASS | Updated full device/wire gate: 102/102, including a 54-core chain carrying 110,592 bytes. |
| 2026-08-24 | 4 | `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe_source_audit.py -q` | PASS | Full source audit: 35/35; validates one Conv3D family/wire and helper-owned exact-group transport. |
| 2026-08-24 | 4 | `python -m tracy -r -m pytest 'tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_weight_mcast_family[dense_weight_group]' -q` | PASS | Dense hardware-multicast case: Conv3D kernel duration 3,030 ns; report `generated/profiler/reports/2026_08_24_01_59_55/ops_perf_results_2026_08_24_01_59_55.csv`. |
| 2026-08-24 | 4 | `python -m tracy -r -m pytest 'tests/ttnn/unit_tests/operations/conv/test_conv3d.py::test_conv3d_weight_mcast_family[irregular_weight_groups]' -q` | PASS | Irregular helper-chain case: Conv3D kernel duration 10,864 ns, consistent with the focused seven-trial 12,206 ns median; report `generated/profiler/reports/2026_08_24_02_00_17/ops_perf_results_2026_08_24_02_00_17.csv`. |

## Decisions and scope changes

Record any decision that changes an invariant, ABI, transport rule, operation
scope, or gate before implementing it.

| Date | Decision/change | Reason | Approved by |
|---|---|---|---|
| 2026-08-23 | Initial design decisions frozen in the linked plan. | Planning discussion. | User |
| 2026-08-23 | Limit API source migration to helper tests, GroupNorm, and Conv3D; do not reconcile the full rollout ledger. | Explicit scope clarification. | User |
| 2026-08-24 | Evaluate multicast density and exact rectangle decomposition in virtual wire coordinates after logical-to-worker conversion; retain logical row-major chain order. | Blackhole logical neighbors may cross harvested/disjoint physical worker columns, so their logical bounding box is not a legal single hardware multicast rectangle. | Plan clarification from the user's “dense rect in disjointgroup” requirement, confirmed by host/device tests. |

## Blockers

| Date | Stage | Blocker | Required resolution | Status |
|---|---|---|---|---|
| | | | | |

## Primary implementation surfaces

- `ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp`
- `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`
- `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl`
- GroupNorm host factory and sender/receiver kernels
- Conv3D host factory and weight writer/receiver kernels
- helper host, device/wire, source-audit, GroupNorm, and Conv3D tests
