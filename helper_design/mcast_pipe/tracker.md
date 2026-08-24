# Multicast family and chain-forwarding tracker

Plan: [`plan.md`](plan.md)
Status: Stage 1 complete; Stage 2 pending
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
- [x] Dense receiver sets always use hardware multicast.
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

### Integration

- [ ] Model the GroupNorm statistics stream as one family across exact groups.
- [ ] Set `use_chain_forwarding == false`.
- [ ] Replace three `Mcast2D` argument blocks with one family argument block.
- [ ] Replace three sender pipes/sends with one selected pipe/send.
- [ ] Remove fake singleton padding for absent first/last rectangles.
- [ ] Pass global-statistics destination and exact byte count to `receive()`.
- [ ] Preserve the early/manual readiness ACK protecting remote Welford reads.
- [ ] Cover legacy and Welford routes.

### Stage 2 validation gate

- [ ] One legacy parametrization passes through the safe pytest wrapper.
- [ ] One Welford parametrization passes through the safe pytest wrapper.
- [ ] Focused GroupNorm POC tests pass.
- [ ] Full GroupNorm unit suite passes.
- [ ] Full GroupNorm nightly suite passes.
- [ ] Relevant helper regressions and source audits pass.
- [ ] GroupNorm performance is recorded and accepted.

Gate: do not begin Stage 3 until Stage 2 is fully green.

## Stage 3 — row-major chain transport

### Implementation

- [ ] Derive logical row-major topology with the active sender as head.
- [ ] Serialize/select predecessor and successor without exposing them to the
  operation.
- [ ] Implement head payload injection with per-hop readiness.
- [ ] Implement middle receive-then-forward from caller-supplied destination.
- [ ] Implement tail receive-only behavior.
- [ ] Relay the same Flag value through middle nodes.
- [ ] Relay one Counter event through middle nodes.
- [ ] Keep dense groups on hardware multicast when chain is enabled.
- [ ] Reject rotating irregular chain groups at host construction.

### Stage 3 validation gate

- [ ] Two-core payload chain passes.
- [ ] Multi-hop payload chain passes.
- [ ] Repeated sends with changing destination and size pass.
- [ ] Flag and Counter chain signal tests pass.
- [ ] Dense-fallback test passes.
- [ ] Rotating-irregular rejection test passes.
- [ ] Full helper host/device suites and build pass.

Gate: do not integrate Conv3D until every applicable Stage 3 item passes.

## Stage 4 — Conv3D dense/chain proof

### Integration

- [ ] Model the weights stream as one family over exact work groups.
- [ ] Enable chain forwarding for the family/groups.
- [ ] Confirm dense groups select multicast and irregular groups select chain.
- [ ] Remove operation-owned predecessor/successor runtime arguments.
- [ ] Consolidate separate Chain/Mcast roles where replaced by the helper.
- [ ] Remove passive bounding-box participants from migrated paths.
- [ ] Receive into the weights circular-buffer write pointer using the exact
  weight-block byte count.

### Stage 4 validation gate

- [ ] One dense Conv3D parametrization passes through the safe pytest wrapper.
- [ ] One irregular Conv3D parametrization passes through the safe pytest
  wrapper.
- [ ] Focused Conv3D POC tests pass.
- [ ] Full Conv3D unit suite passes.
- [ ] Full Conv3D nightly suite passes.
- [ ] Relevant helper regressions and source audits pass.
- [ ] Dense and irregular Conv3D performance is recorded and accepted.

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

## Decisions and scope changes

Record any decision that changes an invariant, ABI, transport rule, operation
scope, or gate before implementing it.

| Date | Decision/change | Reason | Approved by |
|---|---|---|---|
| 2026-08-23 | Initial design decisions frozen in the linked plan. | Planning discussion. | User |
| 2026-08-23 | Limit API source migration to helper tests, GroupNorm, and Conv3D; do not reconcile the full rollout ledger. | Explicit scope clarification. | User |

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
