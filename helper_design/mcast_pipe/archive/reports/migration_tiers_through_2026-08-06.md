# Archived: mcast_pipe rollout tiers through 2026-08-06

## Re-entry — next host/kernel migrations (2026-08-06)

Mode: `halt`, preapproved by the user for fully automated execution. Current
helper: `MCAST_PIPE_API_VERSION=10`. Gate 0 passed before production edits:
`McastHostFixture.*` 25/25 and the complete helper/device-wire suite 77/77
under `--dev`.

Mapping coverage is high for all three units. The previously unmapped sparse
binding now has `MM-SPARSE-IN0`: its exact without-`nnz` case passed from an
empty isolated cache with 0/29 JIT hits and produced sender hash
`14177997899853619887` and receiver hash `18131090411884538285`. No selected
unit has a low-confidence coverage gap.

### Tier 2 — Multicore TopK final readiness

Unit: `topk-multicore-final-readiness`; one factory and two kernels. Use the
existing no-handshake Counter signal channel with an adopted, host-zeroed
readiness semaphore. Preserve the operation-owned arrival counter and value /
index transfers. Validation: one fresh-cache W=8192 case, full
`TOPK-MULTICORE`, host/helper suites, rebuild, and per-kernel perf.

Outcome: **PASS — migrated at API v10** in the current atomic commit. The host
build passed; the exact W=8192, k=50 BFLOAT16_B case passed under `--dev` from
a fresh isolated cache with both JIT artifacts; `TOPK-MULTICORE` passed 14 with
12 expected BFLOAT8_B pad xfails; host/helper suites passed 25/25 and 77/77.
The exact-node device-kernel duration was 238,281 ns. Per-kernel delta is N/A:
there is no operation-matched pre-migration TopK bakeoff and each DM processor
envelope contains a second TopK kernel, so the generic F2 microbenchmark is not
a comparable baseline.

### Tier 3 — Sharded LayerNorm pre-allgather

Unit: `layernorm-sharded-pre-allgather`; one required binding and two kernels.
First split the shared reader compile-time argument builder and validate
`LN-PRE-ALLGATHER`, `LN-POST-ALLGATHER`, and `LN-SHARDED` with no behavior
change. Then migrate the one-stage `Mcast2D` and two-stage `Mcast1D` control
geometries using v10 handshaked Flag signal operations. The shared-builder
split is the load-bearing prerequisite and was landed separately as
`4ef7e9a57a6`.

Outcome: **PASS — migrated at API v10** in the current atomic commit. The
production build passed; the exact 8x4 BFLOAT8_B RMSNorm node passed under
`--dev` from a fresh isolated cache with sender and both receiver-variant JIT
artifacts; the mapped inventories passed 126 pre-allgather, 136
post-allgather, and 208 plain sharded cases. Host/helper suites passed 28/28
and 77/77. The exact-node device-kernel median was 2,563.5 ns across four
calls. Per-kernel delta is N/A because no operation-matched pre-migration
profile exists and the DM envelope contains other LayerNorm kernels.

### Tier 0 re-entry, then Tier 4 — Matmul in0 interleaved

API-007 is an accepted caller-facing extension: Flag `send_signal(value)` and
value-returning `receive_signal()`. Materialize and unit-test it through
`tune-dm-helper`, bump the API version, then re-enter this rollout and remigrate
all 13 kernels and 12 current host bindings as Tier 0 before net-new Matmul.

The Tier-4 Matmul unit contains both kernels and all five required bindings:
1D legacy/descriptor, 2D legacy/descriptor, and sparse 1D. Validation includes
the full `MM-SPARSE-IN0`, `MM-IN0-INTERLEAVED`, and
`MM-BLOCK-SHARDED-HYBRID` guard inventory plus sparse-path perf. The in0 helper
wire is emitted only in interleaved branches; block-sharded ABI remains raw.

Failure policy: the first failing unit stops the rollout with its entire diff
left in the tree and its ledger rows still pending. Device tests and tiers run
strictly sequentially.

---

# Historical mcast_pipe API v9 host-integration tiers — 2026-07-30

## Re-entry — Conv2D width-sharded activation (2026-08-03)

Mode: `run-all`. Selected unit: `conv2d-activation-width-sharded` (Tier 6, refactor-medium).

Binding: `activation-mcast:conv2d-width-sharded:rotating-rect`

Atomic scope:

- `activation_reader_width_sharded.cpp` sender and receiver faces;
- `conv2d_op_width_sharded_program_factory.cpp` host wire and CT/RT packing;
- kernel and host-binding ledger rows.

API-v9 formulation: one rotating, handshaked Flag `Mcast2D` over the dense reader bounding rectangle,
with adopted existing semaphores and `num_active=max(input_cores,output_cores)-1`. The host emits the
full rectangle's sender list; the kernel consumes the first `num_input_cores` coordinates used by the
actual round-robin loop. No helper change or API bump.

Gate evidence: helper 72/72. Unchanged exact BF16/BF16, filter-3, tile-output case passed under
`--dev` at PCC 0.9999565 from an isolated cache, with `activation_reader_width_sharded` confirmed.

Validation: rebuild host code, rerun that exact node from a fresh isolated cache, then run all 64
`test_conv_features` width-sharded selections (historical expected result: 48 pass / 16 legitimate
row-major+bfloat8 skips), followed by the mapped width-sharded DRAM-config route.

Rollback: run-all restores kernel, factory, and ledger together and records quarantine on any failure.

Outcome: **migrated at API v9** in `fe866a1d0c4`. Build passed; the exact fresh-cache `--dev` case
passed at PCC 0.999956503; all 48 runnable feature cases passed with 16 legitimate skips; the mapped
DRAM-config route passed at PCC 0.998234911; post-integration helper coverage passed 72/72. No
rollback or quarantine was needed.

## Re-entry — sort single-row control channel (2026-08-03)

Mode: `halt`, approved through the remaining checkpoints. This re-entry adds one Tier-5 atomic unit
after the four completed host-integration tiers; it does not reopen their historical `run-all` mode.

Unit: `sort-single-row-control`

Binding: `sort:single-row-multi-core:control`

Atomic scope:

- coordinator and reader kernels become the two faces of one no-handshake Counter control Pipe;
- `sort_program_factory.cpp` owns one `Mcast2D` wire over the full worker grid plus coordinator;
- the reader-ready and writer-done counters stay explicit operation protocol;
- the writer remains helper-neutral, but its obsolete control-doorbell runtime word is removed in the
  same host/kernel ABI change;
- all three kernel ledger rows and the new host-binding row move together.

Gate-0 evidence: the unchanged exact
`test_sort_long_tensor[shape=[1, 524288]-dim=-1-descending=False]` case passed under `--dev` from an
isolated JIT cache, which contained `coordinator_single_row_multi_core`,
`reader_single_row_multi_core`, and `writer_single_row_multi_core` artifacts. Step-G helper intake is
72/72, including four control-only Counter cells at 1×2/1×8 and 2/32 back-to-back signals.

Validation: rebuild host code, run the same exact compile-focused node from a fresh isolated cache,
then run both `test_sort_multi_row_multi_core_no_deadlock` descending values. Re-run the complete
helper suite after production integration.

Risk: the host helper derives the multicast fan-out from a rectangle. This factory is selected only
when `Wt` exceeds the hybrid path's `total_cores * 128` capacity, which forces the full worker set;
the rectangle is therefore dense and its EXCLUDE-source fan-out equals `core_range.num_cores()`.

Outcome: **migrated at API v9** in `7337302b564`. Host build passed; exact fresh-cache `--dev` route
passed with all three JIT artifacts; Ht=2 deadlock pair passed 2/2; full long-tensor inventory passed
7/7; post-integration helper suite passed 72/72. Writer remains helper-neutral in the ledger.

Historical initial entry state: the first ten production kernels were current at API v9 while their
paired host-helper rollout was still incomplete. Tiers 1–4 subsequently completed; Tier 5 above is
the 2026-08-03 sort re-entry.

Mode: `run-all`. A failing unit is restored and quarantined without stopping
the remaining units.

## Mapping coverage

All participating kernels reuse device-verified operation inventories from
`test_map.json`; no kernel is newly unmapped:

- Matmul in1: `MM-IN1-ALL` and `MM-IN1-RECEIVER-2D`
  (302 passed, 188 expected skips).
- Conv2d height-sharded weights: `CONV-HEIGHT`
  (49 passed, 16 expected skips) plus 14 DRAM regressions.
- Conv2d block-sharded weights: `CONV-BLOCK`
  (49 passed, 16 expected skips) plus 14 DRAM regressions.
- GroupNorm v2: `GN-SHARDED-PARAMETERIZED`, separately covering legacy and
  Welford sender/receiver paths plus fixed/default-routing nodes.

The helper intake is green: 72/72 `test_mcast_pipe.py` cases passed on 2026-08-03; the prior
`McastHostFixture` inventory remains 19/19 from 2026-07-30.

## Tier 1 — Conv2d single-sender rectangle

Unit: `conv2d-weights-single-sender-rect`

Binding:
`weights-mcast:conv2d-sharded:single-sender-rect`

Why first: one factory branch, one fixed sender, one rectangle, and the
materialized `Mcast2D` directly expresses the existing split between the full
geometric fan-out and the active receiver ACK count.

Atomic scope:

- 1D weights sender and receiver kernels;
- `conv2d_op_sharded_program_factory.cpp`;
- the height-sharded/default writer dispatch;
- its descriptor-time RT packing and buffer bindings;
- both kernel and host-binding ledger rows.

Validation: one exact `CONV-HEIGHT` compile-focused case first, then the
complete mapped `CONV-HEIGHT` inventory and the shared DRAM regression slice.

Risk: noop cores receive the multicast but intentionally do not ACK. The host
wire must preserve `num_active = total_active_num_cores - 1` while retaining
the full rectangle as the data fan-out.

## Tier 2 — Conv2d fixed-line weights

Unit: `conv2d-weights-fixed-line`

Binding:
`weights-mcast:conv2d-sharded:fixed-line`

Atomic scope:

- 2D weights sender and receiver kernels;
- both row and column branches in `conv2d_op_sharded_program_factory.cpp`;
- split-reader pad-out runtime state;
- both kernel and host-binding ledger rows.

Validation: one exact `CONV-BLOCK` compile-focused case first, explicit JIT
confirmation for both row and column routes, then the complete mapped
`CONV-BLOCK` inventory and shared DRAM regression slice.

Gap/risk: `Mcast1D` requires a zero-anchored dense rectangle. The factory
appears to assume that topology but does not state the invariant explicitly.
If an accepted configuration violates it, quarantine this unit and continue.
Split-reader semaphores remain outside the multicast helper.

## Tier 3 — Matmul in1, all live emitters

Unit: `matmul-in1-mcast-padding-host`

Required bindings:

- `matmul-in1-mcast:reuse-2d:legacy`
- `matmul-in1-mcast:reuse-2d:descriptor`
- `matmul-in1-mcast:reuse-1d:mcast-in1:legacy`
- `matmul-in1-mcast:reuse-1d:mcast-in1:descriptor`

Atomic scope:

- in1 sender/writer and receiver/writer kernels;
- legacy and descriptor emitters in both production 1D and 2D factories;
- runtime override and descriptor tensor-reference offsets;
- all six kernel/host ledger rows.

Validation: one exact 1D and one exact 2D compile-focused parameter, including
a non-zero sub-device origin, then `MM-IN1-ALL` and the exact 2D receiver
inventory.

Risks:

- receiver RT grows by two words, shifting output and fused-op slots;
- legacy and descriptor paths duplicate the shared ABI and must move together;
- 2D split receivers compile the same source with two NoCs;
- sub-device offsets rule out direct `Mcast1D`; use one `Mcast2D` per line.

The MCAST_IN0 and sparse bindings are `not_applicable`: those builds compile
the shared sender source with `SKIP_MCAST` and instantiate no in1 receiver.

## Tier 4 — GroupNorm v2 multi-rectangle

Unit: `groupnorm-sharded-v2-mcast-host`

Required bindings:

- `groupnorm-v2:sharded:legacy:mcast`
- `groupnorm-v2:sharded:legacy:degenerate`
- `groupnorm-v2:sharded:welford:mcast`
- `groupnorm-v2:sharded:welford:degenerate`

Atomic scope:

- legacy and Welford sender/receiver kernels;
- the shared sharded GroupNorm factory;
- mcast and group-size-one dispatches;
- variable sender RT packing and cache-rebuild path;
- all eight kernel/host ledger rows.

Validation: one exact legacy parameter and one exact Welford parameter first,
then the complete mapped legacy/Welford inventory and fixed/default-routing
nodes.

Gaps/risks:

- the current sender wire contains up to three optional rectangles and a
  variable gather-coordinate tail, while `Mcast2D` emits one fixed RT block;
- optionality varies per sender and must remain RT-controlled;
- offset-grid and NoC1 cases are mandatory;
- three dead per-rectangle count words can be removed, but the raw pre-gather
  ACK gate and gather-coordinate tail must remain.

If the materialized host helper cannot compose the optional rectangles without
changing observable behavior, quarantine this unit and continue; helper design
belongs back in `tune-dm-helper`.

## Existing deferred kernel backlog

The remaining deferred kernel rows stay outside their completed atomic units. This sort re-entry
changed two kernel rows from pending to migrated, retained its helper-neutral writer as deferred,
and did not change the helper API version.
