# Fabric teardown Watcher incident

## Status

Stage 04 stopped after correct four-device collective results because the
Blackhole fabric router did not satisfy the active-ERISC kernel-handoff
contract. This is an infrastructure defect exposed by the multichip decoder,
not a demonstrated Gemma arithmetic defect. The local TT-Metal repair passes
the model-free control and original EP Watcher probe. No stage pass is implied
until the decoder Watcher suite, capability tests, performance selection, and
independent stage review also pass.

## Symptom and immediate cause

The Watcher-enabled EP probe completed all 16 numerical checks, wrote its JSON,
and printed `EP_PROBE_PASS`. During process teardown, device 0 active Ethernet
core `(28,25)` stopped in subordinate firmware at `NKFW`. The saved Watcher
assert reports that write-capable NoC packet tags were nonzero when
`fabric_erisc_router.cpp` returned. The host then waited 20 seconds for the
core's base-firmware heartbeat, observed no change, and aborted with exit 134.

The active-ERISC firmware checks that NoC reads, writes, and atomics are drained
and that packet tags are cleared before accepting another kernel. The router's
teardown drained write and atomic traffic and synchronized the two local ERISCs,
but did not clear the transaction-ID packet-tag registers before publishing
`TERMINATED`. Fabric traffic deliberately programs explicit transaction IDs,
so a drained command buffer can still retain a nonzero tag.

## Why it appeared in Stage 04

The failure was not expected in Stages 01 through 03: those stages use one
device and never initialize the four-device fabric router. Several early Stage
04 tests ran without Watcher, where the kernel-handoff assertion is not active;
their successful numerical outputs therefore did not validate teardown state.

The first Watcher EP attempt stopped before execution because Watcher
instrumentation made the active-Ethernet binary exceed its configuration
buffer. `TT_METAL_WATCHER_NOINLINE=1` reduced the binary enough to run. That was
the first test to combine all of the following:

1. real collective traffic that programs fabric-router transaction IDs;
2. normal fabric shutdown and active-ERISC kernel return; and
3. Watcher's end-of-kernel packet-tag assertion.

The assert is therefore late by lifecycle, not by workload correctness: it is
checked after correct outputs have already been produced. A no-payload fabric
open/close smoke also passes because it need not leave the same programmed
transaction tag.

## Implementation-cause discrimination

The Gemma expert-parallel implementation is not necessary to reproduce the
failure. `tests/probe_multichip_collective.py` opens the same 1x4 `FABRIC_1D`
mesh and performs only BF16 and FP32 reduce-scatter on ones. Both outputs are
correct, then the process fails on the same device/core during Ethernet
teardown. This control has no model weights, experts, trace capture, or KV
cache.

The model implementation did expose the infrastructure path and still needs
its own remaining validation. In particular, EP improves prefill but its
measured decode is slower than the paired single-device baseline. Resolving
the teardown failure does not by itself select EP or complete Stage 04.

## Recovery and device health

Board recovery was performed, not merely proposed. The Stage 04 command log
records successful bounded `tt-smi -r` commands after the original EP assert,
after the model-free CCL failure, and after the single-ERISC control. Each was
followed by `timeout 60 tt-smi -ls --local`, which listed all four p300c
devices. Relevant artifacts include:

- `ep_reset_assert.log` and `ep_list_assert.log`;
- `ccl_watcher_reset.log` and `ccl_watcher_list.log`;
- `ccl_single_erisc_reset.log` and `final_device_list.log`.

After recovery, `final_mesh_smoke.log` records an exit-0 normal four-chip
fabric mesh open/close. A fresh pre-repair `tt-smi -ls --local` also listed all
four devices. These checks establish recoverability and current visibility;
they do not waive the failed clean-handoff gate.

## Upstream comparison

TT-Metal commit `09ebc3d348af5dbb7badd81f9e534a2c393d83c4`, "Fix
dispatch and fabric teardown for multichip watcher", is already an ancestor of
v0.78.0. It adds full NoC drains and packet-tag cleanup to idle dispatch,
prefetch, and `tt_fabric_mux`. It does not update `fabric_erisc_router`.

The current upstream `main` fetched on 2026-09-26 is 1,088 commits beyond the
v0.78.0 base. Its router teardown is unchanged in the relevant section: it
drains writes and atomics and publishes termination without clearing packet
tags. No later merged commit was found that fixes this exact router exit.

Post-v0.78 commit `e0edc25ea8` catches reset exceptions during host teardown.
That can prevent a secondary `std::terminate`, but it does not repair the
device handoff and cannot satisfy Watcher. Other later CCL fixes address worker
kernel barriers or invalid connections and report different Watcher asserts.

## Workarounds considered

- `TT_METAL_DISABLE_FABRIC_TWO_ERISC=1`: tested with the model-free control;
  correct outputs followed by the same core heartbeat timeout. Rejected.
- Firmware upgrade: installed bundle 19.9.0 exceeds the diagnostic's stated
  minimum 18.10.0. The minimum-version text is an unconditional suffix on the
  heartbeat timeout. Rejected as the evidenced explanation.
- Disable Watcher/asserts or skip fabric termination: normal numerical runs can
  complete, but these choices hide or avoid the invalid handoff. Rejected for
  acceptance.
- Add dummy collectives or change topology until the final transaction ID is
  zero: outcome-dependent masking with no ownership guarantee. Rejected.
- Move to upstream `main`: no exact router cleanup exists there. Rejected as a
  direct remedy.

## Local repair and acceptance plan

The minimal repair mirrors the already-merged mux contract at the actual
failing owner. Each router ERISC performs `noc_async_full_barrier()` and then
`noc_clear_packet_tags(NOC_INDEX)` before the final sibling rendezvous. The
teardown master publishes `TERMINATED` only after both ERISCs reach that clean
state. Dynamic-NoC mode remains statically unsupported by this path, so the
change does not introduce an uncoordinated dynamic-NoC clear.

Validation order:

1. compile/style checks for the changed router kernel;
2. model-free BF16/FP32 CCL control with all Watcher checks and normal exit;
3. original Watcher EP probe with its numerical and teardown checks;
4. Stage 04 resume for batch/reuse, stack, maximum-context, final topology and
   performance selection, target profiles, and clean independent review.

If the model-free control still fails, preserve the new log and reset/list the
boards before further experiments. Do not disable the failing assertion.

## Repair validation results

The repair passed the two discriminating checks on 2026-09-26 with firmware
19.9.0 and all Watcher features enabled (`TT_METAL_WATCHER=10` and
`TT_METAL_WATCHER_NOINLINE=1`):

1. Model-free `probe_multichip_collective` completed BF16 and FP32
   reduce-scatter checks, stopped Watcher normally, closed all device drivers,
   and exited 0. Before the repair, the same command aborted during teardown.
   Generated Watcher/Inspector evidence is under `ccl_watcher_router_fix/`.
2. `probe_multichip_expert_parallel --samples 2` completed all 16 eager,
   trace, empty-rank, and prefill-union comparisons, printed
   `EP_PROBE_PASS`, stopped Watcher normally, closed all device drivers, and
   exited 0. Minimum observed local PCC remained 0.9994366683. The result is
   `ep_watcher_router_fix.json`; generated evidence is under
   `ep_watcher_router_fix/`.

The applicable pre-commit hooks, including clang-format and Metalium include
validation, pass for the router and this report. These results validate the
specific teardown repair on the observed four-chip path. Stage 04 still owns
the broader model and performance gates listed in `stage_review.md`.
