# AUTOTRIAGE: one-worker linear all-gather watcher assertion

Subsequent patch integration and successful native watcher probes are recorded
in `AUTOFIX_watcher_ag.md`; this document preserves the initial diagnosis.

## Diagnosis

The non-mux `minimal_default_writer.cpp` calls a checked fabric-connection
accessor unconditionally, including the outward-facing worker at a linear
endpoint. That worker has no connection in its direction by construction.
`get_forward_connection()` asserts `has_forward_connection()` on line 119 of
`fabric_connection_manager.hpp`. This exactly matches the recorded BRISC
assertion. The corresponding backward endpoint violates the analogous line
123 contract. The kernel already guards every actual fabric send by the
existence of a destination, but does not guard this earlier pointer lookup.

This diagnosis is supported by the original watcher run and the complete
source argument chain. The proposed patch has not been applied or executed by
this source-investigation agent. Hardware A/B and original-case verification
remain with the root agent.

## Triage evidence

- `final_watcher_sliding.command.json` records the final default TP4 layer-0
  4096-token prefill / 128 advancing traced-decode command, eight duplicate
  replays, cache checks, `TT_METAL_WATCHER=10`, and
  `TT_METAL_WATCHER_NOINLINE=1`. Exit code is `-6`.
- `watcher_failure/run.log`: TP1 completes; fabric is reconfigured to a 1x4
  `FABRIC_1D` mesh. At 05:45:12 the watcher identifies physical device 0,
  logical worker `(0,0)`, virtual worker `(1,2)`, **BRISC assertion line 119**
  while running the minimal all-gather writer. Its companion NCRISC kernel is
  `minimal_default_reader.cpp`.
- Waypoints are `K,CRBW,W,W,W`. They are ordered BRISC, NCRISC, TRISC0/1/2:
  **CRBW belongs to the reader**, while the writer has halted on the assertion.
  The literal writer source line 119 is blank. The watcher explicitly warns
  that the assertion line can refer to an included header.
- Searching line 119 in the relevant source headers identifies
  `tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp`:
  `ASSERT(has_forward_connection())`. The dataflow CB reserve implementation
  has no assertion at that line or in its wait loop.
- The host has already aborted, so Inspector-dependent live running-op and
  stack capture fails (`watcher_failure/triage-command.log`). The explicit
  all-device capture succeeds: `watcher_failure/tt-triage-devices.txt` reports
  healthy link status, heartbeats and zero retrains on active Ethernet ports;
  all four ARC heartbeat rates are approximately 10/s. These observations do
  not reconstruct a live CB counter snapshot, but do not suggest a link fault.

## Source evidence

### Host-to-kernel route and connection ledger

1. `multichip_decoder.py:allreduce` uses persistent `reduce_scatter_minimal_async`
   followed by `all_gather_async` for logical decode rows == 1. Sliding decode
   selects `num_workers_per_link=1`; the all-gather takes the mesh overload with
   `persistent_output_tensor`, `cluster_axis=1`, `Topology.Linear`, and two
   all-gather semaphores. The failing writer name independently confirms that
   the minimal-default backend was selected.
2. `all_gather_async_default_program_factory.cpp:create_at` derives the forward
   neighbor with physical-coordinate offset +1 and the backward neighbor with
   offset -1 along the selected mesh axis. `Linear` deliberately leaves the
   outward neighbor absent at each endpoint.
3. The factory always allocates workers for **both directions**. Explicit
   worker count 1 sets `num_mux_cores_per_direction_per_link=0` (lines 253–279),
   selecting the non-`USE_WORKER_MUX` writer branch. More workers use the mux
   path, which already leaves its connection pointer null when its direction
   has no neighbor.
4. Factory runtime-argument construction, lines 782–802, encodes:

   | Worker direction | Forward flag | Backward flag | Connection selected by writer |
   | --- | --- | --- | --- |
   | 0 | `forward_coord.has_value()` | false | `get_forward_connection()` |
   | 1 | false | `backward_coord.has_value()` | `get_backward_connection()` |

   Missing neighbors produce two false flags and no connection arguments.
   `FabricConnectionManager::build_from_args` reads exactly these flags and
   constructs only the connections that exist. `open()` safely checks them.
5. Writer lines 224–227 then request the direction's connection **without** a
   guard. The outward direction therefore trips the accessor assertion before
   reaching any payload transfer. Which logical endpoint maps to physical
   device 0 is not separately captured; the observed forward assertion proves
   that this worker selected the absent forward accessor.

### Why a null pointer for an absent destination is safe

Use the existing `detail::valid_targets(direction)` predicate around the
lookup, preserving the accessor assertions whenever a real route is required.
Every later dereference was checked:

- Startup barrier sends: guarded by `detail::valid_targets(direction)`.
- Packet-header initialization: guarded by the same predicate.
- Local-slice payload forwarding: direction-specific compile-time
  `num_targets_backward_direction` or `num_targets_forward_direction` guards.
  The direction-1 local memory copy still executes at an outward endpoint.
- Local-slice ready-semaphore increments: guarded by `valid_targets`.
- Forwarded-slice payload and semaphore sends: linear `writes_expected` is
  initialized to zero and is assigned only inside
  `valid_targets_backward(direction)` / `valid_targets_forward(direction)`.
  Thus an outward endpoint executes **zero** forwarded-slice iterations.
- Teardown: the non-mux branch uses
  `fabric_connection.is_logically_connected()` before `close()` and never
  dereferences the direction pointer. The mux branch is unchanged.

There is no need to fabricate an endpoint connection, alter routing, change
precision or memory layout, or disable an assertion.

### Circular-buffer producer/consumer ledger

The factory creates one CB0 on every sender worker. Let `B` be the fabric
channel buffer byte size, `T` the input tile/page byte size, and
`P=min(4,floor(B/T))`. The factory checks `B>=T` and allocates **3P pages**.
NCRISC is the producer; BRISC is the consumer.

| Phase | NCRISC production | BRISC consumption |
| --- | --- | --- |
| Local slice | Reserve P, read min(P,remaining) valid tiles, read barrier, push P | Wait P, send/copy min(P,remaining), flush writes, pop P |
| Forwarded slice | Wait for remote readiness; reserve/read/push P per covered group | Wait/send/flush/pop P per matching forwarded group |
| Last partial group | Still reserve/push P; unused padded CB pages are not transmitted | Still wait/pop P; only valid pages are addressed/transmitted |

For the current decode AG input, each channel has logical shape
`[1,1,1,704]`, padded `[1,1,32,704]`: 22 tiles. Attention has one channel,
paired MoE has two. With one worker and link, both directions independently
process all 22 tiles/channel, using `ceil(22/P)` CB groups/channel. Default
`chunks_per_sync` is `min(max(floor(22/P),1),160)`; writer and reader use that
same runtime argument. Persistent output changes the output address and
startup-barrier policy, not the CB allocation or P-page transfer balance.
L1 versus DRAM changes the address generator and placement, not this ledger.

Because BRISC asserts before any CB pop, NCRISC can produce three groups and
then blocks at its next reserve, exactly explaining CRBW. Buffer fullness is
a downstream consequence; there is no evidence here for a CB capacity fix.

## Downstream effects

The stopped writer prevents local input draining and fabric sends. Its reader
waits for CB space; other devices may wait for readiness or fabric completion;
the host eventually aborts after watcher polling. These are consequences of
the invalid connection lookup, not independent network or semaphore failures.
Ordinary non-watcher runs can appear correct because the unused reference is
never dereferenced by the already-guarded sends when device assertions are off.
The single-worker specialization exposes this branch; two-worker mux runs do
not use the failing accessor.

## Proposed fix

`watcher_ag_endpoint.patch` contains a three-line semantic change, only in the
minimal-default all-gather writer: initialize a typed connection pointer to
null and perform the existing lookup only when
`detail::valid_targets(direction)` is true. This follows the existing mux
branch's null-pointer convention and keeps assertions for valid destinations.
No runtime implementation file was edited by this investigation.

## Focused verify/refute experiments

`probe_watcher_all_gather.py` is a bounded TP4 native all-gather test with the
same logical tile shapes, full-grid semaphores, persistent output overload,
and eight traced replays. It compares every output rank bit-for-bit against
the concatenation of the actual uploaded rank tensors, so BF8 upload rounding
cannot create a false failure. It has been syntax/format checked only here.

Run each case in a separate process with watcher enabled and serialize device
use. The root may use the original watcher failure as the pre-fix evidence
instead of intentionally triggering the same abort again.

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/probe_watcher_all_gather.py --workers 1 --output /tmp/ag_endpoint_one.json
```

Predictions / useful controls:

1. Unpatched workers=1, linear, L1 persistent: same absent-connection assertion.
2. Unpatched `--workers 2`: mux path passes (control; unchanged data/layout).
3. Patched workers=1: eager and eight traced replays pass for attention BF16
   and `--dtype bfloat8_b --channels 2` paired MoE.
4. Patched `--no-persistent --memory DRAM --rows 33`: same connection fix plus
   normal startup barrier, partial last packet and nonaligned logical rows.
5. Rerun the exact original 4096/128 layer watcher command and final watcher
   stack checks. Keep final accuracy, trace determinism and cache gates.

The patch modifies a device kernel and must compile through the normal kernel
JIT on those commands. Record the exact build-wrapper outcome separately if
required by the repository's build policy.

## Uncertainty

No live Inspector state survived the host abort, so this report does not claim
measured CB counters or exact native compile-time packet size B. The ledger
uses the factory's exact symbolic formula and actual model tile shapes. The
original failure plus source contract verifies the absent-connection bug;
post-patch watcher and output evidence are still required. This investigation
does not change or independently validate any other CCL path.
