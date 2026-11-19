# AUTOTRIAGE

## Diagnosis

The post-normalization fusion changes CB20's producer from NCRISC to TRISC2 without transferring the producer's private cumulative counter. The original Q input has already published and consumed four pages. The later compute-side `rounded.push_back(4)` publishes four again, not eight. TRISC0 therefore observes zero available pages and waits forever at `rounded.wait_front(4)`. CB21 has the same producer-switch defect and would become the next blocker if only CB20 were repaired.

This diagnosis is supported by both the captured counters and the exact stopped source line; it is not a hardware/fabric diagnosis. No running source or device state was modified during this investigation.

## Triage Evidence

- Evidence: `tt-triage.txt` and `triage-summary.txt` in this directory; original invocation/environment: `../flat_ordered_post_boundary_32_raw_command.json`. The calling agent identified bounded process 3170953 and retained the STOP marker. This investigation made no process-control, capture, or device calls.
- SHA256 at inspection: triage `9eda5203cbf634a5835c0cc6eebe6fedc34fbe979502adc22e066178817fef4e`; compute `52985ed63f39cd2d5b86bee98b74f19fb09f6a46d79a0dc62821aaa3073904ea`; reader `8833f0e2e3f670915d660593697d9dddcc44092c2120c7e18a9adb460b4dd04b`.
- There are **416 TRISC0 observations** stopped in `post_norm_gate`, `tests/kernels/flat_ordered_post/compute.cpp:106:64`, through `llk_wait_tiles`. This covers the 104 active workers on each of four devices.
- The circular-buffer table begins at evidence line 2440. Every one of the 416 recorded CB20 rows has **pushed=4, popped=4, occupancy=0**, although compute has entered its second use. CB21 likewise has 4/4/0. Every CB31 row has 0/0/0.
- Representative device0/core(0,0) rows: CB6=4/4/0 (line2447); CB7=16/0/16 (line2448); CB20=4/4/0 (line2461); CB21=4/4/0 (line2462); CB25=8/8/0 (line2466); CB26=2/2/0 (line2467); CB27=1/0/1 (line2468); CB29 and CB30 each4/0/4 (lines2470–2471); CB31=0/0/0 (line2472).
- NCRISC is back in firmware waiting for BRISC notification. Its gamma/gate payloads have already arrived in CB29/30. BRISC is waiting for CB31 at `writer.cpp:12`; TRISC1/TRISC2 are downstream in the next RMSNorm multiplication/pack. No lack of gamma/gate data explains the stop.
- The summary records successful completion of all capture checks, including binary integrity, DRAM health, NoC status, and the stack/CB dumps. Ethernet counter warnings in the active capture are not needed to explain this deterministic compute-local counter failure.

## Source Evidence

Paths beginning `tests/` are relative to `models/autoports/ornith_ai_ornith_1_5_9b/`; `tt_metal/` paths are relative to the checkout.

- `tests/kernels/flat_ordered_post/reader.cpp` publishes Q/K/V to CB20/21/22 with four pages each. `flat_prepare()` consumes and pops each (`compute.cpp:74–90`).
- `post_norm_gate()` then declares `rounded(20)` and `normalized(21)` (`compute.cpp:99`), packs the recurrence output into CB20, pushes four, and waits for four (`:100–106`). Neither the helper nor kernel implements an ownership/counter handoff.
- `tt_metal/hw/inc/api/dataflow/circular_buffer.h:43–60` dispatches compute push to the pack thread and compute wait to the unpack thread. They are separate producer/consumer counter owners, not ordinary shared local variables.
- NCRISC `cb_push_back` increments the shared received counter (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:208–214`). The pack thread instead increments its **private** `get_local_cb_interface(output).tiles_received`, then stores that cumulative value to the shared counter (`tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_pack.h:55–71`). Its private value was initialized to zero (`tt_metal/hw/inc/internal/circular_buffer_init.h:52–61`) and was never advanced by NCRISC's earlier push.
- TRISC0 waits on shared-received minus its private-acked count (`tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_unpack.h:18–34`); its first Q consumption advanced acked to four (`:49–53`).

Counter ledger for CB20:

| Event | Producer | Shared received | TRISC2 private received | TRISC0 acked | Available |
|---|---|---:|---:|---:|---:|
| Initialize | — | 0 | 0 | 0 | 0 |
| Publish Q | NCRISC | 4 | 0 | 0 | 4 |
| Consume Q | TRISC0 | 4 | 0 | 4 | 0 |
| Publish rounded output | TRISC2 | 4 | 4 | 4 | 0 |
| Wait for rounded output | TRISC0 | 4 | 4 | 4 | 0 forever |

The new producer's reserve does not restore correctness: its stale private-received value0 and shared-acked4 make the free-space calculation report8 pages for this four-page CB (`llk_io_pack.h:33–43`), so the reserve can pass before the subsequent receive counter stalls the consumer.

CB21 repeats this pattern when used for normalized output. CB25/26 retain the same producer and consumer across preparation and post-normalization; their counters continue normally and need no analogous remapping.

## Downstream Effects

CB31 never receives output, so BRISC cannot write the result or drain the already-produced final-state CB7. TRISC1/TRISC2 advance in their instruction streams until they need the next unpack/math result, explaining their RMSNorm frames while TRISC0 remains at line106. These are consequences of the lost CB20 publication, not independent state-write or normalization arithmetic failures.

## Proposed Fix

Use one producer owner per CB for the entire kernel invocation:

| CB | Producer | Consumer | Sequential payloads |
|---|---|---|---|
|20|NCRISC|TRISC0|Q4, then gamma4; received progresses0→4→8|
|21|NCRISC|TRISC0|K4 only|
|22|NCRISC|TRISC0|V4, then gate4; received progresses0→4→8|
|29|TRISC2|TRISC0|Rounded recurrence output4|
|30|TRISC2|TRISC0|Normalized output4|
|31|TRISC2|BRISC|Final gated output4|

Concretely, change reader gamma destination29→20 and gate destination30→22. Change compute rounded20→29, normalized21→30, gamma29→20, and gate30→22, including every pack, input-format configuration, `row_rsqrt`, normalization argument, copy, wait, and pop. These six slots remain BF16 with four-page capacities, so this does not require more L1 or a precision change.

**Preserve reader ordering:** publish the original Q/K/V and scalar gates, then the normalization unit CB27 and identity CB14, **before** attempting blocking second reserves on CB20/22. Otherwise the reader can wait for Q/V consumption while compute waits for the unpublished unit/identity. The current reader already places unit and identity before its post-normalization loads; retain that ordering during remapping.

Do not reset shared counters or artificially add four pushes. Those changes would mask ownership corruption and risk desynchronizing pointers or later phases. The remapping eliminates the unsupported producer transition.

## Uncertainty

- The captured state and source strongly establish this hang mechanism. The proposed remap remains untested in this investigation.
- Completion will not establish numerical correctness of fused RMSNorm/SiLU. Re-run the unchanged PCC/state gates, including the 1024-answer case, and watcher/trace checks after the normal authorized recovery procedure.
- Keep the captured kernel sources/provenance alongside this report before modifying them. The evidence line numbers describe the frozen failing source, not future repaired revisions.
