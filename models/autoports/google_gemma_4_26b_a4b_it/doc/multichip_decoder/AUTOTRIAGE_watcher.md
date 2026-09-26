# AUTOTRIAGE

## Diagnosis

The EP probe exposes a **fabric kernel exit-state violation**: the subordinate ERISC returns from `fabric_erisc_router::kernel_main` with at least one write-capable NoC packet-tag register nonzero. The fabric router uses explicit transaction IDs and drains their transactions during teardown, but its exit path does not clear those persistent registers. Watcher correctly checks that separate kernel-handoff contract. Calling this a harmless Watcher false positive is not supported.

The model/EP algorithm is not necessary to trigger a teardown failure: the parent's CCL-only control computes both expected results and then fails on the same device/core during return to base Ethernet firmware. That control's recorded symptom is a heartbeat timeout, **not a second captured packet-tag assertion**. The same stale-tag cause is strongly supported by the first run and source, but the control alone cannot prove identical device state.

No implementation edits or hardware commands were performed by this source-only investigator. The current model/tests/docs scope excludes a C++ router repair. The parent tested the supported single-fabric-ERISC fallback and reproduced the same-core heartbeat timeout. No source-supported, verified model-scoped workaround remains; this is an infrastructure blocker for clean Watcher acceptance.

## Triage Evidence

Evidence is relative to this stage directory:

- `AUTOFIX_ep.md` records the exact EP command, numerical checks, exit 134 and serialized recovery. `ep_watch_noinline.log` records all 16 correctness comparisons and `EP_PROBE_PASS`, followed at 20:41:10.171 UTC by device 0 acteth logical `(0,8)`, virtual `(28,25)`, `subordinate_erisc` failing the packet-tag-cleared assertion. Both current kernel names are `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`; waypoints are `SEW,NKFW,X,X,X`.
- `ep_watcher_noinline/generated/watcher/watcher.log` provides earlier live snapshots. At its second dump, this core runs fabric kernel IDs 30 and 32. `generated/inspector/kernels.yaml` maps subordinate kernel 32 to JIT directory `4029278992320149898/kernels/fabric_erisc_router/15077944180441090803/`. The last dump aborts while reading device 0; it is not a complete cross-device snapshot.
- The matched JIT `named_ct_arg_map_generated.h` confirms `MY_ERISC_ID=1`, `NUM_ACTIVE_ERISCS=2`, receiver channel 0 serviced, local-write NoC 1/command buffer 0, forwarding NoC 1/data command buffer 2, `ENABLE_DEADLOCK_AVOIDANCE=0`, and `WAIT_FOR_HOST_SIGNAL=1`. This matches the subordinate receiver and the exact checked register bank.
- `tests/probe_multichip_expert_parallel.py` prints `EP_PROBE_PASS` **after** its `finally: close_mesh_device(mesh)`. This differs from the earlier profiler runner's `TP_DONE` marker. The assertion is still a process teardown failure; return from Python mesh close is not a clean process-exit certificate.
- `ep_mesh_watch_noinline.log` is the empty-fabric control: `disabled features: None`, open/close and whole-process UMD shutdown complete. It sends no CCL payload and therefore does not exercise transaction-tag reuse under traffic.
- The parent's subsequent CCL-only control uses `tests/probe_multichip_collective.py` with `TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_NOINLINE=1`, and `TT_METAL_LOGS_PATH=.../ccl_watcher`. `ccl_watcher.log` records `disabled features: None`, correct BF16 and FP32 results, Watcher stopping at 20:47:06.400, then at 20:47:26.403 a timeout returning device 0/core `(28,25)` to base firmware. Heartbeat remains `0xdcba5e00`; RX link is up and retrain count is zero. The parent reports exit 134. No model weights, EP, all-gather or trace execution are present in this control.
- The CCL control's Watcher is already stopped when the timeout is reported, so absence of a printed assertion is not a clean assertion result. Its `generated/watcher` and Inspector artifacts are preserved. The EP run's attempted post-abort tt-triage capture failed after its host disappeared; it supplies no additional live register contents.
- `ccl_watcher_single_erisc.log` confirms `TT_METAL_DISABLE_FABRIC_TWO_ERISC` took effect and all Watcher features remained enabled. BF16/FP32 CCL outputs pass; Watcher stops at 20:49:05.419, and device 0/core `(28,25)` times out returning to base firmware at 20:49:25.421. Heartbeat remains `0xdcba8980`, RX link remains up, and retrain count is zero. Parent reports exit 134. This refutes the tested supported configuration workaround.
- All three workload logs report firmware bundle **19.9.0**. The timeout's minimum-version text says **18.10.0**; installed firmware is newer. `tt_metal/llrt/llrt.cpp:588–594` appends that text to every heartbeat timeout without checking the installed version there. It is not evidence that this system needs a minimum-version update. No new hardware firmware query is needed for that distinction.

## Source Evidence

### Exact assert and downstream waiter

- `tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc:41` calls `kernel_main`; lines 43–52 then enter `NKFW`, check pending reads, nonposted writes, atomics and posted writes, and finally call `ncrisc_noc_packet_tags_cleared(NOC_INDEX)`. The reported assertion is line 51's final check. Thus the subordinate reached the kernel epilogue; it is not parked inside a CCL send/credit loop. Earlier checks did not stop it, although fabric's counter rebasing means this is not independent proof that every historical transaction was accounted correctly.
- On Blackhole, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:255` checks `NOC_PACKET_TAG == 0` on command buffers 0 (large writes), 2 (register/small writes) and 3 (atomics). The failure proves at least one was nonzero, but the log does not record which register or numeric tag.
- `tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc:94` labels the wait for subordinate completion `SEW`. Lines 337–342 wait for that completion after the primary kernel returns and before setting the done signal. The observed primary `SEW` is explained by the subordinate's deliberate assertion halt.

### Tag producer and missing restore boundary

- `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp:179` issues local destination writes with the receiver's explicit transaction ID, `local_chip_data_cmd_buf`, and `edm_to_local_chip_noc`. For the recorded kernel those are command buffer 0 and NoC 1. Forwarding uses command buffer 2 on NoC 1 through `fabric_router_adapter.hpp` and the explicit-ID write APIs.
- `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2454` lowers the explicit-ID packet write to `ncrisc_noc_fast_write<..., true /* use_trid */, ...>`. Blackhole `noc_nonblocking_api.h:554` writes the transaction ID into `NOC_PACKET_TAG`. The with-state variant also sets the tag at `dataflow_api.h:2581`.
- `fabric_erisc_router_ct_args.hpp:552` selects four transaction IDs when deadlock avoidance is disabled. `fabric_erisc_router_transaction_id_tracker.hpp:72` maps successive receiver buffer slots to transaction IDs; its `all_buffer_slot_transactions_acked` at line 109 drains every allocated ID, not just ID zero.
- `fabric_erisc_router.cpp:2879` teardown coordinates the two ERISCs, drains the serviced receiver's TRIDs (line 2911), rebases NoC counters (line 2931), performs write/atomic barriers (lines 2949–2950), then marks termination. `kernel_main` calls it at line 3800 and returns at line 3816. **No tag clear/reset exists on this exit path.**
- The existing drain is not a tag reset: `dataflow_api.h:2603` waits for explicit-ID writes to flush; lines 1780 and 1863 wait for write/atomic completion. None writes `NOC_PACKET_TAG`. `noc_nonblocking_api.h:831` rebases software counters from status registers and does not clear packet tags either. Adding another host synchronization or repeating these barriers would not restore register state.
- A clear helper already exists at `noc_nonblocking_api.h:241`, but it is absent from the fabric router exit. The neighboring fabric mux kernel explicitly performs `noc_async_full_barrier(); noc_clear_packet_tags(noc_index);` at `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp:272–274`. This is supporting precedent for the missing cleanup contract, not a tested patch for the router.

### Ownership and route ledger

| Boundary | Producer and consumer | Contract / observation |
| --- | --- | --- |
| Model to collective | EP emits a local `[1,1,token_rows,2816]` contribution; `MeshConfig.allreduce` calls RS then AG | Existing `models/demos/gemma4/config.py:106–126` path, `dim=3`, `cluster_axis=1`, Linear, one link; all four replicas match in the probe |
| Model-free control | Four replicated `[1,1,32,2816]` ones enter RS; output shards are concatenated on host | Every output equals four for BF16 and FP32; failure persists without EP, AG or trace |
| Fabric receive to destination | Receiver ERISC1 uses NoC1/cmd0 for local writes and NoC1/cmd2 for downstream data | Packet header owns destination; receiver assigns rotating TRID; these same two registers are checked on exit |
| Receiver buffer reuse / termination | TRID tracker polls writes by ID and drains all four IDs during teardown | Completion consumes outstanding transactions, not the persistent last-used command-buffer tag |
| Kernel handoff | Firmware checks write-capable tags are zero | No router exit clear restores the required default; subordinate halts, primary waits at `SEW` |
| Runtime shutdown | Host requests base firmware, waits for heartbeat | CCL-only timeout occurs on the same physical Ethernet core |

No packet-route error is proven: capture has no invalid destination or stuck credit return, and the decisive stop-site is after router teardown returns. Payload destination/first-hop selection is delegated to existing CCL/fabric implementations; no custom packet headers or connection management are added by EP. Per-packet runtime routes cannot be reconstructed from these snapshots, but a missing route calculation is not required to explain this epilogue assertion.

`tt_metal/llrt/llrt.cpp:557` repeatedly requests stop/base-firmware return and waits for the heartbeat to change. A primary stuck waiting for its halted subordinate can explain the CCL control's later timeout. Because that control did not capture `NKFW`/registers, this is an inference rather than a second direct assertion observation.

## Downstream Effects

- Host Watcher throws from its polling thread and aborts the EP process. The primary ERISC completion wait and possible return-to-base-firmware timeout are downstream of the subordinate's failed exit contract.
- Passing output comparisons establish useful numerical correctness but do not erase the teardown failure or count as a clean Watcher pass.
- The no-payload fabric smoke and CCL-only failure fit traffic-dependent stale tags: no payload need leave a nonzero TRID, while legal data traffic can. They do not imply bad hardware or a need to alter EP routing.
- Removing assertions, suppressing Watcher features, exiting before teardown, or omitting fabric termination would hide the failure rather than prove the handoff contract.

## Proposed Fix

### AutoFix experiments and present verdict

1. **Hypothesis: EP sparse execution, zero-rank routing or replay misuse is necessary for the failure.** Experiment: the parent ran the existing model-free `probe_multichip_collective.py` with the same fabric/Watcher settings. Both CCL results were correct, then process exit failed on device 0/core 28-25. **Verdict: refuted as a necessary trigger.** This does not certify all EP behavior; it isolates a sufficient failure outside the EP implementation.
2. **Hypothesis: a legal fabric receiver leaves stale packet tags during shutdown.** The EP assertion plus exact compiled receiver configuration and missing source clear support this directly. CCL-only teardown failure strengthens the subsystem attribution. **Verdict: source-supported cause; smallest C++ repair not experimentally applied**, because user scope excludes C++ changes. Upstream repair should restore zero tags on every router-owned write-capable NoC command buffer only after its traffic is drained and sibling access is coordinated, then rerun CCL-only and the EP probe with all Watcher checks. A blind reset before draining is not proposed.
3. **Scoped fallback hypothesis: supported single-fabric-ERISC mode completes the same workload and teardown with every Watcher feature active.** `TT_METAL_DISABLE_FABRIC_TWO_ERISC=1` is a presence-based runtime option (`tt_metal/llrt/rtoptions.cpp:755`); `erisc_datamover_builder.cpp:90–110` honors it. On this Blackhole runtime, `compute_mesh_router_builder.cpp:973–988` places the single fabric worker on physical ERISC1/NoC1. Thus it changes fabric scheduling while retaining the checked NoC bank. The parent ran the CCL-only control with this additional setting; the log verifies activation and the same-core heartbeat timeout after correct outputs. **Verdict: refuted as a workaround; exit 134.**

A bounded reproducer for the tested item 3 settings is below; the 180-second wrapper is a reproduction bound, not an independently recovered detail of the parent's original shell command:

```bash
TT_METAL_DISABLE_FABRIC_TWO_ERISC=1 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/ccl_watcher_single_erisc timeout 180 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_multichip_collective
```

Acceptance required full process exit 0, correct outputs and no Watcher/firmware teardown error. That failed. No model-level tag-reset API was found to satisfy this contract. Do not carry the single-ERISC environment change into the selected model configuration as an accepted fix.

No supported `set_fabric_config` argument directly resets ERISC packet tags. `FabricRouterConfig` exposes payload size, while fabric manager `INIT_FABRIC` can skip termination; that would avoid the challenged boundary, not validate it. Changing Ring/Linear topology or packet size can change the final tag by accident and is not an evidence-backed repair. Keep normal synchronization and shutdown ownership; do not add unrelated dummy CCL traffic to try to land on TRID zero.

The remaining remedy supported by source is an infrastructure C++ fix at the fabric router's drained exit boundary, followed by the two minimal CCL checks and original EP/full-decoder acceptance. This is outside the user's model/tests/docs scope. Preserve the successful numerical EP evidence separately, but leave clean Watcher and final optimization acceptance open. Further model perturbations are not justified by the three captured outcomes.

## Uncertainty

- The exact nonzero register/value is unavailable. The identified producer writes both checked data buffers; source and assertion constrain the fault but do not identify the last packet.
- CCL-only reproduced a same-core teardown failure with a different final host symptom. A repeated captured assertion or saved register state would prove shared immediate cause more strongly, but EP-specific execution is already unnecessary for the failure.
- The main agent owns all device experiments/recovery and outcome updates. This report records their supplied exit status and independently read artifacts; it does not claim the investigator ran hardware.
- No clean Watcher acceptance or supported fallback has been demonstrated. The blocker conclusion uses the original concrete assertion, the CCL-only reproduction and the failed supported fallback, not source inspection alone. Firmware 19.9.0 exceeds the stated minimum; that excludes the specific old-firmware explanation but does not prove all possible firmware defects absent.
