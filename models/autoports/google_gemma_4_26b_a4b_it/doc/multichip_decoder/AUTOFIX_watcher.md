# AutoFix Report

## Starting Evidence

- Diagnosis: `AUTOTRIAGE_watcher.md`, based on `ep_watch_noinline.log`, saved Watcher/Inspector metadata, exact compiled fabric receiver arguments, and current source.
- Original EP command: `HF_HUB_OFFLINE=1 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/ep_watcher_noinline timeout 300 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_multichip_expert_parallel --samples 2 --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/ep_watch_noinline.json`.
- All 16 numerical checks pass, JSON is written, and `EP_PROBE_PASS` prints. Process then exits 134: device 0/virtual Ethernet core `(28,25)` subordinate is at `NKFW` with nonzero write-capable NoC packet tags; primary is at `SEW` waiting for it.
- All hardware experiments/recovery below were run by the main agent. This investigator read their saved logs and wrote reports only.

## Hypothesis Experiments

### EP or trace misuse is necessary

- Experiment: existing `tests/probe_multichip_collective.py`, `TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1`, dedicated `TT_METAL_LOGS_PATH=.../ccl_watcher`. It opens the same FABRIC_1D 1x4 mesh and runs BF16 then FP32 reduce-scatter on `[1,1,32,2816]` ones with the same CCL manager, Linear topology and one link. There are no experts, weights, trace, or all-gather.
- Result: both outputs equal four; process exits 134. At 20:47:26.403, device 0/core `(28,25)` fails to return to base Ethernet firmware, heartbeat unchanged `0xdcba5e00`. Watcher stopped before this timeout, so no duplicate tag assertion was captured.
- Verdict: **refuted as a necessary trigger**. A minimal, model-free CCL workload suffices for teardown failure. The timeout is consistent with the asserted subordinate preventing primary completion, but same immediate register state is not independently proven in this control.
- Artifacts: `ccl_watcher.log`, `ccl_watcher/generated/`.
- Fix: none to model code is justified by this hypothesis.

### Supported single-fabric-ERISC mode avoids the failure

- Experiment: same CCL-only control, adding only `TT_METAL_DISABLE_FABRIC_TWO_ERISC=1`, with logs at `.../ccl_watcher_single_erisc`.
- Result: runtime explicitly confirms single-fabric-ERISC override; all Watcher checks remain enabled. BF16/FP32 outputs pass. Process exits 134 with the same device/core heartbeat timeout at 20:49:25.421; heartbeat unchanged `0xdcba8980`.
- Verdict: **refuted as a workaround**. Do not retain this mode as an accepted model configuration.
- Artifacts: `ccl_watcher_single_erisc.log`, `ccl_watcher_single_erisc/generated/`.
- Fix: none retained.

### Firmware is below the stated minimum

- Experiment: compare existing UMD discovery records to the timeout source; no hardware access needed.
- Result: all three workload logs report firmware bundle 19.9.0. Timeout text names 18.10.0. `tt_metal/llrt/llrt.cpp:588–594` emits the minimum-version suffix unconditionally on a missing heartbeat; it is not a failed version comparison.
- Verdict: **refuted for the stated minimum-version explanation**. This does not exclude all firmware defects, but a minimum-version upgrade is not supported by these artifacts.

### Fabric fails to restore NoC packet tags at kernel exit

- Evidence: `active_erisck.cc:51` catches the concrete zero-tag contract violation after kernel return. Compiled receiver uses NoC1/cmd0 and cmd2 with rotating explicit TRIDs. Router teardown drains transactions and rebases counters, but neither the drain helpers nor that exit path clear tags. The analogous fabric mux explicitly clears tags after draining.
- Verdict: **source-supported infrastructure defect; C++ repair not applied or verified**. This is a real failed state contract, not a claimed Watcher false positive.
- Scope: proposed repair belongs to the fabric C++ teardown/ownership boundary, outside the permitted model/tests/docs edits. A model-level synchronize, different dummy workload, Watcher suppression, or skipped shutdown does not implement the missing cleanup contract.

## Final Status

**Infrastructure blocker remains; no fix accepted.** The model-free control reproduced the teardown failure, the supported scoped fallback failed, and installed firmware exceeds the generic diagnostic's minimum. No further bounded model-scoped remedy has source or experiment support. Preserve numerical EP/full-decoder evidence independently; clean Watcher acceptance and final optimization acceptance remain open.

The next useful change is an authorized infrastructure repair that clears router-owned write-capable packet tags after draining and coordinating sibling ERISCs. It must be compiled and verified with the minimal CCL control, original EP command and required full-decoder Watcher checks. Do not claim that proposed repair is proven until those pass with normal process shutdown.

## Post-stage authorized infrastructure repair

The user subsequently authorized work outside the original model-only stage
scope. `fabric_erisc_router.cpp` now performs a full NoC barrier and clears the
current ERISC's packet tags before the final sibling rendezvous and
`TERMINATED` publication. This mirrors the handoff contract already used by
the fabric mux without disabling Watcher or skipping shutdown.

The exact model-free BF16/FP32 reduce-scatter control that previously aborted
now exits 0, including normal Watcher stop and device-driver close. The original
EP Watcher probe also completes all 16 comparisons, prints `EP_PROBE_PASS`, and
exits 0. Evidence and the complete causal analysis are in
`FABRIC_TEARDOWN_INCIDENT.md`, `ccl_watcher_router_fix/`,
`ep_watcher_router_fix/`, and `ep_watcher_router_fix.json`.

This resolves the recorded infrastructure blocker. It does not retroactively
turn the earlier stage review into a clean pass or satisfy the remaining
maximum-context, stack, final-selection, profiling, and review gates.
