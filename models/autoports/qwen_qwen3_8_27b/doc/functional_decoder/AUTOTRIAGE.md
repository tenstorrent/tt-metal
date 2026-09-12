# AUTOTRIAGE

## Diagnosis

The device-open failure was a recoverable stale active-Ethernet firmware state on device 0, translated core `(29,25)` (logical Ethernet `(0,9)`, physical NoC0 `(11,1)`): the host requested return to base firmware, but the base-firmware heartbeat remained fixed at `0xdcbad1c0` through the 20-second timeout. A board reset cleared the condition; the same checkout and firmware bundle `19.8.0` then opened and closed the mesh successfully. The originating workload or exact ERISC instruction that created the stale state is not recoverable from these artifacts.

This is an infrastructure recovery result. It provides no model-correctness or model-performance result. There is no demonstrated Qwen implementation bug, and no implementation change was needed for this failure.

Scope: branch `mvasiljevic/qwen38-full-bringup`, checkout `a3a9fb4229a045ad9361b4e39ad854b491346ea9`. Investigation inspected source and prepared logs only; the main agent performed serialized recovery. This report is the investigation's only file change.

## Triage Evidence

### Original failure

`mesh_smoke.log` records an isolated `ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0)` probe using `python_env/bin/python`. It failed before model code or any model operation:

- UMD discovered four local Blackhole chips, firmware bundle `19.8.0`, and KMD `2.8.0`.
- At `2026-09-11 17:23:41 UTC`, the first exception came from `return_to_base_firmware_and_wait_for_heartbeat`, called by `RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset`, `reset_cores`, and `run_launch_phase` during device initialization.
- Device 0/core `29-25`: port status `1`, RX link up `1`, train status `2`, PCS status `1`, retrain count `0`, SerDes reset status `1`, postcode `0xc0dea000`, ERISC0 reset PC `0x357c`, ERISC1 reset PC `0x9330`, and RISC soft reset register `0`.
- Heartbeat start and end both read `0xdcbad1c0`. The host's actual polling loop had already failed to observe advancement for approximately 20 seconds.
- At `17:24:01 UTC`, teardown repeated the same handshake failure and aborted the process. This second exception is downstream of the initialization failure.

`device_list_before.log` reports four Blackhole `p300c` device entries. Discovery and link-up bits establish visibility and trained links; they do not establish ERISC software progress.

### Prepared triage and its limitations

The broad `triage/tt-triage.txt` capture contains system information only: Ubuntu `24.04`, kernel `6.17.0-35-generic`, KMD `2.8.0`, UMD `0.9.9`, and tt-exalens `0.3.30`. `triage/triage-summary.txt` explains that Inspector data was unavailable and the default `--dev=in_use` selection could not resolve devices. Its failed dependent scripts provide no call stacks, running operations, NoC counters, or CB state. Their failures are collection limitations, not evidence of device faults.

The explicit device-0 capture `triage/focused.txt` provides:

- Four active Ethernet rows, including physical `(11,1)` / logical `(0,9)`, all reporting port/RX up and retrain count zero.
- A first mailbox word `0xD0E50002`. The source constants decode this as DONE for RELEASE_CORE, not a currently pending CALL. It records an acknowledged release command, not proof that the core is still progressing.
- ARC heartbeat approximately `9.9984/s`, clock `800 MHz`, and uptime about 4 days 3 hours. This directly supports a live ARC during capture.
- Ethernet `Heartbeat=True` is **not evidence that the heartbeat advanced**. `tools/triage/check_eth_status.py:77` initializes `previous_data = 0` and returns true on the first read that differs from zero. A permanently frozen `0xdcbad1c0` therefore passes this check. This source-proven diagnostic defect reconciles the focused output with the Metal timeout.

The hexadecimal reset-PC fields are reset-vector register contents, not captured current PCs. No RISC-V stop site can be inferred from their values. The postcode is preserved as an observed value; this report does not assign it an undocumented meaning.

### Discriminating recovery evidence

The main agent captured the failure evidence and then performed the bounded reset/recheck sequence:

| Artifact | Observed result |
| --- | --- |
| `reset_1.log` | Reset all PCI device IDs `[0,1,2,3]`, reinitialization completed, exit status `0`. |
| `device_list_after_reset_1.log` | All four expected devices visible. |
| `mesh_smoke_after_reset_1.log` | Firmware still `19.8.0`; `GRID 11-10`; `MESH_SMOKE_OK`; device/cluster close completed; exit status `0`. |

The retry opened at `17:26:30 UTC` and finished successfully at `17:26:35 UTC`. No firmware upgrade or source fix separates the failing and passing probe. This strongly favors resettable stale state over an unconditional firmware incompatibility, permanent link failure, topology mismatch, or model bug.

## Source Evidence

### Lifecycle handshake ledger

| State or signal | Producer / owner | Consumer / expected transition | Observed outcome |
| --- | --- | --- | --- |
| Current active launch slot's `exit_erisc_kernel` | Host `terminate_active_ethernet_cores_on_all_chips()` writes `1`, then performs an L1 barrier | A running active Ethernet kernel must leave its execution loop so firmware can regain control | Source already implements this request; no captured device stack establishes where a prior kernel stopped. |
| `aerisc_run_flag` | Host `set_metal_eth_fw_run_flag(..., false)` writes zero using the current HAL mailbox layout and performs an L1 barrier | Active ERISC firmware's idle loop observes a value other than `1` and returns from `main()` | Host retried this write until timeout; base-firmware progress did not resume. |
| Heartbeat at `0x7CC70` | Base Ethernet firmware is expected to update its heartbeat | Host must observe two unequal samples before resetting/reloading active ERISC resources | Frozen at `0xdcbad1c0` before reset; post-reset full open/close succeeds. |
| Base firmware / ERISC ownership | Return from Metal ERISC startup hands control back to base firmware | Host can safely proceed with subordinate reset and current firmware launch | Original initialization never passed this boundary. |

Relevant code:

- `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:214`: `run_launch_phase()` calls the termination helper before iterating reset and firmware initialization. The failing `reset_cores()` call precedes `initialize_and_launch_firmware()` for device 0.
- `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:432`: the termination helper reads the current launch-ring index, masks it to the ring size, sets `exit_erisc_kernel = 1` in that slot, writes it, and barriers. A proposed fix that merely adds this exit request would duplicate existing behavior.
- `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:376`: in two-ERISC mode, `assert_active_ethernet_cores_to_reset()` waits for base firmware before asserting the non-ERISC0 reset mask.
- `tt_metal/llrt/llrt.cpp:557`: `return_to_base_firmware_and_wait_for_heartbeat()` compares real samples and repeatedly requests stop. `set_metal_eth_fw_run_flag()` at line 604 obtains the run-flag offset from the current HAL/dev-message factory and barriers after the write.
- `tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc:258`: the idle loop invalidates its L1 cache and returns when `flag_disable[0] != 1`. If an inherited application is stuck before this loop, or expects a different mailbox layout, the host's current stop request may not produce the required return. Those mechanisms remain hypotheses because no prior binary or stopped PC was captured.
- `tt_metal/hw/firmware/src/tt-1xx/active_erisc-crt0.cc:23`: `_start()` calls `main()` and returns after it completes; it also provides a `longjmp` return path for `erisc_exit`.

This is a lifecycle state-transition problem. No model CB, semaphore, fabric-route, or tensor-shape producer/consumer mismatch is evidenced; manufacturing such a ledger would go beyond the capture.

### Coordinates, liveness, and firmware ABI

`tt_metal/third_party/umd/device/coordinates/blackhole_coordinate_manager.cpp:234` maps unharvested logical Ethernet channel `n` to translated `(20+n,25)`. Therefore `(29,25)` is logical `(0,9)`, explicitly identified by focused triage as physical `(11,1)`. The apparently different core labels refer to the same core.

`tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_active_eth.cpp:180`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h:21`, and UMD's `blackhole_eth.hpp` agree on base heartbeat address `0x7CC70`. No disagreement between those checked-in host/UMD address definitions explains this failure.

`tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h:355` documents `0xabcdxxxx` for the base-firmware heartbeat and writes the Metal software heartbeat `0xdcbaxxxx` to `heartbeat[1]`, at `0x7CC74`. The captured `0xdcba` prefix at the polled `heartbeat[0]` is suggestive of inherited software/firmware state or a different previous layout, but cannot identify the writer. The current checkout alone is insufficient to establish the resident pre-reset binary or a specific ABI mismatch.

`tt_metal/third_party/umd/device/topology/topology_discovery.cpp:304` explicitly skips Ethernet heartbeat validation on Blackhole. Successful topology discovery before the failing open consequently does not refute the frozen heartbeat. The UMD warm-reset recovery wrapper also uses topology discovery as its success criterion; on this architecture the open/close smoke is stronger evidence that the handshake recovered.

`INSTALLING.md:32` specifies Blackhole firmware `19.8.1`, whereas the logs show `19.8.0`. This is a documented stack-version discrepancy. The error's hardcoded `18.10.0` minimum is not a full compatibility check, but the successful unchanged-19.8.0 retry proves that this particular initialization failure was not an unavoidable consequence of that version. Intermittent version-related behavior remains unproven.

## Downstream Effects

- The teardown exception and final SIGABRT repeat the original failed return-to-base handshake; they are consequences, not an independent root cause.
- Missing Inspector-dependent triage data follows the absence of a successfully initialized live Metal runtime. It supplies no evidence of a dispatch or model hang.
- The unknown-motherboard fallback warning appears in both failing and successful probes. It does not explain the passing/failing contrast.
- Healthy ARC/link status coexists with stale ERISC execution. Neither board enumeration nor the flawed focused heartbeat boolean is sufficient as a recovery gate.

## Proposed Fix

The immediate recovery is complete: one serialized all-device reset restored the base-firmware lifecycle handshake, and the subsequent open/close smoke passed. Resume the model stage from its preserved state. No source change is justified to the model or the lifecycle protocol from this evidence.

A separate, bounded triage-tool improvement is supported by source: seed `previous_data` from an actual initial device read, then compare subsequent samples, preferably logging the sampled values and elapsed interval. A static nonzero register must return false; a changed register must return true. This diagnostic fix was not applied because this investigation was authorized to write only this report.

If this same failure recurs, the next discriminating evidence is the pre-reset ERISC current PC/stack and exact current run-flag/launch-slot addresses compared with the resident firmware ELF, together with repeated raw reads of `0x7CC70` and `0x7CC74`. Recovery should continue through the repository's serialized device-usage procedure. The successful recovery makes further experiments unnecessary for the current failure.

## Uncertainty

- The exact prior workload, prior firmware binary, and instruction that left ERISC unable to return to base firmware are unknown. Reset proves reversibility, not the origin of the stale state.
- No raw pre-reset device call stacks, NoC counters, mailbox-layout comparison, or base-firmware internals were captured. A prior kernel stall, inherited firmware-layout mismatch, and transient base-firmware state cannot be separated further.
- A successful single-device mesh open/close does not validate all four chips under workload, repeated lifecycle stress, model correctness, or performance.
- Firmware `19.8.0` is below this checkout's documented `19.8.1` stack version. The available evidence does not establish that upgrading would prevent recurrence, and no firmware was modified in this recovery.
