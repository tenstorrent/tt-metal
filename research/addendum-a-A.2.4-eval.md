# A.2.4 Evaluation: Per-device fabric heartbeat batching in nsexton/0-racecondition-hunt

## 1. Does the problem described in A.2.4 exist in racecondition-hunt?

**Yes, and it is significantly amplified by the race-condition fixes on this branch.**

A.2.4 describes heartbeat capture doing "one read per router" instead of a batched read. There are two distinct manifestations in this branch:

### (a) Telemetry reader: per-channel sequential `read_from_device`

`tt_metal/fabric/fabric_telemetry_reader.cpp:104-110`:
```
for (const auto& channel : channels) {
    ...
    sample.snapshot = read_fabric_telemetry(umd_cluster, hal, physical_chip_id, channel.channel_id);
}
```

Each call to `read_fabric_telemetry` (line 80) does:
```
cluster.read_from_device(buffer.data(), chip_id, eth_core, telemetry_addr, telemetry_size);
```

This reads the full telemetry struct (128-160 bytes depending on arch) from each ETH core separately via MMIO/UMD. On T3K with 16 active ETH channels per chip, that's 16 separate device reads per chip. The heartbeat counters (`tx_heartbeat`, `rx_heartbeat`) live inside this struct at `FabricTelemetryEriscEntry` (`fabric_telemetry.hpp:60-61`).

### (b) Race-condition heartbeat polls: per-core sequential `read_reg` in tight loops (MUCH WORSE)

The branch introduces **four separate heartbeat polling loops** in `risc_firmware_initializer.cpp`, each doing `cluster_.read_reg()` per core per poll iteration:

1. **FIX TV** (lines 265-314): Polls all MMIO ETH cores during `run_launch_phase()`, `read_reg` at line 271.
2. **FIX AC/AR** Step 2 (lines 584-618): Polls all MMIO ETH cores during teardown, `read_reg` at line 590.
3. **FIX AC/AR** Step 4 timeout path (lines 1047-1079): Same pattern, `read_reg` at line 1053.
4. **FIX PF** (lines 1526-1531): Individual per-core heartbeat checks, `read_reg` at line 1531.

Each poll loop iterates over `poll_states` (one entry per active ETH core on MMIO devices) with a 10ms sleep between iterations, polling for up to 3-5 seconds. Every poll iteration does N separate `read_reg` calls (one per unready core).

**The firmware also writes heartbeat counters in two independent mechanisms:**
- Fabric telemetry heartbeats: `fabric_erisc_router.cpp:1534-1547` — tx/rx heartbeat in the telemetry struct, updated on tx/rx progress or idle.
- Legacy kernel heartbeat: `fabric_erisc_router.cpp:2619-2620` — `0xDCBA0000 | counter` written every 64th iteration to `FABRIC_KERNEL_HEARTBEAT_ADDR`.

The host reads the **legacy heartbeat** (FIX TV/AC/AR polls at `hb_addr`), not the telemetry heartbeats. These are at different L1 addresses:
- WH legacy: `0x1F80` (`fabric_erisc_router.cpp:2268`)
- BH/QA legacy: `0x7CC70` (`fabric_erisc_router.cpp:2266`)
- Telemetry struct: `MEM_AERISC_FABRIC_TELEMETRY_BASE` (arch-dependent, e.g. `dev_mem_map.h:254`)


## 2. Better, worse, or same as batch-t3k-ttnn-unit?

**Significantly worse.** The batch-t3k-ttnn-unit branch has:
- The same `fabric_telemetry_reader.cpp` per-channel loop (identical code).
- **None of the race-condition heartbeat polling loops** (FIX TV/AC/AR/PF don't exist there). The firmware initializer has only a single `return_to_base_firmware_and_wait_for_heartbeat` call at line 309.

The racecondition-hunt branch adds **at least 3 hot polling loops** that each do O(N_cores × poll_iterations) individual MMIO reads during init/teardown. On T3K with ~16 ETH cores per MMIO device, each poll loop can do hundreds of individual `read_reg` calls before all cores report ready. This is a direct multiplier on PCIe bus traffic during critical init/teardown windows.


## 3. What would implementing A.2.4 look like concretely?

Two separate optimizations, targeting the two heartbeat read patterns:

### (a) Telemetry reader batching

In `fabric_telemetry_reader.cpp`, replace the per-channel `read_from_device` loop with a single bulk read that covers all ETH channels' telemetry regions on a given chip. This requires:
- Computing the min/max L1 address range across all active ETH channels (telemetry structs are at the same offset on each ETH core, but each core is a different NOC target).
- Since the cores are at different NOC coordinates (not contiguous memory), true "single read" isn't possible via vanilla MMIO. Instead, use UMD's scatter-gather or multi-core read if available, or at minimum pipeline the reads (issue all, then collect results) instead of serial blocking reads.
- Alternatively, have firmware aggregate all per-erisc heartbeats into a single per-device summary location readable with one MMIO read (e.g., a shared L1 region or ARC scratch register).

### (b) Race-condition poll batching (higher impact)

In `risc_firmware_initializer.cpp`, the FIX TV/AC/AR poll loops already iterate over all cores in a single time window (good — this was FIX AR's contribution). But each iteration still does N individual `read_reg` calls. Options:
- **Firmware-side aggregation**: Have ARC or a designated MMIO ETH core publish a per-device "all channels alive" bitmask at a single well-known address. Host polls one address per device instead of N per device.
- **UMD batch read API**: If UMD supports multi-address reads (reading the same offset from multiple cores in one PCIe transaction), use that instead of looping `read_reg`.
- **Reduced-frequency fallback**: After the first N channels report ready, increase the poll interval for remaining channels (diminishing returns on per-channel polling).


## 4. Interactions with race-condition fixes

The race-condition fixes create a **tension** with A.2.4:

### Pro-batching interactions
- **FIX AR** already consolidates per-core sequential polling into a shared time window. This was specifically designed to avoid the O(N × timeout) problem of sequential polling. Batching the reads within each iteration is the natural next step.
- The 10ms `kPollInterval` sleep between iterations means PCIe bandwidth isn't the bottleneck today — the sleep dominates. But reducing per-iteration read count would allow **shorter poll intervals** without PCIe saturation, enabling faster convergence.

### Anti-batching / caution interactions
- **FIX TW** (line 280-285, 599-603): The two-phase heartbeat detection (wait for nonzero, then wait for change or 0xABCD marker) requires reading the **current** value per core. A batched read that returns a composite "all alive" bitmask would lose the per-core phase tracking. Implementation must preserve per-core state or restructure the state machine.
- **FIX AR2** (line 520-528): The 100ms post-deassert delay is specifically to let L1 clear before reading heartbeats. A batched read doesn't change this requirement — the delay is about timing, not read efficiency.
- **FIX PF** (line 1526-1531): This does individual per-core heartbeat reads as a secondary check during teardown. It's a low-frequency path (once per core during terminate_stale_erisc_routers), so batching here yields minimal benefit.
- **Stale read concern**: The race-condition fixes are extremely sensitive to reading stale values (the entire FIX TW/AR2 chain exists because of stale reads). Any batching mechanism must guarantee freshness — e.g., firmware-side aggregation must use volatile writes, and the host must not cache aggregated values.

### Net assessment
A.2.4 is **applicable and beneficial** for the hot polling loops (FIX TV, FIX AC/AR), but implementation must preserve the per-core two-phase heartbeat state machine (FIX TW). The simplest approach that respects this constraint: have firmware write each core's heartbeat to a per-device shared array at a known offset, so the host reads one contiguous region per device instead of N separate `read_reg` calls to N different NOC coordinates. The telemetry reader path is lower priority since it's not on the init/teardown critical path.
