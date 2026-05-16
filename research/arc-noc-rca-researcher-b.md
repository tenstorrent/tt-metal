<!-- SUMMARY: Deep analysis of why device 4 (non-MMIO far N300) shows all-zero ERISC router states post-quiesce, tracing fabric init/teardown paths for MMIO vs non-MMIO devices, and Option 3 risk assessment -->
<!-- KEYWORDS: ARC NOC bug, ERISC, EDMStatus, non-MMIO, device 4, fabric init, configure_fabric_cores, terminate_stale_erisc_routers, quiesce, zero state -->
<!-- SOURCE: code analysis of nsexton-0-racecondition-hunt branch -->
<!-- SCOPE: ERISC router lifecycle on non-MMIO devices, fabric init/teardown asymmetry, zero-state root cause, Option 3 analysis -->
<!-- USE WHEN: investigating why non-MMIO device ETH routers show zeroed state, or evaluating whether removing TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART fixes the hang -->

# Researcher B Findings: Non-MMIO Device ERISC Router State & Fabric Initialization

## 1. EDMStatus Zero (0x00000000) Is NOT a Valid State

The `EDMStatus` enum is defined at:
`tt_metal/fabric/fabric_edm_packet_header.hpp:48`

```cpp
enum EDMStatus : uint32_t {
    STARTED                      = 0xA0B0C0D0,
    REMOTE_HANDSHAKE_COMPLETE    = 0xA1B1C1D1,
    LOCAL_HANDSHAKE_COMPLETE     = 0xA2B2C2D2,
    READY_FOR_TRAFFIC            = 0xA3B3C3D3,
    TERMINATED                   = 0xA4B4C4D4,
    INITIALIZATION_STARTED       = 0xB0C0D0E0,
    TXQ_INITIALIZED              = 0xB1C1D1E1,
    STREAM_REG_INITIALIZED       = 0xB2C2D2E2,
    DOWNSTREAM_EDM_SETUP_STARTED = 0xB3C3D3E3,
    EDM_VCS_SETUP_COMPLETE       = 0xB4C4D4E4,
    WORKER_INTERFACES_INITIALIZED = 0xB6C6D6E6,
    ETHERNET_HANDSHAKE_COMPLETE  = 0xB7C7D7E7,
    VCS_OPENED                   = 0xB8C8D8E8,
    ROUTING_TABLE_INITIALIZED    = 0xB9C9D9E9,
    INITIALIZATION_COMPLETE      = 0xBACADAEA
};
```

**Every valid EDMStatus value is a distinctive magic number.** Zero (0x00000000) is explicitly NOT in the enum.

The `is_known_edm_status()` function at `fabric_firmware_initializer.cpp:39-57` confirms this -- it checks all 15 enumerants and returns `false` for any other value including zero.

**What zero means:** The `router_sync_address` L1 slot was either:
- (a) **Never written** -- the ERISC router firmware never started on this channel (L1 is in its power-on/reset state of all zeros), OR
- (b) **Explicitly zeroed** by `configure_fabric_cores()` which writes a zero buffer to all `addresses_to_clear` for each active ETH channel

The Phase 2.5 code at `device.cpp:637` treats zero as "already clean":
```cpp
if (status_buf[0] == 0 || status_buf[0] == terminated_val) {
    // skip -- already clean
    continue;
}
```

Similarly, `terminate_stale_erisc_routers` at `fabric_firmware_initializer.cpp:449`:
```cpp
if (status_buf[0] == 0 || status_buf[0] == terminated_val) {
    continue;  // clean -- nothing to do
}
```

**Zero is treated as "clean/uninitialized" by all code paths.** This is semantically correct for a device whose ERISC routers were never programmed, but it is dangerously misleading for device 4 in this scenario -- because device 4's ERISCs WERE once programmed and running, and the zero state means their L1 sync address was cleared (by configure_fabric_cores or by AllGather teardown) without the firmware actually being in a known stopped state.

---

## 2. Full Initialization Path for ETH Fabric Routers

### 2.1 Initial fabric bring-up (FabricFirmwareInitializer::init)

File: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:89-116`

The flow is:
1. `control_plane_.write_routing_tables_to_all_chips()` -- writes routing tables to ALL chips (MMIO and non-MMIO)
2. `compile_and_configure_fabric()` which calls:
   - For EACH device: `dev->compile_fabric()` (creates & compiles the fabric program)
   - For EACH device: `terminate_stale_erisc_routers(dev, builder_context)` -- cleans up any stale state from prior runs
   - For EACH device: `dev->configure_fabric()` -- writes firmware to device

**Critical: Both MMIO and non-MMIO devices go through the same init path.** The `devices_` vector contains ALL devices in the mesh, not just MMIO ones.

### 2.2 configure_fabric_cores() -- The L1 zeroing function

File: `tt_metal/fabric/fabric_init.cpp:58-74`

```cpp
void configure_fabric_cores(tt::tt_metal::IDevice* device) {
    auto soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto addresses_to_clear = builder_context.get_fabric_router_addresses_to_clear();
    const auto& router_config = builder_context.get_fabric_router_config();
    std::vector<uint32_t> router_zero_buf(router_config.router_buffer_clear_size_words, 0);
    for (const auto& [router_chan, _] : router_chans_and_direction) {
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }
    }
}
```

**This function writes ZEROS to specific L1 addresses on every active ETH channel of the device.** It reaches non-MMIO devices because `WriteToDeviceL1` calls `Cluster::write_core`, which for non-MMIO (remote) chips goes through UMD's `write_to_device` + `wait_for_non_mmio_flush`:

```cpp
// tt_metal/llrt/tt_cluster.cpp:785-811
void Cluster::write_core(const void* mem_ptr, uint32_t sz_in_bytes, tt_cxy_pair core, uint64_t addr) const {
    // ... watcher checks ...
    this->driver_->write_to_device(mem_ptr, sz_in_bytes, core.chip, core_coord, addr);
    if (this->cluster_desc_->is_chip_remote(chip_id)) {
        this->driver_->wait_for_non_mmio_flush(chip_id);
    }
}
```

**KEY INSIGHT: `WriteToDeviceL1` for non-MMIO devices uses the UMD ethernet tunnel (dispatch tunnel), NOT the fabric router path.** The UMD ethernet tunnel is a low-level hardware feature of WH that uses dedicated ERISC cores for PCIe-to-remote-chip forwarding. This is SEPARATE from the tt-metal fabric ERISC routers. So L1 reads/writes to non-MMIO chips work even when the fabric routers are not running.

### 2.3 Device::configure_fabric()

File: `tt_metal/impl/device/device.cpp:386-422`

```cpp
void Device::configure_fabric() {
    if (fabric_program_ == nullptr) return;
    tt::tt_fabric::configure_fabric_cores(this);       // Zero out L1 addresses
    fabric_program_->impl().finalize_offsets(this);
    detail::WriteRuntimeArgsToDevice(this, *fabric_program_, using_fast_dispatch_);
    detail::ConfigureDeviceWithProgram(this, *fabric_program_, using_fast_dispatch_);
    cluster.l1_barrier(this->id());
    // Then write launch messages and start ETH cores
    for (each ETH core in fabric_program) {
        write_launch_msg_to_core(...)  // starts the ERISC firmware
    }
}
```

This runs on ALL devices including non-MMIO. After this, ERISC routers on device 4 would have gone through: zero L1 -> write firmware -> launch -> firmware posts INITIALIZATION_STARTED -> ... -> READY_FOR_TRAFFIC.

---

## 3. Why Device 4 Shows ALL ZEROS Post-Quiesce

### 3.1 The env var short-circuit

The fixture sets `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1` in `SetUp()`:
```cpp
// test_ccl_multi_cq_multi_device.cpp:96
setenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART", "1", /*overwrite=*/0);
```

This causes `quiesce_and_restart_fabric_workers()` at `device.cpp:429-437` to return immediately:
```cpp
if (const char* env = std::getenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART");
    env != nullptr && env[0] != '\0' && env[0] != '0') {
    return;  // ALL phases skipped
}
```

So `quiesce_devices()` calls `quiesce_and_restart_fabric_workers()` on each device, but the function is a no-op. **None of Phases 1, 2, 2.5, 3, or 4 execute.**

### 3.2 But the health probe shows zeros!

The health probe at `test_ccl_multi_cq_multi_device.cpp:109-171` reads `edm_status_address` from each device's active ETH channels using `cluster.read_core()`.

For device 4 (chip_id=4, non-MMIO), `read_core` goes through UMD tunnel:
- MMIO device 0 -> UMD ethernet tunnel -> chip 4 -> read ERISC L1 at router_sync_address

The fact that it reads 0x00000000 on ALL channels of device 4 means one of:
1. **The AllGather operation or its teardown zeroed the router_sync_address on device 4**
2. **The ERISC routers on device 4 crashed during AllGather and their L1 was zeroed by a recovery path**
3. **configure_fabric_cores() was invoked on device 4 during the AllGather flow, zeroing addresses without re-launching firmware**

### 3.3 Most likely explanation: AllGather teardown overwriting ERISC L1

The test sequence is:
1. Initial fabric init (all devices get ERISC routers -> READY_FOR_TRAFFIC)
2. `quiesce_devices()` -- no-op due to env var
3. `ttnn::all_gather()` -- this runs the CCL operation using fabric ERISC routers
4. `quiesce_devices()` -- again no-op due to env var
5. Health probe at "post-allgather-post-quiesce" shows device 4 = ALL ZEROS

**Between steps 3 and 5, the AllGather CCL operation (or its cleanup) disrupted device 4's ERISC state.** The AllGather CCL kernel communicates across all devices via the fabric ERISC routers. When AllGather completes, its teardown may overwrite ERISC L1 memory (potentially zeroing the sync address) or send TERMINATE signals that cause ERISC routers to zero their own sync address.

**However**, the fact that devices 0, 1, and 5 show non-zero "OTHER" states (0x3f803f80, 0x40004000, 0x40404040) while device 4 shows exactly 0x00000000 suggests **something went wrong specifically on device 4 during AllGather.** The "OTHER" states on devices 0, 1, 5 are NOT valid EDMStatus values either -- they look like corrupt/stale L1 values (possibly from the CCL operation overwriting portions of ERISC L1 with data payloads). But device 4's perfect zeros are distinctive.

### 3.4 Device 4 is special: it's the far N300 in the tunnel chain

In the T3K topology:
```
mesh[0,0] = chip 0 (N300, MMIO, left)
mesh[0,1] = chip 1 (N300, MMIO, right)
mesh[0,2] = chip 4 (N300, non-MMIO, far left)  <-- HANGS HERE
mesh[0,3] = chip 5 (N300, non-MMIO, far right)
```

Chip 4 being non-MMIO means all host-side L1 reads/writes go through the UMD tunnel. But the fabric ERISC routers are a DIFFERENT communication layer -- they're the software-defined mesh fabric running on top of the ETH hardware.

The zero state on device 4 specifically may indicate that AllGather teardown happened to call `configure_fabric_cores(device_4)` -- which zeros the addresses_to_clear -- as part of the CCL operation's fabric cleanup path, but then did not re-launch the ERISC firmware. Devices 0/1/5 might have their zeroing happen in a different order or their L1 gets partially re-written by CCL data arriving after the zero.

---

## 4. The `terminate_stale_erisc_routers` Function

File: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:423-558`

This function is called **per device** during `compile_and_configure_fabric()` (initial bring-up only):
```cpp
// fabric_firmware_initializer.cpp:588
terminate_stale_erisc_routers(dev, builder_context);
```

It iterates over all active fabric ETH channels on the given device, reads `router_sync_address`, and classifies:
- **Zero or TERMINATED** -> skip (clean) -- line 449
- **Known EDMStatus but not terminated** -> send TERMINATE, poll 50ms for TERMINATED -- line 481-545
- **Unknown value (corrupt L1)** -> send TERMINATE best-effort, skip polling -- line 464-478

The corrupt-L1 handling at line 453-478 is particularly relevant:
```cpp
const bool known_status = is_known_edm_status(status_buf[0]);
if (!known_status) {
    log_error(tt::LogMetal,
        "terminate_stale_erisc_routers: Device {} chan={} edm_status=0x{:08x} is NOT a "
        "valid EDMStatus value -- ERISC L1 appears CORRUPT ...");
    // Send TERMINATE best-effort, NOT polling (would time out)
    detail::WriteToDeviceL1(dev, eth_logical_core, term_addr, term_buf, CoreType::ETH);
    corrupt_count++;
    continue;
}
```

**This function runs on ALL devices, including non-MMIO device 4.** It uses `detail::ReadFromDeviceL1` and `detail::WriteToDeviceL1` which go through UMD tunnels for non-MMIO chips. So there is no asymmetry between MMIO and non-MMIO devices in how stale routers are terminated.

**BUT: This function ONLY runs during initial fabric bring-up, NOT during quiesce_and_restart_fabric_workers.** The quiesce path has its own Phase 2.5 that does similar work but is entirely bypassed by the `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART` env var.

**The comment in teardown_fabric_config mentions terminate_stale_erisc_routers explicitly** (metal_env.cpp:330):
> "causes terminate_stale_erisc_routers on the next fabric bring-up to see 840+ stale channels"

This confirms the function is designed for inter-session cleanup, not intra-session (quiesce-cycle) cleanup.

---

## 5. teardown_fabric_config vs quiesce: Full Comparison

### 5.1 MetalEnvImpl::teardown_fabric_config (full session teardown)

File: `tt_metal/impl/context/metal_env.cpp:280-372`

This iterates ALL chips (`cluster.all_chip_ids()`), including non-MMIO. For each chip with initialized routers:
1. Gets all active ETH channels
2. Reads EDMStatus via `cluster.read_core` (UMD tunnel for non-MMIO)
3. Waits up to 5000ms for TERMINATED
4. On timeout: logs warning, records force-reset (F5a currently skips actual reset)

Then resets fabric config to DISABLED and clears control plane state.

**Both MMIO and non-MMIO devices treated identically.**

### 5.2 quiesce_and_restart_fabric_workers (per-device, during quiesce)

File: `tt_metal/impl/device/device.cpp:425-820`

Full flow (when NOT disabled by env var):
1. Phase 1: Send IMMEDIATELY_TERMINATE to Tensix MUX (skipped if DISABLED)
2. Phase 2: Poll MUX for TERMINATED, assert_risc_reset (skipped if DISABLED)
3. Phase 2.5: Terminate ERISC routers -- ALWAYS runs even without Tensix MUX
4. Phase 3: configure_fabric_cores + reload firmware + re-launch
5. Phase 4: Wait for MUX READY_FOR_TRAFFIC (skipped if DISABLED)

Phase 2.5 uses the same L1 read/write APIs that work transparently for non-MMIO devices.

**There is NO MMIO vs non-MMIO asymmetry in the code paths.** The only difference is the UMD transport layer (MMIO direct vs ETH tunnel), but this is transparent to the caller.

---

## 6. Can write_buffer Commands Reach Device 4 With Zero ERISC State?

### 6.1 Two SEPARATE communication layers (CRITICAL)

**There are TWO independent communication paths to non-MMIO devices:**

**Layer 1: UMD Dispatch Tunnel (ETH hardware tunnel)**
- Low-level hardware path using dedicated ERISC cores for PCIe-to-remote-chip forwarding
- Used by: `write_core`/`read_core`, `l1_barrier`, `WriteToDeviceL1`/`ReadFromDeviceL1`, and **fast dispatch command queues** (prefetch/dispatch/relay kernels)
- These ERISC cores are NOT the fabric ERISC routers
- This path works regardless of fabric ERISC router state

**Layer 2: TT-Metal Fabric ERISC Routers (software mesh fabric)**
- Software-defined mesh routing layer running on additional ERISC cores
- Used by: CCL operations (AllGather, ReduceScatter), fabric-aware dispatch paths
- These are the routers whose `EDMStatus` is probed by the health check
- SEPARATE from the UMD dispatch tunnel

### 6.2 Yes, commands CAN reach device 4 with zero ERISC router state

The "Enqueue dummy ops" at `test_ccl_multi_cq_multi_device.cpp:284-306`:
```cpp
ttnn::write_buffer(QueueId(op_cq_id), dummy_tensor, {dummy_data});
ttnn::test_utils::dispatch_ops_to_device(dummy_tensor, QueueId(op_cq_id));
```

These use the fast dispatch CQ path:
1. `FDMeshCommandQueue::write_shard_to_device` (fd_mesh_command_queue.cpp:585)
2. -> `buffer_dispatch::write_to_device_buffer`
3. -> Dispatch commands through CQ system
4. -> UMD dispatch tunnel to non-MMIO device 4

**The fabric ERISC routers being in zero state does NOT prevent dispatch commands from reaching device 4.** The UMD dispatch tunnel is fully operational independent of the fabric router state.

### 6.3 The hang mechanism

The commands DO reach device 4's dispatch infrastructure. The hang occurs because:
1. The dispatch command arrives at device 4 via UMD dispatch tunnel
2. The dispatch core programs a Tensix worker to execute the write_buffer operation
3. The Tensix worker, whose L1 was corrupted by the AllGather teardown race, executes invalid instructions
4. These invalid instructions generate NOC traffic to ARC_RESET_SCRATCH_ADDR (0x880030060)
5. ARC firmware detects the unsafe NOC access and the chip becomes unresponsive
6. The host never receives the completion acknowledgment -> 5s timeout -> DEVICE_TIMEOUT

**This is scenario (a): commands reach device 4 dispatch but something on-chip goes wrong.**

---

## 7. Option 3 Analysis: Removing TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART

### 7.1 What would happen

If we remove `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1`, then `quiesce_and_restart_fabric_workers()` would execute on each device during both quiesce calls.

Since `FabricTensixConfig::DISABLED` (no Tensix MUX), Phases 1/2/4 are skipped. But Phase 2.5 and Phase 3 WOULD run:

**Phase 2.5** (device.cpp:618-710): For each active ETH channel:
- Read router_sync_address
- If zero or TERMINATED -> skip (no action needed)
- If active -> send TERMINATE to ERISC, wait 500ms for TERMINATED
- On timeout: log warning, continue (no ERISC reset on WH -- would tear down ETH PHY link)

**Phase 3** (device.cpp:712-773):
- `configure_fabric_cores(this)` -> zeros out all fabric router L1 addresses
- `WriteRuntimeArgsToDevice` -> writes new runtime args
- `ConfigureDeviceWithProgram` -> loads fresh firmware into ERISC L1
- `l1_barrier` -> ensures writes committed
- For each ETH core: `write_launch_msg_to_core` -> starts fresh ERISC firmware

### 7.2 Would it fix the device 4 zero-state issue?

**YES, the ERISC router state would be restored.** After Phase 2.5+3:
- Device 4's ERISC routers would be cleanly terminated (Phase 2.5)
- L1 would be zeroed and fresh firmware loaded (Phase 3)
- ERISC routers would boot fresh and eventually reach READY_FOR_TRAFFIC

**BUT this would NOT fix the hang.** The root cause is NOT the ERISC zero state. The root cause is:
1. AllGather CCL teardown leaves Tensix worker L1 corrupted on device 4
2. "Enqueue dummy ops" dispatches to device 4's Tensix workers
3. Corrupted Tensix workers generate invalid NOC traffic to 0x880030060

Phase 2.5+3 only touch ERISC cores, NOT Tensix worker/dispatch cores. The Tensix workers that cause the 0x880030060 NOC write are in the dispatch system, not in the fabric routing layer.

### 7.3 Risks of enabling Phase 2.5+3

1. **Phase 2.5 would mostly no-op on device 4** since its ERISC state is zero (treated as "clean"). The "OTHER" states on devices 0/1/5 (0x3f803f80 etc) are NOT valid EDMStatus, so `is_known_edm_status` returns false, and Phase 2.5 would send TERMINATE best-effort but not poll (matching the corrupt-L1 handling in terminate_stale_erisc_routers). **But wait -- Phase 2.5 doesn't call `is_known_edm_status`.** It simply checks if status is zero or TERMINATED. For devices 0/1/5 with "OTHER" non-zero values, Phase 2.5 would send TERMINATE and poll for 500ms. **Since these aren't running recognizable EDM firmware, the polls would all time out**, adding 500ms * N_channels per device. With ~6 active channels per device, that's ~3 seconds per MMIO device, ~6-9 seconds total for the mesh.

2. **Phase 3 could race with ERISC firmware on devices 0/1/5:** If the "OTHER" states mean ERISC firmware IS actually running (just with a corrupted status word), Phase 3's `configure_fabric_cores` would overwrite running ERISC L1. The Phase 2.5 timeout warning says to continue anyway, and the code comments at device.cpp:696-697 explicitly state: "Do NOT assert_risc_reset_at_core on WH ERISCs: resetting tears down the ETH PHY link, breaking non-MMIO L1 access for the entire mesh."

3. **No Phase 4 (no READY_FOR_TRAFFIC wait):** Since FabricTensixConfig::DISABLED, Phase 4 is skipped. Phase 3 re-launches ETH cores but doesn't wait for them to reach READY_FOR_TRAFFIC. If the next operation requires fabric routing, there could be a startup race. However, `ttnn::write_buffer` uses UMD dispatch, not fabric, so this is probably safe for the dummy ops.

4. **Performance cost:** The Phase 2.5 timeout waits on every non-zero/non-terminated channel add ~3-9 seconds to each quiesce call. With 2 quiesce calls in the test, that's 6-18 seconds of dead time.

### 7.4 Verdict on Option 3

**Removing `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1` would:**
- Fix the ERISC zero-state symptom on device 4 (Phase 3 restores them)
- Restore a known-good ERISC router state on all devices
- Add 6-18 seconds of timeout delays per test iteration
- **NOT fix the root cause hang** (Tensix worker L1 corruption on device 4)

**The ERISC zero state is a red herring for this specific hang.** The zero state indicates something went wrong during AllGather on device 4, but the subsequent hang is caused by Tensix worker corruption, not ERISC router absence. The two symptoms share a root cause (AllGather teardown race) but the fix must address the Tensix side, not just the ERISC side.

---

## 8. How `get_num_fabric_initialized_routers(chip_id)` Works for Non-MMIO Chips

File: `tt_metal/fabric/fabric_builder_context.cpp:159-175`

```cpp
void FabricBuilderContext::set_num_fabric_initialized_routers(ChipId chip_id, size_t num_routers) {
    TT_FATAL(chip_id < num_devices_, "...");
    TT_FATAL(num_initialized_routers_[chip_id] == UNINITIALIZED_ROUTERS, "...");
    num_initialized_routers_[chip_id] = num_routers;
}

uint32_t FabricBuilderContext::get_num_fabric_initialized_routers(ChipId chip_id) const {
    TT_FATAL(chip_id < num_devices_, "...");
    TT_FATAL(num_initialized_routers_[chip_id] != UNINITIALIZED_ROUTERS, "...");
    return num_initialized_routers_[chip_id];
}
```

This is a **host-side data structure** (a simple array indexed by chip_id). It's set during `FabricBuilder::connect_routers()` at `fabric_builder.cpp:110`:
```cpp
builder_context_.set_num_fabric_initialized_routers(device_->id(), routers_.size());
```

This is set for ALL devices during initial fabric compilation. The value reflects how many routers the builder created for the device, NOT the runtime state of those routers. **For device 4 (non-MMIO), this would be non-zero after initial fabric init**, because the fabric builder creates routers for all devices in the mesh.

The check `get_num_fabric_initialized_routers(chip_id) == 0` at various points (device.cpp:469, metal_env.cpp:289, fabric_firmware_initializer.cpp:166/263/308/425) is used to skip devices that have no fabric routers (e.g., single-chip configs). **Device 4 would NOT be skipped by this check** since it was initialized during initial bring-up.

---

## 9. Summary Table

```
Finding                                          | File:Line                          | Impact
-------------------------------------------------+------------------------------------+--------------------------------------------------
Zero is NOT a valid EDMStatus                    | fabric_edm_packet_header.hpp:48    | Zero = never-written or explicitly-zeroed L1
configure_fabric_cores zeros all router L1       | fabric_init.cpp:58-74              | Phase 3 zeros then re-loads firmware
WriteToDeviceL1 uses UMD tunnel for non-MMIO     | tt_cluster.cpp:785-811             | L1 access works regardless of fabric state
UMD dispatch tunnel != fabric ERISC routers      | Two independent comm layers        | write_buffer reaches device 4 even with zero ERISC
terminate_stale_erisc_routers runs on ALL devices | fabric_firmware_init.cpp:423       | No MMIO/non-MMIO asymmetry at init time
Phase 2.5+3 bypassed by env var during quiesce   | device.cpp:429-437                 | No ERISC restart between operations
Commands REACH device 4 but Tensix corrupted     | Dispatch tunnel operational        | Hang is on-chip Tensix corruption, not routing failure
Option 3 fixes ERISC symptom, not Tensix cause   | Phase 2.5+3 only touch ERISC cores | Would add 6-18s delay without fixing the hang
Phase 2.5 treats zero as "clean"                 | device.cpp:637                     | Device 4 zero channels would be skipped in Phase 2.5
get_num_fabric_initialized_routers is host-side  | fabric_builder_context.cpp:168     | Always non-zero for device 4 after fabric init
```

---

## 10. Hypothesis: Why Device 4 Specifically Shows Zeros

Device 4 (far N300, non-MMIO) is uniquely vulnerable because:

1. **AllGather data flow asymmetry:** In a 1x4 mesh AllGather, device 4 is at mesh position [0,2]. The "last hop" of the all_gather ring may terminate differently at this position. If the CCL teardown on device 4 calls a cleanup function that clears the router_sync_address, the timing may differ from other positions.

2. **Higher UMD tunnel latency:** Host-side operations targeting device 4 have higher latency (extra ETH tunnel hop). If AllGather teardown has timing-sensitive cleanup, device 4's operations complete later, potentially leaving it in a different state.

3. **The "OTHER" values on devices 0/1/5 are also corrupt:** The values 0x3f803f80, 0x40004000, 0x40404040 are NOT valid EDMStatus. They look like data payload remnants or partially-overwritten L1 from the AllGather CCL data transfer itself. The AllGather may overwrite ERISC L1 (including the sync address region) with CCL data on ALL devices, but the exact pattern differs based on the CCL ring topology:
   - Devices 0/1/5: L1 overwritten with non-zero CCL data remnants
   - Device 4: L1 overwritten with zeros (perhaps the CCL padding/termination pattern on this position)

This suggests the AllGather CCL operation itself is the source of ERISC L1 corruption on ALL devices, not just device 4. Device 4 happens to get zeros, which is the only value that looks "clean" to the Phase 2.5 checks, making it uniquely dangerous.
