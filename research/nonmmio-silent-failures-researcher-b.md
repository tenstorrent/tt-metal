<!-- SUMMARY: UMD comparison and topology-check-induced corruption analysis for non-MMIO silent failures
KEYWORDS: non-mmio, umd, destructor, topology-check, GetNumAvailableDevices, stale-channels
SOURCE: UMD git diff, CI log analysis, code reading
SCOPE: UMD branch vs main, topology check corruption path
USE WHEN: debugging non-MMIO stale ETH channel state after GetNumAvailableDevices call -->

# Non-MMIO Silent Failures: UMD Branch Comparison & Topology-Check Corruption Analysis

**Researcher B — Opus Swarm Investigation**

---

## 1. UMD Branch vs Main: What's Different

The branch `racecondition-fixes` has **6 commits** on top of UMD `origin/main`:

```
40b5df82  FIX PZ  — process-level cache for FIX AQ unreachable ASIC IDs
34af436d  FIX AF  — add timeout to read_non_mmio() CMD queue full spin loop
5882377c  FIX NU  — capture MMIO EthCoord before relay-safety guards
9758ef99  FIX NT  — preserve EthCoord for FIX-AQ-skipped unhealthy chips
fc5b0776  FIX AE  — relay-broken fast-path for wait_for_non_mmio_flush
1f0f2161  FIX W/X/AQ/AQ-2/AU — topology discovery crash guards (re-applied)
```

Plus one tt-metal-side commit:
```
5a2e723c3 — SiliconSysmemManager destructor: catch exceptions from unpin_or_unmap_sysmem()
```

### Key changes by category:

**Topology Discovery Resilience (FIX W, AQ, AQ-2, AU, NT, NU, PZ):**
- FIX W: Heartbeat check before relay through MMIO ETH channels — skip if firmware is dead (not `0xABCDxxxx`)
- FIX AQ: Catch `std::exception` around `init_tt_device()` for remote devices — degrade topology instead of crashing
- FIX AU: Use 5s timeout instead of 300s ARC_STARTUP_TIMEOUT during topology probing
- FIX NT: Preserve EthCoord for FIX-AQ-skipped devices so chip_locations stays complete
- FIX NU: Capture MMIO EthCoord before relay-safety guards (prevents TT_FATAL on coord lookup)
- FIX PZ: Process-level cache of unreachable ASIC IDs — avoids 5s re-probe per dead device per test

**Relay Communication (FIX AE, AF):**
- FIX AE: `relay_broken_` flag in `RemoteCommunication` — `wait_for_non_mmio_flush()` returns immediately when set
- FIX AF: Timeout added to `read_non_mmio()` CMD queue full spin loop (was missing, causing indefinite hang)

**Destructor Safety:**
- `SiliconSysmemManager::~SiliconSysmemManager()`: wraps `unpin_or_unmap_sysmem()` in try-catch to prevent SIGABRT from ENODEV ioctl failure

---

## 2. The Topology Check Path: How GetNumAvailableDevices Leaves Non-MMIO in Bad State

### Call chain for `GetNumAvailableDevices()`:

```
Python: ttnn.GetNumAvailableDevices()
  → MetalContext::instance().get_cluster().number_of_user_devices()
    → MetalEnv initialization (singleton)
      → UMD Cluster constructor
        → TopologyDiscovery::discover() — probes all ETH connections
        → Cluster::start_device()
          → LocalChip::start_device() × 4 (MMIO chips 0-3)
            → pin hugepages, init iATU, init membars
          → deassert_resets_and_set_power_state()
            → broadcast_tensix_risc_reset_to_cluster(ASSERT)
            → chip->deassert_risc_resets() for ALL chips (0-7)
              → This loads base UMD firmware on ALL ETH cores
              → ETH cores start polling, write 0x49706550 sentinel
            → enable_ethernet_queue() for all chips
            → set_power_state(BUSY)
```

**The critical step**: `deassert_risc_resets()` runs on ALL 8 chips including non-MMIO chips 4-7. This de-asserts the ERISC soft reset, causing each ETH core's BRISC to execute the ROM bootloader which loads the base UMD relay firmware. The firmware writes sentinel `0x49706550` to `edm_status_address` to indicate "base UMD firmware running, waiting for relay commands."

### The teardown path (Python exits, MetalContext destructor fires):

```
MetalContext::~MetalContext()
  → ... device teardown ...
  → UMD Cluster::~Cluster()
    → cluster_desc.reset()  // That's it — no close_device() call!
```

**CRITICAL FINDING**: The `Cluster::~Cluster()` destructor (line 575-579 of cluster.cpp) only does `cluster_desc.reset()`. It does NOT call `close_device()`. The `close_device()` method (line 1064-1075) must be called explicitly. When `GetNumAvailableDevices()` uses `MetalContext` singleton which eventually calls the cluster destructor path, if `close_device()` is not called first, the ETH cores on ALL 8 chips are left running base UMD firmware — they remain in the `0x49706550` sentinel state with the relay polling loop active.

### close_device() — what it does for non-MMIO:

```cpp
// cluster.cpp:1064
void Cluster::close_device() {
    // Close remote (non-MMIO) first
    for (auto remote_chip_id : remote_chip_ids_) {
        get_chip(remote_chip_id)->close_device();  // RemoteChip::close_device()
    }
    // Then close local (MMIO)
    for (auto chip_id : local_chip_ids_) {
        get_chip(chip_id)->close_device();  // LocalChip::close_device()
    }
}
```

**RemoteChip::close_device()** (remote_chip.cpp:89-99):
```cpp
void RemoteChip::close_device() {
    if ((uint32_t)local_chip_->get_clock() != local_chip_->get_tt_device()->get_min_clock_freq()) {
        if ((uint32_t)get_clock() != get_tt_device()->get_min_clock_freq()) {
            set_power_state(DevicePowerState::LONG_IDLE);  // relay write to non-MMIO
            assert_risc_reset(RiscType::ALL);               // relay write to non-MMIO
        }
    }
}
```

For non-MMIO chips, `set_power_state()` and `assert_risc_reset()` use the ETH relay path (write_to_non_mmio). If the relay is broken or the CMD queue is full, these operations can:
1. Hang indefinitely (pre-FIX AF)
2. Fail silently (the writes go into the CMD queue but never get serviced)
3. Throw exceptions (post-fix, caught by FIX AE's relay_broken_ fast-path)

**Even when close_device() IS called**, the non-MMIO `assert_risc_reset()` may fail because:
- The relay CMD queue on the MMIO ETH core servicing that non-MMIO chip may be full
- The ERISC on the gateway channel may have crashed or be in FABRIC mode from a prior Metal session
- The `wait_for_non_mmio_flush()` times out after 5s without confirmation

This leaves non-MMIO ETH cores with base UMD firmware still running — `0x49706550` sentinel persists.

---

## 3. CI Log Evidence

### Timeline from CI run 25246631881 (runner t3k-13):

**07:28:52** — `tt-smi -r` (PCIe reset, 30s timeout)
**07:28:56.638** — "T3K topology OK — 8/8 chips visible" — GetNumAvailableDevices succeeded
**07:28:56.662-56.905** — THREE `TopologyDiscovery` invocations (UMD Cluster, auto-discovery mesh, control plane)
**07:28:56.950** — TopologyMapper physical adjacency for 8 chips

**07:29:02.263-02.269** — `terminate_stale_erisc_routers` finds base-UMD firmware on **ALL 8 devices**:

```
Device 0: base_umd=6 channels (0,1,8,9,14,15)
Device 1: base_umd=6 channels (6,7,8,9,14,15)
Device 2: base_umd=6 channels (0,1,8,9,14,15)
Device 3: base_umd=6 channels (6,7,8,9,14,15)
Device 4: base_umd=4 channels (0,1,6,7)
Device 5: base_umd=4 channels (0,1,6,7)
Device 6: base_umd=4 channels (0,1,6,7)     [inferred from pattern]
Device 7: base_umd=4 channels (0,1,6,7)     [inferred from pattern]
```

**Key observation**: Non-MMIO devices 4-7 each have **4** base-UMD channels, while MMIO devices 0-3 each have **6**. The base-UMD channels are the inter-chip ETH links — these are the channels where UMD loaded relay firmware during `deassert_risc_resets()`. The difference in count reflects the number of ETH channels connected to other chips vs unused/intra-chip channels.

**This confirms**: The topology check (`GetNumAvailableDevices()`) call at 07:28:56 opened a UMD cluster which installed base firmware on ALL ETH cores. When the Python process exited, the non-MMIO chips were left with base UMD firmware running. The next Metal session (actual tests starting at 07:29:02) detects this as `base_umd` channels.

### What happens with the stale base-UMD state:

The Metal `terminate_stale_erisc_routers()` function correctly detects `0x49706550` and:
1. Does NOT add these channels to `probe_dead_channels` (they're alive relay firmware, not dead)
2. Records them in `base_umd_channels` set
3. Skips soft reset for these channels in `configure_fabric_cores()`
4. Sets `fabric_stale_base_umd_channels_ = true` on the Device object (FIX M)
5. Uses `write_launch_msg_to_core` instead of soft reset to transition them to fabric firmware

**This detection/recovery works**. The stale-base-UMD state is not itself the silent failure. The problem is when recovery fails — e.g., when `write_launch_msg_to_core` on a non-MMIO chip goes through a relay path that itself is stale.

---

## 4. Does Main Have the Same Topology Check?

**No.** Main's `run_t3000_ttnn_tests()` does NOT call `GetNumAvailableDevices()`. It simply runs tests directly:

```bash
# main branch:
run_t3000_ttnn_tests() {
  fail=0
  start_time=$(date +%s)
  echo "LOG_METAL: Running run_t3000_ttnn_tests"
  ./build/test/ttnn/unit_tests_ttnn
  ./build/test/ttnn/unit_tests_ttnn_tensor
  ...
```

The branch adds the topology check:
```bash
# racecondition-fixes branch:
run_t3000_ttnn_tests() {
  timeout 30 tt-smi -r || true
  # T3K topology sanity check
  raw_output=$(python3 -u -c "import ttnn; print(ttnn.GetNumAvailableDevices())" 2>/dev/null)
  ...
  echo "LOG_METAL: T3K topology OK — ${n_chips}/8 chips visible."
  ...
```

**This is significant**: The topology check was added to catch degraded hardware early (so CI doesn't silently skip all T3K tests and report green), but it has the side effect of:
1. Opening a full UMD cluster (all 8 devices)
2. Loading base UMD firmware to all ETH cores on all 8 chips
3. Exiting without clean teardown
4. Leaving 40+ ETH channels in base-UMD state that the real test session must handle

**Main doesn't have this problem** because it never does the topology pre-check. The first real test opens the cluster fresh.

---

## 5. SiliconSysmemManager Destructor Fix Analysis

Commit `5a2e723c3` wraps `unpin_or_unmap_sysmem()` in try-catch:

```cpp
SiliconSysmemManager::~SiliconSysmemManager() {
    try {
        SiliconSysmemManager::unpin_or_unmap_sysmem();
    } catch (const std::exception& e) {
        log_warning(LogUMD, "... ignoring exception during cleanup: {}", e.what());
    } catch (...) {
        log_warning(LogUMD, "... ignoring unknown exception during cleanup");
    }
}
```

**Root cause**: `PCIDevice::unmap_for_dma()` calls `TENSTORRENT_IOCTL_UNPIN_PAGES` which returns `ENODEV` when the device was already partially cleaned up (e.g., after `tt-smi -r` or a prior abnormal exit). The ioctl failure throws `UmdException<RuntimeError>`, which in a destructor calls `std::terminate()` → SIGABRT.

**Relationship to non-MMIO stale channels**: This fix is **separate from** the stale channel problem. It prevents a SIGABRT crash during UMD Cluster destruction, which would leave the process in an even worse state (no cleanup at all). However, it does NOT address:
- Why `close_device()` fails for non-MMIO chips (relay path issues)
- Why ETH L1 stays in base-UMD state after teardown
- The root cause of stale channels

The SIGABRT fix is a **prerequisite** for any other cleanup to even run — without it, the process dies before reaching the `close_device()` code path. But it doesn't solve the relay-broken non-MMIO teardown.

---

## 6. Stale Base-UMD Detection in Metal

### Detection code (`fabric_firmware_initializer.cpp:1190+`):

```cpp
const bool is_base_umd = (status_buf[0] == kBaseUmdFirmwareSentinel);  // 0x49706550

if (is_base_umd) {
    // Live relay firmware — touch nothing.
    // FIX M: record this channel so configure_fabric_cores() can skip soft reset.
    base_umd_channels.insert(eth_chan_id);
    // NOTE: do NOT add to probe_dead_channels — base-UMD state is the expected
    // fresh-boot / post-reset condition
}
```

### Propagation path:

1. `terminate_stale_erisc_routers()` returns `{probe_dead_channels, relay_broken, base_umd_channels}`
2. Caller stores `base_umd_channels_map[dev->id()]`
3. `dev->configure_fabric(probe_dead_channels, base_umd_channels)` passes channels to skip soft reset
4. If any `base_umd_channels` found: `device.fabric_stale_base_umd_channels_ = true`
5. The device logs: "Setting fabric_stale_base_umd_channels_=true — configure_fabric_cores will skip soft reset for these channels and use launch_msg instead (FIX M)"

### The FIX M mechanism:

Channels in `base_umd_channels` skip the normal `assert_risc_reset → load firmware → deassert_risc_reset` sequence. Instead, they receive a `write_launch_msg_to_core` which the already-running base UMD firmware's polling loop picks up, transitioning it to fabric firmware without a full reset cycle.

### Dead relay cross-check (`fabric_firmware_initializer.cpp:1667+`):

After the main probe, channels whose relay peer is dead get removed from `base_umd_channels`:
```
"configure_fabric_cores: Device {}/chan={} is in base_umd_channels but its peer connection is
 dead-relay. Removing from base_umd_channels so configure_fabric_cores() will soft-reset it."
```

This prevents the launch_msg path from targeting a channel whose remote endpoint is unreachable.

---

## 7. Summary of Findings

### The corruption chain:

```
tt-smi -r (PCIe reset)
  → All ERISC firmware stopped, ETH L1 zeroed
  ↓
GetNumAvailableDevices() topology check
  → UMD Cluster opens all 8 chips
  → deassert_risc_resets() loads base UMD firmware on ALL ETH cores
  → Base firmware writes 0x49706550 sentinel, starts relay polling
  → Python process exits
  → Cluster::~Cluster() does NOT call close_device()
  → Even if close_device() were called:
    - RemoteChip::close_device() sends set_power_state+assert_risc_reset via relay
    - These relay writes may fail (full CMD queue, dead gateway, etc.)
  → Result: ALL 48+ ETH channels across 8 chips left in base-UMD state
  ↓
Real test session opens
  → terminate_stale_erisc_routers() detects 48+ base_umd channels
  → FIX M: skip soft reset, use launch_msg for transition
  → Non-MMIO channels may fail to transition (relay path itself is stale)
  → Silent degradation: fabric works partially, non-MMIO operations may timeout
```

### Why main doesn't have this:
Main's test script doesn't call `GetNumAvailableDevices()` before running tests. The first test opens the cluster fresh with no stale base-UMD state to recover from.

### The fixes on the branch address:
- **FIX W/AQ/AU**: Topology discovery doesn't crash on dead/stuck non-MMIO devices
- **FIX AE/AF**: Relay communication doesn't hang indefinitely on dead paths
- **FIX NT/NU**: Coordinate metadata preserved even for degraded topology
- **FIX PZ**: Performance — don't re-probe known-dead devices every test
- **SysmemManager destructor**: Process doesn't SIGABRT during cleanup

### What the fixes DON'T address:
- **The topology check itself is the contamination source** — it installs base UMD firmware that persists across process boundaries
- **`Cluster::~Cluster()` doesn't call `close_device()`** — this is by design (caller is expected to call it), but `GetNumAvailableDevices()` never does
- **Non-MMIO `close_device()` relay path is inherently unreliable** — even explicit cleanup can leave stale state
