<!--
SUMMARY: Two detailed designs for extending the dispatch timeout when fabric ERISCs are making progress — MMIO-only polling and rate-based detection — with tradeoff analysis and recommendation.
KEYWORDS: dispatch-timeout, fabric, ERISC, progress-detection, MMIO, rate-based, AllGather, hang-detection, loop_and_wait_with_timeout, get_fabric_erisc_progress, Approach-B
SOURCE: Code analysis of nsexton-0-racecondition-hunt worktree (May 2026), system_memory_manager.cpp, command_queue_common.cpp, fabric_erisc_router.cpp, tt_cluster.hpp
SCOPE: Host-side progress polling for fabric ERISC packet counters, two alternative designs, tradeoff matrix, implementation guidance
USE WHEN: Implementing Approach B for dispatch timeout extension, deciding between MMIO-only vs rate-based progress detection
-->

# Approach B Designs: Extending Dispatch Timeout for Fabric ERISC Progress

## Problem Recap

The dispatch timeout (`loop_and_wait_with_timeout`) resets its clock whenever `get_progress()` returns a changed value. Currently only the dispatch kernel's progress counter (`get_cq_dispatch_progress()`) is read. When a large CCL op (AllGather, ReduceScatter) is in flight, the dispatch kernel is blocked waiting on fabric — its counter stops advancing — and the host times out even though fabric is legitimately moving data.

The previous (reverted) attempt at Approach B:
1. XOR'd `get_cq_dispatch_progress()` with `get_fabric_erisc_progress()` to combine both signals
2. `get_fabric_erisc_progress()` iterated ALL active devices and read ERISC packet counters via `cluster.read_core()`

**Why it was reverted — two fatal flaws:**
1. **Non-MMIO read hang**: `read_core()` on a non-MMIO chip goes through the ERISC relay. If the relay is broken (the very situation causing the hang), `read_core()` blocks indefinitely — the timeout handler itself deadlocks.
2. **Background traffic false-extension**: Fabric keepalive/maintenance traffic increments the packet counter even when no user op is running. The counter constantly changes, preventing the timeout from ever firing even for genuinely stuck dispatches.

---

## Current Code Context

### `loop_and_wait_with_timeout` (system_memory_manager.cpp:56-106)

```
Template params: FuncBody, FuncWait, OnTimeout, GetProgress
Every progress_update_interval (default 100ms):
    current_progress = get_progress()
    if current_progress != last_progress_value:
        reset timeout clock
If timeout_duration elapsed since last progress change:
    on_timeout()
```

Key: `get_progress()` returns a `uint32_t`. **Any** change (even by 1 bit) resets the clock. The function does not care about the magnitude of change.

### `get_fabric_erisc_progress()` (command_queue_common.cpp:187-237)

Iterates `get_all_active_device_ids()`, looks up each device's fabric node ID, then for each active fabric ETH channel reads a 32-bit packet counter at `FABRIC_KERNEL_HEARTBEAT_ADDR + 4`. Returns XOR of all counter values.

### Packet counter (fabric_erisc_router.cpp:2672-2676)

```cpp
if (tx_progress || rx_progress) {
    ++(*fabric_packet_counter_ptr);
}
```

Incremented every main loop iteration where ANY channel made TX or RX progress. On a busy AllGather, this increments millions of times per second. On an idle fabric with only keepalive traffic, this still increments occasionally (whenever keepalive packets trigger tx_progress/rx_progress).

### `read_core()` (tt_cluster.cpp:960-999)

For MMIO chips: goes through `driver_->read_from_device()` — a PCIe BAR read, fast and cannot hang.
For non-MMIO (remote) chips: goes through `driver_->read_from_device()` which internally uses the UMD relay through an ERISC. If the relay ERISC is stuck, this blocks until UMD's internal timeout (5s per read).

### Hardware topology context

- **N150**: 1 MMIO chip. No non-MMIO chips. Fabric is single-chip or disabled.
- **N300**: 2 chips — chip 0 is MMIO, chip 1 is non-MMIO. Fabric flows: chip0-ERISC <-> chip1-ERISC. Both ERISCs process packets.
- **T3K**: 8 chips, typically 4 MMIO + 4 non-MMIO (or 2 MMIO + 6 non-MMIO depending on topology). Each MMIO chip controls 1-2 non-MMIO chips.

### MMIO-capability API

```cpp
cluster.get_cluster_desc()->is_chip_mmio_capable(chip_id)  // bool
cluster.mmio_chip_ids()                                     // std::set<ChipId>
```

Already used in `dump_fabric_erisc_state()` (command_queue_common.cpp:267) to skip non-MMIO chips.

---

## Design 1: MMIO-Only Progress Polling

### Overview

Modify `get_fabric_erisc_progress()` to only read ERISC packet counters from MMIO-capable chips. Reads to MMIO chips are PCIe BAR reads — fast, cannot hang, no relay dependency.

### Implementation Detail

**File**: `tt_metal/impl/dispatch/command_queue_common.cpp`

**Change to `get_fabric_erisc_progress()`:**

```cpp
uint32_t get_fabric_erisc_progress() {
    auto& ctx = MetalContext::instance();
    if (!ctx.is_device_manager_initialized()) return 0;

    auto& control_plane = ctx.get_control_plane();
    if (control_plane.get_fabric_config() == tt_fabric::FabricConfig::DISABLED) return 0;

    const uint32_t heartbeat_addr = ctx.hal().get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
    const uint32_t packet_counter_addr = heartbeat_addr + 4;

    auto& cluster = ctx.get_cluster();
    const auto* cluster_desc = cluster.get_cluster_desc();
    uint32_t combined = 0;

    for (ChipId chip_id : ctx.device_manager()->get_all_active_device_ids()) {
        // DESIGN 1: Only read MMIO-capable chips — PCIe BAR read, cannot hang.
        if (!cluster_desc->is_chip_mmio_capable(chip_id)) {
            continue;
        }

        const auto fabric_node_id = /* same as current code */;
        if (!fabric_node_id) continue;

        for (const auto& [eth_chan_id, _direction] :
             control_plane.get_active_fabric_eth_channels(*fabric_node_id)) {
            // No try/catch needed — MMIO reads cannot hang.
            const CoreCoord eth_logical = /* same coord lookup */;
            uint32_t counter_val = 0;
            cluster.read_core(&counter_val, sizeof(uint32_t), eth_virtual, packet_counter_addr);
            combined ^= counter_val;
        }
    }
    return combined;
}
```

**Integration into `get_progress()` lambda — two options:**

**Option A (XOR combine):** In `fetch_queue_reserve_back` and `completion_queue_wait_front`, change the lambda:
```cpp
auto get_dispatch_progress = [&]() -> uint32_t {
    return get_cq_dispatch_progress(this->device_id, cq_id)
         ^ get_fabric_erisc_progress();
};
```
This is exactly what the reverted code did, but now `get_fabric_erisc_progress()` only reads MMIO chips.

**Option B (separate tracking):** Have `loop_and_wait_with_timeout` accept a vector of progress functions and reset if ANY of them changed. This is cleaner but requires modifying the template — heavier change.

**Recommendation: Option A (XOR combine).** Minimal code change, same integration pattern.

### Does it detect progress on multi-chip fabric ops?

**N300 (1 MMIO + 1 non-MMIO):**
- AllGather sends data from chip 0 -> chip 1 -> chip 0.
- The MMIO chip's (chip 0) fabric ERISC sends and receives packets. Its packet counter advances.
- We only read chip 0's counter. **YES — this detects progress**, because the MMIO-side ERISC must send/receive packets for data to flow.
- Even when the bottleneck is on the non-MMIO side (chip 1 processing slowly), the MMIO side's counter will still advance (it's receiving credits, ACKs, or data from chip 1).

**T3K (4 MMIO + 4 non-MMIO):**
- Data flows through chains of ERISCs across chips.
- MMIO chips at the edges of the mesh have fabric ERISCs that participate in routing.
- **YES — if any fabric traffic is flowing, at least one MMIO chip's ERISC must be participating** (data enters/exits the mesh through MMIO chips' dispatch, or is relayed through MMIO-chip ERISCs).

**Edge case — pure non-MMIO-to-non-MMIO traffic:** If an op sends data between two non-MMIO chips that route through only non-MMIO ERISCs (not passing through any MMIO chip's ERISC), the MMIO-side counters wouldn't advance. **However:** dispatch runs on MMIO chips, and CCL ops are initiated from dispatch. The MMIO chip's dispatch issued the op, and the completion must flow back through MMIO-accessible paths. Some MMIO ERISC must be involved in the data flow for the op to complete. This edge case is theoretical and not reachable in current AllGather/ReduceScatter implementations.

### What happens when the MMIO fabric ERISC itself is hung?

If the MMIO-side ERISC is genuinely stuck:
- Its packet counter stops advancing.
- `get_fabric_erisc_progress()` returns the same value repeatedly.
- The dispatch progress counter (`get_cq_dispatch_progress()`) is also stuck (dispatch is blocked waiting on fabric).
- The XOR combination doesn't change.
- **Timeout fires correctly.** This is the desired behavior.

### Does it reduce false progress from keepalives?

**Partially.** MMIO-chip ERISCs still receive keepalive traffic (they're part of the same fabric mesh). However, the reduction depends on how many ERISCs are read:
- Current (reverted): reads ALL ERISCs on ALL chips. More ERISCs = more XOR entropy from keepalive noise.
- Design 1: reads only MMIO-chip ERISCs. Fewer ERISCs = less XOR entropy.

**But the fundamental problem remains:** even a single MMIO ERISC receiving a keepalive packet increments its counter. The XOR with `get_cq_dispatch_progress()` still changes, resetting the timeout clock.

**This is the key weakness of Design 1.** It solves the hang problem but **does not solve the false-extension problem**. On a system where fabric is enabled but the dispatch is genuinely stuck (not a fabric op), keepalive traffic on MMIO ERISCs will prevent the timeout from firing.

### Complexity

- Lines changed: ~5 (add `is_chip_mmio_capable` check, remove try/catch).
- New state: none.
- New dependencies: none.
- Test surface: existing tests cover this path. Needs a test that verifies timeout fires when dispatch is stuck but fabric keepalives are present (but this test would *fail* with Design 1).

---

## Design 2: Rate-Based Progress Detection

### Overview

Instead of resetting the timeout whenever the packet counter changes by any amount, require the counter to change by more than a threshold rate (delta per polling interval) before treating it as "real progress." Slow background keepalive traffic produces small deltas; a large AllGather produces enormous deltas.

### Implementation Detail

**Where to implement:** Inside `get_fabric_erisc_progress()`, not in `loop_and_wait_with_timeout`.

**Rationale:** `loop_and_wait_with_timeout` is a generic template used by both `fetch_queue_reserve_back` and `completion_queue_wait_front`. Its contract is simple: `get_progress()` returns a `uint32_t`, and if the value changes, progress is assumed. Keeping rate logic inside `get_fabric_erisc_progress()` preserves this contract — the function returns a "progress token" that only changes when real progress is detected.

**New state needed (file-static or class-member):**

```cpp
// In command_queue_common.cpp, at namespace scope:
namespace {
    struct FabricProgressState {
        uint32_t last_raw_counter_sum = 0;    // sum of all ERISC counters
        uint32_t progress_token = 0;          // returned to caller; changes only on "real" progress
        std::chrono::steady_clock::time_point last_sample_time{};
        bool initialized = false;
    };
    FabricProgressState fabric_progress_state;
}
```

**Modified `get_fabric_erisc_progress()`:**

```cpp
uint32_t get_fabric_erisc_progress() {
    auto& ctx = MetalContext::instance();
    if (!ctx.is_device_manager_initialized()) return 0;

    auto& control_plane = ctx.get_control_plane();
    if (control_plane.get_fabric_config() == tt_fabric::FabricConfig::DISABLED) return 0;

    const uint32_t heartbeat_addr = ctx.hal().get_eth_fw_mailbox_val(FWMailboxMsg::HEARTBEAT);
    const uint32_t packet_counter_addr = heartbeat_addr + 4;

    auto& cluster = ctx.get_cluster();
    const auto* cluster_desc = cluster.get_cluster_desc();

    // DESIGN 2: Sum counters (not XOR) so we can compute a meaningful delta.
    uint32_t raw_sum = 0;
    for (ChipId chip_id : ctx.device_manager()->get_all_active_device_ids()) {
        // Also apply MMIO-only filter (Design 1 safety, free to include).
        if (!cluster_desc->is_chip_mmio_capable(chip_id)) continue;

        const auto fabric_node_id = /* same lookup */;
        if (!fabric_node_id) continue;

        for (const auto& [eth_chan_id, _direction] :
             control_plane.get_active_fabric_eth_channels(*fabric_node_id)) {
            uint32_t counter_val = 0;
            cluster.read_core(&counter_val, sizeof(uint32_t), eth_virtual, packet_counter_addr);
            raw_sum += counter_val;  // SUM not XOR — preserves magnitude info
        }
    }

    auto& state = fabric_progress_state;
    auto now = std::chrono::steady_clock::now();

    if (!state.initialized) {
        state.last_raw_counter_sum = raw_sum;
        state.last_sample_time = now;
        state.initialized = true;
        return state.progress_token;
    }

    uint32_t delta = raw_sum - state.last_raw_counter_sum;  // wraps safely
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - state.last_sample_time).count();

    // Threshold: packets per polling interval that constitute "real work."
    // Configurable via env var, default 10 packets per 100ms = 100 packets/sec.
    static const uint32_t PROGRESS_THRESHOLD = []() {
        const char* env = std::getenv("TT_METAL_FABRIC_PROGRESS_THRESHOLD");
        return env ? std::stoul(env) : 10u;
    }();

    // Scale threshold by actual elapsed time (in case polling interval varies).
    uint32_t scaled_threshold = static_cast<uint32_t>(
        PROGRESS_THRESHOLD * (static_cast<float>(elapsed_ms) / 100.0f));
    if (scaled_threshold < 1) scaled_threshold = 1;

    if (delta >= scaled_threshold) {
        ++state.progress_token;  // Signal "real progress" to caller
    }
    // else: delta was below threshold (keepalive-level traffic) — token unchanged,
    // timeout clock continues ticking.

    state.last_raw_counter_sum = raw_sum;
    state.last_sample_time = now;

    return state.progress_token;
}
```

**`loop_and_wait_with_timeout` changes: NONE.** The function already compares `get_progress()` return values. `progress_token` only changes when real progress is detected, so the existing mechanism works as-is.

**Integration:** Same as Design 1 Option A — XOR with `get_cq_dispatch_progress()`:
```cpp
auto get_dispatch_progress = [&]() -> uint32_t {
    return get_cq_dispatch_progress(this->device_id, cq_id)
         ^ get_fabric_erisc_progress();
};
```

### Traffic Rate Analysis: Background vs Real Op

**Background keepalive traffic:**
- The fabric router main loop runs at ~1 GHz ERISC clock speed.
- The heartbeat counter updates every 64 iterations (`(++fabric_heartbeat_counter & 0x3F) == 0`).
- Keepalive-related progress (credit returns, liveness checks) triggers `tx_progress || rx_progress` sporadically — on the order of a few events per millisecond at most when no user data is flowing.
- **Expected rate: ~1-50 counter increments per 100ms per ERISC** when idle.
- With 4 MMIO ERISCs: ~4-200 total increments per 100ms.

**Real AllGather traffic (e.g., 1GB tensor across 8 devices):**
- Fabric packets are typically 4KB-16KB.
- 1GB / 4KB = 262,144 packets.
- At line-rate Ethernet (~12.5 GB/s per link), 1GB takes ~80ms.
- Each ERISC main loop iteration that moves a packet increments the counter.
- **Expected rate: >>10,000 counter increments per 100ms per ERISC** during active transfer.
- With 4 MMIO ERISCs: >>40,000 total increments per 100ms.

**Separation:** The rates differ by 2-3 orders of magnitude. A threshold of 10-100 packets per 100ms interval cleanly separates background from real traffic.

**Recommended default threshold: 10 packets per 100ms (100 packets/sec).** This is:
- Well above the ~1-5 keepalive increments per ERISC per 100ms
- Well below the >>10,000 increments during active data transfer
- Configurable via `TT_METAL_FABRIC_PROGRESS_THRESHOLD` for tuning

### Edge Cases

1. **Very small fabric ops** (e.g., AllGather on a 64-byte tensor): Only a few packets. Delta may not exceed threshold. **Impact:** Timeout fires on small ops that take longer than the timeout. **Mitigation:** The dispatch progress counter (`get_cq_dispatch_progress`) handles this — small ops complete quickly and the dispatch counter advances when the op finishes.

2. **Bursty traffic**: Packets arrive in bursts with idle gaps. Some polling intervals see high delta, others see zero. **Impact:** The timeout clock ticks during idle gaps, but resets during bursts. As long as bursts arrive more frequently than the timeout duration (typically 5-60s), this works correctly.

3. **Counter wrap**: The packet counter is `uint32_t`. At the maximum practical rate (~10M increments/sec), it wraps every ~430 seconds. The `delta = raw_sum - state.last_raw_counter_sum` subtraction handles wrap correctly for unsigned arithmetic (as long as we sample more frequently than ~430s, which we do — every 100ms).

4. **Multiple concurrent ops**: Multiple AllGather ops on different fabric channels all contribute to the sum. The threshold is conservative enough that even a single active channel easily exceeds it.

### Complexity

- Lines changed: ~40 (new state struct, modified function body, env var parsing).
- New state: `FabricProgressState` struct (16 bytes, file-static).
- New dependencies: `<chrono>` (already included), `<cstdlib>` for `getenv` (already included).
- New env var: `TT_METAL_FABRIC_PROGRESS_THRESHOLD` (optional, default 10).
- Test surface: needs a test confirming timeout still fires when only keepalive traffic flows but dispatch is stuck.

---

## Tradeoffs Comparison

```
                          Design 1: MMIO-Only      Design 2: Rate-Based
                          ─────────────────────    ──────────────────────
SAFETY (infinite hang?)
  Non-MMIO read hang      ELIMINATED               ELIMINATED (also MMIO-only)
  Timeout handler hang    IMPOSSIBLE (PCIe only)   IMPOSSIBLE (PCIe only)

CORRECTNESS (extend for large ops?)
  AllGather 1GB           YES (MMIO ERISC active)  YES (delta >> threshold)
  AllGather 64B           YES (counter still moves) MAYBE (delta may be < threshold,
                                                    but dispatch counter handles it)

FALSE POSITIVE (keepalive prevents timeout?)
  Idle fabric + stuck     YES — keepalive traffic   NO — keepalive delta < threshold,
  dispatch                still resets clock         timeout fires correctly
                          *** CRITICAL WEAKNESS ***

COMPLEXITY
  Lines of code           ~5 changed               ~40 changed
  New state               None                     16-byte struct
  New env vars            None                     1 optional
  New concepts            None                     Rate thresholding

HARDWARE ROBUSTNESS
  N150 (single chip)      Works (1 MMIO chip)      Works (1 MMIO chip)
  N300 (1+1)              Works (reads chip 0)     Works (reads chip 0)
  T3K (4+4)               Works (reads 4 MMIO)     Works (reads 4 MMIO)
  Galaxy/TG (complex)     Works (reads MMIO set)   Works (reads MMIO set)

IMPLEMENTATION RISK
  Novel mechanism?        No — just a filter        Moderate — rate thresholding
                                                    needs careful threshold tuning
  Regression risk         Low — removes reads       Low-moderate — new state tracking
  Revert cost             Trivial                  Moderate
```

---

## Recommendation: Combine Both (Design 1 + Design 2)

**Design 1 alone is insufficient.** It solves the non-MMIO hang but not the false-extension problem. A system with enabled fabric and a genuinely stuck dispatch will never timeout because MMIO ERISC keepalive traffic continuously resets the clock.

**Design 2 alone is sufficient** (and includes MMIO-only reading as a safety measure), but it introduces a new concept (rate thresholding) that needs validation.

**Recommended implementation: Design 2 (which subsumes Design 1).**

The rate-based approach solves both problems:
1. Only reads MMIO chips (no hang risk)
2. Only signals progress when the packet rate exceeds the threshold (no keepalive false-extension)

**Implementation order:**
1. Start with Design 1 (5-line change). Get it merged and tested in CI. This immediately eliminates the non-MMIO hang regression.
2. Follow up with Design 2's rate logic on top of Design 1. This addresses the false-extension problem.

If time pressure requires a single PR, go with Design 2 directly — it's ~40 lines, self-contained, and solves both problems. The threshold of 10 packets/100ms is conservative and the env var provides an escape hatch for tuning.

**What NOT to do:** Ship Design 1 alone as the final solution. The false-extension vulnerability means CI tests with `TT_METAL_OPERATION_TIMEOUT_SECONDS=5` will never timeout on a genuinely stuck dispatch when fabric is enabled — this is worse than having no fabric progress extension at all.
