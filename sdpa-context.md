# Ring Joint SDPA: Developer Reference

Essential context for developing and extending the SDPA kernel.


## Current perf metrics

| config                  | math util | delta (%) |
| ----------------------- | --------- | --------- |
| baseline (KV chaining)  | 46.9%     | /         |
| + K-fwd chain           | 54.9%     | +8.0%     |
| + no K DRAM             | 55.1%     | +0.2%     |
| + no K unicast overhead | 56.8%     | +1.7%     |
| + no V DRAM             | 57.1%     | +0.3%     |
| + no V unicast overhead | 59.1%     | +2.0%     |
| + no Q DRAM             | 59.3%     | +0.2%     |


| config                      | Math Util (%) |
| --------------------------- | ------------- |
| baseline (KV chaining)      | 46.9          |
| + K fwd chain               | 54.9          |
| w/ mcast                    | 52.1          |
| w/ mcast, no write          | 53.3          |
| w/ mcast + light Q KV skip  | 51.2          |

Observation: multicast should be faster than unicast. Investigate why.


## Key Files

| File | Purpose |
|------|---------|
| `ring_joint_sdpa_program_factory.cpp` | Host setup, chain construction, work distribution |
| `ring_joint_sdpa_device_operation.cpp` | Parameter validation, constraints enforcement |
| `kernels/compute/ring_joint_sdpa.cpp` | Compute kernel entry point |
| `kernels/compute/compute_common.hpp` | Core attention loop (`sdpa_inner_loop`) |
| `kernels/dataflow/ring_joint_reader.cpp` | KV data movement, chaining protocol |
| `kernels/dataflow/ring_joint_writer.cpp` | Output write, mask generation |
| `kernels/q_chunk_remapping.hpp` | Zigzag index remapping |

---

## Core Concepts

### Two-Level Data Distribution

**Level 1 (Host → Device):** Before sharding, host reorders Q/K/V so each device gets one "light" chunk (early sequence, less causal work) + one "heavy" chunk (late sequence, more work).

```
4 devices, 8 sequence chunks:
  Reorder: [0,7,1,6,2,5,3,4]
  Device 0: orig [0,7], Device 1: orig [1,6], Device 2: orig [2,5], Device 3: orig [3,4]
```

**Level 2 (Kernel, within device):** Zigzag remapping interleaves light/heavy Q chunks across cores.

```cpp
// Even positions → forward from start, Odd positions → backward from end
q_chunk = (pos % 2 == 0) ? pos/2 : num_q_chunks - 1 - pos/2;
```

### Local Memory Layout

Each device's local sequence: first half = light (orig early), second half = heavy (orig late).

```
Local tiles:  [0 ─── N/2-1]  [N/2 ─── N-1]
                  LIGHT          HEAVY
              q_chunk < half   q_chunk >= half
```

`half_sequence = num_q_chunks / 2` is the boundary.

---

## Balanced Mode: Skip Rules

Three rules determine which work is skipped:

| Rule | Condition | Action | Why |
|------|-----------|--------|-----|
| **1** | `q_chunk < half_sequence` AND `ring_index < ring_id` | Skip Q chunk | Light Q doesn't need KV from later devices |
| **2** | `ring_index > ring_id` | `iter_num_kv_chunks /= 2` | Only need light half of earlier device's KV |
| **3** | `ring_index < ring_id` AND `!is_balanced` | Skip ring iteration | Non-balanced fallback |

**Visual (4 devices):**
```
           KV from:   Dev0[0,7]  Dev1[1,6]  Dev2[2,5]  Dev3[3,4]
Dev0  Q[0] light      ✓ causal    ✗          ✗          ✗
      Q[7] heavy      ✓ full      ✓ full     ✓ full     ✓ full
Dev3  Q[3] light      ✓ half      ✓ half     ✓ half     ✓ causal
      Q[4] heavy      ✓ half      ✓ half     ✓ half     ✓ causal
```

---

## Critical Constraint: q_chunk_size

```cpp
// ring_joint_sdpa_device_operation.cpp:115-117
TT_FATAL(!(args.is_balanced && (N_local / 2) % q_chunk_size != 0),
    "q_chunk_size must divide half of local q seq_len in balanced case");
```

**Why:** `half_sequence` must align with light/heavy boundary for zigzag to pair correctly.

```
N_local = 64 tiles, Light: 0-31, Heavy: 32-63

✓ q_chunk_size=4: half_sequence=8, chunks 0-7=light, 8-15=heavy (clean)
✗ q_chunk_size=6: half_sequence=5, chunk 5 spans 30-35 (crosses boundary)
```

---

## Chaining (Store-and-Forward)

Reduces DRAM bandwidth when multiple cores process same (batch, head).

### Topology
```
DRAM → [Injector] ──L1→L1──► [Middle] ──L1→L1──► [Sink]
```

### Protocol
```
Injector:                    Receiver:
1. Read from DRAM            1. Inc sender_sem (ready)
2. Wait sender_sem           2. Wait receiver_sem (data)
3. NOC write to receiver     3. Use data from L1
4. Set receiver_sem
```

### Two-Chain Architecture (MLA Mode)

When `NHK < NH` (K is shared across heads), the system uses **two separate chains**:

| Chain | Scope | Purpose | Semaphores |
|-------|-------|---------|------------|
| **K chain** | Per-batch | Single injector reads K, forwards to all cores in batch | `k_sender_sem`, `k_receiver_sem`, `k_valid_sem` |
| **V chain** | Per-head | Existing per-head chaining for V | `sender_sem`, `receiver_sem`, `valid_sem` |

**K chain data structures:**
```cpp
struct CoreKChainInfo {
    bool participates = false;
    bool is_injector = false;
    bool is_sink = false;
    uint32_t batch = 0;
    CoreCoord prev_physical = CoreCoord{0, 0};
    CoreCoord next_physical = CoreCoord{0, 0};
    uint32_t next_core_q_chunks = 0;  // For forwarding decision (unicast)
    // K multicast fields (for 2D mcast across full grid)
    bool use_k_mcast = false;
    CoreCoord mcast_start = CoreCoord{0, 0};      // Rectangle start (physical)
    CoreCoord mcast_end = CoreCoord{0, 0};        // Rectangle end (physical)
    CoreCoord injector_physical = CoreCoord{0, 0};  // Injector's physical coords
    uint32_t k_mcast_num_dests = 0;               // Receivers count (excludes self)
    uint32_t k_mcast_sender_wait = 0;             // Semaphore wait count
};
```

**K chain construction (per batch):**
1. Group active cores by batch
2. Sort by physical core order (row-major)
3. Find injector: first core with single head segment (not last core)
4. Build linear chain from injector forward to last core

### Chain Forwarding Logic

Both chains use `q_iter`-based forwarding (work completed count, not remapped chunk index):

```cpp
// V chain (head-level)
const bool should_forward_v = is_chain_participant && !is_sink &&
                              (nb == chain_batch && nq == chain_head) &&
                              (q_iter_local < next_core_q_chunks);

// K chain (batch-level, MLA mode)
const bool should_forward_k = k_is_chain_participant && !k_is_sink &&
                              (nb == k_chain_batch) &&
                              (q_iter_local < k_next_core_total_reads);
```

This is critical because:
- `q_iter` monotonically tracks how much work the core has done
- The remapped q_chunk index jumps around (zigzag pattern)
- Chain forwarding must be based on sequential progress, not logical chunk position

### Pair-based Work Distribution

When zigzag balancing is enabled, work is distributed in pairs of (light, heavy) Q chunks:
```cpp
if (enable_zigzag_balancing) {
    const uint32_t total_pairs = total_q_chunks / 2;
    cores_doing_extra_work = total_pairs % num_cores;
    base_chunks_per_core = (total_pairs / num_cores) * 2;
    extra_chunks_per_core = 2;  // Always add pairs, not singles
}
```

### Current Constraints

| Constraint | Location | Reason |
|------------|----------|--------|
| One V chain per core | `CoreChainInfo` struct | Single set of V chain metadata |
| One K chain per core | `CoreKChainInfo` struct | Single set of K chain metadata |
| Head spans ≥2 cores | Chain construction loop | Nothing to forward otherwise |
| Injector = single-head core | `work.head_work.size() == 1` | Avoid head boundary issues |
| Linear only, no wrap | Comment at line 817 | Prevents deadlock |

### Multicast Variants

**V chain (1D multicast):** When all V chain cores are on the same row with uniform work, the injector broadcasts to all receivers simultaneously via `noc_async_write_multicast`.

**K chain (2D multicast):** When `NHK < NH` (MLA mode) and `B == 1` (single batch), K uses 2D multicast across the entire compute grid:
- Injector is selected as the core with **maximum work** (most q_chunks)
- Physical bounds derived from logical grid corners (always rectangular)
- All receivers signal the injector's semaphore, then injector broadcasts to all
- Non-uniform work handled via **loop padding**: all cores iterate `max_q_per_core` times; padded iterations participate in K sync but skip Q/V work

**K mcast logging:** `log_info` shows mode and fallback reason:
```
K chain mode: mcast
K chain mode: unicast (B > 1 (multi-batch not supported))
```

### Known Issue: Non-MLA K Chaining Regression

**Problem:** When `NHK >= NH` (non-MLA case), K is no longer chained.

**Before K chain changes:** K was forwarded via V chain using `should_receive`:
```cpp
if (should_receive) {  // V chain condition applied to both K and V
    // Receive forwarded K chunk from previous core
```

**After K chain changes:** K uses `should_receive_k` which requires K chain to be active:
```cpp
if (should_receive_k) {  // Only true when k_is_chain_participant (NHK < NH)
    // Receive forwarded K chunk from K chain
```

Since K chain is only built when `NHK < NH`, non-MLA configs now read K from DRAM on every core.

**Fix:** Fall back to V chain semantics when K chain is inactive:
```cpp
const bool should_receive_k = k_is_chain_participant
    ? (!k_is_injector && (nb == k_chain_batch))
    : should_receive_v;  // Non-MLA: K chains with V

const bool should_forward_k = k_is_chain_participant
    ? (!k_is_sink && (nb == k_chain_batch) && (q_iter_local < k_next_core_total_reads))
    : should_forward_v;  // Non-MLA: K chains with V
```

---

## Debug Visualization

Debug logging helpers in `ring_joint_sdpa_program_factory.cpp` (enabled with `TT_METAL_LOGGER_LEVEL=Debug`):

| Function | Output |
|----------|--------|
| `print_per_core_work()` | Per-core Q chunk assignments with zigzag indices, light/heavy counts, and role tags `[INJ]`/`[RCV]`/`[SNK]` |
| `print_chains()` | V chains grouped by head with DRAM ratio |
| `print_k_chains()` | K chains grouped by batch (MLA mode) |

**Example output:**
```
Per-Core Workload (zigzag indices)
Core 0 (0,0)[INJ]: H:0-Q:[0,15]{L1+H1} H:1-Q:[0,15]{L1+H1}
Core 1 (1,0)[RCV]: H:1-Q:[1,14]{L1+H1} H:2-Q:[0,15]{L1+H1}
...
V-Chain(head=0: (0,0)[2:INJ]->(1,0)[2:RCV]->(2,0)[2:SNK], dram_ratio=0.33)
K-Chain(batch=0: (0,0)[4:INJ]->(1,0)[4:RCV]->(2,0)[4:SNK], dram_ratio=0.33)
```

---

## Build & Test

### Building
```bash
./build_metal.sh
```

### Performance Testing

**Primary unit test (MLA performance):**
```bash
pytest tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table[mla_100k]
```
Output includes a table with **Math Util** column — the primary performance metric.

**Note:** The `perf_table` test internally spawns a subprocess that runs Tracy profiling. Do NOT wrap it with `python -m tracy` — just run with pytest directly. Device profiler logs are generated at `generated/profiler/ttnn_ring_joint_sdpa_performance/reports/<timestamp>/`.

**Tracy profiling (for standalone tests):**

Use `python -m tracy` only for tests that don't handle profiling internally:
```bash
# Test 1: 131072 sequence length
python -m tracy -p -r -v -m pytest \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py::test_mla_sdpa[blackhole-balanced-rpxup-4x2-line-1link-no_trace-single_run-1-128-1-576-128-131072-256-128-q_bf16_kv_bf8]

# Test 2: 102400 sequence length
python -m tracy -p -r -v -m pytest \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py::test_mla_sdpa[blackhole-balanced-rpxup-4x2-line-1link-no_trace-single_run-1-128-1-576-128-102400-320-64-q_bf16_kv_bf8]
```

**Analyzing device profiler logs with `parse_tracy.py`:**
```bash
python parse_tracy.py <tracy_report_dir>
# Example:
python parse_tracy.py generated/profiler/ttnn_ring_joint_sdpa_performance/reports/2026_04_08_14_48_39/
```

Reads `profile_log_device.csv` and `ops_perf_results_*.csv`, outputs `analysis.md` with per-device kernel duration stats.

**Device profiler log structure (`profile_log_device.csv`):**
- Line 1: Metadata (`ARCH: blackhole, CHIP_FREQ[MHz]: 1350, ...`)
- Line 2: CSV header
- Columns: `PCIe slot` (device), `core_x`, `core_y`, `RISC processor type`, `zone name`, `type` (ZONE_START/ZONE_END), `time[cycles since reset]`

**Important:** Device profiler buffer is limited. Only early iterations are captured — zone counts are not meaningful, but **relative durations** are.

**Filtering by device (post-process):**
```bash
# Keep header + device 0 only
head -2 profile_log_device.csv > profile_log_device_slot0.csv
grep "^0," profile_log_device.csv >> profile_log_device_slot0.csv
```

**After each run:** Create `command_info.txt` in the report folder (`generated/profiler/reports/<timestamp>/`) with:
- Exact command used
- Branch name
- Commit hash

**After all runs:** Archive reports:
```bash
tar -czvf YY-MM-DD-bhlb-<branch>-<commit_short>.tar.gz generated/profiler/reports/*/
```

### Performance Baseline: DRAM Bound

Measurements from `mla_100k` test show the kernel is **DRAM access bound**, not compute or CCL bound:

| DRAM enabled | CCL enabled | Math Util (%) |
|--------------|-------------|---------------|
|     yes      |     yes     |     46.9      |
|     no       |     yes     |     59.3      |
|     no       |     no      |     59.3      |

Disabling DRAM access improves Math Util from 46.9% → 59.3%. CCL has no impact when DRAM is disabled. This motivates chaining optimizations to reduce DRAM bandwidth pressure.

---

## Development Targets

### 1. K Chaining for MLA

**Motivation:** In MLA, K is shared across all heads with shape `(1, 1, <token_count>, 576)`. Chains are already enabled, but each chain has its own injector — meaning multiple injector cores read the same K from DRAM. Since DRAM is the bottleneck, we want a single K read shared across all cores.

**Approach (two chains):**
- **K chain:** Single injector reads K, forwards to all cores (new)
- **V chain:** Existing per-head chaining remains unchanged

**Implementation phases:**

**Phase 1: Enable chain for K** ✅ COMPLETE
- Separate `CoreKChainInfo` struct with batch-level chaining
- Dedicated semaphores (`k_sender_sem`, `k_receiver_sem`, `k_valid_sem`)
- Linear unicast forwarding based on `q_iter` count
- Debug visualization via `print_k_chains()`

**Phase 2: 2D multicast for full grid** ✅ COMPLETE
- Injector (core with max work) broadcasts K to entire compute grid via `noc_async_write_multicast`
- Physical bounds computed from logical grid corners (handles harvested cores)
- Non-uniform work: all cores loop `max_q_per_core` times; padded iterations do K sync only
- Enabled when `NHK < NH` (MLA) and `B == 1` (single batch)
- Fallback to unicast with logged reason when conditions not met

**Phase 3: Light Q causal KV skip** ✅ COMPLETE
- During ring_iter 0 (causal, balanced zigzag), light Q chunks only need the first half of KV chunks
- All cores agree on light/heavy per q_iter (structural guarantee of pair-based zigzag)
- Reader and compute both reduce k_chunk loop bound for light q_iters: reader halves `iter_num_kv_chunks`, compute uses `light_kv_bound`
- Gated on `ring_iter == 0` (reader) / `is_causal` runtime flag (compute)
- Eliminates 30 of 120 KV iterations per core during ring_iter 0 (all pure discard)
- Math util: 51.2% (+4.3% over baseline, but below K unicast chain at 54.9% — mcast overhead still dominates)
- **Prior hang:** Original implementation gated on `!do_joint_kv` in the reader, which evaluated `false` on the last ring device (`ring_id == ring_size - 1`) even with `L=0`. Compute's gate (`iter_k_chunk_end <= num_local_k_chunks`) correctly evaluated `true`. Reader pushed 20 K/V, compute popped 10 → CB desync → K mcast hang. Fixed by using `ring_iter == 0` guard and `iter_num_kv_chunks /= 2` instead.

**Phase 4: Multi-batch K mcast** (future)
- Current limitation: K mcast only works for single batch (`B == 1`)
- Options: multiple injectors (one per batch), sequential mcast per batch, or hybrid approach

### 2. Improving Tests (existing target)

**Key invariants to test:**
- Host reorder + kernel zigzag produce correct attention output
- Skip rules (Rule 1, 2, 3) fire at correct conditions
- `half_sequence` boundary aligns with light/heavy split
- Work is balanced across devices (measure KV chunks processed)
- Chain forwarding synchronizes correctly with zigzag iteration

**Edge cases:**
- `num_q_chunks` odd vs even
- Single device (ring_size=1)
- `q_chunk_size` equals `N_local/2` (only 2 chunks)
- Joint tensors present vs absent
- Chaining with balanced mode enabled

### 3. Enabling Arbitrary q_chunk_size (existing target)

**Current blocker:** `ring_joint_sdpa_device_operation.cpp:115-117`
```cpp
TT_FATAL(!(args.is_balanced && (N_local / 2) % q_chunk_size != 0), ...);
```

**Problem:** When `q_chunk_size` doesn't divide `N_local/2`, some Q chunks contain mixed light/heavy data. The simple `q_chunk < half_sequence` check becomes incorrect.

**Required changes:**

1. **Track per-tile boundaries instead of per-chunk:**
   ```cpp
   // Instead of: q_chunk < half_sequence
   // Use: q_tile_start < N_local/2
   uint32_t q_tile_start = q_chunk * Sq_chunk_t;
   uint32_t q_tile_end = q_tile_start + Sq_chunk_t;
   bool is_pure_light = q_tile_end <= half_tiles;
   bool is_pure_heavy = q_tile_start >= half_tiles;
   bool is_mixed = !is_pure_light && !is_pure_heavy;
   ```

2. **Handle mixed chunks:** A chunk spanning the boundary needs partial processing:
   - Light portion: apply Rule 1 (skip when ring_index < ring_id)
   - Heavy portion: process all ring iterations

3. **Update skip logic in three places:**
   - `compute_common.hpp:1693` (Q skip)
   - `ring_joint_reader.cpp:220` (Q skip in reader)
   - `ring_joint_reader.cpp:206` (KV halving)

4. **Update work distribution:** `half_sequence` may no longer be integer. Use tile-based accounting.

---

## Quick Reference: Loop Structure

### Compute Kernel Loop
```cpp
// Outer: ring iterations (device-to-device)
for (ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
    ring_id = get_next_ring_id_and_sync();

    // Rule 2: halve KV if ring_index > ring_id
    iter_num_kv_chunks = (is_balanced && ring_index > ring_id)
                         ? num_kv_chunks/2 : num_kv_chunks;

    // Middle: Q chunks (distributed across cores via zigzag)
    for (q_iter = q_start; q_iter < q_end; ++q_iter) {
        q_chunk = remap_q_index(q_iter, num_q_chunks, use_zigzag);

        // Rule 1: skip light Q when ring_index < ring_id
        if (is_balanced && q_chunk < half_sequence && ring_index < ring_id)
            continue;

        // Inner: KV chunks
        for (k_chunk = 0; k_chunk < iter_num_kv_chunks; ++k_chunk) {
            // Attention computation
        }
    }
}
```

### Reader Kernel KV Flow (with K chain)
```cpp
// Loop padded to max_q_per_core for K mcast sync
for (q_iter = 0; q_iter < loop_q_count; ++q_iter) {
    const bool is_padded_iter = (q_iter >= q_per_core);

    for (k_chunk ...) {
        // K: receive from K chain or read from DRAM
        if (should_receive_k) {
            noc_semaphore_set(k_receiver_sem, INVALID);
            if (k_use_mcast) {
                // K mcast: signal injector's semaphore
                noc_semaphore_inc(k_injector_sender_sem_noc, 1);
            } else {
                // K unicast: signal previous core's semaphore
                noc_semaphore_inc(k_sender_sem_noc, 1);
            }
            noc_semaphore_wait(k_receiver_sem, VALID);
            if (!is_padded_iter) cb_push_back(cb_k, k_chunk_tiles);
        } else {
            read_block(K_dram, ...);  // Injector reads from DRAM
        }

        // K: forward (mcast or unicast)
        if (should_forward_k) {
            if (k_use_mcast) {
                // Wait for ALL receivers, then broadcast
                noc_semaphore_wait(k_sender_sem, k_mcast_sender_wait);
                noc_async_write_multicast(cb_k, k_mcast_addr, ...);
                noc_semaphore_set_multicast(k_valid_sem, k_mcast_sem_noc, ...);
            } else {
                // Unicast to next core
                noc_semaphore_wait(k_sender_sem, 1);
                noc_async_write(cb_k, next_core_k_addr, ...);
                noc_semaphore_set_remote(k_valid_sem, k_receiver_sem_noc);
            }
        }

        // Skip Q, V for padded iterations (K sync only)
        if (is_padded_iter) continue;

        // Q: download after K (on first K iteration)
        ...

        // V: receive/read and forward (unchanged)
        ...
    }
}
```

---

## Ring Iter 0 Causal Waste Analysis

### Why ring_iter 0 is much slower

During ring_iter 0 (local KV, causal), the caller sets:
```cpp
causality = true;                    // ring_iter == 0
balancing = (ring_index >= ring_id ? false : is_balanced);  // ring_index == ring_id → false
iter_num_kv_chunks = num_kv_chunks;  // ring_index > ring_id is false → no halving
```

So **none of the balanced skip rules fire**: no Q skip (Rule 1), no KV halving (Rule 2). Every Q chunk iterates all 20 KV chunks. The only relief is the causal discard loop in compute:

```cpp
if (k_chunk >= q_high_idx && is_causal) {
    cb_wait_front(cb_k_in, k_chunk_tiles);   // wait for reader
    cb_wait_front(cb_v_in, v_chunk_tiles);
    cb_pop_front(cb_k_in, k_chunk_tiles);    // discard
    cb_pop_front(cb_v_in, v_chunk_tiles);
    continue;
}
```

Each discard still costs a full K DRAM read + mcast + V DRAM read + chain forward. The reader does identical work for discarded and computed KV chunks.

### Loop structure (ring_iter 0, mla_100k)

**Reader:**
```
for q_iter in 0..loop_q_count:           # 6 (padded to max_q_per_core for K mcast)
    q_chunk = zigzag_remap(global_q_start + q_iter) % 20
    nb, nq = extract_batch_head(...)

    # Rule 1: q_chunk < 10 AND ring_index < ring_id
    # ring_iter 0 → ring_index == ring_id → NEVER FIRES

    for k_chunk in 0..20:                # all 20 KV chunks, no halving
        K: mcast_receive or DRAM_read
        K: mcast_forward (injector)
        if is_padded: continue           # K sync only

        if k_chunk == 0: Q: DRAM_read
        V: chain_receive or DRAM_read
        V: chain_forward
```

**Compute:**
```
for q_iter in q_start..q_end:            # 6 iterations
    q_chunk = zigzag_remap(q_iter) % 20
    q_high = q_chunk + 1                 # causal boundary (Sq == Sk → q_high = q_chunk + 1)

    # Balanced skip: balancing == false → NEVER FIRES

    for k_chunk in 0..20:
        if k_chunk >= q_high:            # CAUSAL DISCARD
            wait(K); pop(K)
            wait(V); pop(V)
            continue

        QK = Q @ K^T                    # actual compute
        apply_causal_mask(QK)
        out += softmax(QK) @ V
```

### Per-core waste (example: core 0)

Zigzag gives core 0 q_chunks `[0, 19, 1, 18, 2, 17]`:

| q_iter | q_chunk | q_high | compute K chunks | discarded K chunks |
|--------|---------|--------|------------------|--------------------|
| 0 | 0 (light) | 1 | `[0]` | `[1..19]` → 19 |
| 1 | 19 (heavy) | 20 | `[0..19]` | `[]` → 0 |
| 2 | 1 (light) | 2 | `[0..1]` | `[2..19]` → 18 |
| 3 | 18 (heavy) | 19 | `[0..18]` | `[19]` → 1 |
| 4 | 2 (light) | 3 | `[0..2]` | `[3..19]` → 17 |
| 5 | 17 (heavy) | 18 | `[0..17]` | `[18..19]` → 2 |
| | | | **63 compute** | **57 discards** |

120 total KV iterations, only 63 feed real compute. Each discard costs full DRAM + mcast + chain bandwidth.

### K mcast synchronization amplifies the waste

All cores are coupled by K mcast: injector waits for all receivers to signal "ready" before broadcasting each K chunk. The **heaviest Q chunk in any q_iter gates every core**. During q_iter 0, the injector broadcasts K chunks 0-19 at the rate the slowest consumer (a core with q_chunk 19, doing full matmul) can absorb them, while cores with light Q (q_chunk 0) idle-discard 19 of 20.

### Contrast with ring_iter 1+

| | ring_iter 0 | ring_iter 1+ |
|---|---|---|
| Q skip (Rule 1) | never fires (`balancing=false`) | light Q chunks skipped entirely |
| KV halving (Rule 2) | never fires | halved when `ring_index > ring_id` |
| Causal discard loop | ~47% of KV iterations wasted | no causal → no discards |
| DRAM reads | all K+V for all Q | only heavy Q, half K+V |

### Measured impact: ring_iter 0 is DRAM-bound, rest is compute-bound

| ring iters   | K mcast | K unicast | no DRAM + no FWD |
|--------------|---------|-----------|------------------|
| 0,1,2,3      | 48.4    | 50.4      | 58.2             |
| 1,2,3 only   | 58.0    | 58.1      | 59.7             |

Ring iters 1-3 alone reach 58% math util regardless of K delivery method (mcast vs unicast), nearly matching the 59.7% ceiling with DRAM and forwarding disabled entirely. This confirms ring iters 1-3 are **compute-bound** — DRAM/chain overhead is fully hidden.

Ring iter 0 drags the combined result down to 48-50% because the causal discard loop forces every core to read and forward KV data that is immediately thrown away, making it **DRAM-bound**. The 10% gap between combined (48.4) and ring 1-3 only (58.0) is entirely attributable to ring iter 0's wasted data movement.

### Constraint on fixing this

The K mcast and V chain require all cores to iterate the same number of KV chunks per q_iter. Truncating KV at the causal boundary per-Q-chunk would break chain synchronization — different Q chunks would need different KV counts within the same q_iter.

---

## Glossary

| Term | Definition |
|------|------------|
| `ring_index` | This device's position (0 to ring_size-1) |
| `ring_id` | Source device for current KV |
| `half_sequence` | `num_q_chunks / 2`, light/heavy boundary |
| Light Q | `q_chunk < half_sequence`, early original sequence |
| Heavy Q | `q_chunk >= half_sequence`, late original sequence |
| Injector | Chain head, reads from DRAM |
| Sink | Chain tail, receives but doesn't forward |
| `q_iter` | Iteration counter tracking work completed (used for chain forwarding) |

---

## Appendix A: Ring Iter 0 Execution Trace (mla_100k, Galaxy)

### Grid and work distribution

Galaxy config: grid = (12, 10), sdpa_cols = 11, `num_cores = 110`.

- Total Q chunks = 1 × 32 × 20 = 640, total pairs = 320
- 320 / 110 = 2 remainder 100
- First 100 cores: 3 pairs = 6 chunks; last 10 cores: 2 pairs = 4 chunks
- 3 heads × 10 pairs = 30 pairs → exactly 10 cores at 3 pairs each

### Per-core Q chunk assignments (first 10 cores, heads 0-2)

Zigzag remap is per-head. Even q_iters → light, odd q_iters → heavy (consequence of pair-based distribution).

```
            q_iter 0 (L)  q_iter 1 (H)  q_iter 2 (L)  q_iter 3 (H)  q_iter 4 (L)  q_iter 5 (H)
Core 0  H0: q= 0  qh= 1   q=19  qh=20   q= 1  qh= 2   q=18  qh=19   q= 2  qh= 3   q=17  qh=18
Core 1  H0: q= 3  qh= 4   q=16  qh=17   q= 4  qh= 5   q=15  qh=16   q= 5  qh= 6   q=14  qh=15
Core 2  H0: q= 6  qh= 7   q=13  qh=14   q= 7  qh= 8   q=12  qh=13   q= 8  qh= 9   q=11  qh=12
Core 3  H0→H1:  q= 9  qh=10   q=10  qh=11   q= 0  qh= 1   q=19  qh=20   q= 1  qh= 2   q=18  qh=19
Core 4  H1: q= 2  qh= 3   q=17  qh=18   q= 3  qh= 4   q=16  qh=17   q= 4  qh= 5   q=15  qh=16
Core 5  H1: q= 5  qh= 6   q=14  qh=15   q= 6  qh= 7   q=13  qh=14   q= 7  qh= 8   q=12  qh=13
Core 6  H1→H2:  q= 8  qh= 9   q=11  qh=12   q= 9  qh=10   q=10  qh=11   q= 0  qh= 1   q=19  qh=20
Core 7  H2: q= 1  qh= 2   q=18  qh=19   q= 2  qh= 3   q=17  qh=18   q= 3  qh= 4   q=16  qh=17
Core 8  H2: q= 4  qh= 5   q=15  qh=16   q= 5  qh= 6   q=14  qh=15   q= 6  qh= 7   q=13  qh=14
Core 9  H2: q= 7  qh= 8   q=12  qh=13   q= 8  qh= 9   q=11  qh=12   q= 9  qh=10   q=10  qh=11
```

### Algorithm: one q_iter (all cores in lockstep via K mcast)

```
for k = 0..19:
    ┌─ INJECTOR: read K[k] from DRAM
    │  ALL RECEIVERS: signal injector "ready"
    │  INJECTOR: wait for 9 signals, then multicast K[k] to all 10 cores
    └─ ← BARRIER: no core proceeds until K[k] is delivered to all ←

    // Each core independently:
    if k == 0: read Q from DRAM

    if k < q_high:           // within causal triangle
        V: read/receive via per-head V chain
        V: forward via chain
        COMPUTE: QK = Q @ K^T, mask, softmax, out += scores @ V
    else:                     // CAUSAL DISCARD
        V: read/receive via V chain    ← still happens
        V: forward via chain           ← still happens
        COMPUTE: wait(K); pop(K); wait(V); pop(V)   ← no math
```

### q_iter 0 (all light): compute vs discard grid

`C` = compute, `D` = discard. Each row = one K mcast step.

```
          C0    C1    C2    C3    C4    C5    C6    C7    C8    C9
         qh=1  qh=4  qh=7 qh=10  qh=3  qh=6  qh=9  qh=2  qh=5  qh=8
K[ 0]:    C     C     C     C     C     C     C     C     C     C
K[ 1]:    D     C     C     C     C     C     C     C     C     C
K[ 2]:    D     C     C     C     C     C     C     D     C     C
K[ 3]:    D     C     C     C     D     C     C     D     C     C
K[ 4]:    D     D     C     C     D     C     C     D     C     C
K[ 5]:    D     D     C     C     D     C     C     D     D     C
K[ 6]:    D     D     C     C     D     D     C     D     D     C
K[ 7]:    D     D     D     C     D     D     C     D     D     C
K[ 8]:    D     D     D     C     D     D     C     D     D     D
K[ 9]:    D     D     D     C     D     D     D     D     D     D     ← only C3 computes
K[10]:    D     D     D     D     D     D     D     D     D     D
  ⋮       D     D     D     D     D     D     D     D     D     D
K[19]:    D     D     D     D     D     D     D     D     D     D

Compute:  1     4     7    10     3     6     9     2     5     8  = 55 / 200 = 27.5%
```

K[10]-K[19]: pure waste — read, multicast, chain-forwarded, discarded by ALL cores.

### q_iter 1 (all heavy): compute vs discard grid

```
          C0    C1    C2    C3    C4    C5    C6    C7    C8    C9
        qh=20 qh=17 qh=14 qh=11 qh=18 qh=15 qh=12 qh=19 qh=16 qh=13
K[ 0]:    C     C     C     C     C     C     C     C     C     C
  ⋮       C     C     C     C     C     C     C     C     C     C
K[10]:    C     C     C     C     C     C     C     C     C     C
K[11]:    C     C     C     D     C     C     C     C     C     C
K[12]:    C     C     C     D     C     C     D     C     C     C
K[13]:    C     C     C     D     C     C     D     C     C     D
K[14]:    C     C     D     D     C     C     D     C     C     D
K[15]:    C     C     D     D     C     D     D     C     C     D
K[16]:    C     C     D     D     C     D     D     C     D     D
K[17]:    C     D     D     D     C     D     D     C     D     D
K[18]:    C     D     D     D     D     D     D     C     D     D
K[19]:    C     D     D     D     D     D     D     D     D     D

Compute: 20    17    14    11    18    15    12    19    16    13  = 155 / 200 = 77.5%
```

### Per-core totals (all 6 q_iters)

Each zigzag pair (light `l`, heavy `h`) has `l + h = 19`, so computes = `(l+1) + (h+1) = 21` per pair. All cores have 3 pairs → **63 computes, 57 discards, identical across all cores**.

```
                    Light q_iters    Heavy q_iters    Total
                    (0, 2, 4)        (1, 3, 5)
Compute / core:     ~18              ~45              63
Discard / core:     ~42              ~15              57
Total / core:       60               60               120
Compute fraction:   ~30%             ~75%             52.5%
```

52.5% matches the causal triangle ratio: `sum(1..20) / 20² = 210/400`. The waste is inherent to causality — the problem is that discards still cost full DRAM + mcast + chain bandwidth instead of being free skips.
