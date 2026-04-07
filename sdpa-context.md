# Ring Joint SDPA: Developer Reference

Essential context for developing and extending the SDPA kernel.

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

**Tracy profiling (critical for validating changes):**

Run on Blackhole loud box before and after changes:
```bash
# Test 1: 131072 sequence length
python -m tracy -p -r -v -m pytest \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py::test_mla_sdpa[blackhole-balanced-rpxup-4x2-line-1link-no_trace-single_run-1-128-1-576-128-131072-256-128-q_bf16_kv_bf8]

# Test 2: 102400 sequence length
python -m tracy -p -r -v -m pytest \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py::test_mla_sdpa[blackhole-balanced-rpxup-4x2-line-1link-no_trace-single_run-1-128-1-576-128-102400-320-64-q_bf16_kv_bf8]
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

**Phase 3: Multi-batch K mcast** (future)
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
