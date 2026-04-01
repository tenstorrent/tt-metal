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
1. Read K from DRAM          1. Inc sender_sem (ready)
2. Wait sender_sem           2. Wait receiver_sem (data)
3. NOC write to receiver     3. Use K from L1
4. Set receiver_sem
```

### Current Constraints

| Constraint | Location | Reason |
|------------|----------|--------|
| **Disabled in balanced mode** | `program_factory.cpp:827` | Work distribution complexity |
| One chain per core | `CoreChainInfo` struct | Single set of chain metadata |
| Head spans ≥2 cores | Chain construction loop | Nothing to forward otherwise |
| Injector = single-head core | `work.head_work.size() == 1` | Avoid head boundary issues |
| Linear only, no wrap | Comment at line 817 | Prevents deadlock |

### Multicast Variant
When all chain cores on same row: injector broadcasts to all receivers simultaneously.

---

## Development Targets

### 1. Improving Tests

**Key invariants to test:**
- Host reorder + kernel zigzag produce correct attention output
- Skip rules (Rule 1, 2, 3) fire at correct conditions
- `half_sequence` boundary aligns with light/heavy split
- Work is balanced across devices (measure KV chunks processed)

**Edge cases:**
- `num_q_chunks` odd vs even
- Single device (ring_size=1)
- `q_chunk_size` equals `N_local/2` (only 2 chunks)
- Joint tensors present vs absent

### 2. Enabling Chaining for Causal Zigzag

**Current blocker:** `program_factory.cpp:827`
```cpp
if (segments.size() < 2 || args.is_balanced) {
    continue;  // Chaining disabled for balanced
}
```

**Challenges:**
1. **Non-uniform iteration counts:** In balanced mode, light Q skips some ring iterations (Rule 1). Chain cores must stay synchronized.

2. **Half-KV iterations:** When `ring_index > ring_id`, only half KV is processed. All chain cores must agree on `iter_num_kv_chunks`.

3. **Mixed light/heavy on same core:** With zigzag, a core may have both light and heavy Q chunks. Chain participation depends on which Q is being processed.

**Possible approaches:**
- Per-Q-chunk chain participation (dynamic)
- Restrict chaining to heavy-only cores
- Separate chain metadata for light vs heavy Q

### 3. Enabling Arbitrary q_chunk_size

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
