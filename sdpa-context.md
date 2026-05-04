# Ring Joint SDPA — Mechanics & Developer Reference

How `ring_joint_scaled_dot_product_attention` distributes work across a 2D mesh,
what each layer of remapping does, and the implementation surface for extending
the kernel. Use this for "I need to map an output position back to a compute
unit," "I need to reason about what a core actually runs in a given ring
iteration," or "I'm changing the chain / skip / mcast behavior and need to know
the moving parts."

---

## File map

| File | Role |
|---|---|
| `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py` | Top-level test, sharding setup, host-side reorder branch |
| `models/demos/deepseek_v3_d_p/tt/mla/utils.py` | `create_balanced_chunk_order`, `reorder_tensor_chunks` |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp` | Host: chain construction, position→core distribution, gates |
| `…/sdpa/device/ring_joint_sdpa_device_operation.cpp` | Parameter validation, constraints |
| `…/sdpa/device/kernels/compute/ring_joint_sdpa.cpp` | Compute kernel entry point |
| `…/sdpa/device/kernels/compute/compute_common.hpp` | Core attention loop (`sdpa_inner_loop`) |
| `…/sdpa/device/kernels/dataflow/ring_joint_reader.cpp` | KV data movement, chain receive, CB reservations |
| `…/sdpa/device/kernels/dataflow/ring_joint_writer.cpp` | Output write, mask generation |
| `…/sdpa/device/kernels/dataflow/ring_utils.hpp` | `RingIdSequencer`, `KCausalStraddleInfo` |
| `…/sdpa/device/kernels/dataflow/fused_op_receiver.hpp` | Sequencer init |
| `…/sdpa/device/kernels/q_chunk_remapping.hpp` | `linear_to_zigzag`, `remap_q_index` |
| `context/context.py`, `context/first.py` | Python mirrors of host/kernel logic — use these for offline reasoning instead of redoing math by hand |

---

## Two-axis sharding model

The mesh has two axes used together:

- `rp_axis`: ring-parallel along **sequence**. The ring exchange runs along this axis only.
- `up_axis`: head-parallel. Independent rings, one per up column. Small (typically 2).

Per device:
- **Q, V** are sharded on **both** seq (rp) and heads (up).
- **K** is sharded on seq (rp). When `nhk == 1`, K is **replicated across `up_axis`**
  (the single K head feeds all Q-head subsets) ⇒ devices in the same rp row but
  different up columns hold *identical* K. Anything pinned to a "local head index"
  therefore presents as **two symmetric global heads**, one per up column.

---

## Three layers of redistribution (in order)

These compose. Skipping or reordering any of them invalidates the "where does this
output come from" mapping.

### 1. Host: chunk reorder across rp devices  (gated by `is_balanced`)

The seq is split into `2·RP` chunks (so chunk_size = full_seq / (2·RP), e.g. 2048
when seq=16384, RP=4). `create_balanced_chunk_order` permutes them, e.g.
`[0, 7, 1, 6, 2, 5, 3, 4]` for RP=4. Each rp row gets one EARLY + one LATE chunk so
causal workload across rows is ~50% / 50% / 50% / 50% instead of 12.5 / 37.5 / 62.5 /
87.5%. Without balancing, each device gets one contiguous seq slab and the row index
== the rp ring_index.

```
4 devices, 8 sequence chunks:
  Reorder: [0,7,1,6,2,5,3,4]
  Device 0: orig [0,7], Device 1: orig [1,6], Device 2: orig [2,5], Device 3: orig [3,4]
```

Per-device local layout under balancing: positions `[0, chunk_size)` are the early
("light") chunk; `[chunk_size, 2·chunk_size)` are the late ("heavy") chunk.

```
Local tiles:  [0 ─── N/2-1]  [N/2 ─── N-1]
                  LIGHT          HEAVY
              q_chunk < half   q_chunk >= half
```

`half_sequence = num_q_chunks / 2` is the boundary.

### 2. Host: position → core distribution  (`program_factory.cpp` ~830–895)

Treat `total_q_chunks = batch · per_device_nh · num_q_chunks_per_head` as a flat list
of *positions*. Split across `NUM_CORES`. The unit of split depends on zigzag:

- **No zigzag**: unit = single chunk. `extra=1`,
  `cores_with_extra = total_q_chunks % NUM_CORES`,
  `base = total_q_chunks // NUM_CORES`.
- **Zigzag** (paired): unit = pair of positions. `extra=2`,
  `total_pairs = total_q_chunks / 2`,
  `cores_with_extra = total_pairs % NUM_CORES`,
  `base = (total_pairs // NUM_CORES) * 2`. Each core therefore owns whole
  light/heavy pairs.

```cpp
if (enable_zigzag_balancing) {
    const uint32_t total_pairs = total_q_chunks / 2;
    cores_doing_extra_work = total_pairs % num_cores;
    base_chunks_per_core = (total_pairs / num_cores) * 2;
    extra_chunks_per_core = 2;  // Always add pairs, not singles
}
```

In both cases: first `cores_with_extra` cores get `base + extra` chunks; the rest
get `base`. Position layout: `pos = local_head * num_q_chunks_per_head + pos_in_head`,
so a core typically owns a contiguous position run within one head (possibly
straddling one head boundary).

### 3. Kernel: position → q_chunk  (`q_chunk_remapping.hpp::remap_q_index`)

`use_zigzag = false` ⇒ identity. Cores process q_chunks in linear order ⇒ a core near
position 0 gets all-light q_chunks, a core near the end gets all-heavy. Causal
workload imbalance is exposed at the per-core level.

`use_zigzag = true` (gated on `is_balanced && is_causal && num_q_chunks % 2 == 0`)
⇒ each core's runtime order is *light, heavy, light, heavy, …*  Each pair is
one-light-plus-one-heavy and total work per core is roughly equal.

Concrete formulas (`N = num_q_chunks`):

- **forward** `pos → q_chunk`:
  `pos % 2 == 0` ⇒ `q_chunk = pos / 2`;
  `pos % 2 == 1` ⇒ `q_chunk = N - 1 - pos / 2`.
- **inverse** `q_chunk → pos` (use when starting from a tensor diff):
  `q_chunk < N/2` ⇒ `pos = q_chunk * 2`;
  `q_chunk >= N/2` ⇒ `pos = (N - 1 - q_chunk) * 2 + 1`.

The **causal frontier** of a core's work is its *last* zigzag pair, which contains
its largest q_chunk index — that's where most masking and skip behaviour lives.

---

## Ring exchange: `RingIdSequencer` (Linear topology)

NOT a simple rotation. Behaviour depends on a device's position in the line:

- `ring_index = 0` (start of line): all "backward" — sees `ring_ids = [0, 1, 2, 3, …]`
- `ring_index = ring_size-1` (end of line): all "forward" — sees `[N-1, …, 1, 0]`
- middle devices: **alternate directions**.

Per-device write counts (Linear topology — note the swap):

```
num_targets_forward      = ring_size - 1 - ring_index
num_targets_backward     = ring_index
forward_writes_expected  = num_targets_backward     # ← swap
backward_writes_expected = num_targets_forward      # ← swap
```

Direction-switch rule on each iteration:

- transfer 0: emit `ring_index`. If `expected[curr_dir] == 0`, flip direction.
- transfer k>0: increment `received[curr_dir]`; emit `(ring_index ± received) %
  ring_size` depending on direction; then **switch to the other direction iff
  `received[other] < expected[other]`** (otherwise stay).

For RP=4, the iteration sequences are:

| ring_index | iteration sequence (ring_ids seen) |
|---|---|
| 0 | [0, 1, 2, 3] |
| 1 | [1, 2, 0, 3] |
| 2 | [2, 3, 1, 0] |
| 3 | [3, 2, 1, 0] |

⇒ Never assume "iteration N gets neighbour-N." Use the sequencer (kernel) or its
python mirror in `context/context.py:get_ring_id_sequence` /
`first.py:get_ring_id_sequence_for_device`.

### What a `ring_id` *carries*

- **balanced**: each ring_id is a **pair** (early chunk, late chunk). One iteration
  can therefore deliver two K/V slices with *different* causal statuses ⇒
  partial-and-skip or full-and-partial mixes can appear off the diagonal.
  `KCausalStraddleInfo` (`ring_utils.hpp:162-171`) is what handles per-iteration
  straddle.
- **no_balancing**: each ring_id is one contiguous K block of `seq/RP` tokens. The
  causal status of the whole iteration is exactly:
  - `rid < my ring_index` ⇒ **full**
  - `rid == my ring_index` ⇒ **partial** (diagonal)
  - `rid > my ring_index` ⇒ **skip**

### Inner granularity

Even when an iteration's coarse status is "full" or "partial", masking actually runs
at **k_chunk_size** (e.g. 128) granularity. A 4096-token K block is 32 inner
k-chunks. For a Q chunk on the diagonal, only inner k-chunks up to the q chunk's end
position are non-fully-masked; the rest of the block is skipped. Helpers:
`first.py:num_valid_k_chunks_for_q`.

---

## Balanced mode: skip rules

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

## Critical constraint: q_chunk_size

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

## Mapping `(head_id, seq_pos)` → `(device, core)`

The recipe (balanced; for no_balancing skip steps 2–3 and use linear seq slabs):

1. **up column** = `head_id // per_device_nh`.
2. **original chunk** = `seq_pos // chunk_size`.
3. **reordered position** = `chunk_order.index(original_chunk)`.
4. **rp row** = `reordered_pos // 2` (each row owns two reorder positions).
5. **local seq pos** = `(reordered_pos % 2) * chunk_size + (seq_pos % chunk_size)`.
6. **local q_chunk** = `local_seq_pos // q_chunk_size`.
7. **linear position** = `local_head * num_q_chunks_per_head + zigzag_inverse(local_q_chunk)`.
8. **core** = position-split using `(cores_with_extra, base, extra)`.

Don't open-code this. Use:

- `context/context.py:find_device_and_core_for_q(head_id, seq_pos)`
- `find_core_for_local_q(local_head, local_q_chunk)`
- `q_chunk_to_linear_pos`, `linear_pos_to_q_chunk` (zigzag and inverse)
- `analyze_q_chunk_ring_iterations(device, local_q_chunk)` — what every iteration
  delivers and its causal status

The python mirror is the authoritative offline source of truth; it's been verified
against `RingIdSequencer` traces and `program_factory.cpp` distribution.

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

### Two-chain architecture (MLA mode)

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

### Chain forwarding logic

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

### Current constraints

| Constraint | Location | Reason |
|------------|----------|--------|
| One V chain per core | `CoreChainInfo` struct | Single set of V chain metadata |
| One K chain per core | `CoreKChainInfo` struct | Single set of K chain metadata |
| Head spans ≥2 cores | Chain construction loop | Nothing to forward otherwise |
| Injector = single-head core | `work.head_work.size() == 1` | Avoid head boundary issues |
| Linear only, no wrap | Comment at line 817 | Prevents deadlock |

### Multicast variants

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

### Known issue: non-MLA K chaining regression

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

## Reasoning rules of thumb

1. **Three-layer composition is real.** Output → seq → rp shard (with reorder) →
   local seq → local q_chunk → position (with zigzag) → core. Skip a step and your
   answer is wrong. Use the helpers.
2. **Run ring iteration math through the sequencer.** "Iteration 1 = neighbour 1"
   only holds for `ring_index=0`. For middle devices it's wrong.
3. **Output location ≠ error origin.** The q_chunk where a diff *appears* is just
   where the accumulator was finalised; the value flows from every iteration's
   compute. To attribute by iteration, dump per-iteration accumulators
   (running max, running sum, output) before the final softmax norm.
4. **`nhk == 1` replicates K across up_axis.** Anything that depends on a local head
   index will surface as two symmetric global heads (one per up column). Don't look
   for a per-up-column root cause.
5. **balanced and zigzag are coupled.** Both are gated by `is_balanced`; you cannot
   exercise one without the other from pytest. Code that's only live in this regime
   is the pair-based ring path + `KCausalStraddleInfo` + zigzag-aware core
   distribution.
6. **The causal frontier is at a core's *last* position.** Especially under zigzag,
   the heaviest q_chunk a core processes is the last one it sees, and any
   end-of-work cleanup or boundary handling lives there.
7. **Watch CB reservation sizes in padded iterations.** A reader iteration that
   handles two slices (own + chained) needs the CB reserved for both before either
   producer fires; otherwise you race.

---

## Debug visualization

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

## Build & test

### Building
```bash
./build_metal.sh
```

### Performance testing

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

### Performance baseline: DRAM bound

Measurements from `mla_100k` test show the kernel is **DRAM access bound**, not compute or CCL bound.

---

## Pytest knobs

The test parametrizes `is_balanced`, `is_causal`, `n_iters` (with `n_iters>1` +
`pcc_check` enabling determinism comparison), `trace_enabled`, `num_links`,
`skip_check`, `q_dtype`/`kv_dtype`, mesh shape, and topology. Filter with `-k` to
pin a single combination. Dump-output hooks live near the end of `run_ring_joint_sdpa`
in the test (commented by default) and serialise expected/actual/diff tensors to
`sdpa_determinism_debug/` for offline analysis with `context/determinism.py`.

---

## Development targets

### 1. K chaining for MLA

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

### 2. Improving tests

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

---

## Quick reference: loop structure

### Compute kernel loop
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

### Reader kernel KV flow (with K chain)
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
| `rp_axis` | Ring-parallel axis along sequence; ring exchange runs along this |
| `up_axis` | Head-parallel axis; independent rings, one per up column |
| `ring_index` | This device's position (0 to ring_size-1) |
| `ring_id` | Source device for current KV in a ring iteration |
| `half_sequence` | `num_q_chunks / 2`, light/heavy boundary |
| Light Q | `q_chunk < half_sequence`, early original sequence |
| Heavy Q | `q_chunk >= half_sequence`, late original sequence |
| Injector | Chain head, reads from DRAM |
| Sink | Chain tail, receives but doesn't forward |
| `q_iter` | Iteration counter tracking work completed (used for chain forwarding) |

---

## Appendix A: Ring iter 0 execution trace (mla_100k, Galaxy)

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

---

## Appendix B: Reverse-K algorithm (proposed optimization)

### The real problem: mcast barrier synchronization

The issue with discards is **not primarily DRAM bandwidth** — it's the **synchronization overhead** of K mcast.

K mcast operates with a barrier:
1. Injector waits for ALL cores to signal "ready"
2. Injector broadcasts K to all cores
3. ALL cores must participate, even those that will discard

This means execution is **serialized to the slowest path**:
- Each q_iter has 20 K mcast steps (K0 through K19)
- 6 q_iters × 20 K steps = **120 synchronized barrier cycles**
- Even though actual compute work is only 63 ops per core (3 pairs × 21 ops/pair)

Cores processing light Q (e.g., Q0 needing only K0) must:
- Wait at barrier for K1 mcast → discard
- Wait at barrier for K2 mcast → discard
- ...
- Wait at barrier for K19 mcast → discard

The discards aren't just wasted bandwidth — they're **forced idle time** at synchronization barriers.

### The solution: reverse-K for heavy Q

Process K in **opposite directions** for light vs heavy Q:
- **Light Q** (Q0, Q1, ...): K **forward** (K0 → K1 → K2 → ...)
- **Heavy Q** (Q19, Q18, ...): K **reverse** (K19 → K18 → ... → K0)

With zigzag pairing (light + heavy together), at any timestamp:
- Some cores process light Q, moving forward through K
- Other cores process heavy Q, moving backward through K
- **All cores have useful work** — no one is waiting-and-discarding

### Execution pattern

The pattern has three phases per 21-timestamp cycle:

```
Phase 1: Fwd only (t=0)
t0:   K0 only        ← fwd cores start, rev cores not yet active

Phase 2: Fwd + Rev overlap (t=1 to t=10)
t1:   K1  (fwd) | K19 (rev)   ← 2 K reads needed
t2:   K2  (fwd) | K18 (rev)
...
t10:  K10 (fwd) | K10 (rev)   ← convergence, same K for both

Phase 3: Rev only (t=11 to t=20)
t11:  K9  (rev only)   ← fwd cores finished (light Q causal boundary reached)
t12:  K8  (rev only)
...
t20:  K0  (rev only)

t21:  K0  (fwd) ...    ← next cycle starts
```

- **Phase 1 & 3**: 1 K read per timestamp
- **Phase 2**: 2 K reads per timestamp (memory-intensive)
- Light Q cores finish early (small causal boundary), heavy Q cores continue alone

### Benefits

| Metric | Current | Reverse-K |
|--------|---------|-----------|
| Barrier cycles per core | 120 | 63 |
| Useful work per cycle | ~52.5% | 100% |
| Cores idle at barrier | Many | None |

The total compute work (63 ops/core) is unchanged, but it now happens in **63 synchronized steps** instead of 120. The mcast barrier still exists, but every barrier cycle has all cores doing real work.

### Visualization script

`ring_iter0_trace.py` generates CSV files showing:
- Per-core operation sequence with reverse-K ordering
- `DISTINCT_KV` row: actual K-V pairs used per timestamp (max 2)
- `IDEAL_KV` row: the repeating 21-cycle pattern

Run: `python3 ring_iter0_trace.py`

### Implementation considerations

**K mcast changes:**
- Injector reads K chunks per timestamp: K_fwd and/or K_rev
- Three phases per 21-cycle:
  - **t=0**: K0 only (fwd starts, rev not yet active) — 1 read
  - **t=1 to t=10**: K_fwd and K_rev both active — 2 reads
  - **t=11 to t=20**: K_rev only (fwd cores finished, light Q has small causal boundary) — 1 read
- Net: ~11 timestamps need 2 K reads, ~10 timestamps need 1 K read per cycle

**Cost/benefit analysis (double-buffering model):**

Compute (Tc) and data transfer (Td) happen in parallel. Cycle time = max(Tc, Td).

**Data transfer components:**
```
Td = Tk_mcast + Tk_sem + Tk_dram + Tv_dram + Tv_fwd
```

**Baseline (no causality, compute-bound):**
```
Tc > Tk_mcast + Tk_sem + Tk_dram + Tv_dram + Tv_fwd
```
When this holds, data transfer is hidden behind compute → cycle time = Tc.

**Current algorithm problem:**
- Discard cycles still pay full Td (barrier sync, mcast, chain forwarding)
- But Tc = 0 for discards (no useful compute)
- Cycle time = Td for discards → pure overhead

**Reverse-K algorithm during phase 2 (t=1 to t=10):**
```
Td_phase2 = 2*Tk_mcast + Tk_sem + 2*Tk_dram + 2*Tv_dram + 2*Tv_fwd
```
Semaphore (Tk_sem) stays 1x — one barrier sync, just more data through it.

**Key question:** Does Tc still dominate during phase 2?
```
Tc > 2*Tk_mcast + Tk_sem + 2*Tk_dram + 2*Tv_dram + 2*Tv_fwd
```
- **If yes** → still compute-bound, 2x data hidden, cycle time = Tc
- **If no** → becomes data-bound during phase 2, cycle time = Td_phase2

**Phase comparison:**

| Phase | Cycles | Td | Tc | Cycle Time |
|-------|--------|----|----|------------|
| Current (compute) | 63 | 1x | Tc | max(Tc, Td) |
| Current (discard) | 57 | 1x | 0 | Td (pure overhead) |
| Reverse-K (phase 1 & 3) | 11 per 21 | 1x | Tc | max(Tc, Td) |
| Reverse-K (phase 2) | 10 per 21 | **2x** | Tc | max(Tc, 2×Td) |

**SRAM constraints:**
- Need space for 2 K chunks simultaneously
- Options:
  - Reduce K chunk size to fit both (e.g., 160 → 80 tiles)
  - Require double buffering for K (ping-pong between 2 K slots)
  - Disable Q double buffering to free SRAM for second K slot
- CB sizing for K needs adjustment

**V chain changes:**
- V also processed in reverse order for heavy Q
- V chain may need to:
  - Read 2 V chunks per timestamp (fwd + rev)
  - Forward 2x data down the chain
- Alternative: separate V chains for fwd/rev paths

**Numerical considerations:**
- Online softmax accumulates scores incrementally
- Reverse K order changes accumulation sequence
- Should be mathematically equivalent (addition is commutative), but verify numerical stability

**Open questions:**
1. **Injector selection**: Currently injector = core with max work. With 2 K reads, does selection criteria change?
2. **Ring iterations > 0**: Reverse-K designed for ring_iter=0 (local causal). How does it interact with KV from other devices?
3. **Q reuse**: Q loaded once per q_iter, unchanged. But if K chunk size shrinks, more K iterations per Q — does this affect Q CB?
4. **Mask generation**: Causal mask indices must account for reverse K iteration order
