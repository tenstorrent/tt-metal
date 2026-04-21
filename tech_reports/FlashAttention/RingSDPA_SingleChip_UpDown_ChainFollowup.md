# Follow-up — Wire KV Chain Forwarding into Single-Chip UP/DOWN Proxy

Companion to `RingSDPA_SingleChip_UpDown_Plan.md`. The plan's Step 5 explicitly
skipped chain construction when `flatten_work=true` (the host still pushes 14
zero chain args; reader/writer/compute branches are no-ops). That delivered
correct math but leaves the proxy ~20 pp below `ring_joint_sdpa`'s non-diag
per-iter math util (31–35 % vs 52–56 %).

Root cause: `ring_joint_sdpa` uses L1→L1 store-and-forward of K/V across the
cores that share a (batch, head). One "injector" core reads K/V from DRAM once,
forwards to the other "receiver" cores in the chain via mcast (or unicast) —
cutting DRAM K/V reads by roughly the chain length. For MLA 100k (32 heads
across 110 cores) typical chain length is 3–6, and K/V fetch dominates the
cycle budget since `Sq_chunk_t = Sk_chunk_t = 5` tiles is small and `DHt = 18`
tiles is large. Without chain, every core re-reads K/V from DRAM independently.

Ring kernel inheritance isn't the gap — streaming compute is ineligible for
MLA 100k on both sides (`Sk_chunk_t % (dst_size / qk_out_subblock_h) ≠ 0`), and
the per-K-iter compute body in `sdpa_inner_loop` is the same STANDARD vs RING.
The only structural difference that affects throughput is chain forwarding.

## Goal

Enable KV chain forwarding for the `flatten_work && !is_causal` path in
single-chip SDPA so the UP/DOWN proxy reaches ring_joint's ~52–56 % math util
on MLA 100k. Causal `flatten_work` (iter-0) stays unchanged — chain is already
disabled for causal in both codebases.

## Reference implementation

`ring_joint_sdpa_program_factory.cpp` (flat chain build, lines 766–904) and
`ring_joint_reader.cpp` (per-slot forwarding loop, lines 221–405) are the
exact working template. The plan here is to port their flat-chain topology
build into the single-chip factory and widen the reader's existing chain
branches to also fire in `SDPA_FLAT_WORK` slots.

## Execution steps

### Step 1 — Lift the `!flatten_work` gate

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:911`

```cpp
// Was (current):
if (!is_causal && !is_chunked && !flatten_work) {
    head_segments.resize(total_heads);
    // ... hierarchical core_work population + chain build ...
}

// Should be:
if (!is_causal && !is_chunked) {
    head_segments.resize(total_heads);
    // ... population branches on flatten_work (Step 2) ...
    // ... chain build + mcast eligibility run unchanged ...
}
```

### Step 2 — Branch the population phase on `flatten_work`

File: `sdpa_program_factory.cpp`, inside the chain block (around line 920, the
"First pass: Record work distribution for each core" loop).

**Non-flat (existing)**: keep the hierarchical `local_batch_*, local_nh_*,
local_q_*` triple-loop that pushes one `CoreHeadWork` per `(b, h)` pair.

**Flat**: each core owns `[global_q_start, global_q_start + global_q_count)`
in `B*NQH*q_num_chunks` space. Decompose it into per-head segments the same
way `ring_joint_sdpa_program_factory.cpp:815-839` does:

```cpp
if (flatten_work) {
    uint32_t flat_chunk = global_q_start;
    uint32_t remaining = global_q_count;
    while (remaining > 0) {
        const uint32_t head_index = flat_chunk / q_num_chunks;       // row-major: batch * NQH + head
        const uint32_t q_chunk_idx = flat_chunk % q_num_chunks;
        const uint32_t batch = head_index / NQH;
        const uint32_t head  = head_index % NQH;
        const uint32_t chunk_capacity_in_head = q_num_chunks - q_chunk_idx;
        const uint32_t chunk_take = std::min(remaining, chunk_capacity_in_head);

        work.head_work.push_back(CoreHeadWork{
            .batch = batch,
            .head = head,
            .q_chunk_start = q_chunk_idx,
            .q_chunk_count = chunk_take,
        });
        const uint32_t head_id = batch * NQH + head;
        if (head_id < head_segments.size()) {
            head_segments[head_id].push_back(HeadSegmentRef{
                .core_idx = i, .head_work_index = static_cast<uint32_t>(work.head_work.size() - 1)});
        }
        remaining -= chunk_take;
        flat_chunk += chunk_take;
    }
} else {
    // existing hierarchical push
}
```

**Note the ordering convention.** The existing flat distribution in
`sdpa_program_factory.cpp:396` lays out `total_q_chunks = B * NQH * q_num_chunks`
and decomposes with `nb = _decoded.nb, nq = _decoded.nq` via
`decompose_flat_q_index` in `dataflow_common.hpp`. Double-check that the
`(flat_chunk / q_num_chunks) → head_index; head_index / NQH → batch` convention
here matches — otherwise head_ids will be mis-assigned and chains will build
across the wrong (batch, head) pairs. In ring_joint and the existing
`decompose_flat_q_index` implementation this ordering matches, but verify by
running a small-shape accuracy test before running perf.

**DOWN proxy caveat.** DOWN reduces `total_q_chunks` to
`B * NQH * (q_num_chunks / 2)` (plan Step 7). The head_span for DOWN is
`q_num_chunks / 2` (effective Q chunks), and the decomposed `q_chunk_start`
values are in the heavy half of the Q range. Either:
- use `q_num_effective = is_proxy_down ? q_num_chunks / 2 : q_num_chunks` when
  decomposing, mirroring the reader/writer/compute handling (plan Step 9/10/12),
- or apply the heavy-half offset to `q_chunk_start` after decomposition.

Both work; pick whichever keeps the chain bookkeeping closest to what the
reader sees at runtime.

### Step 3 — Chain build + mcast eligibility: unchanged

The subsequent passes (`Second pass: Build chains for heads spanning multiple
cores`, lines 969-1353, and the mcast eligibility pass) operate purely on
`core_work[].head_work[]` and `head_segments[]`. They don't care whether those
were built from hierarchical or flat ranges. Nothing to change.

The existing DRAM-channel-spreading heuristic and descending-q-count injector
selection will work as-is for flat too.

### Step 4 — Widen reader chain branches to run in `SDPA_FLAT_WORK`

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`

Currently the per-slot chain conditions at line 420-428 compile in both flat
and non-flat branches (they're outside the `#if defined(SDPA_FLAT_WORK)`
block), but `q_iter` means different things:

- Non-flat: `q_iter` is 0-indexed within the core's `(b, h)` head loop.
- Flat: `q_iter = _gq` is 0-indexed across the core's whole flat range, which
  may straddle multiple heads.

The `should_forward` check `(q_iter < next_core_q_chunks)` is correct for
non-flat (it limits forwarding to the first N slots of this head on this
core) but wrong for flat (a core's head_work may start at a non-zero
`_gq` offset, so the 0-based comparison is off by the prior-heads' slot
count).

**Fix (mirrors `ring_joint_reader.cpp:248-252`):**

Inside the flat branch, maintain a `q_iter_local` that resets at every `(nb,
nq)` transition. Simplest placement: reuse the existing `prev_nb_flat`
transition detector in `reader_interleaved.cpp:307-316` to also reset a
`q_iter_local` counter, then use `q_iter_local` in `should_forward` instead
of `q_iter`:

```cpp
#if defined(SDPA_FLAT_WORK)
uint32_t q_iter_local = 0;
uint32_t prev_head_id = static_cast<uint32_t>(-1);
// ...
for (uint32_t _gq = 0; _gq < global_q_count; ++_gq) {
    // ... existing decompose ...
    const uint32_t head_id = nb * NQH + nq;
    if (head_id != prev_head_id) {
        q_iter_local = 0;
        prev_head_id = head_id;
    } else {
        q_iter_local++;
    }
    const uint32_t q_iter = _gq;             // unchanged: used for cb bookkeeping
    // ... and propagate q_iter_local into should_forward below ...
#endif
```

Then in the common body:

```cpp
if constexpr (!is_causal) {
    should_forward = is_chain_participant && !is_sink &&
                     (nb == chain_batch && nq == chain_head) &&
                     (q_iter_local < next_core_q_chunks);
    should_receive = is_chain_participant && !is_injector &&
                     (nb == chain_batch && nq == chain_head);
}
```

For the non-flat branch, `q_iter_local = q_iter` (already equivalent).

### Step 5 — UP proxy: chain forwards only half of K

UP caps the K loop at `k_num_chunks / 2` (plan Step 9 / 11). The chain
forwarding loop must mirror this cap — otherwise the injector forwards K
chunks the receivers will never wait on, or receivers wait for K chunks the
injector never sends. Gate the K loop with the same `k_chunk_end` that Step 9
introduced:

```cpp
#if defined(SDPA_RING_PROXY_UP)
const uint32_t k_chunk_end = k_num_chunks / 2;
#else
const uint32_t k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;
#endif
for (uint32_t k_chunk = 0; k_chunk < k_chunk_end; ++k_chunk) {
    // ... existing forward/receive body, unchanged ...
}
```

This is the same `k_chunk_end` variable Step 9 already introduced — as long as
the forward/receive branches live inside this loop, they automatically
inherit the cap.

### Step 6 — Writer / compute: no change

Writer has no chain logic. Compute's `sdpa_inner_loop` only sees K tiles via
CB push/pop; the reader's chain handles the data path invisibly to compute.
No UP K-cap adjustment needed in writer because writer doesn't iterate K.

### Step 7 — Semaphores already wired

The chain semaphores (`sender_semaphore_id`, `receiver_semaphore_id`,
`valid_semaphore_id`) are created in `sdpa_program_factory.cpp` around line
560 unconditionally for the non-causal path. They're already in compile-time
args 27-29 and referenced in the reader. No host-side plumbing changes needed
beyond lifting the `!flatten_work` gate in Step 1.

`mcast_enabled` (compile-time arg 30) is patched post-chain-build in
`sdpa_program_factory.cpp:1404`:
```cpp
reader_compile_time_args[sem_args_offset + 3] = (mcast_chains > 0) ? 1 : 0;
```
This already works for flat since `mcast_chains` is populated in the
mcast-eligibility pass (Step 3 leaves it untouched).

## Validation

### Accuracy regression

After enabling chain, re-run the existing accuracy tests to confirm no math
change:

```bash
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[mla_100k_ring_iter_up-q160-k160]"
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[mla_100k_ring_iter_down-q160-k160]"
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[mla_100k_ring_iter_0-q160-k160]"
```

Expected: PCC ≥ 0.994 for UP/DOWN, ≥ 0.997 for iter-0 (unchanged).

### Determinism regression

Chain forwarding introduces inter-core synchronization. Ensure bit-exact
output across runs:

```bash
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism[mla_100k_ring_iter_up-q160-k160]"
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism[mla_100k_ring_iter_down-q160-k160]"
```

Expected: bit-exact over 10 iterations (DOWN still ignores light-half rows
per the existing test logic).

### Perf parity

```bash
for name in mla_100k_ring_iter_0 mla_100k_ring_iter_up mla_100k_ring_iter_down; do
  scripts/run_safe_pytest.sh \
    "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[$name]"
done
```

**Expected math util** (from `RingSDPA_IterCases_SingleChip.md` reference):

| Config                     | Current (no chain) | Target (with chain) | Ring op |
|----------------------------|--------------------|---------------------|---------|
| `mla_100k_ring_iter_0`     | 31.5 %             | 31.5 % (unchanged)  | 31.2 %  |
| `mla_100k_ring_iter_up`    | 32.7 %             | ~52 %               | 52.8 %  |
| `mla_100k_ring_iter_down`  | 35.4 %             | ~56 %               | 56.3 %  |

If UP or DOWN lands below ~50 %, investigate:
- chain build: `log_debug(LogOp, "...")` lines in the mcast pass should show
  `mcast_chains = {expected_chain_count}` (32 heads / 110 cores ≈ 3 cores per
  head → chains of length 3, one chain per head → ~32 mcast chains).
- mcast eligibility: all chains should pass `same_row` + `no_gap` +
  `uniform_q_mcast` for MLA 100k's uniform slot distribution.
- UP K-cap: `should_forward` / `should_receive` must respect `k_chunk_end =
  k_num_chunks / 2`, not the full `k_num_chunks`.

## Risks

| Risk | Mitigation |
|------|------------|
| Deadlock from injector/receiver K count mismatch (UP) | Step 5: gate forward/receive inside the UP-capped K loop |
| `q_iter_local` off-by-one at head boundary (flat) | Step 4: reset counter on `(nb, nq)` transition; mirror ring_joint_reader.cpp:249 |
| Head split across cores breaks DOWN heavy-half offset | Step 2 DOWN caveat: decompose against `q_num_chunks / 2` or apply offset after |
| Mcast rectangle includes non-chain cores (gap) | Step 3: existing gap detection already handles this by falling back to unicast |
| Determinism failure under chain (sema races) | The existing chain logic is already deterministic — if it fails, it's a programming error in the q_iter_local wiring |

## Out of scope (deferred further)

1. **Streaming compute + flat_work + chain**: streaming path doesn't honor
   `SDPA_FLAT_WORK` and is ineligible for MLA 100k anyway. Revisit if a
   different model's chunk shapes make streaming eligible.
2. **Joint-L segment**: ring_joint handles a joint-KV tail for `L > 0`. The
   proxy is spatial-only; joint-L is out of scope.
3. **Dynamic K-cap (UP)**: Step 5 bakes `k_num_chunks / 2` as the cap. If a
   future proxy case needs a non-integer K fraction, thread the cap as a
   runtime arg instead.

## Commit split

Single PR, split into reviewable commits:

1. Lift the `!flatten_work` gate + flat core_work population (Steps 1–2).
2. Reader `q_iter_local` wiring for flat mode (Step 4).
3. UP K-cap enforcement inside chain forwarding loop (Step 5).
4. Tech report update: add measured post-chain numbers to
   `RingSDPA_IterCases_SingleChip.md`.
