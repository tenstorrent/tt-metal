# Plan — KV Chain Forwarding for Causal Flat-Work SDPA (single-chip)

Companion to `RingSDPA_SingleChipProxy.md` and
`RingSDPA_SingleChip_UpDown_Plan.md`.

## Goal

Enable the store-and-forward KV chain optimization on the **causal
flat-work** path of single-chip SDPA, so that `mla_100k_ring_iter_0`
(and any other `flatten_work=True, is_causal=True` config) amortizes
DRAM reads across chain members the same way `ring_joint_sdpa` does at
iter 0. Target: close the 27.8 % → 30.8–31.4 % proxy fidelity gap
versus ring-joint, and ideally exceed the pre-4fef10306c 31.6 % baseline.

## Guiding constraint — "A-narrow"

**Do not change hierarchical causal SDPA behavior.** The gate relaxation
is scoped to `flatten_work=true`. `flatten_work=false && is_causal=true`
continues to compile with no chain construction. This keeps WAN-analog
and any existing production causal configs on their tuned path while we
tune flat-work.

Scope in one predicate:

```cpp
const bool chain_enabled = !is_chunked && (flatten_work || !is_causal);
```

## The hard problem and the A-narrow trade-off

Causal truncates each Q chunk's K loop to `ceil(q_high / Sk_chunk_t)`.
For a chain to work, injector + receivers must loop the **same** K range
— they cooperate on the same K/V chunks. Q-chunks at different positions
on different chain members would otherwise walk different K counts.

**Resolution (Option A):** any core that participates in a chain loops
over the **full `k_num_chunks`** regardless of its Q position. Compute's
existing `lightweight_causal_mask` (enabled by default under
`lightweight_causal`, wired up in `fd644afa96`) zeroes out softmax
columns past each Q's true `q_high`.

Cost: early-Q chunks read K they don't need. Savings: only injectors
hit DRAM (≈1/`chain_len` of the cores). Net DRAM reads drop by roughly
`chain_len ×` — the savings from chaining dominate the wasted columns.

Non-chain cores (alone-members, or any core without a chain under a
head) keep the existing per-Q truncated K loop.

## Current vs target state

| | Before this plan | After this plan (A-narrow) |
|---|---|---|
| `flatten_work=false, is_causal=true` (hierarchical causal) | No chains | **No change** (still no chains) |
| `flatten_work=true, is_causal=true` (iter 0 proxy) | No chains; all cores hit DRAM | **Chains active**; injectors hit DRAM, receivers get forwarded K/V |
| `flatten_work=true, is_causal=false, ring_proxy=up/down` | Chains active (5f35f6973b) | No change |
| `flatten_work=false, is_causal=false` (WAN-style non-causal) | Chains active | No change |

---

## Execution steps

### A. Host-side gate relaxation

#### Step 1 — Chain-build gate

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:911`

```cpp
// was:
if (!is_causal && !is_chunked) {
// becomes:
const bool chain_enabled = !is_chunked && (flatten_work || !is_causal);
if (chain_enabled) {
```

Everything inside that block (chain topology, injector/sink selection,
mcast eligibility, `head_segments` population) already works over the
flat-work distribution for the non-causal proxies — reuse it verbatim.

#### Step 2 — Semaphore-creation gate

File: `sdpa_program_factory.cpp:615`

```cpp
// was:
if (!is_causal) { CreateSemaphore(...); ... }
// becomes:
if (chain_enabled) { CreateSemaphore(...); ... }
```

If this block runs, the reader's `sender_semaphore_id` /
`receiver_semaphore_id` / `valid_semaphore_id` CT-args get filled in;
if not, the placeholders (0) stay — matching the current non-chain
behavior.

#### Step 3 — Chain-enabled compile define

File: `sdpa_program_factory.cpp` (near line 722, where `SDPA_RING_PROXY_*`
are emitted)

```cpp
if (chain_enabled) {
    defines["SDPA_KV_CHAIN_ENABLED"] = "1";
}
```

Rationale: today the reader/writer/compute gate chain logic on
`if constexpr (!is_causal)`. We'll switch to
`#if defined(SDPA_KV_CHAIN_ENABLED)` so it's decoupled from causality
and keyed on what the program factory decided.

#### Step 4 — `balanced_q_parallel` interaction check

`sdpa_program_factory.cpp:713` — `balanced_q_parallel` is already
gated off for `flatten_work=true`, so no change needed. Sanity-check
that `flat_zigzag` (emitted at :717) is the correct distribution for
causal chain members too (it already is — flat zigzag pairs Q chunks
i / n-1-i, which is a valid distribution for causal chain semantics).

### B. Reader kernel

#### Step 5 — Chain-condition gate

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp:455`

```cpp
// was:
if constexpr (!is_causal) {
    should_forward = is_chain_participant && !is_sink && ...;
    should_receive = is_chain_participant && !is_injector && ...;
}
// becomes:
#if defined(SDPA_KV_CHAIN_ENABLED)
should_forward = is_chain_participant && !is_sink && ...;
should_receive = is_chain_participant && !is_injector && ...;
#endif
```

Now live for causal chain cores.

#### Step 6 — K-loop length for chain participants

File: `reader_interleaved.cpp:462-468`

```cpp
#if defined(SDPA_RING_PROXY_UP)
const uint32_t k_chunk_end = k_num_chunks / 2;
#else
const uint32_t k_chunk_end =
#if defined(SDPA_KV_CHAIN_ENABLED)
    is_chain_participant
        ? k_num_chunks                                       // full K for chain cores
        : (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;        // truncated causal for alone cores
#else
    (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;
#endif
#endif
```

Important: `valid_Skt_bound` and `kv_row_tile_count` (lines 470–472)
continue to handle per-chunk padding for chunks beyond logical N, so
reading up to full `k_num_chunks` is safe — extra tiles past the
logical-N boundary get zero-padded, not OOB-read.

### C. Writer kernel

#### Step 7 — Mirror K-loop length

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`

The writer has a corresponding `k_chunk_end` computation. Apply the
same conditional so its K-iter count stays aligned with the reader and
compute — otherwise the accumulator-write phase drifts out of sync.

Grep for the existing `(q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t`
formula in the writer and apply the same `is_chain_participant ?
k_num_chunks : truncated` pattern.

### D. Compute kernel

#### Step 8 — Full-K loop under chain

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:1799-1812`

```cpp
if constexpr (sdpa_type == STANDARD) {
#if defined(SDPA_RING_PROXY_UP)
    k_chunk_end = iter_k_chunk_end / 2;
#else
#if defined(SDPA_KV_CHAIN_ENABLED)
    k_chunk_end = is_chain_participant
                      ? k_num_chunks
                      : (q_high_tile + Sk_chunk_t - 1) / Sk_chunk_t;
#else
    k_chunk_end = (q_high_tile + Sk_chunk_t - 1) / Sk_chunk_t;
#endif
#endif
}
```

Compute already receives `is_chain_participant` as a runtime arg (it's
used by the existing non-causal chain path). If not, pass it through
the same mechanism as the reader.

#### Step 9 — Lightweight causal mask must remain enabled

File: `sdpa_program_factory.cpp:475`

```cpp
const bool lightweight_causal = is_causal && !use_provided_mask && sliding_window_size.value_or(0) == 0;
```

Already true for iter 0. Verify no new gate turns it off under
`chain_enabled && is_causal` — this mask is what zeroes softmax
contributions past each Q's real `q_high` when the loop over-reads.

### E. Program-factory — per-chain runtime args

#### Step 10 — Populate chain runtime args for causal cores

The existing chain-topology pass (inside the relaxed block from Step 1)
already emits `is_chain_participant`, `is_injector`, `is_sink`,
`chain_batch`, `chain_head`, `next_physical_x/y`, `prev_physical_x/y`,
`next_core_q_chunks`, `mcast_num_dests`. These flow through runtime
args to reader / writer / compute.

No new runtime arg is required for the kernel-side changes above — they
all key off the existing `is_chain_participant` flag plus the new
compile-time `SDPA_KV_CHAIN_ENABLED` define.

---

## Risks and edge cases

| Risk | Mitigation |
|------|------------|
| Causal correctness — over-read K tiles past `q_high` taint softmax. | `lightweight_causal_mask` already zeroes these columns; verify it's active under `chain_enabled && is_causal`. |
| CB overflow — reader pushes more K chunks than compute pops. | Compute Step 8 mirrors reader Step 6; both loop to `k_num_chunks` for chain cores. Validate cb_reserve/push/pop counts match. |
| Alone-member cores (no chain on their head) — should keep truncated K. | Gate is `is_chain_participant ? full : truncated`. Runtime-true only for chain cores. |
| Mcast eligibility assumes `uniform_q_mcast`. Under causal + flat zigzag, chains may span mixed-q members. | Existing mcast eligibility check already handles non-uniform chains by falling back to unicast chain. Confirm eligibility log output for iter 0. |
| Zigzag distribution (flat_zigzag=1) — pairs Q chunks i / n-1-i on the same core. Chain may span a mix. | Not a correctness issue — chain topology is per (batch, head), and zigzag just determines which Q chunks a core owns. Chain forwarding is agnostic. |
| NOC barrier / linked-write pattern — must not deadlock with causal-path compute. | `read_chunk_for_forwarding` + linked mcast write + semaphore pattern is unchanged from the non-causal path, which is proven under 5f35f6973b + 470f9b88b0. Re-use as-is. |
| 4fef10306c's mid-barrier skip (`enable_mid_barrier=false`) still applies to the non-chain else-branch. | Fine — with chains active, most cores hit the chain branch, not the mid-barrier-skipped branch. |

## Verification plan

### Step V1 — Accuracy (mandatory, first check)

Before any perf measurement, confirm chained causal output matches the
previous chainless output bit-for-bit-ish (PCC ≥ 0.9995).

```bash
source python_env/bin/activate
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[mla_100k_ring_iter_0-q160-k160]"
```

Expect: `PASSED` with PCC ≥ 0.9995 against the torch reference.

Run the longer (128k) variant too:

```bash
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[mla_128k_ring_iter_0-q128-k128]"
```

### Step V2 — Determinism (10× bit-exact)

```bash
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism[mla_100k_ring_iter_0-q160-k160]"
```

Expect: all 10 runs bit-identical.

### Step V3 — Regression guard on UP/DOWN and hierarchical WAN

Confirm the existing non-causal chain path (already working) still
passes, and hierarchical causal (untouched) still passes.

```bash
scripts/run_safe_pytest.sh --run-all \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy" \
  -k "mla_100k or wan2_2"
```

### Step V4 — Math util (the whole point)

Per-config perf table:

```bash
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_100k_ring_iter_0]"
```

Look at the summary line:

```
[mla_100k_ring_iter_0] q=160, k=160: <duration>ms, util=<X>%, cores=110/110 (100%), iters/core=<Y>
```

**Expected numbers:**

- Pre-this-plan (HEAD with 4fef10306c): 2.736 ms, 27.7 % util.
- Pre-4fef10306c (mid-barrier still on, no chains): 2.404 ms, 31.6 % util.
- Ring-joint per-device iter 0: 30.8–31.4 % util.
- Post-this-plan target: ≥ 31 % util; ideally exceeds 31.6 % because
  chains cut DRAM traffic independent of the barrier.

Also check the 128k variant:

```bash
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_128k_ring_iter_0]"
```

### Step V5 — Run all three MLA 100k proxies back-to-back

```bash
scripts/run_safe_pytest.sh --run-all \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_100k_ring_iter_0] \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_100k_ring_iter_up] \
  tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_100k_ring_iter_down]
```

Expect: iter_0 ~31 %+, up ~53 %, down ~56 % (up/down unchanged from
HEAD).

### Step V6 — Cross-check vs multi-chip ring-joint

Re-run the per-device ring-joint measurement to confirm proxy and ring
now agree within noise:

```bash
TT_METAL_RING_ITER_ONLY=0 scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table[mla_100k]
```

Ring-joint per-device iter 0 math util is expected to stay at
30.8–31.4 % (we're not changing ring-joint). The proxy should now land
inside or above that band.

### Step V7 — Update tech report

Update `tech_reports/FlashAttention/RingSDPA_SingleChipProxy.md` §Current
numbers: replace the 27.8 % iter_0 row with the new post-chain number,
and update the proxy-fidelity table to reflect the closed gap.

---

## What a successful completion looks like

1. `test_sdpa_accuracy[mla_100k_ring_iter_0-q160-k160]` passes with
   PCC ≥ 0.9995 (unchanged correctness despite full-K loop).
2. `test_sdpa_determinism[mla_100k_ring_iter_0-q160-k160]` passes 10/10.
3. `test_sdpa_create_perf_table[mla_100k_ring_iter_0]` reports ≥ 31 %
   math util.
4. `test_sdpa_accuracy` across all MLA 100k and WAN configs passes
   (no regression in UP/DOWN or non-causal paths).
5. `RingSDPA_SingleChipProxy.md` proxy-fidelity CAUSAL row shows gap
   ≤ 1 pp instead of the current ~3 pp.
