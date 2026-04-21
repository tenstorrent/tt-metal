# Plan — Profile Chain Injector DRAM Channel Clustering

Companion to `RingSDPA_SingleChip_UpDown_ChainFollowup.md`. Chain forwarding now
engages for single-chip SDPA's flat non-causal path and lifts math util from
~32 % to **39.3 %** (UP) / **43.4 %** (DOWN) on MLA 100k. `ring_joint_sdpa`
reaches **52.8 %** / **56.3 %** on the equivalent per-device work **without
mcast** in both cases.

The gap's most likely cause (working hypothesis): single-chip's chain
construction clusters injectors onto a small set of physical X columns, so the
32 per-iter DRAM reads serialize across fewer banks than on ring_joint. This
plan verifies that hypothesis and picks a fix only if the data supports it.

## Hypothesis

Single-chip (`sdpa_program_factory.cpp:1057-1088`) only runs the
`best_dist` DRAM-channel spread for **uniform-q** chains. In the flat work
distribution, boundary-straddling cores give most chains mixed q_chunk_counts,
so the spread never runs — injectors are picked by descending-q sort, which
places the heaviest (6-slot) cores first. Heavies are a contiguous run in
core-idx → contiguous phys_x → same DRAM channel.

Ring_joint (`ring_joint_sdpa_program_factory.cpp:844-904`) builds chains
linearly (no wrap, no sort) and lets the natural `head_id → ~head_id*3.3`
core-idx spacing spread injectors across columns organically.

## Goal

Confirm or reject the clustering hypothesis with measured data, then decide
between three fixes:

- **A.** Run `best_dist` for all chains, not just uniform-q (needs to preserve
  descending-q invariants on non-uniform chains).
- **B.** Switch flat-mode chains to linear-only (mirror ring_joint exactly).
- **C.** Both.

## Execution

### Step 1 — Log per-chain injector phys_x

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`

Temporarily promote the existing chain-build `log_debug` at line ~1095 to
`log_info`, and add a summary line after the chain-build loop:

```cpp
log_info(
    tt::LogOp,
    "Building chain for head {}: injector core_idx={}, phys=({},{}), chain_size={}, uniform_q={}",
    head_id,
    inj_seg.core_idx,
    core_work[inj_seg.core_idx].physical_core.x,
    core_work[inj_seg.core_idx].physical_core.y,
    chain_order.size(),
    uniform_q);

// After all chains built:
std::map<uint32_t, uint32_t> x_hist;
for (uint32_t x : injector_phys_x) x_hist[x]++;
std::string hist;
for (auto& [x, c] : x_hist) hist += fmt::format("x={}:{} ", x, c);
log_info(tt::LogOp, "Injector phys_x histogram: {}", hist);
```

Run once for each of:

```bash
source python_env/bin/activate
for name in mla_100k_ring_iter_up mla_100k_ring_iter_down; do
  scripts/run_safe_pytest.sh \
    "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[$name-q160-k160]" \
    2>&1 | grep -E "(Building chain|Injector phys_x histogram)"
done
```

Record the histograms. Expected under clustering: 2–4 columns carry most
injectors. Expected under spread: roughly even distribution across ~11
columns.

Also run ring_joint with the same shapes to compare:

```bash
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_ring_iter_matrix[mla_100k]" 2>&1 \
  | grep -E "Injector"
```

(Ring_joint has equivalent log lines in its mcast pass — if not, add them by
the same pattern.)

### Step 2 — Profile per-core reader active cycles

Re-run with the device profiler on UP:

```bash
TT_METAL_DEVICE_PROFILER=1 \
  scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[mla_100k_ring_iter_up]"
```

Open the generated Tracy CSV at
`generated/profiler/ttnn_sdpa_performance/reports/<ts>/ops_perf_results_<ts>.csv`.

Split cores into **injectors** vs **receivers** (cross-reference with the
chain_info dump from Step 1). Compute mean active cycles per group.

**What to look for:**

- If injectors' mean active cycles ≫ receivers' → injectors are the critical
  path. Combined with column clustering (Step 1) this confirms DRAM
  contention.
- If injectors and receivers have similar active cycles → clustering isn't
  the bottleneck; look elsewhere (chain wrap direction, NoC congestion, V
  read interleave).

### Step 3 — Count distinct DRAM-channel-mapped columns

On Blackhole, each worker column maps to a specific DRAM channel via the
memory controller layout. From the injector phys_x histogram, count how
many **distinct DRAM channels** are reached. (Mapping is
arch-specific — query via `tt::tt_metal::hal::get_dram_channel_from_noc_x` or
similar API if programmatic lookup is preferred over hard-coded mapping.)

Expected under clustering: 3–4 channels out of ~8. Ring_joint should hit
close to all 8.

### Step 4 — Decision

| Observation                                                               | Fix                                                         |
|---------------------------------------------------------------------------|-------------------------------------------------------------|
| Clustering confirmed (Step 1 + 3) AND injectors are critical path (Step 2) | Implement **A** (always-spread), then remeasure. If gap remains, add **B**. |
| No clustering but gap remains                                             | Pivot to NoC/chain-order investigation: log per-iter sender-wait cycles, compare wrap vs linear order. |
| Clustering present but injectors aren't critical path                     | Different bottleneck — likely CB depth or Q re-read. Skip A/B; profile Q pipeline. |

## Implementation notes for Fix A (if chosen)

Current code (`sdpa_program_factory.cpp:1056-1088`) conditions
`best_dist` selection on `uniform_q`. For non-uniform chains we can still
spread, but only among cores tied for the max `q_chunk_count`. Sketch:

```cpp
// Collect indices tied for max q
const uint32_t max_q = core_work[segments[chain_order[0]].core_idx]
                           .head_work[segments[chain_order[0]].head_work_index].q_chunk_count;
std::vector<std::size_t> tied_for_max;
for (std::size_t pos = 0; pos < chain_order.size(); ++pos) {
    const auto& seg = segments[chain_order[pos]];
    if (core_work[seg.core_idx].head_work[seg.head_work_index].q_chunk_count == max_q) {
        tied_for_max.push_back(pos);
    }
}
// Spread among tied_for_max via best_dist against injector_phys_x;
// the chosen core replaces chain_order[0] to remain injector.
```

This preserves the invariant that the injector is tied for the max
`q_chunk_count` (so `next_core_q_chunks` bookkeeping stays correct) while
letting us spread across columns.

## Out of scope

- **Linear-only chain construction for flat mode (Fix B)** — a larger
  refactor; only take on if Fix A alone doesn't close the gap.
- **Making mcast eligibility per-chain instead of all-or-nothing** — separate
  follow-up; the user has confirmed the 50 %+ target is achievable without
  mcast, so this is not on the critical path here.

## Expected outcome

If clustering is the root cause and Fix A lands, UP should move from
~39 % → ~50 %+, DOWN from ~43 % → ~55 %+, matching ring_joint within
measurement noise. If it moves by less than ~5 pp, the chain-topology
or NoC-congestion path becomes the more likely culprit and the
investigation shifts accordingly.
