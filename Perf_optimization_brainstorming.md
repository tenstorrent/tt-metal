# Performance Optimization Memory Brainstorming

Date: 2026-06-23

This note captures the working idea for bootstrapping a long-term performance-optimization memory for the TT op-generation framework. It is a checkpoint, not a final design.

## Core Idea

Use the existing TT codebase as a source of human performance expertise.

The proposed pipeline is contrastive:

```text
AI-generated op
vs.
human-written optimized op
    -> identify optimizations present in the human op but missing from the AI op
    -> convert those differences into prioritized refinement hypotheses
    -> let the op-generation framework attempt the refinements
    -> verify correctness and measure performance impact
    -> promote successful, reusable recipes into long-term optimization memory
```

The memory should not be just a collection of prose notes. It should store structured, retrievable optimization knowledge:

- applicability conditions
- required code features
- profiler/runtime evidence
- transformation recipe
- preconditions and veto rules
- expected metric movement
- measured results
- negative examples and failed attempts

The agent can then use this memory to narrow the search space before asking an LLM to plan and edit code.

The memory should also avoid becoming a pile of one-off op-specific tricks. A contrast pair such as generated SDPA vs official SDPA is evidence for a broader optimization pattern. The promoted memory entry should be general when possible, with the concrete op pair stored as supporting evidence and implementation detail.

Example:

```text
General memory:
  tt_reduce_redundant_dram_reads_across_cores

Origin (unverified, reference only — where the idea came from):
  origin/main official SDPA forwards/mcasts K/V across cores sharing a (batch,head);
  generated SDPA rereads K/V from DRAM per (b,h,q_chunk).
  A human reference is a SOURCE, not proof — it never promotes a memory on its own.

Verified evidence (required to promote — machine-actionable, an agent can diff it):
  <refinement SHA on the generated SDPA op that ADDED forwarding,
   golden=pass, device-time before->after>; the diff IS the recipe.
  status: hypothesis  (until such a SHA exists; then -> promoted)
```

The split is load-bearing: prose about *someone else's* code is an unverified hypothesis;
the only thing that earns promotion is a committed refinement on a generated op that an agent
can `git show` / diff before->after, with a measured delta. (Negative results use the same
shape — a SHA whose measured delta is a regression — and become veto evidence.)

## Main Reference: KernelSkill

The motivating paper is:

- KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization
- arXiv: https://arxiv.org/pdf/2603.10085

The relevant idea is its split between long-term and short-term memory.

Long-term memory stores reusable expert optimization knowledge. It is not only a vector database of documents; it includes structured fields, derived bottleneck facts, decision rules, forbidden rules, and method knowledge.

Short-term memory stores the current optimization trajectory: what was tried, what failed, what improved, which kernel version became the new base, and which paths should not be repeated.

For TT, the key adaptation is:

```text
profile/run facts
+ static TT code facts
+ tensor/layout facts
+ architecture facts
    -> derived bottleneck facts
    -> matched optimization memories
    -> vetoed unsafe options
    -> ranked candidate transformations
    -> agent patch plan
```

## Supporting Reference: GPU Optimization Survey

KernelSkill references:

- Optimization Techniques for GPU Programming
- DOI: https://dl.acm.org/doi/10.1145/3570638

The ACM page was not directly accessible from this environment, but an alternate PDF mirror exposed the article structure:

- https://library.qiangtu.com/download/826/pdf/826.pdf

Useful top-level categories from the survey:

- memory access
- irregularity
- balancing
- host interaction

These categories are CUDA/GPU-oriented, but they provide a template for turning expert optimization ideas into a taxonomy. TT-specific categories should be derived from TTNN/TT-Metal code and profiler evidence rather than copied directly.


## Part 1 — Seed Pair: SDPA

The first contrast pair: an AI-generated flash-attention op vs the official C++ SDPA.

## Concrete Seed Pair: Generated SDPA vs Official SDPA

This discussion used scaled-dot-product attention as the first concrete pair.

### AI-Generated SDPA

Grounded in the actual op (verified against the branch directly — not a clone path):

```text
branch:  2026_06_16_1651_run1_flash_attention  @ e7661e4
op dir:  ttnn/ttnn/operations/scaled_dot_product_attention/
```

Files (all verified present on the branch):

```text
scaled_dot_product_attention.py
scaled_dot_product_attention_program_descriptor.py
kernels/scaled_dot_product_attention_{reader,compute,writer}.cpp
op_design.md  op_requirements.md  changelog.md  verification_report.md
tests: tests/ttnn/unit_tests/operations/scaled_dot_product_attention/
```

What the generated op ALREADY implements (grounded read of the kernels + changelog
R0–R5 — this is the **baseline**, NOT the optimization opportunity):

- registry-model op (`SUPPORTED`/`EXCLUSIONS`/`validate()`, `ttnn.generic_op`); **3 kernels**.
- flash-attention **online softmax**: per-unit running max/sum, exp correction, partial
  PV accumulation, final `1/l` normalization (8-CB recurrence:
  `cb_m/cb_l/cb_corr/cb_p/cb_o/cb_pv/cb_scores/cb_recip_l`).
- **`Bkv_t ∈ {1,2,4,8}` KV-chunk blocking** with host-side L1-footprint gating (R5/R4).
- **on-device causal + KV-edge mask generation**, regenerated per work unit (R2/R3).
- `fp32_dest_acc_en` fast-path gating + EXCLUSIONS; bf16/fp32/bf8b dtypes (R1).
- self/cross attention, MHA/GQA/MQA, auto/explicit scale (R2); double-buffered reader (R4).

Work distribution: contiguous `(b,h,q_chunk)` units across the grid via
`split_work_to_cores`, **`Bq_t = 1`** (one Q tile-row per unit), **no cross-core
communication** (`semaphores=[]`, every unit reads its own K/V from DRAM). Those three
facts are exactly what the deltas below act on.

### Human/Official SDPA

Official implementation was inspected from `origin/main` without switching branches.

Key files:

```text
origin/main:ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
origin/main:ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp
origin/main:ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp
origin/main:ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp
```

Observed official-op characteristics:

- Production C++ TTNN transformer SDPA implementation.
- Rich program factory with many compile-time/runtime arguments and kernel variants.
- Streaming compute helper layer in `compute_streaming.hpp`.
- Online softmax flow with row max/sum tracking, exp correction, partial PV accumulation, and final normalization.
- Support for chunked/page-table prefill, flexible chunk starts, sliding window, attention sink, MLA overlap, causal/window/padding behavior, dense and lightweight masks, global Q scheduling, and load-balancing strategies.
- Reader and writer dataflow kernels handle layout-aware reads/writes, mask generation, Q/K/V movement, padding, forwarding/mcast paths, and valid-row output.
- Production implementation separates or gates paths based on constraints such as fp32 destination accumulation.

## SDPA Pair — Grounded Optimization Deltas

The useful artifact is the **semantic delta between the generated and human-optimized
implementations**, derived by diffing the *actual* generated op (see AI-Generated SDPA above
for what it already implements) against the official `sdpa_standard_v2` path. Candidates must
come from this diff, not from an assumed baseline — the generated op is already a streaming
flash-attention op, so its baseline features are not opportunities. Both sides were
inventoried axis-by-axis against the same 11 axes; the deltas below are what the official op
does that the generated op does not, classified **PERF** vs **FEATURE/architectural** per the
`feature delta != optimization delta` rule.

### P0 — use the grid better (structural, biggest wins)

| # | Delta | Generated (grounded) | Official (grounded) | General memory |
|---|-------|----------------------|---------------------|----------------|
| 1 | **Cross-core K/V forwarding + multicast** | every `(b,h,q_chunk)` unit reads its own K/V from DRAM; `semaphores=[]`, no mcast | KV store-and-forward chain + optional linked mcast for cores sharing a `(batch,head)` (`sdpa_program_factory.cpp:787–1100`, `reader_interleaved.cpp:405–697`, 3 semaphores) — **non-causal path** | `tt_reduce_redundant_dram_reads_across_cores` |
| 2 | **Causal work balancing** | contiguous per-core unit range (`split_work_to_cores`), no balancing — causal late-Q chunks attend far more K than early-Q ⇒ per-core skew | global Q scheduling + pair-distribution + `remap_q_index` zigzag (`sdpa_program_factory.cpp:342`, `compute_streaming.hpp:1781`) — **causal path** | `tt_balance_variable_work_units_by_predicted_cost` |

#1 (non-causal) and #2 (causal) are complementary by mask mode.
**Detection:** #1 → `input_read_redundancy = DRAM_K/V_reads / unique_K/V_tiles > 1` + multiple
cores share a `(b,h)` (static from dispatch; npe `dram_bw_util`). #2 → `per_core_predicted_work_skew`
(static from causal loop bounds) + Tracy per-core duration spread.

### P1 — reuse, fusion, overlap

| # | Delta | Generated | Official | General memory |
|---|-------|-----------|----------|----------------|
| 3 | **`Sq_chunk_t>1`: multiple Q rows per K/V chunk** | `Bq_t=1` — one Q tile-row per unit; K/V re-read for adjacent Q chunks | `Sq_chunk_t` Q rows share one K/V chunk; row-group V matmul (`compute_streaming.hpp:1330–1714`) | `tt_reuse_shared_input_tiles_across_multiple_output_blocks` |
| 4 | **Fuse the streaming recurrence** | literal 8-CB recurrence, per-KV-chunk rescale (K→H→J ×`Nkv`) | QK→`cb_qkt_im` direct, in-place softmax, ping-pong accumulators, reduce-trigger overlap, fused SALAD correction, row-wise finalization | `tt_fuse_streaming_recurrence_to_reduce_intermediate_cb_traffic` |
| 5 | **Finer reader/compute overlap** | full-Q push, per-tile barrier, V wait not deferred | Q subblock push, deferred K waits, V DMA overlaps softmax drain (`reader_interleaved.cpp:576–589`, `compute_streaming.hpp:1357`) | `tt_improve_dataflow_overlap_from_cb_wait_signals` |

**Detection:** #3 → `input_bytes_per_output_tile` high + adjacent Q chunks read same K/V
(static; noc_estimator quantifies the read amortization). #4 → `intermediate_cb_traffic_per_output_tile`,
math-vs-pack/unpack fraction (Tracy). #5 → `compute_input_wait_fraction` (Tracy CB `WAIT_*` zones).
**Caveat (from the rms_norm pair):** #4 fusion can *lose* 3-TRISC pipelining — measure before
keeping (a known revert risk, not a guaranteed win).

### P2 — second-order (some coupled)

| # | Delta | Generated | Official | General memory |
|---|-------|-----------|----------|----------------|
| 6 | **Resident mask palette vs per-unit regen** | reader regenerates causal/edge block per work unit (`gen_causal_block`) | lightweight mask palette generated once, resident in `cb_mask_in`, stamped per-row via L1 acc (`writer_interleaved.cpp:43–59`) | `tt_replace_recomputed_structured_metadata_with_resident_palette_or_generator` |
| 7 | **Row-group streaming output drain** | streaming per-tile writes, per-block wait | 2-slot row-group ping-pong `cb_out`, row-by-row finalize+drain | `tt_stream_output_drain_to_reduce_l1_and_backpressure` — **coupled to #3** (only matters once `Sq_chunk_t>1`) |
| 8 | **DEST-tuned subblocks** | fixed `MatmulBlockShape` (1×`Bkv_t`, 1×`d_t`) | `determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size)` | `tt_tune_subblocks_for_dest_utilization` |

**Detection:** #6 → `metadata_overhead_fraction` (Tracy). #8 → `dst_utilization = useful_tiles / DEST_capacity`.

### Excluded — NOT perf deltas (per `feature delta != optimization delta`)

- **Feature modes** (chunked/page-table prefill, sliding window, attention sink, MLA) —
  feature coverage the generated op lacks; mine only if those modes become targets.
- **Streaming-vs-legacy fast-path split** (`use_streaming_compute = !(provided_mask || fp32_acc)`)
  — architectural/correctness gating; the generated op already gates `fp32_dest_acc_en`. Record as
  a **precondition/veto** (the streaming path is invalid under provided-mask or fp32-acc), not a
  refinement candidate.


## Part 2 — Seed Pair: RMS-Norm

The second pair: an AI-generated rms_norm vs a human-vibe-coded refinement of it.

## Second Seed Pair: Generated RMS-Norm vs Advanced (Refined) RMS-Norm

Full contrast report: `rms_norm_contrast_report.md`. This section mirrors the SDPA pair
above and validates that the *method* (enumerate optimizations → derive a detection signal
per optimization → record veto rules) transfers to a different op. The optimizations
themselves are mostly new — they cluster around **work distribution / grid occupancy**
rather than SDPA's input-reuse cluster — but two of them (combine-fusion, output drain)
recur from the SDPA pair, which is the cross-op generality signal we want.

### The Pair

```text
Generated (AI baseline):  ttnn/ttnn/operations/rms_norm/   (framework-generated regen run)
Advanced (refined):       ttnn/ttnn/operations/rms_norm/
                          @ https://github.com/tenstorrent/tt-metal/tree/refinement1-regimeb-fix
```

Both are framework-generated registry-model ops (`ttnn.generic_op`). Unlike the SDPA pair
(generated vs mature production C++), here the advanced side is the **same op taken through
9 measured perf refinements**. Consequences:

- The `feature delta != optimization delta` filter is mostly unneeded — almost every delta
  is a *pre-measured* performance optimization with committed before/after numbers.
- The advanced side **kept its failed experiments as gated-off code with revert verdicts**,
  so this pair yields first-class *negative* memory (veto rules), not just positive recipes.
  It already clears the `no measured result -> no promoted memory` bar.

### Key derived bottleneck fact (attach to the op class)

Timeline profiling proved rms_norm is **DRAM-access-LATENCY bound** (2 KB tile transfers,
~10% NoC-link utilization, ~55–60% of peak DRAM BW with headroom), gated by the serial
per-row chain `read -> square -> reduce -> finalize -> pass2 -> write` starving a
~50%-duty-cycle writer. This single fact predicts which recipes pay off: the wins come from
**using idle cores** and **cutting per-op fixed overhead**, while every **local compute
fusion / input-prefetch** lever measured neutral-to-negative.

### Baseline observation

The generated op is already correct and L1-bounded for wide W, but its work distribution is
**row-parallel with no inter-core communication** (`semaphores=[]` in every path): a single
wide-narrow row (e.g. `(1,1,32,8192)`) is reduced **entirely on one core while 63 sit idle**.
The advanced op's headline change is a second regime that splits that reduction across cores.

### P0: Split A Wide Reduction Across Cores Via Partial-Sum All-Reduce

When the natural work units (rows) underfill the grid and a single unit is large, split the
*reduced dimension* across K cores; each computes a partial reduction on its shard and a
cross-core collective combines them. (Data-parallel → tensor-parallel pivot.)

Generated reduces a wide row on one core; advanced (Regime B) W-splits across K cores with
an mcast all-reduce of partial Σx². Measured 2.1–6.1x on wide-W shapes.

General memory candidate:

```text
tt_split_wide_reduction_across_cores_with_partial_allreduce
```

Applicability:

- a single work unit is large and its core-local op is an associative reduction
- the natural unit count underfills the grid (wide-narrow tensors)
- the reduction can be expressed as combine(partial reductions)

Ideal detection signal:

```text
grid_occupancy = active_cores / total_cores       # low => idle cores to recruit
reducible_unit_tiles                              # large & splittable along the reduce axis
reduction_is_associative                          # precondition (static)
```

Risks/vetoes: introduces mcast/semaphores (hang surface); only wins above an L1/perf
crossover (below).

### P0: Select Parallelization Regime By Grid Occupancy AND L1 Fit (not L1 fit alone)

The generated op routes wide W purely on an **L1-footprint OOM threshold** ("does it fit?").
The advanced op adds an occupancy crossover: a **byte-aware OOM floor** (a row that can't fit
one core's L1 *must* split — counted in real tile-byte formats, dtype-correct) **separate
from** a perf crossover (a row that fits splits only if it adds cores and is wide enough,
and is `fp32_acc`-aware as a precision floor).

General memory candidate:

```text
tt_select_parallelization_regime_by_grid_occupancy_and_l1_fit
```

Ideal detection signal:

```text
must_split  = per_core_resident_bytes(no_split) > L1_resident_budget   # OOM floor
should_split = grid_occupancy(no_split) << 1 AND unit_width >= crossover  # perf floor
```

### P0: Tune The Parallelism Factor Against Communication Cost (do not maximize cores)

The largest single SDPA-pair lesson recurs and sharpens here. The W-split factor K trades
per-core reduce work (~O(Wt/K)) against all-reduce cost (~O(K)): device time ≈ `Wt/K + c·K`,
a U-shape. **"Maximize K" was measured as the worst choice** — K=64 → 109.8 µs vs K=16 →
33.9 µs (3.2x) on `(1,1,32,8192)`. The advanced op proxy-minimizes K; it never argmaxes
core count.

General memory candidate (meta — applies to any tile/tensor-parallel split):

```text
tt_tune_parallelism_factor_against_communication_cost
```

Ideal detection signal:

```text
device_time(K) is U-shaped: compute_term ~ Wt/K (down), comm_term ~ c*K (up)
=> sweep or proxy-minimize; argmax(active_cores) is a trap
```

### P1: Choose The Collective Algorithm By Payload Size And Core Count

Once a cross-core reduction exists, the collective itself is tunable. The advanced op ships
three transports behind one CT arg: O(K) rotating mcast all-gather, O(1) root-relay
gather-then-broadcast, and reduce-then-broadcast (root reduces, mcasts a single tile, peers
skip the combine). For the **1-tile Σx² payload**, reduce-broadcast wins (collapses the
per-peer combine stall ~8 µs → 0.5 µs; root-relay alone is 1.1–1.5x over baseline). Ring was
*not* tried — 1-tile data ⇒ K−1 latency-bound hops.

General memory candidate (the GPU all-reduce algorithm-selection pattern, adapted to mcast):

```text
tt_choose_collective_algorithm_by_payload_and_core_count
```

Ideal detection signal:

```text
collective_payload_tiles (tiny => prefer O(1) reduce-broadcast over O(K) gather)
K (core count in the collective group)
collective_fraction_of_critical_path
```

### P1: Replace Sequential Elementwise Accumulation With A Single Reduce

The advanced combine originally summed K partials with `copy` + (K−1) `add`s — each an L1
round-trip — and was replaced by a single `reduce<SUM,REDUCE_ROW>` over the K-tile block
(DEST-accumulate, pack once). Same family as SDPA's
`tt_fuse_streaming_recurrence_to_reduce_intermediate_cb_traffic` — **cross-op recurrence**.

```text
tt_replace_sequential_elementwise_accumulation_with_single_reduce
```

Ideal detection signal:

```text
n_sequential_binary_adds_to_sum_N_tiles (>=2 => fuse into one DEST-accumulating reduce)
intermediate_cb_roundtrips_per_output_tile
```

### P1: Batch NoC Writes Behind One Barrier + Double-Buffer The Output CB

The generated writer does `noc_async_write_tile` + `noc_async_write_barrier` **per tile**
(~0.75 µs/tile serialized). The advanced writer issues a whole `reduce_block` of writes then
**one** barrier, and grows `cb_output` 2 → `2·reduce_block` so compute fills block N+1 while
the writer drains block N. 1.03–1.18x, plus a bonus pass-2 unstall. Sibling of SDPA's
`tt_stream_output_drain_to_reduce_l1_and_backpressure` /
`tt_improve_dataflow_overlap_from_cb_wait_signals`. Surfaced a library ask: a
`dataflow_kernel_lib::write_tile_block(...)` batched-writer helper (re-implemented inline).

```text
tt_batch_noc_writes_behind_single_barrier   (+ tt_double_buffer_output_cb_to_overlap_compute_and_drain)
```

Ideal detection signal:

```text
noc_barriers_per_output_block (1-per-tile => batch)
WR-wait fraction AND output-CB back-pressure stall on compute pack
```

### P2: Pad The Work Dimension / Flex Team Geometry To Saturate The Grid

The advanced op pads `Wt` up to a multiple of `gx` so any width has a valid split K (odd
widths previously crashed or stuck at K=2), and relaxes the full-width-band requirement so K
may *divide* gx (sub-row 1×K teams) — letting many-row wide-W shapes tile the full grid
instead of falling to single-core-per-row. `(512,8192)` 163.8 → 100.3 µs.

```text
tt_pad_work_dimension_to_unlock_uniform_core_partition
tt_use_flexible_team_geometry_to_saturate_grid
```

### Veto / Method Memories (the negative half — unique strength of this pair)

The advanced side measured these and **kept them off**; they are promotable as veto rules:

```text
v1. tt_do_not_row_block_a_dram_latency_bound_kernel
    evidence: row-blocking bh>1 measured 0.83-0.96x (slower); amortizes compute *init*,
              which is not the bottleneck. (Regime A R7, Regime B R8 K-monotonicity.)
v2. tt_a_large_stall_is_not_always_on_the_critical_path
    evidence: WR-wait, RDR-resv, and a 5.9us "square" zone all looked like bottlenecks but
              were off the critical path; double-buffering input collapsed RDR-resv yet
              regressed total. Reconstruct the wall-clock timeline before optimizing a zone.
v3. tt_do_not_manually_dest_chunk_a_row_reduction
    evidence: a REDUCE_ROW emits 1 output tile, so DEST is never the constraint; the helper
              tiles DEST internally. Manual chunking is only justified when it ALSO bounds a
              resident CB (why PASS-2 chunking was kept but PASS-1 chunking removed).
```

Negative results map onto the same structured shape as positive ones (applicability +
evidence + the signal that *would* have predicted the win), so the method is symmetric:
every hypothesis the loop runs produces either a recipe or a veto, both retrievable.

### What This Pair Adds To The Method

- The enumerate → derive-detection-signal → record-veto loop is **op-agnostic**: it produced
  6 positive + 3 veto candidates here with no SDPA-specific machinery.
- It motivates a **work-distribution / grid-occupancy** category in the taxonomy (distinct
  from SDPA's memory-access/reuse category) covering P0/P1 above.
- It adds one cheap **static** detection primitive to the SDPA list (per-CB timing +
  per-role DRAM read counting): `grid_occupancy = active_cores / total_cores` per dispatched
  shape — a pre-run signal that flags the highest-value optimization here before any
  profiling. `grid_occupancy` is the Tier-0/1 static signal in **Measurement & Tooling
  Landscape** below.


## Part 3 — Synthesis: Method, Tooling, and Proposal (from both pairs)

Conclusions drawn from both seed pairs. They established two optimization clusters — SDPA's *memory-access / reuse* deltas and rms_norm's *work-distribution / grid-occupancy* deltas — with two recipes (combine-fusion, output-drain) recurring across both. The tooling, method, memory schema, risks, and next steps below are written against that combined evidence.

## Measurement & Tooling Landscape

The two pairs above supply *what optimizations exist*. This section supplies *how the
framework detects the need and validates the fix* — the `derived bottleneck facts` arrow in
the Core-Idea pipeline, now backed by real instruments. It is cross-cutting (tied to neither
pair) and informs the detection half of every memory entry.

### The bottleneck-classification ladder (cost-ordered)

Stop at the first tier that is decisive for the decision at hand; only escalate when it isn't.

| Tier | Instrument | Question answered | Cost |
|------|-----------|-------------------|------|
| 0 | Arithmetic intensity (static) | memory- vs compute-bound — *almost always "memory" at the start* | free |
| 1 | Static proxies: `grid_occupancy`, tile/page size, per-role DRAM redundancy, NoC-barrier granularity | latency- vs bandwidth- vs occupancy-limited (candidate generation) | free |
| 2 | **noc_estimator** (per-transfer empirical lookup) | latency/bandwidth of *one* transfer pattern (mechanism × pattern × size × fan-out) — host-runtime, no device | free (table lookup) |
| 3 | **tt-npe** (NoC workload model) | whole-workload NoC / DRAM / **congestion** axis — no-device sim or one trace | low |
| 4 | **Tracy** device zones | per-RISC busy-vs-stall — the compute / sync axis the NoC tools cannot see | one instrumented run |
| 5 | **Tracy** wall-clock timeline | true critical path across overlapping stages | one run + interpretation |

Arithmetic intensity is a roofline prior: rms_norm's `AI ≈ 1 flop/byte` sits far below the
ridge point (hundreds of flops/byte on these accelerators), so "memory-bound" is predictable
statically — and that prior alone would have pre-vetoed the three reverted compute-fusion
attempts (square+reduce, finalize-rsqrt, reader↔square overlap). But AI gives only the
macro-regime; it cannot distinguish **bandwidth-bound from latency-bound** (rms_norm was the
latter — ~55% of peak DRAM BW with headroom, ~10% NoC-link util), nor *which stage* gates.
Tiers 2–5 resolve those. **The two NoC-axis tools (Tiers 2–3) are complementary, not
redundant:** noc_estimator costs a single transfer pattern in isolation (cheapest, runs
*during program construction*); tt-npe simulates the whole workload *with congestion between
concurrent transfers*. Tracy (Tiers 4–5) is the only one that sees compute and sync.

### Instrument 1 — noc_estimator (per-transfer cost, host-runtime, planning-time)

`tt_metal/impl/experimental/noc_estimator/` (introduced in commit `dfc7032`, "[DM] NoC
Estimator v1"). **Not** a simulator and **not** instrumentation — a *data-driven empirical
lookup table*. You hand it a `NocEstimatorParams` describing **one transfer pattern** and it
returns `{latency_cycles, bandwidth_bytes_per_cycle}`:

```text
mechanism: UNICAST | MULTICAST | MULTICAST_LINKED
pattern:   ONE_TO_ONE, ONE_FROM_ALL, ONE_TO_ALL, ALL_TO_ALL, ONE_TO_ROW, ROW_TO_ROW, ...
memory:    L1 | DRAM_INTERLEAVED | DRAM_SHARDED ;  arch: WORMHOLE_B0 | BLACKHOLE
num_transactions, num_transactions_per_barrier, transaction_size_bytes
num_subordinates (fan-out), same_axis, stateful, loopback, noc_index (noc0/noc1)
```

The backing data (`latencies/noc_latencies.yaml`, 740 entries) is **harvested from the
data-movement team's empirical HW perf suite** — real microbenchmarks across transaction
sizes {64 B … 64 KB}. Lookup: exact key → interpolate over size; else interpolate over the
numeric fields (num_transactions, num_subordinates); else **relaxation** — fall back on
similar categorical configs (with a stderr warning). Its stated purpose is to replace the
*"five different estimation approaches with hardcoded values, some off by an order of
magnitude"* with one API usable **"at host runtime during kernel compilation … for kernel
planning and scheduling decisions."**

Two roles in this story:

- **Cheapest NoC-axis detection backend.** It quantifies the latency-vs-transfer-size curve
  we kept invoking. From the unicast one-from-one row: 64 B → 293 cyc, 2 KB → 366 cyc (data
  ×32 for +25% latency → fixed-latency-dominated), 64 KB → 2589 cyc. As bandwidth that is
  0.22 → 5.6 → 25 B/cyc — a ~100× swing. **That is the rms_norm "small 2 KB transfers are
  latency-bound, batch them" finding as a table** — the payoff of coalescing N tiles into one
  transfer is now a number you read *before* writing the kernel.
- **Planning-time decision backend that replaces hand-coded heuristics.** The advanced
  rms_norm hard-coded `_select_k` (minimize `Wt//K + K`) and `_select_transport` (a baked-in
  bake-off verdict) — themselves instances of the "scattered hardcoded approximations" this
  tool exists to kill. noc_estimator can supply the real numbers: cost a `ONE_TO_ALL` mcast
  at `num_subordinates=K` vs a gather, at dispatch time, to pick K and the transport **without
  the on-device sweep the human ran.** So it is not just a detection signal — it is a backend
  a generated op can call *natively* to make these choices.

**Blind spot:** isolated single-transfer cost only — **no workload-level congestion** (that
is tt-npe's job) and no compute/sync (Tracy's). Accuracy is bounded by the empirical grid;
off-grid configs degrade through interpolation/relaxation.

### Instrument 2 — tt-npe (the NoC workload model, with congestion)

A lightweight analytical NoC simulator (`https://github.com/tenstorrent/tt-npe`). It consumes a
workload = a trace of `noc_async_read`/`write` transfers and predicts cycles + utilization +
congestion over a static device model. Modes: **profiler** (re-simulate a captured NoC trace,
enabling free congestion-on/off "what-if") and **synthetic** (construct the transfer pattern
in Python and simulate with **no device at all**).

Signals it emits (`npeStats.hpp`):

```text
dram_bw_util (+ dram_bw_util_per_controller)   # DRAM saturation; per-controller = bank hotspots
overall_avg/max_link_util                      # NoC link utilization (max = hottest link)
link_demand vs link_util                       # demand can exceed 100%; the GAP is contention
getCongestionImpact()                          # % of runtime recoverable if congestion vanished
estimated_cycles vs estimated_cong_free_cycles vs golden_cycles   # built-in ablation + validation
noc0_* vs noc1_*                               # read-NoC vs write-NoC
per_timestep_stats + link/niu demand grids     # spatial+temporal hotspot map
```

In a ttnn perf report this surfaces as `DRAM BW UTIL` and `NOC UTIL`.

**Blind spot (load-bearing):** tt-npe models *only* data movement — **no compute, SFPU,
CB-backpressure, or sync.** rms_norm's true bottleneck (serial read→square→reduce→finalize→
write chain, 2.4 µs SFPU finalize init, one-time first-write burst) is largely outside its
model. So when npe reports "NoC has headroom, low congestion" yet the op is slow, that is
itself a **by-elimination signal** pointing at compute/sync — exactly the rms_norm verdict.
Use npe for "are we NoC/DRAM/congestion bound?" and as a **no-device pre-screen** of a
candidate refinement's movement cost.

### Instrument 3 — Tracy device profiler (the compute / sync / critical-path axis)

`DeviceZoneScopedN("name")` (`tt_metal/tools/profiler/kernel_profiler.hpp`) emits a per-RISC
`ZONE_START`/`ZONE_END` cycle-timestamp pair into an L1 profiler buffer, dumped to
`generated/profiler/.logs/profile_log_device.csv`; op-level `DEVICE KERNEL DURATION [ns]`
comes via `ttnn.ReadDeviceProfiler` + `get_latest_programs_perf_data`. The macro is a no-op
unless JIT-built with the profiler env set, so it is **zero-cost in production / golden runs**.

**Trap:** zones from reader/compute/writer run concurrently and **overlap — they do not sum
to the total**, and a zone that *encloses* a `cb_wait_front` misattributes the blocked time
to the work (the rms_norm "5.9 µs square" that was ~90% read-wait). The fix is to wrap
`cb_wait_front` / `cb_reserve_back` in their own named scopes → per-CB consumer/producer stall
(big `WAIT_*` ⇒ producer too slow; big `RESV_*` ⇒ consumer backpressure). This is the concrete
implementation of measurement primitive #1, and it is **mechanizable** (a deterministic
source transform), with one caveat — per-fire zone overhead + finite L1 buffer mean you wrap
**low-frequency** wait points (row/block granularity), not per-tile inner-loop waits.

**Shared substrate:** Tracy's `--collect-noc-traces` capture is what feeds tt-npe's profiler
mode.

### Signal → tool-backend map

This is the practical payoff: each general memory candidate's *detection* field gets a
concrete backend (and the ones tt-npe cannot see stay on the Tracy track).

| Optimization memory | Detection backend |
|---------------------|-------------------|
| `split_wide_reduction…` / `reduce_redundant_dram_reads…` | `grid_occupancy` (Tier-1 static) + `dram_bw_util`, per-controller (npe) |
| `choose_collective_algorithm…`, `tune_parallelism_factor…` | **noc_estimator** per-transfer cost (mechanism × `num_subordinates`=K × pattern) — cheapest, host-runtime; then npe `getCongestionImpact()`, `max_link_util`, demand>util gap for the *concurrent* contention the per-transfer lookup misses |
| `batch_noc_writes…`, `stream_output_drain…`, `improve_dataflow_overlap…` | **noc_estimator** size→bandwidth curve (quantifies the batch win pre-kernel) + Tracy CB `WAIT_*`/`RESV_*` zones (the `WR-wait`/`RDR-resv` signals) |
| `fuse_streaming_recurrence…`, `replace_seq_accumulation_with_reduce` | Tracy compute zones + intermediate-CB traffic |
| **vetoes** (`do_not_row_block…`, `stall_not_on_critical_path`) | **Tracy timeline (Tier 5) only** — cheaper tiers actively mislead here |
| "which recipes even apply" gate | arithmetic intensity (Tier 0) + npe `estimated_cycles` vs `golden_cycles` |

**Calibration chain** (the three tools are not independent — they form a
measurement → model → validation loop):

```text
Tracy / DM perf microbenchmarks ──feed──► noc_estimator's empirical table
                                ──could calibrate──► tt-npe's device model
Tracy golden_cycles ──validate──► tt-npe (cycle_prediction_error) and noc_estimator
```

Tracy is ground truth; noc_estimator is the cheap memoized per-transfer slice of it;
tt-npe is the analytical extrapolation to whole-workload congestion.

### Cross-cutting lessons (carry into the memory model)

- A signal is a **symptom, not a prescription** — the same 2× DRAM read routes to *split* /
  *share* / *restructure* depending on *why* you re-read.
- **Gate every signal on critical-path membership** — a large-looking stall is not always on
  the critical path (all the rms_norm vetoes — row-blocking, double-buffer, the two fusions —
  came from violating this).
- **Tools are axis-specific**: "NoC has headroom but the op is slow" is a by-elimination
  signal pointing off tt-npe toward compute/sync.
- The **free static prior prunes whole recipe families** before any device time
  (arithmetic intensity → "don't try compute fusion on a memory-bound op").
- The two doc-named primitives now each have a tool backend: per-CB wait/reserve timing →
  Tracy CB-wait zones; per-role DRAM read counting → tt-npe / NoC trace.


## Proposed Pipeline

The contrastive loop: mine a generated-vs-optimized delta, turn it into a refinement
hypothesis, run it, and promote only what measures. Both pairs sharpen two stages that the
SDPA-only version got wrong — *which pairs to mine* and *deriving the bottleneck before
matching recipes*.

### 0. Classify the bottleneck before matching any recipe

Run the cheap detectors first (see Measurement & Tooling Landscape): arithmetic intensity
(memory- vs compute-bound — almost always memory at the start), then `grid_occupancy`,
per-role DRAM redundancy, and the noc_estimator/npe signals. A recipe's applicability is
**conditioned on the regime**: rms_norm being DRAM-latency-bound pre-vetoed every
compute-fusion hypothesis, while the same fusion could help a compute-bound op. Skip recipes
whose regime does not match before spending a refinement on them.

### 1. Generate candidate ops (with full lineage)

Create baseline generated ops across target families, preserving prompt/config, branch/commit,
design + requirements docs, source, tests, refinement history, and correctness/perf
measurements.

### 2. Pair with an optimized reference — provenance matters

Two kinds of pair, with very different signal quality (this is a direct lesson from holding
the two pairs side by side):

- **generated vs mature human op** (SDPA vs official C++): high coverage but **noisy** —
  feature, legacy, architecture-workaround, and cross-paradigm deltas dominate; the
  `feature delta != optimization delta` filter does heavy lifting, and candidates must be
  diffed against the *actual* generated op or already-present features masquerade as
  opportunities.
- **generated vs a vibe-coded refinement of the same generated op** (rms_norm): same
  framework, so the delta is almost purely performance; it is **pre-measured**; and it
  **carries the author's vetoes** (the experiments they tried and reverted). This is the
  cleaner, more scalable source — a human improving a working baseline through the agent is a
  repeatable pair-factory.

```text
feature delta != optimization delta
```

Label every delta — performance / feature / architecture-constraint / correctness-guard /
legacy / API / unknown. Only *performance* becomes a direct refinement candidate; the rest
become preconditions or veto rules. Track per pair: generated branch/commit, reference
branch/commit + **provenance type**, semantic-equivalence level, and missing-features vs
missing-optimizations.

### 3. Extract static features from both sides

op type; TTNN vs TT-Metal layer; kernel count + reader/writer/compute split; CB count/depth;
tile shape + dtype; layouts; sharding/multicast + semaphores; NoC read/write pattern; DRAM vs
L1 staging; block sizes; core grid + work partitioning; reduction axes; streaming-loop
structure; intermediate materialization; mask/edge handling; L1 footprint; output-layout
constraints. (`grid_occupancy` and per-role DRAM-read counts are the two cheapest and
highest-yield — see Tooling.)

### 4. Contrast semantically, not textually

Ask: what algorithmic strategy / memory-movement avoidance / reuse / work-partitioning trick /
encoded hardware constraint / implied negative rule does the reference have that the generated
op lacks? Ground every answer in the *actual* generated op — never an assumed baseline (both
pairs surfaced already-present features that look like opportunities if you skip this).

### 5. Produce refinement hypotheses

```yaml
hypothesis_id: ...
priority: high | medium | low
target_files: [...]
change_intent: ...
bottleneck_regime: <regime this only helps in>    # gate from step 0
detection_signal: <the tool field that flags AND will confirm it>
required_preconditions: [...]
expected_metric_change: [...]
risks: [...]
validation_plan: [...]
```

Prioritize by **value vs blast radius**: prefer low-risk, localized deltas with a clear
detection signal first (prove the loop), defer coupled/architectural bundles. Both pairs show
the biggest wins (cross-core forwarding / split) are also the highest blast radius.

### 6. Run refinements

Apply one hypothesis at a time; record parent attempt, patch summary, files changed, the
**commit SHA**, compile/correctness/performance result, failure mode, and whether the attempt
becomes the new base.

### 7. Promote — verified evidence only

A delta is a *hypothesis* until a committed refinement on a generated op proves it. Promote on:
correctness pass + measured delta beyond a threshold (or a documented tradeoff) + understood
applicability/regime + reusable beyond one op instance. The promoted entry stores the
**refinement SHA** as its evidence (an agent diffs before→after to learn the recipe). Store
failed attempts as **negative/veto evidence** with the same SHA shape — both pairs prove vetoes
are as valuable as wins (rms_norm's row-blocking / double-buffer / fusion reverts; SDPA's
feature-mode exclusions).

## Proposed Memory Entry Shape

The entry separates **`origin`** (where the idea came from — unverified, prose OK, reference
only) from **`verified_evidence`** (committed refinements an agent can diff). `status` is
`promoted` only when `verified_evidence` holds ≥1 entry with `correctness: pass` and a measured
`delta` — `origin` alone keeps it `hypothesis` ("no measured refinement result → no promoted
memory"). The two pairs add three fields: a **`category`** (the clusters they revealed), a
**`bottleneck_regime`** applicability gate, and **`recurs_in`** (cross-op sightings — a recipe
seen in ≥2 ops is a stronger general pattern).

```yaml
memory_id: tt_stream_blockwise_reduction_with_online_state
status: hypothesis | promoted        # promoted requires >=1 verified_evidence (pass + delta)

category: memory_access_reuse | work_distribution_grid_occupancy | dataflow_overlap | numerics
bottleneck_regime: [dram_latency_bound]   # regimes where this recipe can help (gate, step 0)
recurs_in: [sdpa, rms_norm]               # cross-op evidence -> stronger generality
general_pattern:
  Avoid materializing a large intermediate by streaming blocks and maintaining online state.
applies_when:
  op_pattern: [attention, reduction-like blockwise recurrence]
  intermediate_materialization: large
  memory_pressure: high
requires_code_features:
  - blockwise loop over a reduction or sequence dimension
  - partial accumulator state
preconditions:
  - L1 estimate fits selected input and state blocks
  - recurrence is mathematically equivalent to materialized form
  - output layout remains unchanged
transformation:
  - stream input blocks through circular buffers
  - maintain running state in L1/CBs
  - apply correction when new blocks update normalization or scale
  - write final output only
depends_on: [...]                          # prerequisite recipes (coupling — see Risks)
veto_rules:
  - do not select when required accumulator mode is unsupported
  - do not increase block size beyond estimated L1/CB capacity
expected_metric_change:                    # the HYPOTHESIS (pre-measurement)
  - lower DRAM/intermediate traffic
  - lower peak L1/intermediate footprint

# WHERE THE IDEA CAME FROM — unverified, prose OK, NEVER sufficient to promote
origin:
  ref_type: human_cpp | vibe_coded_same_framework
  branch: origin/main
  paths: [...]
  note: official SDPA forwards/mcasts K/V across cores sharing a (batch,head)

# PROOF IT WORKS — machine-actionable; an agent `git show`s the SHA to learn the recipe
verified_evidence:                         # [] while status: hypothesis
  - applied_to: <generated op branch + op dir>
    refinement_sha: <SHA or SHA-range>      # diff before->after = the transformation
    changed_paths: [...]
    correctness: pass
    delta: { metric: device_kernel_time, before_us: ..., after_us: ..., speedup: ... }
    detected_by: <Tracy zone / npe field / static signal that flagged AND confirmed it>

# negative / veto evidence: SAME shape, delta shows a regression -> feeds veto_rules
negative_evidence:
  - applied_to: ...
    refinement_sha: ...
    correctness: pass
    delta: { metric: device_kernel_time, speedup: <1.0 }
    lesson: row-blocking a DRAM-latency-bound kernel only amortizes compute init (not the bottleneck)
```

## Retrieval: Symptom → Strategy Lookup (and the samples branch)

The perf-acceptance gate answers only *"did this attempt win?"* — one number (device time),
plus the correctness/SUPPORTED checks. It does **not** answer *"given what I measured, what
should I try?"* That question is the **long-term memory's retrieval layer**, and it is a
separate component from the gate. The gate *validates*; the memory *proposes*.

Operationally the memory is a **symptom-indexed registry** over the entries defined above,
paired with a **`samples-perf` branch** that stores one before→after commit per strategy:

```yaml
# perf_memory.yaml — keyed by symptom, gated by regime
- symptom: low_grid_occupancy               # active_cores/total < T on a wide-narrow shape
  regime: [dram_latency_bound]
  strategies:                               # ranked
    - id: tt_split_wide_reduction_across_cores
      expected_win: 2-6x
      blast_radius: high                    # semaphores, hang surface
      sample: samples-perf  <sha>^..<sha>   # git diff = the recipe to study
      confirm: [grid_occupancy↑, per_core_dram_bytes↓, device_time↓]
    - id: tt_sequence_parallelism            # alternative strategy for the same symptom
      sample: samples-perf  <sha2>^..<sha2>
- symptom: high_dram_bytes_per_output_tile
  regime: [dram_bandwidth_bound]
  strategies:
    - id: tt_reuse_input_across_output_blocks
      sample: samples-perf  <sha3>^..<sha3>
```

Two mechanics make it operational:

1. **The `samples-perf` branch** — one before/after commit pair per strategy. The `sample`
   field is a SHA range; `git diff <sample>` **is** the machine-actionable recipe (the
   `verified_evidence.refinement_sha` from the entry schema, made browsable). This is where
   the OpComparer's job lands: turn a sample diff into *guidance text* the blind Implementer
   follows — the Implementer sees the description and its confirm-signals, never the diff.

2. **Retrieval flow** — detectors (Tier-0/1 signals + the gate's measured signals) emit a
   **symptom vector** (`grid_occupancy`, `dram_bytes_per_output_tile`, …) → filter by
   `regime` (so a latency-bound op is never handed a row-blocking recipe) → rank the
   surviving strategies by expected-win / blast-radius / applicability-match → hand the top-K
   to the Implementer as guidance + the sample diff + the confirm-signals the gate will check.

```text
detectors -> symptom vector -> filter by regime -> rank strategies
          -> guidance (from sample diff) -> blind Implementer -> perf-acceptance gate
```

**The loop is self-populating.** When the gate confirms a win, commit the before/after to
`samples-perf` and append an entry keyed by the symptom that triggered it; a measured
regression is committed the same way as **negative evidence** (its `confirm` signals show why
it did not pay off). That is the journal's *promote* step made concrete — the registry grows
from the framework's own measured refinements, not from prose.

**PoC seed:** rms_norm already supplies a real first entry — the `low_grid_occupancy →
tt_split_wide_reduction_across_cores` row, whose before/after exists as actual commits and
whose confirm-signals (`grid_occupancy`, per-core DRAM bytes, device time) all move hard and
in the same direction.

## Important Risks

### Coupled optimizations — the dominant risk for the big wins

Human/optimized implementations bundle mutually dependent optimizations; extracting one in
isolation tends to break. The sharpest evidence is rms_norm's headline P0 (cross-core
W-split): it is a *bundle* — regime selection + mcast transport + semaphore protocol + combine
compute + padding — and it was **numerically broken on first implementation** (output too large
by `sqrt(2·num_chunks)`, a cross-thread CB-pointer race) and needed a dedicated debugging pass.
SDPA's cross-core K/V forwarding is the same shape (semaphores, hang surface). So a memory entry
for these must carry its `depends_on` graph **and its known failure modes**, or an agent applying
it will hang/miscompute where the human had to grind. The memory must represent recipe
dependencies.

### Feature support vs performance

Reference code carries broader feature coverage. SDPA is the clean example: chunked/page-table
prefill, sliding window, attention sink, MLA are *feature* deltas, not perf — they are excluded
as refinement candidates (mine only if those modes become targets). Label every delta; only
*performance* is a direct candidate; the rest become preconditions or veto rules.

### Static-analysis overclaiming — measurement gates promotion

Static contrast proposes; only measurement disposes. rms_norm's three reverted compute fusions
(square+reduce, finalize-rsqrt, reader↔square overlap) all looked good statically and were each
net-negative on device — and the trap behind them is that **a large-looking stall is not always
on the critical path** (`WR-wait`, `RDR-resv`, a 5.9 µs "square" zone that was 90 % read-wait).

```text
no measured refinement result -> no promoted optimization memory
```

### Contrast-quality / provenance

Not all pairs are equal. generated-vs-mature-C++ pairs are noisy (feature/legacy/cross-paradigm)
and need heavy filtering; generated-vs-vibe-coded-same-framework pairs are clean, pre-measured,
and carry vetoes. Prefer the latter, and record the `ref_type` so the mining stage knows how
much filtering to apply.

### Bottleneck-class mismatch

The same recipe flips sign with the regime: row-blocking helps a compute-bound op and *regresses*
a DRAM-latency-bound one. A recipe matched without first classifying the bottleneck (step 0) is a
coin flip — applicability must be conditioned on `bottleneck_regime`.

### Sparse initial coverage

Early on, many generated ops match no memory entry. Make that explicit:

```text
unknown bottleneck
no matching memory
route to contrastive mining or expert review
```

Unknown cases are useful — they tell us where the memory must grow.

## PoC Starter Signals & Measurement Recipes

To build lookups you need *some* signals. Grounded in the deltas the two pairs actually
exhibited, start with three (most are not available out of the box — they are computed by
post-processing dispatch metadata, the Tracy device CSV, or a NoC trace):

| # | Signal | Detects (strategy family) | Source / how to build | Effort |
|---|--------|---------------------------|-----------------------|--------|
| 1 | **`core-occupancy`** = **occupancy × balance** = `Σdᵢ/(N·max dᵢ)` — refined from raw `grid_occupancy` | low **occupancy** → cross-core split / parallelize; low **balance** → rebalance work | **BUILT** (see checkpoint below): empirical, per test cell, from Tracy CSVs — no kernel edits. | done ✓ |
| 2 | `device_kernel_time` (the gate, not a symptom) | accept/reject any refinement | Exists — `profiling.py` sums Tracy `DEVICE KERNEL DURATION`. | ~free |
| 3 | `dram_bytes_per_output_tile` (or `input_read_redundancy` = read_bytes / unique_bytes) | high traffic → **reuse / forwarding / kill re-read** (SDPA P0, rms_norm two-pass) | Post-process the NoC trace (`--collect-noc-traces`): Σ read-transfer bytes ÷ output tiles. Or npe `DRAM BW UTIL`. | medium |

`grid_occupancy` + the existing `device_time` gate is enough to run the rms_norm seed
end-to-end; add `dram_bytes_per_output_tile` to prove the lookup handles a second symptom.
Defer per-CB wait-fraction, npe congestion, and arithmetic-intensity until the first loop works.

### Measurement recipes (these should become a `measure-perf-signal` skill)

Each recipe is **instrument → measure (best-of-N, record the noise floor) → revert**. The
ablation recipes edit kernels and MUST preserve the CB protocol — keep every
`cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` with its original
counts; neutralize the *body*, never delete the CB calls, or the kernel deadlocks.

1. **`grid_occupancy`** — static: read the program's core grid usage from the descriptor. No
   kernel edit. (Fallback / confirmation: count distinct cores in `profile_log_device.csv`.)
2. **`device_kernel_time`** — `profiling.py` (Tracy `DEVICE KERNEL DURATION`). No edit.
3. **`dram_bytes_per_output_tile`** — capture the NoC trace, sum read-transfer bytes by tensor
   role ÷ output tiles (or read npe `DRAM BW UTIL`). No kernel edit — it's a trace post-process.
4. **compute-on-critical-path? / compute-vs-transfer split** (ablation builds, revert after):
   - *transfer floor*: neutralize compute — replace the math in the compute kernel with a
     pass-through copy, keep all CB waits/pops/pushes. If total time ≈ unchanged → not
     compute-bound (the movement is the floor).
   - *read floor / write floor*: neutralize writes in the writer (keep `cb_pop_front`) or reads
     in the reader (keep `cb_push_back`, junk data — timing only) to isolate which movement
     dominates.
   - *per-CB stall*: wrap `cb_wait_front` / `cb_reserve_back` in their own `DeviceZoneScopedN`,
     parse `profile_log_device.csv` for the stall fraction. Wrap **low-frequency** points only
     (row/block granularity) — per-fire zone overhead + finite L1 profiler buffer.

**Why a skill:** these are repeatable procedures with sharp, easy-to-get-wrong gotchas
(preserve the CB protocol, revert the instrumented build, best-of-N against a noise floor,
wrap only low-frequency CB points). Packaging them as a `measure-perf-signal` skill — one
section per signal — lets the framework and agents run them consistently and is the natural
home for the ablation discipline.

### Checkpoint — signal #1 built & validated (`core-occupancy` skill)

The first PoC signal is done, and it sharpened in the building:

- **Metric refined to `efficiency = occupancy × balance`** (= `Σdᵢ / (N · max dᵢ)`, where
  `dᵢ` = per-core kernel duration, idle cores = 0): 1.0 perfect, `1/N` worst. It factors
  exactly — `occupancy = n_busy/N`, `balance = mean_busy/max` — and **you must report all three**,
  because a low efficiency alone can't tell *idle cores* (→ parallelize) from *uneven work*
  (→ rebalance).
- **Both factors are NO-EDIT, empirical, from the profiler CSVs** (this was the key discovery —
  no `WORK_UNIT` marker needed for occupancy *or* balance):
  - `occupancy` = `CORE COUNT / AVAILABLE WORKER CORE COUNT` (ops-perf CSV) — cross-checked
    against distinct active cores in the raw `profile_log_device.csv`. Denominator is the op's
    **available** grid, not the physical `Max Compute Cores`.
  - `balance` = per-core `*-KERNEL` zone durations (cycles → ns via `CHIP_FREQ`) from the raw
    CSV; `mean/max`.
  - The `WORK_UNIT` marker is now demoted to a *separate, deeper* recipe — only for the discrete
    **unit-count** per core (to attribute imbalance to count vs per-unit cost), and only it hits
    the ~250-marker/core budget.
- **It's a per-(op × shape × grid) test-cell measurement** — one run → one triple. Occupancy is
  shape-dependent; sweeps are the skill looped over cells.

**Validated end-to-end** on the human SDPA op (`sdpa_standard_v2`, WAN shape `b1·nh10·s9472·d128`,
Blackhole 11×10): a fresh agent, oriented *only* by the skill, reproduced the numbers on two cells —

| cell | occupancy | balance | efficiency | per-core dist. |
|------|-----------|---------|------------|----------------|
| q288 / k512 | 1.000 | 0.995 | **0.995** | single cluster (~1% NoC gradient) |
| q224 / k512 | 1.000 | ~0.80 | **~0.80** | **bimodal** — 100 cores @ 4 chunks, 10 @ 3 (ratio 4/3) |

The punchline: **occupancy alone read 1.000 for *both* cells** — it saw "grid full, perfect" —
yet q224 is ~20 % less efficient. Only the **balance** factor caught it (an even work-count
split at q288 vs a 4-vs-3 imbalance at q224). Concrete proof of why the signal must be the
decomposition, not just cores-used — and a clean symptom→lever map (occ≈1, balance≪1 →
*rebalance the work-split*, not *parallelize*).

Skill: `core-occupancy` (SKILL.md) — op-agnostic; the agent supplies one run-once-on-shape
command. Committed on `dnijemcevic/perf_signals`. Next starter signals (`dram-bytes-per-tile`,
per-CB stall) remain to build.

### Next skill seed — `dram-bytes-per-tile` (DRAM reads by tensor role)

The next signal to build (starter signal #3), grounded and ready. **Goal:** for one op × shape
× config cell, measure **DRAM-sourced read bytes per tensor role (Q/K/V/…) per input (or output)
tile** — the signal that makes the SDPA K/V multicast win visible.

**Why this is the multicast signal (grounded in `reader_interleaved.cpp`, official SDPA):**
- A chain **receiver** core does **not read K/V from DRAM** — it gets them over the NoC (L1→L1)
  from the previous core via a semaphore handshake (`should_receive` → `receiver_sem.wait(VALID)`).
- The **injector** reads K/V from DRAM **once** (`read_chunk_for_forwarding`) then forwards:
  `mcast_enabled` → one `noc.async_write_multicast(… mcast_num_dests …)`; else unicast relay.
  Gated by `mcast_enabled` (CT arg 32), non-causal single-chip only.
- Net: multicast/forwarding turns *"every core reads its own K/V from DRAM"* into *"one injector
  reads K/V from DRAM per `(batch, head)`, the rest receive over the NoC."*

**The discriminating metric — DRAM reads BY SOURCE, not total NoC bytes.** Multicast doesn't
reduce total data moved; it moves K/V over the on-chip NoC instead of DRAM→L1. So:
- headline: `input_read_redundancy(role) = DRAM_read_bytes(role) / unique_bytes(role)` →
  ≈ **1** with mcast (read once, shared), ≈ **num cores sharing the (b,h)** without.
- or `dram_read_bytes_per_output_tile`, split by tensor role.
- complementary tell: **core-to-core NoC (L1→L1) K/V traffic appears** with mcast, absent without.
- **Total NoC bytes would hide the win** — must attribute reads to `src = DRAM`.

**Measurement path:** Tracy `--collect-noc-traces` tags every transfer with src/dst (DRAM bank
vs core L1); filter `src = DRAM`, group by tensor address/role → DRAM bytes per role; divide by
output tiles (or unique tiles for the ratio). tt-npe `dram_bw_util` / per-controller is the
aggregate cross-check. Per-cell like occupancy (one op × shape; agent supplies a run-once
command). Follow the `core-occupancy` skill as the structural template.

**Validation idea:** on the SDPA WAN cell (non-causal, mcast on) K/V redundancy should read ≈ 1;
contrast against a config where forwarding is off (or the generated SDPA, which has none) to
show redundancy ≈ num-sharers — the concrete multicast delta.

### Checkpoint — signal #3 built & validated (`dram-bytes-per-tile` skill)

Built and validated, but the measurement path changed materially from the seed's plan — the
Tracy NoC-trace approach the seed assumed does **not** work for this signal:

- **Metric (per input arg):** `reads_per_output_byte(role) = dram_read_bytes/output_bytes`
  (impact) **and** `read_redundancy(role) = dram_read_bytes/unique_input_bytes` (detector; ≈1
  ideal, >1 = re-read). Report a per-role table — the aggregate hides which tensor is redundant.
- **Load-bearing discovery — the Tracy NoC-event trace is unreliable for DRAM-read accounting.**
  The device profiler is *drop-don't-stall by design* (stalling would perturb the timing it
  measures) and additionally **refuses to flush while a linked multicast is in flight**
  (`quick_push` bails on `NOC_CMD_VC_LINKED`, issue #22578). The heaviest-reading cores
  (forwarding *injectors*) are exactly the ones whose K/V reads get silently dropped — captured
  DRAM reads came out ~⅓ of reality even at tiny scale, and **no runtime knob fixes it**
  (`PROGRAM_SUPPORT_COUNT`, mid-run dump, profiler-sync all had zero effect). So `--collect-noc-traces`
  + tt-npe (or raw-JSON parsing) systematically undercounts the very signal we want.
- **tt-npe status:** it *consumes* the same lossy trace (profiler mode re-simulates the captured
  events — it cannot recover dropped ones), so it inherits the undercount. Its real value is
  congestion / `dram_bw_util` / per-controller / link-util, and *synthetic* mode. It also didn't
  run on this box until patched: `Unknown device model: P150_X4` → fixed by mapping `P150_X4` to
  the single-chip Blackhole model (committed on tt-npe branch `dnijemcevic/p150_x4-device-model`,
  local clone at `/localdev/dnijemcevic/tt-npe`).
- **The method that works (what the skill teaches): temporary in-kernel per-role DRAM-byte
  counters + `DPRINT`, run with the profiler OFF.** A single counter emitted once per core is
  immune to the ring/flush/linked-mcast drop. Count only DRAM-sourced reads (exclude
  forwarded/received L1 data and padding); instrument at the op's *own* reader call sites, not
  shared dataflow headers; capture all worker cores and sum. Reader = NCRISC (`DPRINT_RISCVS=NC`).
- **Revert discipline isolated to a shared reference** — `.claude/skills/shared/revert-temp-edits.md`
  (snapshot-then-restore into the scratchpad, manifest journal, hash-checked, **never**
  `git checkout` — the file may hold the user's uncommitted work). `core-occupancy` now references
  it too (for its optional temp `DeviceZoneScopedN` marker).
- **Validated end-to-end** by fresh agents oriented *only* by the skill (twice, both exact), on the
  SDPA WAN cell `wan2_2_1xGLX_analog-k512-q288-bf16` (b1·nh10·s9472·d128, 11×10 grid):

  | role | dram_read_bytes | reads/output | redundancy | forwarding OFF (redundancy) |
  |------|-----------------|--------------|------------|------------------------------|
  | Q | 24.25 MB | 1.00 | **1.00** | 1.00 |
  | K | 72.7 MB | 3.00 | **3.00** | **33.0** |
  | V | 72.7 MB | 3.00 | **3.00** | **33.0** |

  The multicast (11 cores/head, 1 injector) gives an **11× reduction** in K/V DRAM reads
  (33→3 = cores-per-head); the residual **3×** is the injector re-reading its head's K/V once per
  q_chunk it owns (no caching across q_chunks) — a concrete second lever. Forwarding-off was
  proven by a temporary program-factory toggle (`needs_rebuild`), also reverted.

Skill: `dram-bytes-per-tile` (SKILL.md) — op-agnostic; agent supplies one run-once command,
finds the reader read-sites itself. Committed on `dnijemcevic/perf_signals`. Remaining starter
signal (per-CB stall) still to build.

## Near-Term Implementation Sketch

We now have two validated pairs (SDPA, rms_norm) and the tooling tiers. Next steps:

1. **Build the two cheapest detectors first** — static `grid_occupancy` and per-CB Tracy
   wait/reserve zones (the two named primitives). They cover the highest-value signals at
   near-zero cost.
2. **Prove the loop end-to-end on the cheapest low-blast-radius delta with a clear detection
   signal** — e.g. SDPA P0#2 (causal work balancing, host-side, no semaphores) or a
   batch-writer-class change — apply it as a refinement, measure on device, and store the SHA
   (positive *or* veto). This validates "contrast → hypothesis → refine → measure → promote"
   before betting on a coupled bundle.
3. **Only then attempt a coupled P0** (cross-core forwarding/split), where the failure-mode
   memory matters most and the cost of a missing prerequisite is a hang.
4. **Grow the pair supply** by harvesting vibe-coded refinement sessions (baseline vs
   human-improved-through-the-agent), not only hand-mapped human references.

Promote only measured, reusable deltas — positive or veto — into the first memory entries.

## Working Definition

For this framework, an optimization memory entry is:

```text
a measured, reusable transformation recipe
with a category and a bottleneck-regime it applies in,
explicit applicability conditions, preconditions, dependencies, and veto rules,
implementation cues,
and VERIFIED evidence: the commit(s) where the recipe was applied to a
generated op, passed correctness, and moved a measured metric —
machine-actionable (an agent diffs the SHA before->after), not prose.
```

The memory is built by mining generated-vs-optimized deltas across pairs, but a delta is only a
*hypothesis* until a committed refinement on a generated op proves it. The reference is the
**origin** (a source, reference-only); the **verified_evidence** SHA is what earns promotion;
and a **veto** entry (a measured regression) is as first-class as a win. Two pairs in, the
memory already shows two categories — *memory-access / reuse* (SDPA) and *work-distribution /
grid-occupancy* (rms_norm) — with recipes that recur across both, which is the signal that a
pattern is genuinely general rather than one op's trick.
