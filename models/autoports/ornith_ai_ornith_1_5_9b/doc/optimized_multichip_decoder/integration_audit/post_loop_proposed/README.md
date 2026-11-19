# Pending alternative: fused postnorm and repeated heads in one launch

**Pending cumulative hardware, direct watcher, and final-default validation. Do not select this draft on the strength of the independent postnorm and loop candidates alone.** This directory is a second, independent proposal. The original `integration_audit/proposed`, its patch, and running production/tests are unchanged.

`tt/` contains the proposed production files. `integration.patch` applies directly to the current production baseline, not on top of the first proposal. `source_manifest.json` freezes the copied source hashes and records host-only checks. The builder refuses a production baseline different from the one used for the first proposal. No hardware was run for this preparation.

## Change relative to the first proposal

The tiled-history allocation, compact state reads, direct history row writes, complete-pool predicate, private/subset handling, PartialRoPE splice, production CCL grid fix, KV contract and logical-length handling are identical to the first proposal. The pooled linear path alone replaces `flat_ordered` plus its external output permutation, BF16 conversion, RMSNorm, reshape, and SiLU gate multiplication with:

```python
return flat_ordered_post_loop(
    conv, g, beta, recurrent, z,
    self.weights["linear_attn.norm.weight"], self.config.rms_norm_eps,
    self.decode_recurrent_wave,
)
```

The copied candidate remains in its passing no-gate-padding configuration. The helper returns BF16 L1 interleaved `[1,1,B,1024]` directly, matching the local output-projection input contract. Recurrent state is still updated in place in FP32. This replaces only the complete ordered pooled path; private B1 and incomplete/permuted/mixed pools retain the first proposal's paths.

Promote `tests/multichip_flat_ordered_post_loop.py` to `tt/flat_ordered_post_loop.py`, and its four kernel/header files to `tt/kernels/flat_ordered_post_loop/`. Retain `tt/{tiled_conv,history_row_write,partial_norm_rope}.py` and their eight kernel/header files from the first proposal. The ordinary `flat_ordered` helper/kernel tree is unnecessary for this alternative. All promoted kernels are unchanged copies; no module imports model tests.

## Startup and repeated-head scheduling audit

`kernels/flat_ordered_post_loop/compute.cpp:181–184` calls `compute_kernel_hw_startup(20,27,0)` exactly once in `kernel_main`, before the head loop. Neither `process_head` nor the local norm header calls startup. The shared `chunk_gdn_math.hpp` implementation of `scan_step` also has no startup call. Per-operation data-format reconfiguration remains inside the compute stages; it does not reset kernel-wide synchronization between heads.

The helper launches one program, using `count=min(wave,heads)` cores. Reader and writer handle `first, first+count, ... < heads`; compute receives `(heads-first+count-1)//count` iterations. These schedules agree for every B1..32 and wave1..110: 3,520 host checks. At the selected wave104, B32 has 48 cores processing three heads and 56 processing two; B6 has 48 cores processing one head. B14 and B27 are useful boundary cases for two and three iterations respectively.

Each head owns a distinct 16-tile FP32 state slab at `head*16`. A core can read its next head while the writer drains the previous head because the state slabs are disjoint. The final-state writer barriers before popping CB7; a new compute iteration cannot overwrite CB7 until that pop permits its reservation.

The output writer emits two 32-byte face-row segments for each of four channel tiles. `batch=head/8` selects the physical row and `hv=head%8` selects the four channel tiles (`writer.cpp:9–22`). Different logical heads own disjoint byte intervals even when they share output tiles. Host checks found no overlapping writes for any B1..32. Output physical padding beyond logical B is not written; downstream consumers must continue honoring logical B, as the candidate path does.

## Circular-buffer ownership across iterations

| Buffers | Producer and consumer | Reuse discipline |
|---|---|---|
| CB20 / CB22 | Reader produces Q/V, then gamma/gate, then next-head Q/V; compute consumes all phases | `reader.cpp:88` reserves the full four-page capacity after preparation pops Q/V (`compute.cpp:90`). The next head cannot refill those buffers until postnorm pops gamma/gate (`compute.cpp:125`). No BRISC producer writes these buffers. |
| CB21 / CB23 / CB24 | Reader to compute | Preparation pops each head's K/g/beta. The next reader iteration may fill them early, but compute consumes heads sequentially. |
| CB27 | Reader produces one unit-scaler tile; compute uses it for Q/K and postnorm | Preserved through preparation and popped only after postnorm (`compute.cpp:126`); next-head reservation waits. |
| CB14 | Reader produces identity; compute uses it in scan | Popped after scan (`compute.cpp:177`); each iteration recreates its identity tile. |
| CB28 | Reader-private scratch, no published queue entries | Only NCRISC accesses it, with read barriers before reuse. Its repeated `reserve_back(1)` is scratch reservation; there is no competing producer/consumer. |
| CB29 | Compute only | BF16-rounded recurrent result is popped before reuse for BF16-rounded SiLU gate (`compute.cpp:105–117`), then popped after multiply. |
| CB30 | Compute only | BF16 normalized output is consumed and popped in the same head. |
| CB31 | Compute to writer | Four output pages per head; writer waits, writes disjoint logical rows, barriers, then pops. |
| CB5 / CB7 | Reader-to-compute old state / compute-to-writer final state | Full state slab transfer; scan consumes old state, writer drains final state. Both preserve per-head FIFO ordering. |

There is no source-visible producer-ownership conflict in this arrangement. This is not a substitute for watcher evidence: cumulative looped postnorm adds repeated reuse of all these resources that neither independent candidate fully validates.

## Numerical and memory constraints

The compute path preserves explicit BF16 packing of recurrent output, normalized output, and the fused SiLU activation before multiplication (`compute.cpp:95–126`). Removing any of these packs would change the candidate's numerical behavior. The unit scaler remains live until output normalization. Each head's FP32 recurrent scan remains the ordered implementation, with FP32 state and HiFi4 computation.

Persistent history/recurrent/KV memory is identical to the first proposal, including pooled history's 393,216 bytes per device for B2..32. The fused helper instead allocates a 65,536-byte physical BF16 L1 output tile plane. The first proposal's flat helper allocated `16,384*heads = 131,072*B` bytes of padded FP32 DRAM output before separate postprocessing. The fused compute configuration allocates 434,176 circular-buffer bytes per participating core, excluding program/runtime metadata. Recheck peak L1 availability and context after integration; do not infer capacity from these isolated allocation counts.

## Selection gates

The parent must first finish the cumulative candidate checks, including direct watcher retry. Then run the selected production default with unchanged PCC and rank-consistency gates: private B1; B2/B3/B6/B13/B14/B16/B27/B31/B32 pools; changing values and positions; 1024-answer recurrence; subsets/permutations/mixed pools; reset/re-prefill; live interleaved traces; cache rebinding with changed allocations; non-aligned and maximum-context lengths. Repeat watcher separately from profiling. Use final-default warmed prefill and traced warmed decode measurements; the independent candidate timings cannot establish this cumulative default's result.

Static checks completed: patch apply-check, Python parse/no-test-import audit, single startup call outside the compute loop, 3,520 matching head schedules, and disjoint output ownership for 32 batch sizes. Hardware status remains **pending**.
