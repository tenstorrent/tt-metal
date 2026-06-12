# minimal_matmul K-dim parallelism (split-K) — plan & state

Branch: `cglagovich/minimal-matmul-mcast-prefetch`. Op: `ttnn.experimental.minimal_matmul`
(DRAM matmul). This doc is the resume point for the split-K work; read it + the referenced
commits to continue after a context compaction.

## Why (motivation + measured ceiling)
On **output-starved** shapes — `M_tiles * N_tiles < num_cores` — pure M/N decomposition (incl.
core-grid slicing) can use at most `out_tiles` cores; the rest idle while the busy cores grind the
full sequential K reduction. Split the K reduction across the idle cores.

Emulated ceiling (faithful = `matmul(M, K/P, N*P)` = all 64 WH cores active, real in1 DRAM volume +
contention, PRE-reduction): `32x6144x512` P4 **1.7x**, `32x6144x256` P8 **2.9x**, `64x6144x128` P8
**3.8x**. Win grows with starvation, shrinks with in1 (N) width (wide-N is DRAM-BW-bound under split-K).
Engage when `out_tiles < cores` AND K is large. (Probe: `32xKx512` runtime is ~linear in K with ~3us
intercept → ~95% of it is K-proportional work on 16/64 cores.)

## Core mechanism (applies to A2 and B)
- **K-bands along grid ROWS.** Rows partitioned `num_k_slices` (Pk, OUTER K-bands) x `num_slices`
  (S, N-slice groups, nested) x `rows_per_group` (small-dim parallelism, innermost) =
  `grid.y / (Pk*S)`. Each K-band computes the FULL M/N over a `1/Pk` slice of K. For a given output
  tile the Pk partials live DOWN A COLUMN (same col, the Pk band-rows).
- **No DRAM base-address offset** (DRAM tensors are bank-interleaved → base+offset != page+offset).
  Instead: per-band K-BLOCK START index (`k_block_start`, added to the K-block index in the readers,
  full-K striding preserved) + per-band compile-time K-block COUNT (`padded_K_tiles` set per band) +
  output M-stripe offset (`out_m_tile_offset`).
- **Env knobs (sweepable):** `TT_MM_K_SLICES`=Pk, `TT_MM_NUM_SLICES`=S, `TT_MM_K_FUSED`=1 (B mode).
  All gated to the auto path (no pinned config, no fused ops).
- **Limitation (TT_FATAL-guarded):** no M-padding → `grid.y/(Pk*S)` must divide `M_tiles`. Fine for
  starved M=1–4-tile shapes at Pk=grid.y. (On BH grid.y=10 is not pow2 → only Pk in {1,2}, and M=1
  always pads → K-par effectively WH-only until padded-M is supported.)

## Files
- factory: `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp`
- device op (output spec): `.../device/minimal_matmul_device_operation.cpp`
- kernels: `.../device/kernels/{dm_in0_sender,dm_in1_sender_out,compute}.cpp`
- test: `tests/ttnn/nightly/unit_tests/operations/experimental/test_zz_ksplit.py`
  (runs minimal_matmul with the env set, reshapes the `[Pk*M,N]` output to `[Pk,M,N]`, sums dim 0, PCC)

## Status

### ✅ A2 (host-summed partials) — DONE, commit `de95e3add94`
Each band writes its partial into its own M-stripe of a `[Pk*M, N]` buffer; the **host** reshapes to
`[Pk,M,N]` and sums → `[M,N]`. Output writer = `dm_in0_sender` for non-transpose, `dm_in1_sender_out`
for transpose (BOTH carry the `out_m_tile_offset`). Injector core-spec covers all `Pk*S` group tops.
3 per-core runtime args: `k_block_start`, `out_m_tile_offset`, `out_M_tiles_total` (0/M_tiles when off
→ Pk=1 byte-identical). Validated **PCC 0.99999** on 32/64/128 x6144 x{256,512} at Pk=1/4/8.

### 🚧 B (fused on-device column reduction → single `[M,N]`, no host sum)
Topology: **LINEAR accumulate UP the column.** Bottom band (`kband==Pk-1`) emits its own partial;
each band adds incoming(from below)+own → running sum, forwards up; TOP band (`kband==0`) writes DRAM.

**✅ B half-1 — DONE, commit `b79994e58e2` (compute side + plumbing, GATED/guarded, can't run yet):**
- `compute.cpp`: `reduce_add_block(a,b,out)` elementwise-add helper; under `#ifdef REDUCE_K` the matmul
  output stage emits own partial (`is_reduce_bottom`) or `reduce_add_block(intermediate, cb_reduce,
  out_cb)` (non-bottom). `cb_reduce` = `CBIndex::c_7`. New always-read `is_reduce_bottom` compute arg.
- factory: `TT_MM_K_FUSED` detection (`num_k_fused`) + **TT_FATAL guard "not yet wired"**; per-core
  `is_reduce_bottom = (kband == Pk-1)` pushed into compute args (create path).
- device op: output spec `[M,N]` when B (`out_m_mult=1`), `[Pk*M,N]` for A2.
- Verified normal pcc 0.99998 + A2 pcc 0.99999 still intact.

**✅ B half-2 — DONE (the deadlock-prone half: reduction dataflow + final wiring).** WORKS, PCC 0.99999.
DM = the output-writer kernel (`dm_in0_sender` non-transpose / `dm_in1_sender_out` transpose). Per
output block, per core, under `#ifdef REDUCE_K` (replaces the deferred/immediate write; `defer_write`
forced false so out_cb is handled exactly once):
1. **Receive** (if `!is_reduce_bottom`): `cb_reserve_back(cb_reduce)` (blocks until compute popped the
   prev block → single-slot safe), set local recv-sem INVALID, `noc_semaphore_inc` the band-BELOW's
   ready-sem (tell it our slot is free), wait local recv-sem == VALID, `cb_push_back(cb_reduce)` so the
   compute's `cb_wait_front(cb_reduce)` unblocks.
2. **Emit** (after compute pushes `out_cb`): if `is_reduce_top` → `write_block_sync_granular` to DRAM
   (the final `[M,N]`, `out_m_tile_offset==0`); else → `cb_wait_front(out_cb)`, wait the band-ABOVE's
   ready-sem, `noc_async_write` out_cb into the band-above's `cb_reduce` L1 (`get_noc_addr(up, get_write_ptr(cb_reduce))`
   — single-slot ⇒ constant base addr matches across cores), `noc_async_writes_flushed`,
   `noc_semaphore_set_remote` the band-above's recv-sem VALID (VALID source = reuse `in{0,1}_valid_semaphore`),
   `cb_pop_front(out_cb)`. Wave flows bottom→top; acyclic (bottom never receives, top never sends).

Factory wiring (all DONE): `REDUCE_K` on compute + the output-writer DM only (separate `in0_defines`/
`in1_defines`); `cb_reduce` (`c_7`, 1 block, `output_data_format`) when `num_k_fused`; two semaphores
(`reduce_ready` on the sender/band-below, `reduce_recv` on the receiver/band-above); per-core up/down
neighbor physical coords (`±num_slices*rows_per_group` on `core.y`, clamped at the ends) + `is_reduce_top`
+ `is_reduce_bottom` + the two sem ids → 8 runtime args pushed to the writer kernel under `num_k_fused`;
`out_m_tile_offset=0`, `out_M_tiles_total=M_tiles` for B; guard now requires `num_k_slices>1` for B.

**Validated:** `test_zz_ksplit.py::test_ksplit_fused` ([M,N] direct, vs torch). PCC **0.99999** on
32×6144×{256,512}, 64×6144×{128,512}, 128×4096×512 at Pk∈{2,4,8} incl. `rows_per_group>1` (multi-row
bands). Normal path + A2 unaffected. **Perf (device kernel duration, B Pk=8 vs Pk=1 baseline):**
32×6144×256 **4.32x** (54.0µs→12.5µs); 64×6144×128 **2.64x** (51.0µs→19.3µs). Bench: `test_ksplit_perf`
under tracy. (Note: M=1-tile shapes need Pk=grid.y so `rows_per_group==1`; Pk<grid.y pads M → the
no-M-padding guard fires, same limitation as A2.)

## Gotchas
- **Dirty device → pcc=nan on the NEXT op** after a crashed/guarded/hung run. `tt-smi -r` clears it; NOT
  a regression. Always reset before trusting a fresh failure.
- **Override index:** `override_runtime_arguments` sets `compute_args[4]=ternary scalar`, but
  `is_reduce_bottom` is now compute-arg index 4 → for FUSE_TERNARY the scalar override index must move to
  5. K-par never fuses so it's harmless today; fix when touching ternary.
- **Auto-slicing interferes:** `num_slices` auto-derives (e.g. 4) for skinny shapes; for K-par sweeps set
  `TT_MM_NUM_SLICES=1` explicitly so `Pk*S <= grid.y`.
- **Always verify PCC** — the auto path skips the host `N_block%subblock_w` validator (separate
  silent-corruption class; see the auto-block-sizer work).

## Device
Currently WH B0 8x8 (`wh_env.sh`). The other machine is BH p150b 11x10 (`bh_env.sh`), grid.y=10 not
pow2 → see the limitation note above.
