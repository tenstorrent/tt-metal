# indexer_score — blocked bcast-col gate-mul (compute ceiling 67.7% → 75.8%)

How the per-head gate reduction `acc[r,c] = sum_h relu(q[h,r]·k[c]ᵀ) · w[h,r]` was rewritten from a
per-tile broadcast multiply into a **blocked bcast-col MUL primitive**, lifting the heads8 bfp8
QC2/KC16 compute ceiling from **67.7% → 75.8%** (0.353 → 0.318 ms, DMA off) with no accuracy loss.
Companion to `INDEXER_COMPUTE_CEILING.md` (what the ceiling is made of) and `INDEXER_QCKC_CEILING.md`.

## Result

Compute ceiling (math-util, `INDEXER_DMA_OFF=1`, sp7, 110 cores), before → after:

| config        | before | after  |
|---------------|--------|--------|
| heads8  bfp8 (production) | 67.7% (0.353 ms) | **75.8% (0.318 ms)** |
| heads16 bfp8  | 69.6%  | **77.7%** (0.621 ms) |
| heads64 bfp8  | 72.1%  | **76.6%** (2.522 ms) |

Full kernel (DMA **on**) heads8 QC2/KC16: **0.416 → 0.384 ms** (58.1% → 62.9% util).
Correctness: **25/25** indexer accuracy tests pass (production heads16/64, bfp8_k, knobs,
corner_shapes, multicore_qc2, glx_chunked), rank0 and rank7.

## The diagnosis — the gate-mul was unpack-context-bound, not bandwidth/fidelity/FPU-bound

Device-zone profiling (`DeviceZoneScopedN`, per-RISC, DMA off, QC2/KC16) of the full-unit path:

- The q·kᵀ **matmul runs ~93% efficient in its own window** (16,384 FLOP-floor cyc vs ~17,550-cyc
  measured per row) — no bubbles to mine.
- The **gate-mul phase was the reclaimable region (~21%)** and **unpack-bound**: the standard
  `mul_tiles_bcast_cols` issues one `llk_unpack_AB` context (a `wait_for_next_context` + a
  math↔unpack semaphore handshake + a MOP) **per (column, head) = 128 contexts/row**, and it
  re-unpacks the gate `w[h]` once per column even though `w[h]` is **column-independent**. The FPU
  sat idle ~60% of that phase waiting on the unpacker.

Two "obvious" levers were measured and proven **neutral**, which is what isolated the real cause:

- **bfp8 cb_qk** (halves the relu(qk) round-trip *bandwidth*) → 68.4% ≈ baseline. Not bandwidth-bound.
- **LoFi gate-mul** (kills HiFi2's fidelity replay = fewer math passes) → 68.5% ≈ baseline. Not
  math-pass-bound.

So the limiter is unpack **issue/sync count**, not data volume or math throughput.

## The fix — one unpack context per head instead of per (column, head)

A **blocked bcast-col MUL** primitive (modelled on the SDPA experimental blocked-sub path): one
unpack context loads `w[h]` **once** into SrcB + `ct_dim` consecutive qk columns into SrcA, then
`ct_dim` LoFi `ELWMUL`s MAC head `h` onto `ct_dim` dest tiles. The caller does ONE `tile_regs_acquire`
for a `ct_dim`-column sub-batch and loops the heads — each head one blocked call into the **same**
dest tiles — so `dest[col]` accumulates `sum_h qk[col,h]·w[h]`, one pack per column.

Effect on the bottleneck (heads8, KC=16, ct_dim=8):

- unpack **contexts: 128/row → 16/row** (8 heads × 2 sub-batches); `w[h]` unpacked once/head, not /col.
- gate-mul phase per-thread cycles/row: **unpack 4700→2870, math 2000→2748, pack →2345** — the phase
  went from FPU-starved (~40% FPU-busy) to **FPU-bound (~96%)**, the three threads now co-balanced.
- phase wall **~5044 → ~2870 cyc/row** (~43% faster).

`ELWMUL` accumulates into dest **by default** (the `dest_accum_en` opcode field is 0 — matching the
standard ELWMUL MOP; the per-call "overwrite" only comes from the prior `tile_regs_acquire` ZEROACC).
Setting `dest_accum_en=1` gave overwrite-like behaviour (PCC 0.328 = only the last head survived).

The q·kᵀ matmul stays **HiFi2** (the precision `math_util` credits); only the gate-mul is LoFi
single-pass, validated by the PCC ≥ 0.999 accuracy tests.

## Two bring-up traps (cost ~1 h of bisecting)

1. **cb_qk must be head-major.** The blocked unpack streams `ct_dim` **consecutive** SrcA tiles, so a
   head's columns must be contiguous in cb_qk: layout `[head][col]` (slot `h*cols + c`), not the old
   `[col][head]`. The matmul writes columns one at a time, so its output is repacked head-major.
2. **Matmul output must be repacked with `llk_matmul_pack`, not `pack_tile`.** The generic
   `llk_pack` (`pack_tile`) **misreads the matmul DEST layout** — silent wrong results (PCC 0.328),
   no hang, no error. `llk_matmul_pack<…, out_of_order=true>(dst, cb, 1, slot)` lands each head's dst
   tile at its head-major slot. Bisecting tip: swap the proven standard `mul_tiles_bcast_cols` back in
   over the new head-major layout — if it still fails, the bug is the matmul repack, not the mul.

## Where the cycles go now (heads8 bfp8 QC2/KC16, DMA off, per work unit)

Per-region critical-path (max over unpack/math/pack threads), scaled to one unit (2 rows + 1 untilize):

| region | % of kernel | note |
|--------|-------------|------|
| matmul (q·kᵀ + relu-pack) | **~82%** | of which ~75% is the FLOP floor math_util credits; ~7% is matmul-phase pack/handshake overhead (now mildly pack-bound) |
| gate-mul (Σₕ relu·w)      | **~13%** | now FPU-bound (~96%), co-balanced threads |
| untilize (fast pack_untilize) | **~4%** | |
| reconfigs (matmul↔mul mode switch + half-sync resync) | **~1%** | mode-switch batching + shrunk resync did their job |

Further gains require **fewer matmul FLOPs (algorithmic)**, not more utilization — the non-matmul
overhead is nearly exhausted.

## Files

- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_binary_custom.h` —
  `_llk_math_mul_bcast_cols_reduce_custom_(ct_dim)` (copy of the SDPA `_llk_math_sub_bcast_cols_reuse_custom_`
  scaffold, `TTI_ELWSUB` → `TTI_ELWMUL`, `dest_accum_en=0`).
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_eltwise_binary_custom_api.h` —
  `llk_math_eltwise_binary_mul_bcast_cols_{init_,}custom`.
- `tt_metal/hw/inc/api/compute/experimental/indexer_mul_custom.h` (new) — compute-API wrappers
  `mul_{bcast_cols_init_short,tiles_bcast_cols}_custom`; reuses the op-agnostic blocked unpack
  `llk_unpack_AB_sub_bcast_col_custom`.
- `.../indexer_score/device/kernels/compute_indexer_score.cpp` — `matmul_relu_pass_headmajor`,
  `set_mul_mode_custom`, and the rewritten batched full-strip path.

## Reproduce

```bash
# compute ceiling
INDEXER_DMA_OFF=1 scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::"test_indexer_score_sp7_math_util[heads8_k_bfp8]"
# accuracy (custom path is exercised by any production / bfp8_k config: single-chunk + KC>=2)
scripts/run_safe_pytest.sh --run-all \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_bfp8_k
```

Per-region zones: wrap the matmul / mul / untilize / resync regions in `DeviceZoneScopedN`, run under
tracy (`-p -r`), parse `<dir>/.logs/profile_log_device.csv` (col4 = RISC TRISC_0/1/2 = unpack/math/pack,
col6 = cycles, col11 = zone, col12 = START/END). Take the max over the 3 threads per region for the
critical-path contribution. Keep zones OUT of per-tile/per-head loops (~250-marker/RISC cap).
