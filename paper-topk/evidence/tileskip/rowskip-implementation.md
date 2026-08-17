# Row-Parallel Chunk-Skip Early-Out — Implementation Report

**Date:** 2026-08-16 · **Silicon:** Blackhole p150a (130-core grid) · **Branch:** nkapre/sorting (working tree, not committed)
**Op:** `ttnn.experimental.topk_large_indices`, ROW-PARALLEL path only (`compute.cpp` / `compute_with_values.cpp`). Column-parallel tree kernels untouched (forecast: 0% skip there — their per-slice streams are 1–5 chunks).

## 1. Design

Per chunk `c` of a row's stream, right after the (unavoidable) `topk_xl_copy_tile` of the chunk into DST and before paying the fused sort + index split + merge + rebuild (~2 merge units ≈ 2.2 µs at llk_k=512, ≈ 5.5 µs at llk_k=2048 per this box's baselines):

```
skip  iff  max(chunk) < T          (strict <)
T = running USER_K-th largest survivor
  = DST word of rank USER_K-1 in the resident sorted-descending window
```

On skip, the chunk is popped (CB flow unchanged) and only the MATH-side chunk-base bookkeeping (`topk_xl_separate_indices_row_major_advance_chunk_base`) runs.

**Test eligibility (compile-time):** `chunk >= max(2, USER_K/4)`.
- Floor 2: the threshold address assumes the post-rebuild window layout, which only exists after the first merge+rebuild (chunk 1 tests against a post-local-sort-only window).
- `USER_K/4`: for iid data P(skip at position c) = C(cK, U)/C((c+1)K, U) ≈ e^(−U/(c+1)); below c = U/4 that is < e⁻⁴ ≈ 1.8% and the test is pure overhead. Measured ungated overhead: +6.8% at k512@128-chunks, +1.8% at k1536@25-chunks (§4) — the gate eliminates both while forfeiting < 1 expected skip per row.

### user_k plumbing
`attrs.k` (already in the program hash) is now passed as a compile-time arg to the two row-parallel compute kernels (`topk_large_indices_program_factory.cpp`, CT arg 3 indices-only / 4 with-values). Host change → one `./build_metal.sh` (done). The skip feature itself is a kernel-level `constexpr bool kChunkSkipEnable` in each kernel — the A/B toggle is a one-line JIT-only kernel edit, no rebuild.

## 2. Soundness proof (exact top-k value multiset preserved)

Invariant: at test time T equals the USER_K-th largest of ALL row elements seen so far (processed or skipped). Induction: (i) the window holds the top-llk_K of all *merged* elements and USER_K ≤ llk_K, so window rank USER_K−1 is the USER_K-th largest merged element; (ii) every skipped element was < T at its skip time ≤ current T (T is monotone: merges only improve the window), so skipped elements never belong to the top-USER_K of the seen set, and "merged" and "seen" agree at rank USER_K−1.

Let v_k be the final USER_K-th largest value of the row. T ≤ v_k at all times (at most USER_K−1 elements ever exceed v_k). Any element x of an exact top-k set has x ≥ v_k ≥ T, so its chunk has chunk_max ≥ x ≥ T and the STRICT test `chunk_max < T` never fires on it: every top-k candidate enters the window and, being within the top-USER_K ≤ top-llk_K, survives every subsequent merge to the output. Conversely every skipped element is < T ≤ v_k and belongs to no exact top-k set. Boundary ties (chunk_max == T) are never skipped. Skip decisions are pure functions of the input, so output is deterministic per input; tie membership can differ from the unskipped run only through merge-entrant order among equal values (allowed: stable=false).

## 3. Decision machinery (all components silicon-validated)

- **max(chunk) on the SFPU (MATH):** 32 `SFPLOAD`(INT32, auto-advance-2 walk) per 32-bit DST tile — the cgtceq-validated full-tile walk — lane-max via `SFPSWAP` mod1=VEC_MIN_MAX. **SFPSWAP(0, VC, VD, mod1=1) puts max into VC, min into VD** (SFPSWAP.md functional model; the in-tree comment in `ckernel_sfpu_topk_xl.h:809` reads backwards — the first bring-up run accumulated min ⇒ −inf, caught by the decision tracer). Cross-lane fold: `SFPTRANSP` + 3 SWAPs against −inf partners + 7×(`SFPSHFT2`-ROR1 + `SFPNOP` + SWAP), rotations pulled from the RUNNING max (SFPSWAP clobbers both operands, so cgtceq's rotate-the-rotated sum idiom does not transfer to max). Invoked through `_llk_math_eltwise_unary_sfpu_params_(fn, slot1, RC_custom)` — same bracketing as the op's own `mark_neginf_indices`, so DST base/counter reset and FPU→SFPU ordering are inherited.
- **Store + readback:** the folded max is SFPSTOREd (raw bits) at the walk counter's landing point = row 0 of the chunk's own indices region (dead at test time). MATH RISC: `tensix_sync()` + 2 MMIO reads of Dst @0xFFBD8000 (max word + threshold word) — the 81-cyc S0/R0 rendezvous from `cgtceq_perf.cpp`/CGTCEQ_RUNBOOK.md; the same sync orders the preceding rebuild's stores, so the threshold read needs no extra sync. Window configured once per kernel: `configure_dest_access<MathThreadId>(Float32, swizzle=true)` (the in-tree BH dprint float32 recipe → raw fp32 words at row*16+col).
- **Threshold address — empirically calibrated:** `CHUNK_SKIP_DIAG` dumped the full post-rebuild values region per K with a distinct-monotone bf16 input; every word matched torch rank order. Exact for all ranks, all three windows: **word(r) = (r % (K/16))·16 + r/(K/16)** — the descending window is column-major over the populated physical rows (32/64/128 rows for K=512/1024/2048). Calibration data: `diag/dprint_k{512,1024,2048}.txt`. (My a-priori writer-chain inversion was wrong; the calibration replaced it.)
- **Compare:** on the MATH RISC in sign-magnitude order (monotone bit transform; == IEEE float order; values are bit-exact bf16<<16 payloads end-to-end and the bf16 datapath admits no NaNs).
- **Cross-TRISC propagation:** MATH→UNPACK via the T1→T0 hardware mailbox (`mailbox_write`/blocking `mailbox_read`). UNPACK must branch in tandem — its two `llk_unpack_set_srcb_dummy_valid()` calls (local_sort + rebuild) would otherwise leave SrcA/SrcB banks valid with no consumer and wedge the next real unpack. Audit per mailbox-sync-audit: the op's copy path ALREADY uses this FIFO (unpack-to-dest dst_index per tile); safety holds because the FIFO is order-preserving single-writer/single-reader and both threads issue identical per-chunk sequences under identical compile-time predicates ([dst_index × tiles] then [skip, tested chunks only]); worst-case occupancy 3 ≤ depth 4; overflow would only stall the writer. PACK has no per-chunk leaf work and takes constant false.
- **Decision cost (measured, from the ungated A/B):** ≈ 150 ns/tested chunk at llk_k=512, ≈ 350 ns at llk_k=2048 (126 tested chunks cost +18.9 µs at k512@65536; 5×23 tested cost +24.5 µs at k1536@51200) — versus ≈ 2.0/5.4 µs saved per skipped chunk.

## 4. Measurements (Tracy DEVICE KERNEL DURATION, 3 trials × 5 measured iters, median; spread < 0.1% everywhere; 130-core programs, active cores = min(rows,130); stimulus torch.randn bf16 seed 0)

| cell | baseline | skip ungated | **skip gated (final)** | Δ final |
|---|---|---|---|---|
| rows=2, N=65536, k=32 | 279.14 µs | 153.86 µs (−44.9%) | **153.19 µs** | **−45.1% (1.82×)** |
| rows=2, N=65536, k=512 | 279.18 µs | 298.09 µs (+6.8%) | **280.61 µs** | **+0.51%** |
| rows=8, N=65536, k=32 | 279.14 µs | 178.72 µs (−36.0%) | **177.94 µs** | **−36.3% (1.57×)** |
| rows=640, N=51200, k=1536 (prefill) | 1377.39 µs | 1401.95 µs (+1.8%) | **1377.98 µs** | **+0.04%** |
| rows=2, N=102400, k=1536, valid=56320 (bounded_cache) | 311.72 µs | 317.46 µs (+1.8%) | **312.05 µs** | **+0.11%** |

Notes:
- r8_k32 < r2_k32 win because the op time is the slowest of 8 iid rows (order statistics of per-row skip counts).
- k512@65536 has only 128 chunks; P(skip) ≈ e^(−512/c) ≈ 1.8% even at c=128 — no realistic win exists on that cell, so the gate refuses to pay for it (gate = 128 ⇒ zero tests). The residual +0.51% (+1.4 µs) is the gate-off code path itself (per-chunk compile-time-constant compare + loop code growth on the tiny in-order RISCs); the prefill/bounded_cache cells show the same effect at +0.04%/+0.11%.
- The k32 win is the one that matters for shipping traffic: the ttnn.topk k≤64 routing (commit 15f5659) sends exactly these small-k multi-row shapes down this path.

## 5. Correctness / hang evidence

- Official suite `test_topk_large_indices.py`: **154 passed, 2 deselected** (the IOMMU-gated production_perf_check cells, env-blocked on this box) — run both ungated and gated (suite_run1.log / suite_run2.log).
- Adversarial battery (`_topk_large_indices_skip_adversarial.py`, 36 checks, bit-exact bf16-bits multiset vs torch, distinct-index verification, program-cache on, 2 iters each): **all PASS** in both ungated (run2) and gated (run3) configurations. Cases: iid randn (k32/k512/k1024/k1536-llk2048), all-equal, ascending (0% skip), descending (max skip), top-k entirely in the LAST chunk, top-k split first/last chunk, boundary ties duplicated across 64 chunks (k32/k64), valid_length prefix, return_values variant.
- Decision tracer (`_topk_large_indices_skip_debug.py` + `CHUNK_SKIP_DEBUG`): per-chunk max/threshold/decision match host-computed exact expectations on ascending (no skip) and descending (all tested chunks skip) 4-chunk streams; outputs exact in both.
- Hang battery (`_topk_large_indices_skip_hangbattery.py`): 20 launches alternating max-skip/zero-skip inputs across 5 shapes, program cache on — clean, all results exact.
- Column-parallel guard: canonical sweep k512@65536 op layer (correctness-gated) — **13.153 µs MEASURED**, vs the ~13.1 µs pre-change pin: unchanged. Column-parallel sources (`compute_tree*.cpp`, `writer_tree*`, `reader_local.cpp`, `topk_large_indices_compute_common.hpp`) byte-identical (git-clean).
- Hang battery repeated after gating: 20/20 clean, results exact (hangbattery_gated.log).

## 6. Files touched

- `ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/topk_large_indices_chunk_skip.hpp` (new — all machinery + proof)
- `.../kernels/compute.cpp`, `.../kernels/compute_with_values.cpp` (copy/finish split + gated skip + diag hooks; column-parallel kernels and the shared common header byte-identical)
- `.../device/topk_large_indices_program_factory.cpp` (user_k CT arg; rebuilt once)
- `tests/.../_topk_large_indices_bench.py` (+TOPK_VALID knob), new `_topk_large_indices_skip_{adversarial,debug,diag,hangbattery}.py`

## 7. Verdict

**Keep.** The realistic small-k cells — the shapes the k≤64 ttnn.topk routing actually ships to this path — gain 36–45% device-kernel time on iid data (well past the 5% bar), with exact-top-k correctness proven and adversarially tested. Cells where the skip cannot pay (user_k=512 @ 128 chunks, user_k=1536 prefill/bounded_cache) are protected by the compile-time USER_K/4 amortization gate and sit at +0.04% to +0.51% (the residual is loop-code growth, not the test). Column-parallel path bit-untouched and re-measured unchanged.

Known limits / follow-ups:
- The win is data-dependent: adversarial ascending inputs get 0% skip and pay only the gated test cost (k32: tests from chunk 8; measured exact, no hang).
- k512-class user_k only wins on much longer streams (N ≳ 512·llk_k/4); the gate autoenables there with no code change.
- If the +0.51% k512 residual ever matters, a `kChunkSkipEnable=false` one-line kernel edit restores the exact pre-skip binary (JIT-only, no rebuild).
