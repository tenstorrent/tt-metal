# indexer_score — compute ceiling analysis (heads8 bfp8)

What the matmul **math-utilization** ceiling is for `indexer_score`, why it sits
where it does, what was done to raise it, and what (realistically) is left.
Companion to `INDEXER_PROFILING.md` (how to measure) and `INDEXER_OP.md` (op design).

All numbers are **sp_rank 7, GLX shape, 110 cores, V=35010 valid tiles, HiFi2,
bf16 DEST half-sync**, measured with `INDEXER_DMA_OFF=1` (the compute ceiling —
reader/writer skip NoC but still push/pop CBs, so compute runs unstarved).

## Result

Compute ceiling (math-util), before → after this session's work:

| config        | before | after  |
|---------------|--------|--------|
| heads8  bfp8  | 52.6%  | **66.8%** |
| heads16 bfp8  | ~61%   | **69.6%** |
| heads64 bfp8  | —      | **72.1%** |

Full kernel (DMA **on**) also improved: heads8 bfp8 sp7 **0.827 → 0.723 ms**.

## UPDATE — blocked bcast-col gate-mul broke the gate-mul ceiling (75.8%)

The "practical ceiling ≈ 77%" above assumed the gate-mul's per-tile overhead was irreducible
under the standard API. It is not. Re-profiling QC2/KC16 (device zones IDX_MM/IDX_MUL) showed
the matmul runs **93% efficient in its window** but the gate-mul phase is **unpack-context-bound**:
the standard `mul_tiles_bcast_cols` issues one `llk_unpack_AB` context (wait_for_next_context +
semaphore handshake + MOP) **per (column, head)** = 128 contexts/row, and re-unpacks the gate
`w[h]` once per column even though it is column-independent. bfp8 cb_qk (bandwidth) and LoFi mul
(fidelity replay) were both **neutral** — confirming the limiter is unpack *issue/sync count*, not
bandwidth or math passes.

Fix: a **blocked bcast-col MUL** primitive (`tt-llk .../experimental/llk_math_eltwise_binary_custom.h`
`_llk_math_mul_bcast_cols_reduce_custom_`, mirrored on the SDPA blocked-sub scaffold; compute-API
wrapper `api/compute/experimental/indexer_mul_custom.h`). One unpack context loads `w[h]` **once** +
`ct_dim` qk columns and MAC-reduces head h onto `ct_dim` dest tiles — so contexts drop from per
(col,head) to **per head** (8/row, ct_dim=8) and `w[h]` is unpacked once per head. Requires cb_qk
**head-major** (head h's columns contiguous as the streamed SrcA); the matmul output is repacked
head-major via `llk_matmul_pack<…, out_of_order=true>` (the generic `llk_pack`/`pack_tile` misreads
the matmul DEST layout — that was a 0.328-PCC trap during bring-up). ELWMUL accumulates in dest by
default (dest_accum_en=0), so heads MAC into the same dest with one pack per column.

Compute ceiling (math-util, DMA off, sp7), after the blocked-mul:

| config        | prev (this doc) | blocked-mul |
|---------------|-----------------|-------------|
| heads8  bfp8  | 67.0% (QC2/KC16 67.7%) | **75.8%** |
| heads16 bfp8  | 69.6%           | **77.7%** |
| heads64 bfp8  | 72.1%           | **76.6%** |

heads8 QC2/KC16 full kernel (DMA **on**) **0.416 → 0.384 ms** (58.1% → 62.9% util). Correctness:
25/25 indexer accuracy tests pass (production heads16/64, bfp8_k, knobs, corner_shapes,
multicore_qc2, glx_chunked), rank0 and rank7. The custom op is LoFi single-pass per face; the q.kᵀ
matmul stays HiFi2 (the precision math_util credits). Repro: `INDEXER_DMA_OFF=1 ...
test_indexer_score_sp7_math_util[heads8_k_bfp8]`.

## What landed (5 commits, all numerics-preserving — HiFi2 + bfp8 k + bf16 q/w/out)

1. **`INDEXER_DMA_OFF` ceiling toggle** — env→compile-time flag; reader/writer skip
   NoC reads/writes, still push/pop CBs. Off by default, zero runtime cost. Lets
   `sp7_math_util` report the pure compute ceiling. See `INDEXER_PROFILING.md`.
2. **MAC head reduction into DEST** — `acc_to_dest=1` on the bcast-mul math init
   (`llk_math_eltwise_binary_init<ELWMUL,COL,MATH_FIDELITY>(qk,w,1)`, the
   `deepseek_moe_fast_reduce_nc_fused` idiom): all heads of a chunk accumulate into
   one DEST tile in a single `tile_regs_acquire`, packed **once**; L1-acc only
   across chunks. Only **+0.5%** — the 8-deep L1-acc pack RMW chain was *not* the
   bottleneck (pack overlaps math in half-sync).
3. **Batch k-columns per matmul↔mul mode switch** — *the win, +13.6 pts.* The
   `set_matmul_mode`+`set_mul_mode` switch (unpack/pack reconfig stalls + MOP
   reprograms) was ~345 cyc/output-tile (~18% of compute, non-overlapping) and
   fired **per tile**. The gate `w` is column-independent and the unit's whole k
   chunk is resident, so `produce_full_strip` now runs all of a row's matmuls into
   a bigger `cb_qk` (`qk_col_batch * qk_batch_heads` tiles, 128-tile L1 cap), then
   all the MAC reductions → **one mode switch per row** (heads8 KC=16 → 1/row).
   `qk_col_batch = min(KC, 128/qk_batch_heads)`, single-resident-chunk + fast-strip
   only; otherwise falls back to per-column.
4. **Block-pack qk + guarded reconfig** — `pack_tile_block` instead of an 8× loop;
   4-arg `reconfig_data_format` so the redundant bf16 srcB reconfig (q↔w) is
   skipped by the format guard. +0.1% (noise), but cleaner/correct.

Accuracy after all of the above: **33/33** correctness tests pass (production
heads16-TP-shard + heads64, bfp8_k, knobs, corner_shapes, multicore_qc2,
glx_chunked, invalid/reject), rank0 and rank7.

## Where the cycles go (heads8 bfp8, DMA off, per work unit ≈ 24,524 cyc actual)

| component                                   | cyc/unit | % of total | notes |
|---------------------------------------------|----------|------------|-------|
| matmul q·kᵀ (useful FLOP floor)             | 16,384   | **66.8%**  | the work `math_util` counts |
| gate eltwise-mul (HiFi2 bcast `qk·w`)       | 3,585    | 14.6%      | 28 cyc/tile × 128; necessary, not matmul |
| matmul-phase setup + relu-packs + sched gaps| ~1,918   | ~7.8%      | `set_matmul_mode` (1/row) + qk block-pack |
| untilize (fast W=16 strip)                  | 830      | 3.4%       | already fast-strip optimized |
| mul-phase setup + 16 acc packs              | ~1,155   | ~4.7%      | `set_mul_mode` + one pack/column |
| `mm_block_init` resync after each strip     | 326      | 1.3%       | the fast-untilize half-sync re-init |

The **matmul is ~93% efficient in its own window** (16,384 FLOP-floor cyc vs a
17,550-cyc measured matmul zone) — there are no bubbles to mine inside it.

## The eltwise gate-mul: overhead-bound, not throughput-bound

Isolated measurement (zone around just the 8-head `mul_tiles_bcast_cols` loop):
**28 cyc per 32×32 bcast-col mul tile** at HiFi2.

- Arithmetic floor (1024 multiplies ÷ ~1024-wide FPU × 2 HiFi2 passes) ≈ **2 cyc/tile**.
- Measured **28 cyc/tile** → **~14× overhead.** The ~26 cyc/tile is per-op issue
  overhead: unpacking the qk tile + broadcast-w column, the ELWMUL MOP issue,
  per-face-row `CLR_B`/`SETRWC`, and the fidelity-replay setup — **not** FPU
  throughput (a pessimistic 1 face-row/cyc model predicts 32–64 cyc/tile, *more*
  than measured, so the array is not the limiter).

Implication: the gate-mul's cost is mostly fixed per-tile overhead. The standard
eltwise-binary API issues **one MOP per tile** (unlike `matmul_block`, which does
`rt_dim` tiles per issue), so there's no API knob to amortize it at HiFi2.

## Is 66.8% the maximum? — realistically yes, under "same fidelity + same formats"

This op is fundamentally **matmul + a per-head gated reduction**. The reduction is
real, necessary, non-matmul work that `math_util` doesn't credit, so the metric is
inherently capped well below 100% by the *algorithm*, not by code quality.

- **Practical ceiling ≈ 77%** — if the gate-mul were overhead-free (≈256 cyc/unit)
  but untilize/reinit/setup stayed: `16384 / (16384+256+830+326+~1900) ≈ 77%`.
- **Absolute ceiling ≈ 92%** — gate-mul free *and* untilize/reinit gone.
- We are at **66.8%**, ~10 pts below the practical ceiling, and **all ~10 pts live
  in the gate-mul per-op overhead.**

To recover that without changing numerics you'd need a **blocked HiFi2 eltwise-bcast
LLK** (one MOP across many head-tiles). It does **not** exist in the standard API;
the experimental `llk_math_eltwise_binary_custom` does the single-MOP-over-many-tiles
trick but only at **LoFi** (no fidelity replay). So past 66.8% requires either:

1. **Build/validate a HiFi2 blocked bcast-mul** — fidelity-safe, but speculative
   (days of LLK work, uncertain payoff since per-tile operand unpack is unavoidable).
2. **Drop the gate-mul to LoFi** — biggest raw win (kills the fidelity replay *and*
   enables the custom single-MOP path), but it is a math-fidelity change.

## The real wall-clock lever from here is NOT compute

heads8 full kernel (DMA on) = **0.723 ms** vs **0.361 ms** compute ceiling → it sits
~2× above compute, i.e. **DMA-bound**. Further heads8 wall-clock comes from the
reader/writer NoC path (q/k/w reads, scattered row-major output writes), a different
axis than the compute ceiling. heads16 is closer to compute-bound (0.797 ms full vs
0.694 ms ceiling, 60.6% util).

## Dead ends tried this session (fidelity-safe, did not pan out)

- **Prescale q by |w| on the reader** to fold the gate magnitude into the matmul:
  `relu(qk)·w = sign(w)·relu(|w|·q·kᵀ)`, and |w| is per-(head,row) reused across all
  KC columns. But the **per-row sign** survives as a same-cost per-head bcast-mul, so
  no net reduction in the gate-mul op count. No win.
- **Two-accumulator pos/neg subtract** (accumulate +heads and −heads separately, one
  final `sub_tiles`): would let the matmul DEST-accumulate absorb the reduction, but
  `sign(w)` varies **per row within a tile**, not per tile, so heads can't be cleanly
  partitioned. Dead end.
- **MAC-into-DEST alone** (commit 2): correct and cleaner, but only +0.5% because the
  pack/L1-acc chain overlaps matmul math in half-sync — pack was never the bottleneck.

## How to reproduce these numbers

Ceiling: `INDEXER_DMA_OFF=1 scripts/run_safe_pytest.sh ...::"test_indexer_score_sp7_math_util[heads8_k_bfp8]"`.
Per-zone breakdown: `DeviceZoneScopedN` around the regions + Tracy, parse
`profile_log_device.csv` (see `INDEXER_PROFILING.md`). The matmul-vs-mul-vs-untilize
split above came from zones named `IDX_MM`/`IDX_MUL`/`IDX_UNT`/`IDX_REINIT`, and the
28 cyc/tile mul from an `IDX_MULONLY` zone around just the bcast-mul loop.
