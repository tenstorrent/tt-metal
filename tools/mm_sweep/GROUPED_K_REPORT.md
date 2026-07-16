# Grouped-K compute for Regime-A matmul — report

> **STATUS: REJECTED and REMOVED from the code (2026-07-16).** Grouped-K produced no stable win over `Kg=1`
> in any form (whole-group prewait OR streamed) at fixed configuration, so its implementation was deleted to
> keep the production-adjacent kernels simple. This report (the causal explanation) and the raw benchmark
> JSON (`regime_a_kgroup_bench.json.gz`) are retained as the experimental record; `Kg=1` — the per-block
> progressive schedule — is and remains the production path. **Recover the removed implementation from:**
> `56b37f5d5e6` + `7b3f93ddaa5` (kernels/plan/factory/config) and `a5b7986b18f` + `3593ecd4083`
> (diagnostic tests + `kgroup` benchmark mode + data). Progressive in0 waiting and `DIAG_FULL_IN0_WAIT` were
> kept. **Root cause (why it can't win):** the FP32 pack/L1-accumulation is not on the critical path — the
> packer runs concurrently with the math (TRISC) and data-movement RISCs — so reducing the pack count per
> output tile cannot lower wall time; on narrow-N the compute RISC is the limiter and on wide-N the shape is
> DRAM-read-bound on in1. Prewait additionally reintroduced startup latency; streamed removed that but only
> reached parity with `Kg=1`. Full evidence below (§6 per-RISC, §9 streamed vs prewait).


Change (diagnostic-only): a compute grouping factor `Kg` (in delivered K blocks) that holds DST across `Kg`
consecutive delivered blocks and packs the FP32 output partial to CB3 **once per group** instead of once per
block. Delivery, `kb`, and the ring layout are unchanged; `Kg` is a compute-only grouping. Implemented as
compile-gated modes `DIAG_KGROUP2/4/8` (masks `1<<11/12/13`, program-cache-hashed, absent from the public
API). **Verdict: no grouped mode improves any shape; `Kg=1` (the current per-block progressive schedule)
stays the default. Grouped-K is left diagnostic-only.**

> **UPDATE — streamed grouped-K (§9).** The schedule below was first implemented as a whole-group
> **prewait** (wait all `Kg` blocks, then compute), which reintroduced per-group startup latency. It was
> corrected to **streamed** waits — now the default for `KGROUP`: acquire DST once per output subblock, then
> per-block progressive cumulative waits INSIDE the first subblock so math begins after block 0
> (`DIAG_KGROUP_PREWAIT`=`1<<14` retains the old prewait for A/B). Streamed beats prewait on every shape (it
> recovers the startup overlap) but **still does not beat `Kg=1`**: best −1% (within relaunch noise), `Kg=2`
> ≈ `Kg=1`, larger `Kg` slightly worse on narrow-N. Decision unchanged — `Kg=1` stays default. Details in §9;
> §1–§7 describe the original prewait experiment.

## 1. Old repeated-pack behavior
`matmul_blocks` packs the FP32 partial into CB3 once per delivered K block, accumulating in L1
(`llk_pack_reconfig_l1_acc(1)` after block 0). With `K_num_blocks=8` that is 8 FP32 pack/L1-accumulate
passes per output tile.

## 2. New grouped loop structure
`matmul_group` (compute.cpp): per output subblock, one `tile_regs_acquire()`, then matmul **all `Kg`
delivered blocks** of the group into the same DST (no acquire/pack between blocks), one `pack_tile` at
`tile_regs_commit`. Layouts respected: resident in0 block-major (`(group_start+g)*in0_block_num_tiles`), in1
group at the CB1 front (`g*in1_block_num_tiles`), existing `K_block_tiles` strides within each block. The
math K traversal order (block ascending, inner-K ascending) is unchanged. Schedule: group-level cumulative
in0 wait (`group_end*in0_block_num_tiles`, first N-sub-block only; CB0 stays resident), group in1
wait+pop (`group_size*in1_block_num_tiles`), `l1_acc(1)` transition after group 0. Partial final group
handled. DST usage is unchanged (one subblock region); only the MAC count before the pack grows.

## 3. Pack-count and CB1-depth changes (K_num_blocks=8)
| Kg | FP32 packs/tile | #groups | CB1 depth (blocks) |
|---:|---:|---:|---:|
| 1 | 8 | 8 | 4 |
| 2 | 4 | 4 | 4 |
| 4 | 2 | 2 | 8 |
| 8 | 1 | 1 | 8–16 |

CB1 = `max(4, min(2*Kg, N_bpc*K_num_blocks))` blocks — ≥2 groups so the in1 reader overlaps compute, capped
at the total in1 blocks streamed (so a `Kg==K_num_blocks` group can still double-buffer across N-sub-blocks
when `N_bpc>1`). `Kg=1/2` keep the current 4-block allocation exactly. The planner's L1 check accounts for
the enlarged CB1 and cleanly rejects infeasible `(config, Kg)` (e.g. `Pk=1` large-K + `Kg=8`).

**CB1-sizing note (important):** a first pass sized CB1 at `min(2*Kg, K_num_blocks)` = one group for `Kg=8`
(`Knb=8`), which starved the reader at N-sub-block boundaries — `Kg=8` looked like +12–20% on the wide-N
controls. Sizing CB1 to hold two groups (up to `N_bpc*K_num_blocks`) removed that bubble: those same shapes
went to ~+0%. The numbers below are post-fix; the bubble was a sizing artifact, not grouped-compute cost.

## 4. Startup-versus-compute tradeoff
Grouping trades startup overlap for fewer FP32 materializations: a group waits for `Kg` in0/in1 blocks
before its first matmul, so on the first N-sub-block it forfeits the progressive per-block in0 overlap (the
recently-landed win) — at `Kg=K_num_blocks` this reverts to the full-slice startup wait. The hoped-for
compensation was fewer FP32 pack/L1-accumulate passes.

## 5. Per-shape results (median device-profiler kernel µs, 3 interleaved relaunches, Δ vs Kg=1)
Raw: `regime_a_kgroup_bench.json` (all relaunches, per-RISC, util%512, PCC, CB1 alloc). All PCC ≥ 0.999.
| shape | group | cfg (Ns,Pk,Sm,kb,nsb) | N_bpc | Kg1 | Kg2 | Kg4 | Kg8 |
|---|---|---|---|---|---|---|---|
| 256×2048×1024 | target | 1,4,2,2,2 | 2 | **28.6** | +0% | +2% | +7% |
| 256×2048×1024 (nsb4) | target | 1,4,2,2,4 | 1 | **28.0** | +2% | +5% | +16% |
| 256×6144×768 | target | 1,12,1,2,1 | 3 | **53.1** | +1% | +3% | +5% |
| 256×6144×2304 | control | 1,12,1,2,1 | 9 | **91.7** | +1% | +1% | +0% |
| 256×6144×4608 | control | 1,12,1,2,1 | 18 | 152.8 | +0% | **−1%** | +0% |
| 32×6144×4608 (Mt1) | control | 1,12,1,2,1 | 18 | **118.4** | +0% | +1% | +1% |
| 64×6144×4608 (Mt2) | control | 1,6,1,4,2 | 9 | **119.4** | +0% | +1% | +3% |
| 128×6144×4608 (Mt4) | control | 1,12,1,2,1 | 18 | **129.8** | +0% | +0% | +1% |

`Kg=1` is best or statistically tied on every shape. `Kg=2` (packs halved, **no** CB1 change → no
confound) is already neutral-to-slightly-negative — the clearest evidence the pack count is not the lever.
`Kg=8` regresses narrow-N / `N_bpc=1` (+7 to +16%) and is neutral on wide-N.

## 6. New limiting stage (per-RISC, median µs)
| shape | Kg | wall | BRISC | NCRISC | TRISC (compute) |
|---|---|---|---|---|---|
| 256×6144×768 | 1 | 53.1 | 44.2 | 43.9 | 44.5 |
| 256×6144×768 | 8 | 55.6 | 40.7 | 40.0 | **46.9** |
| 256×2048×1024 | 1 | 28.6 | 22.4 | 22.3 | 23.0 |
| 256×2048×1024 | 8 | 30.8 | 22.1 | 22.1 | **25.2** |
| 256×6144×4608 | 1 | 152.8 | 141.7 | 141.7 | 142.1 |
| 256×6144×4608 | 8 | 152.1 | 140.3 | 139.9 | 143.0 |

The saved pack work is real but **not on the critical path**: fewer packs shrink the data-movement RISC
spans slightly (768: BRISC/NCRISC 44→40) yet the **compute (TRISC) span GROWS** (768: 44.5→46.9;
2048×1024: 23.0→25.2). That growth is the group-level startup in0 wait re-serializing the pipeline (the
overlap progressive waiting had recovered) — i.e. the saved compute work is replaced by an input wait, plus
some. On wide-N (4608) every RISC is ~142µs (DRAM-read-bound on in1), so grouping is simply neutral. So the
FP32 pack/L1-accumulation was never the bottleneck; the packer runs concurrently with the math and DM RISCs.

Implementation was verified (not an accidental serialization): DST is held across the whole group (single
acquire/commit; PCC 0.99999), exactly one pack per output subblock per group, CB1 double-buffers (Kg=8 wide-N
neutral proves the reader isn't stalled), and the only group barrier is the intended `Kg`-block wait.

## 7. Production/default decision
No grouped mode delivers a stable end-to-end improvement on any shape, so per the selection gate **the
default remains `Kg=1`** (the current per-block progressive schedule); the public path is unchanged
(`k_group=1` gives the identical 4-block CB1 and no `KGROUP` define). No exhaustive `Pk×Ns×Sm×kb×nsb×Kg`
re-sweep was run — that step is gated on a grouped mode winning, and none did; a re-sweep cannot rescue a
lever that is neutral even in its no-confound case (`Kg=2`) and whose per-RISC data shows the compute RISC
(not the packer) is the limiter. Grouped-K is retained diagnostic-only (`DIAG_KGROUP2/4/8`) alongside the
other refuted variants. Best candidate for a future picker refresh: **none** — `Kg=1`.

## 8. Correctness
gtest `RegimeADiagFixture.GroupedKCorrectness`: random BF16 vs CPU f32 golden, PCC ≥ 0.999, fresh AND
cached-program, for `Kg=1/2/4/8` across both Mt=8 primaries, `Pk=1`/`Pk>1`, `Sm=1`/`Sm>1`, `N_bpc=1/2/3`,
`W=1`/`W>1`, and balanced K/N tails — all **PCC 0.99999** (infeasible `Kg` combos cleanly L1-skip). Public
20/20 suite unchanged (mask 0 = `Kg=1`). Grouped and baseline are not bit-identical (FP32 pack points move)
but K accumulation stays monotonic, as expected.

## 9. Streamed grouped-K correction (the current default for KGROUP)
§1–§8 measured a **whole-group prewait**: wait all `Kg` in0/in1 blocks, then acquire DST and compute — which
reintroduced the per-group startup latency the progressive in0 schedule had removed (at `Kg=K_num_blocks` it
degenerates to the full-slice startup wait). The corrected **streamed** discipline (now the default for a
`KGROUP` mode; `DIAG_KGROUP_PREWAIT`=`1<<14` selects the old prewait for A/B) instead acquires DST once per
output subblock and consumes the group's blocks with **per-block progressive cumulative waits inside the
first output subblock** (in0 `(group_start+g+1)` over the resident slice, first traversal only; in1 `(g+1)`
over the CB1 group front), so the first matmul begins after block 0. Later output subblocks find all inputs
resident and run with no waits; in1 is popped once after all subblocks; still one pack per output subblock
per group. `matmul_group_streamed` in compute.cpp.

### Results — Kg=1 vs streamed vs prewait (median µs, 3 interleaved relaunches, Δ vs Kg=1)
Raw: `regime_a_kgroup_bench.json` (all relaunches, per-RISC, util%512, PCC, CB1 alloc). All PCC ≥ 0.999.
| shape | grp | Kg1 | strm2 | strm4 | strm8 | pre2 | pre4 | pre8 |
|---|---|---|---|---|---|---|---|---|
| 256×2048×1024 (nsb2) | tgt | **28.6** | +0% | +1% | +4% | −1% | +3% | +9% |
| 256×2048×1024 (nsb4) | tgt | **28.0** | +1% | +5% | +12% | +1% | +5% | +15% |
| 256×6144×768 | tgt | **53.8** | +0% | −1% | +1% | +2% | +0% | +4% |
| 256×6144×2304 | ctl | **91.5** | +0% | +0% | +0% | +1% | +2% | +1% |
| 256×6144×4608 | ctl | 152.8 | +0% | −1% | −1% | +0% | +0% | +0% |
| 32×6144×4608 (Mt1) | ctl | **118.7** | +0% | +1% | +1% | +0% | +1% | +1% |
| 64×6144×4608 (Mt2) | ctl | **119.3** | +0% | +0% | +0% | +0% | +1% | +3% |
| 128×6144×4608 (Mt4) | ctl | **129.6** | +0% | +0% | +0% | +0% | +0% | +1% |

**Streamed ≤ prewait on every shape** (e.g. 768 Kg8: 54.3 vs 56.2; nsb2 Kg8: 29.7 vs 31.1) — the correction
works, the startup overlap is recovered. But **no streamed `Kg` beats `Kg=1`**: `strm2` ties `Kg=1`
everywhere; the only sub-zero cells (768/4608 `strm4/8` at −1%) are within the ~1% relaunch spread; larger
`Kg` on narrow-N is a small regression.

### Per-RISC (median µs) — streamed recovers the compute-start prewait lost, but doesn't beat Kg=1
| shape | variant | wall | BRISC | NCRISC | TRISC (compute) |
|---|---|---|---|---|---|
| 256×6144×768 | kg1 | 53.8 | 44.8 | 44.4 | 45.0 |
| 256×6144×768 | strm8 | 54.3 | 40.2 | 39.2 | **46.5** |
| 256×6144×768 | pre8 | 56.2 | 40.1 | 39.9 | **47.4** |
| 256×2048×1024 | kg1 | 28.6 | 22.4 | 22.6 | 23.1 |
| 256×2048×1024 | strm8 | 29.7 | 21.9 | 21.8 | **24.1** |
| 256×2048×1024 | pre8 | 31.1 | 22.4 | 22.1 | **25.2** |

Streaming drops the compute (TRISC) span vs prewait (768: 47.4→46.5; 2048×1024: 25.2→24.1) — the first
matmul starts earlier. But streamed's TRISC is still **above** `Kg=1` (45.0 / 23.1): grouping shifts fewer
FP32 packs off the data-movement RISCs (BRISC/NCRISC 44→40 on 768) yet the compute RISC — not the packer —
bounds the shape, and holding DST across a group with interleaved waits adds a little math-side stall.
`Kg=2` streamed is indistinguishable from `Kg=1`. Wide-N stays DRAM-read-bound (neutral).

### Decision (unchanged)
Streamed grouped-K is the correct, faster-than-prewait design, but it does not produce a stable end-to-end
win over `Kg=1` at fixed configuration, so per the selection gate the **production default stays `Kg=1`** and
**no exhaustive `Pk×Ns×Sm×kb×nsb×Kg` re-sweep** was run (gated on a fixed-config win; none). Streamed is the
default *within* the diagnostic `KGROUP` modes; prewait is retained as `DIAG_KGROUP_PREWAIT` for A/B. Root
cause is unchanged from §6: the FP32 pack/L1-accumulation is not on the critical path (the packer runs
concurrently with the math), so reducing pack count cannot lower wall time; correcting the startup latency
(streamed) removes the prewait regression but leaves grouped-K at parity, not ahead.
