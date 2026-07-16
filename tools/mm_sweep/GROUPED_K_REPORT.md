# Grouped-K compute for Regime-A matmul — report

Change (diagnostic-only): a compute grouping factor `Kg` (in delivered K blocks) that holds DST across `Kg`
consecutive delivered blocks and packs the FP32 output partial to CB3 **once per group** instead of once per
block. Delivery, `kb`, and the ring layout are unchanged; `Kg` is a compute-only grouping. Implemented as
compile-gated modes `DIAG_KGROUP2/4/8` (masks `1<<11/12/13`, program-cache-hashed, absent from the public
API). **Verdict: no grouped mode improves any shape; `Kg=1` (the current per-block progressive schedule)
stays the default. Grouped-K is left diagnostic-only.**

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
