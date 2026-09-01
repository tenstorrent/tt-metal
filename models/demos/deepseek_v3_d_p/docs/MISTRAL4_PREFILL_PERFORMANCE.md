# Mistral Small 4 119B — prefill performance log

Running record of measured numbers, so each new thing we try can be compared against the last.
**Append, don't overwrite.** Every results block names the machine and the **commit it was measured
at** — that is not the commit that adds the block, so state it explicitly.

Model: `mistralai/Mistral-Small-4-119B-2603`, 36 layers, MLA (`kv_lora_rank=256`,
`qk_rope_head_dim=64` → 320-wide KVPE), MoE 128 routed / top-4 / 1 shared.

Machines: **12 kW** — the box prefill will actually run on, so this is the column that decides things.
**8 kW** (`bh-glx-110-a10u08`) — the dev box; useful as a second data point, not as a target.

---

## 0. READ FIRST — a measurement bug invalidated the earlier PP numbers

`PP_HANDOFF=none` allocated every downstream rank's input as `torch.zeros` and never refreshed it. A
zero hidden state gives every token identical gate logits, so top-k selects the **same experts for the
whole batch** and it all piles onto one chip; with EP > 1 the traffic also converges across the expert
axis, idling the other group. Every PP number measured before 2026-08-20 21:00 carries this artefact.

Reproduced exactly from an independent harness (`pp_contention.py`, `PP_ZERO_HIDDEN=1`):

| shape | stages | real routing | zero-fed | what the driver reported | inflation |
|---|---|---|---|---|---|
| (8,1) | 4 | 134.89 ms | 152.88 ms | **150.6 ms** | **1.13x** |
| (8,2) | 2 | 200.25 ms | 415.04 ms | **414.7 ms** | **2.07x** |

Both reproductions land within 1.5% of the driver, from a separate code path. **Fixed** in
`test_prefill_pipeline_concurrent.py` via `_seed_downstream_inputs`, which replays stage 0 once and
seeds each downstream rank with that real activation before timing.

**Where it bites:** short windows, where MoE dominates (+11.7% at 5,120, +9.3% at 25,600). It
*vanishes* at long context, where attention over accumulated KV dominates (~0% at 102,400 and
261,120). So the 100K/256K conclusions were never affected; the short-window ones were.

Also note the `handoff=none` framing was wrong in **both** directions at once: it omits transport cost
(optimistic) while inflating MoE cost (pessimistic). It is a ceiling for the *transport* only.

---

## 0b. Rebased onto latest main — a NEW baseline, not comparable to §1

Rebasing onto `origin/main` (2026-08-21, 102 commits) forced a fabric change: upstream retired
`FABRIC_1D` for the 8x4 prefill rows in favour of `fabric_profiles.torus_xy_device_params`
(`FABRIC_2D_TORUS_XY`, `RELAXED_INIT`). `FABRIC_1D` at this mesh now reports
"requested combination of fabric config and mesh, unfeasible on the given hardware" and **skips**, so
the torus profile is the only runnable option and everything below §0b was measured on a fabric that
no longer applies.

Measured on 8 kW at `37141629493` (`kmabee/mistral4-prefill-full-rebased`):

| window | single-rank | PP=4 x (8,1) | ratio | (FABRIC_1D ratio) |
|---|---|---|---|---|
| 5,120 | 29,595 | 35,110 | **1.19x** | 1.17x |
| 25,600 | 24,854 | 42,473 | **1.71x** | 1.67x |

Per-config change from the `FABRIC_1D` numbers in §1:

| window | single-rank | PP=4 x (8,1) |
|---|---|---|
| 5,120 | -9.2% | -7.7% |
| 25,600 | -8.3% | -6.1% |

**The absolute drop is uniform across configurations and the ratios are preserved (slightly better),
so this is a re-baseline, not a regression.** Correctness is unchanged: PP layer slicing still samples
**token 2 at p=0.7147**, bit-identical to every prior run, across 102 upstream commits and a different
fabric.

Two things this rebase exposed, both worth knowing:

- **A conflict resolution that passed every static check and failed on hardware.** Upstream's rewritten
  norm ends with `if not use_high_bw_all_gather: ttnn.deallocate(tt_gathered_stats)`; the merged TP=1
  branch never assigned that flag, so every TP=1 forward raised `UnboundLocalError`. AST parses, lint
  and the host-only tests were all green. Only the PP rows caught it. `False` is also the
  *semantically* right value there -- at TP=1 the tensor is freshly allocated per forward, unlike the
  high-bandwidth path's reused buffer, so step 3 should free it.
- **A skip that reports as a pass.** The pre-migration PP rows returned rc=0 with
  "unfeasible on the given hardware". It was only caught because 22 s was implausible against a known
  ~3 min baseline. Any harness reading exit codes alone would have recorded those as successes.

The single-rank question raised by the 12 kW run (-6.6% to -14% vs `c34e372b47d`) is **still open** and
cannot be settled from this run: that A/B needs both sides on the same fabric, and `FABRIC_1D` is no
longer runnable on main. Ruled out by inspection so far: upstream drift (both commits share base
`17407015dab`), the matmul-K guard (present in both), the removed `kt` tag (never populated; the
config table is byte-identical), and rope-table caching (single-shot branch only; every measured
single-rank row is chunked).

---

## 1. The decision table — 8 kW, corrected

All four parallelisations, fixed driver, `PP_HANDOFF=none`, traced, 2026-08-20/21.
**Bold = best in row.** Rows marked \* are still pre-fix.

| measurement | single-rank (8,4) | PP=4 x (8,1) | PP=4 x (4,2) | PP=2 x (8,2) |
|---|---|---|---|---|
| 5,120 single-shot | 32,611 | 38,027 | **40,784** | 27,061 |
| 25,600 single-shot | 27,090 | **45,238** | 24,729 | 13,370\* |
| 102,400 chunked, steady | 19,810 | **23,132** | 13,862 | 11,368\* |
| 102,400 chunked, total | 19,810 | **20,115** | 12,324 | 11,098\* |
| 261,120 chunked, steady | 10,888 | **12,408** | 6,700 | 9,348\* |
| 261,120 chunked, total | 10,888 | **12,192** | 6,457 | 9,227\* |

Ratios vs single-rank (steady where applicable):

| window | (8,1) | (4,2) | (8,2) |
|---|---|---|---|
| 5,120 | 1.17x | **1.25x** | 0.83x |
| 25,600 | **1.67x** | 0.91x | — |
| 102,400 | **1.17x** | 0.70x | — |
| 261,120 | **1.14x** | 0.62x | — |

### Conclusion: `(8,1)` is the candidate

**PP=4 x (8,1) wins every window except one short single-shot case**, including both
production-shaped chunked long contexts. It is the configuration to take forward.

**`(4,2)` is a short-prompt special case, not a general win.** It beats everything at 5,120
single-shot (1.25x single-rank) and then collapses — 0.55x, 0.60x, 0.54x of `(8,1)` at the other three.
The mechanism: at 5,120 its 1,280 tokens/chip fill the core grid better than `(8,1)`'s 640, which is
worth ~7%; but SP=4 gives each chip **half the sequence** SP=8 does, so once KV accumulates across
chunks every chip carries 2x the KV through ring attention, and that dominates everything else. Its
win exists only with an empty KV.

**`(8,2)` is closed** — see the depth argument below.

**Still true of all of it:** `PP_HANDOFF=none`, so no activation crosses a stage boundary and a real
device-to-device transport cost has to come off these margins.

## 2. 12 kW — measured with the fixed driver

Measured 2026-08-21 at `2bc6bc7210b` (`FABRIC_1D`, pre-rebase), 14/14 runs passed, no reset needed,
no other users. Correctness: PP=4 and PP=2 both sample **token 2 at p=0.7147**, matching 8 kW exactly.

**Single-rank was re-measured in the same run**, which matters: the earlier 12 kW single-rank figures
came from a *different commit* (`c34e372b47d`), so quoting PP against them mixes two code states.

| window | single-rank (same commit) | single-rank (earlier, `c34e372b47d`) | PP=4 x (8,1) | PP=4 x (4,2) | PP=2 x (8,2) |
|---|---|---|---|---|---|
| 5,120 | 32,611 | 36,312 | 38,737 | **41,108** | 27,173 |
| 25,600 | 28,225 | 32,821 | **46,629** | 26,163 | — |
| 102,400 | 21,285 | 23,546 | **28,880** steady / 25,225 total | — | — |
| 261,120 | 14,026 | 15,022 | **18,098** steady / 17,107 total | 8,713 steady / 8,373 total | — |

Ratios against the **same-commit** single-rank:

| window | PP=4 x (8,1) | PP=4 x (4,2) | PP=2 x (8,2) |
|---|---|---|---|
| 5,120 | 1.19x | **1.26x** | 0.83x |
| 25,600 | **1.65x** | 0.93x | — |
| 102,400 | **1.36x** steady / 1.19x total | — | — |
| 261,120 | **1.29x** steady / 1.22x total | 0.62x | — |

**The predicted 5,120 flip happened.** Pre-fix it read 0.94x — a loss. It is now 1.19x against the
same-commit baseline, or 1.07x against the older cross-commit one. Either way it is a win, and
**PP=4 x (8,1) now wins all four windows on the target box**, up from one of four.

Two honest caveats on the magnitude:

- **Part of the margin is single-rank getting slower**, not PP getting faster: same-commit single-rank
  is 6.6-14% below the `c34e372b47d` figures at every window. No code path between those commits
  touches chunked single-rank (see §0b), so this is unexplained and most likely environmental or an
  extraction difference. Quote the conservative ratios if the number has to be defended.
- `(4,2)` reproduces the 8 kW crossover exactly: best at 5,120 (1.26x), then 0.93x and 0.62x as
  context accumulates. Short-prompt special case, not a general win.

**These are all `FABRIC_1D` numbers.** Upstream has since retired that fabric for 8x4 (§0b), so this
whole section will need re-baselining on `FABRIC_2D_TORUS_XY` before it describes the shipping path.

### Single-rank scales with power### Single-rank scales with power

| window | 8 kW | 12 kW | 12/8 |
|---|---|---|---|
| 5,120 | 32,611 | 36,312 | 1.11x |
| 25,600 | 27,090 | 32,821 | 1.21x |
| 102,400 | 19,810 | 23,546 | 1.19x |
| 261,120 | 10,888 | 15,022 | **1.38x** |

---

## 3. Experiment queue (TODO)

Ordered. **Tick items off in place and add the measured result inline**, so a later reader sees what
was tried and rejected rather than just what is open.

| # | experiment | why | status |
|---|---|---|---|
| 1 | **Re-run the 12 kW numbers with the fixed driver** | The target box's PP column is stale and the correction may flip the 5,120 loss into a win. Highest value item in this list. | **TODO — needs 12 kW** |
| 2 | **Re-measure `(4,2)`** | It was rejected on buggy numbers, and it is EP=2 — the configuration the bug punished 2.07x. Its rejection is not currently supported. | **IN PROGRESS** — 8 kW |
| 3 | **Profile single-rank TP=4** (op breakdown) | What TP collectives actually cost in the *production* config, i.e. the size of the prize any PP variant chases. Never measured. | TODO |
| 4 | **Measure a real device-to-device handoff** | Every PP number is `PP_HANDOFF=none`. If a real handoff costs >5% it erases the thin margins. | TODO |
| 5 | **Attack MoE Dispatch + Combine** | 33.8% of device time, largest single category. SP-axis, so PP cannot touch it. Biggest lever for the config that ships. | TODO |
| 6 | **Close the ~13.7% host/dispatch overhead** under trace replay | 100.1 ms device busy vs 116.0 ms wall clock. | TODO |
| 7 | **Re-run the 25,600 PP row chunked** (`PP_CONTEXT=25600 PP_WINDOW=5120`) | PP's best window is measured single-shot, not on the production path. | TODO |
| 8 | **Re-measure accuracy (§5) and op breakdown (§6) on 12 kW** | Both have only ever been run on 8 kW. | TODO |
| 9 | **Fix the `analyze_ops_perf.py` CCL regex** | Reports CCL at 8.0% vs the true 4.5%; it string-matches names and `LayerNormPostAllGatherDeviceOperation` contains "AllGather". | TODO |
| — | ~~PP=2 x (8,2)~~ | **DONE — closed on its merits.** 27,061 tok/s corrected (0.83x single-rank, 0.71x PP=4). See below. | done |

### Standing recommendation

**Hold.** The previous recommendation ("keep single-rank; PP=4 wins only 1 of 4 windows on 12 kW")
rested on numbers the §0 bug understated. On 8 kW, corrected, PP=4 wins every window on the
steady-state metric. The 12 kW re-run (queue #1) decides this, and until it lands neither
recommendation is supported. Single-rank remains the default in the meantime because it is the tuned,
validated, single-mesh path — not because PP=4 has been shown to lose.

### Why #3 (PP=2 × (8,2)) is the interesting variant

> **Superseded — kept as the pre-measurement rationale.** The reasoning below was sound but the
> premise was wrong: it treats halving TP as the interesting variable, when the depth term
> (layers per stage) dominates. Measured outcome and the corrected analysis are two sections down.

Put the configurations on a ladder and the gap is obvious:

| config | SP | TP | chips/stage | status |
|---|---|---|---|---|
| PP=1 × (8,4) | 8 | 4 | 32 | production; power-bound, scales +11-38% with power |
| **PP=2 × (8,2)** | **8** | **2** | **16** | **untested — this queue item** |
| PP=4 × (8,1) | 8 | 1 | 8 | flat vs power at 3/4 windows; wins 1/4 on 12 kW |
| PP=4 × (4,2) | 4 | 2 | 8 | rejected, see §7 — worse than no PP at all |

PP=2 × (8,2) is the only untried point that **holds SP=8 fixed** — the variable `(4,2)` violated — while
halving the TP collective width instead of eliminating it. Three reasons it could beat PP=4:

1. **It attacks PP's worst window directly.** The 102,400 total-throughput loss (0.87x on 12 kW) is
   pipeline fill/drain. Two stages instead of four roughly halves that penalty.
2. **Keeping TP=2 should keep some power scaling**, so it need not flatline at 1.00x the way `(8,1)`
   does — and 12 kW is the target.
3. **Cheaper to productise**: 2 driver ranks instead of 4, 2 weight caches instead of 4, and 18 layers
   divides 36 evenly.

Against it: if deleting collectives is what PP buys, TP=2 only deletes half of it. Which is exactly
what queue item #1 is for.

### Result of #3: PP=2 x (8,2) measured on 8 kW — a clear loss at every window

Measured 2026-08-20, `kmabee/mistral4-prefill-full` @ `ca63b3d`, `PP_HANDOFF=none`, traced, same
commands as PP=4 with `-8x2` in place of `-8x1`. Correctness first: the layer-slicing test passes at
**both** pipeline depths — `pp4` (9 layers/stage) and `pp2` (18 layers/stage) each sample token 2 at
p=0.7147, identical to single-rank 36L — so these are throughput numbers for a correct pipeline.

| window | PP=2 x (8,2) | PP=4 x (8,1) | single-rank | PP=2 vs single-rank |
|---|---|---|---|---|
| 5,120 | **27,061** (corrected) | 38,027 | 32,611 | **0.83x** |
| 25,600 | 13,370 (buggy) | 45,238 | 27,090 | — re-run pending |
| 102,400 | 11,098 / 11,368 (buggy) | 20,115 / 23,132 | 19,810 | — re-run pending |
| 261,120 | 9,227 / 9,348 (buggy) | 12,192 / 12,408 | 10,888 | — re-run pending |

Only 5,120 has been re-measured with the fixed driver; it moved 12,346 -> **27,061** (2.19x, the
expected EP=2 correction). The remaining rows are left as measured, marked buggy, and are not worth
re-running because the depth argument below closes the configuration regardless: PP=2 is 0.71x of PP=4
at the one window where both are clean.

### Why PP=2 loses: shorter stages beat wider experts

**A first pass at this concluded EP=2 was pathological. That was wrong**, and the error is worth
recording because it is easy to repeat: per-layer times were derived from *concurrent* runs and
compared against a *single-stage* baseline, which folds contention into what looks like a
parallel-efficiency number. Timing one stage alone with `pp_layer_sweep.py`
(`iter_ms(N) = intercept + slope*N`, layers ∈ {1,2,4,9,18}) says the opposite:

| stage shape | slope ms/layer | intercept ms | 18 layers | chips | scaling vs (8,1) |
|---|---|---|---|---|---|
| (8,1) | 13.84 | -0.46 | 248.71 | 8 | baseline |
| (8,2) | **9.43** | -0.76 | **169.04** | 16 | **73% of linear** |

Both are perfectly linear with a zero intercept, so per-layer cost is constant in stage depth and
trace launch/sync is free at this scale. `(8,2)` is **1.47x faster per layer** than `(8,1)` on 2x the
chips — an ordinary 73% scaling efficiency. There is no EP anomaly.

PP=2 loses for a structural reason instead, and the arithmetic is simple:
throughput = `W / (layers_per_stage x ms_per_layer)`.

| config | ideal iteration | ideal tok/s | measured (fixed driver) | achieved |
|---|---|---|---|---|
| PP=4 x (8,1) | 9 x 13.84 = 124.6 ms | 41,090 | **38,027** | 93% |
| PP=2 x (8,2) | 18 x 9.43 = 169.7 ms | 30,170 | **27,061** | 90% |

**The pipeline-depth term dominates the per-layer term.** Halving the depth doubles the stage's layer
count, which more than cancels the 1.47x per-layer gain from doubling its width.

The model predicts the outcome quantitatively: PP=2/PP=4 should be
`(9 x 13.84)/(18 x 9.43)` = **0.73x**, and the measured ratio with the fixed driver is
27,061/38,027 = **0.71x**. So PP=2 is closed on structure, not on an artefact — its ceiling sits below
both PP=4 and single-rank's 32,611. Both configurations now achieve ~90% of their single-stage
ceiling, which is what a healthy pipeline should look like.

### Contention, and the false trail it led down

**Real contention is unremarkable.** With realistic routing (`pp_contention.py`):

| shape | 1 stage | 2 stages | 4 stages | contention |
|---|---|---|---|---|
| (8,1), 9L | 121.20 ms | 134.24 ms | 134.89 ms | **1.11x**, saturated |
| (8,2), 18L | 168.98 ms | 200.25 ms | — | **1.19x** |

`(8,1)` contention appears entirely at the second stage and then stops — 2 to 4 stages costs 0.5%, so
four stages overlap essentially perfectly.

**The trail worth recording.** Before the §0 bug was found, the same table read 1.24x and 2.45x, and
`414.7 > 2 x 169.04` implied two stages doing worse than running serially. That "backwards" pattern
sent me hunting a mechanism that does not exist — I ruled out an interleaved submesh carve
(`carve_probe.py`: the `(8,4)` parent's logical columns map to physical `0, 4, 12, 8`; `(8,2)` carves
adjacent pairs, cols {0,1} = phys 0&4 and cols {2,3} = phys 12&8) and then column-axis routing-plane
scarcity, before checking what the ranks were actually being fed. **Both figures were contention plus
degenerate routing.** The lesson is the ordinary one: validate the harness's inputs before theorising
about the hardware.

**Ruled out — an interleaved submesh carve.** `carve_probe.py`: the `(8,4)` parent's logical columns
map to physical `0, 4, 12, 8`; `(8,1)` carves one physical column each; `(8,2)` carves sub0 = cols
{0,1} (phys 0&4) and sub1 = cols {2,3} (phys 12&8) — **adjacent pairs, not interleaved**.

**Leading hypothesis, under test.** An `(8,1)` stage has an expert-parallel axis of width 1 and so
emits *no* column-axis traffic at all, while every `(8,2)` stage does; the fabric log reports
`only 2 routing planes are available` on some column directions. Two stages competing for a scarce
shared column-axis resource that four `(8,1)` stages never touch fits every data point, including why
single-rank `(8,4)` is unaffected (one mesh, no sibling). `pp_contention.py` tests the falsifiable
prediction: cost should climb with stage count for `(8,2)` and stay flat for `(8,1)`.

**The 21% I thought was available was mostly this bug.** PP=4 now achieves 93% of its single-stage
ceiling (38,027 of 41,090); the residual ~7% is the real 1.11x cross-stage contention, which
saturates at two stages and is not obviously worth chasing.

**Also ruled out — tuned-matmul fallback.** At EP=2 the MLA weights have Kt=64 (`q_a_proj`,
`kv_a_proj_with_mqa`, `o_proj`) or Kt=32 (`q_b_proj`), and every tuned `in0_block_w` in
`MLA_MATMUL_CONFIG` divides those, so `_cfg_fits_weight` rejects nothing — confirmed by a fallback
count of 0 on a completed run. It does land in the *other* branch of that guard's known gap (at Kt=64
a config tuned for a different K applies silently), but MLA matmuls are ~21.6% of device time and
cannot account for a 2.45x gap.

### Next measurements

1. **Re-run 12 kW with the fixed driver** — queue #1, and the one that decides the recommendation.
2. **Re-measure `(4,2)`** — queue #2, rejected on buggy EP=2 numbers.
3. **PP=2 x (16,1)** — dropped. The depth arithmetic above says a 2-stage pipeline cannot beat a
   4-stage one at equal per-layer cost, so it is not worth the SP=16 ring-attention risk.
4. **A 2-device CCL microbenchmark** — deprioritised. It was motivated by the EP=2 "pathology", which
   turned out not to exist.

### The open question behind all of this

PP=4 measured 1.27-1.53x at 25,600 — **far more than eliminating TP collectives can plausibly buy.**
The 4.5% CCL figure in §6 was measured on a *TP=1 PP stage*, which by construction has no TP
collectives at all, so it says nothing about the production config's collective cost. The likely real
mechanism for PP's gain is **overlapping host dispatch across independent meshes** — four submeshes let
stage N+1's dispatch overlap stage N's execution — rather than deleting CCL. If that is true it
explains PP's flat power scaling exactly (host-bound work does not care about board power), and it
means queue item #5 is cheaper than productising any PP variant and would capture the same win.
Item #1 settles it.

---

## 4. Full results and how to reproduce

### 4.1 Single-rank — SP=8 × TP=4, chunked (the production path)

`prefill_producer.py:534` computes `target_chunks = ceil(real_len / CHUNK_SIZE)` unconditionally with
`CHUNK_SIZE=5120`; there is no single-shot branch in the runner. Numbers measured any other way are
microbenchmarks. Measured at `c34e372b47d`; 12 kW iter0/iter1 agreed within 0.8% on every row.

| window | chunks | 8 kW | 12 kW | 8 kW wall clock |
|---|---|---|---|---|
| 5,120 | 1 | 32,611 | 36,312 | 2 min 22 s |
| 25,600 | 5 | 27,090 | 32,821 | 6 min 22 s |
| 102,400 | 20 | 19,810 | 23,546 | 5 min 10 s |
| 261,120 | 51 | 10,888 | 15,022 | 3 min 37 s |

Most of the wall clock is model build + weight load, not measurement — the 261,120 forward is 24 s.

```bash
T=models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py
ID="$T::test_mistral4_prefill_transformer_chunked_no_pcc[blackhole-mistral4-mesh-8x4-L36-chunks51-two_iters-traced]"
PREFILL_NOPCC_SEQ_CACHE=261120 pytest "$ID" -q -rs -s
```

Swap `chunks51`/`261120` for `chunks01`/`5120`, `chunks05`/`25600`, `chunks20`/`102400`.
`-s` is required or the timing table is swallowed. `traced` is required — the `notrace` row reports
~0.67 s/chunk flat and is measuring host dispatch, not the device.

### 4.2 PP=4 × (8,1) — four 9-layer stages, TP=1, concurrent

Four stages of 9 layers, each SP/CP=8 × TP=1 on 8 chips, tiling the 32-chip galaxy. Motivation: every
collective in this model's dense path is on the TP axis, so TP=1 deletes all of it.

`PP_HANDOFF=none`, traced. The 8 kW column is **corrected** (fixed driver, 2026-08-20 21:00+); the
12 kW column still carries the §0 bug and must be re-run.

| window | 8 kW corrected | 8 kW verbatim | 8 kW buggy | 12 kW (STALE) |
|---|---|---|---|---|
| 5,120 | **38,027** | `min_ms=134.6 med_ms=135.0` | 34,059 | 34,004 |
| 25,600 | **45,238** | `min_ms=565.9 med_ms=566.7` | 41,376 | 41,664 |
| 102,400 | **23,132** steady / 20,115 total | `total_s=5.09 med_ms=221.3` | 23,234 / 19,520 | 23,437 / 20,592 |
| 261,120 | **12,408** steady / 12,192 total | `total_s=21.42 med_ms=412.6` | 12,434 / 12,346 | 16,377 / 15,697 |

The pre-fix 8 kW figures were within 1.4% of the original pre-rebase branch, so the PP integration
itself was always faithful — the driver was feeding it the wrong inputs.

**Read `total` vs `steady` carefully.** `total` is the whole request including pipeline fill/drain;
`steady` is the steady-state median, i.e. the server case with back-to-back requests. At 102,400 the
total is negative on both boxes — fill/drain eats the entire benefit for a single request.

Wall clock: 5,120 → 1 min 49 s (8 kW) / 4 min 26 s (12 kW); 25,600 → 2 min 59 s / 3 min 12 s;
102,400 → 1 min 59 s / 1 min 49 s; 261,120 → 1 min 57 s / 2 min 13 s.

```bash
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/kmabee/mistral4_caches/ttnn_cache_pp   # NOT the 8x4 cache
export PP_HANDOFF=none
T=models/demos/deepseek_v3_d_p/tests/test_prefill_pipeline_concurrent.py

PP_WINDOW=5120  PP_ITERS=12 pytest "$T::test_mistral4_pp4_concurrent_throughput[blackhole-mistral4-mesh-8x4-8x1]" -q -s
PP_WINDOW=25600 PP_ITERS=12 pytest "$T::test_mistral4_pp4_concurrent_throughput[blackhole-mistral4-mesh-8x4-8x1]" -q -s
PP_CONTEXT=102400 PP_WINDOW=5120 pytest "$T::test_mistral4_pp4_concurrent_longctx[blackhole-mistral4-mesh-8x4-8x1]" -q -s
PP_CONTEXT=261120 PP_WINDOW=5120 pytest "$T::test_mistral4_pp4_concurrent_longctx[blackhole-mistral4-mesh-8x4-8x1]" -q -s

# correctness (uses the 8x4 cache, not the pp one)
TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/kmabee/mistral4_caches/ttnn_cache_8x4 \
  pytest models/demos/deepseek_v3_d_p/tests/test_prefill_pipeline_stages.py -q -s
```

The kernel cache (`~/.cache/tt-metal-cache`, ~27 GB) is local per machine and **survives
`tt-smi -r`** — a run straight after a reset still completed in 1:49. A first run on a machine that
has never built these kernels adds a few minutes, no more. If a run takes dramatically longer,
suspect contention, not compilation.

---

## 5. Accuracy

Measured on 8 kW only.

| check | value | threshold |
|---|---|---|
| block, `pcc-prompt_5k` — output | 0.990894 | 0.95 |
| block — KVPE KV part | 0.999896 | 0.999 |
| block — KVPE PE part | 0.999899 | 0.999 |
| chunked per-layer, `chunks03` | passes | raw ≥ 0.88 **or** nPCC ≥ 0.90 |
| PP stages vs single-rank (8 kW) | same token (2), p=0.7147 | exact match |
| PP stages vs single-rank (12 kW) | same token (2), p=0.7147 | exact match |

Runtimes: block PCC ~1 min 25 s, chunked PCC ~4 min 17 s, PP stages 2 min 41 s (8 kW) / 2 min 1 s
(12 kW).

**Late layers are expected to look bad on raw PCC.** Layers 32-35 read raw 0.28 / 0.32 / 0.35 / 0.65
because a few channels carry massive activations (attention sink) and raw PCC is dominated by them.
Their nPCC is 0.9657 / 0.9673 / 0.9692 / 0.9913. Judge depth on nPCC.

---

## 6. Where the time goes (device op breakdown, one PP stage, window 5,120, 8 kW)

Device busy 100.1 ms/forward vs 116.0 ms traced wall clock → ~13.7% host/dispatch overhead survives
even under trace replay.

| op | % device time |
|---|---|
| MatmulDeviceOperation | 21.6 |
| MoE Combine | 20.0 |
| UnifiedRoutedExpertFfn | 15.4 |
| RingJointSDPA (attention) | 14.7 |
| MoE Dispatch | 13.8 |
| AllGather (true CCL) | 4.5 |
| LayerNorm pre/post-gather | 3.4 |

**MoE Dispatch + Combine together are 33.8%** — see recommendation #4.

Caveat: `analyze_ops_perf.py` reports CCL as 8.0% rather than 4.5%, because its regex string-matches
op *names* and `LayerNormPostAllGatherDeviceOperation` contains "AllGather". Fix before trusting it.

---

## 7. Measured and rejected — do not re-run

### PP=4 × (4,2) — re-measured; not rejected, but not the candidate

| window | buggy | corrected | vs (8,1) corrected |
|---|---|---|---|
| 5,120 single-shot | 24,416 | **40,784** | 1.07x — wins |
| 25,600 single-shot | 17,483 | 24,729 | 0.55x |
| 102,400 chunked steady | not run | 13,862 | 0.60x |
| 261,120 chunked steady | not run | 6,700 | 0.54x |

The original verdict — "worse than not using PP at all, do not retry this variant" — **was wrong**, an
artefact of the §0 bug, which inflated this EP=2 configuration by 1.67x at 5,120 and 1.41x at 25,600.
It is in fact the fastest configuration measured at the 5,120 single-shot window.

It is still not the candidate: the advantage is core-utilisation with an **empty** KV, and it inverts
as soon as context accumulates (see §1). Keep it in mind for short-prompt / low-context serving, not
for the 100K-256K path.

### Others

- **Single-shot throughput.** The runner always chunks; single-shot dies on L1 at 102,400
  (`circular buffers grow to 1721216 B beyond max L1 size of 1572864 B`) and peaks at ~33.5k around a
  25k window. Interesting as a curve, irrelevant as a production number.
- **Eager (untraced).** ~0.67 s/chunk flat — measuring host dispatch.
- **`PP_HANDOFF=host`.** 42 MB/hop, ~1121 ms/iteration. Shows what a naive host hand-off costs; not a
  candidate. A *device* hand-off is still unmeasured and is recommendation #2.

---

## 8. Environment requirements

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export LD_LIBRARY_PATH=$PWD/build_Release/lib:$LD_LIBRARY_PATH   # or import ttnn fails on _ttnncpp.so
export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/kmabee/mistral4_caches/ttnn_cache_8x4  # or ttnn_cache_pp for PP
export TT_MISTRAL4_PREFILL_HOST_REF_CACHE=/data/kmabee/mistral4_caches/ref_cache
export PREFILL_TRACE_DIR=/data/kmabee/mistral4_golden_traces/mistral4_15360_36L_fp32rope  # PCC rows only
```

Omitting `MISTRAL4_HF_MODEL` makes the test try to download the model. Omitting
`TT_MISTRAL4_PREFILL_TTNN_CACHE` rebuilds a 65 GB weight cache.

`/data/kmabee` is NFS **shared between the 8 kW and 12 kW boxes** — model weights, ttnn caches and
this checkout are the same files on both. `~/.cache/tt-metal-cache` is *not*; it is local per machine.

A failure at *fixture setup*, before any model code, is usually contention. Two signatures seen:

- `Sysmem mapped at unexpected NOC address` — left by a `kill -9` on a process holding 32 devices.
  One run spent **34 min** retrying before reporting it.
- `TopologyMapper auto-discovery: Downgrading to mesh shape 4x4 (16 total nodes) for 32 physical
  chips` → `Requested more devices (32) than available (16)`. Seen on 12 kW right after another
  user's job exited.

Both are fixed by `tt-smi -r`, and both are worth reaching for immediately rather than debugging.
Check `/dev/shm` for `tt_h2d_<pid>_*` whose pid is still alive — `fuser` will not show another user's
process.

---

## 9. `kmabee/mistral4-prefill-full-rebased.aug27` — rebase onto the refined shared fixes, full current-fabric sweep

Measured 2026-08-28 on `bh-glx-b03u02` (a third 32-chip Blackhole galaxy, distinct from the 8 kW and 12 kW boxes named above — treat as a data point, not a repeat of either target) at `399cc0712f4` (`kmabee/mistral4-prefill-full-rebased.aug27`, `FABRIC_2D_TORUS_XY`), in `/data/kmabee/tt-metal-2`.

This branch rebases `kmabee/mistral4-prefill-full-rebased` onto `kmabee/issue_54515-54519_integration` (squashed to one commit), which carries refined, individually-reviewed versions of five fixes this branch already had cruder versions of from 2026-08-20/21: MoE routed-expert cache dtype-awareness (#54515), the reference rope built from config not model (#54516), the MLA tuned-matmul-config K guard (#54517, byte-identical to what this branch already had), the KVPE-compare fix (#54519), and — the one with an actual device-time delta — the distributed RMSNorm TP=1 guard (#54518). This branch's own version of that guard (`59ac0719ff2`, 2026-08-20) kept the two-op `rms_norm_pre_all_gather`/`rms_norm_post_all_gather` pair and only skipped the collective at TP=1; #54518 instead takes the single fused `ttnn.rms_norm` op at TP=1, verified bit-identical and claimed 2.7x faster device-side.

**Single-layer profile confirms the norm win directly.** Re-running the PP=4-stage (TP=1) single-layer capture from §HANDOFF_TRACY_PERF_SINGLE_LAYER (window 5,120) on this branch: the two `LayerNormPreAllGatherDeviceOperation`/`LayerNormPostAllGatherDeviceOperation` ops (0.359 ms combined pre-rebase) are gone, replaced by a single fused `LayerNormDeviceOperation` at 0.164 ms — **2.19x faster**, consistent with the claimed 2.7x. The single-rank (TP=4) capture is unchanged (3.274 ms vs 3.473 ms pre-rebase, within noise) as expected: the guard only fires at `cluster_axis` length 1, which TP=4 never is.

**Full sweep, both parallelisations, all four production windows:**

| window | single-rank (SP8×TP4) | PP=4×(8,1) | ratio |
|---:|---:|---:|---:|
| 5,120 | 29,767 | 35,003 (min) / 34,583 (med) | 1.17–1.18x |
| 25,600 | 25,754 | 45,604 (min) / 45,541 (med) | 1.77x |
| 102,400 | 19,814 | 24,157 total / **27,791 steady** | 1.22x total / **1.40x steady** |
| 261,120 | 13,407 | 16,794 total / **17,907 steady** | 1.25x total / **1.34x steady** |

All rows `PASSED`, traced, `PP_HANDOFF=none`, single-shot windows via `test_mistral4_pp4_concurrent_throughput`, chunked long-context windows via `test_mistral4_pp4_concurrent_longctx` and `test_mistral4_prefill_transformer_chunked_no_pcc`, same node IDs as §4 except `mesh-8x4` renamed to `torus-xy-8x4` for the chunked-transformer test post-fabric-migration. No PCC re-check in this sweep (the norm fix's correctness is covered by its own bit-identical PCC test, §4.2's node IDs elsewhere carry the model's PCC coverage); this section is throughput-only.

**This closes `LLM_PERF_INVESTIGATION_FINDINGS.md` rev 2's still-open item #1** — current-fabric (`FABRIC_2D_TORUS_XY`) single-rank and PP=4 numbers at 102,400 and 261,120 had never been measured; §0b only had 5,120/25,600 on this fabric (8 kW: single-rank 29,595/24,854, PP=4 35,110/42,473, ratios 1.19x/1.71x), which this sweep's 5,120/25,600 rows match within noise, confirming the rebase changed nothing at short context.

**At long context, both configs beat the stale `FABRIC_1D` baseline** (§1, 8 kW, pre-rebase): single-rank 13,407 here vs 10,888 there at 261,120; PP=4 steady 17,907 here vs 12,408 there. The `FABRIC_1D` numbers were never valid on current `main` anyway (§0b), so this is not a "fabric migration helped" claim — it is the first honest long-context number on the fabric that actually ships.

**Standing recommendation still holds and is now on firmer ground**: PP=4×(8,1) wins every window measured, by a wider margin at long context (1.34–1.40x steady) than the short-context-only data previously supported.

---

## 10. Real device-to-device PP=4 — every `PP_HANDOFF` number in §9 replaced by a measurement

> **Superseded in part by `MISTRAL4_PP4_VS_SINGLE_RANK.md` (2026-08-31).** §10's *conclusion* stands —
> `PP_HANDOFF=none` is not a bound on real PP=4 — but its **ratios** compare this section's runner
> numbers against §9's pytest single-rank figure, and those two harnesses differ: single-rank measures
> **35,597 tok/s** through the runner vs §9's **29,767** at window 5,120, ~20% apart. The newer document
> re-measures BOTH configurations through the same harness and is the one to quote.

Measured 2026-08-31 on `bh-glx-b03u02` at `779a4af546b` (`kmabee/mistral4-prefill-full-rebased.aug27`), in `/data/kmabee/tt-metal-2`.

**Every PP=4 number in §9 (and §0b, §1) came from `test_prefill_pipeline_concurrent.py` with `PP_HANDOFF=none`, which does not move the activation between stages at all.** Mistral4 has now been run through the real pipeline-parallel prefill runner (`models/demos/common/prefill/runners/prefill_runner.py`, 4 processes under `tt-run`, activation carried stage-to-stage by a `ttnn` `MeshSocket` over fabric), so the bracket can be replaced by measurements. New topology config, in-tree:

- `tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x8x1_z_chain_graph_descriptor.textproto` (SP axis LINE) and `..._4x8x1_z_chain_torus_y_graph_descriptor.textproto` (SP axis RING) — one 8x4 galaxy as 4 Z-chained `[8,1]` **column** meshes.
- `models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_intragalaxy_4rank_8x1.yaml` and `..._8x1_torus_y.yaml` — the matching rank bindings.

Use the **`torus_y`** pair for any comparison against §9: §9's submeshes are carved from an 8x4 mesh opened `FABRIC_2D_TORUS_XY`, so their SP-axis ring-attention collectives run Ring. Plain `2d` (LINE) measures 1.27x slower at 5,120 and is not a like-for-like configuration.

**Full sweep, traced, 36 layers, `PP=4 x (8,1)`** (ms/chunk in brackets):

| window | shape | single-rank (§9) | PP=4 `PP_HANDOFF=none` (§9) | **PP=4 REAL D2D** | real / none | real / single-rank |
|---:|---|---:|---:|---:|---:|---:|
| 5,120 | single-shot | 29,767 | 34,583 med [148.0] | **46,269 / 46,182** [110.7] | **1.34x** | **1.55x** |
| 25,600 | single-shot | 25,754 | 45,541 med [562.1] | **48,832** [524.2] | **1.07x** | **1.90x** |
| 102,400 | 20 x 5,120 | 19,814 | 24,157 total / 27,791 steady [184.2] | **24,382 total** / 24,276–25,542 steady [210.9] | 1.01x total / 0.87–0.92x steady | **1.23x** |
| 261,120 | 51 x 5,120 | 13,407 | 16,794 total / 17,907 steady [285.9] | **17,052 total** / 16,958 steady [301.9] | 1.02x total / 0.95x steady | **1.27x** |

Throughput is the **last rank's** chunk-to-chunk interval (the producer's own tok/s includes pipeline fill and a ~6.3 s first-push compile). Single-shot rows are the median steady interval; chunked rows reproduce §9's two metrics exactly (`total = CONTEXT/wall`, `steady = CHUNK/median(interval[PP-1:])`). Reproducible to 0.2% (two independent runs at 5,120). All four stages within 1.5% of each other.

**Two caveats on the `total` column, and one on the whole table.** `total` is a single-prefill latency only when the run drove ONE request: with several requests back to back, request N+1's pipeline fill hides inside request N's drain and `tokens/wall` becomes a stream throughput. The 102,400 row is measured from a 2-request run and is quoted above as the correctly isolated single-prefill figure (24,382 tok/s = 4.200 s for 102,400 tokens); the 261,120 row was a 1-request run and needed no correction. The gap between the two windows is fill/drain: PP=4 loses ~3 chunks to fill and ~3 to drain, 6 of 51 chunks at 261,120 (~12%) but 6 of 20 at 102,400 (~30%). And across the whole table, **no first token is produced** — `PREFILL_KV_ONLY_LAST_LAYER` defaults to 1, so the last rank runs its final layer kv-only and skips the final norm and LM head. These are prefill-completion latencies, **not TTFT**; a true TTFT needs `PREFILL_KV_ONLY_LAST_LAYER=0`, which adds a norm plus an LM-head matmul over the vocab on the last rank and a kernel-compile cost no run here has paid. Expect PP=4 to look *worse* than single-rank for a single-chunk request, where there is nothing to pipeline and the chunk traverses all four stages serially.

**§9's conclusion survives; its numbers do not.** PP=4×(8,1) still wins every window, by 1.27–1.90x over single-rank. But a `PP_HANDOFF` number is not a bound on the real thing, and the error has two unrelated causes pulling opposite ways:

- **At single-shot windows `none` UNDERSTATES real PP=4 by a fixed ~37.5 ms/chunk** — 37.3 ms at 5,120 and 37.9 ms at 25,600, the same number at two windows whose per-chunk work differs 3.5x. That is not transport; it is `test_mistral4_pp4_concurrent_throughput`'s own per-iteration host cost, since one Python thread copies metadata for 4 stages, issues 4 trace replays and calls `synchronize_device` 4 times every iteration. The real runner gives each rank its own process and dispatch thread and that cost disappears. Being a fixed offset, it is 34% of the answer at 5,120 and ~0% by 261,120.
- **At the chunked long-context windows real D2D gives up 16–27 ms/chunk on the *steady* metric**, and this part *is* the transport. With `HANDOFF=none` the longctx loop's downstream stages replay against a fixed seeded input, so stage r's chunk carries no dependency on stage r-1; the real pipeline has both the dependency and the transfer. A direct microbenchmark of the transport (4 ranks, no weights, the runner's own `_d2d_send`/`_d2d_recv`) puts one hop of the 42 MB `[1,1,5120,4096]` bf16 activation at **~11 ms end-to-end** (0.47–0.59 ms host push; 2.25–2.33 ms enqueue + grant + device sync), and `_lease_reclaim` blocks on `wait_for_fabric_links()` at the top of every chunk — so **the transport looks serialised with compute rather than overlapped**. Hypothesis, not measurement, but the right magnitude, and the clearest optimisation lead here: ~5–10% at 102,400+.

**Correctness (single-rank, PCC-gated).** Mistral4 also now passes through the real runner + producer with the per-slot KV read-back gate (`test_producer_runner_e2e.py`, unmodified, via its `PREFILL_PROMPT_FILE` scenario with `PREFILL_REUSE_TRACE_DIR` pointed at a pre-built golden). At 36 layers the deep-KV PCC bottoms at **0.9034**; the per-layer profile runs 0.99992 at layer 0 down to a 0.90–0.95 plateau from layer 15, with the rope half at 0.988–0.9999 throughout. An 8-layer control run reproduces layers 0–7 to all five printed digits and bottoms at 0.9931, which is what establishes this as depth accumulation rather than miswiring. **The producer's default `PREFILL_STANDALONE_CHUNKED_PCC=0.93` is Kimi-calibrated and wrong for a 36-layer Mistral4 read-back — use 0.88**, consistent with `test_prefill_transformer_chunked.py`'s own `KV_CACHE_PCC_THRESHOLD = 0.85` for this quantity (set at GLM-5.2's observed ~0.86 over 78 layers). There is no PCC gate for the PP runs: `prefill_runner` rejects `PREFILL_MOCK_MIGRATION` for `num_ranks>1`.

**Two traps worth knowing before reproducing this** (full list in `~/debug-docs/mistral4_prefill_planning-noissue/perf/RESULTS_PP4_REAL_D2D.md`):

- The weight cache's device-count component is **namespacing only** — `32dev/8x1` and `8dev/8x1` files are byte-identical, so the existing 36-layer `8x1` cache can be hardlinked into the `8dev` namespace each PP rank resolves, with no rebuild. Cache keys are *global* layer indices, so all four ranks share one directory and it must hold layers 0–35.
- A logical `[8,1]` column **spans two trays**, so its device set cannot come from tray/slice discovery; the per-rank `TT_VISIBLE_DEVICES` in the bindings were read off a live 8x4 mesh via `create_submeshes(MeshShape(8,1))`.
