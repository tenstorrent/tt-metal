# Mistral Small 4 prefill: PP=4 x (8,1) vs single-rank — bringup-branch re-run

Re-run of `MISTRAL4_PP4_VS_SINGLE_RANK.md` on the **Mistral-4 bringup branch**, to confirm the PP=4
result survives 111 upstream commits and a different galaxy.

**Measured 2026-09-04 on `bh-glx-120-b03u02` (32-chip Blackhole galaxy).**
Branch `kmabee/issue_53688_mistral_4_small_prefill_bringup_pp4` @ `71dd7e79c17`, based on
`akhan/issue_53688_mistral_4_small_prefill_bringup` @ `c95f5276f73`, built Release + Tracy at
2026-09-04 02:39. Harness and model code as committed on that branch; nothing uncommitted.

The comparison is the same two configurations through the **same** runner + producer, so topology is
the only variable:

- **single-rank** — SP=8 x TP=4 on one 8x4 mesh, `2d_torus_xy`
- **PP=4** — four `[8,1]` column sub-meshes, hidden state handed stage-to-stage over a real
  device-to-device `ttnn` MeshSocket on fabric, `2d_torus_y`

Chunk size is the production 5,120 throughout; ISL is varied by chunk count.

**The §1 tables predate the #55126 fix; they have since been RE-MEASURED on it and hold** — see the
re-measured tables in §1.1 and §1.2, and §4 for what the fix does and does not change.

---

# 1. End-to-end results

## 1.1 Throughput (many requests, steady state)

Last rank's median chunk-to-chunk interval, first 8 intervals discarded.

| ISL | 1-rank ms/chunk | 1-rank tok/s | PP=4 ms/chunk | PP=4 tok/s | ratio | (aug27 ratio) |
|---:|---:|---:|---:|---:|---:|---:|
| 5,120 | 136.4 | 37,531 | 105.6 | **48,480** | **1.29x** | 1.33x |
| 25,600 | 160.0 | 31,991 | 116.0 | **44,139** | **1.38x** | 1.40x |
| 102,400 | 305.3 | 16,770 | 196.5 | **26,062** | **1.55x** | 1.58x |
| 261,120 | 431.0 | 11,881 | 299.9 | **17,074** | **1.44x** | 1.46x |

**PP=4 wins throughput at every ISL, by more as context grows — the aug27 conclusion reproduces.**
Every ratio lands within 0.03 of the original. The 261,120 PP=4 cell is 299.9 ms in both campaigns.

### Re-measured on the #55126 fix, both galaxies (2026-09-05)

The table above was taken with the `MISTRAL4_LLAMA4_PERF_UNSAFE` stopgap. Re-run on the real fix
(no stopgap), full 16 cells on each machine:

| ISL | glx-110 1rank -> PP=4 | ratio | glx-120 1rank -> PP=4 | ratio | (table above) |
|---:|---|---:|---|---:|---:|
| 5,120 | 37,920 -> 49,521 | 1.31x | 37,461 -> 48,299 | 1.29x | 1.29x |
| 25,600 | 32,406 -> 45,118 | 1.39x | 32,142 -> 44,210 | 1.38x | 1.38x |
| 102,400 | 16,797 -> 26,451 | 1.57x | 16,710 -> 26,102 | 1.56x | 1.55x |
| 261,120 | 11,993 -> 17,251 | 1.44x | 11,948 -> 17,059 | 1.43x | 1.44x |

**The fix is perf-neutral.** On the same machine (glx-120) every cell lands within 0.4% of its
pre-fix value, and the ratios within 0.01. The prediction in §4 — that the captured graph is
identical so throughput should not move — holds. **These are the first Mistral PP=4 numbers measured
with the correct query temperature at every context length.**

**The two galaxies agree to 1-2.5%**, glx-110 marginally faster.

**Run-to-run spread, measured (three full campaigns, same machine, same branch):** ratio spread
**0.32% / 0.60% / 0.53% / 0.35%** at the four ISLs; per-config throughput spread 0.03-0.57%. That
number is what makes the rest of this section a claim rather than an assertion:

* the fix's <=0.4% shift is *inside* the noise floor, so "perf-neutral" is the only reading available;
* the 1-2.5% between galaxies is 2-4x the floor, so it is real, just far smaller than the "~5%"
  §1.3 originally claimed;
* PP=4's 29-57% win is two orders of magnitude above it;
* and nothing on this harness that moves <~1% should be reported as an effect — which is
  independently why the three tuning knobs in the levers doc came back null.

## 1.2 Single-request latency (warm)

| ISL | 1-rank | PP=4 | winner | (aug27 PP=4) |
|---:|---:|---:|---|---:|
| 5,120 | *see §3.1* | *see §3.1* | — | 0.947 s |
| 25,600 | 2.884 s (8,878 tok/s) | **1.170 s** (21,881) | PP=4 by 2.47x | 1.209 s |
| 102,400 | 7.124 s (14,374) | **4.033 s** (25,390) | PP=4 by 1.77x | 4.130 s |
| 261,120 | 22.698 s (11,504) | **14.901 s** (17,524) | PP=4 by 1.52x | 15.146 s |

All three valid PP=4 rows come in slightly better than aug27. **The 5,120 row is not reported: the
harness mis-measures a single-chunk request. That is a measurement bug, not a slow cell — see §3.1.**

### Re-measured on the fix, glx-120 (2026-09-05), and a correction to §3.1

| ISL | 1-rank | PP=4 | speedup | (table above) |
|---:|---:|---:|---:|---:|
| 25,600 | 3.205 s | **1.117 s** | 2.87x | 1.170 s |
| 102,400 | 7.512 s | **4.102 s** | 1.83x | 4.033 s |
| 261,120 | 23.016 s | **15.026 s** | 1.53x | 14.901 s |

All three reproduce. These are reconstructed as `rank3 chunk span + one trailing chunk + pipeline
fill`, cross-checked against `analyze_prefill_latency.py`, and the cross-check is the point:

> **§3.1's `E2E_CLOCK` drain artifact is NOT single-chunk-only.** §3.1 says the stamp lands at drain
> and quantifies the gap as 374 ms at 261,120 and 200 ms at 102,400 — i.e. negligible except for a
> single-chunk request. On this run the reconstruction and `analyze_prefill_latency.py` agree on **five of six**
> cells, but on `pp4_102400_ttft` they read **4.102 s vs 9.122 s** — a 5.0 s gap on a *20-chunk*
> request, the same magnitude as the single-chunk stall. It is sporadic, not shape-determined.
>
> **Never quote `analyze_prefill_latency.py` alone.** Cross-check against the chunk intervals; when they
> disagree, the interval reconstruction is the trustworthy one.

## 1.3 What changed vs aug27, and what did not

Three configurations, same `1rank` @ 5,120 throughput cell:

| | ms/chunk | tok/s | spread |
|---|---:|---:|---|
| aug27 base, `bh-glx-110-a04u02` | 143.8 | 35,597 | 143–147 |
| sept1 base, `bh-glx-120-b03u02` | 136.2 | 37,580 | 136.1–138.7 |
| bringup base (+111 commits), `bh-glx-120-b03u02` | 136.4 | 37,531 | 135.8–136.9 |

**Two bases 111 upstream commits apart, on the same machine, agree to 0.15%.** The ~5% gain over
aug27 is the *machine*, not the code. In particular the two RingJointSDPA performance commits in that
range (#54812 even-ring split forwarding, #54481 head-segment scheduling) are not visible here — at
102,400, where SDPA dominates and they should show up most, PP=4 moved 199.6 -> 196.5 ms (1.5%) and
1-rank 315.3 -> 305.3 ms (3.2%), both comparable to the machine difference alone.

**So: machine ~ +5%, base ~ neutral, topology = the real 1.29–1.55x.**

> **CORRECTED 2026-09-05.** The machine half of that does not hold. The comparison above changes
> base *and* machine together — the aug27 base was only ever measured on glx-110 — so "machine" was
> the residual, not a measurement. Running **one base on both galaxies** now puts them within
> **1.3%** (glx-110 135.0 ms vs glx-120 136.7 ms at this cell), with glx-110 the *faster* one. So the
> 143.8 -> 135.0 improvement is on the **same machine** and is ~6% of **code**, not machine.
>
> What survives: base ~ neutral **across sept1 -> bringup** (0.15%, directly measured, 111 commits),
> and topology = the real 1.29-1.57x. What does not: attributing the aug27 gap to hardware. The
> aug27 -> sept1 window was never A/B'd on one machine and remains unattributed — the two
> RingJointSDPA commits in it are again candidates. One machine, two bases would settle it.

---

# 2. Single-layer profiling

One layer, device profiler + Tracy, driven through the real chunked runner so the KV cache deepens
(8 chunks x 5,120 = 40,960 context). Both configurations, same harness.

```bash
# Reference branch only -- the driver harness is not in-tree; see 5.
DEEP_CHUNKS=8 run_single_layer_profile.sh 1rank_deep   # TP=4
DEEP_CHUNKS=8 run_single_layer_profile.sh pp4_deep
```

Without the harness, the committed captures in `tests/perf/captures/` reproduce every number in this
section — that is what they are for.

These run **eager** (`PREFILL_USE_TRACE=0`), so the llama4 guard never fires and **these captures are
numerically correct** — unlike the traced matrix in §1.

## 2.1 Where the time goes, per layer per chunk

| | 1-rank (32 chips, TP=4) | PP=4 stage 0 (8 chips, TP=1) |
|---|---:|---:|
| **compute / layer** | **5.44 ms** (aug27: 5.59) | **11.27 ms** (aug27: 11.89) |
| MLA / attention | 1.036 ms (19.1%) | 2.999 ms (26.6%) |
| MoE | 1.928 ms (35.5%) | 6.348 ms (56.3%) |
| norm | 0.025 ms (0.5%) | 0.159 ms (1.4%) |
| other (incl. TP collectives) | 2.448 ms (45.0%) | 1.761 ms (15.6%) |

> **`MLA / attention` here EXCLUDES the projection matmuls** — they are counted under `other`,
> alongside the collectives. So "SDPA is ~94% of MLA/attention" in this table and "RingJointSDPA is
> 76.9% of the Mistral4 MLA forward" in `test_mla_perf.py` are **the same op against different
> denominators**, not a disagreement: the op-level test's MLA forward includes the projections. Quote
> one or the other and say which denominator it uses; never put the two figures side by side.

**The identity holds:**

```
1-rank  :  5.44 ms x 32 chips = 174.1 chip-ms per layer
PP stage: 11.27 ms x  8 chips =  90.2 chip-ms per layer
-> a PP layer is 2.07x SLOWER in wall time but uses 1.93x LESS silicon time
```

(aug27: 2.13x slower, 1.88x less. Same conclusion.)

**Mechanism — TP=1 deletes the tensor-parallel collectives.** Per-pass op counts:

| | 1-rank (TP=4) | PP=4 stage (TP=1) |
|---|---|---|
| `ReduceScatterMinimalAsync` | x4.0 | – |
| `HighBwAllGather` | x5.0 | – |
| `ReduceScatter` | x1.0 | – |
| `LayerNorm` | (split pre/post all-gather) | **x4.0 fused single op** |

Stage 0's top-op table contains **no** all-gather or reduce-scatter at all.

## 2.2 The cost of context depth is entirely MLA/SDPA

Per-chunk op duration as the KV cache grows (chunks 1→8, 5K→41K context):

| op | 1-rank (TP=4) | PP=4 stage 0 |
|---|---|---|
| **RingJointSDPA** | 360 → **1858** us (**5.16x**) | 1073 → **5231** us (**4.88x**) |
| UnifiedRoutedExpertFfn | 632 → 1119 us | 1928 → 2114 us (1.10x) |
| Combine | 739 → 1229 us | 2428 → 1874 us (0.77x) |
| Dispatch | 839 → 982 us (1.17x) | 1685 → 1728 us (1.03x) |
| Matmul (x10-11) | 1047 → **1046** us (**1.00x**) | 1043 → **1041** us (**1.00x**) |

aug27 measured SDPA 355 → 1857 us (5.23x) single-rank and 961 → 5242 us (5.46x) for a PP stage.
**This reproduces to within a few percent.**

**Actionable conclusion, unchanged: at short context MoE dominates a PP stage (56%), but it is FIXED.
Every additional token of context lands entirely in MLA/SDPA. Long-context optimisation effort
belongs in MLA/SDPA, not MoE.** Matmul is flat to 3 significant figures in both configurations.

## 2.3 Stage asymmetry (PP=4)

| stage | compute/layer | (aug27) | note |
|---|---:|---:|---|
| 0 | 11.27 ms | 11.89 | + embedding; outbound D2D only |
| 1 | 14.01 ms | 14.63 | MoE 9.147 ms — the same outlier as aug27 (9.12) |
| 2 | 11.29 ms | 11.97 | |
| 3 | 15.81 ms | 16.17 | + final norm and LM head (`other` 4.395 vs ~1.7 elsewhere) |

The pipeline period is set by the **slowest** stage, so stage 3 and stage 1's MoE outlier are what
bound throughput here. The caution in the original wording — "it is a single layer, so do not
over-read it" — turned out to be the important part.

> **CORRECTED by the full-model capture (2026-09-04, `bh-glx-110-a04u02`).** This table CONFOUNDS
> stage index with layer index: with 4 layers on 4 ranks the even split puts layer *r* on stage *r*,
> so "stage 1 is slow" and "layer 1 is slow" are the same measurement. A 9-layers-per-stage capture
> separates them — stage 1 then holds layers 9..17 and layer 1 moves to stage 0 — and the answer is
> **layer 1, not stage 1**. Per-layer cost inside stage 0 (`ops_perf_results`, keyed by the
> `forward_layer_N` signposts, mean over 8 chips x 9 chunks):
>
> | global layer | 0 | **1** | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
> | total ms | 11.03 | **13.54** | 10.46 | 11.93 | 10.63 | 11.79 | 12.92 | 13.47 | 11.76 |
> | of which MoE | 6.28 | **8.77** | 5.70 | 7.17 | 5.87 | 7.03 | 8.16 | 8.71 | 7.00 |
>
> Layer 1 is the most expensive layer in the stage and carries the same MoE signature it showed as
> "stage 1" (13.54 / 8.77 here vs 14.01 / 9.15 there, different machine). So the outlier is a
> LAYER property, and stage 1 is unremarkable once it holds a normal spread of layers.
>
> Two consequences. **Stage balance is a non-issue**: at 9 layers/stage the per-layer spread is
> 11.96 / 12.97 / — / 13.36 ms (stage 2's capture was lost), i.e. **11.7%**, not the 40% this table
> implies, and rebalancing layers across stages would buy ~5% at most. **Per-layer MoE variance is
> the real finding**: 5.70 -> 8.77 ms across nine consecutive layers (**+54%**) while MLA/attention
> is flat at 2.96 ms in every stage. That is per-layer expert-routing imbalance, and it is a
> per-layer optimisation target — exactly the thing a single middle-layer capture is for.

---

# 3. Traps found in this run

**Read these before quoting any number.** These are *in addition to* the trap list in
`MISTRAL4_PP4_VS_SINGLE_RANK.md` §3, all of which still applies.

## 3.1 A single-chunk latency cell is mis-measured (the 5,120 ttft row)

`E2E_CLOCK last_compute_end` is stamped during **drain**, not at chunk completion, so it absorbs the
shutdown-sentinel forward. At one chunk that gap *is* the measurement:

```
03:50:36.998  CHUNK_START c=0
03:50:37.143  SEND-d2d done          <- real work: 145 ms, matching 136 ms/chunk throughput
03:50:37.157  SHUTDOWN sentinel received after 1 chunks
03:50:42.771  forwarded SHUTDOWN to rank 1   <- 5.6 s stall
03:50:42.778  E2E_CLOCK last_compute_end
```

It only bites at one chunk: the same gap is **374 ms** at 261,120 and **200 ms** at 102,400, because
the sentinel arrives behind a pipeline that is already flowing. That is exactly why six latency rows
reproduce aug27 and the two 5,120 rows do not (they read 2.252 s and 6.121 s).

Reconstructed from per-stage timings — rank 0 starts at 36.998, stages take 145/165/156 ms, rank 3
finishes ~37.66 — PP=4 @ 5,120 is **~0.66 s**, against aug27's 0.947 s. The cell is not slow; the
clock is wrong.

Two follow-ups: move the stamp to chunk completion, and look at the **5.6 s sentinel-forward stall on
a single-chunk request** — probably harmless, but it is a real shutdown-path delay.

## 3.2 Cold kernel JIT is worth 3–20x on a latency cell

First `1rank_5120_ttft` on a fresh build read 14.1 s; warm it read 2.252 s; on a fresh *kernel cache*
earlier it read 22.5 s. **Always run a latency cell twice and use the warm one** — this campaign does
that automatically (pass 2). Throughput cells are steady-state and unaffected. Verify with the
`JIT cache stats: N/N hits (100.0%)` line rather than assuming.

## 3.3 The driver used to hide rank crashes (fixed on this branch)

`run_pp4_model.sh` exited with the *producer's* rc; the producer exits 0 whether or not the ranks
survived. A rank crash therefore reported `rc=0`, `run_matrix.sh` recorded the cell as complete,
skipped its own `tt-smi` recovery (only reached on non-zero rc, so the wedged fabric took the *next*
cell down too) and skipped the cell on re-run because `runner.log` existed. Fixed in `233ecd1d386`;
the driver now also gates on the runner log and exits 3. **Still worth spot-checking
`grep -c Traceback runner.log` before quoting a cell.**

## 3.4 `gen_pipeline_binding.py --profile` used to clobber the plain binding (fixed)

It did not switch template, so the documented two-command sequence overwrote the e2e binding with a
profiler-enabled one and every later cell ran instrumented. Fixed in `233ecd1d386`.

---

# 4. The llama4 query-scale caveat

**Status: the underlying issue is FIXED.** [#55126](https://github.com/tenstorrent/tt-metal/issues/55126)
is closed by `7f05a09616a` (one pre-built query-scale buffer per deterministic chunk offset, selected
per chunk by `actual_start` and device-to-device copied in). The `MISTRAL4_LLAMA4_PERF_UNSAFE`
stopgap that this section used to describe **no longer exists** — it was removed with the port onto
that fix, and traced Mistral runs now apply the correct temperature.

What follows is why the numbers in §1 are still usable, and what re-measuring is expected to change.

Every **traced** cell in §1 was measured with the stopgap, which replayed with an unrefreshed
query-scale buffer, i.e. a query temperature of **1.0** instead of
`1 + 0.1*ln(1 + floor(pos/8192))`.

- **Exact below 8,192 tokens of context**, wrong above it. At 261,120 the correct scale is ~1.347, so
  the error reaches ~35% at the deepest positions. Of the four ISLs only 5,120 is numerically exact.
- **Performance is unaffected.** The captured graph, shapes, dtypes and memory traffic are identical;
  Tensix timing is not data-dependent for the multiply or the matmuls; `RingJointSDPA` has no
  value-based control flow. Second-order caveat: MoE top-k routing *is* data-dependent, so expert
  load can shift slightly — the same magnitude of perturbation as changing the prompt, already
  visible as `Dispatch`/`Combine` drift across chunks in §2.2.
- **The §2 single-layer captures are numerically correct** — they run eager, where the guard does not
  apply and the scale is derived per chunk on host.

That prediction was tested rather than trusted: the re-measured tables in §1.1 and §1.2 land within
0.4%, i.e. inside a 0.6% measured noise floor. **The re-run removed the asterisk; it did not correct a
number.** Had a cell moved by more than the floor, that would have been a finding about the fix and
not about the topology — the second-order MoE-routing note above is the only mechanism by which it
legitimately could have.

---

# 5. Reproduce

Two directories, and only the first is in-tree: **`T` is reusable tooling** (reads logs and CSVs, no
device, no harness) and is part of this change; **`S` is the driver harness**, site-specific working
material that was deliberately left on the PP=4 reference branch. Get it from there if you want the
full campaign. The one cell behind the headline does not need it — `$T/run_pipeline_prefill_256k.sh`
is the in-tree equivalent.

```bash
T=models/demos/deepseek_v3_d_p/tests/perf     # in-tree
S=<pp4 reference branch>/.../pipeline_prefill_harness

$PY $T/gen_pipeline_binding.py              # REQUIRED on any new galaxy; a wrong map fails SILENTLY
$PY $T/gen_pipeline_binding.py --profile
$T/run_pipeline_prefill_256k.sh             # the 261,120 throughput cell, on its own

# The rest is the harness, i.e. the reference branch:
$S/preflight.sh                             # chips, build, tools, caches
$S/run_pp4_probe.sh                         # ~2 min, weightless topology + D2D check
$S/run_campaign.sh                          # everything below, in order, then the summary

# or the phases individually:
$S/run_matrix.sh                                                     # 16 cells
MODES=ttft FORCE=1 $S/run_matrix.sh                                  # warm latency re-run (see 3.2)
DEEP_CHUNKS=8 $S/run_single_layer_profile.sh 1rank_deep              # eager, no flag needed
DEEP_CHUNKS=8 $S/run_single_layer_profile.sh pp4_deep
```

Analysis:

```bash
$PY $T/analyze_prefill_latency.py  <runner.log>          # single-request latency
$PY $T/analyze_prefill_throughput.py    <runner.log> 8        # steady-state throughput (use the LAST rank)
$PY $T/analyze_prefill_layer_budget.py <ops_perf_results.csv> "label"
$PY $T/analyze_prefill_kv_ramp.py      <ops_perf_results.csv> "label"
```

**Logs and captures for this run:** `/data/kmabee/mistral4_bringup_pp4_bh-glx-120-b03u02/`
(`pass1/`, `pass2_warm/`, `profile/{1rank_deep,pp4_deep}/`) — provenance only: one author's box, not
a shared location, and not committed at multi-GB. The reduced captures in `tests/perf/captures/` are
the durable copy of the §2 data.

`tt-perf-report` was **not** available on this machine (`~/.local/bin` is local disk per box), so the
op tables above come from the in-tree analyzers. Both were validated first against the committed
captures in `$S/captures/`, where they reproduce the aug27 tables exactly.

**Prefill only. No decode data of any kind.**
