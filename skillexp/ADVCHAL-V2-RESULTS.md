# Stage 02b re-run (`advchal-v2`) — live results

> **Start here instead if you want the account rather than the table:**
> **[`ADVCHAL-V2-READ-THIS.md`](ADVCHAL-V2-READ-THIS.md)** — what each of the 15 cells actually did, one test at
> a time, reconstructed from the session transcripts; what makes a model advisor-compatible; and whether the
> winners were simply worse to start with. This file is the headline table that feeds it.

**15 cells planned, 15 complete.** Generated from the published `skillexp/done/advchal-v2/**` tags, not hand-entered.

Stage: tt-metal `mvasiljevic/qb2/skillexp/challenger-skill-v2` @ `db00a44`. v1's nine `done/challenger/**` tags are untouched and remain the audit's evidence base.

## Results

**Multi-kind caveat.** For a cell with two layer kinds, `final.json`'s `incumbent_ms`/`final_ms` are **one kind's** numbers — gemma's `harness_scope` reads *"one sliding_attention decoder layer"* — so its `Δ per layer` is the sliding figure, not a cell-wide one. The other kind is measured against its own incumbent and the two are not comparable to each other; `Δ model` is the properly weighted figure.

The two columns that matter are on the right. **Δ by kind** splits the per-kind saving for multi-kind cells, listed **alphabetically** (same order as the per-kind table below); it is computed on the profile-window basis, which differs from the harness basis by up to ~10 % (one cell: 725 µs window vs 807 µs harness), so the ship decision rests on **Δ per layer**, which is harness-measured.

| cell | batch | v1 | state | outcome | iters | oracle | per layer µs | full model µs | vs band | Δ by kind | Δ per layer | Δ model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-12B | 32 | no_change | tagged | **improved** | 2 | real | 1241.5 → 1218.8 | 58520.1 → 57211.9 | 23.6× (±55.5) | -3.79 % / -1.90 % | **-1.831 %** | **-2.236 %** |
| phi-3.5 exp17 | 32 | no_change | tagged | **no_change** | 1 | real | 1100.9 → 1100.9 | 32515.7 → 32515.7 | 0.0× (±34.9) | +0.00 % | **+0.000 %** | **+0.000 %** |
| llama-3.1-8B | 32 | no_change | tagged | **contribution_zero** | 3 | real | 665.0 → 665.0 | 20747.3 → 20747.3 | 0.0× (±22.3) | +0.00 % | **+0.000 %** | **+0.000 %** |
| llama-3.2-1B | 32 | no_change | tagged | **no_change** | 3 | real | 373.1 → 373.1 | 5672.6 → 5672.6 | 0.0× (±2.3) | +0.00 % | **+0.000 %** | **+0.000 %** |
| phi arm FN | 32 | -7.40 % | tagged | **improved** | 1 | real | 807.2 → 767.5 | 23205.6 → 21938.1 | 37.2× (±34.0) | -5.46 % | **-4.907 %** | **-5.462 %** |
| phi arm B | 32 | -5.20 % | tagged | **improved** | 2 | real | 788.6 → 748.5 | 22392.6 → 21107.7 | 56.3× (±22.8) | -5.74 % | **-5.092 %** | **-5.738 %** |
| phi arm onA | 32 | -6.01 % | tagged | **shipped** | 1 | real | 657.0 → 607.2 | 18210.9 → 16616.7 | 115.9× (±13.8) | -8.75 % | **-7.583 %** | **-8.754 %** |
| qwen fuse-noadvise | 32 | none recorded | tagged | **improved** | 3 | real | 1208.3 → 1181.1 | 935214.5 → 934780.5 | 0.7× (±618.5) | -2.63 % / +0.00 % | **-2.245 %** | **-0.046 %** |
| qwen nofuse-noadvise | 32 | n/a new | tagged | **no_change** | 3 | real | 1449.4 → 1449.4 | 780263.2 → 780263.2 | 0.0× (±290.1) | +0.00 % / +0.00 % | **+0.000 %** | **+0.000 %** |
| gemma-4-26B | 1 | no_change | **complete**, untagged | **SHIPPED** | 1 | real | 1259.9 → 1254.0 | 36224.1 → 36076.2 | 4.0× (±36.5) | +0.00 % / -0.49 % | **-0.469 %** | **-0.408 %** |
| north-mini FN | 1 | n/a new | **complete**, untagged | **improved** | 1 | real | 27634.9 → 25075.7 | 24949.2 → 22397.9 | 34.3× (±74.3) | +0.00 % / -6.74 % / -11.41 % | **-9.261 %** | **-10.226 %** |
| north-mini B | 1 | n/a new | tagged | **no_change** | 3 | real | 613.8 → 613.8 | 28207.3 → 28207.3 | 0.0× (±49.0) | +0.00 % / +0.00 % / +0.00 % | **+0.000 %** | **+0.000 %** |
| gemma-4-26B onA | 1 | n/a new | tagged | **improved** | 2 | real | 1824.0 → 1587.3 | 54633.6 → 47528.2 | 219.4× (±32.4) | -11.98 % / -13.23 % | **-12.980 %** | **-13.006 %** |
| north-mini onA | 1 | n/a new | tagged | **no_change** | 1 | real | 291.8 → 291.8 | 39940.8 → 39940.8 | 0.0× (±534.9) | +0.00 % / +0.00 % / +0.00 % | **+0.000 %** | **+0.000 %** |
| gemma-4-26B FN † | 1 | not recorded | tagged | **improved** | 1 | real | 1341.2 → 1318.4 | 38887.6 → 38095.8 | 9.8× (±80.9) | -2.91 % / -1.69 % | **-1.693 %** | **-2.036 %** |

† **`gemma-4-26B FN` (`fuse-noadvise`) is transcript-derived, not generator-derived.** Its `done` tag
(`skillexp/done/advchal-v2/fuse-noadvise/google_gemma_4_26b_a4b_it`) exists, but `advchal-v2-data.json` was last
generated before that tag landed, so the 14 rows above it come from the tags and this row comes from the cell's
own session log. Re-running the generator will fold it in and supersede this note. Everything else about the cell
— shipped `advisor_concat_projection`, the 88-core norm regression that kept it default-off, the GQA hard wall —
is in [`ADVCHAL-V2-READ-THIS.md`](ADVCHAL-V2-READ-THIS.md) §3.6.

**Complete is not the same as tagged.** Two cells finished their optimization work and passed the gate, but
carry no `done` tag. Their measurements are as real as any other cell's -- same harness, same non-overlap rule,
real-weight oracle -- and they are counted as complete here. What they lack is a publication tag, for reasons
that have nothing to do with the measurement:

- **gemma-4-26B** — optimization complete, gate PASSED; tag withheld by the freshness guard (the 4 identical files are the deterministic advisor output)
- **north-mini FN** — optimization complete, gate PASSED; tag withheld because .agents diverged (reconcile.py extended mid-run)

Still to run: `gemma-4-26B FN`

## Layer ms is per kind, and a single value inverts the comparison

`final.json` carries one `incumbent_ms`/`final_ms` pair — the kind the ship decision was made on. Quoting only
that hides the rest, and on qwen it inverts the reading: the two cells' full-attention layers say `nofuse` is
20 % *worse*, while the model column says 17 % *better*. Both are right; the linear-attention row reconciles them.

| cell | kind | harness median (ms) | note |
|---|---|---|---|
| qwen `fuse-noadvise` | `full_attention` * | 1.208257 -> 1.181132 | decision pair |
| qwen `fuse-noadvise` | `linear_attention` | 19.140244 | before only; 48 of 64 layers, never reconciled in this cell |
| qwen `nofuse-noadvise` | `full_attention` * | 1.449416 -> 1.449416 | decision pair |
| qwen `nofuse-noadvise` | `linear_attention` | 15.852625 | before only; **17 % faster than the fuse arm**, and 97 % of model time |
| gemma-4-26B | `sliding_attention` * | 1.259869 -> 1.253954 | decision pair |
| gemma-4-26B | `full_attention` | 1.261661 | before only |
| gemma-4-12B | `sliding_attention` * | 1.241523 -> 1.218789 | decision pair |
| gemma-4-12B | `full_attention` | **not measured** | 8 of 48 layers never put on the harness; its -3.79 % is profile-basis only |

`*` = the kind the ship decision rested on. Only that kind has an "after" at harness precision; the others have
a measured *before* and an after at the profile basis, which is a few percent off.

**Also worth checking before 0.749 us is treated as qwen `nofuse` full-attention noise:** that cell wrote three
full-attention measurements -- `incumbent.json` 1.449416, `incumbent_profile.json` 1.457902,
`incumbent_profile_trace.json` 1.452329 -- spanning 8.5 us, about 11x the floor the resolvability test uses.

## Per-cell disposition

Why each cell came out as it did -- already shipped, tested and refuted, or out of reach -- is in
[`ADVCHAL-V2-PER-CELL.md`](ADVCHAL-V2-PER-CELL.md), generated from the same artifacts.

## Per layer kind — the feasibility that decides everything

| cell | kind | layers | floor µs | ceiling µs | ceiling/floor | verdict | warm-up suspect | degraded |
|---|---|---|---|---|---|---|---|---|
| gemma-4-12B | `full_attention` | 8 | 3.381 | 25.39 | **7.51×** | **measurable** | no | False |
| gemma-4-12B | `sliding_attention` | 40 | 0.712 | 15.782 | **22.17×** | **measurable** | no | False |
| phi-3.5 exp17 | `dense` | 32 | 1.092 | 83.551 | **76.51×** | **measurable** | no | False |
| llama-3.1-8B | `dense` | 32 | 0.697 | 4.394 | **6.3×** | **measurable** | no | False |
| llama-3.2-1B | `dense` | 16 | 0.146 | 2.822 | **19.33×** | **measurable** | no | False |
| phi arm FN | `dense` | 32 | 1.064 | 71.637 | **67.33×** | **measurable** | no | False |
| phi arm B | `dense` | 32 | 0.713 | 70.732 | **99.2×** | **measurable** | no | False |
| phi arm onA | `dense` | 32 | 0.43 | 70.381 | **163.68×** | **measurable** | no | False |
| qwen fuse-noadvise | `full_attention` | 16 | 1.609 | 34.282 | **21.31×** | **measurable** | yes | False |
| qwen nofuse-noadvise | `full_attention` | 16 | 0.749 | 33.698 | **44.99×** | **measurable** | no | False |
| qwen nofuse-noadvise | `linear_attention` | 48 | 5.795 | 0 | **0.0×** | **not_measurable** | yes | False |
| gemma-4-26B | `full_attention` | 5 | 0.852 | 6.373 | **7.48×** | **measurable** | no | False |
| gemma-4-26B | `sliding_attention` | 25 | 1.291 | 3.832 | **2.97×** | **measurable** | no | False |
| north-mini FN | `dense_full_attention` | 1 | 0.849 | 3.476 | **4.09×** | **measurable** | no | False |
| north-mini FN | `full_attention_moe` | 12 | 1.206 | 0 | **0.0×** | **not_measurable** | no | False |
| north-mini FN | `sliding_attention_moe` | 36 | 1.638 | 1.148 | **0.7×** | **not_measurable** | no | False |
| north-mini B | `dense_full_forced_rope` | 1 | 0.168 | 5.929 | **35.29×** | **measurable** | no | False |
| north-mini B | `full_no_rope_moe` | 12 | 1.263 | 0.563 | **0.45×** | **not_measurable** | yes | False |
| north-mini B | `sliding_rope_moe` | 36 | 0.936 | 1.688 | **1.8×** | **aggregate_only** | no | False |
| gemma-4-26B onA | `full_attention` | 5 | 3.543 | 0 | **0.0×** | **not_measurable** | no | False |
| gemma-4-26B onA | `sliding_attention` | 25 | 0.587 | 0 | **0.0×** | **not_measurable** | no | False |
| north-mini onA | `dense_full_attention` | 1 | 1.841 | 1.709 | **0.93×** | **not_measurable** | no | False |
| north-mini onA | `full_attention_sparse_moe` | 12 | 0.847 | 0.562 | **0.66×** | **not_measurable** | no | False |
| north-mini onA | `sliding_attention_sparse_moe` | 36 | 14.526 | 1.706 | **0.12×** | **not_measurable** | no | False |

## What the three numbers mean

**floor** (`feasibility.noise_floor_us`) — the harness's run-to-run spread: `max − min` of the incumbent's
five timed repeats, each the mean of ≥50 trace replays. It is the smallest effect the ship rule can resolve,
because that rule is non-overlap: an effect below the spread leaves candidate and incumbent ranges
overlapping and cannot be called, however good the advice.

**ceiling / verdict** — the ceiling is the total µs of shipped conversions the advisor does *not* place: the
most this stage could ever attribute to the advisor on that kind. The verdict compares it to the floor.
`measurable` = some chain clears the floor alone. `aggregate_only` = the total clears it but no single chain
does, so chains must be applied together or each returns zero regardless of the advice. `not_measurable` =
even the total is below the floor, so report zero *with the arithmetic* and do not screen.

**vs band** — the model-level effect over its own uncertainty, `|after − before| / band_us`, where
`band_us = Σ over kinds (floor × layers_of_kind)`. Scaling a per-layer measurement to the whole model scales
its error identically, so a model figure quoted without this band reads far more precise than the measurement
supports. Above ~3× is a real result; 0.0× with a `measurable` verdict is a genuine zero.

## What is planned

**The nine configured cells** (a like-for-like re-run of v1's set, so the two are directly comparable):
**14 complete** — `gemma-4-12B`, `phi-3.5 exp17`, `llama-3.1-8B`, `llama-3.2-1B`, `phi arm FN`, `phi arm B`, `phi arm onA`, `qwen fuse-noadvise`, `qwen nofuse-noadvise`, `gemma-4-26B`, `north-mini FN`, `north-mini B`, `gemma-4-26B onA`, `north-mini onA`.  **1 to go** — `gemma-4-26B FN`.

**Three further tests considered, none configured** — decide after the nine land:

| | test | why | availability |
|---|---|---|---|
| A | `nofuse-noadvise-onA/google_gemma_4_26b_a4b_it` | the only cheap bound on the **machine effect**: same model, a second incumbent from machine A against the machine-B one in cell 9 | ref exists, unused |
| B | `nofuse-noadvise-onA/coherelabs_north_mini_code_1_0` | a second model with an independent incumbent | ref exists, but it is MoE (13 `sparse_matmul` / 39 `num_experts` in its decoder) so the tracer reaches little of it |
| C | **multi-layer capture** — `reconcile.py --layers-in-window 2` | the only test of the py-to-IR limitation: with one layer *both* boundaries are pinned to DRAM by the capture; with two, the interior boundary becomes a real choice. One question — does the advisor keep it in L1? | flag implemented, never run. Start on a kind that spills 0-1 times |

**Two planned tests are not possible as written.** An unused `nofuse-noadvise/qwen` incumbent does not exist on
the remote, so cell 8 is the only qwen and has no cross-check. "north-mini x3" needs three independent
incumbents; only one ref exists.

**Out of scope by design:** the `fuse-advise` / `nofuse-advise` arms (the 2x2 matrix), which conflate two
factors. Unresolved: gemma-4-26B runs at batch 1, the only cell not at 32, so it is not tile-comparable with
the rest — trying it at 32 is worth one attempt after the queue, not a mid-run change.

## Does iterating actually help? Yes, and it is measurable

`iters` counts **optimization rounds** — rank, measure, and if something won, re-profile and re-rank against the
new graph — not measurement repeats. Every measurement is 10 untimed warm-ups + 5 timed blocks of 50 replays.

gemma-4-12B, sliding_attention, against an incumbent of 1.241523 ms:

| what | measured | vs incumbent |
|---|---|---|
| best **single** chain (`sliding_attention:b7`) | 1.228846 ms | -1.021 % |
| next best singles (`:1`, `:b6`, `:b22`, `:5`, `:9`) | 1.2322-1.2352 ms | -0.51 to -0.75 % |
| two chains **rejected** (`:7`, `:6`) | 1.241267 ms | no better than incumbent |
| four chains `below_threshold` | not measured | attributable value 0.0 |
| **grouped set shipped** (`sliding Q+K+V+MLP`) | **1.218789 ms** | **-1.831 %** |

**Combining nearly doubled the gain over the best single chain** — 0.81 of the 1.83 points, 44 % of the win,
came from applying chains together rather than from any one of them. That is exactly what v1 lacked: it
screened chains one at a time under an `aggregate_only` verdict, which returns zero regardless of how good the
advice is. `full_attention` shows the same direction (best single 1.319763 ms, grouped 1.306017 ms).

The converse also holds: **llama-3.1-8B ran three rounds and gained nothing.** Its `iterations[]` triggers read
*"initial reconciliation rank 1 / 2 / 3"*, each noting `no prior candidate was kept, so the graph did not
change` — it worked down its ranked list, every chain was rejected or below threshold, and it correctly
declined to re-profile. Iterating pays when there is something to combine and costs only time when there is not.

**One number I cannot yet reconcile:** `model_estimate.per_kind` puts full_attention's after at 1267.1 µs/layer
while the grouped full measurement was 1306.0 µs. A confirmation run exists
(`measurements/confirm_final_full.json`), so the model figure may derive from the shipped config's own profile
rather than the screening measurement. Flagged rather than asserted — it wants checking before this cell's
model-level number is quoted as final.

## Per-op detail

Op-by-op across the cells of each model -- what was tried, kept, rejected as slower, never measured, and what
could have been tried and was not -- is in [`ADVCHAL-V2-PER-OP.md`](ADVCHAL-V2-PER-OP.md). Two things from it
belong here:

**A reporting defect in `reconcile.py`.** An `agrees_with_shipped` row can carry an advised core count that
differs from the shipped one, because the agreement test accepts a **DRAM-sharded program-config family match**
even when the grids differ. So `12 -> 99, agrees` means *both are DS*, not that 87 cores were left unused. It
needs an explicit `agreed_on: grid | ds_family` field; without it the data invites exactly the misreading I made.

**The 1-core RMSNorm is the highest-yield op class in the corpus.** Three cells met one, all three measured a
win, two shipped it. north-mini FN swept 22/32/64 and took 32 (**-10.23 %**); gemma-4-26B onA took 88
(**-12.98 %**); phi arm FN swept **11/12/24** -- every grid faster than its control -- combined the 11-core norm
with its RoPE win for **0.8072 -> 0.7003 ms (-13.24 %)**, and then **discarded it on the correctness oracle**,
shipping RoPE alone for -4.91 %.

> **Corrected 2026-08-03.** This paragraph previously said phi FN "screened only the advised 11, measured it
> slower, and abandoned the op ... the bounded-sweep failure the skill explicitly warns against." The transcript
> shows otherwise: `for cores in 11 12 24`, medians 0.7459 / 0.7490 / 0.7485 ms against a 0.8072 ms control. The
> sweep was compliant. What rejected the win was a **differential** real-weight oracle with the bar set at
> **0.999999** -- the only such bar in the corpus; every other cell used 0.995 or a recorded model value, phi A
> passed its own differential oracle at 0.9999987790, and phi FN's shipped real-weight test passes at PCC
> 0.998902. Full account in [`ADVCHAL-V2-READ-THIS.md`](ADVCHAL-V2-READ-THIS.md) SS3.3 and
> [`ADVCHAL-V2-ORACLES.md`](ADVCHAL-V2-ORACLES.md).

So the corpus's largest measured single win was found and then lost to an **unspecified oracle contract**, which
is a stage defect rather than a screening failure.

## Attribution needs two channels, not one

The stage prices advisor contribution as `us_advisor_drops` -- **conversions the advice does not place**. That
misses the highest-paying class of win entirely, because a **re-grid of an op inside a chain removes no
conversion** and is therefore worth exactly zero to that metric.

Proof from the corpus: north-mini `sliding_attention_moe` realised **59.7 us/layer against a boundary ceiling of
1.148 us**, and `full_attention_moe` realised 33.5 us against a ceiling of **0**. Its win was a 1-core RMSNorm
re-gridded to 32 cores -- 26.1 us to 5.6 us, a **4.65x speedup on one op across 48 of 49 layers**. The advisor
supplied the op and the direction and advised 22; the agent swept 22/32/64 and 32 won, with *"advised 22 and
above-advice 64 were slower"* recorded in the cell's own iteration log. That is the documented division of
labour, and the metric scored it at nothing.

It also under-states gemma-4-12B, the corpus headline: 49.9 and 22.7 us/layer realised against ceilings of 25.4
and 15.8. So part of that win was a re-grid too, unattributed.

**Channel 2 must be split by direction.** The advisor's re-grid advice is mostly *downward* -- fewer cores than
shipped -- which the v1 audit measured at 8 of 8 non-matmul ops and which phi-3.5 exp17 refuted 16 times in a
single cell. Only **UP**, and especially ops **starved on <=2 cores**, is a candidate. The per-cell disposition
document carries the split and the finding class.
## A zero ceiling has two opposite meanings

`us_advisor_drops == 0` means one of two things and the verdict cannot tell them apart. qwen `linear_attention`
has **4886.983 us of boundary time with 3988.953 us of it agreed** across 13 edges (almost all `retilize`) --
the advisor endorses every conversion present, so nothing is attributable and the cost belongs to `$optimize`.
north-mini `full_attention_moe` has **0 agreed**, with 11 boundaries `undetermined` and 3 `unresolved`: not one
of its 14 boundary ops was comparable, because at 68 % untraced the advised-op adjacency keeps breaking. The
first zero is a finding; the second is a blind spot -- and that kind went on to realise 33.5 us/layer from a
re-grid. Report the zero with its reason attached.

## Findings across the seven completed cells

**1. The harness, not the advisor, produced v1's zeros.** gemma-4-12B: same incumbent, same advisor, fixed
harness — its sliding floor fell 18.284 → 0.712 µs (26× tighter), its verdict went `aggregate_only` (1.59×) →
`measurable` (22.2×), and the cell ships **−1.83 % per layer / −2.24 % per model at 23.6× its band**, against an
audit prediction of −1.27 % to −1.64 %. Both llamas were formally unmeasurable in v1 (ceilings at 0.65× and
0.67× of their own floors); their floors are now 0.697 and 0.146 µs, verdicts `measurable` at 6.3× and 19.3×,
three optimization rounds each — and they still return zero. Those are *demonstrated* zeros rather than
artifacts, which is why llama-8B labels its own outcome `contribution_zero`.

**2. All three phi wins are preserved, and v1's error bars were wide in both directions.**

| arm | v1 | run 2 (layer) | run 2 (model) | vs band |
|---|---|---|---|---|
| `fuse-noadvise` | −7.40 % | −4.907 % | −5.462 % | 37.2× |
| `nofuse-noadvise` | −5.20 % | −5.092 % | −5.738 % | 56.3× |
| `nofuse-noadvise-onA` | −6.01 % | **−7.583 %** | **−8.754 %** | **115.9×** |

One lower, one level, one higher. So the earlier reading — that v1 was systematically optimistic — does **not**
hold; what v1 had was wide, unrecheckable uncertainty. Its winners recorded no `repeats_ms`, no
`winning_repeats_ms` and no `confirmed_fresh_process`, and one arm's op profile was 19.1× un-windowed. Run 2's
figures sit at 37–116× their own bands with the non-overlap decision reproducible from recorded repeats.
`onA` at 115.9× is the most solidly established result in the corpus.

**3. Combining chains is where most of a win comes from** — 44 % of gemma's, see the iteration section above.
This is the single behaviour v1 lacked, and it is why an `aggregate_only` verdict must never be screened
chain-by-chain.

**4. A large attributable quantity is not a promise.** phi-3.5 exp17 had a ceiling of **83.551 µs at 76.5× its
floor** — the biggest attributable quantity of any completed cell — a harness precise enough to resolve it, and
returned **zero**. That is the strongest negative result here, and it is only interpretable *because* the floor
was tight: v1 could not have distinguished it from an unmeasurable cell.

## Caveats to render alongside

- **qwen3.6-27B** is the cell the audit called invalid: its run-1 `incumbent_ms` of 937 ms was arithmetic
  (`16 × 1.2077 + 48 × 19.13`) over per-layer medians, not a measurement. Run 2 gives it a real harness, but
  there is no second qwen incumbent on the remote to cross-check it against.
- **gemma-4-26B runs at batch 1**, the only cell not at 32, so its numbers are not comparable to the others on
  tile-quantisation grounds.
- **Three tests from the plan are not configured**: an unused `nofuse-noadvise/qwen` incumbent (does not exist
  on the remote), north-mini ×3 (only one ref exists, and it is MoE so its experts are invisible to the
  tracer), and `onA/gemma-4-26B` (exists, unused — the only cheap bound on the machine effect).
- **A `not_measurable` verdict is a complete result, not a failure.** So is a correctness-gate rejection: some
  ops compute the wrong answer under particular shard specs, and a candidate that is faster but moves PCC is
  rejected on purpose and reported as a tt-metal bug.

