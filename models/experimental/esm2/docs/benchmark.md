# Benchmark — measured evidence for the ESM-2 TTNN port

SPDX-License-Identifier: MIT

A **porting benchmark**, not a biology-leaderboard claim. Gates answer: (1) is
the TT implementation numerically faithful to the FP32 oracle (logits +
final hidden), (2) does it preserve batch/padding/revisit semantics, (3) is it
faster relative to its own recorded baseline, with A100 timings as context.



## Frozen corpus (sha256 `369c475f…`)

| Sequences | Origin | Why |
|---|---|---|
| 14 Swiss-Prot proteins (68–1273 aa; spike truncated to 1024) | UniProtKB, CC-BY-4.0, deterministic accessions | Real, reviewed proteins spanning lengths/families; length spread exercises padding and rotary wrap cases. |
| syn-max1024 / syn-1023 / syn-single / syn-pair / syn-unknown-X | deterministic | Envelope boundary (1023/1024 residues), minimal sequences (1, 2), unusual residues (X/Z/B → unk). |

Deterministic masking: residue positions 5, 21, 37, … are replaced by
mask_token 32 by the evaluator; candidates see only already-masked ids.

## Stages and gates

- `smoke` 1 case; `bringup` 8; `full` 21 (all sequences + batch4/reverse/
  revisit/single-residue; 3 timed repeats).
- logits + hidden max-row NRMSE ≤0.04 vs FP32 oracle; masked-position argmax
  exact; behavior NRMSE ≤0.01; repeat determinism.
- Diagnostics: residues/second, first-call latency, memory snapshots.

## Measured results — current source (`tt/esm2/ttnn_backend.py` sha
`3f422c71…`, bf16 anchor, out_fp32 = {qkv, pv, ao, ffn2} — the measured measured trim; opt-in trace fast path; eager default unchanged)

Rows and tables below that cite the `2be97754…`/`de517a48…` source shas or
the 0.0292 sr-logits anchor belong to the earlier **all-8-out32 era** and are
kept for lineage; see the **anchor correction** note in the out32-trim
section for how to read the two eras.

### Correctness (independent evaluator, fixed suites)

| Stage | Receipt | Result |
|---|---|---|
| smoke (bf16, all-8-out32 source `2be97754`) | `152c03d7b3ab4eef9de3fb7bdd2ef512` | **PASS** — max NRMSE 0.0296, argmax exact, p50 90.0 ms (142 res, 1577 res/s). Not re-run after the numerics-only source changes that stayed inside the measured envelope (later source changes). |
| bringup (bf16, all-8 source) | `fdff371f08a245f9848339ea4b3bf8c7` | **FAIL 4/8** — max NRMSE 0.0433; 3 argmax flips (one shared near-tie row, see limits) + single-residue hidden 0.0433 > 0.04 |
| bringup re-gate (bf16, trace-integrated all-8 source `de517a48`) | `d2ae3c104c424e8c90fe9886498c1845` | **identical to the digit** — same 4 failures, max NRMSE 0.04325492926743312 bit-equal to the prior gate; per-case NRMSEs equal; long 209.5 ms, short 79.1 ms. The opt-in trace integration (shared `_graph_forward`, eager default, Lt≥1024 guard) introduced zero evaluator-path drift. |
| bringup re-gate (bf16, out32-trim source `3f422c71`) | `88c1636ec9084287bd588741b25cbb9f` | **FAIL 4/8 — arbiter KEEP (INTEGRATED_GO)** — max NRMSE **0.0424** (was 0.0433); same 3 intrinsic argmax flips; the hidden miss moved single-residue→long exactly as the sizing predicted (sr hidden 0.0306 now **passes**, long hidden 0.0424 > 0.04). Every currently-passing argmax row stayed exact; numerics matched the A/B sizing to ~5e-6. |

Bringup per-case, all-8 era (logits/hidden NRMSE): short 0.0074/0.0191 ✔,
single1 0.0042/0.0203 ✔ (94.8 ms, 2511 res/s), unknown-residues 0.0048/0.0108
✔ (84.5 ms), long 0.0159/0.0359 ✔ (209.5 ms, 4887 res/s), single0/batch/
reverse logits ≤0.0090 ✔ but argmax flips on the l=22 twin row;
single-residue 0.0291/0.0433 ✘ hidden.

Bringup per-case, current trim source : short
0.0099/0.0227 ✔, unknown-residues 0.0075/0.0122 ✔, single-residue
**0.0170/0.0306 ✔** (the former hidden miss, now passes), long 0.0212/**0.0424
✘ hidden** (the new miss); batch/single0/reverse logits 0.0085–0.0101 ✔ but
argmax flips on the same l=22 twin row (unchanged, intrinsic); single1 argmax
exact (timed row).

### Latency vs the recorded same-host baseline

Baseline = PyTorch FP32 CPU on the TT host, fixed suites, frozen artifact
`benchmarks/artifacts/baseline_pytorch_fp32_20260925T022350Z.json` (job
`503a1a88…`; oracle cross-check 2.75e-05/2.81e-04): single-short **323.9 ms**,
single-max (L=1026) **3269.7 ms**, batch2-max **6431.9 ms**.

| Case (bf16 device) | Device p50 | TT-host CPU FP32 | Speedup |
|---|---|---|---|
| long L=1026 (all-8-out32 source; gate measurement) | 209.5 ms @ 4887 res/s | 3269.7 ms @ 314 res/s | **15.6×** |
| long L=1026 (current trim source {qkv,pv,ao,ffn2}; matched in-process A/B, 1 warm + 5 timed, . | **181.5 ms** (−12.8% vs 208.1 ms same-process all-8; A/B spread 0.009 ms) | 3269.7 ms | **18.0×** |
| single1 L=240 | 94.8 ms @ 2511 res/s | 323.9 ms (short class) | ~3.4× |
| short L=144 | 79.1 ms | 323.9 ms | 4.1× |
| short (eager, in-process, L=4/Lt=32) | 53.1 ms | — | — |
| short (opt-in trace replay, L=4/Lt=32) | **15.4 ms** | 323.9 ms | ~21× |

Gate single-sample timings under the trim source token_ids_to_logits_with_sync protocol, 1 sample/case): short 81.9 ms,
single1 94.0, unknown-residues 65.7, single-residue 51.7; long was not sampled
that run (the timing case pick varies run-to-run). The long-L saving claim
rests on the matched A/B row above, not on cross-run gate samples.

Cost of the out_fp32 site set at the all-8 era (device A/B `1a238191…`): p50 168.5 → 208.6 ms at L=1026 (+24%), 51.7 → 54.5 ms at L=3
(+5%) — the numerics fix is not free; per-site attribution lives in
`tt/esm2/ttnn_backend.py`. The measured trim below buys back −12.8% of
the long-L cost at a measured, accepted numerics trade.

A100 context (recorded CUDA study, different host): FP32 p50 **31.9 ms** on
the max case; its own bf16 path flips 4/8 argmax vs its FP32. The TT device
bf16 long case (209.5 ms all-8 / 181.5 ms trimmed) is ~5.7–6.6× slower than
that A100 FP32 number — recorded honestly as context, not a gate.

### Latency structure (device profile, prior-source lineage `6a9b3cbb`,
.json`)

- ~63 ms fixed floor at L=4–240 from per-op dispatch (~825 dispatched ops ×
  ~50–75 µs), not compute or transfer; compute-bound only at L≈1026
  (33 layers ≈ 166 ms of a 170.5 ms unbroken p50, pre-out32 source).
- Not transfer-bound: warm H2D+D2H ≤ 8.03 MB/call (~3–6 ms); weights one-time
  1.30 GiB bf16 in 2.39 s at build.
- Bitwise-deterministic warm vs cold on all 5 profiled cases.
- Rejected by measurement: ttnn.slice-to-L before D2H (402 ms vs 3.0 ms padded
  pull, 130× worse at identical bytes, bit-identical outputs).

### Opt-in trace fast path — LANDED (hypothesis #1)

Production path in `tt/esm2/ttnn_backend.py`: `set_trace(True)` enables a
capture cache keyed by the full (ids-pattern, mask-pattern, Lt) bytes; eager
stays the default (the evaluator path never enables it); Lt ≥ 1024 declines
(persistent trace intermediates unsized there — sized and confirmed at
Lt=1056, see the long-L section below); one capture attempt per process with
disable-on-failure + eager fallback; ≤4 entries, released on eviction/close.

| Measurement (bringup short, B=1 L=4 Lt=32, bf16, 1 warm + 5 timed) | Value | Receipt |
|---|---|---|
| eager forward() p50 (fresh, integrated source) | 53.1 ms, bit-stable | `e7e373e1…` |
| trace capture, one-time | 74 ms | `e7e373e1…` |
| traced forward() p50 (end-to-end: host embedding + staging write + blocking replay + host read) | **15.4 ms** (15.34–15.45), bit-identical logits+hidden on **every** call | `e7e373e1…` |
| raw execute_trace p50 (device-only arm, sizing job) | 14.4 ms, bit-identical | `537c6992…` |
| speedup | **3.45× vs fresh eager**, **3.54× vs the recorded 54.5 ms bf16 baseline** (`1a238191`) | `e7e373e1…` |
| second pattern (single0, Lt=160) | capture 92 ms + replay, bit-identical vs its own eager reference; 2-entry cache | `e7e373e1…` |
| long (Lt=1056) | declined to eager (`trace_declined_Lt_ge_1024`), bit-stable — guard verified | `e7e373e1…` |

Artifacts: `benchmarks/artifacts/opt_h1_trace.json` (sizing, job
`537c6992a7174647915cdac3490e2ea6`), `benchmarks/artifacts/
opt_h1_trace_integrated.json` (integration verification, job
`e7e373e1c7bc4a8ea6480979423d49c9`, probe `tests/probe_opt_h1_trace_integrated.py`).

Interpretation: the warm short-L forward is host-dispatch-bound (~700 ops);
trace replay removes ~38 ms/forward of per-op dispatch and lands at the
device-execution floor (15.4 ≈ 14.4 ms + ~1 ms of honest end-to-end path
work: embedding gather, byte-verification of inputs, staging write into the
captured x0 device tensor, host reads). The staging write is real on this
runtime (`ttnn.copy` exists; 8 writes recorded) and every
replay stayed byte-identical. Known caveat, recorded: the runtime logs an
allocator warning when transient buffers are allocated with an active trace;
its own `execute_trace` allocation-safety tracker verified every replay, and
byte-identity held across all replays and cases.

### Sized-and-closed optimizations (multiple measurement rounds; one device job each,
artifacts persisted)

#### H1 follow-up: long-L trace replay — GO by rule, NOT integrated (immaterial 1.2%)

Probe `tests/probe_opt_h1_trace_long.py`; **job
`5f63c8f842044f4b9eb01ce6abf0293b`** (bf16, bringup long B=1 L=1026 Lt=1056,
32.8 s); artifact **`benchmarks/artifacts/opt_h1_trace_long.json`**. No
backend change — the probe drove `model._trace_path` directly and re-verified
the production Lt≥1024 decline guard in-probe.

| Measurement (Lt=1056, bf16) | Value |
|---|---|
| traced replay p50 vs matched in-process eager | **205.96 vs 208.40 ms → 1.012×** (1.017× vs the recorded 209.5 ms gate number) |
| why so small | at Lt=1056 device compute dominates; replay saves only ~2.4 ms host dispatch (vs 3.45× at Lt=32) |
| capture cost | one-shot 0.102 s (2nd pattern JIT-warm 0.098 s); first replay warm 517.7 ms; break-even ≈ **170 same-pattern replays** |
| bit-identity | logits+hidden fp32 bytes identical to eager on EVERY call (capture run, warm, 7 timed, pattern B ×2, A-after-B) |
| memory | no OOM with two Lt=1056 entries live; analytic ≈63 MiB pinned/entry (mask 42.5 + rotary 10.3 + x0 5.2 + hidden 5.2, bf16 MiB) + 85.1 MiB transient fp32 scores. LIMITATION: empirical DRAM view not obtained — `ttnn.get_memory_view` needs a `buffer_type` arg on this runtime; immaterial to the decision |
| per-Lt-class (vs per-pattern) cache keys | measured **NEGATIVE**: staging mask 37.6 ms + 4× rotary 1.7 ms ≈ +39 ms per new-pattern replay vs 2.44 ms saved (16×) → per-pattern keys stay |

Decision: rule conditions technically held (bit-identity, < eager, < baseline,
no OOM) but the benefit sits inside eager spread (208.25–210.27 ms) →
**keep the Lt≥1024 guard, do not integrate**. H1 follow-ups (long-L replay,
per-Lt-class keys) measured and closed.

#### H2: out_fp32 site trim {qkv, pv, ao, ffn2} — sized GO, INTEGRATED as the default

Probe `tests/probe_opt_h2_out32_trim.py`; sizing **job
`2ebec6a3af8a423a978370246328fecb`** (bf16, bringup inputs, 56.5 s; A/B
sandwich all8_A → 3 trims → all8_B in ONE process, anchors and spreads
recorded per config); artifact **`benchmarks/artifacts/opt_h2_out32_trim.json`**.
Integration certified by anchor job `ba3ae1e9…` (tiny short = sizing to all
6 digits) plus the ONE bringup re-gate **`88c1636ec9084287bd588741b25cbb9f`**
(FAIL 4/8; arbiter satisfied: failures = 4, every currently-passing argmax row
exact → KEEP).

- Trim ladder: `{ffn2,qkv,ao}` **NO-GO** (long hidden 0.0568, sr logits
  0.0658 > gates — pv is needed at long, decoder-out removal alone would hurt
  sr); `{ffn2,qkv,ao,pv}` **GO**; `+dense` numerically inert (identical to 16
  digits) → the simpler `_pv` set is the shipped default.
- Speed (matched A/B, 1 warm + 5 timed, forward() end-to-end): long eager p50
  **208.1 → 181.5 ms (−26.6 ms, −12.8%, vs 0.009 ms A/B spread)**; sr −2.2 ms;
  short −2.45 ms (inside the 2.23 ms spread — no claim); traced short
  14.8 → 14.7 ms (unchanged; replay is dispatch-bound).
- Numerics trade (recorded honestly): suite max 0.0433 → **0.0424**; the
  hidden miss swaps rows — sr hidden 0.0433 → **0.0306 (now passes)**, long
  hidden 0.0359 → **0.0424 (now misses)**; sr logits 0.0292 → **0.0170**;
  short/long-logits slightly worse. Expected bringup failures stay 4 (3
  intrinsic argmax + one hidden row, sr↔long swap) — confirmed by the
  re-gate to ~5e-6 per row.
- **Anchor correction (explicit):** the **0.0292** sr-logits anchor is the
  **pre-trim (all-8-out32) class**; the current source's sr-logits anchor is
  **0.0170** . Both numbers appear in this file; every
  row above is labeled with its source era. Later A/Bs must anchor against
  0.0170/0.0306 (sr) and 0.0212/0.0424 (long) on source `3f422c71…`.

#### H3: erf-exact gelu composite — NO-GO on every arm; closed WITH attribution

Probe `tests/probe_opt_h3_erf_gelu.py`; **job
`5e9a21d2f7a044afb03509b888c1ce1e`** (bf16, current trim source, 54.5 s);
artifact **`benchmarks/artifacts/opt_h3_erf_gelu.json`**. Gelu swapped
per-instance via a namespace proxy on `model.ttnn` (zero source edits; both
op sites: encoder FFN ×33 + MLM head ×1; trace-capture compatible).

- Op level (vs torch FP32 erf-gelu, bf16 grid n=1237): `ttnn.erf` ≡ `torch.erf`
  to **8.6e-08** NRMSE (max abs 2.4e-07); erf-exact gelu — composite
  `0.5x(1+erf(x/√2))` all-fp32 and fused32 cast→`ttnn.gelu`→cast — matches
  the FP32-erf policy target to **~1.1e-07** vs the default piecewise-CDF
  gelu **1.58e-04** (~1400× closer); composite ≡ fused32. The composite is
  kernel-correct.
- End-to-end A/B (arm A reproduced the current anchors to ≤6e-6):
  **B composite** — long hidden **0.0440** (worse; the unit's target was
  <0.04 from 0.0424), sr logits **0.0436 > 0.04 gate** (A: 0.0170), sr hidden
  0.0398 (A: 0.0306), short improves; cost +24.5 ms long / +13–20 ms short.
  **B2 fused32** (cost rider, same job) — numerically IDENTICAL to B at sr
  (16 digits; same kernel math), +9.2 ms long. **C** (composite + restore
  scores-fp32; conditional arm) — far worse (sr 0.0768/0.0729, long hidden
  0.0447). Argmax held in every arm; all forwards bit-stable; all traced
  replays bit-identical per arm. **All arms NO-GO.**
- **Compensating-error attribution (explicit, the valuable result):** making
  gelu numerically exact makes the END-TO-END suite WORSE. The default
  piecewise-CDF gelu deviations sit inside a **compensating error balance**
  with the other rounding sites — confirmed from both directions: dropping
  rounding sites (measured trim) IMPROVED sr, and adding exactness
  (measured erf-gelu) WORSENED it, including destructively so when combined
  with scores-fp32 (arm C). The long-hidden 0.0424 miss is NOT
  gelu-attributable; it stays implementation-floor / trim-trade class
  (multiple measurement rounds/10). Do not revisit erf-gelu without a changed hypothesis.

### Known limits (measured, documented — gates NOT widened)

1. **Argmax near-tie (batch/single0/reverse)**: one protein row at l=22 has
   twin logits margin 0.0280 (0.0030 rel); clean bf16 stream-only error at
   that row is 0.0961 (≈7× the margin) → flips under any bf16 policy; the
   A100 bf16 study flips 4/8 of its own cases. Intrinsic to bf16 argmax on
   near-ties, not a port defect (sim + device
   `e5b16949`, `fdff371f`, re-confirmed `d2ae3c10`, unchanged under the trim
   source `88c1636e`).
2. **Hidden-row implementation-floor miss (exactly one row per era)**: at the
   all-8-out32 source it was single-residue hidden 0.0433 > 0.04 — CPU floor
   study (`benchmarks/artifacts/sim_floor_study.json`, jobs `9be15994`,
   `f770a9c2`, `087c2b46`) predicts 0.039–0.042 for truncation-mode
   activation rounds + TT piecewise-CDF gelu (RNE-policy floor 0.019); every
   lever sized and rejected (`tests/probe_sim_floor4.py`, round-removal
   levers worsen it). Under the current trim source the miss sits at **long
   hidden 0.0424** with a BETTER suite max (0.0424 vs 0.0433) — the measured
   sr↔long trade of .
   Gelu attribution was tested directly and **rejected** (the erf-exact arm
   made every affected row worse; see H3 above).
3. **Trace path bounds (opt-in only)**: keyed to exact input patterns (no
   generalization across ids/mask/Lt), Lt < 1024 only (long-L replay sized
   immaterial at 1.012× and per-Lt-class keys measured negative — H1
   follow-up above), ≤4 cached traces, capture cost 74–102 ms amortizes from
   the 2nd repeat of a pattern. The evaluator/default path remains eager by
   construction.

## Optimization hypotheses — status after multiple measurement rounds (each closed on
measurement, not opinion)

1. ~~**Dispatch-floor cut at short L via trace capture**~~ — **LANDED**
   (trace fast-path section): 15.4 ms vs 54.5 ms recorded baseline (3.54×),
   bit-identical, gate re-bound to the integrated source (`d2ae3c10…`).
   Follow-ups now CLOSED: Lt≥1024 replay sized immaterial (1.012×, job
   `5f63c8f8…`, artifact `opt_h1_trace_long.json`); per-Lt-class keys
   measured negative (+39 ms/replay vs 2.44 ms saved).
2. ~~**out_fp32 site trim for speed**~~ — **LANDED** as the default
   {qkv, pv, ao, ffn2} (H2 section; sizing …`, artifact
   `opt_h2_out32_trim.json`, re-gate `88c1636e…`): long −12.8% p50 at suite
   max 0.0433→0.0424; the accepted sr↔long hidden trade is on record.
3. ~~**erf-exact gelu composite**~~ — **CLOSED NO-GO with attribution**
   (H3 section; .json`):
   kernel-exact yet end-to-end worse on every affected row — compensating
   error balance; gelu is not the long-hidden miss's cause.
4. ~~**Single-residue hidden levers**~~ — CLOSED NO-GO (measured CPU
   sizing; no lever ≥0.01 shave; round-removal levers worsen).

## Why not big biology benchmarks

PERM/ProteinGym mutation-effect or TAPE measure model quality across
thousands of variants; they are post-port validation, not porting gates.
Coverage beyond 1024 residues (ESM-2's own limit), structure heads and
fine-tuning are explicit out-of-scope extensions.
