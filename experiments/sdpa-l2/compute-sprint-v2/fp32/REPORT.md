# C/D compute-only sprint v2

## Recommendation

Keep the frozen v1 winners. The new whole-query identity branch is exact in
the tested cases but its approximately 2% repeated-KV benefit does not carry
over to ordinary distinct inputs. Its extra code and C regressions are not
justified. Source-A-only reconfiguration and whole-half zero-flag clearing
are neutral and should not be integrated. The corrected internal QK2x2
probe is exact but slower. **No new C/D kernel is retained from v2.**

This is an isolated investigation, not a production modification. All inputs,
CB formats, numeric recipe flags, Q256/K512/D128, one K/V input slot, reader,
and writer are unchanged. `baseline` means the **v1 winner**, not original
main or the original canonical C/D kernel. D retains HiFi4/HiFi4, full FP32
L1 subtraction, unbiased cubic exp, FP32 state, and denominator phases0+2.
C retains HiFi4/HiFi2, cheaper subtraction, biased cubic exp, FP32 state,
and its original LoFi denominator.

Reservation223862, bh-lb-08, logicaldevice0 grid12x10, Blackhole,
FW19.13.1/KMD2.9.0. All device jobs serialize through the root-owned v1
`run_locked.sh`. Profiles report1350MHz. No changes to the frozen v1 tree.

## Screening

Unprofiled repeated resident KV, q_repeats8, k_chunks128, five warmups and
five trace replays. These are preliminary per-process medians, not sustained
ABBA winner claims. Every listed successful screen is raw BF16 uint16-equal
and trace-equal to its paired baseline.

| Candidate | D baseline→candidate ms | C baseline→candidate ms | Decision |
|---|---:|---:|---|
| Blocked state pack |84.055→84.107|61.194→61.187|Neutral|
| Whole-Q identity scan only |84.055→84.399|61.194→60.932|No standalone win|
| Scan + blocked state pack |84.055→84.441|61.194→60.927|No standalone win|
| Whole-Q identity batching |84.079→82.739|61.217→59.964|Resident-only gain; reject below|
| Unary source-A-only reconfig |84.088→83.995|61.230→61.413|Neutral/slower|
| Whole-half ZEROACC flags |84.089→84.104|61.218→61.199|Neutral|
| QK2x2 / unchanged PV1x4 |84.087→93.406|61.228→63.713|11.1%/4.1% slower|

Raw evidence: `screen-v1.json`, `batch-screen-v1.json`,
`srca-screen-v1.json`, `zeroacc-screen-v2.json`, their provenance and logs.
The first zeroacc smoke also passed, but its preprocessor fallback lacked
braces on non-MATH RISCs; the corrected screen restores the unpack MOP
unconditionally and is the relevant implementation. Both preserve the
Blackhole post-unpack zero-flag workaround rather than deleting it.

The QK2x2 probe changes only internal destination microtiles, not Q/K chunks
or input buffering. Four-K-tile split-drain PV grouping stays unchanged.
D needs two-score exp and separate full-L1 subtraction for both query rows;
C needs rowwise1x4 drain calls to avoid processing eight tiles in a four-tile
FP32 destination half. The first C adaptation violated that capacity and
failed the bitwise gate (59% L2); it is explicitly rejected. The device
closed cleanly, the root inspected/cleared the dirty guard without reset,
and the corrected version passed short odd-K/multi-Q and the full screen.
`qk22-smoke-v1.log` preserves the failure; `qk22-smoke-v2.json` and
`qk22-screen-v2.json` preserve corrected equality. No broad stress suite
was spent on the corrected slower candidate. Naive PV2x2 was not launched:
it would change row1's reduction from full-K DST into split partial products.

## Whole-query identity branch: distinct-data qualification

The guarded branch preserves row0's original four partial PV products and
L1 summation order. On bit-identical maxima across all256 query rows, it
computes rows1–7 PV contiguously, then performs exactly the original FP32
old-state L1 additions in four-tile batches. It does not preload old state
into FPU DST or alter reduction order. Changed-max chunks take the original
row-by-row path. Fixed-shape/noncausal/no-mask/no-ring guards are explicit.

Both C and D pass all eight distinct-KV cases against v1 raw output bits:
normal256K; scaled-QK, outliers, common-Q/K/V, constant-V, uniform at32K.
Each passes an independent trace replay. Seed1236, Q256.
`identity-qual-{C,D}-v1.json` contains hashes and L2/PCC. Known numerical
errors are unchanged, including D common-K0.7546%, C common-Q1.1681% and
common-K1.1130%; this is not a new universal0.5% claim.

Distinct multi-Q checks use Q2048, two cores, four different Q chunks/core,
unchanged production-shaped reader/writer, seed1237. All normal,
forced-changing-max, outlier, and common-K cases pass raw bits and trace.
Two ABBA rounds each have five warmups and eight timed replays at K8192;
the long K262144 round uses five timed replays. The changing-max case sets
Q feature0 to8 and raises K feature0 by4 each K chunk, forcing max updates.

| Distinct-data case | D baseline→candidate ms | C baseline→candidate ms |
|---|---:|---:|
| Q2048/K8192 normal |5.7601→5.7455|4.3170→4.3407|
| Q2048/K8192 changing max |5.7549→5.7285|4.2980→4.3265|
| Q2048/K262144 normal |171.9608→171.6306|126.2082→126.8716|
| Q2048/K262144 changing max |181.6411→180.8259|134.9238→135.7257|

Numbers are averages of the two per-round medians. These are **two-core
distinct-data device workloads with DM**, not no-DM TFLOP measurements.
At long normal context D improves only0.19%; C slows0.53%. Raw evidence:
`identity-dm-{C,D}-v1.json`, `identity-long-{C,D}-v1.json`.

Host double-QK with the BF16 online-max recurrence estimates whole-Q identity
on33.66% of normal256K chunks, versus80.87% for individual32-row groups.
Outliers44.42%/80.97%; common-Q97.26%/97.70%; common-K40.51%/83.78%.
This is **not an exact hardware guard counter**: device FPU rounding can
alter ties. `guard-frequency-v1.json` includes chunk-by-chunk estimates.
Repeated resident K has almost100% identity after its first chunk, explaining
why its upper-bound speedup is misleading for this branch.

## Sampled phase evidence

Original SUM zones did not appear in profiler output (`phases-v1/`), so no
phase attribution is inferred from their absence. A subsequent profile adds
six timestamped issue markers on one middle K chunk (index64), independently
on all three compute RISCs. All markers survived and output bits matched.
`timeline-v1-analysis.json` is reproduced by `timeline_analyze.py` from the
raw `timeline-v1/.logs/profile_log_device.csv`.

MATH issue intervals, baseline→identity branch, cycles:

| Interval | D | C |
|---|---:|---:|
| QK with overlapped earlier-row exp |59866→59852|46312→46559|
| Last-row exp / first PV drain |7943→7805|5111→5112|
| Denominator |4933→4976|5153→5164|
| Remaining PV / recurrent state |38467→36341|24049→22108|
| Entire sampled chunk |111238→109005|80654→78974|

These are sampled RISC issue boundaries, not exclusive engine-retirement
costs or a utilization decomposition. The batching gain is localized to the
trailing PV/state phase; it leaves the larger QK/exp region unchanged.
Instrumented timing is not substituted for the unprofiled comparisons.

## What remains promising, and what this sprint ruled out

An exact optimization now needs to tackle QK/exp rather than just recurrent
state setup. That region occupies roughly54% of D and57% of C's sampled
chunk. A deliberately co-designed unpack/MATH/PACK schedule that keeps
producer-consumer ownership explicit across the existing DST halves could
reduce format/setup and handoff bubbles, but no such speedup is demonstrated
here. Historical fixed-half QK/exp and MATH-owned pack experiments were also
slower; replacing them requires a concrete new dependency advantage, not
merely rearranging high-level calls.

Grouping contiguous unchanged32-row runs has a better expected guard hit
rate than whole-Q batching, but this sprint's best resident gain is only
about2% and changed-row SFPU/PV overlap must be preserved. It is lower
priority than the dominant QK/exp region. Reordering FP32 additions, using
different exp/precision, changing Q/K chunks or input buffering would violate
this sprint's contract and was not used to manufacture a gain.

The investigation therefore provides negative evidence against several
plausible small changes, not a proof that80% useful FLOP utilization is
impossible. V1 remains the measured exact improvement; v2 has no accepted
additional improvement and no production patch to merge.

## Reproduction

Frozen numerical baseline source SHA256:

- v1 FP32 header: `fef42cc8a8bfb272e4bdd401902c51ca7c880fc2b4f84f25fc930da69e76462c`
- Canonical recipe adapter: `ec477bb574bd1503320802a400ccfde938eaf8e0cf054c6da39541fd2d779afe`
- Unchanged C refiner: `cf38a29ddb6df1c8e349301012b721ab25c8be47e01ecf1bc10e88b9770b3ee3`

Final rejected-candidate investigation header:
`506e3106647ba4d9d6856937e9bab266547feb228925837acdd53cf0811503da`.
Earlier runs retain their contemporaneous hashes in provenance; the timeline
profile and corrected QK2x2 screen do not have identical source hashes.
Successful C++ candidates were JIT-compiled and executed on Blackhole; no
production source or host library changed, and no full tt-metal rebuild was
required. There is no v2 retained patch or newly qualified winner.

From the remote repository, invoke scripts through
`bash experiments/sdpa-l2/compute-sprint-v1/run_locked.sh`:

```text
experiments/sdpa-l2/compute-sprint-v2/fp32/bench.py --label NEW --candidates identitybatch --q-repeats 8 --k-chunks 128 --warmup 5 --iters 5
experiments/sdpa-l2/compute-sprint-v2/fp32/paired.py --label NEW --variant D --candidate identitybatch --qualify
experiments/sdpa-l2/compute-sprint-v2/fp32/fullchip.py --label NEW --variant D --candidate identitybatch --q-length 2048 --k-length 262144 --cores 2 --distributions normal,changing_max --iters 5 --rounds 2
-m tracy -r --profiler-capture-perf-counters fpu -o NEW_PROFILE experiments/sdpa-l2/compute-sprint-v2/fp32/bench.py --label NEW --candidates timeline,identity_timeline --q-repeats 8 --k-chunks 128 --iters 0
```

Use fresh output labels. Every benchmark stores exact define/fidelity/source
provenance and output hashes. Rejected candidate guards are intentionally
kept isolated for investigation; none is a proposed production patch.
