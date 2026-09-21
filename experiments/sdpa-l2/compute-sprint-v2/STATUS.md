# Compute sprint v2 status

Hardware verified 2026-09-18 15:48 UTC: existing IRD223862 on bh-lb-08,
over six hours remaining. Fresh exact BF16 matmul smoke passed on device0,
12×10 grid. Global lock/dirty guard retained; no simultaneous device tests.

Baselines are v1's retained winners. Canonical and v1 sources are frozen.
All new work is isolated in this directory. [Protocol](PLAN.md).

## Initial hypotheses

- C/D: batch remaining numerator-state packs; amortize exact max-equality
  decisions; inspect in-place identity-state accumulation only if rounding
  order and CB lifecycle are preserved.
- B/E/G: reduce repeated copy/broadcast reconfiguration, overlap independent
  state groups, and improve denominator setup without changing full-column
  state semantics or SFPU macro lifetime.
- Independent review: correction retention across DST releases is unsafe
  without more work because release clears the destination half. This was
  flagged before testing, not treated as a free broadcast-elimination win.

## First screen

- E correction retention across tightly bounded DST lifetimes passed distinct
  multi-Q/changing-max checks, but paired resident timing is neutral:
  287.0174→286.9989 ms (0.00645%, overlapping sample ranges). Not a win.
- Independent transfer of the same correction-retention scheme to B passed
  four held-out distributions at Q2048/K1536 on two cores with trace replay.
  B timing is pending.
- C/D state block-packing was neutral. Whole-Q exact identity scanning alone
  was roughly 0.4% faster C / 0.4% slower D in screening, not a retained gain.
  A more substantial identity-only PV/state batching candidate is being
  investigated, keeping each row's complete-PV-then-state-add ordering.
- Parent [first smoke audit](audit-first-smoke.json) and
  [first screen audit](audit-first-screen.json) passed stored hashes/equality
  gates and throughput arithmetic. They are not additional device runs.

Read-only [telemetry](telemetry-first-round.json): 41 samples, 15:52:36–15:56:32
UTC, observed clocks 800 MHz idle / 1350 MHz active; max sampled ASIC
temperature52.8C. Sampling does not establish every timed replay's frequency.

## Deeper scheduling work

- B correction retention also screens tiny: 337.1159→336.7584 ms (0.1061%).
  Exact on four held-out multi-Q/odd-K distributions, but not a meaningful
  retained gain.
- Whole-identity PV/state batching screens D84.079→82.739 ms (~1.6%) and
  C61.217→59.964 ms (~2.0%), Q repeats8/K chunks128. Initial output/trace
  bits pass; these are preliminary resident results, not final wins or
  distinct-input speedups. SrcA-only reconfiguration separately looked neutral.
- Macro fusion for compensated arithmetic is constrained by four templates
  and explicit dependency latency; the obvious chunk-load/MAD fusion trades
  away another existing fusion, not an established instruction reduction.
- Independent B sampled phase capture found next-batch preparation already
  overlaps prior PACK, motivating cross-half SFPU/pack overlap. The marker
  spans are RISC issue times, not engine-retirement timings, and perturb the
  sampled batches substantially despite only0.578% whole-kernel overhead.
- Cross-half overlap now has an independent source-ordering review and is
  entering a bounded short-loop smoke. Required fences, explicit count2
  readiness, separate SFPU/PACK bank addressing and balanced canonical
  releases are retained. No performance result yet.

## Second screen

- Cross-half overlap passed E short and G/B distinct multi-Q/odd-K output
  and trace comparisons. It did **not** improve timing: release after8vectors
  is ~0.13% slower E / ~0.11% slower B; E split2/4 are neutral. No promotion.
- FP32 half-bank ZEROACC-flag coalescing passed bitwise tests but was neutral.
- Whole-Q identity batching passes eight distinct-KV distributions for both
  C/D. Small two-core real-dataflow timing shows only ~0.25–0.46% improvement
  D and ~0.5–0.7% regression C. Resident gains are not representative of that
  workload. Long distinct-input confirmation pending.
- Host double-QK/BF16-max recurrence estimates whole-Q identity at33.66% of
  256K normal chunks versus80.87% for32-row groups. This is a host surrogate,
  not measured device guard counts.
- Next B/E/G hypothesis: exact finite unchanged-max identity correction,
  preserving every MAD/round/store, existing correction-CB publication fence,
  and nonidentity fallback. Exhaustive finite-BF16 correction-zero device
  proof precedes attention tests. This is not a numerical relaxation.
- Next C/D hypothesis: internal QK2×2 microtiles with unchanged Q256/K512 and
  PV reduction grouping. Naive PV2×2 was rejected in source review because
  it would change one row's split-reduction arithmetic order.

## Correctness-gate pause

QK2×2's first C adapter attempted an eight-tile score-exp drain in a four-tile
FP32 destination half. Its output-bit assertion failed (12,263 differing
values); this candidate was not accepted. Device closed normally at16:18:17
UTC, no process remained, and root cleared the shared dirty marker under the
exclusive lock after reading the failure log. No reset or hardware-failure
claim. One queued HiFi2 correction-proof job was refused before device open;
it will retry under a new evidence label. Corrected C drain is row-wise.

Long distinct-input C timing confirms the whole-Q identity-batching regression
(~0.53% normal/~0.59% forced-changing maxima), despite exact outputs. D gains
only~0.19%/~0.45%. Recommendation: do not promote that specialization on the
basis of its favorable resident microbenchmark alone.

## Third screen and exact identity proof

- Corrected QK2×2 passes exact-output gates, but is11.1% slower D /4.1%
  slower C. The FP32 scheduling report recommends keeping v1 rather than
  its whole-Q identity batching, state-pack, srcA-only, or ZEROACC candidates.
- Exhaustive device checks of all65,280 finite BF16 encodings establish that
  the frozen correction path maps bit-identical finite maxima to BF16 exactly
  1.0, for both LoFi and HiFi2 (including signed zeros and subnormals).
  Identity specialization retains the original MADs, compensation rounding,
  and correction-CB publication fence. This does not relax numerics.
- Initial scalar identity guards regress E6.40% / B4.89%; guard-only E costs
  10.33%. Unrolling the exact comparison reduces the E regression to0.53%,
  while independent B improves only0.32%. These are not retained winners.
- Moving the same comparison earlier on UNPACK, behind already-issued QK/PV
  work, screens E287.0091→278.9719ms (2.80% less time), with transition smoke
  output/trace bits unchanged. B/G transfers and distinct-KV timing are now
  required; repeated resident KV particularly favors identity branches.
- A bounded function-optimization-attribute experiment (not a descriptor or
  linker optimization change) screens D O2 about2.38% faster than original
  O3. Explicit O3 is neutral and Os markedly slower. Changed ELF text sizes
  confirm different code generation. C/D qualification is pending; this is
  not an accepted gain or evidence that code size alone explains performance.
- Two later dirty-marker pauses were clean host/JIT failures: an invalid
  pragma expansion and an `inspect.py` module shadowing Python's stdlib.
  Root reviewed logs and remaining processes before clearing under lock;
  neither required a hardware reset. Failed attempts remain in evidence.

## Final qualification results

- D's scoped O2 function attributes retain a 2.362% sustained resident time
  reduction (0.81911→0.83893 TF/core). Distinct Q2048/K262144 on two cores
  improves 2.52% normal / 2.87% forced-changing maxima. Held-out stress,
  multiple Q jobs and raw trace bits match. This does not qualify changing
  global compiler/linker O2. C is only 0.28% better in screening; E/G compiler
  transfers regress 0.84%/0.79%. Keep their v1 compiler settings.
- Early exact-identity guard final resident gains: B 1.561%, E 2.800%,
  G 2.808%. All compensation arithmetic, publication fences, numerical
  formats/defines and input buffering remain fixed.
- Distinct normal 256K gains are smaller: B 0.569%, E 1.388%, G 1.362%.
  Shorter normal/frequent-max-update cases regress: B 0.52–0.78%, E/G
  approximately 0.23–0.29%. Preserve as an optional specialization, not an
  unconditional default replacement. No model/full-chip speedup is inferred.
- Parent audits passed 35 E/G final-candidate records, 48 B qualification
  plus 15 final timing records, and 54 D sustained/stress/distinct records.
  These independently check saved evidence, not extra device tests.
- V1 source hashes and the preexisting tracked diff remain unchanged.
  Final A/B compiler-transfer screens regress 3.11%/2.19%, with exact outputs
  and neutral O3 controls; rejected. All 22 Python files pass AST parsing.
- Handoff at 16:59 UTC: all agents have completed device work. Root's remote
  read-only check acquired the shared lock nonblocking and confirmed the dirty
  marker absent. No hardware reset occurred in v2. See the completed
  [consolidated report](REPORT.md).
