# Stage review, round 6: iterations 9 and 10 (2026-10-30)

Independent review (fresh subagent, read-only, no device use) of commits d5ba1a1b877 (iteration 9, residual add
fused into an all-core RMSNorm) and 53ec655a9f9 (iteration 10, K-split `skinny_matmul` for the rank-wide LoRA
products) against the stage-review and optimisation-loop skills, `../perf_analysis.md`, `README.md`,
`work_log.md`, the evidence under `evidence/` (logs, ranked tables and the raw Tracy CSVs) and the code.

## Verdict: more-work-needed (no code defects; attribution, evidence and test gaps)

## Findings (as returned)

1. **P2. The iteration-9 cycle-time gain was overstated about 2x.** "kernels 55.63 -> 54.29 ms, -1.34 ms, ~-13 ms
   per 40 layers" compares review 3's chip 2 with iteration 9's chip 0 and includes the collectives class (waiting):
   0.98 of the 1.34 is collective waiting. Re-derived from the CSVs, same chip, collectives excluded: chip 0
   49.84 -> 48.92 (-0.92), chip 2 49.28 -> 48.69 (-0.59) per profiled step. Per op on chip 2: adds 1.770 -> 0.887,
   norm backward 1.537 -> 1.771, norm forward 1.374 (9 calls) -> 0.604 + 0.607 fused + 0.152 plain = 1.363,
   unchanged (2 x 76 us per fused call equals the 153 us 32-core `RMSNormForward`). The gain is the removed adds
   minus the backward's extra read, not a faster forward. Iteration 10 on the same basis is sound (routed rows
   1.404 -> 0.804, chip 0 48.92 -> 48.12).
2. **P3. The loss match after the two numerics-changing iterations existed but was uncommitted and undocumented**
   (`iter10_loss.sh` and its outputs, produced after commit 53ec655a9f9).
3. **P3. `skinny_matmul` had no harness case; N = 32 / 64 (two output tile columns) had never run.** Only the
   standalone script (N = 16, against `ttnn.matmul`) and the routed LoRA tests at rank 16. Also
   `_LoraUpdateAdd.backward` is on the TP path only, so "the LoRA, block and model cases exercise every routed
   product" held only via the 1-layer TP gradient check.
4. **P3. Stale README:** "16 since review round 4" (the harness has 20), and "three kernel projects (... residual-fused
   RMSNorm forward / all-core forward)" after iteration 9 did the third.
5. **P4.** work-log "(this commit)" for iteration 10; `O/iter9_cmake.log` cited but untracked; "-1.5 ms per 8
   layers" is -1.72 by totals; "what ttnn's matmul does for a transposed B" is the attention-matmul idiom.
6. **P4 / residual risk.** The fp32 partial tiles of all three all-core ops are summed in fp32 DEST but read back
   through `copy_tile` with the default unpack mode (SrcA, ~19-bit mantissa), not `UnpackToDestFp32`; invisible in
   the PCCs (0.999997-0.999999).

Verified clean by the reviewer: the CB full-block discipline in all three kernel sets (every push/pop a full block
or the exact capacity; the hazard claim reproduced on paper: 54-tile slices, block 4, second item at offset 6 of an
8-tile CB; `choose_work_split(32, 160, 110)` = 3 x 54, `choose_skinny_split` = 13 x 13 / 13 x 20); HiFi4 + fp32
DEST in all three factories, DEST use <= 4 tiles; the fused-norm math (eps inside the sqrt, fp32 `1/C`, partials
from pre-rounding sums), the backward returning one `dsum` for both inputs (aliasing safe), the two-output
Function's node / dummy-node ordering, frozen gamma without gradient, the block-0 norm on the C++ op; the skinny
math (transpose via the LLK flag, W index `k*Nt + n`, partial page `r*S*Nt + s*Nt + n`, reduce page `r*Nt + n`),
program hashes; the fallbacks (KV cache / memory-efficient runner, padded sequence, batch > 1 via the 64-tile-row
cases, TP via the profile and gradient check); the evidence timeline (iteration-9 chain 10:19-10:39 before the
iteration-10 wiring, iteration-10 chain 10:40-11:00 with both on); every MFU-history number (537.4 = mean of the six
steps, 30.19 %; 536.1, 30.27 %), effective clocks and temperatures; the harness PCCs; the iteration-10 per-op
claims (77 -> 46, 122 -> 65, 76 -> 45 us), op count 508 -> 524 (one extra launch per routed call).

## Response (same day)

- Finding 1: the iteration-9 MFU-history row, its "Result" paragraph, the Loop-resumed sentence and the README
  section restated on the same basis (chip 0, collectives excluded, all four profiled steps averaged, re-derived
  independently: kernels excluding collectives 55.73 -> 54.87 -> 54.19 ms per 4-layer step; norm forward 1.38 ->
  0.15 plain + 0.62 + 0.61 fused = 1.39, unchanged; same-shape adds 2.27 -> 1.29; norm backward 1.64 -> 1.85;
  -0.86 ms per 4 layers, ~-8 ms per 40), with the explanation (the fused forward moves three times the bytes of the
  32-core norm at bandwidth) and the gamma re-read noted as the next rung. Iteration 10's row and paragraph carry
  the same basis (-0.68).
- Finding 2: the loss match recorded in `../perf_analysis.md` ("Loss match after iterations 9 and 10"), the README
  state line and the work log, the evidence committed (max |diff| 0.0273 PASS; TP=4 real weights on the baseline's
  trajectory, 543.0 ms warm).
- Finding 3: `test_skinny_matmul_matches_torch` added (torch fp64 reference; N = 16 / 32 / 64 x both W layouts x
  {K = 5120 at 32 tile-rows, K = 8192 at 64 tile-rows}: 12 cases, `evidence/review6_skinny_tests.log`); the
  coverage sentence corrected (`_LoraUpdateAdd.backward` covered by the TP gradient check).
- Finding 4: README harness count and the kernel-project list corrected.
- Finding 5: hash filled, the cmake-log mention dropped, -1.7 ms, transpose wording.
- Finding 6: recorded in the iteration-10 section as a residual precision note (below bf16 output rounding), not
  changed.

## Re-check of a492accb3a6: clean-pass

Every finding verified fixed by the same reviewer (the restated numbers reproduce from the CSVs on the stated basis;
deltas -0.86 / -0.68 agree with the reviewer's warm-up-2 basis -0.92 / -0.80; 12 skinny cases PCC 0.999999 incl.
N = 64). One P4 note taken in the closing commit: the four-step averages (55.73 / 54.87 / 54.19 ms) fold step 1's
one-off setup ops into the per-step figure; the steady steps 3-4 are 49.84 -> 48.92 -> 48.12 ms per 4-layer step
per chip excluding collectives, now stated alongside.
