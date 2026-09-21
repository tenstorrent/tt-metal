# Numerics-relaxed sprint status

Started 2026-09-18 19:33 UTC. See [contract and ownership](PLAN.md).

## Final handoff — September 19, 00:40 UTC

Completed private compute candidates; see [consolidated report](REPORT.md).
Resident time reductions versus prior optimized controls: D 6.26%, C 8.81%,
B 5.36%, E 5.74%, G 5.73%; A unchanged at 1.99237 TFLOP/s/core. Distinct
normal 256K reductions: D 1.58%, C 2.55%, B 2.39%, E 2.62%, G 2.59%.
Some short/changed-max cases regress, so these are conditional specializations.
Worst final-candidate relative L2 growth was 0.01891%, below the allowed 5%.

All final numerical/replay/input/source integrity gates pass. Frozen v1/v2
source hashes and the preexisting tracked diff are unchanged. Local Python
compile checks and whitespace checks pass. Runtime kernel JIT exercised the
changed compute code; no full host rebuild or production integration claimed.
All subagents have finished. Root verified the device lock is available,
there is no dirty marker and no benchmark process remains. IRD 224379 remains
reserved on bh-lb-08 for subsequent work; no reset was needed after recovery.
Two stale local SSH processes targeting expired reservation 223862 were
terminated; no hardware job or result was deleted.

## Environment and safety

- Reservation 223862 expired during a connection interruption. All agents
  paused device launches. Reacquired the same host as IRD **224379**, container
  `bh-lb-08-special-cglagovich-for-reservation-224379`, for four hours on
  September 19 at 00:18 UTC. Repository, existing build, and JIT cache survived.
  A mistaken allocation on bh-lb-01 (224376) was released without running tests.
- Fresh locked matmul smoke passed at **00:19:09 UTC** on the replacement
  container: same firmware, KMD, and grid; no reset or dirty-marker removal
  needed. Device work resumed under the original exclusive-lock protocol.
  Evidence recovered after the interruption is counted only after inspection.
- Existing IRD 223862 on bh-lb-08 verified active; exclusive device lock was
  available and no dirty marker existed before work.
- Fresh frozen-v1 BF16 matmul smoke passed at 19:36:23 UTC, logical device 0,
  grid 12×10, firmware 19.13.1, KMD 2.9.0. Existing host build is reused;
  changed kernels will compile through device JIT.
- IRD reports successful extension to four hours remaining at 19:42:49 UTC;
  subsequent read-only listing confirms 3:57 remaining on the same reservation.
- First FP32 harness attempt failed during host imports (`core` name collision)
  before opening a device. Root read the log, verified no remaining sprint
  processes, and cleared the safety marker under the exclusive lock. No reset.
  The explicit path-based import fix retries with a fresh evidence label.

## Numerical contract

Shared `numerics.py` passes ten scalar gate self-tests and tensor tests for
finite values, zero references, zero rows, constant-reference PCC and replay
comparison metrics. Independent review found no blocking issue. Frozen SHA256:
`240cf90f5ad978c84720ba4559a2388a234bb95a3d78962c8ca972e716f726f3`.

Per-case L2 must not exceed `1.05 * baseline + 0.0001` percentage points;
improvements are allowed. Exactly zero references have an explicit absolute
gate. Row errors/PCC and coherent-input regressions are separately reported.
Baseline/candidate bit differences are permitted; replay nondeterminism is not.

## Initial designs

- B/E/G: protected high/low numerator plus a small local chunk group, all in
  the existing two output CB allocations. Fold on fixed group boundaries,
  any maximum change, or final K; preserve full denominator compensation.
  Group two's local state holds a single existing BF16 chunk rather than a
  rounded running sum. In-place CB publication/ownership is under independent
  review before device launch.
- A scalar FP32/BF16 recurrence model passes all seven synthetic cases for
  group two, while group four fails four cases. This is a design filter, not
  an attention or hardware-accurate simulation. Group four is deferred.
- C/D: first test full-K row-zero PV accumulation instead of four partial
  reductions; then direct-to-DST old-state preload where justified. The latter
  avoids TF32 operand ingress but changes FPU accumulation order, whose error
  must be measured. D retains its v2 scoped-O2 performance baseline.

## Initial hardware results (screening, not promotion)

- C/D odd-three-K smoke passes normal and constant-V tests for full-row-zero
  PV reduction and direct old-numerator preload. All eight candidate outputs
  match baseline BF16 bits and eager/trace replay in this small smoke.
- Full-row-zero PV reduction is slower in the initial resident screen:
  D 82.11→83.56 ms, C 61.20→61.63 ms. Fewer partial packs do not compensate
  for lost overlap; rejected as a performance candidate.
- Direct old-numerator preload remains slower after batching unpack and
  caching the identity decision: D 82.096→83.390 ms, C 61.207→63.155 ms.
  These are screening timings, not sustained final qualification.
- Next C/D hypothesis: on unchanged maxima, accumulate PV directly into the
  old FP32 L1 numerator bank. This avoids the extra numerator unpack/SFPU
  pass but changes reduction association. CB ownership and final/fallback
  behavior require review. Resident-only wins will not establish general
  distinct-input speedups.

No new implementation has yet earned promotion.

## Promising measured candidates (qualification in progress)

- FP32 both-state in-place fusion, nine paired resident rounds at Q repeats 8,
  K chunks 512: D 327.6585→316.0297 ms (0.838916→0.869785 TF/core, 3.549%
  less time); C 244.2548→231.4558 ms (1.125373→1.187604 TF/core, 5.240%
  less time). All 30 C/D 32K stress cases pass; root independently audited
  their stored gates/replay flags. Held-out and guard-frequency work continues.
- These are conditional compute wins. Distinct-input D normal 256K improves
  only 0.151%, changing maxima regresses 0.274%; normal 32K regresses 0.684%.
  C normal 256K improves 0.637%, changing maxima improves only 0.225%.
  No general/model speedup or automatic default change is justified.
- Group-two direct local writes and dead-clear removal make E approximately
  neutral/slightly faster: a sustained no-clear screen gains 0.460%.
- A separate reviewed 18-instruction SFPU replay for the same even-identity
  fold yields a promising E resident result: 139.57055→131.31908 ms,
  1.96945→2.09321 TF/core, 5.912% less time. E/G wider qualification and
  independent B transfer are now running. Denominator compensation and the
  chosen numerical recipes remain unchanged.

### Follow-up findings

- Independent B replay transfer gains 5.737% resident time versus v2
  (166.0212→156.4964 ms, 1.65568→1.75645 TF/core), but distinct 32K normal
  regresses 14.55%. The scalar changed-max fallback must be improved before
  recommending a general replacement. An empty-local specialization can
  reuse the original paired compensated update; only a changed maximum with
  retained local state needs the heavier fold. This is under implementation.
- Actual profiled D whole-Q guard counts at 256K: normal 1,441/4,088 eligible
  chunks (35.2495%), growing maxima 0/4,088, repeated coherent KV 4,088/4,088.
  Root independently checked the raw device events. Profiling is separate
  from unprofiled performance timing.
- Root audit found the initial FP32 numerical harness recorded source/input
  provenance but did not explicitly reread device inputs afterward or assert
  the complete selected source closure after execution. Those records remain
  numerical/replay evidence, not full integrity qualification. New successor
  harnesses will add the missing assertions and rerun retained candidates;
  earlier evidence and frozen kernels remain untouched.

The original group-two E resident screen is slower by about 4% versus fresh
v2 early-guard (Q repeats 4, K chunks 64: 8.7713→9.1227 ms). A separate
`group2_direct/` sibling is being developed: odd-K PV writes directly into
local slots, avoiding the copy. Original group-two sources remain frozen for
independent B qualification.

FP32 numerator-only in-place screening is neutral/slower (D 82.13→82.54 ms,
C 61.22→61.35 ms). Next candidate reuses both FP32 numerator and denominator
banks on unchanged maxima to remove both recurrent-state passes. Its initial
short smoke passes; sustained performance remains pending. An explicit
correction-CB wait omitted by the earlier shortcut was restored during root
review; only the corrected implementation may be considered for promotion.

## Smoke progress and diagnostics

- Group-two first JIT found ambiguous SFPI addition of two untyped DST proxies.
  Explicit `vFloat` loads fix the compile error without changing the intended
  operations. The failed kernel never ran; device closed cleanly. Root checked
  the log and absence of remaining jobs, then cleared the dirty marker under
  the exclusive lock, without resetting hardware.
- E first-K-only Q512/K512 passes, including distinct Q reset and two exact
  trace replays. Candidate and frozen baseline are bit-identical.
- Independent B first-K-only Q512 passes normal, constant V and zero V;
  v1/v2/group-two outputs and two trace replays match raw bits. These tests do
  not yet exercise grouped recurrence.
- FP32 in-place K2 produced finite but badly inaccurate normal outputs while
  constant V passed. Root identified an L1 accumulation-mode leak between the
  first partial PV and the next softmax write; agent confirmed and corrected
  it. This is an implementation bug, not acceptable numerical relaxation.
  Corrected K2 normal/constant-V tests match baseline bits for C/D. Ten
  subsequent multi-Q cases (five each C/D) pass the budget and exact replay:
  normal, maximum transitions, repeated coherent values, constant V=3.25,
  and cancellation. A D coherent case changes output bits with essentially
  unchanged L2. See `ROOT_REVIEW.md`; sustained timing is pending.
- Frozen v1/v2 source hashes and the complete preexisting tracked diff remain
  identical to the start-of-sprint snapshot as of 19:54 UTC.
- Independent B group-two K2 (five distributions) and K3 (seven distributions)
  pass numerical/replay gates with Q1024 across two cores. Short outputs still
  match baseline bits; this is not proof that longer recurrences remain exact.
- Independent B original group-two qualification now passes 43/43 cases,
  including six at 256K. Normal 256K L2 is 3.312712→3.312464%; common V is
  1.127972→1.128033%; constant V remains 0.548638%. These preserve the
  baseline accuracy band, not a universal small-error guarantee. Common-V
  residual diagnostics expose substantial inherited loss of small variations;
  absolute/global L2 alone must not hide that limitation.
