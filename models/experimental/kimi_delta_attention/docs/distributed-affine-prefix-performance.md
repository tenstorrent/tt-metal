# Distributed affine prefix performance design

## 1. Problem framing

Objective: raise source-derived floor efficiency for the production Kimi-K3
`SP=4, TP=2` distributed affine prefix from 6.61% to at least 60% on LoudBox.
The fixed baseline is 10,535.604 us measured and 696.180 us floor, so the
acceptance threshold is at most 1,160.300 us.

Facts:

- The public operation returns only per-partition entry state and replicated
  global final state; it does not return prefix transforms
  (`chunk_gated_delta_rule.cpp:889-975`).
- Current SP4 execution performs 40 logical transfers, including two local
  final-state copies. Each A, B, or S payload is 3,145,728 B.
- Each cross-device `PointToPointOp` creates one endpoint program pair and one
  global semaphore (`point_to_point_device_op.cpp:215-257`).
- The measured exact function is 10,535.604 us versus a 696.180 us projected
  floor. The six in-function stages are ranked in the tracked log.
- Repository history contains an SP8 socket-based receiver-owned prefix probe
  (`origin/pjosipovic/kda-loudbox-sp-prototype`, `sp_affine_prefix.py`), so a
  persistent-socket schedule is an existing fallback pattern.

Inferred hypothesis: for SP4, a state relay will beat Hillis--Steele because it
removes more P2P launches and matmuls than it adds dependency depth. This must
be accepted or rejected by hardware evidence.

Constraints and invariants:

- Optimize only the SP4xTP2 path; retain the generic implementation as fallback.
- Preserve FP32, TILE, interleaved input/output contracts and both outputs.
- Preserve eager reuse and trace capture; PCC must remain >= 0.999.
- One concern and one validated result per commit. Update the tracked log in
  every retained or rejected experiment commit.
- Measure ten warm trace replays; use device union and source-derived floors.

Non-goals: tuning the full KDA layer outside this operation, changing numerical
precision, or claiming Galaxy performance from LoudBox.

## 2. Workflow

1. Capture the unchanged SP4xTP2 baseline and rank stages by measured time and
   target-closing potential.
2. Form one source-backed hypothesis for the largest exposed cost.
3. Implement the smallest isolated experiment behind the SP4xTP2 condition.
4. Run focused PCC/cache/trace correctness, then ten-replay profiling.
5. Retain and commit only a real improvement; otherwise revert code and record
   the rejected hypothesis.
6. Recompute the breakdown and repeat until efficiency reaches 60% or source-
   backed opportunities are exhausted.

## 3. Data flow

Current:

`(A_i,B_i) -> two Hillis--Steele transform stages -> exclusive shift -> S_entry`

First experiment:

`S_0 -> apply (A_0,B_0) -> send S_1 -> apply (A_1,B_1) -> ... -> S_4`

The first experiment reduces 38 cross-device A/B/S transfers to 12 cross-
device S transfers (six neighbor relays plus six final broadcasts) and reduces
global matmul launches from six to four. It trades logarithmic dependency depth
for fewer operations.

## 4. Domain model

- Affine transform: `(A,B)` representing `S_out = A @ S_in + B`.
- Partition entry state: state before one SP partition.
- Partition exit state: state after applying that partition's transform.
- Prefix schedule: a dependency-preserving way to produce every entry and the
  global exit.
- Projected floor: overlap-aware lower bound from verified compute, DRAM, and
  fabric work.

## 5. Architecture

Keep the public operation and tensor contract unchanged. Add only a localized
SP4xTP2 schedule selection inside `kda_distributed_affine_prefix`. Reuse TTNN
matmul/add/P2P initially. Escalate to a dedicated device operation only if the
profile proves launch/synchronization overhead remains the limiter.

Dependency direction remains public transformer API -> existing TTNN operations
-> device programs. The tracked optimization log owns experiment evidence; the
source owns no benchmark-only policy.

## 6. Alternatives

1. Direct state relay (recommended first): smallest change, 63% fewer cross-
   device transfers, 33% fewer matmuls; risks serial critical path.
2. Adapt persistent socket receiver-owned schedule: reuses repository history
   and overlaps independent TP columns; higher lifecycle and trace complexity.
3. Fused C++ device operation: best launch/communication control and likely
   final route to 60%; highest correctness, semaphore, and maintenance risk.

## 7. Risks and open questions

Hardest-to-change decision: whether SP4xTP2 deserves a specialized device
schedule or should remain inside a general-SP algorithm. The design keeps the
first special case local until measurement justifies promotion.

Open questions resolved only by measurement:

- Does serial state dependency cost more than the removed P2P/matmul launches?
- How much of P2P measured time is fabric transfer versus command/program cost?
- Will reaching 60% require fusing communication and affine compute?

The user's instruction to work autonomously and rank every experiment by time
or potential gain is treated as approval of this iterative architecture.
