# tt-hw-planner (`optimize`, cc engine) — Tool Feedback

Context: Kokoro-82M TTS, single P150 (Blackhole), QB2 `sjc2-t3020`, `TT_VISIBLE_DEVICES=0`,
engine `cc`, metric `device_ms`, `--max-rounds 2`. Branch `tvardhineni/models_bringup` @ `21fb1d6`.
Full run details in `optimization_report_kokoro.md` (Session 3). This doc is feedback for the
tt-hw-planner maintainers, written from one complete on-hardware run that produced **no net win** —
which turned out to be a clean diagnostic of the tool's strengths and blind spots.

## Run outcome (evidence base)
- Baseline `device_ms` = **89.25** (capped perf test: `TT_PERF_LAYERS=2` + tiny phoneme string).
- Baseline e2e full-pipeline = **2345.09 ms** (eager, all 52 layers, prefill + 1 decode).
- Loop ran ~48 min (round 1), stopped manually. Final `device_ms` ~89.26; final e2e 2345 (unchanged).
- Biggest gaps `TilizeWithValPadding` (~14.7 ms) + `UntilizeWithUnpadding` (~4.1 ms) → cleared as
  `structural`/irreducible (no per-op lever).
- tt-lang rung exercised: authored a real fused LSTM-cell kernel (`@ttl.operation grid=auto`,
  compute + 2 datamovement kernels), on-device correct (max abs err ~0.0016), **PCC 0.9910**, but
  `measure_candidate` = 89.26 → `beat_baseline: false` → reverted.
- One candidate gave a `device_ms` micro-gain but **e2e diverged +16.36% (2345 → 2728.84 ms)** →
  reverted by the iron rule.
- Session totals: 4 commits / 3 reverts (all PCC-gated), 2 knobs distilled; all reverted at stop.

## What the tool does WELL
1. **Environment self-heal.** Auto-patched the profiler orphan-marker crash + rebuilt `libtt_metal`,
   auto-installed `tt-lang` (1.0.1, cp312), handled the P150 mesh-graph descriptor, used `claude`
   login for auth. Produced a genuine on-device Tracy profile with minimal hand-holding.
2. **Trustworthy gating.** Iron rule = `device_ms` down AND e2e not diverged AND PCC ok. It caught the
   +16% e2e diverger and reverted; never committed a regression; left HEAD clean.
3. **Deterministic, auditable targeting.** Roofline floor + per-bucket breakdown + largest-gap
   `next_target` + per-op rung ladder (grid→dtype→tt-lang→cpp), `record_kernel_attempt` per rung,
   negative-knowledge recorded. No thrashing.
4. **Real kernel-rung autonomy.** Authored a compiling, on-device-correct fused tt-lang kernel from a
   tutorial, unattended.

## Where it LAGS / is NOT doing well (this case)
1. **Optimizes a capped proxy metric.** `device_ms=89 ms` is a truncated 2-block slice; the real
   workload is `e2e=2345 ms`. The proxy↔e2e correlation is never validated — and here it's inverted.
2. **Wrong objective for the model class.** Primary metric is per-op **device** time, but Kokoro is
   **host/dispatch-bound end-to-end**, so `device_ms` and `e2e` move in *opposite* directions. Chasing
   device micro-gains was actively risky for the metric that matters.
3. **Per-op myopia; can't see graph-level wins.** Dominant costs (layout churn = redundant TILE↔RM
   conversions *between* ops; host round-trips) are inter-op/graph-structure problems. All levers are
   intra-op, so it *identifies* the big rocks but has no lever to move them.
4. **"structural → irreducible" is a dead-end, not a hand-off.** For the single biggest gap
   (`TilizeWithValPadding`, ~15 ms) it marked irreducible and moved on — no attribution of which
   producer/consumer forces the layout, no proposed source diff, no escalation. Zero actionable output
   on the biggest opportunity.
5. **Mis-spent effort on a dispatch-bound op.** The LSTM eltwise gap is count/dispatch-bound (many
   tiny ops); the fix is reducing op count (cross-op fusion / trace / multi-CQ), not one fused cell
   kernel. It spent ~15 min on a kernel that by construction couldn't move the aggregate.
6. **No ROI/Amdahl pre-check.** No estimate of max achievable gain before committing to a rung. A
   "7.9 ms of 89, dispatch-bound, best-case device gain ≈ 0, e2e risk high → skip" check would have
   avoided the detour.
7. **Low yield for high wall-time.** ~48 min/round; each `measure_candidate` is a full ~6-min Tracy
   run; **only device 0 used while 3 P150s sat idle**; no early regime triage.
8. **Robustness defects hit in practice.**
   - Teardown segfault in `_ttnn.so` after a *completed* discovery made `discover()` discard a valid
     manifest (worked around by patching `discover()` in `cc_optimize/run.py`; root segfault remains).
   - Cleanup does `git checkout HEAD -- <model dir>`, which is too broad — it silently clobbered an
     unrelated uncommitted file (this report) that lived in the model dir.

## How to IMPROVE
**Objective & metric**
- Make **e2e a first-class (or primary) objective** when the model is detected not-fully-on-device /
  host- / dispatch-bound. Optimize what ships.
- Profile at a **representative sequence length**, or verify capped `device_ms` correlates with e2e
  before optimizing it.

**Targeting intelligence**
- **Early regime triage** after baseline: classify compute/dispatch/host/layout-bound; if the per-op
  ladder is low-yield, emit the structural report immediately (would have saved ~45 min here).
- **Amdahl/ROI gate per attempt:** skip any rung whose best-case gain < threshold.

**New levers (the real gap)**
- A **graph/inter-op rung** above per-op: detect redundant/adjacent TILE↔RM conversions and
  producer/consumer layout mismatches; auto-insert layout-consistent choices or emit a concrete
  **layout-plan diff** for a human.
- **Structural attribution:** for every "irreducible" op, report the producing/consuming ops and the
  constraint forcing it — turn a dead-end into an actionable finding.
- **Count-bound handling:** when an op class is dispatch-bound, prioritize op-count reduction
  (cross-op fusion, trace, multi-CQ) over per-kernel authoring.

**Throughput & robustness**
- **Parallelize candidate measurement across idle chips** (QB2 has 4 P150s; we used 1) — ~3–4× faster
  inner loop.
- Fix the `_ttnn.so` **teardown segfault** at the source (not just the `discover()` guard).
- **Scope the cleanup revert** to only files the tool touched (or `git stash`), so it never clobbers
  unrelated uncommitted work.

## One-line summary
A rigorous, trustworthy **per-op kernel tuner** that here correctly (and honestly) concluded it had
nothing to give — but it lacks the **graph-/host-level levers** and the **right objective (e2e)** that
Kokoro actually needs, and it spends too long proving a negative it could triage in minutes.

## Enhancements implemented (2026-07-08, in `cc_optimize/perf_mcp.py`)
Additive + defensively wrapped — they NEVER change `can_stop` or the deterministic ladder; on any
error they yield no advice and the gate is byte-identical to before. Validated on the real op mix
(dispatch-bound Kokoro → `low_yield=True`, projected 1.7%; a synthetic compute-bound model →
`low_yield=False`, projected 18%, ladder runs unchanged).

- **P0a — ROI projection per target.** New `_projected_op_gain_ms(gap, rung, bound)`: the fraction of
  an op's roofline gap a per-op device lever can realistically recover. Key rule: kernel rungs
  (tt-lang/cpp) recover **~0** on dispatch/count/host-bound ops. `termination_check` now returns
  `expected_gain_ms` on every blocking op and on `next_target`. (On Kokoro the LSTM tt-lang target →
  `expected_gain_ms = 0.0`, i.e. "don't author this kernel".)
- **P0b — regime triage + low-yield steer.** New `_regime_and_roi(blocking, device_ms)` classifies
  compute / dispatch / host / layout-structural-bound and computes `projected_recoverable_ms/pct`.
  `termination_check` now returns `regime`, `projected_recoverable_ms`, `projected_recoverable_pct`,
  `low_yield`. When `low_yield`, `_low_yield_advisory()` is appended to the directive: don't spend a
  kernel cycle on a dispatch/host/count-bound op; do structural attribution and, for host/dispatch
  regimes, optimize against `check_full_pipeline_latency` (e2e), not device_ms.
- **P1-lite — structural rung is now an ANALYSIS, not a vague directive.** The `_op_ladder_status`
  BOX-4 "structural" text now requires concrete attribution: name the PRODUCER/CONSUMER op, and check
  (1) layout churn (tilize/untilize inserted only for a neighbor's layout → keep layout consistent),
  (2) host round-trips (from_device/to_torch → move on-device), (3) then bound_by hints incl.
  count-bound → reduce op COUNT via fusion/trace, and measure against BOTH device_ms and e2e.

## Enhancements still DESIGNED (not yet implemented)
- **P1-full — `analyze_layout_chain()` MCP tool.** Read-only: from the baseline ops CSV, emit the
  actual producer→consumer layout chain around each tilize/untilize and flag redundant round-trips
  (turn the structural directive's manual step into automated evidence). Needs the ops-CSV schema.
- **P1 — e2e as the primary objective for host/dispatch-bound regimes.** Switch `next_target`
  ranking + the optimize objective from `device_ms` to `check_full_pipeline_latency` when
  `regime ∈ {host-bound, dispatch-bound}` (metric select in `run.py::optimize_pipeline` +
  `termination_check`). Medium risk (changes ranking) → gate behind the regime classifier above.
- **P2 — parallel candidate measurement across idle chips** (QB2 had 4 P150s; we used 1): shard
  `measure_candidate` over idle devices for ~3–4× faster rounds.
- **P2 — robustness:** (a) keep the `discover()` teardown-segfault guard (done in `run.py`) and fix
  the underlying `_ttnn.so` finalize segfault; (b) scope the cleanup revert to only tool-touched
  files (or `git stash`) so it never clobbers unrelated uncommitted files.
