---
name: module-factoring
description: Factor logic that is duplicated across two or more ttnn reference implementations into a single reusable TTTv2 module, and extract each reference's forward into a standalone runnable unit reused as both the perf baseline and an accuracy sanity oracle. Use when designing the abstraction boundary for a new module that generalizes several references (de-duplication, variation matrix, topology-gating, where variation lives), when extracting a reference's ttnn compute out of its model scaffold, or when proving the new module matches every reference at PCC and device-perf parity.
---

# Module Factoring

Bring up one reusable [TTTv2 module](../tttv2-module/SKILL.md) that generalizes the compute duplicated
across **two or more ttnn reference implementations**. This skill owns the two judgments the contract
skill does not: **where the abstraction boundary goes** (which variation lives in config vs. the
forward signature) and **how to prove parity against the real references** at both accuracy and
performance — by extracting each reference's forward into a unit you can run in isolation.

`$tttv2-module` is the contract the result must satisfy (structure, strategy discipline, grep checks,
test shape, low-PCC debugging). This skill is the *path to* a module that satisfies it when the input
is several references rather than one. Use both together.

## Mission

Given N references that each contain a version of the same compute, ship a single
`models/common/modules/<area>/<name>.py` that matches **every** reference at **PCC ≥ 0.99** and at
**device-perf parity within a small tolerance**, on single *and* multi device, with the references'
own logic extracted into a runnable unit that is the parity target for both.

Two failure modes bound every decision: **scope creep** (building config-driven paths no reference
exercises) and **under-proof** (claiming parity against a self-authored golden or a self-baseline
instead of the real references). Avoid both: cover only what the references exercise, and prove it
against the references themselves.

## 1. Factor — find the boundary before writing code

Do this first and write it down; it is the spec the rest of the bringup is graded against.

- **De-duplicate the references first.** Diff them against each other before anything else. Two
  "different" references are often the same implementation (a copy, a fork with one renamed import, a
  perf-tuned variant). Collapse near-duplicates to one column and record exactly where they diverge —
  otherwise the variation matrix invents differences that do not exist.
- **Build a variation matrix:** rows = configurable knobs, columns = references, cells = the value each
  reference uses. Knobs that are universal become defaults; knobs that differ become optional
  `<Name>Config` fields with override hooks.
- **Topology-gating check on every parallelism / strategy knob.** For each knob that depends on mesh
  shape (tensor / expert / sequence parallelism, all-to-all dispatch, ring-vs-linear collectives, …),
  read the reference's mesh-config derivation and decide whether it is *actually exercised on the
  target topology* or *collapses to a no-op* below a larger mesh. A knob that only activates on a bigger
  SKU belongs in a later larger-topology bringup — out of scope, not in the matrix. Run this check
  *before* proposing scope.
- **Decide where variation lives.** Prefer a single canonical forward signature shared across the
  family, absorbing per-reference variation into *optional* construction-time config (default `None` →
  step skipped) — **until** that absorption starts multiplying distinct compute paths, at which point a
  slightly non-standard forward argument is the simpler choice. Optimize for total simplicity, not
  interface purity. State explicitly what the **caller still owns** (residuals, norms, heads, transforms
  outside the unit) so the boundary is unambiguous to whoever wires the module in next.
- **Strategy decisions bind at construction, not in `forward()`.** Where behavior differs across
  references by static config (routing order, activation, normalization variant), model it as an enum
  resolved to a callable when the config resolves, then run it unconditionally in the hot path — no
  static `if` on config inside `forward`. Shape the enum so a third variant is one new value + one
  branch, not a redesign. Implement only the values the references exercise.

When unsure, take the **conservative scope**: the narrower topology, the simpler interface, the path
every reference shares. Defer the uncertain capability with a written reason. Log each non-obvious
decision as (premise → the reference source that supports it → choice).

## 2. Extract — lift each reference's forward into a runnable unit

The references live inside model scaffolding (ccl/mesh/config managers, generators, full-model wiring).
To use a reference as a parity target you must be able to **run its forward in isolation** at matched
shapes/dtypes/mesh. This extracted unit is built **once** and reused twice — as the perf baseline (§4)
and as the accuracy sanity oracle (§3). Skipping this is why a bringup ends up measuring itself.

- Strip the reference down to a callable forward: the minimal weight load + the op sequence, fed
  device tensors directly, with the surrounding model wiring removed or stubbed.
- Keep it faithful — do not re-derive the math; lift the reference's actual ops. Commit it alongside
  the tests so the parity claims are reproducible.
- Match config explicitly: identical input shapes, dtypes, mesh shape, and mode on both sides, and the
  same op boundary (see §4 boundary-equivalence).

## 3. Accuracy — oracle and the free sanity check

- **Primary module gate:** the new module vs. an **HF torch reference**, driven from identical inputs
  and weights, the golden cast to the **same dtype the device path sees**, **PCC ≥ 0.99** per
  (reference × mesh shape × mode) covered. (This is the `$tttv2-module` §3 test; build it there.) A
  higher-precision golden makes quantization noise look like module error — match the dtype.
- **Free extraction sanity:** the **extracted ttnn reference** (§2) vs. the **same HF torch reference**.
  This costs nothing extra — the extracted unit already exists for perf — and it confirms the extraction
  is faithful before you trust it as a baseline. **Mark these so they are not required in CI**
  (e.g. a `slow` / bringup-only marker); they are a bringup aid, not a gate.
- Discrete-selection ops (top-k, argmax, sort) fail differently from smooth ops: a tiny rounding
  difference flips *which* element is chosen, unmoved by fidelity bumps (PCC stuck ~0.97–0.99, spiky
  per sample). Compute the pre-selection math in high precision, cast only for the selection, and match
  that same rounding in the golden so both make the same choice. See `$tttv2-module` for the full
  low-PCC bisection method.

## 4. Performance — module vs. the extracted reference, small tolerance, every mesh

Perf parity is a hard requirement, not a nicety: a module that is accurate but slower than the code it
replaces is not a successful factorization. Measure the module **against the extracted reference**, not
against itself.

- **Boundary-equivalence first.** Decide what counts on each side *before* measuring. Is the
  reference's comparable unit the *same set of ops* as the module's forward? References fuse, split, or
  scope work differently (an extra norm inside the unit, a collective the caller owns). Write the
  matched-config comparison down (which ops count on each side, identical shapes/dtypes/mesh/mode);
  otherwise the two numbers measure different workloads.
- **Mechanism:** build a forward-only, **trace-captured** microbenchmark — see `$tt-enable-tracing` for
  capture/replay (warm-up compile + lazy weight load, capture once, replay in a signposted loop so you
  time device work, not host dispatch). Read device time and the compute-vs-collective split via the
  profiler / `tt-perf-report` path — see `$optimize` for the profiler env and the structured perf
  accounting. (`$optimize`'s LLM/vLLM/context-contract material does not apply; reuse only its
  measurement style.) One parametrization per (mesh shape, mode); never average across them.
- **Time the extracted reference the same way** at the matched config, and tabulate module-vs-reference
  per (mesh × mode), single *and* multi device.
- **Variance discipline:** capture ≥3 samples of the unchanged baseline, report median + range. If a
  delta you care about is smaller than the baseline spread, you are measuring noise — add iterations.
- **Marker budget:** bracket the forward boundary with one signpost pair; the per-op `OP CODE` grouping
  gives attribution without a marker per op.
- **Verdict per row:** within tolerance, or a gap with a **named cause** (fidelity / dtype / chunking /
  collective config) that is **closed** — not deferred. If a port needs a non-default program/compute
  config for perf, plumb it through the config's override slots (already exposed per the contract), not
  by changing the module's resolved defaults.

## Evidence to leave

A scratch `MODULE_NOTES.md` (uncommitted): the (premise → source → choice) decision log, the
de-duplicated reference set + variation matrix, the final **PCC parity table** (per reference × mesh ×
mode), the **perf table** (module vs. extracted reference, per mesh × mode, compute/collective split,
median + range), and the `$tttv2-module` grep-check output. Mirror the PCC + perf numbers into a
structured `<name>_results.json` alongside it. The committed public story is the module docstring, the
co-located tests, and the inventory row in `models/common/modules/README.md`.
