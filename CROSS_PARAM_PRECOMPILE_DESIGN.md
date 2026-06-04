# Cross-Parametrization Precompile with the C++ Collector — Design & Feasibility

**Status:** design / feasibility note (no implementation yet).
Companion to `PRECOMPILE_FINDINGS.md` — this expands **Next Target A**.

## Goal

Today we have two collectors (see `PRECOMPILE_FINDINGS.md` §1):

- **#1 — C++ op collector**, across one run: captures *every* program a forward dispatches,
  op-agnostic. Built; validated on models/blocks.
- **#2 — eval Python plugin**, across a pytest session: captures *one* `generic_op` program per
  test case, deduped across all parametrizations. Pre-existing; `generic_op`-only and shaped
  around the golden-test signature.

We want the **best of both**: drive **mechanism #1** across an entire pytest session so it
captures **all ops of all parametrizations** of *normal* ttnn tests — including multi-op test
bodies. The motivating example: a conv test isn't just `conv`, it's
`interleaved→sharded → halo → move → conv`. We want all of those programs collected and
compiled up front (locally or on the farm), regardless of op type or how many ops a body runs.

## Bottom line

**Medium difficulty, and the hard part is already done.** The op-agnostic multi-op capture is
exactly what mechanism #1 already does (a model forward is just a multi-op body). The new work is
a pytest plugin + a two-pass orchestration; the only genuine uncertainty is how real test bodies
are written (see Risks).

---

## 1. Why the core is already solved

Mechanism #1 captures multi-op chains by construction. Under NO_DISPATCH each op returns a
**spec-correct (address-0) output**; the next op builds its program from that spec; the funnel
hook (`create_and_cache_mesh_workload`) stashes every program. So running a conv test body in
collect mode yields all 5 of `interleaved→sharded→halo→move→conv` with **no new capture logic**.

What's missing is only the **driver**: running that collect mode across a whole pytest session of
arbitrary tests, accumulating one global distinct-program set, then compiling it once.

## 2. The one genuinely tricky part — fixtures

Mechanism #2 sidesteps fixtures by *re-invoking* each body as `func(device=device, **params)` at
`pytest_collection_finish`. That only works because golden tests have a uniform signature and need
only `device`. **Arbitrary ttnn tests use many fixtures and pytest's full resolution machinery**,
so you cannot re-call the function outside pytest.

**Solution:** don't re-invoke bodies in a pre-pass — **hook the call phase**
(`pytest_runtest_call` hookwrapper) so each test runs through pytest's *normal* fixture setup,
but with collect mode active. Fixtures come for free.

## 3. Design — two passes sharing the on-disk cache

Batched compile (collect *everything*, then compile once, then run warm) requires collecting all
programs before running any test for real. With pytest's one-execution-per-test model, that means
two passes over the session, sharing `TT_METAL_CACHE` (the on-disk kernel cache persists across
processes):

- **Pass 1 — collect + compile.** A small plugin enables global NO_DISPATCH collect mode. Every
  test runs via pytest's normal machinery (fixtures resolved); its ops are captured into one
  global program set; the inevitable post-op failure is **swallowed**. At `pytest_sessionfinish`,
  one `up_front_compile(device, workers)` → warms the on-disk cache (local executor or remote
  farm — routing is automatic when `TT_METAL_JIT_SERVER_ENABLE=1`).
- **Pass 2 — real run.** Normal pytest → hits the warm cache.

Two pytest invocations, one shared cache. This is the natural generalization of mechanism #2,
using mechanism #1's capture, with fixtures handled by pytest itself.

Plugin shape (sketch):

```python
# pass 1 only (gated by an env flag)
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    ttnn.graph.up_front_begin_collect()      # global collector; NO_DISPATCH
    try:
        yield                                 # pytest runs the body w/ fixtures; ops captured
    except Exception:
        pass                                  # post-op readback/assert fails on addr-0 — ignore
    finally:
        ttnn.graph.up_front_end_collect_keep()  # accumulate, don't compile yet

def pytest_sessionfinish(session):
    ttnn.graph.up_front_compile(device, max_workers=<≈server cores>)
```

(Exact begin/end/accumulate API may need a small tweak so the collector accumulates across tests
and compiles once at the end, rather than per-test.)

## 4. Risks / edge cases (where the real difficulty lives)

1. **NO_DISPATCH bodies fail after the ops.** A normal body ends with
   `ttnn.to_torch(out)` + `assert_with_pcc(...)`; on address-0 outputs that errors. Must wrap each
   test and **keep the captured programs anyway**, cleanly enough that one failure doesn't poison
   collection. (Easy in principle.)
2. **Intermediate readbacks / value-dependent control flow.** If a body reads a tensor back
   *between* ops, or branches on tensor *values* (`if out.mean() > 0:`), NO_DISPATCH garbage
   breaks it and you capture only the ops *before* that point. Graceful degradation — the missed
   programs simply compile cold in pass 2 — but it caps coverage. **How common this is in real
   ttnn tests is the main unknown** and the first thing to survey.
3. **Double host-side work.** Both passes run the bodies' host logic (input creation, etc.). Pure
   overhead, but the compile savings dominate.
4. **NO_DISPATCH across the whole session.** Device fixture setup/teardown between tests under
   NO_DISPATCH must hold (the mode is designed for whole-model memory analysis, so likely fine —
   verify).
5. **Carried-over caveats:** worker count ≈ server cores; for the remote farm, kernel sources must
   be synced (the FILE_PATH `#include`-by-path rule, `PRECOMPILE_FINDINGS.md` §2.4).

## 5. Effort & de-risking

- **Prototype: days, not weeks.** Plugin (`pytest_runtest_call` hookwrapper +
  `pytest_sessionfinish` compile) + two-pass runner + robust per-test failure swallowing. The op
  capture (mechanism #1) and the farm are reused as-is.
- **De-risk first (quick):** survey a representative slice of real ttnn tests (conv, sharded,
  matmul) for mid-body `ttnn.to_torch` / value-dependent branching. If most are
  "create → ops → readback → assert" with no mid-body readbacks, coverage is high and this is
  straightforward. If many interleave readbacks, expect partial capture and diminishing returns on
  those.

## 6. Alternatives considered

- **Generalize mechanism #2's re-invocation** (call `func` with all fixtures injected manually):
  doesn't generalize — arbitrary fixtures can't be hand-injected.
- **Raise-to-abort at the C++ funnel** (like #2 does at `generic_op`): captures only the *first*
  op of a body → useless for multi-op chains. Multi-op needs the body to *continue* through all
  ops → NO_DISPATCH (return spec-correct output, don't raise).
- **Single pass, collect-during-execution:** can't batch the compile (the whole point), so it
  loses the parallel/farm win. Two-pass is required for batched compile.

## 7. Anchors

| Piece | Where |
|---|---|
| C++ collector (reused) | `ttnn/api/ttnn/up_front_compile.hpp`, `ttnn/core/up_front_compile.cpp` |
| Funnel hook / capture point | `ttnn/api/ttnn/device_operation.hpp` (`create_and_cache_mesh_workload`) |
| Python bindings | `ttnn/core/graph/graph_nanobind.cpp`, `ttnn/ttnn/graph.py` |
| Existing per-case plugin (mechanism #2, to generalize) | `.claude/eval/precompile_plugin.py`, `.claude/eval/precompile.py` |
| NO_DISPATCH semantics / spec propagation | `compute_output_specs`, graph `RunMode::NO_DISPATCH` |
| Remote routing | `tt_metal/impl/program/program.cpp` (~:2016) |
| Overall context + numbers | `PRECOMPILE_FINDINGS.md` |
