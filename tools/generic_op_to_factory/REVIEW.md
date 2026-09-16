# Independent review gate

For acceptance under [the detailed comparison protocol](COMPARISON_GATE.md),
also review its G0–G5 evidence and measured budgets. The receipt below permits
`not_measured` for the existing driver's narrower completion scope; that does
not satisfy the protocol's performance gate.

Before declaring a port complete, ask a separate, read-only agent to review it.
Start the review alongside local inspection; do not give the reviewer an expected
verdict. Re-review fixes against the final source and validation evidence.

Reusable agent task (replace paths, not with production run IDs):

> Independently review the native operation, exact production host factory,
> every constructed kernel, frozen source planner, bindings and cache tests.
> Trace the configured native entry through its binding and launch to the exact
> device operation checked by `factory_contract.cpp`. Check every selected factory
> returns ProgramDescriptor, owns the adapter-compatible per-Program refresh
> hook, uses typed address bindings, and does not rebuild descriptors on hits.
> The compile-time gate proves type shape, not this dispatch or hook-body behavior.
> Trace CB formats/sizes/conditional presence, kernel handles, semaphore IDs,
> compile/runtime argument positions and accessor tails to actual consumers.
> Prefer canonical geometry/format APIs, allocation-returned resource IDs, and
> shared host/kernel argument schemas with derived counts. Check for duplicated
> numeric IDs/offsets/counts rather than treating named literals as sufficient.
> Inspect the framework's hash and hit paths: address/scalar updates, optional
> presence, alias topology, descriptor reconstruction, scans, allocations,
> validation and language-boundary work. Check both miss and return-to-key hits,
> fresh retained allocations, distinct↔aliased transitions, and every legal
> input/output alias class. Identify unsupported cases explicitly. Report
> the descriptor-parity build evidence and exact acceptance cases proving real
> non-no-op cache hits; inspect their selected-call cache deltas and fresh-buffer,
> runtime-value and alias assertions. An enabled build flag or call counter alone
> is not executed parity coverage. Do not accept timings from the parity-enabled
> correctness build as normal cache-hit performance.
> For PR readiness, verify the native hot-versus-cold profile described in
> NATIVE_CACHE_PERFORMANCE.md: same operation/configuration, caching enabled on
> both paths, warm compiled kernels, observed misses/hits, raw paired samples,
> valid host-path boundaries and a statistically supported hot-path improvement.
> Record missing or inconclusive evidence as performance pending, not a pass.
> Report severity, exact file/line evidence, concrete remedies and test gaps. Separate
> measured timings (with method, scope, repetitions and raw evidence) from
> qualitative cost estimates. Do not edit source or run concurrent device tests.

The author reconciles each finding; a test passing is not evidence that a code
quality finding is resolved. Fix defects and run proportionate checks. Source
changes are made in place. Rerun `validate_port run` in the same workspace to rebuild
and pass the original acceptance tests, then `run --through native_compare` for
golden comparison. The driver invalidates previous passes and review approval
when the implementation changes; old logs remain history, not current evidence.
Re-review the corrected factory and provide a receipt for the current plan hash.
No numerical performance claim is required for a behavior-only receipt:
`not_measured` is an honest outcome but is not PR-ready performance evidence.
The required native hot/cold comparison is described in
[NATIVE_CACHE_PERFORMANCE.md](NATIVE_CACHE_PERFORMANCE.md); inspect it under
`cache_hit_overhead` and attach its raw artifacts under `performance.measurements`.
The current driver validates artifact hashes, not the benchmark's statistics or
cache-state provenance. Review must not confuse that receipt check with a passed
performance gate.
Public-call wall time includes binding, allocation, dispatch and possibly device
backpressure; it is not an isolated measurement of the cache-hit hook.

After acceptance passes and `run --through native_compare` finishes, supply `WORKSPACE/review.json`:

```json
{
  "plan_sha256": "COPY_FROM_WORKSPACE_STATE_JSON",
  "author": "author-agent-identity",
  "reviewer": "different-independent-agent-identity",
  "topics": {
    "descriptor_factory": "Binding-to-checked-operation trace, selected factories, typed bindings and refresh-hook evidence",
    "cb_kernel_semaphore_ids": "Evidence and conclusion, with file/line references",
    "argument_wiring": "Evidence and conclusion",
    "cache_hit_overhead": "Evidence and conclusion",
    "test_gaps": "Tests examined, remaining limits and disposition",
    "alias_transitions": "Legal topologies and transitions actually tested",
    "descriptor_cache_hit_parity": "Enabled CMake/TU evidence, binding trace, exact executed cache-hit cases and limitations"
  },
  "findings": [
    {
      "finding": "Concrete finding and source evidence",
      "status": "fixed",
      "disposition": "Fix, re-review and validation evidence"
    }
  ],
  "performance": {
    "classification": "not_measured",
    "assessment": "Qualitative cost assessment only; no measured speedup",
    "measurements": []
  }
}
```

Allowed finding dispositions are `fixed`, `not_a_defect`, and
`accepted_limitation`, each requiring an explanation. Do not relabel an unfixed
correctness defect as a limitation to pass the gate. A measured assessment must
use `classification: "measured"` and list raw evidence records containing
absolute `path` and `sha256`. An empty findings list is valid after an actual
review finds no issues.

The driver checks required topics, distinct declared identities, resolved
dispositions, plan identity and measurement-file hashes. It retains the receipt
and checks it on resume. These are audit checks, not authentication of agent
identity or proof that prose is truthful; the initiating agent/user remains
responsible for genuine independent review. Missing/stale/unresolved review
blocks completion. If execution already reached a blocked review stage, inspect
and supply the receipt, then use `run --through complete --retry`; unchanged
completed tests are not rerun. A plain `run` stops after acceptance.
