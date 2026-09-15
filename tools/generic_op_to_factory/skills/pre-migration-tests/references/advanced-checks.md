# Escalate operation checks by risk

Run the ordinary explicit acceptance assertions first. Diagnostics supplement
those results; they are not numerical or contract oracles. Do not use TTNN
comparison mode in this workflow. Expected values and tolerances must be explicit
in the operation-specific tests and come from an independent CPU/PyTorch reference.

## Required safety pass

Rerun the focused tail, partial-write, alias, fresh-buffer, cache-refresh and
protected-memory cases with `./scripts/run_safe_pytest.sh --dev`. Keep its evidence
separate from the ordinary correctness run. The profile provides serialized device
access, dispatch-timeout handling, automatic triage/reset, Watcher NoC and circular
buffer instrumentation, lightweight kernel assertions and LLK assertions.

Record the exact command, architecture, effective environment, complete log,
JUnit, pytest status and attach/detach or triage output. Simulator execution does
not receive the same NoC-sanitizer coverage. Watcher is not a tensor-allocation
ownership checker or a general device ASan; retain the protected-memory assertions
and limitations described in [cases.md](cases.md).

## Risk-triggered source diagnostics

| Risk or symptom | Check | Evidence and limitation |
| --- | --- | --- |
| L1 capacity, bank pressure or fragmentation | Memory reporter, `ttnn.get_memory_view`, or TTNN no-dispatch graph resource capture | Record peak/category and placement data. No-dispatch capture can disturb cached programs; clear/recreate cache state before cache assertions. |
| Missing NoC barrier or unflushed asynchronous write | `TT_METAL_NOC_DEBUG_DUMP=1` in a separate process | Do not combine with Watcher, DPrint or the profiler. Its timing and acknowledgement limits mean a clean run is supporting evidence, not proof. |
| Suspected device allocation, CB, semaphore or transfer-boundary defect | Emule build with `TT_METAL_EMULE_ASAN=1` and its preflight/postflight procedure | Diagnostic only unless the operation is supported and the emulator is proven to have executed. It does not prove behavior on silicon; note skipped or limited checks such as padding/object-intent coverage. |
| Timing-sensitive device race | Scoped Watcher debug delay | Use only to reproduce/localize. Do not treat a perturbed timing run as performance evidence. |

Run only diagnostics justified by the operation's source, architecture, failure or
memory risk. Record omitted applicable diagnostics and why. Do not turn unavailable
instrumentation into a skip in the shared acceptance suite.

## Checks deferred until native host code exists

- Build and test host C++ with ASan/LSan/UBSan when the port introduces memory-risky
  host code; use TSan when it introduces or changes concurrency.
- Enable descriptor-patching parity checks for descriptor-based native cache-hit
  bugs. This compares native refresh with native reconstruction; it is not
  Python/native parity and is too expensive for ordinary performance evidence.
- Profile only after correctness and safety runs pass. Profiling wrappers and
  instrumentation can alter timing, and a profiler wrapper can mask the underlying
  pytest exit status. Preserve a separate correctness result.

These deferred checks cannot be claimed by this pre-port skill. Include them as
specific handoff recommendations when the operation's risks justify them.
