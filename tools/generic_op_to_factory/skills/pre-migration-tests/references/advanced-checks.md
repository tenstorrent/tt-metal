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

## Required native descriptor cache-hit parity

Once C++ exists, run the acceptance suite with the native correctness build's
`ENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK=ON`. The migration driver's factory
gate requires that CMake setting and compiles a probe requiring the actual
factory translation unit's `TT_DESCRIPTOR_PATCHING_PARITY_CHECK` definition.
See [PORT_FLOW.md](../../../PORT_FLOW.md#descriptor-cache-hit-parity-required-native-configuration)
for build preparation. Do not try to enable this compile-time option through a
pytest environment variable.

On a covered cache hit, the adapter first updates the cached program normally,
then constructs a fresh native reference and compares per-core/common runtime
argument values and tensor-backed CB addresses. A discrepancy fails execution.
The shared acceptance tests must independently assert a real miss-to-hit
sequence, fresh retained buffers, and supported runtime-value/alias transitions.
Keep their numerical, metadata and protected-memory assertions. A first-call-only
suite, a no-op or repeated cache misses does not exercise the parity check.

This is native-refresh versus native-reconstruction, **not** Python/C++ numerical
parity, a complete structural comparison, a hash-collision test, or detection of
unnecessary misses. The current checker also does not establish full common-arg
length or kernel/CB-configuration equivalence. Record the cases and effective
build evidence; the flag alone is not execution coverage. The independent review
must trace the native binding to the instrumented operation and verify actual
cache-hit assertions. There is no per-operation successful-parity counter today.

Use the existing native acceptance run on this correctness build; do not add a
clone or another full golden sweep. Instrumentation reconstructs programs on hits,
so these timings are not production cache-hit performance. Performance requires
a separate configuration with the option OFF and a rebuilt, verified runtime;
do not modify the correctness build beneath a resumable validation receipt.
No native parity execution is claimed by this pre-port skill.

## Other checks deferred until native host code exists

- Build and test host C++ with ASan/LSan/UBSan when the port introduces memory-risky
  host code; use TSan when it introduces or changes concurrency.
- Profile only after correctness and safety runs pass. Profiling wrappers and
  instrumentation can alter timing, and a profiler wrapper can mask the underlying
  pytest exit status. Preserve a separate correctness result.

These deferred checks cannot be claimed by this pre-port skill. Include them as
specific handoff recommendations when the operation's risks justify them.
