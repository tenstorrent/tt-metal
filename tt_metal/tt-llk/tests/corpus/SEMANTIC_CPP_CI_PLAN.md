# Semantic-C++ corpus CI plan

## Pull requests

- Validate the 164 logical rows, 332 architecture paths, enum domains, and
  discovery/audit drift.
- Publish TSV, JSON, and Markdown plans for BH, WH, and QSR.  These include the
  semantic class, exact blocker, selector/test/perf state, correctness contract,
  and current scoped silicon status.
- Compile every explicitly mapped module on each supported architecture.
- Reject new measured silicon states unless the row has an implemented paired
  selector, passing test gate, explicit correctness metric and threshold, and
  a source for both correctness and cycles.

## Simulator tier

- Run paired functional selectors through the pinned CRAQ artifact where the
  ISA path is supported.
- Treat CRAQ as a correctness and optimization discriminator only.  Simulator
  modeled cycles never promote a silicon `win`, `parity`, or `loss`.
- Preserve full nodeids, revisions, simulator SHA-256, manifest SHA-256, and raw
  modeled-cycle traces in the run artifact.

## Serialized Blackhole tier

- Acquire the hardware lock, run the mapped correctness modules, then the
  paired scoped profiler modules from the same revision and compiler artifact.
- Require identical inputs and selectors that differ only in the implementation
  under test.  Store raw/post profiler CSV, ELF/text hashes, disassembly, logs,
  compiler hash, device identity, and a SHA-256 manifest.
- Record the actual contract: exact, PCC plus threshold, or named tolerances.
  “Correctness passed” without its threshold and source is not promotable.
- Compare scoped device cycles only.  Host time, CRAQ cycles, static instruction
  count, and whole-kernel time cannot replace a declared body-zone metric.

## Compiler promotion loop

A corpus conversion may expose architectural operations through typed wrappers,
but fresh C++ must express the algorithm rather than copy a handwritten issue
schedule or hardcode physical LREG allocation.  Repeated blockers graduate to
general compiler work: replay formation/hoisting, SFPLOADMACRO formation,
multi-result paired-state modeling, transpose state, counter/config barriers,
and cross-thread boundary representation.  Each pass needs positive and
adversarial tests plus byte-identical output for ineligible code.
