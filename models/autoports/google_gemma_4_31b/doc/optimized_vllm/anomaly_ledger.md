# Stage 10 anomaly ledger

## Unsafe allocator warning during trace lifetime

- Initial classification: inherited Stage 09 trace-registration warning.
- Disposition: refuted. Fresh Stage 10 diagnostics placed the first occurrence
  at the exact top-k 32 to greedy top-1 transition, after the model trace and
  before sampler trace capture.
- Root cause and repair: sampler trace capture performed a redundant eager
  dispatch after exact prewarm. Capture is now capture-only, ordered after exact
  sampler prewarm and model capture. A fresh full runner passed the same
  transition.

## Unsafe allocator warning at a long-request KV boundary

- Observation: after the sampler repair, the warning moved to the first long
  batch-1 qualitative request and appeared at its first 64-token block-growth
  boundary. Batch-4 sampling and every-token deferred reads had already passed,
  ruling those paths out.
- Root cause and repair: the first `ttnn.copy` from a scheduler-owned page table
  into its persistent trace input compiled/allocated its copy program while the
  traces were live. Trace-state initialization now prewarms each distinct
  source/target pair before identities and generations are recorded and before
  any trace registration. Aliased pairs are deduplicated; refresh counters and
  scheduler-change semantics are unchanged.
- Final evidence: `evidence/final_server.log` has zero
  `Allocating device buffers is unsafe` matches. It records successful exact
  top-k 32 to top-1 sampling, repeated long batch-1 generation across page-table
  boundaries, all primary/CI benchmarks, clean mesh close, and no traceback or
  error match. `evidence/adapter_contract.xml` covers capture ordering and
  pre-trace page-table copy prewarm.

## Device 0 Ethernet resume failures

- Observation: two device-open attempts failed before model execution at
  device 0 Ethernet core 31-25.
- Recovery: only failing processes were terminated; each incident used bounded
  list/reset/list recovery followed by a passing `1x4` mesh open/close smoke.
- Final disposition: infrastructure recovery succeeded. The complete runner
  passed afterward; all four devices listed healthy and no vLLM, API-server,
  runner, or EngineCore process remained.

## Interpreter-shutdown diagnostics

- Observation: nanobind reports reference-leak diagnostics during interpreter
  teardown.
- Control: all requests and gates finished first, the mesh closed, the runner
  exited zero, and the process audit found no live serving process.
- Disposition: known shutdown-only limitation; no serving fallback, correctness
  failure, or resource holder was observed.
