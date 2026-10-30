# AutoFix — trace capture lifetime and cleanup

Diagnosis: AUTODEBUG.md. Contract: a candidate that passes eager execution must
be tested without retaining already-consumed output as unrelated live L1 state;
a capture-body validation exception must leave dispatch usable and must not replay
a partial layer. The original O block16 failed at static end1316992 against live
buffer991232, then hung while closing an active capture.

Priority hypotheses: (1) dead eager output pins a low allocator frontier;
(2) Q lifetime unnecessarily increases pressure; (3) use three readers to reduce
static storage. The discriminating first experiment deletes only consumed eager
output before capture and balances capture cleanup. No model lifetime changes,
geometry changes, tracker exemptions or runtime source changes were made.

Changed tests: run_optimized_decoder.py asserts eager PCC and drops its consumed
output; optimized_trace.py pairs begin/end and discards partial traces on body
exceptions; timing and correctness runners release successful traces on assertion
exits. The sweep has a bounded timeout with live triage and stops for recovery.

Validation: original O block16/BFP8-attention/BFP4-MLP configuration now passes
four trace replays, determinism and changed positions/pages. Exact command and
results are o16_lifetime_fix.log/json; decode PCC0.9995749847521892, changed-input
PCC0.9983147863350963. This proves output lifetime was sufficient to trigger the
observed capture overlap. Block16 is legal under this harness lifetime; it is not
rejected as a physical-device limitation.

Controlled capture-body failure test: check_optimized_capture_cleanup.py; result
in capture_cleanup.log. No arbitrary tracker acknowledgment or unsafe capture
scope was added. This helper handles body validation failures; it does not promise
recovery from an error in end_trace_capture itself.

The model's Q lifetime and multi-reader geometry remain performance experiments,
not necessary repairs for this failure. No context, batch, or cache contract was
reduced. Other optimization gates remain work.

## Larger-batch SDPA grid and matmul contract

Observed real-input sliding B32 prefill PCC .9992738 but decode .921906 with
explicit8x8 SDPA, while default SDPA passes .9994955. Cache BF16, DRAM head
helpers, HiFi4 SDPA and K32 alone do not repair it (`batch32_isolation`).
Source diagnosis: `sdpa_decode_program_factory.cpp` builds Q/output reducer
coordinates from `reducer_idx % grid_size.x` and `/ grid_size.x` (around342).
The Q shards from `_decode_memory` use the actual11-column worker grid.
For B>8, the8-column program's assumed Q locations differ from the actual shards.
Full attention's DRAM Q does not have this mismatch.

The explicit11x10 config repairs B32 and B16 with unchanged BFP4 projections,
BFP8 cache and HiFi2 SDPA. Evidence: `batch_grid_fix/grid11x10_batch{32,16}_correctness.json`.
B1 timing and other batch boundaries are still being checked. No precision
fallback or batch-cap reduction is accepted.

B36 additionally hits the DRAM-sharded matmul validator's `M == 1` tile gate
(`matmul_device_operation.cpp` around1627). The candidate uses an explicit
2D multicore matmul for M>1, with the same DRAM-sharded BFP4 weights, LoFi kernel
and L1 residual boundary. Shared input conversion is performed before the
attention/MLP projection group. Larger-batch acceptance is pending.

## Maximum-context dispatch recovery and final-path validation

A separate long-prefill attempt stalled in host FetchQ reservation before PCC.
`final_context_watcher/AUTOTRIAGE.md` distinguishes a queue-progress contradiction
from an SDPA kernel hang and does not establish a generic runtime source defect.
Live device snapshots and native host stack were preserved. Only the owned test
and queue were terminated; owner scan was empty, and bounded list/reset/list
succeeded. No generic dispatch code was changed and no queue-safety assertion was
suppressed. The retry uses TT_METAL_OPERATION_TIMEOUT_SECONDS=60 for bounded
failure reporting, not as a functional repair.

Meanwhile the independently identified decoder issue—masked historical sliding
prefix work—was repaired with a bounded/rebased SDPA page view and streaming
prefill. `final_context_v2/manifest.json` records the final source and commands.
Both layer kinds complete262143-token and exact262144-token prefill in the same
process, followed by33 deterministic trace replays at the final context position.
This exercises repeated dispatch queue wraps and preserves context. The final
runtime has no per-chunk host synchronization or hidden fallback. The old FetchQ
failure's specific source cause remains unproven; recovery and final-path success
are observed facts, not proof that the generic queue implementation was repaired.

Independent review also found the128-chunk SDPA alignment regression. Choosing32
for page-aligned starts not divisible128 restores the fused public continuation
contract. Tests now resume at32,96,128,1024 and pass. Profiler capture uses the
balanced trace helper/finally release. Distinct-request fixtures now detect
wrong-request page-row reads that repeated batch inputs could hide.

## Short page-table capacity repair

The delivered mixed-process pytest initially passed sliding but failed full decode
(PCC -0.305 initially, or 0.047–0.076 after refreshing page rows). Fresh-process
controls passed. Source-only AutoDebug (`pytest_AUTODEBUG.md`) and independent
review both found full SDPA K128 reading four page IDs from a two-entry table.
Causal SDPA rounds the read extent to a complete chunk and does not validate
capacity divisibility; aligned row padding is not valid page metadata.

The runtime now selects the largest supported chunk no larger than its preferred
64/128 whose size divides the valid page-table capacity. It uses tensor shapes,
not position readback. The exact previously failing eight-case mixed sequence
passes (`pytest_capacity_fix.log`, 8 passed in33.78s). This supports the fix;
it does not prove the precise arithmetic effect of unspecified padding bits.
The 4096/128 headline and8192-page maximum-context configurations retain their
previous chunk sizes and numerical policy. One/three/five-page regression tests
exercise changed-input/page-row replay without imposing a logical-length limit.
No generic TTNN change, padding-ID fabrication, GC workaround, or cache clearing
was used to conceal the failure.
