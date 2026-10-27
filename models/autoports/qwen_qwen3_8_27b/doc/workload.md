# SWE-bench Verified workload

The user requests accurate, practical SWE100 and SWE500 on QB2, with local device
tests and the QB2 dispatch CI authorized. All64 layers and native262144 context
are retained. [context_contract.json](context_contract.json) is copied unchanged
from the retained Experiment26 snapshot; it records prior capability evidence,
not a new native-context accuracy claim.

The initial comparison uses eight fixed tasks, C8 clients, B32 versus B8 resident
capacity, existing prefix caching/shared snapshots/block256, mini-swe-agent2.2.8,
one attempt, seed1234, thinking/default xhigh, temperature1/top-p.95/top-k20,
16384 output tokens, agent6300/request3600/suite7200 seconds. The separate
medium pilot changes the reasoning instruction and has its own score. These
policy/limit choices are described in [SWE investigation](swe_optimization/README.md).

Optimize completed task throughput and normal completion, preserving all
outcomes, rather than token throughput alone. Record startup, prefill, decode,
cached input, malformed/capped outputs, queue/occupancy and verifier/cleanup tail
where observable. HTTP concurrency, resident capacity and actual active rows
are distinct. No numerical accuracy tolerance or missing GPU reference is invented.
The current TTI main checked at67f6e150c0e743eadce9543fe8f0ad7738519bb0
has no Qwen3.8 row in its GPU accuracy-reference file; its Verified catalog row
still cites Pro61.7 provisionally. The published model card also reports Pro,
so no Verified parity verdict follows from that number.

## Representative traffic

Historical measured B32/C8 SWE8 run40395139503 observed median prompt27170
tokens and empirical p95 prompt69313. The per-session maximum prompt+reply p95
was73544.75 tokens (planning proxy73545), maximum78492, with three missing usage
records. These eight sessions include outer timeouts, and their p95 is neither
a population estimate nor simultaneous resident demand. The baseline controls
retained in this checkout expose exact task checksums and request usage.

Fixed native-chat4096/128 full-model runs are regressions and path comparisons;
they do not replace scored SWE or typical long-context traffic. Subsequent replay
must include long retained prefixes, small appends, mixed lengths/partial batches,
slot reuse and pressure. Every recurrent layer needs both its convolution carry
and FP32 state restored at the exact checkpoint, in addition to full-attention KV.

## Existing per-device memory evidence

These are historical selected B32/shared256 preparation measurements from the
Experiment26 `agentic_evals/resume-shared256-memory-ledger.json`, not new serving
peaks. Count unique backing once; logical48 snapshot references share16 pools.

| Component | Bytes/device | Basis |
|---|---:|---|
| Recurrent snapshot backing | 20,153,843,712 | 1571 global IDs ×16 unique pools ×801792 |
| Full-attention KV backing | 3,500,539,904 | 1571 ×256 ×8704 |
| Active recurrent state | 1,231,552,512 | B32 ×48 layers ×801792; independent per layer |
| Remaining weights/runtime/CCL/metadata | 6,511,306,240 | Allocator residual, not a guessed component sum |
| Allocation checkpoint total | 31,397,242,368 | Measured allocator checkpoint |
| Warmed-before-capture checkpoint total | 31,811,013,120 | Measured checkpoint; not serving peak |
| Reported allocator capacity | 34,044,513,280 | Allocated plus free at these checkpoints |

The inherited tensor-payload inventory reports6224093184 weight bytes and
72089600 CCL DRAM bytes per device, with separate prefill/packed decode projection
copies and norm layouts. Those are older inventories and are not remeasured
components of the table's residual. Trace-region configuration is134217728 bytes.
The cache backing budget is23661297664 bytes; actual backing23654383616 leaves
6914048 bytes of rounding. Runtime peaks remain unmeasured.

The worker's sizing hook subtracts seven reservation IDs per resident slot plus
one global null ID. At block256 it returns344576 token units for B32 and373248
for B16. This virtual sizing hook does not establish guaranteed live conversation
capacity: attention IDs and recurrent checkpoints compete in a global allocator,
and retained histories consume IDs too. Target32 long conversations are unproven.

## Capacity hypotheses for wider serving

[Source-based arithmetic](swe_optimization/cache_geometry_forecast.json) holds
the same23.66GB backing budget and shared-state representation:

| Resident capacity / block | Global IDs | Physical attention token slots including null | Worker token sizing hook |
|---|---:|---:|---:|
| B16 /256 |1571|402176|373248|
| B16 /512 |1368|700416|642560|
| B32 /256 |1571|402176|344576|
| B32 /1024 |1088|1114112|883712|

Larger checkpoint blocks buy more token capacity but can replay more of the
prefix, change eager prefill work and increase scratch. These are candidates,
not selected settings or measured memory/quality wins. Validate allocator
checkpoints, exhaustion/backpressure, every layer's restored state and scored
task time before selection. The decoder grouping candidate preserves cache
precision and capacity; it is independently tested and initially disabled.
