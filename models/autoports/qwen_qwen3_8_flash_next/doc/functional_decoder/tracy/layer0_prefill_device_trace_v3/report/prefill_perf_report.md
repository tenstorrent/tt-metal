Readable view of the official tt-perf-report output. Only trailing spaces/tabs were removed inside the fenced text.

Exact original: [prefill_perf_report.txt.xz](prefill_perf_report.txt.xz). TXT SHA256 `e9ed2f95bd3370fd606cc25398c2a613874213ac135a6f183a1dd03fcf84671b`; XZ SHA256 `9fcf9a6d5aaeba75e74ce9b2a31234ac2dbdd82b04fc65f04c1ff29bc95bf046`.

```text
Detected CSV format: v2.1 (with device arch and worker core count)
Using user-specified architecture: blackhole (CSV detected blackhole)
Architecture: blackhole, Worker cores: 120
Tracing mode enabled. CSV will not be sorted by 'HOST START TS'.
Using operations after 'PERF_PREFILL'.
Using operations until 'PERF_PREFILL_END'.
Detected data from 1 devices. Merging device data...
Warning: Unclassified operation 'WHOLE_TRACE_PREFILL_AGGREGATE' found. Please add to OPERATION_CATEGORIES for proper classification.

🚀 Performance Report 🚀
========================

ID  Total %  Bound  OP Code                        Device   Device Time  Op-to-Op Gap  Cores  DRAM  DRAM %  FLOPs  FLOPs %  Math Fidelity
-----------------------------------------------------------------------------------------------------------------------------------------
 3  100.0 %         WHOLE_TRACE_PREFILL_AGGREGATE        0   424,281 μs                  110
-----------------------------------------------------------------------------------------------------------------------------------------
    100.0 %         1 device ops, 0 host ops, 0 signposts            424,281 μs          0 μs
```
