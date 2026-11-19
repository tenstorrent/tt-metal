Readable view of the official tt-perf-report output. Only trailing spaces/tabs were removed inside the fenced text.

Exact original: [decode_perf_report.txt.xz](decode_perf_report.txt.xz). TXT SHA256 `29cf6187eeec18f75b9cd056731f82739ab0a5fcbf7ef0c1d331d80b2e591b03`; XZ SHA256 `b0a9665f546ae6e6e1ba1d542508b9c3472a57c8b0c82ef978757720eabad7af`.

```text
Detected CSV format: v2.1 (with device arch and worker core count)
Using user-specified architecture: blackhole (CSV detected blackhole)
Architecture: blackhole, Worker cores: 120
Tracing mode enabled. CSV will not be sorted by 'HOST START TS'.
Using operations after 'PERF_DECODE'.
Using operations until 'PERF_DECODE_END'.
Detected data from 1 devices. Merging device data...
Warning: Unclassified operation 'WHOLE_TRACE_DECODE_AGGREGATE' found. Please add to OPERATION_CATEGORIES for proper classification.

🚀 Performance Report 🚀
========================

ID  Total %  Bound  OP Code                       Device   Device Time  Op-to-Op Gap  Cores  DRAM  DRAM %  FLOPs  FLOPs %  Math Fidelity
----------------------------------------------------------------------------------------------------------------------------------------
 3  100.0 %         WHOLE_TRACE_DECODE_AGGREGATE        0    47,199 μs                  110
----------------------------------------------------------------------------------------------------------------------------------------
    100.0 %         1 device ops, 0 host ops, 0 signposts             47,199 μs          0 μs
```
