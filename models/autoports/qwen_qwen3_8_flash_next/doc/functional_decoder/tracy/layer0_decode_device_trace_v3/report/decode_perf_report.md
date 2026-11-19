Readable view of the official tt-perf-report output. Only trailing spaces/tabs were removed inside the fenced text.

Exact original: [decode_perf_report.txt.xz](decode_perf_report.txt.xz). TXT SHA256 `256321d795abb92070ea81f8f9730d9cc002c6c6033534b4ffd79d421e4d9e95`; XZ SHA256 `459951b905cf8e74e123810b23213b462a367d96b05783505769fa9f25e96e9d`.

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
 3  100.0 %         WHOLE_TRACE_DECODE_AGGREGATE        0    47,216 μs                  110
----------------------------------------------------------------------------------------------------------------------------------------
    100.0 %         1 device ops, 0 host ops, 0 signposts             47,216 μs          0 μs
```
