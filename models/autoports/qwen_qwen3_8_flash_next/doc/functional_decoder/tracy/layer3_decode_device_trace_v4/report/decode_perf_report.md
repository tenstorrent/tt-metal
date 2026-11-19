Readable view of the official tt-perf-report output. Only trailing spaces/tabs were removed inside the fenced text.

Exact original: [decode_perf_report.txt.xz](decode_perf_report.txt.xz). TXT SHA256 `0f084c63a48563edf03113abfb8aa43d1e48f77ceae7355b45f4f3f2a5808b6d`; XZ SHA256 `7a8c8ba75d8ea62d59835dbf8bcb048c6a2941221269fa8d83df91be8ef16f6e`.

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
 3  100.0 %         WHOLE_TRACE_DECODE_AGGREGATE        0   249,930 μs                  110
----------------------------------------------------------------------------------------------------------------------------------------
    100.0 %         1 device ops, 0 host ops, 0 signposts            249,930 μs          0 μs
```
