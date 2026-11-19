Readable view of the official tt-perf-report output. Only trailing spaces/tabs were removed inside the fenced text.

Exact original: [decode_perf_report.txt.xz](decode_perf_report.txt.xz). TXT SHA256 `6e4f9e9c5fd88b4060b48a34df2ece4c36b7b02d490610202ec779229b45bab6`; XZ SHA256 `a5a6f1e74110dc582b08b43246fa77a7a98cf418b5a5c7dc43bc48f2720a2133`.

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
 3  100.0 %         WHOLE_TRACE_DECODE_AGGREGATE        0    48,119 μs                  110
----------------------------------------------------------------------------------------------------------------------------------------
    100.0 %         1 device ops, 0 host ops, 0 signposts             48,119 μs          0 μs
```
