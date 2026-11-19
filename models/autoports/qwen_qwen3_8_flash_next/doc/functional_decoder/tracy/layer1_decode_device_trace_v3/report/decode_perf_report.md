Readable view of the official tt-perf-report output. Only trailing spaces/tabs were removed inside the fenced text.

Exact original: [decode_perf_report.txt.xz](decode_perf_report.txt.xz). TXT SHA256 `42a8391ab2696121f1540b85d75b88029153452a86a586627ac42eaf135402f9`; XZ SHA256 `7d6bdcd07dfa83b77efa42c6ed1e9169d21d3eb828f15761e78af5ba6fe8168e`.

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
 3  100.0 %         WHOLE_TRACE_DECODE_AGGREGATE        0    48,116 μs                  110
----------------------------------------------------------------------------------------------------------------------------------------
    100.0 %         1 device ops, 0 host ops, 0 signposts             48,116 μs          0 μs
```
