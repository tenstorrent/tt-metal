Readable view of the official tt-perf-report output. Only trailing spaces/tabs were removed inside the fenced text.

Exact original: [prefill_perf_report.txt.xz](prefill_perf_report.txt.xz). TXT SHA256 `c1432630c60d5cee4ee51200ad3863febf39b5d465f001f0d5f1dc202f5f718c`; XZ SHA256 `ab52b67a6316fee65c17ca6f1f2d36703711b80fb7566367a453b32309a80519`.

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
 3  100.0 %         WHOLE_TRACE_PREFILL_AGGREGATE        0   455,556 μs                  110
-----------------------------------------------------------------------------------------------------------------------------------------
    100.0 %         1 device ops, 0 host ops, 0 signposts            455,556 μs          0 μs
```
