Readable view of the official tt-perf-report output. Only trailing spaces/tabs were removed inside the fenced text.

Exact original: [prefill_perf_report.txt.xz](prefill_perf_report.txt.xz). TXT SHA256 `31e96875c5bf3c4206459bd29a28a1b056ea8494b2cbc1d27f0a798b81f98260`; XZ SHA256 `48c1c8e2988853789cfe3378122c072015dd80b986a40523789e1a4a4325e52c`.

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
 3  100.0 %         WHOLE_TRACE_PREFILL_AGGREGATE        0   969,774 μs                  110
-----------------------------------------------------------------------------------------------------------------------------------------
    100.0 %         1 device ops, 0 host ops, 0 signposts            969,774 μs          0 μs
```
