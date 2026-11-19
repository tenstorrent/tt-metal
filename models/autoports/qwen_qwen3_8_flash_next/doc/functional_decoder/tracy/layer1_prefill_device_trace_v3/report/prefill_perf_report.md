Readable view of the official tt-perf-report output. Only trailing spaces/tabs were removed inside the fenced text.

Exact original: [prefill_perf_report.txt.xz](prefill_perf_report.txt.xz). TXT SHA256 `ec11dfff4e77588adcf85cbb4e31c745f61faef71cbe740a943474080d6653a8`; XZ SHA256 `fbb112d40eb723d6e7e38e1c05360ae65ef15db630221d537db3b9dbf8a6d125`.

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
 3  100.0 %         WHOLE_TRACE_PREFILL_AGGREGATE        0   455,605 μs                  110
-----------------------------------------------------------------------------------------------------------------------------------------
    100.0 %         1 device ops, 0 host ops, 0 signposts            455,605 μs          0 μs
```
