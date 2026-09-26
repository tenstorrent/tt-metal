# Final whole-layer device accounting

The CPU audit passes for selected runtime`daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b`.
It independently reconstructs both native4096-input/128-decode,B1,C1 profiles,
including every operation and intra-layer gap. No layer kinds are averaged.
[Machine-readable evidence](final_roofline_audit.json) retains source/raw/report hashes,
per-operation estimates, all128 replay windows and same-run reconciliation.

```sh
python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/audit_final_roofline_v7.py
```

| Quantity | Sliding attention | Full attention |
| --- | ---: | ---: |
| Complete prefill device µs | 221262.879 | 187438.157 |
| Complete decode device µs, mean128 | 881.243 | 899.725 |
| Useful prefill FLOPs | 882,489,950,208 | 1,215,408,635,904 |
| Useful FLOPs / theoretical peak % | 0.601072 | 0.977213 |
| Estimated decode DRAM bytes | 110,493,448 | 106,384,012 |
| Estimated DRAM rate / theoretical peak % | 24.488993 | 23.093873 |
| Native operations per decode | 123 | 126 |

Prefill contains one warmed measured invocation; decode includes all128 traced
positions4096–4223. Each device denominator spans first firmware start to last
firmware end of the complete layer. Inter-replay input refresh and host gaps are
outside the individual device windows. Summed kernel times are diagnostic only,
never a roofline denominator. Raw measured cycle conversion is0.7407407407ns/cycle.

The common theoretical peak is`120 * 4096 * 1.35e9 = 663.552 TFLOP/s` at LoFi and
512GB/s DRAM for one participating Blackhole ASIC. The available runtime grid
has110 workers; the denominator remains the ASIC's theoretical peak.
The [P300 specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html)
lists two120-core ASICs and1024GB/s card bandwidth. Mixed fidelity and SFPU work
mean these ratios are useful-work/peak estimates, not measured FPU utilization
or memory-controller counters. Values are not clamped.

Useful FLOPs count multiply-add as two operations, S4096, hidden2816,16 query
heads, shared width2112, expert width704 and eight active experts per token.
Projection/router terms and causal QK/PV terms are independently reconstructed
in the JSON. Padding, masked attention, extra union-expert work, normalization
and transcendental work are excluded from the useful numerator; their execution
time remains included. DRAM estimates use native operand shapes/dtypes with BFP
exponent storage, indexed8/128 active expert weights, metadata and rounded KV
reads. Extra per-core rereads, NoC traffic and profiler writes are excluded.

Native rows verify BFP8 sliding/BFP4 full expert gates, BFP4 down and LoFi,
UINT16 eight-index operands with compact eight-slot outputs, BF16 attention output
through projection input, selected direct/DRAM projection families and row-major
RoPE lookup. Native minimal prefill QKV is K8/HiFi4 sliding and K16/HiFi2 full;
full minimal output is K8/LoFi. Every explicit block and compute flag is checked
against native attributes by the audit. The report tool's inability to parse
`MinimalMatmulConfig` is documented separately from actual runtime configuration.

| Same profiled run, µs | Sliding attention | Full attention |
| --- | ---: | ---: |
| Theoretical DRAM transfer floor | 215.808 | 207.781 |
| Complete device window | 881.243 | 899.725 |
| Refreshed-position host loop | 896.452 | 914.920 |
| Fixed-position host replay median | 880.335 | 907.798 |

Same-run reconciliation uses raw host signposts and the final two-command journal.
The refreshed host loop and fixed-position replay measure different regimes;
negative arithmetic gaps are retained and are not attributed to a hardware cause.
Separate unprofiled medians are reported as distinct observations, never used as
device latency or in same-run gap arithmetic.

This accounting audit is independent of correctness and trace-lifetime proof.
[Final validation](validated_v7_validation_summary.json) binds current affected-path
checks and explicitly inherited unchanged branches. The earlier v5 accounting
is preserved in `roofline_audit_v5_historical.md/json`; no historical metric is
relabeled as the selected source.
