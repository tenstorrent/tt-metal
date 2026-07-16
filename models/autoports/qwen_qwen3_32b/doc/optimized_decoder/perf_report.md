# Final profiler interpretation

The final Tracy run profiles one real representative decoder layer with the
selected source and keeps prefill/decode signposts separate. Watcher was not
enabled during profiling. The authoritative CSV is:

`tracy_final_all_bfp4/reports/2026_07_16_16_56_12/ops_perf_results_2026_07_16_16_56_12.csv`

## Decode

`tt-perf-report` reports 99 device operations and 3.606 ms device time across
three traced replays: 33 operations and 1.202 ms device time per replay. It also
reports 77 us total op-to-op gaps, or 26 us/replay. The same profiled process
measured 1.243 ms/replay at the wall, leaving about 15 us/replay after device
time plus reported gaps. The longer, unprofiled 200-replay result is 1.217 ms.
This reconciles profiler overhead/jitter without attributing it to model work.

One replay's dominant projection rows are:

| Projection | Device time | Runtime dtype/fidelity | Observed bandwidth |
|---|---:|---|---:|
| Packed QKV, 32x5120x10240 | 100-101 us | BF16 x BFP4, LoFi | 260-261 GB/s |
| O, 32x8192x5120 | 80 us | BF16 x BFP4, LoFi | 263-264 GB/s |
| Gate, 32x5120x25600 | 233-234 us | BF16 x BFP4, LoFi | 281 GB/s |
| Up, 32x5120x25600 | 233 us | BF16 x BFP4, LoFi | 281-282 GB/s |
| Down, 32x25600x5120 | 221 us | BF16 x BFP4, LoFi | 297 GB/s |

These rows prove the selected precision/fidelity policy reached runtime. The
five matmuls consume about 0.87 ms, roughly 72% of device time. Every projection
is visibly BFP4/LoFi at runtime. The rows remain marked `SLOW` at 50-58% of the
tool's reference and are the dominant remaining opportunity.

The tool could not find another output subblock for the BFP4 rows. The selected
DRAM-sharded config class does not expose an output-subblock field; legal
role-specific `in0_block_w`, core geometry, packed gate/up, and advisor 1-D
alternatives were therefore measured instead. HiFi2's accuracy advice was also
tested: it raises latency to 1.971 ms for the MLP and 1.368 ms when only
attention retains HiFi2, while LoFi passes both real and synthetic PCC gates.

The final replay has no tilize/untilize or final copy. Required composite/core
layout boundaries cost 1-3 us. The report's only gap advice is two 8-us gaps
near input/output reshape boundaries per three replays; because the path is
already traced, the tool estimates only 3 us total recoverable gap time.

## Prefill

The signposted prefill contains 217 device operations, 4.922 ms device time,
and 0.433 ms reported gaps. The final 25-run median wall latency is 5.477 ms.
Large rows include:

| Operation | Device time |
|---|---:|
| Packed QKV, 544x5120x10240 | 247 us |
| SDPA | 202 us |
| Head concat | 125 us |
| O, 544x8192x5120 | 198 us |
| Gate/up, 544x5120x25600 | 593 / 593 us |
| Down, 544x25600x5120 | 561 us |

The large op count includes required per-user cache fills for the public
batch-32 cache layout. `tt-perf-report` recommends moving input 0 to L1 for the
large matmuls. An adapted whole MLP chain moved the shared norm result once for
gate/up and the gated activation for down; PCC was preserved but latency
regressed from 5.635 to 6.219 ms, so DRAM-interleaved prefill inputs remain.
The final `in0_block_w=10` wins the 25-iteration legal block-8 comparison
(5.590 versus 5.638 ms in that sweep).

## Roofline/accounting

`profile_run.json` calculates physical-tile weight and KV-read traffic for one
batch-32 decoder-layer step:

- projection weights plus two rounded-position KV reads: 276,496,384 bytes;
- p300c peak DRAM assumption: 512 GB/s decimal;
- traffic-only lower bound: 0.540032 ms.

The profiled wall result is 2.30x that lower bound; the stable 200-replay result
is 2.25x and corresponds to 227.1 GB/s if only the counted bytes are used. This
is deliberately a lower-bound model: it excludes activation, cache update,
norm, SDPA scratch, layout, and output traffic. The per-op report confirms that
all projection rows are bandwidth/compute limited below the tool reference,
with the three large MLP rows consuming most projection time. The remaining
gap is primarily projection kernel/dataflow efficiency plus required
non-matmul work, not host dispatch.

## Artifacts

- Same-run profile metadata, wall result, config, and roofline:
  `results/final/profile_run.json`
- Final functional/optimized PCC and 25/200 wall timing:
  `results/final/before_after.json`
- Final Tracy CSV:
  `tracy_final_all_bfp4/reports/2026_07_16_16_56_12/ops_perf_results_2026_07_16_16_56_12.csv`
- Candidate matrices: `results/candidates/*.json`
