# Reading the profiler on this tower

Companion to [PERF.md](../PERF.md). What `tt-perf-report` and the per-RISC counters do and do not
tell you about the Janus-Pro vision tower, and the three ways their output invites a wrong
conclusion. Each of those cost real time before it was understood.

Every figure was measured on a Wormhole N150. Where a number describes an older tree, it says so.

## Where the remaining time is

### BRISC at 100% of op duration does not mean reader-bound

In every matmul reuse factory, in0 is read on **NCRISC** while in1-read *and the output
writeback* share **BRISC** (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:735,791,808,822`;
1D at `..._1d_program_factory.cpp:599,673`). BRISC's last instruction is the output's final
`async_write_barrier` (`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:692`), so **BRISC ==
op duration is structural**.

**Trap: this table looks like proof that the matmuls are reader-bound and that no math-side
change can help.** That reading is wrong twice over — sharding the outputs removed *writer* work
for −0.375 ms, and change 22 then took 15-17% off the same matmuls by *lowering* fidelity. BRISC
being pinned at 100% tells you nothing about which half of its job is the bottleneck.

Per-RISC means, measured before the output sharding landed:

| shape | op us | TRISC1 busy | BRISC |
|---|---:|---:|---:|
| 576x1024x3072 (qkv) | 81.44 | 68.7% | 99.98% |
| 576x1024x1024 (wo) | 31.52 | 73.7% | 99.95% |
| 576x1024x4096 (c_fc) | 85.08 | 88.1% | 99.98% |
| 576x4096x1024 (c_proj) | 86.45 | 90.5% | 99.98% |

### DRAM bandwidth is not the limit

These ops run at 55-84 GB/s against 288 GB/s peak, 19-29%. BRISC is busy without saturating
bandwidth, i.e. bound by **transaction count and issue overhead** rather than bytes per second.
That is why cutting bytes (bfloat8_b) helped, why blocking parameters mostly did not, and why
removing the writer's one-NOC-write-per-output-tile loop helped as much as it did.

### FLOPs % is the wrong target

Its denominator is `tflops_per_core(fidelity) x cores_used` (`perf_report.py:744,815`), so it does
not penalize leaving 16 of 64 cores idle, and it *rises* when you raise fidelity. Lowering
fidelity to go faster makes the number **worse**.

The whole change log is the strongest example. Between the baseline and the current tree every
body matmul got 2-3x faster and **every one of them reports a lower `FLOPs %`**:

| shape | us before | us after | FLOPs % before | FLOPs % after | fidelity |
|---|---:|---:|---:|---:|---|
| 576 x 1024 x 4096 | 179.2 | 81.6 | 54.7 | 35.2 | HiFi4 → LoFi |
| 576 x 1024 x 3072 | 138.5 | 48.8 | 53.0 | 42.6 | HiFi4 → LoFi |
| 576 x 4096 x 1024 | 130.3 | 55.1 | 75.1 | 50.2 | HiFi4 → LoFi |
| 576 x 1024 x 1024 | 50.8 | 18.2 | 48.2 | 37.9 | HiFi4 → LoFi |
| 576 x 4096 x 4096 | 490.9 | 313.9 | 79.8 | 62.4 | HiFi4 → HiFi2 |

Lowering fidelity raises peak-per-core — 1.028 to 3.639 TFLOPs for HiFi4 to LoFi — so the
denominator grew 3.5x while achieved FLOPs grew 2.2x. c_fc checks out exactly:
`54.7% x (2.20 / 3.54) = 34.0%` against the measured 35.2%. **Reading this column as a score would
say the work made every matmul worse.**

Worked example of the trap. `c_fc` reports 45.2% against `c_proj`'s 74.0%, which reads as a bad
config worth fixing. Moving `c_fc` to 2D reuse changed nothing about its speed — 83 us either
way — but the reported FLOPs% jumped to 59.1%, purely because 2D uses 48 cores where 1D uses 64.
**The outlier was the denominator, not the op.**

### What is structurally closed

- **`NlpCreateHeads` 1.167 ms and `NLPConcatHeads` 0.425 ms.** 576 rows are 18 tile-rows, so
  nothing that parallelizes over rows exceeds 18 cores; the sharded variant caps at 16. Neither
  elimination nor core-count tuning is reachable — see [`DEAD_ENDS.md`](DEAD_ENDS.md) for the citations.
  That is 1.59 ms that cannot be touched from Python.
- **SDPA 1.607 ms.** Fidelity, DST and chunk sizes are all spent.
- **LayerNorm 0.935 ms.** Every program-config knob swept; the 48-core grid is fixed by the chain,
  not by the norm.

### What the profiling did not establish

Stated so the findings above are not over-read:

- **The split of BRISC's time between reading in1 and writing the output is unknown.** TRISC1 has
  not been re-summed since the norm and output sharding landed, so the per-RISC table above
  describes an earlier tree. Any claim about whether further in1 work would pay rests on that
  split, and it has not been measured.
- **The aligner's 0.521 ms of matmul has not been characterised** beyond its op time. It is the
  one module still at HiFi2 with bfloat16 activations, and unlike the body it feeds the language
  model directly, so whether it can absorb narrowing is an open measurement, not an inference.
  **Closed 2026-08-20** — see below.

## The aligner, characterised

`aligner fc1` and `c_fc` are the same 576x1024x4096 with a byte-identical program config, and
fc1 took 235.8 us against `c_fc`'s 72.3 us. Measured on 2026-08-20
([CANDIDATES-2026-08-20.md](CANDIDATES-2026-08-20.md)), the 3.27x splits as:

| difference | worth |
|---|---:|
| `fp32_dest_acc_en` True → False | **−22.4 us** on fc1, −17.9 on the 4096x4096 |
| output bfloat16 → bfloat8_b (DRAM) | −4.9 us on fc1, −13.4 on the 4096x4096 |
| `out_subblock_w` 4 → 8, DST budget freed | **+7.5 us**, i.e. worse |
| fidelity HiFi2 vs LoFi | not measured; it feeds the language model |

Only the second row is in the tree (change 30). The first is measured and deliberately not taken —
it costs 82% of the aligner gate's slack, see [DEAD_ENDS](DEAD_ENDS.md#measured-and-rejected) — so
read the table as a characterisation of where the time is, not as a list of available levers.

So the gap is mostly **fp32 dest accumulation**, and it is not the output format and not the
subblock. The aligner **can** absorb narrowing: bfloat8_b on the read-once intermediate costs
1.5e-5 of the 0.9999 aligner gate, a third of what turning off fp32 accumulation costs.

One thing to carry forward: fc1's reported DRAM utilisation falls 13.1% → 9.9% when its output
narrows — the write genuinely halves — and its time moves 2%. Both aligner matmuls are classified
writer-bound by "BRISC duration dominates", and halving the write did almost nothing. That is
[the section above](#brisc-at-100-of-op-duration-does-not-mean-reader-bound) restated on a
different module: **BRISC dominating tells you nothing about which half of its job binds**, and
here neither half was bytes.
