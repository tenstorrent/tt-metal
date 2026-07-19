# AutoFix: fused all-gather matmul stall

## Failure

The first integrated distributed-RMSNorm plus fused all-gather-matmul decode
stopped during warm decode after host-side rank/shape validation. No PCC or
latency was produced. The process was killed after bounded observation; one
`tt-smi -r all` recovered all four p300c devices, and `tt-smi -ls --local`
confirmed them. A later `out_block_w=10` adaptation also stalled and was
recovered by the same bounded procedure.

Raw evidence is retained under `triage/`. ARC heartbeats and four-device
discovery were healthy. Worker call stacks, running-op aggregation, NoC, and
Ethernet state are not claimed: installed `tt-triage` called a `tt_umd.noc_read`
signature incompatible with UMD 0.9.5 even though its summary labeled those
scripts as passing.

## Isolated hypotheses

| Hypothesis | Isolation | Result |
|---|---|---|
| Two links are the exact fused-graph trigger | Hold graph, buffers, shapes, semaphores, and configs fixed; change fused AGMM `num_links=2` to `1` | verified for this graph: one-link completes |
| Two links are generically unsupported | Source/test audit across other hardware/configurations | refuted; this is not asserted as a universal limitation |
| Packed `per_core_N=50` is the first cause | Run the same full graph with one link and unchanged packed geometry | refuted: it passes |
| Semaphore cycling/lifecycle is unbalanced | Compare the same four-operation AG/barrier sequence with one link | refuted: the identical 0,1,0,1 cycle completes |
| Smaller packed `out_block_w=10` removes pressure | Exact legal retry | refuted by stall; recovered and removed |

The retained repair is narrow: only `_fused_decode_column_parallel` uses one
link. Standalone asynchronous all-gather/reduce-scatter and the selected final
path retain both links.

## Verification and disposition

The repaired fused family completes on the real 1x4 layer path:

- prefill PCC 0.999999954;
- decode PCC 0.999962936;
- warmed prefill 3.464048 ms;
- traced warmed decode 0.778513 ms.

This proves the material fusion was adapted through its first TTNN error and
hang rather than rejected prematurely. It remains rejected from the default
because it is 26.2% slower than the selected 0.616712 ms decode. Artifact:
`results/candidate_distributed_norm_fused_agmm_1link.json`.
