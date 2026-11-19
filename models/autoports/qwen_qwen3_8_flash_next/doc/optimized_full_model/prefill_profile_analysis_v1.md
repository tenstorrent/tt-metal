# Six-bucket reduced-prefill operation analysis

Each row measures one warmed prefix chunk from position0 through embeddings,
real layers0/1/3, final mixer/head and prefill-output copy. Inputs are uploaded
before the marker; drain, finiteness checks and scheduler updates occur after it.
No sampling, checkpoint save, live trace or measured caller read is included.

| Logical rows | Merged ops | Merged device sum, ms | Raw rank sum min–max, ms |
|---:|---:|---:|---:|
| 1 | 1119 | 8.177248750 | 7.886496000–8.015209000 |
| 2 | 1732 | 17.827547250 | 17.374107000–17.442027000 |
| 4 | 1786 | 22.579935250 | 21.995948000–22.139820000 |
| 8 | 1894 | 34.278168750 | 33.386576000–33.621720000 |
| 16 | 2110 | 53.380053250 | 52.463170000–52.502752000 |
| 32 | 2457 | 87.538737250 | 86.355385000–86.408720000 |

Ordinary operations use the report's slowest-rank merge, collectives its rank
mean. These are eager device sums with one measured chunk per bucket. The six
rows are alternative shapes, not sequential chunks to add together. No row is
an all48-layer latency, traced TTFT, statistical throughput or HF quality result.
The physical32-row padding does not establish semantic equivalence of buckets.

Separate all48 public-generator TTFT is **4705.950625 ms**
for the workload and boundary retained in JSON. It includes reset/setup, complete
prompt prefill, first sample and caller read. Do not subtract a reduced eager
sum from it to infer orchestration or multiply a reduced total by48/3.

Whole-body signposts do not supply individual L0/L1/L3 terms. Consequently the
weighted all48 prefill estimate is **unavailable**, not zero. Raw kernel groups,
per-op geometry and advice are retained for source-guided review; repeated GR
kernels cannot be assigned to a class from an op-name match alone. Any later
class reconstruction needs exact source sequence, boundary anchors and a complete
non-overlapping partition checked independently for all six buckets.

See JSON for source-manifest discrepancies, provenance and grouped kernel costs.
Independent advice review is recorded below: keep tracing distinct from eager gaps;
preserve precision and residual policy; validate geometry changes across logical
lengths; retain integrated compaction rejection. This report does not accept the
stage or claim measured full48 device-prefill accounting.


## Completed semantic review

Coverage reconciles89,740 host/device/final operations,12 signposts and all six
buckets. Every bucket has four equal operation counts, one warmed measured
chunk, no live traces, no measured compilation, finite active logits and final
positions equal to its logical length on every rank. All72 frozen model-local
and three shared-source hashes match final_sources_v4 for profile and resident.
This is finiteness/state evidence, not HF accuracy or serving acceptance.

Operation counts are1,119/1,732/1,786/1,894/2,110/2,457 for logical lengths
1/2/4/8/16/32. Eager device sums are8.177248750/17.827547250/22.579935250/
34.278168750/53.380053250/87.538737250 ms. No full48 scaling is applied.

| Bucket | Ordered-indexer kernel, ms | Generic reduction group, ms | Main FP32 multiply kernel group, ms |
|---:|---:|---:|---:|
|1|0.438893|0.797991|0.310725|
|2|1.607789|4.082427|0.205582|
|4|3.106043|4.209081|0.210282|
|8|6.103999|5.604265|0.257259|
|16|12.099725|6.696409|0.310955|
|32|24.091004|9.365586|0.357297|

These kernel groups may span several roles/layers. The indexer identity is
source-specific; multiply/reduction groups must not be labeled solely GR-down
or attributed to L0/L1/L3 without an independently validated sequence partition.
Bucket32 ordered indexer24.091004ms is27.52% of the reduced body. The next two
groups cost19.392713 and9.365586ms. They identify exact-math optimization targets,
not free savings or a full48 functional-floor denominator.

The native ordered indexer uses100 cores for Q1's five512-wide blocks and
60 cores for Q2..32's1024/1024/512 blocks, with20-core ordered combination.
The program passes logical count in runtime args. Some later-bucket raw CSV
input shapes remain32[2], including bucket32 call21713923; the driver records32
logical rows and all-rank position32. Descriptor metadata reuse is a plausible
explanation, not proven from this report. Keep that metadata caveat explicit
and use source/runtime count plus driver checks rather than relabeling the run
as Q2. Independent final validation, not these shape labels, owns HF correctness.

The selected Q1 GR-up/GDN-output fusions do not extend to multirow precise
projection automatically. Code dispatches the native kernel only for count1;
Q2..32 retain inherited accurate FP32 arithmetic. Any multirow kernel must
preserve original reduction order, casts, mixed-prompt behavior and non-aligned
logical lengths. The steep Q1→Q2 step cannot be explained as tile padding alone.

Head matmul remains about0.644ms at both endpoints: Q1 call2413571 costs
0.643387ms, Q32 call22969346 costs0.645659ms. Both retain BF16 operands,
HiFi4/FP32 DEST, approximation off,20 storage cores,three readers,block2,
32x3104 L1 output shards and80-core raw program union. Routed MoE weights
remain BFLOAT4_B; exact projection products/reductions remain FLOAT32.
The head is not the large-bucket bottleneck. Generic DRAM-core/FLOP heuristics
do not justify changing its accepted geometry or precision.

## Advice disposition

- Tracing is already kept in the public generator. Eager gaps are not extra
  achievable trace savings; these bodies contain no measured host transfers.
- Lower-fidelity suggestions are excluded by the explicit user constraint and
  inherited rejection ledger. No precision frontier or replicated stream.
- Head20/3/block2 is retained; larger adapted blocks were measured slower.
- Small-matmul input-L1/block/subblock options remain untried stop-rule handoff.
  They require supported K geometry and exact mixed/non-aligned validation.
- Ordered-indexer row/load reuse and multirow exact projection are actionable
  research targets with unknown gains, not demonstrated fixes.
- Compaction remains an integrated correctness rejection despite isolated gains.
- Preserve selected CCL dtype/transport and four full-width residual streams;
  aggregate kernel groups do not establish a faster collective alternative.

This finishes the six-bucket attribution/advice review. The all48 device-prefill
estimate remains null because there are no layer markers or validated class
partition. Separate all48 public TTFT remains4705.950625ms for128 input tokens,
including reset/setup, full prefill, first sampling and caller read. Neither
this profile nor advice review changes the failed strict decode floor gate.
