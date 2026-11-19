# Full-model final profile reconciliation

Resident token-out: **103.064959568 ms**. Strict useful-weight/KV floor:
**6.414528 ms**; ten-percent ceiling:
**64.145280 ms**. Remaining deficit:
**38.919679568 ms**.
No intermediate traffic or measured kernel time is added to the denominator.

Representative component stack plus entry/terminal/sampler: **104.083627679 ms**;
full-model difference: **-0.978702%**.
Frozen model-local source-manifest match across all inputs: **True**. Consult JSON
for missing/additional/mismatched paths and separate shared-source coverage.
Additional recorded model files are listed; they are not silently removed.

| Region | Merged ops | Merged kernel sum, ms | Full-path weight |
|---|---:|---:|---:|
| token_entry | 2 | 0.002286000 | 1 |
| embedding | 3 | 0.037297250 | 1 |
| layer0 | 256 | 1.768717500 | 35 |
| layer1 | 455 | 2.722637750 | 1 |
| layer3 | 363 | 2.818218500 | 12 |
| final_mixer | 37 | 0.326188750 | 1 |
| lm_head | 3 | 0.655454000 | 1 |
| row_assembly | 2 | 0.046161000 | 1 |
| split_greedy | 31 | 0.426224250 | 1 |
| force_argmax | 15 | 2.680505750 | enclosing/alternative |
| model | 1121 | 8.376960750 | enclosing/alternative |

Weighted35/1/12 eager sum plus entry/embedding/mixer/head/row assembly/split:
**99.939983500 ms**. Raw rank totals: {"0": 96.977957, "1": 95.321708, "2": 96.05837, "3": 97.414919}.
Ordinary merged operations use slowest rank; CCL rows use the report's rank mean.
Enclosing model and terminal regions are never added to their leaves. Eager gaps
are retained in JSON, not treated as clean trace overhead or achievable savings.

Split sampler: **0.426224250 ms** device sum, **0.413549%**
of resident token-out. Force-argmax is a separately measured alternative.
Review raw sampler gather shapes and native operation contributions in JSON.
The all48 boundary includes split sampling/device feedback; caller-read latency
and TTFT are separate fields. Component traces use reduced residency and frozen
reachable inputs; their estimate is not an exact all48 measured lower bound.

Coverage and input hashes, all selected raw geometries, sorted weighted kernel
groups and report advice are in the companion JSON. Kernel groups may combine
multiple semantic uses and are upper bounds, not predicted savings. Independent semantic candidate/advice review is recorded below. Final stage
acceptance remains false; this accounting does not waive the failed floor gate.

## Independent final-source review

The final recipe completed successfully:68,024 host/device/final operations,
24 signposts,12 regions, no trace IDs, and1,121 model operations per rank. All
model leaves partition the model; the terminal parent equals mixer plus head.
The v1 count was1,247, so the126-op reduction is expected from selected exact
fusions. All72 model source hashes and all three shared sampling/CCL hashes
match `final_sources_v4.json` across profile, component and resident artifacts.
The resident before/after snapshots are identical. No missing shared hash
remains in these actual final artifacts.

Natural128-token prefix staging and four eager warmups put the measured decode
at132→133 on every rank. The causal scorer sees33 complete groups/two live
compressed-key tiles. This is one reduced eager sample per region; the clean
full48 trace instead evolves positions128..254 and activations. Weighted eager
attribution fell112.435613500→99.939983500 ms, a12.495630 ms reduction, whereas
full48 resident timing is103.064959568 ms. Their3.124976068 ms difference is a
cross-regime reconciliation residual, not directly isolated host overhead.

Clean component traces give L0=1.847758537, L1=2.860673498 and L3=2.914571163 ms.
The35/1/12 stack is102.507076256 ms; entry/terminal/sampler adds1.576551423 ms,
for104.083627679 ms. Full48 is1.018668111 ms below that representative estimate.
Component inputs are frozen reachable reduced-model activations, not all48
activation history; recurrent/QSA state advances within each window. Exact
first replay, unchanged input snapshots and stable program cache pass. The
estimate refutes a large avoidable orchestration gap under these limits; it is
not a measured physical lower bound.

The strict useful-weight/KV floor stays6.414528 ms. Final resident efficiency
is6.223772%, still below10%; **38.919679568 ms must be removed** to meet64.145280 ms.
No intermediate writes, padded score storage, repeated scans or measured
kernel duration enter that denominator. Overall stage acceptance remains false.

## Final kernel budgets and exact-policy constraints

All values below are35/1/12 weighted merged device costs, with terminal work
once where applicable. They are impossible zero-cost ceilings, not expected
savings. JSON stores the independently checked v2 row ranges and raw call IDs.
GR-down product/reduce is inside its local-chain row; score epilogue is inside
score/selector. Do not add either nested subset again.

| Semantic group | Final budget, ms | Evidence and disposition |
|---|---:|---|
| GR norm |5.264255|RHS fusion selected; retain full2560-feature local norm tree.|
| GR-down local projection |11.455886|Includes10.354546 product/reduce; native R8/R16 probes were slower. Alternative reduction geometry remains research.|
| GR-down four slices + concat |3.310132|Legacy path remains. Fast candidate failed integrated mixed-prefill equality and was reverted.|
| GR-up local projection |3.832833|R16 exact kernel kept in decoder; terminal mixer uses its existing product/reduce path.|
| GR post-gather local sum/scale/cast |1.039800|Exact local rank sum kept; transport unchanged.|
| GDN-output local projection |4.008054|R16 exact kernel and assembly/cast kept.|
| Residual writers |0.558523|Exact writer kept, including original intermediate BF16 rounding.|
| QSA score + selector |3.777024|Native live-prefix dot remains; exact score epilogue subset2.682108 is a bounded untried fusion target.|
| QSA ordered indexer |5.260452|Already native; new read reuse/occupancy must preserve selected arithmetic tree.|
| QSA selected-KV gather |2.865048|65 data-movement cores; reader optimization untried, exact selected-ID/padding semantics required.|
| QSA SDPA |1.662240|Inherited HiFi2/FP32 destination retained; fixed2080 selected slots.|
| GDN QKV |4.764354|Native exact projection; arithmetic/load decomposition remains fixed.|
| GDN recurrence |2.553868|12-core selected state recurrence; finer partitioning untried.|
| Routed MoE SwiGLU |5.547505|Selected BFP4 weights/BF16 activation path retained.|
| Selected CCL transport |6.130895|Synchronous ring transport; overlaps semantically with enclosing layer costs, not an additional layer term.|

The highest-confidence remaining small target is the exact QSA score epilogue:
ReLU44.584us, clone44.295us, fill44.984us, head reduction43.712us and
scale45.934us, across12 layers. Its2.682108 ms zero-cost ceiling cannot close
the38.919680 ms deficit. Preserve exact per-head and padded-zero reduction
order, scalar rounding, selected-ID ordering and device-position transitions.
Skipping ranking when all groups fit can change SDPA accumulation order.
GR-down's10.354546 ms product/reduce ceiling is larger, but tested R8/R16 fused
geometries lose; an alternative exact reduction decomposition is untried, not
an established fix. None of these budgets establishes a feasible38.9 ms gain.

## Selected runtime precision and geometry

The following IDs refer to the joined raw final CSV, not only high-level op
names. Generic operations are identified by their native source and operands.

| Calls / semantic role | Raw runtime evidence |
|---|---|
|16047105/16048128 GR down|FP32 `[1,1,1,10240]` activation and `[1,1,96,10240]` weights/products. Multiply110 cores; accurate SFPU W-sum3 cores, HiFi4, FP32 DEST, approximation off.|
|16062467 GR up|FP32 activation320, FP32 weights2560x320; exact native projection110 cores, FP32 row-major scratch; source selects16 rows/worker with exact fold/tree.|
|16182272 GDN out|BF16 activation6144 and weights640x6144; exact native40-core projection produces FP32 scratch,20-core assembly then explicit BF16 cast.|
|16070658 local rank sum|BF16 `[1,4,1,2560]` gathered streams, FP32 local residual operand/output,80 cores; source preserves exact four-rank tree.|
|16186371 residual write|BF16 full2560-wide local row and gate,80-core exact writer with original BF16 rounding boundaries.|
|16992256 packed causal scores|BF16 query `[1,4,1,128]`, BF16 key capacity65536x128;110-core native packed scorer emits FP32 logical `[1,1,4,65536]`. Device live prefix bounds dot work; output storage remains capacity-sized.|
|16999424 selected KV|BF16 K/V `[8192,1,32,256]` to2080x256 selected rows,65 data-movement cores.|
|17007618 SDPA|BF16 Q six local heads, selected BF16 K/V2080 slots;64 cores, HiFi2, FP32 DEST, approximation off, qchunk32/kchunk128.|
|17175553 LM head|BF162560x62080 rank-local weights in DRAM width shards;20 storage cores,3 readers,block2, per_core_M1/N97,32x3104 L1 output shards; HiFi4/FP32 DEST, approximation off. Raw program union80 cores.|
|16265216 routed SwiGLU|BFLOAT4_B weights, BF16 input/output,80 cores; inherited routed policy preserved.|

The report's8-core DRAM-sharded heuristic is not the head's actual80-core
program union or20-core shard layout. Modeled FLOP utilization above100% must
not be treated as physical achievement or a reason to lower fidelity. Native
program flags not serialized in generic-op attributes are verified from the
source hashes and implementation; they are not falsely described as raw CSV
attributes. No `compact_gr_down` kernel appears in the selected raw graph.

## Sampler and advice closure

Split greedy uses0.426224250 ms device time and0.481938393 ms in the separate
clean component trace. It is0.413549% of full48 token time. Local
`TopkLargeIndicesDeviceOperation` is0.266644 ms on110 cores, physical k32,
row-major BF16 `[1,1,32,62080]`. It dominates the small sampler region, not decode.
The two selected gathers are32→128 candidates per physical row, BF16 values
and UINT16 indices,6.26975/6.8835us, synchronous two-link ring. No generic
`TopKDeviceOperation`, force-argmax or full-vocabulary gather is selected.

The deliberately measured force alternative costs2.680505750 ms, including
1.441860750 ms full-vocabulary gather and0.938621 ms argmax. Both strategies
select token1292, the real frozen logits' global maximum4.34375. These eager
numbers remain distinct from historical clean split/force alternative timing.

Advice dispositions are recorded in JSON and `perf_summary.json`:

- Tracing is already kept; eager op gaps do not imply new trace savings.
- Head20/3/block2 is kept. Larger adapted blocks8/10 are measured slower;
  specific L1 collisions are recorded without rejecting an entire family.
- Exact GR-up/GDN-output, norm RHS, writer and local rank sum are integrated.
  GR-down/router native variants are measured rejections.
- Compaction is an integrated correctness rejection, despite isolated speedups.
- Async sampling gather is measured slower; persistent-output-buffer alternatives
  remain untried, not implicitly rejected.
- Generic lower-fidelity advice conflicts with the explicit user constraint and
  inherited rejection ledger. No precision frontier or replicated stream is used.
- Small-matmul L1/block/subblock configurations and exact QSA epilogue/reader
  changes remain stop-rule hand-off work with unknown gains. Source/config and
  non-aligned logical-length checks remain necessary.

This closes final decode attribution and advice review, not the failed strict
FUNCTIONAL gate. Six-bucket prefill attribution is complete in [prefill_profile_analysis_v1.md](prefill_profile_analysis_v1.md), with its separate scope and metadata caveats.
The analyzer and subsequent semantic row/operand audit use no hardware.
