# Optimized decoder work log

## Operation-topology audit

| current topology | candidate/action | evidence and decision |
|---|---|---|
| packed QKV: one 2048x5120 matmul | retain; test BFP8/BFP4 and advisor DRAM sharding | already removes three same-input matmuls; BFP8 wins |
| split heads, optional device RoPE, paged update, SDPA | retain composite device ops; explicit decode SDPA grid/config | no torch/host boundary; correctness and trace replay pass |
| O projection: 4096x2048 | test reduced precision and advisor DRAM sharding | corrected advisor `(16,1,1)` retained for B1 dense decode |
| packed dense gate/up: 2048x6144 | retain single projection; test BFP8/BFP4/LoFi and DRAM sharding | corrected advisor `(8,1,2)` retained for B1 dense decode |
| device fused SwiGLU | retain | no intermediate host or separate SiLU round trip |
| dense down: 3072x2048 | test BFP4/LoFi, explicit 1D geometries, and advisor DRAM sharding | corrected advisor `(12,1,1)` retained |
| packed sparse expert gate/up and sparse down | reduce precision and sweep explicit blocks/subblocks | BFP8 retained; explicit program configs hit the batched-weight contract blocker below |
| residual/norm/layout edges | retain fused path; apply advisor reshards only in B1 dense specialization | corrected complete advisor chain wins 0.254850 ms |

The audit found no remaining repeated same-input projection matmuls or host
conversion. Sparse routing has required row-major/tiled boundaries described
below; no safe adjacent op remains to fold beyond Stage-02 projections/SwiGLU.

### Sparse expert geometry sweep

The final profiler marks packed expert gate/up (`2048x1536`, 61.57%) and
expert down (`768x2048`, 25.74%) SLOW at BFP8/HiFi2, block 2, subblock 1x1.
The following precision-locked explicit configs were tried at decode batch 1
and 32:

| program/config | gate/up | down | result |
|---|---|---|---|
| helper/default | auto block 2, subblock 1x1 | auto block 2, subblock 1x1 | correct; B1 6.662622 ms |
| 1D 8x8, block 2/4, per-M 2, per-N 2, subblock 1x2 | same | same | rejected: 1536 output blocks exceed 64 cores |
| 1D 8x8, per-M 128 | block 2/4 | block 2/4 | rejected: fused-batch program requires unbatched RHS |
| 2D 8x8, per-M 1, per-N 6/8, subblock 1x2/1x4 | block 2/4, per-N 6 | block 2/4, per-N 8 | rejected: 128 expert-batch Y blocks exceed grid |
| 2D 8x8, per-M 16, per-N 6/8, subblock 1x2/1x4 | block 2/4 | block 2/4 | rejected at B1 and B32: explicit program selects fused-batch factory, which requires RHS batch size 1; expert RHS batch is 128 |

Thus current TTNN explicit multicast program configs cannot tune this batched
expert-weight matmul without changing routed-expert semantics. The multiple
independent configurations establish an exact op-contract blocker rather than
using the first error as rejection evidence. Logs are `sparse_geometry_*.log`.

## Candidate sweep

Warmed dense trace replay used 50 samples for baselines/final and the same
harness for intermediate candidates.

| candidate | batch 1 (ms) | batch 32 (ms) | correctness/decision |
|---|---:|---:|---|
| fused BF16 baseline | 0.335773 | 6.026796 | correct baseline |
| BFP8/HiFi2 | 0.287308 | 4.963969 | general-path precision baseline; correct |
| BFP8/LoFi | 0.287742 | not promoted | no useful B1 gain |
| BFP4 attention/LoFi | 0.288024 | not promoted | slower than BFP8 |
| BFP4 MLP/LoFi | 0.287139 | 4.933672 | real dense PCC 0.996756, but real sparse PCC 0.988299 |
| all BFP4/LoFi | 0.287693 | 4.901102 | real dense PCC 0.996201, but real sparse PCC 0.947683 |
| corrected advisor, all four dense matmuls | 0.254850 | n/a | selected for dense B1; PCC 0.999680 |
| explicit 1D block 2 | 0.298478 | n/a | correct-family trial, slower |
| explicit 1D block 4 | 0.256100 | n/a | rejected: output PCC 0.478119 |
| explicit 1D block 8/16 | no run | n/a | exact blocker: 4-tile activation shard width is not divisible by 8/16 |

Separate batch-1 and batch-32 sweeps were required because decode activations
have different shard geometry. No batch-1 matmul family was presumed illegal.
Prefill retained the runtime-selected large-M configs: BF16 fused baseline to
BFP8 optimized is 0.596576 to 0.579367 ms at batch 1 and 12.568575 to
12.137082 ms at batch 32.

## Required shard advisor

Command (bootstrap and capture executed in a separate shell):

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
source tools/ttnn-jit/integrations/agentic-research/shard-advise/scripts/bootstrap.sh
ttnn-advise capture \
  models/autoports/coherelabs_north_mini_code_1_0/tests/advise_north_mini.py:decode \
  --out models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/shard_advise
```

The corrected final report has 22 ops, 19 choices, 4/4 dense matmuls considered and
advised for DRAM sharding, and 2 modeled spills. Recommendations:

- QKV `in0_block_w=8, per_core_M=1, per_core_N=2`: applied in advisor seed.
- O `16,1,1`: applied in advisor seed.
- packed gate/up `8,1,2`: applied.
- down `12,1,1`: applied.
- The complete advisor path is retained for batch-1 dense decode at
  0.254850 ms and 0.999680 PCC. Interleaved copies preserve prefill and the
  general layer/batch contracts.

The earlier 8192-wide erroneous capture was taken through AutoFix.
`AUTODEBUG.md` records a possible multicast partition defect but labels the
root cause uncertain; it is not used to reject the corrected recommendation.

## Correctness, stress, and capacity

`correctness.xml` records 14/14 passing optimized-path tests:

- non-aligned logical prefill lengths 33 and 65;
- dense traced decode;
- sparse sliding/RoPE layer 1 and full/no-RoPE layer 4;
- sparse dynamic trace replay at batch 32 and batch 1;
- paged physical cache slots and bitwise deterministic repeat decode.
- real checkpoint layer-0 weights with token-embedding activations at length
  65 for BFP8 and all BFP4 groupwise policies;
- real checkpoint sparse layer-1 decode for the selected BFP8 policy.

Real-weight precision evidence resolves the synthetic discrepancy: BFP4
dense policies pass (0.996201--0.999045), but real sparse decode fails the
0.995 functional bar (attention-only 0.952695, MLP-only 0.988299, all-BFP4
0.947683). BFP8 passes at 0.999705. Cache precision was swept separately;
the final cache remains BF16.

The final cache remains BF16, so `doc/context_contract.json` is unchanged.
The attempted post-allocation BFP8 cache conversion is rejected: it creates a
transient duplicate cache and fails the 500k serving boundary. The existing
hard fused sparse batch-32 limit of 496928 is neither reduced nor widened;
the 500000 attempt is recorded in `context500000_attempt.log`.

## Profiler conclusions

Tracy captures and `tt-perf-report` tables/CSVs exist for final dense decode
batch 1 and 32, dense prefill batch 1 and 32, and sparse decode batch 1.
Reports were produced with advice enabled:

```bash
tt-perf-report <ops_perf_results.csv> \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --no-color --csv perf_report.csv --summary-file perf_summary.csv
```

The final batch-1 dense report models 160 GB/s (31.2% DRAM roofline). Weight traffic
and numerous short operations dominate; BFP8 reduces traffic without adding
host conversions. Sparse top-k/scatter requires row-major routing tensors,
while expert matmuls require tiled inputs, so its profiler-visible
tilize/untilize boundaries are producer/consumer contract transitions rather
than host fallback; the absolute no-conversion claim was removed.

## Optimize checklist

- [x] Started from fused decoder and kept scope to optimized decoder/tests/docs.
- [x] Audited topology before local tuning.
- [x] Swept precision/fidelity, layouts, program/compute/memory configs at B1/B32.
- [x] Tested BFP4/LoFi and DRAM-sharded dense matmuls instead of assuming illegality.
- [x] Ran mandatory shard advisor and saved `report.json` plus `final_ir.mlir`.
- [x] Preserved non-aligned prefill, paged cache, trace replay, determinism, layer kinds.
- [x] Preserved the context contract and documented the rejected cache policy.
- [x] Measured warmed before/after prefill and traced decode at B1/B32.
- [x] Collected Tracy and advice-enabled `tt-perf-report` evidence.
- [x] Added repeated-run and watcher-clean coverage.
- [x] Independent stage review clean-pass (no required work after two remediation rounds).
- [ ] Local stage-only commit recorded.
