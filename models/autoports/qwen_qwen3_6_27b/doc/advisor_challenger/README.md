# Qwen3.6-27B advisor challenger

Result: **measured no-change**. The shipped `bfp4_all_dram_w8` decoder
remains the best measured decoder. Ties go to the incumbent.

## Frozen incumbent

Everything ran at decode batch 32. Three warmed 30-replay measurements
were made for both layer kinds before advisor capture. The model metric is
`16 * full_attention + 48 * linear_attention`.

| repeat | full layer (ms) | linear layer (ms) | weighted sum (ms) |
|---:|---:|---:|---:|
| 1 | 1.208874 | 19.135090 | 937.826304 |
| 2 | 1.208242 | 19.120764 | **937.128544** |
| 3 | 1.208673 | 19.128778 | 937.520112 |

The frozen best repeat is 937.128544 ms and the repeat spread/noise floor
is 0.697760 ms. Both kinds passed PCC >= 0.995, exact repeated-trace
determinism, and the batch-row alias check.

The executed policy is sourced from the final batch-32 tt-perf-report
CSVs and selected-policy record: BFP4 attention/MLP weights, LoFi
compute, DRAM-sharded decode, and `max_in0_block_w=8`.

## Capture and reconciliation

Each layer kind was captured at batch 32 after the incumbent was frozen,
using the shipped BFP4 policy. The corrected full-attention capture
contains the packed QKV split and both per-head norms; the earlier
skeleton capture was superseded. Both final captures advise all five
projection matmuls as DRAM-sharded with zero spills.

| kind | layer share | DS considered/advised | recall disagreement |
|---|---:|---:|---|
| full attention | 16/64 (25%) | 5/5 | Q/K per-head RMSNorm chain |
| linear attention | 48/64 (75%) | 5/5 | terminal recurrent norm is 0.432635% |

The ten projection matmuls and four outer norms already match the
advisor's recall classes. Advisor geometry and the traced
`compute_config`/`math_fidelity` were not treated as advice.

`scripts/reconcile.py` was run against each frozen incumbent CSV. Its
parser does not understand this revision's MLIR layout aliases or
`Device Time` column, so the same inputs were reconciled directly. The
Q/K norm rows are one chain: 0.619698% + 0.619781% = **1.239479%**, above
the 1% materiality threshold. The linear recurrence becomes
tracer-terminal at `softplus`/state copy; its gated RMSNorm is 0.432635%
of the linear window and is below threshold.

## Screening and selection

The material Q/K chain was implemented as a single variable: Q uses an
8x8 block-sharded RMSNorm, K uses 8x4, and both cross back at the RoPE
boundary. Its batch-32 correctness oracle passed.

| repeat | full layer (ms) | linear layer (ms) | weighted sum (ms) |
|---:|---:|---:|---:|
| 1 | 1.209297 | 19.133226 | 937.743600 |
| 2 | 1.208835 | 19.125551 | **937.367808** |
| 3 | 1.208976 | 19.127716 | 937.473984 |

The candidate's best is 0.239264 ms slower than the incumbent and lies
inside the 0.697760 ms noise floor. It was therefore reverted. The
candidate op-level report remains in
`tracy/qk_norm_candidate_full_b32/perf_report.csv`.

There was one material chain, so the single-chain measurement is also
the only cumulative challenger set and no distinct pairwise combination
exists. With no screened winner applied, the graph did not change and a
second profiling/re-ranking iteration was not triggered.
