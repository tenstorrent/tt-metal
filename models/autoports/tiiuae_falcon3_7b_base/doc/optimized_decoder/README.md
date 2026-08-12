# Falcon3-7B-Base optimized decoder

This directory contains the single-device optimized-decoder stage for
`tiiuae/Falcon3-7B-Base`. The implementation is `tt/optimized_decoder.py`.
It owns the measured TTNN prefill/decode paths and preserves paged KV-cache,
determinism, public DRAM output, and arbitrary logical sequence lengths.

## Selected implementation

- All QKV/O/gate/up/down weights use BFP4 and LoFi; KV cache uses BFP8.
- Prefill weights are DRAM interleaved. Decode owns separate 8-bank DRAM
  width-sharded copies, preventing the large-M prefill corruption diagnosed by
  AutoFix.
- Decode QKV/O use 32-way activation/output shards; gate/up/down use the
  selected 48-way shard geometry. The profiler reports 80 program
  participants for these DRAM matmuls. The residual/norm chain is 32-core L1
  width-sharded.
- Gate-product and down-input share the same physical 48-way shard map,
  eliminating one decode reshard.
- Prefill uses grid 11, `in0_block_w=8`, and internal row chunking. Public
  callers have no alignment or `seq_len % chunk == 0` restriction.
- Maskless causal decode SDPA and dedicated TTNN prefill/decode SDPA remain;
  split gate/up beats the legal packed alternative.

The required shard-advisor pass was run on the rewritten dense block. Its
`report.json` and `final_ir.mlir` are preserved. Exact advisor matmul and
residual/input recommendations remain executable controls, but the coherent
DRAM-48 decode family wins primary batch-1 latency.

## Headline evidence

All final comparisons use real layer-14 weights and recorded layer-14 inputs
from HF revision `bf3d7ed586cb22a921520e2d681a9d3d7642cde8`.

| Batch | Path | Prefill PCC | Decode PCC | Warmed prefill | Traced decode |
|---:|---|---:|---:|---:|---:|
| 32 | Functional BF16 | 0.99999636 | 0.99999972 | 4.148475 ms | 1.797630 ms |
| 32 | DRAM BFP8 control | 0.99999637 | 0.99999995 | 4.530043 ms | 1.134656 ms |
| 32 | Advisor all-BFP4 control | 0.99998189 | 0.99999842 | 3.274530 ms | 0.773084 ms |
| 32 | Selected DRAM-48 all-BFP4 | 0.99998189 | 0.99999839 | **3.262737 ms** | **0.768483 ms** |
| 1 | Optimized BF16 advisor | n/a | 0.99999995 | n/a | 1.402281 ms |
| 1 | Advisor all-BFP4 control | n/a | 0.99999882 | n/a | 0.652849 ms |
| 1 | Selected DRAM-48 all-BFP4 | n/a | 0.99999881 | n/a | **0.644047 ms** |

Against functional BF16 at batch 32, the selected path is 21.4% faster in
prefill and 57.2% faster in traced decode. Aligning the down-input shard removes
one reshard and makes DRAM-48 0.6% faster than advisor at batch 32 and 1.3%
faster at primary batch 1. Both tests use 100 trace replays and assert the
ordering.

A genuine 31-token activation plus two following tokens validates prefill,
queries, cache updates, and sequential decode. All selected-policy metrics are
above 0.99; the minimum K/V metric is 0.995557. Genuine max-contract
128-token prefill reaches 0.999806 PCC. Eight repeated same-slot writes and
trace replays are bitwise deterministic.

The final same-run Tracy capture reports 215 prefill ops and 42 decode ops per
iteration, with zero host ops. Device-plus-gap is 3.717517 ms prefill and
0.792598 ms decode; same-run wall is 3.836255/0.801771 ms. Every dominant
attention and MLP matmul is BFP4/LoFi. Human-readable `tt-perf-report` tables
and advice are saved beside the CSV reports.

## Key artifacts

- `shard_advise/report.json`, `shard_advise/final_ir.mlir`: mandatory advisor
  output for the rewritten dense attention+MLP block.
- `activations/layer14_inputs.{safetensors,json}`: reproducible 17-, 31-, and
  128-token target activations.
- `results/precision_frontier_seq31/`: recorded two-step precision/fidelity
  selection with durable policy identities.
- `results/final_policy_topology/`: final-policy advisor/DRAM, grid-8/11, and
  packed/split controls.
- `AUTODEBUG.md`, `AUTOFIX.md`, `results/autofix/`: diagnosis and repair of
  the original DRAM-BFP4 prefill/cache corruption.
- `results/final/`: final batch-32/batch-1 and profile summaries.
- `tracy/dense_layer/ops.csv`, `*_perf_report.{csv,txt}`: selected-path
  profiler evidence.
- `watcher.log`, `watcher_device.log`: watcher-clean optimized correctness run.

See `work_log.md` for exact commands, the topology audit, advisor decisions,
all selected/rejected configurations, hashes, anomaly ledger, and completed
optimize checklist.

## Scope and limitations

The preserved single-device contract is batch 32 and cache length 128, with a
separately measured batch-1 decode path. DRAM decode keeps phase-specific
duplicate projection weights, but post-selection length-128 validation shows
that advertised capability is unchanged. Multichip, full-model assembly, LM
head, sampling, generation, and vLLM are outside this stage.
