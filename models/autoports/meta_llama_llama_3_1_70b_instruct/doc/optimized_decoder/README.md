# Llama 3.1 70B Instruct optimized decoder

This directory contains the evidence for the optimized, single-device dense decoder layer. The stage is intentionally limited to `tt/optimized_decoder.py`, its tests, and this documentation; it does not start multichip, full-model, generator, or vLLM work.

The selected default uses BF16 activations and norms, BFP4/LoFi weights for every dense projection, an 11x10 2-D prefill matmul family, composite SDPA, and the current-pass shard-advisor 1-D decode topology. The final down projection uses an 11x6 grid, `in0_block_w=4`, `per_core_N=4`, and a 1x4 output subblock. QKV remains packed. Gate/up remains split: full-phase packing requires 2,263,296 bytes of circular buffers per affected core, above Blackhole's 1,572,864-byte L1 limit, while an adapted decode-only packed path is correct but measures 2.1220 ms versus 2.1201 ms for the split checkpoint over 500 replays.

## Result

All numbers use real Hugging Face layer-39 weights, logical prefill sequence 18, five warmed prefill repetitions, and 500 traced decode replays. The primary single-user result is:

| Path, batch 1 | Warmed prefill | Traced warmed decode |
| --- | ---: | ---: |
| functional, BF16 KV cache | 5.059 ms | 4.917 ms |
| final optimized, BF16 KV cache | 3.179 ms | 1.846 ms |
| final optimized, BFP8 KV cache | 3.204 ms | 1.846 ms |

The final BF16-cache path is 1.59x faster in prefill and 2.66x faster in decode at batch 1. BF16 is the selected primary cache policy because BFP8 is 0.78% slower in prefill and 0.015% slower in decode at the primary target. Cache allocation remains caller-owned, and BFP8 remains a supported throughput policy because it wins at batch 32.

The batch-32 scaling result is:

| Path, batch 32 | Warmed prefill | Traced warmed decode |
| --- | ---: | ---: |
| functional baseline | 143.845 ms | 142.906 ms |
| advisor BFP8-attention/BFP4-MLP baseline | 8.692 ms | 2.360 ms |
| all-BFP4 advisor checkpoint, 11x8 down/block 2 | 7.954 ms | 2.281 ms |
| final default, BF16 KV cache | 7.969 ms | 2.114 ms |
| final default, BFP8 KV cache | 8.025 ms | 2.079 ms |

At batch 32, the final default improves prefill by 18.1x and traced decode by 67.6x over the functional path. It improves decode by 10.4% over the 2.360 ms BFP8-attention baseline, 7.3% over the first correct all-BFP4 checkpoint, and 0.30% over the immediately preceding correct 2.1203 ms block-4/11x8 topology. Its real-weight PCC is 0.9999023 prefill and 0.9999545 decode; appended K/V PCC is 0.9929778/0.9933865. The cache delta is material but remains above the functional test contract's 0.99 bar. An eight-step recurrent real-weight test stays above 0.99 at every step, three nonaligned seq=7/batch=13 repetitions are bitwise deterministic, and five traced replays plus input refresh pass.

The exact required `TT_METAL_WATCHER=10` run completed in 62.94 s with 6 passes, 2 expected opt-in skips, watcher asserts enabled, and no watcher sanitizer/assert/error report on the final all-BFP4 11x6 code state.

## Evidence map

- `work_log.md`: topology audit, rewrite decisions, shard-advisor application/rejections, sweep matrix, commands, profiler conclusions, contract analysis, and completed optimization checklist.
- `shard_advise/report.json` and `shard_advise/final_ir.mlir`: mandatory current-pass advisor artifacts.
- `shard_advise/advise_llama70b.py`: capture workload for the rewritten dense block.
- `tracy/*_report.csv`: advice-enabled `tt-perf-report` rows for functional/final prefill/decode.
- `tracy/*/reports/**/ops_perf_results_*.csv.gz`: losslessly compressed raw reduced-layer Tracy op reports.
- `watcher_test_report.xml` and `watcher.log`: preserved exact `TT_METAL_WATCHER=10` final gate.
- `candidate_evidence/README.md` and `candidate_evidence/*.xml`: auditable geometry, packing, recurrence, cache, and 500-replay results.
- `AUTODEBUG.md`: fresh-context diagnosis of the silent DRAM-sharded prefill corruption fixed during this stage.

The repository `doc/context_contract.json` is unchanged. The public layout, sequence, cache, and capacity contract is unchanged; seq=7 proves nonaligned logical lengths remain accepted. The final advisor path also uses fewer persistent weight copies than the rejected DRAM-sharded decode candidate, so there is no evidence-based capacity reduction to advertise.
