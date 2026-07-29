# North-Mini-Code-1.0 optimized decoder

The single-device optimized decoder is implemented in
`tt/optimized_decoder.py`. It preserves the functional decoder's prefill,
decode, paged BF16 KV-cache, trace, determinism, non-aligned sequence, and
context contracts.

Final phase policy:

- attention QKV/output weights: BFP8; exact advisor 11x8/11x6 configs were
  applied and measured, then faster eight-bank DRAM-sharded configs won;
  BF16 activations/outputs;
- dense batch-1 decode: advisor-seeded DRAM-sharded BFP4/LoFi packed gate/up
  and BFP4/LoFi down;
- dense prefill and serving batch: BFP8 projections;
- sparse batch-1 decode: routed active-expert `sparse_matmul`, BFP4/LoFi,
  11-core 32/24 K-block policy, L1 intermediates;
- sparse prefill/serving batch: device-resident BF16 batched experts in DRAM.
  Grouped routed sparse and explicit 2D expert candidates were executed but
  lost on TTNN sparse-A/batched-weight contracts and measured latency.

Same-session traced batch-1 decode improves from 0.362345 to 0.200406 ms for
the dense layer, 9.526286 to 0.881623 ms for sliding MoE, and 9.523028 to
0.885650 ms for full-attention/no-RoPE MoE. Every batch-32 point improves.
Final3 routed batch-1 prefill is within 0.41% of baseline (earlier final2
repeats were faster), documented as run-to-run variance in the work log.

See `work_log.md` for commands, PCC/performance matrices, topology audit,
candidate decisions, shard-advisor disposition, watcher/stress evidence,
Tracy reports, and the completed optimization checklist. Mandatory advisor
artifacts are `shard_advise/report.json` and `shard_advise/final_ir.mlir`.
