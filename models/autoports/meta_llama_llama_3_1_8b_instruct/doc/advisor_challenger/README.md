# Llama 3.1 8B Advisor Challenger

Outcome: **no change**. The frozen batch-32 incumbent remains the measured
winner and `tt/optimized_decoder.py` is intentionally shipped unchanged.

## Frozen incumbent

The incumbent was measured before the advisor capture with the model-local
batch-32 traced-decode harness and full production tensor shapes.

| Configuration | Best of repeats | Spread | HF-reference PCC |
| --- | ---: | ---: | ---: |
| Shipped 32-core norm/residual chain | 0.621305 ms | 0.006282 ms | 0.999604 |

The effective executed policy is BFP8 attention, BFP4 gate/up/down, BF16
activations and norms, BFP8 KV cache and MLP multiply, and LoFi MLP math. It is
recorded in `incumbent_run.json` and sourced by `incumbent.json`; it was not
inferred from constructor defaults.

The real Llama checkpoint is not cached on this host and outbound Hugging Face
access is disabled. Correctness therefore uses the incumbent's synthetic
full-shape `LlamaDecoderLayer` oracle at PCC >= 0.995. This limitation is
recorded rather than presenting synthetic weights as real-weight coverage.

## Advisor capture and reconciliation

There is one layer kind: dense attention plus SwiGLU MLP, used by all 32
layers. It was captured once at batch 32 after the incumbent measurement,
using the executed shipped policy. Because the traced norm weights are BF16,
capture passed `allow-bf16-dram-sharded-matmul=true`.

The advisor found all five attention/MLP matmuls and selected the same
DRAM-sharded program family already shipped by the incumbent. Those findings
have no set difference to screen. `compute_config` and `math_fidelity` in
`final_ir.mlir` were treated as traced state, not advice; the shipped LoFi MLP
policy was retained.

The remaining material disagreement was the complete norm/residual chain.
From the incumbent `tt-perf-report` CSV it accounts for 3.183% of the measured
603.414 us window when the two norms and adjacent conversion/residual
boundaries are summed. The current reconciliation script was run as required,
but its legacy IR regex did not parse the current `#ttnn_layout` declarations;
`reconciliation.json` records that result and the manual chain reconciliation
against the same CSV.

## Screening and combination

One variable was changed per measurement: norm-chain core count. The advisor's
geometry was used as recall, so legal width-sharded points on both sides of the
incumbent were measured instead of copying its irregular 11/22-core geometry.

| Measured set | Best latency | PCC | Decision |
| --- | ---: | ---: | --- |
| Incumbent, 32 cores | 0.621305 ms | 0.999604 | ship |
| Norm/residual chain, 16 cores | 0.630563 ms | 0.999604 | reject |
| Norm/residual chain, 64 cores | 0.628329 ms | 0.999604 | reject |

The best challenger is 0.007024 ms slower than the incumbent, exceeding the
0.006282 ms incumbent noise floor. Only one material chain existed and it
lost, so cumulative and pairwise winner combinations are not applicable.
`final.json.combination.measured_sets` nevertheless records every measured
set and identifies the incumbent as the best measured set.

No topology rewrite won, so there was no second iteration, re-profile, or
re-capture. Capture count is one for the dense layer kind.

## Artifacts

- `incumbent.json` and `incumbent_run.json`: frozen ordering, policy, repeats,
  noise floor, and oracle.
- `shard_advise/dense/report.json` and `final_ir.mlir`: batch-32 capture at
  shipped precision.
- `tracy/incumbent.csv`: durable `tt-perf-report` op-level ranking oracle.
- `reconciliation.json`: chain-level ranking and screening results.
- `norm16_run.json`, `norm64_run.json`: rejected candidate measurements.
- `final.json`: measured-set comparison, invariant, and no-change decision.

