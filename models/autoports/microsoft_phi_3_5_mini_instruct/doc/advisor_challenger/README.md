# Phi-3.5 mini advisor challenger

The advisor challenger improved batch-32 traced decode from a frozen best of
0.806270 ms to a shipped best of 0.796420 ms, a 0.009850 ms (1.22%) reduction.
The incumbent noise floor was 0.000501 ms, so this is a measured win rather
than a tie.

The incumbent was measured three times before capture: 0.806270, 0.806771,
and 0.806426 ms. Its executed op CSV established BFP4 weights for qkv,
o_proj, gate_up, and down; down already used a DRAM width-sharded weight.

There is one layer kind (`dense`) across all 32 layers and no sparse, SSM, or
otherwise uncapturable kind. The batch-32 shipped-precision capture considered
and advised DRAM sharding for four of four linears. The supplied reconcile
script was run against the incumbent CSV, but its inline-layout regex cannot
resolve the `#ttnn_layoutN` aliases emitted by this advisor build. The
authoritative IR and executed CSV were therefore reconciled manually by
weight shape. The resulting shares were qkv 4.249%, gate_up 6.247%, and
o_proj 1.118%; each cleared the 1% chain threshold.

Single-chain whole-decoder results were:

- qkv DRAM sharding: 0.798748 ms, kept.
- gate_up DRAM sharding: 0.803843 ms, kept.
- o_proj DRAM sharding: 0.809640 ms, rejected alone.

All pairwise combinations and the cumulative set were nevertheless measured.
The cumulative qkv + o_proj + gate_up set was best at 0.796261 ms, illustrating
the interaction that single-chain screening misses. The production decoder
was then measured independently three times, with 0.796420 ms as its best.

Geometry was derived and measured locally rather than copied from the advisor:
16-core L1 width-sharded activations/outputs, block widths 3/3/6 for
qkv/o_proj/gate_up, and an explicit LoFi compute configuration. The
`compute_config` in `final_ir.mlir` was treated as traced state, not advice.

The shipped decoder passed its incumbent real-weight batch-32 oracle at PCC
0.998931 against the PyTorch reference (required PCC >= 0.995). The final
op-level report is `tracy/final_ops.csv`; kept single-chain reports are
`tracy/qkv_ds_ops.csv` and `tracy/gate_up_ds_ops.csv`. No raw Tracy captures
are retained.

The new projection path is deliberately selected only for batch 32. Other
decode batches retain the incumbent path; the complete optimized-decoder test
module passed 10/10, including batch 1, batch 32, trace replay, prefill/decode
transition, and the advertised 131072-token context check.
