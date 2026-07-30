# Phi-3.5 Mini advisor challenger

This fresh batch-32 run shipped one measured change: keep the manual Q/K
rotate-half chain in L1. The frozen incumbent was 0.656191 ms (best of
0.656836, 0.656873, and 0.656191 ms; 0.000682 ms spread). The winning chain's
best repeat was 0.616772 ms, a 0.039419 ms / 6.007% improvement.

## Capture and reconciliation

Phi has one decoder layer kind: dense, 32 of 32 layers. It was captured once at
batch 32 after the incumbent timestamp, with the executed BFP4 projection
policy and the BF16 DRAM-sharded eligibility option enabled. All five
projection weights were considered and advised for DRAM sharding; all five
were already DRAM-sharded in the incumbent. There are no sparse-matmul or SSM
layer kinds. `nlp_concat_heads_decode` was the one unfixable traced op because
the advisor did not synthesize its required sharded input.

The stock reconcile command was run against the incumbent CSV. Its parser
cannot currently resolve the captured IR's `#ttnn_layout` aliases, so it
emitted no rows; `reconciliation.json` records that incompatibility and the
manual chain reconciliation from the same IR and perf window.

The material chains were:

- RoPE, 28.536% of the device window: kept. L1 placement measured 0.616772 ms
  and passed real-weight batch-32 PCC 0.9999926011 at the 0.995 bar. LongRoPE
  PCC was 0.9999964813.
- MLP projection geometry, 23.420%: rejected at 0.664378 ms.
- Attention projection geometry, 12.391%: rejected at 0.663052 ms.

The advisor's block widths 12/12/12/32 are not legal on the shipped 16-core
partition because the per-core K tile counts are smaller. Following the legal
family instead of treating the first constraint failure as rejection, an
8-core control and each width were measured independently. The full advisor
geometry measured 0.660539 ms. Pairing it with RoPE measured 0.621091 ms;
RoPE-plus-attention was 0.623875 ms and RoPE-plus-MLP was 0.623654 ms. All
interactions lost to RoPE alone.

The `compute_config` and LoFi fidelity in `final_ir.mlir` were treated as traced
state, not advice. Every candidate retained the shipped BFP4/LoFi precision.

## Artifacts

- `incumbent.json`: frozen policy, timestamps, repeats, and noise floor.
- `shard_advise/dense/{report.json,final_ir.mlir}`: batch-32 dense capture.
- `reconciliation.json`: chain-ranked screening results.
- `final.json`: every measured set, pairwise interactions, oracle, and bounded
  iteration history.
- `tracy/incumbent/` and `tracy/final/`: op-level profiler and tt-perf-report
  CSVs for the incumbent and kept winner.

The shipped source is `tt/optimized_decoder.py`; it defaults
`advisor_rope_l1_chain` on and retains every other incumbent policy setting.
