# Gemma-4 26B A4B advisor challenger

Outcome: **NO CHANGE**. The frozen incumbent remains the fastest measured
oracle-passing decoder. Its 46-layer weighted batch-1 decode latency is
1.275091 ms, with a 0.001094 ms same-config noise floor.

The incumbent was measured three times before any advisor process ran. The
metric weights the two representative layer kinds by the shipped topology:
39 sliding-attention layers and 7 full-attention layers. The best repeat is the
incumbent, and ties within the measured noise floor go to it.

## Capture and reconciliation

One successful shipped-precision capture was produced for each layer kind at
batch 1. BF16 DRAM-sharded matmul eligibility was explicitly enabled. The
sliding capture considered four DRAM-sharded candidates and the full capture
considered four. `final_ir.mlir` compute configs were treated as traced state,
not advice; all candidates retained the incumbent fidelity.

The advisor recalled two material disagreement chains:

- Norm/residual sharding covers 29.666% of the weighted incumbent device
  window. The coherent R11 candidate measured 1.177125 ms, but sliding decode
  PCC was 0.9947948362 against the incumbent 0.995 bar. It is rejected.
- Attention-projection DRAM sharding covers 16.481% of the weighted window.
  Sliding QKV+O plus full QKV measured 1.329224 ms, 4.244% slower than the
  incumbent. It is rejected on performance.

The advisor agreed with the incumbent on the packed MLP gate/up and down
DRAM-sharded strategy, and on the full-attention O projection. Those are not
disagreements and were not reintroduced as fake candidates.

Both layer kinds contain routed `ttnn.sparse_matmul` experts. That op is
terminal in the tracer, so the expert path is uncapturable and carries 100% of
the layer topology. No topology rewrite won, so there was no valid trigger for
another capture and no stale-profile reranking iteration.

## Combination decision

No challenger survived both performance and correctness screening. Therefore
there are no screened winners to combine or pairwise-test. `final.json`
records every measured set and selects the measured, oracle-passing incumbent.
The shipped `tt/optimized_decoder.py` is intentionally unchanged.

The capture target needed two mechanical accommodations: explicit declared
Q/K/V head memory configs while tracing, and stopping immediately before the
terminal sparse expert op. Neither accommodation was shipped into the decoder.
Failed pre-capture compatibility attempts produced no advisor report and are
not counted as completed captures.
