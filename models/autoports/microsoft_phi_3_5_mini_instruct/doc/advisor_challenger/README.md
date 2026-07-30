# Phi-3.5 Mini advisor challenger

The shipped batch-32 decoder improved from a frozen best of **0.806821 ms** to
**0.747121 ms**, a reduction of **0.059700 ms (7.40%)**. The incumbent noise
floor was 0.001729 ms. Three final production-harness repeats were 0.748518,
0.747459, and 0.747121 ms.

The incumbent was frozen before capture. The one dense layer kind was captured
at batch 32 with the shipped BFLOAT4_B precision. The advisor considered and
selected DRAM sharding for all four linears. The down projection already
shipped that policy. QKV, output, and gate/up were screened as geometry-only
changes at their shipped HiFi2 fidelity; the IR's traced LoFi compute config
was explicitly not accepted as advice.

## Screening and combinations

The corrected single-chain results were:

- QKV DRAM sharding: 0.837050 ms, rejected.
- Output DRAM sharding: 0.816853 ms, rejected.
- Gate/up DRAM sharding: 0.880201 ms, rejected.
- Two width-sharded RMSNorms: 0.747679 ms, kept.

All six pairwise sets and the cumulative four-chain set were measured at the
same shipped precision and fidelity. No combination beat the norm-only
candidate; the best pair containing it was output-plus-norm at 0.756583 ms.
The norm-only set was therefore written into `tt/optimized_decoder.py`. Its
final standalone production-harness best was 0.747121 ms.

The direct advised RoPE-internal ops summed to 0.948% of the incumbent measured
window and were recorded below the 1% materiality threshold. There are no
sparse-matmul or SSM layer kinds in this model. `nlp_concat_heads_decode` was
the capture's one unfixable op; the production path already supplies its
required sharded input.

## Correctness and iteration

The complete real-weight optimized decoder suite passed 10/10. Batch-32 real
decode PCC was 0.998892 and traced-replay PCC was 0.998864 against the
incumbent 0.995 bar. Batch 1, non-aligned prefill, prefill/decode transition,
and 131072-context coverage also passed.

After applying the norm winner, a fresh op-level report was collected and
`scripts/reconcile.py` was re-run. The script currently does not parse this
branch's MLIR alias and tt-perf CSV spellings, so both machine runs returned
zero rows; `reconciliation.json` contains the audited IR-to-CSV chain
reconciliation used for the measurements.
