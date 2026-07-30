# Qwen3.6-27B advisor challenger

The incumbent remains the shipped decoder after measuring every material
advisor recall chain and their pair. This is a measured no-change result.

At decode batch 32 the frozen model-weighted repeats were 937.803776,
937.597152, and 937.344112 ms across 16 full-attention and 48
linear-attention layers. The incumbent is the best repeat, 937.344112 ms, and
the frozen spread is 0.459664 ms. Both layer kinds passed the incumbent PCC
0.995 oracle with exact repeated-state determinism.

## Capture

The incumbent was frozen before either valid capture. Each representative layer
kind was captured once at batch 32 with the shipped
`bfp4_all_dram_w8` policy. The full-attention capture contains the shipped
cache, SDPA, RoPE, per-head norms, projections, and MLP; static TTNN slices
replace Python tensor subscripting solely because `TracedTensor` does not
implement Python indexing. All five full-attention and all five
linear-attention material projections were traced as BFP4, considered for DRAM
sharding, and advised DRAM-sharded. The incumbent CSVs prove all ten already
ship DRAM-sharded.

Linear attention's gated-delta core is terminal to the tracer (`softplus`,
prefix-scan/reduction, and state assignment). It occurs in 48 of 64 layers
(75%). Its input/output projections and MLP envelope were still captured at
the shipped shapes and precision. Full attention is capturable;
`nlp_concat_heads_decode` is recorded as an unfixable optimizer constraint
because it requires sharded input, but the trace continues through the output
projection and MLP.

## Material chains

The supplied `reconcile.py` was run against both incumbent CSVs. This branch's
`tt-perf-report` emits shape-qualified `OP Code` plus `Device Time`, and its IR
uses aliased `#ttnn_layout` definitions; the supplied parser recognizes neither
format and emits a zero window. `reconciliation.json` therefore records the
direct, reproducible mapping from the same CSV row indices and authoritative
IR.

Two full-attention chains cleared the 1% summed-share threshold:

- RoPE: advised transpose/repeat/neg/multiply/add rows sum to 3.675% of the
  decode window; the complete edge-to-edge RoPE envelope is 9.377%. Keeping
  the chain sharded passed PCC/determinism but produced 1.234214, 1.234547,
  and 1.234506 ms. Its best is 26.645 us slower than the frozen full-layer
  best of 1.207569 ms.
- Q/K per-head RMSNorm: 1.315% of the decode window. The legal 64-core-Q /
  32-core-K block-sharded chain produced 1.208878, 1.209890, and 1.208918 ms,
  regressing the best full-layer result by 1.309 us. The lower 32/16-core
  bracket was slower at 1.210932 ms best. Height sharding is rejected by a
  hard TTNN LayerNorm constraint. The advisor point is already the largest
  integral-tile block geometry, so there is no legal above bracket.

Both candidates retained the shipped LoFi compute configuration. The
`compute_config` and `math_fidelity` printed in `final_ir.mlir` were treated as
captured state, never as advice.

## Combination and result

The two top chains were also measured together. Their full-layer medians were
1.233023, 1.232352, and 1.232993 ms, still slower than the incumbent.
`final.json` records the incumbent, both norm geometries, the RoPE chain, and
the pair as measured sets. The best challenger single is 937.360064 ms
model-weighted; the pair is 937.745264 ms. Neither beats the frozen incumbent,
and values inside its 0.459664 ms global spread are ties. Ties go to the
incumbent.

No winner was applied, so no profile became stale, no second reconcile
iteration was permitted, and `tt/optimized_decoder.py` is intentionally
byte-identical to the frozen incumbent (SHA-256
`fe5ae5dc9505130bb22511803f48b091df075e40d8601a003af7e0aff8ae4169`).

## Evidence

- `incumbent.json`: frozen batch-32 timing, executed policy, weighting, and
  oracle.
- `tracy/incumbent_{full,linear}_b32.csv`: incumbent op-level reports only.
- `shard_advise/{full_attention,linear_attention}/`: batch-32 reports and
  authoritative final IR.
- `reconciliation.json`: chain shares, screening numbers, and rejections.
- `final.json`: every measured set, best single, best measured set, invariant,
  and bounded iteration record.
