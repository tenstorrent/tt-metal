# ESM-2 TTNN implementation notes

SPDX-License-Identifier: MIT

Companion to `docs/esm2.md` (op map + oracle-verified semantics) and
`docs/benchmark.md` (measured evidence). This file records HOW the port is
engineered and WHY each non-obvious choice was made.

## Graph shape

Per layer (identical op order to `reference_layers.py`, the CPU twin):

```
cast(stream, bf16) -> layer_norm -> fused QKV linear [1280 -> 3*1280]
  (q-scale 64**-0.5 folded into the fused q rows; scale commutes with rotary)
  -> split_query_key_value_and_split_heads (q,v [B,H,L,D]; k [B,H,D,L])
  -> rotary: 6-op halves rotation per tensor on cached sign-folded bf16
     tables (pre-duplicated cos / cat(-sin,sin); 9 -> 6 ops, bit-identical,
     .json)
  -> matmul(Q,K^T) + additive -1e9 mask -> fp32-compute softmax composite
  -> matmul(P,V) -> concatenate_heads -> attn_out linear (fp32 out)
  -> fp32 residual add -> layer_norm -> ffn1 -> gelu -> ffn2 (fp32 out)
  -> fp32 residual add            (x33)
head: final LN -> dense -> gelu -> LN -> decoder (tied) -> logits
```

`ttnn.transformer.scaled_dot_product_attention` is NOT used: its attn_mask
input is broken on tt-metal-fd80faa3 (additive 5.3e-1 / multiplicative
3.5e-1 NRMSE, .md), so the port keeps the BERT-style
manual attention path.

## Key engineering decisions (all measured)

1. **Tile-padding to Lt=32k with pad tokens + mask, slice back to L.**
   ttnn eltwise ops honor logical bounds but matmuls compute over physical
   tiles; garbage rows beyond L pollute real outputs (L=3 logits 0.134).
   The additive mask covers every physical column; outputs sliced to L.
2. **fp32 residual stream** with casts at bf16 consumers: the bf16 residual
   add dominates hidden error otherwise (sim 3.31e-2 -> 1.94e-2).
3. **fp32-compute softmax composite** (cast/max/sub/exp/sum/divide): the
   runtime's `ttnn.softmax` measures ~2.2e-2/site with 0.5–0.7% row-mass
   deficit vs an exact fp32 softmax of identical bf16-valued scores; no
   kernel-math variant reproduces it, so the port composes it from
   fp32-valued ops.
4. **out_fp32 matmul sites (fp32 outputs where device bf16-out excess
   bites)**: device bf16-out matmuls carry contraction-dependent excess (up
   to 3.9× over the clean model at the gate misses). Per-site A/B
   (`tests/probe_out32_ab.py`, .: base misses the gates
   (long 0.0673 / single-residue 0.0929); all-8 sites land 0.0359/0.0292 at
   +24% p50 (L=1026). **Measured trim (+
   `88c1636e` re-gate, artifact `benchmarks/artifacts/opt_h2_out32_trim.json`):
   the default is now {qkv, pv, ao, ffn2}** — long −12.8% p50 with suite max
   0.0433 → 0.0424; the hidden miss swaps single-residue (0.0433 → 0.0306,
   passes) ↔ long (0.0359 → 0.0424, misses); sr-logits anchor 0.0292 →
   0.0170 (all-8 numbers remain the pre-trim era's anchors). The scores
   fp32 path must round to bf16 BEFORE the additive mask (fp32 [B,H,L,L] add
   throws on this runtime, ..
5. **Host-side fp32 exceptions**: rotary tables, additive mask, embedding
   token-dropout rescale, final logits/hidden cast — matching the oracle.

## Known measured limits (gates NOT widened; details in docs/benchmark.md)

- **Argmax near-tie row (l=22 of the shared bringup protein)**: margin
  0.0280 abs / 0.0030 rel vs clean-bf16 error 0.0961 at that row → flips
  under any bf16 policy. A100 bf16 flips 4/8 of its own cases.
- **Hidden-row implementation-floor miss (one row per source era)**: at the
  all-8 era, single-residue hidden 0.0433 > 0.04 sat at the runtime's
  truncation-mode activation-rounding floor (CPU floor studies predict
  0.039–0.042; RNE floor 0.019; every candidate lever sized and rejected —
  round-removal levers do not help under truncation semantics at this case;
  the rounding mode itself is runtime/kernel-level, outside the in-graph
  lever set). Under the current trim source the miss is long hidden 0.0424
  (suite max improved to 0.0424) — the accepted, measured sr↔long trade of
  .
- **erf-exact gelu tested and REJECTED — compensating error balance
  (precision note)**: `ttnn.erf` is exact (8.6e-08 NRMSE vs torch) and an
  erf-exact gelu composite matches the FP32-erf policy to ~1.1e-07 (vs
  1.58e-04 for the default piecewise-CDF bf16 gelu), yet END-TO-END every
  affected NRMSE got WORSE (sr logits 0.0170 → 0.0436, long hidden 0.0424 →
  0.0440, plus a destructive interaction with scores-fp32). The default
  gelu's deviations sit inside a compensating balance with the other
  rounding sites; "fixing" gelu to exact erf is a measured regression.
  Receipt `5e9a21d2…`, artifact `benchmarks/artifacts/opt_h3_erf_gelu.json`.

## Tests

- `tests/test_portable_model.py` — portable, host-only, no device.
- `tests/test_ttnn_bringup.py`, `tests/test_op_fidelity.py`,
  `tests/test_offline_encoder_layer.py`, `tests/test_hidden_mini.py` —
  component gates used during bring-up.
- Development probes (attribution, floor studies, A/B comparisons) were used
  during porting but are not part of the shipped test set.

All Python files carry `SPDX-License-Identifier: MIT` headers.
