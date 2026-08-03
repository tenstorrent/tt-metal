# AutoDebug: Phi-3.5 recorded-state decode accuracy

## Headline

The original b32 PCC near `0.90` and the later cache-consistent PCC near
`0.988` are not one undifferentiated precision problem.

1. The `0.9018359` log used a zero prefix and predates the current RoPE
   batch-layout repair. Its lane pattern is consistent with the now-removed
   direct sharded transpose: users 0-7 are near `0.9997+`, then user 8 falls
   to `0.8972`. Duplicate recorded inputs at users 3/15 and 13/19 have
   identical references but different actual norms. That is lane-dependent
   corruption, not ordinary BFP quantization.
2. With 127 real recorded prefix activations in the paged cache, the remaining
   optimized b32 failure is localized by controlled contrasts to **two decode
   attention optimizations**:
   - fused adjacent-pair RoPE; and
   - the explicit `(8,8)`, dynamic-chunk decode SDPA configuration.
3. Each optimization is insufficient on its own for the recorded full-state
   b32 gate. Disabling both gives PCC `0.9999631` with BFP8/HiFi2/BF16 cache.
   Fused cache update can remain enabled.

The smallest evidence-backed shipped defaults are therefore:

```text
fused_rope = False
explicit_decode_sdpa = False
fused_paged_cache_update = True
```

The current `OptimizationPolicy` has those defaults. The final BFP4/LoFi/BFP8
policy passes recorded prefill/decode at b1/b32, including recorded-state
decode PCC `0.9992643 / 0.9989930`.

## Direct observations

### Recorded data and harness state are coherent

- `layer0_inputs.safetensors` contains BF16:
  `token_embeddings [159,3072]`, `prefill_128 [128,3072]`, and
  `decode_position_127 [1,3072]`.
- A static equality check proves:
  `prefill_128 == token_embeddings[:128]` and
  `decode_position_127 == token_embeddings[127:128]`.
- The b32 harness constructs user `u` from prefix
  `token_embeddings[u:u+127]` and matching next activation
  `token_embeddings[127+u]`. The last window is rows 31-157 with next row 158,
  so all accesses fit the 159 recorded rows
  (`optimized_decoder_perf.py:41-67`).
- The reference cache fill converts canonical reference K into the same
  adjacent-pair basis only when fused RoPE is selected
  (`optimized_decoder_perf.py:70-102`). Optimized-prefill-populated cache gives
  essentially the same failing result as reference-filled cache, so the
  reference-fill adapter is not the cause.

### The old functional “control” was not a real-state control

`recorded_activation_functional_control.log` compares recorded current
activations against `_reference_decode_zero_prefix`; its cache prefix is zero.
It passes b32 at `0.9996182`, but does not test attention over the 127 recorded
keys.

The corrected control uses functional prefill to populate those keys and then
decodes the matching next activations. It passes b32 at `0.9997310`
(`recorded_cached_functional_b32.log`). The anomaly is therefore optimized-path
specific, not bad recorded data or an HF reference error.

### Pairwise matrix

All rows below use the same recorded b32 windows and matching next
activations:

| Change from optimized BFP8/HiFi2/BF16-cache path | PCC | Verdict |
|---|---:|---|
| Fused RoPE + explicit SDPA | `0.9880456` | fail |
| Manual RoPE only; explicit SDPA retained | `0.9862429` | fail |
| Default SDPA only; fused RoPE retained | `0.9912487` | fail |
| Manual RoPE + default SDPA; fused cache retained | `0.9999631` | pass |
| Full functional-attention trio, including non-fused cache | `0.9999631` | pass |

The equality of the last two rows refutes fused cache update as a necessary
part of the failure. Additional controls also refute:

- BF16 versus BFP8 cache: `0.9880456` versus `0.9880681`;
- BFP8/HiFi2 versus BFP4/LoFi projection policy: approximately `0.9880`
  versus `0.9876`;
- permuted versus identity page tables: exactly `0.9880456`;
- fused versus separate cache update: `0.9880456` versus `0.9885409`;
- reference-filled versus optimized-prefill-filled cache: `0.9880456` versus
  `0.9886233`.

The passing manual/default pair also passes b1 at `0.9999488`. With the shipped
BFP4/LoFi/BFP8-cache defaults, recorded-state decode passes b1/b32 at
`0.9992643 / 0.9989930`
(`recorded_cached_final_correct_attention_decode.log` and
`final_recorded_perf_correctness.log`).

## Source-level boundary

The optimized fused path permutes Q/K weight output coordinates and RoPE
tables into adjacent real/imaginary pairs
(`optimized_decoder.py:300-306`), then uses
`rotary_embedding_llama` in decode (`optimized_decoder.py:824-869`).
The permutation is mathematically valid and is applied consistently to Q, K,
tables, and injected cached K. Source inspection does not prove a logical
pairing error.

The optimized SDPA override fixes an 8x8 grid, disables exp approximation, and
selects dynamic Q/K chunks (`optimized_decoder.py:207-212`); it is passed at
the paged decode call when `explicit_decode_sdpa=True`
(`optimized_decoder.py:958-967`). The functional path uses canonical
rotate-half and leaves the SDPA program configuration at its default
(`functional_decoder.py:400-420,481-489`).

The controlled matrix proves the intervention boundary: both optimized
attention choices are below the `0.995` contract under 127-key recorded state,
and the canonical/default pair passes without changing cache update, paging,
weights, MLP, or downstream head concatenation.

It does **not** yet prove whether either miss is:

- finite-precision/reduction-order sensitivity (the adjacent-pair permutation
  changes tile-local reduction order);
- a particular explicit-SDPA grid/chunk lowering issue; or
- a low-level layout defect before SDPA.

That distinction is not required to choose safe shipped defaults.

## Ranked hypotheses

1. **Confirmed policy boundary: explicit decode SDPA and fused adjacent-pair
   RoPE are each unsafe for this full-state b32 gate.** High confidence.
   Single-toggle failures and the joint fallback pass directly establish this.
2. **The first large divergence is inside RoPE/QK/SDPA, most likely a
   layout- or reduction-order-sensitive lowered path.** Medium confidence.
   The functional real-state path passes, while paging, cache dtype, cache
   producer, update mode, and projection precision contrasts do not explain
   the failure.
3. **Downstream head concat/projection amplifies a smaller attention error.**
   Low-to-medium confidence. It may explain final PCC magnitude, but cannot be
   the primary policy boundary because the passing pair leaves that downstream
   code unchanged.
4. **Recorded metadata, shifted-window selection, page routing, or reference
   cache construction is wrong.** Refuted by static metadata checks,
   functional real-state pass, identity-page contrast, and optimized-prefill
   cache contrast.

## Focused follow-up experiments

These are useful for recovering either optimization later; they are not
release blockers for the safe defaults.

1. Capture per-user/per-head PCC at these exact boundaries: Q/K/V head
   creation, post-RoPE Q/K (inverse-permute fused outputs in test code),
   paged-SDPA output before concat, post-concat, and output projection. The
   first bad boundary assigns ownership.
2. Sweep the SDPA override one field at a time against the manual-RoPE control:
   `(8,8)` versus device grid, `k_chunk_size=0` versus 32, exp mode, and
   explicit versus default compute config. Do not re-enable explicit SDPA
   based on zero-prefix tests.
3. Replicate one recorded prefix/current activation across all 32 users and
   assert per-lane equality; then permute user order. This distinguishes
   remaining core/lane mapping from data-dependent numerical sensitivity.
4. Re-run the historical zero-prefix duplicate-input b32 diagnostic on current
   code to retire the old `0.9018` artifact explicitly.
5. If fused RoPE is revisited, require both raw post-RoPE Q/K parity and
   nonzero-cache b1/b32 decode. Dot-product invariance alone is not sufficient
   evidence under tiled finite-precision SDPA.

## Conclusion

Ship manual Phi rotate-half plus default paged decode SDPA, retain fused cache
update, and keep the recorded 127-prefix b1/b32 gate. This is the smallest
configuration change supported by the full pairwise evidence and the final
passing recorded-state run.
