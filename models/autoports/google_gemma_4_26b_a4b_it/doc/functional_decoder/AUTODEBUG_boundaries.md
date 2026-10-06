# AutoDebug: wider decode boundary failures

**Resolution:** the integrated repair now passes all nine real reuse cases for
both attention kinds and all 64 batch32 decode slots. See
[`AUTOFIX_decode.md`](AUTOFIX_decode.md) for the measured RoPE, softmax ranking,
cache conversion and router projection causes. The hypotheses below record the
investigation's starting point, rather than the final implementation policy.

Source-only follow-up to `AUTOFIX_decode.md`, prompted by
`reuse_sliding.json` and `reuse_sliding.log` (real layer 0, seed 19). Decode
fails PCC 0.995 at lengths 31 (0.984888), 1025 (0.992513), and 2049 (0.992244),
while lengths 32, 33, 1023, 1024 and a repeated 33 pass. All prefill cases pass.

## Evidence and competing explanations

- The first S31 request fails before any trace reuse or page-owner reuse.
  Reuse cannot explain that failure. Its physical prefill length is 32; decode
  overwrites padding row 31, and attention must read positions 0 through 31.
- Lengths 1025 and 2049 are just beyond chunk/window boundaries. Prefill uses
  1024-token chunks with carried sliding K/V tails; the decode cache path reads
  the paged prefix with a sliding mask. A page-table/rounded-read-window defect
  remains possible, and needs exact-shape cache evidence.
- Cache allocation in this harness covers 4096 tokens, so these cases already
  have ample rounded-window coverage; a simple missing final allocation is
  unlikely. Fresh tables are random permutations of the same 128 physical pages.
- The trace is captured once at S31 and replayed with mutable token, position,
  page-table and K/V buffers. Captured positions are tensor values, not Python
  scalars. Eager-versus-traced output on the same current cache will distinguish
  a trace-state defect from arithmetic shared by both paths.
- The earlier full-attention S33 failure was caused by discrete route flips
  despite high residual PCC. QKV FP32 accumulation and FP32 router arithmetic
  fixed that case; no evidence yet proves the new cases have the same cause.
- HF reference execution is FP32, although checkpoint weights are BF16.
  BF16 HF is a useful control, but does not replace the existing target or
  relax its 0.995 gate.

## Discriminating next experiments

Reproduce exact seed19 inputs, including RNG consumption by page permutations.
Start at S31, which is independent of reuse and long-context behavior.

1. Capture HF and TT attention/residual/router/expert boundaries and compare
   eager output with the recorded trace. Log original HF routes, exact CPU
   routes on the TT residual, and TT routes; include rank-8/9 margins.
2. Upload original HF routing weights while keeping TT expert input and the
   remaining TT branch. A recovered final PCC establishes route selection as
   causal; failure redirects attention to experts or cache state.
3. Read current TT paged K/V in logical order using the actual page table;
   compare against HF cache and current-token K/V. If attention drift is first,
   compare CPU attention on exact TT Q/K/V, then an oracle-cache substitution.
4. Repeat the same controls at S1025 and S2049 with exact RNG inputs. Separate
   fresh-cache/eager results from reused trace results before changing precision.

No additional production change is justified by this source-only pass.
