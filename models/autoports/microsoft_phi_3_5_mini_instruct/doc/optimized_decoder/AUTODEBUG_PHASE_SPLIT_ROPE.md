# AutoDebug: phase-specific prefill RoPE

## Hypothesis

Fused adjacent-pair RoPE can be used only in prefill while decode and the paged
KV cache retain canonical HF split-half coordinates.

## Adaptation

- Keep canonical `qkv` and row-major RoPE tables for decode.
- Add an opt-in pair-permuted `qkv_prefill`, tiled pair-basis tables, and the
  fused prefill transformation.
- After fused Q/K RoPE, convert each tensor once to row-major, take BF16
  even/odd strided slices, concatenate even then odd, and tilize. This restores
  canonical Q/K before unchanged SDPA and paged-cache code.
- Reject simultaneous phase-specific and legacy global fused modes.

The API is legal: strided slice supports row-major BF16, and each 48-element
half-row is 96 bytes and buffer aligned.

## Evidence

`autofix_phase_split_prefill_retry.log`:

- b1 prefill PCC 0.998572, 1.626378 ms
- b32 prefill PCC 0.998578, 28.357615 ms

`autofix_phase_split_cache_decode.log`:

- optimized-prefill cache decode b1 PCC 0.998924
- optimized-prefill cache decode b32 PCC 0.998949

The selected manual path is 1.596/19.670 ms in the paired comparison. The
adapter therefore preserves semantics but regresses both batches, severely at
b32, and is rejected from the default.

At max context the opt-in candidate adds 116,590,592 persistent bytes per
decoder instance: 100,663,296 bytes of pair tables, 15,925,248 bytes for the
BFP4 QKV copy, and a 2,048-byte transform. None is allocated by the selected
default.
