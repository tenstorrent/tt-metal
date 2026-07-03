# MoE expert-layout transpose: investigation (negative result)

**Context (#47465):** Tracy profiling showed `PermuteDeviceOperation` = ~97% of per-denoise-step
device-fw (~130 ms/step, ~0 kernel = pure data movement). Origin (via monkeypatching
`ttnn.permute`/`ttnn.transpose`): the shared gemma4 sparse-MoE `Gemma4Experts.prefill_forward` →
`_process_prefill_chunk`, which does `ttnn.transpose(gate, 1, 3)` and `ttnn.transpose(up, 1, 3)`
after the gate/up `sparse_matmul` to reorder into expert-major `[1, E, chunk, I]` for the down
projection.

## Attempt: replace the transpose with a zero-copy reshape (in-repo MoE, no gemma4 edits)
For the denoise canvas every prefill chunk is exactly one tile group (`chunk_len == TILE_SIZE == 32`
⇒ `group_size == 1`), so the dims the transpose swaps have size-1 neighbors and `transpose(gate,1,3)`
is **bit-exactly** reproducible by a direct `reshape` (verified: `diag_moe_transpose.py`, PCC=1.0,
max|diff|=0). Built an in-repo `denoise_moe.py` doing exactly this and routed the denoise path to it.

## Result: bit-exact, but a perf WASH — reverted
- Full-MoE output bit-exact vs shared (`diag_verify_moe.py`: 4/4 calls PCC=1.0, diff=0).
- `PermuteDeviceOperation` dropped 130.4 ms → 8.9 ms (n 150→54). **But** `UnaryDeviceOperation`
  rose 1.3 ms → 122.7 ms at the *same* n=213, and **total device-fw was unchanged** (133.9 → 133.8 ms),
  **eager_ms_per_step unchanged** (339.30 → 339.26 ms, 2-layer).
- `ttnn.reshape` here is a lazy view: the data reorder the transpose was doing is not eliminated,
  just **deferred to the consuming op** (the GeGLU `Unary`), which now pays it. Total work conserved.

## Conclusion
The shared MoE transpose is **necessary data movement** (expert-major reorder feeding the down
`sparse_matmul`), not a wasteful op. No Python-level transpose/reshape rewrite reduces the ~130 ms;
it only relocates it. A real reduction needs a **kernel-level** change — a `sparse_matmul` variant
that emits the down-projection's expected layout directly, or a fused MoE — which lives in shared
ttnn/gemma4, outside the diffusion_gemma no-shared-edits scope. Recorded for #47465 so the layout
reorder is understood as intrinsic rather than a quick Python win.

(Separately landed, real & bit-exact but only on the SDPA *fallback* path: fusing Kᵀ into
`matmul(transpose_b=True)` — commit 981820808bc.)
