# Stage review

Verdict: **clean-pass**

## Required work

None.

## Verified closure

- Standalone optimized implementation with no functional fallback.
- Sixteen exact-current-source correctness/stress tests pass; the watcher-10
  subset passes five tests cleanly.
- Non-aligned, page-boundary, multi-user, paged-cache consumption,
  long-context, trace determinism, and b1/b32 coverage are present.
- Final nonprofiled timing is 1.609/0.466 ms prefill/decode at b1 and
  19.745/0.794 ms at b32, beating functional 2.084/1.009 and 36.950/1.332 ms.
- The final profiler proves BFP4/LoFi material matmuls, BFP8 cache,
  DRAM-sharded decode matmuls, packed QKV/gate-up, composite SDPA, and zero
  host operations.
- The phase-specific RoPE opportunity is adapted, measured, semantically
  validated, allocation-accounted, and evidence-backed as slower.
- Human-readable profiler reports contain advice tables; console logs are
  populated.
- Block-width descriptions, functional command environment, and default-SDPA
  prose are corrected.
- The 131072-token context capability is preserved, with default and opt-in
  persistent allocations documented.

## Residual risks

- Exact-context prefill at 131071/131072 uses zero inputs; nonzero semantic
  coverage reaches 32769. Exact shapes and the shared long-context
  implementation pass.
- Functional prefill timing uses synthetic values while optimized prefill uses
  recorded activations. Shapes, dtypes, context, and execution regime are
  comparable, and the difference is disclosed.
- `tt-smi` was unavailable. Hardware runs closed cleanly and watcher level 10
  reported no errors.

The reviewer performed only read-only inspection and did not open hardware.
