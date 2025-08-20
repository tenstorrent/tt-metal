Moreh Softmax — Program Cache Review

- Status: Reviewed — no program-cache issues found (representative H-small variant inspected)

Summary
- In `MorehSoftmaxHSmallFactory`, reader runtime args start with input base address and per-core tile counts/offsets and static Ht/Wt/scaler/mask. Writer args start with output base address and tile metadata.
- Override updates base addresses for both reader and writer consistently across cores. Other args are derived from hashed shape/op attributes and remain unchanged on cache hits.

Key references
- Factory create: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/softmax_h_small/softmax_h_small.cpp`
  - Reader arg order (input addr first): around L144–L151.
  - Writer arg order (output addr first): around L153–L157.
- Cache-hit override updates:
  - Reader base address updated: around L177–L179.
  - Writer base address updated: around L181–L183.

Notes
- Other parallelization variants follow the same override pattern (update buffer addresses per core); no evidence of stale arguments on cache hit.
