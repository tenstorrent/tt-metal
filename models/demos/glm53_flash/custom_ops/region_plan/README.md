# region_plan

The regions prefill path's bookkeeping (`TtRoutedExperts._forward_regions`) as one single-core generic op: the
router's top-k expert ids (`[1, T*k]` uint32, row-major `(t, j)`) and the chip's local expert list (`gidx`) in;
`counts [E]`, the tile-aligned exclusive `offsets [E]`, the region slot of every pick `pos [T*k]` (the dummy slot
`max_buf - 1` at non-local picks) and the local-pick mask `loc` (`[1,1,T,k]` fp32 TILE) out. Two passes over the ids
in 64 KB chunks: counts, then ranks in `(t, j)` order with the pos chunks and the loc tiles written as they complete.
Replaces the one-hot / cumsum / gather chain measured at 27 + 8 ms per MoE layer at 16 K gathered tokens on the 8x4
(2026-11-01). Golden: `RegionPlan.golden` (the exact arithmetic of the ttnn chain).
