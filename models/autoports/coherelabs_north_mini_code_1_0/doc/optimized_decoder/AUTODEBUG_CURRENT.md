# Current-arm AutoDebug

Fresh source/artifact diagnosis after `STAGE_REVIEW_CURRENT.md` found three
independent issues.

1. Attention used one dtype/fidelity for packed QKV and O. The retained BFP4
   veto used random activations and an older O geometry. The required
   experiment is to split QKV/O, keep SDPA/RMSNorm fixed, and cross BFP4
   LoFi/HiFi2 on checkpoint-propagated activations through non-aligned prefill
   and cache-consuming traced decode at batches 1 and 32.
2. Sparse gate/up/down all used `per_core_M=per_core_N=1`, making larger
   subblocks illegal on the selected grids. Height 2 remains illegal because M
   is one tile. Reducing gate/up from 24 to 12 cores and down from 64 to 32
   cores makes `per_core_N=2`, permitting independent 1x2 candidates.
3. Batch-32 dense all-expert dispatch is caused by a missing TTNN output
   contract. Sparse matmul retains the full token-by-expert surface;
   `moe_compute(compute_only=True)` exposes only a rolling two-expert buffer;
   its complete consumer requires fabric. A compact persistent output or
   local-only fused combine requires shared TTNN changes excluded by this
   stage.

Focused evidence is retained in `SPARSE_SUBBLOCK_HYPOTHESIS.md` and
`ROUTED_MOE_HYPOTHESIS.md`.
