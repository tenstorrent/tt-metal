# Fused single-token GDN epilogue: simulator correctness

The new opt-in standalone operator consumes the recurrence's FP32 row-major
output and performs output layout conversion, gated RMSNorm and the final z
multiply in one program. It reads only the live token row and produces the
existing BF16 tiled output geometry with zero padding. FP32 recurrent state and
BFP8 model policy are unchanged. **The model does not select this operator yet.**

Simulator v4 completed at **21:55:17 UTC Oct 9, 2026**. All **nine comparisons**
were bit-identical to the actual native TTNN layout + gated-norm + multiply
sequence: B1/B16/B32, two live independent allocations, then returning to the
first allocation. Pre-multiply normalized outputs also match bit-for-bit.
Inputs were unchanged and padded rows were zero. B16/B32 exercise multiple
worker waves; cases include zero and near-zero norms and extreme gate values.

These are synthetic CPU simulator results, not model evals or physical timing.
The simulator uses `TT_METAL_DISABLE_SFPLOADMACRO=1`; production instruction
parity still requires hardware. The next gates are physical output/rebinding
checks and matched trace timings, then real-weight layer/model integration and
reference evaluation. There is no claimed TSU uplift from this operator yet.

## Diagnosed attempts

1. **v1:** the reference-test setup explicitly deallocated an intermediate
   padded gate that aliased the original gate. The subsequent tensor read
   segfaulted before candidate execution. Retaining those reference views fixes
   their ownership; v2/v3 confirmed the alias and read the original inputs safely.
2. **v2:** the kernel compiled and executed, but the first case had 0.28624%
   maximum per-head relative RMS error, above the unchanged 0.1% gate.
3. **v3:** exposing the intermediate proved the gated norm was bit-identical;
   the difference arose in the final multiply. Candidate FPU multiplication and
   packing differed from TTNN's SFPU multiply with BF16 round-to-nearest-even.
4. **v4:** use SFPU multiplication plus explicit BF16 RNE before packing, while
   preserving the native BF16 intermediate. All nine comparisons pass without
   relaxing any threshold, including repeated allocations and zero padding.

Every attempt's result, log, launch and exact source manifest is retained.
The frozen source files are compressed verbatim for each attempt so repository
formatting cannot change their provenance. The standalone implementation is
under `tt/gdn_epilogue/`; `demo/probe_gdn_epilogue.py` is the simulator screen.

Runs used the existing pinned virtual Blackhole library in separate persistent
units with one CPU, 8-GiB RAM, a private JIT cache and a 45-minute bound. They did
not open physical devices or take the physical-device lock. Existing hardware
qualification and BFP8 follow-up snapshots remain unchanged.
