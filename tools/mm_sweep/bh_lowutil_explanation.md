# Why the low-util BH shapes are stuck — roofline explanation

**TL;DR:** the bottom-10 util shapes are all **M=1-2 tiles** (M=32/64 rows). A matmul's
arithmetic intensity is **AI ≈ M (rows)** — K and N cancel — so small-M shapes are
DRAM-bandwidth-bound on the in1 (weight) read at ~280 GB/s effective, with a single-digit-%
util ceiling. This is structural (these are GEMV-like), **not** a sweep-coverage or blocking gap.

## Evidence 1 — util scales with M at FIXED (K,N)
```
K=6144 N=4608:  M=1t→2.9%   M=2t→5.7%   M=4t→11.3%   M=16t→38.3%   M=516t→60.5%
K=6144 N=9216:  M=1t→3.0%   M=2t→6.0%   M=16t→43.0%  M=18t→47.9%
K=6144 N=1536:  M=1t→2.4%   M=2t→4.9%   M=16t→29.0%
K=2304 N=6144:  M=4t→10.5%  M=512t→53.1%
K=6144 N=768:   M=4t→6.6%   M=128t→50.2%
```
Doubling M doubles util in the memory-bound regime, then saturates (compute-bound) at large M.

## Evidence 2 — roofline: AI = 2·M·K·N / (2·K·N) = M
```
shape              AI(flop/byte)  util    regime
32×256×6144        28             1.6%    DRAM-bound (in1 read)
32×6144×6144       32             2.9%    DRAM-bound
32×6144×9216       32             3.0%    DRAM-bound
512×6144×9216      450            43.0%   still ~DRAM-bound (~290 GB/s)
16512×6144×4608    2271           60.5%   compute/other-bound (~81 GB/s)
```
util ≈ AI × (BW_eff / peak), BW_eff ≈ 280 GB/s (~55% of BH DRAM peak), peak = 304 TFLOP/s.
=> util ≈ 0.09% per row of M, until compute saturates near M≈600 rows.

**K and N do not help util** — they appear in both the FLOPs and the in1 bytes and cancel. Larger
K·N just means more weight bytes to stream at the same low intensity. Only **M** (the reuse
dimension) buys arithmetic intensity.

## Not a tuning gap
- **(S,Pk):** all 9 feasible combos swept (incl. max-parallel (10,1)). Slicing (5,1) helps ~1.5x by
  spreading the in1 read across more DRAM banks / NoC injectors (~190→~280 GB/s) but cannot beat the
  DRAM wall.
- **Blocking:** M=1 tile pins mb=1; blocking creates L1 *reuse*, but there is no in1 reuse to capture
  when M=1. Best blocks are degenerate (1/4/{1..4}). Not the limiter.

## What would actually raise it
1. **Lower-precision weights** — bf8 in1 halves bytes → ~2x AI → ~2x util; bf4 → ~4x. Main lever for M≤2.
2. **Increase effective M** — batch GEMV calls if the model allows.
3. **Modest kernel headroom** — DRAM BW is ~55% of peak; better read scheduling / inter-core
   forwarding could give ~1.3-1.4x, but the ceiling stays single-digit % at M=1.
