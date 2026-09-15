# Isolated BFP8 RNE pre-rounding prototype

This prototype keeps the existing attention variants and quantizers unchanged.
It computes the maximum independently in every native group of 16 adjacent
columns within each tile face, rounds and saturates onto that group's BFP8 grid
in SFPU, then uses the ordinary native BFP8 packer. There is no scale tensor,
intermediate DRAM pass, or packed-exponent feedback loop.

For `E = floor(log2(max(abs(group))))`, the grid spacing is `2^(E-6)`.
Separate FP32 additions `(abs(x) + 2^(E+17)) - 2^(E+17)` give RNE to that grid.
Clamping to `1.984375 * 2^E` prevents an exponent carry. The pre-rounded nonzero
group retains exponent E and is exactly representable through all native
packing stages. Sign restoration occurs after rounding and saturation.

There are seven magnitude bits plus sign, not a per-value eight-bit float.
This removes the ties-away rule of native packing by making packing an identity
on pre-rounded values. It does **not** guarantee zero aggregate bias: saturation
at magnitude 127 still clips the highest bin, and correlations/distributions
can affect net gain. The CLI reports least-squares quantized/input gain and
relative absolute-magnitude drift alongside L2 to characterize that effect.

The exponent bounds are recalculated for this grid: `E-6 >= -126` ensures a
normal minimum grid step, and `E+17 <= 127` keeps the magic constant finite.
Thus the supported group-maximum exponent range is `[-120,110]`. Individual
nonzero BF16 inputs may be smaller, down to `2^-126`; subnormals/NaN/Inf are
excluded. All-zero groups are supported explicitly.

The implemented contract is finite BF16 normal inputs plus zero, with each
nonzero group's maximum exponent in [-120, 110]. Smaller individual normal
inputs are allowed. BF16 DST is the default because all pre-rounded values are
exact BF16; `--fp32-dst` supplies an independent control. `--output-format bf16`
exposes the pre-rounded values without BFP8 packing. The Python host oracle uses
integer alignment/RNE rather than sharing the kernel's magic-add algorithm.

From the repository root in the existing device environment:

```sh
python experiments/sdpa-l2/bfp8-lofi-v2/bfp8_round.py --label smoke-normal-v1 --length 4096
python experiments/sdpa-l2/bfp8-lofi-v2/bfp8_round.py --label smoke-ties-v1 --distribution thresholds --length 4096
python experiments/sdpa-l2/bfp8-lofi-v2/bfp8_round.py --label smoke-wide-v1 --distribution wide --length 4096
python experiments/sdpa-l2/bfp8-lofi-v2/bfp8_round.py --label smoke-zero-v1 --distribution zeros --length 4096
python experiments/sdpa-l2/bfp8-lofi-v2/bfp8_round.py --label fullchip-timing-v1 --length 262144 --cores 110 --batch 4 --iters 10
```

`--host-only` checks the two CPU oracles without importing TTNN or opening a
device. Device JIT compilation and execution must be performed by the card's
current owner; preparing this prototype did not run either.
