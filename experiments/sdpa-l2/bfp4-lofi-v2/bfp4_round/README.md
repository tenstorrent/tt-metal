# Isolated BFP4 RNE pre-rounding prototype

This prototype keeps the existing attention variants and quantizers unchanged.
It computes the maximum independently in every native group of 16 adjacent
columns within each tile face, rounds and saturates onto that group's BFP4 grid
in SFPU, then uses the ordinary native BFP4 packer. There is no scale tensor,
intermediate DRAM pass, or packed-exponent feedback loop.

For `E = floor(log2(max(abs(group))))`, the grid spacing is `2^(E-2)`.
Separate FP32 additions `(abs(x) + 2^(E+21)) - 2^(E+21)` give RNE to that grid.
Clamping to `1.75 * 2^E` prevents an exponent carry. The pre-rounded nonzero
group retains exponent E and is exactly representable through all native
packing stages. Sign restoration occurs after rounding and saturation.

The implemented contract is finite BF16 normal inputs plus zero, with each
nonzero group's maximum exponent in [-124, 106]. Smaller individual normal
inputs are allowed. BF16 DST is the default because all pre-rounded values are
exact BF16; `--fp32-dst` supplies an independent control. `--output-format bf16`
exposes the pre-rounded values without BFP4 packing. The Python host oracle uses
integer alignment/RNE rather than sharing the kernel's magic-add algorithm.

From the repository root in the existing device environment:

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_round.py --label smoke-normal-v1 --length 4096
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_round.py --label smoke-ties-v1 --distribution thresholds --length 4096
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_round.py --label smoke-wide-v1 --distribution wide --length 4096
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_round.py --label smoke-zero-v1 --distribution zeros --length 4096
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_round.py --label fullchip-timing-v1 --length 262144 --cores 110 --batch 4 --iters 10
```

`--host-only` checks the two CPU oracles without importing TTNN or opening a
device. Device JIT compilation and execution must be performed by the card's
current owner; preparing this prototype did not run either.
