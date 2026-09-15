# Fused two/three-component RNE BFP4 preprocessor

New isolated prototype; existing attention kernels and the qualified
`bfp4_round` implementation are not changed or imported.

For each BF16 input tile, read once into DST. Each four-row/16-column native
group is processed completely in SFPU: compute its shared-exponent RNE BFP4
component, store that component to its own DST tile, subtract in FP32 SFPU
registers, then repeat on the live residual. Every component has its own native
group exponent. Residuals never round-trip through DST, L1, or DRAM.

By default all output CBs and tensors are BFP4, requiring no pack format switch.
BF16 DST defaults to batch4 for two components (eight DST tiles) and batch2
for three (six DST tiles). FP32 DST control defaults to batch2 / batch1.
The writer drains all component CBs in the same order as the producer.

## Contract and checks

Finite BF16 normal values and zero only. Each residual passed to another stage
must remain exactly BF16-representable, with nonzero native-group maximum
exponent in [-124, 106]. The CPU oracle checks this stage by stage before the
CLI opens a device. Arbitrary underflowing recursive decompositions are not
qualified: SFPU flushes subnormals. The wide generator uses group exponents
[-90, 90], leaving headroom for the residual stages.

The reusable `build` API does not download inputs to validate values; its caller
must enforce this numerical contract. It returns `(outputs, invoke, cores)`,
where `outputs` is a list of two or three BFP4 tensors.

Optional `build(..., second_format="b8_rne5")` / `--second-format b8_rne5`
requires two components. The first remains direct RNE BFP4. Its actual decoded
value is subtracted from the original in live FP32 SFPU registers. The second
residual is rounded per value to five significant bits with integer RNE, then
packed natively into BFP8. The output list is then `[BFP4, BFP8]`. The oracle
models the native BFP8 shared-exponent **nearest-away** rounding independently;
it deliberately does not substitute host BFP8 ties-even packing or the older
test quantizer's -100 exponent clamp.

The mixed second stage uses 17 SFPU instructions per 64 input values, replacing
the second native-group max/magic-RNE/saturation stage. It adds two pack format
reconfigurations per batch (BFP4 to BFP8, then back). Output storage is 1664 bytes
per input tile versus 1152 for two BFP4 or 1728 for three BFP4; both mixed and
two-BFP4 paths retain two output DST tiles per input and default batch4. These
are static costs, not measured speed claims. Separate tensor accessors carry
the differing 576/1088-byte output page sizes.

The independent integer-alignment/RNE oracle and FP32 magic-add oracle agree
on each component; device outputs are compared separately, not merely after
component summation. Reconstruction errors after each component are also
reported using FP64 accumulation.

## Commands

Run only when the device owner grants the card:

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_residual_preprocess.py --label two-normal-v1 --components 2
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_residual_preprocess.py --label two-ties-v1 --components 2 --distribution thresholds
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_residual_preprocess.py --label two-wide-v1 --components 2 --distribution wide
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_residual_preprocess.py --label three-normal-v1 --components 3
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_residual_preprocess.py --label two-throughput-v1 --components 2 --length 262144 --cores 110 --iters 10
python experiments/sdpa-l2/bfp4-lofi-v2/bfp4_residual_preprocess.py --label mixed48-normal-v1 --components 2 --second-format b8_rne5
```

Controls: `--distribution zeros`, `--fp32-dst`, `--host-only`.
Host-only mode does not import TTNN or open a device. Preparing this prototype
did not compile JIT kernels or launch device jobs; the card's current owner
performs compilation and testing.
