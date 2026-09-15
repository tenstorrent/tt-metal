# FP32 matmul floor: product alignment, not FP32 output quantization

## Result

The CPU model reproduces the Blackhole device output **bit-for-bit for all
15 original case/fidelity combinations**, including dense HiFi4/FP32. This is
an exact SHA256 match over every output, not merely agreement in L2. Results:
[model JSON](matmul-fp32-alignment-model-v2.json),
[CPU-only model](matmul_fp32_alignment_model.py),
[original device records](matmul-fp32-floor-v1.jsonl).

| HiFi4/FP32 case | Device and model L2 | Model with 13 extra product-alignment bits |
| --- | ---: | ---: |
| Identity | 0% | 0% |
| One BF16 product | 0% | 0% |
| Dense K=32 | 0.031518409% | 0.000004746% |
| Dense K=128 | 0.031674011% | 0.000012093% |
| Positive K=128 | 0.011827141% | 0.000022670% |

The ablation preserves the modeled final FP32 accumulator and instruction
order; it changes only precision retained during product alignment. It is a
theoretical ablation, **not an available hardware mode**.

## Mechanism and evidence strength

The detailed local simulator model at
`ttsim-private/src/tensix.cpp:1300–1373` computes each MVMUL dot16 as **two groups
of eight** products. For each fidelity phase:

1. Form up-to-12-bit magnitudes from 5-bit SrcA and 7-bit SrcB pieces.
2. Find the largest product exponent independently in each eight-product group.
3. Round every product magnitude onto that exponent's integer grid **before**
   applying signs and summing. In integer notation, for exponent difference d,
   the code implements `((m << 1) + (1 << d)) >> (d + 1)`.
4. Widen each signed group sum by 13 bits, then merge both sums with FP32 DST
   using the alignment/normalization rules at `src/tensix.cpp:1101–1202`.

Step 3 does not depend on FP32 DST. Low bits survive step 4 and subsequent
fidelity phases, explaining why the output is genuinely FP32 while still
carrying much larger early-rounding error. A single nonzero product has no
competing exponent and therefore avoids this loss. HiFi4 consumes all BF16
operand bits, but cannot recover information already lost within each phase.

The simulator has a corresponding RTL-reference comparison in
`tests/rtl/tt_fp_lane/sim_main.cpp:202–282,439–449`; its wrapper explicitly wires
products 0–7 and 8–15 into paired lanes. I inspected this test but did not run
RTL verification. The stronger evidence here is independent CPU reproduction
of the existing **Blackhole hardware** results.

The selected replay order is K tile32 → fidelity phase → lower/upper K half16,
consistent with `tt_llk_blackhole/llk_lib/llk_math_matmul.h:320–449`. Alternative
phase orders have similar L2 but fail dense HiFi4 output hashes, confirming that
low-order final accumulation rounding is also represented by the model.

## Configuration and ISA limits

`llk_math_common.h:44–45` enables FP32 FPU/SFPU destination interpretation.
It does not widen the initial product-alignment grid. The disabled
`packer_l1_acc` and exact one-product outputs also rule out a BF16 partial-output
spill as the explanation for these particular tests.

The public Blackhole MatrixUnit page delegates to the Wormhole-oriented shared
documentation. The shared MVMUL page explicitly says its floating-point
pseudocode is approximate: operation order and precision can differ. It does
**not** promise an IEEE FP32 dot product or specify this exact eight-product
alignment mechanism. See [MatrixUnit](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/MatrixUnit.md)
and [MVMUL](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md).

LLK exposes `fpu_srnd_en`, default false, in
`cunpack_common.h:766,844–850`. The inspected simulator rejects enabling it;
neither an accuracy guarantee nor a wider-grid mode follows from this flag.
No supported switch to remove the observed alignment loss was identified.

## Consequences and discriminator

The approximately 0.03044% Q-mean-correction residual is consistent with this
hardware mechanism, but its complete attention path has not been modeled here.
This is not a universal SDPA accuracy floor. Centering or transformations can
change which products compete for an exponent; final-output scaling cannot
reconstruct already discarded terms. Cancellation and large product-exponent
spread remain useful stress cases even with HiFi4/FP32.

Sharp independent prediction: with power-of-two BF16 operands, compute
`1 + 2^-12`. Put both nonzero products in reduction positions 0/1: predicted
output **1.0**. Put them in positions 0/8: predicted output **1.000244140625**,
the exact answer. The independent device check now confirms both predictions:
all 2,048 outputs are 1.0 for positions 0/1, and all 2,048 are exactly
1.000244140625 for positions 0/8, in each of LoFi, HiFi2 and HiFi4 with FP32
DST. The first case has 0.02440810349% L2; the second has zero error. See
`same_eight` / `opposite_eight` in `matmul-fp32-floor-v2.jsonl`.
Positions 0/7 and 0/15 are additional model predictions, not device-tested here.

## Reproducibility

CPU-only execution used `/opt/venv/bin/python`, four torch threads and one
interop thread on the reserved host; no ttnn/device import or device job.
The output pins its own source hash and original device-record provenance.

- Simulator commit: `8c47553e0c28e4b2239497ffa4f5a44f02e1683f`.
- Simulator `src/tensix.cpp` SHA256:
  `73facffeb1e7d508a256dcc8cc1bba1120e870964283b1e5ef65c9ce71e7b9d5`.
- tt-metal checkout commit: `637d956c8874d356c9080e467b9a1664133fa780`.
- LLK `llk_math_matmul.h` SHA256:
  `41e1b2e27caa7177f4c8c8e8b7b8f0c3e1475a4faab7b8b9e34053f0b25afa3d`.
- ISA checkout commit: `5287a62727350bcef35f7b411d1b8a706172ec4c`.

The JSON additionally pins both RTL-reference files and relevant LLK
configuration hashes. Local private simulator details are internal research
evidence; do not treat them as an already-published ISA contract or automatically
include private-source citations in a public PR.
