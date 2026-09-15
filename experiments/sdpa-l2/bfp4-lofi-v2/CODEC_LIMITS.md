# Four-bit codec limits: TT BFP4 versus NVFP4

## What is already established

**TT BFP4 is not NVFP4 with a different name. Neither four-bit codebook is
universally more accurate.** The relevant axes are codebook, scale precision,
scale-selection rule, group orientation and actual rounding path. We should
measure those separately before attributing all error to a hardware limitation.

| Representation | Nonnegative element levels before scaling | Shared scale | Group |
|---|---|---|---|
| TT BFP4_b | 0,1,2,3,4,5,6,7 | Power of two; native exponent E means step2^(E−2) | 16 consecutive columns within a tile face |
| MXFP4 | 0,.5,1,1.5,2,3,4,6 | E8M0 power of two | 32 elements |
| NVFP4 | 0,.5,1,1.5,2,3,4,6 | E4M3 block scale plus FP32 global scale | 16 elements |

NVIDIA's official description specifies the E2M1 codebook, block sizes and
two-level NVFP4 scaling. Fractional E4M3 scales allow a closer match to block
amplitude than power-of-two scales. This does not make the four-bit elements
exact, and an illustrative scale-MSE improvement is not an attention-L2 claim.
[NVIDIA format documentation](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
MXFP4's group32/E8M0 layout is also specified by the
[OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

The TT statement is grounded in the exact current CPU/device-qualified codecs:
`bfp4_round.py:host_rne_bfp4`, `adaptive_bfp4_round.py:oracle`, and frozen
`../bfp4-lofi-v1/probe.py:quantize`. Default native packing is not unbiased
nearest-grid quantization: the qualified path first rounds each datum to E8M6,
rounds the shared BFP8 representation, then truncates to three magnitude bits.
The RNE preprocessor removes that avoidable rounding bias. Adaptive exponent
search changes the clipping/resolution tradeoff but retains a power-of-two grid.

For a block with maximum M, ideal absmax E2M1 scaling has smallest nonzero step
M/12 but its largest spacing is M/3. Continuous uniform7 scaling has spacing
M/7 throughout. Thus E2M1 resolves small values more finely at the expense of
large-value resolution. Two witnesses rule out universal dominance:

- A block containing every integer0…7 is exact in native TT BFP4 with step1.
  E2M1 with scale7/6 is not exact for those integers.
- A block containing0,.5,1,1.5,2,3,4,6 is exact in E2M1 with scale1. Native TT
  BFP4 selects step1 and loses the half-integer values.

These examples compare specified scaling recipes, not an exhaustive optimal
scale search. Nor do they imply that either codec will meet a0.5% attention goal.

## Why V orientation could matter as much as codebook

Current TT V tiles share exponents across16 output features D, not16 tokens N.
One persistent high-amplitude channel can therefore coarsen15 neighboring
channels at every token. Grouping across N instead confines its exponent to
that channel. This is a hypothesis with a clear controlled test, not a claim
that the existing reader/PV kernel supports the alternative layout for free.

SageAttention3 explicitly transposes V during quantization to satisfy FP4 MMA's
sequence-contiguous operand layout. It also uses Q/K smoothing and a separate
two-level probability quantizer. Its reported attention results cannot be
reproduced by changing only K/V codec; our comparisons omit those algorithms
and do not claim Sage3 emulation. [SageAttention3 §3 and appendix A.5](https://arxiv.org/html/2505.11594v3).

## New matched CPU experiment

`codec_limits_models.py` reuses pinned TT oracle/input/reference function bodies
through AST extraction, without importing TTNN or opening a device. Q/K/V start
from identical original BF16 inputs. QK, softmax, P, PV and recurrent arithmetic
are FP64; P is unquantized. Both FP64 output and final-BF16 output metrics are
reported, against the same original-input FP64 attention reference.

Seven codecs separate the questions:

1. `tt_native`: actual qualified biased native packing model.
2. `tt_rne`: qualified shared-exponent RNE/saturation.
3. `tt_adaptive_pm`: qualified E/E−1/E+1 search with FP32 score-tree selection.
4. `uniform7_continuous`: absmax/7 FP64 scale, isolating unrestricted scaling
   without changing TT's uniform codebook.
5. `e2m1_continuous`: absmax/6 FP64 scale and E2M1 RNE codebook. This is an
   **optimistic representability comparison**, not a rigorous lower bound or
   an MSE-optimal scale search. Removing scale rounding need not improve every
   block or downstream attention result.
6. `e2m1_power2_g16`: power-of-two scale2^(floor(log2(M))−2), E2M1 RNE/saturation,
   group16. This isolates codebook under a coarse scale; it is **not MXFP4**,
   which also uses group32.
7. `nvfp4_e4m3`: representative absmax NVFP4 recipe, FP32 global scale
   tensor_absmax/(6×448), E4M3FN-RNE local scale M/(6×global), E2M1-RNE elements,
   then exact FP64 reconstruction of those encoded values. No NV hardware
   packing or tensor-core arithmetic is emulated. This is a legal-format
   representative recipe, not a bit-exact model of every NVIDIA software path.

Every codec reports K-only, V-only and K+V attention errors, independently of
optional Q-RNE7. K groups always follow D. `--v-axes D N` applies both orientations
to **all selected codecs**, including TT, with explicit labels. N-group TT is a
representation experiment requiring a different actual dataflow/packing layout.

Inputs include normal, sparse outliers from the original repro, commonV32, and
`channel_v`: original normal BF16 V with channels0,16,… multiplied by32. The
last pattern deliberately places one persistent feature outlier in every TT
D-group16. Quiet-channel errors are reported separately; commonV reports exact-
FP64-mean-subtracted residual L2, avoiding a misleading large-offset denominator.

## Execution and evidence status

No model runtime or device jobs were started by the implementation agent:
local Torch is unavailable and the parent owns scheduling to avoid disturbing
device timings. Syntax and selected source paths are checked locally. Numerical
conclusions and tables must wait for passing `self_tests` and a final `complete`
record in the output JSONL. No format superiority factor is asserted yet.

Priority orientation audit,72 attention variants plus controls:

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/codec_limits_models.py --label codec-vaxis-v1 --lengths 4096 --q-rows 64 --seeds 1240 --distributions normal outliers channel_v common_v --codecs tt_rne tt_adaptive_pm nvfp4_e4m3 --q-modes bf16 --v-axes D N --threads 4
```

Matched codec comparison,84 variants plus controls:

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/codec_limits_models.py --label codec-formats-v1 --lengths 4096 --q-rows 128 --threads 4
```

The JSONL records exact input hashes, every reused source hash, group/codebook/
scale metadata, reconstruction L2, attention L2/PCC/gain, finite checks and source
stability. No gain alignment is applied to the reported primary L2. Longer N
and additional seeds can follow only if this small comparison is informative.

## Hardware implication

NVFP4-scale flexibility is not a drop-in TT BFP4 optimization. A non-power-of-two
scale per reduction group cannot generally be replaced by one output scale;
it must be applied to each group's partial dot product or absorbed through a
higher-precision representation. E2M1 also cannot be losslessly relabeled as
TT's uniform BFP4 codebook. Emulation would introduce conversion/scaling and
potentially higher-width storage or multiple matmuls. This CPU test intentionally
assigns no speed to such an implementation. The practical TT levers remain
better rounding, adaptive power-of-two scaling, grouping/layout, smoothing and
residual components, assessed against real kernel costs.
