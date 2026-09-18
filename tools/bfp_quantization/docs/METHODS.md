# Methods and provenance

## Exponent search

For each group of 16 packed values, Emax is the largest IEEE-754 FP32 biased
exponent in the group. Candidate quantization matches TT B-format's exponent
alignment, round-to-nearest-even and saturation. BFP4 has three magnitude bits
plus sign; BFP8 has seven magnitude bits plus sign, in addition to shared-exponent
overhead. Denormal inputs follow the baseline flush-to-zero behavior.

The standalone default tries Emax and Emax−1, computes float64 summed squared
weight error for each, and keeps the lower-error candidate. Exact ties retain
Emax. This is Matt's proposed max/max−1 choice; the lower exponent can clip
outliers while giving the other values finer steps. No activations are needed.
The API's `exponent_deltas` are **added** to Emax: `(0,-1)`, not the historical
experiment's subtractive `offsets=(0,1)`. `(0,)` gives ordinary rounding.

## GPTQ + search

Collect H = XᵀX / token_count for each distinct layer input. Factor the Hessian
with 1% diagonal damping by default and order input columns by decreasing
diagonal magnitude. At each sequential column, quantize its groups of 16 outputs
using Emax/Emax−1/Emax−2, then propagate the quantization error into remaining
columns using the upper Cholesky factor of the inverse damped Hessian. Process
blocks of 128 input columns; use a matrix multiply for compensation outside the
current block. Restore the original input-column order before exporting.

Within one group/column, candidate selection is by weight SSE. The Hessian enters
through GPTQ's sequential error compensation; this is not a separate sweep of
activation-scored clipping thresholds. “Clipping” in the earlier experiment's
GPTQ + clipping label referred to these smaller-exponent candidates. Channels
with zero observed energy are treated as dead and their weights set to zero,
following the experiment's GPTQ routine. Poor coverage therefore matters.

`factor_hessian` is reusable across projections with identical inputs. Passing a
factor to `gptq_search` uses its recorded damping and activation order; those
choices were already made during factorization. The shipped GPTQ path is BFP4,
matching the tested method. Standalone exponent search supports BFP4 and BFP8.

Algorithm reference: [Frantar et al., GPTQ, ICLR 2023](https://arxiv.org/abs/2210.17323).
TT format reference: [blockfloat_common.cpp at the experiment's pinned commit](https://github.com/tenstorrent/tt-metal/blob/e44536782e14b501e2d8bc231c60c5808371e399/tt_metal/impl/data_format/blockfloat_common.cpp).

## Origin and scope

Extracted from the completed Qwen3.6-27B experiment, with model-specific loaders,
paths, machine names, saved weights, credentials and datasets removed. The
original TT implementation used commit `e44536782e14b501e2d8bc231c60c5808371e399`.
The pretrained model revision was `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
No checkpoint, calibration corpus, or model results need to be downloaded to
run the synthetic example and CPU test suite.

The native CPU loop preserves separate float32 multiply/subtract operations
(`-ffp-contract=off`). Dense factorization/matmul are supplied by PyTorch, so
cross-platform BLAS differences may change borderline choices; numerical
agreement on a platform does not establish identical full-model accuracy on
another platform. The test suite checks the native loop against the NumPy/
PyTorch implementation, calibration behavior, repacking, shard boundaries,
frozen experiment fixtures, CLI export and held-out reconstruction improvement.
`tests/test_ttnn.py` is optional and checks the installed native TT host packer.

See [VALIDATION.md](VALIDATION.md) for checks actually run when this package was prepared.
