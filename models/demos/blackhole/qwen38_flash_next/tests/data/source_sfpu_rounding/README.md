Synthetic FP32 input/output fixtures for `tests/test_ttnn_source_sfpu_rounding.py`, recorded on the source commit
`cd9a11771107ea2c27da3303a0556ff7343e4af5` with its SFPI 7.74 toolchain ([PROVENANCE.md](../../../PROVENANCE.md)).
`manifest.json` holds each fixture's file, input and output hashes.

The first reciprocal input is a real GDN norm sqrt, `0x3a834422` (source output `0x4479a001`); the first softplus
input is a real GDN gate sum, `-5.17578125` (source output `0x3bb8ad42`). The reciprocal fixtures check exact source
words. The source's three-step inverse does not meet a 1e-4 relative bound over every mantissa; the test measures
and reports that instead of passing it, and the real-norm mantissa keeps the 1e-4 gate. Softplus is checked against
FP64 at rtol 1e-3 / atol 1e-5.
