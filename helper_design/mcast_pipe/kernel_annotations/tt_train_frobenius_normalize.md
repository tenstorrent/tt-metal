# TT-Train Frobenius normalization multicast

Path: `tt-train/sources/ttml/metal/ops/frobenius_normalize/device/kernels/dataflow/reader_frobenius_normalize.cpp`

Current ledger status (2026-08-23): deferred, repository-recall addition.

One reader binary serves both roles. The origin gathers per-core partials,
computes the normalization scalar, loopback-multicasts the scalar to the dense
active-core bounding box, then multicasts a level flag. Non-origin cores wait
on the flag before consuming the scalar. The reduction semaphore and gather
writes are operation-owned phases outside the broadcast channel.

Migration requires a decision on whether TT-Train may depend on the TTNN
kernel helper. The atomic source scope is the reader plus
`frobenius_normalize_program_factory.cpp`; the mapped coverage is
`tt-train/tests/ops/frobenius_normalize_test.cpp`. No build or device test was
performed during this static reconciliation.
