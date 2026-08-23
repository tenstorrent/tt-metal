# TT-Train K-split Gram Matmul multicast

Paths:

- `tt-train/sources/ttml/metal/ops/k_split_gram_matmul/device/kernels/mcast_sender.cpp`
- `tt-train/sources/ttml/metal/ops/k_split_gram_matmul/device/kernels/mcast_receiver.cpp`
- `tt-train/sources/ttml/metal/ops/k_split_gram_matmul/device/kernels/mcast_receiver_writer.cpp`

Current ledger status (2026-08-23): all three deferred, repository-recall
additions.

The sender drives independent row and column channels with upper/lower
rectangles, four semaphore pairs, repeated CB-wrap sends, and one loopback path
where a sending core also consumes a receive flag. The two receiver binaries
share the clear/ready-increment/VALID-wait protocol; partner reduction and
DRAM/mirror output remain operation-owned.

Treat all three kernels and `k_split_gram_matmul_program_factory.cpp` as one
atomic protocol unit. Migration also requires the TT-Train dependency-boundary
decision. Mapped coverage is
`tt-train/tests/ops/k_split_gram_matmul_op_test.cpp`; it was not run during this
static reconciliation.
