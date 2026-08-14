# block-sharded Conv2D activation multicast — source integrated, validation pending

## Current verdict

The former R4 design gap is resolved in source. Commit `f3361f57596` added
`SenderPipe::send_from_cb`, migrated the block-sharded Conv2D activation reader,
and connected the factory to a rotating `Mcast1D` channel. Commit `ccd7b597e92`
kept CB type resolution at the call site without changing the wire or API
version.

`send_from_cb` preserves the original producer/NoC overlap: it waits for each
monotonically growing CB frontier and multicasts that burst before the complete
tilized block is ready. The helper also owns loopback, degenerate local copy,
handshake, semaphore, and sender/receiver geometry behavior.

## Rollout state

- Kernel: `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`
- Host binding: `activation-mcast:conv2d-block-sharded:rotating-lines`
- Atomic unit: `conv2d-activation-block-sharded`
- Ledger status: `pending`
- Required inventory: `CONV-BLOCK`, including exact post-integration JIT proof
  and the mapped performance gate

Do not mark this unit migrated from source inspection or helper-only tests. The
host binding and kernel must be validated and written back together when the
user resumes the apply workflow.
