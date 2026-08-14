# block-sharded Conv2D activation multicast — reverted experiment

> **Historical only (reverted 2026-08-14).** The experimental `f3361f57596`,
> `ccd7b597e92`, `8ae4604379e`, and `9686814ea22` feature chain was reverted.
> This file preserves the attempted design and is not current rollout evidence
> or instructions.

## Experimental verdict at the time

The former R4 design gap is resolved in source. Commit `f3361f57596` added
`SenderPipe::send_from_cb`, migrated the block-sharded Conv2D activation reader,
and connected the factory to a rotating `Mcast1D` channel. Commit `ccd7b597e92`
kept CB type resolution at the call site without changing the wire or API
version.

`send_from_cb` preserves the original producer/NoC overlap: it waits for each
monotonically growing CB frontier and multicasts that burst before the complete
tilized block is ready. The helper also owns loopback, degenerate local copy,
handshake, semaphore, and sender/receiver geometry behavior.

## Attempted rollout state

- Kernel: `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`
- Host binding: `activation-mcast:conv2d-block-sharded:rotating-lines`
- Atomic unit: `conv2d-activation-block-sharded`
- Ledger status at the time: `pending`
- Required inventory: `CONV-BLOCK`, including exact post-integration JIT proof
  and the mapped performance gate

The attempt required the host binding and kernel to be validated and written
back together. It was reverted before that apply validation and write-back.
