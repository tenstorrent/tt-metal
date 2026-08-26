# activation_reader_width_sharded.cpp — MIGRATED API v10 (verified 2026-08-05)

- Unit: `conv2d-activation-width-sharded` (Tier 6, `run-all`).
- Kernel role: hybrid rotating sender/receiver.
- Factory: `conv2d_op_width_sharded_program_factory.cpp`.
- Production commit: `fe866a1d0c4c32b78aae8a76e875c0da109f51c8`.
- Final status: **migrated**, fully end-to-end at `MCAST_PIPE_API_VERSION=10`.

## API-v10 wire update

The helper CT block now carries the full rotating rectangle area in its sixth `rotating_span` word.
The kernel uses `McastArgs<12, 3>` and derives its RT width and post-helper CT/RT boundaries from
that wire. The operation's actual loop count remains `num_input_cores`; the receiver coordinate
table may contain additional inert output-only/noop rectangle cores, but the helper owns and skips
the complete runtime block before operation-specific arguments.

The v10 host build, 25 host gtests, and complete 73-case helper device suite passed. The exact
fresh-JIT width case passed at PCC `0.9999992597711427` with 0/26 cache hits; the complete width
feature inventory passed 48/16, its DRAM-config route passed, and shared DRAM passed 14/14.

## Historical API-v9 formulation

The factory retains its two existing semaphore descriptors and adopts them in helper order
`[data_ready=act_mcast_receiver_semaphore, consumer_ready=act_mcast_sender_semaphore]`. One host
`Mcast2D` describes the dense reader bounding rectangle with a rotating sender, handshaked Flag
signalling, the activation NoC, and the divergent active ACK count
`max(input_num_cores, output_num_cores)-1`.

The factory appends the five-word helper CT block after the operation's first 12 CT words and appends
the rotating RT block after the existing `(core_x, core_y, full_grid_x)` prefix. The old explicit
semaphore/rectangle CT block and separable physical-X/physical-Y lookup arrays are gone. The helper
emits all full-rectangle sender coordinates; `McastArgs<12, 3, num_input_cores>` consumes the prefix
used by the operation's actual round-robin loop.

The kernel constructs both faces after the inactive-core return. Its sender branch is one
`send(src, dst, size)` call and its receiver branch is `receive(round)`. This removes the raw
multicast endpoint/destination, `wait_min`/reset/up/wait semaphore sequence, explicit signal
multicast, endpoint lookup, and write barrier. Skip-mcast remains compile-time guarded as before.

The load-config CT offsets moved by one word because the five-word helper block replaces the old
six-word semaphore-plus-rectangle block. No helper implementation or API version changed.

## Required semantics and the resolved historical failure

The geometric data fan-out is the full reader rectangle, while only active cores ACK. The helper
therefore derives the multicast count from rectangle area and carries the smaller ACK count
separately. Every input core takes one sender turn and receives the other turns over the same Flag
cell, including a real INCLUDE-source data loopback on its sender turn.

The prior API-v9 attempt produced 25 numerical failures and was restored. It predated the current
rotating `SenderPipe` completion rule: a real loopback is now ACK-fenced before `send()` returns, and
the sender then resets its local Flag cell for its next receiver turn. The complete width-sharded
inventory now proves that this was the missing completion invariant rather than an operation-level
API gap.

## Validation

- `./build_metal.sh`: passed.
- Exact BF16/BF16, filter-3, TILE-output, `packer_l1_acc=True`, `fp32_accum=False` node under
  `--dev` from fresh isolated `TT_METAL_CACHE`: passed, PCC `0.999956503`; the cache contains
  `activation_reader_width_sharded`.
- Complete `test_conv_features and WIDTH_SHARDED`: **48 passed / 16 legitimate row-major+bfloat8
  skips / 0 failed**.
- `test_conv_dram_config and WIDTH_SHARDED`: **1 passed**, PCC `0.998234911`; current JIT cache path
  `built/tt-metal-cache828016524674873717/kernels/activation_reader_width_sharded` was refreshed by
  the route.
- Post-integration `test_mcast_pipe.py`: **72 passed**.

Production diff: **+48 / -150**, a net reduction of **102 lines** across the kernel and factory.

## Historical v8 context

The successful 2026-06-20 v8 port first demonstrated that both sender and receiver faces fit the
helper when fan-out and ACK count are represented separately. It used direct `SenderPipe` and
`ReceiverPipe` construction while retaining the operation's X/Y sender lookup wire. The later v9
host-helper port made that full host/device wire helper-owned, but its first attempt lacked the
loopback completion guarantee and was quarantined after the 25 numerical regressions above.

The current migration preserves the useful v8 insight—full rectangle fan-out plus the smaller
active ACK count—while replacing its hand-packed device-only integration with `Mcast2D` and
`McastArgs` end to end.
