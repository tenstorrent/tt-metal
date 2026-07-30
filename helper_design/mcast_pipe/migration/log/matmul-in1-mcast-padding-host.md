# matmul-in1-mcast-padding-host

- Kind: host integration
- Helper API: `MCAST_PIPE_API_VERSION=9`
- Bindings: all four `matmul-in1-mcast` legacy/descriptor 1D/2D bindings
- Status: migrated, fully end-to-end
- Code commit: `2d0280d3dacf8a2ba24882b35816c6a1fbffb7dd`
- Verified: 2026-07-30

## Atomic scope

- `matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- `matmul_multicore_reuse_mcast_2d_program_factory.cpp`
- `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
- `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`

The 1D factory's legacy and descriptor mcast-in1 paths now construct one host
`Mcast2D` over the actual offset worker bounding rectangle. The 2D legacy and
descriptor paths construct one `Mcast2D` per in1 line, preserving
`transpose_mcast`, sender placement, the preferred in1 NoC, subdevice offsets,
and active receiver acknowledgement counts. All four bindings adopt the
existing receiver/sender semaphore IDs.

Both kernels decode the helper's five-word CT/four-word RT wire through
`McastArgs`. Downstream compile-time and runtime argument offsets, tensor
bindings, optional fused bias arguments, and cached output-address overrides
were shifted to match. The `MCAST_IN0`, sparse, and non-multicast
`SKIP_MCAST` paths retain their previous wire and behavior.

## Validation

- `./build_metal.sh`: passed.
- Exact 1D mcast-in1 non-zero-subdevice node under
  `scripts/run_safe_pytest.sh --dev`: passed.
  - sender JIT hash `10580236968838213332`;
  - receiver JIT hash `6510408673418518324`.
- Exact 2D `transpose_mcast=false` and `transpose_mcast=true` descriptor nodes
  under `scripts/run_safe_pytest.sh --dev`: both passed.
  - sender JIT hash `4616781822959825899`;
  - receiver JIT hash `4167676435791909128`.
- `MM-IN1-ALL`: 302 passed, 188 expected skips, 490 selected.
- `McastHostFixture.*`: 19 passed.
- `test_mcast_pipe.py`: 68 passed.

The mapped matmul inventory exercises the descriptor constructors at runtime,
including both 2D multicast orientations and offset 1D subdevice placement.
The legacy constructors compile in the full host build, but their only current
callers are fused CCL factories, so no mapped single-chip pytest provides
legacy-constructor device-runtime proof.

## Diff and coverage

- Production diff: 196 insertions, 70 deletions.
- Coverage gap: yes; legacy factory constructors have build coverage but no
  mapped device-runtime coverage.
- Result: PASS; no rollback or quarantine required.
