# groupnorm-sharded-v2-mcast-host

- Kind: host integration
- Helper API: `MCAST_PIPE_API_VERSION=10`
- Bindings: legacy/Welford multicast and sender-only degenerate GroupNorm v2 paths
- Status: migrated, fully end-to-end
- Code commit: `0a796a025c9dc678387e2a7fa52518c898737dc9`
- Verified: 2026-07-30

## Atomic scope

- `groupnorm_sharded_program_factory.cpp`
- `reader_mcast_sender_unary_sharded_gn_v2.cpp`
- `reader_mcast_receiver_unary_sharded_gn_v2.cpp`
- `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp`
- `welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp`

GroupNorm multicast groups can wrap across the shard-grid boundary. The factory
now represents every group with a fixed three-block `Mcast2D` wire ordered
middle, first edge, and last edge. Each block adopts the existing
`reduce_sender_sem=0` and `reduce_receiver_sem=1` IDs and the preferred reader
NoC. Missing edge rectangles are encoded as sender-only singleton rectangles,
whose in-place sender operation is a no-op.

The legacy and Welford senders decode the three five-word CT/four-word RT
blocks with chained `McastArgs`. Their original gather-coordinate tail begins
at fixed RT offset 12. The raw pre-gather acknowledgement gate remains and
uses the helper's adopted consumer-ready semaphore. Sender compile-time blocks
override pre-handshake off; receiver blocks retain pre-handshake on. Both
receiver kernels decode the middle block; all three blocks share the same
sender and semaphore IDs, so first/last-rectangle receivers retain the same
receiver protocol.

Per-group helpers own logical-to-physical worker translation, NoC corner
ordering, offset grids, and active receiver counts. Tensor bindings and cached
buffer-address overrides are unchanged. When `use_mcast=false`, the group is a
sender-only singleton, no receiver kernel is emitted, and the existing
`num_mcast_cores > 1` compile-time guard bypasses all sends.

## Validation

- Gate 6 geometry classification found that every mapped production GroupNorm v2 configuration is
  zero-edge. The factory's actual splitter has direct synthetic host coverage for zero-, one-, and
  two-edge coordinate sequences: `GroupNormMcastGeometry` passed 3/3; `McastHostFixture` passed 25/25.
- The supported zero-edge performance class retains its matched Blackhole p100a measurements:
  legacy +0.248% and Welford -0.485% versus baseline, both within the 1.5% gate.
- `./build_metal.sh`: passed.
- Exact legacy 8x4 node under `scripts/run_safe_pytest.sh --dev`: passed.
  - sender JIT hash `665208170585676730`;
  - receiver JIT hash `4997071892188578060`.
- Exact Welford 8x4 node under `scripts/run_safe_pytest.sh --dev`: passed.
  - sender JIT hash `12485185751240554029`;
  - receiver JIT hash `14565098489008052769`.
- `GN-SHARDED-PARAMETERIZED -k legacy`: 108 passed, 2 expected skips.
- `GN-SHARDED-PARAMETERIZED -k welford`: 108 passed, 2 expected skips.
- Fixed/default-routing set: 19 passed, 6 expected skips.
- `McastHostFixture.*`: 19 passed, including `Mcast2DDegenerate`,
  pre-handshake override, and the 1D/rotating degenerate cases.
- `test_mcast_pipe.py`: 68 passed, including `test_f3_degenerate`.

The optional-weight/bias GroupNorm test routes through the older non-v2
factory, as confirmed by its isolated JIT cache, so it is not counted as v2
degenerate device coverage.

## Diff and coverage

- Production diff: 168 insertions, 376 deletions (net 208 lines removed).
- Coverage gap: the legacy and Welford `use_mcast=false` host bindings have
  direct host-helper and device-wire degenerate coverage, but no mapped device
  operation test reaches the GroupNorm v2 sender-only route. The same sender
  kernels have exact JIT and broad operation coverage through `use_mcast=true`.
- Result: PASS; no rollback or quarantine required.
