# Conv3D weight sharing — migrated at API v11

Date: 2026-08-16

## Verdict

Migrated in `a290ce202811f6867a0c18e1ebcb19285369881c`. Only `WeightShareMode::Mcast` moved to
`Mcast2D`; the Disabled path and the point-to-point Chain transport remain operation-owned and
unchanged. The historical v7 runtime-recipient-count gap is obsolete because API v11 derives fan-out
from the runtime rectangle and accepts the ACK override carried by `McastArgs`.

## Protocol mapping

- Each logical group strip is one independent dense `Mcast2D` rectangle with a fixed, staggered sender.
  The host retains the original semaphore allocation order and adopts those IDs as helper data-ready
  and consumer-ready semaphores.
- One representative helper compile-time block is appended after the three TensorAccessor blocks. Each
  group is asserted active and compile-time-identical to the representative, so all strips safely share
  one writer binary.
- The four-word helper runtime block starts at operation slot 19 on every core, including Chain and
  Disabled modes. The kernel resumes operation-owned parsing at
  `weights_mcast_args.next_runtime_args_offset()`; the trailing iteration/worker ABI remains fixed.
- The sender reads the weight block from DRAM, then uses default `SourceL1Guard` for
  `send(local_addr, local_addr, bytes)`. This preserves source lifetime until the next receiver-ACK gate;
  using caller-managed source lifetime here would permit the next DRAM fill to reuse the one-block CB
  before the next round's ACK wait.
- Active and passive receivers call the helper receiver once per multicast iteration. Passive cores
  consume no CB slots and retain the original final atomic barrier.
- The factory asserts that the weight CB is allocated over the complete core rectangle. Therefore every
  destination—including passive cores—has a valid, identical destination L1 address.

## Validation

- Production LOC: factory 44 additions / 49 deletions; kernel 15 additions / 58 deletions.
- `./build_metal.sh` passed.
- A cold exact `k333_s111_g1_zeros_c64_c64` run passed with PCC `0.9999914190473849` and 0/25 JIT
  hits. Compile output identified the writer's Mcast variant (`CT[22]=2`) on NOC1, proving the intended
  branch was newly built.
- The focused shape sweep passed 12/12. The complete unit file passed 27 with one pre-existing
  Blackhole skip. The complete nightly file passed 1606 with five expected skips and two pre-existing
  width-sharded page-alignment xfails.
- Shared guards passed: `McastHostFixture.*` 32/32, `test_mcast_pipe.py` 80/80 under Watcher, and the
  source audit 18/18 after adding the Conv3D fixed-ABI guard.
- Matched Tracy at 800 MHz used three independent 25-iteration sessions per source state and shape,
  discarding the first five Conv3d samples in every session:
  - non-grouped `k333_s222_g1_zeros_c64_c64`: raw 14,977 ns, migrated 14,855 ns, -0.815%;
  - grouped `k333_s111_g4_replicate_c64_c64`: raw 70,343 ns, migrated 70,133.5 ns, -0.298%.

## Claude consultation

The architecture review returned REVISE and required default `SourceL1Guard`, complete rectangle CB
allocation, a fixed four-word helper ABI with named next offset, compile-time equality checks across
strips, and adoption of the existing semaphore IDs without changing allocation order. All five were
implemented. Two broad final-review attempts timed out without verdict and were not treated as approval;
a final bounded decision returned PASS, API EXPANSION NO, LEDGER YES.
