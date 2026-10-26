# NOC line sweep

A per-chip NOC data-integrity test. It checks that every core of a chip can move data to every other core over
each NOC without corruption or loss, using plain NOC reads and writes with host-generated data and host-side
verification. Run it on each chip after board bring-up and before any fabric or multi-chip test.

Source: `tests/tt_metal/tt_metal/noc/test_noc_line_sweep.cpp` (host) and
`tests/tt_metal/tt_metal/test_kernels/dataflow/noc_line_sweep.cpp` (device). Binary: `unit_tests_noc`.

## Purpose

Higher layers (fabric routers, CCL ops, dispatch) move data over the NOC and usually wait for it to arrive by
polling on a word in the destination buffer. If a NOC path corrupts data, the polled word may never take its
expected value and the failure appears as a hang far from its cause. This test removes that indirection: nothing
on the device waits for data from another core, every byte is checked by the host, and a bad path is reported as a
data mismatch with a signature that points at the kind of fault.

Faults it is designed to expose:

* A data line stuck at 0 or 1, or inverted, on one NOC path: the same XOR mask on every bad word, classified as
  `stuck-at-0`, `stuck-at-1`, `inverted` or `intermittent`.
* A fault tied to a position inside a NOC flit: bad words concentrated on one flit-word index.
* A fault that depends on the source core, the destination core, or the NOC: a bad (source, destination, NOC)
  subset while the same cores are clean in every other combination.
* A write or read that never completes: a slot that still holds the poison pattern.

## What it does

* A **line** is an ordered set of cores: a logical tensix row, a logical tensix column, the active-Ethernet cores,
  or the idle-Ethernet cores of a chip.
* A **step** takes an active line and a passive line.
  * PUSH: every core of the active line runs the kernel and writes its own host-generated payload into a dedicated
    slot on every core of the passive line.
  * PULL: every core of the active line reads every passive core's payload into a dedicated local slot.
* A **sweep** iterates: for each transfer size, for each active line, for each other passive line, for each NOC
  (and RISC) combination. The default sizes are 1, 2, 4 and 8 KB per (source, destination) pair, each sent as a
  single NOC transaction, so every path carries packets of 16, 32, 64 and 128 flits.
* After every step the host reads every slot back and compares it word for word with the expected payload.
* Payloads are random flits, each followed by its bitwise complement. Every bit at every position inside a 64-byte
  flit is therefore sent as both 0 and 1 in every 128 bytes, so a stuck-at in either direction cannot hide behind
  the data, and the fault classification below is exact.

Memory layout in the unreserved L1 region of every core: `[payload][slot 0][slot 1]...[slot N-1]`. Slots are
pre-filled with `0xDEADBEEF`, so a slot that never received data is reported as `UNDELIVERED` rather than as random
corruption.

| Test | Active (runs kernel) | Passive | Direction |
|---|---|---|---|
| `TensixRowMajorPush` | each tensix row | every other row | write |
| `TensixColMajorPush` | each tensix column | every other column | write |
| `TensixRowMajorPull` | each tensix row | every other row | read |
| `TensixColMajorPull` | each tensix column | every other column | read |
| `EthToTensixPush` | active-eth, idle-eth | every row and column | write |
| `EthFromTensixPull` | active-eth, idle-eth | every row and column | read |
| `TensixToEthPush` | every row and column | active-eth, idle-eth | write |
| `EthToEthPush` | active-eth, idle-eth | the other Ethernet group | write |

The Ethernet-to-tensix write direction is the one used by fabric routers to deliver packets into local L1, so
`EthToTensixPush` covers the same path with the same RISC and NOC selection a router would use.

NOC / RISC combinations per line kind:

* Tensix: BRISC on NOC0 and BRISC on NOC1.
* Active Ethernet on architectures where base firmware owns ERISC0/NOC0: ERISC0+NOC0 and ERISC1+NOC1 when two
  ERISCs are exposed, otherwise ERISC1+NOC1 only.
* Active Ethernet on Wormhole: ERISC on NOC0 and on NOC1.
* Idle Ethernet: ERISC0 on NOC0 and on NOC1 (Blackhole only; idle-ERISC kernels on Wormhole are blocked on #14640).

## Running

Slow dispatch is recommended: every tensix core and every Ethernet core is then available. Under fast dispatch the
tensix sweeps still run (minus the dispatch cores) and the Ethernet sweeps are skipped.

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_noc --gtest_filter='NocLineSweepFixture.*'
```

The suite opens every chip visible to the host and sweeps them one after another. Restrict to some chips with
`TT_NOC_SWEEP_DEVICES=0,5`.

Output is deliberately compact. Each sweep logs one header line and one summary line per chip, a progress line
every 10 s while it runs, and the bad pairs of any failing step (capped by `TT_NOC_SWEEP_MAX_REPORT`):

```
[dev 0] sweep 'tensix row-major push': 8 active x 8 passive lines x 4 sizes, 448 steps, grid 8x8, harvesting mask 0x204
[dev 0] sweep 'tensix row-major push': step 230/448 (row4 -> row0 noc1 4096 B), 0 bad so far, 10.0 s elapsed
[dev 0] sweep 'tensix row-major push': 448 steps, 28672 pairs, 0 bad, 1.8 s (4.1 ms/step: host-write 0.2, build+run 1.0, readback+check 2.8)
```

The summary splits the per-step time into host L1 writes (payload and poison), program build + launch + wait for completion, and L1 readback + compare, which shows directly which part is slow on a given platform. The progress line only appears for sweeps that run longer than 10 s. A progress line that stops advancing means
a kernel on the active line never finished; the step named in the last progress line brackets where to look.
`TT_NOC_SWEEP_VERBOSE=1` logs every step and the cores of every line.

| Variable | Default | Meaning |
|---|---|---|
| `TT_NOC_SWEEP_TRANSFER_BYTES` | `1024,2048,4096,8192` | comma list of bytes per (source, destination) pair, each a multiple of 128; every step runs once per size |
| `TT_NOC_SWEEP_CHUNK_BYTES` | 0 | bytes per NOC transaction; 0 = whole transfer per call |
| `TT_NOC_SWEEP_ITERS` | 1 | re-send the same data this many times per step |
| `TT_NOC_SWEEP_SEED` | 0xC0FFEE | payload RNG seed |
| `TT_NOC_SWEEP_NOCS` | `0,1` | which NOCs to test |
| `TT_NOC_SWEEP_DEVICES` | all | comma list of chip ids |
| `TT_NOC_SWEEP_MAX_REPORT` | 16 | bad pairs logged per step and in the final summary |
| `TT_NOC_SWEEP_VERBOSE` | 0 | 1 = log every step and the cores of every line |
| `TT_NOC_SWEEP_INJECT_XOR` | 0 | checker self-test, see below |
| `TT_NOC_SWEEP_INJECT_MODE` | `set` | self-test fault type: `set` (stuck-at-1), `clear` (stuck-at-0), `flip` (inverted) |
| `TT_NOC_SWEEP_INJECT_FLIT_WORD` | 15 | checker self-test, see below |

The transfer size is capped automatically when a sink line cannot hold one slot per source core (small Ethernet L1
on some architectures); a warning says so.

Cost: about 10 s per chip on Wormhole N300 with the default four sizes (about 4 ms per step, independent of size
there). On Blackhole the per-step cost grows with the transfer size because host L1 readback dominates; a single
1 KB size measured 10-21 ms per step and about 15-20 s per chip, so budget several times that for the default
sizes, or set `TT_NOC_SWEEP_TRANSFER_BYTES=1024` for a quick pass. Send the output to a file: on a slow terminal
the log writes themselves can take several times longer than the test.

## Reading a failure

Every bad (source, destination) pair is logged once, the step it belongs to is logged after its pairs, and the
fixture prints a summary at the end:

```
[dev 0] active-eth log(0,2) virt(...) phys(2,1) -> row0 log(3,0) virt(...) phys(4,2) over noc1 (PUSH/write, erisc kernel, 1024 B):
    8/256 bad words, xor_or=0x00000800, fault=stuck-at-1 [0->1: 8/8, 1->0: 0/8], flit-word{idx:count}={15:8}, first bad @+0x3c exp=... got=...
```

* `log` / `virt` / `phys` are the logical, virtual (what the kernel addresses) and physical NOC0 coordinates of the
  core, so the result can be mapped onto the die layout directly.
* `xor_or` is the OR of `expected ^ got` over all bad words: a single bit means a stuck or flipped data line.
* `fault=` classifies the mismatch. For the bits in `xor_or` at the affected flit positions, the host counts how
  many times a 0 was sent (a chance for a 0->1 flip) and how many arrived as 1, and the same for 1->0. Because
  every payload carries each flit and its complement, the two chance counts are equal.
  * `stuck-at-1 [0->1: n/n, 1->0: 0/n]`: every 0 sent on that line arrived as 1, every 1 was fine.
  * `stuck-at-0 [0->1: 0/n, 1->0: n/n]`: the reverse.
  * `inverted [0->1: n/n, 1->0: n/n]`: every sample flipped, both directions.
  * `intermittent 0->1` / `1->0` / `mixed [k/n, ...]`: only some chances were taken, i.e. a marginal or
    timing-dependent fault rather than a hard one. Re-running with `TT_NOC_SWEEP_ITERS` or another seed shows
    whether the ratio is stable.
* `flit-word{idx:count}` is a histogram of which 32-bit word of each 64-byte NOC flit was wrong. A single index
  means the fault is tied to a position inside the flit, i.e. the NOC data path rather than memory.
* `UNDELIVERED` means the slot still holds the poison pattern: the write never arrived (or the read never
  completed), which points at routing or completion rather than data corruption.

To localise a fault, compare rows against columns and NOC0 against NOC1. A destination that is bad from one source
line over one NOC but clean from every other line and NOC isolates the fault to that source's output on that NOC,
or to a router on the route between them. A destination that is bad from everything isolates it to the
destination's inbound side.

## Checker self-test

To confirm the checker reports what you expect, make the host write a corrupted payload to every source core while
still expecting the pristine one. Every pair must then be reported with exactly that signature:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 TT_NOC_SWEEP_TRANSFER_BYTES=1024 \
TT_NOC_SWEEP_INJECT_XOR=0x800 TT_NOC_SWEEP_INJECT_FLIT_WORD=15 \
  ./build/test/tt_metal/unit_tests_noc --gtest_filter='NocLineSweepFixture.TensixRowMajorPush'
# ... 8/256 bad words, xor_or=0x00000800, fault=stuck-at-1 [0->1: 8/8, 1->0: 0/8], flit-word{idx:count}={15:8} ...
```

A 1 KB payload has 16 flits, so 16 samples of word 15, of which exactly 8 carry a 0 at bit 11 and 8 carry a 1
(each flit is paired with its complement). `set` corrupts the 8 zeros and reports `stuck-at-1 [0->1: 8/8, 1->0: 0/8]`;
`TT_NOC_SWEEP_INJECT_MODE=clear` corrupts the 8 ones and reports `stuck-at-0 [0->1: 0/8, 1->0: 8/8]`; `flip`
corrupts all 16 and reports `inverted [0->1: 8/8, 1->0: 8/8]` with 16/256 bad words. Larger sizes scale the
counts: 8 KB gives `[0->1: 64/64, 1->0: 0/64]`.

## Limitations

* Unicast only. A multicast variant of the kernel would be a small addition.
* One line at a time. Background traffic on the other NOC or from other lines is not generated, so faults that only
  appear under contention need at least `TT_NOC_SWEEP_ITERS` and a larger `TT_NOC_SWEEP_TRANSFER_BYTES`.
* Ethernet kernels need slow dispatch. Kernels cannot be launched on Ethernet cores under fast dispatch.
* DRAM endpoints are not included; the test covers L1-to-L1 paths only.
