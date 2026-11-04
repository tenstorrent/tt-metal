# NOC Deployment Tests
On-chip NOC data integrity tests. They find NOC paths that corrupt data or
hang, even when every Ethernet link is healthy, and report exactly which chip,
source, destination and NOC are affected.

## What it does
For every chip in the mesh, on NOC0 and then NOC1:

1. Each source core writes data patterns (walking 1s, walking 0s,
   checkerboard and its inverse, address-as-data, PRBS) to every destination.
   Writes are split into packets with several in flight before the barrier,
   similar to fabric traffic.
2. It reads the data back into a separate L1 buffer and compares it
   on-device.
3. It records per destination: mismatch count, first bad offset, expected and
   actual word, pattern and iteration, or a barrier timeout.
4. The host reads the results back, prints every failure plus summaries
   grouped by source and by destination, and fails the test if anything
   mismatched or timed out.

Destinations on each chip:
- Every Tensix core (all worker rows by default)
- The unreserved L1 of every other Ethernet core
- The DRAM NOC endpoints in NOC rows 0/1, addressed by coordinate inside an
  allocated DRAM buffer

PCIe, ARC and router-only tiles are never targeted. Each source writes to its
own slice of every destination, so concurrent sources never overlap.
Destinations run in batches of up to 64 per source.

## Tests
| Test | Sources | Use it to |
|------|---------|-----------|
| `TensixDeploymentNoc00EthSourceIntegrity` | Every Ethernet core, active and idle | Qualify the system; find bad Ethernet-core NOC paths |
| `TensixDeploymentNoc01TensixSourceIntegrity` | The four corner Tensix cores | Tell a source-side fault from a destination-side one |

## Usage
From the repository root:

```sh
./build_metal.sh --build-metal-tests
tt-smi -glx_reset
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_deployment --gtest_filter='*DeploymentNoc*'
```

- Slow dispatch is required: kernels on active Ethernet cores need it. Without
  it both tests are skipped.
- Run one test with `--gtest_filter='*DeploymentNoc00*'` or `'*DeploymentNoc01*'`.
- Test only the chips on one tray with `--ubb N` (N = 1-4). All trays are tested by default.
- Add `TT_LOGGER_TYPES=Test` for quieter output.
- `test_runner.sh` in the parent directory runs these as the "NOC tests" step.

Example, more passes and only worker rows 0/1:
```sh
NOC_TEST_ITERATIONS=10 NOC_TEST_TENSIX_ROWS=2 TT_METAL_SLOW_DISPATCH_MODE=1 \
    ./build/test/tt_metal/unit_tests_deployment --gtest_filter='*DeploymentNoc00*'
```

## Environment variables
| Variable | Default | Meaning |
|----------|---------|---------|
| `NOC_TEST_ITERATIONS` | 1 | Pattern passes per source/destination |
| `NOC_TEST_SLICE_SIZE` | 4096 | Bytes per source in each destination (multiple of 64) |
| `NOC_TEST_PACKET_SIZE` | 2048 | NOC write size |
| `NOC_TEST_BURST` | 8 | Writes of the slice in flight before each barrier |
| `NOC_TEST_VC` | 2 | Unicast write VC (2 matches fabric) |
| `NOC_TEST_TENSIX_ROWS` | 0 | Worker rows used as destinations, from row 0 (0 = all) |
| `NOC_TEST_TIMEOUT_MS` | 2000 | Barrier timeout in the kernel |
| `NOC_TEST_DRAM_ALL_NIUS` | 0 | Also target non-preferred DRAM NIUs when DRISC firmware is off |
| `NOC_TEST_WATCHDOG_S` | 120 | Fail and exit if one host step (compile, launch, finish, read) makes no progress for this long; reports the chip and UBB (0 = off) |

## Reading the output
Per-batch progress:
```
noc 1 batch 1/3: 384 sources ran, 26 failing source/destination pairs
```

Each failure names the chip (id, PCI BDF, tray/ASIC), source, destination and NOC:
```
NOC mismatch: chip 26 (0000:c3:00.0, ubb: 3, chip: 3) src eth ch 2 [noc0 2-1, logical 0-2, active]
  -> dst tensix [noc0 1-2, logical 0-0] @0x1fdc0 noc 1: 215 words in 6 transfers,
  first at +0x3c, expected 0x00008000, got 0x00008800 (WalkingOnes, iteration 0)
```

- `NOC mismatch`: data read back differs from what was written.
- `NOC timeout`: a write or read barrier did not complete. That source stops
  and its remaining destinations in the batch show as `NOC not run`.
- `NOC hang`: a source stopped making progress entirely. The line names the
  destination it was on; its later destinations show as `NOC not run`.

On Blackhole Galaxy, Ethernet channels 10 and 11 are unused by design. Failures
whose source or destination is one of them are logged as warnings and do not
fail the test or appear in the summary.

At the end, a `NOC FAILURE SUMMARY` groups failures per chip and NOC:
- **Recurring sources**: one source failing to several destinations, listed
  with those destinations. Points at the source core's NOC path.
- **Recurring destinations**: one destination failing from several sources,
  listed with those sources. Points at the destination core's NOC path.
- **Isolated pairs**: failures whose source and destination both fail only once.
- **Wedged chips**: chips that hung, with the NOC, batch and reason.

Each section is only printed if it has entries.

```
chip 26 (0000:c3:00.0, ubb: 3, chip: 3) noc 1: 26 failing pair(s), 1 source(s), 26 destination(s)
  Recurring sources (one source -> many destinations: suspect the source's path):
    src eth ch 2 [noc0 2-1, logical 0-2, active] -> 26 destinations
        dst tensix [noc0 1-2, logical 0-0]
        dst tensix [noc0 3-2, logical 1-0]
        ...
        dst tensix [noc0 16-3, logical 12-1]
```

Failures on every chip are reported in one run.

## Wedged chips
A chip is marked wedged when one of its sources stops making progress, or a
host read of the chip fails or does not return. It is also marked wedged when a
source aborted on a NOC timeout and the core's firmware does not finish within
`NOC_TEST_TIMEOUT_MS`: with transactions still outstanding, the firmware's
NOC-idle checks can trap the core, and waiting on it would block every other
chip. Each chip is polled
separately, so a wedged chip does not block the others:
```
chip 31 (0000:c8:00.0, ubb: 3, chip: 8) wedged in noc 0 batch 1/3: 1 source(s) stopped making progress; skipping it for the rest of the test
```
- The wedged chip is skipped for the remaining batches and NOC phases; every
  other chip keeps running, so the final report covers the whole system.
- If the chip can still be read, its results up to the hang are reported,
  including the `NOC hang` line for the stuck source.
- Wedged chips can't be closed cleanly, so after the report the test exits
  with code 2 without the usual teardown (the watchdog uses the same code).
  `test_runner.sh` runs the NOC tests last in each iteration and asks for a
  reset if this happens. Later tests in the same run (for
  example `Noc01` after `Noc00`) don't run. Reset the boards before the next
  run.
