# test_x280_counter — X280 bare-metal proof-of-life (step 1)

Boots a single-hart bare-metal firmware onto the Blackhole **X280 (L2CPU)** and
polls, on the host, a heartbeat counter the firmware publishes to L3 LIM over the
NOC. A steadily increasing value proves the X280 firmware is alive.

Everything runs through tt-metal's UMD (the low-level `Cluster`) — a single
access path. We do **not** use pyluwen here: pyluwen + UMD in one process
corrupts ARC firmware. A board reset (`tt-smi -r`) runs before the device is
opened so the L2CPU is resettable (re-asserting reset on a running L2CPU is a
no-op on this hardware).

## Pieces

- **Firmware** — `tools/x280_bm/` (separate, bare-metal riscv64 build):
  `entry.S` + `src/counter.c`, linked at LIM `0x08000000` by `ld/x280-lim.ld`.
  Hart 0 increments a u64 at `0x08010000` ~every 1 ms (rdcycle-paced at the
  1000 MHz boot PLL); harts 1-3 park.
- **Host** — this example. Boot sequence mirrors tt-llm-engine
  `x280/host/loader.py`: assert L2CPU reset → NOC-write `counter.bin` to LIM →
  set the four hart reset vectors → step the L2CPU PLL → release reset → poll.
  The ARC reset-unit / PLL registers are reached as NOC reg writes to the ARC
  tile `(8,0)`; LIM + reset vectors as NOC ops to the L2CPU tile `(8,3)`.

## Build

```sh
make -C tools/x280_bm                                   # builds counter.bin
cmake --build build_Release --target test_x280_counter  # builds this example
```

## Run (from the repo root)

```sh
export TT_METAL_HOME=$PWD
export LD_LIBRARY_PATH=$(find build_Release -name '*.so*' -type f \
    -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
./build_Release/programming_examples/profiler/test_x280_counter --count 20
```

Expected: the counter starts near 1 and climbs by ~100 each 100 ms poll.

Flags: `--bin <path>` `--device N` `--l2cpu N` `--pll {200,800,1000,1750}`
`--interval-ms N` `--count N` `--no-reset` `--no-boot`.
