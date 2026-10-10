# Reproducer: dispatcher startup NOC-counter race (device close hang)

**Evidence class:** a forced-timing hardware reproducer.
- A test-only delay makes dispatch_s take its NOC-counter baseline late. Under that delay, device close hangs without the fix and completes with it.
- How often the race occurs under natural timing has **not** been established.

## Files

| File | What it is |
|---|---|
| `dispatch_startup_race_repro.py` | Opens one chip with fast dispatch and two command queues (`ttnn.CreateDevice(id, num_command_queues=2)`), then closes it, in a child process under a watchdog. It always exits. |
| Test-only delay (below) | **Repro only, never merge.** In `ncrisck.cc`, under `DISPATCH_KERNEL`, a busy-wait of about 4e9 wall-clock ticks (a few seconds) before `noc_local_state_init`. On the dispatch core this delays dispatch_s's NOC-counter baseline, so dispatch_d (same core) issues on that NOC first. It is not part of this commit. |

## Hardware

- **Required:** one Blackhole chip with the default (worker-core) fast dispatch. The race needs dispatch_s on the dispatch_d core, which is the Blackhole default.
- **Fresh upstream result:** forced-delay pairs on two P300 chips, one chip per run; natural controls on one of them.
- **Historical result:** four Blackhole chips on two P300 boards, kept separate below.

## Build and run

Check out this branch and build tt-metal normally (first time only: `./build_metal.sh && ./create_venv.sh && source python_env/bin/activate`). Only device-side sources change, and they are JIT-compiled at run time. Use a new `--jit-cache` directory for each arm.

1. Add the test-only delay. In `tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc`, in `_start()`, insert this right after the `do_crt1(...)` call and before the `noc_local_state_init` block:

   ```cpp
   #ifdef DISPATCH_KERNEL
       // REPRO ONLY - DO NOT MERGE. Forces the dispatcher startup race: start the NCRISC dispatch kernel
       // (dispatch_s) late, so its NOC counter baseline below is taken after dispatch_d, on the same core,
       // has already issued on that NOC. About 4e9 wall-clock ticks (a few seconds).
       {
           const uint64_t repro_delay_end = c_tensix_core::read_wall_clock() + 4000000000ull;
           while (c_tensix_core::read_wall_clock() < repro_delay_end) {
           }
       }
   #endif
   ```

2. Stock arm (expected: hang at close). Restore the two dispatcher files from the parent commit, then run:

   ```bash
   git checkout HEAD~1 -- tt_metal/impl/dispatch/kernels/cq_dispatch.cpp tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp
   python repro/dispatch_startup_handshake/dispatch_startup_race_repro.py --device-id 0 --jit-cache ./repro-cache-stock-delay
   ```

3. Fixed arm (expected: close completes). After the stock run and any required chip reset, restore this commit's dispatcher files and run again:

   ```bash
   git checkout HEAD -- tt_metal/impl/dispatch/kernels/cq_dispatch.cpp tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp
   python repro/dispatch_startup_handshake/dispatch_startup_race_repro.py --device-id 0 --jit-cache ./repro-cache-fixed-delay
   ```

Without the delay (`git checkout HEAD -- tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc`), both arms are expected to close.

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | PASS | the device opened and closed |
| 2 | HANG_AT_CLOSE | close did not finish within `--close-timeout` (default 60 s), or teardown reported that the dispatch cores did not finish |
| 3 | HANG_AT_OPEN | the device did not open within `--open-timeout` (default 900 s, which includes the JIT build) |
| 1 | ERROR | anything else |

Run behavior:
- **Fresh JIT cache.** Each run JIT-compiles firmware and dispatch kernels into a new temporary cache (`TT_METAL_CACHE`), so the patched sources are always the ones that run. Pass `--jit-cache DIR` to reuse a directory, but only with one variant per directory.
- **Operation timeout unset.** The script unsets `TT_METAL_OPERATION_TIMEOUT_SECONDS` for the child, so a stuck teardown blocks and is caught by the watchdog. If that timeout were set, close would instead return after logging "Exception waiting for dispatch cores to finish during teardown", which the script also reports as HANG_AT_CLOSE.
- **Runtime.** About 1–3 minutes for the first open (JIT build), then a few seconds per open and close. A hanging run ends at the close timeout.

**After HANG_AT_CLOSE, the chip's dispatch cores are stuck. Reset it (`tt-smi -r <device id>`) before running anything else on it.**

## Fresh upstream results (Blackhole, 2026-10-06)

Using the identical external 4-billion-tick delay before the NCRISC dispatch counter snapshot, stock opens in 2.86/2.88 s on two P300 chips but emits no CLOSED marker within the 60 s close budget. Both watchdog verdicts are HANG_AT_CLOSE, exit 2, with child exit -15; each full process takes 65.0 s. Fixed opens in 2.86/2.83 s and closes in 2.16 s on those same chips, exit 0 (process times 7.8/7.7 s). The natural-timing pair on one chip passes both arms: open 2.75/2.78 s, close about 0.013 s from the phase markers (rounded to 0.01 in the verdict), process 5.6 s each. Targeted resets after the two stock close hangs succeed in 41.1/44.2 s. No run hits the outer process timeout.

All six attempts, their exact source roots and 132 retained ELF hashes were independently checked. The kit production patch equals the original tested peer port; the script is byte-identical, and each forced arm contains the exact separate timing patch. The only matched `.text` differences are `cq_dispatch/BRISC` and `cq_dispatch_subordinate/NCRISC`, including XIP images. This comparison does not prove that every compiled image was loaded. Original phase observations, image manifests and reset receipts are retained; no fresh stack or NoC-counter capture is claimed.

## Historical results

Measured on the earlier tree on which this fix was developed; these are separate from the fresh upstream pair above.

| Variant | Chips | Result |
|---|---|---|
| stock + forced delay | 4 / 4 | **hang at close**: open about 2.5 s; close not finished after 60 s; host blocked in `MeshDevice::close` → `DispatchKernelInitializer::teardown` → `wait_for_dispatch_cores` |
| fixed + forced delay | 4 / 4 | **closes**: open about 2.4–2.5 s, close about 2.07 s (the close waits out the remaining delay) |
| stock, no delay | 1 | closes: open 2.48 s, close 0.115 s |
| fixed, no delay | 1 | closes: open 2.44 s, close 0.114 s |

- **Tree:** an earlier tt-metal tree carrying the same dispatch_d/dispatch_s export/merge logic and the same `ncrisck.cc` wrapper.
- **Delay:** the identical delay block.
- **Arms:** one chip per arm and a fresh JIT cache per arm. Each batch of arms started from a full reset, and no arm ran on a chip after a hang on that chip.
- **Fresh port:** the separate `4ff4adaa` stock/fixed results above reproduce the forced-delay failure and successful repair on two chips. Natural frequency remains unknown.

## Reproducer-only delay

The timing injection is the block in step 1 above (the same block is in the PR description). It is not part of this fix commit. Apply the same block to both arms when running the forced-timing pair. Use a new persistent `--jit-cache` directory per arm to retain the compiled images; do not reuse one across source variants. The script and this README are part of the fix branch so the public source is directly reviewable.

For one visible chip of a P300 board, set `TT_MESH_GRAPH_DESC_PATH` to `tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto`; without it, tt-metal fails with a custom fabric mesh graph descriptor error. Select the physical chip with `TT_VISIBLE_DEVICES`; it appears as device 0 inside the process. This descriptor does not change the measured hardware type.
