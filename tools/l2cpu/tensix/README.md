<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
SPDX-License-Identifier: Apache-2.0
-->

# Tensix <-> L2CPU link

Kernels and host helpers that let Tensix data-movement kernels exchange data and requests with the SiFive x280
harts of a Blackhole L2CPU tile, without the host in the loop. The x280 side is whatever runs on the harts (see the
bring-up component in `tools/l2cpu`); `responder/` is a minimal test responder.

## Reaching the L2CPU tile from a kernel

| What | NoC tile | Address | Coherent with x280 caches |
|---|---|---|---|
| Local GDDR, Memory Port alias | (8,3) for CPUs 0-3 | `0x4000_3000_0000 + offset` (47 bits) | yes |
| Local GDDR, System Port alias | (8,3) | `0x3000_0000 + offset` | no |
| Local GDDR through the DRAM tile | DRAM bank (for CPUs 0-3: bank 5 = D5) | `offset` | no |
| MSI catcher (doorbell) | (8,3) | `0x2006_0000` | n/a |

The NoC address at the L2CPU tile equals the x280 physical address. Offset 0 of the local GDDR is offset 0 of the
DRAM bank, so a ttnn interleaved DRAM buffer's bank-5 page at `buffer_address()` is visible to the x280 at
`0x4000_3000_0000 + buffer_address()` (Blackhole bank base offset 0).

`get_noc_addr()` cannot express the coherent alias: tt-metal's Blackhole NoC API carries 36 local address bits
(`NOC_ADDR_LOCAL_BITS`) and its command-buffer writers mask `NOC_*_ADDR_MID` with `NOC_PCIE_MASK`.
`kernels/l2cpu_noc.h` programs the command buffer directly, with the upper 32 address bits in the MID register
(`l2cpu_noc_write`, `l2cpu_noc_read`, `l2cpu_noc_write_bulk`); the writes and reads are counted like
`noc_async_write` / `noc_async_read`, so the usual barriers cover them. The doorbell fits in 36 bits and uses the
stock `noc_async_write`.

**Doorbell:** a NoC write of a non-zero u32 to (8,3) `0x2006_0000` pushes it into the MSI catcher's 16-entry FIFO,
which raises PLIC source 6 while non-empty (machine external interrupt of the hart whose context enables it). A 0
would read back as "empty" on the x280 side, so the helpers never push 0.

## Channel and ordering protocol (`kernels/l2cpu_link.h`)

`kernels/l2cpu_link.h` is the single definition of the link block (offsets and meaning of every word). An
application that has its own shared region (e.g. a firmware's arena) embeds the block at a fixed offset of that
region and passes `region_base + offset` as the channel base to the kernels and to `ops.Link`; its other data
(push destinations, rings) can live anywhere in the region, given to the push program as an offset from that base.

A channel is a region of L2CPU memory with one 64-byte line per shared word: `req_seq` (producer), `done_seq`
(responder), `wait_status`, `landed` (streamed push), diagnostics, push source table, reply line. Rules:
1. A producer writes its data, waits for the write acks (`noc_async_write_barrier`), then publishes the new
   sequence number (`l2cpu_link_notify`: req_seq + 1, then the doorbell). The x280 side writes its data, `fence`,
   then `done_seq`.
2. A consumer reads the sequence number first and the data only after it matched (`l2cpu_link_wait`).
3. Sequence numbers only increase.
4. Every wait is bounded (table below); a dead responder never hangs the device.

Kernels: `l2cpu_notify.cpp`, `l2cpu_wait.cpp` (optionally copies the reply line into an output page),
`l2cpu_push.cpp` (ROW_MAJOR interleaved DRAM pages -> L2CPU memory through either alias; optional streamed mode:
doorbell first, then rows in a fixed order, each followed by a write barrier and a `landed` count),
`l2cpu_link_stress.cpp`. Python program builders: `ops.py` (`Link`), all trace-safe (per-step state lives in the
channel, runtime arguments are constant). generic_op's program cache ignores runtime-argument values, so `ops.py`
adds a `L2CPU_ARGS_ID` define per argument set; the push kernel reads its source table from the channel so that
an eager warm-up and a trace capture share one program even when the source tensor moved.

### Waits and their bounds

| Where | Waits for | Bound | At the bound | How the host sees it |
|---|---|---|---|---|
| `l2cpu_link_wait` (wait kernel, stress kernel) | `done_seq == req_seq` | `timeout_us` of wall clock, default 50 ms (`ops.WAIT_TIMEOUT_US`), runtime argument, <= 3 s; ticks = us x `L2CPU_WAIT_TICKS_PER_US` (1350, the measured AI clock; a lower AI clock only lengthens it). One poll is one 64 B NoC read (~1 us) | writes `0xDEAD0000 \| (req & 0xFFFF)` to `wait_status`, returns without copying the reply; the program ends normally, later programs in the trace run | read `LINK_OFF_WAIT_STATUS` (non-zero = timeout of request `low 16 bits`); a monitor thread polls it |
| `l2cpu_link_stress.cpp` | each round's wait | same per round | stops at the first timeout; `timeouts` = 1 in the DIAG result line | `test_link_stress.py` fails |
| `l2cpu_push.cpp`, `l2cpu_notify.cpp` | NoC write/read acks only (`noc_async_*_barrier`) | none in software: a NoC transaction to a live tile completes; to a clock-gated L2CPU tile it never does (chip hang, `tt-smi -r`) | n/a | keep the device open (clock on) for the whole session |
| host: `link_setup.open_link` | responder ready (status magic) | 5 s (`bringup_ttnn` ready_timeout) | exception | test fails |
| host: `test_link_timeout.py` | responder stop acknowledgement | 2 s | FAIL | exit 1 |
| host: `ttnn.synchronize_device` after a trace | queued programs | none on the host; bounded by the device-side bounds above (N x timeout_us worst case) | | wrap runs in `timeout <s>` |

`tests/test_link_timeout.py` stops the responder and replays a [notify, wait] trace: every wait returns at its
bound with the status word set and the trace completes.

**Uncached zones:** rows that will be read by the x280 uncached must be written through the System Port alias and
must never be touched through the coherent alias (and vice versa): the caches do not know the aliases are the same
memory.

## Measured (P300, chip 0, L2CPU tile 0, x280 at 1750 MHz, Tensix wall clock 1350 MHz; `tests/`)

| Measurement | Result |
|---|---|
| `test_link_stress.py --rounds 1000000`: notify -> x280 responder -> wait -> check 16 reply words | 1,000,000 rounds, 0 stale, 0 timeouts; 2.02 us per round trip |
| `test_link_trace.py --iters 10000`: notify + wait as two programs in a trace, `execute_trace(blocking=False)` | 5.97 us per iteration; seq words, wait status and output checked |
| one program launch inside a trace (100 notify programs) | 1.55 us |
| `test_link_timeout.py`: responder stopped, 20 traced [notify, wait] | trace completes, 50.0 ms per iteration (= the bound), wait status `0xDEAD0000 \| req` |

`bench_push.py`: 303,872-byte rows (one page each), trace-timed, destination bytes checked:

| Rows | Alias | Cores | ms per push | GB/s |
|---|---|---|---|---|
| 1 | coherent | 1 | 0.047 | 6.4 |
| 1 | uncached | 1 | 0.025 | 12.2 |
| 32 | coherent | 1 / 2 / 4 | 19.20 / 19.16 / 19.17 | 0.5 (the 2 MiB L3 write-allocates and thrashes) |
| 32 | uncached | 1 / 2 / 4 | 0.527 / 0.504 / 0.495 | 18.4 / 19.3 / 19.6 |

## Limits

- Verified on one P300 board, chip 0 (PCIe 0), L2CPU tile 0 (CPUs 0-3, NoC (8,3)) only. Tiles 1-3 have other
  coordinates and DRAM banks (tiles 2 and 3 share D7).
- Run with the watcher off: its device-side sanitizer classifies (8,3) as a Tensix tile and does not know the
  47-bit Memory Port addresses or the MSI catcher, so it flags (or stalls on) these accesses.
- Coordinates: kernels use translated coordinates; (8,3) is the same in raw and translated space on Blackhole.
  DRAM banks from `ttnn.cluster.get_dram_bank_table` are translated (the x280's own TLB windows need raw ones).
- The L2CPU clock runs only while a process holds the chip open (KMD power flag); a NoC access to the tile while
  its clock is off hangs the chip until `tt-smi -r`. Keep the device open for the whole session.
- `l2cpu_push.cpp` handles at most 64 rows per program.

## Tests (one process each from a fresh chip reset (`l2cpu_run.sh` does it), watcher off)

    make -C tools/l2cpu/tensix/responder
    source tools/l2cpu/scripts/l2cpu_env.sh
    tools/l2cpu/scripts/l2cpu_run.sh "link stress" $PY tools/l2cpu/tensix/tests/test_link_stress.py [--rounds 1000000]
    tools/l2cpu/scripts/l2cpu_run.sh "link trace" $PY tools/l2cpu/tensix/tests/test_link_trace.py [--iters 10000]
    tools/l2cpu/scripts/l2cpu_run.sh "link timeout" $PY tools/l2cpu/tensix/tests/test_link_timeout.py [--iters 20 --timeout-us 50000]
    tools/l2cpu/scripts/l2cpu_run.sh "push bench" $PY tools/l2cpu/tensix/tests/bench_push.py
