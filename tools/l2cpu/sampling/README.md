<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
SPDX-License-Identifier: Apache-2.0
-->

# l2cpu/sampling: token sampling on the Blackhole L2CPU

An application of the L2CPU firmware runtime (`tools/l2cpu`, see its README) that turns rows of logits into next
tokens on the four x280 harts of L2CPU tile 0: greedy, temperature, top-k, top-p and a seeded draw, batch 1 to 32,
bit-exact with a host build of the same library. Logits arrive in the L2CPU's local DRAM (pushed there by the
Tensix link kernels of `tools/l2cpu/tensix`, or written by the host), tokens go back to the region and, optionally,
into a ttnn DRAM tensor through a TLB window.

    lib/       x280s: the sampling library (one C source for rv64gcv firmware and the x86-64 host .so), its
               specification (SPEC_NOTES.md), ctypes wrapper, NumPy model, unit tests, replay tool, QEMU corpus test
    include/   l2cpu_sampling.h: the region layout (embeds the link block of tensix/kernels/l2cpu_link.h)
    fw/        the firmware application (app.c on fw/app.h), its build fragment and QEMU test driver
    tools/     gen_sampling_py.py: the generated Python mirror host/l2cpu/sampling/layout.py
    scripts/   replay_on_chip.py
    tests/     test_sampling_mirror.py
    ../host/l2cpu/sampling/   host package: SamplingFw (boot through L2cpuCtl, descriptors, params, timing), replay

## What the library guarantees

- **Bit-exact host == RISC-V.** The same `x280s.c` built for x86-64 (`libx280s_host.so`) and for rv64gcv (scalar and
  RVV, any VLEN) returns the same token and the same statistics bytes for every input: only binary32 `+ - * /`,
  comparisons and exact conversions, no FMA (`-ffp-contract=off`), round-to-nearest-even forced at the entry points,
  an own `expf` (99.19 % correctly rounded over all inputs, max 1.023 ulp), RVV only for exact operations
  (argmax, NaN count, top-k prefilter, fused copy + row statistics). Checked by the QEMU corpus (1243 cases, VLEN
  128 / 512 / 1024, 4 harts) and on the chip (replay below).
- **Corner cases decided** (SPEC_NOTES.md): NaN logits count as `-inf`; `!(T > 0)` is greedy (lowest index of the
  maximum); `+inf` temperature is `FLT_MAX`; `top_k = 0` or `> 1024` means `K_MAX = 1024`; `top_p >= 1` or NaN means
  1; ties broken by index; the draw is SplitMix64 of `(seed, user, step)`; padding columns `>= vocab` are never read.
- **Fast paths change no result.** `SAMPLING_FAST=1` (default) enables the bf16 integer-key argmax, the exact top-k
  prefilter and the fused copy; every condition they cannot handle exactly falls back to the reference path.

## Request protocol

The region (64 MiB, `L2S_REGION_SIZE`) follows `include/l2cpu_sampling.h`:

| Offset | Content |
|---|---|
| 0x0001000 | link block (`l2cpu_link.h`): `req_seq`, `done_seq`, `wait_status`, `landed`, diag, push source table; the Tensix kernels' channel base |
| 0x0002000 | control: batch, vocab, vocab_padded, step_seq_base, flags, ring_wr, timing_count, seq_skips, stream config, ring geometry, work_timeout_us, logits and tokens descriptors, 32 per-user params, 32 `next_tokens` |
| 0x0008000 | timing ring (one 64 B record per published request) |
| 0x0020000 | output ring (2048 slots of 32 tokens) |
| 0x1000000 | coherent logits zone (Memory Port alias; the x280 reads rows in place) |
| 0x2000000 | uncached logits zone (System Port alias; read uncached only, at most 2 harts at a time) |

1. Host, once: descriptors (logits: `REGION`, `REGION_UC` or `NOC` = an interleaved ttnn tensor read through TLB
   windows; tokens: optional `NOC` tensor), per-user params, batch, vocab, flags.
2. Producer: write the rows, write barrier, `req_seq + 1`, doorbell (MSI catcher). The Tensix notify / push
   kernels do this; the host can too (`SamplingFw.issue`).
3. Hart 0 sees `req_seq` move, snapshots the control block, validates it when it changed, gives users 8h..8h+7 to
   hart h, samples users 0..7, waits for the workers, writes the ring slot and the timing record, then `done_seq`.
   Step of user u = `req_seq - step_seq_base`; tokens always go to `next_tokens[u]`.
4. Consumer: wait for `done_seq == req_seq` (the Tensix wait kernel or the host), read the tokens.

Streamed requests (`FLAG_STREAMED`): the producer publishes `req_seq` first and then `landed = (req & 0xFFFF) << 16 |
count` after each row (in the push kernel's order with 4 groups of 8); each hart starts on its rows as they land. A restart is transparent to requests: a WARM restart keeps the
sequence words and rings, and a request published while the harts were parked is served after it.

## Measured x280 time per request

P300, tile 0, 1750 MHz, image `bh-irq-sampling` (SAMPLING_FAST=1, ZB=1), Qwen3-8B recorded logits (V = 151936,
bf16), not streamed. Batch 1: coherent zone, 1000 rows per setting; batch 32: uncached zone, 100 requests x 32 users,
per-user settings cycling over the four ("mix"). x280 time = hart 0 from seeing `req_seq` to `done_seq`, medians, us.

| Case | read | sampling | wait workers | x280 total | rv64gcv build (ZB=0) |
|---|---|---|---|---|---|
| b1 greedy | 0 (in place) | 27.9 | - | 32.7 | 32.6 |
| b1 T0.7 k50 p0.9 | 0 | 86.6 | - | 91.5 | 92.1 |
| b1 T1.0 k0 p1.0 | 0 | 591.2 | - | 596.1 | 632.2 |
| b1 T0.6 k20 p0.95 | 0 | 72.9 | - | 78.0 | 78.3 |
| b32 mix (hart 0's 8 users) | 699.8 | 1339.5 | 128.8 | 2180.5 | 2246.2 |

Wake-up (host doorbell write to hart 0 seeing the request) 5.6 us at batch 1, 8.0 us at batch 32; host round trip
(host `req_seq` write to `done_seq` read) 43 / 101 / 606 / 88 us at batch 1 and 2194 us at batch 32. The rv64gcv
column is the same code without Zba/Zbb.

## ISA extensions of the x280 (measured)

Probed on the chip with guarded instructions (one instruction or a short sequence each, tile 0 hart 0):

| Extension | Result |
|---|---|
| Zba, Zbb, Zfh, Zfhmin, Zvfh, Zvfhmin, Zvfbfmin, Xsfvfnrclipxfqf | execute with correct results (Zvfbfmin f32 -> bf16 rounds ties to even, NaN -> canonical) |
| Zbs, Zicond, Zbc, Zfbfmin (scalar bf16), Zvfbfwma, Zvkb, Zvbb, Zicbom, Zfa | trap (illegal instruction) |

`misa` (0x8000000000b4112d: RV64 IMAFDCV + S, U, X) does not show the Z extensions. The library and the bulk copy
are built with `-march=rv64gcv_zba_zbb` (`ZB=1`, default; gcc 13.2 emits ~120 Zba/Zbb instructions in `x280s.c`);
`ZB=0` builds plain rv64gcv. Both builds are bit-exact with the host build (the gates below ran for both).

## Waits on the request path and their bounds

In addition to the runtime's waits (tools/l2cpu/README.md "Waits and their bounds"):

| Wait (where) | Bound | At the bound | Host sees it by | Recovery | QEMU test |
|---|---|---|---|---|---|
| stream consumer waits for `landed` (each hart) | `stream.timeout_us` without progress (default 100 ms) | hart parks with `L2S_ERR_STREAM_TIMEOUT` (arg = count needed); hart 0 then does not publish (`WORKER_DEAD`) | error word; `done_seq` stays | producer finishes or the host rewrites `landed`, then WARM restart: the request is served | stalled producer (5 ms bound) |
| uncached-reader gate (at most 2 harts) | 100 ms (`L2S_GATE_TIMEOUT_US`) | hart parks with `L2S_ERR_GATE_TIMEOUT` (arg = gate value) | error word | WARM restart (the gate is reset) | gate forced full |
| hart 0 waits for its workers | `work_timeout_us` (default 1 s) + the stream timeout for streamed requests | `L2CPU_ERR_WORK_TIMEOUT` / `WORKER_DEAD` (arg = hart), not published, hart 0 stays alive | error word | L2 (RNMI) park + restart | stalled worker (50 ms bound) |
| validation of a changed descriptor | no wait | hart 0 parks with `L2S_ERR_BAD_DESC` (arg = reason), nothing written | error word | fix the descriptor, WARM restart | bad descriptor |
| token write-back through a TLB window, NoC reads of `NOC` logits | **not bounded** (a load or the write barrier's read-back whose target never answers has no instruction boundary) | - | `done_seq` stays; heartbeat stall | chip reset | not modelled |
| host / Tensix wait for `done_seq` | the caller's timeout (`SamplingFw.wait_done`: 5 s; wait kernel: its own, writes `wait_status`) | exception / `wait_status = 0xDEAD0000 \| req` | - | read the error word | - |

Environment-dependent: the stream wait depends on the Tensix producer, the reader gate on the other harts, the NoC
accesses on the targets, and everything on the clock holder (tools/l2cpu README).

## Build and test

    make -C tools/l2cpu/sampling            # bh-irq-sampling, bh-poll-sampling (+ QEMU images), libx280s_host.so
    make -C tools/l2cpu/sampling test PY=<python with numpy + pytest>
    make -C tools/l2cpu/fw image PLATFORM=bh NOTIFY=irq APP=sampling [SAMPLING_FAST=0] [ZB=0]

`test` = mirror test (host gcc, rv64, rv32 JIT compiler), library host tests (43) + RISC-V build + `nm` check +
QEMU corpus (scalar VLEN 512, RVV VLEN 128 / 512 / 1024), and the firmware QEMU suites (irq, poll): 15 request
cases (batch 1 / 4 / 9 / 13 / 32, coherent / uncached / NoC zones, row-major and TILE, fp32 and bf16, remote tokens,
no ring), ordering stress (20000 batch-1 and 1000 batch-32 round trips), 5 stream cases (fast, slow and
stress producers), the restart suite, trap isolation, and the forced bounds above.

The library's QEMU corpus is synthetic (Qwen3-shaped rows made by `lib/tests/gen_corpus.py`). Recorded rows can be
added with `CORPUS_NPY=rows.npy` (uint16 bf16 `[N, V]`). A rows file (1000 Qwen3-8B rows = 303 MB) is not part of
the tree: record one with the decode harness of the Qwen3 example (`tools/l2cpu/examples/qwen3/decode_harness.py
--record-rows 1000 --rows-out ROWS.npy`, the logits row of each decode step), or make a synthetic one with
`lib/replay.py --make-synthetic ROWS.npy --n 1000 --bf16`.

## Run on the chip

    source tools/l2cpu/scripts/l2cpu_env.sh
    tools/l2cpu/scripts/l2cpu_run.sh "replay" $PY tools/l2cpu/sampling/scripts/replay_on_chip.py ROWS.npy \
        --rows 1000 --b32 100 --restart-b1 500 --restart-b32 50 --json replay.json

From Python (fresh chip reset, ttnn device open): `fw, region, info = l2cpu.sampling.boot(device)`; keep `region`
alive for the whole session and allocate it before any trace capture.

## Limits

- `top_k = 0` (or > 1024) costs about 5x top-k 50: up to `K_MAX` = 1024 candidates are sorted and exponentiated
  in scalar code (an exact vector `expf` would need its own bit-exactness proof).
- At most 2 harts read the uncached zone at a time (3 or more collapse 20x): batch 32 is bound by about 1.1 ms of
  row copies. ROWS streaming hides the push behind that bound but cannot go below it.
- TILE-layout logits on the chip and a bfloat8_b direct path are not done (TILE is tested in QEMU only).
- Streamed requests and NoC (`LOC_NOC`) logits are tested in QEMU only in this tree; host-written rows were
  replayed on the chip.
- Verified on one P300 chip, L2CPU tile 0, single-chip open.

Hardware: P300
