# Hang capture: routers and test workers

`hang_capture.py` (capture, read-only), `hang_decode.py` + `hang_workers.py` (offline decode), `vc1_capture_lib.sh`
(arms them in VC1 runs), `test_hang_workers.py` (offline tests). RESULTS §94 has the router part; this file covers the
worker (Tensix) part added after the fourth VC1 hang (§98: all 48 routers alive and drained, so the wait is outside
the routers).

## What is captured

Routers: unchanged (stream registers, whole 512 KiB L1, TX-queue registers, heartbeat; same reads, same order).

Workers, after the routers (`--no-workers` turns them off). Reads only, tt-exalens safe mode, no halts, no writes:

| Phase | Cores | Reads |
| --- | --- | --- |
| `wscan` | every Tensix core of every chip (`functional_workers`) | mailbox head (0x60, 1488 B: launch-message ring, go messages, read pointer, go index), the test's result status word (0x23580), soft-reset register |
| `wl1` | selected cores | L1 [0, 256 KiB) (`--worker-window`); the whole 1.5 MiB for cores still running a test kernel (at most `--worker-full-max`, 32) |
| `wpass2` | selected cores | mailbox head, status word, soft reset and the routers' credit words (fabric connection table) again |
| `wniu` | selected cores | NIU counters of both NoCs (NOC_STATUS 0..15 and 48..61: requests sent, acks and responses received; `--no-worker-niu` skips) |

A core is selected if its go signal is GO with kernels enabled, or it ran the newest program on its chip, or its
status word carries the test's `0xabc0000x` status (`--workers-all` selects every core). "Still running" = GO in the
newest program, or status STARTED.

Program membership (capture and decoder alike): the host-assigned program id is 0 on this setup (job 308), so
programs are told apart by the launch slots' watcher kernel ids, which come from a process-wide counter (one per
`CreateKernel`) and so grow with each test. All Tensix cores execute the launch ring in lockstep (every go signal
advances every core's read pointer); a core left out of later programs keeps executing its stale slots with nothing
enabled. A core ran the chip's newest program when the slot it last executed is also its most recent one (highest
kernel id of its 8) and carries the newest program's kernel config base (one ring allocation per program). The
newest program is the highest kernel id among such slots. Cores outside it are "left out": decoded from their own most
recent slot, for reference, without checks. Dispatch kernels (launched by the host, mode host) are never part of it.

The 256 KiB window holds: mailbox, the fabric connection table (0x9170; per Ethernet channel the router's
coordinates, buffer, connection-info and producer-cursor addresses, and at 0x9200 + 16 x channel the credit word the
router writes), the kernel-config ring (0xa080: runtime args, semaphores, kernel binaries), and the test's
CommonMemoryMap from 0x1b480: local args, result buffer (0x23480), kernel config buffer (0x24480: the kernel's
config struct with every connection object and traffic config), sender credit / header / sync regions, the start
of the receiver payload buffers. The whole L1 adds the receivers' payload chunks (what a receiver polls).

Then, files only: the run's non-router kernel ELFs (`--kernel-cache`, default the `--meta cache=` value the lib
passes) are copied to `OUT/cache/kernels/<name>/<hash>/<risc>/`: test kernels keep their DWARF types (0.5 MB each),
dispatch kernels are stripped.

Output: `chip<C>/tensix_heads.bin` (scan, 1488 B per core (1232 before 2026-10-29) in `index.json` order), `chip<C>/tensix_<x>-<y>_l1.bin`
(noc0 x, y), `index.json` key `workers` (constants, per chip the scan list and selected cores with reasons, pass1 /
pass2 words, NIU counters), `format` 2. Timings `wscan_s`, `wl1_s`, `wpass2_s`, `wniu_s`, `kernel_elfs_s`.

Expected cost on P150x8: about 960 cores scanned in well under 1 s; ~10 to 15 selected cores per chip at 256 KiB
(~25 MB, ~1 s at the 27 MB/s the router L1 reads get), plus 1.5 MiB per still-running core; NIU reads ~0.1 s.
About 3 to 5 s for the whole capture (routers 1.7 s).

## What the decoder adds

Per captured core (`== Workers: Tensix cores of the last program (per core) ==`, `--quiet-workers` drops it):
- go signal, the launch slot used (current if GO, previous if DONE, the core's own most recent one if it is left
  out), kernel ids, kernel config base, enables, soft reset (which RISCs run);
- per RISC with a kernel, the binary at `kernel_config_base + kernel_text_offset` matched against the cached ELFs
  (`= tt_fabric_test_sender/<hash> (exact)`);
- for the test kernels (sender, receiver, sync): runtime args (CommonMemoryMap), result status / packets, compile-time
  args (from DWARF), and the kernel config struct decoded with the layouts gdb prints for that ELF (`ptype /o`):
  - sender: per traffic config packets sent / expected, destination (mesh, chip, core, address from its packet
    header and NoC fields), credit manager if enabled; local-sync semaphore;
  - receiver: per config packets validated / expected, current seed (checked against the initial seed advanced once
    per packet), buffer offset, the word it polls, the value it wants and the value there, and which packet's data
    the slot actually holds (seed sequence search: a later packet = overrun, none = stale or foreign data); for
    atomics the counter and its target;
  - sync core: global sync semaphore against `line_sync_val` x rounds (10 start + 1 end), local sync semaphore
    against N x (0, 1, 2, 3), the local cores;
  - every connection object (WorkerToFabricEdmSender): router, Ethernet channel, slots N, write counter, slot index,
    the credit word the router writes (free slots = N - (sent - credit)), teardown word, cursor read at open;
- NIU counters per NoC and any requests not answered;
- a WAITS line: what the core waits for, inferred from its L1 state.

After `== Traffic waiting ==`:
- `== Workers not finished (who waits for what) ==`: every test core of the newest program that has not written
  PASS (or is still GO), with its wait; non-test kernels still running (dispatch, expected); if every test core
  reported DONE with PASS, a line saying the wait is outside the test kernels (dispatch completion, host, other rank).
- `== Workers by rank (mesh) ==`: per rank (`--rank-bindings`, default the VC1 split bindings; mesh from the routing
  tables) and chip, counts of senders / receivers / sync cores by state, and the sync core's semaphores.
- `== Worker flows: sender vs receiver (packets) ==`: each sender config matched to its receiver by (mesh, chip,
  core, address): sent vs validated; only rows that differ are listed (`--all-worker-rows` lists all).
- `== Worker-router connections ==`: each connection object against the router's worker channel (E0 ch0: connection
  info at `LOCAL_SENDER_CH_0_CONN_INFO_ADDR`, read counter at +48, state at +96, producer cursor at +112, free-slots
  register, completions), and routers whose worker channel is open or closing with the core they point at; only rows
  that disagree are listed.

Worker anomalies join `== Anomalies ==`: credits the router counted but the worker never received (or the reverse),
packets the worker counted that the router never got, a connection open after its worker finished or closed while a
config still has packets, a receiver slot overwritten by a later packet, a sender that sent everything to a receiver
still waiting, a receiver with more packets than its sender sent, semaphores past an equality target, NoC requests
not acknowledged on a running core, DATA_MISMATCH, go DONE with status STARTED, configs not built.

Credit lag: the router writes the worker's credit word only while the worker is connected (`credit_worker` in the
v3 sender ERISC; the generic loop does the same). The teardown (`teardown_worker_connection`) marks the channel unused,
copies the read counter to the router's connection info (`edm_read_counter`, +48) and acks the worker; completions
of packets still in flight at the close are then credited to that connection info only, for the next worker to read
at its open. So a worker that closed its connection may hold a credit word up to N (the channel's slots) behind the
router's read counter. That is reported as an expected lag (a note, and a count in the connection summary). It is an
anomaly when the connection is still open, when the worker still runs and is out of free slots or has packets left
for that connection, or when the lag exceeds N. The hardware self-test shows lags of 2 (senders) and 1 (sync cores).

A capture without a `workers` part decodes byte-identically to before.

## Known limits and assumptions

- RISC PCs and BRISC-private memory (kernel globals, stack, the firmware's NoC counters) are not read: PCs need the
  debug interface, and tt-exalens marks Blackhole private memory unsafe. The wait point is inferred from L1 state.
- Benchmark-mode senders work on local copies of their connections; L1 holds the start state until the loop ends.
- Mux connections, latency kernels and 1D (low-latency header) destinations are not decoded.
- Fixed Blackhole addresses (dev_mem_map.h / dev_msgs.h / HAL), as of main 6d8a103 (merged 2026-10-28): mailbox at
  0x60 (launch at +16, 144 B each; go messages at +1168 as rings, go_messages[9].slot[8], one per launch slot per
  sub-device, so the current slot's go message is slot[rd_ptr]; go index at +1472), routing table 0x86e0, fabric
  connection table 0x9170, kernel-config ring from 0xa180, DEFAULT_UNRESERVED 0x1b580 (the test's memory map base;
  only the scan's status-word address depends on it, the decode takes addresses from the runtime args). Before the
  merge the go messages were one word per sub-device (go index at +1216) and every base after the mailbox was 0x100
  lower; captures from then decode with the older tools (git history). The ERISC mailbox kept one go message per
  sub-device (unchanged). A capture made with the old constants after the merge (job 142's) found no running test
  core: it read go_messages[0].slot[0] (DONE) for cores running in slot 1.
- Router side: channel 0 is the worker channel; its producer cursor sits 112 B after the connection info
  (erisc_datamover_builder.cpp layout). Counters compare directly only when the worker is the router's last producer
  (worker_xy at +32); a sender and the sync core can share a router channel one after the other.
- Kernel identification needs the ELFs of the run (the capture copies them; the full cache also works); struct
  layouts need the toolchain's gdb (`runtime/sfpi/compiler/bin/riscv-tt-elf-gdb`, `--gdb`); without it the worker
  decode falls back to mailbox, status and NIU only.
- Program membership relies on kernel ids growing with program creation and on one kernel config base per program.
  A left-out core whose most recent launch is exactly one ring lap old and happens to sit at the same kernel config
  base would be counted in; the decode would then show it with that older program's counts.
- The NIU counters cover every RISC and every kernel since reset (the self-test shows more acks than requests on
  some cores), so only a running test core with requests outstanding is flagged.

## Hardware self-test

`fabric_exp/consolidation/wcap_runs.sh selftest` (as `p1cap_runs.sh selftest`, RESULTS §94: one VC1 sanity run of
v3v2 at -O3 in a fresh cache, then the capture of the idle device and its decode, into
`results/wcapv3v2_o3_capselftest_hangcap/`). First run: job 308, 4.0 s, 26640 reads, 0 read errors, 1040 Tensix
cores scanned. With this decoder the capture shows:
- per core: `BRISC: ... = tt_fabric_test_sender|receiver|sync/<hash> (exact)`, soft reset `running: none` (teardown
  puts the RISCs in reset), `test memory map: local args 0x1b480, results 0x23480, kernel config 0x24480 | status
  PASS`, the compile-time args; for the last test (MultiMeshNeighborExchangeInter unicast_scatter_write) per chip
  a sender with `sent 3000/3000`, a receiver with `validated 3000/3000` and a sync core at `33/33`, `6/6`;
- `Workers not finished`: earlier tests' cores (the all-to-all receivers) listed as left out, and `none: every
  captured test kernel wrote status PASS and reported DONE (24 cores)`;
- `Workers by rank`: chips 0-3 rank 0 mesh 0, chips 4-7 rank 1 mesh 1, one sender, receiver and sync core each;
- `all 8 flow(s) complete on both ends`; `all 24 connection(s) quiet and consistent ...; in 10 of them the worker's
  credit word trails by up to the slot count ...`;
- `Anomalies`: the router liveness line and `none`.
An `unidentified` kernel, implausible counts or a decoder note about runtime args point at a wrong address
assumption (list above).

## Offline tests

`python_env/bin/python fabric_exp/consolidation/test_hang_workers.py [--keep DIR]` (no device; ~10 s): router-only
captures (p15cvc1_auto_hangcap, v3l3_o3_capselftest_hangcap) decode as their stored decode.txt and identically to git
HEAD's decoder; the hardware worker self-test (wcapv3v2_o3_capselftest_hangcap) decodes clean; then
`hang_capture.main()` runs against a fake tt-exalens device serving the fourth hang's router state and synthetic
Tensix L1 built from the real kernel ELFs' layouts (8 chips: sender with 7 configs, 7 receivers, sync core,
dispatch core, stale core): a clean run must decode with no worker anomaly, and a run with injected faults (sync
packet missing, sender parked in local sync, receiver slot overrun, lost router credits, a credit lag beyond the slot
count, NoC write not acked, router connection left open, an unreachable core) must name each one; the clean run
carries host-id 0 launches, a left-out core with a stale last-executed slot, and a closed connection with a credit
lag of 2 that must stay a note.
