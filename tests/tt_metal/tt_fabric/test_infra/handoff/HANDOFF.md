# Fabric test infra speed-up: handoff

Branch `aagarwal/fabric-test-infra-opts` on top of `ff89f4eaec4` (main, 2026-10-27): commit `e3539e34225` has the
changes, the next one adds this `handoff/` folder.
All work is in `tests/tt_metal/tt_fabric/test_infra`; nothing outside the test infra was changed. This `handoff/`
folder (this doc, validation yamls, log-parsing scripts) is temporary: delete it before the PR merges.

## Motivation

The fabric sanity and stability suites took a long time on BH Galaxy (8x4): about 15 min per CI run of the short
stability suite (11 min after the first allocator fix), and 6.3 h for the long suite. The goal was to cut that without touching tt-metal, and to size the
infra for 64x4 meshes, not just one Galaxy. Ground rules followed throughout:
- No changes outside the test infra.
- No program reuse across iterations (the number of configs would blow up).
- Measure before optimizing.

## Results (8x4 BH Galaxy)

| Suite | Original | Now | Speedup | Measure |
|---|---|---|---|---|
| Stability short (CI) | 64.7 s | 11.4 s | 5.7x | Per iteration, warm cache, device open and one-time compile excluded |
| Stability mid (100k packets) | 221 s | 56.9 s | 3.9x | Device time: sum of each test's slowest receiver loop |
| Stability long (1M packets) | 2259 s | ~580 s | ~3.9x | Per iteration; estimated from mid (last measured: 717 s, before the last two device changes) |
| Sanity (1300 tests on 8x4) | ~541 s | ~351 s | 1.5x | Cold, node-local cache, node-specific control-plane time excluded; original includes the CSV fix |

- "Original" for the short suite already includes the allocator fix. With the old allocator, 10 short iterations took
  about 15 min.
- Without the CSV fix, the sanity suite took 1946 s on an NFS-backed setup, mostly opening the latency CSV once per
  sample. With the fix: 581 s. The gain on CI's local disk will be smaller.

## What worked, and how much each step gave

All of these are in the commit. Everything except the CSV and allocator fixes is behind a `TT_FABRIC_TEST_*` env knob,
default off.

Stability short suite, seconds per iteration (host time dominates here):

| Step | Knob | s / iteration |
|---|---|---|
| Original (with allocator fix: lazy multicast destination search, ~1 s -> 20 ms per mcast test) | none | 64.7 |
| Batched non-blocking host writes (`create_kernels` 355 -> 62 ms/test), SFPU gen/verify, sender hot state | `BATCHED_HOST_WRITES`, ... | 22.4 |
| Shared kernels: one kernel object per identical kernel per chip (1192 -> 220 per test) | `SHARED_KERNELS` | 13.9 |
| Parallel per-chip program build, host writes deferred and issued in order | `PARALLEL_BUILD` | 11.4 |

Stability mid suite, device time (device time dominates mid and long):

| Step | Knob | Device time |
|---|---|---|
| Original | none | 221.1 s |
| SFPU payload generation (sender core's compute engine) and verification (receiver's), 8 mux full-size passes | `SFPU_PAYLOAD`, `SFPU_VERIFY`, `MUX_FULL_SIZE_ITERS=8` | 70.1 s |
| Sender hot state in local memory instead of L1 (0.45 -> 0.24 us/packet; Linear unicast 1.7x) | `SENDER_HOT` | 65.4 s |
| SFPU verify for scatter writes | (part of `SFPU_VERIFY`) | 56.2 s |
| Sender micro-opts: slot cache, batched flush (neighbor 215 -> 176 ns/packet; suite is fabric-bound) | `SENDER_SLOT_CACHE`, `SENDER_BATCH` | 56.3 s |
| Per-config SFPU choice and multi-tile payloads (below) | | 56.9 s (no runtime cost) |

SFPU details (the largest device win; e.g. 4 KB Mesh unicast 10.7 -> 1.39 s):
- SFPU vs local fill/check is chosen per config at runtime (a 4th metadata word). Every sender/receiver core declares
  the same CBs (placeholders where unused), because the CBs on a core are part of every kernel's build hash. So SFPU
  and local cores share one kernel build. This removed the compile overhead SFPU used to add: cold sanity went from 41
  to 28 tests that stop to compile (125 -> 108 s; the original is 107 s).
- Payloads in 16 B steps, up to 4 Int32 tiles (15232 B, the largest BH packet). The verifier masks the last partial
  64-word block.
- Configs fall back to the local path when the payload isn't a multiple of 16 B, the receive ring has 65536+ slots
  (`cb_reserve_back` counts free pages in 16 bits), a receiver has more than 8 configs (only the extra ones fall back)
  or a sender more than 64 (the whole core falls back). A per-test `SFPU coverage: X/Y` log line shows the split; on
  sanity, 74% of sender and 77% of receiver configs use SFPU.
- At 16-256 B on Mesh, SFPU is 2-12% slower than the local path: the sender must send SFPU pages in strict round-robin
  order, so it waits on a full connection instead of skipping it. SFPU wins from 32 B on Linear and from 512 B on
  Mesh, by up to 7-10x at 4 KB.

Tried and not adopted:

| Experiment | Result |
|---|---|
| Several sender cores per link (`SENDERS_PER_LINK`) | Helped before hot state and SFPU; no gain after (mid 56.2 -> 56.9 s) |
| Skip-when-full and fair rotation between connections (`SENDER_SKIP`, `SENDER_ROT_*`) | Mesh atomic_inc got worse (blocking sends advance every flow in lockstep, already at the link's packet rate) |
| Extra command buffers, batched credit returns, connection ordering (`SENDER_CMDBUFS`, `SENDER_CREDIT_BATCH`, `SENDER_CONN_ORDER`) | No gain |
| Second data-movement core pipeline for the sender | Parked: only pays if the sender can skip blocked configs |
| No flush on credit-return packets (`CREDIT_RETURN_NO_FLUSH`) | No effect |
| Mux v2 for 8x4 throughput | Not needed at 8x4 |
| Sorting tests by fabric setup | ~4 s on sanity: tt-metal already reuses router builds within a process |
| Latency reporting in a thread or at suite end | At most 5-7 s once the CSV fix was in |
| Compiling a fabric setup's kernels in one batch | Needs an internal API (`ProgramImpl::compile`); the public `program_preparation::prepare` is per workload and not safe to call from several threads |

## Where we stand

Performance:
- Mid and long: about 70% of the remaining time is limited by the fabric (busiest link bandwidth, single-link packet
  rate), not by test code. Linear groups are still bound by the sender's per-packet cost.
- Sanity and short: device reopens (~150 s of router compiles cold, inside tt-metal), kernel compile (~108 s cold,
  serial test to test), and host setup.
- Known regressions against the original: Mesh atomic_inc mcast +6% (recovered with atomic_inc flush off);
  atomic_inc unicast -9% from the hot sender (still 1.6x faster than original; accepted for 1.7x on Linear).

Code readiness:
- Validated on Galaxy: sanity 1300/1300; payload sweeps 16 B-15232 B (unicast/mcast, write/fused/scatter); a core
  mixing SFPU and local configs; 21 corruption canaries caught; mid perf unchanged by the per-config change.
- Not PR-ready yet: prototype knobs, instrumentation still in, and the base is 200+ commits behind main. Main has since changed test_infra (#62323 binds test programs to their MeshDevice with
  `CreateProgram(MeshDevice&)`, #60499 multimesh, #60814 `tt_align.hpp`), so expect rebase conflicts in program
  creation.

## Next steps (in priority order)

1. **64x4 sizing.**
   - The hot sender path only fits ~39 configs per sender (2 KB of local memory at 48 B + 4 B per config,
     `tt_fabric_test_sender.cpp:117`); a 64x4 all-to-all has 255 per sender, so it falls back to the slow L1 path.
     Options: keep the L1 state as the fallback for big configs, shrink the hot state (~32 B), consolidate it (e.g. per
     connection), or split a sender's configs across cores by budget (the `SENDERS_PER_LINK` split code can seed this).
   - Mux v1 tops out around 20 clients, so 64x4 needs mux v2. Mux v2 has no header-only channels, and receivers return
     credits over header-only channels today: work out how receivers connect (e.g. credits as full-size packets, or a
     separate path).
   - SFPU caps (8 configs per receiver, 64 per sender) need revisiting at that size.
2. **Productize the knobs.**
   - Make default: `SHARED_KERNELS`, `PARALLEL_BUILD`, `BATCHED_HOST_WRITES`, `SENDER_HOT`, `SENDER_SLOT_CACHE`,
     `SENDER_BATCH`, `SFPU_PAYLOAD`, `SFPU_VERIFY` (depth 4), `MUX_FULL_SIZE_ITERS=8`, `FAST_PATTERN` (unrolled local
     fill 1.14x / validate 1.7x, for configs that fall back from SFPU).
   - Drop: `SENDER_CMDBUFS`, `SENDER_CREDIT_BATCH`, `SENDER_CONN_ORDER`, `SENDER_SKIP`, `SENDER_ROT_BURST`,
     `SENDER_ROT_WAIT`, `CREDIT_RETURN_NO_FLUSH`, `SENDERS_PER_LINK`.
   - `ATOMIC_INC_NO_FLUSH`: turn into a yaml `flush:` option on plain `atomic_inc`. Main's yaml toggle covers only
     `fused_atomic_inc` and scatter with `atomic_inc: true`.
   - Keep `CORRUPT_PAYLOAD` (1 = mid word, 2 / 3 = second / third-to-last word) as a test hook for the verifiers.
3. **Remove instrumentation** once you're done measuring: `PERF_BREAKDOWN` and its `_HOST_DETAIL`, `_SEND_DETAIL`,
   `_HOT_DETAIL`, `_SCHED` timers (`tt_fabric_test_perf_breakdown.hpp` and the kernel-side macros), the kernel
   signature counter, the per-generator `SFPU_DIAG` info log, the extra `program.impl()` call in `run_programs`, and
   this `handoff/` folder. Keep the SFPU coverage line at debug level if useful.
4. **Test coverage.**
   - atomic_inc: make flow control a no-op (the receiver only polls a counter; confirm on bigger meshes) and use the
     saved time to run both flush modes.
   - Port the new sanity coverage on main (scatter chunk options, flush variants, strict fused validation) to the
     stability yamls.
   - Decide whether the SFPU size sweeps and canaries (see below) go into the repo.
5. **Smaller items.**
   - One sharded SFPU buffer per (address, size) per chip instead of ~1400 single-core buffers per test (~0.5-0.8 s
     per short iteration).
   - Infra bug: the memory maps' highest usable address overshoots real L1 by 0x11400 (SFPU code caps at real L1).
   - Fewer kernel builds: turn count compile-time args (traffic configs, connections, sync cores, muxes to terminate,
     config size) into runtime args, so fewer tests stop to compile. Re-measure the cold short suite's compile stops
     first; needs a hot-path perf check.
   - Sort tests by fabric setup: add the router bandwidth-telemetry flag to the reopen key first, or bandwidth groups
     can land on non-telemetry routers.
   - Keep SFPU page order per connection, so the sender can skip a full connection (fixes the small-payload Mesh loss).
6. **Open investigations.**
   - A ~5 s step on some test launches, on both NFS and local disk; cause unknown.
   - Kernel launch +20 ms/test after scatter SFPU verify (may be gone with shared kernels).
   - Parallel build threads each run ~5x slower than serial (contention, not identified).
   - atomic packets cost more than writes (Mesh 1.30 vs 1.17 s); Mesh atomic_inc mcast is credit-bound (1149 ms with
     flow control vs 678 without).

Everything can land as a single PR; it doesn't need to be split.

## Gotchas

- The receiver's `poll()` treats a packet as landed when its LAST word is right, so corrupting the last word hangs the
  test instead of failing it.
- LLK addresses `copy_tile` / out-of-order `pack_tile` tile indices in CB pages, not tiles. Multi-tile payloads set
  the CB's page size to one tile around those calls (`TileSizedPages`; not on the math thread, which has no CB
  interface).
- A CB must have fewer than 65536 pages (16-bit free-page count in `cb_reserve_back`).
- `read_tile_value` in compute kernels is an unpack-to-math/pack handshake: every TRISC must make every call.
- Writes without flow control plus a fast (SFPU) sender overrun the receive ring and hang. That's expected, not a bug.
- An NFS-backed kernel cache inflates compile and reopen times (warm reopens 79-162 s vs 19-35 s on local disk).
- A yaml with explicit `allocation_policies` switches the allocator to static policies; with flow control, a sender
  policy can hit the mux header-only channel limit.

## Measuring and validating

Everything below is in this folder. Pass yamls to `test_tt_fabric --test_config` by absolute path: a relative path is
looked up in `test_infra/test_yamls/`.

- Measure with a node-local kernel cache (`TT_METAL_CACHE` on local disk). An NFS-backed cache inflates compile and
  reopen times.
- Run the "now" numbers with the knobs being made default: `TT_FABRIC_TEST_SHARED_KERNELS=1 PARALLEL_BUILD=1
  BATCHED_HOST_WRITES=1 SENDER_HOT=1 SENDER_SLOT_CACHE=1 SENDER_BATCH=1 SFPU_PAYLOAD=1 SFPU_VERIFY=1
  MUX_FULL_SIZE_ITERS=8` (each prefixed `TT_FABRIC_TEST_`).
- Scripts (they read the run's log; take baselines before the instrumentation is removed):
  - `scripts/log_phases.py [--stops] <log>...`: fabric opens, program launch (where kernels compile on a cold cache;
    `--stops` lists each test that waited on a compile) and device wait.
  - `scripts/device_time_by_group.py label=<log>...` and `scripts/device_time_by_test.py label=<log>...`: device time
    (slowest receiver loop) per test group or per test, from runs with `TT_FABRIC_TEST_PERF_BREAKDOWN=1`. The second
    also shows each test's SFPU coverage.
- Perf slices: `yamls/stab_short_1it.yaml` and `yamls/stab_mid_1it.yaml` are the stability suite's short tests and a
  100k-packet version, at one iteration each. The mid numbers above come from the latter.
- Validation (all must pass with SFPU on), worth re-running after the rebase and knob changes:
  - `yamls/sfpu_sizes.yaml`: 16 B-4352 B in 16 B steps, write/fused/scatter, Linear and Mesh, enough packets to wrap
    the receive rings.
  - `yamls/sfpu_small.yaml`: 16-272 B with 4 configs per receiver core, so 16 B rings stay under 65536 slots and use
    SFPU.
  - `yamls/sfpu_multitile.yaml`: 4112 B-15232 B (custom max packet size), unicast/mcast, write/fused/scatter.
  - `yamls/sfpu_mixed.yaml`: one sender core and one receiver core mixing SFPU and local configs.
  - `yamls/sfpu_overhead.yaml`: run with SFPU on and off to compare device time per payload size.
- Corruption canaries (each must fail with a data mismatch): `yamls/canary_tail_*.yaml` with
  `TT_FABRIC_TEST_CORRUPT_PAYLOAD=2` and `=3`, `yamls/canary_mid_local_16.yaml` with `=1` (a config that falls back to
  local checking), and `yamls/sfpu_mixed.yaml` with `=2`. Run each on its own: the first failure ends the run.
- The `SFPU coverage: X/Y` log line shows which configs actually took the SFPU path.
