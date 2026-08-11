# Real-Time Profiler: Dispatch Core, Profiler Core, and Host Interaction

This document describes how the **dispatch core** (dispatch_s), **real-time profiler core**, and **host** interact to stream program execution timestamps and metadata to the host for profiling (e.g. Tracy).

---

## 1. High-Level Architecture

```
+-----------------------------------------------------------------------------+
| HOST                                                                        |
|                                                                             |
|   +-------------------+  +------------------+  +-------------------------+  |
|   | Init               |  | D2H Socket       |  | Receiver thread         |  |
|   | - Pick profiler   |  | - Config buffer  |  | - Drain pages           |  |
|   |   core            |  | - Page flow      |  | - Probe device clock    |  |
|   | - Create D2H      |  |   (PCIe)         |  | - Map + publish records |  |
|   |   socket          |  |                  |  |   to the record ring    |  |
|   | - Warm up clock   |  |                  |  |                         |  |
|   | - Start recv      |  |                  |  |                         |  |
|   +--------+----------+  +--------+---------+  +------------+------------+  |
|            |                     |                          |               |
|            | L1 write            | PCIe read                | PCIe read     |
|            | (config_buffer_addr)| (timestamp pages)        | (timestamp    |
|            |                     |                          |  pages +      |
|            |                     |                          |  clock reg)   |
+------------+---------------------+--------------------------+---------------+
             |                     |                          |               |
             v                     |                          |               |
+-----------------------------------------------------------------------------+
| DEVICE (per chip)                                                           |
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   | REAL-TIME PROFILER CORE (Tensix, closest to PCIe)                   |   |
|   | Kernel: cq_realtime_profiler.cpp                                    |   |
|   |                                                                     |   |
|   |   +---------------+  +-------------------------------------------+  |   |
|   |   | Mailbox (L1)  |  | Loop:                                     |  |   |
|   |   | - config_buf  |  |   PUSH_A -> NOC read buf A -> L1 ring     |  |   |
|   |   |   _addr       |  |   PUSH_B -> NOC read buf B -> L1 ring     |  |   |
|   |   | - state (R/W) |  |   TERMINATE -> exit                       |  |   |
|   |   +-------+-------+  +-------------------------------------------+  |   |
|   |           |                        ^ NOC read (timestamp data)      |   |
|   +-----------+------------------------+--------------------------------+   |
|               |                        |                                    |
|               | state (PUSH_A/B)       |                                    |
|               | NOC write              |                                    |
|               v                        |                                    |
|   +---------------------------------------------------------------------+   |
|   | DISPATCH CORE (dispatch_s)                                          |   |
|   | Kernel: cq_dispatch_subordinate.cpp                                 |   |
|   |                                                                     |   |
|   |   L1 carve-out realtime_profiler_msg_t:                              |   |
|   |     Ping-pong: kernel_start_a/b, kernel_end_a/b                     |   |
|   |     program_id_fifo, realtime_profiler_core_noc_xy,                 |   |
|   |     realtime_profiler_remote_state_addr                             |   |
|   |                                                                     |   |
|   |   Per-command: record start ts, FIFO program id, process cmd,       |   |
|   |     record end ts, signal_realtime_profiler_and_switch()            |   |
|   |     ... process command ...                                         |   |
|   |     record_realtime_timestamp(false); signal_realtime_profiler_and_ |   |
|   |     switch();  (NOC-write state to profiler core)                   |   |
|   +---------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------+
```

---

## 2. Data Flow: Program Timestamp to Host

```
  DISPATCH_S                 REAL-TIME PROFILER CORE              HOST
  (dispatch_s)               (cq_realtime_profiler)               (receiver thread)

       |                              |                                  |
       | 1. Record start ts,          |                                  |
       |    program_id into           |                                  |
       |    mailbox buf A or B        |                                  |
       | 2. Process command           |                                  |
       | 3. Record end ts             |                                  |
       | 4. Update state PUSH_A/B     |                                  |
       | 5. NOC write state --------> |                                  |
       |                              | 6. See state PUSH_A or PUSH_B    |
       |                              | 7. NOC read timestamp data       |
       | <----------------------------|    from dispatch_s L1 (buf A/B)  |
       |                              | 8. Push page to D2H socket       |
       |                              |    (PCIe write to host buffer)   |
       |                              | -------------------------------> | 9. wait_for_pages
       |                              |                                  |    get_read_ptr
       |                              |                                  | 10. Parse start/end ts,
       |                              |                                  |     program_id
       |                              |                                  | 11. InvokeProgramRealtime
       |                              |                                  |     Callbacks(record)
       |                              | <------------------------------- | pop_pages, notify_sender
```

---

## 3. Sync (Timestamp Calibration)

Host and device timestamps are aligned so consumers can relate device cycles to host time. The host retains a per-chip ring of clock **probes** — paired (host time, device ticks) reads — and places each record between the two probes around it: every record is published with its own affine mapping `device_cycle = frequency * host_ns + device_cycle_offset`, whose rate is the secant of the probe pair (or, for a record spanning several pairs, of its own two placements). There is no fitted model and no re-anchor policy: the mapping is pure interpolation over what was measured (`ClockSyncMapping`, owned by `DeviceClockSync`).

The tensix free-running cycle counter is a hardware register the NOC serves directly, so the host reads it through its own uncached TLB window with a single load, bracketed between two `steady_clock` reads. The device timestamp is known to have been sampled somewhere inside that bracket, so the probe goes at its midpoint and contributes half its width to `clock_sync.error`.

```
  HOST                                   REAL-TIME PROFILER CORE

    |  t1 = steady_clock::now()            |
    |  load WALL_CLOCK_L  ---------------> |  (served by the NOC; no RISC involved)
    | <--------------------- D ----------- |
    |  t2 = steady_clock::now()            |
    |  load WALL_CLOCK_H (latched by the L read, so outside the bracket)
    |  anchor D at (t1+t2)/2               |
```

Reading the low word latches the high word, so the pair is coherent and only the low read has to sit inside the bracket. The window is allocated once at configure time and the mapped address resolved once: the generic UMD register read holds a chip-wide mutex and rewrites the TLB configuration registers over PCIe on every call, all of which would land inside the bracket. The bracket *is* the error bound, so that width is the whole quantity being minimised.

Because no device software is in the path, a sample cannot be delayed by whatever the profiler core's push loop is doing, and sync costs the device nothing at all.

**Probe cadence.** The receiver probes every device strictly on the 250 us sync-interval cadence — serviced between drain phases so it holds regardless of publish latency, and never per-drain, since back-to-back drains would mint microseconds-wide chords whose noisy secants poison neighboring chords' rate brackets. Uniform chord width keeps certified bounds at the read-noise scale, and certification rests on adjacent probe-gap pairs staying under the DVFS transition spacing (the idle backoff sleep is capped at the earliest probe deadline so the cadence doesn't inherit its jitter). Each probe is the tightest of up to a few reads, taken only while the bracket is still wider than reads have recently been coming back at; warm-up at device construction gives the first record finalized chords to land between. Records in flight during a rare host-side receiver stall carry the honest fallback-tier bound instead of a certified one (the stress tests bound such records to <0.1% of the run; observed ~0.02% worst case).

**`clock_sync.error`** is an upper bound on the placement error of both record endpoints, built from two terms per **chord** (the interval between adjacent probes): the anchoring probes' half-brackets, plus an allowance for the device clock's rate changing inside the chord.

The rate allowance rests on how the clock actually misbehaves. DVFS steps AICLK from the ARC firmware's 1 ms timer, one **monotone** PLL glide per tick, and consecutive ticks can close to no less than ~0.95 ms (verified against Blackhole firmware; assumed for Wormhole, whose firmware ships as a blob). A chord is **certified** when the two-chord windows on both sides of it — widened by read noise — measure shorter than that spacing: at most one transition can then touch the chord, its neighbors are transition-free, and the neighbor secants (widened by their own read noise) bracket every rate inside the chord. For rates confined to `[f_lo, f_hi]` with `rho = f_hi/f_lo`, the worst in-chord misplacement is `span * (sqrt(rho)-1)/(sqrt(rho)+1)`, further refined per record by `(1/f_lo - 1/f_hi) * distance-to-nearest-probe` (both maps agree at the probes and their slopes live in the same band, so their difference grows at most at the slope width). On a quiet clock the whole bound sits near read noise (~1 us); across a real transition it grows to exactly what the probes cannot resolve.

**`record.frequency`** is smoothed without giving up transition honesty: consecutive certified chords whose rate brackets all *intersect* form a frequency window (the running intersection makes hidden-step creep impossible — every rate in the window provably lies inside it), and records publish the window's secant. Quiet clock: ~1 ppm jitter over the sliding 1–1.6 s baseline (read error / span). At a detected transition the intersection empties, the window restarts, and records around the step get the local chord's rate with its honest bracket, re-smoothing as the window regrows. The exactly-known skew between the smoothed slope and the placement interpolation is added to `clock_sync.error` so consumer-reconstructed endpoints stay covered.

A chord that cannot be certified — a receiver stall, a history edge — keeps a fallback allowance instead: the same formula evaluated with the envelope of every rate ever observed (the platform's AICLK range read from UMD until anything has been observed; the chord's own span when neither is available — the single-transition worst case is extremal among all rate trajectories inside a band, so the formula holds for any number of transitions). This tier deliberately trades the hardware-clamp guarantee for realism: it assumes an unprobed window holds no rate the clock has never visited, so it understates only when a first-ever rate excursion lands inside an unprobed gap — and in exchange it stays near the read-noise scale instead of pricing in an idle-frequency dive that almost certainly did not happen. Records whose start predates the 2 s probe history ride back from the oldest retained probe at the same envelope; `pin_start` exists so long-running programs avoid that path while their start is still covered.

Certification needs a chord's *successor* probe, so the receiver holds records whose end lies past `finalized_device_timestamp()` back for one probe (a fixed-capacity per-device buffer that provably flushes in full at every probe) rather than publishing them with the fallback bound.

The bound's guarantee excludes host-forced clock operations — forced AICLK/VDD, AICLK sweep, clock-scheme switches — which bypass the DVFS timer entirely; nothing in normal tt-metal operation does so.

---

## 4. Carve-out layout (conceptual)

| Location | Contents (`realtime_profiler_msg_t`) |
|----------|----------------------------------------|
| **Dispatch_s L1** | Ping-pong buffers, program_id_fifo, **realtime_profiler_core_noc_xy**, **realtime_profiler_remote_state_addr**, realtime_profiler_state. Host writes NOC XY and the profiler tensix L1 address of `realtime_profiler_state` for NOC signaling. |
| **Profiler tensix L1** | **config_buffer_addr**, **realtime_profiler_state**. |

Layout: `tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h`. HAL: `tt::tt_metal::realtime_profiler_msgs`. Not in `mailboxes_t`.

---

## 5. File / Component Reference

| Component | File(s) |
|-----------|--------|
| Dispatch_s (timestamp record + signal) | `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp`, `realtime_profiler.hpp` |
| Real-time profiler kernels | `tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp`, `cq_realtime_profiler_push.cpp` |
| Host init and receiver thread | `tt_metal/impl/realtime_profiler/realtime_profiler_receiver.cpp` |
| Clock read path + probe-interpolation mapping | `tt_metal/impl/realtime_profiler/device_clock_sync.cpp` |
| Shared struct + HAL accessors | `realtime_profiler_msgs.h` → `realtime_profiler_msgs` (generated) |
| Public API (register / unregister / is-active) | `tt_metal/impl/realtime_profiler/realtime_profiler.cpp` |
| Callback registration and record delivery | `tt_metal/impl/realtime_profiler/realtime_profiler_service.cpp` |
