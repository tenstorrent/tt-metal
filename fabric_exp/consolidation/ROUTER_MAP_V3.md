# Router cost map: v3q, measured by PC sampling (2026-10-26)

Where each ERISC's time goes, per traffic configuration, for the lead design `fabric_exp/variants/v3q` at -O3 on
bh-lb-120-a09u32 (2D 2x4 mesh and 1D line, one link, keepalive 1000, telemetry off). This replaces the v2-era estimates
in ROUTER_MAP.md for v3. Method and trials: RESULTS §98 ("Phase/cost analysis"). Data: `fabric_exp/results/pcs3_*`.

## Method

- **Sampling.** While one long row runs (about 8 s of traffic), `pc_sampler.py` reads every router ERISC's
  program counter through its Ethernet core's debug bus (select `erisc0_pc` / `erisc1_pc`, read the data register).
  The core is not halted or stepped. About 78k reads a second over 96 ERISCs, so about 5,500 samples per ERISC per
  row. Rates with and without the sampler were equal to 0.01 cycles/packet, and every row's rate matches the suites.
- **Mapping.** Kernels run from the kernel config window, so a PC maps to the ELF through the core's launch message
  (config base + text offset). Each core's build is identified by its loaded image (equal to the cache's `.xip.elf`).
  The run's own compile commands are replayed with `-g` (identical loadable code, checked on all 68 builds), and
  `addr2line -i` gives each address's inline chain.
- **Sections.** By the helper an address is inlined from, else by the loop-body line (v3q sources); see
  `pcs_sections.py`. "Checks" are what every pass does whether or not there is work (is there a packet and room? a
  completion? a connection change? any forwarding work?). Work sections are sends, credits, local delivery, forwards,
  decode and admission, completion frames and retire.
- **Cycles per packet.** A section's share of the ERISC's samples times the row's cycles/packet. The first and last 5%
  of each ERISC's samples are dropped (outside the traffic window). One standard error on a 10% section is about 0.4
  percentage points (0.5 cycles).

## Caveats

- **Waiting shows up in polling sections.** An ERISC that does not set the rate spends its spare time polling: E1 in
  "poll", E0 in its checks. So E0's checks are about 45 cycles/packet where E0 sets the rate (1-hop writes) and 70–80
  where E1 does (scatter, fused). The difference is waiting, not extra work.
- **Stalls are counted on the stalled instruction.** The PC is the instruction in decode/execute, so time waiting on a
  register read or L1 load is charged to the instruction waiting for it. Line-level attributions can be one
  instruction late (skid); section totals are not affected.
- **Routers carrying two flows** (the forwarded row's middle E0, both bidirectional ERISCs) do two packets per
  interval; their per-packet costs are half the figures shown.
- **Multicast rows** show the mean over all busy ERISCs of that kind (12 E0s and 20 E1s in 2D).
- **Multi-mesh (VC1-enabled) routers** are not covered yet.

## E0 (sender_erisc), cycles per packet

The busiest E0 of each row: the sender for 1-hop rows, a middle router for 3-hop and forwarded rows.

| Row (cycles/packet) | E0 work | send | credits | checks | loop |
| --- | --- | --- | --- | --- | --- |
| 1-hop E/W write 4352 B (126.9) | 52% | 57.9 | 8.3 | 54.4 | 6.4 |
| 1-hop E/W write 16 B (115.7) | 57% | 56.7 | 9.3 | 44.9 | 4.8 |
| 1-hop N/S write 4352 B (127.3) | 50% | 56.0 | 7.9 | 56.7 | 6.6 |
| 1-hop E/W scatter 4352 B (143.9) | 45% | 56.9 | 8.4 | 71.0 | 7.7 |
| 1-hop E/W fused write+atomic (153.3) | 43% | 57.9 | 8.3 | 78.7 | 8.4 |
| 1-hop E/W atomic inc (124.9) | 53% | 58.2 | 8.6 | 52.6 | 5.5 |
| 3-hop E/W write, middle router (142.8) | 64% | 63.0 | 28.1 | 47.4 | 4.3 |
| multicast all-to-all, mean (141.1) | 59% | 41.2 | 41.9 | 53.2 | 4.8 |
| forwarded write, middle router, 2 flows (125.7) | 72% | 59.8 | 31.2 | 31.9 | 2.9 |
| 1-hop E/W bidirectional (135.0) | 50% | 57.6 | 9.8 | 61.6 | 6.0 |
| 1D 1-hop write (129.2) | 48% | 53.1 | 8.4 | 60.3 | 7.5 |
| 1D 3-hop write, middle router (140.2) | 66% | 60.4 | 32.7 | 40.1 | 6.8 |
| 1D multicast, mean (145.0) | 40% | 35.6 | 22.2 | 77.5 | 9.6 |

- **send:** `send_packet` (worker or forwarding): the pre-send group (payload size, TX-queue status), the data
  command, slot bookkeeping, the post-send group and the far receiver's packets-received update.
- **credits:** the worker credit write, or for forwarding channels the completion read and bookkeeping plus the credit
  to the upstream router.
- **checks:** the worker's packet-and-room test, its completion test, the connection semaphore poll, the VC0
  free-slots group read and the forwarding channels' idle test.
- **loop:** loop control and the per-round work (out of line).

Within the 1-hop E/W 4352 B sender (126.9 cycles/packet, no spare time): send 57.9 · worker send check 15.8 ·
forwarding idle test 11.1 · connection poll 10.7 · group read 9.5 · worker credit 8.3 · completion check 7.2 ·
loop 3.9 · per-round 1.8. Inside the send, TX-queue register writes are the largest item (about 13), then TX-queue
reads (about 7) and the payload-size load (about 6.5).

## E1 (receiver_erisc), cycles per packet

The busiest E1 of each row: the receiver for 1-hop rows, a middle router for 3-hop and forwarded rows.

| Row (cycles/packet) | E1 work | poll | decode+admit | local | forward | completions | retire+atomic | loop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1-hop E/W write 4352 B (126.9) | 78% | 15.4 | 21.8 | 35.4 | 0.0 | 29.0 | 19.7 | 5.7 |
| 1-hop E/W write 16 B (115.7) | 82% | 11.1 | 19.3 | 36.9 | 0.0 | 26.3 | 17.3 | 4.8 |
| 1-hop N/S write 4352 B (127.3) | 80% | 13.2 | 22.2 | 38.3 | 0.0 | 28.8 | 18.9 | 5.9 |
| 1-hop E/W scatter 4352 B (143.9) | 89% | 6.7 | 19.0 | 74.0 | 0.0 | 23.0 | 16.8 | 4.5 |
| 1-hop E/W fused write+atomic (153.3) | 91% | 5.6 | 19.4 | 80.2 | 0.0 | 27.3 | 16.3 | 4.4 |
| 1-hop E/W atomic inc (124.9) | 86% | 7.9 | 19.5 | 56.9 | 0.0 | 19.7 | 16.0 | 4.9 |
| 3-hop E/W write, middle router (142.8) | 89% | 6.8 | 33.8 | 0.8 | 51.7 | 24.2 | 20.6 | 4.8 |
| multicast all-to-all, mean (141.1) | 83% | 8.0 | 50.4 | 16.9 | 19.7 | 27.8 | 10.7 | 7.5 |
| forwarded write, middle router, 2 flows (125.7) | 85% | 2.8 | 50.9 | 0.3 | 25.9 | 26.8 | 12.2 | 6.7 |
| 1-hop E/W bidirectional (135.0) | 78% | 16.3 | 21.9 | 37.1 | 0.0 | 33.1 | 20.2 | 6.4 |
| 1D 1-hop write (129.2) | 75% | 17.0 | 24.8 | 34.7 | 0.0 | 31.7 | 13.8 | 7.3 |
| 1D 3-hop write, middle router (140.2) | 86% | 9.2 | 35.6 | 0.0 | 51.0 | 24.2 | 14.7 | 5.4 |
| 1D multicast, mean (145.0) | 63% | 30.3 | 33.9 | 19.1 | 9.8 | 27.1 | 7.8 | 17.1 |

- **poll:** reading the packets-received register and the empty pass, including waiting for packets (spare time).
- **decode+admit:** the header word and route bytes, the downstream credit test and its reload (1D: the route
  decision).
- **local:** the NoC commands of local delivery. **forward:** the copy to the next router (one per direction).
- **completions:** the per-pass batch check plus proving a batch's writes have left and sending its credit frame.
- **retire+atomic:** retiring the slot (packets-received register, slot record) and deferred flushed atomics.

## What the map says

1. **E0 sets the rate on 1-hop writes; E1 on scatter, fused writes and multi-hop.** On 1-hop writes E0 has no spare
   time while E1 polls for 11–17 cycles a packet. On scatter and fused writes E1 polls for 6 or less and E0's checks
   absorb 25–35 cycles of waiting. 3-hop, forwarded and multicast routers are near their limit on both ERISCs.
2. **The send is a little under half of the sender's time.** `send_packet` costs 53–63 cycles per packet in every
   row. On the 1-hop sender the per-pass checks cost about 45–55 more: worker send check about 13–16, connection poll
   about 10 (the semaphore load is used at once), forwarding idle test about 11 (three idle forwarding channels),
   group read about 9, completion check about 4–7.
3. **E1's completion credits cost 20–33 cycles per packet even with batches of 4,** about 80–130 per credit frame:
   the per-pass check (4–9), the newest-write check, the TX-queue test and the frame itself.
4. **Retiring a slot costs E1 14–20 cycles per packet** (the packets-received register update and slot bookkeeping),
   and decode 18–22 (header loads used at once).
5. **Local delivery and forwarding are the expected large items:** a write 35–38, an atomic 57, a scatter 74, a fused
   write 80; a forward about 51 per copy, plus 12–20 of admission on forwarding routers (more where downstream credits
   are reloaded: about 24 on the forwarded row's middle router).

## Files

- `fabric_exp/consolidation/pc_sampler.py`: the sampler. `pcs3_runs.sh 2D|1D`: the stage-3 jobs (fresh cache per
  topology; the first row compiles and logs the routers).
- `fabric_exp/consolidation/pcs_sections.py`: `twins` (rebuild the run's routers with `-g`) and `rows` (sections per
  ERISC); `pcs_tables.py`: the tables above. `pcs_report.py`: per-target overview (function level).
- `fabric_exp/pcs3/*.yaml`: one long row each. Results: `pcs3_<row>_13129088_trial.csv` (rate),
  `pcs3_<row>_samples.json` (targets, load addresses), `pcs3_2d_sections.json` / `pcs3_1d_sections.json` (sections per
  ERISC). Raw samples (`*_samples.bin`) stay local.
