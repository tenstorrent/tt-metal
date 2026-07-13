# SP3 (small-operand broadcast) + SP4 (output write) — contention checks

**Scope:** bf16 in/out, HiFi2, fp32 accum. Both are **contention checks**: do these small side-flows
steal bandwidth from the big read? Method: extend the SP5 interleaved-read harness (kernel-time BW of
the reader's BRISC zone), add the flow concurrently on NCRISC, compare to read-only baseline.
Board: BH p150b, gap-free rectangular compute grids (physical worker cols 1–7 / 10–13; a gap at
physical x=8–9, so rects must not straddle it).

## SP4 — output write (MEASURED, both regimes)
Each core writes its output band to a DRAM-interleaved buffer (NCRISC/NOC1) while reading the big
operand (BRISC/NOC0).

| regime | cores | big read | + output write | baseline BW | +write BW | degradation |
|---|---:|---|---|---:|---:|---:|
| B (16384×6144×128) | 32 (8×4) | 192 MB in0 | 4 MB out (64 t/core) | 362.2 | 357.4 | **1.3 %** |
| B | 30 (6×5) | 192 MB in0 | 6 MB out (96 t/core) | 319.8 | 317.5 | **0.7 %** |
| A* (32×6144×9216) | 16 (4×4) | 108 MB in1 | 0.5 MB out (18 t/core) | 297.2 | 295.9 | **0.4 %** |

**Output write overlaps the big read; cost < 1.5 % in all cases.** No GDDR read/write-turnaround
pathology at these lopsided volumes (output is 0.3–3 % of the read). ⇒ the op writes output
DRAM-interleaved on the second RISC with no special path; no batching/deferral needed at this scale.

**\* IMPORTANT — the Regime-A baseline (297) is NOT Regime A's read ceiling.** This harness only has
the SP5 **interleaved** reader (per-tile 2 KB reads; the Regime-B mechanism, ceiling ~419). Regime A's
real read is **in1 DRAM-width-sharded, one adjacent reader per bank, 16 KB bursts = ~510 GB/s** (SP1,
re-confirmed). On top of the wrong mechanism, the 16-core compact 4×4 corner is both under-saturated
(interleaved needs ~24–32 cores) and poorly placed — clustering readers in a corner funnels traffic to
the spread-out DRAM banks through few NoC links. Measured at 108 MB / depth 32 / 16 cores:
compact 4×4 = 299, same 16 cores row-major spread = 375, 8 adjacent per-bank sharded readers = **510.7**.
So the SP4-A number should be read as "write added 0.4 % to a 297 GB/s *interleaved* read"; against
the true ~510 sharded read the write fraction is even smaller. A proper Regime-A contention test needs
the sharded/adjacent reader wired into the harness (built in the Regime-A prototype).

## SP3 — small-operand broadcast (DERIVED + partial)
The small operand must reach **all** compute cores (each owns a full-small-dim × big-band output).
Two delivery options:

**Redundant per-core DRAM read (anti-pattern):** every core re-reads the full small operand.
- Regime B: in1 = 1.5 MB × 32 cores = **48 MB extra DRAM = +25 % on the 192 MB read → +25 % runtime.**
- Regime A: in0 = 0.4 MB × 16 = 6 MB extra = +5.5 % on 108 MB.
This cost is linear in the redundant volume (read BW is volume-linear, SP1), so it needs no separate
measurement — and it is clearly unacceptable for Regime B.

**Multicast once, reuse (the design choice):** one source core reads the small operand and NoC-mcasts
to all compute cores.
- Broadcast volume: Regime B 1.5 MB (×re-broadcast factor if it doesn't stay L1-resident → 3–6 MB);
  Regime A 0.4–0.8 MB. That is the **same magnitude as SP4's output write** (6 MB → 0.7 %).
- Expected read-BW degradation: **~1–3 % (B), < 1 % (A)** — small and overlappable, by analogy to the
  measured SP4 write of equal volume on the same NCRISC/NOC path.

**⇒ Use mcast, not redundant reads.** Redundant reads cost +25 % (B); mcast costs ~1–3 %.

### Caveat — mcast microbench not cleanly measured on silicon
The faithful mcast microbench (`--mc-tiles`) **hangs**: `noc_async_write_multicast` to the physical
bounding-box of the compute rectangle stalls the flush barrier. Root cause is a Blackhole
coordinate/rectangle issue — the physical worker grid is non-contiguous (gap at x=8–9; possible
harvested rows), so a naive bounding-box mcast targets non-worker cores whose acks never arrive. This
is an **integration detail, not a finding blocker**: the real op will use the proven mcast path
(minimal_matmul's semaphore-handshake sender to a validated worker CoreRangeSet), and the volume math
+ SP4 analog already bound the cost. Fixing the microbench = derive the mcast rect from the device's
actual worker set (not a physical bounding box) + receiver-ready semaphores.

## Transfer to the op / composition
- **Output path:** interleaved output write on the 2nd RISC, overlapped — free (<1.5 %). Honors the
  output-interleaved constraint with no special layout.
- **Small-operand path:** mcast once and reuse in L1; ~1–3 % overlapped. Never redundant-read
  (Regime B would pay +25 %). Mcast source/injector placement should avoid the hot DRAM columns
  (SP5) and the physical-grid gap.
- **The "memory is the only limiter" check:** big read (SP1/SP5) is ~99 %+ of the cost; compute
  (SP2, ~90 % util, hidden), broadcast (~1–3 %), and output write (<1.5 %) all overlap under it. So
  the op is read-bound at ~419 GB/s (B) / ~510 GB/s (A) as designed. The final validation is a full
  read+mcast+write+compute run confirming the achieved BW still equals the read ceiling.

## Artifacts
- Harness: `sp5_interleaved_read/test_interleaved_read` with `--gx --gy` (rect grid), `--wr-tiles`
  (SP4), `--mc-tiles --mc-reps --mc-chunk` (SP3, currently hangs — see caveat). Kernels:
  `writer_interleaved.cpp`, `mcast_src.cpp`.
- `parse_kernel_bw.py <csv> <bytes> BRISC` isolates the reader BW.
