# Streaming profiler: the zone wire format, and what it measures like

This documents the variable-width zone packet family the SPSC streaming profiler puts on the wire
(producer: `kernel_profiler.hpp`; decoder: `spsc_marker_decode.hpp`; plain-C constants:
`spsc_packet.h`), and the measurements that shaped it. For how to *consume* the stream, see
[STREAMING_PROFILER.md](STREAMING_PROFILER.md) — consumers see complete zones either way and none of
this changes that contract. For what zones and point markers look like in the Tracy GUI,
see [docs/zone_gifs/README.md](docs/zone_gifs/README.md).

## Zone packets

Every zone ships WHOLE, in one packet, emitted at scope **close** — the RAII scope object carries the
start timestamp (`start_hi`/`start_lo`, 8 B of member state), so the open touches nothing but the wall
clock. Packets are sized by need. `word0 = type(5) | id27` in all of them; the id is the full 27-bit
structural zone id (`tu_id(17) << 10 | local(10)`), ELF-name-resolved on the host.

| type | name | words | payload after word0 | expresses |
|---|---|---|---|---|
| 3 | `ZONE_S` | 2 | `end_delta16 << 16 \| dur16` | end within 2^16 cycles (~48 µs @1.35 GHz) of the lane cursor AND duration ≤ 2^16 cycles |
| 2 | `ZONE_ATOMIC` ("M") | 3 | `end_lo32`, `dur32` | duration < 2^32 cycles (~3.2 s), any gap; **re-anchors the cursor** |
| 4 | `ZONE_L` | 5 | `end_lo`, `end_hi`, `dur_lo`, `dur_hi` | anything — two full 64-bit values, self-contained |
| 0/1 | legacy `ZONE_START`/`END` pair | 2+2 | `timer_lo` each | no emitter left on workers; decode-only until retired |

### The lane cursor

Producer and decoder each keep, per lane, a 64-bit **cursor = the end of the last S or M zone**. Zones
are emitted at close, and closes happen in end order — the same per-lane monotonicity invariant the
host's order-regression check relies on — so end-to-end deltas are unsigned. This is why the delta is
**end-relative, never start-relative**: a closing *parent* zone's start lies before its already-closed
children's ends, so start deltas go negative; ends never do. Start is always reconstructed as
`end − duration`.

- Only S and M move the cursor, identically on both sides. The M packet's absolute `end_lo` is the
  re-anchor (`cursor = sticky_hi << 32 | end_lo`).
- L (and the legacy pair) leave the cursor alone — a stale cursor is merely conservative: the next
  zone's delta overflows 16 bits and falls back to M, which re-anchors.
- The producer invalidates its cursor (`hi = ~0`) at `init_profiler()` and on the idle-launch rewind,
  which makes the S class test fail arithmetically — the first zone after any launch is always an
  absolute re-anchor, with no extra branch.
- There is deliberately **no sticky-lo**: a separate re-anchor packet can never beat a 3-word M that
  also carries a zone. `STICKY_TIMER` (the 27-bit high half, ~one per 3.2 s) survives for M/`PP_DATA`/
  `PP_EVENT`, which carry absolute low words; S never needs it — the 64-bit cursor add crosses the
  2^32 lo-wrap for free.

The producer's class test is one OR-tree into one branch, laid out as the fall-through:
`(((c_lo_d | lo_d) >> 16) | c_hi_d | hi_d) == 0`.

### The stall zone (PRODUCER-STALL)

Pinned to **M with a saturating duration**, written straight into the ring's stall reserve with no
room check — a room check from inside the full-ring path recurses into another stall scope, which is
exactly what the reserve exists to prevent. Because the stall OPEN writes nothing (option-C member
state like every zone), the reserve covers only the CLOSE: **4 words** (3-word packet + 1 sticky),
down from the 6 the old START/END pair needed. Neither S (buys zero reserve — the reserve must cover
the deterministic worst case) nor L (grows the reserve to 5) is worth it there; a ≥2^32-cycle stall is
a wedged drainer, not a measurement, so saturation loses nothing real.

### ZONE_L and the >3.2 s fallback

A zone whose duration overflows 32 bits ships as one self-contained ZONE_L — no stickies, no cursor.
The decoder normalizes it to a synthetic START/END pair for the delivery-side pairing stack (a 64-bit
duration cannot ride the 32-bit dur argument); the in-the-past synthetic START trips the per-lane
order-regression diagnostic exactly once per nesting *parent*, kept on purpose as wedge visibility.

The decoder **normalizes at the emit boundary** — S emits as wire-type ATOMIC with the cursor-resolved
end, L as the synthetic pair — so the receiver, every consumer, and the stall classifier see only the
types they always saw.

## Findings (bh-26, Blackhole, 1.35 GHz, 2026-08-26)

### Which classes real workloads actually use

The receiver reports a per-stream histogram
(`zone classes: S n (x%) | M n (x%) | L n | legacy pair halves n | zone wire x MB`):

| workload | S | M |
|---|---|---|
| dense synthetic (test_perf_debug_zones knee mode, 10 zones/iter back-to-back) | **100.0%** | exactly 1,200 = 600 per-lane first-zone re-anchors + 600 trailing zones |
| **ResNet-50 inference (batch 16)** | **0%** | **100%** (26k–99k zones/run, lossless) |

Real-model zones at FW/kernel granularity have end-to-end deltas — dispatch gaps and kernel
durations — far beyond the 16-bit/48 µs S window. **S is a dense-instrumentation feature; op-level
captures ride M entirely.** If S should ever matter for models, the lever is the delta width or
denser in-kernel instrumentation, not the duration field.

### Producer cost per class (`--empty` calibration, 1×1 grid, 500 iters, median dur+gap)

| class | cycles | ns | ring words |
|---|---|---|---|
| S | 50–51 | ~37 | 2 |
| M | 41–44 measured with the S test compiled out; ~45–47 in the shipping build | ~31 | 3 (+1 sticky, rare) |
| L | 74 | ~55 | 5 |

**ZONE_S is not a producer-cycle win**: its cursor bookkeeping (two RAM stores + a 64-bit delta + the
class test) slightly outweighs the one saved L1 store. What S buys is **wire volume** — and volume is
what the pipeline scales with.

### What the family did to the pipeline

- Wire: −33% on a dense capture (2 words/zone instead of 3); a 2000-iter full-grid decode smoke went
  159.1 → 104.9 MB.
- Producer-stall onset knee (10k iters, RING=448, 6F+1M, gate off, NO_DECODE, slow dispatch): clean
  **d7 → d2**, onset d1 — five delay steps, entirely attributable to the volume drop (ring runway,
  filler bandwidth and egress all scale with words), not to per-zone cycles.
- The stall-reserve shrink (6 → 4 words) alone was measured worth one full knee step: pinning the
  reserve back at 6 with the atomic stall packet in place reproduced the old knee exactly.
- Accounting identities used to verify all of this, worth keeping: a clean run decodes exactly
  `iters × 6000 + 600` records (10 zones × 600 lanes + 1 trailing per lane), and a stalling run's
  record surplus equals the device L1 stall counter **to the unit** (one atomic stall zone per stall;
  under the old pair wire the surplus was exactly 2× the stalls).

### Rendering long zones in Tracy

Tracy's server carries an unwrap heuristic for wrapping GPU timestamp counters
(`TracyWorker.cpp ProcessGpuTime`): a backwards jump > 2^31 ticks in one context's GpuTime stream is
read as a counter wrap and everything after is shifted up by a power of two, cumulatively. Flushing
lane-by-lane jumps back to capture start at every lane boundary, so any capture whose per-lane span
exceeds **2^31 ticks (~1.6 s)** staggered its RISCs by huge power-of-two offsets. The sink
(`perf_debug_tracy_consumer.cpp flush_zones`) therefore merges the lanes of each context by timestamp
before pushing — each lane's bracket sequence is already non-decreasing in ts, so a ts-only stable
sort is a correct k-way merge — making the context stream monotone so the heuristic can never fire.
Verified: a 5 s-zone capture's five RISC kernels render within ~100 ticks of one another.
`tracy_ctx_inspect` (with `CTX_ALL_THREADS=1`) prints per-thread zone spans to check exactly this.
