# Streaming profiler: the zone wire format, and what it measures like

This documents the variable-width zone packet family the SPSC streaming profiler puts on the wire
(producer: `kernel_profiler.hpp`; decoder: `spsc_marker_decode.hpp`; plain-C constants:
`spsc_packet.h`). For how to consume the stream, see
[STREAMING_PROFILER.md](STREAMING_PROFILER.md) — consumers see complete zones either way and none of
this changes that contract. For what zones and point markers look like in the Tracy GUI,
see [docs/zone_gifs/README.md](docs/zone_gifs/README.md).

## Zone packets

Every zone ships whole, in one packet, emitted at scope **close** — the RAII scope object carries the
start timestamp (`start_hi`/`start_lo`, 8 B of member state), so the open touches nothing but the wall
clock. Packets are sized by need. `word0 = type(5) | id27` in all of them; the id is the full 27-bit
structural zone id (`tu_id(17) << 10 | local(10)`), ELF-name-resolved on the host.

| type | name | words | payload after word0 | expresses |
|---|---|---|---|---|
| 3 | `ZONE_S` | 2 | `end_delta16 << 16 \| dur16` | end within 2^16 cycles (~48 µs @1.35 GHz) of the lane cursor and duration ≤ 2^16 cycles |
| 2 | `ZONE_ATOMIC` ("M") | 3 | `end_lo32`, `dur32` | duration < 2^32 cycles (~3.2 s), any gap; **re-anchors the cursor** |
| 4 | `ZONE_L` | 5 | `end_lo`, `end_hi`, `dur_lo`, `dur_hi` | anything — two full 64-bit values, self-contained |
| 0/1 | `ZONE_START`/`END` pair | 2+2 | `timer_lo` each | no worker emits it; the decoder still accepts it |

### The lane cursor

Producer and decoder each keep, per lane, a 64-bit **cursor = the end of the last S or M zone**. Zones
are emitted at close, and closes happen in end order — the same per-lane monotonicity invariant the
host's order-regression check relies on — so end-to-end deltas are unsigned. This is why the delta is
**end-relative, never start-relative**: a closing *parent* zone's start lies before its already-closed
children's ends, so start deltas go negative; ends never do. Start is always reconstructed as
`end − duration`.

- Only S and M move the cursor, identically on both sides. The M packet's absolute `end_lo` is the
  re-anchor (`cursor = sticky_hi << 32 | end_lo`).
- L (and the paired form) leave the cursor alone — a stale cursor is merely conservative: the next
  zone's delta overflows 16 bits and falls back to M, which re-anchors.
- The producer invalidates its cursor (`hi = ~0`) at `init_profiler()` and on the idle-launch rewind,
  which makes the S class test fail arithmetically — the first zone after any launch is always an
  absolute re-anchor, with no extra branch.
- There is deliberately **no sticky-lo**: a separate re-anchor packet can never beat a 3-word M that
  also carries a zone. `STICKY_TIMER` (the 27-bit high half, ~one per 3.2 s) applies to M/`PP_DATA`/
  `PP_EVENT`, which carry absolute low words; S never needs one — the 64-bit cursor add crosses the
  2^32 lo-wrap for free.

The producer's class test is one OR-tree into one branch, laid out as the fall-through:
`(((c_lo_d | lo_d) >> 16) | c_hi_d | hi_d) == 0`.

### The stall zone (PRODUCER-STALL)

Pinned to **M with a saturating duration**, written straight into the ring's stall reserve with no
room check — a room check from inside the full-ring path recurses into another stall scope, which is
exactly what the reserve exists to prevent. The stall open writes nothing (member state, like every
zone), so the reserve covers only the close: **4 words** (3-word packet + 1 sticky). S buys zero
reserve, since the reserve must cover the deterministic worst case, and L grows it to 5; a ≥2^32-cycle
stall is a wedged relay rather than a measurement, so saturating the duration loses nothing real.

### ZONE_L and the >3.2 s fallback

A zone whose duration overflows 32 bits ships as one self-contained ZONE_L — no stickies, no cursor.
The decoder normalizes it to a synthetic START/END pair for the delivery-side pairing stack (a 64-bit
duration cannot ride the 32-bit dur argument); the in-the-past synthetic START trips the per-lane
order-regression diagnostic once per nesting parent, which is kept as wedge visibility.

The decoder **normalizes at the emit boundary** — S emits as wire-type ATOMIC with the cursor-resolved
end, L as the synthetic pair — so the receiver, every consumer, and the stall classifier see only the
types they always saw.

## Measured behavior (Blackhole, 1.35 GHz)

### Which classes real workloads use

The receiver's per-stream decode-path line breaks records down by class (`decode paths: ... zoneS16 +
zone8 + atomic16 ...`):

| workload | S | M |
|---|---|---|
| dense synthetic (test_streaming_profiler_zones, 10 zones/iter) | ~100% | per-lane first-zone re-anchors, trailing zones |
| op-level model capture (FW/kernel-granularity zones) | 0% | 100% |

Zones at FW/kernel granularity have end-to-end deltas — dispatch gaps and kernel durations — far
beyond the 16-bit/48 µs S window. **S is a dense-instrumentation feature; op-level captures ride M
entirely.** If S should ever matter for models, the lever is the delta width or denser in-kernel
instrumentation, not the duration field.

### Producer cost per class (`--empty` calibration, 1×1 grid, 500 iters, median dur+gap)

| class | cycles | ns | ring words |
|---|---|---|---|
| S | 50–51 | ~37 | 2 |
| M | 45–47 | ~34 | 3 (+1 sticky, rare) |
| L | 74 | ~55 | 5 |

**ZONE_S is not a producer-cycle win**: its cursor bookkeeping (two RAM stores + a 64-bit delta + the
class test) slightly outweighs the one saved L1 store. What S buys is **wire volume** — and volume is
what the pipeline scales with: 2 words per zone instead of 3 is a third off a dense capture, and the
producer-stall onset knee moves with the volume, not with per-zone cycles.

Accounting identities worth keeping for verification: a clean run decodes exactly
`iters × 6000 + 600` records (10 zones × 600 lanes + 1 trailing per lane), and a stalling run's
record surplus equals the device L1 stall counter to the unit, one atomic stall zone per stall.

### Rendering long zones in Tracy

Tracy's server carries an unwrap heuristic for wrapping GPU timestamp counters
(`TracyWorker.cpp ProcessGpuTime`): a backwards jump > 2^31 ticks in one context's GpuTime stream is
read as a counter wrap, and everything after it is shifted up by a power of two, cumulatively. So the
timestamps pushed for one GPU context must be monotone: a sink that flushes lane by lane jumps back
to capture start at every lane boundary, and any capture whose per-lane span exceeds **2^31 ticks
(~1.6 s)** then staggers its RISCs by huge power-of-two offsets. Each lane's bracket sequence is
already non-decreasing in ts, so merging a context's lanes by timestamp before pushing is a correct
k-way merge and keeps the heuristic from firing. `tracy_ctx_inspect` (with `CTX_ALL_THREADS=1`)
prints per-thread zone spans to check exactly this.
